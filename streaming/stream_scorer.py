"""Module 4 — live scoring of Kafka events (SPEC.md Section 5, Streaming).

Structured Streaming over the topic. At start, Module 2's build_serving_tables turns the saved model
named by streaming.model plus Module 1's id mappings into two cached serving tables (raw id ->
index, bias, vector). Per micro-batch (foreachBatch): parse JSON, score with predict_events (batch
broadcast, serving tables never shuffled, so per-batch cost follows the batch size), count unknown
ids, write predictions + per-event latency + the batch's processing time.
Writes are idempotent (batch N overwrites its own folder), so a restart that replays a batch never
duplicates output.
"""
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from centralized.train_als import build_serving_tables, load_model, predict_events
from evaluation.metrics import write_json

EVENT_SCHEMA = "event_id string, user_id string, item_id string, rating double, event_time long"


def _log(msg: str) -> None:
    print(f"[scorer] {msg}", flush=True)


def _join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def kafka_source(spark: SparkSession, config: dict) -> DataFrame:
    """Streaming DataFrame (value string, timestamp) from the topic, earliest offsets first."""
    st = config["streaming"]
    return (spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", st["kafka_bootstrap_servers"])
            .option("subscribe", st["kafka_topic"])
            .option("startingOffsets", "earliest")
            .option("maxOffsetsPerTrigger", int(st["max_offsets_per_trigger"]))   # backpressure
            .load()
            .select(F.col("value").cast("string").alias("value"), F.col("timestamp")))


def run_stream_scorer(config: dict, source_override: DataFrame = None, max_seconds: float = None) -> dict:
    spark = SparkSession.builder.getOrCreate()
    st, d = config["streaming"], config["data"]
    out = _join(d["results_prefix"], "streaming")
    model_name = st["model"]
    t_setup = time.time()
    bundle = load_model(_join(d["results_prefix"], f"models/{model_name}"))
    user_map = spark.read.parquet(_join(d["processed_prefix"], "user_id_mapping.parquet"))
    item_map = spark.read.parquet(_join(d["processed_prefix"], "item_id_mapping.parquet"))
    serving = build_serving_tables(bundle, user_map, item_map)      # once; cached on the executors
    _log(f"model {model_name} ({bundle['meta'].get('model_type')}); serving tables: {serving['n_users']:,} users, "
         f"{serving['n_items']:,} items, built in {time.time() - t_setup:.0f}s; checkpoint {st['checkpoint_location']}")
    rows_per_file = int(st["output_rows_per_file"])

    state = {"busy": False, "last_activity": None, "batches": 0, "events": 0, "scored": 0}

    def process(batch_df: DataFrame, batch_id: int) -> None:
        state["busy"] = True
        t0 = time.time()
        try:
            ev = (batch_df.select(F.from_json("value", EVENT_SCHEMA).alias("e"), F.col("timestamp").alias("kafka_ts"))
                          .select("e.*", "kafka_ts")).cache()
            n = ev.count()
            if n == 0:
                ev.unpersist()
                return
            preds = predict_events(serving, ev).cache()
            n_scored = preds.count()
            scored_ts = time.time()                      # predictions materialised
            (preds.select("event_id", "user_id", "item_id", "user_idx", "item_idx", "rating", "prediction",
                          F.col("kafka_ts").cast("double").alias("kafka_ts"))
                  .withColumn("scored_ts", F.lit(scored_ts))
                  .withColumn("latency_sec", F.lit(scored_ts) - F.col("kafka_ts"))
                  .coalesce(max(1, -(-n_scored // rows_per_file)))
                  .write.mode("overwrite").parquet(_join(out, f"predictions/batch_id={batch_id}")))
            stats = {"batch_id": int(batch_id), "received": int(n), "scored": int(n_scored),
                     "dropped_unknown": int(n - n_scored), "scored_ts": scored_ts,
                     "processing_sec": time.time() - t0}
            write_json(_join(out, f"batches/batch_{batch_id}.json"), stats)
            preds.unpersist(); ev.unpersist()
            state["batches"] += 1; state["events"] += n; state["scored"] += n_scored
            _log(f"batch {batch_id}: {n:,} events, {n_scored:,} scored, {n - n_scored:,} unknown dropped, "
                 f"{stats['processing_sec']:.1f}s")
        finally:
            state["busy"] = False
            state["last_activity"] = time.time()         # a batch that ran counts as activity

    source = source_override if source_override is not None else kafka_source(spark, config)
    q = (source.writeStream.foreachBatch(process)
               .option("checkpointLocation", st["checkpoint_location"])
               .trigger(processingTime=f"{int(st['trigger_seconds'])} seconds")
               .start())
    t0 = time.time()
    limit = float(max_seconds) if max_seconds else float(st["max_run_seconds"])
    idle = float(st["idle_stop_seconds"])
    try:
        while q.isActive:
            time.sleep(1)
            now = time.time()
            # idle = no batch running AND no batch with events finished for `idle` seconds
            if (state["events"] > 0 and not state["busy"] and state["last_activity"]
                    and now - state["last_activity"] > idle and not q.status.get("isDataAvailable", False)):
                _log(f"no new events for {idle:.0f}s and no batch running, stopping"); break
            if now - t0 > limit:
                _log(f"reached the {limit:.0f}s limit, stopping"); break
        if q.exception() is not None:
            raise RuntimeError(f"streaming query failed: {q.exception()}")
    finally:
        q.stop()
        # stop() can return before Spark unregisters the query; wait, so a restart on the same
        # checkpoint in this session is not rejected as "concurrently using" it
        for _ in range(60):
            if all(x.id != q.id for x in spark.streams.active):
                break
            time.sleep(0.5)
    serving["users"].unpersist(); serving["items"].unpersist()
    _log(f"stopped: {state['batches']} batches, {state['events']:,} events, {state['scored']:,} scored")
    return {k: state[k] for k in ("batches", "events", "scored")}


def summarize_stream(config: dict) -> dict:
    """results/streaming_latency/ (CSV per event) + streaming_summary.json, all Spark aggregations
    (no per-event collect on the driver, so it scales with the number of events)."""
    spark = SparkSession.builder.getOrCreate()
    d, st = config["data"], config["streaming"]
    out = _join(d["results_prefix"], "streaming")
    p = spark.read.parquet(_join(out, "predictions")).cache()
    b = spark.read.json(_join(out, "batches/*.json"), multiLine=True)   # one JSON object per file
    bs = b.agg(F.sum("received").alias("received"), F.sum("dropped_unknown").alias("dropped"),
               F.count("*").alias("batches"), F.avg("processing_sec").alias("proc_mean"),
               F.expr("percentile_approx(processing_sec, 0.5, 10000)").alias("proc_p50"),
               F.max("processing_sec").alias("proc_max")).first()
    agg = p.agg(F.count("*").alias("n"), F.countDistinct("event_id").alias("distinct"),
                F.avg("latency_sec").alias("mean"), F.max("latency_sec").alias("max"),
                F.expr("percentile_approx(latency_sec, array(0.5, 0.95, 0.99), 100000)").alias("q"),
                F.sqrt(F.avg(F.pow(F.col("prediction") - F.col("rating"), 2))).alias("rmse"),
                F.expr("percentile_approx(kafka_ts, array(0.1, 0.9), 100000)").alias("pub"),
                F.expr("percentile_approx(scored_ts, array(0.1, 0.9), 100000)").alias("sco"),
                F.min("kafka_ts").alias("first_pub"), F.max("scored_ts").alias("last_sco")).first()
    n = int(agg["n"])
    pub_lo, pub_hi = agg["pub"]; sco_lo, sco_hi = agg["sco"]
    in_pub = p.where((F.col("kafka_ts") >= pub_lo) & (F.col("kafka_ts") <= pub_hi)).count()
    publish_rate = in_pub / (pub_hi - pub_lo) if pub_hi > pub_lo else None
    if sco_hi - sco_lo >= float(st["trigger_seconds"]):
        in_sco = p.where((F.col("scored_ts") >= sco_lo) & (F.col("scored_ts") <= sco_hi)).count()
        throughput, how = in_sco / (sco_hi - sco_lo), "stable window (10th-90th percentile of scored time)"
    else:   # scoring concentrated in very few batches: whole-run rate instead of a degenerate window
        throughput, how = n / max(agg["last_sco"] - agg["first_pub"], 1e-9), "whole run (last scored - first published)"
    summary = {
        "model": st["model"],
        "events_received": int(bs["received"] or 0), "events_scored": n,
        "dropped_unknown": int(bs["dropped"] or 0), "batches": int(bs["batches"]),
        "duplicate_event_ids": int(n - agg["distinct"]),
        "batch_processing_sec": {"mean": bs["proc_mean"], "p50": bs["proc_p50"], "max": bs["proc_max"]},
        "latency_sec": {"mean": agg["mean"], "p50": agg["q"][0], "p95": agg["q"][1], "p99": agg["q"][2],
                        "max": agg["max"]},
        "publish_rate_per_sec": publish_rate,
        "scoring_throughput_per_sec": throughput, "throughput_window": how,
        "rmse_streamed": agg["rmse"],
        "replay_rate_per_sec": int(st["replay_rate_per_sec"]),
        "trigger_seconds": int(st["trigger_seconds"]),
    }
    (p.select("event_id", "kafka_ts", "scored_ts", "latency_sec")
       .coalesce(max(1, -(-n // int(st["output_rows_per_file"]))))
       .write.mode("overwrite").option("header", True).csv(_join(d["results_prefix"], "streaming_latency")))
    write_json(_join(d["results_prefix"], "streaming_summary.json"), summary)
    p.unpersist()
    _log(f"summary: {n:,} scored of {summary['events_received']:,} received ({summary['dropped_unknown']:,} unknown "
         f"dropped, {summary['duplicate_event_ids']} duplicates) in {summary['batches']} batches "
         f"(processing p50 {bs['proc_p50']:.1f}s); latency p50 {agg['q'][0]:.2f}s p95 {agg['q'][1]:.2f}s; "
         f"throughput {throughput:.1f}/s ({how}); RMSE {agg['rmse']:.4f}")
    return summary
