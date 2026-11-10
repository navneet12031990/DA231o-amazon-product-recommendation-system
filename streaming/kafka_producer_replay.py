"""Module 4 — replay held-out test ratings into Kafka (SPEC.md Section 5, Streaming).

Distributed, so it scales with the number of events and the replay rate:
  1. the earliest streaming.replay_max_events test ratings are numbered in time order with a
     distributed sort + zipWithIndex (nothing is collected on the driver);
  2. each event gets a one-second slot (seq // replay_rate_per_sec); the slotted events are cached;
  3. every second, the executors write that slot's events with Spark's Kafka sink.
Messages are JSON {event_id, user_id, item_id, rating, event_time}, keyed by user_id; the Kafka
record timestamp (set when sent) is the publish time used for latency.
"""
import time

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def _log(msg: str) -> None:
    print(f"[producer] {msg}", flush=True)


def _join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def slotted_events(config: dict) -> DataFrame:
    """(slot, key, value) for the replay, numbered in time order without any driver collect."""
    spark = SparkSession.builder.getOrCreate()
    st = config["streaming"]
    n_max, rate = int(st["replay_max_events"]), int(st["replay_rate_per_sec"])
    test = spark.read.parquet(_join(config["data"]["processed_prefix"], "test.parquet"))
    ordered = test.select("user_id", "item_id", F.col("rating").cast("double").alias("rating"),
                          F.col("timestamp").cast("long").alias("event_time")) \
                  .orderBy("event_time", "user_id", "item_id")
    numbered = (ordered.rdd.zipWithIndex()                       # distributed, keeps the sort order
                .filter(lambda x: x[1] < n_max)
                .map(lambda x: (x[1],) + tuple(x[0]))
                .toDF(["seq", "user_id", "item_id", "rating", "event_time"]))
    value = F.to_json(F.struct(F.concat_ws("|", "user_id", "item_id").alias("event_id"),
                               "user_id", "item_id", "rating", "event_time"))
    return numbered.select((F.col("seq") / rate).cast("long").alias("slot"),
                           F.col("user_id").alias("key"), value.alias("value"))


def kafka_sink(config: dict):
    st = config["streaming"]

    def send(slot_df: DataFrame) -> None:
        (slot_df.select("key", "value").write.format("kafka")
                .option("kafka.bootstrap.servers", st["kafka_bootstrap_servers"])
                .option("topic", st["kafka_topic"])
                .option("kafka.linger.ms", "0")
                .save())
    return send


def replay_to_kafka(config: dict, sink=None) -> dict:
    st = config["streaming"]
    rate = int(st["replay_rate_per_sec"])
    send = sink or kafka_sink(config)
    ev = slotted_events(config).persist(StorageLevel.MEMORY_AND_DISK)
    total = ev.count()
    n_slots = -(-total // rate) if total else 0
    _log(f"replaying {total:,} events to {st['kafka_topic']} at {rate}/s in {n_slots} one-second slots "
         f"(~{n_slots}s) via {st['kafka_bootstrap_servers']}")
    t0 = time.time()
    max_lag = 0.0
    for k in range(n_slots):
        delay = t0 + k - time.time()
        if delay > 0:
            time.sleep(delay)
        else:
            max_lag = max(max_lag, -delay)       # behind schedule: send now, record the lag
        send(ev.where(F.col("slot") == k))
        if k % 30 == 0:
            _log(f"slot {k + 1}/{n_slots} sent after {time.time() - t0:.0f}s")
    end = t0 + n_slots
    if end > time.time():
        time.sleep(end - time.time())            # the last slot lasts a full second
    elapsed = time.time() - t0
    ev.unpersist()
    _log(f"done: {total:,} events in {elapsed:.1f}s ({total / max(elapsed, 1e-9):.1f}/s); "
         f"max lag behind schedule {max_lag:.2f}s")
    return {"events": total, "seconds": elapsed, "max_lag_sec": max_lag}
