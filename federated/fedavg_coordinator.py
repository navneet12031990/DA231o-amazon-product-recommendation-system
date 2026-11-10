"""Module 3 — FedAvg coordinator for bias + residual matrix factorization (SPEC.md Section 4).

This SIMULATES federated averaging on data that is already centralized. It measures accuracy and
communication cost; it provides no privacy guarantee (CLAUDE.md Rule 6).
"""
import gc
import json
import time

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from evaluation.coldstart_eval import user_buckets
from evaluation.metrics import mae, read_text, rmse, write_json, write_text
from federated.local_als import (build_routes, build_user_blocks, deliver, init_item_factors,
                                 item_pass, make_identity, solve_users, user_pass)
from federated.partition_clients import partition_clients

FLOAT32_BYTES = 4
FLOAT64_BYTES = 8
INT32_BYTES = 4

NOTE = ("Federated averaging is simulated on an already-centralized public dataset to measure "
        "accuracy and communication cost. No privacy guarantee is provided or implied.")


def _log(msg: str) -> None:
    print(f"[federated] {msg}", flush=True)


def _join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


# ---------------------------------------------------------------- biases (round 0)
def federated_biases(part_df: DataFrame, damping: float) -> tuple:
    """mu and b_i aggregated from per-client sums (same formulas as Module 2); b_u computed per
    client from its own users. Returns (mu, item_bias(item_idx, b_i), user_bias(user_idx, b_u),
    bytes_transferred)."""
    per_client = part_df.groupBy("client_id").agg(F.sum("rating").alias("s"), F.count(F.lit(1)).alias("n"))
    tot = per_client.agg(F.sum("s").alias("s"), F.sum("n").alias("n"), F.count(F.lit(1)).alias("c")).first()
    mu = float(tot["s"]) / float(tot["n"])
    n_clients = int(tot["c"])

    per_client_item = (part_df.groupBy("client_id", "item_idx")
                       .agg(F.sum(F.col("rating") - mu).alias("s"), F.count(F.lit(1)).alias("n"))).cache()
    client_item_rows = per_client_item.count()
    item_bias = (per_client_item.groupBy("item_idx")
                 .agg((F.sum("s") / (F.sum("n") + damping)).alias("b_i")))
    user_bias = (part_df.join(item_bias, "item_idx")
                 .groupBy("user_idx")
                 .agg((F.sum(F.col("rating") - mu - F.col("b_i")) / (F.count(F.lit(1)) + damping)).alias("b_u")))
    # clients send (sum, count) and receive mu; then send (sum, count) per local item, receive b_i
    bias_bytes = FLOAT64_BYTES * (3 * n_clients + 3 * client_item_rows)
    per_client_item.unpersist()
    return mu, item_bias, user_bias, bias_bytes


# ---------------------------------------------------------------- shared item factors
# ---------------------------------------------------------------- evaluation
def _dot(a: str, b: str):
    return F.aggregate(F.zip_with(F.col(a), F.col(b), lambda x, y: x * y), F.lit(0.0),
                       lambda acc, x: acc + x)


def score_test(test_df: DataFrame, user_f: DataFrame, item_f: DataFrame, mu: float,
               user_bias: DataFrame, item_bias: DataFrame, config: dict) -> DataFrame:
    """Prediction = clip(mu + b_u + b_i + u . v); rows with an item missing from V are dropped.
    user_f/item_f may be None for the bias-only model (round 0)."""
    a = config["als"]
    df = test_df.select("user_idx", "item_idx", "rating", "bucket")
    if item_f is not None:
        df = (df.join(user_f.select("user_idx", F.col("features").alias("uf")), "user_idx")
                .join(item_f.select("item_idx", F.col("features").alias("vf")), "item_idx")
                .withColumn("_dot", _dot("uf", "vf")).drop("uf", "vf"))
    else:
        df = df.withColumn("_dot", F.lit(0.0))
    if user_bias is not None:
        df = (df.join(user_bias, "user_idx", "left").join(item_bias, "item_idx", "left")
                .withColumn("_b", F.lit(mu) + F.coalesce("b_u", F.lit(0.0)) + F.coalesce("b_i", F.lit(0.0)))
                .drop("b_u", "b_i"))
    else:
        df = df.withColumn("_b", F.lit(0.0))
    pred = F.least(F.greatest(F.col("_b") + F.col("_dot"), F.lit(float(a["prediction_min"]))),
                   F.lit(float(a["prediction_max"])))
    return df.withColumn("prediction", pred).drop("_b", "_dot")


def _eval(preds: DataFrame) -> dict:
    preds = preds.cache()
    out = {"rmse": rmse(preds, "prediction"), "mae": mae(preds, "prediction"),
           "rmse_cold": rmse(preds.where(F.col("bucket") == "cold"), "prediction"),
           "rmse_warm": rmse(preds.where(F.col("bucket") == "warm"), "prediction"),
           "rows": preds.count()}
    preds.unpersist()
    return out


def _release_round_garbage(sc) -> None:
    """Delete the finished round's shuffle files and blocks now. Spark's ContextCleaner removes
    them only after the driver's JVM garbage-collects the RDDs that produced them, which by default
    can lag far behind; meanwhile shuffle files from every round pile up on the executors' disks
    and later rounds get slower (observed: 423 s -> 616 s over rounds 1-5 at full size). Dropping
    the Python references and forcing a JVM GC lets the cleaner act every round."""
    gc.collect()
    sc._jvm.System.gc()


def _save_federated_model(path: str, u_df: DataFrame, v_df: DataFrame, mu: float, user_bias,
                          item_bias, config: dict, num_clients: int, rounds: int, residual: bool) -> None:
    """Federated model bundle, same conventions as Module 2's (SPEC.md Section 3)."""
    spark = SparkSession.builder.getOrCreate()
    a = config["als"]
    u_df.write.mode("overwrite").parquet(_join(path, "user_factors.parquet"))
    v_df.write.mode("overwrite").parquet(_join(path, "item_factors.parquet"))
    if residual:
        user_bias.write.mode("overwrite").parquet(_join(path, "user_bias.parquet"))
        item_bias.write.mode("overwrite").parquet(_join(path, "item_bias.parquet"))
    else:   # plain model: empty bias tables keep the bundle layout uniform
        spark.createDataFrame([], "user_idx int, b_u double").write.mode("overwrite").parquet(_join(path, "user_bias.parquet"))
        spark.createDataFrame([], "item_idx int, b_i double").write.mode("overwrite").parquet(_join(path, "item_bias.parquet"))
    write_json(_join(path, "meta.json"), {
        "model_type": "federated", "mu": float(mu), "residual_on_bias": bool(residual),
        "prediction_min": float(a["prediction_min"]), "prediction_max": float(a["prediction_max"]),
        "rank": int(a["rank"]), "reg_param": float(a["reg_param"]),
        "num_clients": int(num_clients), "rounds": int(rounds)})


# ---------------------------------------------------------------- training
def run_federated_training(config: dict, num_clients: int, save_model_to: str = None) -> dict:
    spark = SparkSession.builder.getOrCreate()
    a, fed = config["als"], config["federated"]
    rank, reg = int(a["rank"]), float(a["reg_param"])
    residual = bool(a["residual_on_bias"])
    rounds_n, epochs = int(fed["communication_rounds"]), int(fed["local_epochs"])
    eval_every = bool(fed["evaluate_every_round"])
    p = config["data"]["processed_prefix"]
    train = spark.read.parquet(_join(p, "train.parquet"))
    test = spark.read.parquet(_join(p, "test.parquet"))

    buckets = user_buckets(train, int(config["evaluation"]["cold_user_threshold"]))
    test_b = test.join(buckets, "user_idx").cache()
    n_test = test.count()

    t0 = time.time()
    part = partition_clients(train, int(num_clients), int(config["seed"])).cache()
    sizes = part.groupBy("client_id").agg(F.countDistinct("user_idx").alias("u"), F.count(F.lit(1)).alias("r")).collect()
    _log(f"{num_clients} clients: users per client {min(r['u'] for r in sizes):,}-{max(r['u'] for r in sizes):,}, "
         f"ratings per client {min(r['r'] for r in sizes):,}-{max(r['r'] for r in sizes):,}")

    if residual:
        mu, item_bias, user_bias, bias_bytes = federated_biases(part, float(a["baseline_damping"]))
        item_bias, user_bias = item_bias.localCheckpoint(), user_bias.localCheckpoint()
        ratings = (part.join(item_bias, "item_idx").join(user_bias, "user_idx")
                   .withColumn("target", F.col("rating") - (F.lit(mu) + F.col("b_u") + F.col("b_i"))))
    else:
        mu, item_bias, user_bias, bias_bytes = 0.0, None, None, 0
        ratings = part.withColumn("target", F.col("rating"))
    ratings = ratings.select("user_idx", "item_idx", "client_id", "target").localCheckpoint()

    client_items = ratings.select("client_id", "item_idx").distinct()
    n_client_items = client_items.count()
    round_bytes = n_client_items * (2 * rank * FLOAT32_BYTES + INT32_BYTES)

    sc = spark.sparkContext
    n_ratings = ratings.count()
    parts = max(1, int(sc.defaultParallelism) * int(fed["partitions_per_core"]),
                -(-n_ratings // int(fed["rows_per_partition"])))
    ident = make_identity()
    C = int(num_clients)
    # MEMORY_AND_DISK: under memory pressure these spill instead of being evicted and recomputed
    blocks = build_user_blocks(ratings, parts, ident).persist(StorageLevel.MEMORY_AND_DISK)
    n_users = blocks.map(lambda kv: len(kv[1][0])).sum()
    routes = build_routes(blocks, parts, ident, C, per_client=False).persist(StorageLevel.MEMORY_AND_DISK)
    n_routes = routes.map(lambda kv: len(kv[1][0])).sum()
    routes_c = (build_routes(blocks, parts, ident, C, per_client=True).persist(StorageLevel.MEMORY_AND_DISK)
                if epochs > 1 else None)
    v = init_item_factors(routes, rank, float(fed["init_std"]), int(config["seed"])) \
        .persist(StorageLevel.MEMORY_AND_DISK)
    n_items = v.map(lambda kv: len(kv[1][0])).sum()
    _log(f"{n_users:,} users, {n_ratings:,} ratings in {parts} partitions; V = {n_items:,} items x {rank}, "
         f"sharded by item ({n_routes:,} item rows delivered per round); vectorised array blocks")

    saved = {}

    def evaluate(v_cur):
        u_rdd = solve_users(blocks, deliver(v_cur, routes, ident, parts), reg, rank, parts)
        u_df = spark.createDataFrame(u_rdd, "user_idx int, features array<double>")
        v_rows = v_cur.flatMap(lambda kv: [(int(i), [float(x) for x in row])
                                           for i, row in zip(kv[1][0].tolist(), kv[1][1])])
        v_df = spark.createDataFrame(v_rows, "item_idx int, features array<double>")
        saved["u"], saved["v"] = u_df, v_df          # the last evaluation's factors = final model
        return _eval(score_test(test_b, u_df, v_df, mu, user_bias, item_bias, config))

    rounds = []
    eval_time = 0.0                       # excluded from train_time_sec (SPEC.md Section 4)
    te = time.time()
    r0 = _eval(score_test(test_b, None, None, mu, user_bias, item_bias, config)) if eval_every else {}
    eval_time += time.time() - te
    rounds.append({"round": 0, "rmse": r0.get("rmse"), "rmse_cold": r0.get("rmse_cold"),
                   "rmse_warm": r0.get("rmse_warm"), "bytes_transferred": int(bias_bytes)})
    _log(f"round 0 (bias exchange, {bias_bytes:,} bytes)"
         + (f": RMSE {r0['rmse']:.4f} (cold {r0['rmse_cold']:.4f}, warm {r0['rmse_warm']:.4f})" if r0 else ""))

    final = None
    for rnd in range(1, rounds_n + 1):
        t_round = time.time()
        cur, cur_routes, local = v, routes, False
        for epoch in range(1, epochs + 1):
            rows = deliver(cur, cur_routes, ident, parts)                        # download
            emitted = user_pass(blocks, rows, reg, rank, parts, C, local)        # user step, local
            last = epoch == epochs
            cur = item_pass(v, emitted, ident, reg, rank, parts, C, average=last)  # upload + server
            cur_routes, local = routes_c, True                                   # next epoch: client-local
        new_v = cur.persist(StorageLevel.MEMORY_AND_DISK)
        new_v.localCheckpoint()                                    # cut lineage: depends only on itself
        new_v.count()                                              # materialize this round's V
        v.unpersist(blocking=True)                                 # free the old V before the next round
        v = new_v
        del rows, emitted, cur, new_v
        _release_round_garbage(sc)                                 # old shuffle files deleted now, not later
        _log(f"round {rnd}/{rounds_n} trained in {time.time() - t_round:.1f}s")
        row = {"round": rnd, "rmse": None, "rmse_cold": None, "rmse_warm": None,
               "bytes_transferred": int(round_bytes)}
        if eval_every or rnd == rounds_n:
            te = time.time()
            m = evaluate(v)
            row.update({"rmse": m["rmse"], "rmse_cold": m["rmse_cold"], "rmse_warm": m["rmse_warm"]})
            final = m
            eval_time += time.time() - te
            _log(f"round {rnd}: RMSE {m['rmse']:.4f} (cold {m['rmse_cold']:.4f}, warm {m['rmse_warm']:.4f}), "
                 f"{round_bytes:,} bytes")
        rounds.append(row)
    train_time = time.time() - t0 - eval_time

    if save_model_to:                     # after the timer: never part of train_time_sec
        _save_federated_model(save_model_to, saved["u"], saved["v"], mu, user_bias, item_bias,
                              config, int(num_clients), rounds_n, residual)
        _log(f"federated model bundle ({num_clients} clients) written to {save_model_to}")
    # release only after the model is saved: the saved tables are computed from V
    blocks.unpersist(); routes.unpersist(); v.unpersist()
    if routes_c is not None:
        routes_c.unpersist()
    dropped = n_test - final["rows"]
    total_bytes = sum(r["bytes_transferred"] for r in rounds)
    _log(f"{num_clients} clients: training {train_time:.1f}s (+ {eval_time:.1f}s evaluation, not counted); total {total_bytes:,} bytes; "
         f"test rows dropped (item not in train): {dropped:,}")
    part.unpersist(); test_b.unpersist()
    return {"num_clients": int(num_clients), "residual_on_bias": residual, "test_rows_dropped": int(dropped),
            "rounds": rounds, "final_rmse": final["rmse"], "final_mae": final["mae"],
            "final_rmse_cold": final["rmse_cold"], "final_rmse_warm": final["rmse_warm"],
            "total_bytes": int(total_bytes), "train_time_sec": train_time}


# ---------------------------------------------------------------- experiment driver
def _read_json(path: str) -> dict:
    try:
        return json.loads(read_text(path))
    except Exception as e:  # noqa: BLE001 - surface a clear dependency error
        raise FileNotFoundError(
            f"Required Module 2 output not found or unreadable: {path}. Run "
            f"scripts/run_centralized.py with the same config first (SPEC.md Section 6).") from e


def run_federated_experiment(config: dict, client_counts: list = None) -> dict:
    results = config["data"]["results_prefix"]
    central = _read_json(_join(results, "centralized_metrics.json"))
    coldwarm = _read_json(_join(results, "coldstart_metrics.json"))
    if bool(central.get("residual_on_bias")) != bool(config["als"]["residual_on_bias"]):
        raise ValueError("centralized_metrics.json was produced with a different als.residual_on_bias; "
                         "rerun Module 2 with this config so the comparison is like-for-like")

    lines = ["num_clients,round,rmse,rmse_cold,rmse_warm,bytes_transferred,cumulative_bytes"]
    fed = {}
    model_n = int(config["federated"]["model_num_clients"])
    model_path = _join(results, "models/federated")
    for n in (client_counts or config["federated"]["num_clients"]):
        res = run_federated_training(config, int(n), save_model_to=model_path if int(n) == model_n else None)
        cum = 0
        for r in res["rounds"]:
            cum += r["bytes_transferred"]
            vals = ["" if r[k] is None else f"{r[k]:.6f}" for k in ("rmse", "rmse_cold", "rmse_warm")]
            lines.append(f"{n},{r['round']},{','.join(vals)},{r['bytes_transferred']},{cum}")
        fed[str(n)] = {k: res[k] for k in ("final_rmse", "final_rmse_cold", "final_rmse_warm",
                                           "final_mae", "total_bytes")}
        fed[str(n)]["rounds"] = len(res["rounds"]) - 1

    comparison = {
        "centralized": {"rmse": central["rmse"], "rmse_cold": coldwarm["cold"]["rmse"],
                        "rmse_warm": coldwarm["warm"]["rmse"]},
        "federated": fed,
        "residual_on_bias": bool(config["als"]["residual_on_bias"]),
        "note": NOTE,
    }
    write_text(_join(results, "federated_convergence.csv"), "\n".join(lines) + "\n")
    write_json(_join(results, "comparison.json"), comparison)
    _log(f"centralized RMSE {central['rmse']:.4f} (cold {coldwarm['cold']['rmse']:.4f}, "
         f"warm {coldwarm['warm']['rmse']:.4f})")
    for n, m in fed.items():
        _log(f"federated {n} clients: RMSE {m['final_rmse']:.4f} (cold {m['final_rmse_cold']:.4f}, "
             f"warm {m['final_rmse_warm']:.4f}), {m['total_bytes']:,} bytes")
    _log(f"federated_convergence.csv and comparison.json written under {results}")
    return comparison
