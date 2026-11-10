"""Module 2 — centralized recommender: user/item bias baseline + ALS on residuals (SPEC.md Section 3).

With als.residual_on_bias: false the model is plain ALS on raw ratings (kept for the report).
"""
import json
import time

import numpy as np
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from evaluation.coldstart_eval import evaluate_coldstart, user_buckets
from evaluation.metrics import mae, read_text, rmse, write_json

# Memory guard for top-k scoring: users per block = this / n_items. Not a model parameter —
# it changes only how the work is chunked, never the result.
SCORE_BLOCK_ELEMENTS = 20_000_000


def _log(msg: str) -> None:
    print(f"[centralized] {msg}", flush=True)


def _join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def load_split(spark: SparkSession, config: dict) -> tuple:
    p = config["data"]["processed_prefix"]
    return (spark.read.parquet(_join(p, "train.parquet")),
            spark.read.parquet(_join(p, "test.parquet")))


def _clip(col, config: dict):
    a = config["als"]
    return F.least(F.greatest(col, F.lit(float(a["prediction_min"]))), F.lit(float(a["prediction_max"])))


# ---------------------------------------------------------------- bias baseline
def fit_bias_baseline(train_df: DataFrame, damping: float) -> tuple:
    """mu, item biases, user biases (damped means of residuals), from train only."""
    mu = float(train_df.agg(F.avg("rating")).first()[0])
    item_bias = (train_df.groupBy("item_idx")
                 .agg((F.sum(F.col("rating") - mu) / (F.count(F.lit(1)) + damping)).alias("b_i")))
    user_bias = (train_df.join(item_bias, "item_idx")
                 .groupBy("user_idx")
                 .agg((F.sum(F.col("rating") - mu - F.col("b_i"))
                       / (F.count(F.lit(1)) + damping)).alias("b_u")))
    return mu, item_bias, user_bias


def _with_bias(df: DataFrame, mu: float, item_bias: DataFrame, user_bias: DataFrame) -> DataFrame:
    """Adds column _bias = mu + b_u + b_i (unknown user/item bias = 0)."""
    return (df.join(item_bias, "item_idx", "left").join(user_bias, "user_idx", "left")
            .withColumn("_bias", F.lit(mu) + F.coalesce(F.col("b_u"), F.lit(0.0))
                        + F.coalesce(F.col("b_i"), F.lit(0.0)))
            .drop("b_u", "b_i"))


# ---------------------------------------------------------------- model
def fit_model(train_df: DataFrame, config: dict, rank: int, reg_param: float) -> dict:
    """Fit the configured model; returns an in-memory bundle (same shape as load_centralized_model)."""
    a = config["als"]
    residual = bool(a["residual_on_bias"])
    mu, item_bias, user_bias = fit_bias_baseline(train_df, float(a["baseline_damping"]))
    item_bias, user_bias = item_bias.cache(), user_bias.cache()
    if residual:
        fit_df = (_with_bias(train_df, mu, item_bias, user_bias)
                  .withColumn("_target", F.col("rating") - F.col("_bias")))
    else:
        fit_df = train_df.withColumn("_target", F.col("rating"))
    als = ALS(userCol="user_idx", itemCol="item_idx", ratingCol="_target",
              rank=int(rank), maxIter=int(a["max_iter"]), regParam=float(reg_param),
              coldStartStrategy=a["cold_start_strategy"], seed=int(config["seed"]))
    model = als.fit(fit_df)
    model.userFactors.count()  # materialize the fit (keeps it inside any timer around this call)
    meta = {"model_type": "centralized", "mu": mu, "residual_on_bias": residual,
            "prediction_min": float(a["prediction_min"]), "prediction_max": float(a["prediction_max"]),
            "rank": int(rank), "reg_param": float(reg_param)}
    return {"als": model, "user_bias": user_bias, "item_bias": item_bias, "meta": meta}


def _dot(a: str, b: str):
    return F.aggregate(F.zip_with(F.col(a), F.col(b), lambda x, y: x * y), F.lit(0.0),
                       lambda acc, x: acc + x)


def predict_pairs(bundle: dict, pairs_df: DataFrame) -> DataFrame:
    """Adds a clipped 'prediction' column to (user_idx, item_idx, ...) rows, for a centralized or
    a federated bundle. Pairs with an unknown user or item are dropped (coldStartStrategy=drop
    semantics for both)."""
    m = bundle["meta"]
    if m.get("model_type", "centralized") == "federated":
        scored = (pairs_df
                  .join(bundle["user_factors"].select("user_idx", F.col("features").alias("_uf")), "user_idx")
                  .join(bundle["item_factors"].select("item_idx", F.col("features").alias("_vf")), "item_idx")
                  .withColumn("prediction", _dot("_uf", "_vf")).drop("_uf", "_vf"))
    else:
        scored = bundle["als"].transform(pairs_df)  # 'prediction' = u . v (residual or raw)
    if m["residual_on_bias"]:
        scored = (_with_bias(scored, m["mu"], bundle["item_bias"], bundle["user_bias"])
                  .withColumn("prediction", F.col("prediction") + F.col("_bias")).drop("_bias"))
    lo, hi = F.lit(m["prediction_min"]), F.lit(m["prediction_max"])
    return scored.withColumn("prediction", F.least(F.greatest(F.col("prediction"), lo), hi))


def build_serving_tables(bundle: dict, user_map: DataFrame, item_map: DataFrame) -> dict:
    """Streaming: two cached tables keyed by RAW ids holding everything a prediction needs
    (users: user_id, user_idx, b_u, uf; items: item_id, item_idx, b_i, vf), built ONCE per scorer
    run from a saved bundle of either type. Users/items absent from the model are absent here."""
    m = bundle["meta"]
    if m.get("model_type", "centralized") == "federated":
        uf = bundle["user_factors"].select("user_idx", F.col("features").cast("array<double>").alias("uf"))
        vf = bundle["item_factors"].select("item_idx", F.col("features").cast("array<double>").alias("vf"))
    else:
        uf = bundle["als"].userFactors.select(F.col("id").alias("user_idx"),
                                             F.col("features").cast("array<double>").alias("uf"))
        vf = bundle["als"].itemFactors.select(F.col("id").alias("item_idx"),
                                             F.col("features").cast("array<double>").alias("vf"))
    users = user_map.select("user_id", "user_idx").join(uf, "user_idx")
    items = item_map.select("item_id", "item_idx").join(vf, "item_idx")
    residual = bool(m["residual_on_bias"])
    if residual:     # unknown bias = 0, exactly as predict_pairs
        users = (users.join(bundle["user_bias"], "user_idx", "left")
                      .withColumn("b_u", F.coalesce(F.col("b_u"), F.lit(0.0))))
        items = (items.join(bundle["item_bias"], "item_idx", "left")
                      .withColumn("b_i", F.coalesce(F.col("b_i"), F.lit(0.0))))
    else:
        users = users.withColumn("b_u", F.lit(0.0)); items = items.withColumn("b_i", F.lit(0.0))
    users = users.select("user_id", "user_idx", "b_u", "uf").cache()
    items = items.select("item_id", "item_idx", "b_i", "vf").cache()
    n_u, n_i = users.count(), items.count()
    return {"users": users, "items": items, "mu": float(m["mu"]) if residual else 0.0,
            "lo": float(m["prediction_min"]), "hi": float(m["prediction_max"]), "meta": m,
            "n_users": n_u, "n_items": n_i}


def predict_events(serving: dict, events_df: DataFrame) -> DataFrame:
    """Same prediction as predict_pairs, for a SMALL batch of events with raw user_id/item_id.
    Both joins broadcast the (small) batch side, so the large serving tables are scanned in place
    and never shuffled. Unknown ids drop out (inner joins = coldStartStrategy drop). Adds
    user_idx, item_idx and a clipped 'prediction'."""
    with_user = serving["users"].join(F.broadcast(events_df), "user_id")
    scored = (serving["items"].join(F.broadcast(with_user), "item_id")
              .withColumn("prediction", F.lit(serving["mu"]) + F.col("b_u") + F.col("b_i") + _dot("uf", "vf")))
    lo, hi = F.lit(serving["lo"]), F.lit(serving["hi"])
    return (scored.withColumn("prediction", F.least(F.greatest(F.col("prediction"), lo), hi))
                  .drop("uf", "vf", "b_u", "b_i"))


def save_centralized_model(bundle: dict, model_path: str) -> None:
    bundle["als"].write().overwrite().save(_join(model_path, "als"))
    bundle["user_bias"].write.mode("overwrite").parquet(_join(model_path, "user_bias.parquet"))
    bundle["item_bias"].write.mode("overwrite").parquet(_join(model_path, "item_bias.parquet"))
    write_json(_join(model_path, "meta.json"), bundle["meta"])


def load_model(model_path: str) -> dict:
    """Loads a centralized or federated model bundle (type from meta.json)."""
    spark = SparkSession.builder.getOrCreate()
    meta = json.loads(read_text(_join(model_path, "meta.json")))
    bundle = {"user_bias": spark.read.parquet(_join(model_path, "user_bias.parquet")),
              "item_bias": spark.read.parquet(_join(model_path, "item_bias.parquet")),
              "meta": meta}
    if meta.get("model_type", "centralized") == "federated":
        bundle["user_factors"] = spark.read.parquet(_join(model_path, "user_factors.parquet"))
        bundle["item_factors"] = spark.read.parquet(_join(model_path, "item_factors.parquet"))
    else:
        bundle["als"] = ALSModel.load(_join(model_path, "als"))
    return bundle


def load_centralized_model(model_path: str) -> dict:
    """Kept for compatibility; load_model handles both bundle types."""
    return load_model(model_path)


# ---------------------------------------------------------------- recommendations
def top_k_excluding_train(bundle: dict, test_df: DataFrame, train_df: DataFrame,
                          k: int, pool: int) -> tuple:
    """Rank every catalog item for each test user by b_i + u . v (mu and b_u are constant per
    user and cannot change the order; b_i = 0 for plain ALS), keep the top `pool`, drop items the
    user rated in train, cut to k. Scoring runs on executors in blocks of users against a
    broadcast item matrix. Returns (recs_df(user_idx, item_idx, score), n_users_short_of_k)."""
    spark = SparkSession.builder.getOrCreate()
    model, meta = bundle["als"], bundle["meta"]

    items = model.itemFactors.withColumnRenamed("id", "item_idx")
    if meta["residual_on_bias"]:
        items = items.join(bundle["item_bias"], "item_idx", "left")
    else:
        items = items.withColumn("b_i", F.lit(0.0))
    items_pdf = items.select("item_idx", "features", F.coalesce("b_i", F.lit(0.0)).alias("b_i")).toPandas()
    item_ids = items_pdf["item_idx"].to_numpy(dtype=np.int32)
    item_mat = np.vstack(items_pdf["features"].to_numpy()).astype(np.float32)
    item_b = items_pdf["b_i"].to_numpy(dtype=np.float32)
    bc = spark.sparkContext.broadcast((item_ids, item_mat, item_b))
    n_items = len(item_ids)
    top = int(min(pool, n_items))
    block = max(1, SCORE_BLOCK_ELEMENTS // max(n_items, 1))

    def score_partition(rows):
        # Plain RDD partition function (no Arrow/pandas dependency, so it runs on any JVM).
        import numpy as _np
        ids, mat, b = bc.value
        users, feats = [], []
        for r in rows:
            users.append(r[0]); feats.append(r[1])
            if len(users) == block:
                yield from _score(_np, ids, mat, b, users, feats)
                users, feats = [], []
        if users:
            yield from _score(_np, ids, mat, b, users, feats)

    def _score(_np, ids, mat, b, users, feats):
        U = _np.asarray(feats, dtype=_np.float32)
        S = U @ mat.T + b
        idx = _np.argpartition(-S, top - 1, axis=1)[:, :top]
        sc = _np.take_along_axis(S, idx, axis=1)
        for row_u, row_i, row_s in zip(users, ids[idx], sc):
            for i, sv in zip(row_i.tolist(), row_s.tolist()):
                yield (int(row_u), int(i), float(sv))

    test_users = test_df.select(F.col("user_idx").alias("id")).distinct()
    user_f = model.userFactors.join(test_users, "id").select("id", "features")
    raw = spark.createDataFrame(user_f.rdd.mapPartitions(score_partition),
                                "user_idx int, item_idx int, score double")
    unseen = raw.join(train_df.select("user_idx", "item_idx"), ["user_idx", "item_idx"], "left_anti")
    w = Window.partitionBy("user_idx").orderBy(F.col("score").desc(), F.col("item_idx"))
    recs = unseen.withColumn("_r", F.row_number().over(w)).where(F.col("_r") <= k).drop("_r").cache()
    short = recs.groupBy("user_idx").count().where(F.col("count") < k).count()
    return recs, short


# ---------------------------------------------------------------- breakdowns
def metrics_by_category(preds: DataFrame, item_cat: DataFrame) -> dict:
    """RMSE/MAE of the model and the bias baseline, grouped by the test item's category."""
    rows = (preds.join(item_cat.select("item_idx", "category"), "item_idx")
            .groupBy("category")
            .agg(F.count(F.lit(1)).alias("n"),
                 F.sqrt(F.avg(F.pow(F.col("prediction") - F.col("rating"), 2))).alias("rmse"),
                 F.avg(F.abs(F.col("prediction") - F.col("rating"))).alias("mae"),
                 F.sqrt(F.avg(F.pow(F.col("pred_bias") - F.col("rating"), 2))).alias("bias_rmse"),
                 F.avg(F.abs(F.col("pred_bias") - F.col("rating"))).alias("bias_mae"))
            .orderBy("category").collect())
    return {r["category"]: {"n_test_rows": int(r["n"]), "rmse": float(r["rmse"]), "mae": float(r["mae"]),
                            "bias_rmse": float(r["bias_rmse"]), "bias_mae": float(r["bias_mae"])}
            for r in rows}


def cross_category_users(train_df: DataFrame, item_cat: DataFrame) -> dict:
    """How many train users rated items in more than one category."""
    per_user = (train_df.join(item_cat.select("item_idx", "category"), "item_idx")
                .groupBy("user_idx").agg(F.countDistinct("category").alias("n_cat")))
    row = per_user.agg(F.count(F.lit(1)).alias("users"),
                       F.sum((F.col("n_cat") > 1).cast("long")).alias("multi")).first()
    users, multi = int(row["users"]), int(row["multi"] or 0)
    return {"count": multi, "share": (multi / users) if users else 0.0}


def sample_precision_users(test_df: DataFrame, train_df: DataFrame, config: dict) -> DataFrame:
    """At most evaluation.precision_user_sample test users per cold/warm bucket, chosen by a seeded
    hash of user_idx (same users every run). Returns (user_idx)."""
    n = int(config["evaluation"]["precision_user_sample"])
    buckets = user_buckets(train_df, int(config["evaluation"]["cold_user_threshold"]))
    users = test_df.select("user_idx").distinct().join(buckets, "user_idx")
    w = Window.partitionBy("bucket").orderBy(F.xxhash64(F.col("user_idx"), F.lit(int(config["seed"]))),
                                             F.col("user_idx"))
    return users.withColumn("_r", F.row_number().over(w)).where(F.col("_r") <= n).select("user_idx")


# ---------------------------------------------------------------- entry points
def train_centralized(config: dict) -> dict:
    spark = SparkSession.builder.getOrCreate()
    a, ev = config["als"], config["evaluation"]
    results = config["data"]["results_prefix"]
    train, test = load_split(spark, config)
    train = train.cache()
    n_train, n_test = train.count(), test.count()
    _log(f"train rows {n_train:,}, test rows {n_test:,}")

    t0 = time.time()
    bundle = fit_model(train, config, a["rank"], a["reg_param"])
    train_time = time.time() - t0
    mode = "bias + ALS on residuals" if a["residual_on_bias"] else "plain ALS"
    _log(f"fit {mode} (rank={a['rank']}, reg_param={a['reg_param']}) in {train_time:.1f}s")

    meta = bundle["meta"]
    preds = predict_pairs(bundle, test)
    preds = (_with_bias(preds, meta["mu"], bundle["item_bias"], bundle["user_bias"])
             .withColumn("pred_bias", _clip(F.col("_bias"), config))
             .withColumn("pred_global_mean", _clip(F.lit(meta["mu"]), config))
             .drop("_bias")).cache()
    n_scored = preds.count()
    dropped = n_test - n_scored
    _log(f"test rows scored {n_scored:,}; dropped by coldStartStrategy=drop: {dropped:,}")

    overall = {
        "rmse": rmse(preds, "prediction"), "mae": mae(preds, "prediction"),
        "train_time_sec": train_time, "test_rows_dropped": dropped,
        "residual_on_bias": bool(a["residual_on_bias"]),
        "baselines": {
            "global_mean": {"rmse": rmse(preds, "pred_global_mean"), "mae": mae(preds, "pred_global_mean")},
            "bias": {"rmse": rmse(preds, "pred_bias"), "mae": mae(preds, "pred_bias")},
        },
    }
    _log(f"overall: model RMSE {overall['rmse']:.4f} | bias {overall['baselines']['bias']['rmse']:.4f} "
         f"| global mean {overall['baselines']['global_mean']['rmse']:.4f}")

    item_cat = spark.read.parquet(_join(config["data"]["processed_prefix"], "item_category.parquet"))
    overall["by_category"] = metrics_by_category(preds, item_cat)
    for cat, m in overall["by_category"].items():
        _log(f"category {cat}: rows {m['n_test_rows']:,} | model RMSE {m['rmse']:.4f} | bias {m['bias_rmse']:.4f}")
    overall["cross_category_users"] = cross_category_users(train, item_cat)
    cc = overall["cross_category_users"]
    _log(f"train users rating in more than one category: {cc['count']:,} ({cc['share']:.2%})")

    sampled = sample_precision_users(test, train, config).cache()
    test_sample = test.join(sampled, "user_idx")
    _log(f"precision@k on {sampled.count():,} sampled test users "
         f"(at most {int(ev['precision_user_sample']):,} per cold/warm bucket)")
    recs, short = top_k_excluding_train(bundle, test_sample, train, int(ev["precision_at_k"]),
                                        int(ev["precision_candidate_pool"]))
    if short:
        _log(f"WARNING: {short:,} users had fewer than k unseen candidates; "
             f"consider raising evaluation.precision_candidate_pool")
    cold_warm = evaluate_coldstart(preds, train, test_sample, recs, config)
    for b in ("cold", "warm"):
        m = cold_warm[b]
        _log(f"{b}: users {m['n_users']:,} rows {m['n_test_rows']:,} | model RMSE {m['rmse']:.4f} "
             f"| bias {m['baselines']['bias']['rmse']:.4f} | global {m['baselines']['global_mean']['rmse']:.4f} "
             f"| P@{ev['precision_at_k']} {m['precision_at_k']:.4f} ({m['precision_users']:,} users)")

    save_centralized_model(bundle, _join(results, "models/centralized"))
    write_json(_join(results, "centralized_metrics.json"), overall)
    write_json(_join(results, "coldstart_metrics.json"), cold_warm)
    _log(f"model bundle and metrics written under {results}")
    recs.unpersist(); preds.unpersist(); train.unpersist(); sampled.unpersist()
    return overall


def split_train_for_validation(train_df: DataFrame, fraction: float) -> tuple:
    """Per-user temporal split of TRAIN into (sub_train, validation). Users with 1 train rating
    stay entirely in sub_train; others keep >= 1 row on each side."""
    w_order = Window.partitionBy("user_idx").orderBy(F.col("timestamp").asc(), F.col("item_id").asc())
    ranked = (train_df.withColumn("_rank", F.row_number().over(w_order))
              .withColumn("_n", F.count(F.lit(1)).over(Window.partitionBy("user_idx"))))
    n_keep = F.when(F.col("_n") < 2, F.col("_n")).otherwise(
        F.least(F.greatest(F.floor(F.col("_n") * F.lit(fraction)), F.lit(1)), F.col("_n") - 1))
    ranked = ranked.withColumn("_sub", F.col("_rank") <= n_keep)
    cols = train_df.columns
    return (ranked.where(F.col("_sub")).select(*cols),
            ranked.where(~F.col("_sub")).select(*cols))


def tune_als(config: dict) -> list:
    """One-time grid on a validation split carved from TRAIN. Never reads test.parquet."""
    spark = SparkSession.builder.getOrCreate()
    a = config["als"]
    train = spark.read.parquet(_join(config["data"]["processed_prefix"], "train.parquet"))
    sub_train, val = split_train_for_validation(train, float(a["tuning_train_fraction"]))
    sub_train, val = sub_train.cache(), val.cache()
    mode = "bias + ALS on residuals" if a["residual_on_bias"] else "plain ALS"
    _log(f"tuning {mode}: sub-train {sub_train.count():,} rows, validation {val.count():,} rows")
    buckets = user_buckets(sub_train, int(config["evaluation"]["cold_user_threshold"])).cache()

    rows = []
    for rank in a["tuning_ranks"]:
        for reg in a["tuning_reg_params"]:
            bundle = fit_model(sub_train, config, rank, reg)
            p = predict_pairs(bundle, val).join(buckets, "user_idx").cache()
            row = {"rank": int(rank), "reg_param": float(reg),
                   "val_rmse": rmse(p, "prediction"),
                   "val_rmse_cold": rmse(p.where(F.col("bucket") == "cold"), "prediction"),
                   "val_rmse_warm": rmse(p.where(F.col("bucket") == "warm"), "prediction")}
            p.unpersist(); bundle["user_bias"].unpersist(); bundle["item_bias"].unpersist()
            rows.append(row)
            _log(f"rank={rank:<3} reg={reg:<5} val RMSE {row['val_rmse']:.4f} "
                 f"(cold {row['val_rmse_cold']:.4f}, warm {row['val_rmse_warm']:.4f})")

    best = min(rows, key=lambda r: r["val_rmse"])
    _log(f"best: rank={best['rank']} reg_param={best['reg_param']} val RMSE {best['val_rmse']:.4f} "
         f"— copy into als.rank / als.reg_param in both config files if the team agrees")
    write_json(_join(config["data"]["results_prefix"], "als_tuning.json"),
               {"residual_on_bias": bool(a["residual_on_bias"]), "grid": rows, "best": best})
    return rows
