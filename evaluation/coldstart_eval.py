"""Module 2 — cold vs. warm evaluation (SPEC.md Section 3, CLAUDE.md Rule 5).

Cold/warm is decided by each user's number of TRAIN ratings, never test ratings.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from evaluation.metrics import mae, precision_at_k, rmse

NOTE = (
    "Cold = fewer than {thr} train ratings; warm = {thr} or more (train counts only). "
    "precision@{k} is averaged over users with at least one relevant test item (rating >= {rel}); "
    "a user with only r relevant test items can score at most r/{k}, so cold users have a "
    "mechanically low precision ceiling — this is a property of the metric, not necessarily a "
    "model failure. The per-user temporal split gives users with 2-4 ratings a larger test share "
    "(a 2-rating user is split 1/1), so the overall train/test ratio is below the configured "
    "{split}. Test rows whose user or item never appears in train cannot be scored by ALS and "
    "are excluded (see test_rows_dropped); baselines are evaluated on the same scored rows."
)


def user_buckets(train_df: DataFrame, cold_threshold: int) -> DataFrame:
    """(user_idx, bucket) with bucket in {'cold', 'warm'} from train-set counts."""
    return (train_df.groupBy("user_idx").agg(F.count(F.lit(1)).alias("n_train"))
            .select("user_idx",
                    F.when(F.col("n_train") < cold_threshold, F.lit("cold"))
                     .otherwise(F.lit("warm")).alias("bucket")))


def evaluate_coldstart(predictions_df: DataFrame, train_df: DataFrame, test_df: DataFrame,
                       recommendations_df: DataFrame, config: dict) -> dict:
    ev = config["evaluation"]
    k = int(ev["precision_at_k"])
    rel = float(ev["relevance_threshold"])
    thr = int(ev["cold_user_threshold"])

    buckets = user_buckets(train_df, thr).cache()
    preds = predictions_df.join(buckets, "user_idx").cache()
    test_b = test_df.join(buckets, "user_idx")
    recs_b = recommendations_df.join(buckets, "user_idx")

    result = {}
    for bucket in ("cold", "warm"):
        p = preds.where(F.col("bucket") == bucket)
        prec, prec_users = precision_at_k(recs_b.where(F.col("bucket") == bucket),
                                          test_b.where(F.col("bucket") == bucket), k, rel)
        result[bucket] = {
            "n_users": p.select("user_idx").distinct().count(),
            "n_test_rows": p.count(),
            "rmse": rmse(p, "prediction"),
            "mae": mae(p, "prediction"),
            "precision_at_k": prec,
            "precision_users": prec_users,
            "baselines": {
                "global_mean": {"rmse": rmse(p, "pred_global_mean"), "mae": mae(p, "pred_global_mean")},
                "bias": {"rmse": rmse(p, "pred_bias"), "mae": mae(p, "pred_bias")},
            },
        }
    result["note"] = NOTE.format(thr=thr, k=k, rel=rel, split=config["data"]["train_test_split"])
    preds.unpersist()
    buckets.unpersist()
    return result
