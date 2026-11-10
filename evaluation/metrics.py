"""Module 2 — metric functions (SPEC.md Section 3). All computed as Spark aggregations;
no rating data is collected to the driver."""
import json
import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def rmse(df: DataFrame, pred_col: str, label_col: str = "rating") -> float:
    """Root mean squared error; NaN if df is empty."""
    v = df.agg(F.sqrt(F.avg(F.pow(F.col(pred_col) - F.col(label_col), 2)))).first()[0]
    return float(v) if v is not None else float("nan")


def mae(df: DataFrame, pred_col: str, label_col: str = "rating") -> float:
    """Mean absolute error; NaN if df is empty."""
    v = df.agg(F.avg(F.abs(F.col(pred_col) - F.col(label_col)))).first()[0]
    return float(v) if v is not None else float("nan")


def precision_at_k(recommendations_df: DataFrame, test_df: DataFrame, k: int,
                   relevance_threshold: float) -> tuple:
    """precision@k = (# recommended items that are relevant test items) / k, averaged over users
    with >= 1 relevant test item. recommendations_df: (user_idx, item_idx), <= k rows per user.
    Returns (mean_precision, n_users)."""
    relevant = (test_df.where(F.col("rating") >= relevance_threshold)
                .select("user_idx", "item_idx").distinct())
    eligible_users = relevant.select("user_idx").distinct()
    hits = (recommendations_df.select("user_idx", "item_idx")
            .join(relevant, ["user_idx", "item_idx"])
            .groupBy("user_idx").agg(F.count(F.lit(1)).alias("hits")))
    per_user = (eligible_users.join(hits, "user_idx", "left")
                .select((F.coalesce(F.col("hits"), F.lit(0)) / F.lit(k)).alias("p")))
    row = per_user.agg(F.avg("p").alias("p"), F.count(F.lit(1)).alias("n")).first()
    p = float(row["p"]) if row["p"] is not None else float("nan")
    return p, int(row["n"])


def _has_scheme(path: str) -> bool:
    return "://" in path


def read_text(path: str) -> str:
    """Read a small text file from gs://, s3://, hdfs:// (via Spark's Hadoop layer) or a local path."""
    if _has_scheme(path):
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        return spark.read.text(path, wholetext=True).first()[0]
    with open(path) as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    """Write a small text file to gs://, s3://, hdfs:// (via Spark's Hadoop layer) or a local path."""
    if _has_scheme(path):
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        jvm = spark._jvm
        jpath = jvm.org.apache.hadoop.fs.Path(path)
        fs = jpath.getFileSystem(spark._jsc.hadoopConfiguration())
        out = fs.create(jpath, True)  # overwrite
        try:
            out.write(bytearray(text.encode("utf-8")))
        finally:
            out.close()
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


def write_json(path: str, obj: dict) -> None:
    """Write a small JSON result to gs://, s3://, hdfs:// or a local/Drive path."""
    write_text(path, json.dumps(obj, indent=2, default=float))
