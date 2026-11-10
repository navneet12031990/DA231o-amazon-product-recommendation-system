"""Module 1 — one-time ratings-only CSV -> Parquet conversion (SPEC.md Section 2).

Never called inside a scaling-benchmark timer.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, LongType, StringType, StructField, StructType

# The UCSD ratings-only CSVs are headerless and put the ITEM first: item, user, rating, timestamp.
RAW_CSV_SCHEMA = StructType([
    StructField("item_id", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("rating", FloatType(), True),
    StructField("timestamp", LongType(), True),
])


def _log(msg: str) -> None:
    print(f"[etl.convert] {msg}", flush=True)


def _join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def path_exists(spark: SparkSession, path: str) -> bool:
    """Existence check that works for local, Drive and s3:// paths (via Hadoop FileSystem)."""
    jpath = spark._jvm.org.apache.hadoop.fs.Path(path)
    return bool(jpath.getFileSystem(spark._jsc.hadoopConfiguration()).exists(jpath))


def read_category_csv(spark: SparkSession, raw_prefix: str, category: str) -> DataFrame:
    """Read one category's CSV with the explicit schema and tag it with its category."""
    path = _join(raw_prefix, f"{category}.csv")
    df = (spark.read
          .schema(RAW_CSV_SCHEMA)
          .option("header", "false")
          .option("mode", "PERMISSIVE")   # unparseable fields become null; ETL drops them later
          .csv(path))
    return df.withColumn("category", F.lit(category))


def convert_to_parquet(config: dict) -> None:
    """Convert each configured category's CSV to Parquet under data.raw_parquet_prefix,
    partitioned by category. Re-running overwrites only the categories being converted."""
    spark = SparkSession.builder.getOrCreate()
    # Dynamic mode: overwriting category=Books leaves already-converted categories untouched.
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    data_cfg = config["data"]
    raw_prefix = data_cfg["raw_prefix"]
    out_prefix = data_cfg["raw_parquet_prefix"]
    categories = data_cfg["categories"]

    missing = [c for c in categories if not path_exists(spark, _join(raw_prefix, f"{c}.csv"))]
    if missing:
        raise SystemExit(f"[etl.convert] No CSV found under {raw_prefix} for: {', '.join(missing)}. "
                         f"Download <Category>.csv for these or fix data.categories.")

    for category in categories:
        df = read_category_csv(spark, raw_prefix, category)
        _log(f"{category}: sample parsed rows (check item_id looks like an ASIN, "
             f"user_id like a reviewer ID, rating in 1-5):")
        for row in df.limit(3).collect():
            _log(f"  {row.asDict()}")
        df.write.mode("overwrite").partitionBy("category").parquet(out_prefix)
        _log(f"{category}: written to {out_prefix}")

    # Post-write summary from the Parquet itself (cheap: columnar read).
    written = spark.read.parquet(out_prefix).where(F.col("category").isin(categories))
    summary = (written.groupBy("category")
               .agg(F.count(F.lit(1)).alias("rows"),
                    F.sum(F.col("rating").isNull().cast("long")).alias("null_ratings"),
                    F.sum(F.col("timestamp").isNull().cast("long")).alias("null_timestamps"))
               .orderBy("category")
               .collect())
    for r in summary:
        _log(f"{r['category']}: rows={r['rows']:,} null_ratings={r['null_ratings']:,} "
             f"null_timestamps={r['null_timestamps']:,}")
