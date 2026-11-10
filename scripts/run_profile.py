"""Read-only dataset profile for the notebooks' data-engineering charts (SPEC.md Section 7).

Reads Module 1's inputs and outputs, changes nothing, and writes ONE small JSON:
data.results_prefix/data_profile.json. All work is Spark aggregations (nothing per-row reaches the
driver), so it scales with the cluster like the ETL itself.

  spark-submit scripts/run_profile.py --config config.yaml           (Dataproc: --config gs://...)

Contents:
  funnel        raw rows -> after cleaning -> distinct (user, item) pairs -> final (train + test)
                The two middle numbers reuse Module 1's own load_raw/clean functions, so they are
                the same rules the ETL applied (dedup keeps one row per pair = distinct pairs).
  per_category  raw rows, final rows, users and items per configured category
  totals        users, items, train/test rows, density = ratings / (users x items)
  ratings       count per star value (train + test)
  user_activity number of users by training-rating count bucket, plus cold / warm users
  item_activity number of items by training-rating count bucket
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import read_text, write_json  # noqa: E402

REQUIRED_KEYS = ["data.raw_parquet_prefix", "data.processed_prefix", "data.results_prefix",
                 "data.categories", "evaluation.cold_user_threshold"]
BUCKETS = [1, 2, 3, 4, 5, 10, 20, 50, 100, 1000]   # lower edges of the activity buckets


def load_config(path: str) -> dict:
    config = yaml.safe_load(read_text(path)) or {}
    missing = []
    for dotted in REQUIRED_KEYS:
        node = config
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node or node[part] is None:
                missing.append(dotted)
                break
            node = node[part]
    if missing:
        sys.exit(f"Config error in {path}: missing required key(s): {', '.join(missing)}")
    return config


def _join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def _bucket_label(lo: int, hi) -> str:
    return f"{lo}" if hi == lo + 1 else (f"{lo}+" if hi is None else f"{lo}-{hi - 1}")


def _buckets(counts_df, col: str) -> list:
    from pyspark.sql import functions as F
    edges = BUCKETS
    expr = None
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        cond = (F.col(col) >= lo) if hi is None else ((F.col(col) >= lo) & (F.col(col) < hi))
        expr = F.when(cond, F.lit(i)) if expr is None else expr.when(cond, F.lit(i))
    rows = {r["b"]: r["n"] for r in counts_df.groupBy(expr.alias("b")).agg(F.count("*").alias("n")).collect()}
    out = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        out.append({"bucket": _bucket_label(lo, hi), "count": int(rows.get(i, 0))})
    return out


def profile(config: dict) -> dict:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from etl.build_rating_matrix import clean, load_raw   # Module 1's own rules, unchanged

    spark = SparkSession.builder.getOrCreate()
    d = config["data"]
    cats = list(d["categories"])
    p = d["processed_prefix"]

    raw = load_raw(spark, d["raw_parquet_prefix"], cats)
    raw_per_cat = {r["category"]: int(r["n"]) for r in raw.groupBy("category").agg(F.count("*").alias("n")).collect()}
    n_raw = sum(raw_per_cat.values())
    cleaned = clean(raw)
    n_clean = cleaned.count()
    n_pairs = cleaned.select("user_id", "item_id").distinct().count()

    train = spark.read.parquet(_join(p, "train.parquet")).withColumn("split", F.lit("train"))
    test = spark.read.parquet(_join(p, "test.parquet")).withColumn("split", F.lit("test"))
    item_cat = spark.read.parquet(_join(p, "item_category.parquet")).select("item_idx", "category")
    both = train.unionByName(test).join(item_cat, "item_idx").cache()
    n_train, n_test = train.count(), test.count()
    n_users = both.select("user_idx").distinct().count()
    n_items = both.select("item_idx").distinct().count()

    per_cat = {r["category"]: {"final_rows": int(r["rows"]), "users": int(r["users"]), "items": int(r["items"])}
               for r in both.groupBy("category").agg(F.count("*").alias("rows"),
                                                     F.countDistinct("user_idx").alias("users"),
                                                     F.countDistinct("item_idx").alias("items")).collect()}
    for c in cats:
        per_cat.setdefault(c, {"final_rows": 0, "users": 0, "items": 0})["raw_rows"] = raw_per_cat.get(c, 0)

    ratings = {str(float(r["rating"])): int(r["n"])
               for r in both.groupBy("rating").agg(F.count("*").alias("n")).orderBy("rating").collect()}
    user_counts = train.groupBy("user_idx").agg(F.count("*").alias("n"))
    item_counts = train.groupBy("item_idx").agg(F.count("*").alias("n"))
    thr = int(config["evaluation"]["cold_user_threshold"])
    cold = user_counts.where(F.col("n") < thr).count()
    warm = user_counts.where(F.col("n") >= thr).count()

    result = {
        "categories": cats,
        "funnel": [{"stage": "raw rows", "rows": n_raw},
                   {"stage": "after cleaning", "rows": n_clean},
                   {"stage": "distinct (user, item)", "rows": n_pairs},
                   {"stage": "after min-interaction filter (train + test)", "rows": n_train + n_test},
                   {"stage": "train", "rows": n_train},
                   {"stage": "test", "rows": n_test}],
        "per_category": per_cat,
        "totals": {"users": n_users, "items": n_items, "train_rows": n_train, "test_rows": n_test,
                   "density": (n_train + n_test) / float(n_users * n_items) if n_users and n_items else None},
        "ratings": ratings,
        "user_activity": _buckets(user_counts, "n"),
        "item_activity": _buckets(item_counts, "n"),
        "cold_warm_users": {"cold": int(cold), "warm": int(warm), "threshold": thr},
    }
    both.unpersist()
    out = _join(d["results_prefix"], "data_profile.json")
    write_json(out, result)
    print(f"[profile] raw {n_raw:,} -> clean {n_clean:,} -> distinct pairs {n_pairs:,} -> final "
          f"{n_train + n_test:,} (train {n_train:,}, test {n_test:,}); users {n_users:,}, items {n_items:,}; "
          f"written {out}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only dataset profile (data_profile.json)")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    profile(load_config(args.config))


if __name__ == "__main__":
    main()
