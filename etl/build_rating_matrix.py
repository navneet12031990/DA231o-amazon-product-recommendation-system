"""Module 1 — build the rating matrix from the raw Parquet (SPEC.md Section 2).

Reads data.raw_parquet_prefix, writes train.parquet, test.parquet, user_id_mapping.parquet,
item_id_mapping.parquet and item_category.parquet to data.processed_prefix.

Data flow (built for scaling, SPEC.md Section 2): the cleaned ratings are hash-partitioned by
user_id once. Dedup, user counts and the temporal split are windows over user_id on that
partitioning, so they move no ratings. Each filter pass shuffles only partial per-item counts and
applies the kept-item list by broadcast. Item ids are attached by broadcast; the user mapping is
partitioned like the ratings.
"""
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

INT_MAX = 2**31 - 1  # ALS requires user/item ids that fit in a 32-bit int


def _log(msg: str) -> None:
    print(f"[etl.build] {msg}", flush=True)


def _join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def load_raw(spark: SparkSession, raw_parquet_prefix: str, categories: list) -> DataFrame:
    """Read the converted Parquet, keeping only the configured categories. Fails fast if any
    configured category has not been converted yet."""
    from etl.convert_to_parquet import path_exists
    missing = [c for c in categories if not path_exists(spark, _join(raw_parquet_prefix, f"category={c}"))]
    if missing:
        raise SystemExit(f"[etl.build] Not yet converted to Parquet: {', '.join(missing)}. "
                         f"Run scripts/run_etl.py --step convert with this config first.")
    df = spark.read.parquet(raw_parquet_prefix).where(F.col("category").isin(categories))
    for r in df.groupBy("category").count().orderBy("category").collect():
        _log(f"raw rows {r['category']}: {r['count']:,}")
    return df


def clean(df: DataFrame) -> DataFrame:
    """Drop nulls, empty IDs, ratings outside [1, 5] and non-positive timestamps."""
    valid = (
        F.col("user_id").isNotNull() & (F.trim(F.col("user_id")) != "")
        & F.col("item_id").isNotNull() & (F.trim(F.col("item_id")) != "")
        & F.col("rating").isNotNull() & F.col("rating").between(1.0, 5.0)
        & F.col("timestamp").isNotNull() & (F.col("timestamp") > 0)
    )
    return df.where(valid).select("user_id", "item_id", "rating", "timestamp", "category")


def deduplicate(df: DataFrame) -> DataFrame:
    """Keep one row per (user_id, item_id): the most recent timestamp.
    Ties (same day — Amazon timestamps are day-granular) are broken deterministically, by rating
    and then category (the same pair can appear in two category files)."""
    w = Window.partitionBy("user_id", "item_id").orderBy(
        F.col("timestamp").desc(), F.col("rating").desc(), F.col("category").asc())
    return (df.withColumn("_rn", F.row_number().over(w))
              .where(F.col("_rn") == 1)
              .drop("_rn"))


def filter_min_interactions(df: DataFrame, min_user: int, min_item: int, num_partitions: int,
                            broadcast_max_items: int) -> DataFrame:
    """Iteratively drop users with < min_user and items with < min_item interactions until a
    full pass removes nothing (dropping items can push users below threshold and vice versa).

    df must be hash-partitioned by user_id. The user filter is a window count over user_id (no
    shuffle); the item filter shuffles only partial per-item counts and applies the kept-item list
    by broadcast, so no pass moves the ratings themselves."""
    rows = df.count()
    n_pass = 0
    w_user = Window.partitionBy("user_id")
    while True:
        n_pass += 1
        df_u = (df.withColumn("_n", F.count(F.lit(1)).over(w_user))
                  .where(F.col("_n") >= min_user).drop("_n")
                  .localCheckpoint())
        rows_after_user = df_u.count()

        keep_items = (df_u.groupBy("item_id").agg(F.count(F.lit(1)).alias("_n"))
                          .where(F.col("_n") >= min_item).select("item_id")
                          .localCheckpoint())
        n_keep = keep_items.count()
        if n_keep <= broadcast_max_items:
            df_i = df_u.join(F.broadcast(keep_items), "item_id", "left_semi")
        else:  # too many items to broadcast: plain join, then restore the user partitioning
            _log(f"filter pass {n_pass}: {n_keep:,} items > broadcast_max_items, using a shuffle join")
            df_i = (df_u.join(keep_items, "item_id", "left_semi")
                        .repartition(num_partitions, "user_id"))
        df_i = df_i.localCheckpoint()
        rows_after_item = df_i.count()

        _log(f"filter pass {n_pass}: rows {rows:,} -> {rows_after_user:,} (user filter) "
             f"-> {rows_after_item:,} (item filter)")
        df = df_i
        if rows_after_item == rows:    # nothing removed by either side: stable
            break
        rows = rows_after_item
    return df


def build_id_mapping(df: DataFrame, id_col: str, idx_col: str) -> DataFrame:
    """Deterministic string -> int mapping: sort distinct ids, number them 0..n-1.
    Uses zipWithIndex (distributed) rather than a global window (single partition)."""
    distinct_sorted = df.select(id_col).distinct().orderBy(id_col)
    mapping = (distinct_sorted.rdd.zipWithIndex().toDF(["_row", "_idx"])
               .select(F.col(f"_row.{id_col}").alias(id_col),
                       F.col("_idx").cast("int").alias(idx_col)))
    return mapping


def temporal_split(df: DataFrame, train_fraction: float) -> tuple:
    """Per-user temporal split: each user's oldest floor(n * fraction) interactions go to train,
    the rest to test, clamped so every user (n >= 2) has >= 1 row in each split.
    Ties on timestamp are broken by item_id so the split is deterministic."""
    # Windows over user_id (1:1 with user_idx) reuse the ratings' user partitioning: no shuffle.
    w_order = Window.partitionBy("user_id").orderBy(F.col("timestamp").asc(), F.col("item_id").asc())
    w_user = Window.partitionBy("user_id")
    ranked = (df.withColumn("_rank", F.row_number().over(w_order))
                .withColumn("_n", F.count(F.lit(1)).over(w_user)))
    n_train = F.least(F.greatest(F.floor(F.col("_n") * F.lit(train_fraction)), F.lit(1)),
                      F.col("_n") - 1)
    ranked = ranked.withColumn("_is_train", F.col("_rank") <= n_train)
    cols = ["user_id", "item_id", "user_idx", "item_idx", "rating", "timestamp"]
    train = ranked.where(F.col("_is_train")).select(*cols)
    test = ranked.where(~F.col("_is_train")).select(*cols)
    return train, test


def build_item_category(filtered: DataFrame, item_map: DataFrame) -> DataFrame:
    """One category per final item: the category with the most ratings for that item
    (ties broken alphabetically). Returns (item_idx, item_id, category)."""
    counts = filtered.groupBy("item_id", "category").agg(F.count(F.lit(1)).alias("_n"))
    w = Window.partitionBy("item_id").orderBy(F.col("_n").desc(), F.col("category").asc())
    best = (counts.withColumn("_r", F.row_number().over(w)).where(F.col("_r") == 1)
            .select("item_id", "category"))
    multi = counts.groupBy("item_id").count().where(F.col("count") > 1).count()
    _log(f"items found in more than one category file: {multi:,} (assigned to their majority category)")
    return item_map.join(best, "item_id").select("item_idx", "item_id", "category")


def build_rating_matrix(config: dict) -> None:
    """Runs the build with adaptive query execution (AQE) switched off, then restores the previous
    setting. With AQE on, a plan's partitioning is unknown until it runs, so localCheckpoint()
    would forget the user_id partitioning and every window would reshuffle the ratings. The build
    sets its own partition count, so AQE's partition coalescing is not needed here."""
    spark = SparkSession.builder.getOrCreate()
    previous = spark.conf.get("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    try:
        _build(spark, config)
    finally:
        spark.conf.set("spark.sql.adaptive.enabled", previous)


def _build(spark: SparkSession, config: dict) -> None:
    data_cfg = config["data"]
    min_user = int(data_cfg["min_user_interactions"])
    min_item = int(data_cfg["min_item_interactions"])
    bmax = int(data_cfg["broadcast_max_items"])
    out = data_cfg["processed_prefix"]
    parts = max(1, int(spark.sparkContext.defaultParallelism) * int(data_cfg["shuffle_partitions_per_core"]))
    spark.conf.set("spark.sql.shuffle.partitions", str(parts))
    _log(f"{parts} shuffle partitions ({spark.sparkContext.defaultParallelism} cores x "
         f"{data_cfg['shuffle_partitions_per_core']})")

    raw = load_raw(spark, data_cfg["raw_parquet_prefix"], data_cfg["categories"])
    n_raw = raw.count()

    # The one shuffle of the ratings: everything after this is partitioned by user_id.
    cleaned = clean(raw).repartition(parts, "user_id").localCheckpoint()
    n_clean = cleaned.count()
    _log(f"malformed rows dropped: {n_raw - n_clean:,} ({n_raw:,} -> {n_clean:,})")

    deduped = deduplicate(cleaned).localCheckpoint()
    n_dedup = deduped.count()
    _log(f"duplicate (user, item) rows removed: {n_clean - n_dedup:,} ({n_clean:,} -> {n_dedup:,})")

    users_before = deduped.select("user_id").distinct().count()
    items_before = deduped.select("item_id").distinct().count()

    filtered = filter_min_interactions(deduped, min_user, min_item, parts, bmax)

    user_map = build_id_mapping(filtered, "user_id", "user_idx").localCheckpoint()
    item_map = build_id_mapping(filtered, "item_id", "item_idx").localCheckpoint()
    n_users, n_items = user_map.count(), item_map.count()
    if n_users > INT_MAX or n_items > INT_MAX:
        raise ValueError(f"Too many ids for ALS int indices: users={n_users}, items={n_items}")
    _log(f"users {users_before:,} -> {n_users:,} (dropped {users_before - n_users:,}); "
         f"items {items_before:,} -> {n_items:,} (dropped {items_before - n_items:,})")

    # User ids: the mapping is partitioned like the ratings, so the join moves only the mapping.
    user_map_part = user_map.repartition(parts, "user_id")
    item_side = F.broadcast(item_map) if n_items <= bmax else item_map
    indexed = (filtered.join(user_map_part, "user_id").join(item_side, "item_id")
               .select("user_id", "item_id", "user_idx", "item_idx",
                       F.col("rating").cast("float"), F.col("timestamp").cast("long")))

    train, test = temporal_split(indexed, float(data_cfg["train_test_split"]))
    item_cat = build_item_category(filtered, item_map)

    user_map.write.mode("overwrite").parquet(_join(out, "user_id_mapping.parquet"))
    item_map.write.mode("overwrite").parquet(_join(out, "item_id_mapping.parquet"))
    item_cat.write.mode("overwrite").parquet(_join(out, "item_category.parquet"))
    train.write.mode("overwrite").parquet(_join(out, "train.parquet"))
    test.write.mode("overwrite").parquet(_join(out, "test.parquet"))

    # Counts from the written files, so the log reflects exactly what downstream modules read.
    n_train = spark.read.parquet(_join(out, "train.parquet")).count()
    n_test = spark.read.parquet(_join(out, "test.parquet")).count()
    _log(f"final rows: train={n_train:,} test={n_test:,} total={n_train + n_test:,} "
         f"(written to {out})")
    per_cat = (spark.read.parquet(_join(out, "train.parquet"))
               .unionByName(spark.read.parquet(_join(out, "test.parquet")))
               .join(spark.read.parquet(_join(out, "item_category.parquet")).select("item_idx", "category"),
                     "item_idx")
               .groupBy("category").count().orderBy("category").collect())
    for r in per_cat:
        _log(f"final rows {r['category']}: {r['count']:,}")
