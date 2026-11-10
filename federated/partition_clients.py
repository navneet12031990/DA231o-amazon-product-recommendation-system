"""Module 3 — client partitioning (SPEC.md Section 4)."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def partition_clients(train_df: DataFrame, num_clients: int, seed: int) -> DataFrame:
    """Assign each user to one of num_clients cohorts (random_cohort: a seeded hash of user_idx,
    so assignment is deterministic and independent of data order or partitioning).
    Returns train_df with an added client_id (int) column."""
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    h = F.xxhash64(F.col("user_idx"), F.lit(int(seed)))
    return train_df.withColumn("client_id", F.pmod(h, F.lit(int(num_clients))).cast("int"))
