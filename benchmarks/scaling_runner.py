"""Module 4 — scaling benchmarks, cluster side (SPEC.md Section 5).

Runs inside Dataproc jobs submitted by scripts/run_benchmarks.py. Every timed run is its own Spark
application with a fixed executor count; nothing here is timed except the region in timed_run.
"""
import copy
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

HASH_BUCKETS = 1_000_000_000  # resolution of the [0, 1) user-hash used for subset thresholds


def _log(msg: str) -> None:
    print(f"[scaling] {msg}", flush=True)


def _join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


# ---------------------------------------------------------------- targets
def sweep_targets(config: dict, sweep: str, unit: str) -> dict:
    """{executors: target_rows} for one sweep. unit 'train' (model sweeps) or 'raw' (ETL sweep)."""
    sc = config["scaling"]
    counts = [int(n) for n in sc["executor_counts"]]
    if unit == "train":
        strong, per_exec = int(sc["strong_fixed_train_rows"]), int(sc["weak_train_rows_per_executor"])
    else:
        strong, per_exec = int(sc["etl_strong_fixed_raw_rows"]), int(sc["etl_weak_raw_rows_per_executor"])
    if sweep == "strong":
        return {n: strong for n in counts}
    return {n: per_exec * n for n in counts}


def subset_path(config: dict, unit: str, target_rows: int) -> str:
    return _join(config["scaling"]["scratch_prefix"], f"subsets/{unit}_{int(target_rows)}")


# ---------------------------------------------------------------- capacity
def check_scaling_capacity(config: dict, units: list = None) -> dict:
    """Fail fast if the data can't supply the largest subset (never silently run a smaller one).
    units: which row units the requested modes need ("train" for centralized/federated, "raw" for
    etl); default = train, plus raw when the ETL sweep is enabled."""
    spark = SparkSession.builder.getOrCreate()
    if units is None:
        units = ["train"] + (["raw"] if bool(config["scaling"]["etl_sweep_enabled"]) else [])
    have, need = {}, {}
    for unit in units:
        need[unit] = max(max(sweep_targets(config, s, unit).values()) for s in ("strong", "weak"))
        if unit == "train":
            have[unit] = spark.read.parquet(_join(config["data"]["processed_prefix"], "train.parquet")).count()
        else:
            have[unit] = (spark.read.parquet(config["data"]["raw_parquet_prefix"])
                          .where(F.col("category").isin(config["data"]["categories"])).count())
        _log(f"capacity ({unit} rows): available {have[unit]:,}, largest target {need[unit]:,}")
        if have[unit] < need[unit]:
            raise SystemExit(
                f"[scaling] Not enough {unit} rows: {have[unit]:,} available, {need[unit]:,} needed. "
                f"Configure more categories or lower the scaling targets in the config.")
    return have


# ---------------------------------------------------------------- subsets
def _hash_fraction(col: str, seed: int):
    return F.pmod(F.xxhash64(F.col(col), F.lit(int(seed))), F.lit(HASH_BUCKETS)) / F.lit(float(HASH_BUCKETS))


def build_scaling_subset(config: dict, target_rows: int, unit: str) -> str:
    """Whole-user sample by hash threshold (SPEC.md Section 5). Untimed. Returns the subset path."""
    spark = SparkSession.builder.getOrCreate()
    seed = int(config["seed"])
    out = subset_path(config, unit, target_rows)
    if unit == "train":
        p = config["data"]["processed_prefix"]
        train = spark.read.parquet(_join(p, "train.parquet"))
        test = spark.read.parquet(_join(p, "test.parquet"))
        frac = min(1.0, float(target_rows) / float(train.count()))
        keep = train.select("user_idx").distinct().where(_hash_fraction("user_idx", seed) < frac)
        sub_train = train.join(keep, "user_idx", "left_semi")
        sub_train.write.mode("overwrite").parquet(_join(out, "train.parquet"))
        test.join(keep, "user_idx", "left_semi").write.mode("overwrite").parquet(_join(out, "test.parquet"))
        got = spark.read.parquet(_join(out, "train.parquet")).count()
    else:
        raw = (spark.read.parquet(config["data"]["raw_parquet_prefix"])
               .where(F.col("category").isin(config["data"]["categories"])))
        frac = min(1.0, float(target_rows) / float(raw.count()))
        sub = raw.where(_hash_fraction("user_id", seed) < frac)
        sub.write.mode("overwrite").partitionBy("category").parquet(out)
        got = spark.read.parquet(out).count()
    _log(f"subset {unit} target {target_rows:,}: {got:,} rows -> {out}")
    return out


def prepare_subsets(config: dict, modes: list, sweeps: list) -> None:
    units = sorted({"raw" if m == "etl" else "train" for m in modes})
    check_scaling_capacity(config, units)
    for unit in units:
        targets = sorted({t for s in sweeps for t in sweep_targets(config, s, unit).values()})
        for t in targets:
            build_scaling_subset(config, t, unit)


# ---------------------------------------------------------------- timing
def executors_registered(spark: SparkSession) -> int:
    """Executors currently registered (the driver's entry excluded)."""
    return max(0, spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size() - 1)


def wait_for_executors(spark: SparkSession, requested: int, timeout_sec: float) -> int:
    t0 = time.time()
    n = executors_registered(spark)
    while n < requested and time.time() - t0 < timeout_sec:
        time.sleep(2)
        n = executors_registered(spark)
    if n < requested:
        _log(f"WARNING: only {n} of {requested} executors registered after {timeout_sec:.0f}s; "
             f"timing anyway and recording executors_granted={n}")
    else:
        _log(f"{n} executors registered after {time.time() - t0:.1f}s")
    return n


def timed_run(config: dict, mode: str, subset: str, executors: int) -> dict:
    """Time exactly one region (SPEC.md Section 5 'Timed regions'); everything else is untimed."""
    spark = SparkSession.builder.getOrCreate()
    granted = wait_for_executors(spark, int(executors), float(config["scaling"]["executor_wait_sec"]))
    cfg = copy.deepcopy(config)
    scratch_out = _join(config["scaling"]["scratch_prefix"], f"runs/{mode}/{int(executors)}")

    if mode == "centralized":
        from centralized.train_als import fit_model
        train = spark.read.parquet(_join(subset, "train.parquet")).cache()
        t0 = time.time()
        rows = train.count()                       # one read of the subset, inside the timer
        fit_model(train, cfg, cfg["als"]["rank"], cfg["als"]["reg_param"])
        elapsed = time.time() - t0
        unit = "train"
    elif mode == "federated":
        from federated.fedavg_coordinator import run_federated_training
        cfg["data"]["processed_prefix"] = subset
        cfg["data"]["results_prefix"] = scratch_out
        cfg["federated"]["evaluate_every_round"] = False
        rows = spark.read.parquet(_join(subset, "train.parquet")).count()
        res = run_federated_training(cfg, int(cfg["federated"]["scaling_fixed_num_clients"]))
        elapsed = float(res["train_time_sec"])      # excludes evaluation (SPEC.md Section 4)
        unit = "train"
    elif mode == "etl":
        from etl.build_rating_matrix import build_rating_matrix
        cfg["data"]["raw_parquet_prefix"] = subset
        cfg["data"]["processed_prefix"] = _join(scratch_out, "processed")
        rows = spark.read.parquet(subset).count()
        t0 = time.time()
        build_rating_matrix(cfg)
        elapsed = time.time() - t0
        unit = "raw"
    else:
        raise ValueError(f"unknown mode {mode!r}")

    _log(f"{mode}: {int(executors)} executors requested, {granted} granted, {rows:,} {unit} rows, "
         f"{elapsed:.1f}s")
    return {"mode": mode, "executors": int(executors), "executors_granted": int(granted),
            "rows": int(rows), "row_unit": unit, "time_sec": float(elapsed)}
