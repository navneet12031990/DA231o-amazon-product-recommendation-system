"""Cluster-side entry point for the scaling benchmarks (SPEC.md Section 5). Submitted as a Dataproc
job by scripts/run_benchmarks.py; one invocation = one Spark application.

  --action prepare  capacity check + all subsets the enabled sweeps need (untimed)
  --action run      one timed run; writes one JSON result under scaling.scratch_prefix/results/
  --action collect  gathers all JSON results into results/scaling_results.csv
"""
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import read_text, write_text  # noqa: E402

REQUIRED_KEYS = [
    "seed", "data.processed_prefix", "data.results_prefix", "data.raw_parquet_prefix", "data.categories",
    "scaling.executor_counts", "scaling.strong_fixed_train_rows", "scaling.weak_train_rows_per_executor",
    "scaling.etl_sweep_enabled", "scaling.etl_strong_fixed_raw_rows", "scaling.etl_weak_raw_rows_per_executor",
    "scaling.repetitions", "scaling.executor_wait_sec", "scaling.scratch_prefix",
    "federated.scaling_fixed_num_clients", "als.rank", "als.reg_param",
]
CSV_COLUMNS = ["mode", "sweep", "repetition", "executors", "executors_granted", "rows", "row_unit", "time_sec"]


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaling benchmark job (cluster side)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--action", required=True, choices=["prepare", "run", "collect"])
    ap.add_argument("--modes", nargs="+", default=["centralized", "federated"])
    ap.add_argument("--sweeps", nargs="+", default=["strong", "weak"])
    ap.add_argument("--mode", choices=["centralized", "federated", "etl"])
    ap.add_argument("--sweep", choices=["strong", "weak"])
    ap.add_argument("--executors", type=int)
    ap.add_argument("--repetition", type=int, default=1)
    args = ap.parse_args()
    config = load_config(args.config)

    from pyspark.sql import SparkSession
    from benchmarks.scaling_runner import (prepare_subsets, subset_path, sweep_targets, timed_run)
    spark = SparkSession.builder.getOrCreate()
    results_dir = _join(config["scaling"]["scratch_prefix"], "results")

    if args.action == "prepare":
        prepare_subsets(config, args.modes, args.sweeps)
    elif args.action == "run":
        if not (args.mode and args.sweep and args.executors):
            sys.exit("--action run needs --mode, --sweep and --executors")
        unit = "raw" if args.mode == "etl" else "train"
        target = sweep_targets(config, args.sweep, unit)[args.executors]
        res = timed_run(config, args.mode, subset_path(config, unit, target), args.executors)
        res.update({"sweep": args.sweep, "repetition": args.repetition})
        name = f"{args.mode}_{args.sweep}_{args.executors}_r{args.repetition}.json"
        write_text(_join(results_dir, name), json.dumps(res))
    else:  # collect
        try:
            rows = [json.loads(r.value) for r in spark.read.text(_join(results_dir, "*.json")).collect()]
        except Exception:  # noqa: BLE001 - no results yet
            rows = []
        rows.sort(key=lambda r: (r["mode"], r["sweep"], r["executors"], r["repetition"]))
        lines = [",".join(CSV_COLUMNS)] + [",".join(str(r[c]) for c in CSV_COLUMNS) for r in rows]
        out = _join(config["data"]["results_prefix"], "scaling_results.csv")
        write_text(out, "\n".join(lines) + "\n")
        print(f"[scaling] collected {len(rows)} runs -> {out}", flush=True)


if __name__ == "__main__":
    main()
