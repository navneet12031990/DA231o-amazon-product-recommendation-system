"""Run Module 1's ETL: convert_to_parquet, then build_rating_matrix.

  spark-submit --master local[*] --driver-memory 8g scripts/run_etl.py --config config.dev.yaml
  spark-submit scripts/run_etl.py --config config.yaml --step build
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import read_text  # noqa: E402  (reads gs:// or local configs)

from etl.build_rating_matrix import build_rating_matrix  # noqa: E402
from etl.convert_to_parquet import convert_to_parquet  # noqa: E402

REQUIRED_KEYS = [
    "data.raw_prefix",
    "data.raw_parquet_prefix",
    "data.processed_prefix",
    "data.categories",
    "data.available_categories",
    "data.min_user_interactions",
    "data.min_item_interactions",
    "data.train_test_split",
    "data.shuffle_partitions_per_core",
    "data.broadcast_max_items",
]


def load_config(path: str) -> dict:
    """Load YAML and fail fast with a clear message if any required key is missing or invalid."""
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

    d = config["data"]
    errors = []
    if not isinstance(d["categories"], list) or not d["categories"]:
        errors.append("data.categories must be a non-empty list")
    else:
        unknown = [c for c in d["categories"] if c not in (d["available_categories"] or [])]
        if unknown:
            errors.append(f"unknown categor{'y' if len(unknown) == 1 else 'ies'} {unknown} "
                          f"(names must match data.available_categories exactly, case-sensitive)")
        if len(set(d["categories"])) != len(d["categories"]):
            errors.append("data.categories contains duplicates")
    if not 0.0 < float(d["train_test_split"]) < 1.0:
        errors.append("data.train_test_split must be strictly between 0 and 1")
    if int(d["min_user_interactions"]) < 2:
        errors.append("data.min_user_interactions must be >= 2 (each user needs a train and a test row)")
    if int(d["min_item_interactions"]) < 1:
        errors.append("data.min_item_interactions must be >= 1")
    if errors:
        sys.exit(f"Config error in {path}: " + "; ".join(errors))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 1 ETL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--step", choices=["convert", "build", "all"], default="all",
                        help="convert = CSV->Parquet only; build = rating matrix only; all = both")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.step in ("convert", "all"):
        convert_to_parquet(config)
    if args.step in ("build", "all"):
        build_rating_matrix(config)


if __name__ == "__main__":
    main()
