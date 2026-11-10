"""Run Module 2: train + evaluate the centralized ALS baseline, or the one-time tuning grid.

  spark-submit --master local[*] --driver-memory 8g scripts/run_centralized.py --config config.dev.yaml
  spark-submit --master local[*] --driver-memory 8g scripts/run_centralized.py --config config.dev.yaml --tune
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import read_text  # noqa: E402  (reads gs:// or local configs)

from centralized.train_als import train_centralized, tune_als  # noqa: E402

REQUIRED_KEYS = [
    "seed",
    "data.processed_prefix", "data.results_prefix", "data.train_test_split",
    "als.rank", "als.max_iter", "als.reg_param", "als.cold_start_strategy",
    "als.baseline_damping", "als.residual_on_bias", "als.prediction_min", "als.prediction_max",
    "evaluation.cold_user_threshold", "evaluation.precision_at_k",
    "evaluation.relevance_threshold", "evaluation.precision_candidate_pool",
    "evaluation.precision_user_sample",
]
TUNE_KEYS = ["als.tuning_ranks", "als.tuning_reg_params", "als.tuning_train_fraction"]


def load_config(path: str, required: list) -> dict:
    """Load YAML and fail fast with a clear message if any required key is missing."""
    config = yaml.safe_load(read_text(path)) or {}
    missing = []
    for dotted in required:
        node = config
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node or node[part] is None:
                missing.append(dotted)
                break
            node = node[part]
    if missing:
        sys.exit(f"Config error in {path}: missing required key(s): {', '.join(missing)}")
    if config["als"]["cold_start_strategy"] != "drop":
        sys.exit(f"Config error in {path}: als.cold_start_strategy must be 'drop' (SPEC.md Section 3)")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 2 centralized ALS")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tune", action="store_true",
                        help="run the one-time rank/reg_param grid on a validation split of train")
    args = parser.parse_args()
    config = load_config(args.config, REQUIRED_KEYS + (TUNE_KEYS if args.tune else []))
    if args.tune:
        tune_als(config)
    else:
        train_centralized(config)


if __name__ == "__main__":
    main()
