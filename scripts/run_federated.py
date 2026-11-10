"""Run Module 3: simulated FedAvg for every federated.num_clients value, then compare with
Module 2's centralized results (run scripts/run_centralized.py with the same config first).

  spark-submit --master local[*] --driver-memory 8g scripts/run_federated.py --config config.dev.yaml
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import read_text  # noqa: E402  (reads gs:// or local configs)

from federated.fedavg_coordinator import run_federated_experiment  # noqa: E402

REQUIRED_KEYS = [
    "seed",
    "data.processed_prefix", "data.results_prefix",
    "als.rank", "als.reg_param", "als.residual_on_bias", "als.baseline_damping",
    "als.prediction_min", "als.prediction_max",
    "federated.num_clients", "federated.communication_rounds", "federated.local_epochs",
    "federated.init_std", "federated.evaluate_every_round", "federated.partitions_per_core", "federated.model_num_clients", "federated.rows_per_partition",
    "evaluation.cold_user_threshold",
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
    fed = config["federated"]
    errors = []
    if not isinstance(fed["num_clients"], list) or not fed["num_clients"] or min(fed["num_clients"]) < 1:
        errors.append("federated.num_clients must be a non-empty list of integers >= 1")
    if int(fed["communication_rounds"]) < 1:
        errors.append("federated.communication_rounds must be >= 1")
    if int(fed["local_epochs"]) < 1:
        errors.append("federated.local_epochs must be >= 1")
    if errors:
        sys.exit(f"Config error in {path}: " + "; ".join(errors))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 3 simulated FedAvg")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-clients", nargs="+", type=int,
                        help="run only these client counts (default: federated.num_clients); the run "
                             "with federated.model_num_clients saves the federated model")
    args = parser.parse_args()
    run_federated_experiment(load_config(args.config), args.num_clients)


if __name__ == "__main__":
    main()
