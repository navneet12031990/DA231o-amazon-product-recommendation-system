"""Module 4 streaming entry point (SPEC.md Section 5). Each action is one Dataproc job.

  --action producer    replay test events into Kafka (driver on the master, next to the broker)
  --action scorer      Structured Streaming scorer; needs spark.jars.packages=streaming.spark_kafka_package
  --action summarize   streaming_latency.csv + streaming_summary.json from the written batches
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import read_text  # noqa: E402

REQUIRED_KEYS = [
    "data.processed_prefix", "data.results_prefix",
    "streaming.kafka_bootstrap_servers", "streaming.model", "streaming.kafka_topic",
    "streaming.replay_rate_per_sec", "streaming.replay_max_events", "streaming.trigger_seconds",
    "streaming.idle_stop_seconds", "streaming.max_run_seconds", "streaming.kafka_home",
    "streaming.checkpoint_location",
]


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
    if config["streaming"]["model"] not in ("centralized", "federated"):
        sys.exit(f"Config error in {path}: streaming.model must be 'centralized' or 'federated'")
    return config


def main() -> None:
    ap = argparse.ArgumentParser(description="Kafka streaming: producer | scorer | summarize")
    ap.add_argument("--config", required=True)
    ap.add_argument("--action", required=True, choices=["producer", "scorer", "summarize"])
    ap.add_argument("--max-seconds", type=float, help="scorer: override streaming.max_run_seconds")
    args = ap.parse_args()
    config = load_config(args.config)
    if args.action == "producer":
        from streaming.kafka_producer_replay import replay_to_kafka
        replay_to_kafka(config)
    elif args.action == "scorer":
        from streaming.stream_scorer import run_stream_scorer
        run_stream_scorer(config, max_seconds=args.max_seconds)
    else:
        from streaming.stream_scorer import summarize_stream
        summarize_stream(config)


if __name__ == "__main__":
    main()
