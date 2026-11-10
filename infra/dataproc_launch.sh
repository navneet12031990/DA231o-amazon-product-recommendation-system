#!/usr/bin/env bash
# Create the Dataproc cluster for full-scale runs / scaling sweeps (SPEC.md Section 5).
# Usage: bash infra/dataproc_launch.sh <worker_count>        (CONFIG=other.yaml to use another config)
# Everything except the worker count comes from the config. Delete with infra/dataproc_delete.sh.
set -euo pipefail
WORKERS="${1:?usage: dataproc_launch.sh <worker_count>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
IFS='|' read -r PROJECT REGION IMAGE MACHINE DISK_TYPE DISK_GB MAX_IDLE NAME SA LABELS < <(
  python3 "$HERE/_cfg.py" infra.gcp_project infra.region infra.image_version infra.machine_type \
    infra.boot_disk_type infra.boot_disk_size_gb infra.max_idle infra.cluster_name \
    infra.service_account infra.labels)
SA_FLAG=()
if [[ -n "${SA:-}" ]]; then SA_FLAG=(--service-account="$SA"); fi
# KAFKA=1: install a Kafka broker on the master (infra/kafka_init.sh, uploaded first)
KAFKA_FLAGS=()
if [[ "${KAFKA:-0}" == "1" ]]; then
  IFS='|' read -r INIT_PREFIX K_URL K_HOME K_TOPIC K_PARTS < <(python3 "$HERE/_cfg.py" \
    infra.init_actions_prefix streaming.kafka_download_url streaming.kafka_home \
    streaming.kafka_topic streaming.kafka_partitions)
  INIT_URI="${INIT_PREFIX%/}/kafka_init.sh"
  gcloud storage cp "$HERE/kafka_init.sh" "$INIT_URI"
  KAFKA_FLAGS=(--initialization-actions="$INIT_URI" --initialization-action-timeout=15m
               --metadata="kafka-url=$K_URL,kafka-home=$K_HOME,kafka-topic=$K_TOPIC,kafka-partitions=$K_PARTS")
  echo "Kafka: init action $INIT_URI, topic $K_TOPIC ($K_PARTS partitions)"
fi
echo "Creating $NAME in $REGION: 1 master + $WORKERS workers ($MACHINE, ${DISK_GB}GB $DISK_TYPE), labels $LABELS, idle delete $MAX_IDLE"
gcloud dataproc clusters create "$NAME" --project="$PROJECT" --region="$REGION" \
  --num-workers="$WORKERS" --master-machine-type="$MACHINE" --worker-machine-type="$MACHINE" \
  --master-boot-disk-type="$DISK_TYPE" --worker-boot-disk-type="$DISK_TYPE" \
  --master-boot-disk-size="${DISK_GB}GB" --worker-boot-disk-size="${DISK_GB}GB" \
  --image-version="$IMAGE" --labels="$LABELS" --max-idle="$MAX_IDLE" \
  --enable-component-gateway --properties=spark:spark.dynamicAllocation.enabled=false \
  "${SA_FLAG[@]}" "${KAFKA_FLAGS[@]}"
