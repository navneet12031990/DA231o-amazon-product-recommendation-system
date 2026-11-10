#!/usr/bin/env bash
# Dataproc initialization action (SPEC.md Section 5, Streaming): one Kafka broker on the MASTER
# node, in KRaft mode (no ZooKeeper), plus the topic. Workers do nothing.
# Settings arrive as cluster metadata (set by infra/dataproc_launch.sh when KAFKA=1):
#   kafka-url, kafka-home, kafka-topic, kafka-partitions
set -euxo pipefail

meta() { /usr/share/google/get_metadata_value "attributes/$1"; }

ROLE="$(meta dataproc-role)"
if [[ "${ROLE}" != "Master" ]]; then
  echo "kafka_init: ${ROLE} node, nothing to do"
  exit 0
fi

KAFKA_URL="$(meta kafka-url)"
KAFKA_HOME="$(meta kafka-home)"
TOPIC="$(meta kafka-topic)"
PARTITIONS="$(meta kafka-partitions)"
HOST="$(hostname)"
LOGDIR=/var/lib/kafka-logs

mkdir -p "${KAFKA_HOME}" "${LOGDIR}"
# The cluster has no internet route: fetch from the bucket (gs://); plain URLs are a fallback only
if [[ "${KAFKA_URL}" == gs://* ]]; then
  gcloud storage cp "${KAFKA_URL}" /tmp/kafka.tgz
else
  curl -fsSL --retry 2 --connect-timeout 20 "${KAFKA_URL}" -o /tmp/kafka.tgz
fi
tar -xzf /tmp/kafka.tgz -C "${KAFKA_HOME}" --strip-components=1

CONF="${KAFKA_HOME}/config/kraft/server.properties"
sed -i "s#^log.dirs=.*#log.dirs=${LOGDIR}#" "${CONF}"
sed -i "s#^advertised.listeners=.*#advertised.listeners=PLAINTEXT://${HOST}:9092#" "${CONF}"
sed -i "s#^listeners=.*#listeners=PLAINTEXT://0.0.0.0:9092,CONTROLLER://localhost:9093#" "${CONF}"
grep -q '^num.partitions=' "${CONF}" && sed -i "s#^num.partitions=.*#num.partitions=${PARTITIONS}#" "${CONF}"

CLUSTER_ID="$("${KAFKA_HOME}/bin/kafka-storage.sh" random-uuid)"
"${KAFKA_HOME}/bin/kafka-storage.sh" format -t "${CLUSTER_ID}" -c "${CONF}"
# run the broker as a systemd service, so it outlives this script and restarts on failure
JAVA_BIN="$(readlink -f "$(command -v java || echo "${JAVA_HOME:-/usr/lib/jvm/default-java}/bin/java")")"
cat > /etc/systemd/system/kafka.service <<UNIT
[Unit]
Description=Kafka broker (KRaft, single node) for E231project streaming
After=network-online.target
[Service]
Environment=JAVA_HOME=$(dirname "$(dirname "${JAVA_BIN}")")
ExecStart=${KAFKA_HOME}/bin/kafka-server-start.sh ${CONF}
Restart=on-failure
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now kafka

# wait for the broker, then create the topic
for _ in $(seq 1 60); do
  if "${KAFKA_HOME}/bin/kafka-topics.sh" --bootstrap-server "${HOST}:9092" --list >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
"${KAFKA_HOME}/bin/kafka-topics.sh" --bootstrap-server "${HOST}:9092" --create --if-not-exists \
  --topic "${TOPIC}" --partitions "${PARTITIONS}" --replication-factor 1
"${KAFKA_HOME}/bin/kafka-topics.sh" --bootstrap-server "${HOST}:9092" --describe --topic "${TOPIC}"
echo "kafka_init: broker ${HOST}:9092 up, topic ${TOPIC} (${PARTITIONS} partitions)"
