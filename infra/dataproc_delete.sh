#!/usr/bin/env bash
# Delete the cluster named in the config (CONFIG env var, default config.yaml). Run right after every run.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IFS='|' read -r PROJECT REGION NAME < <(python3 "$HERE/_cfg.py" infra.gcp_project infra.region infra.cluster_name)
gcloud dataproc clusters delete "$NAME" --project="$PROJECT" --region="$REGION" --quiet
gcloud dataproc clusters list --project="$PROJECT" --region="$REGION"
