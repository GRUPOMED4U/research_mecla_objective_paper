#!/usr/bin/env sh
set -eu

# Google Cloud authentication setup on startup
KEY="/app/key.json"

if [ -f "$KEY" ]; then
  echo "[entrypoint] Found GCP key at $KEY"
  gcloud auth activate-service-account --key-file="$KEY"
else
  echo "[entrypoint] ERROR: expected GCP key at $KEY but file not found"
  ls -la /app || true
  exit 1
fi

# enable DVC automatic staging configuration
uv run dvc config core.autostage true

exec "$@"