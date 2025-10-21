#!/usr/bin/env bash
# Start Playwright server wrapper (POSIX shell version)
# Runs the data pipeline first, then starts Streamlit for Playwright tests.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Ensure RUN_NAMESPACE is set for output separation in CI
if [ -z "${RUN_NAMESPACE:-}" ]; then
  RUN_NAMESPACE="ci_$(cat /proc/sys/kernel/random/uuid | tr -d '-')"
  export RUN_NAMESPACE
  echo "[INFO] Generated RUN_NAMESPACE=$RUN_NAMESPACE"
else
  echo "[INFO] Using RUN_NAMESPACE=$RUN_NAMESPACE"
fi

# Enable per-run subdir and run lock by default in CI
: "${PIPELINE_USE_RUN_SUBDIR:=1}"
: "${PIPELINE_USE_RUN_LOCK:=1}"
export PIPELINE_USE_RUN_SUBDIR
export PIPELINE_USE_RUN_LOCK

echo "[$(date --iso-8601=seconds)] Running full pipeline: run_all_systems_today.py (this may take a few minutes)..."
# Use test-mode mini by default in CI to bound execution time
python3 -u scripts/run_all_systems_today.py --test-mode mini --save-csv || {
  rc=$?
  echo "[ERROR] run_all_systems_today.py failed with exit code $rc. Aborting web server startup." >&2
  exit $rc
}

echo "[$(date --iso-8601=seconds)] Pipeline completed successfully. Starting Streamlit..."
# Start Streamlit in foreground so Playwright can detect the server
python3 -u -m streamlit run apps/app_integrated.py --server.headless true --server.port 8501

# When Streamlit exits, script ends
exit $?
