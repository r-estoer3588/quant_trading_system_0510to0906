#!/usr/bin/env bash
# Local helper: build the Playwright CI Docker image and run tests inside it.
# Usage: ./ci/run_in_docker.sh [--namespace NAME] [--skip-pipeline]

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NAMESPACE="local"
SKIP_PIPELINE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --skip-pipeline)
      SKIP_PIPELINE=1
      shift
+      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

IMAGE_TAG="playci/pts-playwright-ci:local"

echo "Building Docker image: $IMAGE_TAG"
docker build -t "$IMAGE_TAG" -f ci/Dockerfile.playwright-ci .

echo "Running container (namespace=$NAMESPACE)"
mkdir -p playwright-report results_csv
if [ "$SKIP_PIPELINE" = "1" ]; then
  docker run --rm \
    -e CI=1 \
    -e RUN_NAMESPACE="$NAMESPACE" \
    -e PIPELINE_USE_RUN_SUBDIR=1 \
    -e PIPELINE_USE_RUN_LOCK=1 \
    -v "$PWD/playwright-report:/workspace/playwright-report" \
    -v "$PWD/results_csv:/workspace/results_csv" \
    "$IMAGE_TAG" /bin/bash -lc "npx playwright test --config=playwright.config.ci.ts --workers=1"
else
  docker run --rm \
    -e CI=1 \
    -e RUN_NAMESPACE="$NAMESPACE" \
    -e PIPELINE_USE_RUN_SUBDIR=1 \
    -e PIPELINE_USE_RUN_LOCK=1 \
    -v "$PWD/playwright-report:/workspace/playwright-report" \
    -v "$PWD/results_csv:/workspace/results_csv" \
    "$IMAGE_TAG"
fi
