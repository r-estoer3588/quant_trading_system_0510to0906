#!/usr/bin/env bash

# Lightweight Codacy Analysis runner for WSL (Ubuntu) using Docker image
# - Copies the repo (excluding heavy caches) into /tmp to avoid /mnt/c file quirks
# - Runs codacy/codacy-analysis-cli inside Docker
# - Writes SARIF to /reports (bind-mounted to host temp dir)
#
# Usage (inside WSL):
#   bash tools/codacy_wsl_analyze.sh                # all tools, SARIF out to codacy_report/results.sarif
#   TOOL=pylint bash tools/codacy_wsl_analyze.sh    # only pylint
#   FORMAT=json bash tools/codacy_wsl_analyze.sh    # JSON format
#
# Env vars:
#   SRC    - Source repo path (default: /mnt/c/Repos/quant_trading_system)
#   DST    - Temp working copy (default: /tmp/codacy_src)
#   TOOL   - Single tool to run (e.g., pylint). Empty = all tools
#   FORMAT - Output format (sarif|json). Default: sarif
#
set -euo pipefail

SRC=${SRC:-/mnt/c/Repos/quant_trading_system}
# 一時ディレクトリ（衝突・権限問題を避けるため毎回新規に作成）
DST=${DST:-}
if [[ -z "$DST" ]]; then
  DST=$(mktemp -d /tmp/codacy_src_XXXXXX)
fi
TOOL=${TOOL:-}
FORMAT=${FORMAT:-sarif}

echo "[codacy-wsl] SRC=$SRC"
echo "[codacy-wsl] DST=$DST"
echo "[codacy-wsl] TOOL=${TOOL:-<all>}"
echo "[codacy-wsl] FORMAT=$FORMAT"

# Prepare temp workspace
# 構成
mkdir -p "$DST/reports"

echo "[codacy-wsl] Syncing source -> temp ..."
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude .git \
    --exclude codacy_report \
    --exclude data_cache \
    --exclude data_cache_recent \
    --exclude results_csv \
    --exclude results_csv_test \
    --exclude logs \
    "$SRC"/ "$DST"/
else
  tar -C "$SRC" \
    --exclude=.git \
    --exclude=codacy_report \
    --exclude=data_cache \
    --exclude=data_cache_recent \
    --exclude=results_csv \
    --exclude=results_csv_test \
    --exclude=logs \
    -cf - . | tar -C "$DST" -xf -
fi

mkdir -p "$SRC/codacy_report"

export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

DOCKER_IMAGE=codacy/codacy-analysis-cli:latest
echo "[codacy-wsl] Pulling docker image if needed: $DOCKER_IMAGE"
docker pull "$DOCKER_IMAGE" >/dev/null 2>&1 || true

CMD=(analyze --format "$FORMAT" --output /reports/results.$FORMAT)
if [[ -n "$TOOL" ]]; then
  CMD+=(--tool "$TOOL")
fi

EXIT_CODE=0
# 1) ネイティブJARがあれば優先（PATH不要）
if [[ -f "$HOME/.codacy/codacy-analysis-cli.jar" ]]; then
  echo "[codacy-wsl] Running analysis via native JAR ..."
  # dockerが使えるかを先に確認
  if ! command -v docker >/dev/null 2>&1; then
    echo "[codacy-wsl] ERROR: docker コマンドが見つかりません (WSL内にDockerが必要)" >&2
    EXIT_CODE=127
  else
    set +e
    java -Dfile.encoding=UTF-8 -jar "$HOME/.codacy/codacy-analysis-cli.jar" analyze \
      --directory "$DST" \
      --format "$FORMAT" \
      --output "$DST/reports/results.$FORMAT" \
      ${TOOL:+--tool "$TOOL"}
    NATIVE_CODE=$?
    set -e
    EXIT_CODE=$NATIVE_CODE
  fi
fi

# 2) フォールバック: コンテナ版CLI + ホストDockerを透過利用
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[codacy-wsl] Running analysis in Docker (with host docker socket) ..."
  set +e
  docker run --rm \
    --user 0:0 \
    -e DOCKER_HOST=unix:///var/run/docker.sock \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$(command -v docker)":/usr/bin/docker \
    -v "$DST":/src \
    -v "$DST/reports":/reports \
    -w /src \
    "$DOCKER_IMAGE" "${CMD[@]}"
  EXIT_CODE=$?
  set -e
fi

# Ensure reports are owned by the current user to avoid cleanup failures on next run
sudo chown -R "$(id -u):$(id -g)" "$DST" 2>/dev/null || true

if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[codacy-wsl] WARNING: codacy-analysis-cli exited with code $EXIT_CODE (partial results may still exist)." >&2
fi

if [[ -s "$DST/reports/results.$FORMAT" ]]; then
  cp -f "$DST/reports/results.$FORMAT" "$SRC/codacy_report/results.sarif"
  echo "[codacy-wsl] Wrote SARIF: $SRC/codacy_report/results.sarif"
  exit 0
else
  echo "[codacy-wsl] ERROR: No results file produced at $DST/reports/results.$FORMAT" >&2
  exit 2
fi
#!/usr/bin/env bash
set -euo pipefail

# Codacy analysis runner for WSL (ext4) to avoid /mnt/c issues and enforce UTF-8
# Usage:
#   bash tools/codacy_wsl_analyze.sh [head|copy|micro]
#   - head: analyze HEAD snapshot via git archive (default)
#   - copy: analyze minimal working copy (apps/common/core/strategies/scripts/tests, config files)
#   - micro: analyze only UTF-8 safe files (auto-selected) to avoid charset errors

MODE=${1:-head}

WORKDIR="/tmp/codacy_ws_$$"
REPO_WIN="/mnt/c/Repos/quant_trading_system"
SARIF_OUT="$REPO_WIN/codacy_report/results.sarif"

mkdir -p "$WORKDIR"
echo "[codacy] WORKDIR=$WORKDIR MODE=$MODE"

case "$MODE" in
  head)
    # Create clean snapshot from HEAD
    git -C "$REPO_WIN" rev-parse HEAD >/dev/null
    git -C "$REPO_WIN" archive --format=tar HEAD | tar -xf - -C "$WORKDIR"
    ;;
  copy)
    # Copy a minimal set of folders/files from working copy (no .git)
    for path in \
      apps \
      common \
      core \
      strategies \
      scripts \
      tests \
      pyproject.toml \
      mypy.ini \
      requirements.txt \
      requirements-dev.txt \
      .codacy.yml
    do
      if [ -e "$REPO_WIN/$path" ]; then
        cp -a "$REPO_WIN/$path" "$WORKDIR/" || true
      fi
    done
    ;;
  micro)
    # Copy only files that are valid UTF-8 to avoid MalformedInputException in Codacy CLI
    include_list=(
      apps/*.py
      common/*.py
      core/*.py
      strategies/*.py
      scripts/*.py
      tests/*.py
      *.py
      pyproject.toml
      mypy.ini
      requirements.txt
      requirements-dev.txt
      .codacy.yml
    )
    shopt -s nullglob
    for pattern in "${include_list[@]}"; do
      for src in "$REPO_WIN"/$pattern; do
        rel="${src#$REPO_WIN/}"
        dest_dir="$WORKDIR/$(dirname "$rel")"
        mkdir -p "$dest_dir"
        # Validate UTF-8 readability using Python; skip files that fail
        if [[ "$src" == *.py ]]; then
          if python3 - "$src" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    s = p.read_text(encoding='utf-8')
    # Additionally ensure ASCII-only to avoid JVM charset hiccups
    _ = s.encode('ascii')
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
          then
            : # ok
          else
            echo "[skip-nonutf8] $rel"
            continue
          fi
        fi
        cp -a "$src" "$dest_dir/" || true
      done
    done
    ;;
  *)
    echo "Unknown mode: $MODE (use head|copy)" >&2
    exit 2
    ;;
esac

# Enforce UTF-8 for JVM and locale to avoid MalformedInputException
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export JAVA_TOOL_OPTIONS="-Dfile.encoding=UTF-8"

# Run Codacy CLI
if ! command -v codacy-analysis-cli >/dev/null 2>&1; then
  if [ -x "$HOME/.codacy/bin/codacy-analysis-cli" ]; then
    CODACY_BIN="$HOME/.codacy/bin/codacy-analysis-cli"
  else
    echo "codacy-analysis-cli not found. Ensure it is installed under ~/.codacy/bin or in PATH." >&2
    exit 3
  fi
else
  CODACY_BIN="codacy-analysis-cli"
fi

echo "[codacy] Running analysis..."
"$CODACY_BIN" analyze \
  --directory "$WORKDIR" \
  --tool pylint \
  --format sarif \
  --output "$SARIF_OUT" \
  --verbose || {
    echo "[codacy] Analysis failed" >&2
    exit 4
  }

echo "[codacy] SARIF written to $SARIF_OUT"
exit 0
