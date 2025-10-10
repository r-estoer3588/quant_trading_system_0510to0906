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
