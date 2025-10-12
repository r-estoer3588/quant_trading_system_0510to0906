"""成果物スナップショットを保存する簡易ツール。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
DEFAULT_SOURCES = [
    "results_csv",
    "results_csv_test",
    "logs",
    "results_images",
    "outputs/images",
    "artifacts/images",
    "images",
]


def _copy_tree(src: Path, dst: Path, max_bytes: int) -> None:
    if not src.exists():
        return
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        out_dir = dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            src_path = Path(root) / name
            try:
                size = src_path.stat().st_size
            except OSError:
                continue
            if size > max_bytes and src_path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                shutil.copy2(src_path, out_dir / name)
            except OSError:
                continue


def _parse_sources(args: argparse.Namespace) -> list[str]:
    sources: list[str] = []
    if args.source:
        sources.extend(args.source)
    if args.sources:
        for chunk in args.sources.split(","):
            value = chunk.strip()
            if value:
                sources.append(value)
    return sources or DEFAULT_SOURCES


def _ensure_manifest(base: Path, sources: Iterable[str], timestamp: str) -> None:
    manifest = {
        "timestamp": timestamp,
        "sources": list(sources),
        "cwd": str(base),
        "git_commit": None,
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(base))
        manifest["git_commit"] = commit.decode("utf-8").strip()
    except (subprocess.SubprocessError, OSError, UnicodeDecodeError):
        pass
    (base / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture project snapshots into snapshots/<timestamp>/."
    )
    parser.add_argument(
        "--sources",
        help="Comma separated list of directories to copy.",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Directory to copy (repeatable).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="Maximum size per non-image file in bytes (default: 10MB).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target_root = project_root / "snapshots" / timestamp
    target_root.mkdir(parents=True, exist_ok=True)

    sources = _parse_sources(args)
    for src_name in sources:
        _copy_tree(project_root / src_name, target_root / src_name, args.max_bytes)

    _ensure_manifest(target_root, sources, timestamp)
    print(f"Snapshot saved to: {target_root}")


if __name__ == "__main__":
    main()
