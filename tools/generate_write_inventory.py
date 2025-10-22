#!/usr/bin/env python3
"""Generate an inventory of file-write call sites for migration.

Scans Python source files for common write/serialization call patterns
(`to_csv`, `json.dump(s)`, `write_text`, `.encode(...)` etc.) and emits a
JSON report that lists file, line number and a short code snippet. The
output is saved to repro_payloads/migration_inventory.json to drive the
manual migration to `common.io_utils` helpers.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

# Ensure repository root is on sys.path so `from common.io_utils` works when
# running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# local imports of project modules are performed inside main() after
# inserting the repo root into sys.path.
OUT_PATH = REPO_ROOT / "repro_payloads" / "migration_inventory.json"

# Directories to skip while scanning
EXCLUDE_DIR_PARTS = {
    "venv",
    "venv_old",
    ".git",
    "node_modules",
    "htmlcov",
    "playwright-report",
}

# Patterns to look for (regex, short-name)
PATTERNS: list[tuple[str, str]] = [
    (r"\bto_csv\s*\(", "to_csv("),
    (r"\.to_csv\s*\(", ".to_csv("),
    (r"\bjson\.dump[s]?\s*\(", "json.dump(s)"),
    (r"\.write_text\s*\(", "Path.write_text"),
    (r"\.write_bytes\s*\(", "Path.write_bytes"),
    (r"\.to_json\s*\(", ".to_json("),
    (r"\.encode\s*\(", ".encode("),
    (r"open\([^)]*[\'\"]w[\'\"][^)]*\)", "open(..., 'w')"),
    (r"write_json\s*\(", "write_json("),
    (r"write_text\s*\(", "write_text("),
]


def _should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_DIR_PARTS for part in path.parts)


def scan_repo() -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    files_scanned = 0
    pattern_counts: dict[str, int] = defaultdict(int)

    for p in sorted(REPO_ROOT.rglob("*.py")):
        if _should_skip(p):
            continue
        files_scanned += 1
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            # Fallback: read as bytes then decode
            try:
                text = p.read_bytes().decode("utf-8", errors="replace")
            except Exception:
                text = ""
        lines = text.splitlines()
        for ln, line in enumerate(lines, start=1):
            for pat, name in PATTERNS:
                if re.search(pat, line):
                    context_prev = lines[ln - 2] if ln > 1 else ""
                    context_next = lines[ln] if ln < len(lines) else ""
                    matches.append(
                        {
                            "file": str(p.relative_to(REPO_ROOT)),
                            "line_no": ln,
                            "pattern": name,
                            "snippet": line.strip(),
                            "context": {
                                "prev": context_prev.strip(),
                                "next": context_next.strip(),
                            },
                        }
                    )
                    pattern_counts[name] += 1

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "repo_root": str(REPO_ROOT),
        "files_scanned": files_scanned,
        "matches_found": len(matches),
        "pattern_counts": dict(pattern_counts),
        "matches": matches,
    }
    return out


def main() -> int:
    out = scan_repo()
    try:
        # Local import here ensures REPO_ROOT was added to sys.path above.
        from common.io_utils import write_json

        write_json(OUT_PATH, out, ensure_ascii=False, indent=2, default=str)
        print(f"Wrote inventory to: {OUT_PATH} ({out['matches_found']} matches)")
    except Exception as e:
        print(f"Failed to write inventory JSON: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
