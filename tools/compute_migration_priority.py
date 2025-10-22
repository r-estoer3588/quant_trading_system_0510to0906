"""Compute priority scores for migration inventory and print top files and suggested batches."""

import json
from pathlib import Path

INV = Path("repro_payloads/migration_inventory.json")


def score_files(inv_path: Path):
    j = json.loads(inv_path.read_text(encoding="utf-8"))
    matches = j["matches"]
    counts = {}
    patterns = {}
    for m in matches:
        f = m["file"]
        counts[f] = counts.get(f, 0) + 1
        patterns.setdefault(f, {}).setdefault(m["pattern"], 0)
        patterns[f][m["pattern"]] += 1

    items = []
    for f, c in counts.items():
        score = float(c)
        # directory weights
        if f.startswith("tools\\") or f.startswith("tools/"):
            score += 3.0
        if f.startswith("scripts\\") or f.startswith("scripts/"):
            score += 2.0
        if f.startswith("apps\\") or f.startswith("apps/"):
            score += 2.5
        if f.startswith("common\\") or f.startswith("common/"):
            score += 1.5
        # bump if mentions key directories when file exists
        p = Path(f)
        text = ""
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                text = ""
        if "results_csv" in text or "repro_payloads" in text:
            score += 2.0
        items.append((f, c, score, patterns[f]))

    items.sort(key=lambda x: x[2], reverse=True)
    return items


if __name__ == "__main__":
    out = score_files(INV)
    print("Top 50 files by priority score:\n")
    for i, (f, c, s, pats) in enumerate(out[:50], 1):
        print(
            f"{i:2d}. {f:50s} count={c:3d} score={s:.1f} patterns={list(pats.keys())}"
        )

    # suggest batches of 6
    batch_size = 6
    batches = [out[i : i + batch_size] for i in range(0, min(60, len(out)), batch_size)]
    print("\nSuggested batches (6 files each):")
    for bi, b in enumerate(batches, 1):
        print(f"Batch {bi}:")
        for f, c, s, _ in b:
            print(f"  - {f} (score={s:.1f}, count={c})")
