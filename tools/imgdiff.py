"""スナップショット間の画像差分を確認するユーティリティ。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def _latest_snapshots(base: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if not base.exists():
        return None, None
    items = [p for p in base.iterdir() if p.is_dir()]
    items.sort(key=lambda p: p.name)
    if len(items) < 2:
        return None, None
    return items[-2], items[-1]


def _collect_images(base: Path, rel_dir: str) -> List[Path]:
    target = base / rel_dir
    if not target.exists():
        return []
    images: List[Path] = []
    for path in target.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return images


def _rel_path(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def _compare_bytes(path_a: Path, path_b: Path) -> bool:
    try:
        if path_a.stat().st_size != path_b.stat().st_size:
            return False
        return path_a.read_bytes() == path_b.read_bytes()
    except OSError:
        return False


def _compare_pixels(path_a: Path, path_b: Path, diff_path: Path) -> bool:
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return _compare_bytes(path_a, path_b)

    try:
        image_a = Image.open(path_a).convert("RGBA")
        image_b = Image.open(path_b).convert("RGBA")
        if image_a.size != image_b.size:
            return False
        diff = ImageChops.difference(image_a, image_b)
        if diff.getbbox() is None:
            return True
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff.save(diff_path)
        return False
    except OSError:
        return False


def _build_report(
    mismatches: List[Tuple[str, str, Path, Path, Path]],
    snap_a: Path,
    snap_b: Path,
    src_dir: str,
    report_path: Path,
) -> None:
    rows: List[str] = []
    for rel, reason, path_a, path_b, diff_path in mismatches:
        cell_a = path_a.as_posix() if path_a.exists() else ""
        cell_b = path_b.as_posix() if path_b.exists() else ""
        cell_d = diff_path.as_posix() if diff_path.exists() else ""
        html_a = "" if not cell_a else f'<img src="{cell_a}" height="120"/>'
        html_b = "" if not cell_b else f'<img src="{cell_b}" height="120"/>'
        html_d = "" if not cell_d else f'<img src="{cell_d}" height="120"/>'
        rows.append(f"<tr><td>{rel}</td><td>{reason}</td><td>{html_a}</td><td>{html_b}</td><td>{html_d}</td></tr>")

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Image Diff Report</title>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    body {{
      background: #1e1e1e;
      color: #d4d4d4;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      padding: 20px;
      line-height: 1.6;
    }}
    h2 {{
      color: #4ec9b0;
      margin-bottom: 16px;
      font-size: 28px;
      font-weight: 600;
    }}
    .info {{
      background: #252526;
      border-left: 4px solid #4ec9b0;
      padding: 12px 16px;
      margin-bottom: 24px;
      border-radius: 4px;
    }}
    .info p {{
      margin: 4px 0;
      font-size: 14px;
    }}
    .info strong {{
      color: #569cd6;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #252526;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }}
    th {{
      background: #2d2d30;
      color: #4ec9b0;
      padding: 12px 16px;
      text-align: left;
      font-weight: 600;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 2px solid #3e3e42;
    }}
    td {{
      padding: 12px 16px;
      border-bottom: 1px solid #3e3e42;
      vertical-align: top;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    tr:hover {{
      background: #2d2d30;
    }}
    td:nth-child(1) {{
      font-family: 'Consolas', 'Courier New', monospace;
      color: #ce9178;
      font-size: 13px;
      max-width: 300px;
      word-break: break-all;
    }}
    td:nth-child(2) {{
      color: #dcdcaa;
      font-weight: 500;
      font-size: 13px;
    }}
    img {{
      display: block;
      border: 1px solid #3e3e42;
      border-radius: 4px;
      background: #1e1e1e;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
      transition: transform 0.2s ease;
    }}
    img:hover {{
      transform: scale(1.05);
      box-shadow: 0 4px 16px rgba(78, 201, 176, 0.3);
    }}
    .no-diff {{
      text-align: center;
      padding: 32px;
      color: #4ec9b0;
      font-size: 16px;
      font-weight: 500;
    }}
    .no-diff::before {{
      content: "✓ ";
      font-size: 24px;
    }}
  </style>
</head>
<body>
  <h2>Image Diff Report</h2>
  <div class="info">
    <p><strong>Snapshot A:</strong> {snap_a.as_posix()}</p>
    <p><strong>Snapshot B:</strong> {snap_b.as_posix()}</p>
    <p><strong>Source dir:</strong> {src_dir}</p>
  </div>
  <table>
    <thead>
      <tr>
        <th>Relative Path</th>
        <th>Status</th>
        <th>A</th>
        <th>B</th>
        <th>Diff</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows) if rows else '<tr><td colspan="5" class="no-diff">差分はありません</td></tr>'}
    </tbody>
  </table>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare images between two snapshot directories.")
    parser.add_argument("--snap-a", help="Older snapshot directory.")
    parser.add_argument("--snap-b", help="Newer snapshot directory.")
    parser.add_argument(
        "--src-dir",
        default="results_images",
        help="Relative directory to search within each snapshot.",
    )
    parser.add_argument(
        "--report",
        help="HTML report output path (default: stored under snapshot B).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    snapshots_dir = project_root / "snapshots"

    snap_a = Path(args.snap_a) if args.snap_a else None
    snap_b = Path(args.snap_b) if args.snap_b else None
    if not snap_a or not snap_b:
        older, newer = _latest_snapshots(snapshots_dir)
        snap_a = snap_a or older
        snap_b = snap_b or newer

    if not snap_a or not snap_b or not snap_a.exists() or not snap_b.exists():
        print("No pair of snapshots available. Run snapshot first.")
        return

    images_a = _collect_images(snap_a, args.src_dir)
    images_b = _collect_images(snap_b, args.src_dir)
    if not images_a and not images_b:
        print(f"No images found under {args.src_dir}.")
        return

    map_b: Dict[str, Path] = {str(_rel_path(p, snap_b)): p for p in images_b}
    mismatches: List[Tuple[str, str, Path, Path, Path]] = []
    matched = 0
    diff_root = snap_b / "imgdiff" / args.src_dir

    for path_a in images_a:
        rel = str(_rel_path(path_a, snap_a))
        path_b = map_b.get(rel)
        if not path_b:
            mismatches.append((rel, "missing in B", path_a, Path(""), Path("")))
            continue
        matched += 1
        diff_path = diff_root / rel
        result = _compare_pixels(path_a, path_b, diff_path)
        if not result:
            mismatches.append((rel, "pixels differ", path_a, path_b, diff_path))

    paths_a = {str(_rel_path(p, snap_a)) for p in images_a}
    for rel, path_b in map_b.items():
        if rel not in paths_a:
            mismatches.append((rel, "new in B", Path(""), path_b, Path("")))

    print(f"Compared: matched={matched}, mismatches={len(mismatches)}")
    for rel, reason, _a, _b, _d in mismatches[:20]:
        print(f"- {rel}: {reason}")
    if len(mismatches) > 20:
        print("(showing top 20)")

    report_path = Path(args.report) if args.report else snap_b / "imgdiff_report.html"
    try:
        _build_report(mismatches, snap_a, snap_b, args.src_dir, report_path)
        print(f"HTML report: {report_path}")
    except OSError:
        print("Failed to write HTML report.")


if __name__ == "__main__":
    main()
