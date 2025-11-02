"""Build a ZIP containing the Playwright CI templates and the mini repro folder.

Run from the repository root:
  python tools/build_playwright_repro_zip.py

Output:
  docs/playwright_mini_repro.zip
"""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_ZIP = OUT_DIR / "playwright_mini_repro.zip"

INCLUDE = [
    "playwright.config.ci.ts",
    "ci/Dockerfile.playwright-ci",
    ".github/workflows/playwright.yml",
    "repro/playwright-mini",
]

print("Creating", OUT_ZIP)
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel in INCLUDE:
        p = ROOT / rel
        if not p.exists():
            print("Warning: path not found:", rel)
            continue
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    arc = f.relative_to(ROOT)
                    zf.write(f, arc)
        else:
            zf.write(p, p.relative_to(ROOT))
print("Wrote", OUT_ZIP)
