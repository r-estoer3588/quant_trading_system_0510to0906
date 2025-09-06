import ast
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def iter_py_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.py'):
                yield Path(dirpath) / fn

def top_level_modules_from_file(path: Path):
    try:
        src = path.read_text(encoding='utf-8')
    except Exception:
        return set()
    try:
        tree = ast.parse(src, filename=str(path))
    except Exception:
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                top = (n.name or '').split('.')[0]
                if top:
                    mods.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split('.')[0]
                if top:
                    mods.add(top)
    return mods

def main():
    mods = set()
    for f in iter_py_files(ROOT):
        if f.parts and any(part.startswith('.') for part in f.parts):
            # skip hidden directories like .venv, .git
            continue
        if any(part in {'.venv', '.git', '__pycache__'} for part in f.parts):
            continue
        mods |= top_level_modules_from_file(f)
    for m in sorted(mods):
        print(m)

if __name__ == '__main__':
    main()

