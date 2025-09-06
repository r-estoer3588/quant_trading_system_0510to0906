import sys
from pathlib import Path


# プロジェクトルートを import パスに追加（pytest 実行場所に依存しないため）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

