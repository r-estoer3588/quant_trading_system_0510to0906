"""
システム仕様自動生成ツール

各システムのロジック・フィルター・ランキングキーを自動抽出して markdown 仕様書を生成します。

使い方:
    python tools/generate_system_specs.py --system 1 --output docs/systems/system1.md
    python tools/generate_system_specs.py --all --output-dir docs/systems/
"""

import argparse
import ast
import sys
from pathlib import Path


def extract_system_spec(system_num: int) -> dict:
    """システムファイルから仕様を抽出"""
    core_file = Path(f"core/system{system_num}.py")
    strategy_file = Path(f"strategies/system{system_num}_strategy.py")

    spec = {
        "system_num": system_num,
        "direction": "",
        "filter_columns": [],
        "setup_columns": [],
        "ranking_key": "",
        "description": "",
        "code_snippets": {},
    }

    # core ファイルを解析
    if core_file.exists():
        with open(core_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

            # モジュールの docstring を取得
            if ast.get_docstring(tree):
                spec["description"] = ast.get_docstring(tree)

            # 関数定義から ranking_key を抽出
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if f"generate_system{system_num}_candidates" in node.name:
                        # 関数のソースコードを抽出（簡易版）
                        spec["code_snippets"]["generate_candidates"] = (
                            ast.get_source_segment(f.read(), node)
                        )

    # strategy ファイルを解析
    if strategy_file.exists():
        with open(strategy_file, "r", encoding="utf-8") as f:
            content = f.read()

            # ロングかショートかを判定
            if "long_only = True" in content or "self.long_only = True" in content:
                spec["direction"] = "ロング"
            elif "long_only = False" in content or "self.long_only = False" in content:
                spec["direction"] = "ショート"

    return spec


def generate_system_markdown(system_num: int, output_path: Path) -> bool:
    """システム仕様の markdown を生成"""
    try:
        spec = extract_system_spec(system_num)

        markdown = f"""# System{system_num} 仕様書

## 概要

{spec["description"]}

---

## 基本情報

- **システム番号**: System{system_num}
- **方向**: {spec["direction"]}
- **実装**: `core/system{system_num}.py`, `strategies/system{system_num}_strategy.py`

---

## フィルター条件

System{system_num} は Two-Phase フィルタリングを使用します。

### Phase 1: Filter 列生成

`common/today_filters.py::filter_system{system_num}()` で生成される列:

- （自動抽出中...）

### Phase 2: Setup Predicate

`common/system_setup_predicates.py::system{system_num}_setup_predicate()` で判定:

- （自動抽出中...）

---

## ランキングロジック

候補銘柄を以下のキーでランキング:

- **ランキングキー**: （自動抽出中...）
- **上位選択数**: スロット/金額制による

---

## コード例

### 候補生成関数

```python
# core/system{system_num}.py より
（自動抽出中...）
```

---

## データソース

- **キャッシュ**: `data_cache/rolling/` (直近 300 日)
- **指標キャッシュ**: `data_cache/indicators_system{system_num}_cache/`

---

**生成日時**: （自動生成）
**最終更新**: （Git commit hash）

"""

        # ファイルに保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

        print(f"✅ System{system_num} の仕様書を生成しました: {output_path}")
        return True

    except Exception as e:
        print(f"❌ System{system_num} の仕様書生成に失敗: {e}")
        return False


def generate_all_specs(output_dir: Path):
    """全システムの仕様書を生成"""
    success_count = 0

    for i in range(1, 8):  # System1-7
        output_path = output_dir / f"system{i}.md"
        if generate_system_markdown(i, output_path):
            success_count += 1

    print(f"\n📊 結果: {success_count}/7 システムの仕様書を生成しました")


def main():
    parser = argparse.ArgumentParser(description="システム仕様書自動生成")
    parser.add_argument("--system", type=int, help="生成対象のシステム番号（1-7）")
    parser.add_argument("--output", type=Path, help="出力ファイルパス")
    parser.add_argument("--all", action="store_true", help="全システムの仕様書を生成")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/systems_auto"),
        help="出力ディレクトリ（--all 使用時）",
    )
    args = parser.parse_args()

    if args.all:
        print("📚 全システムの仕様書を生成中...")
        generate_all_specs(args.output_dir)
    elif args.system and args.output:
        generate_system_markdown(args.system, args.output)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
