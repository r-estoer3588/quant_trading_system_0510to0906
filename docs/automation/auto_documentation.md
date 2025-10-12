# Automated Documentation Generation

## 🎯 目的

コードから自動でドキュメントを生成し、常に最新の状態を維持。

## 📋 実装プラン

### Phase 1: API ドキュメント自動生成

```python
# tools/generate_api_docs.py
"""
Python コードから API ドキュメントを自動生成
"""
import ast
import inspect
from pathlib import Path
from typing import Dict, List

def extract_function_info(node: ast.FunctionDef) -> dict:
    """関数情報を抽出"""
    docstring = ast.get_docstring(node) or "No description"

    # パラメータ抽出
    params = []
    for arg in node.args.args:
        param_info = {'name': arg.arg}

        # 型ヒント
        if arg.annotation:
            param_info['type'] = ast.unparse(arg.annotation)

        params.append(param_info)

    # 戻り値の型
    return_type = None
    if node.returns:
        return_type = ast.unparse(node.returns)

    return {
        'name': node.name,
        'docstring': docstring,
        'params': params,
        'return_type': return_type,
        'is_public': not node.name.startswith('_')
    }

def extract_class_info(node: ast.ClassDef) -> dict:
    """クラス情報を抽出"""
    docstring = ast.get_docstring(node) or "No description"

    methods = []
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            methods.append(extract_function_info(item))

    return {
        'name': node.name,
        'docstring': docstring,
        'methods': methods,
        'is_public': not node.name.startswith('_')
    }

def generate_markdown(module_path: Path) -> str:
    """モジュールのMarkdownドキュメント生成"""
    content = module_path.read_text(encoding='utf-8')
    tree = ast.parse(content)

    md = f"# {module_path.stem}\n\n"

    # モジュールのdocstring
    module_doc = ast.get_docstring(tree)
    if module_doc:
        md += f"{module_doc}\n\n"

    # クラス
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if classes:
        md += "## Classes\n\n"
        for cls_node in classes:
            cls_info = extract_class_info(cls_node)
            if cls_info['is_public']:
                md += f"### `{cls_info['name']}`\n\n"
                md += f"{cls_info['docstring']}\n\n"

                if cls_info['methods']:
                    md += "**Methods:**\n\n"
                    for method in cls_info['methods']:
                        if method['is_public']:
                            params_str = ", ".join([p['name'] for p in method['params']])
                            md += f"- `{method['name']}({params_str})`"
                            if method['return_type']:
                                md += f" → `{method['return_type']}`"
                            md += f"\n  - {method['docstring'].split(chr(10))[0]}\n"
                    md += "\n"

    # 関数
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if functions:
        md += "## Functions\n\n"
        for func_node in functions:
            func_info = extract_function_info(func_node)
            if func_info['is_public']:
                params_str = ", ".join([f"{p['name']}: {p.get('type', 'Any')}" for p in func_info['params']])
                md += f"### `{func_info['name']}({params_str})`\n\n"
                if func_info['return_type']:
                    md += f"**Returns:** `{func_info['return_type']}`\n\n"
                md += f"{func_info['docstring']}\n\n"

    return md

def main():
    """全モジュールのドキュメント生成"""
    output_dir = Path("docs/api")
    output_dir.mkdir(exist_ok=True)

    # core/ モジュール
    for py_file in Path("core").glob("system*.py"):
        md_content = generate_markdown(py_file)
        output_file = output_dir / f"{py_file.stem}.md"
        output_file.write_text(md_content, encoding='utf-8')
        print(f"✅ Generated: {output_file}")

    # common/ の主要モジュール
    for module in ["cache_manager", "today_signals", "final_allocation"]:
        py_file = Path(f"common/{module}.py")
        if py_file.exists():
            md_content = generate_markdown(py_file)
            output_file = output_dir / f"{module}.md"
            output_file.write_text(md_content, encoding='utf-8')
            print(f"✅ Generated: {output_file}")

    # インデックス生成
    index_md = "# API Reference\n\n"
    index_md += "## Core Systems\n\n"
    for doc in sorted(output_dir.glob("system*.md")):
        index_md += f"- [{doc.stem}](./{doc.name})\n"

    index_md += "\n## Common Modules\n\n"
    for doc in sorted(output_dir.glob("*.md")):
        if not doc.stem.startswith("system") and doc.stem != "index":
            index_md += f"- [{doc.stem}](./{doc.name})\n"

    (output_dir / "index.md").write_text(index_md, encoding='utf-8')
    print(f"✅ Generated: {output_dir / 'index.md'}")

if __name__ == "__main__":
    main()
```

### Phase 2: システム仕様書自動生成

```python
# tools/generate_system_specs.py
"""
各システムの仕様書を自動生成
"""
import re
from pathlib import Path

def extract_system_specs(system_file: Path) -> dict:
    """システムファイルから仕様を抽出"""
    content = system_file.read_text(encoding='utf-8')

    spec = {
        'name': system_file.stem,
        'direction': 'Long' if 'ロング' in content or 'Long' in content else 'Short',
        'filters': [],
        'setup_conditions': [],
        'ranking_key': None
    }

    # フィルター条件抽出（Filter列の生成ロジックから）
    filter_pattern = r"df\['(\w+)'\]\s*([<>=!]+)\s*([\d.]+)"
    for match in re.finditer(filter_pattern, content):
        spec['filters'].append(f"{match.group(1)} {match.group(2)} {match.group(3)}")

    # ランキングキー抽出
    ranking_match = re.search(r"\.sort_values\(['\"](\w+)['\"]", content)
    if ranking_match:
        spec['ranking_key'] = ranking_match.group(1)

    return spec

def generate_spec_markdown(spec: dict) -> str:
    """仕様書Markdownを生成"""
    md = f"# {spec['name']}\n\n"
    md += f"**Direction:** {spec['direction']}\n\n"

    if spec['filters']:
        md += "## Filter Conditions\n\n"
        for f in spec['filters']:
            md += f"- {f}\n"
        md += "\n"

    if spec['ranking_key']:
        md += f"**Ranking Key:** `{spec['ranking_key']}`\n\n"

    return md

def main():
    output_dir = Path("docs/systems_auto")
    output_dir.mkdir(exist_ok=True)

    for system_file in Path("core").glob("system*.py"):
        spec = extract_system_specs(system_file)
        md_content = generate_spec_markdown(spec)

        output_file = output_dir / f"{system_file.stem}_spec.md"
        output_file.write_text(md_content, encoding='utf-8')
        print(f"✅ Generated: {output_file}")

if __name__ == "__main__":
    main()
```

### Phase 3: GitHub Actions で自動更新

```yaml
# .github/workflows/docs-auto-update.yml
name: Auto-Update Documentation

on:
  push:
    branches: [branch0906]
    paths:
      - "core/**.py"
      - "common/**.py"

jobs:
  update-docs:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Generate API docs
        run: python tools/generate_api_docs.py

      - name: Generate system specs
        run: python tools/generate_system_specs.py

      - name: Commit updated docs
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add docs/api/ docs/systems_auto/
          git diff-index --quiet HEAD || git commit -m "docs: Auto-update API documentation [skip ci]"
          git push
```

## 🔧 使い方

### 自動更新（GitHub Actions）

```powershell
# core/ や common/ を変更してプッシュ
git add core/system1.py
git commit -m "Update System1 logic"
git push

# → 自動でドキュメント更新・コミット
```

### 手動生成

```powershell
# API ドキュメント
python tools/generate_api_docs.py

# システム仕様書
python tools/generate_system_specs.py
```

## 📈 メリット

- ✅ 常に最新のドキュメント
- ✅ コードとドキュメントの乖離を防止
- ✅ オンボーディング資料の自動生成
- ✅ レビュー時の参照資料
