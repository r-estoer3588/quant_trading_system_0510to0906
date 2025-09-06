from __future__ import annotations

from pathlib import Path


def _get_function_source_text(path: Path, func_name: str) -> str:
    # BOM を除去しつつテキスト抽出（AST は BOM/全角コメントで壊れやすい）
    src = path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")
    key = f"def {func_name}("
    i = src.find(key)
    if i < 0:
        return ""
    # 次の top-level 定義までをざっくり抽出
    j = src.find("\ndef ", i + 1)
    k = src.find("\nclass ", i + 1)
    end = len(src)
    for pos in (j, k):
        if pos != -1:
            end = min(end, pos)
    return src[i:end]


def test_trade_logs_are_in_expander_in_common_ui_components():
    path = Path("common/ui_components.py")
    assert path.exists(), "common/ui_components.py が見つかりません"
    func_src = _get_function_source_text(path, "run_backtest_with_logging")
    # UI 契約: 取引ログはエクスパンダー内で表示されること
    assert func_src and "expander(" in func_src, "取引ログの表示が expander で包まれていません"
    # なるべく見やすい text_area を使っていること（任意だが回 regress 用）
    assert "text_area(" in func_src, "取引ログの表示に text_area が使われていません"


def test_trade_logs_are_in_expander_in_common_ui_bridge():
    # System2 向けログもエクスパンダーであることを確認
    path = Path("common/ui_bridge.py")
    assert path.exists(), "common/ui_bridge.py が見つかりません"
    src = path.read_text(encoding="utf-8", errors="ignore")
    assert "st.expander(" in src or ".expander(" in src, "ui_bridge のログ表示が expander で包まれていません"
