"""Public API export regression guard for ui_components.

目的: テストが内部実装詳細に侵入せず、公開API境界が不用意に変わっていないかを検知する。
このテストは *関数名の存在* のみを確認し、実装詳細や副作用には立ち入らない。

CI/ローカルのどちらでも高速に終了する（import のみ）。
"""

from __future__ import annotations

import inspect

from common import ui_components as uic

# 公開として維持したい関数シグネチャ名（引数までは縛らない）
EXPECTED_EXPORTS = {
    "run_backtest_app",
    "prepare_backtest_data",
    "fetch_data",
    "show_results",
    "show_signal_trade_summary",
    "clean_date_column",
    "save_signal_and_trade_logs",
}

# 内部利用扱いで外へ依存を増やしたくないもの（露出していたら通知）
INTERNAL_SHOULD_NOT_EXPORT = {
    # summarize_results は内部利用想定。現状は globals に存在するが __all__ 管理が無いため情報目的のみ。
    "summarize_results",
}

# 追加: 公開 API は docstring を持つことを推奨
MIN_DOCSTRING_RATIO = 0.8  # 推奨しきい値（情報目的、FAIL ではない）


def test_expected_public_api_presence():
    missing = [name for name in EXPECTED_EXPORTS if not hasattr(uic, name)]
    assert not missing, f"公開APIが欠落: {missing}"


def test_internal_functions_not_exposed():
    # __all__ 未運用のため、単純 hasattr で失敗させるとノイズが大きい。
    # ここでは "漏れているならテストは PASS だが pytest -q 出力に info を表示" だけ行う。
    leaked = [name for name in INTERNAL_SHOULD_NOT_EXPORT if hasattr(uic, name)]
    if leaked:  # 許容: テスト内軽量分岐
        print(f"[INFO] 内部想定関数が現状公開スコープに存在: {leaked}")


ALLOWLIST_PREFIXES = (
    "display_",
    "generate_",
    "get_",
    "run_backtest_",
    "load_",
    "log_",
    "round_",
    "base_",
    "default_",
    "extract_",
    "safe_",
)

EXACT_ALLOWLIST = {"tr", "safe_filename", "extract_zero_reason_from_logs"}


def test_no_unexpected_new_callables():
    public_callables = {name for name, obj in vars(uic).items() if callable(obj) and not name.startswith("_")}
    filtered = {
        n
        for n in public_callables
        if not (
            n in EXPECTED_EXPORTS
            or n in INTERNAL_SHOULD_NOT_EXPORT
            or any(n.startswith(pref) for pref in ALLOWLIST_PREFIXES)
        )
    }
    # 型ヒント由来 / import 由来の名前を除外
    noise = {"Any", "ThreadPoolExecutor", "as_completed", "cast"}
    unexpected = sorted(filtered - noise - EXACT_ALLOWLIST)
    assert not unexpected, f"監視対象外の callable が追加: {unexpected}"


def test_public_api_docstring_coverage():
    """公開 API に Docstring が存在するかの情報テスト（不足しても失敗させない）。"""
    docs = []
    missing = []
    for name in EXPECTED_EXPORTS:
        obj = getattr(uic, name, None)
        if obj is None:
            continue
        ds = inspect.getdoc(obj)
        if ds:
            docs.append(name)
        else:
            missing.append(name)
    ratio = len(docs) / max(1, len(EXPECTED_EXPORTS))
    if missing:  # 許容: テスト内軽量分岐
        print(f"[INFO] Docstring 無し: {missing} (ratio={ratio:.2f})")
    if ratio < MIN_DOCSTRING_RATIO:
        print(f"[INFO] Docstring カバレッジ比 {ratio:.2f} < 推奨 {MIN_DOCSTRING_RATIO}")


def test_callable_prefix_classification():
    """公開 callable を prefix ベースに分類し情報表示（監視用）。"""
    categories = {p: [] for p in ALLOWLIST_PREFIXES}
    uncategorized = []
    for name, obj in vars(uic).items():
        if not callable(obj) or name.startswith("_"):
            continue
        matched = False
        for p in ALLOWLIST_PREFIXES:
            if name.startswith(p):
                categories[p].append(name)
                matched = True
                break
        if not matched and name not in EXPECTED_EXPORTS and name not in INTERNAL_SHOULD_NOT_EXPORT:
            if name not in EXACT_ALLOWLIST:
                uncategorized.append(name)
    # 情報出力
    print("[INFO] Callable prefix classification summary:")
    for p, items in categories.items():
        if items:
            print(f"  {p}: {len(items)}")
    if uncategorized:
        print(f"[INFO] 未分類 callable: {sorted(uncategorized)[:15]} ... (total {len(uncategorized)})")
