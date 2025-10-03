import pandas as pd

from core.system1 import generate_candidates_system1
from core.system6 import generate_candidates_system6

# シナリオ a: 最終日の setup 条件が偽で双方候補ゼロ


def _make_df_no_setup_latest():
    dates = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=5, freq="D")
    # 必要最小列 (system1 は close と rsi4 等; safety: include generic columns)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [10, 11, 12, 13, 14],
            "high": [10, 11, 12, 13, 14],
            "low": [9, 10, 11, 12, 13],
            "close": [10, 11, 12, 13, 14],
            "volume": [1000] * 5,
            # gating を確実に外すための列 (例: system1 フィルタを満たさない値を置く)
            "rsi4": [90, 90, 90, 90, 90],  # 高すぎてロング条件入らない想定
        }
    )
    return df


# シナリオ b: System6 用で return_6d が最終行 NaN -> fast でも full でも当日エントリなし


def _make_df_system6_nan():
    dates = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=7, freq="D")
    returns6 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, float("nan")]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [10, 11, 12, 13, 14, 15, 16],
            "high": [10, 11, 12, 13, 14, 15, 16],
            "low": [9, 10, 11, 12, 13, 14, 15],
            "close": [10, 11, 12, 13, 14, 15, 16],
            "volume": [1000] * 7,
            "return_6d": returns6,
            "adx7": [50] * 7,
        }
    )
    return df


def test_system1_negative_parity_empty_latest_day():
    prepared = {"TEST1": _make_df_no_setup_latest()}
    fast_by_date, _ = generate_candidates_system1(prepared, latest_only=True)
    full_by_date, _ = generate_candidates_system1(prepared, latest_only=False)

    # fast path dict empty
    assert fast_by_date == {} or all(len(v) == 0 for v in fast_by_date.values())
    # full path has no latest date key producing entries
    assert full_by_date == {} or all(len(v) == 0 for v in full_by_date.values())


def test_system6_negative_parity_no_entry_due_to_nan():
    prepared = {"TEST6": _make_df_system6_nan()}
    fast_by_date, _ = generate_candidates_system6(prepared, latest_only=True)
    full_by_date, _ = generate_candidates_system6(prepared, latest_only=False)

    assert fast_by_date == {} or all(len(v) == 0 for v in fast_by_date.values())
    assert full_by_date == {} or all(len(v) == 0 for v in full_by_date.values())
