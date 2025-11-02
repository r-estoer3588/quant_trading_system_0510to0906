import pandas as pd

from core.system5 import generate_candidates_system5


def test_system5_latest_only_zero_candidates_sets_ranking_source():
    # 準備: 最新行の setup=False かつ 指標が無効値 → latest_only で候補0件にする
    idx = pd.to_datetime(["2025-10-30", "2025-10-31"])  # 最新が False なら除外対象
    df = pd.DataFrame(
        {
            "setup": [False, False],
            # System5 は ADX/ATR 系を使う。NaN や極端に低い値で predicate/manual を不成立に寄せる
            "adx7": [float("nan"), float("nan")],
            "atr_pct": [float("nan"), float("nan")],
            "Close": [1.0, 1.0],
        },
        index=idx,
    )

    prepared = {"SAMPLE": df}
    (by_date, df_out, diags) = generate_candidates_system5(  # type: ignore[misc]
        prepared, latest_only=True, include_diagnostics=True
    )

    # 0件であること
    assert diags.get("ranked_top_n_count", 0) == 0
    # latest_only 起因であることが診断に反映されること
    assert diags.get("ranking_source") == "latest_only"
    # 一意シンボル数も 0 で安定
    assert diags.get("setup_unique_symbols", 0) == 0
