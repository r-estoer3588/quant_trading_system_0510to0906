import pandas as pd

from core.system4 import generate_candidates_system4


def test_system4_latest_only_zero_candidates_sets_ranking_source():
    # 準備: setup=False の最終行を持つシンボル → latest_only で候補0件にする
    idx = pd.to_datetime(
        ["2025-10-30", "2025-10-31"]
    )  # 2行あっても最新がFalseなら除外される
    df = pd.DataFrame(
        {
            "setup": [False, False],
            "rsi4": [50.0, 49.0],  # 値は任意（setup False を優先）
            "Close": [100.0, 101.0],
        },
        index=idx,
    )

    prepared = {"SAMPLE": df}
    (by_date, df_out, diags) = generate_candidates_system4(
        prepared, latest_only=True, include_diagnostics=True
    )

    # 0件であること
    assert diags.get("ranked_top_n_count", 0) == 0
    # latest_only 起因であることが診断に反映されること
    assert diags.get("ranking_source") == "latest_only"
    # 一意シンボル数も 0 で安定
    assert diags.get("setup_unique_symbols", 0) == 0
