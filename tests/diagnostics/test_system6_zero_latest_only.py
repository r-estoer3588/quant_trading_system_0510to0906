import pandas as pd

from core.system6 import generate_candidates_system6


def test_system6_latest_only_zero_candidates_sets_ranking_source():
    # 準備: 最新行の setup=False → latest_only で候補0件にする
    idx = pd.to_datetime(["2025-10-30", "2025-10-31"])  # 最新が False なら除外対象
    df = pd.DataFrame(
        {
            "setup": [False, False],
            # System6 は return_6d を参照するが、ゼロ候補化の主因は setup False で十分
            "return_6d": [float("nan"), float("nan")],
            "Close": [100.0, 100.0],
        },
        index=idx,
    )

    prepared = {"SAMPLE": df}
    (by_date, df_out, diags) = generate_candidates_system6(  # type: ignore[misc]
        prepared, latest_only=True, include_diagnostics=True
    )

    # 0件であること
    assert diags.get("ranked_top_n_count", 0) == 0
    # latest_only 起因であることが診断に反映されること
    assert diags.get("ranking_source") == "latest_only"
    # 一意シンボル数も 0 で安定
    assert diags.get("setup_unique_symbols", 0) == 0
