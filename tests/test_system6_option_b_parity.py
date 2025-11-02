from __future__ import annotations

import pandas as pd

from strategies.system6_strategy import System6Strategy


def _make_df(rows: int = 5, last_return6d: float = 0.25) -> pd.DataFrame:
    dates = pd.date_range("2025-10-20", periods=rows, freq="B")
    df = pd.DataFrame(
        {
            "Close": [30.0] * rows,
            # System6 の主要ランク指標
            "return_6d": [0.05] * (rows - 1) + [last_return6d],
            # 仕様に必要な基本列（最低限）
            "atr10": [1.0] * rows,
            "dollarvolume50": [20_000_000.0] * rows,
            "hv50": [0.20] * rows,
        },
        index=dates,
    )
    # setup/filter 互換（コア側でも再計算されるが列の存在だけ担保）
    df["filter"] = True
    df["setup"] = True
    return df


def test_system6_option_b_parity_latest_only():
    prepared = {"AAA": _make_df()}

    strat = System6Strategy()

    # OFF: 明示的に無効化
    by_date_off, df_off = strat.generate_candidates(
        prepared,
        latest_only=True,
        top_n=5,
        use_option_b_utils=False,
    )

    # ON: 明示的に有効化
    by_date_on, df_on = strat.generate_candidates(
        prepared,
        latest_only=True,
        top_n=5,
        use_option_b_utils=True,
    )

    # 候補 DataFrame の形状は一致
    assert (df_off is not None) and (df_on is not None)
    assert len(df_off) == len(df_on)

    # by-date 構造とシンボル集合が一致
    assert set(by_date_off.keys()) == set(by_date_on.keys())
    for dt in by_date_off.keys():
        assert set(by_date_off[dt].keys()) == set(by_date_on[dt].keys())
