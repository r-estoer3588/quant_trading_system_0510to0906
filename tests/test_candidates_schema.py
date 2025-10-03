import pandas as pd

from core import system6, system7

# 簡易ダミー DataFrame 生成ヘルパ


def _df(setup: bool = True):
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102],
            "Open": [100, 100, 101],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
        },
        index=idx,
    )
    # system7 用列
    df["atr50"] = 2.5
    df["ATR50"] = 2.5
    df["min_50"] = df["Low"]
    df["max_70"] = df["High"]
    # system6 用列 (return_6d, atr10, setupなど)
    df["return_6d"] = [0.05, 0.04, 0.03]
    df["atr10"] = [1.0, 1.0, 1.0]
    df["filter"] = 1
    df["setup"] = 1 if setup else 0
    return df


def test_system7_latest_only_schema():
    prepared = {"SPY": _df(True)}
    out, df_fast = system7.generate_candidates_system7(prepared, latest_only=True)
    assert isinstance(out, dict)
    if out:  # 立っている場合
        for k, v in out.items():
            assert isinstance(k, pd.Timestamp)
            assert isinstance(v, dict)
            assert "SPY" in v
            assert isinstance(v["SPY"], dict)
            assert "entry_date" in v["SPY"]
            assert "ATR50" in v["SPY"] or "atr50" in v["SPY"]


def test_system7_full_schema():
    prepared = {"SPY": _df(True)}
    out, _ = system7.generate_candidates_system7(prepared, latest_only=False)
    assert isinstance(out, dict)
    for k, v in out.items():
        assert isinstance(k, pd.Timestamp)
        assert isinstance(v, dict)
        assert "SPY" in v
        assert isinstance(v["SPY"], dict)


def test_system6_latest_only_schema():
    prepared = {"AAA": _df(True), "BBB": _df(False)}
    out, df_fast = system6.generate_candidates_system6(prepared, latest_only=True)
    assert isinstance(out, dict)
    if out:
        for k, v in out.items():
            assert isinstance(k, pd.Timestamp)
            assert isinstance(v, dict)
            for sym, payload in v.items():
                assert isinstance(sym, str)
                assert isinstance(payload, dict)
                assert "return_6d" in payload


def test_system6_full_schema():
    prepared = {"AAA": _df(True)}
    out, _ = system6.generate_candidates_system6(prepared, latest_only=False)
    assert isinstance(out, dict)
    for k, v in out.items():
        assert isinstance(k, pd.Timestamp)
        assert isinstance(v, dict)
