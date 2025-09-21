from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from common import indicators_precompute as ip


def test_precompute_shared_indicators_enriches_frames_and_writes_cache(
    tmp_path, monkeypatch
):
    """precompute_shared_indicators should add common factors and persist them."""

    settings_stub = SimpleNamespace(outputs=SimpleNamespace(signals_dir=tmp_path))
    monkeypatch.setattr(ip, "get_settings", lambda create_dirs=True: settings_stub)

    written_paths: list[Path] = []

    def _fake_to_feather(self, path, *args, **kwargs):  # type: ignore[override]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        written_paths.append(target)
        target.write_bytes(b"")

    def _fake_to_parquet(self, path, *args, **kwargs):  # type: ignore[override]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        written_paths.append(target)
        target.write_bytes(b"")

    monkeypatch.setattr(pd.DataFrame, "to_feather", _fake_to_feather, raising=False)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet, raising=False)

    periods = 250
    dates = pd.date_range("2022-01-03", periods=periods, freq="B")
    prices = np.linspace(100, 200, periods)
    volumes = np.linspace(1_000_000, 2_000_000, periods)
    raw = pd.DataFrame(
        {
            "date": dates,
            "open": prices + 1,
            "high": prices + 2,
            "low": prices - 2,
            "close": prices,
            "volume": volumes,
        }
    )

    enriched_map = ip.precompute_shared_indicators({"AAA": raw}, parallel=False)
    enriched = enriched_map["AAA"]

    expected = set(ip.PRECOMPUTED_INDICATORS)
    assert expected.issubset(set(enriched.columns))

    last_row = enriched.iloc[-1]
    numeric_cols = [
        "ATR10",
        "SMA25",
        "ROC200",
        "RSI3",
        "ADX7",
        "DollarVolume20",
        "AvgVolume50",
        "ATR_Ratio",
        "Return_3D",
        "6D_Return",
        "HV50",
        "min_50",
        "max_70",
    ]
    for col in numeric_cols:
        assert col in enriched.columns
        assert pd.notna(last_row[col]), f"{col} should have a numeric value"

    assert "UpTwoDays" in enriched.columns
    assert "TwoDayUp" in enriched.columns
    pd.testing.assert_series_equal(
        enriched["UpTwoDays"], enriched["TwoDayUp"], check_names=False
    )
    assert enriched["UpTwoDays"].dtype == bool

    ratio = enriched["ATR_Ratio"].iloc[-1]
    pct = enriched["ATR_Pct"].iloc[-1]
    assert pd.notna(ratio)
    assert pd.notna(pct)
    np.testing.assert_allclose(ratio, pct)

    cache_dir = tmp_path / "shared_indicators"
    cache_file = cache_dir / "AAA.feather"
    assert cache_dir.exists()
    assert cache_file.exists()
    assert any(path.suffix == ".feather" for path in written_paths)
