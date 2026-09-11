from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from scripts import export_alpaca_snapshot as exporter


def _write_status(root, *, generated_at: datetime) -> None:
    status_dir = root / "results_csv"
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "soft_trailing_status/v1",
        "generated_at": generated_at.isoformat(),
        "paper": True,
        "systems": ["system1", "system4"],
        "rows": [
            {
                "symbol": "AMCR",
                "system": "system1",
                "hwm": 46.95,
                "trail_pct": 0.25,
                "stop_price": 35.2125,
            },
            {
                "symbol": "AXSM",
                "system": "system4",
                "hwm": 210.86,
                "trail_pct": 0.20,
                "stop_price": 168.688,
            },
        ],
    }
    (status_dir / "soft_trailing_status_20260911.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_fresh_shared_status_becomes_soft_hwm_state(monkeypatch, tmp_path):
    now = datetime(2026, 9, 11, 1, 0, tzinfo=timezone.utc)
    _write_status(tmp_path, generated_at=now - timedelta(minutes=30))
    monkeypatch.setattr(exporter, "ROOT", tmp_path)

    state = exporter._load_fresh_soft_status(now=now)

    assert state["AMCR"]["highest_price"] == 46.95
    assert state["AMCR"]["trailing_stop_pct"] == 0.25
    assert state["AXSM"]["highest_price"] == 210.86
    assert state["AXSM"]["trailing_stop_pct"] == 0.20
    assert state["AMCR"]["source"] == "soft_trailing_status"


def test_stale_shared_status_is_not_authority(monkeypatch, tmp_path):
    now = datetime(2026, 9, 11, 1, 0, tzinfo=timezone.utc)
    _write_status(tmp_path, generated_at=now - timedelta(minutes=91))
    monkeypatch.setattr(exporter, "ROOT", tmp_path)

    assert exporter._load_fresh_soft_status(now=now) == {}


def test_empty_tracked_state_falls_back_to_fresh_shared_status(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "trailing_stops.json").write_text("{}\n", encoding="utf-8")
    _write_status(tmp_path, generated_at=now - timedelta(minutes=5))
    monkeypatch.setattr(exporter, "ROOT", tmp_path)

    state = exporter._load_soft_state()

    assert state["AMCR"]["highest_price"] == 46.95
    assert state["AXSM"]["trailing_stop_pct"] == 0.20
