import json
from pathlib import Path

from common.perf_snapshot import PerfSnapshot


def test_perf_snapshot_writes_file(tmp_path, monkeypatch):
    # 強制的に logs/perf_snapshots を tmp 配下に向けるため settings 偽装
    class DummySettings:
        LOGS_DIR = str(tmp_path / "logs")

    def fake_get_settings(create_dirs=False):  # noqa: D401
        if create_dirs:
            (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        return DummySettings()

    monkeypatch.setattr("common.perf_snapshot.get_settings", fake_get_settings)

    ps = PerfSnapshot(enabled=True)
    with ps.run(latest_only=True):
        ps.mark_system_start("system1")
        ps.mark_system_end("system1", symbol_count=0, candidate_count=0)

    assert ps.output_path is not None
    data = json.loads(Path(ps.output_path).read_text(encoding="utf-8"))
    for key in ["timestamp", "latest_only", "total_time_sec", "per_system", "cache_io"]:
        assert key in data
    assert data["latest_only"] is True
    assert "system1" in data["per_system"]
    assert "candidate_count" in data["per_system"]["system1"]
    assert "read_feather" in data["cache_io"]
