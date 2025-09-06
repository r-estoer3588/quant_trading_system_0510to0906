import os
from config.settings import get_settings


def test_get_settings_reads_yaml_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    s = get_settings(create_dirs=False)
    assert s.outputs is not None
    assert str(s.RESULTS_DIR).endswith("results")
    assert s.risk.max_positions >= 1
    assert s.logging.level in {"INFO", "DEBUG", "WARN", "ERROR"}

