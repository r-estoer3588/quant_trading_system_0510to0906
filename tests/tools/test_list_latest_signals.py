from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import types

from _pytest.monkeypatch import MonkeyPatch


def _install_fake_settings_module(
    signals_dir: Path, results_dir: Path
) -> types.ModuleType:
    """Install a fake config.settings module exposing get_settings()."""
    fake = types.ModuleType("config.settings")

    class _Outputs:
        def __init__(self, s: Path, r: Path) -> None:
            self.signals_dir = Path(s)
            self.results_csv_dir = Path(r)

    class _Settings:
        def __init__(self, s: Path, r: Path) -> None:
            self.outputs = _Outputs(s, r)

    def get_settings(create_dirs: bool = False) -> _Settings:  # noqa: FBT001, FBT002
        return _Settings(signals_dir, results_dir)

    fake.get_settings = get_settings  # type: ignore[attr-defined]
    # Ensure package structure exists
    pkg = types.ModuleType("config")
    sys.modules.setdefault("config", pkg)
    sys.modules["config.settings"] = fake
    return fake


def _write_csv(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("col1,col2\n")
        for i in range(rows):
            f.write(f"{i},x\n")


def test_specific_date_outputs(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    signals_dir = tmp_path / "signals"
    results_dir = tmp_path / "results"
    # Prepare files
    date = "2025-10-31"
    csv_path = signals_dir / f"signals_final_{date}.csv"
    _write_csv(csv_path, 2)
    validation_path = results_dir / "validation" / f"validation_report_{date}.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

    # Inject fake settings
    _install_fake_settings_module(signals_dir, results_dir)

    # Import the tool after injection
    from tools import list_latest_signals as tool

    # Run main with --date
    argv_backup = sys.argv[:]
    try:
        sys.argv = ["list_latest_signals.py", "--date", date]
        buf = io.StringIO()
        # Capture stdout
        monkeypatch.setattr(sys, "stdout", buf)
        rc = tool.main()
        out = buf.getvalue()
    finally:
        sys.argv = argv_backup

    assert rc == 0
    assert "File:" in out and str(csv_path) in out
    assert "Rows: 2" in out
    assert str(validation_path) in out


def test_latest_top2(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    signals_dir = tmp_path / "signals"
    results_dir = tmp_path / "results"

    # two files with different mtimes
    csv_new = signals_dir / "signals_final_2025-10-31.csv"
    csv_old = signals_dir / "signals_final_2025-10-24.csv"
    _write_csv(csv_old, 4)
    _write_csv(csv_new, 1)
    # adjust mtimes: make old older
    os.utime(csv_old, (csv_old.stat().st_atime - 1000, csv_old.stat().st_mtime - 1000))

    _install_fake_settings_module(signals_dir, results_dir)
    from tools import list_latest_signals as tool

    argv_backup = sys.argv[:]
    try:
        sys.argv = ["list_latest_signals.py", "--latest", "2"]
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        rc = tool.main()
        out = buf.getvalue()
    finally:
        sys.argv = argv_backup

    assert rc == 0
    # First should be the newer file
    assert "LatestFile:" in out and str(csv_new) in out
    # Second should list the older one
    assert "File#2:" in out and str(csv_old) in out
    # Row counts
    assert "Rows: 1" in out
    assert "Rows: 4" in out
