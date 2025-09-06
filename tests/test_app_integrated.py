from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import app_integrated


class DummyStreamlit:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_page_config(self, **kwargs):
        pass

    def title(self, *args, **kwargs):
        pass

    def expander(self, *args, **kwargs):
        return self

    def columns(self, n):
        return [self for _ in range(n)]

    def write(self, *args, **kwargs):
        pass

    def tabs(self, labels):
        return [self for _ in labels]

    def subheader(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass

    def checkbox(self, *args, **kwargs):
        return False


def test_app_integrated_main_runs(monkeypatch):
    dummy_st = DummyStreamlit()
    monkeypatch.setattr(app_integrated, "st", dummy_st)

    settings = SimpleNamespace(
        RESULTS_DIR=Path("results"),
        LOGS_DIR=Path("logs"),
        DATA_CACHE_DIR=Path("data_cache"),
        THREADS_DEFAULT=1,
        ui=SimpleNamespace(default_capital=100000),
        logging=SimpleNamespace(level="INFO"),
    )
    monkeypatch.setattr(app_integrated, "get_settings", lambda create_dirs=True: settings)

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass

        def exception(self, *args, **kwargs):
            pass

    monkeypatch.setattr(app_integrated, "setup_logging", lambda s: DummyLogger())
    monkeypatch.setattr(app_integrated, "Notifier", lambda platform="discord": object())
    monkeypatch.setattr(app_integrated, "get_spy_data_cached", lambda: None)
    monkeypatch.setattr(app_integrated, "render_integrated_tab", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_integrated, "render_batch_tab", lambda *args, **kwargs: None)

    for i in range(1, 8):
        mod = ModuleType(f"app_system{i}")
        mod.run_tab = lambda *args, **kwargs: None
        sys.modules[f"app_system{i}"] = mod

    app_integrated.main()
