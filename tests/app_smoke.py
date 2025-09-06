from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import app_integrated as app

settings = SimpleNamespace(
    RESULTS_DIR=Path("results"),
    LOGS_DIR=Path("logs"),
    DATA_CACHE_DIR=Path("data_cache"),
    THREADS_DEFAULT=1,
    ui=SimpleNamespace(default_capital=100000),
    logging=SimpleNamespace(level="INFO"),
)
app.get_settings = lambda create_dirs=True: settings


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


def _dummy_logger(_: SimpleNamespace) -> DummyLogger:
    return DummyLogger()


app.setup_logging = _dummy_logger
app.Notifier = lambda platform="auto": object()
app.get_spy_data_cached = lambda: None
app.render_integrated_tab = lambda *args, **kwargs: None
app.render_batch_tab = lambda *args, **kwargs: None

for i in range(1, 8):
    mod = ModuleType(f"app_system{i}")
    mod.run_tab = lambda *args, **kwargs: None
    sys.modules[f"app_system{i}"] = mod

app.main()
