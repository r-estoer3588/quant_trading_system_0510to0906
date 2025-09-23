import os
from common.utils import is_today_run


def test_is_today_run_true(monkeypatch):
    monkeypatch.setenv("TODAY_RUN", "1")
    assert is_today_run() is True


def test_is_today_run_false(monkeypatch):
    monkeypatch.delenv("TODAY_RUN", raising=False)
    assert is_today_run() is False


def test_is_today_run_true_values(monkeypatch):
    for v in ("1", "true", "yes", "True", "YES"):
        monkeypatch.setenv("TODAY_RUN", v)
        assert is_today_run() is True


def test_is_today_run_false_values(monkeypatch):
    for v in ("0", "false", "no", "", "random"):
        monkeypatch.setenv("TODAY_RUN", v)
        assert is_today_run() is False
