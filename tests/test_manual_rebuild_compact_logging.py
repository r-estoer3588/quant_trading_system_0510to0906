from unittest.mock import Mock

from apps.app_today_signals import _log_manual_rebuild_notice


def test_manual_rebuild_notice_compact(monkeypatch, caplog):
    monkeypatch.setenv("COMPACT_TODAY_LOGS", "1")
    detail = {
        "status": "nan_columns",
        "rows_before": 70,
        "missing_required": "",
        "missing_optional": "ema20",
        "nan_columns": "sma25:30%",
    }
    # ログ関数は呼ばれない（aggregator 経由）想定
    mock_log = Mock()
    msg = _log_manual_rebuild_notice("TESTX", detail, log_fn=mock_log)
    assert "rolling未整備" in msg
    # 直接ログ出力は最初の数件は WARNING になるため caplog に何らか入る
    assert any("manual_rebuild" in r.message for r in caplog.records)
    # 直接 log_fn は呼ばれない
    mock_log.assert_not_called()


def test_manual_rebuild_notice_verbose(monkeypatch, caplog):
    monkeypatch.delenv("COMPACT_TODAY_LOGS", raising=False)
    detail = {
        "status": "nan_columns",
        "rows_before": 70,
        "missing_required": "",
        "missing_optional": "ema20",
        "nan_columns": "sma25:30%",
    }
    mock_log = Mock()
    msg = _log_manual_rebuild_notice("TESTY", detail, log_fn=mock_log)
    assert "rolling未整備" in msg
    mock_log.assert_called_once()
