from common.ui_components import extract_zero_reason_from_logs


def test_extract_from_selection_log():
    logs = [
        "[2025-09-20] info: something",
        "🛈 選定結果: 候補0件理由: no_candidate_dates",
    ]
    reason = extract_zero_reason_from_logs(logs)
    assert reason == "no_candidate_dates"


def test_extract_from_setup_log():
    logs = [
        "start",
        "🛈 セットアップ不成立: setup_fail: SPY close <= SMA100",
    ]
    reason = extract_zero_reason_from_logs(logs)
    assert reason == "setup_fail: SPY close <= SMA100"


def test_no_reason_returns_none():
    logs = ["start", "processing", "done"]
    assert extract_zero_reason_from_logs(logs) is None
