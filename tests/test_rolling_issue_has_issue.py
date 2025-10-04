from common.cache_warnings import get_rolling_issue_aggregator, report_rolling_issue


def test_has_issue_basic():
    agg = get_rolling_issue_aggregator()
    # 前提: まだ報告されていないシンボル
    assert not agg.has_issue("missing_rolling", "SYM_TEST")
    report_rolling_issue("missing_rolling", "SYM_TEST", "fallback")
    assert agg.has_issue("missing_rolling", "SYM_TEST")
    # 別カテゴリは未登録
    assert not agg.has_issue("manual_rebuild", "SYM_TEST")
