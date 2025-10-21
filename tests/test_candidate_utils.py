import pandas as pd

from common.candidate_utils import normalize_candidate_frame, validate_candidate_frame


def test_normalize_close_from_lowercase():
    df = pd.DataFrame({"symbol": ["A"], "close": [100.5]})
    norm = normalize_candidate_frame(df, system_name="systemx")
    assert "Close" in norm.columns
    # Compare value directly to avoid type conversion issues in static analysis
    assert norm.at[0, "Close"] == 100.5


def test_validate_candidate_frame_reports_missing():
    df = pd.DataFrame({"symbol": ["A"], "close": [None]})
    norm = normalize_candidate_frame(df, system_name="systemx")
    diag = validate_candidate_frame(norm)
    assert diag["rows_total"] == 1
    assert "Close" in diag["missing_counts"]
    assert diag["rows_missing_entry"] == 1
