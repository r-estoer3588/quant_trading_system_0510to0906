from common.candidate_utils import normalize_candidate_frame, validate_candidate_frame
from tools.debug_allocation import create_strategies, generate_simple_test_signals


def test_pipeline_normalization_smoke():
    strategies = create_strategies()
    # Small deterministic universe used in tests
    universe = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    per_system = generate_simple_test_signals(strategies, universe, test_mode="mini")

    for sys_name, df in per_system.items():
        norm = normalize_candidate_frame(df, system_name=sys_name)
        # Check canonical columns exist
        assert "symbol" in norm.columns
        assert "Close" in norm.columns
        assert "atr10" in norm.columns
        assert "entry_price" in norm.columns
        # Validate returns a diagnostic with consistent row counts
        diag = validate_candidate_frame(norm)
        assert diag["rows_total"] == len(norm)
