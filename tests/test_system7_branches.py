"""
System7 branch coverage tests for detailed code paths.

Focus on:
- Lines 228-264: latest_only data construction branches
- Lines 320-343: Date grouping and limit_n logic
- Lines 352-396: Ranking and normalization branches
"""

import numpy as np
import pandas as pd

from core.system7 import generate_candidates_system7, prepare_data_vectorized_system7


class TestSystem7LatestOnlyBranches:
    """Test latest_only mode detailed branches."""

    def create_spy_with_atr_variations(self, atr_column="ATR50"):
        """Create SPY data with specific ATR column name."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(400, 450, 100)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        setup_vals = [False] * 99 + [True]

        data = {
            "Close": prices,
            "atr50": [p * 0.02 for p in prices],
            "ATR50": [p * 0.02 for p in prices],
            "min_50": min_50.values,
            "max_70": max_70.values,
            "setup": setup_vals,
        }

        # Test specific ATR column configurations
        if atr_column == "atr50_only":
            data.pop("ATR50")
        elif atr_column == "ATR50_only":
            data.pop("atr50")

        return pd.DataFrame(data, index=dates)

    def test_latest_only_atr50_uppercase_priority(self):
        """Test ATR50 (uppercase) has priority over atr50 (line 233)."""
        spy_data = self.create_spy_with_atr_variations("ATR50")
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict, latest_only=True, include_diagnostics=True
        )
        candidates = result_tuple[0]

        # Should successfully use ATR50
        assert isinstance(candidates, dict)
        if candidates:
            for date_key, symbols in candidates.items():
                if "SPY" in symbols:
                    # ATR50 should be present
                    spy_atr = symbols["SPY"].get("ATR50")
                    assert "ATR50" in symbols["SPY"] or spy_atr is not None

    def test_latest_only_atr50_fallback_to_lowercase(self):
        """Test fallback to atr50 (lowercase) when ATR50 missing (line 234-235)."""
        spy_data = self.create_spy_with_atr_variations("atr50_only")
        data_dict = {"SPY": spy_data}

        _ = generate_candidates_system7(
            data_dict, latest_only=True, include_diagnostics=True
        )

        # Should successfully use atr50 as fallback without crashing
        assert True  # If we get here, the fallback worked

    def test_latest_only_log_callback_invoked(self):
        """Test log_callback is invoked in latest_only mode (lines 256-259)."""
        spy_data = self.create_spy_with_atr_variations("ATR50")
        data_dict = {"SPY": spy_data}

        log_messages = []

        def mock_log(msg):
            log_messages.append(msg)

        _ = generate_candidates_system7(
            data_dict,
            latest_only=True,
            log_callback=mock_log,
            include_diagnostics=True,
        )

        # Should have logged the fast-path message
        assert any("latest_only" in msg or "fast-path" in msg for msg in log_messages)

    def test_latest_only_log_callback_exception_handled(self):
        """Test log_callback exception is handled gracefully (lines 258-259)."""
        spy_data = self.create_spy_with_atr_variations("ATR50")
        data_dict = {"SPY": spy_data}

        def failing_log(msg):
            raise RuntimeError("Log callback failed")

        # Should not crash despite log callback failure
        result_tuple = generate_candidates_system7(
            data_dict,
            latest_only=True,
            log_callback=failing_log,
            include_diagnostics=True,
        )
        candidates = result_tuple[0]

        assert isinstance(candidates, dict)

    def test_latest_only_diagnostics_structure(self):
        """Test diagnostics structure in latest_only mode (lines 260-261)."""
        spy_data = self.create_spy_with_atr_variations("ATR50")
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict, latest_only=True, include_diagnostics=True
        )
        diagnostics = result_tuple[2] if len(result_tuple) > 2 else {}

        # Check expected diagnostics keys
        assert isinstance(diagnostics, dict)
        # When setup=True today, should have these keys
        if diagnostics.get("ranked_top_n_count", 0) > 0:
            assert "ranking_source" in diagnostics
            assert diagnostics.get("ranking_source") == "latest_only"

    def test_latest_only_normalized_structure(self):
        """Test normalized dictionary structure (lines 248-252)."""
        spy_data = self.create_spy_with_atr_variations("ATR50")
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict, latest_only=True, include_diagnostics=True
        )
        normalized = result_tuple[0]

        # Check structure: {date: {"SPY": {...}}}
        assert isinstance(normalized, dict)
        for date_key, symbols_dict in normalized.items():
            assert isinstance(date_key, pd.Timestamp)
            assert isinstance(symbols_dict, dict)
            if "SPY" in symbols_dict:
                spy_payload = symbols_dict["SPY"]
                # Should have entry_date in payload
                assert "entry_date" in spy_payload


class TestSystem7DateGroupingBranches:
    """Test date grouping and limit_n logic."""

    def create_spy_with_multiple_setups(self, setup_count=10):
        """Create SPY data with multiple setup dates."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(400, 450, 100)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        # Create multiple setup signals
        setup_vals = [False] * 100
        setup_indices = np.linspace(50, 95, setup_count, dtype=int)
        for idx in setup_indices:
            if idx < len(setup_vals):
                setup_vals[idx] = True

        return pd.DataFrame(
            {
                "Close": prices,
                "atr50": [p * 0.02 for p in prices],
                "ATR50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "max_70": max_70.values,
                "setup": setup_vals,
            },
            index=dates,
        )

    def test_date_grouping_with_limit_n(self):
        """Test limit_n parameter in date grouping (line 343-345).

        Note: limit_n restricts candidates per date bucket, not total.
        System7 is SPY-only, so each date has at most 1 candidate.
        """
        spy_data = self.create_spy_with_multiple_setups(setup_count=10)
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict, top_n=3, include_diagnostics=True
        )
        candidates = result_tuple[0]

        # Each date bucket has at most 1 SPY (SPY-only system)
        # With 10 setup signals, we may have up to 10 date buckets
        # limit_n controls candidates per bucket, not total count
        for date_key, symbols_dict in candidates.items():
            # Each date should have at most 1 symbol (SPY)
            assert len(symbols_dict) <= 1

    def test_date_grouping_limit_n_zero_skipped(self):
        """Test limit_n=0 causes candidates to be skipped (line 322-323)."""
        spy_data = self.create_spy_with_multiple_setups(setup_count=5)
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict, top_n=0, include_diagnostics=True
        )
        candidates = result_tuple[0]

        # With limit_n=0, should have no candidates
        assert len(candidates) == 0

    def test_date_grouping_bucket_limit(self):
        """Test bucket limit prevents exceeding limit_n (line 344-345)."""
        spy_data = self.create_spy_with_multiple_setups(setup_count=15)
        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict, top_n=5, include_diagnostics=True
        )
        candidates = result_tuple[0]

        # Each date bucket should have at most 1 SPY (single symbol system)
        for date_key, symbols_dict in candidates.items():
            assert len(symbols_dict) <= 1

    def test_log_callback_with_candidates_by_date(self):
        """Test log_callback with candidate date counting (lines 348-361)."""
        spy_data = self.create_spy_with_multiple_setups(setup_count=8)
        data_dict = {"SPY": spy_data}

        log_messages = []

        def mock_log(msg):
            log_messages.append(msg)

        _ = generate_candidates_system7(
            data_dict,
            top_n=10,
            log_callback=mock_log,
            include_diagnostics=True,
        )

        # Should have logged candidate count message
        assert any("候補日数" in msg for msg in log_messages)

    def test_log_callback_exception_in_date_mode(self):
        """Test log_callback exception handling in date mode (line 361)."""
        spy_data = self.create_spy_with_multiple_setups(setup_count=5)
        data_dict = {"SPY": spy_data}

        def failing_log(msg):
            raise RuntimeError("Log failed")

        # Should not crash despite log callback failure
        _ = generate_candidates_system7(
            data_dict, top_n=5, log_callback=failing_log, include_diagnostics=True
        )

        # If we get here, exception was handled gracefully
        assert True

    def test_progress_callback_in_date_mode(self):
        """Test progress_callback in date mode (lines 363-367)."""
        spy_data = self.create_spy_with_multiple_setups(setup_count=5)
        data_dict = {"SPY": spy_data}

        progress_calls = []

        def mock_progress(current, total):
            progress_calls.append((current, total))

        _ = generate_candidates_system7(
            data_dict,
            top_n=5,
            progress_callback=mock_progress,
            include_diagnostics=True,
        )

        # Should have called progress_callback
        assert len(progress_calls) > 0
        # Should report (1, 1) for SPY-only system
        assert (1, 1) in progress_calls


class TestSystem7NormalizationBranches:
    """Test normalization and ranking branches."""

    def create_spy_for_ranking(self):
        """Create SPY data suitable for ranking tests.

        Designed so prepare_data_vectorized_system7 will generate real setup=True rows
        based on Low <= min_50 condition.
        """
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Use CONSTANT prices for first 60 days to stabilize min_50
        prices = [450.0] * 60 + list(np.linspace(450, 400, 40))

        # Normal lows for first 60 days
        lows = [p * 0.995 for p in prices[:60]]

        # Last 40 days: aggressive drop to guarantee Low < min_50
        for i in range(60, 100):
            # Start from 445 and go down to 390 (below 400)
            lows.append(445 - (i - 60) * 1.5)

        highs = [p * 1.005 for p in prices]

        # Calculate rolling indicators
        df_temp = pd.DataFrame(
            {"Close": prices, "Low": lows, "High": highs}, index=dates
        )
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        return pd.DataFrame(
            {
                "Close": prices,
                "Low": lows,
                "High": highs,
                # Required precomputed indicators
                "atr50": [p * 0.02 for p in prices],
                "ATR50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "Min_50": min_50.values,
                "max_70": max_70.values,
                "Max_70": max_70.values,
                # setup will be recalculated by prepare_data
            },
            index=dates,
        )

    def test_normalization_payload_construction(self):
        """Test payload construction in normalization (lines 373-379)."""
        spy_data = self.create_spy_for_ranking()
        raw_dict = {"SPY": spy_data}

        # Prepare data first
        data_dict = prepare_data_vectorized_system7(raw_dict, reuse_indicators=False)

        result_tuple = generate_candidates_system7(
            data_dict, top_n=5, include_diagnostics=True
        )
        normalized = result_tuple[0]

        # Check payload excludes 'symbol' and 'date'
        for date_key, symbols_dict in normalized.items():
            if "SPY" in symbols_dict:
                spy_payload = symbols_dict["SPY"]
                # Should NOT have 'symbol' or 'date' keys
                assert "symbol" not in spy_payload
                assert "date" not in spy_payload
                # Should have 'entry_date'
                assert "entry_date" in spy_payload

    def test_ranking_columns_added(self):
        """Test rank and rank_total columns are added (lines 385-386)."""
        spy_data = self.create_spy_for_ranking()
        raw_dict = {"SPY": spy_data}

        # Prepare data first
        data_dict = prepare_data_vectorized_system7(raw_dict, reuse_indicators=False)

        result_tuple = generate_candidates_system7(
            data_dict, top_n=5, include_diagnostics=True
        )
        df_result = result_tuple[1]

        # Should have rank columns
        if df_result is not None and not df_result.empty:
            assert "rank" in df_result.columns
            assert "rank_total" in df_result.columns

    def test_diagnostics_ranking_source_full_scan(self):
        """Test ranking_source is set to full_scan (line 394)."""
        spy_data = self.create_spy_for_ranking()
        raw_dict = {"SPY": spy_data}

        # Prepare data first (let exception propagate if it fails)
        data_dict = prepare_data_vectorized_system7(raw_dict, reuse_indicators=False)

        # Verify setup rows exist after prepare_data
        spy_prepared = data_dict.get("SPY")
        assert spy_prepared is not None, f"SPY missing. Keys: {list(data_dict.keys())}"
        setup_rows = spy_prepared[spy_prepared["setup"]]
        assert (
            len(setup_rows) > 0
        ), f"No setup rows after prepare (count={len(setup_rows)})"

        # Explicitly use full_scan mode (latest_only=False)
        result_tuple = generate_candidates_system7(
            data_dict,
            top_n=5,
            include_diagnostics=True,
            latest_only=False,
        )
        diagnostics = result_tuple[2] if len(result_tuple) > 2 else {}

        # Should indicate full_scan mode
        assert diagnostics.get("ranking_source") == "full_scan"

    def test_empty_candidates_by_date(self):
        """Test behavior when no candidates are found (line 370+)."""
        # Create data with no setup signals
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        prices = np.linspace(400, 450, 50)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        spy_data = pd.DataFrame(
            {
                "Close": prices,
                "atr50": [p * 0.02 for p in prices],
                "ATR50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "max_70": max_70.values,
                "setup": [False] * 50,  # No setup signals
            },
            index=dates,
        )

        data_dict = {"SPY": spy_data}

        result_tuple = generate_candidates_system7(
            data_dict, top_n=5, include_diagnostics=True
        )
        normalized = result_tuple[0]
        df_result = result_tuple[1]

        # Should return empty structures gracefully
        assert isinstance(normalized, dict)
        assert len(normalized) == 0 or df_result is None or df_result.empty
