"""
System7 final push to 65% coverage.

Focus on:
- Lines 367-381: Full-scan normalization and diagnostics
- Lines 215-216: latest_only log callback
- Lines 226-246: latest_only detailed data construction
"""

import numpy as np
import pandas as pd

from core.system7 import generate_candidates_system7, prepare_data_vectorized_system7


class TestSystem7FullScanNormalization:
    """Test full-scan normalization logic (lines 367-381)."""

    def create_spy_with_multiple_dates(self):
        """Create SPY data with setups on multiple dates.

        Setup condition for System7: Low <= min_50
        """
        dates = pd.date_range("2023-01-01", periods=150, freq="D")
        prices = np.linspace(400, 450, 150)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        df = pd.DataFrame(
            {
                "Close": prices,
                "Low": lows,
                "High": highs,
                "atr50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "max_70": max_70.values,
            },
            index=dates,
        )

        # Create setup conditions on multiple dates (days 70, 100, 130)
        # Setup condition: Low <= min_50
        df.loc[dates[70], "Low"] = min_50.iloc[70]  # Exactly equal
        df.loc[dates[100], "Low"] = min_50.iloc[100] * 0.999  # Below min_50
        df.loc[dates[130], "Low"] = min_50.iloc[130]  # Exactly equal

        return df

    def test_full_scan_normalization_with_diagnostics(self):
        """Test full-scan normalization loop and diagnostics (lines 367-381)."""
        spy_data = self.create_spy_with_multiple_dates()
        raw_data = {"SPY": spy_data}

        # Prepare data first
        prepared_data = prepare_data_vectorized_system7(raw_data, reuse_indicators=False)

        # Generate candidates with full_scan mode and diagnostics
        result_tuple = generate_candidates_system7(
            prepared_data, top_n=5, latest_only=False, include_diagnostics=True
        )

        assert len(result_tuple) == 3
        candidates_dict = result_tuple[0]
        diagnostics = result_tuple[2]

        # Should have candidates from multiple dates
        assert isinstance(candidates_dict, dict)
        assert len(candidates_dict) > 0

        # Lines 378-379: diagnostics["ranking_source"] should be "full_scan"
        assert diagnostics.get("ranking_source") == "full_scan"

        # Line 378: final_top_n_count should be set
        assert "final_top_n_count" in diagnostics
        assert diagnostics["final_top_n_count"] >= 0

    def test_full_scan_normalization_loop_details(self):
        """Test normalization loop details (lines 367-374)."""
        spy_data = self.create_spy_with_multiple_dates()
        raw_data = {"SPY": spy_data}

        prepared_data = prepare_data_vectorized_system7(raw_data, reuse_indicators=False)

        result_tuple = generate_candidates_system7(
            prepared_data, top_n=3, latest_only=False, include_diagnostics=False
        )

        candidates_dict = result_tuple[0]

        # Lines 367-374: Normalization loop processes records by date
        assert isinstance(candidates_dict, dict)

        # Each date key should have a dict with "SPY" as key
        for date_key, symbol_dict in candidates_dict.items():
            assert isinstance(date_key, pd.Timestamp)
            assert isinstance(symbol_dict, dict)

            # Line 369: sym_val check
            if "SPY" in symbol_dict:
                # Line 372: payload construction
                payload = symbol_dict["SPY"]
                assert isinstance(payload, dict)

                # Payload should not contain "symbol" key (line 372)
                assert "symbol" not in payload

    def test_full_scan_max_exception_handling(self):
        """Test exception handling in max(normalized_full.keys()) (line 381)."""
        # Create minimal data that results in empty candidates
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        prices = [400.0] * 30
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=30, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=30, min_periods=1).max()

        spy_data = pd.DataFrame(
            {
                "Close": prices,
                "Low": [p * 1.01 for p in prices],  # Lows > min_50, no setup
                "High": highs,
                "atr50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "max_70": max_70.values,
            },
            index=dates,
        )

        raw_data = {"SPY": spy_data}
        prepared_data = prepare_data_vectorized_system7(raw_data, reuse_indicators=False)

        result_tuple = generate_candidates_system7(
            prepared_data, top_n=5, latest_only=False, include_diagnostics=True
        )

        # Should not crash even if normalized_full is empty
        assert len(result_tuple) == 3
        diagnostics = result_tuple[2]

        # Line 381: Exception handling should prevent crash
        # ranking_source might be None if exception occurred
        assert "ranking_source" in diagnostics


class TestSystem7LatestOnlyLogCallback:
    """Test latest_only log callback (line 215-216)."""

    def create_spy_for_latest_only_with_setup(self):
        """Create SPY data with setup on last day.

        Setup condition for System7: Low <= min_50
        """
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(400, 450, 100)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        df = pd.DataFrame(
            {
                "Close": prices,
                "Low": lows,
                "High": highs,
                "atr50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "max_70": max_70.values,
            },
            index=dates,
        )

        # Setup on last day: Low <= min_50
        df.loc[dates[-1], "Low"] = min_50.iloc[-1] * 0.999  # Below min_50

        return df

    def test_latest_only_log_callback_called(self):
        """Test log callback is called in latest_only mode (line 215-216)."""
        spy_data = self.create_spy_for_latest_only_with_setup()
        raw_data = {"SPY": spy_data}

        prepared_data = prepare_data_vectorized_system7(raw_data, reuse_indicators=False)

        log_messages = []

        def log_cb(msg):
            log_messages.append(msg)

        result_tuple = generate_candidates_system7(
            prepared_data, top_n=5, latest_only=True, log_callback=log_cb
        )

        # Should have generated candidate
        candidates_dict = result_tuple[0]
        assert len(candidates_dict) > 0

        # Line 215-216: Log callback should have been called
        assert len(log_messages) > 0


class TestSystem7LatestOnlyDataConstruction:
    """Test latest_only detailed data construction (lines 226-246)."""

    def create_spy_for_data_construction(self):
        """Create SPY data for testing data construction details.

        Setup condition for System7: Low <= min_50
        """
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = np.linspace(400, 450, 100)
        lows = [p * 0.992 for p in prices]
        highs = [p * 1.008 for p in prices]

        df_temp = pd.DataFrame({"Low": lows, "High": highs}, index=dates)
        min_50 = df_temp["Low"].rolling(window=50, min_periods=1).min()
        max_70 = df_temp["High"].rolling(window=70, min_periods=1).max()

        df = pd.DataFrame(
            {
                "Close": prices,
                "Low": lows,
                "High": highs,
                "atr50": [p * 0.02 for p in prices],
                "ATR50": [p * 0.02 for p in prices],
                "min_50": min_50.values,
                "max_70": max_70.values,
            },
            index=dates,
        )

        # Setup on last day: Low <= min_50
        df.loc[dates[-1], "Low"] = min_50.iloc[-1] * 0.999  # Below min_50

        return df

    def test_latest_only_entry_price_construction(self):
        """Test entry_price extraction in latest_only mode (line 228-230)."""
        spy_data = self.create_spy_for_data_construction()
        raw_data = {"SPY": spy_data}

        prepared_data = prepare_data_vectorized_system7(raw_data, reuse_indicators=False)

        result_tuple = generate_candidates_system7(
            prepared_data, top_n=5, latest_only=True, include_diagnostics=False
        )

        candidates_dict = result_tuple[0]

        # Should have one candidate
        assert len(candidates_dict) == 1

        # Extract the candidate
        date_key = list(candidates_dict.keys())[0]
        spy_payload = candidates_dict[date_key]["SPY"]

        # Line 228-230: entry_price should be set from Close column
        assert "entry_price" in spy_payload
        assert spy_payload["entry_price"] is not None
        assert isinstance(spy_payload["entry_price"], (int, float, np.number))

    def test_latest_only_atr_fallback(self):
        """Test ATR50 fallback logic (line 231-233)."""
        spy_data = self.create_spy_for_data_construction().copy()

        # Remove uppercase ATR50 if it exists
        if "ATR50" in spy_data.columns:
            spy_data = spy_data.drop(columns=["ATR50"])

        raw_data = {"SPY": spy_data}
        prepared_data = prepare_data_vectorized_system7(raw_data, reuse_indicators=False)

        result_tuple = generate_candidates_system7(prepared_data, top_n=5, latest_only=True)

        candidates_dict = result_tuple[0]

        if len(candidates_dict) > 0:
            date_key = list(candidates_dict.keys())[0]
            spy_payload = candidates_dict[date_key]["SPY"]

            # Line 231-233: ATR50 should fallback to atr50
            assert "ATR50" in spy_payload
            assert spy_payload["ATR50"] is not None

    def test_latest_only_df_fast_construction(self):
        """Test df_fast DataFrame construction (line 234-241)."""
        spy_data = self.create_spy_for_data_construction()
        raw_data = {"SPY": spy_data}

        prepared_data = prepare_data_vectorized_system7(raw_data, reuse_indicators=False)

        result_tuple = generate_candidates_system7(
            prepared_data, top_n=5, latest_only=True, include_diagnostics=True
        )

        # Access diagnostics (tuple index 2)
        candidates_dict = result_tuple[0]
        diagnostics = result_tuple[2]

        # Line 241-242: If setup condition is met, should have candidate
        if len(candidates_dict) > 0:
            assert diagnostics.get("final_top_n_count") == 1
        # Diagnostics should always be returned even if no candidates
        assert "final_top_n_count" in diagnostics

    def test_latest_only_symbol_payload_construction(self):
        """Test symbol_payload construction (line 246-250)."""
        spy_data = self.create_spy_for_data_construction()
        raw_data = {"SPY": spy_data}

        prepared_data = prepare_data_vectorized_system7(raw_data, reuse_indicators=False)

        result_tuple = generate_candidates_system7(prepared_data, top_n=5, latest_only=True)

        candidates_dict = result_tuple[0]
        date_key = list(candidates_dict.keys())[0]
        spy_payload = candidates_dict[date_key]["SPY"]

        # Line 246-250: Payload should not contain "symbol" or "date"
        assert "symbol" not in spy_payload
        assert "date" not in spy_payload

        # Should contain entry_date
        assert "entry_date" in spy_payload
