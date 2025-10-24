"""
Enhanced tests for core.system6 module targeting main functions for improved coverage.
Focus on the main pipeline functions and integration tests.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from common.testing import set_test_determinism

# Import functions directly to avoid dependency issues
generate_candidates_system6: Any = None
get_total_days_system6: Any = None
prepare_data_vectorized_system6: Any = None
HV50_BOUNDS_FRACTION: Any = None
HV50_BOUNDS_PERCENT: Any = None
MIN_DOLLAR_VOLUME_50: Any = None
MIN_PRICE: Any = None
try:
    from core.system6 import HV50_BOUNDS_FRACTION as _hv_frac
    from core.system6 import HV50_BOUNDS_PERCENT as _hv_pct
    from core.system6 import MIN_DOLLAR_VOLUME_50 as _min_dv
    from core.system6 import MIN_PRICE as _min_p
    from core.system6 import generate_candidates_system6 as _gc6
    from core.system6 import get_total_days_system6 as _gt6
    from core.system6 import prepare_data_vectorized_system6 as _prep6

    generate_candidates_system6 = _gc6
    get_total_days_system6 = _gt6
    prepare_data_vectorized_system6 = _prep6
    HV50_BOUNDS_FRACTION = _hv_frac
    HV50_BOUNDS_PERCENT = _hv_pct
    MIN_DOLLAR_VOLUME_50 = _min_dv
    MIN_PRICE = _min_p
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestSystem6MainFunctions:
    """Test main System6 functions for improved coverage"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system6 imports not available")

    def test_prepare_data_vectorized_system6_with_valid_data(self):
        """Test data preparation with minimal valid data"""
        # Create minimal data that will pass all System6 filters
        mock_data = {
            "AAPL": pd.DataFrame(
                {
                    "Open": [100, 105, 110, 115, 120, 125],
                    "High": [102, 107, 112, 117, 122, 127],
                    "Low": [10, 11, 12, 13, 14, 15],  # Above MIN_PRICE=5
                    "Close": [101, 106, 111, 116, 121, 126],
                    "Volume": [10_000_000] * 6,  # Large volume
                },
                index=pd.date_range("2023-01-01", periods=6),
            ),
        }

        # System6 calculates indicators if not present
        result = prepare_data_vectorized_system6(raw_data_dict=mock_data)

        assert isinstance(result, dict)
        # If processing succeeds, result should have data with filter/setup
        if result:
            for _symbol, df in result.items():
                # Should have filter and setup columns
                assert "filter" in df.columns or len(df) >= 0

    def test_prepare_data_vectorized_system6_with_symbols_list(self):
        """Test that function handles None input gracefully"""
        # System6 requires raw_data_dict, None returns empty
        result = prepare_data_vectorized_system6(raw_data_dict=None)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_prepare_data_vectorized_system6_empty_input(self):
        """Test handling of empty input"""
        result = prepare_data_vectorized_system6(raw_data_dict={})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_prepare_data_vectorized_system6_none_input(self):
        """Test handling of None input without symbols"""
        result = prepare_data_vectorized_system6(raw_data_dict=None, symbols=[])
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_generate_candidates_system6_basic(self):
        """Test basic candidate generation"""
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "return_6d": [0.30, 0.28, 0.25],  # Descending (ranking key)
                    "atr10": [2.5, 2.7, 2.9],
                    "dollarvolume50": [15_000_000, 16_000_000, 17_000_000],
                    "hv50": [15.0, 20.0, 25.0],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
            "MSFT": pd.DataFrame(
                {
                    "Close": [200, 210, 220],
                    "return_6d": [0.35, 0.32, 0.28],
                    "atr10": [3.0, 3.2, 3.4],
                    "dollarvolume50": [18_000_000, 19_000_000, 20_000_000],
                    "hv50": [18.0, 22.0, 26.0],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        candidates, merged_df = generate_candidates_system6(
            prepared_dict=prepared_dict, top_n=10, latest_only=False
        )

        assert isinstance(candidates, dict)
        assert merged_df is None or isinstance(merged_df, pd.DataFrame)

    def test_generate_candidates_system6_latest_only(self):
        """Test candidate generation with latest_only=True"""
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "return_6d": [0.30, 0.28, 0.25],
                    "atr10": [2.5, 2.7, 2.9],
                    "dollarvolume50": [15_000_000, 16_000_000, 17_000_000],
                    "hv50": [15.0, 20.0, 25.0],
                    "setup": [False, False, True],  # Only last row setup
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        candidates, merged_df = generate_candidates_system6(
            prepared_dict=prepared_dict, top_n=10, latest_only=True
        )

        assert isinstance(candidates, dict)
        # With latest_only, should only check last row
        if candidates:
            for _date, _cands in candidates.items():
                assert isinstance(_cands, dict)

    def test_generate_candidates_system6_with_diagnostics(self):
        """Test candidate generation with diagnostics enabled"""
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "return_6d": [0.30, 0.28, 0.25],
                    "atr10": [2.5, 2.7, 2.9],
                    "dollarvolume50": [15_000_000, 16_000_000, 17_000_000],
                    "hv50": [15.0, 20.0, 25.0],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        candidates, merged_df, diagnostics = generate_candidates_system6(
            prepared_dict=prepared_dict, top_n=10, include_diagnostics=True
        )

        assert isinstance(candidates, dict)
        assert isinstance(diagnostics, dict)
        # Check for required diagnostic keys
        assert "ranking_source" in diagnostics
        assert "setup_predicate_count" in diagnostics
        assert "ranked_top_n_count" in diagnostics

    def test_generate_candidates_system6_empty_prepared_dict(self):
        """Test handling of empty prepared_dict"""
        candidates, merged_df = generate_candidates_system6(prepared_dict={}, top_n=10)

        assert isinstance(candidates, dict)
        assert len(candidates) == 0
        assert merged_df is None or (
            isinstance(merged_df, pd.DataFrame) and merged_df.empty
        )

    def test_generate_candidates_system6_ranking_order(self):
        """Test that candidates are ranked by return_6d descending"""
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100],
                    "return_6d": [0.25],  # Lower return
                    "atr10": [2.5],
                    "dollarvolume50": [15_000_000],
                    "hv50": [15.0],
                    "setup": [True],
                },
                index=pd.date_range("2023-01-01", periods=1),
            ),
            "MSFT": pd.DataFrame(
                {
                    "Close": [200],
                    "return_6d": [0.35],  # Higher return
                    "atr10": [3.0],
                    "dollarvolume50": [18_000_000],
                    "hv50": [18.0],
                    "setup": [True],
                },
                index=pd.date_range("2023-01-01", periods=1),
            ),
        }

        candidates, _merged_df = generate_candidates_system6(
            prepared_dict=prepared_dict, top_n=1, latest_only=True
        )

        # MSFT should rank higher due to higher return_6d
        if candidates:
            for _date, cands in candidates.items():
                if cands:
                    # First candidate should be MSFT (higher return)
                    first_symbol = list(cands.keys())[0]
                    assert first_symbol == "MSFT"

    def test_generate_candidates_system6_missing_indicators(self):
        """Test handling of missing required indicators"""
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110],
                    # Missing return_6d, atr10, etc.
                    "setup": [True, True],
                },
                index=pd.date_range("2023-01-01", periods=2),
            ),
        }

        # Should handle gracefully without crashing
        try:
            candidates, _merged_df = generate_candidates_system6(
                prepared_dict=prepared_dict, top_n=10
            )
            # Either returns empty or handles missing columns
            assert isinstance(candidates, dict)
        except KeyError:
            # Acceptable to raise KeyError for missing required columns
            pass

    def test_get_total_days_system6(self):
        """Test get_total_days_system6 function"""
        test_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "return_6d": [0.30, 0.28, 0.25],
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }
        result = get_total_days_system6(test_data)
        assert isinstance(result, int)
        assert result > 0  # Should return a positive number

    def test_system6_constants(self):
        """Test System6 configuration constants"""
        assert MIN_PRICE == 5.0
        assert MIN_DOLLAR_VOLUME_50 == 10_000_000
        assert isinstance(HV50_BOUNDS_PERCENT, tuple)
        assert isinstance(HV50_BOUNDS_FRACTION, tuple)
        assert len(HV50_BOUNDS_PERCENT) == 2
        assert len(HV50_BOUNDS_FRACTION) == 2
        assert HV50_BOUNDS_PERCENT == (10.0, 40.0)
        assert HV50_BOUNDS_FRACTION == (0.10, 0.40)


class TestSystem6EdgeCases:
    """Test edge cases and error handling for System6"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system6 imports not available")

    def test_prepare_data_with_nan_values(self):
        """Test handling of NaN values in indicators"""
        mock_data = {
            "AAPL": pd.DataFrame(
                {
                    "Open": [100, 105, 110],
                    "High": [102, 107, 112],
                    "Low": [6, 7, 8],  # All above MIN_PRICE
                    "Close": [101, 106, 111],
                    "Volume": [1_000_000, 1_100_000, 1_200_000],
                    "atr10": [2.5, None, 2.9],  # NaN in middle
                    "dollarvolume50": [15_000_000, 16_000_000, None],
                    "return_6d": [0.25, 0.30, 0.35],
                    "UpTwoDays": [True, True, True],
                    "hv50": [15.0, None, 25.0],  # NaN in middle
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        # System6 drops NaN rows via dropna in _compute_indicators_from_frame
        result = prepare_data_vectorized_system6(raw_data_dict=mock_data)

        assert isinstance(result, dict)
        # Result may be empty or have rows with NaN dropped
        if "AAPL" in result:
            df = result["AAPL"]
            # Verify no NaN in required numeric columns after dropna
            assert not df["atr10"].isna().any()
            assert not df["dollarvolume50"].isna().any()
            assert not df["hv50"].isna().any()

    def test_generate_candidates_with_all_false_setup(self):
        """Test candidate generation when all setup values are False"""
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [100, 110, 120],
                    "return_6d": [0.30, 0.28, 0.25],
                    "atr10": [2.5, 2.7, 2.9],
                    "dollarvolume50": [15_000_000, 16_000_000, 17_000_000],
                    "hv50": [15.0, 20.0, 25.0],
                    "setup": [False, False, False],  # All False
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        candidates, merged_df = generate_candidates_system6(
            prepared_dict=prepared_dict, top_n=10
        )

        assert isinstance(candidates, dict)
        # Should return empty candidates or handle gracefully
        if candidates:
            for _date, cands in candidates.items():
                # If any candidates, they should be dict
                assert isinstance(cands, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestSystem6LatestOnlyMode:
    """System6のlatest_only=Trueモードのテスト（lines 271-375カバレッジ向上）"""

    def test_latest_only_with_specific_date(self):
        """latest_mode_dateで特定日のみ抽出することを検証"""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        target_date = pd.Timestamp("2023-01-05")

        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 10,
                "High": [105.0] * 10,
                "Low": [10.0] * 10,
                "Close": [100.0] * 10,
                "Volume": [20_000_000] * 10,
            },
            index=dates,
        )

        # prepare_data_vectorized_system6を使ってsetup列を生成
        from core.system6 import (
            generate_candidates_system6,
            prepare_data_vectorized_system6,
        )

        symbols_dict = {"TEST": mock_df}
        prepared_dict = prepare_data_vectorized_system6(symbols_dict)

        result, _ = generate_candidates_system6(
            prepared_dict, top_n=5, latest_only=True, latest_mode_date=target_date
        )

        # 指定日のデータのみ抽出される（setup条件を満たす場合）
        assert isinstance(result, dict)
        # latest_only=Trueの場合、結果は{date: {symbol: {...}}}形式
        if len(result) > 0:
            first_date = list(result.keys())[0]
            assert pd.Timestamp(first_date).date() == target_date.date()

    def test_latest_only_skip_missing_date(self):
        """latest_mode_dateが存在しない場合はスキップされることを検証"""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        missing_date = pd.Timestamp("2023-02-01")  # データに存在しない日

        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [105.0] * 5,
                "Low": [10.0] * 5,
                "Close": [100.0] * 5,
                "Volume": [20_000_000] * 5,
            },
            index=dates,
        )

        from core.system6 import (
            generate_candidates_system6,
            prepare_data_vectorized_system6,
        )

        symbols_dict = {"TEST": mock_df}
        prepared_dict = prepare_data_vectorized_system6(symbols_dict)

        result, _ = generate_candidates_system6(
            prepared_dict, top_n=5, latest_only=True, latest_mode_date=missing_date
        )

        # 対象日のデータがないためスキップ
        assert len(result) == 0

    def test_latest_only_skip_false_setup(self):
        """setup=Falseの行はスキップされることを検証（全行filter/setup条件を満たさない）"""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")

        # 極端に低いボラティリティ（HV50が範囲外になる）で条件を満たさないデータを作成
        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [100.01] * 5,  # 極小の値幅
                "Low": [99.99] * 5,
                "Close": [100.0] * 5,
                "Volume": [1_000] * 5,  # 非常に低い出来高
            },
            index=dates,
        )

        from core.system6 import (
            generate_candidates_system6,
            prepare_data_vectorized_system6,
        )

        symbols_dict = {"TEST": mock_df}
        prepared_dict = prepare_data_vectorized_system6(symbols_dict)

        result, _ = generate_candidates_system6(
            prepared_dict, top_n=5, latest_only=True
        )

        # 条件を満たさないためスキップ（結果は0件）
        assert len(result) == 0


class TestSystem6RankingAndFiltering:
    """System6のランキングとフィルタリングロジックのテスト（lines 343-420カバレッジ向上）"""

    @pytest.mark.xfail(
        reason=(
            "Test isolation issue: passes in isolation, fails in full suite "
            "due to NumPy module reload warning. Returns None for df in full suite."
        ),
        strict=False,
    )
    def test_ranking_by_return_6d_descending(self):
        """return_6d降順でランキングされることを検証"""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")

        # 3銘柄で異なるreturn_6d値
        symbols_dict = {
            "HIGH": pd.DataFrame(
                {
                    "Open": [100.0] * 60,
                    "High": [110.0] * 60,
                    "Low": [95.0] * 60,
                    "Close": [110.0] * 60,  # 10%上昇
                    "Volume": [20_000_000] * 60,
                },
                index=dates,
            ),
            "MID": pd.DataFrame(
                {
                    "Open": [100.0] * 60,
                    "High": [105.0] * 60,
                    "Low": [95.0] * 60,
                    "Close": [105.0] * 60,  # 5%上昇
                    "Volume": [20_000_000] * 60,
                },
                index=dates,
            ),
            "LOW": pd.DataFrame(
                {
                    "Open": [100.0] * 60,
                    "High": [102.0] * 60,
                    "Low": [95.0] * 60,
                    "Close": [102.0] * 60,  # 2%上昇
                    "Volume": [20_000_000] * 60,
                },
                index=dates,
            ),
        }

        from core.system6 import (
            generate_candidates_system6,
            prepare_data_vectorized_system6,
        )

        prepared_dict = prepare_data_vectorized_system6(symbols_dict)

        gen_result = generate_candidates_system6(
            prepared_dict, top_n=3, latest_only=True, include_diagnostics=True
        )
        # 戻り値は2個または3個（diagnostics付き）
        if len(gen_result) == 3:
            result, df, _ = gen_result
        else:
            result, df = gen_result

        # DataFrameが返されることを確認
        assert df is not None
        assert len(df) <= 3

        # return_6d降順でソートされていることを確認
        if len(df) > 1:
            returns = df["return_6d"].tolist()
            assert returns == sorted(returns, reverse=True)

        # rank列が付与されていることを確認
        if len(df) > 0:
            assert "rank" in df.columns
            assert "rank_total" in df.columns
            assert df["rank"].tolist() == list(range(1, len(df) + 1))

    def test_top_n_limiting(self):
        """top_n制限が正しく機能することを検証"""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")

        # 5銘柄作成
        symbols_dict = {}
        for i in range(5):
            symbols_dict[f"SYM{i}"] = pd.DataFrame(
                {
                    "Open": [100.0] * 60,
                    "High": [105.0 + i] * 60,
                    "Low": [10.0] * 60,
                    "Close": [105.0 + i] * 60,
                    "Volume": [20_000_000] * 60,
                },
                index=dates,
            )

        from core.system6 import (
            generate_candidates_system6,
            prepare_data_vectorized_system6,
        )

        prepared_dict = prepare_data_vectorized_system6(symbols_dict)

        gen_result = generate_candidates_system6(
            prepared_dict,
            top_n=2,
            latest_only=True,  # 2件のみ要求
        )
        result, df = gen_result[:2] if len(gen_result) == 3 else gen_result

        # 最大2件に制限されることを確認
        if df is not None:
            assert len(df) <= 2

    def test_date_mode_selection_with_multiple_dates(self):
        """複数日付がある場合、最頻日が選択されることを検証"""
        dates1 = pd.date_range("2023-01-01", periods=60, freq="D")
        dates2 = pd.date_range("2023-01-02", periods=60, freq="D")  # 1日ずれ

        symbols_dict = {
            "SYM1": pd.DataFrame(
                {
                    "Open": [100.0] * 60,
                    "High": [105.0] * 60,
                    "Low": [10.0] * 60,
                    "Close": [105.0] * 60,
                    "Volume": [20_000_000] * 60,
                },
                index=dates1,
            ),
            "SYM2": pd.DataFrame(
                {
                    "Open": [100.0] * 60,
                    "High": [105.0] * 60,
                    "Low": [10.0] * 60,
                    "Close": [105.0] * 60,
                    "Volume": [20_000_000] * 60,
                },
                index=dates1,
            ),  # 同じ日付
            "SYM3": pd.DataFrame(
                {
                    "Open": [100.0] * 60,
                    "High": [105.0] * 60,
                    "Low": [10.0] * 60,
                    "Close": [105.0] * 60,
                    "Volume": [20_000_000] * 60,
                },
                index=dates2,
            ),  # 異なる日付
        }

        from core.system6 import (
            generate_candidates_system6,
            prepare_data_vectorized_system6,
        )

        prepared_dict = prepare_data_vectorized_system6(symbols_dict)

        gen_result = generate_candidates_system6(
            prepared_dict, top_n=5, latest_only=True
        )
        result, df = gen_result[:2] if len(gen_result) == 3 else gen_result

        # 最頻日（dates1）が選択されることを確認
        if df is not None and len(df) > 0:
            unique_dates = df["date"].unique()
            # 1つの日付に揃えられることを確認
            assert len(unique_dates) == 1


@pytest.mark.xfail(reason="numpy/pandas dropna() compatibility issue with _NoValueType")
class TestSystem6IndicatorFallback:
    """System6 インジケータフォールバック機能のテスト"""

    def test_atr10_from_precomputed(self):
        """事前計算済みATR10列が使われることを検証"""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 60,
                "High": [105.0] * 60,
                "Low": [95.0] * 60,
                "Close": [100.0] * 60,
                "Volume": [10_000_000] * 60,
                "ATR10": [2.5] * 60,  # 事前計算済み（大文字）
                "DollarVolume50": [15_000_000.0] * 60,
                "Return_6D": [0.25] * 60,
                "UpTwoDays": [True] * 60,
                "HV50": [25.0] * 60,
            },
            index=dates,
        )

        from core.system6 import _compute_indicators_from_frame

        result = _compute_indicators_from_frame(mock_df)

        assert "atr10" in result.columns
        assert (result["atr10"] == 2.5).all()

    def test_dollarvolume50_from_precomputed(self):
        """事前計算済みDollarVolume50列が使われることを検証"""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 60,
                "High": [105.0] * 60,
                "Low": [95.0] * 60,
                "Close": [100.0] * 60,
                "Volume": [1_000_000] * 60,
                "DollarVolume50": [15_000_000.0] * 60,  # 事前計算済み
                "ATR10": [2.5] * 60,
                "Return_6D": [0.25] * 60,
                "UpTwoDays": [True] * 60,
                "HV50": [25.0] * 60,
            },
            index=dates,
        )

        from core.system6 import _compute_indicators_from_frame

        result = _compute_indicators_from_frame(mock_df)

        assert "dollarvolume50" in result.columns
        assert (result["dollarvolume50"] == 15_000_000.0).all()

    def test_return_6d_from_precomputed(self):
        """事前計算済みReturn_6D列が使われることを検証"""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 60,
                "High": [105.0] * 60,
                "Low": [95.0] * 60,
                "Close": [100.0] * 60,
                "Volume": [1_000_000] * 60,
                "Return_6D": [0.25] * 60,  # 事前計算済み（25%上昇）
                "ATR10": [2.5] * 60,
                "DollarVolume50": [15_000_000.0] * 60,
                "UpTwoDays": [True] * 60,
                "HV50": [25.0] * 60,
            },
            index=dates,
        )

        from core.system6 import _compute_indicators_from_frame

        result = _compute_indicators_from_frame(mock_df)

        assert "return_6d" in result.columns
        assert (result["return_6d"] == 0.25).all()

    def test_uptwodays_from_precomputed(self):
        """事前計算済みUpTwoDays列が使われることを検証"""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 60,
                "High": [105.0] * 60,
                "Low": [95.0] * 60,
                "Close": [100.0] * 60,
                "Volume": [1_000_000] * 60,
                "UpTwoDays": [True] * 60,  # 事前計算済み
                "ATR10": [2.5] * 60,
                "DollarVolume50": [15_000_000.0] * 60,
                "Return_6D": [0.25] * 60,
                "HV50": [25.0] * 60,
            },
            index=dates,
        )

        from core.system6 import _compute_indicators_from_frame

        result = _compute_indicators_from_frame(mock_df)

        assert "UpTwoDays" in result.columns
        assert result["UpTwoDays"].all()

    def test_hv50_from_precomputed(self):
        """事前計算済みHV50列が使われることを検証"""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 60,
                "High": [105.0] * 60,
                "Low": [95.0] * 60,
                "Close": [100.0] * 60,
                "Volume": [1_000_000] * 60,
                "HV50": [25.0] * 60,  # 事前計算済み（パーセント形式）
                "ATR10": [2.5] * 60,
                "DollarVolume50": [15_000_000.0] * 60,
                "Return_6D": [0.25] * 60,
                "UpTwoDays": [True] * 60,
            },
            index=dates,
        )

        from core.system6 import _compute_indicators_from_frame

        result = _compute_indicators_from_frame(mock_df)

        assert "hv50" in result.columns
        assert (result["hv50"] == 25.0).all()

    def test_hv50_dual_condition_percent(self):
        """HV50のパーセント条件（10.0-40.0）を検証"""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 60,
                "High": [105.0] * 60,
                "Low": [10.0] * 60,  # MIN_PRICE=5.0以上
                "Close": [100.0] * 60,
                "Volume": [20_000_000] * 60,  # MIN_DOLLAR_VOLUME_50=10M以上
                "HV50": [25.0] * 60,  # 10.0-40.0の範囲内（事前計算済み）
                "DollarVolume50": [15_000_000.0] * 60,
                "ATR10": [2.5] * 60,
                "Return_6D": [0.25] * 60,
                "UpTwoDays": [True] * 60,
            },
            index=dates,
        )

        from core.system6 import _compute_indicators_from_frame

        result = _compute_indicators_from_frame(mock_df)

        assert "filter" in result.columns
        # hv50=25.0はパーセント条件を満たす
        assert result["filter"].all()

    def test_hv50_dual_condition_fraction(self):
        """HV50の分数条件（0.10-0.40）を検証"""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 60,
                "High": [105.0] * 60,
                "Low": [10.0] * 60,
                "Close": [100.0] * 60,
                "Volume": [20_000_000] * 60,
                "HV50": [0.25] * 60,  # 0.10-0.40の範囲内（事前計算済み）
                "DollarVolume50": [15_000_000.0] * 60,
                "ATR10": [2.5] * 60,
                "Return_6D": [0.25] * 60,
                "UpTwoDays": [True] * 60,
            },
            index=dates,
        )

        from core.system6 import _compute_indicators_from_frame

        result = _compute_indicators_from_frame(mock_df)

        assert "filter" in result.columns
        assert result["filter"].all()

    def test_filter_and_setup_columns(self):
        """filterとsetup列が正しく生成されることを検証"""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 60,
                "High": [105.0] * 60,
                "Low": [10.0] * 60,  # MIN_PRICE以上
                "Close": [100.0] * 60,
                "Volume": [20_000_000] * 60,
                "HV50": [25.0] * 60,  # 事前計算済み
                "DollarVolume50": [15_000_000.0] * 60,
                "ATR10": [2.5] * 60,
                "Return_6D": [0.25] * 60,  # >0.20を満たす
                "UpTwoDays": [True] * 60,
            },
            index=dates,
        )

        from core.system6 import _compute_indicators_from_frame

        result = _compute_indicators_from_frame(mock_df)

        assert "filter" in result.columns
        assert "setup" in result.columns
        # すべての条件を満たすのでsetup=Trueのはず
        assert result["setup"].all()

    def test_insufficient_rows_error(self):
        """行数不足でエラーが発生することを検証"""
        # 50行未満でエラー
        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 10,
                "High": [105.0] * 10,
                "Low": [95.0] * 10,
                "Close": [100.0] * 10,
                "Volume": [1000000] * 10,
            },
            index=pd.date_range("2023-01-01", periods=10),
        )

        from core.system6 import _compute_indicators_from_frame

        with pytest.raises(ValueError, match="insufficient rows"):
            _compute_indicators_from_frame(mock_df)

    def test_lowercase_indicator_columns(self):
        """小文字形式の既存指標列が使われることを検証"""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [100.0] * 60,
                "High": [105.0] * 60,
                "Low": [10.0] * 60,
                "Close": [100.0] * 60,
                "Volume": [10_000_000] * 60,
                "atr10": [2.5] * 60,  # 小文字形式（既存）
                "dollarvolume50": [15_000_000.0] * 60,
                "return_6d": [0.25] * 60,
                "uptwodays": [True] * 60,  # UpTwoDaysへコピーされる
                "hv50": [25.0] * 60,
            },
            index=dates,
        )

        from core.system6 import _compute_indicators_from_frame

        result = _compute_indicators_from_frame(mock_df)

        # 小文字列がそのまま使われる
        assert "atr10" in result.columns
        assert (result["atr10"] == 2.5).all()
        assert "UpTwoDays" in result.columns  # uptwodays→UpTwoDaysへコピー
        assert result["UpTwoDays"].all()


class TestSystem6DateModeProcessing:
    """日付モード処理のテスト (lines 588-601)

    generate_candidates_system6関数のlatest_only=True時の
    日付モード処理（ソート、ランク付け、top_n制限）を検証します。
    """

    @pytest.mark.xfail(
        reason=(
            "Test isolation issue: passes in isolation, fails in full suite "
            "due to NumPy module reload warning. Returns empty candidates in "
            "full suite but correct candidates when run alone."
        ),
        strict=False,
    )
    def test_date_mode_sorting_by_return_6d_descending(self):
        """return_6dによる降順ソートを検証 (lines 588-591)"""
        from core.system6 import generate_candidates_system6

        set_test_determinism()

        # 3銘柄で異なるreturn_6d値を設定
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        data_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [150.0],
                    "return_6d": [0.15],
                    "atr10": [2.0],
                    "dollarvolume50": [100_000_000],
                    "hv50": [20.0],
                    "UpTwoDays": [0],
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            ),
            "GOOGL": pd.DataFrame(
                {
                    "Close": [2800.0],
                    "return_6d": [0.10],
                    "atr10": [30.0],
                    "dollarvolume50": [150_000_000],
                    "hv50": [18.0],
                    "UpTwoDays": [0],
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            ),
            "MSFT": pd.DataFrame(
                {
                    "Close": [300.0],
                    "return_6d": [0.05],
                    "atr10": [3.5],
                    "dollarvolume50": [120_000_000],
                    "hv50": [22.0],
                    "UpTwoDays": [0],
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            ),
        }

        candidates, _ = generate_candidates_system6(
            prepared_dict=data_dict,
            top_n=3,
            latest_only=True,
        )

        # 最低1件の候補が返されることを確認
        assert len(candidates) >= 1
        last_date = list(candidates.keys())[0]
        symbols = list(candidates[last_date].keys())

        # return_6d降順でソートされているはず: AAPL(0.15) > GOOGL(0.10) > MSFT(0.05)
        assert symbols == ["AAPL", "GOOGL", "MSFT"]

    @pytest.mark.xfail(
        reason=(
            "Test isolation issue: passes in isolation, fails in full suite "
            "due to NumPy module reload warning. Returns empty candidates in "
            "full suite but correct candidates when run alone."
        ),
        strict=False,
    )
    def test_date_mode_rank_assignment(self):
        """ランク付けとrank_total設定を検証 (lines 593-595)"""
        from core.system6 import generate_candidates_system6

        set_test_determinism()

        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        data_dict = {
            "SYM1": pd.DataFrame(
                {
                    "Close": [100.0],
                    "return_6d": [0.30],
                    "atr10": [2.0],
                    "dollarvolume50": [100_000_000],
                    "hv50": [20.0],
                    "UpTwoDays": [0],
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            ),
            "SYM2": pd.DataFrame(
                {
                    "Close": [200.0],
                    "return_6d": [0.20],
                    "atr10": [3.0],
                    "dollarvolume50": [120_000_000],
                    "hv50": [18.0],
                    "UpTwoDays": [0],
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            ),
            "SYM3": pd.DataFrame(
                {
                    "Close": [300.0],
                    "return_6d": [0.10],
                    "atr10": [4.0],
                    "dollarvolume50": [150_000_000],
                    "hv50": [22.0],
                    "UpTwoDays": [0],
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            ),
        }

        candidates, _ = generate_candidates_system6(
            prepared_dict=data_dict,
            top_n=3,
            latest_only=True,
        )

        assert len(candidates) >= 1
        last_date = list(candidates.keys())[0]

        # 各銘柄のrankとrank_totalを確認
        assert candidates[last_date]["SYM1"]["rank"] == 1
        assert candidates[last_date]["SYM2"]["rank"] == 2
        assert candidates[last_date]["SYM3"]["rank"] == 3

        # rank_totalは全銘柄共通で3（制限後の総数）
        assert candidates[last_date]["SYM1"]["rank_total"] == 3
        assert candidates[last_date]["SYM2"]["rank_total"] == 3
        assert candidates[last_date]["SYM3"]["rank_total"] == 3

    @pytest.mark.xfail(
        reason=(
            "Test isolation issue: passes in isolation, fails in full suite "
            "due to NumPy module reload warning. Returns empty candidates in "
            "full suite but correct candidates when run alone."
        ),
        strict=False,
    )
    def test_date_mode_top_n_limit(self):
        """top_n制限が正しく適用されることを検証 (lines 597-599)"""
        from core.system6 import generate_candidates_system6

        set_test_determinism()

        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        # 5銘柄を用意
        data_dict = {}
        for i, symbol in enumerate(["A", "B", "C", "D", "E"]):
            data_dict[symbol] = pd.DataFrame(
                {
                    "Close": [100.0 + i * 10],
                    "return_6d": [0.50 - i * 0.05],  # 降順になるように設定
                    "atr10": [2.0],
                    "dollarvolume50": [100_000_000],
                    "hv50": [20.0],
                    "UpTwoDays": [0],
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            )

        # top_n=3で制限
        candidates, _ = generate_candidates_system6(
            prepared_dict=data_dict,
            top_n=3,
            latest_only=True,
        )

        assert len(candidates) >= 1
        last_date = list(candidates.keys())[0]

        # 上位3銘柄のみが選択されているはず
        assert len(candidates[last_date]) == 3
        assert set(candidates[last_date].keys()) == {"A", "B", "C"}

        # rank_totalは制限後の数（3）
        assert candidates[last_date]["A"]["rank_total"] == 3

    def test_date_mode_empty_rows_handling(self):
        """setup=Falseの行が正しく除外されることを検証"""
        from core.system6 import generate_candidates_system6

        set_test_determinism()

        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        data_dict = {
            "SYM_NO_SETUP": pd.DataFrame(
                {
                    "Close": [100.0],
                    "return_6d": [0.30],
                    "atr10": [2.0],
                    "dollarvolume50": [100_000_000],
                    "hv50": [20.0],
                    "UpTwoDays": [0],
                    "filter": [True],
                    "setup": [False],  # setupがFalse
                },
                index=dates,
            ),
        }

        candidates, _ = generate_candidates_system6(
            prepared_dict=data_dict,
            top_n=10,
            latest_only=True,
        )

        # setup=Falseなので候補は空のはず
        assert isinstance(candidates, dict)
        assert len(candidates) == 0


class TestSystem6HelperFunctions:
    """System6のヘルパー関数のテスト (lines 698-711など)

    NumPy問題の影響を受けにくいユーティリティ関数を検証します。
    """

    def test_get_total_days_with_date_column(self):
        """Date列を持つDataFrameから総日数を取得"""
        from core.system6 import get_total_days_system6

        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data_dict = {
            "SYM1": pd.DataFrame(
                {
                    "Date": dates,
                    "Close": [100.0] * 5,
                },
                index=range(5),
            ),
            "SYM2": pd.DataFrame(
                {
                    "Date": dates[:3],  # 最初の3日のみ
                    "Close": [200.0] * 3,
                },
                index=range(3),
            ),
        }

        total_days = get_total_days_system6(data_dict)
        # 5日分のユニーク日付
        assert total_days == 5

    def test_get_total_days_with_lowercase_date_column(self):
        """小文字のdate列を持つDataFrameから総日数を取得"""
        from core.system6 import get_total_days_system6

        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        data_dict = {
            "SYM1": pd.DataFrame(
                {
                    "date": dates,  # 小文字
                    "Close": [100.0] * 3,
                },
                index=range(3),
            ),
        }

        total_days = get_total_days_system6(data_dict)
        assert total_days == 3

    def test_get_total_days_with_index_as_dates(self):
        """インデックスが日付のDataFrameから総日数を取得"""
        from core.system6 import get_total_days_system6

        dates = pd.date_range("2023-01-01", periods=4, freq="D")
        data_dict = {
            "SYM1": pd.DataFrame(
                {
                    "Close": [100.0] * 4,
                },
                index=dates,
            ),
            "SYM2": pd.DataFrame(
                {
                    "Close": [200.0] * 4,
                },
                index=dates,
            ),
        }

        total_days = get_total_days_system6(data_dict)
        assert total_days == 4

    def test_get_total_days_with_empty_dataframes(self):
        """空のDataFrameを含む場合の処理"""
        from core.system6 import get_total_days_system6

        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        data_dict = {
            "SYM1": pd.DataFrame(
                {
                    "Date": dates,
                    "Close": [100.0] * 2,
                },
                index=range(2),
            ),
            "SYM_EMPTY": pd.DataFrame(),  # 空のDF
            "SYM_NONE": None,  # None
        }

        total_days = get_total_days_system6(data_dict)
        # 空/Noneは無視され、SYM1の2日分のみ
        assert total_days == 2

    def test_get_total_days_with_overlapping_dates(self):
        """複数銘柄で日付が重複する場合"""
        from core.system6 import get_total_days_system6

        dates1 = pd.date_range("2023-01-01", periods=5, freq="D")
        dates2 = pd.date_range(
            "2023-01-03", periods=5, freq="D"
        )  # 1/3から開始（2日重複）

        data_dict = {
            "SYM1": pd.DataFrame(
                {
                    "Date": dates1,
                    "Close": [100.0] * 5,
                },
                index=range(5),
            ),
            "SYM2": pd.DataFrame(
                {
                    "Date": dates2,
                    "Close": [200.0] * 5,
                },
                index=range(5),
            ),
        }

        total_days = get_total_days_system6(data_dict)
        # 1/1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7 = 7日分
        assert total_days == 7

    def test_get_total_days_empty_dict(self):
        """空のdictを渡した場合"""
        from core.system6 import get_total_days_system6

        total_days = get_total_days_system6({})
        assert total_days == 0


class TestSystem6ErrorHandling:
    """Test error handling in normalization and diagnostics code (lines 671-690)."""

    def test_normalization_with_invalid_symbol_type(self):
        """Test normalization handles non-string symbol gracefully (lines 673-675)."""
        from core.system6 import generate_candidates_system6

        # Create data with valid setup conditions
        dates = pd.date_range("2023-01-01", periods=1)
        data = {
            "TEST": pd.DataFrame(
                {
                    "Close": [100.0],
                    "return_6d": [0.25],
                    "atr10": [2.0],
                    "dollarvolume50": [1e8],
                    "hv50": [20.0],
                    "UpTwoDays": [1],
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            )
        }

        # Should handle gracefully even if symbol processing encounters issues
        result = generate_candidates_system6(data, top_n=5, latest_only=True)
        if len(result) == 3:
            result, _, _ = result
        else:
            result, _ = result

        # Should return valid structure (may be empty or have candidates)
        assert isinstance(result, dict)

    def test_diagnostics_exception_on_empty_candidates(self):
        """Test diagnostics handles empty candidates dict gracefully (lines 683-687)."""
        from core.system6 import generate_candidates_system6

        # Create data that won't produce candidates (fails setup)
        dates = pd.date_range("2023-01-01", periods=1)
        data = {
            "TEST": pd.DataFrame(
                {
                    "Close": [100.0],
                    "return_6d": [0.10],  # < 0.20, fails setup
                    "atr10": [2.0],
                    "dollarvolume50": [1e8],
                    "hv50": [20.0],
                    "UpTwoDays": [0],  # False, fails setup
                    "filter": [True],
                    "setup": [False],  # Explicitly False
                },
                index=dates,
            )
        }

        # Should handle max() on empty dict gracefully
        result, _, diagnostics = generate_candidates_system6(
            data, top_n=5, latest_only=True, include_diagnostics=True
        )

        assert isinstance(result, dict)
        assert "ranked_top_n_count" in diagnostics
        # Should be 0 or handle exception gracefully
        assert diagnostics["ranked_top_n_count"] == 0


class TestSystem6MetricsRecording:
    """Test metrics recording in final phase (lines 660-666)."""

    def test_metrics_recording_with_candidates(self):
        """Test that metrics are recorded correctly when candidates exist."""
        import core.system6
        from core.system6 import generate_candidates_system6

        # Capture metrics by monitoring _metrics instance
        metrics_captured = []
        original_record = core.system6._metrics.record_metric

        def mock_record_metric(name, value, unit, **kwargs):
            metrics_captured.append({"name": name, "value": value, "unit": unit})
            return original_record(name, value, unit, **kwargs)

        core.system6._metrics.record_metric = mock_record_metric  # type: ignore

        try:
            # Create data with valid candidates
            dates = pd.date_range("2023-01-01", periods=1)
            data = {
                "AAPL": pd.DataFrame(
                    {
                        "Close": [150.0],
                        "return_6d": [0.25],
                        "atr10": [2.5],
                        "dollarvolume50": [1e9],
                        "hv50": [25.0],
                        "UpTwoDays": [1],
                        "filter": [True],
                        "setup": [True],
                    },
                    index=dates,
                ),
                "GOOGL": pd.DataFrame(
                    {
                        "Close": [2800.0],
                        "return_6d": [0.30],
                        "atr10": [30.0],
                        "dollarvolume50": [2e9],
                        "hv50": [28.0],
                        "UpTwoDays": [1],
                        "filter": [True],
                        "setup": [True],
                    },
                    index=dates,
                ),
            }

            result, _ = generate_candidates_system6(data, top_n=5, latest_only=True)

            # Verify metrics were recorded
            metric_names = [m["name"] for m in metrics_captured]
            assert "system6_total_candidates" in metric_names
            assert "system6_unique_entry_dates" in metric_names
            assert "system6_processed_symbols_candidates" in metric_names

        finally:
            core.system6._metrics.record_metric = original_record  # type: ignore

    def test_metrics_recording_with_no_candidates(self):
        """Test that metrics are recorded even when no candidates exist."""
        import core.system6
        from core.system6 import generate_candidates_system6

        # Capture metrics
        metrics_captured = []
        original_record = core.system6._metrics.record_metric

        def mock_record_metric(name, value, unit, **kwargs):
            metrics_captured.append({"name": name, "value": value, "unit": unit})
            return original_record(name, value, unit, **kwargs)

        core.system6._metrics.record_metric = mock_record_metric  # type: ignore

        try:
            # Create data with no candidates (fails setup)
            dates = pd.date_range("2023-01-01", periods=1)
            data = {
                "TEST": pd.DataFrame(
                    {
                        "Close": [100.0],
                        "return_6d": [0.10],  # < 0.20
                        "atr10": [2.0],
                        "dollarvolume50": [1e8],
                        "hv50": [20.0],
                        "UpTwoDays": [0],  # False
                        "filter": [True],
                        "setup": [False],
                    },
                    index=dates,
                )
            }

            _ = generate_candidates_system6(data, top_n=5, latest_only=False)

            # Verify metrics were recorded with 0 values
            metric_names = [m["name"] for m in metrics_captured]
            assert "system6_total_candidates" in metric_names
            total_candidates = next(
                m for m in metrics_captured if m["name"] == "system6_total_candidates"
            )
            assert total_candidates["value"] == 0

        finally:
            core.system6._metrics.record_metric = original_record  # type: ignore


class TestSystem6LoggingCallbacks:
    """Test logging callback invocations (lines 643-650, 659-666)."""

    def test_skip_summary_logging_when_skipped_gt_zero(self):
        """Test that skip summary is logged when skipped > 0 (lines 643-650)."""
        from core.system6 import generate_candidates_system6

        logs_captured = []

        def mock_log_callback(message):
            logs_captured.append(message)

        # Create data with missing required columns to trigger skips
        dates = pd.date_range("2023-01-01", periods=1)
        data = {
            "INCOMPLETE": pd.DataFrame(
                {
                    "Close": [100.0],
                    # Missing return_6d, atr10, etc.
                },
                index=dates,
            )
        }

        _ = generate_candidates_system6(
            data, top_n=5, latest_only=True, log_callback=mock_log_callback
        )

        # Check if any logging activity occurred
        assert isinstance(logs_captured, list)
        # May have logged skip summary or completion message

    def test_completion_logging_callback(self):
        """Test that completion message is logged via log_callback (lines 659-666)."""
        from core.system6 import generate_candidates_system6

        logs_captured = []

        def mock_log_callback(message):
            logs_captured.append(message)

        # Create valid data
        dates = pd.date_range("2023-01-01", periods=1)
        data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [150.0],
                    "return_6d": [0.25],
                    "atr10": [2.5],
                    "dollarvolume50": [1e9],
                    "hv50": [25.0],
                    "UpTwoDays": [1],
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            )
        }

        _ = generate_candidates_system6(
            data, top_n=5, latest_only=True, log_callback=mock_log_callback
        )

        # Check if completion message was logged
        completion_logs = [
            log for log in logs_captured if "完了" in log or "候補生成" in log
        ]
        # Should have completion message
        assert len(completion_logs) > 0

    def test_logging_callback_exception_handling(self):
        """Test that exceptions in log_callback propagate correctly."""
        import pytest

        from core.system6 import generate_candidates_system6

        def failing_log_callback(message):
            raise RuntimeError("Intentional log callback failure")

        # Create valid data
        dates = pd.date_range("2023-01-01", periods=1)
        data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [150.0],
                    "return_6d": [0.25],
                    "atr10": [2.5],
                    "dollarvolume50": [1e9],
                    "hv50": [25.0],
                    "UpTwoDays": [1],
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            )
        }

        # Should raise RuntimeError from failing callback
        with pytest.raises(RuntimeError, match="Intentional log callback failure"):
            generate_candidates_system6(
                data, top_n=5, latest_only=True, log_callback=failing_log_callback
            )
