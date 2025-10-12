"""
System7 latest_only fast-path coverage tests.

Target: Lines 219-262 (約18%カバレッジ向上)
"""

import pandas as pd

from core.system7 import generate_candidates_system7, prepare_data_vectorized_system7


class TestSystem7LatestOnlyFastPath:
    """latest_only=True でのfast-path実行をテスト"""

    def create_spy_data_for_latest_only(self, setup_today: bool = True) -> pd.DataFrame:
        """latest_only fast-pathに必要な最小SPYデータを生成"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = [100 + i * 0.5 for i in range(100)]
        lows = [p * 0.98 for p in prices]
        closes = [p * 1.01 for p in prices]

        # min_50: 50日最小値
        min_50_series = pd.Series(lows).rolling(window=50, min_periods=1).min()
        # max_70: 70日最大値
        max_70_series = pd.Series(prices).rolling(window=70, min_periods=1).max()

        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.02 for p in prices],
                "Low": lows,
                "Close": closes,
                "Volume": [1000000] * 100,
                "atr50": [p * 0.02 for p in prices],  # Required by System7
                "min_50": min_50_series.values,
                "max_70": max_70_series.values,
            },
            index=dates,
        )

        # setup条件: Low <= min_50 を満たすよう最終日のLowを調整
        if setup_today:
            # 最終日のLowをmin_50以下にする
            final_min50 = df["min_50"].iloc[-1]
            df.loc[df.index[-1], "Low"] = final_min50 * 0.99  # min_50より少し低く

        return df

    def test_latest_only_fast_path_basic(self):
        """latest_only=True で fast-path が実行されることを確認 (Lines 219-262)"""
        spy_data = self.create_spy_data_for_latest_only(setup_today=True)
        raw_dict = {"SPY": spy_data}

        # prepare_data で指標計算 (既にatr50, min_50, max_70は含まれている)
        prepared_dict = prepare_data_vectorized_system7(
            raw_dict, reuse_indicators=False
        )

        # latest_only=True で呼び出し
        result = generate_candidates_system7(
            prepared_dict, top_n=5, latest_only=True, include_diagnostics=True
        )

        # 結果検証
        assert len(result) == 3, "Should return (normalized, df, diagnostics)"
        normalized, df_result, diagnostics = result

        # fast-path が実行されたことを確認
        ranking_src = diagnostics.get("ranking_source")
        assert (
            ranking_src == "latest_only"
        ), f"Expected 'latest_only', got {ranking_src}"
        assert (
            diagnostics.get("ranked_top_n_count") == 1
        ), "latest_only should return 1 candidate"

        # normalized に SPY が含まれる
        assert len(normalized) > 0, "Should have at least one date"
        first_date = list(normalized.keys())[0]
        assert "SPY" in normalized[first_date], "Should contain SPY"

        # DataFrame result
        assert not df_result.empty, "Result DataFrame should not be empty"
        assert "SPY" in df_result["symbol"].values, "SPY should be in result"

    def test_latest_only_with_close_price(self):
        """latest_only で Close 価格が正しく entry_price にセットされる (Line 228-229)"""
        spy_data = self.create_spy_data_for_latest_only(setup_today=True)
        expected_close = spy_data["Close"].iloc[-1]
        raw_dict = {"SPY": spy_data}

        prepared_dict = prepare_data_vectorized_system7(
            raw_dict, reuse_indicators=False
        )

        result = generate_candidates_system7(
            prepared_dict, top_n=5, latest_only=True, include_diagnostics=True
        )
        normalized, df_result, diagnostics = result

        # entry_price が Close と一致
        first_date = list(normalized.keys())[0]
        spy_payload = normalized[first_date]["SPY"]
        actual_price = spy_payload.get("entry_price")
        assert (
            actual_price == expected_close
        ), f"Expected {expected_close}, got {actual_price}"

    def test_latest_only_with_atr50_lowercase(self):
        """latest_only で atr50 (lowercase) が ATR50 として取得される (Line 230-232)"""
        spy_data = self.create_spy_data_for_latest_only(setup_today=True)
        expected_atr = spy_data["atr50"].iloc[-1]
        data_dict = {"SPY": spy_data}

        result = generate_candidates_system7(
            data_dict, top_n=5, latest_only=True, include_diagnostics=True
        )
        normalized, df_result, diagnostics = result

        # ATR50 が正しく設定されている
        first_date = list(normalized.keys())[0]
        spy_payload = normalized[first_date]["SPY"]
        assert (
            spy_payload.get("ATR50") == expected_atr
        ), f"Expected ATR50={expected_atr}, got {spy_payload.get('ATR50')}"

    def test_latest_only_df_result_structure(self):
        """latest_only の DataFrame 結果が正しい構造を持つ (Lines 233-244)"""
        spy_data = self.create_spy_data_for_latest_only(setup_today=True)
        data_dict = {"SPY": spy_data}

        result = generate_candidates_system7(
            data_dict, top_n=5, latest_only=True, include_diagnostics=True
        )
        normalized, df_result, diagnostics = result

        # DataFrame の必須カラム
        assert "symbol" in df_result.columns
        assert "date" in df_result.columns
        assert "ATR50" in df_result.columns
        assert "entry_price" in df_result.columns
        assert "rank" in df_result.columns  # Line 242
        assert "rank_total" in df_result.columns  # Line 243

        # rank が 1 に設定されている
        assert df_result["rank"].iloc[0] == 1
        assert df_result["rank_total"].iloc[0] == 1

    def test_latest_only_log_callback_invocation(self):
        """latest_only で log_callback が呼ばれる (Lines 250-254)"""
        spy_data = self.create_spy_data_for_latest_only(setup_today=True)
        data_dict = {"SPY": spy_data}

        log_messages = []

        def mock_log_callback(msg: str):
            log_messages.append(msg)

        _ = generate_candidates_system7(
            data_dict,
            top_n=5,
            latest_only=True,
            include_diagnostics=True,
            log_callback=mock_log_callback,
        )

        # log_callback が呼ばれたことを確認
        assert len(log_messages) > 0, "log_callback should be called"
        assert any(
            "latest_only" in msg for msg in log_messages
        ), f"Expected 'latest_only' in log messages, got {log_messages}"

    def test_latest_only_progress_callback_invocation(self):
        """latest_only で progress_callback が呼ばれる (Lines 257-261)"""
        spy_data = self.create_spy_data_for_latest_only(setup_today=True)
        data_dict = {"SPY": spy_data}

        progress_calls = []

        def mock_progress_callback(current: int, total: int):
            progress_calls.append((current, total))

        _ = generate_candidates_system7(
            data_dict,
            top_n=5,
            latest_only=True,
            include_diagnostics=True,
            progress_callback=mock_progress_callback,
        )

        # progress_callback が呼ばれたことを確認
        assert len(progress_calls) > 0, "progress_callback should be called"
        assert progress_calls[0] == (
            1,
            1,
        ), f"Expected progress (1, 1), got {progress_calls[0]}"

    def test_latest_only_no_setup_today(self):
        """latest_only で最終日に setup=False の場合、候補なし"""
        spy_data = self.create_spy_data_for_latest_only(setup_today=False)
        # 最終日を setup=False に明示的に設定
        spy_data.loc[spy_data.index[-1], "setup"] = False
        data_dict = {"SPY": spy_data}

        result = generate_candidates_system7(
            data_dict, top_n=5, latest_only=True, include_diagnostics=True
        )
        normalized, df_result, diagnostics = result

        # setup が False なので候補なし
        assert (
            diagnostics.get("ranked_top_n_count") == 0
        ), "Should have 0 candidates when setup=False"
        assert (
            diagnostics.get("ranking_source") is None
            or diagnostics.get("ranking_source") != "latest_only"
        ), "Should not use latest_only path"

    def test_latest_only_normalized_structure(self):
        """latest_only の normalized dict が正しい構造を持つ (Lines 244-248)"""
        spy_data = self.create_spy_data_for_latest_only(setup_today=True)
        data_dict = {"SPY": spy_data}

        result = generate_candidates_system7(
            data_dict, top_n=5, latest_only=True, include_diagnostics=True
        )
        normalized, df_result, diagnostics = result

        # normalized の構造確認
        assert len(normalized) == 1, "Should have exactly 1 entry date"
        first_date = list(normalized.keys())[0]
        assert isinstance(first_date, pd.Timestamp)

        spy_payload = normalized[first_date]["SPY"]
        # entry_date が含まれる (Line 247)
        assert "entry_date" in spy_payload
        assert isinstance(spy_payload["entry_date"], pd.Timestamp)

        # symbol/date 以外のフィールドが含まれる (Lines 245-246)
        assert "ATR50" in spy_payload
        assert "entry_price" in spy_payload
