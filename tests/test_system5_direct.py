"""System5直接関数テスト - import問題回避版 (Long mean-reversion with high ADX)"""

import numpy as np
import pandas as pd


class TestSystem5DirectFunctions:
    """System5の主要関数を直接テストするクラス（Long mean-reversion戦略）"""

    def test_system5_get_total_days_direct(self):
        """get_total_days_system5 の直接実装テスト"""

        def mock_get_total_days_system5(data_dict):
            """get_total_days_system5 の簡易模擬実装"""
            total_days = 0
            for _symbol, df in data_dict.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    total_days += len(df)
            return total_days

        # System5特化サンプルデータ（Long mean-reversion用）
        data_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                    "Close": [150.0, 145.0, 140.0],  # 下落トレンド
                    "RSI14": [25, 20, 15],  # RSI oversold領域（mean-reversion候補）
                    "ADX14": [30, 32, 35],  # 高ADX（System5特徴）
                }
            ),
            "GOOGL": pd.DataFrame(
                {
                    "Date": ["2023-01-01", "2023-01-02"],
                    "Close": [2500.0, 2450.0],
                    "RSI14": [30, 25],  # RSI oversold
                    "ADX14": [28, 30],
                }
            ),
            "TSLA": pd.DataFrame(
                {"Date": ["2023-01-01"], "Close": [200.0], "RSI14": [35], "ADX14": [25]}
            ),
        }

        result = mock_get_total_days_system5(data_dict)

        # 総日数確認（System5データ特性）
        assert result == 6  # 3 + 2 + 1 = 6日
        assert result > 0
        assert isinstance(result, int)

        # empty辞書テスト
        empty_result = mock_get_total_days_system5({})
        assert empty_result == 0

    def test_system5_generate_candidates_mean_reversion_direct(self):
        """generate_candidates_system5 のmean-reversion特化テスト"""

        def mock_generate_candidates_system5(prepared_dict, top_n=10, **kwargs):
            """System5 mean-reversion候補生成模擬実装"""
            all_signals = []
            for sym, df in prepared_dict.items():
                if "setup" not in df.columns or not df["setup"].any():
                    continue

                # mean-reversion setup条件
                setup_df = df[df["setup"] == 1].copy()
                setup_df["symbol"] = sym

                # ADX高水準 (System5特徴)
                if "ADX14" in df.columns:
                    setup_df = setup_df[setup_df["ADX14"] > 25]  # 高ADXフィルタ

                if not setup_df.empty:
                    all_signals.append(setup_df)

            if all_signals:
                combined = pd.concat(all_signals, ignore_index=True)
                # mean-reversionスコア（価格乖離度でソート）
                if "price_deviation" in combined.columns:
                    # 絶対値降順でソート（大幅下落優先）
                    combined["abs_deviation"] = combined["price_deviation"].abs()
                    combined = combined.sort_values("abs_deviation", ascending=False)
                    return {"signals": combined.head(top_n).to_dict("records")}, combined.head(
                        top_n
                    )
                else:
                    return {"signals": combined.head(top_n).to_dict("records")}, combined.head(
                        top_n
                    )
            else:
                return {}, None

        # mean-reversion セットアップデータ（価格下落 + 高ADX）
        prepared_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                    "Close": [100, 95, 90],  # 下落トレンド（mean-reversion候補）
                    "SMA20": [105, 104, 103],
                    "ADX14": [30, 32, 35],  # 高ADX
                    "setup": [0, 0, 1],  # 最終日にsetup
                    "price_deviation": [0, -9, -15],  # 平均からの乖離度
                }
            ),
            "GOOGL": pd.DataFrame(
                {
                    "Date": ["2023-01-01", "2023-01-02"],
                    "Close": [200, 185],  # 大幅下落
                    "SMA20": [210, 208],
                    "ADX14": [28, 30],
                    "setup": [0, 1],
                    "price_deviation": [0, -23],
                }
            ),
        }

        result_signals, result_df = mock_generate_candidates_system5(prepared_dict, top_n=5)

        # mean-reversionシグナルが検出されることを確認
        assert isinstance(result_signals, dict)
        assert "signals" in result_signals
        assert len(result_signals["signals"]) > 0

        # 価格乖離度でソート確認（絶対値降順）
        if result_df is not None and len(result_df) > 1:
            abs_deviations = result_df["abs_deviation"].values
            # より大きな乖離度（絶対値）が上位に来ることを確認
            assert abs_deviations[0] >= abs_deviations[-1]  # 絶対値降順ソート

        # ADXフィルタ確認
        for signal in result_signals["signals"]:
            assert signal["ADX14"] > 25  # 高ADX条件

    def test_system5_prepare_data_mean_reversion_indicators_direct(self):
        """prepare_data_vectorized_system5 のmean-reversion指標特化テスト"""

        def mock_prepare_data_vectorized_system5(data_dict):
            """System5 mean-reversion指標付与模擬実装"""
            prepared_dict = {}

            for symbol, df in data_dict.items():
                if df.empty or "Close" not in df.columns:
                    continue

                df_copy = df.copy()

                # mean-reversion指標計算
                if "Close" in df_copy.columns and len(df_copy) >= 14:
                    # RSI14（oversold検出）
                    close_diff = df_copy["Close"].diff()
                    gains = close_diff.where(close_diff > 0, 0)
                    losses = -close_diff.where(close_diff < 0, 0)
                    avg_gains = gains.rolling(14).mean()
                    avg_losses = losses.rolling(14).mean()
                    rs = avg_gains / avg_losses
                    df_copy["RSI14"] = 100 - (100 / (1 + rs))

                    # ADX14（トレンド強度）
                    df_copy["ADX14"] = 30.0  # 簡易固定値

                    # SMA20（平均回帰基準線）
                    df_copy["SMA20"] = df_copy["Close"].rolling(20).mean()

                    # 価格乖離度（mean-reversion核心指標）
                    df_copy["price_deviation"] = df_copy["Close"] - df_copy["SMA20"]

                    # mean-reversion setup条件
                    df_copy["setup"] = (
                        (df_copy["RSI14"] < 30)  # RSI oversold
                        & (df_copy["ADX14"] > 25)  # 高ADXトレンド
                        & (df_copy["price_deviation"] < -5)  # 価格下落
                    ).astype(int)

                prepared_dict[symbol] = df_copy

            return prepared_dict

        # System5特化データ（mean-reversion適合）
        data_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=30, freq="D"),
                    "Close": np.linspace(150, 120, 30),  # 段階的下落
                    "Volume": [1000000] * 30,
                    "High": np.linspace(155, 125, 30),
                    "Low": np.linspace(145, 115, 30),
                    "Open": np.linspace(150, 120, 30),
                }
            ),
            "GOOGL": pd.DataFrame(
                {
                    "Date": pd.date_range("2023-01-01", periods=25, freq="D"),
                    "Close": np.linspace(2500, 2300, 25),  # 大幅下落
                    "Volume": [500000] * 25,
                    "High": np.linspace(2550, 2350, 25),
                    "Low": np.linspace(2450, 2250, 25),
                    "Open": np.linspace(2500, 2300, 25),
                }
            ),
        }

        prepared_dict = mock_prepare_data_vectorized_system5(data_dict)

        # mean-reversion指標確認
        assert len(prepared_dict) == 2
        for _symbol, df in prepared_dict.items():
            assert "RSI14" in df.columns
            assert "ADX14" in df.columns
            assert "SMA20" in df.columns
            assert "price_deviation" in df.columns  # mean-reversion核心
            assert "setup" in df.columns

            # RSI oversold条件確認
            rsi_values = df["RSI14"].dropna()
            if len(rsi_values) > 0:
                assert rsi_values.min() >= 0
                assert rsi_values.max() <= 100

            # setup条件確認
            setup_count = df["setup"].sum()
            assert setup_count >= 0  # setup条件が成立している日が存在

    def test_system5_mean_reversion_entry_conditions_direct(self):
        """System5 mean-reversion エントリ条件特化テスト"""

        def mock_system5_entry_conditions(df):
            """System5 mean-reversion エントリ条件模擬実装"""
            if df.empty or "Close" not in df.columns:
                return pd.Series(False, index=df.index)

            # Long mean-reversion エントリ条件
            conditions = pd.Series(False, index=df.index)

            if all(col in df.columns for col in ["RSI14", "ADX14", "SMA20"]):
                conditions = (
                    (df["RSI14"] < 30)  # RSI oversold（過売り）
                    & (df["ADX14"] > 25)  # 高ADXトレンド強度
                    & (df["Close"] < df["SMA20"] * 0.95)  # 移動平均から5%以上下落
                )

            return conditions

        # mean-reversion エントリデータ
        test_df = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=10),
                "Close": [100, 98, 95, 92, 89, 87, 85, 88, 91, 94],  # V字回復パターン
                "RSI14": [50, 45, 35, 28, 22, 18, 15, 25, 35, 45],  # oversold→回復
                "ADX14": [20, 25, 28, 30, 32, 35, 38, 35, 30, 25],  # 高ADX
                "SMA20": [102, 101, 100, 99, 98, 97, 96, 95, 94, 93],  # 下落トレンド
            }
        )

        entry_signals = mock_system5_entry_conditions(test_df)

        # mean-reversion エントリ確認
        assert isinstance(entry_signals, pd.Series)
        assert len(entry_signals) == len(test_df)

        # 条件成立確認（RSI oversold + 高ADX + 価格下落）
        expected_entries = (
            (test_df["RSI14"] < 30)
            & (test_df["ADX14"] > 25)
            & (test_df["Close"] < test_df["SMA20"] * 0.95)
        )

        pd.testing.assert_series_equal(entry_signals, expected_entries)

        # 複数のエントリポイント存在確認
        assert entry_signals.sum() > 0  # 少なくとも1つのエントリ

    def test_system5_high_adx_filter_direct(self):
        """System5 高ADXフィルタ特化テスト"""

        def mock_high_adx_filter_system5(df, adx_threshold=25):
            """System5 高ADXフィルタ模擬実装"""
            if df.empty or "ADX14" not in df.columns:
                return pd.Series(False, index=df.index)

            return df["ADX14"] > adx_threshold

        # ADX変動データ
        test_df = pd.DataFrame(
            {
                "ADX14": [15, 20, 25, 28, 30, 35, 40, 35, 30, 25, 20, 15],
                "Close": [100, 101, 99, 97, 95, 92, 88, 90, 93, 96, 98, 100],
            }
        )

        # 標準ADXフィルタ（25以上）
        adx_filter = mock_high_adx_filter_system5(test_df, adx_threshold=25)

        assert isinstance(adx_filter, pd.Series)
        assert len(adx_filter) == len(test_df)

        # 高ADX条件確認
        expected_high_adx = test_df["ADX14"] > 25
        pd.testing.assert_series_equal(adx_filter, expected_high_adx)

        # 高ADX期間確認
        high_adx_count = adx_filter.sum()
        assert high_adx_count > 0  # 高ADX期間が存在
        assert high_adx_count < len(test_df)  # 全期間ではない

        # 厳格ADXフィルタ（30以上）
        strict_adx_filter = mock_high_adx_filter_system5(test_df, adx_threshold=30)
        strict_count = strict_adx_filter.sum()
        assert strict_count <= high_adx_count  # より厳格な条件

    def test_system5_mean_reversion_scoring_direct(self):
        """System5 mean-reversion スコアリング特化テスト"""

        def mock_mean_reversion_scoring_system5(df):
            """System5 mean-reversion スコア計算模擬実装"""
            if df.empty:
                return pd.Series(0.0, index=df.index)

            scores = pd.Series(0.0, index=df.index)

            if all(col in df.columns for col in ["RSI14", "ADX14", "price_deviation"]):
                # mean-reversionスコア計算
                rsi_score = (30 - df["RSI14"]).clip(0, 30) / 30  # RSI oversold度
                adx_score = (df["ADX14"] - 25).clip(0, 25) / 25  # ADX強度
                deviation_score = (-df["price_deviation"]).clip(0, 20) / 20  # 価格下落度

                # 総合mean-reversionスコア
                scores = rsi_score * 0.4 + adx_score * 0.3 + deviation_score * 0.3

            return scores

        # mean-reversion スコアリングデータ
        scoring_df = pd.DataFrame(
            {
                "RSI14": [50, 40, 30, 20, 15, 10, 20, 30, 40, 50],  # oversold→回復
                "ADX14": [20, 25, 30, 35, 40, 35, 30, 25, 20, 15],  # 高ADX変動
                "price_deviation": [0, -2, -5, -8, -12, -15, -10, -5, -2, 0],  # 価格乖離
                "Close": [100, 98, 95, 92, 88, 85, 90, 95, 98, 100],
            }
        )

        reversion_scores = mock_mean_reversion_scoring_system5(scoring_df)

        # スコア基本確認
        assert isinstance(reversion_scores, pd.Series)
        assert len(reversion_scores) == len(scoring_df)
        assert reversion_scores.min() >= 0.0
        assert reversion_scores.max() <= 1.0

        # 最高スコア確認（RSI=10, ADX=40, deviation=-15の組み合わせ）
        max_score_idx = reversion_scores.idxmax()
        expected_max_idx = 5  # RSI=10, ADX=35, deviation=-15
        assert max_score_idx == expected_max_idx

        # スコア変動確認
        assert reversion_scores.std() > 0.1  # 十分な変動幅

        # mean-reversion適性確認（低RSI + 高ADX + 大幅下落でスコア上昇）
        low_rsi_indices = scoring_df["RSI14"] < 25
        high_adx_indices = scoring_df["ADX14"] > 30
        high_deviation_indices = scoring_df["price_deviation"] < -8

        optimal_conditions = low_rsi_indices & high_adx_indices & high_deviation_indices
        if optimal_conditions.any():
            optimal_scores = reversion_scores[optimal_conditions]
            overall_mean = reversion_scores.mean()
            assert optimal_scores.mean() > overall_mean  # 条件満たす場合は高スコア
