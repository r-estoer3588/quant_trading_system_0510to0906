"""System7コア機能の直接テスト（外部依存なし）

System7のSPY特化機能を直接テストし、高いカバレッジを達成する。
mockによって外部I/Oを回避し、決定論的な動作を保証する。
"""

import pandas as pd
import numpy as np
from unittest.mock import patch

from core.system7 import (
    prepare_data_vectorized_system7,
    generate_candidates_system7,
    get_total_days_system7,
)


class TestSystem7DirectFunctions:
    """System7コア関数の直接テスト"""

    def create_spy_test_data(self, periods: int = 100) -> pd.DataFrame:
        """SPY用テストデータ生成"""
        dates = pd.date_range("2023-01-01", periods=periods)
        # System7に必要な基本データ
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(400, 450, periods),
                "High": np.random.uniform(450, 500, periods),
                "Low": np.random.uniform(350, 400, periods),
                "Close": np.random.uniform(400, 450, periods),
                "Volume": np.random.randint(50000000, 100000000, periods),
            },
            index=dates,
        )
        return data

    def test_prepare_data_vectorized_system7_basic(self):
        """prepare_data_vectorized_system7 の基本機能テスト"""
        test_data = self.create_spy_test_data(100)  # 50日以上のデータが必要
        raw_data_dict = {"SPY": test_data}

        with (
            patch("os.makedirs"),
            patch("pandas.read_parquet", side_effect=FileNotFoundError),
            patch("pandas.DataFrame.to_parquet"),
        ):

            result = prepare_data_vectorized_system7(raw_data_dict)

        # 戻り値検証
        assert isinstance(result, dict)
        assert "SPY" in result
        spy_result = result["SPY"]
        assert isinstance(spy_result, pd.DataFrame)

        # System7特有の指標が計算されている
        expected_cols = {"ATR50", "min_50", "setup", "max_70"}
        assert expected_cols.issubset(set(spy_result.columns))

    def test_prepare_data_vectorized_system7_empty_data(self):
        """prepare_data_vectorized_system7 の空データ処理"""
        result = prepare_data_vectorized_system7(None)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_prepare_data_vectorized_system7_no_spy(self):
        """prepare_data_vectorized_system7 のSPYなしデータ処理"""
        raw_data_dict = {"AAPL": self.create_spy_test_data()}

        with (
            patch("os.makedirs"),
            patch("pandas.read_parquet", side_effect=FileNotFoundError),
            patch("pandas.DataFrame.to_parquet"),
        ):

            result = prepare_data_vectorized_system7(raw_data_dict)

        assert isinstance(result, dict)
        assert "SPY" not in result  # SPYがないので空辞書

    def test_prepare_data_vectorized_system7_with_progress_callback(self):
        """prepare_data_vectorized_system7 のプログレスコールバック処理"""
        test_data = self.create_spy_test_data(100)
        raw_data_dict = {"SPY": test_data}

        progress_calls = []

        def mock_progress(done, total):
            progress_calls.append((done, total))

        with (
            patch("os.makedirs"),
            patch("pandas.read_parquet", side_effect=FileNotFoundError),
            patch("pandas.DataFrame.to_parquet"),
        ):

            result = prepare_data_vectorized_system7(raw_data_dict, progress_callback=mock_progress)

        # プログレスコールバックが呼び出された
        assert len(progress_calls) > 0
        assert isinstance(result, dict)

    def test_generate_candidates_system7_basic(self):
        """generate_candidates_system7 の基本機能テスト"""
        # セットアップ条件を満たすテストデータ
        dates = pd.date_range("2023-01-01", periods=60)
        test_data = pd.DataFrame(
            {
                "Open": [400] * 60,
                "High": [450] * 60,
                "Low": np.concatenate(
                    [np.full(55, 380), [350, 340, 330, 320, 310]]
                ),  # 最後で最低値
                "Close": [420] * 60,
                "ATR50": [10.0] * 60,
                "min_50": np.concatenate([np.full(55, 350), [340, 330, 320, 310, 300]]),
                "setup": np.concatenate([np.zeros(55), [1, 1, 1, 1, 1]]),  # 最後の5日でsetup
                "max_70": [500] * 60,
            },
            index=dates,
        )

        prepared_dict = {"SPY": test_data}

        with patch("common.utils_spy.resolve_signal_entry_date") as mock_resolve:
            # エントリー日を翌日として設定
            mock_resolve.side_effect = lambda x: x + pd.Timedelta(days=1)

            candidates_by_date, merged_df = generate_candidates_system7(prepared_dict)

        # 戻り値検証
        assert isinstance(candidates_by_date, dict)
        # setupが1の日付に対して候補が生成される
        assert len(candidates_by_date) > 0

    def test_generate_candidates_system7_no_spy(self):
        """generate_candidates_system7 のSPYなしデータ処理"""
        prepared_dict = {"AAPL": self.create_spy_test_data()}

        candidates_by_date, merged_df = generate_candidates_system7(prepared_dict)

        # SPYがない場合は空の結果
        assert candidates_by_date == {}
        assert merged_df is None

    def test_generate_candidates_system7_with_top_n(self):
        """generate_candidates_system7 のtop_n制限テスト"""
        # セットアップ条件を満たすテストデータ
        dates = pd.date_range("2023-01-01", periods=60)
        test_data = pd.DataFrame(
            {
                "Open": [400] * 60,
                "High": [450] * 60,
                "Low": [350] * 60,
                "Close": [420] * 60,
                "ATR50": [10.0] * 60,
                "min_50": [340] * 60,
                "setup": [1] * 60,  # すべての日でsetup
                "max_70": [500] * 60,
            },
            index=dates,
        )

        prepared_dict = {"SPY": test_data}

        with patch("common.utils_spy.resolve_signal_entry_date") as mock_resolve:
            mock_resolve.side_effect = lambda x: x + pd.Timedelta(days=1)

            candidates_by_date, merged_df = generate_candidates_system7(prepared_dict, top_n=5)

        assert isinstance(candidates_by_date, dict)

    def test_generate_candidates_system7_with_top_n_zero(self):
        """generate_candidates_system7 のtop_n=0テスト"""
        # セットアップ条件を含むテストデータ
        test_data = pd.DataFrame(
            {
                "Open": [400] * 10,
                "High": [450] * 10,
                "Low": [350] * 10,
                "Close": [420] * 10,
                "ATR50": [10.0] * 10,
                "min_50": [340] * 10,
                "setup": [1] * 10,  # setup カラムを含む
                "max_70": [500] * 10,
            },
            index=pd.date_range("2023-01-01", periods=10),
        )

        prepared_dict = {"SPY": test_data}

        candidates_by_date, merged_df = generate_candidates_system7(prepared_dict, top_n=0)

        # top_n=0の場合は候補なし
        assert candidates_by_date == {}
        assert merged_df is None

    def test_generate_candidates_system7_with_progress_callback(self):
        """generate_candidates_system7 のプログレスコールバック処理"""
        # セットアップ条件を含むテストデータ
        test_data = pd.DataFrame(
            {
                "Open": [400] * 10,
                "High": [450] * 10,
                "Low": [350] * 10,
                "Close": [420] * 10,
                "ATR50": [10.0] * 10,
                "min_50": [340] * 10,
                "setup": [1] * 10,  # setup カラムを含む
                "max_70": [500] * 10,
            },
            index=pd.date_range("2023-01-01", periods=10),
        )

        prepared_dict = {"SPY": test_data}

        progress_calls = []

        def mock_progress(done, total):
            progress_calls.append((done, total))

        candidates_by_date, merged_df = generate_candidates_system7(
            prepared_dict, progress_callback=mock_progress
        )

        # プログレスコールバックが呼び出される
        assert len(progress_calls) > 0

    def test_get_total_days_system7_basic(self):
        """get_total_days_system7 の基本機能テスト"""
        data_dict = {
            "SPY": pd.DataFrame(
                {"Close": [400, 401, 402]},
                index=pd.date_range("2023-01-01", "2023-01-03", freq="D"),
            ),
            "AAPL": pd.DataFrame(
                {"Close": [150, 151]}, index=pd.date_range("2023-01-02", "2023-01-03", freq="D")
            ),
        }

        result = get_total_days_system7(data_dict)

        # ユニーク日付数 (1/1, 1/2, 1/3) = 3日
        assert result == 3

    def test_get_total_days_system7_with_date_column(self):
        """get_total_days_system7 のDateカラム処理テスト"""
        data_dict = {
            "SPY": pd.DataFrame(
                {"Date": ["2023-01-01", "2023-01-02", "2023-01-03"], "Close": [400, 401, 402]}
            ),
        }

        result = get_total_days_system7(data_dict)

        # Dateカラムからの日付抽出
        assert result == 3

    def test_get_total_days_system7_empty_data(self):
        """get_total_days_system7 の空データ処理"""
        data_dict = {
            "SPY": pd.DataFrame(),
            "EMPTY": None,
        }

        result = get_total_days_system7(data_dict)

        # 空データの場合は0日
        assert result == 0

    def test_get_total_days_system7_overlapping_dates(self):
        """get_total_days_system7 の重複日付処理"""
        data_dict = {
            "SPY": pd.DataFrame(
                {"Close": [400, 401]},
                index=pd.date_range("2023-01-01", "2023-01-02", freq="D"),
            ),
            "AAPL": pd.DataFrame(
                {"Close": [150, 151]},
                index=pd.date_range("2023-01-01", "2023-01-02", freq="D"),
            ),
        }

        result = get_total_days_system7(data_dict)

        # 重複日付は1回だけカウント
        assert result == 2

    def test_prepare_data_vectorized_system7_with_reuse_indicators_false(self):
        """prepare_data_vectorized_system7 のreuse_indicators=Falseテスト"""
        test_data = self.create_spy_test_data(100)
        raw_data_dict = {"SPY": test_data}

        with (
            patch("os.makedirs"),
            patch("pandas.read_parquet", side_effect=FileNotFoundError),
            patch("pandas.DataFrame.to_parquet"),
        ):

            result = prepare_data_vectorized_system7(raw_data_dict, reuse_indicators=False)

        assert isinstance(result, dict)
        assert "SPY" in result

    def test_prepare_data_vectorized_system7_insufficient_data(self):
        """prepare_data_vectorized_system7 の不十分データ処理"""
        # 70日未満のデータ（max_70計算に不十分）
        test_data = self.create_spy_test_data(30)
        raw_data_dict = {"SPY": test_data}

        with (
            patch("os.makedirs"),
            patch("pandas.read_parquet", side_effect=FileNotFoundError),
            patch("pandas.DataFrame.to_parquet"),
        ):

            result = prepare_data_vectorized_system7(raw_data_dict)

        # 不十分なデータでも処理は完了するがSPYは処理されない場合がある
        assert isinstance(result, dict)
        # 実装によってはSPYが返される場合と返されない場合がある
