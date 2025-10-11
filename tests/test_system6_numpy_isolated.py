"""System6 NumPy-dependent tests in isolation

このファイルは、NumPy モジュールリロード問題の影響を受けやすいテストを
メインテストスイートから分離して実行するためのものです。

別ファイルで実行することで、NumPy の状態が他のテストから影響を受けず、
クリーンな環境でテストできる可能性があります。

実行方法:
    pytest tests/test_system6_numpy_isolated.py -v

注意:
    このファイルは test_system6_enhanced.py の一部テストを複製しています。
    NumPy 問題が解決したら、元のファイルに統合することを推奨します。
"""

import pandas as pd

from common.testing import set_test_determinism


class TestSystem6DateModeProcessingIsolated:
    """日付モード処理のテスト (lines 588-601)

    generate_candidates_system6関数のlatest_only=True時の
    日付モード処理（ソート、ランク付け、top_n制限）を検証します。

    このクラスはtest_system6_enhanced.pyから分離されたものです。
    """

    def test_date_mode_sorting_by_return_6d_descending(self):
        """return_6dによる降順ソートを検証 (lines 588-591)"""
        from core.system6 import generate_candidates_system6

        set_test_determinism()

        # 3銘柄で異なるreturn_6d値を設定
        # System6 setup条件: return_6d > 0.20 AND UpTwoDays == True
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        data_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [150.0],
                    "return_6d": [0.30],  # > 0.20
                    "atr10": [2.0],
                    "dollarvolume50": [100_000_000],
                    "hv50": [20.0],
                    "UpTwoDays": [1],  # True
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            ),
            "GOOGL": pd.DataFrame(
                {
                    "Close": [2800.0],
                    "return_6d": [0.25],  # > 0.20
                    "atr10": [30.0],
                    "dollarvolume50": [150_000_000],
                    "hv50": [18.0],
                    "UpTwoDays": [1],  # True
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            ),
            "MSFT": pd.DataFrame(
                {
                    "Close": [300.0],
                    "return_6d": [0.22],  # > 0.20
                    "atr10": [3.5],
                    "dollarvolume50": [120_000_000],
                    "hv50": [22.0],
                    "UpTwoDays": [1],  # True
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

        # return_6d降順でソートされているはず: AAPL(0.30) > GOOGL(0.25) > MSFT(0.22)
        assert symbols == ["AAPL", "GOOGL", "MSFT"]

    def test_date_mode_rank_assignment(self):
        """ランク付けの正確性を検証 (lines 592-593)"""
        from core.system6 import generate_candidates_system6

        set_test_determinism()

        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        data_dict = {
            "SYM1": pd.DataFrame(
                {
                    "Close": [100.0],
                    "return_6d": [0.30],  # > 0.20
                    "atr10": [1.5],
                    "dollarvolume50": [90_000_000],
                    "hv50": [15.0],
                    "UpTwoDays": [1],  # True
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            ),
            "SYM2": pd.DataFrame(
                {
                    "Close": [200.0],
                    "return_6d": [0.25],  # > 0.20
                    "atr10": [2.5],
                    "dollarvolume50": [110_000_000],
                    "hv50": [18.0],
                    "UpTwoDays": [1],  # True
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            ),
        }

        candidates, _ = generate_candidates_system6(
            prepared_dict=data_dict,
            top_n=5,
            latest_only=True,
        )

        assert len(candidates) >= 1
        last_date = list(candidates.keys())[0]

        # ランク1がreturn_6d最大の銘柄に割り当てられているか
        sym1_rank = candidates[last_date]["SYM1"]["rank"]
        sym2_rank = candidates[last_date]["SYM2"]["rank"]

        assert sym1_rank == 1  # return_6d=0.30が最大
        assert sym2_rank == 2  # return_6d=0.25が2位

    def test_date_mode_top_n_limit(self):
        """top_n制限が正しく機能するか検証 (lines 594-601)"""
        from core.system6 import generate_candidates_system6

        set_test_determinism()

        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        data_dict = {
            f"SYM{i}": pd.DataFrame(
                {
                    "Close": [100.0 + i * 10],
                    "return_6d": [0.40 - i * 0.05],  # 全て > 0.20
                    "atr10": [2.0],
                    "dollarvolume50": [100_000_000],
                    "hv50": [20.0],
                    "UpTwoDays": [1],  # True
                    "filter": [True],
                    "setup": [True],
                },
                index=dates,
            )
            for i in range(5)
        }

        # top_n=2に制限
        candidates, _ = generate_candidates_system6(
            prepared_dict=data_dict,
            top_n=2,
            latest_only=True,
        )

        assert len(candidates) >= 1
        last_date = list(candidates.keys())[0]

        # 2銘柄のみ返されるべき
        assert len(candidates[last_date]) == 2

        # return_6dが最大の2銘柄(SYM0, SYM1)が返されるべき
        symbols = list(candidates[last_date].keys())
        assert "SYM0" in symbols
        assert "SYM1" in symbols


class TestSystem6RankingAndFilteringIsolated:
    """ランキングとフィルタリングのテスト (lines 343-420)

    このクラスもtest_system6_enhanced.pyから分離されたものです。
    """

    def test_ranking_by_return_6d_descending(self):
        """return_6dによる降順ランキングを検証 (lines 343-370)"""
        from core.system6 import generate_candidates_system6

        set_test_determinism()

        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        data_dict = {
            "HIGH": pd.DataFrame(
                {
                    "Close": [100.0, 105.0, 110.0],
                    "return_6d": [0.25, 0.26, 0.27],  # > 0.20
                    "atr10": [2.0, 2.0, 2.0],
                    "dollarvolume50": [100_000_000] * 3,
                    "hv50": [20.0] * 3,
                    "UpTwoDays": [1, 1, 1],  # True
                    "filter": [True, True, True],
                    "setup": [True, True, True],
                },
                index=dates,
            ),
            "LOW": pd.DataFrame(
                {
                    "Close": [200.0, 195.0, 190.0],
                    "return_6d": [0.21, 0.22, 0.23],  # > 0.20 だがHIGHより小さい
                    "atr10": [3.0, 3.0, 3.0],
                    "dollarvolume50": [150_000_000] * 3,
                    "hv50": [18.0] * 3,
                    "UpTwoDays": [1, 1, 1],  # True
                    "filter": [True, True, True],
                    "setup": [True, True, True],
                },
                index=dates,
            ),
        }

        candidates, _ = generate_candidates_system6(
            prepared_dict=data_dict,
            top_n=5,
            latest_only=True,
        )

        assert len(candidates) >= 1
        last_date = list(candidates.keys())[0]

        # HIGHがLOWより高いランクを持つべき
        high_rank = candidates[last_date]["HIGH"]["rank"]
        low_rank = candidates[last_date]["LOW"]["rank"]

        assert high_rank < low_rank  # ランク1が最高


class TestSystem6NormalizationLogicIsolated:
    """正規化ロジックのテスト (lines 633-652)

    このクラスもtest_system6_enhanced.pyから分離されたものです。
    """

    def test_a_normalization_basic_structure(self):
        """正規化が基本的な構造を持つことを検証 (lines 633-643)"""
        from core.system6 import generate_candidates_system6

        set_test_determinism()

        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        data_dict = {
            "AAPL": pd.DataFrame(
                {
                    "Close": [150.0, 152.0, 155.0],
                    "return_6d": [0.22, 0.24, 0.26],  # > 0.20
                    "atr10": [2.0, 2.0, 2.0],
                    "dollarvolume50": [100_000_000] * 3,
                    "hv50": [20.0] * 3,
                    "UpTwoDays": [1, 1, 1],  # True
                    "filter": [True, True, True],
                    "setup": [True, True, True],
                },
                index=dates,
            ),
        }

        candidates, _ = generate_candidates_system6(
            prepared_dict=data_dict,
            top_n=5,
            latest_only=True,
        )

        assert len(candidates) >= 1
        last_date = list(candidates.keys())[0]
        assert "AAPL" in candidates[last_date]

        cand = candidates[last_date]["AAPL"]

        # 正規化された候補が必須フィールドを持つことを検証
        # latest_only=Trueでは、正規化時にシンボル名はキーとして使われるため、
        # 辞書のペイロード自体には'symbol'キーは含まれない
        assert "rank" in cand
        assert "entry_price" in cand
        assert "atr10" in cand  # ATRは'atr10'として格納される
        assert "rank_total" in cand
