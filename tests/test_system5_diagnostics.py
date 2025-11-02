"""System5 診断整合性テスト - STUpass/TRDlist逆転バグ検証用。

このテストは以下のバグを検出・防止します:
- STUpass < TRDlist という論理的に不可能な状態
- 重複カウントによる診断情報の不整合
- ランキング前後での候補数の逆転
"""

import pandas as pd

from core.system5 import generate_candidates_system5


class TestSystem5DiagnosticsConsistency:
    """System5の診断整合性テスト群。"""

    def test_stupass_never_less_than_trdlist(self):
        """基本不変条件: STUpass >= TRDlist が常に成立。"""
        # Given: 10件のセットアップ通過データ
        prepared_dict = {
            f"SYM{i:03d}": pd.DataFrame(
                {
                    "Close": [10.0],
                    "adx7": [40.0 + i * 5],  # 40, 45, 50, ...
                    "atr_pct": [0.03],
                    "atr10": [0.5],
                },
                index=[pd.Timestamp("2025-01-01")],
            )
            for i in range(10)
        }

        # When: top_n=20 で候補生成（候補数より大きい）
        result = generate_candidates_system5(
            prepared_dict,
            top_n=20,
            latest_only=True,
            include_diagnostics=True,
        )

        _, df_all, diagnostics = result

        # Then: setup件数 >= ランキング後件数
        assert diagnostics["setup_predicate_count"] >= diagnostics["ranked_top_n_count"]
        assert diagnostics["setup_predicate_count"] == 10
        assert diagnostics["ranked_top_n_count"] == 10

    def test_no_duplicate_ranking_unique_symbols(self):
        """重複ランキングが発生しないこと - ユニーク銘柄の検証。"""
        # Given: 5件のユニーク銘柄
        prepared_dict = {
            f"SYM{i:03d}": pd.DataFrame(
                {
                    "Close": [10.0],
                    "adx7": [40.0 + i * 5],
                    "atr_pct": [0.03],
                    "atr10": [0.5],
                },
                index=[pd.Timestamp("2025-01-01")],
            )
            for i in range(5)
        }

        # When: top_n=10 で候補生成
        result = generate_candidates_system5(
            prepared_dict,
            top_n=10,
            latest_only=True,
            include_diagnostics=True,
        )

        _, df_all, diagnostics = result

        # Then: 重複なし、setup件数 = ランキング件数
        assert len(df_all) == 5
        assert diagnostics["setup_predicate_count"] == 5
        assert diagnostics["ranked_top_n_count"] == 5
        assert len(df_all["symbol"].unique()) == 5  # 重複なし
        assert diagnostics.get("setup_unique_symbols", 0) == 5

    def test_setup_unique_symbols_tracking(self):
        """setup_unique_symbols が正しくトラッキングされること。"""
        # Given: 8件の銘柄
        prepared_dict = {
            f"SYM{i:03d}": pd.DataFrame(
                {
                    "Close": [10.0],
                    "adx7": [50.0],
                    "atr_pct": [0.03],
                    "atr10": [0.5],
                },
                index=[pd.Timestamp("2025-01-01")],
            )
            for i in range(8)
        }

        # When: 候補生成
        result = generate_candidates_system5(
            prepared_dict,
            top_n=10,
            latest_only=True,
            include_diagnostics=True,
        )

        _, _, diagnostics = result

        # Then: ユニーク銘柄数が正しく記録
        assert "setup_unique_symbols" in diagnostics
        assert diagnostics["setup_unique_symbols"] == 8
        assert diagnostics["setup_predicate_count"] == 8

    def test_top_n_requested_recorded(self):
        """top_n_requested が診断情報に記録されること。"""
        # Given: 5件の銘柄
        prepared_dict = {
            f"SYM{i:03d}": pd.DataFrame(
                {
                    "Close": [10.0],
                    "adx7": [45.0],
                    "atr_pct": [0.03],
                    "atr10": [0.5],
                },
                index=[pd.Timestamp("2025-01-01")],
            )
            for i in range(5)
        }

        # When: top_n=15 で候補生成
        result = generate_candidates_system5(
            prepared_dict,
            top_n=15,
            latest_only=True,
            include_diagnostics=True,
        )

        _, _, diagnostics = result

        # Then: top_n_requested が記録
        assert "top_n_requested" in diagnostics
        assert diagnostics["top_n_requested"] == 15

    def test_insufficient_candidates_handling(self):
        """候補不足時の正しい処理。"""
        # Given: 3件の銘柄（top_n=10未満）
        prepared_dict = {
            f"SYM{i:03d}": pd.DataFrame(
                {
                    "Close": [10.0],
                    "adx7": [40.0 + i * 5],
                    "atr_pct": [0.03],
                    "atr10": [0.5],
                },
                index=[pd.Timestamp("2025-01-01")],
            )
            for i in range(3)
        }

        # When: top_n=10 で候補生成
        result = generate_candidates_system5(
            prepared_dict,
            top_n=10,
            latest_only=True,
            include_diagnostics=True,
        )

        _, df_all, diagnostics = result

        # Then: 整合性保証
        assert diagnostics["setup_predicate_count"] == 3
        assert diagnostics["ranked_top_n_count"] == 3
        assert len(df_all) == 3

    def test_edge_case_zero_setup_pass(self):
        """エッジケース: セットアップ通過候補がゼロ。"""
        # Given: セットアップ条件を満たさないデータ
        prepared_dict = {
            f"SYM{i:03d}": pd.DataFrame(
                {
                    "Close": [3.0],  # < 5.0
                    "adx7": [30.0],  # < 35.0
                    "atr_pct": [0.01],  # < 0.025
                    "atr10": [0.5],
                },
                index=[pd.Timestamp("2025-01-01")],
            )
            for i in range(5)
        }

        # When: 候補生成
        result = generate_candidates_system5(
            prepared_dict,
            top_n=10,
            latest_only=True,
            include_diagnostics=True,
        )

        by_date, df_all, diagnostics = result

        # Then: 全て0件
        assert diagnostics["setup_predicate_count"] == 0
        assert diagnostics["ranked_top_n_count"] == 0
        assert len(by_date) == 0
        assert df_all is None or len(df_all) == 0

    def test_exact_top_n_match(self):
        """境界ケース: 候補数がちょうど top_n と一致。"""
        # Given: 12件の銘柄
        prepared_dict = {
            f"SYM{i:03d}": pd.DataFrame(
                {
                    "Close": [10.0],
                    "adx7": [40.0 + i * 2],
                    "atr_pct": [0.03],
                    "atr10": [0.5],
                },
                index=[pd.Timestamp("2025-01-01")],
            )
            for i in range(12)
        }

        # When: top_n=12 で候補生成
        result = generate_candidates_system5(
            prepared_dict,
            top_n=12,
            latest_only=True,
            include_diagnostics=True,
        )

        _, df_all, diagnostics = result

        # Then: 完全一致
        assert diagnostics["setup_predicate_count"] == 12
        assert diagnostics["ranked_top_n_count"] == 12
        assert len(df_all) == 12

    def test_multi_date_latest_only_behavior(self):
        """複数日付データでlatest_onlyが最終日のみを使うことを検証。"""
        # Given: 複数日付を持つ銘柄（latest_only=True なので最終日のみ）
        prepared_dict = {
            f"SYM{i:03d}": pd.DataFrame(
                {
                    "Close": [10.0, 11.0, 12.0],
                    "adx7": [40.0, 45.0, 50.0],
                    "atr_pct": [0.03, 0.03, 0.03],
                    "atr10": [0.5, 0.5, 0.5],
                },
                index=[
                    pd.Timestamp("2025-01-01"),
                    pd.Timestamp("2025-01-02"),
                    pd.Timestamp("2025-01-03"),
                ],
            )
            for i in range(5)
        }

        # When: latest_only=True で候補生成
        result = generate_candidates_system5(
            prepared_dict,
            top_n=10,
            latest_only=True,
            include_diagnostics=True,
        )

        _, df_all, _ = result

        # Then: 各銘柄は最終日のみ（重複なし）
        assert len(df_all) == 5
        assert len(df_all["symbol"].unique()) == 5
