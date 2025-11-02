"""System3 診断整合性テスト - STUpass/TRDlist不一致バグ検証用。

このテストは以下のバグを検出・防止します:
- STUpass (setup_predicate_count) とTRDlist (ranked_top_n_count) の不一致
- Top-off処理の不具合による候補数減少
- 診断カウンタの計算タイミング不整合
"""

import pandas as pd

from core.system3 import generate_candidates_system3


def _create_system3_test_data(
    n_symbols: int,
    base_drop3d: float = 0.15,
    date: str = "2025-01-01",
    close: float = 10.0,
    sma150: float = 8.0,
) -> dict[str, pd.DataFrame]:
    """System3テスト用のデータ生成ヘルパー。

    Args:
        n_symbols: 生成する銘柄数
        base_drop3d: 基準となるdrop3d値（i番目の銘柄は base_drop3d + i*0.01）
        date: データの日付
        close: 終値
        sma150: 150日移動平均（Close > sma150 を満たす必要あり）

    Returns:
        prepared_dict: System3が期待するフォーマットのデータ辞書
    """
    prepared_dict = {}
    for i in range(n_symbols):
        df = pd.DataFrame(
            {
                "Close": [close],
                "dollarvolume20": [30_000_000],
                "atr_ratio": [0.06],
                "drop3d": [base_drop3d + i * 0.01],
                "atr10": [0.5],
                "sma150": [sma150],
            },
            index=[pd.Timestamp(date)],
        )
        # System3 ロジック再現: filter と setup を計算
        df["filter"] = (
            (df["Close"] >= 5.0)
            & (df["dollarvolume20"] > 25_000_000)
            & (df["atr_ratio"] >= 0.05)
        )
        df["setup"] = df["filter"] & (df["drop3d"] >= 0.125)
        prepared_dict[f"SYM{i:03d}"] = df
    return prepared_dict


class TestSystem3DiagnosticsConsistency:
    """System3の診断整合性テスト群。"""

    def test_stupass_trdlist_consistency_normal_case(self):
        """通常ケース: STUpass >= TRDlist が保証されること。"""
        # Given: 20件のセットアップ通過データ（top_n以上）
        # 注: System3は filter & drop3d>=0.125 & Close>sma150 を評価
        #     テストデータには sma150 を含める必要がある
        prepared_dict = {}
        for i in range(20):
            df = pd.DataFrame(
                {
                    "Close": [10.0],
                    "dollarvolume20": [30_000_000],
                    "atr_ratio": [0.06],
                    "drop3d": [0.15 + i * 0.01],  # 0.15, 0.16, ...
                    "atr10": [0.5],
                    "sma150": [8.0],  # Close > sma150 を満たすため
                },
                index=[pd.Timestamp("2025-01-01")],
            )
            # System3 ロジック再現: filter と setup を計算
            df["filter"] = (
                (df["Close"] >= 5.0)
                & (df["dollarvolume20"] > 25_000_000)
                & (df["atr_ratio"] >= 0.05)
            )
            df["setup"] = df["filter"] & (df["drop3d"] >= 0.125)
            prepared_dict[f"SYM{i:03d}"] = df

        # When: top_n=10 で候補生成
        result = generate_candidates_system3(
            prepared_dict,
            top_n=10,
            latest_only=True,
            include_diagnostics=True,
        )

        _, df_all, diagnostics = result  # type: ignore[misc]

        # Then: 基本不変条件
        assert diagnostics["setup_predicate_count"] >= diagnostics["ranked_top_n_count"]
        assert diagnostics["ranked_top_n_count"] == 10  # top_n通り
        assert len(df_all) == 10  # type: ignore[arg-type]

    def test_stupass_trdlist_consistency_insufficient_candidates(self):
        """不足ケース: 候補が top_n より少ない場合。"""
        # Given: 5件のセットアップ通過データ（top_n=10未満）
        prepared_dict = _create_system3_test_data(n_symbols=5)

        # When: top_n=10 で候補生成
        result = generate_candidates_system3(
            prepared_dict,
            top_n=10,
            latest_only=True,
            include_diagnostics=True,
        )

        _, df_all, diagnostics = result  # type: ignore[misc]

        # Then: 候補不足でも整合性保証
        assert diagnostics["setup_predicate_count"] == 5
        assert diagnostics["ranked_top_n_count"] == 5  # 5件しかない
        assert diagnostics["ranked_top_n_count"] <= diagnostics["setup_predicate_count"]
        assert len(df_all) == 5  # type: ignore[arg-type]

    def test_stupass_equals_trdlist_exact_match(self):
        """境界ケース: 候補数がちょうど top_n と一致。"""
        # Given: 10件のセットアップ通過データ（top_n=10）
        prepared_dict = _create_system3_test_data(n_symbols=10)

        # When: top_n=10 で候補生成
        result = generate_candidates_system3(
            prepared_dict,
            top_n=10,
            latest_only=True,
            include_diagnostics=True,
        )

        _, df_all, diagnostics = result  # type: ignore[misc]

        # Then: 完全一致
        assert diagnostics["setup_predicate_count"] == 10
        assert diagnostics["ranked_top_n_count"] == 10
        assert len(df_all) == 10  # type: ignore[arg-type]

    def test_ranking_breakdown_populated(self):
        """診断内訳情報が正しく記録されること。"""
        # Given: 15件のセットアップ通過データ
        prepared_dict = _create_system3_test_data(n_symbols=15)

        # When: top_n=10 で候補生成
        result = generate_candidates_system3(
            prepared_dict,
            top_n=10,
            latest_only=True,
            include_diagnostics=True,
        )

        _, _, diagnostics = result  # type: ignore[misc]

        # Then: ranking_breakdown が存在し、内訳が正しい
        assert "ranking_breakdown" in diagnostics
        breakdown = diagnostics["ranking_breakdown"]

        assert "original_filtered" in breakdown
        assert "top_cut_before_topoff" in breakdown
        assert "extras_added" in breakdown
        assert "final_count" in breakdown

        # final_count = top_cut + extras
        # final_count = top_cut + extras
        assert breakdown["final_count"] == (
            breakdown["top_cut_before_topoff"] + breakdown["extras_added"]
        )

    def test_no_duplicate_symbols_in_trdlist(self):
        """TRDlistに重複シンボルが含まれないこと。"""
        # Given: 同一銘柄が複数日付で出現するケース（複数行のデータ）
        prepared_dict = {}
        for i in range(10):
            df = pd.DataFrame(
                {
                    "Close": [10.0, 11.0],
                    "dollarvolume20": [30_000_000, 30_000_000],
                    "atr_ratio": [0.06, 0.06],
                    "drop3d": [0.15 + i * 0.01, 0.16 + i * 0.01],
                    "atr10": [0.5, 0.5],
                    "sma150": [8.0, 8.5],  # Close > sma150 を満たす
                },
                index=[pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
            )
            df["filter"] = (
                (df["Close"] >= 5.0)
                & (df["dollarvolume20"] > 25_000_000)
                & (df["atr_ratio"] >= 0.05)
            )
            df["setup"] = df["filter"] & (df["drop3d"] >= 0.125)
            prepared_dict[f"SYM{i:03d}"] = df

        # When: latest_only=True で候補生成
        result = generate_candidates_system3(
            prepared_dict,
            top_n=10,
            latest_only=True,
            include_diagnostics=True,
        )

        _, df_all, _ = result  # type: ignore[misc]

        # Then: 各銘柄は1回のみ出現
        assert df_all is not None  # type: ignore[misc]
        assert len(df_all) == len(df_all["symbol"].unique())  # type: ignore[arg-type]

    def test_edge_case_zero_candidates(self):
        """エッジケース: セットアップ通過候補がゼロ。"""
        # Given: セットアップ条件を満たさないデータ
        prepared_dict = {
            f"SYM{i:03d}": pd.DataFrame(
                {
                    "Close": [3.0],  # < 5.0
                    "dollarvolume20": [10_000_000],  # < 25M
                    "atr_ratio": [0.03],  # < 0.05
                    "drop3d": [0.05],  # < 0.125
                    "atr10": [0.5],
                },
                index=[pd.Timestamp("2025-01-01")],
            )
            for i in range(5)
        }

        # When: 候補生成
        result = generate_candidates_system3(
            prepared_dict,
            top_n=10,
            latest_only=True,
            include_diagnostics=True,
        )

        _, df_all, diagnostics = result  # type: ignore[misc]

        # Then: 全て0件
        assert diagnostics["setup_predicate_count"] == 0
        assert diagnostics["ranked_top_n_count"] == 0
        assert df_all is None or len(df_all) == 0  # type: ignore[arg-type]
        assert df_all is None or len(df_all) == 0
