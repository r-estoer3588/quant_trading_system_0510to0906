"""
System7 キャッシュ増分更新ロジックのカバレッジ向上テスト

ターゲット: core/system7.py Lines 99-116
期待カバレッジ向上: +8% (18行/243行)

テスト戦略:
1. キャッシュ済みデータ生成 (cached.feather)
2. 新規データ追加 (new_rows)
3. 増分計算の検証 (Lines 99-116)
"""

import pandas as pd

from core.system7 import prepare_data_vectorized_system7


class TestSystem7CacheIncremental:
    """キャッシュ増分更新の動作確認"""

    def create_cached_spy_data(self, days: int = 100) -> pd.DataFrame:
        """キャッシュ済みSPYデータを生成 (指標計算済み)"""
        dates = pd.date_range("2024-01-01", periods=days, freq="D")
        prices = [100 + i * 0.5 for i in range(days)]
        lows = [p * 0.98 for p in prices]

        # 指標計算済み
        min_50_series = pd.Series(lows).rolling(window=50, min_periods=1).min()
        max_70_series = pd.Series(prices).rolling(window=70, min_periods=1).max()

        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.02 for p in prices],
                "Low": lows,
                "Close": [p * 1.01 for p in prices],
                "Volume": [1000000] * days,
                "atr50": [p * 0.02 for p in prices],
                "min_50": min_50_series.values,
                "max_70": max_70_series.values,
            },
            index=dates,
        )
        return df

    def create_new_spy_data(self, start_date: str, days: int = 10) -> pd.DataFrame:
        """新規SPYデータを生成 (指標なし)"""
        dates = pd.date_range(start_date, periods=days, freq="D")
        base_price = 150.0
        prices = [base_price + i * 0.5 for i in range(days)]
        lows = [p * 0.98 for p in prices]

        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.02 for p in prices],
                "Low": lows,
                "Close": [p * 1.01 for p in prices],
                "Volume": [1000000] * days,
                # atr50, min_50, max_70 は未計算
            },
            index=dates,
        )
        return df

    def test_cache_incremental_update_basic(self, tmp_path):
        """キャッシュ増分更新の基本動作 (Lines 99-116)"""
        # キャッシュディレクトリ作成
        cache_dir = tmp_path / "indicators_system7_cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "SPY.feather"

        # 既存キャッシュ作成 (100日分)
        cached_df = self.create_cached_spy_data(days=100)
        cached_df.reset_index().to_feather(cache_path)

        # 新規データ追加 (110日分 = 既存100日 + 新規10日)
        full_df = pd.concat(
            [
                cached_df,
                self.create_new_spy_data(start_date="2024-04-11", days=10),  # 100日後
            ]
        )

        # 増分更新実行 (dict形式で渡す)
        data_dict = {"SPY": full_df}
        result_dict = prepare_data_vectorized_system7(data_dict, use_cache=True, cache_path=str(cache_path))

        # 検証
        assert result_dict is not None, "Result should not be None"
        assert "SPY" in result_dict, "SPY should be in result"
        result_df = result_dict["SPY"]

        assert len(result_df) == 110, f"Expected 110 rows, got {len(result_df)}"
        assert "atr50" in result_df.columns, "atr50 should be present"
        assert "min_50" in result_df.columns, "min_50 should be present"
        assert "max_70" in result_df.columns, "max_70 should be present"

        # 新規行に指標が計算されている (Lines 105-110)
        new_rows = result_df.iloc[100:]
        assert not new_rows["atr50"].isna().all(), "New rows should have atr50"
        assert not new_rows["min_50"].isna().all(), "New rows should have min_50"
        assert not new_rows["max_70"].isna().all(), "New rows should have max_70"

    def test_cache_incremental_no_new_rows(self, tmp_path):
        """新規データなしの場合 (Line 101-102)"""
        cache_dir = tmp_path / "indicators_system7_cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "SPY.feather"

        # 既存キャッシュ作成
        cached_df = self.create_cached_spy_data(days=100)
        cached_df.reset_index().to_feather(cache_path)

        # 同じデータで呼び出し (new_rows.empty == True)
        result_df = prepare_data_vectorized_system7(cached_df, use_cache=True, cache_path=str(cache_path))

        # キャッシュそのまま返却 (Line 102)
        assert result_df is not None
        assert len(result_df) == 100, "Should return cached data as-is"

        # データ内容一致確認
        pd.testing.assert_frame_equal(
            result_df.reset_index(drop=True),
            cached_df.reset_index(drop=True),
            check_dtype=False,
        )

    def test_cache_incremental_max70_priority(self, tmp_path):
        """max_70 の優先保持ロジック (Lines 109-111)"""
        cache_dir = tmp_path / "indicators_system7_cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "SPY.feather"

        # 既存キャッシュ作成
        cached_df = self.create_cached_spy_data(days=100)
        original_max70 = cached_df["max_70"].copy()
        cached_df.reset_index().to_feather(cache_path)

        # 新規データ追加
        full_df = pd.concat([cached_df, self.create_new_spy_data(start_date="2024-04-11", days=10)])

        # 増分更新実行
        result_df = prepare_data_vectorized_system7(full_df, use_cache=True, cache_path=str(cache_path))

        # cached 部分の max_70 が保持されている (Line 111)
        cached_indices = cached_df.index
        for idx in cached_indices:
            if idx in result_df.index:
                expected_val = original_max70.loc[idx]
                actual_val = result_df.loc[idx, "max_70"]
                assert abs(expected_val - actual_val) < 0.01, f"max_70 at {idx} should be preserved"

    def test_cache_incremental_feather_save(self, tmp_path):
        """増分更新後のFeather保存 (Lines 112-114)"""
        cache_dir = tmp_path / "indicators_system7_cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "SPY.feather"

        # 既存キャッシュ作成
        cached_df = self.create_cached_spy_data(days=100)
        cached_df.reset_index().to_feather(cache_path)

        # 新規データ追加
        full_df = pd.concat([cached_df, self.create_new_spy_data(start_date="2024-04-11", days=10)])

        # 増分更新実行
        _ = prepare_data_vectorized_system7(full_df, use_cache=True, cache_path=str(cache_path))

        # Featherファイルが更新されている (Line 113)
        assert cache_path.exists(), "Cache file should exist"

        # 更新されたキャッシュを読み込み
        reloaded_df = pd.read_feather(cache_path)
        assert len(reloaded_df) == 110, "Saved cache should have 110 rows"

    def test_cache_incremental_context_window(self, tmp_path):
        """70日コンテキストウィンドウ (Lines 105-107)"""
        cache_dir = tmp_path / "indicators_system7_cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "SPY.feather"

        # 既存キャッシュ作成 (200日分で十分なコンテキスト確保)
        cached_df = self.create_cached_spy_data(days=200)
        last_cached_date = cached_df.index.max()
        cached_df.reset_index().to_feather(cache_path)

        # 新規データ追加 (10日分)
        full_df = pd.concat(
            [
                cached_df,
                self.create_new_spy_data(
                    start_date=(last_cached_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    days=10,
                ),
            ]
        )

        # 増分更新実行
        result_df = prepare_data_vectorized_system7(full_df, use_cache=True, cache_path=str(cache_path))

        # 70日コンテキストで再計算されている (Lines 105-107)
        # context_start = last_date - 70日
        context_start = last_cached_date - pd.Timedelta(days=70)
        assert result_df is not None
        assert len(result_df) == 210, "Should have 210 rows total"

        # コンテキスト範囲内のデータも含まれている
        context_rows = result_df[result_df.index >= context_start]
        assert len(context_rows) >= 80, "Should have at least 80 rows in context"

    def test_cache_incremental_use_cache_false(self, tmp_path):
        """use_cache=False時は増分更新スキップ"""
        cache_dir = tmp_path / "indicators_system7_cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "SPY.feather"

        # 既存キャッシュ作成
        cached_df = self.create_cached_spy_data(days=100)
        cached_df.reset_index().to_feather(cache_path)

        # 新規データ追加
        full_df = pd.concat([cached_df, self.create_new_spy_data(start_date="2024-04-11", days=10)])

        # use_cache=False で呼び出し
        result_df = prepare_data_vectorized_system7(full_df, use_cache=False, cache_path=str(cache_path))

        # 全データが再計算される (Lines 99-116をスキップ)
        assert result_df is not None
        assert len(result_df) == 110
        # キャッシュは使われていない (全行計算)
        assert not result_df["atr50"].isna().any()

    def test_cache_incremental_cache_not_exists(self, tmp_path):
        """キャッシュファイル不存在時"""
        cache_dir = tmp_path / "indicators_system7_cache"
        cache_dir.mkdir()
        cache_path = cache_dir / "SPY_NOTEXIST.feather"

        # データ生成
        df = self.create_cached_spy_data(days=100)

        # キャッシュなしで呼び出し
        result_df = prepare_data_vectorized_system7(df, use_cache=True, cache_path=str(cache_path))

        # 全データが新規計算される (増分更新パスはスキップ)
        assert result_df is not None
        assert len(result_df) == 100
        assert "atr50" in result_df.columns
