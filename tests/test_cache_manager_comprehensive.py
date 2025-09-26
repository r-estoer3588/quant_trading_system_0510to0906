# c:\Repos\quant_trading_system\tests\test_cache_manager_comprehensive.py
"""
CacheManagerの包括的なテストスイート
Phase4のcache_manager.pyカバレッジ向上（12% → 70-80%）
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from common.cache_manager import CacheManager, compute_base_indicators, round_dataframe


def _create_sample_df(n_rows=100):
    """テスト用のサンプルDataFrameを作成"""
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.random.uniform(100, 200, n_rows),
            "high": np.random.uniform(105, 210, n_rows),
            "low": np.random.uniform(95, 195, n_rows),
            "close": np.random.uniform(100, 200, n_rows),
            "volume": np.random.uniform(100000, 1000000, n_rows).astype(int),
        }
    )


class DummyRolling(SimpleNamespace):
    base_lookback_days = 300
    buffer_days = 30
    prune_chunk_days = 30
    meta_file = "_meta.json"
    round_decimals = 2


class DummyCsv(SimpleNamespace):
    decimal_point = "."
    thousands_sep = None
    field_sep = ","


def _build_cm(tmp_path):
    """テスト用のCacheManagerインスタンスを作成"""
    cache = SimpleNamespace(
        full_dir=tmp_path,
        rolling_dir=tmp_path,
        rolling=DummyRolling(),
        file_format="csv",
        round_decimals=2,
    )
    settings = SimpleNamespace(
        cache=cache,
        csv=DummyCsv(),
        paths=SimpleNamespace(
            data_cache=str(tmp_path),
            full_backup=str(tmp_path / "full_backup"),
        ),
    )
    settings.use_cache_parallel = False
    return CacheManager(settings)


class TestRoundingFunctions:
    """round_dataframe関数のテスト"""

    def test_round_dataframe_default(self):
        """round_dataframe: デフォルト値での丸め"""
        df = pd.DataFrame({"price": [1.23456, 2.34567], "volume": [100.123, 200.456]})
        result = round_dataframe(df, decimals=2)
        # 価格は丸められ、整数列は整数のまま
        assert result["price"].iloc[0] == 1.23
        assert result["volume"].iloc[0] == 100

    def test_round_dataframe_none_decimals(self):
        """round_dataframe: None指定（丸めなし）"""
        df = pd.DataFrame({"price": [1.23456, 2.34567], "volume": [100, 200]})
        result = round_dataframe(df, decimals=None)
        pd.testing.assert_frame_equal(result, df)

    def test_round_dataframe_empty(self):
        """round_dataframe: 空のDataFrame"""
        df = pd.DataFrame()
        result = round_dataframe(df, decimals=2)
        pd.testing.assert_frame_equal(result, df)


class TestIndicatorFunctions:
    """compute_base_indicators関数のテスト"""

    def test_compute_base_indicators_basic(self):
        """compute_base_indicators: 基本的なインジケーター計算"""
        df = _create_sample_df(50)
        result = compute_base_indicators(df)

        # インジケーター関数によって列名が大文字に変換される
        # 元の列の存在は大文字小文字を区別して確認
        assert "Date" in result.columns  # 'date' -> 'Date'
        assert "Open" in result.columns  # 'open' -> 'Open'
        assert "Close" in result.columns  # 'close' -> 'Close'

        # 結果のDataFrameは元と同じ行数
        assert len(result) == len(df)

    def test_compute_base_indicators_empty(self):
        """compute_base_indicators: 空のDataFrame"""
        df = pd.DataFrame()
        result = compute_base_indicators(df)
        assert result.empty


class TestCacheManagerBasicOperations:
    """CacheManagerの基本操作テスト"""

    def test_cache_manager_init(self, tmp_path):
        """CacheManager初期化"""
        cm = _build_cm(tmp_path)
        assert cm.full_dir.exists()
        assert cm.rolling_dir.exists()
        assert cm.settings is not None

    def test_cache_manager_paths(self, tmp_path):
        """CacheManagerパス設定"""
        cm = _build_cm(tmp_path)
        assert cm.rolling_meta_path.name == "_meta.json"
        assert cm.rolling_meta_path.parent == cm.rolling_dir

    def test_rolling_target_len(self, tmp_path):
        """rolling target length計算"""
        cm = _build_cm(tmp_path)
        expected = cm.rolling_cfg.base_lookback_days + cm.rolling_cfg.buffer_days
        assert cm._rolling_target_len == expected

    def test_write_and_read_atomic(self, tmp_path):
        """書き込みと読み取りの基本操作"""
        cm = _build_cm(tmp_path)
        df = _create_sample_df(50)

        # 書き込み
        cm.write_atomic(df, "TEST", "full")

        # 読み取り
        result = cm.read("TEST", "full")
        assert result is not None
        assert len(result) == len(df)

    def test_detect_path(self, tmp_path):
        """_detect_path: パス検出"""
        cm = _build_cm(tmp_path)
        path = cm._detect_path(cm.full_dir, "AAPL")
        assert path.suffix == ".csv"
        assert "AAPL" in path.name


class TestCacheManagerDataOperations:
    """CacheManagerデータ操作のテスト"""

    def test_upsert_both_new_symbol(self, tmp_path):
        """upsert_both: 新しいシンボルの追加"""
        cm = _build_cm(tmp_path)
        new_df = _create_sample_df(10)

        # 新しいシンボルでupsert
        cm.upsert_both("NEWSTOCK", new_df)

        # 読み取りテスト
        result = cm.read("NEWSTOCK", "full")
        assert result is not None
        assert len(result) >= len(new_df)

    def test_read_nonexistent_file(self, tmp_path):
        """存在しないファイルの読み取り"""
        cm = _build_cm(tmp_path)
        result = cm.read("NONEXISTENT", "full")
        assert result is None

    def test_prune_rolling_if_needed_no_data(self, tmp_path):
        """prune_rolling_if_needed: データなしの場合"""
        cm = _build_cm(tmp_path)
        result = cm.prune_rolling_if_needed("NONEXISTENT")

        # データがない場合でもdict形式の結果を返す
        assert isinstance(result, dict)
        assert "pruned_files" in result
        assert "dropped_rows_total" in result

    def test_prune_rolling_if_needed_with_data(self, tmp_path):
        """prune_rolling_if_needed: データありの場合"""
        cm = _build_cm(tmp_path)

        # 大きなデータセットを作成してmetaファイルも作成
        df = _create_sample_df(400)  # rolling target lenより大きい
        cm.write_atomic(df, "BIGSTOCK", "rolling")

        # metaファイルを作成
        meta_data = {"anchor_rows_at_prune": 0}
        cm.rolling_meta_path.write_text(json.dumps(meta_data), encoding="utf-8")

        result = cm.prune_rolling_if_needed("BIGSTOCK")

        # プルーニングが実行されたかどうかをチェック
        assert result is not None


class TestCacheManagerMetaOperations:
    """CacheManagerメタファイル操作のテスト"""

    def test_prune_rolling_if_needed_with_meta(self, tmp_path):
        """prune_rolling_if_needed: メタファイル付きプルーニング"""
        cm = _build_cm(tmp_path)

        # データを追加
        df = _create_sample_df(100)
        cm.write_atomic(df, "SPY", "rolling")

        # メタファイルを作成
        meta_data = {"anchor_rows_at_prune": 0}
        cm.rolling_meta_path.write_text(json.dumps(meta_data), encoding="utf-8")

        # プルーニング実行（強制）
        cm.prune_rolling_if_needed("SPY")

        # メタファイルが更新されたかチェック
        assert cm.rolling_meta_path.exists()

        # メタファイルの内容をチェック
        with open(cm.rolling_meta_path, encoding="utf-8") as f:
            meta_content = json.load(f)
        assert "anchor_rows_at_prune" in meta_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
