# c:\Repos\quant_trading_system\tests\test_exceptions_comprehensive.py

"""
common/exceptions.py の包括的テストスイート
例外階層、デコレータ、並列ユーティリティの完全カバレッジ
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch

from common.exceptions import (
    TradingError,
    DataValidationError,
    TaskTimeoutError,
    handle_exceptions,
    run_with_timeout,
    map_with_timeout,
)


class TestTradingErrorHierarchy:
    """例外階層の完全テスト"""

    def test_trading_error_basic(self):
        """TradingError基本機能テスト"""
        msg = "取引エラーが発生しました"
        error = TradingError(msg)
        assert str(error) == msg
        assert isinstance(error, Exception)

    def test_trading_error_inheritance_chain(self):
        """例外継承チェーンの検証"""
        # DataValidationError
        data_error = DataValidationError("データ検証エラー")
        assert isinstance(data_error, TradingError)
        assert isinstance(data_error, Exception)

        # TaskTimeoutError
        timeout_error = TaskTimeoutError("タイムアウト")
        assert isinstance(timeout_error, TradingError)

    def test_specific_error_messages(self):
        """特定エラーメッセージの保持"""
        test_cases = [
            (DataValidationError, "価格データが見つかりません"),
            (TaskTimeoutError, "処理がタイムアウトしました"),
        ]

        for error_class, message in test_cases:
            error = error_class(message)
            assert str(error) == message
            assert isinstance(error, TradingError)


class TestHandleExceptionsDecorator:
    """handle_exceptions デコレータの包括テスト"""

    def test_normal_function_execution(self):
        """正常実行時の挙動"""

        @handle_exceptions()
        def normal_func(x, y):
            return x + y

        result = normal_func(3, 5)
        assert result == 8

    def test_exception_handling_with_default_value(self):
        """例外処理時のデフォルト値返却"""

        @handle_exceptions(default="エラー発生")
        def failing_func():
            raise ValueError("何らかのエラー")

        result = failing_func()
        assert result == "エラー発生"

    def test_exception_handling_without_default_value(self):
        """default未指定時のNone返却"""

        @handle_exceptions()
        def failing_func():
            raise RuntimeError("エラー")

        result = failing_func()
        assert result is None

    def test_reraise_flag_true(self):
        """reraise=True時の例外再発生"""

        @handle_exceptions(reraise=True)
        def trading_error_func():
            raise DataValidationError("データエラー")

        with pytest.raises(DataValidationError):
            trading_error_func()

    def test_reraise_flag_false_default(self):
        """reraise=False(デフォルト)時の例外吸収"""

        @handle_exceptions()
        def error_func():
            raise RuntimeError("エラー")

        result = error_func()
        assert result is None

    @patch("common.exceptions.logging.getLogger")
    def test_default_logger_usage(self, mock_get_logger):
        """デフォルトロガーの使用"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        @handle_exceptions()
        def error_func():
            raise ValueError("テストエラー")

        result = error_func()
        assert result is None

        # モジュール名でロガー取得
        mock_get_logger.assert_called_once_with("test_exceptions_comprehensive")

        # 例外ログ記録
        mock_logger.exception.assert_called_once_with(
            "Unhandled exception in %s: %s", "error_func", mock_logger.exception.call_args[0][2]
        )

    def test_custom_logger_usage(self):
        """カスタムロガーの使用"""
        custom_logger = Mock()

        @handle_exceptions(logger=custom_logger)
        def error_func():
            raise ValueError("カスタムログテスト")

        result = error_func()
        assert result is None

        # カスタムロガーが使用される
        custom_logger.exception.assert_called_once()

    def test_function_args_kwargs_preserved(self):
        """関数の引数・キーワード引数の保持"""

        @handle_exceptions()
        def complex_func(a, b, c=None, d=10):
            return f"a={a}, b={b}, c={c}, d={d}"

        result = complex_func("hello", 42, c="world", d=100)
        assert result == "a=hello, b=42, c=world, d=100"

    def test_decorator_preserves_function_metadata(self):
        """デコレータが関数のメタデータを保持"""

        @handle_exceptions()
        def documented_func():
            """これはテスト関数です"""
            return "test"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "これはテスト関数です"


class TestRunWithTimeout:
    """run_with_timeout ユーティリティの包括テスト"""

    def test_fast_function_completion(self):
        """高速完了関数のテスト"""

        def fast_func(x, y):
            return x * y

        result = run_with_timeout(fast_func, 1.0, 5, 6)
        assert result == 30

    def test_function_with_args_kwargs(self):
        """引数・キーワード引数の受け渡し"""

        def complex_func(a, b, c=None, multiplier=1):
            return (a + b) * multiplier if c is None else (a + b + c) * multiplier

        # 位置引数のみ
        result1 = run_with_timeout(complex_func, 1.0, 2, 3)
        assert result1 == 5

        # キーワード引数込み
        result2 = run_with_timeout(complex_func, 1.0, 2, 3, c=1, multiplier=2)
        assert result2 == 12

    def test_timeout_exception(self):
        """タイムアウト時の例外発生"""

        def slow_func():
            time.sleep(2.0)
            return "完了"

        with pytest.raises(TaskTimeoutError) as exc_info:
            run_with_timeout(slow_func, 0.1)

        assert "timeout after 0.1 seconds" in str(exc_info.value)
        assert "slow_func" in str(exc_info.value)

    def test_function_exception_propagation(self):
        """関数内例外の伝播"""

        def error_func():
            raise ValueError("関数内エラー")

        with pytest.raises(ValueError) as exc_info:
            run_with_timeout(error_func, 1.0)

        assert str(exc_info.value) == "関数内エラー"

    def test_function_returns_none(self):
        """None返却関数の処理"""

        def none_func():
            return None

        result = run_with_timeout(none_func, 1.0)
        assert result is None

    def test_empty_result_container(self):
        """空の結果コンテナのハンドリング"""

        def void_func():
            pass  # 何も返さない

        result = run_with_timeout(void_func, 1.0)
        assert result is None


class TestMapWithTimeout:
    """map_with_timeout 並列ユーティリティの包括テスト"""

    def test_simple_mapping(self):
        """単純なマッピング処理"""

        def square(x):
            return x**2

        items = [1, 2, 3, 4, 5]
        results, errors = map_with_timeout(square, items)

        assert results == [1, 4, 9, 16, 25]
        assert errors == []

    def test_mixed_success_and_errors(self):
        """成功とエラーの混在処理"""

        def conditional_func(x):
            if x % 2 == 0:
                raise ValueError(f"偶数エラー: {x}")
            return x * 10

        items = [1, 2, 3, 4, 5]
        results, errors = map_with_timeout(conditional_func, items)

        # 成功: 奇数のインデックス
        assert results[0] == 10  # 1 * 10
        assert results[2] == 30  # 3 * 10
        assert results[4] == 50  # 5 * 10

        # エラー: 偶数のインデックス
        assert results[1] is None  # エラーでNone
        assert results[3] is None  # エラーでNone

        # エラーリスト検証
        assert len(errors) == 2
        error_items = [item for item, exc in errors]
        assert 2 in error_items
        assert 4 in error_items

    def test_per_item_timeout(self):
        """アイテム毎タイムアウト設定"""

        def conditional_slow_func(x):
            if x == 3:
                time.sleep(0.2)  # 0.1秒でタイムアウト
            return x * 2

        items = [1, 2, 3, 4]
        results, errors = map_with_timeout(conditional_slow_func, items, per_item_timeout=0.1)

        # 1, 2, 4 は成功
        assert results[0] == 2
        assert results[1] == 4
        assert results[3] == 8

        # 3 はタイムアウト
        assert results[2] is None
        assert len(errors) == 1
        assert errors[0][0] == 3  # タイムアウトしたアイテム
        assert isinstance(errors[0][1], TaskTimeoutError)

    def test_max_workers_configuration(self):
        """ワーカー数設定の動作確認"""

        def worker_id_func(x):
            return threading.current_thread().ident

        items = list(range(10))
        results, errors = map_with_timeout(worker_id_func, items, max_workers=2)

        assert len(results) == 10
        assert errors == []

        # 最大2つのワーカーIDのみ使用されることを確認
        unique_worker_ids = set(results)
        assert len(unique_worker_ids) <= 2

    def test_progress_callback(self):
        """進捗コールバックの動作"""
        progress_calls = []

        def progress_tracker(done, total):
            progress_calls.append((done, total))

        def simple_func(x):
            return x

        items = [1, 2, 3, 4, 5]
        results, errors = map_with_timeout(simple_func, items, progress=progress_tracker)

        assert results == [1, 2, 3, 4, 5]
        assert errors == []

        # 進捗コールバックが呼ばれたことを確認
        assert len(progress_calls) == 5
        assert (5, 5) in progress_calls  # 最終コール

    def test_progress_callback_exception_handling(self):
        """進捗コールバック内例外の安全な処理"""

        def failing_progress(done, total):
            raise RuntimeError("進捗エラー")

        def simple_func(x):
            return x * 2

        items = [1, 2, 3]
        # 進捗コールバックがエラーしても処理は続行
        results, errors = map_with_timeout(simple_func, items, progress=failing_progress)

        assert results == [2, 4, 6]
        assert errors == []

    def test_return_exceptions_false(self):
        """return_exceptions=False時の例外再発生"""

        def error_func(x):
            if x == 2:
                raise ValueError("テストエラー")
            return x

        items = [1, 2, 3]

        with pytest.raises(ValueError):
            map_with_timeout(error_func, items, return_exceptions=False)

    def test_empty_iterable(self):
        """空の反復可能オブジェクト処理"""

        def any_func(x):
            return x

        results, errors = map_with_timeout(any_func, [])

        assert results == []
        assert errors == []

    def test_single_item_processing(self):
        """単一アイテム処理"""

        def increment(x):
            return x + 1

        results, errors = map_with_timeout(increment, [42])

        assert results == [43]
        assert errors == []


class TestIntegrationScenarios:
    """統合シナリオテスト"""

    def test_decorator_with_timeout_utility(self):
        """デコレータとタイムアウトユーティリティの組み合わせ"""

        @handle_exceptions(default="タイムアウト処理済み")
        def timeout_prone_func():
            def slow_operation():
                time.sleep(0.2)
                return "完了"

            return run_with_timeout(slow_operation, 0.1)

        result = timeout_prone_func()
        assert result == "タイムアウト処理済み"

    def test_nested_exception_handling(self):
        """ネストした例外処理シナリオ"""

        def inner_func(x):
            if x < 0:
                raise DataValidationError("負の値は処理できません")
            return x**0.5

        @handle_exceptions(default=-1)
        def outer_func(values):
            results, errors = map_with_timeout(inner_func, values)
            if errors:
                raise TaskTimeoutError("一部処理でエラーが発生しました")
            return sum(results)

        # 正常ケース
        result1 = outer_func([4, 9, 16])
        assert result1 == 9.0  # 2 + 3 + 4

        # エラーケース
        result2 = outer_func([4, -1, 16])  # 負の値でDataValidationError
        assert result2 == -1  # デコレータのdefault
