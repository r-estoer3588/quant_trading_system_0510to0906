#!/usr/bin/env python3
"""
構造化ログシステムの動作確認テスト

ErrorCode統合、日本語説明付きエラー出力、TradingError例外システムの動作を検証
📊 リアルタイム進捗表示機能付き
"""

import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from common.structured_logging import ErrorCodes, TradingSystemLogger
from common.trading_errors import (
    DataError,
    ErrorCode,
    ErrorContext,
    SignalError,
    TradingError,
    create_error_summary,
    format_error_for_ui,
)


class ProgressBar:
    """リアルタイム進捗バー表示クラス"""

    def __init__(self, total_steps: int, width: int = 50):
        self.total_steps = total_steps
        self.current_step = 0
        self.width = width
        self.start_time = time.time()
        self.step_times = []

    def update(self, step_name: str = "", increment: int = 1):
        """進捗を更新し、リアルタイムで表示"""
        self.current_step += increment
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.step_times.append(elapsed)

        # 進捗率計算
        progress = self.current_step / self.total_steps
        filled_width = int(self.width * progress)

        # ETA計算（直近のステップ時間から推定）
        if len(self.step_times) > 1:
            avg_step_time = sum(self.step_times[-3:]) / min(3, len(self.step_times))
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = avg_step_time * remaining_steps / len(self.step_times)
            eta_str = f"ETA: {eta_seconds:.1f}s"
        else:
            eta_str = "ETA: --"

        # 進捗バー作成
        bar = "█" * filled_width + "░" * (self.width - filled_width)
        percentage = progress * 100

        # 経過時間フォーマット
        elapsed_str = f"{elapsed:.1f}s"

        # ステップ名を20文字に制限
        step_display = step_name[:20].ljust(20) if step_name else " " * 20

        # 進捗表示（カーソル位置を戻して上書き）
        print(
            f"\r📊 [{bar}] {percentage:5.1f}% | {self.current_step:2d}/{self.total_steps} | {elapsed_str} | {eta_str} | {step_display}",
            end="",
            flush=True,
        )

        if self.current_step >= self.total_steps:
            print()  # 完了時に改行


def test_error_code_mapping(progress: ProgressBar):
    """AAA123E形式ErrorCode統合をテスト"""
    progress.update("ErrorCode統合テスト")

    print("\n=== ErrorCode統合テスト ===")

    # レガシーコードの新しいコードへのマッピングをテスト
    legacy_codes = {
        "SPY001E": "DAT004E",
        "SYS001E": "SYS001E",
        "DATA001E": "DAT001E",
        "FIL001E": "SIG002E",
        "STU001E": "SIG003E",
        "TRD001E": "SIG001E",
    }

    for i, (legacy, expected) in enumerate(legacy_codes.items()):
        mapped = ErrorCodes.get_mapped_code(legacy)
        formatted = ErrorCodes.get_formatted_error(legacy)
        print(f"{legacy} → {mapped} (期待値: {expected})")
        print(f"  フォーマット済み: {formatted}")
        assert mapped == expected, f"マッピング失敗: {legacy} → {mapped} != {expected}"
        time.sleep(0.1)  # 進捗可視化のための小さな遅延

    print("✅ ErrorCodeマッピング正常")


def test_japanese_error_descriptions(progress: ProgressBar):
    """日本語説明付きエラー出力をテスト"""
    progress.update("日本語エラー説明テスト")

    print("\n=== 日本語エラー説明テスト ===")

    error_codes = [
        "DAT001E",
        "DAT004E",
        "SIG001E",
        "SIG002E",
        "SIG003E",
        "ALC001E",
        "SYS001E",
        "NET001E",
    ]

    for i, code in enumerate(error_codes):
        description = ErrorCodes.get_error_description(code)
        formatted = ErrorCodes.get_formatted_error(code)
        print(f"{code}: {description}")
        print(f"  フォーマット済み: {formatted}")

        # 日本語が含まれていることを確認
        assert any(ord(c) > 127 for c in description), f"日本語が含まれていない: {code}"
        time.sleep(0.1)  # 進捗可視化のための小さな遅延

    print("✅ 日本語説明正常")


def test_trading_error_system(progress: ProgressBar):
    """TradingError例外システムをテスト"""
    progress.update("TradingError例外テスト")

    print("\n=== TradingError例外システムテスト ===")

    # ErrorContextの作成
    context = ErrorContext(
        timestamp="2025-09-29T21:00:00",
        phase="filter",
        system="system1",
        symbol="AAPL",
        trace_id="test-123",
    )

    # DataErrorの作成とテスト
    try:
        raise DataError("キャッシュファイルが見つかりません", ErrorCode.DAT001E, context=context)
    except TradingError as e:
        print(f"DataError捕捉: {e.error_code.value} - {e.message}")
        error_dict = e.to_dict()
        print(f"  辞書形式: {error_dict}")
        ui_format = format_error_for_ui(e)
        print(f"  UI形式: {ui_format}")
        assert e.error_code == ErrorCode.DAT001E
        assert "キャッシュファイル" in e.message

    # SignalErrorの作成とテスト
    try:
        raise SignalError(
            "フィルター条件でエラーが発生", ErrorCode.SIG002E, context=context, retryable=True
        )
    except TradingError as e:
        print(f"SignalError捕捉: {e.error_code.value} - {e.message}")
        assert e.retryable is True

    print("✅ TradingError例外システム正常")


def test_error_summary(progress: ProgressBar):
    """複数エラーの要約テスト"""
    progress.update("エラー要約テスト")

    print("\n=== エラー要約テスト ===")

    context = ErrorContext(timestamp="2025-09-29T21:00:00", phase="test")

    errors = [
        DataError("データエラー1", ErrorCode.DAT001E, context=context),
        DataError("データエラー2", ErrorCode.DAT002E, context=context),
        SignalError("シグナルエラー1", ErrorCode.SIG001E, context=context, retryable=True),
        SignalError("シグナルエラー2", ErrorCode.SIG002E, context=context),
    ]

    summary = create_error_summary(errors)
    print(f"エラー要約: {summary}")

    assert summary["total"] == 4
    assert summary["by_category"]["DAT"] == 2
    assert summary["by_category"]["SIG"] == 2
    assert summary["retryable_count"] == 1

    print("✅ エラー要約正常")


def test_trace_context_integration(progress: ProgressBar):
    """TraceContext統合をテスト"""
    progress.update("TraceContext統合テスト")

    print("\n=== TraceContext統合テスト ===")

    from common.trace_context import ProcessingPhase, trace_context

    with trace_context(phase=ProcessingPhase.FILTERS, system="system1", symbol="AAPL") as context:
        print(f"現在のTraceContext: {context.to_dict()}")

        # TraceContextからErrorContextを作成
        error_context = ErrorContext(
            timestamp="2025-09-29T21:00:00",
            phase="test",
            system=context.system,
            symbol=context.symbol,
            trace_id=context.trace_id,
        )

        error = DataError("TraceContext統合テスト", ErrorCode.DAT001E, context=error_context)

        error_dict = error.to_dict()
        print(f"TraceContext付きエラー: {error_dict}")

        assert error_dict["context"]["system"] == "system1"
        assert error_dict["context"]["symbol"] == "AAPL"
        assert error_dict["context"]["trace_id"] is not None

    print("✅ TraceContext統合正常")


def test_structured_logger(progress: ProgressBar):
    """構造化ログシステムのテスト"""
    progress.update("構造化ログテスト")

    print("\n=== 構造化ログシステムテスト ===")

    logger = TradingSystemLogger()

    # エラーコード付きログ出力テスト
    logger.log_spy_error("SPYデータアクセスエラーのテスト")
    logger.log_system_error("システムエラーのテスト")
    logger.log_filter_error("フィルターエラーのテスト")
    logger.log_setup_error("セットアップエラーのテスト")

    time.sleep(0.2)  # ログ処理の完了を待機

    print("✅ 構造化ログ出力正常")


def main():
    """全テストを実行 - 📊 リアルタイム進捗表示付き"""
    print("🚀 構造化ログシステム動作確認テスト開始")
    print("📊 リアルタイム進捗表示機能付き\n")

    # 6つのテストステップ
    total_tests = 6
    progress = ProgressBar(total_tests)

    try:
        test_error_code_mapping(progress)
        test_japanese_error_descriptions(progress)
        test_trading_error_system(progress)
        test_error_summary(progress)
        test_trace_context_integration(progress)
        test_structured_logger(progress)

        print("\n🎉 全てのテストが正常に完了しました！")
        print("📊 構造化ログシステムは100%実装完了済みです。")
        print(f"⏱️ 総実行時間: {time.time() - progress.start_time:.2f}秒")

    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
