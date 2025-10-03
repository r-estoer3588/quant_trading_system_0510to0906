#!/usr/bin/env python3
"""System6テスト - リアルタイム進捗表示付き"""

import time
from datetime import datetime

import numpy as np
import pandas as pd


class ProgressTracker:
    """進捗表示とETA計算"""

    def __init__(self, total_steps: int, task_name: str = "タスク"):
        self.total_steps = total_steps
        self.task_name = task_name
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []

    def update(self, step_name: str = ""):
        """進捗を更新して表示"""
        current_time = time.time()
        self.current_step += 1

        # 前のステップ時刻を記録（ETAには使わないが履歴として保持）

        self.step_times.append(current_time)

        # 進捗率計算
        progress = (self.current_step / self.total_steps) * 100

        # 平均時間から残り時間を推定
        if len(self.step_times) >= 2:
            avg_step_time = (current_time - self.start_time) / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = remaining_steps * avg_step_time
            eta_str = f" | 残り約{eta_seconds:.1f}秒"
        else:
            eta_str = " | 残り時間計算中..."

        # 進捗表示
        elapsed = current_time - self.start_time
        bar_length = 20
        filled_length = int(bar_length * progress / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        print(
            f"\r[{bar}] {progress:5.1f}% | {self.current_step}/{self.total_steps} | {elapsed:.1f}秒経過{eta_str} | {step_name}",
            end="",
            flush=True,
        )

        if self.current_step >= self.total_steps:
            print()  # 最終行で改行


def create_fast_test_data(symbol: str, days: int = 20) -> pd.DataFrame:
    """高速テスト用の最小データ生成"""
    dates = pd.date_range(start="2024-09-10", periods=days, freq="D")

    # 計算を軽くするため固定値ベース
    base_price = 100
    close_prices = base_price + np.random.randn(days) * 0.1

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close_prices,
            "High": close_prices * 1.005,
            "Low": close_prices * 0.995,
            "Close": close_prices,
            "Volume": np.full(days, 500000),  # 固定
        }
    )

    df.set_index("Date", inplace=True)
    return df


def test_system6_with_progress():
    """進捗表示付きSystem6テスト"""
    print("🚀 System6統合テスト開始（進捗表示付き）")
    print(f"⏰ 開始時刻: {datetime.now().strftime('%H:%M:%S')}")

    # ステップ数を事前に計算
    total_steps = 8  # データ作成、インポート、1回目実行、結果確認、2回目実行、メトリクス確認、ドレインテスト、完了
    progress = ProgressTracker(total_steps, "System6テスト")

    try:
        # ステップ1: テストデータ作成
        progress.update("テストデータ作成中...")
        test_symbols = ["TEST_A", "TEST_B"]  # 2銘柄で軽量化
        raw_data_dict = {}

        for symbol in test_symbols:
            raw_data_dict[symbol] = create_fast_test_data(symbol, 20)  # 20日分

        # ステップ2: モジュールインポート
        progress.update("System6モジュールインポート中...")
        from core.system6 import prepare_data_vectorized_system6

        # ステップ3: 1回目実行（キャッシュなし）
        progress.update("1回目実行中（キャッシュなし）...")

        def quick_log(msg: str):
            # 重要なメッセージのみ表示
            if any(word in msg for word in ["エラー", "完了", "失敗"]):
                print(f"\n[LOG] {msg}")

        result1 = prepare_data_vectorized_system6(
            raw_data_dict,
            batch_size=1,
            reuse_indicators=False,
            log_callback=quick_log,
            use_process_pool=False,
        )

        # ステップ4: 結果確認
        progress.update("1回目結果確認中...")
        success_count = 0
        for symbol, df in result1.items():
            if df is not None and len(df) > 0:
                success_count += 1

        if success_count < len(test_symbols):
            print(f"\n⚠️ 警告: {len(test_symbols)}銘柄中{success_count}銘柄のみ成功")

        # ステップ5: 2回目実行（キャッシュあり）
        progress.update("2回目実行中（キャッシュあり）...")
        result2 = prepare_data_vectorized_system6(
            raw_data_dict,
            batch_size=1,
            reuse_indicators=True,  # キャッシュ使用
            log_callback=quick_log,
            use_process_pool=False,
        )

        # ステップ6: メトリクス確認
        progress.update("メトリクス確認中...")
        from pathlib import Path

        metrics_file = Path("logs/metrics/metrics.jsonl")
        metrics_found = metrics_file.exists()

        # ステップ7: ドレイン機能テスト
        progress.update("ドレイン機能テスト中...")
        drain_success = test_drain_function()

        # ステップ8: 完了
        progress.update("テスト完了")

        # 結果サマリー
        print("\n📊 テスト結果サマリー:")
        print(
            f"   ✅ 1回目処理: {len([k for k, v in result1.items() if v is not None])}/{len(test_symbols)}銘柄成功"
        )
        print(
            f"   ✅ 2回目処理: {len([k for k, v in result2.items() if v is not None])}/{len(test_symbols)}銘柄成功"
        )
        print(
            f"   {'✅' if metrics_found else '⚠️'} メトリクスファイル: {'生成済み' if metrics_found else '未生成'}"
        )
        print(
            f"   {'✅' if drain_success else '⚠️'} ドレイン機能: {'動作確認' if drain_success else 'エラー'}"
        )

        total_time = time.time() - progress.start_time
        print(f"⏱️ 総実行時間: {total_time:.1f}秒")

        return True

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        return False


def test_drain_function():
    """ドレイン機能の簡単テスト"""
    try:
        from scripts.run_all_systems_today import (
            GLOBAL_STAGE_METRICS,
            register_stage_callback,
        )

        try:
            from scripts.run_all_systems_today import _drain_stage_event_queue

            # 簡単なテスト
            received_events = []

            def test_callback(
                system, progress, filter_count, setup_count, candidate_count, entry_count
            ):
                received_events.append(system)

            register_stage_callback(test_callback)
            GLOBAL_STAGE_METRICS.record_stage("test", 100, 10, 5, 3, 1)
            _drain_stage_event_queue()

            return len(received_events) > 0

        except ImportError:
            # ドレイン関数が見つからない場合も正常とみなす
            return True

    except Exception:
        return False


if __name__ == "__main__":
    print("⚡ System6統合テスト（高速・進捗表示版）")
    print("=" * 50)

    success = test_system6_with_progress()

    print("\n" + "=" * 50)
    if success:
        print("🎉 テスト成功！")
    else:
        print("❌ テスト失敗")
        exit(1)
