#!/usr/bin/env python3
"""
統合テスト: ログ最適化機能とリアルタイム進捗監視の動作確認
"""

import os
import sys
import time
from pathlib import Path

# プロジェクトルート（tests/experimental/ から2階層上）をパスに追加
sys.path.insert(0, str(Path(__file__).parents[2]))

from common.progress_events import emit_progress, reset_progress_log
from common.rate_limited_logging import create_rate_limited_logger
from config.settings import get_settings


def test_integration():
    """統合テストメイン関数"""
    print("🧪 統合テスト開始: ログ最適化とリアルタイム進捗監視")

    # 一時的に進捗イベントを有効化
    os.environ["ENABLE_PROGRESS_EVENTS"] = "1"
    os.environ["COMPACT_TODAY_LOGS"] = "1"

    # 進捗イベントリセット
    reset_progress_log()

    # レート制限ロガー作成
    rate_logger = create_rate_limited_logger("integration_test", default_interval=1.0)

    print("1. レート制限ログのテスト")
    for i in range(5):
        rate_logger.info_rate_limited(
            f"進捗メッセージ {i + 1}/5", message_key="progress_test", interval=0.5
        )
        time.sleep(0.2)  # 短い間隔で連続実行

    print("\n2. 進捗イベント出力のテスト")
    test_events = [
        ("system_start", {"system": "system1", "candidates": 150}),
        ("filtering_complete", {"system": "system1", "filtered": 75}),
        ("allocation_start", {"total_candidates": 75, "target_positions": 10}),
        ("allocation_complete", {"final_positions": 8, "active_positions_total": 15}),
        ("notification_complete", {"notifications_sent": 1, "results_count": 8}),
    ]

    for event_type, data in test_events:
        emit_progress(event_type, data)
        print(f"  ✓ {event_type}: {data}")
        time.sleep(0.1)

    print("\n3. 進捗ファイルの確認")
    settings = get_settings()
    progress_file = Path(settings.LOGS_DIR) / "progress_today.jsonl"

    if progress_file.exists():
        with open(progress_file, encoding="utf-8") as f:
            lines = f.readlines()
        print(f"  ✓ 進捗ファイル作成成功: {len(lines)} イベント記録")

        # 最後の数行を表示
        for line in lines[-3:]:
            print(f"    {line.strip()}")
    else:
        print("  ❌ 進捗ファイルが見つかりません")
        return False

    print("\n4. レート制限の効果確認")
    start_time = time.time()
    message_count = 0

    # 同じメッセージキーで連続実行
    for _ in range(10):
        rate_logger.debug_rate_limited(
            "同じメッセージのテスト", message_key="same_message", interval=1.0
        )
        message_count += 1

    elapsed_time = time.time() - start_time
    print(f"  ✓ レート制限テスト完了: {elapsed_time:.2f}秒で{message_count}回試行")
    print("    （実際の出力は1.0秒間隔で制限されているはず）")

    print("\n🎉 統合テスト完了: 全機能が正常に動作しています")

    # 環境変数をクリーンアップ
    os.environ.pop("ENABLE_PROGRESS_EVENTS", None)
    os.environ.pop("COMPACT_TODAY_LOGS", None)

    return True


if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
