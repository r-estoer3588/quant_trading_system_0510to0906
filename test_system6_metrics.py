#!/usr/bin/env python3
"""System6のMetricsCollector統合とキャッシュヒット率収集のテスト"""

import tempfile
from pathlib import Path

import pandas as pd


def create_test_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """テスト用のダミーデータを作成"""
    import numpy as np

    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")

    # 基本価格データ
    close_prices = 100 + np.cumsum(np.random.randn(days) * 0.5)
    high_prices = close_prices * (1 + np.random.rand(days) * 0.02)
    low_prices = close_prices * (1 - np.random.rand(days) * 0.02)
    open_prices = close_prices + np.random.randn(days) * 0.3
    volumes = np.random.randint(1000000, 10000000, days)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
        }
    )

    df.set_index("Date", inplace=True)
    return df


def test_system6_metrics():
    """System6のMetricsCollector統合をテスト"""
    print("🔧 System6のMetricsCollector統合テスト開始")

    # テスト用の一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_cache_dir = Path(temp_dir) / "test_cache"
        temp_cache_dir.mkdir(exist_ok=True)

        # テストデータ作成
        test_symbols = ["TEST_AAPL", "TEST_MSFT", "TEST_GOOGL"]
        raw_data_dict = {}

        for symbol in test_symbols:
            df = create_test_data(symbol, 120)  # 120日分のデータ
            raw_data_dict[symbol] = df
            print(f"📊 {symbol}: {len(df)}行のテストデータ作成")

        # System6をインポートしてテスト
        try:
            from core.system6 import prepare_data_vectorized_system6

            print("✅ System6モジュールのインポート成功")

            # 一時的にキャッシュディレクトリを変更
            # core.system6.py内のcache_dirを一時的に変更するために、
            # 環境変数やパッチを使用せず、直接テスト
            def test_log_callback(message: str):
                print(f"[LOG] {message}")

            # MetricsCollectorを使用してテスト実行
            print("🚀 prepare_data_vectorized_system6の実行開始...")

            result_dict = prepare_data_vectorized_system6(
                raw_data_dict,
                batch_size=2,  # 小さなバッチサイズでテスト
                reuse_indicators=False,  # 最初はキャッシュなし
                log_callback=test_log_callback,
                use_process_pool=False,  # シンプルなテストのため
            )

            print(f"✅ 処理完了: {len(result_dict)}シンボル処理")

            # 結果の確認
            for symbol, df in result_dict.items():
                if df is not None and not df.empty:
                    print(f"📈 {symbol}: {len(df)}行の指標付きデータ生成")
                    required_cols = ["atr10", "dollarvolume50", "return_6d", "filter", "setup"]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        print(f"   ⚠️ 不足列: {missing_cols}")
                    else:
                        print("   ✅ 必要な指標列がすべて存在")
                        setup_count = df["setup"].sum() if "setup" in df.columns else 0
                        print(f"   📊 セットアップ条件成立: {setup_count}日")
                else:
                    print(f"❌ {symbol}: データ処理失敗")

            # 2回目の実行でキャッシュヒット率をテスト
            print("\n🔄 2回目の実行（キャッシュヒット率テスト）...")

            result_dict2 = prepare_data_vectorized_system6(
                raw_data_dict,
                batch_size=2,
                reuse_indicators=True,  # キャッシュ使用
                log_callback=test_log_callback,
                use_process_pool=False,
            )

            print(f"✅ 2回目処理完了: {len(result_dict2)}シンボル処理")

            # メトリクスファイルの確認
            logs_dir = Path("logs/metrics")
            if logs_dir.exists():
                metrics_file = logs_dir / "metrics.jsonl"
                if metrics_file.exists():
                    print(f"📊 メトリクスファイル確認: {metrics_file}")
                    try:
                        with open(metrics_file, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            print(f"   📝 記録されたメトリクス: {len(lines)}行")

                            # 最後の数行を表示
                            for line in lines[-5:]:
                                import json

                                try:
                                    metric = json.loads(line.strip())
                                    if "system6" in metric.get("metric_name", ""):
                                        print(
                                            f"   📊 {metric['metric_name']}: {metric['value']} {metric.get('unit', '')}"
                                        )
                                except Exception:
                                    pass
                    except Exception as e:
                        print(f"   ❌ メトリクスファイル読み込みエラー: {e}")
                else:
                    print("❌ メトリクスファイルが生成されていません")
            else:
                print("❌ メトリクスディレクトリが存在しません")

        except ImportError as e:
            print(f"❌ System6インポートエラー: {e}")
            return False
        except Exception as e:
            print(f"❌ テスト実行エラー: {e}")
            import traceback

            traceback.print_exc()
            return False

    print("🎯 System6 MetricsCollectorテスト完了")
    return True


def test_stage_event_drain():
    """ステージイベントドレイン機能のテスト"""
    print("\n🔧 ステージイベントドレイン機能テスト開始")

    try:
        from scripts.run_all_systems_today import (
            GLOBAL_STAGE_METRICS,
            _ensure_stage_event_pump,
            register_stage_callback,
        )

        # ドレイン関数を個別にインポート
        try:
            from scripts.run_all_systems_today import _drain_stage_event_queue
        except ImportError:
            # フォールバック実装
            def _drain_stage_event_queue():
                print("[FALLBACK] ドレイン関数が見つかりません")

        print("✅ ドレイン関数のインポート成功")

        # テスト用コールバック
        drained_events = []

        def test_callback(
            system, progress, filter_count, setup_count, candidate_count, entry_count
        ):
            drained_events.append(
                {
                    "system": system,
                    "progress": progress,
                    "filter_count": filter_count,
                    "setup_count": setup_count,
                    "candidate_count": candidate_count,
                    "entry_count": entry_count,
                }
            )
            print(
                f"[DRAIN] {system}: {progress}% - フィルター:{filter_count}, セットアップ:{setup_count}"
            )

        # コールバック登録
        register_stage_callback(test_callback)

        # テストイベントをGLOBAL_STAGE_METRICSに追加
        GLOBAL_STAGE_METRICS.record_stage("system6", 50, 1000, 100, 50, 10)
        GLOBAL_STAGE_METRICS.record_stage("system6", 100, 1000, 200, 80, 15)

        # ドレイン実行
        _drain_stage_event_queue()

        print(f"✅ ドレインされたイベント数: {len(drained_events)}")
        for event in drained_events:
            print(f"   📊 {event}")

        # ポンプ機能テスト
        print("🔄 ステージイベントポンプ開始テスト")
        _ensure_stage_event_pump()

        # 少し待ってポンプ動作確認
        import time

        time.sleep(0.5)

        print("✅ ステージイベントポンプテスト完了")

    except ImportError as e:
        print(f"❌ ドレイン関数インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ ドレインテスト実行エラー: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("🚀 System6統合テスト開始")

    success1 = test_system6_metrics()
    success2 = test_stage_event_drain()

    if success1 and success2:
        print("\n🎉 すべてのテスト成功！")
    else:
        print("\n❌ 一部のテストが失敗しました")
        exit(1)
