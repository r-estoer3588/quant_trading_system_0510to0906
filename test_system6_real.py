#!/usr/bin/env python3
"""System6 実運用パフォーマンステスト

実際のSystem6当日実行の速度を直接測定
"""
import os
import time

# プロセスプール強制有効
os.environ["USE_PROCESS_POOL"] = "1"


def test_system6_real_performance():
    """System6を含む当日パイプラインの実測テスト"""
    print("=== System6 実運用パフォーマンステスト ===")

    # 実際の当日パイプラインの実行（System6のみ）
    from scripts.run_all_systems_today import main
    import sys

    # 元の引数を保存
    original_argv = sys.argv.copy()

    try:
        # 当日パイプライン実行（並列のみ）
        sys.argv = ["run_all_systems_today.py", "--parallel"]

        print("� System6当日パイプライン実行開始...")
        start_time = time.time()

        # 実際の実行
        result = main()

        end_time = time.time()
        execution_time = end_time - start_time

        print("✅ System6実行完了!")
        print(f"⏱️  実行時間: {execution_time:.2f}秒 ({execution_time/60:.1f}分)")

        return execution_time, result

    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        return 0, None

    finally:
        # 引数を元に戻す
        sys.argv = original_argv


def test_system6_task_performance():
    """VSCodeタスクでのSystem6実行テスト"""
    print("=== System6 VSCodeタスク実行テスト ===")

    import subprocess

    # 当日実行タスクをPowerShellで実行
    cmd = ["python", "scripts/run_all_systems_today.py", "--parallel"]

    print("� System6タスク実行開始...")
    print(f"💻 コマンド: {' '.join(cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd()  # 10分タイムアウト
        )

        end_time = time.time()
        execution_time = end_time - start_time

        print("✅ System6タスク実行完了!")
        print(f"⏱️  実行時間: {execution_time:.2f}秒 ({execution_time/60:.1f}分)")

        if result.returncode == 0:
            print("✅ 正常終了")
            # 最後の部分のみ表示
            if result.stdout:
                lines = result.stdout.strip().split("\n")
                print("� 出力（最後の10行）:")
                for line in lines[-10:]:
                    print(f"   {line}")
        else:
            print(f"❌ エラー終了 (code: {result.returncode})")
            if result.stderr:
                print(f"⚠️  エラー内容: {result.stderr}")

        return execution_time, result.returncode == 0

    except subprocess.TimeoutExpired:
        print("⏰ タイムアウト: 10分以上実行")
        return 600, False

    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"❌ 実行エラー: {e}")
        return execution_time, False


if __name__ == "__main__":
    print("テスト方法を選択:")
    print("1. VSCodeタスク実行テスト（推奨）")
    print("2. 直接実行テスト")

    try:
        choice = input("選択 (1 or 2): ").strip()

        if choice == "1":
            exec_time, success = test_system6_task_performance()
        elif choice == "2":
            exec_time, result = test_system6_real_performance()
            success = result is not None
        else:
            print("無効な選択")
            exit(1)

        if success:
            print(f"\n🎯 System6実測結果: {exec_time:.2f}秒")
            if exec_time > 60:
                print(f"   📏 分換算: {exec_time/60:.1f}分")
        else:
            print("\n❌ テスト失敗")

    except KeyboardInterrupt:
        print("\n⏹️  テスト中断")
