#!/usr/bin/env python3
"""スケジュール実行用の安全な日次更新スクリプト

Bulk APIを試み、品質チェックで問題があれば個別APIにフォールバック。
実行結果をログに記録し、継続的な品質モニタリングを可能にします。
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

# プロジェクトルートをPYTHONPATHに追加
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from scripts.verify_bulk_accuracy import BulkDataVerifier  # noqa: E402


class SafeDailyUpdater:
    """安全な日次更新を実行するクラス"""

    def __init__(self):
        self.settings = get_settings()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(self.settings.LOGS_DIR) / f"daily_update_{timestamp}.log"
        self.stats: dict[str, str | bool | float | None] = {
            "start_time": None,
            "end_time": None,
            "method_used": None,
            "success": False,
            "errors": [],
            "bulk_reliability_score": None,
        }

    def log(self, message: str, level: str = "INFO"):
        """ログ出力（コンソールとファイル）"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message, flush=True)

        # ファイルにも記録
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_message + "\n")
        except Exception as e:
            print(f"⚠️ ログファイル書き込みエラー: {e}")

    def wait_for_market_data(self) -> bool:
        """市場データが安定するまで待機（簡易チェック）"""
        now = datetime.now()

        # 日本時間で朝6時以降を安定時刻とする
        # （米国市場クローズから十分な時間が経過）
        if now.hour < 6:
            wait_hours = 6 - now.hour
            self.log(
                f"市場データ安定化のため、推奨実行時刻は朝6時以降です（現在から約{wait_hours}時間後）",
                "WARNING",
            )
            self.log("処理は続行しますが、データが不完全な可能性があります", "WARNING")
            return True  # 警告のみで続行

        self.log("実行時刻は推奨範囲内です（市場データは安定していると想定）", "INFO")
        return True

    def verify_bulk_quality(self) -> tuple[bool, float]:
        """Bulk APIの品質を事前検証"""
        self.log("=" * 60)
        self.log("Bulk APIデータ品質の事前検証を開始", "INFO")
        self.log("=" * 60)

        try:
            verifier = BulkDataVerifier()
            verification_result = verifier.verify_sample_symbols()

            reliability_score = verification_result.get("reliability_score", 0.0)
            self.stats["bulk_reliability_score"] = reliability_score

            self.log(f"信頼性スコア: {reliability_score:.1%}", "INFO")

            if reliability_score >= 0.95:
                self.log("品質チェック合格: Bulk APIは高品質です", "SUCCESS")
                return True, reliability_score
            elif reliability_score >= 0.80:
                self.log(
                    "品質チェック注意: 一部銘柄で差異がありますが、許容範囲内です",
                    "WARNING",
                )
                return True, reliability_score
            else:
                self.log(
                    f"品質チェック不合格: 信頼性が低いです（{reliability_score:.1%}）",
                    "ERROR",
                )
                return False, reliability_score

        except Exception as e:
            self.log(f"品質検証中にエラーが発生: {e}", "ERROR")
            return False, 0.0

    def run_bulk_update(self) -> bool:
        """Bulk API更新を実行"""
        self.log("=" * 60)
        self.log("Bulk API更新を開始", "INFO")
        self.log("=" * 60)

        try:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "update_from_bulk_last_day.py"),
                "--workers",
                "16",
                "--tail-rows",
                "240",
            ]

            self.log(f"実行コマンド: {' '.join(cmd)}", "INFO")

            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=ROOT)
            elapsed = time.time() - start_time

            self.log(f"Bulk更新完了: {elapsed:.1f}秒", "INFO")

            if result.returncode != 0:
                self.log(
                    f"Bulk更新が失敗しました（終了コード: {result.returncode}）",
                    "ERROR",
                )
                if result.stderr:
                    self.log(f"エラー出力:\n{result.stderr}", "ERROR")
                return False

            # Rolling cache更新
            self.log("Rolling cache更新を開始", "INFO")
            cmd_rolling = [
                sys.executable,
                str(ROOT / "scripts" / "build_rolling_with_indicators.py"),
                "--workers",
                "4",
            ]

            start_rolling = time.time()
            result_rolling = subprocess.run(
                cmd_rolling, capture_output=True, text=True, timeout=900, cwd=ROOT
            )
            elapsed_rolling = time.time() - start_rolling

            if result_rolling.returncode != 0:
                self.log(
                    f"Rolling cache更新が失敗しました（終了コード: {result_rolling.returncode}）",
                    "WARNING",
                )
                if result_rolling.stderr:
                    self.log(f"エラー出力:\n{result_rolling.stderr}", "WARNING")
            else:
                self.log(f"Rolling cache更新完了: {elapsed_rolling:.1f}秒", "INFO")

            self.log("✅ Bulk API更新が正常に完了しました", "SUCCESS")
            return True

        except subprocess.TimeoutExpired:
            self.log("Bulk更新がタイムアウトしました（30分制限）", "ERROR")
            return False
        except Exception as e:
            self.log(f"Bulk更新で予期しないエラー: {e}", "ERROR")
            import traceback

            self.log(traceback.format_exc(), "ERROR")
            return False

    def run_individual_update(self) -> bool:
        """個別API更新を実行（フォールバック）"""
        self.log("=" * 60)
        self.log("⚠️ 個別API更新（フォールバック）を開始", "WARNING")
        self.log("=" * 60)
        self.log("注意: この方法は大量のAPIコールを消費します", "WARNING")

        try:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "update_cache_all.py"),
                "--parallel",
                "--workers",
                "4",
            ]

            self.log(f"実行コマンド: {' '.join(cmd)}", "INFO")

            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=ROOT)
            elapsed = time.time() - start_time

            self.log(f"個別API更新完了: {elapsed / 60:.1f}分", "INFO")

            if result.returncode != 0:
                self.log(
                    f"個別API更新が失敗しました（終了コード: {result.returncode}）",
                    "ERROR",
                )
                if result.stderr:
                    self.log(f"エラー出力:\n{result.stderr}", "ERROR")
                return False

            self.log("✅ 個別API更新が正常に完了しました", "SUCCESS")
            return True

        except subprocess.TimeoutExpired:
            self.log("個別API更新がタイムアウトしました（2時間制限）", "ERROR")
            return False
        except Exception as e:
            self.log(f"個別API更新で予期しないエラー: {e}", "ERROR")
            import traceback

            self.log(traceback.format_exc(), "ERROR")
            return False

    def run_post_update_checks(self):
        """更新後の検証（シグナル生成テスト）"""
        self.log("=" * 60)
        self.log("更新後の検証を開始", "INFO")
        self.log("=" * 60)

        try:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_all_systems_today.py"),
                "--test-mode",
                "mini",
                "--skip-external",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=ROOT)

            if result.returncode == 0:
                self.log("✅ 更新後検証: シグナル生成テスト成功", "SUCCESS")
            else:
                self.log("⚠️ 更新後検証: シグナル生成テストに問題があります", "WARNING")
                if result.stderr:
                    self.log(f"テスト出力:\n{result.stderr}", "WARNING")

        except subprocess.TimeoutExpired:
            self.log("更新後検証がタイムアウトしました", "WARNING")
        except Exception as e:
            self.log(f"更新後検証でエラー: {e}", "WARNING")

    def save_statistics(self):
        """統計情報を保存"""
        try:
            stats_file = Path(self.settings.LOGS_DIR) / "daily_update_stats.json"
            stats_file.parent.mkdir(parents=True, exist_ok=True)

            existing_stats = []
            if stats_file.exists():
                try:
                    with open(stats_file, "r", encoding="utf-8") as f:
                        existing_stats = json.load(f)
                except Exception:
                    existing_stats = []

            existing_stats.append(self.stats)

            # 最新30日分のみ保持
            existing_stats = existing_stats[-30:]

            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(existing_stats, f, indent=2, ensure_ascii=False)

            self.log(f"統計情報を保存しました: {stats_file}", "INFO")

        except Exception as e:
            self.log(f"統計情報の保存に失敗: {e}", "WARNING")

    def execute(self) -> bool:
        """メイン実行フロー"""
        self.stats["start_time"] = datetime.now().isoformat()

        self.log("=" * 60)
        self.log("🚀 日次更新処理を開始します", "INFO")
        self.log("=" * 60)
        self.log(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
        self.log(f"ログファイル: {self.log_file}", "INFO")

        # 市場データの安定性確認
        self.wait_for_market_data()

        # Bulk APIの品質を事前検証
        bulk_quality_ok, reliability_score = self.verify_bulk_quality()

        if bulk_quality_ok:
            # Bulk API更新を試行
            if self.run_bulk_update():
                self.stats["method_used"] = "bulk"
                self.stats["success"] = True
            else:
                # フォールバック: 個別API更新
                self.log("=" * 60)
                self.log(
                    "⚠️ Bulk API更新に失敗したため、個別APIにフォールバックします",
                    "WARNING",
                )
                self.log("=" * 60)
                if self.run_individual_update():
                    self.stats["method_used"] = "individual_fallback"
                    self.stats["success"] = True
                else:
                    self.stats["method_used"] = "failed"
                    self.stats["success"] = False
                    self.log("❌ すべての更新方法が失敗しました", "ERROR")
        else:
            # 品質が低いため、最初から個別APIを使用
            self.log("=" * 60)
            self.log("⚠️ Bulk APIの品質が低いため、個別APIを使用します", "WARNING")
            self.log("=" * 60)
            if self.run_individual_update():
                self.stats["method_used"] = "individual_quality"
                self.stats["success"] = True
            else:
                self.stats["method_used"] = "failed"
                self.stats["success"] = False
                self.log("❌ 個別API更新が失敗しました", "ERROR")

        # 更新後の検証
        if self.stats["success"]:
            self.run_post_update_checks()

        self.stats["end_time"] = datetime.now().isoformat()

        # 統計情報を保存
        self.save_statistics()

        # 完了メッセージ
        self.log("=" * 60)
        if self.stats["success"]:
            self.log(
                f"✅ 日次更新が正常に完了しました（方法: {self.stats['method_used']}）",
                "SUCCESS",
            )
            if self.stats.get("bulk_reliability_score"):
                self.log(
                    f"   Bulk信頼性スコア: {self.stats['bulk_reliability_score']:.1%}",
                    "INFO",
                )
        else:
            self.log("❌ 日次更新が失敗しました", "ERROR")
            self.log("💡 手動で update_cache_all.py の実行を検討してください", "INFO")

        self.log("=" * 60)
        self.log(f"ログファイル: {self.log_file}", "INFO")

        return self.stats["success"]


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(
        description="安全な日次更新スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 通常の日次更新
  python scripts/scheduled_daily_update.py

  # Windows タスクスケジューラーに登録
  schtasks /create /tn "QuantTradingDailyUpdate" ^
    /tr "C:\\Repos\\quant_trading_system\\venv\\Scripts\\python.exe ^
    C:\\Repos\\quant_trading_system\\scripts\\scheduled_daily_update.py" ^
    /sc daily /st 06:00

  # cron（Linux/Mac）に登録
  0 6 * * * cd /path/to/quant_trading_system && ./venv/bin/python scripts/scheduled_daily_update.py
        """,
    )

    parser.add_argument(
        "--force-bulk",
        action="store_true",
        help="品質チェックをスキップしてBulk APIを強制使用",
    )
    parser.add_argument(
        "--force-individual",
        action="store_true",
        help="Bulk APIをスキップして個別APIを強制使用",
    )

    args = parser.parse_args()

    updater = SafeDailyUpdater()

    if args.force_individual:
        updater.log("個別API更新を強制実行します", "INFO")
        updater.stats["start_time"] = datetime.now().isoformat()
        success = updater.run_individual_update()
        updater.stats["method_used"] = "individual_forced"
        updater.stats["success"] = success
        updater.stats["end_time"] = datetime.now().isoformat()
        updater.save_statistics()
        return 0 if success else 1

    if args.force_bulk:
        updater.log("Bulk API更新を強制実行します（品質チェックスキップ）", "WARNING")
        updater.stats["start_time"] = datetime.now().isoformat()
        success = updater.run_bulk_update()
        updater.stats["method_used"] = "bulk_forced"
        updater.stats["success"] = success
        updater.stats["end_time"] = datetime.now().isoformat()
        updater.save_statistics()
        return 0 if success else 1

    # 通常の実行
    success = updater.execute()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
