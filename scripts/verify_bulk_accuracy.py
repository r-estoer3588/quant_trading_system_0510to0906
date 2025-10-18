#!/usr/bin/env python3
"""Bulk APIデータの精度を検証するスクリプト

個別APIまたは既存キャッシュとの差異を確認し、
Bulk APIの信頼性を数値化します。
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
import pandas as pd
import requests

# プロジェクトルートをPYTHONPATHに追加
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.cache_manager import CacheManager  # noqa: E402
from config.environment import get_env_config  # noqa: E402
from config.settings import get_settings  # noqa: E402
from scripts.update_from_bulk_last_day import fetch_bulk_last_day  # noqa: E402

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")


class BulkDataVerifier:
    """Bulk APIデータの品質を検証するクラス"""

    def __init__(self):
        self.settings = get_settings()
        self.cm = CacheManager(self.settings)
        self.discrepancies: list[dict[str, Any]] = []

        # 環境変数から設定を読み込み
        env_config = get_env_config()

        # Volume差異の許容範囲（環境変数で制御可能）
        self.volume_tolerance = env_config.bulk_api_volume_tolerance / 100.0
        # 価格データの許容範囲（従来通り）
        self.price_tolerance = env_config.bulk_api_price_tolerance / 100.0
        # 信頼性スコアの最低基準
        self.min_reliability = env_config.bulk_api_min_reliability / 100.0

    def fetch_individual_eod(self, symbol: str, date: str | None = None) -> dict[str, Any]:
        """個別APIで最新データを取得（検証用）"""
        if not API_KEY:
            return {}

        url = f"https://eodhistoricaldata.com/api/eod/{symbol.lower()}.US"
        params = {
            "api_token": API_KEY,
            "fmt": "json",
            "from": (date if date else (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")),
            "to": date if date else datetime.now().strftime("%Y-%m-%d"),
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data[-1] if data else {}
        except Exception as e:
            print(f"  ⚠️ 個別API取得エラー {symbol}: {e}")
        return {}

    def compare_prices(
        self,
        bulk_row: pd.Series,
        reference: dict[str, Any] | pd.Series,
        symbol: str,
        tolerance: float = 0.01,
    ) -> dict[str, Any]:
        """価格データを比較（許容誤差: デフォルト1%）"""
        issues = []

        # 価格フィールドのマッピング
        price_fields = {
            "open": ["open", "Open"],
            "high": ["high", "High"],
            "low": ["low", "Low"],
            "close": ["close", "Close"],
            "adjusted_close": ["adjusted_close", "adjclose", "AdjClose"],
            "volume": ["volume", "Volume"],
        }

        for field, aliases in price_fields.items():
            bulk_val = None
            ref_val = None

            # Bulkデータから値を取得
            for alias in aliases:
                if alias in bulk_row.index and pd.notna(bulk_row[alias]):
                    try:
                        bulk_val = float(bulk_row[alias])
                        break
                    except (ValueError, TypeError):
                        continue

            # 参照データから値を取得
            if isinstance(reference, dict):
                for alias in aliases:
                    if alias in reference and reference[alias] is not None:
                        try:
                            ref_val = float(reference[alias])
                            break
                        except (ValueError, TypeError):
                            continue
            else:
                for alias in aliases:
                    if alias in reference.index and pd.notna(reference[alias]):
                        try:
                            ref_val = float(reference[alias])
                            break
                        except (ValueError, TypeError):
                            continue

            # 比較（Volumeは専用許容範囲を使用）
            if bulk_val is not None and ref_val is not None and ref_val > 0:
                diff_pct = abs(bulk_val - ref_val) / ref_val
                # Volumeは緩和した許容範囲、価格データは厳格な許容範囲
                field_tolerance = self.volume_tolerance if field == "volume" else tolerance

                if diff_pct > field_tolerance:
                    issues.append(
                        {
                            "field": field,
                            "bulk": bulk_val,
                            "reference": ref_val,
                            "diff_pct": diff_pct,
                        }
                    )

        return {"symbol": symbol, "has_issues": len(issues) > 0, "issues": issues}

    def verify_sample_symbols(
        self, sample_symbols: list[str] | None = None, use_individual_api: bool = False
    ) -> dict[str, Any]:
        """サンプル銘柄でBulkデータの精度を検証"""

        if sample_symbols is None:
            # デフォルトのサンプル銘柄（主要指数・大型株・小型株の混合）
            sample_symbols = [
                "SPY",
                "QQQ",
                "IWM",
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "NVDA",
                "META",
            ]

        print("=" * 60)
        print("🔍 Bulk APIデータ品質検証")
        print("=" * 60)
        print("📊 Bulk APIデータを取得中...", flush=True)

        bulk_df = fetch_bulk_last_day()

        if bulk_df is None or bulk_df.empty:
            print("❌ Bulk データの取得に失敗しました")
            return {"success": False, "message": "Bulk data fetch failed"}

        # 日付を確認
        bulk_date = None
        if "date" in bulk_df.columns:
            try:
                bulk_date = pd.to_datetime(bulk_df["date"].iloc[0]).strftime("%Y-%m-%d")
                print(f"📅 Bulk データ日付: {bulk_date}")
            except Exception:
                print("⚠️ 日付の解析に失敗しました")

        print(f"📦 Bulk データ: {len(bulk_df)}行取得")
        print()

        results = {
            "date": bulk_date,
            "total_symbols": len(sample_symbols),
            "verified": 0,
            "issues": [],
            "missing": [],
            "perfect_match": [],
        }

        for idx, symbol in enumerate(sample_symbols, 1):
            print(f"[{idx}/{len(sample_symbols)}] 🔍 検証中: {symbol}")

            # Bulkデータから該当銘柄を抽出
            bulk_sym = bulk_df[bulk_df["code"].str.upper() == symbol] if "code" in bulk_df.columns else pd.DataFrame()

            if bulk_sym.empty:
                print("  ⚠️ Bulkデータに存在しません")
                results["missing"].append(symbol)
                continue

            bulk_row = bulk_sym.iloc[0]
            comparison_done = False

            # 方法1: 既存キャッシュと比較（APIコール不要）
            try:
                cached = self.cm.read(symbol, "full")
                if cached is not None and not cached.empty:
                    # 最新行と比較
                    latest_cached = cached.iloc[-1]

                    # 日付を確認
                    cached_date = None
                    if "Date" in cached.columns:
                        cached_date = pd.to_datetime(cached["Date"].iloc[-1]).strftime("%Y-%m-%d")
                    elif "date" in cached.columns:
                        cached_date = pd.to_datetime(cached["date"].iloc[-1]).strftime("%Y-%m-%d")

                    if cached_date:
                        print(f"  📁 キャッシュ最新日: {cached_date}")

                    comparison = self.compare_prices(bulk_row, latest_cached, symbol)
                    comparison_done = True

                    if comparison["has_issues"]:
                        print("  ⚠️ キャッシュとの差異を検出:")
                        for issue in comparison["issues"]:
                            print(
                                f"    - {issue['field']}: Bulk={issue['bulk']:.2f}, "
                                f"Cache={issue['reference']:.2f} "
                                f"({issue['diff_pct']:.2%}差)"
                            )
                        results["issues"].append(comparison)
                    else:
                        print("  ✅ キャッシュと一致")
                        results["perfect_match"].append(symbol)
            except Exception as e:
                print(f"  ⚠️ キャッシュ比較エラー: {e}")

            # 方法2: 個別APIで再取得して比較（オプション、APIコール消費注意）
            if use_individual_api and not comparison_done:
                print("  🌐 個別APIで検証中...")
                individual = self.fetch_individual_eod(symbol, bulk_date)
                if individual:
                    comparison = self.compare_prices(bulk_row, individual, symbol)
                    if comparison["has_issues"]:
                        print("  ⚠️ 個別APIとの差異を検出:")
                        for issue in comparison["issues"]:
                            print(
                                f"    - {issue['field']}: Bulk={issue['bulk']:.2f}, "
                                f"API={issue['reference']:.2f} "
                                f"({issue['diff_pct']:.2%}差)"
                            )
                        results["issues"].append(comparison)
                    else:
                        print("  ✅ 個別APIと一致")
                        results["perfect_match"].append(symbol)
                else:
                    print("  ⚠️ 個別APIからのデータ取得失敗")

            results["verified"] += 1

        # サマリー出力
        print("\n" + "=" * 60)
        print("📋 検証結果サマリー")
        print("=" * 60)
        print(f"  検証銘柄数: {results['verified']}/{results['total_symbols']}")
        print(f"  完全一致: {len(results['perfect_match'])}件")
        print(f"  問題検出: {len(results['issues'])}件")
        print(f"  データ欠損: {len(results['missing'])}件")

        if results["perfect_match"]:
            print(f"\n✅ 完全一致した銘柄: {', '.join(results['perfect_match'])}")

        if results["issues"]:
            print("\n⚠️ 問題のある銘柄:")
            for item in results["issues"]:
                print(f"  - {item['symbol']}: {len(item['issues'])}項目で差異")

        if results["missing"]:
            print(f"\n⚠️ Bulkデータに存在しない銘柄: {', '.join(results['missing'])}")

        # 信頼性スコアの算出
        verified_count = results["verified"]
        issue_count = len(results["issues"])

        if verified_count > 0:
            total_symbols = results["total_symbols"]
            reliability_score = (verified_count - issue_count) / total_symbols
        else:
            reliability_score = 0.0

        results["reliability_score"] = reliability_score

        print("\n" + "=" * 60)
        # 環境変数で設定された最低基準と比較
        if reliability_score >= 0.95:
            print(f"✅ 信頼性スコア: {reliability_score:.1%}")
            print("👍 Bulk APIは高品質です。安心して使用できます。")
        elif reliability_score >= self.min_reliability:
            print(f"⚠️ 信頼性スコア: {reliability_score:.1%}")
            print("💡 一部銘柄で差異があります。重要銘柄は個別確認を推奨。")
            print(f"   （基準: {self.min_reliability:.0%}以上で使用可能）")
        else:
            print(f"❌ 信頼性スコア: {reliability_score:.1%}")
            print("🚨 Bulk APIの品質が低いです。個別API使用を推奨します。")
            print(f"   （基準: {self.min_reliability:.0%}未満）")
        print("=" * 60)

        return results

    def verify_timing_impact(self):
        """取得タイミングによる影響を調査"""
        print("\n" + "=" * 60)
        print("🕐 取得タイミング影響調査")
        print("=" * 60)

        # 現在時刻を確認
        now = datetime.now()

        # 米国市場のクローズ時刻（ET 4PM）
        # 日本時間で考えると、夏時間: 翌朝5時、冬時間: 翌朝6時
        print(f"現在時刻: {now.strftime('%Y-%m-%d %H:%M:%S')}")

        # 簡易的な判定（実際の運用では市場カレンダーAPIを使用すべき）
        hour = now.hour

        if 6 <= hour < 10:
            print("✅ 推奨実行時間帯です（米国市場クローズ後、データ安定）")
            print("💡 この時間帯のBulk API取得は信頼性が高いです。")
        elif 0 <= hour < 6:
            print("⚠️ 市場クローズ直後の時間帯です")
            print("💡 データが不完全な可能性があります。6時以降の実行を推奨。")
        else:
            print("ℹ️ 通常の実行時間帯です")
            print("💡 前営業日のデータが取得されます。")

        print("=" * 60)

    def analyze_bulk_coverage(self):
        """Bulkデータのカバレッジを分析"""
        print("\n" + "=" * 60)
        print("📊 Bulkデータカバレッジ分析")
        print("=" * 60)

        bulk_df = fetch_bulk_last_day()
        if bulk_df is None or bulk_df.empty:
            print("❌ Bulk データの取得に失敗しました")
            return

        # ユニバースと比較
        try:
            from common.symbol_universe import build_symbol_universe_from_settings

            universe = build_symbol_universe_from_settings(self.settings)
            universe_set = set(s.upper() for s in universe)

            if "code" in bulk_df.columns:
                bulk_symbols = set(bulk_df["code"].str.upper())

                coverage = len(bulk_symbols & universe_set) / len(universe_set) if universe_set else 0
                missing = universe_set - bulk_symbols

                print(f"ユニバース銘柄数: {len(universe_set)}")
                print(f"Bulk取得銘柄数: {len(bulk_symbols)}")
                print(f"カバレッジ: {coverage:.1%}")

                if missing:
                    print(f"\n⚠️ Bulkに存在しない銘柄: {len(missing)}件")
                    if len(missing) <= 20:
                        print(f"  {', '.join(sorted(missing))}")
                    else:
                        sample = sorted(missing)[:20]
                        print(f"  (最初の20件) {', '.join(sample)}")
                        print(f"  ... 他 {len(missing) - 20}件")
        except Exception as e:
            print(f"⚠️ カバレッジ分析エラー: {e}")

        print("=" * 60)


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bulk APIデータの精度検証",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルトのサンプル銘柄で検証
  python scripts/verify_bulk_accuracy.py

  # 特定銘柄を指定して検証
  python scripts/verify_bulk_accuracy.py --symbols AAPL,MSFT,SPY

  # 個別APIでも検証（APIコール消費注意）
  python scripts/verify_bulk_accuracy.py --use-api

  # タイミング影響を調査
  python scripts/verify_bulk_accuracy.py --timing

  # カバレッジ分析
  python scripts/verify_bulk_accuracy.py --coverage

  # 全機能実行
  python scripts/verify_bulk_accuracy.py --full
        """,
    )

    parser.add_argument("--symbols", type=str, help="検証する銘柄（カンマ区切り）例: AAPL,MSFT,SPY")
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="個別APIでも検証（APIコール消費に注意）",
    )
    parser.add_argument("--timing", action="store_true", help="取得タイミングの影響を調査")
    parser.add_argument("--coverage", action="store_true", help="Bulkデータのカバレッジを分析")
    parser.add_argument(
        "--full",
        action="store_true",
        help="全機能を実行（タイミング・カバレッジ・精度検証）",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="価格差異の許容誤差（デフォルト: 0.01 = 1%%）",
    )

    args = parser.parse_args()

    verifier = BulkDataVerifier()

    # タイミング影響調査
    if args.timing or args.full:
        verifier.verify_timing_impact()

    # カバレッジ分析
    if args.coverage or args.full:
        verifier.analyze_bulk_coverage()

    # 精度検証
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        results = verifier.verify_sample_symbols(symbols, use_individual_api=args.use_api)
    else:
        results = verifier.verify_sample_symbols(use_individual_api=args.use_api)

    # 終了コード
    reliability_score = results.get("reliability_score", 0)
    if reliability_score >= 0.80:
        return 0  # 成功
    else:
        return 1  # 失敗（品質が低い）


if __name__ == "__main__":
    sys.exit(main())
