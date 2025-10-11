#!/usr/bin/env python
"""配分ロジックのデバッグユーティリティ

TRDlist は生成されるのに Entry が 0 件になる問題を調査するツール。
候補生成から最終配分までの各ステップを詳細にトレースし、
どの段階で候補が除外されるかを特定します。

使い方:
  python tools/debug_allocation.py --test-mode mini --verbose

機能:
  - 候補生成から最終配分までのプロセスを詳細にトレース
  - 各システムの候補と、配分後の結果を比較
  - なぜ候補が最終リストに選ばれなかったかの理由を分析
  - symbol_system_map との突き合わせ
  - ポジションサイズ計算の詳細診断
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd

from common.cache_manager import CacheManager
from common.symbol_universe import build_symbol_universe_from_settings
from config.environment import get_env_config
from config.settings import get_settings
from core.final_allocation import (
    DEFAULT_LONG_ALLOCATIONS,
    DEFAULT_SHORT_ALLOCATIONS,
    finalize_allocation,
    load_symbol_system_map,
)

# 戦略クラスを直接インポート（run_all_systems_today.py と同じ方法）
from strategies.system1_strategy import System1Strategy
from strategies.system2_strategy import System2Strategy
from strategies.system3_strategy import System3Strategy
from strategies.system4_strategy import System4Strategy
from strategies.system5_strategy import System5Strategy
from strategies.system6_strategy import System6Strategy
from strategies.system7_strategy import System7Strategy

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """ロギング設定"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def create_strategies() -> Dict[str, Any]:
    """戦略インスタンスを作成"""
    strategy_objs = [
        System1Strategy(),
        System2Strategy(),
        System3Strategy(),
        System4Strategy(),
        System5Strategy(),
        System6Strategy(),
        System7Strategy(),
    ]
    return {getattr(s, "SYSTEM_NAME", "").lower(): s for s in strategy_objs}


def generate_simple_test_signals(
    strategies: Dict[str, Any],
    symbol_universe: List[str],
    test_mode: str = "mini",
) -> Dict[str, pd.DataFrame]:
    """テスト用の簡易シグナル生成"""
    per_system = {}

    # 簡易的なダミーシグナルを生成
    for system_name, strategy in strategies.items():
        if test_mode == "mini":
            # ミニモードでは少数のシンボルのみ
            test_symbols = symbol_universe[:5] if len(symbol_universe) >= 5 else symbol_universe
        else:
            test_symbols = symbol_universe

        # システムの取引方向を取得
        side = "short" if system_name in ["system2", "system6", "system7"] else "long"

        # ダミーデータを作成
        signals = []
        for i, symbol in enumerate(test_symbols):
            signals.append(
                {
                    "symbol": symbol,
                    "system": system_name,
                    "side": side,
                    "entry_price": 100.0 + i,  # ダミー価格
                    "stop_price": 90.0 + i if side == "long" else 110.0 + i,
                    "score": 0.5 + (i * 0.1),  # ダミースコア
                    "atr": 2.0 + (i * 0.1),  # ダミーATR
                }
            )

        if signals:
            per_system[system_name] = pd.DataFrame(signals)
            logger.info(f"✅ {system_name}: {len(signals)}件のテストシグナル生成")
        else:
            per_system[system_name] = pd.DataFrame()
            logger.info(f"⚠️ {system_name}: シグナルなし")

    return per_system
    """候補の詳細分析"""
    logger.info("=" * 50)
    logger.info("🔍 候補詳細分析")
    logger.info("=" * 50)

    total_candidates = 0

    for system_name, df in per_system.items():
        if df is None or df.empty:
            logger.info(f"❌ {system_name}: 候補なし")
            continue

        count = len(df)
        total_candidates += count
        logger.info(f"✅ {system_name}: {count}件の候補")

        # データ型と必須列の確認
        required_cols = ["symbol", "side", "score"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ {system_name}: 必須列が不足 - {missing_cols}")
            continue

        # side列の値を確認
        side_values = df["side"].unique() if "side" in df.columns else []
        logger.debug(f"   📊 side値: {side_values}")

        # score列の統計
        if "score" in df.columns:
            score_stats = df["score"].describe()
            nan_count = df["score"].isna().sum()
            logger.debug(f"   📈 score統計: mean={score_stats['mean']:.3f}, NaN={nan_count}件")

        # ATRとCloseの確認（ポジションサイズ計算に必要）
        atr_col = None
        close_col = None
        for col in df.columns:
            if "atr" in col.lower():
                atr_col = col
            if "close" in col.lower():
                close_col = col

        if atr_col:
            atr_stats = df[atr_col].describe()
            atr_nan_count = df[atr_col].isna().sum()
            logger.debug(f"   💹 {atr_col}: mean={atr_stats['mean']:.3f}, NaN={atr_nan_count}件")
        else:
            logger.warning(f"   ⚠️ {system_name}: ATR列が見つかりません")

        if close_col:
            close_stats = df[close_col].describe()
            close_nan_count = df[close_col].isna().sum()
            logger.debug(
                f"   💰 {close_col}: mean=${close_stats['mean']:.2f}, NaN={close_nan_count}件"
            )
        else:
            logger.warning(f"   ⚠️ {system_name}: Close列が見つかりません")

        if verbose and count > 0:
            # サンプル表示
            sample = df.head(3)
            logger.debug(f"   📋 {system_name} サンプル:")
            for _, row in sample.iterrows():
                symbol = row["symbol"]
                side = row.get("side", "N/A")
                score = row.get("score", "N/A")
                atr = row.get(atr_col, "N/A") if atr_col else "N/A"
                close = row.get(close_col, "N/A") if close_col else "N/A"
                logger.debug(
                    f"      {symbol}: side={side}, score={score}, atr={atr}, close={close}"
                )

    logger.info(f"📊 総候補数: {total_candidates}件")


def check_symbol_system_map_compatibility(
    per_system: Dict[str, pd.DataFrame], symbol_system_map: Dict[str, List[str]]
) -> None:
    """symbol_system_mapとの互換性確認"""
    logger.info("=" * 50)
    logger.info("🗺️ symbol_system_map互換性確認")
    logger.info("=" * 50)

    logger.info(f"マップ登録銘柄数: {len(symbol_system_map)}銘柄")

    for system_name, df in per_system.items():
        if df is None or df.empty:
            continue

        system_key = system_name.lower()
        blocked_symbols = []
        allowed_symbols = []
        unmapped_symbols = []

        for _, row in df.iterrows():
            symbol = row["symbol"]
            allowed_systems = symbol_system_map.get(symbol, [])

            if not allowed_systems:  # マップに登録されていない
                unmapped_symbols.append(symbol)
            elif system_key not in allowed_systems:
                blocked_symbols.append(symbol)
            else:
                allowed_symbols.append(symbol)

        logger.info(f"{system_name}:")
        logger.info(f"  ✅ 許可: {len(allowed_symbols)}件")
        logger.info(f"  ❌ ブロック: {len(blocked_symbols)}件")
        logger.info(f"  ❓ 未登録: {len(unmapped_symbols)}件")

        if blocked_symbols:
            logger.warning(f"  🚫 ブロックされた銘柄: {blocked_symbols[:5]}")
        if unmapped_symbols:
            logger.warning(f"  📋 未登録銘柄: {unmapped_symbols[:5]}")


def simulate_position_size_calculation(
    per_system: Dict[str, pd.DataFrame], strategies: Dict[str, Any]
) -> None:
    """ポジションサイズ計算のシミュレーション"""
    logger.info("=" * 50)
    logger.info("💰 ポジションサイズ計算シミュレーション")
    logger.info("=" * 50)

    test_budget = 100000.0  # $100,000のテスト予算

    for system_name, df in per_system.items():
        if df is None or df.empty:
            continue

        logger.info(f"--- {system_name} ---")

        strategy = strategies.get(system_name)
        if strategy is None:
            logger.error("❌ 戦略インスタンスが見つかりません")
            continue

        # ポジションサイズ計算関数の確認
        calc_fn = getattr(strategy, "calculate_position_size", None)
        if not callable(calc_fn):
            logger.error("❌ calculate_position_size関数が見つかりません")
            continue

        # 各候補でポジションサイズ計算を試行
        success_count = 0
        error_count = 0
        zero_size_count = 0

        for i, (_, row) in enumerate(df.head(5).iterrows()):  # 上位5件で確認
            symbol = row["symbol"]

            try:
                size_result = calc_fn(row, test_budget)

                if size_result is None or size_result <= 0:
                    zero_size_count += 1
                    logger.debug(f"  💸 {symbol}: 無効なポジションサイズ {size_result}")
                else:
                    success_count += 1
                    percentage = (size_result / test_budget) * 100
                    logger.debug(f"  💵 {symbol}: ${size_result:.0f} ({percentage:.1f}%)")

            except Exception as e:
                error_count += 1
                logger.debug(f"  ❌ {symbol}: 計算エラー {e}")

        logger.info(
            f"  結果: 成功={success_count}, ゼロサイズ={zero_size_count}, エラー={error_count}"
        )


def trace_allocation_step_by_step(
    per_system: Dict[str, pd.DataFrame], strategies: Dict[str, Any], verbose: bool = False
) -> None:
    """配分プロセスのステップバイステップトレース"""
    logger.info("=" * 50)
    logger.info("🚀 配分プロセス詳細トレース")
    logger.info("=" * 50)

    # 環境設定の確認
    env = get_env_config()
    logger.info("環境設定:")
    logger.info(f"  COMPACT_LOGS: {env.compact_logs}")
    logger.info("  DEBUG系フラグ: チェック中...")

    # symbol_system_mapの読み込み
    symbol_system_map = load_symbol_system_map()
    logger.info(f"symbol_system_map: {len(symbol_system_map)}銘柄登録済み")

    # 最終配分の実行（デバッグフラグ有効）
    import os

    old_debug = os.environ.get("ALLOCATION_DEBUG")
    os.environ["ALLOCATION_DEBUG"] = "1"

    try:
        logger.info("🎯 最終配分実行中...")

        final_df, summary = finalize_allocation(
            per_system=per_system,
            strategies=strategies,
            symbol_system_map=symbol_system_map,
            long_allocations=DEFAULT_LONG_ALLOCATIONS,
            short_allocations=DEFAULT_SHORT_ALLOCATIONS,
        )

        # 結果の詳細分析
        logger.info("=" * 30)
        logger.info("📋 最終配分結果")
        logger.info("=" * 30)
        logger.info(f"モード: {summary.mode}")
        logger.info(f"最終件数: {summary.final_counts}")

        if hasattr(summary, "slot_allocation"):
            logger.info(f"スロット配分: {summary.slot_allocation}")
        if hasattr(summary, "budgets"):
            logger.info(f"予算配分: {summary.budgets}")

        if final_df.empty:
            logger.error("❌ 最終候補が0件 - 配分ロジックで全て除外されました")
            logger.error("🔍 詳細な原因分析が必要です")
        else:
            logger.info(f"✅ 最終候補: {len(final_df)}件")
            if verbose:
                logger.info("最終候補リスト:")
                for _, row in final_df.head(10).iterrows():
                    logger.info(
                        f"  {row['symbol']} ({row['system']}, {row['side']}, score={row.get('score', 'N/A')})"
                    )

    finally:
        # 環境変数を元に戻す
        if old_debug is not None:
            os.environ["ALLOCATION_DEBUG"] = old_debug
        else:
            if "ALLOCATION_DEBUG" in os.environ:
                del os.environ["ALLOCATION_DEBUG"]


def validate_data_consistency() -> None:
    """データ整合性の検証"""
    logger.info("=" * 50)
    logger.info("🔍 データ整合性検証")
    logger.info("=" * 50)

    # キャッシュマネージャーの日付確認
    cache_manager = CacheManager()

    # rolling キャッシュの最新日付を確認
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    valid_count = 0

    for symbol in test_symbols:
        try:
            df = cache_manager.load_rolling(symbol)
            if not df.empty:
                latest_date = df.index[-1].strftime("%Y-%m-%d")
                row_count = len(df)
                logger.debug(f"✅ {symbol}: 最新={latest_date}, 行数={row_count}")
                valid_count += 1
            else:
                logger.warning(f"❌ {symbol}: rolling データが空")
        except Exception as e:
            logger.warning(f"❌ {symbol}: rolling データ読み込みエラー {e}")

    logger.info(f"キャッシュ検証: {valid_count}/{len(test_symbols)}銘柄が有効")


def generate_debug_report(per_system: Dict[str, pd.DataFrame], final_result: Any = None) -> None:
    """デバッグレポートの生成"""
    logger.info("=" * 50)
    logger.info("📊 デバッグレポート生成")
    logger.info("=" * 50)

    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "summary": {
            "total_systems": len(per_system),
            "systems_with_candidates": len(
                [df for df in per_system.values() if df is not None and not df.empty]
            ),
            "total_candidates": sum(
                len(df) for df in per_system.values() if df is not None and not df.empty
            ),
            "final_allocations": (
                0
                if final_result is None
                else len(final_result[0]) if hasattr(final_result[0], "__len__") else 0
            ),
        },
        "per_system": {},
    }

    for system_name, df in per_system.items():
        if df is not None and not df.empty:
            report["per_system"][system_name] = {
                "candidate_count": len(df),
                "has_required_columns": all(
                    col in df.columns for col in ["symbol", "side", "score"]
                ),
                "unique_symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
                "side_distribution": (
                    df["side"].value_counts().to_dict() if "side" in df.columns else {}
                ),
            }
        else:
            report["per_system"][system_name] = {
                "candidate_count": 0,
                "has_required_columns": False,
                "unique_symbols": 0,
                "side_distribution": {},
            }

    # レポートをファイルに保存
    settings = get_settings()
    report_path = Path(settings.project_root) / "results_csv_test" / "debug_allocation_report.json"
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"📄 デバッグレポート保存: {report_path}")


def main():
    """簡易テスト用メイン関数"""
    parser = argparse.ArgumentParser(description="配分ロジック簡易デバッグ")
    parser.add_argument("--test-mode", default="mini", help="テストモード")
    parser.add_argument("--verbose", action="store_true", help="詳細ログ")

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info("🚀 配分デバッグ（簡易版）開始")

    # デバッグモードを有効化
    import os

    os.environ["ALLOCATION_DEBUG"] = "1"
    logger.info("🐛 ALLOCATION_DEBUG=1 設定完了")

    # 戦略作成
    strategies = create_strategies()
    logger.info(f"✅ 戦略作成完了: {list(strategies.keys())}")

    # シンボル一覧取得
    settings = get_settings(create_dirs=False)
    symbol_universe = build_symbol_universe_from_settings(settings)
    logger.info(f"✅ シンボル一覧: {len(symbol_universe)}件")

    # テストシグナル生成
    per_system = generate_simple_test_signals(strategies, symbol_universe, args.test_mode)

    # TRDlist状況確認
    total_candidates = sum(len(df) for df in per_system.values() if not df.empty)
    logger.info(f"📊 TRDlist総件数: {total_candidates}件")

    # 配分実行テスト
    logger.info("\n🎯 配分プロセス開始")

    # デフォルト配分設定で実行
    try:
        final_df = finalize_allocation(
            per_system,
            capital_long=100000,  # $100k
            capital_short=100000,  # $100k
            positions_long=10,
            positions_short=10,
        )

        entry_count = len(final_df) if final_df is not None and not final_df.empty else 0
        logger.info(f"🎯 Entry最終件数: {entry_count}件")

        if entry_count > 0:
            logger.info("✅ 成功: TRDlist → Entry変換完了")
            if args.verbose and not final_df.empty:
                logger.info("\n📋 Entry詳細:")
                for _, row in final_df.head(10).iterrows():
                    logger.info(f"  {row.get('symbol', 'N/A')} ({row.get('system', 'N/A')})")
        else:
            logger.warning("⚠️ 問題: TRDlistあるが、Entry 0件")

    except Exception as e:
        logger.error(f"❌ 配分プロセスエラー: {e}")
        import traceback

        logger.error(traceback.format_exc())

    logger.info("\n✅ 簡易デバッグ完了")


if __name__ == "__main__":
    main()
