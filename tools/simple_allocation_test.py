#!/usr/bin/env python
# ruff: noqa: E402
"""簡単な配分テスト - TRDlist 10件→Entry 0件問題の検証"""

from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
import os

import pandas as pd

from core.final_allocation import (  # noqa: E402
    finalize_allocation,
    load_symbol_system_map,
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_test_signals() -> dict:
    """テスト用のシグナルデータを生成"""

    # 各システムのテストシグナルを作成
    per_system = {
        "system1": pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "side": "long",
                    "entry_price": 150.0,
                    "stop_price": 140.0,
                    "score": 0.8,
                    "atr": 3.0,
                },
                {
                    "symbol": "MSFT",
                    "side": "long",
                    "entry_price": 300.0,
                    "stop_price": 285.0,
                    "score": 0.7,
                    "atr": 4.0,
                },
            ]
        ),
        "system2": pd.DataFrame(
            [
                {
                    "symbol": "TSLA",
                    "side": "short",
                    "entry_price": 200.0,
                    "stop_price": 210.0,
                    "score": 0.9,
                    "atr": 8.0,
                },
            ]
        ),
        "system3": pd.DataFrame(
            [
                {
                    "symbol": "GOOGL",
                    "side": "long",
                    "entry_price": 2500.0,
                    "stop_price": 2400.0,
                    "score": 0.6,
                    "atr": 30.0,
                },
                {
                    "symbol": "NVDA",
                    "side": "long",
                    "entry_price": 400.0,
                    "stop_price": 380.0,
                    "score": 0.75,
                    "atr": 15.0,
                },
            ]
        ),
        "system4": pd.DataFrame(
            [
                {
                    "symbol": "META",
                    "side": "long",
                    "entry_price": 350.0,
                    "stop_price": 330.0,
                    "score": 0.65,
                    "atr": 12.0,
                },
            ]
        ),
        "system5": pd.DataFrame(
            [
                {
                    "symbol": "AMZN",
                    "side": "long",
                    "entry_price": 3000.0,
                    "stop_price": 2850.0,
                    "score": 0.85,
                    "atr": 50.0,
                },
            ]
        ),
        "system6": pd.DataFrame(
            [
                {
                    "symbol": "NFLX",
                    "side": "short",
                    "entry_price": 400.0,
                    "stop_price": 420.0,
                    "score": 0.7,
                    "atr": 20.0,
                },
            ]
        ),
        "system7": pd.DataFrame(
            [
                {
                    "symbol": "SPY",
                    "side": "short",
                    "entry_price": 450.0,
                    "stop_price": 460.0,
                    "score": 0.6,
                    "atr": 10.0,
                },
            ]
        ),
    }

    # 各DataFrameにsystemカラムを追加
    for system_name, df in per_system.items():
        if not df.empty:
            df["system"] = system_name

    return per_system


def main():
    """簡単な配分テスト実行"""

    logger.info("🚀 簡単な配分テスト開始")

    # デバッグモード有効化
    os.environ["ALLOCATION_DEBUG"] = "1"
    logger.info("🐛 ALLOCATION_DEBUG=1 設定")

    # テストシグナル作成
    per_system = create_test_signals()

    # TRDlist件数確認
    total_candidates = sum(len(df) for df in per_system.values() if not df.empty)
    logger.info(f"📊 TRDlist総件数: {total_candidates}件")

    for system_name, df in per_system.items():
        count = len(df) if not df.empty else 0
        logger.info(f"  {system_name}: {count}件")

    # 戦略インスタンスを作成
    from strategies.system1_strategy import System1Strategy
    from strategies.system2_strategy import System2Strategy
    from strategies.system3_strategy import System3Strategy
    from strategies.system4_strategy import System4Strategy
    from strategies.system5_strategy import System5Strategy
    from strategies.system6_strategy import System6Strategy
    from strategies.system7_strategy import System7Strategy

    strategy_objs = [
        System1Strategy(),
        System2Strategy(),
        System3Strategy(),
        System4Strategy(),
        System5Strategy(),
        System6Strategy(),
        System7Strategy(),
    ]
    strategies = {getattr(s, "SYSTEM_NAME", "").lower(): s for s in strategy_objs}
    logger.info(f"✅ 戦略インスタンス作成: {list(strategies.keys())}")

    # 配分実行
    logger.info("\n🎯 配分プロセス実行")

    try:
        try:
            symbol_system_map = load_symbol_system_map()
        except Exception:
            symbol_system_map = {}

        final_result = finalize_allocation(
            per_system,
            strategies=strategies,  # 戦略インスタンスを渡す
            symbol_system_map=symbol_system_map,
            capital_long=100000,  # $100k
            capital_short=50000,  # $50k
            slots_long=5,  # 5スロット
            slots_short=3,  # 3スロット
        )

        # 結果確認（finalize_allocationはtupleを返す）
        final_df, allocation_summary = final_result

        if final_df is not None and not final_df.empty:
            entry_count = len(final_df)
            logger.info(f"✅ Entry最終件数: {entry_count}件")

            # 詳細表示
            logger.info("\n📋 Entry詳細:")
            for i, row in final_df.iterrows():
                symbol = row.get("symbol", "N/A")
                system = row.get("system", "N/A")
                side = row.get("side", "N/A")
                position_size = row.get("position_size", 0)
                logger.info(f"  {symbol} ({system}, {side}) - ${position_size:.0f}")

        else:
            logger.warning("❌ Entry 0件 - 問題確認が必要")

    except Exception as e:
        logger.error(f"❌ 配分エラー: {e}")
        import traceback

        logger.error(traceback.format_exc())

    logger.info("\n✅ テスト完了")


if __name__ == "__main__":
    main()
