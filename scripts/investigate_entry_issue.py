"""
エントリー数が0になる問題の調査スクリプト
Copilot Instructions準拠版
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logging_utils import SystemLogger
from config.settings import get_settings

# ロガー設定
logger = logging.getLogger(__name__)
sys_logger = SystemLogger.create("investigate_entry", logger=logger)


def investigate_entry_issue(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    エントリー数0の問題を段階的に調査

    Returns:
        調査結果の辞書
    """
    settings = get_settings(create_dirs=True)

    # results_csv_testディレクトリを使用（最新のテスト実行結果を確認）
    results_dir = Path("results_csv_test")

    investigation_results = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "results_dir": str(results_dir),
        "findings": {},
    }

    sys_logger.info("エントリー数0問題の調査開始", results_dir=str(results_dir))
    print("=== エントリー数0問題の調査 ===")
    print(f"調査対象ディレクトリ: {results_dir}\n")

    # 1. per_system_*.featherの確認
    print("1. 各システムの候補データ確認:")
    per_system_data = {}
    system_findings = {}

    for sys_num in range(1, 8):
        system_name = f"system{sys_num}"
        feather_path = results_dir / f"per_system_{system_name}.feather"

        if feather_path.exists():
            try:
                df = pd.read_feather(feather_path)
                per_system_data[system_name] = df

                system_info = {
                    "candidate_count": len(df),
                    "columns": list(df.columns),
                    "has_entry_price": "entry_price" in df.columns,
                    "has_stop_price": "stop_price" in df.columns,
                }

                print(f"\n{system_name}:")
                print(f"  - 候補数: {len(df)}")
                print(f"  - カラム数: {len(df.columns)}")

                # entry_price/stop_priceの分析
                if system_info["has_entry_price"] and system_info["has_stop_price"]:
                    valid_entries = df["entry_price"].notna().sum()
                    valid_stops = df["stop_price"].notna().sum()
                    system_info["valid_entry_prices"] = int(valid_entries)
                    system_info["valid_stop_prices"] = int(valid_stops)

                    print(f"  - entry_price存在: ✓ (有効: {valid_entries}/{len(df)})")
                    print(f"  - stop_price存在: ✓ (有効: {valid_stops}/{len(df)})")

                    # エントリー価格が無効な銘柄をサンプル表示
                    if valid_entries < len(df):
                        invalid_entries = df[df["entry_price"].isna()]
                        if "symbol" in invalid_entries.columns:
                            sample_symbols = list(invalid_entries["symbol"][:3])
                        else:
                            sample_symbols = list(invalid_entries.index[:3])
                        print(f"  - entry_price無効な銘柄例: {sample_symbols}")
                else:
                    entry_mark = "✓" if system_info["has_entry_price"] else "✗"
                    stop_mark = "✓" if system_info["has_stop_price"] else "✗"
                    print(f"  - entry_price存在: {entry_mark}")
                    print(f"  - stop_price存在: {stop_mark}")

                # attrsの確認（スキップ理由）
                if hasattr(df, "attrs"):
                    skip_counts = df.attrs.get("entry_skip_counts", {})
                    if skip_counts:
                        system_info["skip_reasons"] = skip_counts
                        print("  - スキップ理由:")
                        for reason, count in sorted(
                            skip_counts.items(), key=lambda x: x[1], reverse=True
                        ):
                            print(f"    * {reason}: {count}")

                    # スキップ詳細のサンプル
                    skip_details = df.attrs.get("entry_skip_details", [])
                    if skip_details:
                        system_info["skip_detail_samples"] = skip_details[:3]
                        print("  - スキップ詳細（最初の3件）:")
                        for detail in skip_details[:3]:
                            print(f"    * {detail}")

                system_findings[system_name] = system_info

            except Exception as e:
                sys_logger.error(
                    f"per_system_{system_name}.feather読込エラー", error=str(e)
                )
                print(f"  - エラー: {e}")
        else:
            print(f"\n{system_name}: ファイルが存在しません")

    investigation_results["findings"]["per_system_analysis"] = system_findings

    # 2. skip_summary_*.csvの確認
    print("\n\n2. スキップサマリーファイルの確認:")
    skip_summaries = {}

    for sys_num in range(1, 8):
        system_name = f"system{sys_num}"
        skip_summary_path = results_dir / f"skip_summary_{system_name}.csv"

        if skip_summary_path.exists():
            try:
                skip_df = pd.read_csv(skip_summary_path)
                skip_summaries[system_name] = skip_df.to_dict("records")

                print(f"\n{system_name} スキップサマリー:")
                if not skip_df.empty:
                    print(skip_df.to_string())
                else:
                    print("  （空のデータフレーム）")

            except Exception as e:
                sys_logger.error(
                    f"skip_summary_{system_name}.csv読込エラー", error=str(e)
                )
                print(f"  - エラー: {e}")

    investigation_results["findings"]["skip_summaries"] = skip_summaries

    # 3. symbol_system_map.jsonの確認
    print("\n\n3. symbol_system_map.jsonの確認:")
    map_path = Path("data/symbol_system_map.json")
    fixed_allocations = {}

    if map_path.exists():
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                symbol_map = json.load(f)

            # 各システムに固定された銘柄数を集計
            system_counts = {}
            for symbol, sys_name in symbol_map.items():
                sys_key = str(sys_name).lower()
                system_counts[sys_key] = system_counts.get(sys_key, 0) + 1

            fixed_allocations = system_counts
            print("固定銘柄数:")
            for sys_name, count in sorted(system_counts.items()):
                print(f"  - {sys_name}: {count}銘柄")

            # 固定銘柄と候補の重複を確認
            print("\n固定銘柄と候補の重複確認:")
            for system_name, df in per_system_data.items():
                if system_name in system_counts:
                    fixed_symbols = [
                        sym
                        for sym, sys_val in symbol_map.items()
                        if str(sys_val).lower() == system_name
                    ]
                    if "symbol" in df.columns:
                        candidate_symbols = set(df["symbol"])
                    else:
                        candidate_symbols = set(df.index)
                    overlap = candidate_symbols & set(fixed_symbols)
                    if overlap:
                        sample = list(overlap)[:3]
                        msg = f"  - {system_name}: {len(overlap)}銘柄が重複 (例: {sample})"
                        print(msg)

        except Exception as e:
            sys_logger.error("symbol_system_map.json読込エラー", error=str(e))
            print(f"  - エラー: {e}")
    else:
        print("  - ファイルが存在しません")

    investigation_results["findings"]["fixed_allocations"] = fixed_allocations

    # 4. final_df_*.csvの確認
    print("\n\n4. 最終配分データの確認:")
    final_df_info = {}

    final_csv_files = list(results_dir.glob("final_df_*.csv"))
    if final_csv_files:
        latest_final = sorted(final_csv_files)[-1]
        print(f"最新ファイル: {latest_final.name}")

        try:
            final_df = pd.read_csv(latest_final)
            final_df_info["total_entries"] = len(final_df)
            print(f"  - 総エントリー数: {len(final_df)}")

            if "system" in final_df.columns:
                system_counts = final_df["system"].value_counts()
                final_df_info["system_distribution"] = system_counts.to_dict()
                print("  - システム別エントリー数:")
                for sys_name, count in system_counts.items():
                    print(f"    * {sys_name}: {count}")
            else:
                print("  - 'system'カラムが見つかりません")

        except Exception as e:
            sys_logger.error("final_df読込エラー", error=str(e))
            print(f"  - エラー: {e}")
    else:
        print("  - final_df_*.csvファイルが見つかりません")

    investigation_results["findings"]["final_df"] = final_df_info

    # 5. 設定ファイルの確認
    print("\n\n5. 戦略設定の確認:")
    strategy_configs = {}

    try:
        for sys_num in range(1, 8):
            system_name = f"system{sys_num}"
            # デフォルト値を設定
            config_info = {
                "max_positions": 10,
                "allocation_mode": "slot",
            }

            # 設定から値を取得（存在する場合）
            try:
                if hasattr(settings, "strategies"):
                    strategy_config = getattr(settings.strategies, system_name, None)
                    if strategy_config:
                        if hasattr(strategy_config, "max_positions"):
                            config_info["max_positions"] = strategy_config.max_positions
                        if hasattr(strategy_config, "allocation_mode"):
                            mode = strategy_config.allocation_mode
                            config_info["allocation_mode"] = mode
            except Exception:
                pass

            strategy_configs[system_name] = config_info

            print(f"\n{system_name}:")
            for key, value in config_info.items():
                print(f"  - {key}: {value}")

    except Exception as e:
        sys_logger.error("設定読込エラー", error=str(e))
        print(f"  - エラー: {e}")

    investigation_results["findings"]["strategy_configs"] = strategy_configs

    # 6. diagnostics_snapshot_*.jsonの確認
    print("\n\n6. diagnostics_snapshotの確認:")
    diagnostics_info = {}

    snapshot_files = list(results_dir.glob("diagnostics_snapshot_*.json"))
    if snapshot_files:
        # 最新のファイルを使用
        latest_snapshot = sorted(snapshot_files)[-1]
        print(f"最新ファイル: {latest_snapshot.name}")

        try:
            with open(latest_snapshot, "r", encoding="utf-8") as f:
                diag = json.load(f)

            for system_name, system_diag in diag.items():
                if system_name.startswith("system"):
                    diagnostics_info[system_name] = system_diag
                    print(f"\n{system_name}:")
                    for key, value in system_diag.items():
                        print(f"  - {key}: {value}")

        except Exception as e:
            sys_logger.error("diagnostics_snapshot読込エラー", error=str(e))
            print(f"  - エラー: {e}")
    else:
        print("  - スナップショットファイルが見つかりません")

    investigation_results["findings"]["diagnostics"] = diagnostics_info

    # 7. 問題の診断
    print("\n\n=== 診断結果 ===")

    # 各段階での候補数の推移を確認
    print("\n候補数の推移:")
    issues_found = []

    for system_name in [f"system{i}" for i in range(1, 8)]:
        if system_name in system_findings:
            candidate_count = system_findings[system_name].get("candidate_count", 0)
            valid_entries = system_findings[system_name].get("valid_entry_prices", 0)
            final_dist = final_df_info.get("system_distribution", {})
            final_entries = final_dist.get(system_name, 0)

            flow = f"{system_name}: 候補{candidate_count} → 有効エントリー{valid_entries} → 最終配分{final_entries}"
            print(flow)

            # 問題を特定
            if candidate_count > 0 and valid_entries == 0:
                issue = f"{system_name}: エントリー価格計算で全て失敗"
                print(f"  ⚠️ {issue}")
                issues_found.append(issue)

                # スキップ理由を表示
                skip_reasons = system_findings[system_name].get("skip_reasons", {})
                if skip_reasons:
                    top_reason = max(skip_reasons.items(), key=lambda x: x[1])
                    print(f"     主な理由: {top_reason[0]} ({top_reason[1]}件)")

            elif valid_entries > 0 and final_entries == 0:
                issue = f"{system_name}: 配分段階で全て除外"
                print(f"  ⚠️ {issue}")
                issues_found.append(issue)

    investigation_results["findings"]["issues_found"] = issues_found

    # 結果をJSONファイルに保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"investigation_entry_issue_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(investigation_results, f, indent=2, ensure_ascii=False)

    print(f"\n\n調査結果を保存: {output_path}")
    sys_logger.info("調査完了", output_file=str(output_path))

    return investigation_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="エントリー数0問題の調査")
    parser.add_argument("--run-id", type=str, help="実行IDの指定")
    args = parser.parse_args()

    investigate_entry_issue(run_id=args.run_id)
