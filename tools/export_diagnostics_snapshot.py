"""Export diagnostics snapshot to JSON for comparison and analysis.

このツールは、Mini パイプライン実行後の diagnostics を JSON 形式で保存します。
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from common.system_diagnostics import get_diagnostics_with_fallback

logger = logging.getLogger(__name__)


def export_diagnostics_snapshot(
    allocation_summary: Any,
    output_path: Path,
) -> None:
    """全システムの diagnostics をスナップショット JSON として保存。

    Args:
        allocation_summary: compute_today_signals() が返す allocation_summary
        output_path: 出力先 JSON パス（例: results_csv_test/diagnostics_snapshot.json）
    """
    systems_list: list[dict[str, Any]] = []
    snapshot: dict[str, Any] = {
        "export_date": datetime.now().isoformat(),
        "systems": systems_list,
    }

    # allocation_summary がオブジェクトの場合、属性を取得
    if hasattr(allocation_summary, "__dict__"):
        # AllocationSummary オブジェクトの場合
        systems_data = getattr(allocation_summary, "systems", {})
    elif isinstance(allocation_summary, dict):
        # 辞書の場合
        systems_data = allocation_summary
    else:
        logger.warning(
            f"Unexpected allocation_summary type: {type(allocation_summary)}"
        )
        systems_data = {}

    # 各システムの diagnostics を収集
    for system_id, system_info in systems_data.items():
        if isinstance(system_info, dict):
            diag = system_info.get("diagnostics", {})
            candidates = system_info.get("candidates", [])
        elif hasattr(system_info, "__dict__"):
            # オブジェクトの場合
            diag = getattr(system_info, "diagnostics", {})
            candidates = getattr(system_info, "candidates", [])
        else:
            logger.warning(
                f"Unexpected system_info type for {system_id}: {type(system_info)}"
            )
            diag = {}
            candidates = []

        # フォールバック適用（標準化キー）
        diag_safe = get_diagnostics_with_fallback(diag, system_id)
        # 追加の生診断（標準化されない任意フィールド）
        try:
            extras = (
                {k: v for k, v in (diag or {}).items() if k not in diag_safe}
                if isinstance(diag, dict)
                else {}
            )
        except Exception:
            extras = {}

        systems_list.append(
            {
                "system_id": system_id,
                "diagnostics": diag_safe,
                **({"diagnostics_extra": extras} if extras else {}),
                "candidate_count": (
                    len(candidates) if isinstance(candidates, (list, tuple)) else 0
                ),
            }
        )

    # 出力先ディレクトリが存在しない場合は作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    logger.info(f"Diagnostics snapshot exported to {output_path}")


def main():
    """CLI エントリーポイント（スタンドアロン実行用）。"""
    parser = argparse.ArgumentParser(description="Export diagnostics snapshot to JSON")
    parser.add_argument(
        "--input",
        type=Path,
        help="Input allocation summary JSON (if available)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output snapshot JSON path",
    )
    args = parser.parse_args()

    # スタンドアロン実行の場合は入力 JSON を読み込み
    if args.input and args.input.exists():
        with open(args.input, "r", encoding="utf-8") as f:
            allocation_summary = json.load(f)
    else:
        logger.warning("No input file provided, creating empty snapshot")
        allocation_summary = {}

    export_diagnostics_snapshot(allocation_summary, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
