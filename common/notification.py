"""Zero TRD escalation notification utilities.

このモジュールは、全システムで候補がゼロになった際の通知機能を提供します。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def notify_zero_trd_all_systems(
    ctx: Any,
    final_df: Any,
) -> None:
    """全システムで候補ゼロの際にエスカレーション通知を送る。

    Args:
        ctx: TodayRunContext（通知設定を含む）
        final_df: compute_today_signals() が返す最終 DataFrame
    """
    # 判定基準を変更: per_system ベースでトレード候補の有無を判定
    # final_df は配分・フィルター後の結果なので、TRDlist 段階では存在する場合がある
    per_system = getattr(ctx, "per_system_frames", {})
    has_any_candidates = False
    if isinstance(per_system, dict):
        for df in per_system.values():
            if df is not None and (hasattr(df, "empty") and not df.empty):
                has_any_candidates = True
                break

    # per_system に候補がある場合はアラート不要（配分段階でフィルターされた可能性）
    if has_any_candidates:
        return

    # final_df が空の場合（全システムで候補ゼロ）
    if final_df is None or (hasattr(final_df, "empty") and final_df.empty):
        # モードとテスト設定を取得
        test_mode = getattr(ctx, "test_mode", None)
        today = getattr(ctx, "today", None)
        current_date = str(today) if today is not None else "unknown"
        run_id = getattr(ctx, "run_id", "unknown")

        # 警告メッセージを作成
        message = (
            "⚠️ Zero TRD Alert: All systems returned zero candidates.\n"
            f"Mode: {test_mode or 'production'}\n"
            f"Date: {current_date}\n"
            f"Run ID: {run_id}\n"
            "Action: Check filters, data freshness, and indicator calculation."
        )

        # ログに警告を記録
        logger.warning(message)

        # 通知が有効な場合に送信
        notify_enabled = getattr(ctx, "notify", False)
        if notify_enabled:
            # 既存の通知機能を使用（notifier が ctx に存在する場合）
            notifier = getattr(ctx, "notifier", None)
            if notifier is not None and hasattr(notifier, "send_message"):
                try:
                    notifier.send_message(
                        title="⚠️ Zero TRD Alert",
                        message=message,
                        level="warning",
                    )
                    logger.info("Zero TRD notification sent successfully")
                except Exception as e:
                    logger.error(f"Failed to send zero TRD notification: {e}")
            else:
                logger.debug(
                    "Notifier not available or does not support send_message, skipping notification"
                )
        else:
            logger.debug("Notifications disabled, skipping zero TRD alert")
