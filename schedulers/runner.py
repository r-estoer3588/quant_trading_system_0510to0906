"""Simple scheduler runner using YAML scheduler config.

Supports a minimal subset of cron: "m h * * d".
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import logging
import sys
import time
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

from common.logging_utils import setup_logging
from config.settings import get_settings

Field = tuple[int, ...] | Literal["*"]


def is_us_dst(dt: datetime | None = None) -> bool:
    """米国がDST(夏時間)中かどうかを判定。

    Args:
        dt: 判定する日時(None の場合は現在時刻)

    Returns:
        米国が夏時間中なら True
    """
    if dt is None:
        dt = datetime.now(ZoneInfo("America/New_York"))
    ny_tz = ZoneInfo("America/New_York")
    try:
        ny_time = dt.astimezone(ny_tz)
        # DSTオフセットが0でなければ夏時間
        return ny_time.dst() is not None and ny_time.dst().total_seconds() > 0
    except Exception:
        return False


def adjust_cron_for_dst(cron: str, is_dst: bool) -> str:
    """DSTの場合、時刻を1時間早める。

    Args:
        cron: cron文字列 (例: "30 23 * * 1-5")
        is_dst: 夏時間中かどうか

    Returns:
        調整後のcron文字列
    """
    if not is_dst:
        return cron
    parts = cron.split()
    if len(parts) != 5:
        return cron
    try:
        hour = int(parts[1])
        parts[1] = str((hour - 1) % 24)
        return " ".join(parts)
    except (ValueError, IndexError):
        return cron


def parse_cron(cron: str):
    """Parse a very small subset of cron: "m h * * dow".
    - minute: 0-59 or "*"
    - hour: 0-23 or "*"
    - dow: 0-7, list (e.g., 1-5), comma-separated, or "*". 0/7 = Sunday.
    Returns a predicate function (dt: datetime) -> bool
    """
    parts = cron.split()
    if len(parts) != 5:
        raise ValueError(f"Unsupported cron format: {cron}")
    m_s, h_s, _, _, d_s = parts

    def parse_field(val: str, min_v: int, max_v: int) -> Field:
        if val.strip() == "*":
            return "*"
        vals = set()
        for tok in val.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-", 1)
                a_i, b_i = int(a), int(b)
                vals.update(range(a_i, b_i + 1))
            else:
                vals.add(int(tok))
        return tuple(sorted(v for v in vals if min_v <= v <= max_v))

    m_val: Field = parse_field(m_s, 0, 59)
    h_val: Field = parse_field(h_s, 0, 23)
    d_val: Field = parse_field(d_s, 0, 7)

    def _match(value: int, allowed: Field) -> bool:
        if allowed == "*":
            return True
        return value in allowed

    def pred(dt: datetime) -> bool:
        minute = dt.minute
        hour = dt.hour
        dow = dt.weekday() + 1  # Monday=1 ... Sunday=7
        dow = 0 if dow == 7 else dow  # accept 0 as Sunday
        if not _match(minute, m_val):
            return False
        if not _match(hour, h_val):
            return False
        if not _match(dow, d_val):
            return False
        return True

    return pred


def task_cache_daily_data():
    import scripts.cache_daily_data as cache_daily_data

    cache_daily_data._cli_main()


def task_notify_signals():
    try:
        from tools.notify_signals import notify_signals
    except Exception:
        logging.warning(
            "notify_signals タスクが未実装です。tools/notify_signals.py を用意してください。"
        )
        return
    notify_signals()


def task_notify_metrics():
    try:
        from tools.notify_metrics import notify_metrics
    except Exception:
        logging.warning(
            "notify_metrics タスクが未実装です。tools/notify_metrics.py を用意してください。"
        )
        return
    notify_metrics()


def task_build_metrics_report():
    try:
        from tools.build_metrics_report import build_metrics_report

        build_metrics_report()
    except Exception:
        logging.exception("build_metrics_report タスクが失敗しました")


def task_daily_run():
    try:
        from scripts.daily_run import main as daily_main

        exit_code = daily_main()
        if exit_code != 0:
            logging.error("daily_run タスクが異常終了しました (code=%s)", exit_code)
    except Exception:
        logging.exception("daily_run タスクが失敗しました")


def task_run_today_signals():
    try:
        from scripts.run_all_systems_today import compute_today_signals
        from tools.notify_signals import send_signal_notification

        final_df, _ = compute_today_signals(None, save_csv=True, notify=False)

        # Slack通知を送信
        if final_df is not None and not final_df.empty:
            send_signal_notification(final_df)
            logging.info(f"シグナル生成完了: {len(final_df)}件 (Slack通知送信済み)")
        else:
            logging.info("シグナル生成完了: 0件")
    except Exception:
        logging.exception("run_today_signals タスクが失敗しました")


def task_bulk_last_day():
    try:
        from scripts.update_from_bulk_last_day import main as bulk_update

        bulk_update()
    except Exception:
        logging.exception("bulk_last_day タスクが失敗しました")


def task_update_tickers():
    try:
        from scripts.tickers_loader import update_ticker_list

        update_ticker_list()
    except Exception:
        logging.exception("update_tickers タスクが失敗しました")


def task_update_trailing_stops():
    try:
        from scripts.update_trailing_stops import update_trailing_stops

        update_trailing_stops()
    except Exception:
        logging.exception("update_trailing_stops タスクが失敗しました")


def task_precompute_shared_indicators():
    try:
        from tools.precompute_shared_indicators import main as warmup

        warmup()
    except Exception:
        logging.exception("precompute_shared_indicators タスクが失敗しました")


def task_run_auto_rule():
    """自動ルールに基づいてポジションをエグジット"""
    try:
        from scripts.run_auto_rule import main as run_auto_rule_main

        # paper=True でペーパートレーディング、dry_run=False で実際に注文送信
        import sys

        sys.argv = ["run_auto_rule", "--paper"]  # 本番の場合は --paper を削除
        run_auto_rule_main()
    except Exception:
        logging.exception("run_auto_rule タスクが失敗しました")
        _notify_task_error("run_auto_rule")


def task_daily_summary_report():
    """日次サマリーレポートをSlackに送信"""
    try:
        from scripts.daily_summary_report import send_report

        send_report(paper=True)
    except Exception:
        logging.exception("daily_summary_report タスクが失敗しました")
        _notify_task_error("daily_summary_report")


def task_sync_positions():
    """Alpacaポジションをトラッカーに同期"""
    try:
        from scripts.sync_positions_to_tracker import sync_positions

        sync_positions(paper=True)
    except Exception:
        logging.exception("sync_positions タスクが失敗しました")
        _notify_task_error("sync_positions")


def _notify_task_error(task_name: str):
    """タスク失敗をSlack通知"""
    try:
        from common.error_notifier import notify_error
        import traceback

        notify_error(
            task_name, f"タスク {task_name} が失敗しました", traceback.format_exc()
        )
    except Exception:
        logging.exception("エラー通知の送信に失敗しました")


def task_weekly_summary_report():
    """週次サマリーレポートをSlackに送信"""
    try:
        from scripts.weekly_summary_report import send_weekly_report

        send_weekly_report(paper=True)
    except Exception:
        logging.exception("weekly_summary_report タスクが失敗しました")
        _notify_task_error("weekly_summary_report")


def task_monthly_detailed_report():
    """月次詳細レポートをExcel/CSVで生成"""
    try:
        from scripts.monthly_detailed_report import generate_monthly_report
        from scripts.monthly_detailed_report import send_notification

        report_files = generate_monthly_report(paper=True)
        send_notification(report_files, paper=True)
    except Exception:
        logging.exception("monthly_detailed_report タスクが失敗しました")
        _notify_task_error("monthly_detailed_report")


def task_monitor_portfolio():
    """ポートフォリオPnL監視・アラート"""
    try:
        from scripts.monitor_portfolio import check_and_alert

        check_and_alert(paper=True)
    except Exception:
        logging.exception("monitor_portfolio タスクが失敗しました")
        _notify_task_error("monitor_portfolio")


TASKS: dict[str, Callable[[], None]] = {
    "cache_daily_data": task_cache_daily_data,
    "warm_cache": task_cache_daily_data,
    "notify_signals": task_notify_signals,
    "run_today_signals": task_run_today_signals,
    "bulk_last_day": task_bulk_last_day,
    "update_tickers": task_update_tickers,
    "update_trailing_stops": task_update_trailing_stops,
    "precompute_shared_indicators": task_precompute_shared_indicators,
    "notify_metrics": task_notify_metrics,
    "build_metrics_report": task_build_metrics_report,
    "daily_run": task_daily_run,
    "run_auto_rule": task_run_auto_rule,
    "daily_summary_report": task_daily_summary_report,
    "sync_positions": task_sync_positions,
    "weekly_summary_report": task_weekly_summary_report,
    "monthly_detailed_report": task_monthly_detailed_report,
    "monitor_portfolio": task_monitor_portfolio,
}


def main():
    settings = get_settings(create_dirs=True)
    setup_logging(cast(Any, settings))
    tz_name = settings.scheduler.timezone or "America/New_York"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        logging.warning("未知のタイムゾーン '%s'、ローカル時刻を使用します", tz_name)
        tz = None
    jobs = settings.scheduler.jobs
    if not jobs:
        logging.warning(
            "scheduler.jobs が空です。config/config.yaml を確認してください。"
        )
        return 0

    compiled = []
    current_dst = is_us_dst()
    if current_dst:
        logging.info(
            "📍 米国は現在サマータイム(DST)中です。対象タスクは1時間早く実行されます。"
        )
    for job in jobs:
        func = TASKS.get(job.task)
        if not func:
            logging.warning(f"未知のタスク '{job.task}' はスキップします。")
            continue
        try:
            # dst_aware が True の場合、夏時間中は1時間早く実行
            cron_to_use = job.cron
            dst_aware = getattr(job, "dst_aware", False)
            if dst_aware and current_dst:
                cron_to_use = adjust_cron_for_dst(job.cron, True)
            pred = parse_cron(cron_to_use)
        except Exception as e:
            logging.error(f"cron 解析失敗 ({job.name}): {e}")
            continue
        display_cron = cron_to_use if cron_to_use != job.cron else job.cron
        dst_marker = " [DST調整済]" if (dst_aware and current_dst) else ""
        compiled.append((job.name, pred, func))
        logging.info(f"登録: {job.name} ({display_cron}){dst_marker} -> {job.task}")

    # 簡易ポーリングループ（30秒）
    logging.info("スケジューラー開始")
    last_minute = None
    try:
        while True:
            now = datetime.now(tz) if tz is not None else datetime.now()
            # 1分に1回だけ起動判定
            if last_minute != (now.year, now.month, now.day, now.hour, now.minute):
                last_minute = (now.year, now.month, now.day, now.hour, now.minute)
                for name, pred, func in compiled:
                    try:
                        if pred(now):
                            logging.info(f"起動: {name}")
                            func()
                    except Exception:
                        logging.exception(f"タスク失敗: {name}")
            time.sleep(30)
    except KeyboardInterrupt:
        logging.info("スケジューラー停止")
        return 0


if __name__ == "__main__":
    sys.exit(main())
