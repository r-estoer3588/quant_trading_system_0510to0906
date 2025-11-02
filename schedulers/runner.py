"""Simple scheduler runner using YAML scheduler config.

Supports a minimal subset of cron: "m h * * d".
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

from common.logging_utils import setup_logging
from config.settings import get_settings

Field = tuple[int, ...] | Literal["*"]


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

        compute_today_signals(None, save_csv=True, notify=False)
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
    for job in jobs:
        func = TASKS.get(job.task)
        if not func:
            logging.warning(f"未知のタスク '{job.task}' はスキップします。")
            continue
        try:
            pred = parse_cron(job.cron)
        except Exception as e:
            logging.error(f"cron 解析失敗 ({job.name}): {e}")
            continue
        compiled.append((job.name, pred, func))
        logging.info(f"登録: {job.name} ({job.cron}) -> {job.task}")

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
