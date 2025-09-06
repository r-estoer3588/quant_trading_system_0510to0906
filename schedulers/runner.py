"""Simple scheduler runner using YAML scheduler config.

Supports a minimal subset of cron: "m h * * d" where d is 0-6 (Mon=1 .. Sun=0/7 accepted).
Runs a polling loop and triggers tasks when minute/hour/dow match.
"""

from __future__ import annotations

import sys
import time
import logging
from datetime import datetime
from typing import Callable, Dict, Iterable

from config.settings import get_settings
from common.logging_utils import setup_logging


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

    def parse_field(val: str, min_v: int, max_v: int) -> Iterable[int] | str:
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

    m_val = parse_field(m_s, 0, 59)
    h_val = parse_field(h_s, 0, 23)
    d_val = parse_field(d_s, 0, 7)

    def pred(dt: datetime) -> bool:
        minute = dt.minute
        hour = dt.hour
        dow = dt.weekday() + 1  # Monday=1 ... Sunday=7
        dow = 0 if dow == 7 else dow  # accept 0 as Sunday
        if m_val != "*" and minute not in m_val:
            return False
        if h_val != "*" and hour not in h_val:
            return False
        if d_val != "*" and dow not in d_val:
            return False
        return True

    return pred


def task_cache_daily_data():
    from scripts.cache_daily_data import warm_cache_default

    warm_cache_default()


def task_notify_signals():
    try:
        from tools.notify_signals import notify_signals
    except Exception:
        logging.warning("notify_signals タスクが未実装です。tools/notify_signals.py を用意してください。")
        return
    notify_signals()


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


TASKS: Dict[str, Callable[[], None]] = {
    "cache_daily_data": task_cache_daily_data,
    "warm_cache": task_cache_daily_data,
    "notify_signals": task_notify_signals,
    "run_today_signals": task_run_today_signals,
    "bulk_last_day": task_bulk_last_day,
    "update_tickers": task_update_tickers,
}


def main():
    settings = get_settings(create_dirs=True)
    setup_logging(settings)
    tz_name = settings.scheduler.timezone
    jobs = settings.scheduler.jobs
    if not jobs:
        logging.warning("scheduler.jobs が空です。config/config.yaml を確認してください。")
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
            now = datetime.now()
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

