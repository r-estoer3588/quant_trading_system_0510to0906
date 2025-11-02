"""
米国市場(XNYS)の当日が休場日かを判定し、結果を標準出力に返すユーティリティ。

戻り値:
- exit code 0: 取引日 (trading day)
- exit code 2: 休場日 (holiday / 非取引日)

出力:
- JSON 1行: {"date": "YYYY-MM-DD", "is_trading_day": bool, "reason": str}

依存:
- pandas_market_calendars (存在しない場合は土日判定にフォールバック)
"""

from __future__ import annotations

import json
from datetime import date, datetime


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def main() -> int:
    today = date.today()
    is_trading_day = False
    reason = ""

    try:
        import pandas_market_calendars as mcal  # type: ignore

        xnys = mcal.get_calendar("XNYS")
        sched = xnys.schedule(start_date=today, end_date=today)
        is_trading_day = not sched.empty
        reason = "market_open" if is_trading_day else "us_holiday_or_weekend"
    except Exception:
        # フォールバック: 土日以外は取引日とみなす
        is_trading_day = today.weekday() < 5
        reason = "weekday_fallback" if is_trading_day else "weekend_fallback"

    payload = {"date": _today_str(), "is_trading_day": is_trading_day, "reason": reason}
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if is_trading_day else 2


if __name__ == "__main__":
    raise SystemExit(main())
