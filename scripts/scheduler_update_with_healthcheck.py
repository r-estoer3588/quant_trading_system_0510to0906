from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
import traceback

# プロジェクトルートを import パスに追加
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from common.cache_manager import CacheManager  # type: ignore
from config.settings import get_settings  # type: ignore


def _prev_business_day_usa(day: datetime) -> datetime:
    """米国営業日の前営業日を返す（簡易版：土日のみ除外、祝日は無視）"""
    from datetime import timedelta

    d = day - timedelta(days=1)
    # 土日をスキップ
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d = d - timedelta(days=1)
    return d


def _read_latest_date_from_cache(cm: CacheManager, symbol: str) -> str | None:
    try:
        # CacheManager.read() を使用（profile="rolling"）
        df = cm.read(symbol, profile="rolling")
        if df is None or df.empty:
            return None
        date_col = None
        for c in ("Date", "date", "timestamp", "time"):
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            return None
        s = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if s.empty:
            return None
        return str(s.max().date())
    except Exception as e:
        # デバッグ用に例外を表示
        print(f"Warning: Failed to read {symbol} from cache: {e}")
        return None


def _run_update_once() -> None:
    # 既存スクリプトを直接呼び出す
    from scripts.update_from_bulk_last_day import main as bulk_update_main  # type: ignore

    # 環境変数で worker/tail を調整（未設定なら既定値維持）
    if os.getenv("SCHEDULER_WORKERS"):
        os.environ["BULK_UPDATE_WORKERS"] = os.getenv("SCHEDULER_WORKERS", "4")
    if os.getenv("SCHEDULER_TAIL_ROWS"):
        os.environ["BULK_TAIL_ROWS"] = os.getenv("SCHEDULER_TAIL_ROWS", "240")

    bulk_update_main()


def main() -> None:
    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    # 1回目の実行
    _run_update_once()

    # ヘルスチェック: SPY の rolling 最新日付が前営業日になっているか
    latest = _read_latest_date_from_cache(cm, "SPY")
    prev_bd = _prev_business_day_usa(datetime.now()).date().isoformat()

    if latest != prev_bd:
        # 1 回だけ再試行
        _run_update_once()
        latest = _read_latest_date_from_cache(cm, "SPY")

    ok = latest == prev_bd
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # settings に LOG_DIR が無い場合は既定の logs/
    log_base = getattr(settings, "LOG_DIR", None)
    log_dir = Path(log_base) if log_base else (ROOT / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"scheduler_update_health_{datetime.now().strftime('%Y%m%d')}.log"
    with out.open("a", encoding="utf-8") as f:
        status = "OK" if ok else "NG"
        f.write(f"[{now}] prev_bd={prev_bd} latest(SPY)={latest} " f"status={status}\n")

    # 環境によっては終了コードで成否を知らせたい場合もある
    if not ok:
        # ここでは非0で終了（スケジューラの再試行ポリシーでリトライ可能）
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # 予期せぬ例外はログに落として失敗終了
        try:
            settings = get_settings(create_dirs=True)
            log_base = getattr(settings, "LOG_DIR", None)
            log_dir = Path(log_base) if log_base else (ROOT / "logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            out = log_dir / (f"scheduler_update_health_{datetime.now().strftime('%Y%m%d')}.log")
            with out.open("a", encoding="utf-8") as f:
                f.write("[EXCEPTION]\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
        sys.exit(1)
