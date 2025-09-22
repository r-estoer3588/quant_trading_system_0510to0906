"""日次の株価データをフル/ローリング両キャッシュに反映するツール."""

from __future__ import annotations

import logging

import pandas as pd

from common.cache_manager import CacheManager
from config.settings import get_settings

logger = logging.getLogger(__name__)


def main(fetched: dict[str, pd.DataFrame]) -> None:
    """EODHD等から取得した日次データをキャッシュへ書き込む."""
    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)
    logger.info("[cache_daily_data] ⏳ upsert開始: items=%s", len(fetched))
    for ticker, df in fetched.items():
        cm.upsert_both(ticker, df)
    cm.prune_rolling_if_needed(anchor_ticker="SPY")
    logger.info("[cache_daily_data] ✅ 完了: items=%s", len(fetched))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main({})
