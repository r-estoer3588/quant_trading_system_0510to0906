"""System1 strategy wrapper class using shared core functions.

This class integrates with YAML-driven settings for backtest parameters
and relies on StrategyBase to inject risk/system-specific config.  As an
extension example, Alpaca 発注処理も組み込み、バックテストと実売双方に
対応できるようにする。
"""

from __future__ import annotations

import pandas as pd

from .base_strategy import StrategyBase
from core.system1 import (
    prepare_data_vectorized_system1,
    generate_roc200_ranking_system1,
    get_total_days_system1,
)
from common.alpaca_order import AlpacaOrderMixin
from common.backtest_utils import simulate_trades_with_risk


class System1Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system1"

    def prepare_data(self, raw_data_dict, **kwargs):
        progress_callback = kwargs.pop("progress_callback", None)
        log_callback = kwargs.pop("log_callback", None)
        skip_callback = kwargs.pop("skip_callback", None)

        return prepare_data_vectorized_system1(
            raw_data_dict,
            progress_callback=progress_callback,
            log_callback=log_callback,
            skip_callback=skip_callback,
            **kwargs,
        )

    def generate_candidates(self, prepared_dict, market_df=None, **kwargs):
        # Pull top-N from YAML backtest config
        try:
            from config.settings import get_settings

            top_n = get_settings(create_dirs=False).backtest.top_n_rank
        except Exception:
            top_n = 10
        if market_df is None:
            market_df = prepared_dict.get("SPY")
            if market_df is None:
                raise ValueError("SPY data not found in prepared_dict.")
        return generate_roc200_ranking_system1(
            prepared_dict, market_df, top_n=top_n, **kwargs
        )

    def run_backtest(
        self, prepared_dict, candidates_by_date, capital, on_progress=None, on_log=None
    ) -> pd.DataFrame:
        trades_df, _ = simulate_trades_with_risk(
            candidates_by_date,
            prepared_dict,
            capital,
            self,
            on_progress=on_progress,
            on_log=on_log,
        )
        return trades_df

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system1(data_dict)
