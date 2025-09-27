#!/usr/bin/env python3
"""Test AlpacaOrderMixin + StrategyBase combination."""

from common.alpaca_order import AlpacaOrderMixin
from strategies.base_strategy import StrategyBase


class TestStrategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "test"

    def prepare_data(self, *args, **kwargs):
        return {}

    def generate_candidates(self, *args, **kwargs):
        return pd.DataFrame()

    def run_backtest(self, *args, **kwargs):
        return {}


print("Testing AlpacaOrderMixin + StrategyBase...")
import pandas as pd

t = TestStrategy()
print("TestStrategy OK")
