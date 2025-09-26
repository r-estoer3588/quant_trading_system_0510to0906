#!/usr/bin/env python3
"""Test StrategyBase instantiation."""

import pandas as pd
from strategies.base_strategy import StrategyBase


class MockStrategy(StrategyBase):
    SYSTEM_NAME = "test"

    def prepare_data(self, *args, **kwargs):
        return {}

    def generate_candidates(self, *args, **kwargs):
        return pd.DataFrame()

    def run_backtest(self, *args, **kwargs):
        return {}


print("Creating MockStrategy instance...")
m = MockStrategy()
print("MockStrategy OK")
print(f"Config: {m.config}")
