import pandas as pd

from strategies.base_strategy import StrategyBase


def test_compute_candidate_count_dataframe():
    class Dummy(StrategyBase):
        def prepare_data(self, raw_data_or_symbols, reuse_indicators=None, **kwargs):  # type: ignore[override]
            return {}

        def generate_candidates(self, data_dict, market_df=None, **kwargs):  # type: ignore[override]
            return ({}, None)

        def run_backtest(self, *args, **kwargs):  # type: ignore[override]
            raise NotImplementedError

    dummy = Dummy()
    df = pd.DataFrame({"a": [1, 2, 3]})
    empty = pd.DataFrame()

    # DataFrame 3行 -> 3
    assert dummy._compute_candidate_count(df) == 3
    # 空 DataFrame -> 0 (Noneではない)
    assert dummy._compute_candidate_count(empty) == 0


def test_compute_candidate_count_non_supported():
    class Dummy(StrategyBase):
        def prepare_data(self, raw_data_or_symbols, reuse_indicators=None, **kwargs):  # type: ignore[override]
            return {}

        def generate_candidates(self, data_dict, market_df=None, **kwargs):  # type: ignore[override]
            return ({}, None)

        def run_backtest(self, *args, **kwargs):  # type: ignore[override]
            raise NotImplementedError

    dummy = Dummy()

    # None -> None
    assert dummy._compute_candidate_count(None) is None

    # 不明型 -> None
    class X:  # noqa: D401
        pass

    assert dummy._compute_candidate_count(X()) is None
