from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

def fetch_data(
    symbols: Iterable[str], max_workers: int = 8, ui_manager: object | None = None
) -> dict[str, pd.DataFrame]: ...
def prepare_backtest_data(
    strategy: Any,
    symbols: Iterable[str],
    system_name: str = "SystemX",
    spy_df: pd.DataFrame | None = None,
    ui_manager: object | None = None,
    **kwargs: Any,
) -> tuple[dict[str, pd.DataFrame] | None, Any | None, pd.DataFrame | None]: ...
def run_backtest_with_logging(
    strategy: Any,
    prepared_dict: dict[str, pd.DataFrame] | None,
    candidates_by_date: Any,
    capital: float,
    system_name: str = "SystemX",
    ui_manager: object | None = None,
) -> pd.DataFrame: ...
def run_backtest_app(
    strategy: Any,
    system_name: str = "SystemX",
    limit_symbols: int | None = None,
    system_title: str | None = None,
    spy_df: pd.DataFrame | None = None,
    ui_manager: object | None = None,
    **kwargs: Any,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    dict[str, pd.DataFrame] | None,
    float,
    Any | None,
]: ...
def show_results(
    results_df: pd.DataFrame,
    capital: float,
    system_name: str = "SystemX",
    *,
    key_context: str = "main",
) -> None: ...
def show_signal_trade_summary(
    source_df: pd.DataFrame | dict[str, pd.DataFrame] | None,
    trades_df: pd.DataFrame | None,
    system_name: str,
    display_name: str | None = None,
) -> pd.DataFrame: ...
def save_signal_and_trade_logs(
    signal_counts_df: pd.DataFrame | None,
    results: pd.DataFrame | list[dict[str, Any]] | None,
    system_name: str,
    capital: float,
) -> None: ...
def save_prepared_data_cache(
    data_dict: dict[str, pd.DataFrame], system_name: str = "SystemX"
) -> None: ...
def display_roc200_ranking(
    ranking_df: pd.DataFrame,
    years: int = 5,
    top_n: int = 10,
    title: str = "System1 ROC200ランキング",
) -> None: ...
def clean_date_column(df: pd.DataFrame, col_name: str = "Date") -> pd.DataFrame: ...
