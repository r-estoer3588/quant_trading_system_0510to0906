from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceSummary:
    trades: int
    total_return: float
    win_rate: float
    max_drawdown: float
    sharpe: float
    sortino: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    cagr: float | None

    def to_dict(self) -> Dict[str, float | int | None]:
        return {
            "trades": self.trades,
            "total_return": self.total_return,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "cagr": self.cagr,
        }


def _equity_from_trades(trades_df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Build equity curve indexed by exit_date from trades dataframe.

    Assumes `trades_df` has columns: `exit_date`, `pnl`.
    """
    if trades_df.empty:
        return pd.Series([initial_capital])

    df = trades_df.copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])  # ensure datetime
    df = df.sort_values("exit_date")

    equity = initial_capital + df["pnl"].cumsum()
    equity.index = df["exit_date"].values
    return equity


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    return float(drawdown.min())


def _sharpe_daily(
    returns: pd.Series, risk_free: float = 0.0, periods: int = 252
) -> float:
    if returns.empty:
        return 0.0
    r = returns - risk_free / periods
    denom = r.std(ddof=0)
    return float(np.sqrt(periods) * (r.mean() / denom)) if denom > 0 else 0.0


def _sortino_daily(
    returns: pd.Series, risk_free: float = 0.0, periods: int = 252
) -> float:
    if returns.empty:
        return 0.0
    r = returns - risk_free / periods
    downside = r[r < 0]
    denom = downside.std(ddof=0)
    return float(np.sqrt(periods) * (r.mean() / denom)) if denom > 0 else 0.0


def _cagr(equity: pd.Series) -> float | None:
    if equity.empty:
        return None
    start, end = equity.index.min(), equity.index.max()
    try:
        years = (end - start).days / 365.25
        if years <= 0:
            return None
        return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
    except Exception:
        return None


def summarize(
    trades_df: pd.DataFrame, initial_capital: float
) -> Tuple[PerformanceSummary, pd.DataFrame]:
    """トレード一覧からパフォーマンスを集計し、概要と拡張済みDFを返す。

    - `df` の各行はトレードで、`exit_date` と `pnl` が必要。
    - 累積損益やドローダウン列を付与して返す。
    - 日次リターンの計算は重複インデックスでも安全な `resample("D").last().ffill()` を用いる。
    """
    if trades_df is None or trades_df.empty:
        return (
            PerformanceSummary(
                trades=0,
                total_return=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                sharpe=0.0,
                sortino=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                cagr=None,
            ),
            pd.DataFrame(),
        )

    df = trades_df.copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])  # ensure
    df = df.sort_values("exit_date")

    # equity/drawdown 列を追加（equity は各トレード終了時点の累積資産）
    equity = _equity_from_trades(df, initial_capital)
    df["cumulative_pnl"] = (equity - initial_capital).values
    df["cum_max"] = df["cumulative_pnl"].cummax()
    df["drawdown"] = df["cumulative_pnl"] - df["cum_max"]

    # 集計値
    total_return = float(df["pnl"].sum())
    wins = df[df["pnl"] > 0]["pnl"]
    losses = df[df["pnl"] <= 0]["pnl"]
    win_rate = float((df["pnl"] > 0).mean() * 100)
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    gross_profit = float(wins.sum())
    gross_loss = float(losses.abs().sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    # 日次リターン計算（重複index容認）
    # - 同一日の複数トレードがあっても最後の観測値を採用
    equity_for_daily = equity.sort_index()
    daily_equity = equity_for_daily.resample("D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()

    # リスク指標など
    sharpe = _sharpe_daily(daily_returns)
    sortino = _sortino_daily(daily_returns)
    mdd = _max_drawdown(equity)  # トレード粒度で計算
    cagr_val = _cagr(equity)

    summary = PerformanceSummary(
        trades=int(len(df)),
        total_return=total_return,
        win_rate=win_rate,
        max_drawdown=mdd,
        sharpe=sharpe,
        sortino=sortino,
        profit_factor=float(profit_factor),
        avg_win=avg_win,
        avg_loss=avg_loss,
        cagr=cagr_val,
    )
    return summary, df


def to_frame(summary: PerformanceSummary) -> pd.DataFrame:
    d = summary.to_dict()
    return pd.DataFrame([d])


def save_summary_csv(summary: PerformanceSummary, path: str) -> None:
    to_frame(summary).to_csv(path, index=False)
