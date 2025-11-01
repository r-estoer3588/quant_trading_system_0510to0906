# ============================================================================
# 🧠 Context Note
# このファイルは core/system4.py（ロング トレンド ロー・ボラティリティ）を UI 用に適応させるラッパー層
#
# 前提条件：
#   - ロジック本体は core/system4.py。このファイルは orchestration のみ
#   - 低ボラティリティ収縮期（HV50: 10-40%）を検出してエントリー
#   - トレンド確認（Close > SMA200）が必須
#   - 最終配分は finalize_allocation() で一元化
#
# ロジック単位：
#   generate_signals()    → prepare_data + generate_candidates を順序実行
#   apply_allocation()    → 当日配分情報をまとめて渡す
#   _build_diagnostics()  → setup count など診断情報構築
#
# Copilot へ：
#   → core のロジック変更は core/system4.py で実施
#   → ボラティリティ収縮判定（HV50 %ile）は厳格に守る
#   → DollarVolume50 の高閾値（100M）を変更する場合は制御テストで確認
# ============================================================================

# strategies/system4_strategy.py
from __future__ import annotations

import numpy as np
import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
from common.system_diagnostics import (
    SystemDiagnosticSpec,
    build_system_diagnostics,
    numeric_is_finite,
)
from common.utils import resolve_batch_size
from core.system4 import (
    generate_candidates_system4,
    get_total_days_system4,
    prepare_data_vectorized_system4,
)

from .base_strategy import StrategyBase
from .constants import STOP_ATR_MULTIPLE_SYSTEM4


class System4Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system4"

    # インジケータ計算（コア委譲）
    def prepare_data(
        self,
        raw_data_or_symbols,
        reuse_indicators: bool | None = None,
        **kwargs,
    ):
        """System4のデータ準備（共通テンプレート使用）"""
        return self._prepare_data_template(
            raw_data_or_symbols,
            prepare_data_vectorized_system4,
            reuse_indicators=reuse_indicators,
            **kwargs,
        )

    # 候補抽出（SPYフィルタ適用。market_df 後方互換あり）
    def generate_candidates(
        self,
        data_dict,
        market_df=None,
        progress_callback=None,
        log_callback=None,
        batch_size: int | None = None,
        **kwargs,
    ):
        prepared_dict = data_dict
        # 他システム(system1-3)と同様に共通の取得ロジックを使用
        top_n = self._get_top_n_setting(kwargs.pop("top_n", None))
        # market_df 未指定時は prepared_dict から SPY を使用（後方互換）
        if market_df is None:
            market_df = prepared_dict.get("SPY")
        if market_df is None or getattr(market_df, "empty", False):
            raise ValueError("System4 には SPYデータ (market_df) が必要です")
        # top_n は上で確定（明示指定 > strategies.<system>.top_n_rank > backtest.top_n_rank）
        if batch_size is None:
            try:
                from config.settings import get_settings

                batch_size = get_settings(create_dirs=False).data.batch_size
            except Exception:
                batch_size = 100
            batch_size = resolve_batch_size(len(prepared_dict), batch_size)
        # kwargs から取り出して重複渡しを防止
        latest_only = bool(kwargs.pop("latest_only", False))
        try:  # noqa: SIM105
            from common.perf_snapshot import get_global_perf

            _perf = get_global_perf()
            if _perf is not None:
                _perf.mark_system_start(self.SYSTEM_NAME)
        except Exception:  # pragma: no cover
            pass
        result = generate_candidates_system4(
            prepared_dict,
            top_n=top_n,
            progress_callback=progress_callback,
            log_callback=log_callback,
            batch_size=batch_size,
            latest_only=latest_only,
            include_diagnostics=True,
            **kwargs,
        )
        if isinstance(result, tuple) and len(result) == 3:
            candidates_by_date, merged_df, diagnostics = result
            self.last_diagnostics = diagnostics
            result = (candidates_by_date, merged_df)
        elif isinstance(result, tuple) and len(result) == 2:
            candidates_by_date, merged_df = result
            self.last_diagnostics = build_system_diagnostics(
                self.SYSTEM_NAME,
                prepared_dict,
                candidates_by_date,
                top_n=top_n,
                latest_only=latest_only,
                spec=SystemDiagnosticSpec(
                    rank_metric_name="rsi4",
                    rank_predicate=numeric_is_finite("rsi4"),
                ),
            )
            result = (candidates_by_date, merged_df)
        else:
            self.last_diagnostics = None
        try:  # noqa: SIM105
            from common.perf_snapshot import get_global_perf as _gpf

            _p2 = _gpf()
            if _p2 is not None:
                candidate_count = self._compute_candidate_count(result)
                _p2.mark_system_end(
                    self.SYSTEM_NAME,
                    symbol_count=len(prepared_dict or {}),
                    candidate_count=candidate_count,
                )
        except Exception:  # pragma: no cover
            pass
        return result

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_price: float,
        *,
        risk_pct: float | None = None,
        max_pct: float | None = None,
        **kwargs,
    ) -> int:
        risk = self._resolve_pct(risk_pct, "risk_pct", 0.02)
        max_alloc = self._resolve_pct(max_pct, "max_pct", 0.10)
        return self._calculate_position_size_core(
            capital,
            entry_price,
            stop_price,
            risk,
            max_alloc,
        )

    # システムフック群
    def compute_entry(self, df: pd.DataFrame, candidate: dict, _current_capital: float):
        try:
            entry_loc = df.index.get_loc(candidate["entry_date"])
        except Exception:
            return None
        if isinstance(entry_loc, slice) or isinstance(entry_loc, np.ndarray):
            return None
        if not isinstance(entry_loc, int | np.integer):
            return None
        entry_idx = int(entry_loc)
        if entry_idx <= 0 or entry_idx >= len(df):
            return None
        entry_price = float(df.iloc[entry_idx]["Open"])
        atr40 = None
        for col in ("atr40", "ATR40"):
            try:
                atr40 = float(df.iloc[entry_idx - 1][col])
                break
            except Exception:
                continue
        if atr40 is None:
            return None
        stop_mult = float(
            getattr(self, "config", {}).get(
                "stop_atr_multiple",
                STOP_ATR_MULTIPLE_SYSTEM4,
            )
        )
        stop_price = entry_price - stop_mult * atr40
        if entry_price - stop_price <= 0:
            return None
        return entry_price, stop_price

    def compute_exit(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_price: float,
    ):
        trail_pct = float(getattr(self, "config", {}).get("trailing_pct", 0.20))
        highest = entry_price
        for idx2 in range(entry_idx + 1, len(df)):
            close = float(df.iloc[idx2]["Close"])
            if close > highest:
                highest = close
            if close <= highest * (1 - trail_pct):
                return close, df.index[idx2]
            if close <= stop_price:
                return close, df.index[idx2]
        last_close = float(df.iloc[-1]["Close"])
        return last_close, df.index[-1]

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        """ロングのPnL - 基底クラスのメソッドを使用。"""
        return self.compute_pnl_long(entry_price, exit_price, shares)

    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out = {}
        for sym, df in raw_data_dict.items():
            # テスト用の軽量処理では浅いコピーで十分
            x = df.copy(deep=False)
            x["sma200"] = x["Close"].rolling(200).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system4(data_dict)
