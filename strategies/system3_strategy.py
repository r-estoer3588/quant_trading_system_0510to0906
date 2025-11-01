# ============================================================================
# 🧠 Context Note
# このファイルは core/system3.py（ロング ミーン・リバージョン）を UI 用に適応させるラッパー層
#
# 前提条件：
#   - ロジック本体は core/system3.py。このファイルは orchestration のみ
#   - 3 日連続下落（drop3d）を売却シグナル検出
#   - ロング戦略（売却後の回復狙い）
#   - 最終配分は finalize_allocation() で一元化
#
# ロジック単位：
#   generate_signals()    → prepare_data + generate_candidates を順序実行
#   apply_allocation()    → 当日配分情報をまとめて渡す
#   _build_diagnostics()  → setup count など診断情報構築
#
# Copilot へ：
#   → core のロジック変更は core/system3.py で実施
#   → candidates が 0 の場合は正常。エラーと混同するな
#   → drop3d 判定の敏感性は慎重に調整してから提案
# ============================================================================

# strategies/system3_strategy.py
from __future__ import annotations

import numpy as np
import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
from common.system_diagnostics import (
    SystemDiagnosticSpec,
    build_system_diagnostics,
    numeric_greater_equal,
)
from core.system3 import (
    generate_candidates_system3,
    get_total_days_system3,
    prepare_data_vectorized_system3,
)

from .base_strategy import StrategyBase
from .constants import (
    MAX_HOLD_DAYS_DEFAULT,
    PROFIT_TAKE_PCT_DEFAULT_4,
    STOP_ATR_MULTIPLE_SYSTEM3,
)


class System3Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system3"

    # データ準備（共通コアへ委譲）
    def prepare_data(
        self,
        raw_data_or_symbols,
        reuse_indicators: bool | None = None,
        **kwargs,
    ):
        """System3のデータ準備（共通テンプレート使用）"""
        return self._prepare_data_template(
            raw_data_or_symbols,
            prepare_data_vectorized_system3,
            reuse_indicators=reuse_indicators,
            **kwargs,
        )

    # 候補生成（共通コアへ委譲）
    def generate_candidates(
        self,
        data_dict,
        market_df=None,
        **kwargs,
    ):
        """候補生成（共通メソッド使用）"""
        top_n = self._get_top_n_setting(kwargs.pop("top_n", None))
        batch_size = self._get_batch_size_setting(len(data_dict))
        # 重複渡し防止: kwargs に残っている latest_only を取り除いてから明示引数で渡す
        latest_only = bool(kwargs.pop("latest_only", False))
        try:  # noqa: SIM105
            from common.perf_snapshot import get_global_perf

            _perf = get_global_perf()
            if _perf is not None:
                _perf.mark_system_start(self.SYSTEM_NAME)
        except Exception:  # pragma: no cover
            pass
        result = generate_candidates_system3(
            data_dict,
            top_n=top_n,
            batch_size=batch_size,
            latest_only=latest_only,
            include_diagnostics=True,
            **kwargs,
        )
        try:
            from config.environment import get_env_config

            env = get_env_config()
            standardize = bool(getattr(env, "standardize_strategy_output", False))
        except Exception:
            standardize = False

        if isinstance(result, tuple) and len(result) >= 2:
            if len(result) == 3:
                candidates_by_date, merged_df, diagnostics = result
                self.last_diagnostics = diagnostics
            else:
                candidates_by_date, merged_df = result
                self.last_diagnostics = build_system_diagnostics(
                    self.SYSTEM_NAME,
                    data_dict,
                    candidates_by_date,
                    top_n=top_n,
                    latest_only=latest_only,
                    spec=SystemDiagnosticSpec(
                        rank_metric_name="drop3d",
                        rank_predicate=numeric_greater_equal("drop3d", 0.125),
                    ),
                )
            if standardize:
                try:
                    from common.candidates_schema import normalize_candidates_to_list

                    candidates_by_date = normalize_candidates_to_list(
                        candidates_by_date or {}
                    )
                except Exception:
                    pass
            result = (candidates_by_date, merged_df)
        else:
            self.last_diagnostics = None
            if standardize and isinstance(result, dict):
                try:
                    from common.candidates_schema import normalize_candidates_to_list

                    result = (normalize_candidates_to_list(result), None)
                except Exception:
                    result = (result, None)
        try:  # noqa: SIM105
            from common.perf_snapshot import get_global_perf as _gpf

            _p2 = _gpf()
            if _p2 is not None:
                candidate_count = self._compute_candidate_count(result)
                _p2.mark_system_end(
                    self.SYSTEM_NAME,
                    symbol_count=len(data_dict or {}),
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

    # 共通シミュレーター用フック（System3）
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
        prev_close = float(df.iloc[entry_idx - 1]["Close"])
        ratio = float(self.config.get("entry_price_ratio_vs_prev_close", 0.93))
        entry_price = round(prev_close * ratio, 2)
        atr = None
        for col in ("atr10", "ATR10"):
            try:
                atr = float(df.iloc[entry_idx - 1][col])
                break
            except Exception:
                continue
        if atr is None:
            return None
        stop_mult = float(
            self.config.get("stop_atr_multiple", STOP_ATR_MULTIPLE_SYSTEM3)
        )
        stop_price = entry_price - stop_mult * atr
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
        """利確/損切りロジック。
        - 終値ベースで4%以上の利益なら翌日大引けで決済
        - 損切り価格到達時は当日決済
        - 3日経過しても未達なら4日目の大引けで決済
        """
        profit_take_pct = float(
            self.config.get("profit_take_pct", PROFIT_TAKE_PCT_DEFAULT_4)
        )
        max_hold_days = int(self.config.get("max_hold_days", MAX_HOLD_DAYS_DEFAULT))

        for offset in range(max_hold_days + 1):
            idx = entry_idx + offset
            if idx >= len(df):
                break
            row = df.iloc[idx]
            if float(row["Low"]) <= stop_price:
                return stop_price, df.index[idx]
            gain = (float(row["Close"]) - entry_price) / entry_price
            if gain >= profit_take_pct:
                exit_idx = min(idx + 1, len(df) - 1)
                return float(df.iloc[exit_idx]["Close"]), df.index[exit_idx]

        exit_idx = min(entry_idx + max_hold_days + 1, len(df) - 1)
        return float(df.iloc[exit_idx]["Close"]), df.index[exit_idx]

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        """ロングのPnL - 基底クラスのメソッドを使用。"""
        return self.compute_pnl_long(entry_price, exit_price, shares)

    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out = {}
        for sym, df in raw_data_dict.items():
            x = df.copy()
            x["sma150"] = x["Close"].rolling(150).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system3(data_dict)
