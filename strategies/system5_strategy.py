# ============================================================================
# 🧠 Context Note
# このファイルは core/system5.py（ロング ミーン・リバージョン 高 ADX）を UI 用に適応させるラッパー層
#
# 前提条件：
#   - ロジック本体は core/system5.py。このファイルは orchestration のみ
#   - 高 ADX 環境（ADX7 > 55）でのミーン・リバージョン狙い
#   - ATR_Pct による変動性フィルター（> 2.5%）
#   - RSI3 < 50 で過売り確認
#   - 最終配分は finalize_allocation() で一元化
#
# ロジック単位：
#   generate_signals()    → prepare_data + generate_candidates を順序実行
#   apply_allocation()    → 当日配分情報をまとめて渡す
#   _build_diagnostics()  → setup count など診断情報構築
#
# Copilot へ：
#   → core のロジック変更は core/system5.py で実施
#   → ADX 閾値（55）の変更は慎重に。他システムとの競合検証必須
#   → RSI3 条件の役割は「リバージョン環境確認」。ロジック変更禁止
# ============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
from common.system_diagnostics import (
    SystemDiagnosticSpec,
    build_system_diagnostics,
    numeric_greater_than,
)
from common.utils import resolve_batch_size
from core.system5 import (
    generate_candidates_system5,
    get_total_days_system5,
    prepare_data_vectorized_system5,
)

from .base_strategy import StrategyBase
from .constants import FALLBACK_EXIT_DAYS_DEFAULT, STOP_ATR_MULTIPLE_DEFAULT


class System5Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system5"

    def __init__(self):
        """System5初期化、_last_entry_atr属性を追加。"""
        super().__init__()
        self._last_entry_atr: float = 0.0

    def prepare_data(
        self,
        raw_data_or_symbols,
        reuse_indicators: bool | None = None,
        **kwargs,
    ):
        """System5のデータ準備（共通テンプレート使用）"""
        return self._prepare_data_template(
            raw_data_or_symbols,
            prepare_data_vectorized_system5,
            reuse_indicators=reuse_indicators,
            **kwargs,
        )

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
        # 共通ロジックで上限件数を決定（明示指定 > strategies.<system>.top_n_rank > backtest.top_n_rank）
        top_n = self._get_top_n_setting(kwargs.pop("top_n", None))

        if batch_size is None:
            try:
                from config.settings import get_settings

                default_bs = int(get_settings(create_dirs=False).data.batch_size)
            except Exception:
                default_bs = 100
            batch_size = resolve_batch_size(len(prepared_dict or {}), default_bs)

        latest_only = bool(kwargs.pop("latest_only", False))

        # 環境フラグが立っている場合、Option-B ユーティリティを core 側へ渡す
        try:
            from config.environment import get_env_config as _get_env

            _env = _get_env()
            if (
                bool(getattr(_env, "enable_option_b_system5", False))
                and "use_option_b_utils" not in kwargs
            ):
                kwargs["use_option_b_utils"] = True
        except Exception:
            pass

        try:  # noqa: SIM105
            from common.perf_snapshot import get_global_perf

            _perf = get_global_perf()
            if _perf is not None:
                _perf.mark_system_start(self.SYSTEM_NAME)
        except Exception:  # pragma: no cover
            pass

        result = generate_candidates_system5(
            prepared_dict,
            top_n=top_n,
            progress_callback=progress_callback,
            log_callback=log_callback,
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
                    prepared_dict,
                    candidates_by_date,
                    top_n=top_n,
                    latest_only=latest_only,
                    spec=SystemDiagnosticSpec(
                        rank_metric_name="adx7",
                        rank_predicate=numeric_greater_than("adx7", 0.0),
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
        ratio = float(
            getattr(self, "config", {}).get("entry_price_ratio_vs_prev_close", 0.97)
        )
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
            getattr(self, "config", {}).get(
                "stop_atr_multiple",
                STOP_ATR_MULTIPLE_DEFAULT,
            )
        )
        stop_price = entry_price - stop_mult * atr
        if entry_price - stop_price <= 0:
            return None
        self._last_entry_atr = atr
        return entry_price, stop_price

    def compute_exit(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_price: float,
    ):
        """System5 の利確・損切り・時間退出ロジック。

        - 利益目標: 過去10日ATR×設定倍率を上回ったら翌営業日の寄り付きで決済
        - 損切り: 当日の安値がストップ以下になった時点でストップ価格で決済
        - 時間退出: 6営業日経過後も未決済なら7日目の寄り付きで決済
        """

        atr = getattr(self, "_last_entry_atr", None)
        if atr is None:
            try:
                atr = None
                for col in ("atr10", "ATR10"):
                    try:
                        atr = float(df.iloc[entry_idx - 1][col])
                        break
                    except Exception:
                        continue
                if atr is None:
                    return None
            except Exception:
                atr = 0.0
        target_mult = float(getattr(self, "config", {}).get("target_atr_multiple", 1.0))
        target_price = entry_price + target_mult * atr
        fallback_days = int(
            getattr(self, "config", {}).get(
                "fallback_exit_after_days",
                FALLBACK_EXIT_DAYS_DEFAULT,
            )
        )

        last_idx = len(df) - 1

        for offset in range(1, fallback_days + 1):
            idx = entry_idx + offset
            if idx >= len(df):
                break
            row = df.iloc[idx]

            if float(row["Low"]) <= stop_price:
                return float(stop_price), df.index[idx]

            if float(row["High"]) >= target_price:
                exit_idx = idx + 1
                if exit_idx < len(df):
                    exit_price = float(df.iloc[exit_idx]["Open"])
                    exit_date = df.index[exit_idx]
                else:
                    exit_price = float(df.iloc[idx]["Close"])
                    exit_date = df.index[idx]
                return exit_price, exit_date

        fallback_exit_idx = entry_idx + fallback_days + 1
        if fallback_exit_idx < len(df):
            exit_price = float(df.iloc[fallback_exit_idx]["Open"])
            exit_date = df.index[fallback_exit_idx]
        else:
            fallback_idx = min(entry_idx + fallback_days, last_idx)
            exit_price = float(df.iloc[fallback_idx]["Close"])
            exit_date = df.index[fallback_idx]

        return exit_price, exit_date

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        """ロングのPnL - 基底クラスのメソッドを使用。"""
        return self.compute_pnl_long(entry_price, exit_price, shares)

    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out = {}
        for sym, df in raw_data_dict.items():
            x = df.copy()
            x["sma100"] = x["Close"].rolling(100).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system5(data_dict)
