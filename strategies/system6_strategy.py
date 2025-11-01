# ============================================================================
# 🧠 Context Note
# このファイルは core/system6.py（ショート ミーン・リバージョン 高シックスデイサージ）を UI 用に適応させるラッパー層
#
# 前提条件：
#   - ロジック本体は core/system6.py。このファイルは orchestration のみ
#   - 6 日連続上昇サージを検出して空売りシグナル生成
#   - 高変動性環境で機能（HV 上昇確認）
#   - 最終配分は finalize_allocation() で一元化
#   - candidates 数がゼロになることは正常（サージ条件が厳しい）
#
# ロジック単位：
#   generate_signals()    → prepare_data + generate_candidates を順序実行
#   apply_allocation()    → 当日配分情報をまとめて渡す
#   _build_diagnostics()  → setup count など診断情報構築
#
# Copilot へ：
#   → core のロジック変更は core/system6.py で実施
#   → candidates が 0 でもエラーではない。正常な現象
#   → 6 日サージ検出ロジックは絶対変更禁止（他システムとの差別化要素）
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
from core.system6 import (
    generate_candidates_system6,
    get_total_days_system6,
    prepare_data_vectorized_system6,
)

from .base_strategy import StrategyBase
from .constants import (
    MAX_HOLD_DAYS_DEFAULT,
    PROFIT_TAKE_PCT_DEFAULT_5,
    STOP_ATR_MULTIPLE_DEFAULT,
)


class System6Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system6"

    def get_trading_side(self) -> str:
        """System6 はショート戦略"""
        return "short"

    def prepare_data(
        self,
        raw_data_or_symbols: dict | list,
        reuse_indicators: bool | None = None,
        **kwargs,
    ) -> dict:
        """System6のデータ準備（共通テンプレート使用、特殊分岐廃止）"""
        # パフォーマンス最適化: プロセスプール使用制御（型安全な環境アクセス）
        try:
            from config.environment import get_env_config  # 遅延importで循環回避

            env = get_env_config()
            use_process_pool = bool(getattr(env, "system6_use_process_pool", False))
        except Exception:
            # フォールバック（互換性維持）
            import os  # noqa: WPS433

            use_process_pool = (
                os.environ.get("SYSTEM6_USE_PROCESS_POOL", "false").lower() == "true"
            )

        # System6専用のパフォーマンス設定
        kwargs.setdefault("use_process_pool", use_process_pool)
        kwargs.setdefault("max_workers", 2)  # プロセスプール使用時も控えめに

        return self._prepare_data_template(
            raw_data_or_symbols,
            prepare_data_vectorized_system6,
            reuse_indicators=reuse_indicators,
            **kwargs,
        )

    def generate_candidates(
        self,
        data_dict: dict,
        market_df: pd.DataFrame | None = None,
        **kwargs,
    ) -> tuple[dict, pd.DataFrame | None]:
        """候補生成（共通メソッド使用、特殊分岐廃止）"""
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
        result = generate_candidates_system6(
            data_dict,
            top_n=top_n,
            batch_size=batch_size,
            latest_only=latest_only,
            include_diagnostics=True,
            **kwargs,
        )
        if isinstance(result, tuple) and len(result) == 3:
            candidates_by_date, merged_df, diag = result
            self.last_diagnostics = diag
            result = (candidates_by_date, merged_df)
        elif isinstance(result, tuple) and len(result) == 2:
            candidates_by_date, merged_df = result
            # fallback to computed diagnostics if core didn't return it
            self.last_diagnostics = build_system_diagnostics(
                self.SYSTEM_NAME,
                data_dict,
                candidates_by_date,
                top_n=top_n,
                latest_only=latest_only,
                spec=SystemDiagnosticSpec(
                    rank_metric_name="return_6d",
                    rank_predicate=numeric_greater_than("return_6d", 0.20),
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
                    symbol_count=len(data_dict or {}),
                    candidate_count=candidate_count,
                )
        except Exception:  # pragma: no cover
            # 戻り値の型をタプルに統一（互換維持）
            if isinstance(result, dict):
                result = (result, None)
        # 出力スキーマの標準化（フラグで有効化）
        try:
            from config.environment import get_env_config  # 遅延import

            env = get_env_config()
            if getattr(env, "standardize_strategy_output", False):
                from common.candidates_schema import normalize_candidates_to_list

                if isinstance(result, tuple) and len(result) == 2:
                    _c, _m = result
                    result = (normalize_candidates_to_list(_c or {}), _m)
                elif isinstance(result, dict):
                    # 返却型をタプルに揃える
                    result = (normalize_candidates_to_list(result), None)
        except Exception:
            # 標準化に失敗しても従来出力をそのまま返す（安全側）
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
        result: int = self._calculate_position_size_core(
            capital,
            entry_price,
            stop_price,
            risk,
            max_alloc,
        )
        return result

    # シミュレーター用フック（System6: Short）
    def compute_entry(
        self, df: pd.DataFrame, candidate: dict, _current_capital: float
    ) -> tuple[float, float] | None:
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
        ratio = float(self.config.get("entry_price_ratio_vs_prev_close", 1.05))
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
            self.config.get("stop_atr_multiple", STOP_ATR_MULTIPLE_DEFAULT)
        )
        stop_price = entry_price + stop_mult * atr
        # ショート戦略: ストップロスはエントリー価格より上に設定される
        if stop_price <= entry_price:
            return None
        return entry_price, stop_price

    def compute_exit(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_price: float,
    ) -> tuple[float, pd.Timestamp]:
        """System6 の利確・損切り・時間退出ルールを実装。"""

        profit_take_pct = float(
            self.config.get("profit_take_pct", PROFIT_TAKE_PCT_DEFAULT_5)
        )
        max_days = int(self.config.get("profit_take_max_days", MAX_HOLD_DAYS_DEFAULT))
        last_idx = len(df) - 1

        for offset in range(1, max_days + 1):
            idx = entry_idx + offset
            if idx >= len(df):
                break
            row = df.iloc[idx]

            # ショート戦略: 損切り（価格上昇で損失）
            if float(row["High"]) >= stop_price:
                return float(stop_price), df.index[idx]

            # ショート戦略: 利食い（価格下落で利益）
            current_price = float(row["Close"])
            gain = (entry_price - current_price) / entry_price
            if gain >= profit_take_pct:
                # 翌日の大引けで決済（仕様通り）
                exit_idx = idx + 1
                if exit_idx < len(df):
                    exit_price = float(df.iloc[exit_idx]["Close"])
                    exit_date = df.index[exit_idx]
                else:
                    exit_price = current_price
                    exit_date = df.index[idx]
                return exit_price, exit_date

        fallback_idx = entry_idx + max_days
        if fallback_idx < len(df):
            exit_price = float(df.iloc[fallback_idx]["Close"])
            exit_date = df.index[fallback_idx]
        else:
            exit_price = float(df.iloc[last_idx]["Close"])
            exit_date = df.index[last_idx]

        return exit_price, exit_date

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        """ショートのPnL - 基底クラスのメソッドを使用。"""
        return self.compute_pnl_short(entry_price, exit_price, shares)

    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out = {}
        for sym, df in raw_data_dict.items():
            x = df.copy()
            high, low, close = x["High"], x["Low"], x["Close"]
            tr = pd.concat(
                [
                    (high - low),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)
            x["atr10"] = tr.rolling(10).mean()
            returns = close.pct_change()
            x["hv50"] = returns.rolling(50).std() * (252**0.5) * 100
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return int(get_total_days_system6(data_dict))
