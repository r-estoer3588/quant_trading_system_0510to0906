# ============================================================================
# 🧠 Context Note
# このファイルは core/system1.py を Streamlit UI 用に適応させるラッパー層。バックテスト＆当日実行両対応
#
# 前提条件：
#   - UI からのシグナル呼び出しフロー: symbol list → setup → rank → signals
#   - ロジックの本体は core/system1.py。このファイルは orchestration のみ
#   - Alpaca 発注対応。YAML 設定経由のパラメータ注入
#   - 最終配分は finalize_allocation() で一元化（API 契約厳守）
#
# ロジック単位：
#   generate_signals() → prepare_data + generate_candidates を順序実行
#   apply_allocation() → 当日配分・ポジション情報をまとめて finalize_allocation() へ
#   prepare_data()    → キャッシュから指標ロード
#
# Copilot へ：
#   → core のロジック変更は core/system1.py で実施（このファイルは変更禁止）
#   → finalize_allocation() API 契約は変更するな
#   → UI 用の検証は簡潔に。複雑な検査は core に任せる
# ============================================================================

"""System1 strategy wrapper class using shared core functions.

This class integrates with YAML-driven settings for backtest parameters
and relies on StrategyBase to inject risk/system-specific config.  As an
extension example, Alpaca 発注処理も組み込み、バックテストと実売双方に
対応できるようにする。
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
from core.system1 import (
    generate_candidates_system1,
    get_total_days_system1,
    prepare_data_vectorized_system1,
)

from .base_strategy import StrategyBase
from .constants import STOP_ATR_MULTIPLE_SYSTEM1


class System1Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system1"

    def __init__(self) -> None:
        super().__init__()

    def prepare_data(
        self,
        raw_data_or_symbols: dict | list[str],
        reuse_indicators: bool | None = None,
        **kwargs: Any,
    ) -> dict:
        """System1のデータ準備（共通テンプレート + フォールバック対応）"""
        return cast(
            dict,
            self._prepare_data_template(
                raw_data_or_symbols,
                prepare_data_vectorized_system1,
                reuse_indicators=reuse_indicators,
                **kwargs,
            ),
        )

    def generate_candidates(self, data_dict, market_df=None, **kwargs):
        """候補生成（共通メソッド使用）"""
        top_n = self._get_top_n_setting(kwargs.get("top_n"))
        latest_only = bool(kwargs.get("latest_only", False))

        # Extract progress/log callbacks from kwargs if present
        progress_callback = kwargs.get("progress_callback", kwargs.get("on_progress"))
        log_callback = kwargs.get("log_callback", kwargs.get("on_log"))

        # perf snapshot 計測（存在しない場合はノーオペ）
        try:  # noqa: SIM105
            from common.perf_snapshot import get_global_perf

            _perf = get_global_perf()
            if _perf is not None:
                _perf.mark_system_start(self.SYSTEM_NAME)
        except Exception:  # pragma: no cover
            pass
        # 未知の追加キーワード（latest_mode_date / max_date_lag_days 等）もコアへ透過
        # ただし、明示引数として渡すキーは衝突を避けるため除外
        extra_kwargs = dict(kwargs)
        for k in (
            "latest_only",
            "top_n",
            "progress_callback",
            "on_progress",
            "log_callback",
            "on_log",
        ):
            if k in extra_kwargs:
                extra_kwargs.pop(k, None)
        result = generate_candidates_system1(
            data_dict,
            top_n=top_n,
            latest_only=latest_only,
            progress_callback=progress_callback,
            log_callback=log_callback,
            **extra_kwargs,
        )
        if isinstance(result, tuple) and len(result) == 3:
            candidates_by_date, merged_df, diagnostics = result
            self.last_diagnostics = diagnostics
            if merged_df is not None:
                try:
                    merged_df.attrs["system1_diagnostics"] = diagnostics
                except Exception:
                    pass
            result_tuple = (candidates_by_date, merged_df)
        else:  # Fallback for unexpected shapes
            self.last_diagnostics = None
            # 型が想定外の場合はそのまま返す（呼び出し側が安全に扱う）
            result_tuple = result
        try:  # noqa: SIM105
            from common.perf_snapshot import get_global_perf as _gpf

            _p2 = _gpf()
            if _p2 is not None:
                candidate_count = self._compute_candidate_count(result_tuple)
                _p2.mark_system_end(
                    self.SYSTEM_NAME,
                    symbol_count=len(data_dict or {}),
                    candidate_count=candidate_count,
                )
        except Exception:  # pragma: no cover
            pass
        return result_tuple

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

    def compute_entry(
        self,
        df: pd.DataFrame,
        candidate: dict,
        _current_capital: float,
    ) -> tuple[float, float] | None:
        """
        翌日寄り付きで成行仕掛けし、ATR20×5 を損切りに設定。

        Args:
            df: 価格データ
            candidate: エントリー候補情報
            _current_capital: 現在資本（未使用、インターフェース互換性のため）

        Returns:
            (entry_price, stop_price) または None
        """
        result = self._compute_entry_common(
            df,
            candidate,
            atr_column="atr20",
            stop_multiplier=self.config.get(
                "stop_atr_multiple",
                STOP_ATR_MULTIPLE_SYSTEM1,
            ),
        )
        if result is None:
            return None
        entry_price, stop_price, _ = result
        return entry_price, stop_price

    def get_total_days(self, data_dict: dict) -> int:
        return int(get_total_days_system1(data_dict))

    def compute_exit(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        _entry_price: float,
        stop_price: float,
    ) -> tuple[float, pd.Timestamp]:
        """
        Day-based exit for System1 (long):
        - Stop hit: if Low <= stop -> exit same day at stop_price
        - Otherwise, max-hold days then exit on close

        Args:
            df: 価格データ
            entry_idx: エントリーインデックス
            _entry_price: エントリー価格（未使用、インターフェース互換性のため）
            stop_price: ストップ価格

        Returns:
            (exit_price, exit_date): 決済価格と日付のタプル
        """
        try:
            from .constants import MAX_HOLD_DAYS_DEFAULT
        except Exception:
            MAX_HOLD_DAYS_DEFAULT = 3
        max_hold_days = int(self.config.get("max_hold_days", MAX_HOLD_DAYS_DEFAULT))
        n = len(df)
        for offset in range(max_hold_days):
            idx = entry_idx + offset
            if idx >= n:
                break
            row = df.iloc[idx]
            try:
                if float(row["Low"]) <= float(stop_price):
                    return float(stop_price), pd.Timestamp(str(df.index[idx]))
            except Exception:
                pass
        exit_idx = min(entry_idx + max_hold_days, n - 1)
        return float(df.iloc[exit_idx]["Close"]), pd.Timestamp(str(df.index[exit_idx]))
