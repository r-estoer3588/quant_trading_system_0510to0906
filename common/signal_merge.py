from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import logging

try:  # pragma: no cover - fallback for older repos
    from common.progress import log_with_progress  # type: ignore
except Exception:  # pragma: no cover
    def log_with_progress(*args, **kwargs):
        pass

from common.utils import clamp01

try:  # pragma: no cover - optional dependency
    from common.regime import MarketRegime  # type: ignore
except Exception:  # pragma: no cover
    class MarketRegime:  # minimal fallback
        def __init__(self, state: Dict | None = None):
            self._state = state or {}
            self.severity = clamp01(self._state.get("severity", 0.0))

        def is_bearish(self) -> bool:
            return bool(self._state.get("bearish"))

try:  # pragma: no cover - name compatibility
    from common.config_loader import load_yaml  # type: ignore
except Exception:  # pragma: no cover
    from common.config_loader import load_config_file as load_yaml  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    system_id: int       # 1..7
    symbol: str
    side: str            # 'BUY' or 'SELL'
    strength: float      # raw strength, ~0..1
    meta: Dict


def _score_for_signal(sig: Signal, weights: Dict[str, float], regime: MarketRegime) -> float:
    meta = sig.meta or {}
    sid = int(sig.system_id)
    if sid == 1:
        w_rank = float(weights.get("w_rank", 0.0))
        w_trend = float(weights.get("w_trend", 0.0))
        w_filter = float(weights.get("w_filter", 0.0))
        rank_pct = clamp01(meta.get("rank_pct_ROC200", 1.0))
        trend_ok = 1.0 if meta.get("trend_ok") else 0.0
        spy_ok = 1.0 if meta.get("spy_filter_ok") else 0.0
        score = w_rank * (1.0 - rank_pct) + w_trend * trend_ok + w_filter * spy_ok
    elif sid == 2:
        w_thrust = float(weights.get("w_thrust", 0.0))
        w_vol = float(weights.get("w_vol", 0.0))
        w_filter = float(weights.get("w_filter", 0.0))
        rsi_thrust = clamp01(meta.get("rsi_thrust", 0.0))
        adx_pct = clamp01(meta.get("adx_pct", 0.0))
        filt = 1.0 if regime.is_bearish() else 0.0
        score = w_thrust * rsi_thrust + w_vol * adx_pct + w_filter * filt
    elif sid == 6:
        w_trend = float(weights.get("w_trend", 0.0))
        w_vol = float(weights.get("w_vol", 0.0))
        w_filter = float(weights.get("w_filter", 0.0))
        down = clamp01(meta.get("downtrend_strength", 0.0))
        atr = clamp01(meta.get("atr_norm", 0.0))
        filt = 1.0 if regime.is_bearish() else 0.0
        score = w_trend * down + w_vol * atr + w_filter * filt
    elif sid == 7:
        score = clamp01(getattr(regime, "severity", 0.0))
    else:  # 3,4,5
        w_core = float(weights.get("w_core", 0.0))
        w_filter = float(weights.get("w_filter", 0.0))
        strength = clamp01(meta.get("strength", sig.strength))
        spy_ok = 1.0 if meta.get("spy_filter_ok") else 0.0
        score = w_core * strength + w_filter * spy_ok
    return clamp01(score)


def merge_signals(all_signals: List[List[Signal]], portfolio_state, market_state) -> List[Dict]:
    """
    Build only an execution queue across systems.
    - Do not filter or block other systems' signals.
    - If opposite to existing position, always enqueue EXIT then ENTER.
    - Within the same tier, order by score(desc).
    """

    signals: List[Signal] = [s for arr in all_signals for s in arr]
    rules = load_yaml("config/merge_rules.yaml") or {}
    priority_cfg = {int(k): int(v) for k, v in (rules.get("priority") or {}).items()}
    scoring_cfg = rules.get("scoring", {})
    regime = MarketRegime(market_state)

    log_with_progress("merge.start", total=len(signals))

    scored: List[tuple[int, float, Signal]] = []
    for sig in signals:
        log_with_progress(
            "merge.step",
            increment=1,
            extra={"symbol": sig.symbol, "sys": sig.system_id, "side": sig.side},
        )
        weights = scoring_cfg.get(str(sig.system_id), {})
        score = _score_for_signal(sig, weights, regime)
        logger.info(f"[S{sig.system_id}] score={score:.3f} meta={sig.meta}")
        scored.append((priority_cfg.get(sig.system_id, 999), score, sig))

    scored.sort(key=lambda x: (x[0], -x[1]))

    queue: List[Dict] = []
    for _, score, sig in scored:
        existing = (portfolio_state or {}).get(sig.symbol)
        sig_side = "LONG" if sig.side.upper() == "BUY" else "SHORT"
        if existing and str(existing).upper() != sig_side:
            queue.append(
                {
                    "action": "EXIT",
                    "symbol": sig.symbol,
                    "reason": "reverse_signal",
                    "system": sig.system_id,
                }
            )
        queue.append(
            {
                "action": "ENTER",
                "symbol": sig.symbol,
                "side": sig.side,
                "system": sig.system_id,
                "score": score,
            }
        )

    log_with_progress("merge.done", total=len(queue))
    return queue


__all__ = ["Signal", "merge_signals"]

