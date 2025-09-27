"""Trading strategy constants module.

This module centralizes magic numbers used across trading strategies
to improve maintainability and make business rules explicit.
"""

# Profit take percentage thresholds
PROFIT_TAKE_PCT_DEFAULT_4 = 0.04  # 4% profit threshold (systems 2, 3)
PROFIT_TAKE_PCT_DEFAULT_5 = 0.05  # 5% profit threshold (system 6)

# Entry gap thresholds
ENTRY_MIN_GAP_PCT_DEFAULT = 0.04  # 4% minimum gap for entry (system 2)

# Maximum holding period constants
MAX_HOLD_DAYS_DEFAULT = 3  # Default maximum days to hold a position
FALLBACK_EXIT_DAYS_DEFAULT = 6  # Fallback exit period for system 5

# ATR-based stop loss multipliers
STOP_ATR_MULTIPLE_DEFAULT = 3.0  # Standard ATR multiplier (systems 2, 6, 7)
STOP_ATR_MULTIPLE_SYSTEM1 = 5.0  # System 1 specific ATR multiplier
STOP_ATR_MULTIPLE_SYSTEM3 = 2.5  # System 3 specific ATR multiplier
STOP_ATR_MULTIPLE_SYSTEM4 = 1.5  # System 4 specific ATR multiplier

# System-specific configuration mapping
SYSTEM_SPECIFIC_CONFIG = {
    "system1": {
        "stop_atr_multiple": STOP_ATR_MULTIPLE_SYSTEM1,
        "max_hold_days": MAX_HOLD_DAYS_DEFAULT,
        "fallback_enabled": True,
    },
    "system2": {
        "stop_atr_multiple": STOP_ATR_MULTIPLE_DEFAULT,
        "entry_min_gap_pct": ENTRY_MIN_GAP_PCT_DEFAULT,
        "profit_take_pct": PROFIT_TAKE_PCT_DEFAULT_4,
        "max_hold_days": MAX_HOLD_DAYS_DEFAULT,
    },
    "system3": {
        "stop_atr_multiple": STOP_ATR_MULTIPLE_SYSTEM3,
        "profit_take_pct": PROFIT_TAKE_PCT_DEFAULT_4,
        "max_hold_days": MAX_HOLD_DAYS_DEFAULT,
    },
    "system4": {
        "stop_atr_multiple": STOP_ATR_MULTIPLE_SYSTEM4,
        "max_hold_days": MAX_HOLD_DAYS_DEFAULT,
    },
    "system5": {
        "stop_atr_multiple": STOP_ATR_MULTIPLE_DEFAULT,
        "fallback_exit_days": FALLBACK_EXIT_DAYS_DEFAULT,
    },
    "system6": {
        "stop_atr_multiple": STOP_ATR_MULTIPLE_DEFAULT,
        "profit_take_pct": PROFIT_TAKE_PCT_DEFAULT_5,
    },
    "system7": {
        "stop_atr_multiple": STOP_ATR_MULTIPLE_DEFAULT,
        "max_hold_days": MAX_HOLD_DAYS_DEFAULT,
    },
}
