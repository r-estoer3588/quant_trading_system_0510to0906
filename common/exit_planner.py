"""Utilities for deciding when to exit live positions.

This module contains small, dependency-light helpers that can be unit tested
without importing the heavy Streamlit application.  The helpers encapsulate the
business rules for determining whether a position should be closed immediately
and which session (today's close, tomorrow's close, or tomorrow's open) should
be targeted for the order.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

_TOMORROW_CLOSE_SYSTEMS: Iterable[str] = {
    "system1",
    "system2",
    "system3",
    "system6",
}


def decide_exit_schedule(
    system: str, exit_date: object, today: pd.Timestamp | None
) -> tuple[bool, str]:
    """Return whether the exit is due today and the suggested order timing.

    Parameters
    ----------
    system:
        Strategy system name (e.g. ``"system2"``).  The value is normalised to
        lower-case internally, but the comparison is case-sensitive with respect
        to the canonical system identifiers used throughout the project.
    exit_date:
        The timestamp returned by the strategy's ``compute_exit`` hook.  Any
        object accepted by :func:`pandas.to_datetime` is supported.
    today:
        Trading day used as the reference point for "today".  ``None`` disables
        the comparison and always marks the exit as not due.

    Returns
    -------
    tuple[bool, str]
        A tuple ``(is_due, when)`` where ``is_due`` indicates whether the exit
        should be executed immediately and ``when`` is the suggested timing
        string (``"today_close"``, ``"tomorrow_close"`` or ``"tomorrow_open"``).
    """

    system_norm = (system or "").lower()
    exit_ts = pd.to_datetime(exit_date).normalize()
    is_due = False
    if today is not None and not pd.isna(exit_ts):
        is_due = bool(exit_ts <= today)

    if system_norm == "system5":
        when_due = "tomorrow_open"
        when_future = "tomorrow_open"
    else:
        when_due = "today_close"
        when_future = (
            "tomorrow_close"
            if system_norm in _TOMORROW_CLOSE_SYSTEMS
            else "today_close"
        )

    return is_due, when_due if is_due else when_future


__all__ = ["decide_exit_schedule"]
