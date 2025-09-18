from __future__ import annotations

import pandas as pd

from common.position_age import fetch_entry_dates_from_alpaca


class _Activity:
    def __init__(self, ts: str | None) -> None:
        self.transaction_time = ts


class _DummyClient:
    def __init__(self, mapping: dict[str, list[_Activity]]) -> None:
        self._mapping = mapping
        self.calls: list[tuple[str, str | None]] = []

    def get_activities(
        self, symbol: str, activity_types: str | None = None
    ) -> list[_Activity]:
        self.calls.append((symbol, activity_types))
        if symbol not in self._mapping:
            return []
        return list(self._mapping[symbol])


class _FailClient:
    def get_activities(
        self, symbol: str, activity_types: str | None = None
    ) -> list[_Activity]:
        raise RuntimeError("boom")


def test_fetch_entry_dates_returns_oldest_fill() -> None:
    client = _DummyClient(
        {
            "AAA": [
                _Activity("2024-05-03T15:00:00Z"),
                _Activity("2024-05-01T15:00:00Z"),
            ],
            "BBB": [_Activity("2024-06-01")],
        }
    )
    result = fetch_entry_dates_from_alpaca(client, ["aaa", "AAA", "BBB"])
    assert result["AAA"] == pd.Timestamp("2024-05-01T15:00:00Z")
    assert result["BBB"] == pd.Timestamp("2024-06-01")
    # ensure client was only queried once per symbol despite duplicates
    assert client.calls == [("AAA", "FILL"), ("BBB", "FILL")]


def test_fetch_entry_dates_handles_errors_gracefully() -> None:
    client = _FailClient()
    result = fetch_entry_dates_from_alpaca(client, ["ZZZ"])
    assert result == {}
