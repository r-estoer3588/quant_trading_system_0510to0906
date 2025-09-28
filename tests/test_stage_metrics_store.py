from __future__ import annotations

import pytest

from common.stage_metrics import StageEvent, StageMetricsStore


@pytest.fixture()
def store() -> StageMetricsStore:
    return StageMetricsStore()


def test_record_stage_sets_target_from_filter_when_progress_positive(
    store: StageMetricsStore,
) -> None:
    snapshot = store.record_stage("system1", 25, filter_count=7)

    assert snapshot.target == 7
    assert snapshot.filter_pass == 7

    events = store.drain_events()
    assert events == [StageEvent("system1", 25, 7, None, None, None)]


def test_record_stage_preserves_initial_target_during_later_updates(
    store: StageMetricsStore,
) -> None:
    first_snapshot = store.record_stage("system1", 0, filter_count=5)
    assert first_snapshot.target == 5
    assert first_snapshot.filter_pass is None

    second_snapshot = store.record_stage("system1", 25, filter_count=8)

    assert second_snapshot.target == 5
    assert second_snapshot.filter_pass == 8

    events = store.drain_events()
    assert events[-1] == StageEvent("system1", 25, 8, None, None, None)
