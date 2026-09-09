"""Every destructive preflight waits out the same processor backlog (F-N1)."""

from __future__ import annotations

from scripts.e2e.regional.acceptance_runner_common import processor_queue_backlog


def test_the_gate_reads_the_fault_tier_when_the_snapshot_carries_it() -> None:
    assert processor_queue_backlog({"depth": 1, "fault_backlog_depth": 0}) == 0, (
        "routine telemetry above the reserved tier is not a backlog"
    )
    assert processor_queue_backlog({"depth": 5, "fault_backlog_depth": 2}) == 2
    assert processor_queue_backlog({"depth": "3", "fault_backlog_depth": None}) == 0


def test_the_gate_falls_back_to_total_depth_for_an_older_snapshot() -> None:
    assert processor_queue_backlog({"depth": 3}) == 3
    assert processor_queue_backlog({"depth": "0"}) == 0
    assert processor_queue_backlog({}) == 0
    assert processor_queue_backlog(None) == 0
