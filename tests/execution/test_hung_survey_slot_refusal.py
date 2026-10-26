"""A node that refused a survey command before any worker ran is asked again.

Live (GF-GENERAL-LIVE-011 ``fr_temporal_completion``, HyperPod, 2026-10-26
04:21:31Z) both nodes answered COLLECT_HUNG_TRIAGE with ``WORKER_STILL_RUNNING``
1.2 s after the step started -- their single observation worker slot was held by
the passive communication sample -- and the survey adapter recorded both under
``node_failures`` at once, inside a 30-second per-node wait it never used. The
refusal an upgraded Agent sends is a retryable failure with ``worker_started``
false; the adapter keeps such a node pending and re-sends within the node wait,
the way it already waits on a node that has accepted the command.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from gpu_fault.adapters.node_action import hung_survey
from gpu_fault.adapters.node_action.adapter import NodeActionWorkflowAdapter
from gpu_fault.hung_summary_codec import decode_summary
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowStepStatus
from gpu_fault.node_agent.protocol import NodeActionResult, NodeActionStatus
from tests.execution.test_full_hung_survey import context
from tests.node_agent._support import SECRET

REFUSAL = {
    "status": "WORKER_STILL_RUNNING",
    "evidence_complete": False,
    "ranks": [],
    "error": "observation worker slot held by sample for 2.1 s; waited 6.0 s",
    "failed_component": "observation_supervisor",
    "failure_stage": "WORKER_NOT_STARTED",
    "worker_started": False,
    "slot_wait_seconds": 6.0,
}
ERROR = "ObservationSlotBusy: WORKER_STILL_RUNNING: " + REFUSAL["error"]


def refusal(envelope, *, retryable: bool) -> NodeActionResult:
    return NodeActionResult(
        command_id=envelope.command.command_id,
        operation=Op.COLLECT_HUNG_SUMMARY,
        status=NodeActionStatus.FAILED,
        error=ERROR if retryable else "ObservationIncomplete: WORKER_STILL_RUNNING",
        retryable=retryable,
        details=dict(REFUSAL)
        if retryable
        else {
            "status": "WORKER_STILL_RUNNING",
            "evidence_complete": False,
            "ranks": [],
        },
    )


def collected(envelope) -> NodeActionResult:
    return NodeActionResult(
        command_id=envelope.command.command_id,
        operation=Op.COLLECT_HUNG_SUMMARY,
        status=NodeActionStatus.SUCCEEDED,
        details={"status": "COLLECTED", "ranks": [{"pid": 100, "rank": 0}]},
    )


def test_a_retryable_slot_refusal_is_resent_within_the_node_wait():
    sent: list[float] = []
    lock = threading.Lock()

    def sender(endpoint, envelope):
        with lock:
            sent.append(time.monotonic())
            attempt = len(sent)
        return (
            refusal(envelope, retryable=True) if attempt == 1 else collected(envelope)
        )

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "https://test.invalid"}, SECRET, sender=sender
    )
    result = adapter.execute(context(["node-a"]))
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert result.details["node_failures"] == {}, result.details
    document = decode_summary(result.details["node_results"]["node-a"])
    assert document["ranks"] == [{"pid": 100, "rank": 0}], document
    assert len(sent) == 2, sent
    assert 0.9 <= sent[1] - sent[0] < 3, "a refused node is re-sent after a pause"


def test_a_refusal_that_outlasts_the_node_wait_is_recorded_with_its_cause(monkeypatch):
    monkeypatch.setattr(hung_survey, "NODE_TIMEOUT_SECONDS", 2)
    sent: list[Any] = []

    def sender(endpoint, envelope):
        sent.append(envelope.command.command_id)
        return refusal(envelope, retryable=True)

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "https://test.invalid"}, SECRET, sender=sender
    )
    started = time.monotonic()
    result = adapter.execute(context(["node-a"]))
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert result.details["node_failures"] == {"node-a": ERROR}, result.details
    assert result.details["undetermined_nodes"] == ["node-a"], result.details
    # The refusal itself is retained, as every incomplete node answer is.
    refused = decode_summary(result.details["node_results"]["node-a"])
    assert refused["worker_started"] is False and refused["ranks"] == [], refused
    assert 2 <= time.monotonic() - started < 6, "re-sent across the node wait"
    assert len(set(sent)) == 1 and 2 <= len(sent) <= 4, sent


def test_an_older_agents_refusal_is_recorded_at_once():
    sent: list[Any] = []

    def sender(endpoint, envelope):
        sent.append(envelope.command.command_id)
        return refusal(envelope, retryable=False)

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "https://test.invalid"}, SECRET, sender=sender
    )
    result = adapter.execute(context(["node-a"]))
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert result.details["node_failures"] == {
        "node-a": "ObservationIncomplete: WORKER_STILL_RUNNING"
    }, result.details
    assert len(sent) == 1, sent
