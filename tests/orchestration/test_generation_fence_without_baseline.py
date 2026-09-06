"""F-B8 tail: the attempt-generation fence with no recovery workflow to compare.

An event that predates the running attempt used to be fenced only when a
ranked recovery workflow for that attempt existed as the baseline. Without one
the stale event was admitted and built a fresh recovery for a fault the
attempt restart had already answered. The restart that *did* happen is the
baseline: a stale candidate that does not out-rank RESTART_WORKLOAD is recorded
as an ignored incident with no workflow; an escalating one still runs.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from gpu_fault.models import IncidentState, WorkflowOperation
from gpu_fault.watcher import WorkloadPhase
from tests._builders import build_context
from tests.orchestration._cross_fault_support import (
    NOW,
    observation,
    post_faults,
    xid_payload,
)


def _restarted_without_recovery():
    """Attempt a001 stopped, a002 running since NOW+60 s, no workflow anywhere."""

    context = build_context()
    old = observation(started_at=NOW - timedelta(minutes=1))
    context.store.save_attempt_observation(
        old.model_copy(
            update={
                "workload_phase": WorkloadPhase.STOPPED,
                "observed_at": NOW + timedelta(seconds=10),
            }
        )
    )
    context.store.save_attempt_observation(
        observation(
            attempt_id="train-a002",
            started_at=NOW + timedelta(seconds=60),
            observed_at=NOW + timedelta(seconds=70),
        )
    )
    return context


def _stale(payload: dict) -> dict:
    payload.update(
        {
            "observed_at": (NOW + timedelta(seconds=40)).isoformat(),
            "collected_at": (NOW + timedelta(seconds=69)).isoformat(),
        }
    )
    return payload


def test_stale_same_rank_event_without_baseline_is_recorded_not_recovered() -> None:
    context = _restarted_without_recovery()

    result = asyncio.run(
        post_faults(
            context,
            [("/v1/gpu-events/xid", _stale(xid_payload(11, "stale-no-baseline")))],
        )
    )[0]

    incident = context.store.get_incident(result["incident_id"])
    assert incident.state is IncidentState.RECOVERED
    assert not incident.workflow_request_id, (
        "a fenced stale event must build no workflow"
    )
    assert context.store.list_workflows() == []
    assert any(
        "Ignored stale XID 11 action" in reason
        and "current_attempt=train-a002" in reason
        for reason in incident.reasons
    ), incident.reasons
    assert (
        context.store.get_incident_by_event("stale-no-baseline").incident_id
        == incident.incident_id
    )


def test_stale_escalating_event_without_baseline_still_recovers() -> None:
    context = _restarted_without_recovery()

    result = asyncio.run(
        post_faults(
            context,
            [("/v1/gpu-events/xid", _stale(xid_payload(79, "stale-escalation")))],
        )
    )[0]

    workflow = context.store.get_workflow(result["workflow_request_id"])
    operations = {step.operation for step in workflow.official_steps}
    assert operations & {
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.REPLACE_NODE,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    }, operations


def test_fresh_event_after_the_attempt_started_is_untouched() -> None:
    context = _restarted_without_recovery()
    payload = xid_payload(11, "fresh-after-restart")
    payload.update(
        {
            "observed_at": (NOW + timedelta(seconds=90)).isoformat(),
            "collected_at": (NOW + timedelta(seconds=91)).isoformat(),
        }
    )

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))[0]

    assert result["workflow_request_id"], (
        "an event inside the attempt keeps its recovery"
    )
