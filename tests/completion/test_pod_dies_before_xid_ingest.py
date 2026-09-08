"""The Pod died before its XID reached the control plane.

An application-level XID (RESTART_APP class) kills the training process; the
Pod is observed terminated, and only then does the kernel event arrive. The
topology service therefore resolves the node as IDLE (no live container).
The XID must not quarantine the node, and the attempt's own failure must
still restart it through the passive path (quick triage -> RESTART_WORKLOAD).

Before the policy judged RESTART_APP on IDLE as MONITOR_ONLY, this sequence
ended with the node cordoned and quarantined and the job never restarted.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    DecisionStatus,
    IncidentState,
    RecoveryAction,
    TerminalEvent,
    TriageFinding,
    TriageOutcome,
    TriageReport,
    WorkloadState,
)
from gpu_fault.watcher import WorkloadPhase
from tests._builders import asgi_client, attempt_observation, container_observation

WORKLOAD_ID = "training/pytorchjob/distributed-training"


def _observe_running_then_dead(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    common = {
        "cluster_id": failed_event.cluster_id,
        "workload_ids": [WORKLOAD_ID],
        "started_at": ended_at - timedelta(minutes=10),
    }
    context.store.save_attempt_observation(
        attempt_observation(
            failed_event.job_id,
            failed_event.attempt_id,
            ended_at - timedelta(seconds=60),
            containers=[
                container_observation(
                    "pod-a", "worker-0", 0, "node-a", gpu_uuids=["GPU-a"]
                )
            ],
            **common,
        )
    )
    # The process died: the newest observation carries a terminated container
    # and a FAILED phase. Nothing live is left on node-a.
    context.store.save_attempt_observation(
        attempt_observation(
            failed_event.job_id,
            failed_event.attempt_id,
            ended_at,
            workload_phase=WorkloadPhase.FAILED,
            containers=[
                container_observation(
                    "pod-a",
                    "worker-0",
                    0,
                    "node-a",
                    gpu_uuids=["GPU-a"],
                    terminated=True,
                    exit_code=1,
                    finished_at=ended_at,
                )
            ],
            **common,
        )
    )


def _post_late_xid(context: ApplicationContext, observed_at: datetime) -> dict:
    """Deliver the kernel line the way the node collector does.

    The collector route is the production path: it resolves the workload
    state from the attempt observations (the collector itself only knows
    UNKNOWN) before the XID reaches the policy.
    """

    async def scenario() -> dict:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/nvidia-kernel",
                json={
                    "cluster_id": "cluster-a",
                    "node_id": "node-a",
                    "record_id": "kmsg-xid13-after-pod-death",
                    "observed_at": observed_at.isoformat(),
                    "message": "NVRM: Xid (PCI:0000:59:00): 13, pid=4242, name=python",
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                },
            )
        assert response.status_code == 200, response.text
        return response.json()

    return asyncio.run(scenario())


def test_pod_dies_before_xid_ingest_keeps_node_and_restarts_via_triage(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    _observe_running_then_dead(context, failed_event, ended_at)

    payload = _post_late_xid(context, ended_at + timedelta(seconds=5))

    (event,) = payload["normalized"]["xid_events"]
    (decision,) = payload["decisions"]
    assert event["xid"] == 13
    # The dead Pod is no live container, so the node resolves as IDLE; the
    # cluster is still covered by the attempt's own (terminated) observation.
    assert event["workload_state"] == WorkloadState.IDLE.value
    assert event["affected_workload_ids"] == []
    assert decision["disposition"] == "MONITOR_ONLY"
    assert decision["action"] == RecoveryAction.NO_ACTION.value
    assert decision["workflow_request_id"] is None
    incident = context.store.get_incident(decision["incident_id"])
    assert incident.state is IncidentState.RECOVERED
    # No workflow at all means no MARK_UNSCHEDULABLE / QUARANTINE was planned.
    assert incident.workflow_request_id is None

    # The attempt's own terminal event is not owned by the XID incident
    # (its marker is inactive), so it goes to quick triage, and a PASS
    # restarts the job on its original allocation.
    pending = context.completion.handle_terminal(failed_event)
    assert pending.status is DecisionStatus.PENDING_TRIAGE, pending

    resolved = context.completion.handle_triage(
        TriageReport(
            request_id=pending.diagnostic_request_id,
            attempt_id=failed_event.attempt_id,
            completed_at=ended_at + timedelta(seconds=30),
            findings=[
                TriageFinding(node_id="node-a", outcome=TriageOutcome.PASS),
                TriageFinding(node_id="node-b", outcome=TriageOutcome.PASS),
            ],
        )
    )
    plan = context.store.get_plan(resolved.recovery_plan_id)

    assert resolved.status is DecisionStatus.PLAN_CREATED
    assert plan.trigger == "quick-triage:PASS"
    assert [step.action for step in plan.steps] == [RecoveryAction.RESTART_WORKLOAD]
    assert plan.steps[0].parameters["reuse_allocation"] is True
