"""``POST /v1/workload-observations`` opens a placement hold and wakes the
dispatcher; a failing hold never fails the observation itself.

The observation is the data plane's statement of where an attempt runs. The
hold is the control plane's reaction to it, so it rides the same request but
is subordinate to it: the observation is stored first and returned 200 even
when the hold cannot be opened, which is logged and counted instead.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from gpu_fault.channel_registry import WORKLOAD_OBSERVATIONS_PATH
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_context,
    container_observation,
    fault_incident,
    workflow_request,
    workflow_step,
)

NOW = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
REBOOT = WorkflowOperation.RESTART_NODE


def _busy_node(store) -> None:
    incident = fault_incident(
        "inc-node",
        "event-node",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-node",
        node_ids=["node-a"],
        fencing_token=1,
    )
    workflow = workflow_request(
        "wf-node",
        "inc-node",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[workflow_step(REBOOT, node_ids=["node-a"])],
        execution_owner_id="executor-elsewhere",
        execution_lease_expires_at=NOW + timedelta(minutes=10),
    )
    store.save_incident_and_workflow(incident, workflow)


def _observation():
    return attempt_observation(
        "train-1",
        "train-1-a1",
        NOW,
        expected_critical_ranks=2,
        containers=[
            container_observation("pod-0", "train-1-0", 0, "node-a"),
            container_observation("pod-1", "train-1-1", 1, "node-b"),
        ],
        workload_ids=["training/job/train-1"],
    )


def _post(context) -> int:
    async def scenario() -> int:
        async with asgi_client(context) as client:
            response = await client.post(
                WORKLOAD_OBSERVATIONS_PATH, json=_observation().model_dump(mode="json")
            )
        return response.status_code

    return asyncio.run(scenario())


def test_observing_an_attempt_on_a_repairing_node_opens_a_hold_and_wakes():
    context = build_context()
    _busy_node(context.store)
    wakes: list[str] = []
    context.dispatcher.wake = lambda: wakes.append("woken")

    status = _post(context)

    assert status == 200
    hold = context.store.get_incident("hold-cluster-a-train-1-a1")
    assert hold.state is IncidentState.ACTION_PENDING
    workflow = context.store.get_workflow(hold.workflow_request_id)
    assert workflow.placement_hold is True
    assert workflow.status is WorkflowStatus.PENDING
    assert wakes == ["woken"]
    assert context.orchestrator.placement_holds_opened_total == 1
    # The observation itself was stored before the hold was considered.
    observed = context.store.list_attempt_observations("cluster-a")
    assert [item.attempt_id for item in observed] == ["train-1-a1"]


def test_an_observation_on_a_free_node_neither_holds_nor_wakes():
    context = build_context()
    wakes: list[str] = []
    context.dispatcher.wake = lambda: wakes.append("woken")

    assert _post(context) == 200
    assert wakes == []
    assert context.store.list_active_workflow_incidents("cluster-a") == []


def test_a_failing_hold_does_not_fail_the_observation():
    context = build_context()
    _busy_node(context.store)
    wakes: list[str] = []
    context.dispatcher.wake = lambda: wakes.append("woken")

    def broken(observation):
        raise RuntimeError("store unavailable")

    context.orchestrator.hold_attempt_on_repairing_nodes = broken

    assert _post(context) == 200
    observed = context.store.list_attempt_observations("cluster-a")
    assert [item.attempt_id for item in observed] == ["train-1-a1"]
    assert wakes == []
    assert context.orchestrator.placement_holds_failed_total == 1
