"""A validated restore whose collectors never report fails fast, naming them.

Live (gpu-eks-cluster, release 5377615bd785): the operator's ``restore`` of
the parked XID 46 incident ran VALIDATE_GPU on a node whose exporter was still
evicted; the step waited out the 600 s cap ("stayed non-terminal for 604s")
with ``missing_or_stale_collectors`` as its only word, and the incident ended
ESCALATED. An operator-driven restore learns nothing from the cap: after
``restore_collector_grace`` of silence it fails with the collectors named and
the things to check. The chain's own post-action validation keeps the cap --
the exporter it waits for is the one its RESTORE just let back.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.adapters import GpuValidationAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.validated_restore import RESTORE_WORKFLOW_PREFIX
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

OP = WorkflowOperation
NODE = "ip-10-0-41-164"
STARTED = datetime(2026, 10, 5, 23, 50, tzinfo=timezone.utc)


class NoMetrics:
    def __init__(self, store) -> None:
        self.store = store

    def latest(self, cluster_id, node_id):
        return []

    def findings(self, cluster_id, node_id):
        return []


def _context(store, *, request_id: str, waited: timedelta) -> WorkflowStepContext:
    step = workflow_step(
        OP.VALIDATE_GPU, "gpu-fault-validation-adapter", node_ids=[NODE]
    )
    workflow = workflow_request(
        request_id,
        "inc-xid46",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[step, workflow_step(OP.RESTORE_SCHEDULING, node_ids=[NODE])],
        step_executions=[
            workflow_step_execution(
                0,
                OP.VALIDATE_GPU,
                WorkflowStepStatus.WAITING,
                started_at=STARTED,
                updated_at=STARTED + waited,
            )
        ],
    )
    incident = fault_incident(
        "inc-xid46",
        "event-xid46",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=request_id,
        node_ids=[NODE],
        fencing_token=1,
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=1),
        idempotency_key=f"{request_id}/validate",
    )


def _adapter(store) -> GpuValidationAdapter:
    return GpuValidationAdapter(
        NoMetrics(store), store=store, restore_collector_grace=timedelta(seconds=120)
    )


def _execute_at(adapter: GpuValidationAdapter, context: WorkflowStepContext, now):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001 - datetime signature
            return now if tz is None else now.astimezone(tz)

    import gpu_fault.adapters.gpu_validation as module

    original = module.datetime
    module.datetime = Clock  # type: ignore[misc]
    try:
        return adapter.execute(context)
    finally:
        module.datetime = original  # type: ignore[misc]


def test_a_silent_validated_restore_fails_after_the_grace_naming_the_collectors() -> (
    None
):
    store = build_store()
    context = _context(
        store, request_id=f"{RESTORE_WORKFLOW_PREFIX}abc", waited=timedelta(seconds=130)
    )

    outcome = _execute_at(_adapter(store), context, STARTED + timedelta(seconds=130))

    assert outcome.status is WorkflowStepStatus.FAILED, outcome.details
    assert outcome.details["failed_nodes"] == [NODE]
    assert outcome.details["collectors_not_reporting"] == {NODE: ["GPU_METRICS"]}
    assert outcome.details["restore_collector_grace_seconds"] == 120
    assert "have not reported for 130s" in (outcome.error or "")
    assert "DaemonSets are not running" in (outcome.error or "")
    assert "gpu-fault.io/quiesce taint" in (outcome.error or "")


def test_the_grace_is_still_a_wait_before_it_runs_out() -> None:
    store = build_store()
    context = _context(
        store, request_id=f"{RESTORE_WORKFLOW_PREFIX}abc", waited=timedelta(seconds=30)
    )

    outcome = _execute_at(_adapter(store), context, STARTED + timedelta(seconds=30))

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_pending"][NODE]["missing_or_stale_collectors"] == [
        "GPU_METRICS"
    ]


def test_a_chains_own_validation_keeps_waiting_under_the_step_cap() -> None:
    store = build_store()
    context = _context(
        store, request_id="wf-reset-chain", waited=timedelta(seconds=400)
    )

    outcome = _execute_at(_adapter(store), context, STARTED + timedelta(seconds=400))

    assert outcome.status is WorkflowStepStatus.WAITING, (
        "the exporter the chain waits for is the one its RESTORE just let back"
    )


def test_the_grace_has_a_floor() -> None:
    with pytest.raises(ValueError, match="restore collector grace"):
        GpuValidationAdapter(
            NoMetrics(build_store()),
            store=build_store(),
            restore_collector_grace=timedelta(seconds=5),
        )
