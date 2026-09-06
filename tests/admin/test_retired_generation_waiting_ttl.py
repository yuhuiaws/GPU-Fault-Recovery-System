"""The operator's retired-generation apply can be told how long a restart may
wait before its reservation counts as never used (F-C9, log 60 item 2).

``retired_generation`` has no executor config to read the cap from, so the
entry point takes ``waiting_ttl`` and leaves the default -- keep a WAITING
record's reservation -- exactly as it was.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.retired_generation import (
    apply_retired_generation_plan,
    build_retired_generation_plan,
)
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

CLUSTER = "cluster-a"
JOB = "training-a"
RETIRED = "workflow-retired"
CURRENT = "workflow-current"
INCIDENT = "incident-a"
RESTART = WorkflowOperation.RESTART_WORKLOAD
TTL = timedelta(minutes=10)


def _store_with_waiting_restart(age: timedelta):
    store = build_store()
    now = datetime.now(timezone.utc)
    store.save_incident(
        fault_incident(
            INCIDENT,
            "event-a",
            cluster_id=CLUSTER,
            state=IncidentState.ACTION_PENDING,
            fencing_token=4,
            workflow_request_id=CURRENT,
        )
    )
    store.save_workflow(
        workflow_request(
            RETIRED,
            INCIDENT,
            status=WorkflowStatus.RUNNING,
            fencing_token=1,
            official_action="RESTART_APP",
            official_steps=[
                workflow_step(WorkflowOperation.FREEZE_EVIDENCE),
                workflow_step(
                    RESTART,
                    parameters={
                        "cluster_id": CLUSTER,
                        "job_id": JOB,
                        "restart_budget": 1,
                    },
                ),
            ],
            completed_step_indexes=[0],
            completed_operations=[WorkflowOperation.FREEZE_EVIDENCE],
            step_executions=[
                workflow_step_execution(
                    1,
                    RESTART,
                    WorkflowStepStatus.WAITING,
                    adapter_operation_id="remote/cmd-restart",
                    details={
                        "remote_status": "WAITING",
                        "remote_command_id": "cmd-restart",
                    },
                    started_at=now - age,
                    updated_at=now - age,
                )
            ],
        )
    )
    store.save_workflow(
        workflow_request(
            CURRENT,
            INCIDENT,
            fencing_token=4,
            official_action="RUN_DIAGNOSTICS",
            official_steps=[workflow_step(WorkflowOperation.VALIDATE_GPU)],
        )
    )
    store.reserve_job_restart(CLUSTER, JOB, 1, f"{RETIRED}/1/{RESTART.value}")
    return store


def _budget_is_free(store) -> bool:
    _, accepted = store.reserve_job_restart(CLUSTER, JOB, 1, "a-later-real-restart")
    return accepted


@pytest.mark.parametrize(
    "age, waiting_ttl, released",
    [
        (TTL + timedelta(seconds=1), TTL, True),
        (timedelta(seconds=10), TTL, False),
        (TTL + timedelta(seconds=1), None, False),
    ],
    ids=["older-than-ttl-released", "fresher-than-ttl-kept", "no-ttl-keeps-as-before"],
)
def test_apply_threads_the_waiting_ttl_into_the_reservation_release(
    age: timedelta, waiting_ttl: timedelta | None, released: bool
) -> None:
    store = _store_with_waiting_restart(age)
    plan = build_retired_generation_plan(store, [RETIRED])

    result = apply_retired_generation_plan(
        store,
        workflow_ids=[RETIRED],
        expected_plan_sha256=plan["plan_sha256"],
        reference="pre-deploy-6459c07ea279",
        waiting_ttl=waiting_ttl,
    )

    assert result["applied_workflow_ids"] == [RETIRED]
    assert result["restart_reservation_warnings"] == []
    assert store.get_workflow(RETIRED).status is WorkflowStatus.SUPERSEDED
    assert _budget_is_free(store) is released
