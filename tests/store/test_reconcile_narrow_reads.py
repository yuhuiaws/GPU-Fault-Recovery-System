"""The two operator reconciles read only the workflow's own remote commands.

FINAL-建议汇总 F-J5 / F-D-log (P1-78D, P1-60D, P1-27C). Both reconcile
transactions loaded and decoded *every* remote command in the store while
holding the workflow's advisory lock and row lock; a 32-cluster fleet carries
tens of thousands of them per day. ``list_remote_commands(workflow_request_ids=
...)`` has existed on every backend all along.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.store import SqliteStore
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 5, 22, 0, tzinfo=timezone.utc)
RETIRED, CURRENT, OTHER = "wf-retired", "wf-current", "wf-other"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "narrow.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def _command(store, workflow, incident, command_id: str) -> None:
    store.ensure_remote_command(
        RemoteActionCommand(
            command_id=command_id,
            cluster_id=incident.cluster_id,
            workflow_request_id=workflow.request_id,
            incident_id=incident.incident_id,
            step_index=0,
            fencing_token=workflow.fencing_token,
            idempotency_key=f"remote/{command_id}",
            step=workflow.official_steps[0],
            workflow=workflow,
            incident=incident,
        )
    )


def _scenario(store):
    incident = fault_incident(
        "inc-r",
        "event-r",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=CURRENT,
        fencing_token=4,
        created_at=NOW,
        updated_at=NOW,
    )
    steps = [workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=["node-a"])]
    retired = workflow_request(
        RETIRED,
        "inc-r",
        status=WorkflowStatus.PENDING,
        fencing_token=1,
        official_steps=steps,
        created_at=NOW,
        updated_at=NOW,
    )
    current = workflow_request(
        CURRENT,
        "inc-r",
        status=WorkflowStatus.PENDING,
        fencing_token=4,
        official_steps=steps,
        created_at=NOW + timedelta(minutes=1),
        updated_at=NOW + timedelta(minutes=1),
    )
    other_incident = fault_incident(
        "inc-o",
        "event-o",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=OTHER,
        fencing_token=1,
        created_at=NOW,
        updated_at=NOW,
    )
    other = workflow_request(
        OTHER,
        "inc-o",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=steps,
        created_at=NOW,
        updated_at=NOW,
    )
    store.save_incident_and_workflow(incident, current)
    store.save_workflow(retired)
    store.save_incident_and_workflow(other_incident, other)
    _command(store, other, other_incident, "remote-other")
    return retired, current


class _Recorder:
    def __init__(self, store) -> None:
        self.narrow_calls: list[list[str]] = []
        self.full_loads: list[str] = []
        original_narrow = store.list_remote_commands
        original_list = getattr(store, "_list", None)

        def narrow(*, workflow_request_ids=None):
            self.narrow_calls.append(sorted(workflow_request_ids or []))
            return original_narrow(workflow_request_ids=workflow_request_ids)

        store.list_remote_commands = narrow
        if original_list is not None:

            def full(kind):
                self.full_loads.append(kind)
                return original_list(kind)

            store._list = full


def test_retired_generation_reconcile_reads_only_its_own_remote_commands(store):
    _scenario(store)
    recorder = _Recorder(store)

    revoked, _ = store.reconcile_retired_generation_workflow(
        RETIRED,
        CURRENT,
        expected_fencing_token=1,
        reference="operator-narrow-read",
        reconciled_at=NOW + timedelta(hours=1),
    )

    assert revoked.status is WorkflowStatus.SUPERSEDED
    assert recorder.narrow_calls == [[RETIRED]]
    assert "remote_command" not in recorder.full_loads
