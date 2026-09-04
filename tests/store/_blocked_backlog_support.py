"""Shared record builders for the BLOCKED backlog gauge across backends.

``blocked_workflows_without_verified_restore`` is hand-written once per backend,
so the memory/SQLite cases and the Postgres case have to be driven by the same
records to mean anything. The Postgres case lives in ``test_postgres_store.py``
because ``POSTGRES_TESTS`` in the Makefile is the only pytest invocation that
runs with ``GPU_FAULT_TEST_POSTGRES_URL`` set, and it runs serially -- every
parallel invocation clears the variable.
"""

from __future__ import annotations

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from tests._builders import fault_incident, workflow_request

FLAWS = (
    "successor_status",
    "successor_incident",
    "successor_operation",
    "successor_fencing_token",
    "incident_state",
    "incident_fencing_token",
    "successor_is_itself",
    "successor_missing",
)


def blocked(store, name: str, *, fencing_token: int = 3) -> None:
    """One BLOCKED workflow whose incident has not recovered."""
    store.save_incident(
        fault_incident(
            f"incident-{name}",
            f"event-{name}",
            state=IncidentState.QUARANTINED,
            fencing_token=fencing_token,
        )
    )
    store.save_workflow(
        workflow_request(
            f"workflow-{name}",
            f"incident-{name}",
            status=WorkflowStatus.BLOCKED,
            fencing_token=fencing_token,
        )
    )


def restore(store, name: str) -> None:
    """Close ``name`` the way a successor workflow does when it restores it."""
    store.save_workflow(
        workflow_request(
            f"restore-{name}",
            f"incident-{name}",
            status=WorkflowStatus.SUCCEEDED,
            fencing_token=3,
            completed_operations=[WorkflowOperation.RESTORE_SCHEDULING],
        )
    )
    incident = store.get_incident(f"incident-{name}")
    store.save_incident(
        incident.model_copy(
            update={
                "state": IncidentState.RECOVERED,
                "workflow_request_id": f"restore-{name}",
            }
        )
    )


def break_one_clause(store, flaw: str, name: str) -> None:
    """Undo exactly one clause of ``verified_restore_successor`` for ``name``."""
    successor = store.get_workflow(f"restore-{name}")
    incident = store.get_incident(f"incident-{name}")
    if flaw == "successor_status":
        store.save_workflow(
            successor.model_copy(update={"status": WorkflowStatus.FAILED})
        )
    elif flaw == "successor_incident":
        # Names that must not resolve stay scoped to ``name``: on Postgres the
        # table is shared with every other test in the serial invocation.
        store.save_workflow(
            successor.model_copy(update={"incident_id": f"other-incident-{name}"})
        )
    elif flaw == "successor_operation":
        store.save_workflow(successor.model_copy(update={"completed_operations": []}))
    elif flaw == "successor_fencing_token":
        store.save_workflow(successor.model_copy(update={"fencing_token": 4}))
    elif flaw == "incident_state":
        store.save_incident(
            incident.model_copy(update={"state": IncidentState.ESCALATED})
        )
    elif flaw == "incident_fencing_token":
        store.save_incident(incident.model_copy(update={"fencing_token": 4}))
    elif flaw == "successor_is_itself":
        store.save_incident(
            incident.model_copy(update={"workflow_request_id": f"workflow-{name}"})
        )
    elif flaw == "successor_missing":
        store.save_incident(
            incident.model_copy(
                update={"workflow_request_id": f"workflow-missing-{name}"}
            )
        )
    else:  # pragma: no cover - guards the parametrisation itself
        raise AssertionError(f"unknown flaw: {flaw}")
