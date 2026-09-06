"""The apply path compares on what the approval bound (F-K1 / log §61 未做 1).

``apply_workflow_reconcile_plan`` rebuilds the plan, checks its digest against
the approval and then hands each verified-restore item to the Store. The Store
compare-and-set must be keyed on ``execution_epoch`` -- which the digest covers
-- and not on a re-read ``updated_at`` that a heartbeat restamps between the
rebuild and the write. The restart-reservation release the same path runs
afterwards takes the executor's waiting TTL from the caller (log §60 未做 2);
this module has no executor config of its own.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault import workflow_reconcile
from gpu_fault.models import (
    IncidentState,
    PlanStatus,
    RecoveryPlan,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.workflow_reconcile import (
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
)
from tests._builders import fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 9, 6, 11, 0, tzinfo=timezone.utc)
INCIDENT_ID = "incident-apply-epoch"
BLOCKED_ID = "workflow-apply-epoch-blocked"
SUCCESSOR_ID = "workflow-apply-epoch-restored"
PLAN_ID = "plan-apply-epoch"


def _restored_state(store: InMemoryStore) -> None:
    store.save_plan(
        RecoveryPlan(
            plan_id=PLAN_ID,
            incident_id=INCIDENT_ID,
            attempt_id="attempt-a",
            trigger="test",
            runtime_profile_version="profile-v1",
            steps=[],
            workflow_request_id=BLOCKED_ID,
            status=PlanStatus.FAILED,
            created_at=NOW - timedelta(hours=2),
        )
    )
    store.save_workflow(
        workflow_request(
            BLOCKED_ID,
            INCIDENT_ID,
            status=WorkflowStatus.BLOCKED,
            fencing_token=7,
            execution_epoch=4,
            source_plan_id=PLAN_ID,
            official_action=WorkflowOperation.QUARANTINE.value,
            official_steps=[workflow_step(WorkflowOperation.QUARANTINE)],
            updated_at=NOW - timedelta(hours=1),
        )
    )
    store.save_workflow(
        workflow_request(
            SUCCESSOR_ID,
            INCIDENT_ID,
            status=WorkflowStatus.SUCCEEDED,
            fencing_token=7,
            predecessor_workflow_id=BLOCKED_ID,
            completed_operations=[WorkflowOperation.RESTORE_SCHEDULING],
            updated_at=NOW - timedelta(minutes=30),
        )
    )
    store.save_incident(
        fault_incident(
            INCIDENT_ID,
            "event-apply-epoch",
            state=IncidentState.RECOVERED,
            workflow_request_id=SUCCESSOR_ID,
            fencing_token=7,
            updated_at=NOW - timedelta(minutes=30),
        )
    )


class _RecordingStore(InMemoryStore):
    """Records the compare-and-set keys the apply path hands to the Store."""

    def __init__(self) -> None:
        super().__init__()
        self.reconcile_calls: list[dict[str, Any]] = []

    def reconcile_restored_workflow(self, *args: Any, **kwargs: Any):
        self.reconcile_calls.append(dict(kwargs))
        return super().reconcile_restored_workflow(*args, **kwargs)


def test_apply_compares_on_the_epoch_the_approval_bound_not_a_reread_updated_at() -> (
    None
):
    store = _RecordingStore()
    _restored_state(store)
    plan = build_workflow_reconcile_plan(store, [BLOCKED_ID], now=NOW)
    assert plan["items"][0]["execution_epoch"] == 4

    result = apply_workflow_reconcile_plan(
        store,
        workflow_ids=[BLOCKED_ID],
        expected_plan_sha256=plan["plan_sha256"],
        reference="CHG-EPOCH",
        now=NOW + timedelta(minutes=1),
    )

    assert result["applied_workflow_ids"] == [BLOCKED_ID]
    (call,) = store.reconcile_calls
    assert call["expected_fencing_token"] == 7
    assert call["expected_execution_epoch"] == 4
    assert call.get("expected_workflow_updated_at") is None, (
        "the apply must not compare on a timestamp the approval never covered"
    )


def test_apply_threads_the_callers_waiting_ttl_to_the_reservation_release(
    monkeypatch,
) -> None:
    store = InMemoryStore()
    _restored_state(store)
    plan = build_workflow_reconcile_plan(store, [BLOCKED_ID], now=NOW)
    seen: list[dict[str, Any]] = []

    def record(store_arg, workflow, *args: Any, **kwargs: Any) -> None:
        seen.append({"workflow": workflow.request_id, **kwargs})

    monkeypatch.setattr(
        workflow_reconcile, "release_unattempted_restart_reservations", record
    )

    apply_workflow_reconcile_plan(
        store,
        workflow_ids=[BLOCKED_ID],
        expected_plan_sha256=plan["plan_sha256"],
        reference="CHG-TTL",
        now=NOW + timedelta(minutes=1),
        waiting_ttl=timedelta(minutes=45),
    )

    (call,) = seen
    assert call["workflow"] == BLOCKED_ID
    assert call["waiting_ttl"] == timedelta(minutes=45)


def test_apply_without_a_waiting_ttl_keeps_waiting_reservations(monkeypatch) -> None:
    store = InMemoryStore()
    _restored_state(store)
    plan = build_workflow_reconcile_plan(store, [BLOCKED_ID], now=NOW)
    seen: list[dict[str, Any]] = []

    def record(store_arg, workflow, *args: Any, **kwargs: Any) -> None:
        seen.append(dict(kwargs))

    monkeypatch.setattr(
        workflow_reconcile, "release_unattempted_restart_reservations", record
    )

    apply_workflow_reconcile_plan(
        store,
        workflow_ids=[BLOCKED_ID],
        expected_plan_sha256=plan["plan_sha256"],
        reference="CHG-NO-TTL",
        now=NOW + timedelta(minutes=1),
    )

    (call,) = seen
    assert call.get("waiting_ttl") is None


def test_the_shipped_apply_script_hands_the_executors_waiting_cap_to_the_apply(
    monkeypatch,
) -> None:
    """The in-Pod script is the only caller that knows the executor config."""

    import io
    import json
    import sys
    from types import SimpleNamespace

    from gpu_fault import app as app_module
    from gpu_fault.admin import workflow_reconcile as admin_reconcile
    from gpu_fault.execution.config import ProductionExecutorConfig

    seen: list[dict[str, Any]] = []

    def fake_apply(
        store,
        *,
        workflow_ids: list[str],
        expected_plan_sha256: str,
        reference: str,
        waiting_ttl: timedelta | None = None,
    ) -> dict[str, Any]:
        seen.append(
            {
                "workflow_ids": workflow_ids,
                "expected_plan_sha256": expected_plan_sha256,
                "reference": reference,
                "waiting_ttl": waiting_ttl,
            }
        )
        return {"mode": "workflow-reconcile-apply", "applied_workflow_ids": []}

    config = ProductionExecutorConfig(
        enabled=False,
        executor_id="executor-test",
        allowed_operations=frozenset(),
        step_waiting_timeout_seconds=600,
        step_waiting_timeout_overrides={WorkflowOperation.RESTART_WORKLOAD: 900},
    )
    fake_context = SimpleNamespace(
        store=InMemoryStore(), production_executor_config=config
    )
    monkeypatch.setattr(
        app_module.ApplicationContext,
        "from_environment",
        classmethod(lambda cls: fake_context),
    )
    monkeypatch.setattr(workflow_reconcile, "apply_workflow_reconcile_plan", fake_apply)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "mode": "apply",
                    "workflow_ids": [BLOCKED_ID],
                    "plan_sha256": "b" * 64,
                    "reference": "CHG-SCRIPT",
                }
            )
        ),
    )

    exec(compile(admin_reconcile.RECONCILE_SCRIPT, "<workflow-reconcile>", "exec"), {})

    (call,) = seen
    assert call["workflow_ids"] == [BLOCKED_ID]
    assert call["reference"] == "CHG-SCRIPT"
    assert call["waiting_ttl"] == timedelta(seconds=900)
