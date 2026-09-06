"""The deadline watchdog acts under the executor's own identity, PENDING age is
observed rather than terminalized, and the retired generations waiting on an
operator are counted.

FINAL-建议汇总 F-A5 (P0-22A, P2-70H, P1-58D, P1-53E, P1-53G). The watchdog
claimed a stuck workflow as ``confirm_cluster_name`` -- one identity shared by
every replica in the region -- so nothing in the store could tell two
watchdogs apart. PENDING had no watchdog at all, which is why "running for
ever" had a backstop and "pending for ever" did not; the fix is a gauge and a
warning, never a terminal write. The records the retired-generation sweep
refuses to close had no gauge either.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store.shared.errors import WorkflowLeaseError
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome

OP = WorkflowOperation.FREEZE_EVIDENCE
STOP = WorkflowOperation.STOP_WORKLOADS


class _ClaimRecordingStore:
    def __init__(self, store) -> None:
        self._store = store
        self.claimants: list[str] = []

    def claim_workflow(self, request_id, executor_id, *args, **kwargs):
        self.claimants.append(executor_id)
        return self._store.claim_workflow(request_id, executor_id, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._store, name)


class _RecordingExecutor:
    def __init__(self) -> None:
        self.requested: list[str] = []
        self.config = SimpleNamespace(executor_id="executor-recording")

    def execute(self, request_id, request):
        self.requested.append(request_id)
        raise WorkflowLeaseError("recorded only")


def _save(store, request_id: str, **values):
    incident_id = f"inc-{request_id}"
    now = datetime.now(timezone.utc)
    incident = fault_incident(
        incident_id,
        f"event-{request_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=values.pop("incident_points_at", request_id),
        fencing_token=values.pop("incident_token", 1),
        created_at=now,
        updated_at=now,
    )
    workflow = workflow_request(
        request_id,
        incident_id,
        status=values.pop("status", WorkflowStatus.PENDING),
        fencing_token=values.pop("fencing_token", 1),
        official_steps=values.pop("official_steps", [workflow_step(OP)]),
        created_at=values.pop("created_at", now),
        updated_at=values.pop("updated_at", now),
        **values,
    )
    if incident.workflow_request_id == request_id:
        store.save_incident_and_workflow(incident, workflow)
    else:
        # The incident names another generation; the paired write refuses
        # that shape on purpose, so the two rows are written one by one.
        store.save_incident(incident)
        store.save_workflow(workflow)
    return workflow


def test_the_watchdog_claims_as_the_executor_not_as_the_cluster() -> None:
    inner = build_store()
    now = datetime.now(timezone.utc)
    _save(
        inner,
        "wf-stuck",
        status=WorkflowStatus.RUNNING,
        execution_deadline=now - timedelta(minutes=10),
    )
    store = _ClaimRecordingStore(inner)
    adapter = FakeAdapter({OP: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [adapter], [OP], executor_id="executor-a"),
        WorkflowDispatcherConfig(
            enabled=True,
            batch_size=10,
            max_workers=1,
            confirm_cluster_name="cluster-shared-by-every-replica",
        ),
    )

    report = dispatcher.run_once()

    assert report.failed == 1
    assert inner.get_workflow("wf-stuck").status is WorkflowStatus.FAILED
    assert store.claimants, "the watchdog must take the lease before it writes"
    assert set(store.claimants) == {"executor-a"}, store.claimants


def test_pending_age_is_observed_and_warned_but_never_terminalized(caplog) -> None:
    store = build_store()
    now = datetime.now(timezone.utc)
    _save(store, "wf-stale", created_at=now - timedelta(hours=2))
    _save(store, "wf-fresh", created_at=now - timedelta(seconds=5))
    executor = _RecordingExecutor()
    dispatcher = WorkflowDispatcher(
        store,
        executor,  # type: ignore[arg-type]
        WorkflowDispatcherConfig(
            enabled=True, batch_size=10, max_workers=1, pending_age_warning_seconds=600
        ),
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.execution.dispatcher"):
        dispatcher.run_once()

    assert dispatcher.pending_age_seconds_max >= 7000, (
        dispatcher.pending_age_seconds_max
    )
    assert dispatcher.oldest_pending_workflow_id == "wf-stale"
    assert dispatcher.pending_age_warnings_total == 1
    assert store.get_workflow("wf-stale").status is WorkflowStatus.PENDING, (
        "the PENDING watchdog observes only"
    )
    warnings = [
        record for record in caplog.records if "wf-stale" in record.getMessage()
    ]
    assert warnings, "the oldest PENDING row past the threshold is named in a warning"


def test_pending_age_below_the_threshold_is_recorded_without_a_warning() -> None:
    store = build_store()
    _save(
        store, "wf-fresh", created_at=datetime.now(timezone.utc) - timedelta(seconds=5)
    )
    dispatcher = WorkflowDispatcher(
        store,
        _RecordingExecutor(),  # type: ignore[arg-type]
        WorkflowDispatcherConfig(
            enabled=True, batch_size=10, max_workers=1, pending_age_warning_seconds=600
        ),
    )

    dispatcher.run_once()

    assert 0 <= dispatcher.pending_age_seconds_max < 600
    assert dispatcher.pending_age_warnings_total == 0


def test_retired_generations_awaiting_an_operator_are_counted_each_tick() -> None:
    store = build_store()
    now = datetime.now(timezone.utc)
    steps = [
        workflow_step(OP, node_ids=["node-a"]),
        workflow_step(STOP, node_ids=["node-a"]),
    ]
    # Generation 1 already stopped workloads; its incident moved on to
    # generation 4 and names another workflow. Nothing may close this record
    # but an operator (see test_retired_generation.py).
    _save(
        store,
        "wf-retired",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        incident_token=4,
        incident_points_at="wf-current",
        official_steps=steps,
        completed_step_indexes=[0, 1],
        completed_operations=[OP, STOP],
        execution_owner_id="executor-b",
        execution_lease_expires_at=now + timedelta(minutes=3),
    )
    store.save_workflow(
        workflow_request(
            "wf-current",
            "inc-wf-retired",
            status=WorkflowStatus.PENDING,
            fencing_token=4,
            official_steps=[workflow_step(WorkflowOperation.VALIDATE_GPU)],
        )
    )
    dispatcher = WorkflowDispatcher(
        store,
        _RecordingExecutor(),  # type: ignore[arg-type]
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )

    dispatcher.run_once()
    assert dispatcher.retired_generation_awaiting_operator == 1

    store.save_workflow(
        store.get_workflow("wf-retired").model_copy(
            update={"status": WorkflowStatus.SUCCEEDED}
        )
    )
    dispatcher.run_once()
    assert dispatcher.retired_generation_awaiting_operator == 0, (
        "a gauge, re-derived every tick"
    )


def test_the_production_config_reads_the_pending_age_warning(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_WORKFLOW_PENDING_AGE_WARNING_SECONDS", "1234")

    config = WorkflowDispatcherConfig.from_environment(executor_enabled=True)

    assert config.pending_age_warning_seconds == 1234.0
