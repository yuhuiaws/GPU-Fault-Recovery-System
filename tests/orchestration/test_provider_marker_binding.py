"""The finalized marker follows execution ownership after semantic association."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.app.ingest.faults import FaultIngestionService
from gpu_fault.models import WorkflowStatus
from gpu_fault.policy import ActionDisposition, FaultPolicyDecision, XidEvent
from gpu_fault.store import InMemoryStore, NotFoundError, SqliteStore
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.store.shared.errors import StaleWriteError
from gpu_fault.watcher import WorkloadPhase
from tests._builders import build_context
from tests.orchestration.test_provider_correlation_arbitration import (
    NOW,
    RESET,
    current_attempt,
    ingest_warning,
    live_reset_gpus,
    sbe_warning,
    xid,
)


@pytest.fixture(params=["memory", "sqlite"])
def binding_store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[ControlPlaneStore]:
    store = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "binding.db"))
    )
    store.save_profile(default_simulated_profile())
    try:
        yield store
    finally:
        if isinstance(store, SqliteStore):
            store.close()


def test_finalized_xid_marker_belongs_to_its_new_attempt_recovery(
    binding_store: ControlPlaneStore,
) -> None:
    correlation_store = binding_store
    context = build_context(store=correlation_store)
    correlation_store.save_attempt_observation(current_attempt())
    warning, prior = ingest_warning(context.orchestrator, sbe_warning())
    correlation_store.save_attempt_observation(
        current_attempt(
            observed_at=NOW + timedelta(seconds=5), phase=WorkloadPhase.STOPPED
        )
    )
    correlation_store.save_attempt_observation(
        current_attempt(
            "train-a002",
            observed_at=NOW + timedelta(seconds=20),
            started_at=NOW + timedelta(seconds=10),
        )
    )
    event = xid(
        "finalized-new-attempt", at=NOW + timedelta(seconds=21), attempt_id="train-a002"
    )
    service = FaultIngestionService(context)

    finalized = service.finalize_xid(event, context.policy.evaluate_xid(event))

    assert finalized.workflow_request_id is not None
    workflow = correlation_store.get_workflow(finalized.workflow_request_id)
    assert workflow.request_id != prior.request_id
    assert workflow.incident_id != warning.incident_id
    assert live_reset_gpus(workflow, RESET) == {"GPU-a"}
    persisted_marker = next(
        marker
        for marker in correlation_store.list_markers()
        if marker.marker_id == finalized.marker.marker_id
    )
    assert finalized.marker.incident_id == workflow.incident_id, (
        "the finalized decision retained the old attempt's marker association"
    )
    assert persisted_marker.incident_id == workflow.incident_id, (
        "marker closure and completion correlation still point at the old attempt"
    )
    assert correlation_store.get_xid_policy_decision(event.event_id) == finalized, (
        "the persisted decision must retain the corrected marker binding"
    )
    assert any(
        f"Provider marker association with incident {warning.incident_id}" in reason
        for reason in finalized.reasons
    ), "binding execution ownership must preserve its evidence association"
    assert correlation_store.get_incident(warning.incident_id) == warning


def test_failed_incident_ingestion_does_not_publish_a_marker(
    binding_store: ControlPlaneStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = build_context(store=binding_store)
    service = FaultIngestionService(context)
    event = xid("failed-incident", active=False)
    decision = context.policy.evaluate_xid(event)

    def fail(*args: object) -> None:
        raise RuntimeError("incident persistence failed")

    monkeypatch.setattr(context.orchestrator, "ingest", fail)
    with pytest.raises(RuntimeError, match="incident persistence failed"):
        service.finalize_xid(event, decision)

    assert binding_store.list_markers() == [], (
        "no marker may claim ownership before the incident exists"
    )
    assert binding_store.get_xid_policy_decision(event.event_id) is None, (
        "failed incident ingestion must not finalize the policy decision"
    )
    assert binding_store.get_incident_by_event(event.event_id) is None, (
        "the failed write must not mark the event as handled"
    )


def test_marker_write_failure_refuses_finalization_and_retries_same_workflow(
    binding_store: ControlPlaneStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = build_context(store=binding_store)
    service = FaultIngestionService(context)
    event = xid("failed-marker", active=False)
    decision = context.policy.evaluate_xid(event)

    def fail(*args: object) -> None:
        raise RuntimeError("marker persistence failed")

    with monkeypatch.context() as failure:
        failure.setattr(context.completion, "add_marker", fail)
        with pytest.raises(RuntimeError, match="marker persistence failed"):
            service.finalize_xid(event, decision)

    incident = binding_store.get_incident_by_event(event.event_id)
    assert incident is not None, "the persisted incident remains available for retry"
    assert binding_store.list_markers() == [], (
        "the failed marker write must not publish"
    )
    assert binding_store.get_xid_policy_decision(event.event_id) is None, (
        "a marker write failure must not publish a finalized policy decision"
    )
    finalized = service.finalize_xid(event, decision)
    assert finalized.incident_id == incident.incident_id, "retry reuses its incident"
    assert finalized.workflow_request_id == incident.workflow_request_id, (
        "retry must not create a second recovery"
    )
    assert finalized.marker.incident_id == incident.incident_id, (
        "the retried marker must bind to its persisted incident"
    )
    assert binding_store.list_markers() == [finalized.marker], (
        "retry must publish exactly one correctly bound marker"
    )


def observation_event(
    event_id: str, *, seconds: int = 0, source: str = "KERNEL_LOG"
) -> XidEvent:
    return xid(event_id, code=94, at=NOW + timedelta(seconds=seconds)).model_copy(
        update={
            "event_source": source,
            "source_boot_id": "boot-a",
            "source_monotonic_us": 10_000_000 + (5_000_000 if seconds else 0),
        }
    )


@pytest.mark.parametrize(
    ("source", "seconds"),
    [("KERNEL_LOG", 1), ("FABRIC_MANAGER_LOG", 1), ("KERNEL_LOG", 3600)],
    ids=["same-source", "cross-source", "wall-clock-jump"],
)
def test_equivalent_provider_observations_share_an_unstarted_plan(
    binding_store: ControlPlaneStore, source: str, seconds: int
) -> None:
    context = build_context(store=binding_store)
    service = FaultIngestionService(context)
    first = service.ingest_xid(observation_event("first-observation"))
    assert first.workflow_request_id is not None, "the first XID requires a plan"
    workflow = binding_store.get_workflow(first.workflow_request_id)

    second = service.ingest_xid(
        observation_event("second-observation", seconds=seconds, source=source)
    )

    assert second.duplicate, "source-aware marker correlation must remain visible"
    assert second.incident_id == first.incident_id, (
        "equivalent observations must not create independent recovery incidents"
    )
    assert second.workflow_request_id == workflow.request_id, (
        "equivalent observations must not duplicate an unstarted recovery"
    )
    assert binding_store.get_workflow(workflow.request_id) == workflow, (
        "coalescing observations must not rewrite the pending plan or its history"
    )
    assert second.marker.incident_id == first.incident_id, (
        "the second marker must follow the shared execution owner"
    )
    if seconds == 3600:
        with pytest.raises(NotFoundError):
            binding_store.get_xid_event(first.event_id)


@pytest.mark.parametrize(
    "changes",
    [
        {"gpu_uuid": "GPU-b"},
        {"pci_bdf": "0000:60:00.0"},
        {"source_boot_id": "boot-b"},
        {"runtime_profile_version": "unknown-profile"},
        {"product": "H100"},
        {"driver_branch": 580},
        {"cuda_version": "13.0"},
        {"affected_workload_ids": ["training/pytorchjob/another-job"]},
        {"job_id": "train", "attempt_id": "train-a002"},
        {"pod_uid": "replacement-pod"},
        {"synthetic": True},
        {"xid": 48},
    ],
    ids=[
        "gpu",
        "pci",
        "boot",
        "profile",
        "product",
        "driver",
        "cuda",
        "workload",
        "attempt",
        "pod",
        "synthetic",
        "action",
    ],
)
def test_semantic_drift_does_not_reuse_an_independent_provider_plan(
    binding_store: ControlPlaneStore, changes: dict[str, object]
) -> None:
    context = build_context(store=binding_store)
    service = FaultIngestionService(context)
    first = service.ingest_xid(observation_event("original-observation"))
    assert first.workflow_request_id is not None, "the initial fault requires a plan"
    workflow = binding_store.get_workflow(first.workflow_request_id)
    event = observation_event("changed-observation", seconds=1).model_copy(
        update=changes
    )

    second = service.ingest_xid(event)

    assert second.incident_id != first.incident_id, (
        "a semantic or identity change must reach independent candidate arbitration"
    )
    assert second.workflow_request_id != workflow.request_id, (
        "the prior marker must not count as coverage for the changed candidate"
    )
    assert binding_store.get_workflow(workflow.request_id) == workflow, (
        "the existing recovery must not be overwritten by broad marker association"
    )


@pytest.mark.parametrize(
    "change",
    [
        "event",
        "incident",
        "marker-incident",
        "workflow",
        "policy",
        "disposition",
        "trust",
        "scope",
    ],
)
def test_retained_policy_must_match_the_correlated_execution_owner(
    binding_store: ControlPlaneStore, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    context = build_context(store=binding_store)
    service = FaultIngestionService(context)
    first = service.ingest_xid(observation_event("policy-owner"))
    updates = {
        "event": {"event_id": "other-event"},
        "incident": {"incident_id": "inc-other"},
        "workflow": {"workflow_request_id": "other-workflow"},
        "policy": {"policy_version": "other-policy"},
        "disposition": {"disposition": ActionDisposition.MONITOR_ONLY},
    }
    marker_updates = {
        "marker-incident": {"incident_id": "inc-other"},
        "trust": {"trusted": False},
        "scope": {
            "scope": first.marker.scope.model_copy(update={"gpu_uuids": ["GPU-b"]})
        },
    }
    changed = first.model_copy(
        update=updates.get(
            change,
            {"marker": first.marker.model_copy(update=marker_updates.get(change, {}))},
        )
    )
    get_policy = binding_store.get_xid_policy_decision

    def policy(event_id: str) -> FaultPolicyDecision | None:
        return changed if event_id == first.event_id else get_policy(event_id)

    monkeypatch.setattr(binding_store, "get_xid_policy_decision", policy)
    second = service.ingest_xid(observation_event("new-policy-evidence", seconds=1))

    assert second.incident_id != first.incident_id, (
        "an incoherent retained decision must not prove execution ownership"
    )
    assert second.workflow_request_id != first.workflow_request_id, (
        "policy, identity and marker drift must re-enter candidate arbitration"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"execution_owner_id": "executor-a", "execution_epoch": 1},
        {"completed_step_indexes": [0]},
        {"superseded_step_indexes": [0]},
        {"status": WorkflowStatus.SUCCEEDED},
        {"workload_withdrawn_at": NOW},
        {"lifetime_deadline_at": NOW},
    ],
    ids=["leased", "executed", "superseded-step", "terminal", "withdrawn", "expired"],
)
def test_progressed_provider_plan_cannot_absorb_another_observation(
    binding_store: ControlPlaneStore, changes: dict[str, object]
) -> None:
    context = build_context(store=binding_store)
    service = FaultIngestionService(context)
    first = service.ingest_xid(observation_event("before-progress"))
    assert first.workflow_request_id is not None, "the initial fault requires a plan"
    progressed = binding_store.amend_workflow(first.workflow_request_id, changes)

    second = service.ingest_xid(observation_event("after-progress", seconds=1))

    assert second.workflow_request_id != progressed.request_id, (
        "a newly observed fault cannot reuse execution coverage that already advanced"
    )
    assert binding_store.get_workflow(progressed.request_id) == progressed, (
        "a new observation must retain the existing execution history"
    )


@pytest.mark.parametrize("guard", ["save_incident", "save_workflow"])
def test_correlated_event_link_waits_for_both_snapshot_guards(
    binding_store: ControlPlaneStore, monkeypatch: pytest.MonkeyPatch, guard: str
) -> None:
    context = build_context(store=binding_store)
    service = FaultIngestionService(context)
    first = service.ingest_xid(observation_event("guarded-first"))
    event = observation_event("guarded-second", seconds=1)

    def refuse(*args: object, **kwargs: object) -> None:
        raise StaleWriteError("correlation snapshot changed")

    with monkeypatch.context() as failure:
        failure.setattr(binding_store, guard, refuse)
        with pytest.raises(StaleWriteError, match="correlation snapshot changed"):
            service.ingest_xid(event)

    assert binding_store.get_incident_by_event(event.event_id) is None, (
        "a failed incident or workflow CAS must not publish a handled-event link"
    )
    assert all(
        marker.marker_id != f"marker-{event.event_id}"
        for marker in binding_store.list_markers()
    ), "a failed execution binding must not publish the new marker"
    second = service.ingest_xid(event)
    assert second.workflow_request_id == first.workflow_request_id, (
        "the retried observation must coalesce after both guards succeed"
    )
