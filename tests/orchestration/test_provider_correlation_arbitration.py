"""Provider marker association must not suppress a new recovery candidate."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryAction,
    Severity,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.policy import (
    FaultPolicyDecision,
    GpuFaultPolicyEngine,
    SxidClassification,
    SxidEvent,
    SxidLinkScope,
    XidEvent,
)
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.watcher import AttemptObservation, WorkloadPhase
from tests._builders import (
    attempt_observation,
    container_observation,
    node_health_finding,
    workflow_step_execution,
)

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
WORKLOAD = "training/pytorchjob/train"
RESET = WorkflowOperation.RESET_GPU
RESET_ALL = WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES


@pytest.fixture(params=["memory", "sqlite"])
def correlation_store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[ControlPlaneStore]:
    store = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "correlation.db"))
    )
    store.save_profile(default_simulated_profile())
    try:
        yield store
    finally:
        if isinstance(store, SqliteStore):
            store.close()


def current_attempt(
    attempt_id: str = "train-a001",
    *,
    observed_at: datetime = NOW,
    started_at: datetime = NOW - timedelta(minutes=1),
    phase: WorkloadPhase = WorkloadPhase.RUNNING,
) -> AttemptObservation:
    return attempt_observation(
        "train",
        attempt_id,
        observed_at,
        started_at=started_at,
        workload_phase=phase,
        containers=[
            container_observation(
                f"pod-{attempt_id}",
                f"trainer-{attempt_id}",
                0,
                "node-a",
                gpu_uuids=["GPU-a", "GPU-b"],
                cgroup_path=f"/kubepods/pod-{attempt_id}/trainer",
            )
        ],
        workload_ids=[WORKLOAD],
        restart_budget=3,
    )


def sbe_warning(
    *, active: bool = True, event_id: str = "dcgm-sbe", at: datetime = NOW
) -> NodeHealthFinding:
    return node_health_finding(
        event_id,
        event_id,
        observed_at=at,
        category=NodeHealthCategory.GPU,
        severity=Severity.WARNING,
        reason="DCGM ECC single-bit counter increased",
        metric_name="ecc_sbe_volatile_total",
        gpu_uuids=["GPU-a"],
        pci_bdf="0000:59:00.0",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        policy_source="SITE_DCGM_METRIC",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE if active else WorkloadState.IDLE,
        affected_workload_ids=[WORKLOAD] if active else [],
    )


def xid(
    event_id: str,
    *,
    code: int = 48,
    at: datetime = NOW + timedelta(seconds=1),
    active: bool = True,
    attempt_id: str | None = None,
    workload_state: WorkloadState | None = None,
) -> XidEvent:
    return XidEvent(
        event_id=event_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=at,
        collected_at=at,
        event_source="NVIDIA_KERNEL_LOG",
        xid=code,
        gpu_uuid="GPU-a",
        pci_bdf="0000:59:00.0",
        product="H200",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
        workload_state=workload_state
        or (WorkloadState.ACTIVE if active else WorkloadState.IDLE),
        affected_workload_ids=[WORKLOAD] if active else [],
        job_id="train" if attempt_id else None,
        attempt_id=attempt_id,
    )


def sxid(
    event_id: str, *, trunk: bool = False, active: bool = True, at: datetime = NOW
) -> SxidEvent:
    return SxidEvent(
        event_id=event_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=at,
        event_source="FABRIC_MANAGER_LOG",
        sxid=11001,
        classification=SxidClassification.FATAL,
        classification_source="NVIDIA_FABRIC_MANAGER_CATALOG",
        link_scope=SxidLinkScope.TRUNK if trunk else SxidLinkScope.ACCESS,
        link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
        fabric_partition="cluster-a/node-a/local-nvswitch" if trunk else None,
        participating_gpu_uuids=["GPU-a", "GPU-b"] if trunk else ["GPU-a"],
        product="H200",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE if active else WorkloadState.IDLE,
        affected_workload_ids=[WORKLOAD] if active else [],
    )


def provider_decision(
    orchestrator: IncidentOrchestrator, event: XidEvent | SxidEvent
) -> FaultPolicyDecision:
    policy = GpuFaultPolicyEngine()
    decision = (
        policy.evaluate_xid(event)
        if isinstance(event, XidEvent)
        else policy.evaluate_sxid(event)
    )
    return orchestrator.correlate_provider_event(decision, cluster_id=event.cluster_id)


def ingest_provider(
    orchestrator: IncidentOrchestrator, event: XidEvent | SxidEvent
) -> tuple[FaultIncident, WorkflowRequest]:
    decision = provider_decision(orchestrator, event)
    incident, workflow = orchestrator.ingest(event, decision)
    assert workflow is not None, "a recovery candidate must retain a workflow"
    orchestrator.store.add_marker(
        decision.marker.model_copy(update={"incident_id": incident.incident_id})
    )
    return incident, workflow


def ingest_warning(
    orchestrator: IncidentOrchestrator, finding: NodeHealthFinding
) -> tuple[FaultIncident, WorkflowRequest]:
    incident, workflow = orchestrator.ingest_node_health(finding)
    assert workflow is not None, "the SBE finding must retain its diagnostic plan"
    orchestrator.store.add_marker(
        finding.marker().model_copy(update={"incident_id": incident.incident_id})
    )
    return incident, workflow


def live_reset_gpus(
    workflow: WorkflowRequest, operation: WorkflowOperation
) -> set[str]:
    resolved = set(workflow.completed_step_indexes) | set(
        workflow.superseded_step_indexes
    )
    return {
        gpu
        for index, step in enumerate(workflow.official_steps)
        if index not in resolved
        and step.operation is operation
        and "node-a" in step.node_ids
        for gpu in step.parameters.get("gpu_uuids_by_node", {}).get(
            "node-a", step.gpu_uuids
        )
    }


@pytest.mark.parametrize("active", [False, True], ids=["idle", "attempt"])
def test_sbe_marker_cannot_swallow_xid48_reset(
    correlation_store: ControlPlaneStore, active: bool
) -> None:
    if active:
        correlation_store.save_attempt_observation(current_attempt())
    orchestrator = IncidentOrchestrator(correlation_store)
    warning, _ = ingest_warning(orchestrator, sbe_warning(active=active))
    event = xid("after-sbe", active=active)
    decision = provider_decision(orchestrator, event)
    assert decision.marker.incident_id == warning.incident_id

    incident, workflow = orchestrator.ingest(event, decision)

    assert workflow is not None, "semantic association discarded the XID48 workflow"
    assert live_reset_gpus(workflow, RESET) == {"GPU-a"}, (
        "the SBE diagnostic marker suppressed the required XID48 reset"
    )
    assert workflow.status is WorkflowStatus.PENDING
    assert incident.effective_action is RecoveryAction.RESET_GPU
    assert correlation_store.get_incident_by_event(event.event_id) == incident
    assert orchestrator.ingest(event, decision) == (incident, workflow)


def test_later_sbe_warning_preserves_the_xid48_reset(
    correlation_store: ControlPlaneStore,
) -> None:
    correlation_store.save_attempt_observation(current_attempt())
    orchestrator = IncidentOrchestrator(correlation_store)
    first, workflow = ingest_provider(orchestrator, xid("before-sbe", at=NOW))

    incident, merged = ingest_warning(
        orchestrator, sbe_warning(at=NOW + timedelta(seconds=1))
    )

    assert incident.incident_id == first.incident_id
    assert merged.request_id == workflow.request_id
    assert live_reset_gpus(merged, RESET) == {"GPU-a"}
    assert incident.effective_action is RecoveryAction.RESET_GPU


@pytest.mark.parametrize("active", [False, True], ids=["idle", "attempt"])
def test_access_marker_does_not_hide_trunk_reset_kind_or_gpu_b(
    correlation_store: ControlPlaneStore, active: bool
) -> None:
    if active:
        correlation_store.save_attempt_observation(current_attempt())
    orchestrator = IncidentOrchestrator(correlation_store)
    first, access = ingest_provider(orchestrator, sxid("access-a", active=active))
    event = sxid("trunk-ab", trunk=True, active=active, at=NOW + timedelta(seconds=1))
    decision = provider_decision(orchestrator, event)
    assert decision.marker.incident_id == first.incident_id

    incident, workflow = orchestrator.ingest(event, decision)

    assert workflow is not None
    assert live_reset_gpus(workflow, RESET_ALL) == {"GPU-a", "GPU-b"}, (
        "an intersecting ACCESS marker must not cover a TRUNK reset of both GPUs"
    )
    assert not live_reset_gpus(workflow, RESET), "a pending ACCESS reset survived"
    assert workflow.request_id == access.request_id
    assert incident.gpu_uuids == ["GPU-a", "GPU-b"]


@pytest.mark.parametrize(
    "step_status", [WorkflowStepStatus.WAITING, WorkflowStepStatus.SUCCEEDED]
)
@pytest.mark.parametrize("active", [False, True], ids=["idle", "attempt"])
def test_correlated_trunk_candidate_never_rewrites_a_started_access_reset(
    correlation_store: ControlPlaneStore, step_status: WorkflowStepStatus, active: bool
) -> None:
    if active:
        correlation_store.save_attempt_observation(current_attempt())
    orchestrator = IncidentOrchestrator(correlation_store)
    _, access = ingest_provider(orchestrator, sxid("access-running", active=active))
    reset_index = next(
        index
        for index, step in enumerate(access.official_steps)
        if step.operation is RESET
    )
    started = correlation_store.amend_workflow(
        access.request_id,
        {
            "status": WorkflowStatus.RUNNING,
            "execution_owner_id": "executor-a",
            "step_executions": [
                workflow_step_execution(reset_index, RESET, step_status)
            ],
            "completed_step_indexes": (
                [reset_index] if step_status is WorkflowStepStatus.SUCCEEDED else []
            ),
        },
    )
    frozen = started.official_steps[reset_index].model_copy(deep=True)
    event = sxid(
        "trunk-after-start", trunk=True, active=active, at=NOW + timedelta(seconds=1)
    )

    _, workflow = ingest_provider(orchestrator, event)

    retained = correlation_store.get_workflow(access.request_id).official_steps[
        reset_index
    ]
    # Adding a DAG branch changes graph metadata, not an issued command's scope.
    assert retained.operation is frozen.operation
    assert retained.node_ids == frozen.node_ids
    assert retained.gpu_uuids == frozen.gpu_uuids
    assert retained.workload_ids == frozen.workload_ids
    assert retained.execution_owner == frozen.execution_owner
    assert retained.parameters == frozen.parameters, "the issued reset was rewritten"
    assert live_reset_gpus(workflow, RESET_ALL) == {"GPU-a", "GPU-b"}
    if workflow.request_id != access.request_id:
        assert workflow.predecessor_workflow_id == access.request_id
    assert (
        orchestrator.ingest(event, provider_decision(orchestrator, event))[1]
        == workflow
    )


@pytest.mark.parametrize(
    "status",
    [WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.SUPERSEDED],
)
def test_terminal_marker_incumbent_does_not_count_as_new_reset_coverage(
    correlation_store: ControlPlaneStore, status: WorkflowStatus
) -> None:
    correlation_store.save_attempt_observation(current_attempt())
    orchestrator = IncidentOrchestrator(correlation_store)
    first, workflow = ingest_provider(orchestrator, xid("settled", at=NOW))
    terminal = correlation_store.amend_workflow(workflow.request_id, {"status": status})

    incident, successor = ingest_provider(orchestrator, xid("after-terminal"))

    assert successor.request_id != workflow.request_id
    assert incident.incident_id != first.incident_id
    assert live_reset_gpus(successor, RESET) == {"GPU-a"}
    assert correlation_store.get_workflow(terminal.request_id) == terminal
    assert correlation_store.get_incident(first.incident_id) == first


@pytest.mark.parametrize("explicit", [False, True], ids=["inferred", "explicit"])
def test_fresh_new_attempt_xid48_does_not_bind_to_old_attempt_warning(
    correlation_store: ControlPlaneStore, explicit: bool
) -> None:
    correlation_store.save_attempt_observation(current_attempt())
    orchestrator = IncidentOrchestrator(correlation_store)
    warning, prior = ingest_warning(orchestrator, sbe_warning())
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
        "new-attempt-reset",
        at=NOW + timedelta(seconds=21),
        attempt_id="train-a002" if explicit else None,
    )
    decision = provider_decision(orchestrator, event)
    assert decision.marker.incident_id == warning.incident_id

    incident, workflow = orchestrator.ingest(event, decision)

    assert workflow is not None
    assert workflow.request_id != prior.request_id
    assert incident.incident_id != warning.incident_id
    assert incident.attempt_id == "train-a002"
    assert live_reset_gpus(workflow, RESET) == {"GPU-a"}
    restart = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    assert restart.parameters["source_attempt_id"] == "train-a002"
    assert correlation_store.get_workflow(prior.request_id) == prior
    assert correlation_store.get_incident(warning.incident_id) == warning
    assert orchestrator.ingest(event, decision) == (incident, workflow)


def test_stale_semantic_event_is_fenced_without_overwriting_the_associated_incident(
    correlation_store: ControlPlaneStore,
) -> None:
    orchestrator = IncidentOrchestrator(correlation_store)
    warning, prior = ingest_warning(orchestrator, sbe_warning(active=False))
    correlation_store.save_attempt_observation(
        current_attempt(
            "train-a002",
            observed_at=NOW + timedelta(seconds=20),
            started_at=NOW + timedelta(seconds=10),
        )
    )
    event = xid("stale-memory-report", code=63, at=NOW + timedelta(seconds=1))
    decision = provider_decision(orchestrator, event)
    assert decision.marker.incident_id == warning.incident_id

    incident, workflow = orchestrator.ingest(event, decision)

    assert workflow is None
    assert incident.state is IncidentState.RECOVERED
    assert incident.incident_id != warning.incident_id
    assert any("Ignored stale XID 63" in reason for reason in incident.reasons), (
        "stale evidence must retain the generation-fence reason"
    )
    assert correlation_store.get_incident(warning.incident_id) == warning
    assert correlation_store.get_workflow(prior.request_id) == prior
    assert orchestrator.ingest(event, decision) == (incident, None)


@pytest.mark.parametrize("first_code", [48, 64])
def test_exact_companion_still_shares_one_reset_with_aggregation_disabled(
    correlation_store: ControlPlaneStore, first_code: int
) -> None:
    orchestrator = IncidentOrchestrator(
        correlation_store, multi_node_aggregation_window_seconds=0
    )
    first_event = xid("companion-first", code=first_code, active=False)
    first, workflow = ingest_provider(orchestrator, first_event)
    second_event = xid(
        "companion-second",
        code=64 if first_code == 48 else 48,
        at=NOW + timedelta(seconds=2),
        active=False,
    )
    decision = GpuFaultPolicyEngine().evaluate_xid(
        second_event, companion_events=[first_event]
    )
    assert decision.correlated_event_id == first_event.event_id
    decision = orchestrator.correlate_provider_event(
        decision, cluster_id=second_event.cluster_id
    )

    incident, companion = orchestrator.ingest(second_event, decision)

    assert incident.incident_id == first.incident_id
    assert companion is not None
    assert companion.request_id == workflow.request_id
    assert len(correlation_store.list_workflows()) == 1
    assert orchestrator.ingest(second_event, decision) == (incident, companion)


def test_missing_correlated_workflow_cannot_bypass_the_unknown_workload_gate(
    correlation_store: ControlPlaneStore,
) -> None:
    orchestrator = IncidentOrchestrator(correlation_store)
    first, _ = ingest_provider(orchestrator, xid("missing-owner", active=False))
    dangling = first.model_copy(update={"workflow_request_id": "workflow-vanished"})
    correlation_store.save_incident(dangling, expected=first)
    event = xid(
        "unknown-after-loss", active=False, workload_state=WorkloadState.UNKNOWN
    )

    incident, workflow = ingest_provider(orchestrator, event)

    assert workflow.status in {WorkflowStatus.SAFETY_PENDING, WorkflowStatus.BLOCKED}
    assert "node workload state is UNKNOWN" in workflow.blocked_reasons
    assert incident.incident_id != first.incident_id
    assert correlation_store.get_incident(first.incident_id) == dangling
    assert orchestrator.ingest(event, provider_decision(orchestrator, event)) == (
        incident,
        workflow,
    )


@pytest.mark.parametrize("active", [False, True], ids=["idle", "attempt"])
def test_later_access_marker_does_not_downgrade_the_trunk_reset(
    correlation_store: ControlPlaneStore, active: bool
) -> None:
    if active:
        correlation_store.save_attempt_observation(current_attempt())
    orchestrator = IncidentOrchestrator(correlation_store)
    first, trunk = ingest_provider(
        orchestrator, sxid("trunk-first", trunk=True, active=active)
    )

    incident, workflow = ingest_provider(
        orchestrator,
        sxid("access-second", active=active, at=NOW + timedelta(seconds=1)),
    )

    assert incident.incident_id == first.incident_id
    assert workflow.request_id == trunk.request_id
    assert live_reset_gpus(workflow, RESET_ALL) == {"GPU-a", "GPU-b"}
    assert not live_reset_gpus(workflow, RESET), "the TRUNK reset was downgraded"
    assert workflow.merge_revision > trunk.merge_revision, (
        "semantic association bypassed the Store merge"
    )


def test_execution_progress_after_marker_lookup_is_seen_by_atomic_arbitration(
    correlation_store: ControlPlaneStore,
) -> None:
    correlation_store.save_attempt_observation(current_attempt())
    orchestrator = IncidentOrchestrator(correlation_store)
    _, access = ingest_provider(orchestrator, sxid("access-before-lookup"))
    event = sxid("trunk-after-lookup", trunk=True, at=NOW + timedelta(seconds=1))
    decision = provider_decision(orchestrator, event)
    reset_index = next(
        index
        for index, step in enumerate(access.official_steps)
        if step.operation is RESET
    )
    progressed = correlation_store.amend_workflow(
        access.request_id,
        {
            "status": WorkflowStatus.RUNNING,
            "execution_owner_id": "another-executor",
            "execution_epoch": access.execution_epoch + 1,
            "step_executions": [
                workflow_step_execution(reset_index, RESET, WorkflowStepStatus.WAITING)
            ],
        },
    )

    incident, workflow = IncidentOrchestrator(correlation_store).ingest(event, decision)

    assert workflow is not None
    retained = correlation_store.get_workflow(access.request_id)
    assert retained.execution_epoch == progressed.execution_epoch
    assert retained.step_executions == progressed.step_executions
    assert retained.official_steps[reset_index].gpu_uuids == ["GPU-a"]
    assert retained.official_steps[reset_index].operation is RESET
    assert live_reset_gpus(workflow, RESET_ALL) == {"GPU-a", "GPU-b"}
    assert workflow.merge_revision > access.merge_revision
    assert correlation_store.get_incident_by_event(event.event_id) == incident


def test_failed_candidate_merge_does_not_mark_the_event_handled_and_retry_is_idempotent(
    correlation_store: ControlPlaneStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    correlation_store.save_attempt_observation(current_attempt())
    orchestrator = IncidentOrchestrator(correlation_store)
    first, access = ingest_provider(orchestrator, sxid("access-before-abort"))
    event = sxid("trunk-retried", trunk=True, at=NOW + timedelta(seconds=1))
    decision = provider_decision(orchestrator, event)
    merge = correlation_store.merge_attempt_fault_workflow

    def abort_merge(
        group_key: str,
        event_id: str,
        builder: Callable[
            [FaultIncident | None, WorkflowRequest | None],
            tuple[FaultIncident, WorkflowRequest],
        ],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        def abort_after_build(
            incident: FaultIncident | None, workflow: WorkflowRequest | None
        ) -> tuple[FaultIncident, WorkflowRequest]:
            candidate, planned = builder(incident, workflow)
            assert live_reset_gpus(planned, RESET_ALL) == {"GPU-a", "GPU-b"}
            assert candidate.workflow_request_id == planned.request_id
            raise RuntimeError("injected failure before atomic merge commit")

        return merge(group_key, event_id, abort_after_build)

    monkeypatch.setattr(correlation_store, "merge_attempt_fault_workflow", abort_merge)
    with pytest.raises(RuntimeError, match="before atomic merge commit"):
        orchestrator.ingest(event, decision)
    assert correlation_store.get_incident_by_event(event.event_id) is None
    assert correlation_store.get_incident(first.incident_id) == first
    assert correlation_store.get_workflow(access.request_id) == access

    monkeypatch.setattr(correlation_store, "merge_attempt_fault_workflow", merge)
    incident, workflow = orchestrator.ingest(event, decision)

    assert workflow is not None
    assert live_reset_gpus(workflow, RESET_ALL) == {"GPU-a", "GPU-b"}
    assert len(correlation_store.list_workflows()) == 1
    assert orchestrator.ingest(event, decision) == (incident, workflow)
    assert correlation_store.get_workflow(workflow.request_id) == workflow


def test_exact_event_replay_returns_current_execution_without_replanning(
    correlation_store: ControlPlaneStore,
) -> None:
    correlation_store.save_attempt_observation(current_attempt())
    orchestrator = IncidentOrchestrator(correlation_store)
    event = xid("exact-replay", at=NOW)
    first, workflow = ingest_provider(orchestrator, event)
    completed = correlation_store.amend_workflow(
        workflow.request_id, {"status": WorkflowStatus.SUCCEEDED}
    )
    ingest_warning(
        orchestrator,
        sbe_warning(event_id="later-warning", at=NOW + timedelta(seconds=1)),
    )
    replay_payload = event.model_dump(mode="json")
    replayed_event = XidEvent.model_validate(replay_payload)
    assert replayed_event == event
    decision = provider_decision(orchestrator, replayed_event)

    replay = IncidentOrchestrator(correlation_store).ingest(replayed_event, decision)

    assert replay == (first, completed)
    assert correlation_store.get_workflow(workflow.request_id) == completed


def test_companion_event_id_wins_over_an_unrelated_semantic_marker(
    correlation_store: ControlPlaneStore,
) -> None:
    orchestrator = IncidentOrchestrator(
        correlation_store, multi_node_aggregation_window_seconds=0
    )
    first_event = xid("exact-primary", at=NOW - timedelta(seconds=1), active=False)
    primary, workflow = ingest_provider(orchestrator, first_event)
    warning, _ = ingest_warning(orchestrator, sbe_warning(active=False))
    event = xid("exact-companion", code=64, active=False)
    decision = GpuFaultPolicyEngine().evaluate_xid(
        event, companion_events=[first_event]
    )
    decision = orchestrator.correlate_provider_event(
        decision, cluster_id=event.cluster_id
    )
    assert decision.correlated_event_id == first_event.event_id
    assert decision.marker.incident_id == warning.incident_id

    incident, companion = orchestrator.ingest(event, decision)

    assert incident.incident_id == primary.incident_id
    assert companion == workflow
    assert live_reset_gpus(companion, RESET) == {"GPU-a"}
    assert correlation_store.get_incident_by_event(event.event_id) == primary


def test_companion_link_to_an_old_diagnostic_cannot_suppress_the_required_reset(
    correlation_store: ControlPlaneStore,
) -> None:
    orchestrator = IncidentOrchestrator(
        correlation_store, multi_node_aggregation_window_seconds=0
    )
    warning, diagnostic = ingest_warning(orchestrator, sbe_warning(active=False))
    primary = xid("historically-mislinked-64", code=64, at=NOW, active=False)
    correlation_store.link_event_to_incident(primary.event_id, warning.incident_id)
    event = xid("companion-needs-reset", active=False)
    decision = GpuFaultPolicyEngine().evaluate_xid(event, companion_events=[primary])
    assert decision.correlated_event_id == primary.event_id
    decision = orchestrator.correlate_provider_event(
        decision, cluster_id=event.cluster_id
    )

    incident, workflow = orchestrator.ingest(event, decision)

    assert workflow is not None
    assert workflow.request_id != diagnostic.request_id
    assert live_reset_gpus(workflow, RESET) == {"GPU-a"}
    assert correlation_store.get_incident(warning.incident_id) == warning
    assert orchestrator.ingest(event, decision) == (incident, workflow)


@pytest.mark.parametrize("active", [False, True], ids=["idle", "attempt"])
def test_new_access_event_with_intersecting_scope_resets_both_gpus(
    correlation_store: ControlPlaneStore, active: bool
) -> None:
    if active:
        correlation_store.save_attempt_observation(current_attempt())
    orchestrator = IncidentOrchestrator(correlation_store)
    first, access = ingest_provider(orchestrator, sxid("one-gpu-access", active=active))
    event = sxid(
        "two-gpu-access", active=active, at=NOW + timedelta(seconds=1)
    ).model_copy(update={"participating_gpu_uuids": ["GPU-a", "GPU-b"]})
    decision = provider_decision(orchestrator, event)
    assert decision.marker.incident_id == first.incident_id
    assert decision.official_action == access.official_action

    incident, workflow = orchestrator.ingest(event, decision)

    assert workflow is not None
    assert workflow.request_id == access.request_id
    assert workflow.merge_revision > access.merge_revision
    reset_steps = [step for step in workflow.official_steps if step.operation is RESET]
    assert len(reset_steps) == 1
    assert reset_steps[0].node_ids == ["node-a"]
    assert reset_steps[0].gpu_uuids == ["GPU-a", "GPU-b"]
    assert reset_steps[0].parameters["gpu_uuids_by_node"] == {
        "node-a": ["GPU-a", "GPU-b"]
    }
    assert incident.gpu_uuids == ["GPU-a", "GPU-b"]
    replayed = SxidEvent.model_validate(event.model_dump(mode="json"))
    assert orchestrator.ingest(replayed, decision) == (incident, workflow)


@pytest.mark.parametrize(
    "change",
    ["missing", "terminal", "profile", "boot", "attempt", "generation", "incident"],
)
def test_unsafe_exact_companion_cannot_bypass_a_fresh_candidates_evidence_gate(
    correlation_store: ControlPlaneStore, change: str
) -> None:
    orchestrator = IncidentOrchestrator(
        correlation_store, multi_node_aggregation_window_seconds=0
    )
    primary_event = xid("companion-incumbent", code=64, at=NOW, active=False)
    primary_event = primary_event.model_copy(update={"source_boot_id": "boot-a"})
    incident, original = ingest_provider(orchestrator, primary_event)
    if change == "missing":
        incident = incident.model_copy(
            update={"workflow_request_id": "workflow-no-longer-present"}
        )
        correlation_store.save_incident(incident)
    elif change == "terminal":
        correlation_store.amend_workflow(
            original.request_id, {"status": WorkflowStatus.SUCCEEDED}
        )
    elif change == "profile":
        correlation_store.amend_workflow(
            original.request_id, {"runtime_profile_version": "previous-profile"}
        )
    elif change == "boot":
        incident = incident.model_copy(update={"source_boot_id": "previous-boot"})
        correlation_store.save_incident(incident)
    elif change == "attempt":
        incident = incident.model_copy(
            update={"job_id": "previous-job", "attempt_id": "previous-attempt"}
        )
        correlation_store.save_incident(incident)
    elif change == "generation":
        correlation_store.amend_workflow(
            original.request_id, {"fencing_token": original.fencing_token + 1}
        )
    elif change == "incident":
        correlation_store.amend_workflow(
            original.request_id, {"incident_id": "different-incident"}
        )
    before = correlation_store.get_workflow(original.request_id)
    event = xid(
        "fresh-companion-report", active=False, workload_state=WorkloadState.UNKNOWN
    ).model_copy(update={"source_boot_id": "boot-a"})
    decision = GpuFaultPolicyEngine().evaluate_xid(
        event, companion_events=[primary_event]
    )
    assert decision.correlated_event_id == primary_event.event_id

    current_incident, workflow = orchestrator.ingest(event, decision)

    assert workflow is not None
    assert workflow.request_id != original.request_id
    assert current_incident.incident_id != incident.incident_id
    assert workflow.status in {WorkflowStatus.BLOCKED, WorkflowStatus.SAFETY_PENDING}
    assert "node workload state is UNKNOWN" in workflow.blocked_reasons
    assert correlation_store.get_workflow(original.request_id) == before
    assert correlation_store.get_incident(incident.incident_id) == incident
    replayed = XidEvent.model_validate(event.model_dump(mode="json"))
    assert orchestrator.ingest(replayed, decision) == (current_incident, workflow)
