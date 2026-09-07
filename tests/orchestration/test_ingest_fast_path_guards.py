"""Ingestion fast paths survive a dangling pointer; derived ids are stable (F-B7).

Three places on the ingestion path read a record through a pointer they never
checked: ``NodeHealthIngestionService._existing_or_grouped`` (incident found
by event, its workflow row gone), ``ResetOperationService.ingest_distributed_xids``
(same shape for the XID batch id), and ``active_workflow_covers_inventory_finding``
(the incumbent's incident deleted between the listing and the lookup). Each
one raised ``NotFoundError`` out of the ingest, the data plane re-posted, and
the same pointer raised again -- a stable poison message (P2-51H).

The rebuild that repairs the chain is only idempotent when the ids it mints
are: ``incident-<uuid4>`` would give a re-posted batch a second incident and a
second workflow for one fault (P2-49H). Family ids are now derived from the
event identity.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.store import InMemoryStore, NotFoundError, SqliteStore
from gpu_fault.watcher import AttemptObservation
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_store,
    container_observation,
    fault_incident,
    node_health_finding,
    workflow_request,
    workflow_step,
)

NOW = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
CLUSTER = "cluster-a"
NODES = ["node-0", "node-1", "node-2"]


def _nccl_finding(node_id: str = "node-a") -> NodeHealthFinding:
    return node_health_finding(
        f"finding-log-{node_id}",
        f"log-{CLUSTER}-{node_id}-0123456789abcdef0123456789abcdef",
        cluster_id=CLUSTER,
        node_id=node_id,
        observed_at=NOW,
        category=NodeHealthCategory.NCCL,
        severity="warning",
        reason="NCCL collective timeout",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        runtime_profile_version="simulated-v1",
    )


def _replacement_observation() -> AttemptObservation:
    return attempt_observation(
        "train-multi",
        "train-multi-a001",
        NOW,
        cluster_id=CLUSTER,
        expected_critical_ranks=3,
        containers=[
            container_observation(
                f"pod-{rank}",
                f"worker-{rank}",
                rank,
                node_id,
                gpu_uuids=[f"GPU-{rank}"],
            )
            for rank, node_id in enumerate(NODES)
        ],
        workload_ids=["training/job/train-multi"],
        restart_budget=2,
    )


def _replacement_finding(rank: int) -> NodeHealthFinding:
    return node_health_finding(
        f"finding-node-{rank}",
        f"event-node-{rank}",
        cluster_id=CLUSTER,
        node_id=f"node-{rank}",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason=f"unrecoverable GPU fault on node-{rank}",
        recommended_action=RecoveryAction.REPLACE_NODE,
        gpu_uuids=[f"GPU-{rank}"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/train-multi"],
    )


def _distributed_batch() -> dict:
    return {
        "batch_id": "xid-batch-training-1",
        "job_id": "training-job-1",
        "attempt_id": "training-job-1-a001",
        "restart_budget": 2,
        "affected_workload_ids": ["training/pytorchjob/training-job-1"],
        "allocation": [
            {"node_id": node_id, "rank": rank, "gpu_uuids": [f"GPU-{rank}"]}
            for rank, node_id in enumerate(NODES)
        ],
        "events": [
            {
                "event_id": f"xid95-{node_id}",
                "cluster_id": CLUSTER,
                "node_id": node_id,
                "observed_at": NOW.isoformat(),
                "xid": 95,
                "gpu_uuid": f"GPU-{rank}",
                "product": "H200",
                "driver_branch": 575,
                "cuda_version": "12.9",
                "job_id": "training-job-1",
                "runtime_profile_version": "simulated-v1",
                "workload_state": "ACTIVE",
                "affected_workload_ids": ["training/pytorchjob/training-job-1"],
            }
            for rank, node_id in enumerate(NODES[:2])
        ],
    }


def _post_distributed(context: ApplicationContext) -> dict:
    async def scenario() -> dict:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/gpu-events/xid/distributed", json=_distributed_batch()
            )
        assert response.status_code == 200, response.text
        return response.json()

    return asyncio.run(scenario())


# --- health family: incident found by event, workflow row gone ---------------


def test_node_health_re_post_rebuilds_a_workflow_whose_row_is_gone(tmp_path) -> None:
    store = SqliteStore(str(tmp_path / "health.db"))
    store.save_profile(default_simulated_profile())
    orchestrator = IncidentOrchestrator(store)
    finding = _nccl_finding()
    try:
        incident, workflow = orchestrator.ingest_node_health(finding)
        assert workflow is not None
        # The pointer dangles: the shape a cross-group rewrite or a failed
        # ``_put`` leaves behind (P0-57A).
        store.save_incident(
            incident.model_copy(update={"workflow_request_id": "workflow-vanished"})
        )

        again_incident, again_workflow = orchestrator.ingest_node_health(finding)

        assert again_workflow is not None
        assert again_incident.incident_id == incident.incident_id
        assert again_incident.workflow_request_id == again_workflow.request_id
        assert store.get_workflow(again_workflow.request_id) == again_workflow
        assert store.get_incident_by_event(finding.event_id) == again_incident
        assert store.stale_event_link_repairs == 1
    finally:
        store.close()


def test_node_health_re_post_of_a_healthy_event_still_returns_the_stored_pair(
    tmp_path,
) -> None:
    store = SqliteStore(str(tmp_path / "health-dup.db"))
    store.save_profile(default_simulated_profile())
    orchestrator = IncidentOrchestrator(store)
    finding = _nccl_finding()
    try:
        first = orchestrator.ingest_node_health(finding)
        second = orchestrator.ingest_node_health(finding)

        assert first == second
        assert len(store.list_workflows()) == 1
        assert store.stale_event_link_repairs == 0
    finally:
        store.close()


# --- reset family: XID batch found by batch id, workflow row gone ------------


def test_distributed_xid_re_post_rebuilds_a_workflow_whose_row_is_gone(
    tmp_path,
) -> None:
    store = SqliteStore(str(tmp_path / "reset.db"))
    context = ApplicationContext(store=store)
    try:
        first = _post_distributed(context)
        stored = store.get_incident(first["incident"]["incident_id"])
        store.save_incident(
            stored.model_copy(update={"workflow_request_id": "workflow-vanished"})
        )

        second = _post_distributed(context)

        assert second["incident"]["incident_id"] == first["incident"]["incident_id"]
        assert store.get_workflow(second["workflow"]["request_id"]) is not None
        assert (
            store.get_incident_by_event("xid-batch-training-1").workflow_request_id
            == second["workflow"]["request_id"]
        )
    finally:
        store.close()


def test_distributed_xid_ids_are_a_function_of_the_batch() -> None:
    first = _post_distributed(ApplicationContext())
    second = _post_distributed(ApplicationContext())

    assert first["incident"]["incident_id"] == second["incident"]["incident_id"]
    assert first["workflow"]["request_id"] == second["workflow"]["request_id"]
    assert first["incident"]["incident_id"].startswith("incident-"), (
        'expected first["incident"]["incident_id"].startswith("incident-") to be true'
    )
    assert first["workflow"]["request_id"].startswith("workflow-"), (
        'expected first["workflow"]["request_id"].startswith("workflow-") to be true'
    )


# --- conflicts: incumbent incident vanishes between listing and lookup -------


class _IncidentVanishesAfterListing(InMemoryStore):
    """Deletes one incident the moment the active pairs have been listed."""

    def __init__(self, incident_id: str) -> None:
        super().__init__()
        self._vanishing = incident_id
        self._listed = False

    def list_active_workflow_incidents(self, *args, **kwargs):
        pairs = super().list_active_workflow_incidents(*args, **kwargs)
        self._listed = True
        return pairs

    def get_incident(self, incident_id: str):
        if self._listed and incident_id == self._vanishing:
            raise NotFoundError(incident_id)
        return super().get_incident(incident_id)


def _inventory_finding() -> NodeHealthFinding:
    return node_health_finding(
        "finding-inventory",
        "event-inventory",
        cluster_id=CLUSTER,
        node_id="node-a",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        metric_name="gpu_inventory_mismatch",
        reason="active GPU inventory does not match the configured node invariant",
        recommended_action=RecoveryAction.REBOOT_NODE,
        runtime_profile_version="simulated-v1",
    )


def _running_reboot(store) -> None:
    incident = fault_incident(
        "inc-run",
        "event-run",
        cluster_id=CLUSTER,
        node_ids=["node-a"],
        workflow_request_id="wf-run",
    )
    workflow = workflow_request(
        "wf-run",
        "inc-run",
        status=WorkflowStatus.RUNNING,
        official_steps=[
            workflow_step(WorkflowOperation.RESTART_NODE),
            workflow_step(WorkflowOperation.VALIDATE_GPU),
        ],
    )
    store.save_incident_and_workflow(incident, workflow)


def test_inventory_coverage_treats_a_vanished_incumbent_incident_as_no_incumbent() -> (
    None
):
    store = _IncidentVanishesAfterListing("inc-run")
    _running_reboot(store)
    service = NodeConflictService(store, RecoveryArbiter())

    assert (
        service.active_workflow_covers_inventory_finding(_inventory_finding()) is None
    )


def test_inventory_coverage_still_reports_a_present_incumbent() -> None:
    store = build_store()
    _running_reboot(store)
    service = NodeConflictService(store, RecoveryArbiter())

    covered = service.active_workflow_covers_inventory_finding(_inventory_finding())

    assert covered is not None
    assert covered[0].incident_id == "inc-run"
    assert covered[1].request_id == "wf-run"


# --- deterministic family ids -----------------------------------------------


def _replacement_ids(rank: int) -> tuple[str, str]:
    store = build_store()
    store.save_profile(default_simulated_profile())
    store.save_attempt_observation(_replacement_observation())
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)
    incident, workflow = orchestrator.ingest_node_health(_replacement_finding(rank))
    assert workflow is not None
    return incident.incident_id, workflow.request_id


def test_node_replacement_ids_are_a_function_of_the_finding() -> None:
    assert _replacement_ids(0) == _replacement_ids(0)
    assert _replacement_ids(0) != _replacement_ids(1)


def test_node_health_workflow_id_is_a_function_of_the_finding() -> None:
    first = ApplicationContext().orchestrator.ingest_node_health(_nccl_finding())
    second = ApplicationContext().orchestrator.ingest_node_health(_nccl_finding())

    assert first[1] is not None and second[1] is not None
    assert first[1].request_id == second[1].request_id
    assert first[0].incident_id == second[0].incident_id
