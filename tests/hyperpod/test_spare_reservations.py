"""Warm-spare reservations must be released on every path, or reclaimed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.adapters.hyperpod.lifecycle import HyperPodLifecycleStepAdapter
from gpu_fault.hyperpod import HyperPodNode
from gpu_fault.hyperpod_spares import (
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    SPARE_RESERVED_AT_ANNOTATION,
    HyperPodSpareCoordinator,
    SparePoolState,
)
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.spare_health import HyperPodSpareHealthController
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


class FakeLifecycle:
    def __init__(self, nodes):
        self.nodes = nodes
        self.config = SimpleNamespace(cluster_name="hp-cluster")

    def list_nodes(self, *, enrich=False):
        return self.nodes


class FakeCore:
    def __init__(self, nodes, *, fail_on=None, conflict_on=None):
        self.nodes = nodes
        self.patches = []
        self.fail_on = fail_on
        self.conflict_on = conflict_on

    def read_node(self, node_id):
        return self.nodes[node_id]

    def patch_node(self, node_id, body):
        if node_id == self.fail_on:
            raise ValueError("apiserver unavailable")
        if node_id == self.conflict_on:
            self.conflict_on = None
            error = RuntimeError("the object has been modified")
            error.status = 409  # type: ignore[attr-defined]
            raise error
        node = self.nodes[node_id]
        annotations = node["metadata"]["annotations"]
        for key, value in body["metadata"].get("annotations", {}).items():
            if value is None:
                annotations.pop(key, None)
            else:
                annotations[key] = value
        node["spec"].update(body.get("spec", {}))
        node["metadata"]["resourceVersion"] = str(
            int(node["metadata"]["resourceVersion"]) + 1
        )
        self.patches.append((node_id, body))

    def list_pod_for_all_namespaces(self, field_selector=None):
        return {"items": []}


def spare(instance_id: str) -> HyperPodNode:
    return HyperPodNode(
        node_logical_id=f"worker-{instance_id}",
        instance_id=instance_id,
        instance_group_name="workers",
        instance_type="ml.p5.48xlarge",
        status="Running",
        kubernetes_labels={
            "kubernetes.io/hostname": f"hyperpod-{instance_id}",
            "gpu-fault.io/spare": "true",
        },
    )


def reserved_node(incident_id: str, *, reserved_at: datetime | None = NOW):
    annotations = {
        SPARE_RESERVATION_ANNOTATION: incident_id,
        SPARE_POOL_STATE_ANNOTATION: SparePoolState.ALLOCATED.value,
    }
    if reserved_at is not None:
        annotations[SPARE_RESERVED_AT_ANNOTATION] = reserved_at.isoformat()
    return {
        "metadata": {
            "resourceVersion": "1",
            "annotations": annotations,
            "labels": {"sagemaker.amazonaws.com/node-health-status": "Schedulable"},
        },
        "spec": {"unschedulable": False},
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }


def _released(node) -> bool:
    return (
        SPARE_RESERVATION_ANNOTATION not in node["metadata"]["annotations"]
        and SPARE_RESERVED_AT_ANNOTATION not in node["metadata"]["annotations"]
        and node["spec"]["unschedulable"] is True
    )


def _controller(core_nodes, nodes, store, *, now=None, ttl_seconds=3600.0):
    core = FakeCore(core_nodes)
    coordinator = HyperPodSpareCoordinator(
        FakeLifecycle(nodes), store, core, now=now or (lambda: NOW)
    )
    controller = HyperPodSpareHealthController(
        coordinator,
        SimpleNamespace(ingest_node_health=None),
        store,
        now=now or (lambda: NOW),
        reservation_ttl_seconds=ttl_seconds,
    )
    return controller, core


def _incident_with_workflow(store, incident_id, status, *, executions=()):
    workflow = workflow_request(
        f"workflow-{incident_id}",
        incident_id,
        status=status,
        official_steps=[workflow_step(WorkflowOperation.REPLACE_NODE)],
        step_executions=list(executions),
    )
    incident = fault_incident(
        incident_id, f"event-{incident_id}", workflow_request_id=workflow.request_id
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


# --- (a) rollback releases every node even when one release fails ---------


def test_rollback_releases_the_remaining_nodes_when_one_release_fails() -> None:
    core_nodes = {
        "hyperpod-i-1": reserved_node("incident-a"),
        "hyperpod-i-2": reserved_node("incident-a"),
    }
    core = FakeCore(core_nodes, fail_on="hyperpod-i-2")
    coordinator = HyperPodSpareCoordinator(FakeLifecycle([]), build_store(), core)

    with pytest.raises(ValueError, match="apiserver unavailable"):
        coordinator.release(["hyperpod-i-1", "hyperpod-i-2"], "incident-a")

    assert _released(core_nodes["hyperpod-i-1"]), "the first node was not released"
    assert not _released(core_nodes["hyperpod-i-2"]), "the failing node changed"


def test_rollback_retries_a_resource_version_conflict() -> None:
    core_nodes = {"hyperpod-i-1": reserved_node("incident-a")}
    core = FakeCore(core_nodes, conflict_on="hyperpod-i-1")
    coordinator = HyperPodSpareCoordinator(FakeLifecycle([]), build_store(), core)

    coordinator.release(["hyperpod-i-1"], "incident-a")

    assert _released(core_nodes["hyperpod-i-1"]), "conflict was not retried"


def test_reservation_writes_its_timestamp() -> None:
    core_nodes = {
        "hyperpod-i-1": {
            "metadata": {
                "resourceVersion": "1",
                "annotations": {},
                "labels": {"sagemaker.amazonaws.com/node-health-status": "Schedulable"},
            },
            "spec": {"unschedulable": True},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]},
        }
    }
    core = FakeCore(core_nodes)
    coordinator = HyperPodSpareCoordinator(
        FakeLifecycle([]), build_store(), core, now=lambda: NOW
    )

    coordinator.reserve(spare("i-1"), "hyperpod-i-1", "incident-a")

    annotations = core_nodes["hyperpod-i-1"]["metadata"]["annotations"]
    assert annotations[SPARE_RESERVATION_ANNOTATION] == "incident-a"
    assert annotations[SPARE_RESERVED_AT_ANNOTATION] == NOW.isoformat()


# --- (b) the health scan reclaims orphaned reservations -------------------


def test_scan_reclaims_a_reservation_whose_workflow_failed() -> None:
    store = build_store()
    _incident_with_workflow(store, "incident-failed", WorkflowStatus.FAILED)
    core_nodes = {"hyperpod-i-1": reserved_node("incident-failed")}
    controller, _ = _controller(core_nodes, [spare("i-1")], store)

    results = controller.scan()

    assert _released(core_nodes["hyperpod-i-1"]), "reservation was not reclaimed"
    assert results[0]["incident_id"] == "incident-failed"
    assert any("reclaimed" in reason for reason in results[0]["reasons"]), results


def test_scan_keeps_a_reservation_whose_workflow_is_running() -> None:
    store = build_store()
    _incident_with_workflow(store, "incident-live", WorkflowStatus.RUNNING)
    core_nodes = {"hyperpod-i-1": reserved_node("incident-live")}
    controller, core = _controller(core_nodes, [spare("i-1")], store)

    controller.scan()

    assert core.patches == []
    annotations = core_nodes["hyperpod-i-1"]["metadata"]["annotations"]
    assert annotations[SPARE_RESERVATION_ANNOTATION] == "incident-live"


def test_scan_never_reclaims_a_spare_a_successful_failover_consumed() -> None:
    store = build_store()
    _incident_with_workflow(
        store,
        "incident-done",
        WorkflowStatus.SUCCEEDED,
        executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.REPLACE_NODE,
                WorkflowStepStatus.SUCCEEDED,
                details={
                    "action": "SPARE_FAILOVER",
                    "activated_spare_nodes": ["hyperpod-i-1"],
                },
            )
        ],
    )
    core_nodes = {
        "hyperpod-i-1": reserved_node(
            "incident-done", reserved_at=NOW - timedelta(days=30)
        )
    }
    controller, core = _controller(core_nodes, [spare("i-1")], store)

    controller.scan()

    assert core.patches == []
    annotations = core_nodes["hyperpod-i-1"]["metadata"]["annotations"]
    assert annotations[SPARE_RESERVATION_ANNOTATION] == "incident-done"


def test_scan_reclaims_an_unknown_incident_only_after_the_ttl() -> None:
    store = build_store()
    core_nodes = {
        "hyperpod-i-1": reserved_node(
            "incident-unknown", reserved_at=NOW - timedelta(minutes=30)
        ),
        "hyperpod-i-2": reserved_node(
            "incident-unknown", reserved_at=NOW - timedelta(hours=2)
        ),
    }
    controller, _ = _controller(
        core_nodes, [spare("i-1"), spare("i-2")], store, ttl_seconds=3600.0
    )

    controller.scan()

    assert not _released(core_nodes["hyperpod-i-1"]), "young reservation reclaimed"
    assert _released(core_nodes["hyperpod-i-2"]), "expired reservation kept"


def test_scan_publishes_a_reservation_gauge() -> None:
    store = build_store()
    _incident_with_workflow(store, "incident-live", WorkflowStatus.RUNNING)
    _incident_with_workflow(store, "incident-failed", WorkflowStatus.FAILED)
    core_nodes = {
        "hyperpod-i-1": reserved_node("incident-live"),
        "hyperpod-i-2": reserved_node("incident-failed"),
    }
    controller, _ = _controller(core_nodes, [spare("i-1"), spare("i-2")], store)

    controller.scan()
    snapshot = controller.metrics_snapshot()

    assert snapshot["spare_reservations_active"] == 1
    assert snapshot["spare_reservations_reclaimed_total"] == 1
    assert snapshot["spare_reservations_observed_at"] == NOW.isoformat()


# --- executor-side hook ---------------------------------------------------


def test_release_spare_reservations_frees_unconsumed_waiting_spares() -> None:
    class Coordinator:
        def __init__(self) -> None:
            self.releases: list[tuple[list[str], str]] = []

        def release(self, node_ids, incident_id):
            self.releases.append((list(node_ids), incident_id))

    coordinator = Coordinator()
    adapter = HyperPodLifecycleStepAdapter(
        SimpleNamespace(), spare_coordinator=coordinator
    )
    workflow = workflow_request(
        "workflow-1",
        "incident-1",
        status=WorkflowStatus.FAILED,
        official_steps=[workflow_step(WorkflowOperation.REPLACE_NODE)],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.REPLACE_NODE,
                WorkflowStepStatus.WAITING,
                details={
                    "spare_failover_pending": True,
                    "activated_spare_nodes": ["hyperpod-i-1", "hyperpod-i-2"],
                },
            )
        ],
    )

    released = adapter.release_spare_reservations(workflow, "incident-1")

    assert released == ["hyperpod-i-1", "hyperpod-i-2"]
    assert coordinator.releases == [(["hyperpod-i-1", "hyperpod-i-2"], "incident-1")]


def test_release_spare_reservations_keeps_consumed_spares() -> None:
    class Coordinator:
        def __init__(self) -> None:
            self.releases: list[tuple[list[str], str]] = []

        def release(self, node_ids, incident_id):
            self.releases.append((list(node_ids), incident_id))

    coordinator = Coordinator()
    adapter = HyperPodLifecycleStepAdapter(
        SimpleNamespace(), spare_coordinator=coordinator
    )
    workflow = copy_model(
        workflow_request(
            "workflow-1",
            "incident-1",
            status=WorkflowStatus.SUCCEEDED,
            official_steps=[workflow_step(WorkflowOperation.REPLACE_NODE)],
        ),
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.REPLACE_NODE,
                WorkflowStepStatus.SUCCEEDED,
                details={
                    "action": "SPARE_FAILOVER",
                    "activated_spare_nodes": ["hyperpod-i-1"],
                },
            )
        ],
    )

    assert adapter.release_spare_reservations(workflow, "incident-1") == []
    assert coordinator.releases == []


# --- A2: the health controller's own node patch retries conflicts ---------


def test_spare_health_patch_retries_a_resource_version_conflict() -> None:
    store = build_store()
    core_nodes = {
        "hyperpod-i-1": {
            "metadata": {
                "resourceVersion": "1",
                "annotations": {},
                "labels": {"sagemaker.amazonaws.com/node-health-status": "Schedulable"},
            },
            "spec": {"unschedulable": True},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]},
        }
    }
    core = FakeCore(core_nodes, conflict_on="hyperpod-i-1")
    coordinator = HyperPodSpareCoordinator(
        FakeLifecycle([spare("i-1")]), store, core, now=lambda: NOW
    )
    controller = HyperPodSpareHealthController(
        coordinator, SimpleNamespace(ingest_node_health=None), store, now=lambda: NOW
    )

    results = controller.scan()

    assert results[0]["state"] == "SUSPECT"
    annotations = core_nodes["hyperpod-i-1"]["metadata"]["annotations"]
    assert annotations["gpu-fault.io/spare-health"] == "SUSPECT"
    assert len(core.patches) == 1
