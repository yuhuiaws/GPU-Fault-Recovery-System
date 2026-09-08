from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.hyperpod import HyperPodNode
from gpu_fault.hyperpod_spares import (
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    HyperPodSpareCoordinator,
    SpareHealthPending,
    SparePoolState,
)
from gpu_fault.models import (
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.spare_health import (
    FAILURES_ANNOTATION,
    HEALTH_ANNOTATION,
    LAST_ALERT_AT_ANNOTATION,
    UNAVAILABLE_AT_ANNOTATION,
    HyperPodSpareHealthController,
    SpareHealthState,
)
from tests._builders import build_store, copy_model, node_health_finding


class FakeLifecycle:
    def __init__(self, nodes):
        self.nodes = nodes
        self.config = SimpleNamespace(cluster_name="hp-cluster")

    def list_nodes(self, *, enrich=False):
        return self.nodes

    def resolve_nodes(self, identifiers, *, nodes=None):
        source = nodes or self.nodes
        resolved = []
        for identifier in identifiers:
            matches = [node for node in source if identifier in node.aliases]
            if len(matches) != 1:
                raise ValueError(identifier)
            resolved.append(matches[0])
        return resolved


class FakeCore:
    def __init__(self, nodes, *, fail_on=None, pod_batches=None):
        self.nodes = nodes
        self.patches = []
        self.fail_on = fail_on
        self.pod_batches = list(pod_batches or [[]])

    def read_node(self, node_id):
        return self.nodes[node_id]

    def patch_node(self, node_id, body):
        if node_id == self.fail_on:
            self.fail_on = None
            raise ValueError("resourceVersion conflict")
        node = self.nodes[node_id]
        expected = body["metadata"]["resourceVersion"]
        if expected != node["metadata"]["resourceVersion"]:
            raise ValueError("resourceVersion conflict")
        annotations = node["metadata"]["annotations"]
        for key, value in body["metadata"].get("annotations", {}).items():
            if value is None:
                annotations.pop(key, None)
            else:
                annotations[key] = value
        node["spec"].update(body.get("spec", {}))
        node["metadata"]["resourceVersion"] = str(int(expected) + 1)
        self.patches.append((node_id, body))

    def list_node(self, label_selector=None):
        items = []
        for name, node in self.nodes.items():
            labels = node["metadata"].get("labels", {})
            if label_selector:
                key, value = label_selector.split("=", 1)
                if labels.get(key) != value:
                    continue
            item = {**node, "metadata": {**node["metadata"], "name": name}}
            items.append(item)
        return {"items": items}

    def list_pod_for_all_namespaces(self, field_selector=None):
        pods = self.pod_batches[0]
        if len(self.pod_batches) > 1:
            self.pod_batches.pop(0)
        return {"items": pods}


class FakeRegistry:
    def readiness(self, _cluster_id, _node_ids):
        return SimpleNamespace(ready=True)


def hyperpod_node(
    logical_id,
    instance_id,
    *,
    spare=False,
    status="Running",
    group="workers",
    instance_type="ml.p5.48xlarge",
):
    labels = {"kubernetes.io/hostname": f"hyperpod-{instance_id}"}
    if spare:
        labels["gpu-fault.io/spare"] = "true"
    return HyperPodNode(
        node_logical_id=logical_id,
        instance_id=instance_id,
        instance_group_name=group,
        instance_type=instance_type,
        status=status,
        kubernetes_labels=labels,
    )


def kubernetes_node(*, ready=True, unschedulable=True, hyperpod_health="Schedulable"):
    labels = {}
    if hyperpod_health is not None:
        labels["sagemaker.amazonaws.com/node-health-status"] = hyperpod_health
    return {
        "metadata": {"resourceVersion": "1", "annotations": {}, "labels": labels},
        "spec": {"unschedulable": unschedulable},
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}]
        },
    }


def coordinator(nodes, core_nodes, sent=None, *, core=None):
    store = build_store()
    for node in nodes:
        if node.kubernetes_labels.get("gpu-fault.io/spare"):
            store.save_agent(
                SimpleNamespace(
                    cluster_id="hp-cluster",
                    node_id=node.kubernetes_labels["kubernetes.io/hostname"],
                    runtime_profile_version="simulated-v1",
                )
            )
    return (
        HyperPodSpareCoordinator(
            FakeLifecycle(nodes),
            store,
            core or FakeCore(core_nodes),
            registry=FakeRegistry(),
            alert_sender=(sent.append if sent is not None else None),
        ),
        store,
    )


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs) -> None:
        self.value += timedelta(**kwargs)


def test_allocates_one_healthy_compatible_spare_per_fault_node():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-fault2"),
        hyperpod_node("worker-3", "i-spare1", spare=True),
        hyperpod_node("worker-4", "i-spare2", spare=True),
    ]
    core_nodes = {
        "hyperpod-i-spare1": kubernetes_node(),
        "hyperpod-i-spare2": kubernetes_node(),
    }
    service, _ = coordinator(nodes, core_nodes)

    first = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-a",
        fault_node_ids=["worker-1", "worker-2"],
    )
    repeated = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-a",
        fault_node_ids=["worker-1", "worker-2"],
    )

    assert first.applicable, "expected first.applicable to be truthy"
    assert first.sufficient, "expected first.sufficient to be truthy"
    assert first.required == 2
    assert first.selected_node_ids == ("hyperpod-i-spare1", "hyperpod-i-spare2")
    assert repeated.selected_node_ids == first.selected_node_ids
    for node in core_nodes.values():
        assert not node["spec"]["unschedulable"]
        assert (
            node["metadata"]["annotations"][SPARE_RESERVATION_ANNOTATION]
            == "incident-a"
        )
        assert (
            node["metadata"]["annotations"][SPARE_POOL_STATE_ANNOTATION]
            == SparePoolState.ALLOCATED.value
        )


def test_local_only_allocation_uses_kubernetes_node_inventory():
    source_name = "hyperpod-i-fault1"
    spare_name = "hyperpod-i-spare1"
    source = kubernetes_node(unschedulable=True)
    source["metadata"]["labels"].update(
        {
            "sagemaker.amazonaws.com/instance-group-name": "workers",
            "node.kubernetes.io/instance-type": "ml.p5.48xlarge",
        }
    )
    spare = kubernetes_node(unschedulable=True)
    spare["metadata"]["labels"].update(
        {
            "gpu-fault.io/spare": "true",
            "sagemaker.amazonaws.com/instance-group-name": "workers",
            "node.kubernetes.io/instance-type": "ml.p5.48xlarge",
        }
    )
    nodes = {source_name: source, spare_name: spare}
    lifecycle = FakeLifecycle([])
    lifecycle.list_nodes = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("provider inventory must not be used")
    )
    service = HyperPodSpareCoordinator(
        lifecycle,
        build_store(),
        FakeCore(nodes),
        remote_health_provider=SimpleNamespace(
            spare_health_reasons=lambda **_kwargs: []
        ),
    )

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-local",
        fault_node_ids=[source_name],
        local_only=True,
    )

    assert result.sufficient, "expected result.sufficient to be truthy"
    assert result.selected_node_ids == (spare_name,)
    assert (
        spare["metadata"]["annotations"][SPARE_RESERVATION_ANNOTATION]
        == "incident-local"
    )
    assert spare["spec"]["unschedulable"] is False


def test_insufficient_healthy_spares_blocks_and_alerts_admin():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-fault2"),
        hyperpod_node("worker-3", "i-spare1", spare=True),
        hyperpod_node("worker-4", "i-spare2", spare=True),
    ]
    core_nodes = {
        "hyperpod-i-spare1": kubernetes_node(),
        "hyperpod-i-spare2": kubernetes_node(ready=False),
    }
    sent = []
    service, store = coordinator(nodes, core_nodes, sent)

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-shortage",
        fault_node_ids=["worker-1", "worker-2"],
    )

    assert result.applicable, "expected result.applicable to be truthy"
    assert not result.sufficient, "expected result.sufficient to be falsy"
    assert result.required == 2
    assert "required=2, healthy=1" in result.reason
    assert len(store.list_notifications()) == 1
    assert sent == [result.notification_id]
    assert all(node["spec"]["unschedulable"] for node in core_nodes.values()), (
        'expected all(node["spec"]["unschedulable"] for node in core_nodes.values()) to be truthy'
    )


def test_hyperpod_unschedulable_spare_is_rejected():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    service, _ = coordinator(
        nodes, {"hyperpod-i-spare1": kubernetes_node(hyperpod_health="Unschedulable")}
    )

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-unhealthy",
        fault_node_ids=["worker-1"],
    )

    assert not result.sufficient, "expected result.sufficient to be falsy"
    assert "healthy=0" in result.reason


def test_spare_without_hyperpod_health_label_is_rejected():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node(hyperpod_health=None)}
    service, _ = coordinator(nodes, core_nodes)

    reasons = service.health_reasons("hp-cluster", nodes[1], "hyperpod-i-spare1")

    assert "HyperPod node health is unknown" in reasons


def test_incompatible_spare_does_not_satisfy_target_group():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("other-1", "i-spare1", spare=True, group="other-workers"),
    ]
    service, _ = coordinator(nodes, {"hyperpod-i-spare1": kubernetes_node()})

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-incompatible",
        fault_node_ids=["worker-1"],
    )

    assert not result.sufficient, "expected result.sufficient to be falsy"
    assert "workers/ml.p5.48xlarge" in result.reason


def test_concurrent_reservation_conflict_rolls_back_partial_set():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-fault2"),
        hyperpod_node("worker-3", "i-spare1", spare=True),
        hyperpod_node("worker-4", "i-spare2", spare=True),
    ]
    core_nodes = {
        "hyperpod-i-spare1": kubernetes_node(),
        "hyperpod-i-spare2": kubernetes_node(),
    }
    core = FakeCore(core_nodes, fail_on="hyperpod-i-spare2")
    service, store = coordinator(nodes, core_nodes, core=core)

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-conflict",
        fault_node_ids=["worker-1", "worker-2"],
    )

    assert not result.sufficient, "expected result.sufficient to be falsy"
    assert "resourceVersion conflict" in result.reason
    assert len(store.list_notifications()) == 1
    assert all(
        node["spec"]["unschedulable"]
        and SPARE_RESERVATION_ANNOTATION not in node["metadata"]["annotations"]
        for node in core_nodes.values()
    ), (
        'expected all( node["spec"]["unschedulable"] and SPARE_RESERVATION_ANNOTATION not in node["metadata"]["annotations"] for node in core_nodes.values() ) to be truthy'
    )


def test_pending_gpu_client_check_rolls_back_partial_set():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-fault2"),
        hyperpod_node("worker-3", "i-spare1", spare=True),
        hyperpod_node("worker-4", "i-spare2", spare=True),
    ]
    core_nodes = {
        "hyperpod-i-spare1": kubernetes_node(),
        "hyperpod-i-spare2": kubernetes_node(),
    }
    service, store = coordinator(nodes, core_nodes)

    def checker(_node, node_name, phase):
        if phase == "activation" and node_name.endswith("spare2"):
            raise SpareHealthPending("node action is still running")
        return []

    with pytest.raises(SpareHealthPending):
        service.allocate(
            cluster_id="hp-cluster",
            incident_id="incident-pending",
            fault_node_ids=["worker-1", "worker-2"],
            gpu_client_checker=checker,
        )

    assert store.list_notifications() == []
    assert all(
        node["spec"]["unschedulable"]
        and SPARE_RESERVATION_ANNOTATION not in node["metadata"]["annotations"]
        for node in core_nodes.values()
    ), (
        'expected all( node["spec"]["unschedulable"] and SPARE_RESERVATION_ANNOTATION not in node["metadata"]["annotations"] for node in core_nodes.values() ) to be truthy'
    )


def pod(name, *, phase="Running", requests=None, limits=None, init_limits=None):
    return {
        "metadata": {"namespace": "gpu-fault-system", "name": name},
        "status": {"phase": phase},
        "spec": {
            "containers": [
                {
                    "name": "main",
                    "resources": {"requests": requests or {}, "limits": limits or {}},
                }
            ],
            "initContainers": (
                [{"name": "init", "resources": {"limits": init_limits}}]
                if init_limits is not None
                else []
            ),
        },
    }


def test_monitoring_pod_without_gpu_resource_does_not_occupy_spare():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node()}
    core = FakeCore(core_nodes, pod_batches=[[pod("nvidia-smi-monitor")]])
    service, _ = coordinator(nodes, core_nodes, core=core)

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-monitor",
        fault_node_ids=["worker-1"],
    )

    assert result.sufficient, "expected result.sufficient to be truthy"
    assert result.selected_node_ids == ("hyperpod-i-spare1",)


def test_active_gpu_resource_pod_blocks_spare_allocation():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node()}
    core = FakeCore(
        core_nodes, pod_batches=[[pod("training", requests={"nvidia.com/gpu": "1"})]]
    )
    service, _ = coordinator(nodes, core_nodes, core=core)

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-occupied",
        fault_node_ids=["worker-1"],
    )

    assert not result.sufficient, "expected result.sufficient to be falsy"
    assert "healthy=0" in result.reason
    assert core.patches == []


def test_terminal_gpu_pod_does_not_occupy_spare():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node()}
    core = FakeCore(
        core_nodes,
        pod_batches=[
            [pod("finished", phase="Succeeded", limits={"nvidia.com/gpu": 8})]
        ],
    )
    service, _ = coordinator(nodes, core_nodes, core=core)

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-finished",
        fault_node_ids=["worker-1"],
    )

    assert result.sufficient, "expected result.sufficient to be truthy"


def test_activation_rechecks_gpu_pods_before_uncordon():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node()}
    core = FakeCore(
        core_nodes,
        pod_batches=[[], [pod("racing-init", init_limits={"nvidia.com/gpu": 1})]],
    )
    service, store = coordinator(nodes, core_nodes, core=core)

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-race",
        fault_node_ids=["worker-1"],
    )

    assert not result.sufficient, "expected result.sufficient to be falsy"
    assert "became occupied" in result.reason
    assert "gpu-fault-system/racing-init" in result.reason
    assert core.patches == []
    assert len(store.list_notifications()) == 1


def test_occupied_candidate_is_not_downgraded_to_pending():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node()}
    core = FakeCore(
        core_nodes, pod_batches=[[pod("training", requests={"nvidia.com/gpu": "1"})]]
    )
    service, store = coordinator(nodes, core_nodes, core=core)
    phases = []

    def checker(_node, _name, phase):
        phases.append(phase)
        raise SpareHealthPending("node agent GPU client check is pending")

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-occupied-pending",
        fault_node_ids=["worker-1"],
        gpu_client_checker=checker,
    )

    # Kubernetes already disqualified the spare, so the agent is never asked
    # and its pending answer can never mask the shortage.
    assert phases == []
    assert not result.sufficient, "expected result.sufficient to be falsy"
    assert "active GPU resource pods exist" in result.reason
    assert result.notification_id is not None
    assert len(store.list_notifications()) == 1
    assert core.patches == []


def test_occupied_activation_is_not_downgraded_to_pending():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node()}
    core = FakeCore(
        core_nodes, pod_batches=[[], [pod("racing", requests={"nvidia.com/gpu": "1"})]]
    )
    service, store = coordinator(nodes, core_nodes, core=core)
    phases = []

    def checker(_node, _name, phase):
        phases.append(phase)
        if phase == "activation":
            raise SpareHealthPending("node agent GPU client check is pending")
        return []

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-race-pending",
        fault_node_ids=["worker-1"],
        gpu_client_checker=checker,
    )

    assert phases == ["candidate"]
    assert not result.sufficient, "expected result.sufficient to be falsy"
    assert "became occupied" in result.reason
    assert len(store.list_notifications()) == 1
    assert core.patches == []


def test_busy_spare_rejection_reason_alerts_instead_of_waiting():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node()}
    service, store = coordinator(nodes, core_nodes)

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-busy-spare",
        fault_node_ids=["worker-1"],
        gpu_client_checker=lambda _node, _name, _phase: [
            "node agent GPU client check rejected the spare: "
            "GPU compute clients are still active: GPU-abc:4242"
        ],
    )

    assert not result.sufficient, "expected result.sufficient to be falsy"
    assert "insufficient healthy HyperPod spares" in result.reason
    assert "GPU compute clients are still active" in result.reason
    assert result.notification_id is not None
    assert len(store.list_notifications()) == 1


def test_gpu_client_checker_runs_for_candidate_and_activation():
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node()}
    service, _ = coordinator(nodes, core_nodes)
    phases = []

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-client-check",
        fault_node_ids=["worker-1"],
        gpu_client_checker=lambda _node, _name, phase: (phases.append(phase) or []),
    )

    assert result.sufficient, "expected result.sufficient to be truthy"
    assert phases == ["candidate", "activation"]


def test_no_declared_spare_pool_keeps_existing_replace_behavior():
    nodes = [hyperpod_node("worker-1", "i-fault1")]
    service, store = coordinator(nodes, {})

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-no-pool",
        fault_node_ids=["worker-1"],
    )

    assert not result.applicable, "expected result.applicable to be falsy"
    assert result.sufficient, "expected result.sufficient to be truthy"
    assert store.list_notifications() == []


def test_discovers_spare_from_kubernetes_when_provider_label_missing():
    fault = hyperpod_node("worker-1", "i-fault1")
    spare = hyperpod_node("worker-2", "i-spare1")
    core_nodes = {"hyperpod-i-spare1": kubernetes_node()}
    core_nodes["hyperpod-i-spare1"]["metadata"]["labels"].update(
        {"gpu-fault.io/spare": "true"}
    )
    store = build_store()
    store.save_agent(
        SimpleNamespace(
            cluster_id="hp-cluster",
            node_id="hyperpod-i-spare1",
            runtime_profile_version="simulated-v1",
        )
    )
    service = HyperPodSpareCoordinator(
        FakeLifecycle([fault, spare]),
        store,
        FakeCore(core_nodes),
        registry=FakeRegistry(),
    )

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-k8s-label",
        fault_node_ids=["worker-1"],
    )

    assert result.sufficient, "expected result.sufficient to be truthy"
    assert result.selected_node_ids == ("hyperpod-i-spare1",)


def test_spare_health_reboots_after_consecutive_failures_then_recovers():
    nodes = [hyperpod_node("worker-spare", "i-spare1", spare=True)]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node(ready=False)}
    coordinator_service, store = coordinator(nodes, core_nodes)
    store.save_profile(default_simulated_profile())
    controller = HyperPodSpareHealthController(
        coordinator_service, IncidentOrchestrator(store), store, failure_threshold=2
    )

    suspect = controller.scan()[0]
    rebooting = controller.scan()[0]

    assert suspect["state"] == SpareHealthState.SUSPECT.value
    assert rebooting["state"] == (SpareHealthState.REBOOT_PENDING.value)
    incident = store.get_incident(rebooting["incident_id"])
    workflow = store.get_workflow(incident.workflow_request_id)
    assert WorkflowOperation.RESTART_NODE in {
        step.operation for step in workflow.official_steps
    }

    core_nodes["hyperpod-i-spare1"]["status"]["conditions"][0]["status"] = "True"
    store.save_workflow(copy_model(workflow, status=WorkflowStatus.SUCCEEDED))
    rechecking = controller.scan()[0]
    recovered = controller.scan()[0]

    assert rechecking["state"] == (SpareHealthState.RECHECKING.value)
    assert recovered["state"] == SpareHealthState.HEALTHY.value
    assert (
        core_nodes["hyperpod-i-spare1"]["metadata"]["annotations"][HEALTH_ANNOTATION]
        == SpareHealthState.HEALTHY.value
    )


def test_spare_configuration_error_notifies_without_reboot() -> None:
    nodes = [hyperpod_node("worker-spare", "i-spare1", spare=True)]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node(unschedulable=False)}
    sent = []
    coordinator_service, store = coordinator(nodes, core_nodes)
    controller = HyperPodSpareHealthController(
        coordinator_service,
        IncidentOrchestrator(store),
        store,
        failure_threshold=1,
        alert_sender=sent.append,
    )

    result = controller.scan()[0]

    assert result["state"] == SpareHealthState.SUSPECT.value
    assert result["incident_id"] is None
    assert result["notification_id"]
    assert store.list_workflows() == []
    assert sent == [result["notification_id"]]
    notification = store.list_notifications()[0]
    assert "No physical recovery action was submitted" in (notification.body_text)


def test_spare_observation_error_notifies_without_reboot() -> None:
    class UnobservableCore(FakeCore):
        def list_pod_for_all_namespaces(self, field_selector=None):
            raise RuntimeError("Kubernetes API timeout")

    nodes = [hyperpod_node("worker-spare", "i-spare1", spare=True)]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node()}
    core = UnobservableCore(core_nodes)
    coordinator_service, store = coordinator(nodes, core_nodes, core=core)
    controller = HyperPodSpareHealthController(
        coordinator_service, IncidentOrchestrator(store), store, failure_threshold=1
    )

    result = controller.scan()[0]

    assert result["state"] == SpareHealthState.SUSPECT.value
    assert result["incident_id"] is None
    assert store.list_workflows() == []
    assert result["notification_id"]
    assert "OBSERVATION" in store.list_notifications()[0].body_text


def test_spare_health_marks_unavailable_and_alerts_after_reboot_failure():
    nodes = [hyperpod_node("worker-spare", "i-spare1", spare=True)]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node(ready=False)}
    sent = []
    coordinator_service, store = coordinator(nodes, core_nodes)
    store.save_profile(default_simulated_profile())
    controller = HyperPodSpareHealthController(
        coordinator_service,
        IncidentOrchestrator(store),
        store,
        failure_threshold=1,
        alert_sender=sent.append,
    )

    rebooting = controller.scan()[0]
    incident = store.get_incident(rebooting["incident_id"])
    workflow = store.get_workflow(incident.workflow_request_id)
    store.save_workflow(copy_model(workflow, status=WorkflowStatus.FAILED))
    unavailable = controller.scan()[0]
    repeated = controller.scan()[0]

    assert unavailable["state"] == (SpareHealthState.UNAVAILABLE.value)
    assert repeated["state"] == SpareHealthState.UNAVAILABLE.value
    assert core_nodes["hyperpod-i-spare1"]["spec"]["unschedulable"]
    assert (
        core_nodes["hyperpod-i-spare1"]["metadata"]["annotations"][
            SPARE_POOL_STATE_ANNOTATION
        ]
        == SparePoolState.UNAVAILABLE.value
    )
    assert len(store.list_notifications()) == 1
    assert sent == [unavailable["notification_id"]]


def test_spare_health_rechecks_unavailable_and_recovers() -> None:
    nodes = [hyperpod_node("worker-spare", "i-spare1", spare=True)]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node(ready=False)}
    clock = MutableClock(datetime(2026, 8, 13, 12, tzinfo=timezone.utc))
    coordinator_service, store = coordinator(nodes, core_nodes)
    store.save_profile(default_simulated_profile())
    controller = HyperPodSpareHealthController(
        coordinator_service,
        IncidentOrchestrator(store),
        store,
        failure_threshold=1,
        unavailable_recheck_seconds=3600,
        now=clock,
    )

    rebooting = controller.scan()[0]
    incident = store.get_incident(rebooting["incident_id"])
    workflow = store.get_workflow(incident.workflow_request_id)
    store.save_workflow(copy_model(workflow, status=WorkflowStatus.FAILED))
    unavailable = controller.scan()[0]
    assert unavailable["state"] == SpareHealthState.UNAVAILABLE.value

    core_nodes["hyperpod-i-spare1"]["status"]["conditions"][0]["status"] = "True"
    clock.advance(hours=1, seconds=1)
    rechecking = controller.scan()[0]
    recovered = controller.scan()[0]

    assert rechecking["state"] == SpareHealthState.RECHECKING.value
    assert recovered["state"] == SpareHealthState.HEALTHY.value
    annotations = core_nodes["hyperpod-i-spare1"]["metadata"]["annotations"]
    assert UNAVAILABLE_AT_ANNOTATION not in annotations
    assert LAST_ALERT_AT_ANNOTATION not in annotations


def test_spare_health_repeats_unavailable_alert_by_interval() -> None:
    nodes = [hyperpod_node("worker-spare", "i-spare1", spare=True)]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node(ready=False)}
    sent = []
    clock = MutableClock(datetime(2026, 8, 13, 12, tzinfo=timezone.utc))
    coordinator_service, store = coordinator(nodes, core_nodes)
    store.save_profile(default_simulated_profile())
    controller = HyperPodSpareHealthController(
        coordinator_service,
        IncidentOrchestrator(store),
        store,
        failure_threshold=1,
        unavailable_recheck_seconds=3600,
        unavailable_alert_seconds=86400,
        alert_sender=sent.append,
        now=clock,
    )

    rebooting = controller.scan()[0]
    incident = store.get_incident(rebooting["incident_id"])
    workflow = store.get_workflow(incident.workflow_request_id)
    store.save_workflow(copy_model(workflow, status=WorkflowStatus.FAILED))
    unavailable = controller.scan()[0]
    clock.advance(days=1, seconds=1)
    repeated = controller.scan()[0]
    same_bucket = controller.scan()[0]

    assert unavailable["notification_id"]
    assert repeated["notification_id"]
    assert same_bucket["notification_id"] is None
    assert len(store.list_notifications()) == 2
    assert len(sent) == 2
    annotations = core_nodes["hyperpod-i-spare1"]["metadata"]["annotations"]
    assert annotations[LAST_ALERT_AT_ANNOTATION] == (clock.value.isoformat())


def test_invalid_spare_annotations_do_not_stop_other_nodes() -> None:
    nodes = [
        hyperpod_node("worker-bad", "i-bad", spare=True),
        hyperpod_node("worker-good", "i-good", spare=True),
    ]
    core_nodes = {
        "hyperpod-i-bad": kubernetes_node(),
        "hyperpod-i-good": kubernetes_node(),
    }
    core_nodes["hyperpod-i-bad"]["metadata"]["annotations"].update(
        {
            HEALTH_ANNOTATION: "not-a-state",
            FAILURES_ANNOTATION: "not-an-integer",
            UNAVAILABLE_AT_ANNOTATION: "not-a-timestamp",
        }
    )
    coordinator_service, store = coordinator(nodes, core_nodes)
    controller = HyperPodSpareHealthController(
        coordinator_service, IncidentOrchestrator(store), store
    )

    results = controller.scan()

    assert [item["node_id"] for item in results] == [
        "hyperpod-i-bad",
        "hyperpod-i-good",
    ]
    assert all(item["state"] == SpareHealthState.HEALTHY.value for item in results), (
        'expected all(item["state"] == SpareHealthState.HEALTHY.value for item in results) to be truthy'
    )


def test_spare_reconcile_exception_does_not_stop_other_nodes() -> None:
    class ReadFailureCore(FakeCore):
        def __init__(self, nodes):
            super().__init__(nodes)
            self.failed = False

        def read_node(self, node_id):
            if node_id == "hyperpod-i-bad" and not self.failed:
                self.failed = True
                raise RuntimeError("temporary Kubernetes API error")
            return super().read_node(node_id)

    nodes = [
        hyperpod_node("worker-bad", "i-bad", spare=True),
        hyperpod_node("worker-good", "i-good", spare=True),
    ]
    core_nodes = {
        "hyperpod-i-bad": kubernetes_node(),
        "hyperpod-i-good": kubernetes_node(),
    }
    core = ReadFailureCore(core_nodes)
    coordinator_service, store = coordinator(nodes, core_nodes, core=core)
    controller = HyperPodSpareHealthController(
        coordinator_service, IncidentOrchestrator(store), store
    )

    results = controller.scan()

    assert results[0]["node_id"] == "hyperpod-i-bad"
    assert results[0]["state"] == SpareHealthState.SUSPECT.value
    assert results[1]["node_id"] == "hyperpod-i-good"
    assert results[1]["state"] == SpareHealthState.HEALTHY.value


def test_spare_health_waits_for_existing_specific_remediation():
    nodes = [hyperpod_node("worker-spare", "i-spare1", spare=True)]
    core_nodes = {"hyperpod-i-spare1": kubernetes_node(ready=False)}
    coordinator_service, store = coordinator(nodes, core_nodes)
    store.save_profile(default_simulated_profile())
    orchestrator = IncidentOrchestrator(store)
    finding = node_health_finding(
        "existing-reset",
        "existing-reset",
        cluster_id="hp-cluster",
        node_id="hyperpod-i-spare1",
        observed_at=datetime.now(timezone.utc),
        category=NodeHealthCategory.GPU,
        severity=Severity.CRITICAL,
        reason="uncorrectable ECC error",
        recommended_action=RecoveryAction.RESET_GPU,
        gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )
    store.add_marker(finding.marker())
    incident, _ = orchestrator.ingest_node_health(finding)
    controller = HyperPodSpareHealthController(
        coordinator_service, orchestrator, store, failure_threshold=1
    )

    result = controller.scan()[0]

    assert result["state"] == SpareHealthState.REBOOT_PENDING.value
    assert result["incident_id"] == incident.incident_id
    assert len(store.list_workflows()) == 1


def test_spare_health_state_machine_never_mutates_training_nodes():
    nodes = [
        hyperpod_node("worker-active", "i-active"),
        hyperpod_node("worker-spare", "i-spare1", spare=True),
    ]
    active = kubernetes_node(ready=False, unschedulable=False)
    spare = kubernetes_node(ready=True)
    core_nodes = {"hyperpod-i-active": active, "hyperpod-i-spare1": spare}
    coordinator_service, store = coordinator(nodes, core_nodes)
    store.save_profile(default_simulated_profile())
    controller = HyperPodSpareHealthController(
        coordinator_service, IncidentOrchestrator(store), store
    )

    results = controller.scan()

    assert [item["node_id"] for item in results] == ["hyperpod-i-spare1"]
    assert active["metadata"]["annotations"] == {}
    assert SPARE_POOL_STATE_ANNOTATION not in active["metadata"]["annotations"]


def test_shortage_reason_names_each_rejected_candidate_gate():
    """A blocked failover must say which gate each spare failed.

    Without this, an operator only learns "insufficient healthy
    spares" and has to re-inject a fault to discover the cause.
    """
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    service, _ = coordinator(
        nodes,
        {
            "hyperpod-i-spare1": kubernetes_node(
                ready=False, hyperpod_health="Unschedulable"
            )
        },
    )

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-diagnosable",
        fault_node_ids=["worker-1"],
    )

    assert not result.sufficient, "expected result.sufficient to be falsy"
    assert "rejected candidates" in result.reason
    assert "hyperpod-i-spare1" in result.reason
    assert "Kubernetes node is not Ready" in result.reason
    assert dict(result.rejected_candidates).keys() == {"hyperpod-i-spare1"}


def _advisory_marker(node_id: str, *, action, severity):
    return NodeMarker(
        cluster_id="hp-cluster",
        source="gpu-fault-policy/site-node-health",
        trusted=True,
        incident_id=f"inc-{node_id}-{action.value.lower()}",
        observed_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=50),
        scope=MarkerScope(node_ids=[node_id]),
        severity=severity,
        recommended_action=action,
        mapping_version="site-node-health-policy/v1",
        action_disposition=(
            "MONITOR_ONLY" if action is RecoveryAction.RUN_DIAGNOSTICS else "EXECUTABLE"
        ),
        raw_reason="TCP retransmissions increased",
    )


def test_monitor_only_marker_does_not_disqualify_a_healthy_spare():
    """Advisory markers must not veto the only replacement path.

    Site node-health policy records a trusted, unexpired
    ``RUN_DIAGNOSTICS`` marker on healthy nodes whenever TCP
    retransmissions blip, and such markers never produce a workflow, so
    nothing ever clears them before their TTL. Counting them left warm
    spares ineligible for most of their lifetime and made real
    replacements fail with "insufficient healthy HyperPod spares".
    """
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    service, store = coordinator(
        nodes, {"hyperpod-i-spare1": kubernetes_node(ready=True)}
    )
    store.add_marker(
        _advisory_marker(
            "hyperpod-i-spare1",
            action=RecoveryAction.RUN_DIAGNOSTICS,
            severity=Severity.WARNING,
        )
    )

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-advisory",
        fault_node_ids=["worker-1"],
    )

    assert result.sufficient, "expected result.sufficient to be truthy"
    assert result.selected_node_ids == ("hyperpod-i-spare1",)
    assert result.rejected_candidates == ()


def test_actionable_marker_still_disqualifies_and_is_named():
    """A marker demanding remediation must block, and say which one.

    The bare "a marker exists" reason gave an operator no way to decide
    whether the spare could be released by hand -- the marker only
    lives in the control-plane store.
    """
    nodes = [
        hyperpod_node("worker-1", "i-fault1"),
        hyperpod_node("worker-2", "i-spare1", spare=True),
    ]
    service, store = coordinator(
        nodes, {"hyperpod-i-spare1": kubernetes_node(ready=True)}
    )
    marker = _advisory_marker(
        "hyperpod-i-spare1",
        action=RecoveryAction.REPLACE_NODE,
        severity=Severity.CRITICAL,
    )
    store.add_marker(marker)

    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="incident-actionable",
        fault_node_ids=["worker-1"],
    )

    assert not result.sufficient, "expected result.sufficient to be falsy"
    reason = dict(result.rejected_candidates)["hyperpod-i-spare1"]
    joined = " ".join(reason)
    assert "active trusted node fault marker exists" in joined
    assert marker.marker_id in joined
    assert "critical/REPLACE_NODE" in joined
