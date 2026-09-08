"""Architecture-review guards for the Kubernetes and HyperPod adapters.

Every case here reads the node back instead of trusting what the workflow
remembers, and every failure path must leave the node clean.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from kubernetes import client as kubernetes_client

from gpu_fault.adapters.kubernetes.primitives import (
    build_kubernetes_clients,
    kubernetes_request_timeout_seconds,
)
from tests._builders import build_store, copy_model

from ._support import (
    FakeCoreApi,
    FakeHyperPodLifecycle,
    HyperPodLifecycleStepAdapter,
    KubernetesWorkflowAdapter,
    SimpleNamespace,
    UnusedApi,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepContext,
    WorkflowStepStatus,
    quarantine_taint_value,
    workflow_state,
)

EFA_OPERATION_ANNOTATION = "gpu-fault.io/efa-plugin-restart-operation"
EFA_INCIDENT_ANNOTATION = "gpu-fault.io/efa-plugin-restart-incident"
EFA_STARTED_ANNOTATION = "gpu-fault.io/efa-plugin-restart-started-at"


class _Stop(Exception):
    pass


def _api_error(status: int, message: str = "api error") -> Exception:
    error = RuntimeError(message)
    error.status = status  # type: ignore[attr-defined]
    return error


def _context(store, operation, *, parameters=None, node_ids=None):
    incident, workflow = workflow_state(store, [operation])
    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-kubernetes-adapter",
        parameters=parameters or {},
        node_ids=node_ids or ["node-a"],
    )
    return WorkflowStepContext(
        workflow=copy_model(workflow, official_steps=[step]),
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key=f"workflow-active/0/{operation.value}",
    )


def _isolated_node(incident_id: str, fencing_token: int = 3) -> FakeCoreApi:
    core = FakeCoreApi()
    core.node["metadata"]["annotations"] = {
        "gpu-fault.io/incident-id": incident_id,
        "gpu-fault.io/fencing-token": str(fencing_token),
        "gpu-fault.io/previous-unschedulable": "false",
    }
    core.node["spec"]["unschedulable"] = True
    core.node["spec"]["taints"].append(
        {
            "key": "gpu-fault.io/quarantined",
            "value": quarantine_taint_value(incident_id),
            "effect": "NoSchedule",
        }
    )
    return core


def _adapter(core) -> KubernetesWorkflowAdapter:
    return KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )


# --- A1: every API request is bounded -------------------------------------


def test_kubernetes_clients_bound_every_request_with_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = kubernetes_client.Configuration(host="https://kubernetes.invalid")
    core, batch, custom = build_kubernetes_clients(
        configuration, request_timeout_seconds=12.5
    )
    recorded: list[object] = []

    def request(method, url, **kwargs):
        recorded.append(kwargs.get("_request_timeout"))
        raise _Stop()

    for api in (core, batch, custom):
        monkeypatch.setattr(api.api_client, "request", request)

    with pytest.raises(_Stop):
        core.read_node("node-a")
    with pytest.raises(_Stop):
        batch.read_namespaced_job("job", "training")
    with pytest.raises(_Stop):
        custom.get_namespaced_custom_object("g", "v1", "training", "things", "x")

    assert recorded == [12.5, 12.5, 12.5]


def test_kubernetes_request_timeout_is_read_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GPU_FAULT_KUBERNETES_REQUEST_TIMEOUT_SECONDS", raising=False)
    assert kubernetes_request_timeout_seconds() == 30.0
    monkeypatch.setenv("GPU_FAULT_KUBERNETES_REQUEST_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError):
        kubernetes_request_timeout_seconds()
    monkeypatch.setenv("GPU_FAULT_KUBERNETES_REQUEST_TIMEOUT_SECONDS", "abc")
    with pytest.raises(ValueError):
        kubernetes_request_timeout_seconds()


def test_pod_log_read_is_bounded_by_timeout_and_bytes() -> None:
    class Core:
        def __init__(self) -> None:
            self.log_calls: list[dict[str, object]] = []
            self.pods = [
                {
                    "metadata": {
                        "name": "worker-0",
                        "namespace": "training",
                        "uid": "pod-uid-0",
                        "annotations": {"gpu-fault.io/training-container": "trainer"},
                    },
                    "spec": {"nodeName": "node-a", "containers": [{"name": "trainer"}]},
                }
            ]

        def list_namespaced_pod(self, *_args, **_kwargs):
            return SimpleNamespace(items=list(self.pods))

        def read_namespaced_pod_log(self, *_args, **kwargs):
            self.log_calls.append(kwargs)
            return "line\n"

        def patch_namespaced_pod(self, *_args, **_kwargs):
            return None

        def delete_namespaced_pod(self, *_args, **_kwargs):
            self.pods = []

    class Custom:
        workload = {
            "metadata": {
                "resourceVersion": "1",
                "labels": {
                    "gpu-fault.io/job-id": "training-job",
                    "gpu-fault.io/attempt-id": "attempt-a",
                },
                "annotations": {},
            },
            "spec": {"runPolicy": {"suspend": False}},
            "status": {
                "conditions": [{"type": "Running", "status": "True"}],
                "replicaStatuses": {"Master": {"active": 1}},
            },
        }

        def get_namespaced_custom_object(self, *_args):
            return self.workload

        def patch_namespaced_custom_object(self, *_args):
            body = _args[-1]
            self.workload["metadata"]["annotations"].update(
                body["metadata"]["annotations"]
            )
            self.workload["spec"] = body["spec"]

    store = build_store()
    core = Core()
    adapter = KubernetesWorkflowAdapter(
        core_api=core,
        batch_api=UnusedApi(),
        custom_api=Custom(),
        store=store,
        request_timeout_seconds=7.5,
    )
    incident, workflow = workflow_state(store, [WorkflowOperation.STOP_WORKLOADS])
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        workload_ids=["training/pytorchjob/training-job"],
        parameters={"termination_initiator_incident_id": incident.incident_id},
    )
    context = WorkflowStepContext(
        workflow=copy_model(workflow, official_steps=[step]),
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="workflow-log-bound/STOP_WORKLOADS",
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert len(core.log_calls) == 1
    assert core.log_calls[0]["_request_timeout"] == 7.5
    assert core.log_calls[0]["limit_bytes"] == adapter.workload_log_s3_max_bytes


# --- A2: one patch_node retry primitive -----------------------------------


def test_restore_reapplies_on_conflict_with_a_fresh_resource_version() -> None:
    class ConflictCore(FakeCoreApi):
        def __init__(self) -> None:
            super().__init__()
            self.bodies: list[dict[str, object]] = []

        def patch_node(self, node_id, body):
            self.bodies.append(body)
            if len(self.bodies) == 1:
                self.node["metadata"]["resourceVersion"] = "2"
                raise _api_error(409, "the object has been modified")
            return super().patch_node(node_id, body)

    store = build_store()
    context = _context(store, WorkflowOperation.RESTORE_SCHEDULING)
    core = ConflictCore()
    core.node["metadata"]["annotations"] = _isolated_node(
        context.incident.incident_id
    ).node["metadata"]["annotations"]
    core.node["spec"]["unschedulable"] = True

    outcome = _adapter(core).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert [body["metadata"]["resourceVersion"] for body in core.bodies] == ["1", "2"]
    assert core.node["spec"]["unschedulable"] is False


def test_restore_still_conflicting_after_bounded_attempts_waits() -> None:
    class AlwaysConflict(FakeCoreApi):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def patch_node(self, _node_id, _body):
            self.attempts += 1
            raise _api_error(409, "the object has been modified")

    store = build_store()
    context = _context(store, WorkflowOperation.RESTORE_SCHEDULING)
    core = AlwaysConflict()
    core.node["metadata"]["annotations"] = _isolated_node(
        context.incident.incident_id
    ).node["metadata"]["annotations"]

    outcome = _adapter(core).execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["patch_conflict_retry"] == ["node-a"]
    assert core.attempts == 3


def test_restore_treats_an_absent_node_as_nothing_to_restore() -> None:
    class MissingCore(FakeCoreApi):
        def read_node(self, _node_id):
            raise _api_error(404, "node is gone")

    store = build_store()
    context = _context(store, WorkflowOperation.RESTORE_SCHEDULING)

    outcome = _adapter(MissingCore()).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert outcome.details["absent_nodes"] == ["node-a"]
    assert outcome.details["restored_nodes"] == []


# --- A3: device-plugin restart annotation never poisons the node -----------


class _PluginCore:
    def __init__(self, *, allocatable: str = "0", pods=None) -> None:
        self.node = {
            "metadata": {"resourceVersion": "1", "annotations": {}},
            "status": {"allocatable": {"vpc.amazonaws.com/efa": allocatable}},
        }
        self.pods = (
            [
                {
                    "metadata": {"name": "efa-plugin-old", "uid": "pod-old"},
                    "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                }
            ]
            if pods is None
            else pods
        )
        self.deleted: list[tuple[str, str]] = []
        self.delete_error: Exception | None = None

    def read_node(self, _node_id):
        return self.node

    def patch_node(self, _node_id, body):
        for key, value in body["metadata"]["annotations"].items():
            if value is None:
                self.node["metadata"]["annotations"].pop(key, None)
            else:
                self.node["metadata"]["annotations"][key] = value

    def list_namespaced_pod(self, *_args, **_kwargs):
        return SimpleNamespace(items=self.pods)

    def delete_namespaced_pod(self, name, namespace, **_kwargs):
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append((namespace, name))


def test_plugin_restart_clears_its_annotation_when_the_pod_delete_fails() -> None:
    store = build_store()
    context = _context(
        store,
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
        parameters={"expected_count": 16},
    )
    core = _PluginCore()
    core.delete_error = _api_error(500, "etcd leader changed")

    with pytest.raises(RuntimeError):
        _adapter(core).execute(context)

    assert core.node["metadata"]["annotations"] == {}


def test_plugin_restart_timeout_fails_and_clears_the_annotation() -> None:
    store = build_store()
    context = _context(
        store,
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
        parameters={"expected_count": 16, "restart_timeout_seconds": 60},
    )
    core = _PluginCore()
    started = datetime.now(timezone.utc) - timedelta(minutes=5)
    core.node["metadata"]["annotations"] = {
        EFA_OPERATION_ANNOTATION: context.idempotency_key,
        EFA_INCIDENT_ANNOTATION: context.incident.incident_id,
        "gpu-fault.io/efa-plugin-restart-pod-uid": "pod-old",
        EFA_STARTED_ANNOTATION: started.isoformat(),
    }
    core.pods = []

    outcome = _adapter(core).execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert core.node["metadata"]["annotations"] == {}


def test_plugin_restart_annotation_carries_the_incident() -> None:
    store = build_store()
    context = _context(
        store,
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
        parameters={"expected_count": 16},
    )
    core = _PluginCore()

    outcome = _adapter(core).execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    annotations = core.node["metadata"]["annotations"]
    assert annotations[EFA_INCIDENT_ANNOTATION] == context.incident.incident_id
    assert annotations[EFA_OPERATION_ANNOTATION] == context.idempotency_key


def test_plugin_restart_takes_over_a_stale_foreign_annotation() -> None:
    store = build_store()
    context = _context(
        store,
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
        parameters={"expected_count": 16, "restart_timeout_seconds": 60},
    )
    core = _PluginCore()
    stale = datetime.now(timezone.utc) - timedelta(minutes=10)
    core.node["metadata"]["annotations"] = {
        EFA_OPERATION_ANNOTATION: "workflow-dead/0/RESTART_EFA_DEVICE_PLUGIN",
        EFA_INCIDENT_ANNOTATION: "incident-dead",
        EFA_STARTED_ANNOTATION: stale.isoformat(),
    }

    outcome = _adapter(core).execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert core.deleted == [("kube-system", "efa-plugin-old")]
    annotations = core.node["metadata"]["annotations"]
    assert annotations[EFA_OPERATION_ANNOTATION] == context.idempotency_key
    assert annotations[EFA_INCIDENT_ANNOTATION] == context.incident.incident_id
    assert outcome.details["node_results"]["node-a"]["took_over_incident"] == (
        "incident-dead"
    )


def test_plugin_restart_refuses_a_fresh_foreign_annotation() -> None:
    store = build_store()
    context = _context(
        store,
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
        parameters={"expected_count": 16, "restart_timeout_seconds": 600},
    )
    core = _PluginCore()
    core.node["metadata"]["annotations"] = {
        EFA_OPERATION_ANNOTATION: "workflow-live/0/RESTART_EFA_DEVICE_PLUGIN",
        EFA_INCIDENT_ANNOTATION: "incident-live",
        EFA_STARTED_ANNOTATION: datetime.now(timezone.utc).isoformat(),
    }

    outcome = _adapter(core).execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert core.deleted == []


def test_plugin_restart_without_an_expected_count_fails_closed() -> None:
    store = build_store()
    context = _context(store, WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN)
    core = _PluginCore(allocatable="1")

    outcome = _adapter(core).execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert "expected_count" in str(outcome.error)
    assert core.deleted == []
    assert core.node["metadata"]["annotations"] == {}


# --- A5: isolation is observed, not remembered -----------------------------


def test_isolating_an_absent_node_fails_closed() -> None:
    class MissingCore(FakeCoreApi):
        def read_node(self, _node_id):
            raise _api_error(404, "node is gone")

    store = build_store()
    context = _context(store, WorkflowOperation.MARK_UNSCHEDULABLE)

    outcome = _adapter(MissingCore()).execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["safety_rejection"] is True
    assert outcome.details["node_id"] == "node-a"


def _hyperpod_context(store, request):
    incident, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    step = copy_model(
        workflow.official_steps[0], execution_owner="gpu-fault-hyperpod-adapter"
    )
    workflow = copy_model(
        workflow,
        official_steps=[step],
        completed_operations=[WorkflowOperation.MARK_UNSCHEDULABLE],
    )
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=request,
        idempotency_key="workflow-active/0/RESTART_NODE",
    )


def test_hyperpod_preflight_refuses_a_node_that_is_not_actually_isolated() -> None:
    store = build_store()
    request = WorkflowExecutionRequest(
        expected_fencing_token=3,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )
    context = _hyperpod_context(store, request)
    core = FakeCoreApi()
    lifecycle = FakeHyperPodLifecycle()
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle, registry=object(), kubernetes_adapter=_adapter(core)
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["safety_rejection"] is True
    assert lifecycle.calls == 0
    assert lifecycle.preflight_kwargs == {}


def test_hyperpod_preflight_refuses_when_no_kubernetes_adapter_can_observe() -> None:
    store = build_store()
    request = WorkflowExecutionRequest(
        expected_fencing_token=3,
        isolation_verified_nodes=["node-a"],
        confirm_cluster_name="hp-cluster",
    )
    context = _hyperpod_context(store, request)
    lifecycle = FakeHyperPodLifecycle()
    adapter = HyperPodLifecycleStepAdapter(lifecycle, registry=object())

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["safety_rejection"] is True
    assert lifecycle.calls == 0


def test_hyperpod_preflight_passes_observed_isolation_to_the_provider() -> None:
    store = build_store()
    request = WorkflowExecutionRequest(
        expected_fencing_token=3, confirm_cluster_name="hp-cluster"
    )
    context = _hyperpod_context(store, request)
    core = _isolated_node(context.incident.incident_id)
    lifecycle = FakeHyperPodLifecycle()
    adapter = HyperPodLifecycleStepAdapter(lifecycle, kubernetes_adapter=_adapter(core))

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert lifecycle.preflight_kwargs["isolation_verified_nodes"] == ["node-a"]
    assert outcome.details["observed_isolation"]["node-a"]["unschedulable"] is True


# --- A6: STOP_WORKLOADS survives a pod vanishing between list and patch -----


def test_stop_workloads_ignores_a_pod_that_vanished_before_the_patch() -> None:
    class Core:
        def __init__(self) -> None:
            self.pods = [
                {
                    "metadata": {
                        "name": name,
                        "namespace": "training",
                        "uid": f"uid-{name}",
                        "annotations": {"gpu-fault.io/training-container": "trainer"},
                    },
                    "spec": {"nodeName": "node-a", "containers": [{"name": "trainer"}]},
                }
                for name in ("worker-0", "worker-1")
            ]
            self.patched: list[str] = []
            self.deleted: list[str] = []

        def list_namespaced_pod(self, *_args, **_kwargs):
            return SimpleNamespace(items=list(self.pods))

        def read_namespaced_pod_log(self, *_args, **_kwargs):
            return "line\n"

        def patch_namespaced_pod(self, name, _namespace, _body, **_kwargs):
            if name == "worker-1":
                raise _api_error(404, "pod not found")
            self.patched.append(name)

        def delete_namespaced_pod(self, name, *_args, **_kwargs):
            self.deleted.append(name)
            self.pods = []

    class Custom:
        workload = {
            "metadata": {
                "resourceVersion": "1",
                "labels": {
                    "gpu-fault.io/job-id": "training-job",
                    "gpu-fault.io/attempt-id": "attempt-a",
                },
                "annotations": {},
            },
            "spec": {"runPolicy": {"suspend": False}},
            "status": {
                "conditions": [{"type": "Running", "status": "True"}],
                "replicaStatuses": {"Master": {"active": 1}},
            },
        }

        def get_namespaced_custom_object(self, *_args):
            return self.workload

        def patch_namespaced_custom_object(self, *_args):
            body = _args[-1]
            self.workload["metadata"]["annotations"].update(
                body["metadata"]["annotations"]
            )
            self.workload["spec"] = body["spec"]

    store = build_store()
    core = Core()
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=Custom(), store=store
    )
    incident, workflow = workflow_state(store, [WorkflowOperation.STOP_WORKLOADS])
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        workload_ids=["training/pytorchjob/training-job"],
        parameters={"termination_initiator_incident_id": incident.incident_id},
    )
    context = WorkflowStepContext(
        workflow=copy_model(workflow, official_steps=[step]),
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="workflow-vanish/STOP_WORKLOADS",
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert core.patched == ["worker-0"]
    assert core.deleted == ["worker-0"]


# --- A8: the scheduling baseline is kept in the step outcome --------------


def test_isolate_and_restore_record_the_node_scheduling_baseline() -> None:
    store = build_store()
    core = FakeCoreApi()
    adapter = _adapter(core)
    isolate = _context(store, WorkflowOperation.MARK_UNSCHEDULABLE)

    isolated = adapter.execute(isolate)

    baseline = isolated.details["node_baselines"]["node-a"]
    assert baseline["before"]["unschedulable"] is False
    assert baseline["before"]["taint_keys"] == [
        "sagemaker.amazonaws.com/node-health-status"
    ]
    assert baseline["before"]["resource_version"] == "1"
    assert baseline["after"]["unschedulable"] is True
    assert "gpu-fault.io/quarantined" in baseline["after"]["taint_keys"]

    restore = _context(store, WorkflowOperation.RESTORE_SCHEDULING)
    restored = adapter.execute(restore)

    baseline = restored.details["node_baselines"]["node-a"]
    assert baseline["before"]["unschedulable"] is True
    assert "gpu-fault.io/quarantined" in baseline["before"]["taint_keys"]
    assert baseline["after"]["unschedulable"] is False
    assert "gpu-fault.io/quarantined" not in baseline["after"]["taint_keys"]


def test_restore_keeps_an_unreserved_warm_spare_cordoned() -> None:
    """Live 2026-09-08 (DESTR-003): a declared spare, uncordoned by its
    failover and later quarantined, was restored with previous-unschedulable
    false and left labeled but schedulable -- a state the pool health check
    refuses and no supported path re-cordons. An unreserved spare stays
    cordoned; only the quarantine and ownership come off."""
    store = build_store()
    context = _context(store, WorkflowOperation.RESTORE_SCHEDULING)
    core = _isolated_node(context.incident.incident_id)
    core.node["metadata"]["labels"] = {"gpu-fault.io/spare": "true"}
    core.node["metadata"]["annotations"]["gpu-fault.io/spare-pool-state"] = "AVAILABLE"

    outcome = _adapter(core).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert core.node["spec"]["unschedulable"] is True, (
        "an unreserved spare stays cordoned"
    )
    assert "gpu-fault.io/incident-id" not in core.node["metadata"]["annotations"]
    assert not any(
        item["key"] == "gpu-fault.io/quarantined"
        for item in core.node["spec"]["taints"]
    ), "the quarantine still comes off"
    assert core.node["metadata"]["annotations"]["gpu-fault.io/spare-pool-state"] == (
        "AVAILABLE"
    ), "the pool state is the pool's to change, not the restore's"


def test_restore_uncordons_an_allocated_spare_and_an_ordinary_node() -> None:
    """A spare ALLOCATED to an incident is serving as the replacement node and
    is uncordoned like any other node; so is a node without the spare label."""
    for labels, pool_state in (
        ({"gpu-fault.io/spare": "true"}, "ALLOCATED"),
        ({}, None),
    ):
        store = build_store()
        context = _context(store, WorkflowOperation.RESTORE_SCHEDULING)
        core = _isolated_node(context.incident.incident_id)
        core.node["metadata"]["labels"] = labels
        if pool_state:
            core.node["metadata"]["annotations"]["gpu-fault.io/spare-pool-state"] = (
                pool_state
            )

        outcome = _adapter(core).execute(context)

        assert outcome.status is WorkflowStepStatus.SUCCEEDED
        assert core.node["spec"]["unschedulable"] is False, (labels, pool_state)
