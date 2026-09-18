from __future__ import annotations

from copy import deepcopy

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.adapters.kubernetes.stop_ownership import (
    KubernetesStopOwnershipValidator,
    stop_ownership_scope,
)
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.store import InMemoryStore

NODES = ["node-a", "node-b"]
WORKLOAD = "training/pytorchjob/job"


class KubernetesState:
    def __init__(self):
        self.job = {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {
                "name": "job",
                "namespace": "training",
                "uid": "job-uid",
                "resourceVersion": "1",
                "labels": {
                    "gpu-fault.io/managed": "true",
                    "gpu-fault.io/attempt-id": "attempt",
                },
                "annotations": {},
            },
            "spec": {"runPolicy": {"suspend": False}},
            "status": {
                "replicaStatuses": {"Master": {"active": 1}, "Worker": {"active": 1}}
            },
        }
        self.pods = [self.pod(node, f"pod-{node}") for node in NODES]
        self.nodes = {
            node: {
                "metadata": {
                    "name": node,
                    "uid": f"{node}-uid",
                    "resourceVersion": "1",
                },
                "status": {"nodeInfo": {"bootID": f"{node}-boot"}},
            }
            for node in NODES
        }
        self.calls = []
        self.read_error = None

    @staticmethod
    def pod(node, uid):
        return {
            "metadata": {
                "name": uid,
                "namespace": "training",
                "uid": uid,
                "resourceVersion": "1",
                "labels": {
                    "gpu-fault.io/managed": "true",
                    "gpu-fault.io/attempt-id": "attempt",
                },
                "ownerReferences": [
                    {
                        "apiVersion": "kubeflow.org/v1",
                        "kind": "PyTorchJob",
                        "name": "job",
                        "uid": "job-uid",
                        "controller": True,
                    }
                ],
            },
            "spec": {
                "nodeName": node,
                "containers": [
                    {
                        "name": "training",
                        "resources": {"limits": {"nvidia.com/gpu": "1"}},
                    }
                ],
            },
            "status": {"phase": "Running"},
        }

    def get_namespaced_custom_object(self, group, version, namespace, plural, name):
        self.calls.append(("get-workload", namespace, name))
        if self.read_error is not None:
            raise self.read_error
        return deepcopy(self.job)

    def patch_namespaced_custom_object(
        self, group, version, namespace, plural, name, body
    ):
        self.calls.append(("patch-workload", namespace, name))
        assert body["metadata"]["uid"] == self.job["metadata"]["uid"]
        assert (
            body["metadata"]["resourceVersion"]
            == self.job["metadata"]["resourceVersion"]
        )
        self.job["metadata"]["annotations"].update(body["metadata"]["annotations"])
        self.job["metadata"]["resourceVersion"] = "2"
        self.job["spec"] = deepcopy(body["spec"])
        self.job["status"] = {"conditions": [{"type": "Suspended", "status": "True"}]}

    def list_namespaced_pod(self, namespace, **kwargs):
        self.calls.append(("list-pods", namespace))
        return {"items": deepcopy(self.pods), "metadata": {}}

    def list_pod_for_all_namespaces(self, *, field_selector, **kwargs):
        node = field_selector.removeprefix("spec.nodeName=")
        self.calls.append(("list-node-pods", node))
        return {
            "items": deepcopy(
                [pod for pod in self.pods if pod["spec"]["nodeName"] == node]
            ),
            "metadata": {},
        }

    def read_node(self, name):
        self.calls.append(("get-node", name))
        return deepcopy(self.nodes[name])

    def read_namespaced_pod_log(self, *args, **kwargs):
        return "local training\n"

    def patch_namespaced_pod(self, name, namespace, body):
        self.calls.append(("patch-pod", name))

    def delete_namespaced_pod(self, name, namespace, **kwargs):
        self.calls.append(("delete-pod", name))
        self.pods = [pod for pod in self.pods if pod["metadata"]["name"] != name]


def runtime():
    state = KubernetesState()
    adapter = KubernetesWorkflowAdapter(
        core_api=state, batch_api=state, custom_api=state, store=InMemoryStore()
    )
    validator = KubernetesStopOwnershipValidator.from_adapter(
        adapter, cluster_id="cluster-local", allowed_namespaces=frozenset({"training"})
    )
    stop = WorkflowStepSpec(
        operation=WorkflowOperation.STOP_WORKLOADS,
        execution_owner=adapter.owner,
        node_ids=NODES,
        workload_ids=[WORKLOAD],
        parameters={"termination_initiator_incident_id": "incident-local"},
    )
    reset = WorkflowStepSpec(
        operation=WorkflowOperation.RESET_GPU,
        execution_owner="gpu-fault-node-agent",
        node_ids=[NODES[0]],
        gpu_uuids=["GPU-local"],
        workload_ids=[WORKLOAD],
    )
    workflow = WorkflowRequest(
        request_id="workflow-local",
        incident_id="incident-local",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        execution_epoch=1,
        official_action="RESET_GPU",
        official_steps=[stop, reset],
    )
    incident = FaultIncident(
        incident_id="incident-local",
        event_id="event-local",
        event_type="LOCAL_ACCEPTANCE",
        cluster_id="cluster-local",
        node_ids=NODES,
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
        workflow_request_id=workflow.request_id,
        attempt_id="attempt",
        policy_version="local-test",
        policy_source="LOCAL_TEST",
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=stop,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=1),
        idempotency_key="workflow-local/0/STOP_WORKLOADS",
    )
    return state, adapter, validator, context


def stopped_runtime():
    state, adapter, validator, context = runtime()
    with stop_ownership_scope(validator):
        outcome = adapter.execute(context)
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    workflow = context.workflow.model_copy(
        update={
            "completed_step_indexes": [0],
            "completed_operations": [WorkflowOperation.STOP_WORKLOADS],
            "step_executions": [
                WorkflowStepExecution(
                    step_index=0,
                    operation=WorkflowOperation.STOP_WORKLOADS,
                    status=WorkflowStepStatus.SUCCEEDED,
                    phase="official",
                    details=outcome.details,
                )
            ],
        }
    )
    reset = WorkflowStepContext(
        workflow=workflow,
        incident=context.incident,
        step=workflow.official_steps[1],
        step_index=1,
        request=context.request,
        idempotency_key="workflow-local/1/RESET_GPU",
    )
    return state, adapter, validator, reset
