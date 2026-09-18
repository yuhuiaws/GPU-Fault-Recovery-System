"""Fresh, all-participant STOP ownership checks through the local Kubernetes API.

Node Action transport invokes this both before a new submission and in response
to the Agent's native post-queue/pre-spawn challenge. The Agent enforces that
challenge; this module alone cannot close an asynchronous queue boundary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from functools import wraps
from typing import Annotated, Any, Literal, TypeVar

from pydantic import ConfigDict, Field

from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    ANNOTATION_OPERATION,
    ANNOTATION_WORKFLOW,
    LABEL_ATTEMPT_ID,
)
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import (
    IncidentState,
    StrictModel,
    WorkflowOperation,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    execution_phase,
)
from gpu_fault.operation_registry import (
    DEVICE_PLUGIN_RESTART_OPERATIONS,
    NODE_MUTATING_OPERATIONS,
)
from gpu_fault.adapters.node_action.lease_guard import lease_hold_reason
from gpu_fault.adapters.kubernetes.restart_source_guard import _WorkloadMutation
from gpu_fault.adapters.kubernetes.stop_boot_transition import (
    REBOOT_AUTHORIZATION_KEY,
    authorized_boot_transition,
)
from gpu_fault.hyperpod import HyperPodAction, hyperpod_submission_idempotency_key
from gpu_fault.restart_containment import STOP_OWNERSHIP_RECEIPT_KEY

# One name for both layers: the control plane signs the predecessor
# containment's receipt under it, the guard reads it back under it.
STOP_RECEIPT_KEY = STOP_OWNERSHIP_RECEIPT_KEY
SUBMISSION_BOUNDARY = "EXECUTOR_PRE_SUBMIT"
GUARD_VERSION = "kubernetes-stop-ownership/v1"
# Match the exact API endpoints used by KubernetesPrimitivesMixin._read_workload.
WORKLOAD_GVK = {
    "job": ("batch/v1", "Job"),
    "pytorchjob": ("kubeflow.org/v1", "PyTorchJob"),
    "jobset": ("jobset.x-k8s.io/v1alpha2", "JobSet"),
}
Text = Annotated[str, Field(strict=True, min_length=1, max_length=512)]
Positive = Annotated[int, Field(strict=True, gt=0)]
PreparedWorkload = tuple[str, str, str, str, Any]


@dataclass
class _RebootSubmissionCapture:
    context: WorkflowStepContext
    authorization: dict[str, Any] | None = None


_reboot_submission_capture: ContextVar[_RebootSubmissionCapture | None] = ContextVar(
    "gpu_fault_reboot_submission_capture", default=None
)


class StopIdentity(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class StopOwnerReference(StopIdentity):
    api_version: Text
    kind: Text
    name: Text
    uid: Text
    controller: bool


class StopWorkloadIdentity(StopIdentity):
    workload_id: Text
    namespace: Text
    kind: Literal["job", "pytorchjob", "jobset"]
    name: Text
    uid: Text
    attempt_id: Text
    owners: tuple[StopOwnerReference, ...]


class StopPodIdentity(StopIdentity):
    workload_id: Text
    namespace: Text
    name: Text
    uid: Text
    node_id: Text
    owners: tuple[StopOwnerReference, ...]


class StopNodeIdentity(StopIdentity):
    name: Text
    uid: Text
    boot_id: Text


class StopOwnershipReceipt(StopIdentity):
    version: Literal[1] = 1
    cluster_id: Text
    workflow_id: Text
    incident_id: Text
    fencing_token: Positive
    execution_epoch: Positive
    stop_step_index: Annotated[int, Field(strict=True, ge=0)]
    stop_idempotency_key: Text
    phase: Literal["official", "safety"]
    workloads: tuple[StopWorkloadIdentity, ...]
    pods: tuple[StopPodIdentity, ...]
    nodes: tuple[StopNodeIdentity, ...]
    contained: bool = False
    completed_at: datetime | None = None

    def digest(self) -> str:
        return sha256(
            json.dumps(
                self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()


class StopOwnershipError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def refusal(reason: str) -> WorkflowStepOutcome:
    return WorkflowStepOutcome.failed(
        "workload containment does not authorize a new node action",
        details={
            "reason": reason,
            "safety_rejection": True,
            "manual_confirmation_required": True,
            "node_action_not_started": True,
            "ownership_guard": GUARD_VERSION,
            "ownership_check_boundary": SUBMISSION_BOUNDARY,
            "agent_queue_ownership_checked": False,
        },
    )


def _metadata(
    document: dict[str, Any], *, allow_deleting: bool = False
) -> dict[str, Any]:
    value = document.get("metadata")
    if (
        not isinstance(value, dict)
        or not all(
            isinstance(value.get(key), str) and value[key]
            for key in ("uid", "name", "resourceVersion")
        )
        or (value.get("deletionTimestamp") and not allow_deleting)
    ):
        raise StopOwnershipError("STOP_OWNERSHIP_IDENTITY_UNKNOWN")
    return value


def _owners(metadata: dict[str, Any]) -> tuple[StopOwnerReference, ...]:
    raw = metadata.get("ownerReferences", [])
    if not isinstance(raw, list):
        raise StopOwnershipError("STOP_OWNERSHIP_IDENTITY_UNKNOWN")
    owners = tuple(
        StopOwnerReference(
            api_version=item["apiVersion"],
            kind=item["kind"],
            name=item["name"],
            uid=item["uid"],
            controller=item.get("controller", False),
        )
        for item in raw
    )
    if len({item.uid for item in owners}) != len(owners):
        raise StopOwnershipError("STOP_OWNERSHIP_IDENTITY_UNKNOWN")
    return tuple(sorted(owners, key=lambda item: (item.uid, item.kind, item.name)))


class KubernetesStopOwnershipValidator:
    """Read local Kubernetes through the already-configured adapter clients."""

    def __init__(
        self,
        *,
        cluster_id: str,
        allowed_namespaces: frozenset[str],
        core_api: Any,
        read_workload: Callable[[str, str, str], Any],
        serialize: Callable[[Any], dict[str, Any]],
        workload_active: Callable[..., bool | None],
        before_recheck: Callable[[WorkflowStepContext, StopOwnershipReceipt], None]
        | None = None,
    ) -> None:
        if not cluster_id:
            raise ValueError("STOP ownership validator requires a cluster")
        self.cluster_id = cluster_id
        self.allowed_namespaces = allowed_namespaces
        self.core = core_api
        self.read_workload = read_workload
        self.serialize = serialize
        self.workload_active = workload_active
        self.before_recheck = before_recheck

    @classmethod
    def from_adapter(
        cls, adapter: Any, *, cluster_id: str, allowed_namespaces: frozenset[str]
    ) -> KubernetesStopOwnershipValidator:
        return cls(
            cluster_id=cluster_id,
            allowed_namespaces=allowed_namespaces,
            core_api=adapter.core,
            read_workload=adapter._read_workload,
            serialize=adapter._serialize_workload,
            workload_active=adapter._workload_active,
        )

    def _workload(
        self, namespace: str, kind: str, name: str, workload_id: str, raw: Any
    ) -> StopWorkloadIdentity:
        if (
            namespace not in self.allowed_namespaces
            or workload_id != f"{namespace}/{kind}/{name}"
        ):
            raise StopOwnershipError("STOP_OWNERSHIP_SCOPE_MISMATCH")
        document = self.serialize(raw)
        if (document.get("apiVersion"), document.get("kind")) != WORKLOAD_GVK.get(kind):
            raise StopOwnershipError("STOP_OWNERSHIP_SCOPE_MISMATCH")
        metadata = _metadata(document)
        labels = metadata.get("labels") or {}
        if (
            metadata.get("namespace") != namespace
            or metadata["name"] != name
            or labels.get("gpu-fault.io/managed") != "true"
            or not labels.get(LABEL_ATTEMPT_ID)
        ):
            raise StopOwnershipError("STOP_OWNERSHIP_IDENTITY_UNKNOWN")
        return StopWorkloadIdentity.model_validate(
            {
                "workload_id": workload_id,
                "namespace": namespace,
                "kind": kind,
                "name": name,
                "uid": metadata["uid"],
                "attempt_id": labels[LABEL_ATTEMPT_ID],
                "owners": _owners(metadata),
            }
        )

    def _pods(
        self,
        workloads: tuple[StopWorkloadIdentity, ...],
        known_pods: tuple[StopPodIdentity, ...] = (),
    ) -> list[tuple[StopPodIdentity, str]]:
        result = []
        for namespace in sorted({item.namespace for item in workloads}):
            response = self.serialize(self.core.list_namespaced_pod(namespace))
            items = response.get("items")
            if not isinstance(items, list) or (response.get("metadata") or {}).get(
                "continue"
            ):
                raise StopOwnershipError("STOP_PARTICIPANTS_UNKNOWN")
            for item in items:
                metadata = _metadata(item, allow_deleting=True)
                labels = metadata.get("labels") or {}
                owner_refs = _owners(metadata)
                candidates = [
                    workload
                    for workload in workloads
                    if workload.namespace == namespace
                    and (
                        labels.get(LABEL_ATTEMPT_ID) == workload.attempt_id
                        or any(owner.uid == workload.uid for owner in owner_refs)
                        or any(
                            pod.uid == metadata["uid"]
                            and pod.workload_id == workload.workload_id
                            for pod in known_pods
                        )
                    )
                ]
                if not candidates:
                    continue
                if len(candidates) != 1:
                    raise StopOwnershipError("STOP_PARTICIPANTS_UNKNOWN")
                workload = candidates[0]
                if labels.get(LABEL_ATTEMPT_ID) != workload.attempt_id:
                    raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")
                owners = owner_refs
                if len(owners) != 1 or not owners[0].controller:
                    raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")
                owner = owners[0]
                if owner.uid == workload.uid and (
                    (owner.api_version, owner.kind) != WORKLOAD_GVK[workload.kind]
                    or owner.name != workload.name
                ):
                    raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")
                if owner.uid != workload.uid:
                    if (
                        workload.kind != "jobset"
                        or (owner.api_version, owner.kind) != WORKLOAD_GVK["job"]
                    ):
                        raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")
                    parent_document = self.serialize(
                        self.read_workload(namespace, "job", owner.name)
                    )
                    parent = _metadata(parent_document)
                    if (
                        (parent_document.get("apiVersion"), parent_document.get("kind"))
                        != WORKLOAD_GVK["job"]
                        or parent.get("namespace") != namespace
                        or parent["name"] != owner.name
                    ):
                        raise StopOwnershipError("STOP_OWNERSHIP_SCOPE_MISMATCH")
                    controllers = [ref for ref in _owners(parent) if ref.controller]
                    if (
                        parent["uid"] != owner.uid
                        or len(controllers) != 1
                        or controllers[0].uid != workload.uid
                        or controllers[0].name != workload.name
                        or (controllers[0].api_version, controllers[0].kind)
                        != WORKLOAD_GVK[workload.kind]
                    ):
                        raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")
                if metadata.get("namespace") != namespace:
                    raise StopOwnershipError("STOP_OWNERSHIP_SCOPE_MISMATCH")
                result.append(
                    (
                        StopPodIdentity(
                            workload_id=workload.workload_id,
                            namespace=namespace,
                            name=metadata["name"],
                            uid=metadata["uid"],
                            node_id=(item.get("spec") or {}).get("nodeName", ""),
                            owners=owners,
                        ),
                        str((item.get("status") or {}).get("phase") or "UNKNOWN"),
                    )
                )
        if len({pod.uid for pod, _ in result}) != len(result):
            raise StopOwnershipError("STOP_PARTICIPANTS_UNKNOWN")
        return result

    def _active_gpu_pods(self, nodes: tuple[StopNodeIdentity, ...]) -> bool:
        active = False
        for node in nodes:
            response = self.serialize(
                self.core.list_pod_for_all_namespaces(
                    field_selector=f"spec.nodeName={node.name}",
                    _request_timeout=(5, 10),
                )
            )
            pods = response.get("items")
            if not isinstance(pods, list) or (response.get("metadata") or {}).get(
                "continue"
            ):
                raise StopOwnershipError("STOP_PARTICIPANTS_UNKNOWN")
            for pod in pods:
                _metadata(pod, allow_deleting=True)
                spec = pod.get("spec") or {}
                if spec.get("nodeName") != node.name:
                    raise StopOwnershipError("STOP_OWNERSHIP_SCOPE_MISMATCH")
                containers = spec.get("containers")
                if not isinstance(containers, list) or not containers:
                    raise StopOwnershipError("STOP_PARTICIPANTS_UNKNOWN")
                containers = [
                    *containers,
                    *(spec.get("initContainers") or []),
                    *(spec.get("ephemeralContainers") or []),
                ]
                requested = False
                for container in containers:
                    resources = container.get("resources") or {}
                    for amounts in (
                        resources.get("requests") or {},
                        resources.get("limits") or {},
                    ):
                        if not isinstance(amounts, dict):
                            raise StopOwnershipError("STOP_PARTICIPANTS_UNKNOWN")
                        for resource, amount in amounts.items():
                            if resource != "nvidia.com/gpu" and not resource.startswith(
                                "nvidia.com/mig-"
                            ):
                                continue
                            if (
                                isinstance(amount, bool)
                                or not isinstance(amount, (str, int))
                                or re.fullmatch(r"[0-9]+", str(amount)) is None
                            ):
                                raise StopOwnershipError("STOP_PARTICIPANTS_UNKNOWN")
                            requested = requested or int(amount) > 0
                phase = (pod.get("status") or {}).get("phase")
                if requested and phase not in {"Succeeded", "Failed"}:
                    active = True
        return active

    def check_idle(self, context: WorkflowStepContext) -> WorkflowStepOutcome | None:
        try:
            if (
                context.incident.cluster_id != self.cluster_id
                or context.workflow.incident_id != context.incident.incident_id
                or context.workflow.fencing_token != context.incident.fencing_token
                or context.request.expected_fencing_token
                != context.workflow.fencing_token
                or not context.step.node_ids
                or len(set(context.step.node_ids)) != len(context.step.node_ids)
            ):
                raise StopOwnershipError("STOP_OWNERSHIP_SCOPE_MISMATCH")
            nodes = tuple(self._node(name) for name in context.step.node_ids)
            if self._active_gpu_pods(nodes):
                raise StopOwnershipError("STOP_PARTICIPANTS_ACTIVE")
        except StopOwnershipError as exc:
            return refusal(exc.reason)
        except Exception:
            return refusal("STOP_OWNERSHIP_UNVERIFIABLE")
        return None

    def _node(self, name: str) -> StopNodeIdentity:
        value = self.serialize(self.core.read_node(name))
        metadata = _metadata(value)
        if metadata["name"] != name:
            raise StopOwnershipError("STOP_OWNERSHIP_SCOPE_MISMATCH")
        return StopNodeIdentity(
            name=name,
            uid=metadata["uid"],
            boot_id=((value.get("status") or {}).get("nodeInfo") or {}).get(
                "bootID", ""
            ),
        )

    def capture(
        self, context: WorkflowStepContext, prepared: Sequence[PreparedWorkload]
    ) -> StopOwnershipReceipt:
        if (
            context.incident.cluster_id != self.cluster_id
            or context.workflow.incident_id != context.incident.incident_id
            or context.workflow.fencing_token != context.incident.fencing_token
            or context.request.expected_fencing_token != context.workflow.fencing_token
        ):
            raise StopOwnershipError("STOP_OWNERSHIP_SCOPE_MISMATCH")
        previous = [
            item
            for item in context.workflow.step_executions
            if item.operation is WorkflowOperation.STOP_WORKLOADS
            and item.step_index == context.step_index
            and item.phase == execution_phase(context.workflow)
        ]
        workloads = tuple(self._workload(*item) for item in prepared)
        if not workloads or {item.workload_id for item in workloads} != set(
            context.step.workload_ids
        ):
            raise StopOwnershipError("STOP_OWNERSHIP_IDENTITY_UNKNOWN")
        if context.incident.attempt_id and any(
            item.attempt_id != context.incident.attempt_id for item in workloads
        ):
            raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")
        if previous:
            receipt = self._decode(previous[-1])
            self._binding(context, receipt, previous[-1], capturing=True)
            if workloads != receipt.workloads:
                raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")
            return receipt
        pods = tuple(pod for pod, _ in self._pods(workloads))
        nodes = sorted(set(context.step.node_ids) | {pod.node_id for pod in pods})
        return StopOwnershipReceipt(
            cluster_id=self.cluster_id,
            workflow_id=context.workflow.request_id,
            incident_id=context.incident.incident_id,
            fencing_token=context.workflow.fencing_token,
            execution_epoch=context.workflow.execution_epoch,
            stop_step_index=context.step_index,
            stop_idempotency_key=context.idempotency_key,
            phase=execution_phase(context.workflow),
            workloads=workloads,
            pods=pods,
            nodes=tuple(self._node(node) for node in nodes),
        )

    @staticmethod
    def _decode(execution: WorkflowStepExecution) -> StopOwnershipReceipt:
        raw = (execution.details or {}).get(STOP_RECEIPT_KEY)
        if raw is None:
            raise StopOwnershipError("STOP_OWNERSHIP_RECEIPT_MISSING")
        return StopOwnershipReceipt.model_validate_json(
            json.dumps(raw, allow_nan=False)
        )

    def _binding(
        self,
        context: WorkflowStepContext,
        receipt: StopOwnershipReceipt,
        execution: WorkflowStepExecution,
        *,
        capturing: bool = False,
    ) -> None:
        workflow = context.workflow
        inherited = (
            not capturing
            and execution.step_index in workflow.inherited_step_indexes
            and execution.details.get("preemption_reuse") is True
            and execution.details.get("inherited_from_workflow_id")
            == workflow.predecessor_workflow_id
            == receipt.workflow_id
            and receipt.fencing_token < workflow.fencing_token
        )
        if (
            receipt.cluster_id != self.cluster_id
            or context.incident.cluster_id != self.cluster_id
            or receipt.phase != execution_phase(workflow)
            or workflow.fencing_token != context.incident.fencing_token
            or workflow.incident_id != context.incident.incident_id
            or context.request.expected_fencing_token != workflow.fencing_token
            or (
                not inherited
                and (
                    receipt.workflow_id != workflow.request_id
                    or receipt.incident_id != context.incident.incident_id
                    or receipt.fencing_token != workflow.fencing_token
                    or receipt.execution_epoch > workflow.execution_epoch
                    or receipt.stop_step_index != execution.step_index
                )
            )
        ):
            raise StopOwnershipError("STOP_OWNERSHIP_SCOPE_MISMATCH")
        if context.incident.attempt_id and any(
            item.attempt_id != context.incident.attempt_id for item in receipt.workloads
        ):
            raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")

    def _current(
        self,
        receipt: StopOwnershipReceipt,
        *,
        node_action: bool = True,
        context: WorkflowStepContext | None = None,
        observed_nodes: dict[str, StopNodeIdentity] | None = None,
    ) -> bool:
        current = []
        inactive = True
        for expected in receipt.workloads:
            raw = self.read_workload(expected.namespace, expected.kind, expected.name)
            observed = self._workload(
                expected.namespace,
                expected.kind,
                expected.name,
                expected.workload_id,
                raw,
            )
            if observed != expected:
                raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")
            document = self.serialize(raw)
            annotations = _metadata(document).get("annotations") or {}
            if any(
                annotations.get(key) != value
                for key, value in (
                    (ANNOTATION_WORKFLOW, receipt.workflow_id),
                    (ANNOTATION_INCIDENT, receipt.incident_id),
                    (ANNOTATION_FENCING, str(receipt.fencing_token)),
                    (ANNOTATION_OPERATION, receipt.stop_idempotency_key),
                )
            ):
                raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")
            spec = document.get("spec") or {}
            policy = (
                spec.get("runPolicy") or {} if expected.kind == "pytorchjob" else spec
            )
            state = self.workload_active(
                expected.namespace, expected.kind, expected.name, raw
            )
            if state is None:
                raise StopOwnershipError("STOP_PARTICIPANTS_UNKNOWN")
            inactive = inactive and policy.get("suspend") is True and state is False
            current.append(observed)
        if node_action:
            for expected_node in receipt.nodes:
                observed_node = self._node(expected_node.name)
                if observed_nodes is not None:
                    observed_nodes[expected_node.name] = observed_node
                if observed_node == expected_node:
                    continue
                if (
                    observed_node.uid != expected_node.uid
                    or context is None
                    or receipt.completed_at is None
                    or not authorized_boot_transition(
                        context,
                        node_id=expected_node.name,
                        node_uid=expected_node.uid,
                        receipt_digest=receipt.digest(),
                        previous_boot_id=expected_node.boot_id,
                        current_boot_id=observed_node.boot_id,
                        stopped_at=receipt.completed_at,
                    )
                ):
                    raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")
        for pod, phase in self._pods(tuple(current), receipt.pods):
            if pod not in receipt.pods:
                raise StopOwnershipError("STOP_PARTICIPANTS_CHANGED")
            if phase not in {"Succeeded", "Failed"}:
                inactive = False
        if node_action and self._active_gpu_pods(receipt.nodes):
            inactive = False
        return inactive

    def finish(
        self,
        context: WorkflowStepContext,
        receipt: StopOwnershipReceipt,
        outcome: WorkflowStepOutcome,
    ) -> WorkflowStepOutcome:
        if outcome.status is not WorkflowStepStatus.SUCCEEDED:
            return replace(
                outcome,
                details={
                    **(outcome.details or {}),
                    STOP_RECEIPT_KEY: receipt.model_dump(mode="json"),
                },
            )
        if not self._current(receipt):
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    **(outcome.details or {}),
                    "reason": "STOP_PARTICIPANTS_ACTIVE",
                    STOP_RECEIPT_KEY: receipt.model_dump(mode="json"),
                },
            )
        completed = receipt.model_copy(
            update={"contained": True, "completed_at": datetime.now(timezone.utc)}
        )
        return replace(
            outcome,
            details={
                **(outcome.details or {}),
                STOP_RECEIPT_KEY: completed.model_dump(mode="json"),
            },
        )

    def _own_receipt(
        self, context: WorkflowStepContext, steps: Sequence[WorkflowStepSpec]
    ) -> tuple[StopOwnershipReceipt, list[str]]:
        """The receipt of this workflow's own completed STOP and its scope."""
        stops = [
            index
            for index, step in enumerate(steps)
            if step.operation is WorkflowOperation.STOP_WORKLOADS
        ]
        if len(stops) != 1 or stops[0] not in context.workflow.completed_step_indexes:
            raise StopOwnershipError("STOP_OWNERSHIP_RECEIPT_MISSING")
        executions = [
            item
            for item in context.workflow.step_executions
            if item.step_index == stops[0]
            and item.operation is WorkflowOperation.STOP_WORKLOADS
            and item.status is WorkflowStepStatus.SUCCEEDED
            and item.phase == execution_phase(context.workflow)
        ]
        if not executions:
            raise StopOwnershipError("STOP_OWNERSHIP_RECEIPT_MISSING")
        receipt = self._decode(executions[-1])
        self._binding(context, receipt, executions[-1])
        return receipt, list(steps[stops[0]].workload_ids)

    def _predecessor_receipt(
        self, context: WorkflowStepContext
    ) -> tuple[StopOwnershipReceipt, list[str]]:
        """The predecessor containment's receipt a passive restart carries.

        A passive recovery workflow (``no-hardware-evidence:RESTART``) has one
        step and no STOP of its own: the attempt was contained by the
        predecessor workflow named in ``predecessor_workflow_id``. The control
        plane signs that containment's contained receipt into the restart
        authorization when the predecessor SUCCEEDED, and the step's premise
        (``requires_incident_state=RECOVERED`` on the containment incident)
        names the same incident. Everything else is refused as before.
        """
        authorization = context.request.restart_authorization
        proof = authorization.containment if authorization is not None else None
        if (
            context.step.operation is not WorkflowOperation.RESTART_WORKLOAD
            or proof is None
        ):
            raise StopOwnershipError("STOP_OWNERSHIP_RECEIPT_MISSING")
        receipt = StopOwnershipReceipt.model_validate_json(
            json.dumps(proof.receipt, allow_nan=False)
        )
        workflow = context.workflow
        parameters = context.step.parameters
        if (
            workflow.predecessor_workflow_id is None
            or proof.workflow_id != workflow.predecessor_workflow_id
            or receipt.workflow_id != proof.workflow_id
            or receipt.incident_id != proof.incident_id
            or parameters.get("requires_incident_state")
            != IncidentState.RECOVERED.value
            or parameters.get("incident_id") != proof.incident_id
            or receipt.cluster_id != self.cluster_id
            or context.incident.cluster_id != self.cluster_id
            or workflow.incident_id != context.incident.incident_id
            or workflow.fencing_token != context.incident.fencing_token
            or context.request.expected_fencing_token != workflow.fencing_token
        ):
            raise StopOwnershipError("STOP_OWNERSHIP_SCOPE_MISMATCH")
        if context.incident.attempt_id and any(
            item.attempt_id != context.incident.attempt_id for item in receipt.workloads
        ):
            raise StopOwnershipError("STOP_OWNERSHIP_DRIFT")
        return receipt, list(context.step.workload_ids)

    def check(self, context: WorkflowStepContext) -> WorkflowStepOutcome | None:
        try:
            steps = (
                context.workflow.safety_steps
                if context.workflow.executes_safety_steps
                else context.workflow.official_steps
            )
            if any(
                step.operation is WorkflowOperation.STOP_WORKLOADS for step in steps
            ):
                receipt, stop_scope = self._own_receipt(context, steps)
            else:
                receipt, stop_scope = self._predecessor_receipt(context)
            # The signed restart authorization validates the destination after
            # recovery. Original Node incarnations cannot survive a legitimate
            # reboot/replacement; source ownership and all source Pods still must.
            node_action = (
                context.step.operation is not WorkflowOperation.RESTART_WORKLOAD
            )
            if (
                not receipt.contained
                or receipt.completed_at is None
                or receipt.completed_at.tzinfo is None
                or receipt.completed_at > datetime.now(timezone.utc)
                or not receipt.workloads
                or not receipt.nodes
                or (
                    node_action
                    and not set(context.step.node_ids)
                    <= {node.name for node in receipt.nodes}
                )
                or not set(context.step.workload_ids)
                <= {item.workload_id for item in receipt.workloads}
                or set(stop_scope) != {item.workload_id for item in receipt.workloads}
            ):
                raise StopOwnershipError("STOP_PARTICIPANTS_CHANGED")
            if self.before_recheck is not None:
                self.before_recheck(context, receipt)
            observed_nodes: dict[str, StopNodeIdentity] = {}
            if not self._current(
                receipt,
                node_action=node_action,
                context=context,
                observed_nodes=observed_nodes,
            ):
                raise StopOwnershipError("STOP_PARTICIPANTS_ACTIVE")
            capture = _reboot_submission_capture.get()
            if capture is not None and capture.context is context:
                capture.authorization = {
                    "version": 1,
                    "cluster_id": self.cluster_id,
                    "workflow_id": context.workflow.request_id,
                    "incident_id": context.incident.incident_id,
                    "fencing_token": context.workflow.fencing_token,
                    "execution_epoch": context.workflow.execution_epoch,
                    "phase": execution_phase(context.workflow),
                    "step_index": context.step_index,
                    "execution_owner": context.step.execution_owner,
                    "submission_idempotency_key": hyperpod_submission_idempotency_key(
                        context.workflow.request_id,
                        context.step_index,
                        context.step.operation,
                    ),
                    "stop_receipt_sha256": receipt.digest(),
                    "nodes": {
                        node: {
                            "uid": observed_nodes[node].uid,
                            "boot_id": observed_nodes[node].boot_id,
                        }
                        for node in context.step.node_ids
                    },
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                }
        except StopOwnershipError as exc:
            return refusal(exc.reason)
        except Exception:
            return refusal("STOP_OWNERSHIP_UNVERIFIABLE")
        return None


@dataclass(frozen=True)
class _GuardScope:
    validator: KubernetesStopOwnershipValidator | None


_active_scope: ContextVar[_GuardScope | None] = ContextVar(
    "gpu_fault_stop_ownership", default=None
)


@contextmanager
def stop_ownership_scope(
    validator: KubernetesStopOwnershipValidator | None,
) -> Iterator[None]:
    token = _active_scope.set(_GuardScope(validator))
    try:
        yield
    finally:
        _active_scope.reset(token)


def capture_stop_ownership(
    context: WorkflowStepContext, prepared: Sequence[PreparedWorkload]
) -> StopOwnershipReceipt | WorkflowStepOutcome | None:
    scope = _active_scope.get()
    if scope is None:
        return None
    if scope.validator is None:
        return refusal("STOP_OWNERSHIP_VALIDATOR_UNAVAILABLE")
    try:
        return scope.validator.capture(context, prepared)
    except StopOwnershipError as exc:
        return refusal(exc.reason)
    except Exception:
        return refusal("STOP_OWNERSHIP_UNVERIFIABLE")


def prepare_stop_ownership(
    context: WorkflowStepContext,
    state: _WorkloadMutation,
    mark_terminating: Callable[..., Any],
) -> StopOwnershipReceipt | WorkflowStepOutcome | None:
    receipt = capture_stop_ownership(context, state.workloads)
    if isinstance(receipt, WorkflowStepOutcome):
        return receipt
    if context.step.parameters.get("termination_initiator_incident_id"):
        (
            state.terminating_pods,
            state.log_evidence,
            state.log_errors,
        ) = mark_terminating(state.workloads, context)
    return receipt


def finish_stop_ownership(
    context: WorkflowStepContext,
    receipt: StopOwnershipReceipt | None,
    outcome: WorkflowStepOutcome,
) -> WorkflowStepOutcome:
    if receipt is None:
        return outcome
    scope = _active_scope.get()
    if scope is None or scope.validator is None:
        return refusal("STOP_OWNERSHIP_VALIDATOR_UNAVAILABLE")
    try:
        return scope.validator.finish(context, receipt, outcome)
    except StopOwnershipError as exc:
        return refusal(exc.reason)
    except Exception:
        return refusal("STOP_OWNERSHIP_UNVERIFIABLE")


def node_submission_ownership_guard(
    context: WorkflowStepContext,
) -> WorkflowStepOutcome | None:
    scope = _active_scope.get()
    if (
        scope is None
        or context.step.operation not in NODE_MUTATING_OPERATIONS
        or context.step.operation is WorkflowOperation.RESTORE_GPU_SERVICES
    ):
        return None
    steps = (
        context.workflow.safety_steps
        if context.workflow.executes_safety_steps
        else context.workflow.official_steps
    )
    stopped = any(step.operation is WorkflowOperation.STOP_WORKLOADS for step in steps)
    if not stopped and context.step.operation in DEVICE_PLUGIN_RESTART_OPERATIONS:
        # A workflow that stopped nothing has no receipt to bind, and
        # re-registering devices disturbs no device holder: the spec runs the
        # EFA plugin restart beside live training on purpose (COLLECT-017 C).
        # Behind a STOP the same restart still binds to the receipt so an
        # owner or Pod drift is caught before the adapter acts.
        return None
    if scope.validator is None:
        return refusal("STOP_OWNERSHIP_VALIDATOR_UNAVAILABLE")
    if not context.step.workload_ids and not stopped:
        return scope.validator.check_idle(context)
    return scope.validator.check(context)


def regional_ownership_enforced() -> bool:
    return _active_scope.get() is not None


def guard_new_node_submission(
    context: WorkflowStepContext, node_id: str
) -> WorkflowStepOutcome | None:
    # The Kubernetes read or acceptance rendezvous may outlast the lease.
    checks: tuple[Callable[[], str | WorkflowStepOutcome | None], ...] = (
        lease_hold_reason,
        lambda: node_submission_ownership_guard(context),
        lease_hold_reason,
    )
    for check in checks:
        denied = check()
        if isinstance(denied, WorkflowStepOutcome):
            return denied
        if denied is not None:
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    "node_action_state": "LEASE_LOST",
                    "node_action_not_started": True,
                    "waiting_node": node_id,
                    "reason": denied,
                },
            )
    return None


class OwnershipMutationRefused(RuntimeError):
    def __init__(self, outcome: WorkflowStepOutcome) -> None:
        super().__init__("ownership mutation refused")
        self.outcome = outcome


def require_mutation_ownership(context: WorkflowStepContext) -> None:
    if regional_ownership_enforced() and lease_hold_reason() is not None:
        raise OwnershipMutationRefused(refusal("OWNERSHIP_LEASE_LOST"))
    outcome = node_submission_ownership_guard(context)
    if outcome is not None:
        raise OwnershipMutationRefused(outcome)


T = TypeVar("T")


def ownership_outcome(
    method: Callable[[T, WorkflowStepContext], WorkflowStepOutcome],
) -> Callable[[T, WorkflowStepContext], WorkflowStepOutcome]:
    @wraps(method)
    def guarded(self: T, context: WorkflowStepContext) -> WorkflowStepOutcome:
        capture = (
            _RebootSubmissionCapture(context)
            if context.step.operation is WorkflowOperation.RESTART_NODE
            else None
        )
        token = _reboot_submission_capture.set(capture)
        try:
            outcome = method(self, context)
        except OwnershipMutationRefused as exc:
            return exc.outcome
        finally:
            _reboot_submission_capture.reset(token)
        # Only an actual, matching provider submission can retain this proof.
        # Polls reuse the original result binding, never a fresh current-boot read.
        if (
            capture is not None
            and capture.authorization is not None
            and outcome.status
            in {WorkflowStepStatus.WAITING, WorkflowStepStatus.SUCCEEDED}
            and (outcome.details or {}).get("action") == HyperPodAction.REBOOT.value
            and (outcome.details or {}).get("submission_idempotency_key")
            == capture.authorization["submission_idempotency_key"]
        ):
            if (outcome.details or {}).get("provider_submission_duplicate") is True:
                details = dict(outcome.details or {})
                details.pop(REBOOT_AUTHORIZATION_KEY, None)
                return replace(
                    outcome,
                    status=WorkflowStepStatus.FAILED,
                    error=(
                        "cached reboot submission lacks its original STOP ownership "
                        "authorization; manual confirmation is required"
                    ),
                    details={
                        **details,
                        "reason": "STOP_REBOOT_AUTHORIZATION_MISSING",
                        "outcome_unknown": True,
                        "manual_confirmation_required": True,
                        "safety_rejection": True,
                    },
                )
            return replace(
                outcome,
                details={
                    **(outcome.details or {}),
                    REBOOT_AUTHORIZATION_KEY: capture.authorization,
                },
            )
        return outcome

    return guarded
