from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store import NotFoundError


from gpu_fault.adapters.common import (
    ANNOTATION_EFA_PLUGIN_RESTART_OPERATION,
    ANNOTATION_EFA_PLUGIN_RESTART_POD_UID,
    ANNOTATION_EFA_PLUGIN_RESTART_STARTED_AT,
    ANNOTATION_FENCING,
    ANNOTATION_GPU_PLUGIN_RESTART_OPERATION,
    ANNOTATION_GPU_PLUGIN_RESTART_POD_UID,
    ANNOTATION_GPU_PLUGIN_RESTART_STARTED_AT,
    ANNOTATION_INCIDENT,
    ANNOTATION_MECHANICAL_INSPECTION_COMPLETE,
    ANNOTATION_PREVIOUS_UNSCHEDULABLE,
    NodeIsolationRejected,
    QUARANTINE_TAINT,
    quarantine_taint_value,
)


class KubernetesNodeOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    store: Any

    _annotations: Callable[..., Any]
    _efa_plugin_pods: Callable[..., Any]
    _node_allocatable: Callable[..., Any]
    _pod_name: Callable[..., Any]
    _pod_ready: Callable[..., Any]
    _pod_uid: Callable[..., Any]
    _resource_version: Callable[..., Any]
    _taints: Callable[..., Any]
    _unschedulable: Callable[..., Any]
    alert_sender: Callable[..., Any]
    core: Any
    mechanical_email_builder: Any
    notification_sink: Any
    ownership_provider: Any

    def _restart_device_plugin(
        self, context: WorkflowStepContext
    ) -> WorkflowStepOutcome:
        parameters = context.step.parameters
        efa = context.step.operation is WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN
        namespace = str(parameters.get("plugin_namespace", "kube-system"))
        selector = str(
            parameters.get(
                "plugin_label_selector",
                (
                    "name=hyperpod-dependencies-aws-efa-k8s-device-plugin"
                    if efa
                    else "app.kubernetes.io/name=nvidia-device-plugin"
                ),
            )
        )
        resource_name = str(
            parameters.get(
                "resource_name",
                ("vpc.amazonaws.com/efa" if efa else "nvidia.com/gpu"),
            )
        )
        operation_annotation = (
            ANNOTATION_EFA_PLUGIN_RESTART_OPERATION
            if efa
            else ANNOTATION_GPU_PLUGIN_RESTART_OPERATION
        )
        pod_uid_annotation = (
            ANNOTATION_EFA_PLUGIN_RESTART_POD_UID
            if efa
            else ANNOTATION_GPU_PLUGIN_RESTART_POD_UID
        )
        started_annotation = (
            ANNOTATION_EFA_PLUGIN_RESTART_STARTED_AT
            if efa
            else ANNOTATION_GPU_PLUGIN_RESTART_STARTED_AT
        )
        expected = int(parameters.get("expected_count", 1))
        timeout_seconds = int(parameters.get("restart_timeout_seconds", 180))
        node_results: dict[str, dict[str, Any]] = {}
        waiting = False
        now = datetime.now(timezone.utc)
        for node_id in context.step.node_ids:
            node = self.core.read_node(node_id)
            allocatable = self._node_allocatable(node, resource_name)
            annotations = self._annotations(node)
            operation = annotations.get(operation_annotation)
            old_uid = annotations.get(pod_uid_annotation, "")
            started_raw = annotations.get(started_annotation)
            pods = self._efa_plugin_pods(
                node_id,
                namespace=namespace,
                label_selector=selector,
            )
            ready_new_pods = [
                pod
                for pod in pods
                if self._pod_ready(pod) and self._pod_uid(pod) != old_uid
            ]
            if allocatable >= expected and (
                operation is None
                or operation == context.idempotency_key
                and ready_new_pods
            ):
                if operation == context.idempotency_key:
                    self.core.patch_node(
                        node_id,
                        {
                            "metadata": {
                                "resourceVersion": (self._resource_version(node)),
                                "annotations": {
                                    operation_annotation: None,
                                    pod_uid_annotation: None,
                                    started_annotation: None,
                                },
                            }
                        },
                    )
                node_results[node_id] = {
                    "allocatable": allocatable,
                    "expected": expected,
                    "already_healthy": operation is None,
                    "replacement_pod_uids": [
                        self._pod_uid(pod) for pod in ready_new_pods
                    ],
                }
                continue
            if operation and operation != context.idempotency_key:
                return WorkflowStepOutcome.failed(
                    f"node {node_id} has another EFA plugin restart "
                    f"in progress: {operation}"
                )
            if operation == context.idempotency_key:
                try:
                    started = datetime.fromisoformat(
                        str(started_raw).replace("Z", "+00:00")
                    )
                except (TypeError, ValueError):
                    started = now
                if (now - started).total_seconds() >= timeout_seconds:
                    return WorkflowStepOutcome.failed(
                        f"device plugin did not restore {resource_name} on {node_id}",
                        details={
                            "node_id": node_id,
                            "allocatable": allocatable,
                            "expected": expected,
                            "plugin_pods": [self._pod_name(pod) for pod in pods],
                        },
                    )
                waiting = True
                node_results[node_id] = {
                    "allocatable": allocatable,
                    "expected": expected,
                    "waiting_for_replacement_pod": True,
                }
                continue
            if not pods:
                self.core.patch_node(
                    node_id,
                    {
                        "metadata": {
                            "resourceVersion": self._resource_version(node),
                            "annotations": {
                                operation_annotation: (context.idempotency_key),
                                pod_uid_annotation: "",
                                started_annotation: now.isoformat(),
                            },
                        }
                    },
                )
                waiting = True
                node_results[node_id] = {
                    "allocatable": allocatable,
                    "expected": expected,
                    "waiting_for_daemonset_pod": True,
                }
                continue
            if len(pods) != 1:
                return WorkflowStepOutcome.failed(
                    f"expected at most one device plugin Pod on "
                    f"{node_id}, found {len(pods)}"
                )
            pod = pods[0]
            pod_uid = self._pod_uid(pod)
            pod_name = self._pod_name(pod)
            self.core.patch_node(
                node_id,
                {
                    "metadata": {
                        "resourceVersion": self._resource_version(node),
                        "annotations": {
                            operation_annotation: context.idempotency_key,
                            pod_uid_annotation: pod_uid,
                            started_annotation: now.isoformat(),
                        },
                    }
                },
            )
            self.core.delete_namespaced_pod(
                pod_name,
                namespace,
                grace_period_seconds=0,
            )
            waiting = True
            node_results[node_id] = {
                "deleted_pod": f"{namespace}/{pod_name}",
                "deleted_pod_uid": pod_uid,
                "allocatable": allocatable,
                "expected": expected,
            }
        if waiting:
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={"node_results": node_results},
            )
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={"node_results": node_results},
        )

    def _check_mechanicals(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        expected = f"{context.incident.incident_id}:{context.workflow.fencing_token}"
        pending = []
        confirmations = {}
        for node_id in context.step.node_ids:
            node = self.core.read_node(node_id)
            actual = self._annotations(node).get(
                ANNOTATION_MECHANICAL_INSPECTION_COMPLETE
            )
            if actual != expected:
                pending.append(node_id)
            else:
                confirmations[node_id] = actual
        if pending:
            notification_id = None
            if self.notification_sink is not None:
                notification = self.notification_sink.save_notification_if_absent(
                    self.mechanical_email_builder.build(
                        cluster_id=context.incident.cluster_id,
                        incident_id=context.incident.incident_id,
                        workflow_id=context.workflow.request_id,
                        node_ids=pending,
                        link_id=context.step.parameters.get("nvlink_link_id"),
                        pci_bdf=context.step.parameters.get("pci_bdf"),
                        occurrence_counts=context.step.parameters.get(
                            "nvlink_occurrence_counts", {}
                        ),
                        annotation=(ANNOTATION_MECHANICAL_INSPECTION_COMPLETE),
                        annotation_value=expected,
                        xid=int(context.step.parameters.get("xid", 74)),
                    )
                )
                notification_id = notification.notification_id
                if self.alert_sender is not None:
                    self.alert_sender(notification_id)
            return WorkflowStepOutcome.waiting(
                details={
                    "pending_nodes": pending,
                    "required_annotation": (ANNOTATION_MECHANICAL_INSPECTION_COMPLETE),
                    "required_annotation_value": expected,
                    "acknowledgement_commands": [
                        "kubectl annotate node "
                        f"{node_id} "
                        f"{ANNOTATION_MECHANICAL_INSPECTION_COMPLETE}"
                        f"='{expected}' --overwrite"
                        for node_id in pending
                    ],
                    "required_evidence": (
                        "physical seating and applicable NVLink "
                        "connections inspected; reseating completed "
                        "when required"
                    ),
                    "notification_id": notification_id,
                }
            )
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "confirmed_nodes": sorted(confirmations),
                "confirmation_values": confirmations,
            },
        )

    def _isolate(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        absent_nodes = []
        for node_id in context.step.node_ids:
            for attempt in range(3):
                try:
                    node = self.core.read_node(node_id)
                except Exception as exc:
                    if getattr(exc, "status", None) == 404:
                        absent_nodes.append(node_id)
                        break
                    raise
                try:
                    body = self._node_isolation_patch(node, context)
                except NodeIsolationRejected as exc:
                    return WorkflowStepOutcome.failed(
                        str(exc),
                        details={
                            "safety_rejection": True,
                            "node_id": node_id,
                        },
                    )
                try:
                    self.core.patch_node(node_id, body)
                    break
                except Exception as exc:
                    if getattr(exc, "status", None) != 409 or attempt == 2:
                        raise
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "isolated_nodes": [
                    node_id
                    for node_id in context.step.node_ids
                    if node_id not in absent_nodes
                ],
                "already_absent_nodes": absent_nodes,
            },
        )

    def _restore(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        for node_id in context.step.node_ids:
            node = self.core.read_node(node_id)
            annotations = self._annotations(node)
            if annotations.get(
                ANNOTATION_INCIDENT
            ) != context.incident.incident_id or annotations.get(
                ANNOTATION_FENCING
            ) != str(context.workflow.fencing_token):
                return WorkflowStepOutcome.failed(
                    f"node {node_id} isolation ownership does not "
                    "match incident/fencing token"
                )
            taints = [
                item
                for item in self._taints(node)
                if item.get("key") != QUARANTINE_TAINT
            ]
            was_unschedulable = (
                annotations.get(
                    ANNOTATION_PREVIOUS_UNSCHEDULABLE,
                    "false",
                ).lower()
                == "true"
            )
            self.core.patch_node(
                node_id,
                {
                    "metadata": {
                        "resourceVersion": self._resource_version(node),
                        "annotations": {
                            ANNOTATION_INCIDENT: None,
                            ANNOTATION_FENCING: None,
                            ANNOTATION_PREVIOUS_UNSCHEDULABLE: None,
                        },
                    },
                    "spec": {
                        "unschedulable": was_unschedulable,
                        "taints": taints,
                    },
                },
            )
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={"restored_nodes": context.step.node_ids},
        )

    def _incident_workflow_is_terminal(self, incident_id: str) -> bool:
        """Whether another incident's workflow has finished.

        Fails closed on every error path: an unreachable control plane or
        an unknown incident must not license taking a node away from a
        workflow that could still be running.
        """
        if self.store is not None:
            try:
                incident = self.store.get_incident(incident_id)
                workflow = self.store.get_workflow(incident.workflow_request_id)
            except (NotFoundError, KeyError, TypeError):
                return False
            return workflow.status in {
                WorkflowStatus.BLOCKED,
                WorkflowStatus.SUCCEEDED,
                WorkflowStatus.FAILED,
                WorkflowStatus.SUPERSEDED,
            }
        if self.ownership_provider is None:
            return False
        try:
            return bool(
                self.ownership_provider.incident_workflow_is_terminal(incident_id)
            )
        except Exception:
            return False

    def _incident_ownership(self, incident_id: str) -> tuple[bool, bool]:
        if self.store is not None:
            try:
                incident = self.store.get_incident(incident_id)
                workflow = self.store.get_workflow(incident.workflow_request_id)
            except (NotFoundError, KeyError, TypeError):
                return False, False
            terminal = workflow.status in {
                WorkflowStatus.BLOCKED,
                WorkflowStatus.SUCCEEDED,
                WorkflowStatus.FAILED,
                WorkflowStatus.SUPERSEDED,
            }
            quarantine_hold = (
                incident.state is IncidentState.QUARANTINED
                or (
                    WorkflowOperation.QUARANTINE in workflow.completed_operations
                    and WorkflowOperation.RESTORE_SCHEDULING
                    not in workflow.completed_operations
                )
                or any(
                    execution.operation is WorkflowOperation.REPLACE_NODE
                    and execution.details.get("action") == "SPARE_FAILOVER"
                    for execution in workflow.step_executions
                )
            )
            return terminal, quarantine_hold
        if self.ownership_provider is None:
            return False, False
        try:
            if hasattr(
                self.ownership_provider,
                "incident_ownership",
            ):
                report = self.ownership_provider.incident_ownership(incident_id)
                return (
                    bool(report.known and report.terminal),
                    bool(report.quarantine_hold),
                )
            return (
                bool(
                    self.ownership_provider.incident_workflow_is_terminal(incident_id)
                ),
                False,
            )
        except Exception:
            return False, False

    @staticmethod
    def _workflow_preserves_quarantine(
        workflow: WorkflowRequest,
    ) -> bool:
        operations = {step.operation for step in workflow.official_steps}
        return (
            WorkflowOperation.QUARANTINE in operations
            and WorkflowOperation.RESTORE_SCHEDULING not in operations
        )

    def _can_take_over_node_isolation(
        self,
        node: Any,
        incident_id: str,
        context: WorkflowStepContext,
    ) -> bool:
        terminal, quarantine_hold = self._incident_ownership(incident_id)
        return terminal and (
            not quarantine_hold or self._workflow_preserves_quarantine(context.workflow)
        )

    def _node_isolation_patch(
        self, node: Any, context: WorkflowStepContext
    ) -> dict[str, Any]:
        annotations = self._annotations(node)
        existing_incident = annotations.get(ANNOTATION_INCIDENT)
        existing_fencing = annotations.get(ANNOTATION_FENCING)
        newer_same_incident_generation = False
        existing_fencing_token = None
        if existing_fencing is not None:
            try:
                existing_fencing_token = int(existing_fencing)
            except ValueError as exc:
                raise ValueError("node has invalid gpu-fault fencing token") from exc
        if (
            existing_incident == context.incident.incident_id
            and existing_fencing_token is not None
        ):
            if context.workflow.fencing_token < existing_fencing_token:
                raise NodeIsolationRejected(
                    "node is controlled by a newer workflow generation"
                )
            newer_same_incident_generation = (
                context.workflow.fencing_token > existing_fencing_token
            )
        takeover = bool(
            existing_incident
            and (
                existing_incident != context.incident.incident_id
                or existing_fencing != str(context.workflow.fencing_token)
            )
            and (
                newer_same_incident_generation
                or (
                    existing_incident != context.incident.incident_id
                    and self._can_take_over_node_isolation(
                        node, existing_incident, context
                    )
                )
                or (
                    existing_incident == context.incident.incident_id
                    and existing_fencing_token is None
                    and self._can_take_over_node_isolation(
                        node, existing_incident, context
                    )
                )
            )
        )
        if (
            existing_incident
            and not takeover
            and (
                existing_incident != context.incident.incident_id
                or existing_fencing != str(context.workflow.fencing_token)
            )
        ):
            raise NodeIsolationRejected(
                "node is already isolated by another incident/token"
            )
        previous_unschedulable = annotations.get(ANNOTATION_PREVIOUS_UNSCHEDULABLE)
        # This annotation is the scheduling baseline from before the first
        # isolation. A successor incident takes over an already cordoned node,
        # so resampling here would turn that intermediate state into the new
        # baseline and make validation-first restore leave the node cordoned.
        if previous_unschedulable is None:
            previous_unschedulable = str(self._unschedulable(node)).lower()
        taints = self._taints(node)
        if takeover:
            taints = [item for item in taints if item.get("key") != QUARANTINE_TAINT]
        owned_taint_values = {
            context.incident.incident_id,
            quarantine_taint_value(context.incident.incident_id),
        }
        if not any(
            item.get("key") == QUARANTINE_TAINT
            and item.get("value") in owned_taint_values
            for item in taints
        ):
            taints.append(
                {
                    "key": QUARANTINE_TAINT,
                    "value": quarantine_taint_value(context.incident.incident_id),
                    "effect": "NoSchedule",
                }
            )
        return {
            "metadata": {
                "resourceVersion": self._resource_version(node),
                "annotations": {
                    ANNOTATION_INCIDENT: context.incident.incident_id,
                    ANNOTATION_FENCING: str(context.workflow.fencing_token),
                    ANNOTATION_PREVIOUS_UNSCHEDULABLE: (previous_unschedulable),
                },
            },
            "spec": {
                "unschedulable": True,
                "taints": taints,
            },
        }
