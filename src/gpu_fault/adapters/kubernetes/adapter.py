from __future__ import annotations

from typing import Any

from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.telemetry import (
    EvidenceService,
)
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepSpec,
)
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)
from gpu_fault.notifications import (
    Nvlink74MechanicalEmailBuilder,
    RestartGuardEmailBuilder,
)
from gpu_fault.adapters.kubernetes.node_operations import KubernetesNodeOperationsMixin
from gpu_fault.adapters.kubernetes.primitives import KubernetesPrimitivesMixin
from gpu_fault.adapters.kubernetes.restart_operations import (
    KubernetesRestartOperationsMixin,
)
from gpu_fault.adapters.kubernetes.workload_operations import (
    KubernetesWorkloadOperationsMixin,
)


class KubernetesWorkflowAdapter(
    KubernetesNodeOperationsMixin,
    KubernetesWorkloadOperationsMixin,
    KubernetesRestartOperationsMixin,
    KubernetesPrimitivesMixin,
):
    """Kubernetes scheduler and workload mutation adapter.

    It writes only the gpu-fault.io taint/annotations and never removes
    provider-owned taints.
    """

    OPERATIONS = operations_for_adapter(OperationAdapter.KUBERNETES)

    def __init__(
        self,
        *,
        owner: str = "gpu-fault-kubernetes-adapter",
        core_api: Any | None = None,
        batch_api: Any | None = None,
        custom_api: Any | None = None,
        store: Any | None = None,
        notification_sink: Any | None = None,
        alert_sender=None,
        ownership_provider: Any | None = None,
        evidence_sink: Any | None = None,
        workload_log_tail_lines: int = 2000,
        workload_log_max_bytes: int = 262144,
        workload_log_s3_uri: str | None = None,
        workload_log_s3_max_bytes: int = 104857600,
        workload_log_uploader=None,
    ) -> None:
        if workload_log_tail_lines < 1:
            raise ValueError("workload log tail lines must be positive")
        if workload_log_max_bytes < 4096:
            raise ValueError("workload log max bytes must be at least 4096")
        if workload_log_s3_max_bytes < workload_log_max_bytes:
            raise ValueError("workload log S3 max bytes must cover the tail")
        self.owner = owner
        self.store = store
        self.notification_sink = notification_sink or store
        self.evidence_sink = evidence_sink
        self.evidence_service = (
            EvidenceService.from_environment(store) if store is not None else None
        )
        self.workload_log_tail_lines = workload_log_tail_lines
        self.workload_log_max_bytes = workload_log_max_bytes
        self.workload_log_s3_uri = (
            workload_log_s3_uri.rstrip("/") if workload_log_s3_uri else None
        )
        self.workload_log_s3_max_bytes = workload_log_s3_max_bytes
        self.workload_log_uploader = workload_log_uploader
        # Answers "is the incident that currently owns this node/workload
        # already finished?" when there is no local store to ask. The
        # regional executor runs with REMOTE_STATE=true and therefore
        # store=None, which made every takeover branch below dead code:
        # a node left annotated by a workflow that died before
        # RESTORE_SCHEDULING was refused forever.
        self.ownership_provider = ownership_provider
        self.alert_sender = alert_sender
        self.restart_email_builder = RestartGuardEmailBuilder()
        self.mechanical_email_builder = Nvlink74MechanicalEmailBuilder()
        if core_api is None or batch_api is None or custom_api is None:
            core_api, batch_api, custom_api = self._clients()
        self.core = core_api
        self.batch = batch_api
        self.custom = custom_api

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == self.owner and step.operation in self.OPERATIONS

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        operation = context.step.operation
        if operation in {
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUARANTINE,
        }:
            return self._isolate(context)
        if operation is WorkflowOperation.RESTORE_SCHEDULING:
            return self._restore(context)
        if operation is WorkflowOperation.CHECKPOINT_WORKLOADS:
            checkpoint = context.step.parameters.get("checkpoint_manifest_ref")
            if not checkpoint:
                return WorkflowStepOutcome.failed(
                    "checkpoint manifest evidence is missing"
                )
            return WorkflowStepOutcome.succeeded(
                operation_id=context.idempotency_key,
                details={"checkpoint_manifest_ref": checkpoint},
            )
        if operation is WorkflowOperation.STOP_WORKLOADS:
            return self._set_workloads(context, suspend=True)
        if operation is WorkflowOperation.CHECK_MECHANICALS:
            return self._check_mechanicals(context)
        if operation in {
            WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
            WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
        }:
            return self._restart_device_plugin(context)
        return self._set_workloads(context, suspend=False)
