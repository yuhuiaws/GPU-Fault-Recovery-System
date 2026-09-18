from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from gpu_fault.adapters.gpu_validation import GpuValidationAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.gpu_metrics import GpuMetricSample, GpuMetricSource
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import workflow_step_execution
from tests.execution._cov95_runtime_workflows import FlowHarness


class ValidationFeed:
    def __init__(self, nodes: list[str], sampled_at: datetime) -> None:
        self.store = self
        self.sampled_at = sampled_at
        self.statuses = {
            node: [
                CollectorStatus(
                    cluster_id="cluster-a",
                    node_id=node,
                    collector=kind,
                    observed_at=sampled_at,
                    ingested_at=sampled_at,
                    last_success_at=sampled_at,
                )
                for kind in (CollectorKind.GPU_METRICS, CollectorKind.HOST_TELEMETRY)
            ]
            for node in nodes
        }
        self.host: dict[str, list[Any]] = {node: [] for node in nodes}
        self.gpu: dict[str, list[Any]] = {node: [] for node in nodes}
        self.gpu_reads: list[str] = []

    def list_collector_statuses(self, cluster_id: str, node_id: str):
        assert cluster_id == "cluster-a", "collector lookup must retain cluster scope"
        return self.statuses[node_id]

    def list_telemetry_metrics_latest(self, cluster_id: str, node_id: str):
        assert cluster_id == "cluster-a", (
            "host telemetry lookup must retain cluster scope"
        )
        return self.host[node_id]

    def latest(self, cluster_id: str, node_id: str):
        assert cluster_id == "cluster-a", (
            "GPU telemetry lookup must retain cluster scope"
        )
        self.gpu_reads.append(node_id)
        return self.gpu[node_id]

    def findings(self, cluster_id: str, node_id: str):
        assert cluster_id == "cluster-a" and node_id in self.gpu, (cluster_id, node_id)
        return []

    def host_metric(
        self, node: str, name: str, value: float, *, observed_at=None
    ) -> None:
        self.host[node] = [item for item in self.host[node] if item.name != name]
        self.host[node].append(
            SimpleNamespace(
                name=name, value=value, observed_at=observed_at or self.sampled_at
            )
        )

    def gpu_metric(
        self, node: str, name: str, *, source=GpuMetricSource.DCGM_EXPORTER
    ) -> None:
        self.gpu[node].append(
            SimpleNamespace(
                observed_at=self.sampled_at,
                source=source,
                sample=GpuMetricSample(
                    metric_name=name, canonical_name=name, value=0, gpu_uuid="GPU-a"
                ),
            )
        )


class ValidationHarness:
    def __init__(
        self,
        operation: WorkflowOperation,
        *,
        action: WorkflowOperation | None = None,
        requirements: dict[str, Any] | None = None,
        nodes: list[str] | None = None,
        **config: Any,
    ) -> None:
        nodes = nodes or ["node-a"]
        self.action_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        self.feed = ValidationFeed(nodes, self.action_at + timedelta(seconds=5))
        for node in nodes:
            for name, value in {
                "load1_per_cpu": 0.2,
                "memory_used_percent": 30,
                "filesystem_used_percent": 40,
                "network_link_up": 1,
            }.items():
                self.feed.host_metric(node, name, value)
            for name in (
                "gpu_temperature_c",
                "nvlink_crc_aggregate_error_total",
                "nvlink_recovery_aggregate_error_total",
                "nvlink_replay_aggregate_error_total",
            ):
                self.feed.gpu_metric(node, name)
        flow = FlowHarness(([action] if action is not None else []) + [operation])
        index = 1 if action is not None else 0
        self.adapter = GpuValidationAdapter(self.feed, **config)
        step = flow.workflow.official_steps[index].model_copy(
            update={
                "execution_owner": self.adapter.owner,
                "node_ids": nodes,
                "gpu_uuids": ["GPU-a"],
                "parameters": {"inventory_requirements_by_node": requirements or {}},
            }
        )
        workflow = flow.workflow.model_copy(
            update={
                "official_steps": [*flow.workflow.official_steps[:index], step],
                "step_executions": (
                    [
                        workflow_step_execution(
                            0,
                            action,
                            WorkflowStepStatus.SUCCEEDED,
                            updated_at=self.action_at,
                        )
                    ]
                    if action is not None
                    else []
                ),
            }
        )
        self.context = WorkflowStepContext(
            workflow=workflow,
            incident=flow.incident,
            step=step,
            step_index=index,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key="validation/unit",
        )

    def execute(self):
        return self.adapter.execute(self.context)
