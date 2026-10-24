"""Real API/policy/executor/validation; scheduler client and device I/O are local."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from gpu_fault import training_health
from gpu_fault.adapters import ControlPlaneEvidenceAdapter, gpu_validation
from gpu_fault.adapters.kubernetes.adapter import KubernetesWorkflowAdapter
from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.execution import ProductionExecutorConfig, WorkflowStepOutcome
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
)
from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
    NodeLogBatch,
    NodeLogEntry,
)
from gpu_fault.models import WorkflowExecutionRequest
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.store import SqliteStore
from tests._builders import asgi_client, attempt_observation, container_observation
from tests.execution._support import FakeCoreApi, UnusedApi

JOB = "training/job/train"


class SchedulerAdapter(KubernetesWorkflowAdapter):
    def supports(self, step):
        return step.operation in {
            Op.MARK_UNSCHEDULABLE,
            Op.QUARANTINE,
            Op.RESTORE_SCHEDULING,
        } and super().supports(step)


class DeviceBoundary:
    owner = "simulated-runtime"

    def __init__(self):
        self.calls = []
        self.running = True

    def supports(self, step):
        return step.operation in {
            Op.STOP_WORKLOADS,
            Op.RESTART_WORKLOAD,
            Op.COLLECT_DIAGNOSTIC_BUNDLE,
            Op.COLLECT_HUNG_TRIAGE,
            Op.COLLECT_HUNG_SUMMARY,
            Op.TRIGGER_HEALTH_SNAPSHOT,
            Op.ESCALATE_SUPPORT,
        }

    def execute(self, context):
        self.calls.append(context.step.operation)
        if context.step.operation in {Op.STOP_WORKLOADS, Op.RESTART_WORKLOAD}:
            assert context.step.workload_ids == [JOB], context.step
            self.running = context.step.operation is Op.RESTART_WORKLOAD
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={"node_results": {"node-a": {"ranks": []}}},
        )


class HealthRig:
    def __init__(self, path, monkeypatch):
        self.clock = datetime.now(timezone.utc)
        self.sequence = 0
        self.boundary = DeviceBoundary()
        self.core = FakeCoreApi()
        self.store = SqliteStore(str(path / "health.db"))
        owner = "simulated-runtime"
        self.context = ApplicationContext(
            store=self.store,
            production_executor_config=ProductionExecutorConfig(
                enabled=True,
                executor_id="health-policy-test",
                allowed_operations=frozenset(Op),
            ),
            production_adapters=[
                ControlPlaneEvidenceAdapter(owner=owner),
                gpu_validation.GpuValidationAdapter(
                    GpuMetricsService(store=self.store), store=self.store, owner=owner
                ),
                self.boundary,
                SchedulerAdapter(
                    owner=owner,
                    core_api=self.core,
                    batch_api=UnusedApi(),
                    custom_api=UnusedApi(),
                    store=self.store,
                ),
            ],
            execution_token="local-health-policy-" + "x" * 32,
        )
        self.store.save_profile(default_simulated_profile())
        rig = self

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return rig.clock if tz else rig.clock.replace(tzinfo=None)

        monkeypatch.setattr(gpu_validation, "datetime", Clock)
        monkeypatch.setattr(training_health, "datetime", Clock)
        self.observe()
        self.host()

    def observe(self):
        self.store.save_attempt_observation(
            attempt_observation(
                "train",
                "attempt-a",
                self.clock,
                expected_critical_ranks=1,
                workload_ids=[JOB],
                started_at=self.clock - timedelta(minutes=10),
                containers=[
                    container_observation(
                        "pod-a",
                        "trainer",
                        0,
                        "node-a",
                        gpu_uuids=["GPU-a"],
                        host_pid=123,
                    )
                ],
            )
        )

    def advance(self, seconds):
        self.clock += timedelta(seconds=seconds)
        self.observe()

    def post(self, path, payload):
        async def submit():
            async with asgi_client(self.context) as client:
                response = await client.post(path, json=payload)
                assert response.status_code == 200, response.text
                return response.json()

        return asyncio.run(submit())

    def host(self):
        batch = HostTelemetryBatch(
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=self.clock,
            received_at=self.clock,
            runtime_profile_version="simulated-v1",
            samples=[
                HostMetricSample(name=name, value=value)
                for name, value in (
                    ("load1_per_cpu", 0.1),
                    ("memory_used_percent", 20),
                    ("filesystem_used_percent", 20),
                    ("rdma_link_down", 0),
                    ("rdma_errors_delta", 0),
                    ("network_link_up", 1),
                )
            ],
        )
        return self.post(
            "/v1/collector-events/host-telemetry", batch.model_dump(mode="json")
        )

    def bind_task_fabric(self, workflow):
        current = self.store.get_workflow(workflow.request_id)
        scope = {
            "basis": "TASK_DEVICE_BINDING",
            "network_interfaces": ["eth-training"],
            "rdma_ports": ["efa0/1"],
        }
        self.store.save_workflow(
            current.model_copy(
                update={
                    "official_steps": [
                        step.model_copy(
                            update={
                                "parameters": {
                                    **step.parameters,
                                    "training_fabric_by_node": {"node-a": scope},
                                }
                            }
                        )
                        if step.operation is Op.VALIDATE_FABRIC
                        else step
                        for step in current.official_steps
                    ]
                }
            )
        )
        self.post(
            "/v1/collector-events/host-telemetry",
            {
                "cluster_id": "cluster-a",
                "node_id": "node-a",
                "observed_at": self.clock.isoformat(),
                "runtime_profile_version": "simulated-v1",
                "samples": [
                    {"name": "network_link_up", "value": 1, "device": "eth-training"},
                    {
                        "name": "network_errors_delta",
                        "value": 0,
                        "device": "eth-training",
                    },
                    {
                        "name": "network_drops_delta",
                        "value": 0,
                        "device": "eth-training",
                    },
                    {"name": "rdma_link_down", "value": 0, "device": "efa0/1"},
                    {"name": "rdma_errors_delta", "value": 0, "device": "efa0/1"},
                ],
            },
        )

    def gpu(self, values, *, errors=None):
        self.sequence += 1
        batch = GpuMetricBatch(
            batch_id=f"metric-{self.sequence}",
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=self.clock,
            source=GpuMetricSource.DCGM_EXPORTER,
            product="H100",
            runtime_profile_version="simulated-v1",
            samples=[
                GpuMetricSample(
                    metric_name=name,
                    canonical_name=name,
                    value=value,
                    gpu_uuid="GPU-a",
                    pci_bdf="0000:81:00.0",
                )
                for name, value in values.items()
            ],
            collection_errors=errors or [],
        )
        return self.post(
            "/v1/collector-events/gpu-metrics", batch.model_dump(mode="json")
        )

    def log(self, text, *, source="dmesg", fields=None):
        self.sequence += 1
        batch = NodeLogBatch(
            cluster_id="cluster-a",
            node_id="node-a",
            collected_at=self.clock,
            runtime_profile_version="simulated-v1",
            entries=[
                NodeLogEntry(
                    entry_id=f"log-{self.sequence}",
                    observed_at=self.clock,
                    source=source,
                    message=text,
                    fields=fields or {},
                )
            ],
        )
        return self.post(
            "/v1/collector-events/node-logs", batch.model_dump(mode="json")
        )

    def execute(self, workflow):
        return self.context.workflow_executor.execute(
            workflow.request_id,
            WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        )
