from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu_fault.adapters import ControlPlaneEvidenceAdapter
from gpu_fault.adapters.gpu_validation import GpuValidationAdapter
from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.collectors import CollectorContext, HostTelemetryCollector
from gpu_fault.collectors.host import system_metrics
from gpu_fault.execution import ProductionExecutorConfig
from gpu_fault.gpu_metrics import GpuMetricsService
from gpu_fault.host_health import HostMetricSample
from gpu_fault.models import WorkflowExecutionRequest, WorkflowStatus
from gpu_fault.models import WorkflowOperation as Op
from tests._builders import asgi_client, build_store


@pytest.mark.parametrize(
    "recovered", [False, True], ids=["still-unhealthy", "recovered"]
)
def test_swap_fault_from_collector_to_api_and_real_validation(
    tmp_path, monkeypatch, recovered
):
    memory = tmp_path / "meminfo"
    memory.write_text(
        "MemTotal: 1000 kB\nMemAvailable: 800 kB\nMemFree: 700 kB\n"
        "Cached: 100 kB\nSwapTotal: 1000 kB\nSwapFree: 100 kB\n"
    )
    monkeypatch.setattr(
        system_metrics,
        "Path",
        lambda value: memory if value == "/proc/meminfo" else Path(value),
    )
    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ("_memory",))

    class Sink:
        def post(self, *_):
            return {}

    collector = HostTelemetryCollector(
        Sink(),
        CollectorContext(
            cluster_id="cluster-a", runtime_profile_version="simulated-v1"
        ),
        node_id="node-a",
        edge_filter_enabled=False,
        now=lambda: datetime.now(timezone.utc),
    )
    store = build_store()
    owner = "simulated-runtime"
    ctx = ApplicationContext(
        store=store,
        production_executor_config=ProductionExecutorConfig(
            enabled=True,
            executor_id="host-e2e",
            allowed_operations=frozenset({Op.FREEZE_EVIDENCE, Op.VALIDATE_HOST}),
        ),
        production_adapters=[
            ControlPlaneEvidenceAdapter(owner=owner),
            GpuValidationAdapter(
                GpuMetricsService(store=store), store=store, owner=owner
            ),
        ],
        execution_token="local-host-test-" + "x" * 32,
    )
    store.save_profile(default_simulated_profile())

    async def scenario():
        async with asgi_client(ctx) as client:

            async def publish():
                batch = collector.collect_once()
                batch.samples.extend(
                    [
                        HostMetricSample(name="load1_per_cpu", value=0.1),
                        HostMetricSample(name="filesystem_used_percent", value=20),
                    ]
                )
                response = await client.post(
                    "/v1/collector-events/host-telemetry",
                    json=batch.model_dump(mode="json"),
                )
                assert response.status_code == 200, response.text
                return response.json()

            decision = await publish()
            assert len(decision["workflow_request_ids"]) == 1, decision
            if recovered:
                memory.write_text(
                    memory.read_text().replace("SwapFree: 100", "SwapFree: 900")
                )
                await publish()
            request_id = decision["workflow_request_ids"][0]
            result = ctx.workflow_executor.execute(
                request_id, WorkflowExecutionRequest(expected_fencing_token=1)
            )
            assert result.status is (
                WorkflowStatus.SUCCEEDED if recovered else WorkflowStatus.FAILED
            ), result
            workflow = store.get_workflow(request_id)
            assert [step.operation for step in workflow.official_steps] == [
                Op.FREEZE_EVIDENCE,
                Op.VALIDATE_HOST,
            ]
            assert (
                workflow.official_steps[1].parameters["host_diagnostic_probe"]["metric"]
                == "swap_used_percent"
            )

    asyncio.run(scenario())
