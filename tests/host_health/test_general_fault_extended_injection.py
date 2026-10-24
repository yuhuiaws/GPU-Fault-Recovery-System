"""Additional raw host injections through API, workflow and real validation."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.collectors import CollectorContext
from gpu_fault.collectors.gpu.dcgm import DcgmMetricsCollector
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowStatus
from tests._builders import asgi_client
from tests.host_health._general_fault_e2e import HostFaultRig


@pytest.mark.parametrize(
    "kind,metric,seconds",
    [
        ("page-cache", "page_cache_percent", 300),
        ("shared", "shared_filesystem_used_percent", 0),
        ("disk-io", "disk_io_util_percent", 120),
        ("net-errors", "network_errors_delta", 0),
        ("rdma-errors", "rdma_errors_delta", 0),
    ],
    ids=["003", "005", "006", "009", "010"],
)
def test_extended_host_injection(tmp_path, monkeypatch, kind, metric, seconds):
    rig = HostFaultRig(tmp_path, monkeypatch, kind)
    rig.clock = datetime.now(timezone.utc) - timedelta(seconds=seconds + 45)

    async def scenario():
        async with asgi_client(rig.context) as client:
            await rig.publish(client)
            rig.clock += timedelta(seconds=15)
            rig.set_fault(True)
            _, first = await rig.publish(client)
            if seconds:
                assert not first["workflow_request_ids"], first
                rig.clock += timedelta(seconds=seconds)
                rig.set_fault(True)
                await rig.publish(client)
            workflows = rig.store.list_workflows()
            assert workflows, (kind, first)
            for workflow in workflows:
                operations = [step.operation for step in workflow.official_steps]
                assert set(operations) <= {
                    Op.FREEZE_EVIDENCE,
                    Op.VALIDATE_HOST,
                    Op.VALIDATE_FABRIC,
                }, operations
            rig.set_fault(False)
            rig.clock += timedelta(seconds=15)
            await rig.publish(client)
            # A zero delta after the recovery sample proves that the same
            # counter source, rather than a different interface, stopped failing.
            rig.clock += timedelta(seconds=15)
            batch, _ = await rig.publish(client)
            assert any(sample.name == metric for sample in batch.samples), batch
            if kind in {"net-errors", "rdma-errors"}:

                class Sink:
                    def post(self, *_):
                        return {}

                gpu = DcgmMetricsCollector(
                    Sink(),
                    CollectorContext(
                        cluster_id="cluster-a",
                        product="H100",
                        runtime_profile_version="simulated-v1",
                    ),
                    node_id="node-a",
                    edge_filter_enabled=False,
                ).collect_text(
                    "".join(
                        f'{name}{{UUID="GPU-a"}} {value}\n'
                        for name, value in {
                            "DCGM_FI_DEV_GPU_TEMP": 40,
                            "DCGM_FI_DEV_POWER_MGMT_LIMIT": 500,
                            "DCGM_FI_DEV_ROW_REMAP_FAILURE": 0,
                            "DCGM_FI_DEV_ROW_REMAP_PENDING": 0,
                            "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": 0,
                            "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": 0,
                            "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": 0,
                            "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL": 0,
                            "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL": 0,
                            "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL": 0,
                        }.items()
                    ),
                    observed_at=rig.clock,
                )
                response = await client.post(
                    "/v1/collector-events/gpu-metrics", json=gpu.model_dump(mode="json")
                )
                assert response.status_code == 200, response.text
            for workflow in workflows:
                result = rig.execute(workflow.request_id)
                assert result.status is WorkflowStatus.SUCCEEDED, (
                    rig.store.get_workflow(workflow.request_id).step_executions
                )
            assert rig.core.node["spec"]["unschedulable"] is False, rig.core.node
            assert rig.store.list_remote_commands() == [], (
                "diagnosis cannot dispatch hardware recovery"
            )

    try:
        asyncio.run(scenario())
    finally:
        rig.close()
