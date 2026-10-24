from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.models import WorkflowOperation as Op
from tests._builders import asgi_client
from tests.host_health._general_fault_e2e import HostFaultRig


@pytest.mark.parametrize("kind", ["cpu", "memory", "swap"])
@pytest.mark.parametrize("recovered", [False, True], ids=["unhealthy", "recovered"])
def test_resource_fault_reaches_real_validation(tmp_path, monkeypatch, kind, recovered):
    rig = HostFaultRig(tmp_path, monkeypatch, kind)

    async def scenario():
        async with asgi_client(rig.context) as client:
            if kind == "cpu":
                _, baseline = await rig.publish(client)
                assert baseline["workflow_request_ids"] == [], baseline
            rig.clock += timedelta(seconds=15)
            rig.set_fault(True)
            _, decision = await rig.publish(client)
            if kind in {"cpu", "memory"}:
                assert decision["workflow_request_ids"] == [], decision
                rig.clock += timedelta(seconds=120)
                if kind == "cpu":
                    (rig.proc / "stat").write_text("cpu 12079 0 0 921 0 0 0 0\n")
                _, decision = await rig.publish(client)
            expected_metric = {
                "cpu": "cpu_usage_percent",
                "memory": "memory_used_percent",
                "swap": "swap_used_percent",
            }[kind]
            workflows = rig.store.list_workflows()
            targets = [
                workflow
                for workflow in workflows
                if workflow.official_steps[-1].parameters["host_diagnostic_probe"][
                    "metric"
                ]
                == expected_metric
            ]
            assert len(targets) == 1, (decision, workflows)
            request_id = targets[0].request_id
            if recovered:
                rig.set_fault(False)
                if kind == "cpu":
                    (rig.proc / "stat").write_text("cpu 12080 0 0 1020 0 0 0 0\n")
                rig.clock += timedelta(seconds=15)
                await rig.publish(client)
            result = rig.execute(request_id)
            assert result.status is (
                WorkflowStatus.SUCCEEDED if recovered else WorkflowStatus.FAILED
            ), result
            workflow = rig.store.get_workflow(request_id)
            assert [step.operation for step in workflow.official_steps] == [
                Op.FREEZE_EVIDENCE,
                Op.VALIDATE_HOST,
            ], workflow
            assert rig.core.node["spec"]["unschedulable"] is False, rig.core.node
            assert rig.store.list_remote_commands() == [], "no hardware dispatch"
            assert len(rig.store.list_workflows()) == len(workflows), (
                "no hardware escalation"
            )

    try:
        asyncio.run(scenario())
    finally:
        rig.close()


@pytest.mark.parametrize(
    "kind", ["filesystem", "smart", "bmc", "net-missing", "net-down"]
)
def test_hardware_fault_reaches_real_quarantine_adapter(tmp_path, monkeypatch, kind):
    rig = HostFaultRig(tmp_path, monkeypatch, kind)
    rig.set_fault(True)

    async def scenario():
        async with asgi_client(rig.context) as client:
            _, decision = await rig.publish(client)
            assert len(decision["workflow_request_ids"]) == 1, decision
            request_id = decision["workflow_request_ids"][0]
            result = rig.execute(request_id)
            assert result.status is WorkflowStatus.SUCCEEDED, result
            workflow = rig.store.get_workflow(request_id)
            assert [step.operation for step in workflow.official_steps] == [
                Op.FREEZE_EVIDENCE,
                Op.MARK_UNSCHEDULABLE,
                Op.QUARANTINE,
            ], workflow
            assert rig.core.node["spec"]["unschedulable"] is True, rig.core.node
            taints = rig.core.node["spec"]["taints"]
            assert {item["key"] for item in taints} == {
                "gpu-fault.io/quarantined",
                "sagemaker.amazonaws.com/node-health-status",
            }, taints
            incident = rig.store.get_incident(workflow.incident_id)
            assert incident.state is IncidentState.QUARANTINED, incident
            _, replay = await rig.publish(client)
            assert replay["workflow_request_ids"] == [], replay
            assert len(rig.store.list_workflows()) == 1, "no duplicate workflow"
            assert rig.store.list_remote_commands() == [], "no reset or reboot"

    try:
        asyncio.run(scenario())
    finally:
        rig.close()


@pytest.mark.parametrize("kind", ["smart-query", "bmc-query", "net-unknown"])
def test_observation_failure_does_not_isolate_node(tmp_path, monkeypatch, kind):
    rig = HostFaultRig(tmp_path, monkeypatch, kind)
    rig.set_fault(True)

    async def scenario():
        async with asgi_client(rig.context) as client:
            for _ in range(2):
                batch, decision = await rig.publish(client)
                assert batch.collection_errors, batch
                assert decision["workflow_request_ids"] == [], decision
                rig.clock += timedelta(seconds=15)
            assert rig.store.list_workflows() == [], "unknown is not hardware failure"
            assert rig.core.node["spec"]["unschedulable"] is False, rig.core.node

    try:
        asyncio.run(scenario())
    finally:
        rig.close()
