"""NET-008 operation ownership and generation selection through real executors."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.orchestration.collector_outbox_maintenance import (
    build_collector_outbox_workflow,
)
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import StubFleetRegistry
from tests.node_agent._support import node_action_executor
from tests.regional._cov95_collect_net import (  # noqa: F401
    forbidden,
    no_external_effects,
)

OPERATION = WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE


def outbox_pair(action: str = "stats") -> Any:
    return build_collector_outbox_workflow(
        "cluster-a",
        "node-a",
        collector="kernel",
        action=action,
        confirm=action == "requeue-dead",
        path="/v1/kernel-logs",
        operator="fixture-operator",
        reference="fixture-reference",
        now=datetime.now(timezone.utc),
    )


def outbox_records(path: Path) -> None:
    path.parent.mkdir()
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "path": "/v1/kernel-logs",
                    "payload": {"record_id": "dead-a"},
                    "replayable": False,
                    "error": "HTTP 422",
                    "failed_at": "2026-09-01T00:00:00Z",
                },
                {
                    "path": "/v1/other",
                    "payload": {"record_id": "foreign"},
                    "replayable": False,
                    "error": "HTTP 404",
                },
                {
                    "path": "/v1/kernel-logs",
                    "payload": {"payload_event_key": "truncated"},
                    "payload_truncated": True,
                    "replayable": False,
                    "error": "HTTP 422",
                },
            ]
        )
        + "\n"
    )


@pytest.mark.parametrize("action", ["stats", "list", "requeue-dead"])
def test_outbox_operation_selects_live_generation_not_quiesce_generation(
    tmp_path: Path, action: str
) -> None:
    secret = "example-only-" * 4
    outbox = tmp_path / "outbox/kernel.ndjson"
    outbox_records(outbox)
    agent = node_action_executor(
        tmp_path,
        "ledger.db",
        allowed_operations={OPERATION},
        now=lambda: datetime.now(timezone.utc),
        secret=secret,
        agent_generation=7,
        collector_outbox_directory=str(outbox.parent),
        runner=forbidden,
        device_client_finder=forbidden,
        gpu_device_path_finder=forbidden,
    )
    registry = StubFleetRegistry({"node-a": "http://node.invalid:9099"}, generation=7)
    sent = []

    def sender(endpoint: str, envelope: Any) -> Any:
        assert endpoint == "http://node.invalid:9099"
        sent.append(envelope.command)
        return agent.execute(envelope)

    adapter = NodeActionWorkflowAdapter({}, secret, registry=registry, sender=sender)
    incident, workflow = outbox_pair(action)
    step = workflow.official_steps[0]
    assert adapter.supports(step), (
        "outbox workflow owner must select NodeAction adapter"
    )
    assert not adapter.supports(copy_model(step, execution_owner="foreign-owner")), (
        "another owner must not select the NodeAction adapter"
    )
    workflow = copy_model(
        workflow,
        official_steps=[workflow_step(WorkflowOperation.QUIESCE_GPU_SERVICES), step],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                details={
                    "agent_generations": {"node-a": 3},
                    "maintenance_window_expires_at": (
                        datetime.now(timezone.utc) - timedelta(hours=1)
                    ).isoformat(),
                },
            )
        ],
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=1,
        request=WorkflowExecutionRequest(expected_fencing_token=1),
        idempotency_key="workflow/outbox",
    )
    first = adapter.execute(context)
    assert first.status is WorkflowStepStatus.SUCCEEDED, first.error
    assert registry.restart_agent() == 8
    agent.set_agent_generation(8)
    second = adapter.execute(context)
    assert second.status is WorkflowStepStatus.SUCCEEDED, second.error
    assert [command.agent_generation for command in sent] == [7, 8]
    assert [command.command_id for command in sent] == [
        "workflow/outbox/node-a/agent-7",
        "workflow/outbox/node-a/agent-8",
    ]
    assert registry.maintenance_calls == [], (
        "outbox maintenance must not use quiesce generation"
    )
    assert registry.endpoint_calls == [("cluster-a", "node-a"), ("cluster-a", "node-a")]
    assert all(
        command.node_id == "node-a" and command.gpu_uuids == [] for command in sent
    ), "maintenance must retain node scope without acquiring GPU targets"
    if action == "requeue-dead":
        assert first.details["node_results"]["node-a"]["requeued"] == 1
        assert second.details["node_results"]["node-a"]["requeued"] == 0
        written = [json.loads(line) for line in outbox.read_text().splitlines()]
        assert [record["replayable"] for record in written] == [True, False, False]
    elif action == "list":
        records = first.details["node_results"]["node-a"]["records"]
        assert len(records) == 2
        assert all("payload" not in record for record in records), "metadata only"
    else:
        assert first.details["node_results"]["node-a"]["stats"]["dead"] == 3


@pytest.mark.parametrize(
    "ready", [False, True], ids=["stale-readiness", "generation-race"]
)
def test_outbox_generation_uncertainty_refuses_before_transport(
    tmp_path: Path, ready: bool
) -> None:
    class Registry(StubFleetRegistry):
        def endpoint(self, cluster_id: str, node_id: str) -> Any:
            if ready:
                self.restart_agent()
            return super().endpoint(cluster_id, node_id)

    registry = Registry(
        {"node-a": "http://node.invalid:9099"}, generation=7, ready=ready
    )
    adapter = NodeActionWorkflowAdapter(
        {}, "example-only-" * 4, registry=registry, sender=forbidden
    )
    incident, workflow = outbox_pair()
    outcome = adapter.execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=workflow.official_steps[0],
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=1),
            idempotency_key="workflow/outbox",
        )
    )
    assert outcome.status is WorkflowStepStatus.FAILED
    assert (
        "generation changed" if ready else "fleet consistency gate"
    ) in outcome.error
    assert registry.maintenance_calls == []


def test_outbox_wrong_owner_is_not_dispatched_by_workflow_executor() -> None:
    store = build_store()
    incident, workflow = outbox_pair()
    step = copy_model(workflow.official_steps[0], execution_owner="foreign-owner")
    workflow = copy_model(workflow, official_steps=[step])
    store.save_incident_and_workflow(incident, workflow)
    adapter = NodeActionWorkflowAdapter(
        {"node-a": "http://node.invalid:9099"}, "example-only-" * 4, sender=forbidden
    )
    executor = active_workflow_executor(store, [adapter], [OPERATION])
    result = execute_workflow(executor, workflow.request_id, expected_fencing_token=1)
    assert result.status is WorkflowStatus.FAILED
    assert store.get_workflow(workflow.request_id).completed_operations == []
