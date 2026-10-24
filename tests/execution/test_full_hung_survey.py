from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Lock
from types import SimpleNamespace

import pytest

from gpu_fault.adapters.node_action.adapter import NodeActionWorkflowAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepOutcome
from gpu_fault.execution.flight_temporal import membership_digest
from gpu_fault.hung_survey_analysis import survey_analysis
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowStepStatus
from gpu_fault.node_agent.protocol import NodeActionResult, NodeActionStatus
from tests._builders import workflow_step, workflow_step_execution
from tests.execution._hung_survey_support import add_communication, passive_document
from tests.execution._support import (
    WorkflowStepContext,
    fault_incident,
    workflow_request,
)
from tests.node_agent._support import SECRET


def context(nodes):
    step = workflow_step(
        Op.COLLECT_HUNG_SUMMARY,
        execution_owner="gpu-fault-node-agent",
        node_ids=nodes,
        parameters={
            "snapshot_epoch": "epoch",
            "job_id": "job-a",
            "attempt_id": "attempt-a",
            "workload_cgroup_paths_by_node": {node: ["/job"] for node in nodes},
        },
    )
    incident = fault_incident("inc-survey", "event-survey", "NODE_HEALTH")
    workflow = workflow_request(
        "workflow-survey", incident.incident_id, "RUNNING", 1, official_steps=[step]
    )
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=1),
        idempotency_key="survey",
    )


@pytest.mark.parametrize(
    "communication", [False, True], ids=["proc-only", "three-process-groups"]
)
def test_ten_thousand_ranks_are_all_visited_with_bounded_inflight_and_scoped_commands(
    communication,
):
    nodes = [f"node-{index}" for index in range(1250)]
    calls = []
    lock = Lock()
    inflight = 0
    peak = 0

    def sender(endpoint, envelope):
        nonlocal inflight, peak
        command = envelope.command
        with lock:
            inflight += 1
            peak = max(peak, inflight)
            calls.append(command.node_id)
        assert set(command.parameters["workload_cgroup_paths_by_node"]) == {
            command.node_id
        }, "a node does not receive the whole task's cgroup map"
        index = int(command.node_id.split("-")[1])
        document = passive_document(
            command.node_id, range(index * 8, index * 8 + 8), 10000
        )
        if communication:
            add_communication(document, 10000)
        result = NodeActionResult(
            command_id=command.command_id,
            operation=command.operation,
            status=NodeActionStatus.SUCCEEDED,
            details=document,
        )
        with lock:
            inflight -= 1
        return result

    adapter = NodeActionWorkflowAdapter(
        {node: "https://test.invalid" for node in nodes},
        SECRET,
        sender=sender,
        max_parallel_node_actions=16,
    )
    ctx = context(nodes)
    result = adapter.execute(ctx)
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert sorted(calls) == sorted(nodes), (
        "no total-node cap or duplicate successful read"
    )
    assert peak <= 16, peak
    analysis = survey_analysis(ctx.step.parameters, nodes, result.details)
    assert analysis["rank_inventory_complete"] is True, analysis
    assert analysis["valid_rank_count"] == analysis["expected_rank_count"] == 10000
    if communication:
        assert (
            analysis["gpu_observed_device_count"]
            == analysis["rdma_valid_port_count"]
            == 10000
        )
    assert analysis["communication_progress_rank_count"] == (
        10000 if communication else 0
    ), "FR disabled is missing evidence, not fabricated communication progress"


@pytest.mark.parametrize(
    "defect",
    [
        "node-missing",
        "rank-duplicate",
        "no-world",
        "bad-epoch",
        "pid-changed",
        "truncated",
    ],
)
def test_coverage_never_turns_missing_or_ambiguous_evidence_into_complete(defect):
    nodes = ["node-a", "node-b"]
    documents = {
        node: passive_document(node, [rank], 2) for rank, node in enumerate(nodes)
    }
    if defect == "node-missing":
        documents.pop("node-b")
    elif defect == "rank-duplicate":
        documents["node-b"]["ranks"][0]["rank"] = 0
    elif defect == "no-world":
        for document in documents.values():
            document["ranks"][0]["world_size"] = None
    elif defect == "bad-epoch":
        documents["node-b"]["snapshot_epoch"] = "old-attempt"
    elif defect == "pid-changed":
        documents["node-b"]["ranks"][0]["process_identity"]["verified"] = False
    else:
        documents["node-b"]["scan_complete"] = False
    report = survey_analysis(
        context(nodes).step.parameters, nodes, {"node_results": documents}
    )
    assert report["rank_inventory_complete"] is False, report


def test_group_comparison_requires_complete_membership_and_comparable_times():
    nodes = ["node-a", "node-b", "node-c"]
    stamp = datetime.now(timezone.utc)
    documents = {
        node: passive_document(node, [rank], 3, stamp=stamp)
        for rank, node in enumerate(nodes)
    }
    for index, document in enumerate(documents.values()):
        document["groups"] = {
            "dp": {
                "member_count": 3,
                "membership_sha256": membership_digest([0, 1, 2]),
                "name": "dp",
            }
        }
        sequence = 9 if index == 0 else 10
        document["ranks"][0]["communication"] = [
            {
                "group": "dp",
                "before": [sequence, sequence, sequence - 1],
                "after": [sequence, sequence, sequence - 1],
                "first_observed_at": stamp.isoformat(),
                "second_observed_at": (stamp + timedelta(seconds=2)).isoformat(),
                "clock_uncertainty_seconds": 0,
            }
        ]
    report = survey_analysis(
        context(nodes).step.parameters, nodes, {"node_results": documents}
    )
    assert report["candidate_ranks"] == [0], report
    late = deepcopy(documents)
    late["node-c"].update(
        sample_started_at=(stamp + timedelta(seconds=60)).isoformat(),
        sample_finished_at=(stamp + timedelta(seconds=61)).isoformat(),
    )
    late["node-c"]["ranks"][0]["communication"][0].update(
        first_observed_at=(stamp + timedelta(seconds=60)).isoformat(),
        second_observed_at=(stamp + timedelta(seconds=61)).isoformat(),
    )
    report = survey_analysis(
        context(nodes).step.parameters, nodes, {"node_results": late}
    )
    assert report["candidate_ranks"] == [], report
    assert report["cross_node_time_comparable"] is False


def test_unrelated_late_node_does_not_discard_a_time_aligned_small_group():
    stamp = datetime.now(timezone.utc)
    first = passive_document("node-a", [0, 1, 2], 6, stamp=stamp)
    first["groups"] = {
        "local": {
            "member_count": 3,
            "name": "local",
            "membership_sha256": membership_digest([0, 1, 2]),
        }
    }
    for row in first["ranks"]:
        sequence = 9 if row["rank"] == 0 else 10
        row["communication"] = [
            {
                "group": "local",
                "before": [sequence] * 3,
                "after": [sequence] * 3,
                "first_observed_at": stamp.isoformat(),
                "second_observed_at": (stamp + timedelta(seconds=2)).isoformat(),
            }
        ]
    second = passive_document(
        "node-b", [3, 4, 5], 6, stamp=stamp + timedelta(seconds=60)
    )
    report = survey_analysis(
        context(["node-a", "node-b"]).step.parameters,
        ["node-a", "node-b"],
        {"node_results": {"node-a": first, "node-b": second}},
    )
    assert report["cross_node_time_comparable"] is False
    assert report["candidate_ranks"] == [0], report
    assert report["process_groups"]["local"]["time_comparable"] is True


def test_unreachable_agent_does_not_hide_healthy_survey_nodes():
    nodes = ["node-a", "node-b"]

    def sender(endpoint, envelope):
        if envelope.command.node_id == "node-a":
            return WorkflowStepOutcome.failed("node unavailable")
        return NodeActionResult(
            command_id=envelope.command.command_id,
            operation=Op.COLLECT_HUNG_SUMMARY,
            status=NodeActionStatus.SUCCEEDED,
            details=passive_document("node-b", [1], 2),
        )

    adapter = NodeActionWorkflowAdapter(
        {node: "https://test.invalid" for node in nodes}, SECRET, sender=sender
    )
    result = adapter.execute(context(nodes))
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert set(result.details["node_results"]) == {"node-b"}
    assert set(result.details["node_failures"]) == {"node-a"}


def test_resume_keeps_completed_nodes_and_expires_only_the_timed_out_pending_node():
    ctx = context(["node-a", "node-b", "node-c"])
    before = datetime.now(timezone.utc) - timedelta(seconds=31)
    completed = passive_document("node-a", [0], 3)
    ctx = replace(
        ctx,
        workflow=ctx.workflow.model_copy(
            update={
                "step_executions": [
                    workflow_step_execution(
                        0,
                        Op.COLLECT_HUNG_SUMMARY,
                        WorkflowStepStatus.WAITING,
                        details={
                            "node_results": {"node-a": completed},
                            "pending_since": {"node-b": before.isoformat()},
                            "survey_started_at": before.isoformat(),
                            "retained_bytes": 1000,
                        },
                    )
                ]
            }
        ),
    )
    calls = []

    def sender(endpoint, envelope):
        node = envelope.command.node_id
        calls.append(node)
        if node == "node-b":
            return WorkflowStepOutcome.waiting(details={"node_action_state": "PENDING"})
        return NodeActionResult(
            command_id=envelope.command.command_id,
            operation=Op.COLLECT_HUNG_SUMMARY,
            status=NodeActionStatus.SUCCEEDED,
            details=passive_document(node, [2], 3),
        )

    adapter = NodeActionWorkflowAdapter(
        {node: "https://test.invalid" for node in ctx.step.node_ids},
        SECRET,
        sender=sender,
    )
    result = adapter.execute(ctx)
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert set(calls) == {"node-b", "node-c"}, calls
    assert result.details["node_failures"] == {"node-b": "NODE_TIMEOUT"}, result
    assert result.details["node_results"]["node-a"] == completed


@pytest.mark.parametrize("operation", [Op.COLLECT_HUNG_SUMMARY, Op.COLLECT_HUNG_TRIAGE])
def test_remote_pending_commands_occupy_the_window_until_timeout(
    operation, monkeypatch
):
    import gpu_fault.adapters.node_action.hung_survey as fanout

    ticks = iter([0, 6])
    monkeypatch.setattr(fanout, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    nodes = [f"node-{index}" for index in range(128)]
    ctx = context(nodes)
    ctx = replace(
        ctx,
        step=ctx.step.model_copy(
            update={
                "operation": operation,
                "parameters": {**ctx.step.parameters, "hung_wave": 2},
            }
        ),
    )
    calls = []

    def sender(endpoint, envelope):
        calls.append(envelope.command.node_id)
        return WorkflowStepOutcome.waiting(details={"node_action_state": "PENDING"})

    adapter = NodeActionWorkflowAdapter(
        {node: "https://test.invalid" for node in nodes},
        SECRET,
        sender=sender,
        max_parallel_node_actions=4,
    )
    result = adapter.execute(ctx)
    assert result.status is WorkflowStepStatus.WAITING, result
    assert len(calls) == 4 and set(calls) == set(nodes[:4]), calls
    assert set(result.details["pending_since"]) == set(nodes[:4]), result
    assert result.details["remaining_node_count"] == 128, result
