from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.adapters.gpu_validation import GpuValidationAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStepStatus
from gpu_fault.orchestration.validated_restore import build_validated_restore_workflow
from tests._builders import fault_incident
from tests.execution._cov95_runtime_validation import ValidationFeed


def _gpu_outcomes(nodes, incident_gpus, inventory, sampled):
    now = datetime.now(timezone.utc)
    incident = fault_incident(
        "inc-runtime-restore",
        "event-runtime-restore",
        node_ids=nodes,
        gpu_uuids=incident_gpus,
        state=IncidentState.QUARANTINED,
        fencing_token=7,
    )
    _, workflow = build_validated_restore_workflow(
        incident,
        operator="test-operator",
        reference="CHG-runtime-restore",
        now=now,
        node_gpu_uuids=inventory,
    )
    feed = ValidationFeed(nodes, now)
    for node, gpus in sampled.items():
        feed.gpu[node] = [
            SimpleNamespace(
                observed_at=now,
                sample=SimpleNamespace(
                    canonical_name="gpu_temperature_c", gpu_uuid=gpu
                ),
            )
            for gpu in gpus
        ]
    adapter = GpuValidationAdapter(feed)
    outcomes = []
    for index, step in enumerate(workflow.official_steps):
        if step.operation is WorkflowOperation.VALIDATE_GPU:
            outcomes.append(
                adapter.execute(
                    WorkflowStepContext(
                        incident=incident,
                        workflow=workflow,
                        step=step,
                        step_index=index,
                        request=WorkflowExecutionRequest(expected_fencing_token=7),
                        idempotency_key=f"restore-validation/{index}",
                    )
                )
            )
    return outcomes


@pytest.mark.parametrize(
    ("inventory", "sampled", "missing"),
    [
        ({"node-a": ["GPU-a"]}, {"node-a": ["GPU-a"]}, ["GPU-b"]),
        ({"node-a": ["GPU-new"]}, {"node-a": ["GPU-new"]}, ["GPU-a", "GPU-b"]),
        ({}, {"node-a": ["GPU-a"]}, ["GPU-b"]),
    ],
    ids=["partial-dropout", "all-original-gpus-absent", "unknown-inventory"],
)
def test_disappeared_gpus_cannot_be_validated_by_the_remaining_healthy_gpu(
    inventory, sampled, missing
):
    outcomes = _gpu_outcomes(["node-a"], ["GPU-a", "GPU-b"], inventory, sampled)

    assert len(outcomes) == 1
    assert outcomes[0].status is WorkflowStepStatus.WAITING
    assert outcomes[0].details["node_pending"]["node-a"][
        "missing_recent_metrics_by_gpu"
    ] == {gpu: ["gpu_temperature_c"] for gpu in missing}


def test_complete_multi_node_restore_does_not_wait_for_a_siblings_gpu():
    inventory = {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}

    outcomes = _gpu_outcomes(
        ["node-a", "node-b"], ["GPU-a", "GPU-b"], inventory, inventory
    )

    assert len(outcomes) == 2
    assert all(
        outcome.status is WorkflowStepStatus.SUCCEEDED for outcome in outcomes
    ), outcomes


def test_inventory_presence_does_not_replace_fresh_telemetry_for_each_gpu():
    inventory = {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}

    outcomes = _gpu_outcomes(
        ["node-a", "node-b"],
        ["GPU-a", "GPU-b"],
        inventory,
        {"node-a": ["GPU-a"], "node-b": ["GPU-unrelated"]},
    )

    assert [outcome.status for outcome in outcomes] == [
        WorkflowStepStatus.SUCCEEDED,
        WorkflowStepStatus.WAITING,
    ]
    assert outcomes[1].details["node_pending"]["node-b"][
        "missing_recent_metrics_by_gpu"
    ] == {"GPU-b": ["gpu_temperature_c"]}


def test_stale_inventory_assignment_cannot_validate_the_wrong_node():
    outcomes = _gpu_outcomes(
        ["node-a", "node-b"],
        ["GPU-a", "GPU-b"],
        {"node-a": ["GPU-a"], "node-b": ["GPU-b"]},
        {"node-a": ["GPU-b"], "node-b": ["GPU-a"]},
    )

    assert len(outcomes) == 2
    assert all(outcome.status is WorkflowStepStatus.WAITING for outcome in outcomes), (
        outcomes
    )
