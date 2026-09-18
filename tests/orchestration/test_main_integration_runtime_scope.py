from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStepStatus
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.orchestration.validated_restore import (
    VALIDATED_RESTORE_OPERATIONS,
    build_validated_restore_workflow,
)
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

GPU = WorkflowOperation.VALIDATE_GPU
FABRIC = WorkflowOperation.VALIDATE_FABRIC
REBOOT = WorkflowOperation.RESTART_NODE


def _incident():
    return fault_incident(
        "inc-runtime-scope",
        "event-runtime-scope",
        node_ids=["node-a", "node-b"],
        gpu_uuids=["GPU-a", "GPU-b"],
        state=IncidentState.QUARANTINED,
        fencing_token=7,
    )


def _restore(incident, inventory):
    return build_validated_restore_workflow(
        incident,
        operator="test-operator",
        reference="CHG-runtime-scope",
        now=datetime.now(timezone.utc),
        node_gpu_uuids=inventory,
    )


def test_complete_inventory_binds_each_restore_validation_to_its_own_node():
    incident = _incident()
    inventory = {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}
    before = deepcopy((incident, inventory))

    updated, workflow = _restore(incident, inventory)

    assert (incident, inventory) == before, "scope binding must not mutate evidence"
    assert updated.gpu_uuids == incident.gpu_uuids
    assert workflow.fencing_token == incident.fencing_token
    assert [step.operation for step in workflow.official_steps] == [
        GPU,
        GPU,
        WorkflowOperation.VALIDATE_HOST,
        FABRIC,
        FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
    ], "every hardware validation must finish before any scheduling restore"
    for operation in (GPU, FABRIC):
        assert {
            tuple(step.node_ids): step.gpu_uuids
            for step in workflow.official_steps
            if step.operation is operation
        } == {("node-a",): ["GPU-a"], ("node-b",): ["GPU-b"]}
    assert workflow.official_steps[-1].node_ids == ["node-a", "node-b"]


@pytest.mark.parametrize(
    "inventory",
    [
        None,
        {},
        {"node-a": ["GPU-a"]},
        {"node-a": [], "node-b": ["GPU-b"]},
        {"node-a": ["GPU-a"], "node-b": ["GPU-new"]},
        {"node-a": ["GPU-a", "GPU-b"], "node-b": ["GPU-b"]},
        {"node-a": "GPU-a", "node-b": ["GPU-b"]},
        {"node-a": ["GPU-a"], "node-b": [None, "GPU-b"]},
        {"node-a": ["GPU-a"], "node-b": ["", "GPU-b"]},
        {"node-a": ["GPU-a"], "unrelated-node": ["GPU-b"]},
    ],
    ids=[
        "omitted",
        "empty",
        "missing-node",
        "empty-node",
        "missing-gpu",
        "ambiguous-owner",
        "scalar",
        "invalid-uuid",
        "empty-uuid",
        "outside-target",
    ],
)
def test_incomplete_or_ambiguous_inventory_never_shrinks_the_restore_scope(inventory):
    incident = _incident()

    updated, workflow = _restore(incident, inventory)

    assert [step.operation for step in workflow.official_steps] == list(
        VALIDATED_RESTORE_OPERATIONS
    )
    assert all(
        step.node_ids == incident.node_ids and step.gpu_uuids == incident.gpu_uuids
        for step in workflow.official_steps
    ), "unproven ownership must retain the original validation scope"
    assert updated.gpu_uuids == incident.gpu_uuids


def test_unrelated_inventory_cannot_remove_a_single_node_incident_gpu():
    incident = _incident().model_copy(update={"node_ids": ["node-a"]})

    _, workflow = _restore(incident, {"node-a": ["GPU-a"], "node-b": ["GPU-b"]})

    assert all(
        step.node_ids == ["node-a"] and step.gpu_uuids == ["GPU-a", "GPU-b"]
        for step in workflow.official_steps
    ), "an extra mapping key cannot authorize removing an incident GPU"


def _scope(steps, *, failed_nodes=None, incident=None):
    source = incident or _incident()
    execution = workflow_step_execution(
        0,
        steps[0].operation,
        WorkflowStepStatus.FAILED,
        details={} if failed_nodes is None else {"failed_nodes": failed_nodes},
    )
    workflow = workflow_request(
        "workflow-runtime-scope",
        source.incident_id,
        official_steps=steps,
        step_executions=[execution],
    )
    return HardwareEscalationService.collect_scope(workflow, source, [execution])


def test_multi_node_steps_are_not_guessed_to_be_single_node_gpu_evidence():
    scope = _scope(
        [
            workflow_step(
                REBOOT, node_ids=["node-a", "node-b"], gpu_uuids=["GPU-a", "GPU-b"]
            )
        ],
        failed_nodes=["node-a"],
    )

    assert scope["ordered_failed_nodes"] == ["node-a"]
    assert scope["gpu_uuids"] == ["GPU-a", "GPU-b"]
    assert scope["gpu_uuids_by_node"] == {}


def test_explicit_gpu_mapping_takes_precedence_over_derived_step_scope():
    scope = _scope(
        [
            workflow_step(
                REBOOT,
                node_ids=["node-a", "node-b"],
                gpu_uuids=["GPU-a", "GPU-b"],
                parameters={
                    "gpu_uuids_by_node": {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}
                },
            ),
            workflow_step(GPU, node_ids=["node-a"], gpu_uuids=["GPU-a", "GPU-b"]),
        ],
        failed_nodes=["node-a"],
    )

    assert scope["gpu_uuids"] == ["GPU-a"]
    assert scope["gpu_uuids_by_node"] == {"node-a": ["GPU-a"]}


def test_unassigned_gpu_scope_is_retained_instead_of_assumed_to_be_a_sibling():
    source = _incident().model_copy(
        update={"gpu_uuids": ["GPU-a", "GPU-b", "GPU-missing"]}
    )
    scope = _scope(
        [
            workflow_step(REBOOT, node_ids=["node-a"], gpu_uuids=["GPU-a"]),
            workflow_step(REBOOT, node_ids=["node-b"], gpu_uuids=["GPU-b"]),
        ],
        incident=source,
    )

    assert scope["ordered_failed_nodes"] == ["node-a"]
    assert scope["gpu_uuids"] == ["GPU-a", "GPU-b", "GPU-missing"]
    assert scope["gpu_uuids_by_node"] == {}


def test_partially_mapped_failed_nodes_do_not_lose_the_unmapped_nodes_gpus():
    scope = _scope(
        [
            workflow_step(
                REBOOT,
                node_ids=["node-a", "node-b"],
                gpu_uuids=["GPU-a", "GPU-b"],
                parameters={"gpu_uuids_by_node": {"node-a": ["GPU-a"]}},
            )
        ]
    )

    assert scope["ordered_failed_nodes"] == ["node-a", "node-b"]
    assert scope["gpu_uuids"] == ["GPU-a", "GPU-b"]
    assert scope["gpu_uuids_by_node"] == {}


def test_a_mapping_cannot_assign_gpus_to_a_node_outside_its_step():
    scope = _scope(
        [
            workflow_step(
                REBOOT,
                node_ids=["node-a"],
                gpu_uuids=["GPU-a"],
                parameters={"gpu_uuids_by_node": {"node-b": ["GPU-a"]}},
            ),
            workflow_step(REBOOT, node_ids=["node-b"], gpu_uuids=["GPU-b"]),
        ]
    )

    assert scope["gpu_uuids"] == ["GPU-a"]
    assert scope["gpu_uuids_by_node"] == {"node-a": ["GPU-a"]}
