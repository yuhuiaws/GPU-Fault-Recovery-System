"""The validated restore an operator (or the acceptance fixture) requests for a
QUARANTINED incident is built in one place, ``validated_restore``, so the
product verb and the fixture yield the same workflow: four steps under the same
incident and fencing token, the incident moved to ACTION_PENDING and pointed at
the workflow."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration import validated_restore as module
from tests._builders import fault_incident

NOW = datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)
OPERATOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"


def _quarantined(**values):
    defaults = dict(
        node_ids=["node-a"],
        gpu_uuids=["GPU-1", "GPU-2"],
        state=IncidentState.QUARANTINED,
        workflow_request_id="wf-quarantine",
        fencing_token=5,
        reasons=["quarantined after repeated resets"],
    )
    defaults.update(values)
    return fault_incident("inc-q", "event-q", **defaults)


def test_the_workflow_runs_the_four_steps_under_the_incident_and_its_token() -> None:
    incident = _quarantined()

    updated, workflow = module.build_validated_restore_workflow(
        incident, operator=OPERATOR, reference="CHG-7", now=NOW
    )

    assert workflow.request_id.startswith(module.RESTORE_WORKFLOW_PREFIX), (
        workflow.request_id
    )
    assert module.is_validated_restore_workflow(workflow.request_id) is True
    assert workflow.incident_id == "inc-q"
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.official_action == "RESTORE_SCHEDULING"
    assert workflow.fencing_token == 5, "the token the node's annotation carries"
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]
    assert [step.execution_owner for step in workflow.official_steps] == [
        module.VALIDATION_OWNER,
        module.VALIDATION_OWNER,
        module.VALIDATION_OWNER,
        module.KUBERNETES_OWNER,
    ]
    assert all(step.node_ids == ["node-a"] for step in workflow.official_steps), (
        "every step targets the incident's node"
    )
    assert all(
        step.gpu_uuids == ["GPU-1", "GPU-2"] for step in workflow.official_steps
    ), "the incident's GPU scope goes on the steps when they cover its nodes"
    assert workflow.created_at == workflow.updated_at == NOW
    assert workflow.runtime_profile_version is None


def test_the_incident_moves_to_action_pending_and_records_the_operator() -> None:
    incident = _quarantined()

    updated, workflow = module.build_validated_restore_workflow(
        incident, operator=OPERATOR, reference="CHG-7", now=NOW
    )

    assert updated.state is IncidentState.ACTION_PENDING
    assert updated.workflow_request_id == workflow.request_id
    assert updated.fencing_token == 5
    assert updated.node_ids == ["node-a"]
    assert updated.reasons == [
        "quarantined after repeated resets",
        f"operator restore requested by {OPERATOR} (CHG-7)",
    ]
    assert updated.updated_at == NOW
    assert incident.state is IncidentState.QUARANTINED, "the input is not mutated"


def test_without_a_reference_the_reason_says_so() -> None:
    assert module.restore_reason(OPERATOR, None) == (
        f"operator restore requested by {OPERATOR} (no reference)"
    )


def test_the_fixture_shape_restores_another_node_without_the_incident_gpus() -> None:
    """The warm-spare fixture restores a spare under the fault incident: the
    spare joins the incident's nodes, the steps target the spare only, and the
    fault node's GPU UUIDs must not be pinned on it (DESTR-003)."""

    incident = _quarantined()

    updated, workflow = module.build_validated_restore_workflow(
        incident,
        operator="acceptance-fixture",
        reference=None,
        now=NOW,
        node_ids=["node-spare"],
        runtime_profile_version="hyperpod-v1",
        reason="acceptance restore",
    )

    assert all(step.node_ids == ["node-spare"] for step in workflow.official_steps), (
        "the steps target the spare only"
    )
    assert all(step.gpu_uuids == [] for step in workflow.official_steps), (
        "the fault node's GPUs are not pinned on the spare"
    )
    assert workflow.runtime_profile_version == "hyperpod-v1"
    assert updated.node_ids == ["node-a", "node-spare"]
    assert updated.reasons[-1] == "acceptance restore"


def test_an_incident_naming_no_nodes_cannot_be_restored() -> None:
    with pytest.raises(ValueError, match="names no nodes"):
        module.build_validated_restore_workflow(
            _quarantined(node_ids=[]), operator=OPERATOR, reference=None, now=NOW
        )


def test_the_fixture_script_uses_the_product_builder() -> None:
    from scripts.e2e.regional import warm_spare_fixture

    script = warm_spare_fixture.CREATE_RESTORE_WORKFLOW
    assert "build_validated_restore_workflow" in script
    assert "WorkflowStepSpec(" not in script, "no hand-built steps left in the fixture"
    assert "store.save_incident_and_workflow(incident, workflow)" in script
    assert "context.dispatcher.wake()" in script
    for key in ('"workflow_request_id"', '"incident_id"', '"node_id"'):
        assert key in script, f"the printed contract keeps {key}"
    compile(script, "<create-restore-workflow>", "exec")
