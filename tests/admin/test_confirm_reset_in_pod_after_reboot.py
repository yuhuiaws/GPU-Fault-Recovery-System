"""The in-Pod write accepts a reboot-settled RESET_GPU confirmation.

Live (gpu-eks-cluster, XID 46): the deploy-host plan judged the reboot
evidence (``settled_by: reboot_since_action``) and the Pod script -- shipped
with the CLI, running against the deployed image -- refused it for want of a
terminal Node Agent result, with the refusal blacked out. The script now
proves the same reboot the admin side proved (boot before and after) for any
node action settled by a reboot, and a refusal reaches the operator readably.
The incident's pointer sits on a failed restore attempt, as it did live; the
parked record is found among the incident's BLOCKED workflows.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.admin import submit_remediation as module
from gpu_fault.execution.node_action_uncertainty import has_unresolved_node_action
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.admin.test_admin_confirm_node_action import (
    NEW_BOOT,
    NODE,
    OLD_BOOT,
    _agent_record,
    _nodes,
    _pod,
)

OP = WorkflowOperation
NOW = datetime(2026, 10, 6, 1, 30, tzinfo=timezone.utc)
STARTED = NOW - timedelta(hours=2)
INCIDENT = "inc-3731-xid-46"
RESET_WORKFLOW = "workflow-ec95fa59"
RESTORE_ATTEMPT = "workflow-validated-restore-attempt"
OPERATOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"


def _reset_store():
    store = build_store()
    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=[NODE]),
        workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=[NODE]),
        workflow_step(OP.RESET_GPU, node_ids=[NODE], gpu_uuids=["GPU-3df3"]),
        workflow_step(OP.RESTORE_GPU_SERVICES, node_ids=[NODE]),
    ]
    store.save_workflow(
        workflow_request(
            RESET_WORKFLOW,
            INCIDENT,
            status=WorkflowStatus.BLOCKED,
            blocked_kind=BlockedKind.NEEDS_OPERATOR,
            fencing_token=1,
            execution_epoch=4,
            merge_revision=1,
            official_steps=steps,
            completed_step_indexes=[0, 1],
            completed_operations=[OP.MARK_UNSCHEDULABLE, OP.QUIESCE_GPU_SERVICES],
            step_executions=[
                workflow_step_execution(0, OP.MARK_UNSCHEDULABLE, phase="official"),
                workflow_step_execution(1, OP.QUIESCE_GPU_SERVICES, phase="official"),
                workflow_step_execution(
                    2,
                    OP.RESET_GPU,
                    WorkflowStepStatus.FAILED,
                    phase="official",
                    adapter_operation_id="remote/remote-reset",
                    error="OWNERSHIP_FINAL_CLIENTS_CHANGED",
                    details={
                        "outcome_unknown": True,
                        "manual_confirmation_required": True,
                        "remote_command_id": "remote-reset",
                    },
                    started_at=STARTED,
                    updated_at=STARTED + timedelta(minutes=3),
                ),
            ],
            blocked_reasons=["node action outcome unknown"],
            execution_deadline=NOW - timedelta(hours=1),
            created_at=STARTED,
            updated_at=STARTED + timedelta(minutes=5),
        )
    )
    store.save_workflow(
        workflow_request(
            RESTORE_ATTEMPT,
            INCIDENT,
            status=WorkflowStatus.FAILED,
            fencing_token=1,
            official_steps=[workflow_step(OP.VALIDATE_GPU, node_ids=[NODE])],
            created_at=STARTED + timedelta(hours=1),
            updated_at=NOW - timedelta(minutes=20),
        )
    )
    store.save_incident(
        fault_incident(
            INCIDENT,
            "kernel-log-kmsg-xid-46",
            cluster_id="gpu-a",
            node_ids=[NODE],
            gpu_uuids=["GPU-3df3"],
            state=IncidentState.ESCALATED,
            workflow_request_id=RESTORE_ATTEMPT,
            fencing_token=1,
            source_boot_id=OLD_BOOT,
            created_at=STARTED,
            updated_at=NOW - timedelta(minutes=20),
        )
    )
    _agent_record(store, boot_id=NEW_BOOT)
    return store


def _payload(run):
    inspection = run(
        {
            "mode": "inspect",
            "incident_id": INCIDENT,
            "action_incident_id": f"inc-operator-{INCIDENT}-confirm-node-action",
        }
    )
    plan = module.build_confirm_node_action_plan(
        inspection,
        incident_id=INCIDENT,
        node_id=NODE,
        operator=OPERATOR,
        reference="CHG-2026-1006-01",
        live_nodes=_nodes(),
        now=datetime.now(timezone.utc),
    )
    assert plan.workflow_request_id == RESET_WORKFLOW, (
        "the parked record is found behind the pointer at the failed restore"
    )
    (confirmation,) = plan.confirmations
    assert confirmation["settled_by"] == "reboot_since_action"
    assert confirmation["previous_boot_id"] == OLD_BOOT
    return module.confirmation_payload(
        plan, operator=OPERATOR, reference="CHG-2026-1006-01"
    )


def test_the_pod_writes_the_reboot_settled_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _reset_store()
    run = _pod(monkeypatch, capsys, store)

    result = run(_payload(run))

    assert result["no_op"] is False
    (written,) = result["confirmations"]
    assert written["settled_by"] == "reboot_since_action"
    saved = store.get_workflow(RESET_WORKFLOW)
    reset = next(
        item for item in saved.step_executions if item.operation is OP.RESET_GPU
    )
    assert reset.details["operator_confirmed"]["operation"] == "RESET_GPU"
    assert reset.details["outcome_unknown"] is False
    assert has_unresolved_node_action(saved) is False, (
        "the release probe sees no blocker"
    )
    assert saved.status is WorkflowStatus.BLOCKED, (
        "the record itself waits for the close"
    )


def test_the_pod_still_refuses_a_reboot_it_cannot_prove_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _reset_store()
    run = _pod(monkeypatch, capsys, store)
    payload = _payload(run)
    forged = json.loads(json.dumps(payload))
    forged["confirmations"][0]["confirmation"]["previous_boot_id"] = NEW_BOOT

    with pytest.raises(SystemExit, match="still runs boot") as refused:
        run(forged)

    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["error_kind"] == "refused"
    assert printed["error"] == str(refused.value), (
        "the same refusal travels on stdout, where the admin side can show it"
    )
    assert "recorded before RESET_GPU" in printed["error"]
    assert has_unresolved_node_action(store.get_workflow(RESET_WORKFLOW)), (
        "a refusal writes nothing"
    )


HEAL_INCIDENT = "inc-drain-after-workflow-82ab3511"
HEAL_WORKFLOW = "workflow-heal-541560c45428"


def _heal_store():
    """The same parked reset, no reboot since -- but the product's own drain
    incident on the node ran a heal that validated and uncordoned it after the
    refused reset (live, gpu-eks-cluster)."""

    store = _reset_store()
    _agent_record(store, boot_id=OLD_BOOT)
    validated_at = NOW - timedelta(minutes=10)
    operations = [
        OP.VALIDATE_GPU,
        OP.VALIDATE_HOST,
        OP.VALIDATE_FABRIC,
        OP.RESTORE_SCHEDULING,
    ]
    store.save_workflow(
        workflow_request(
            HEAL_WORKFLOW,
            HEAL_INCIDENT,
            status=WorkflowStatus.SUCCEEDED,
            fencing_token=2,
            official_steps=[
                workflow_step(item, node_ids=[NODE]) for item in operations
            ],
            completed_step_indexes=[0, 1, 2, 3],
            completed_operations=operations,
            step_executions=[
                workflow_step_execution(
                    index,
                    item,
                    phase="official",
                    started_at=validated_at - timedelta(minutes=1),
                    updated_at=validated_at,
                )
                for index, item in enumerate(operations)
            ],
            created_at=STARTED + timedelta(minutes=30),
            updated_at=validated_at,
        )
    )
    store.save_incident(
        fault_incident(
            HEAL_INCIDENT,
            "dcgm-fields-82ab3511",
            cluster_id="gpu-a",
            node_ids=[NODE],
            state=IncidentState.RECOVERED,
            workflow_request_id=HEAL_WORKFLOW,
            fencing_token=2,
            created_at=STARTED + timedelta(minutes=30),
            updated_at=validated_at,
        )
    )
    return store


def _old_boot_nodes() -> dict:
    nodes = _nodes()
    nodes[NODE]["status"]["nodeInfo"]["bootID"] = OLD_BOOT
    return nodes


def test_a_later_heal_of_another_incident_settles_the_reset_without_a_reboot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _heal_store()
    run = _pod(monkeypatch, capsys, store)

    inspection = run(
        {
            "mode": "inspect",
            "incident_id": INCIDENT,
            "action_incident_id": f"inc-operator-{INCIDENT}-confirm-node-action",
        }
    )
    assert [item["request_id"] for item in inspection["node_validations"]] == [
        HEAL_WORKFLOW
    ], "the Pod lists the other incident's validated restore of the node"

    plan = module.build_confirm_node_action_plan(
        inspection,
        incident_id=INCIDENT,
        node_id=NODE,
        operator=OPERATOR,
        reference="CHG-2026-1006-02",
        live_nodes=_old_boot_nodes(),
        now=datetime.now(timezone.utc),
    )
    (confirmation,) = plan.confirmations
    assert confirmation["settled_by"] == "validated_restore"
    assert confirmation["validated_by_workflow"]["request_id"] == HEAL_WORKFLOW
    assert confirmation["validated_by_workflow"]["incident_id"] == HEAL_INCIDENT
    assert confirmation["validated_by_workflow"]["cross_incident"] is True
    assert confirmation["observed_boot_id"] == OLD_BOOT, "no reboot was needed"

    result = run(
        module.confirmation_payload(
            plan, operator=OPERATOR, reference="CHG-2026-1006-02"
        )
    )

    assert result["no_op"] is False
    saved = store.get_workflow(RESET_WORKFLOW)
    reset = next(
        item for item in saved.step_executions if item.operation is OP.RESET_GPU
    )
    assert reset.details["operator_confirmed"]["validated_by_workflow"][
        "incident_id"
    ] == (HEAL_INCIDENT)
    assert has_unresolved_node_action(saved) is False


def test_a_validation_of_another_cluster_or_before_the_action_is_not_listed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _heal_store()
    elsewhere = store.get_incident(HEAL_INCIDENT).model_copy(
        update={"cluster_id": "gpu-b"}
    )
    store.save_incident(elsewhere, expected=store.get_incident(HEAL_INCIDENT))
    run = _pod(monkeypatch, capsys, store)

    inspection = run(
        {
            "mode": "inspect",
            "incident_id": INCIDENT,
            "action_incident_id": f"inc-operator-{INCIDENT}-confirm-node-action",
        }
    )

    assert inspection["node_validations"] == [], (
        "another cluster's node is not this one"
    )
    with pytest.raises(Exception, match="no later validated restore of this or any"):
        module.build_confirm_node_action_plan(
            inspection,
            incident_id=INCIDENT,
            node_id=NODE,
            operator=OPERATOR,
            reference="CHG-2026-1006-02",
            live_nodes=_old_boot_nodes(),
            now=datetime.now(timezone.utc),
        )
