"""The orphaned-commands reconcile: commands a terminal workflow left open are
cancelled; anything a live workflow still owns is not."""

from __future__ import annotations

import pytest

from gpu_fault.admin.orphaned_commands import (
    APPLY_MODE,
    PLAN_MODE,
    apply_orphaned_commands_plan,
    build_orphaned_commands_plan,
)
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests._builders import build_store, fault_incident, workflow_request, workflow_step

WORKFLOW = "workflow-f48baa91-63e4-432b-9648-1469d9e7eb39"
INCIDENT = "inc-kernel-log-kmsg-xid-54"
REFERENCE = "pre-deploy-step-cap-orphan"


def _orphan(
    store,
    *,
    workflow_status=WorkflowStatus.FAILED,
    command_status=RemoteCommandStatus.WAITING,
):
    incident = fault_incident(
        INCIDENT, "event-xid54", node_ids=["node-a"], state=IncidentState.ESCALATED
    )
    steps = [
        workflow_step(WorkflowOperation.FREEZE_EVIDENCE),
        workflow_step(WorkflowOperation.CHECK_MECHANICALS),
    ]
    workflow = workflow_request(
        WORKFLOW,
        INCIDENT,
        status=workflow_status,
        fencing_token=1,
        official_action="CHECK_MECHANICALS",
        official_steps=steps,
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    store.ensure_remote_command(
        RemoteActionCommand(
            command_id="remote-2cbdab99",
            cluster_id="cluster-a",
            workflow_request_id=WORKFLOW,
            incident_id=INCIDENT,
            step_index=1,
            fencing_token=1,
            idempotency_key=f"{WORKFLOW}/1/CHECK_MECHANICALS",
            step=steps[1],
            workflow=workflow,
            incident=incident,
            status=command_status,
        )
    )


def _command_status(store) -> str:
    (command,) = store.list_remote_commands(workflow_request_ids=[WORKFLOW])
    return command.status.value


def test_an_open_command_of_a_failed_workflow_is_cancelled() -> None:
    store = build_store()
    _orphan(store)

    plan = build_orphaned_commands_plan(store, [WORKFLOW])
    (item,) = plan["items"]
    assert plan["mode"] == PLAN_MODE, plan
    assert item["eligible"] is True, item["reasons"]
    assert item["open_commands"][0]["operation"] == "CHECK_MECHANICALS", item

    result = apply_orphaned_commands_plan(
        store,
        workflow_ids=[WORKFLOW],
        expected_plan_sha256=plan["plan_sha256"],
        reference=REFERENCE,
    )

    assert result["mode"] == APPLY_MODE, result
    assert result["applied_workflow_ids"] == [WORKFLOW], result
    assert result["failed_workflow_ids"] == [], result
    assert result["records_deleted"] == 0, result
    assert _command_status(store) not in {"PENDING", "LEASED", "WAITING"}, (
        "the orphan is still open after the apply"
    )
    # The workflow itself is not touched.
    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.FAILED, (
        "the terminal workflow must be left as it was"
    )


@pytest.mark.parametrize(
    "workflow_status",
    [WorkflowStatus.RUNNING, WorkflowStatus.PENDING, WorkflowStatus.SAFETY_PENDING],
)
def test_a_live_workflow_keeps_its_commands(workflow_status: WorkflowStatus) -> None:
    store = build_store()
    _orphan(store, workflow_status=workflow_status)

    plan = build_orphaned_commands_plan(store, [WORKFLOW])
    (item,) = plan["items"]
    assert item["eligible"] is False, item
    assert any("live work" in reason for reason in item["reasons"]), item["reasons"]
    with pytest.raises(ValueError, match="ineligible records"):
        apply_orphaned_commands_plan(
            store,
            workflow_ids=[WORKFLOW],
            expected_plan_sha256=plan["plan_sha256"],
            reference=REFERENCE,
        )
    assert _command_status(store) == "WAITING", (
        "a live workflow's command was cancelled"
    )


def test_a_workflow_without_open_commands_is_refused_not_no_opped() -> None:
    store = build_store()
    _orphan(store, command_status=RemoteCommandStatus.SUCCEEDED)
    (item,) = build_orphaned_commands_plan(store, [WORKFLOW])["items"]
    assert item["eligible"] is False, item
    assert "workflow has no open remote commands" in item["reasons"], item["reasons"]


def test_apply_refuses_a_plan_that_changed_under_it() -> None:
    store = build_store()
    _orphan(store)
    plan = build_orphaned_commands_plan(store, [WORKFLOW])
    # The command settled between review and apply: nothing left to approve.
    store.cancel_remote_commands_for_workflow(WORKFLOW, reason="settled elsewhere")
    with pytest.raises(ValueError, match="changed before apply"):
        apply_orphaned_commands_plan(
            store,
            workflow_ids=[WORKFLOW],
            expected_plan_sha256=plan["plan_sha256"],
            reference=REFERENCE,
        )


def test_plan_requires_explicit_workflow_ids_and_reports_unknown_ones() -> None:
    store = build_store()
    with pytest.raises(ValueError, match="explicit workflow IDs"):
        build_orphaned_commands_plan(store, [])
    (item,) = build_orphaned_commands_plan(store, ["workflow-missing"])["items"]
    assert item["eligible"] is False, item
    assert item["reasons"] == ["workflow does not exist"], item


def test_a_lease_flap_between_review_and_apply_does_not_change_the_digest() -> None:
    """The executor keeps re-leasing an orphaned CHECK_MECHANICALS to poll for
    an acknowledgement, so the command flips WAITING <-> LEASED under the
    operator; the approval binds which command is cancelled, not its phase."""

    store = build_store()
    _orphan(store)
    reviewed = build_orphaned_commands_plan(store, [WORKFLOW])
    # A WAITING command is re-claimable; the executor's poll is exactly this.
    claimed = store.claim_remote_commands(
        "cluster-a", "executor-a", limit=1, lease_seconds=60
    )
    assert [item.command_id for item in claimed] == ["remote-2cbdab99"], claimed
    flapped = build_orphaned_commands_plan(store, [WORKFLOW])
    assert flapped["items"][0]["open_commands"][0]["status"] == "LEASED", flapped
    assert flapped["plan_sha256"] == reviewed["plan_sha256"], (
        "a lease flap must not invalidate the review"
    )

    result = apply_orphaned_commands_plan(
        store,
        workflow_ids=[WORKFLOW],
        expected_plan_sha256=reviewed["plan_sha256"],
        reference=REFERENCE,
    )
    assert result["applied_workflow_ids"] == [WORKFLOW], result
