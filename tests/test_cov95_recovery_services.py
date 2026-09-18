from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault import planner
from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    CompletionDecision,
    DecisionStatus,
    MarkerScope,
    RecoveryAction,
    TerminalStatus,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.planner import PlanBuilder, UnsupportedPlanError
from gpu_fault.service import CompletionService
from gpu_fault.store import InMemoryStore, NotFoundError, SqliteStore
from tests._builders import workflow_step
from tests._cov95_recovery_services import (
    active_recovery,
    exercise_failed_withdrawal,
    failure_for,
    marker_for,
    profile_without,
)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("NO_ACTION", ["RESTART_WORKLOAD"]),
        ("RESTART_WORKLOAD", ["RESTART_WORKLOAD"]),
        ("MARK_UNSCHEDULABLE", ["MARK_UNSCHEDULABLE"]),
        ("STOP_WORKLOAD", ["STOP_WORKLOAD"]),
        ("COLLECT_EVIDENCE", ["COLLECT_EVIDENCE"]),
        ("VALIDATE_NODE", ["VALIDATE_NODE"]),
        ("RESTORE_SCHEDULING", ["RESTORE_SCHEDULING"]),
        ("QUARANTINE", ["MARK_UNSCHEDULABLE", "QUARANTINE"]),
        (
            "ESCALATE_OPERATOR",
            ["MARK_UNSCHEDULABLE", "QUARANTINE", "ESCALATE_OPERATOR"],
        ),
    ],
)
def test_marker_plans_preserve_the_selected_action_and_explicit_scope(
    failed_event, action, expected
) -> None:
    selected = RecoveryAction(action)
    value = marker_for(failed_event, selected)
    plan = PlanBuilder().from_marker(failed_event, value, profile_without())
    assert [step.action.value for step in plan.steps] == expected, (
        "a marker's declared action must not silently become a different recovery"
    )
    assert all(step.node_ids == ["node-a"] for step in plan.steps), (
        "hardware scope comes from the marker, not the wider allocation"
    )
    assert plan.checkpoint_manifest_ref == failed_event.checkpoint_manifest_ref, (
        "checkpoint identity must survive planning"
    )


@pytest.mark.parametrize("workload", [False, True])
@pytest.mark.parametrize("can_stop", [False, True])
def test_drain_stops_only_known_workloads_with_an_executable_owner(
    failed_event, workload, can_stop
) -> None:
    event = failed_event.model_copy(
        update={"workload_ids": ["training/job/unit"] if workload else []}
    )
    profile = profile_without(*([] if can_stop else [CapabilityName.WORKLOAD_STOP]))
    plan = PlanBuilder().from_marker(
        event, marker_for(event, RecoveryAction.DRAIN), profile
    )
    expected = [RecoveryAction.MARK_UNSCHEDULABLE]
    if workload and can_stop:
        expected.append(RecoveryAction.STOP_WORKLOAD)
    expected.append(RecoveryAction.QUARANTINE)
    assert [step.action for step in plan.steps] == expected, (
        "missing optional STOP must not remove the drain containment"
    )


@pytest.mark.parametrize(
    "action",
    [
        RecoveryAction.RESET_GPU,
        RecoveryAction.REBOOT_NODE,
        RecoveryAction.REPLACE_NODE,
        RecoveryAction.REMEDIATE_EFA_DRIVER,
        RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
        RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
    ],
)
def test_hardware_plan_orders_containment_before_action_and_restore_before_restart(
    failed_event, action
) -> None:
    event = failed_event.model_copy(update={"workload_ids": ["training/job/unit"]})
    plan = PlanBuilder().from_marker(
        event, marker_for(event, action), profile_without()
    )
    assert [step.action for step in plan.steps] == [
        RecoveryAction.MARK_UNSCHEDULABLE,
        RecoveryAction.COLLECT_EVIDENCE,
        RecoveryAction.STOP_WORKLOAD,
        action,
        RecoveryAction.VALIDATE_NODE,
        RecoveryAction.RESTORE_SCHEDULING,
        RecoveryAction.RESTART_WORKLOAD,
    ], "a destructive plan must retain the stop/action/restore/restart order"
    assert plan.steps[3].gpu_uuids == ["GPU-a"], (
        "duplicate GPU identities cannot multiply the planned action"
    )


@pytest.mark.parametrize(
    ("absent", "expected"),
    [
        (
            [CapabilityName.SCHEDULER_DRAIN],
            [
                RecoveryAction.COLLECT_EVIDENCE,
                RecoveryAction.REBOOT_NODE,
                RecoveryAction.VALIDATE_NODE,
            ],
        ),
        (
            [CapabilityName.DEEP_DIAGNOSTICS],
            [
                RecoveryAction.MARK_UNSCHEDULABLE,
                RecoveryAction.COLLECT_EVIDENCE,
                RecoveryAction.REBOOT_NODE,
                RecoveryAction.RESTORE_SCHEDULING,
            ],
        ),
        (
            [CapabilityName.EVIDENCE_CAPTURE],
            [
                RecoveryAction.MARK_UNSCHEDULABLE,
                RecoveryAction.REBOOT_NODE,
                RecoveryAction.VALIDATE_NODE,
                RecoveryAction.RESTORE_SCHEDULING,
            ],
        ),
    ],
)
def test_unavailable_auxiliary_capabilities_do_not_invent_an_execution_owner(
    failed_event, absent, expected
) -> None:
    plan = PlanBuilder().from_marker(
        failed_event,
        marker_for(failed_event, RecoveryAction.REBOOT_NODE),
        profile_without(*absent),
    )
    assert [step.action for step in plan.steps] == expected, (
        "only available auxiliary steps may enter the plan"
    )


def test_destructive_plan_without_stop_uses_only_containment_and_support(
    failed_event,
) -> None:
    event = failed_event.model_copy(update={"workload_ids": ["training/job/unit"]})
    plan = PlanBuilder().from_marker(
        event,
        marker_for(event, RecoveryAction.RESET_GPU),
        profile_without(CapabilityName.WORKLOAD_STOP, CapabilityName.SCHEDULER_DRAIN),
    )
    assert [step.action for step in plan.steps] == [
        RecoveryAction.COLLECT_EVIDENCE,
        RecoveryAction.ESCALATE_OPERATOR,
    ], "a live workload with no STOP owner must never reach RESET"
    assert plan.steps[-1].parameters["blocked_action"] == "RESET_GPU", (
        "support must retain the action that was refused"
    )


def test_absent_primary_and_fallback_owners_refuse_instead_of_an_empty_plan(
    failed_event,
) -> None:
    with pytest.raises(UnsupportedPlanError, match="no containment/support"):
        PlanBuilder().from_marker(
            failed_event,
            marker_for(failed_event, RecoveryAction.REBOOT_NODE),
            profile_without(
                CapabilityName.NODE_REBOOT,
                CapabilityName.SCHEDULER_DRAIN,
                CapabilityName.SUPPORT_ESCALATION,
            ),
        )


def test_diagnostic_plan_can_omit_evidence_but_not_the_diagnostic_owner(
    failed_event,
) -> None:
    marker = marker_for(failed_event, RecoveryAction.RUN_DIAGNOSTICS)
    plan = PlanBuilder().from_marker(
        failed_event, marker, profile_without(CapabilityName.EVIDENCE_CAPTURE)
    )
    assert [step.action for step in plan.steps] == [RecoveryAction.RUN_DIAGNOSTICS], (
        "the diagnostic action remains available without auxiliary capture"
    )
    with pytest.raises(UnsupportedPlanError, match="no executable owner"):
        PlanBuilder().from_marker(
            failed_event, marker, profile_without(CapabilityName.DEEP_DIAGNOSTICS)
        )


def test_operator_escalation_does_not_require_an_auxiliary_scheduler_owner(
    failed_event,
) -> None:
    plan = PlanBuilder().from_marker(
        failed_event,
        marker_for(failed_event, RecoveryAction.ESCALATE_OPERATOR),
        profile_without(CapabilityName.SCHEDULER_DRAIN),
    )
    assert [step.action for step in plan.steps] == [RecoveryAction.ESCALATE_OPERATOR], (
        "lack of a containment capability must not prevent asking an operator"
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("conflict", ["disabled", "ambiguous"])
def test_public_plan_entrypoint_rejects_incoherent_capability_claims(
    failed_event, conflict, reverse
) -> None:
    profile = profile_without()
    claim = next(
        item
        for item in profile.capabilities
        if item.capability is CapabilityName.DEEP_DIAGNOSTICS
    )
    peer = claim.model_copy(
        update={
            "mode": (
                CapabilityMode.DISABLED
                if conflict == "disabled"
                else CapabilityMode.DELEGATE
            ),
            "owner": "other-owner",
        }
    )
    claims = [*profile.capabilities, peer]
    if reverse:
        claims.reverse()
    with pytest.raises(UnsupportedPlanError, match=conflict):
        PlanBuilder().from_marker(
            failed_event,
            marker_for(failed_event, RecoveryAction.RUN_DIAGNOSTICS),
            profile.model_copy(update={"capabilities": claims}),
        )


def test_missing_action_registration_is_a_named_public_refusal(
    failed_event, monkeypatch
) -> None:
    monkeypatch.delitem(planner.ACTION_CAPABILITY, RecoveryAction.RUN_DIAGNOSTICS)
    with pytest.raises(UnsupportedPlanError, match="no capability mapping"):
        PlanBuilder().from_marker(
            failed_event,
            marker_for(failed_event, RecoveryAction.RUN_DIAGNOSTICS),
            profile_without(),
        )


def test_workload_only_marker_uses_allocation_when_no_node_scope_is_claimed(
    failed_event,
) -> None:
    marker = marker_for(
        failed_event, RecoveryAction.RESTART_WORKLOAD, scope=MarkerScope()
    )
    plan = PlanBuilder().from_marker(failed_event, marker, profile_without())
    assert plan.steps[0].node_ids == ["node-a", "node-b"], (
        "a workload restart may use its captured allocation"
    )
    assert plan.avoid_node_ids == [], "workload-only recovery does not condemn nodes"


@pytest.mark.parametrize("seconds", [0, -1])
def test_nonpositive_marker_correlation_window_is_refused(seconds) -> None:
    with pytest.raises(ValueError, match="marker_window must be positive"):
        CompletionService(InMemoryStore(), marker_window=timedelta(seconds=seconds))


@pytest.mark.parametrize("missing", ["runtime_profile_version", "workload_ids"])
def test_failure_detection_requires_profile_and_owning_workload(failed_event, missing):
    store = InMemoryStore()
    store.save_profile(profile_without())
    event = failure_for(
        failed_event, **{missing: "" if missing == "runtime_profile_version" else []}
    )
    with pytest.raises(ValueError, match="requires"):
        CompletionService(store).handle_failure_detected(event)
    assert store.list_workflows() == [], (
        "incomplete ownership cannot publish a containment workflow"
    )


@pytest.mark.parametrize("prior_disable", [logging.NOTSET, logging.CRITICAL])
def test_bad_workload_log_snapshots_do_not_discard_valid_sibling_evidence(
    failed_event, caplog, prior_disable
) -> None:
    store = InMemoryStore()
    store.save_profile(profile_without())
    captured = []
    service = CompletionService(
        store,
        evidence_service=SimpleNamespace(capture=lambda **item: captured.append(item)),
    )
    event = failure_for(
        failed_event,
        workload_log_snapshots=[
            {"capture_error": "synthetic log read failure"},
            {"record_id": "no-node"},
            {"node_id": "node-a"},
            {"record_id": "bad-time", "node_id": "node-a", "captured_at": "invalid"},
            {
                "record_id": "good",
                "node_id": "node-a",
                "captured_at": failed_event.ended_at.isoformat(),
            },
        ],
    )
    inherited_disable = logging.root.manager.disable
    try:
        logging.disable(prior_disable)
        with caplog.at_level(logging.ERROR, logger="gpu_fault.service"):
            decision = service.handle_failure_detected(event)
        assert logging.root.manager.disable == prior_disable, (
            "diagnostic capture must restore the caller's logging suppression"
        )
    finally:
        logging.disable(inherited_disable)
    assert [item["record_id"] for item in captured] == ["good"], (
        "only complete, timestamped log snapshots may be persisted"
    )
    assert (
        store.get_workflow(decision.workflow_request_id).official_action
        == "STOP_WORKLOAD"
    ), "log capture failure cannot remove the required containment"
    assert "cannot persist emergency workload log evidence" in caplog.text, (
        "a malformed sibling must remain diagnosable"
    )
    diagnostics = [
        record
        for record in caplog.records
        if record.name == "gpu_fault.service"
        and "cannot persist emergency workload log evidence" in record.getMessage()
    ]
    assert len(diagnostics) == 1, (
        "the malformed timestamp must emit one diagnostic without duplicating valid evidence"
    )
    diagnostic = diagnostics[0]
    assert diagnostic.levelno == logging.ERROR and diagnostic.exc_info is not None, (
        "the malformed sibling's error must retain its exception diagnostic"
    )
    assert diagnostic.exc_info[0] is ValueError, (
        "the diagnostic must identify the actual malformed timestamp"
    )
    assert event.attempt_id in diagnostic.getMessage(), (
        "the malformed sibling's diagnostic must identify the owning attempt"
    )


def test_failure_without_log_capture_still_creates_one_owned_containment(
    failed_event,
) -> None:
    store = InMemoryStore()
    store.save_profile(profile_without())
    service = CompletionService(store)
    event = failure_for(failed_event)
    first = service.handle_failure_detected(event)
    second = service.handle_failure_detected(event)
    assert first.duplicate is False and second.duplicate is True, (
        "missing optional evidence capture does not remove containment idempotency"
    )
    assert first.workflow_request_id == second.workflow_request_id, (
        "repeated failure detection must not create a second STOP"
    )


def test_terminal_reconstructs_a_containment_with_a_missing_workflow_pointer(
    failed_event,
) -> None:
    store = InMemoryStore()
    store.save_profile(profile_without())
    service = CompletionService(store)
    containment = service.handle_failure_detected(failure_for(failed_event))
    incident = store.get_incident(containment.incident_id)
    store.save_incident(
        incident.model_copy(update={"workflow_request_id": None}), expected=incident
    )
    event = failed_event.model_copy(update={"workload_ids": ["training/job/unit"]})
    decision = service.handle_terminal(event)
    restored = store.get_incident(containment.incident_id)
    assert restored.workflow_request_id == containment.workflow_request_id, (
        "a historical dangling containment pointer must be reconstructed"
    )
    assert store.get_plan(decision.recovery_plan_id).restart_after_incident_id == (
        containment.incident_id
    ), "the repaired containment still gates the subsequent workload recovery"


def test_unknown_explicit_initiator_cannot_create_recursive_recovery(
    failed_event,
) -> None:
    store = InMemoryStore()
    store.save_profile(profile_without())
    event = failed_event.model_copy(
        update={
            "workload_ids": ["training/job/unit"],
            "termination_initiator_incident_id": "unavailable-initiator",
        }
    )
    decision = CompletionService(store).handle_terminal(event)
    assert decision.status is DecisionStatus.NO_ACTION, (
        "missing initiator evidence must not recursively create another recovery"
    )
    assert store.list_workflows() == [], (
        "the foreign initiator declaration cannot create a new passive STOP"
    )


@pytest.mark.parametrize("pointer", ["unassigned", "missing-workflow", "running"])
def test_marker_incident_reads_preserve_unfinished_recovery_evidence(
    failed_event, pointer
) -> None:
    store = InMemoryStore()
    store.save_profile(profile_without())
    incident_values = {"attempt_id": "earlier-attempt", "event_type": "XID"}
    incident, _ = active_recovery(
        store,
        failed_event,
        incident_values=incident_values,
        official_steps=[workflow_step(WorkflowOperation.RESET_GPU)],
        status=WorkflowStatus.RUNNING,
    )
    if pointer != "running":
        changed = incident.model_copy(
            update={
                "workflow_request_id": (
                    None if pointer == "unassigned" else "missing-workflow"
                )
            }
        )
        store.save_incident(changed, expected=incident)
        incident = changed
    marker = marker_for(
        failed_event, RecoveryAction.RESET_GPU, incident_id=incident.incident_id
    )
    service = CompletionService(store)
    service.add_marker(marker)
    if pointer == "missing-workflow":
        with pytest.raises(NotFoundError):
            service.handle_terminal(failed_event)
        assert store.get_decision_by_event(failed_event.event_key) is None, (
            "a missing recovery owner cannot become a completed decision"
        )
    else:
        decision = service.handle_terminal(failed_event)
        plan = store.get_plan(decision.recovery_plan_id)
        assert plan.restart_after_incident_id == incident.incident_id, (
            "unfinished node recovery remains the workload restart's premise"
        )
    assert store.list_markers() == [marker], (
        "an unfinished or unavailable workflow is not proof that a marker is obsolete"
    )


def test_a_marker_without_an_incident_can_create_only_its_own_plan(failed_event):
    store = InMemoryStore()
    store.save_profile(profile_without())
    service = CompletionService(store)
    service.add_marker(
        marker_for(failed_event, RecoveryAction.MARK_UNSCHEDULABLE, incident_id="")
    )
    decision = service.handle_terminal(failed_event)
    plan = store.get_plan(decision.recovery_plan_id)
    assert plan.incident_id and plan.incident_id.startswith("inc-"), (
        "an observational marker must not create an empty incident identity"
    )
    assert [step.action for step in plan.steps] == [
        RecoveryAction.MARK_UNSCHEDULABLE
    ], "the unassigned marker cannot invent an existing incident's action"


def test_terminal_rechecks_a_peer_decision_after_acquiring_the_event_lock(
    failed_event, monkeypatch
) -> None:
    store = InMemoryStore()
    store.save_profile(profile_without())
    decided = CompletionDecision(
        cluster_id=failed_event.cluster_id,
        attempt_id=failed_event.attempt_id,
        event_key=failed_event.event_key,
        status=DecisionStatus.NO_ACTION,
        reason="peer already decided",
    )
    transaction = store.completion_transaction

    @contextmanager
    def peer_then_lock(event_key):
        store.save_decision(decided)
        with transaction(event_key):
            yield

    def no_second_event(event):
        pytest.fail("the peer decision must be consumed before another event write")

    monkeypatch.setattr(store, "completion_transaction", peer_then_lock)
    monkeypatch.setattr(store, "save_event_if_absent", no_second_event)
    result = CompletionService(store).handle_terminal(failed_event)
    assert result.duplicate is True and result.reason == decided.reason, (
        "an in-lock peer decision remains authoritative"
    )


@pytest.mark.parametrize("foreign_cluster", ["other-cluster", None])
def test_terminal_cannot_use_a_foreign_or_unbound_marker_on_a_same_named_node(
    failed_event, foreign_cluster
) -> None:
    store = InMemoryStore()
    store.save_profile(profile_without())
    service = CompletionService(store)
    marker = marker_for(
        failed_event,
        RecoveryAction.REBOOT_NODE,
        cluster_id=foreign_cluster,
        incident_id="",
    )
    service.add_marker(marker)
    result = service.handle_terminal(failed_event)
    assert result.matched_marker_ids == [], (
        "node names and GPU aliases cannot substitute for the marker's cluster identity"
    )
    plan = store.get_plan(result.recovery_plan_id)
    assert [step.action for step in plan.steps] == [RecoveryAction.RESTART_WORKLOAD], (
        "an unrelated marker cannot promote a workload failure to a node reboot"
    )
    assert store.list_markers() == [marker], (
        "one tenant must not retire or modify another tenant's marker"
    )


@pytest.mark.parametrize(
    ("variant", "withdrawn"),
    [
        ("current", True),
        ("old-passive", True),
        ("other-attempt", False),
        ("no-restart", False),
        ("already-withdrawn", True),
    ],
)
def test_user_stop_withdraws_only_applicable_recovery_workflows(
    failed_event, variant, withdrawn
) -> None:
    store = InMemoryStore()
    store.save_profile(profile_without())
    incident_values, changes = {}, {}
    if variant in {"old-passive", "other-attempt"}:
        incident_values["attempt_id"] = "earlier-attempt"
    if variant == "other-attempt":
        incident_values["event_type"] = "XID"
    if variant == "no-restart":
        changes["official_steps"] = [workflow_step(WorkflowOperation.FREEZE_EVIDENCE)]
    if variant == "already-withdrawn":
        changes["workload_withdrawn_at"] = failed_event.ended_at
    _, workflow = active_recovery(
        store, failed_event, incident_values=incident_values, **changes
    )
    stopped = failed_event.model_copy(
        update={"terminal_status": TerminalStatus.STOPPED}
    )
    decision = CompletionService(store).handle_terminal(stopped)
    assert decision.status is DecisionStatus.NO_ACTION, (
        "a user stop must not plan another workload restart"
    )
    after = store.get_workflow(workflow.request_id)
    assert (after.workload_withdrawn_at is not None) is withdrawn, (
        "withdrawal must retain attempt and recovery ownership boundaries"
    )
    if variant == "already-withdrawn":
        assert after.workload_withdrawn_at == failed_event.ended_at, (
            "repeated terminal reports must not reset the withdrawal timestamp"
        )


@pytest.mark.parametrize(
    "boundary", ["list_active_workflow_incidents", "amend_workflow"]
)
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_failed_withdrawal_cannot_commit_a_terminal_decision_that_suppresses_retry(
    failed_event, monkeypatch, boundary, backend, tmp_path
) -> None:
    store = (
        InMemoryStore()
        if backend == "memory"
        else SqliteStore(str(tmp_path / "withdrawal.db"))
    )
    try:
        exercise_failed_withdrawal(store, failed_event, monkeypatch, boundary)
    finally:
        if isinstance(store, SqliteStore):
            store.close()
