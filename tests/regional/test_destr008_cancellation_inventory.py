from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.models import (
    PlanStatus,
    RecoveryPlan,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from gpu_fault.processor.models import ProcessorRequest
from gpu_fault.remote_command_models import BatchedStep, RemoteCommandStatus
from gpu_fault.store.shared.errors import NotFoundError
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.probes import destr008_cancellation_store as core
from tests.regional.test_destr008_cancellation_probe import (
    NOW,
    STAMP,
    acknowledge,
    armed_submission,
    claim,
    command,
    parent_write,
    plan,
    seed,
    setup,
)


@pytest.mark.parametrize(
    "change",
    [
        {"cluster_id": "wrong"},
        {"job_id": "wrong"},
        {"attempt_id": "wrong"},
        {"node_ids": ["spare-a"]},
        {"node_ids": ["fault-a", "foreign"]},
        {"node_ids": ["fault-a", "fault-a"]},
        {"policy_source": "NVIDIA"},
    ],
)
def test_wrong_incident_scope_refuses_all_store_mutation(
    change: dict[str, Any],
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    foreign = incident.model_copy(update=change)
    store.save_incident(foreign, expected=incident)
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "INCIDENT_SCOPE"
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"runtime_profile_version": "changed"}, "WORKFLOW_SCOPE"),
        ({"fencing_token": 2}, "WORKFLOW_SCOPE"),
        ({"workload_withdrawn_at": STAMP}, "WITHDRAWAL_SHAPE"),
        ({"predecessor_workflow_id": "foreign"}, "ROOT_BINDING"),
    ],
)
def test_wrong_workflow_binding_is_refusal(change: dict[str, Any], code: str) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    altered = workflow.model_copy(update=change)
    store.save_workflow(altered, expected=workflow)
    remote = command(store, incident, workflow)
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == code
    assert (
        store.get_remote_command(remote.command_id).status
        is RemoteCommandStatus.PENDING
    )


@pytest.mark.parametrize("field", ["node_ids", "branch_node_ids", "workload_ids"])
def test_step_targets_cannot_expand_the_plan(field: str) -> None:
    bound, _, _, store, watchdog, _, workflow = armed_submission()
    step = workflow.official_steps[0].model_copy(update={field: ["foreign"]})
    store.save_workflow(
        workflow.model_copy(update={"official_steps": [step]}), expected=workflow
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "STEP_SCOPE"
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None


@pytest.mark.parametrize(
    "parameters",
    [
        {"cluster_id": "wrong"},
        {"job_id": "wrong"},
        {"attempt_id": "wrong"},
        {"source_attempt_id": "wrong"},
        {"fault_node": "wrong"},
        {"spare_node": "wrong"},
        {"runtime_profile_version": "wrong"},
        {"node_id": "wrong"},
        {"target_node_id": "wrong"},
        {"node_ids": "fault-a"},
        {"target_node_ids": ["wrong"]},
        {"branch_node_ids": [None]},
        {"workload_ids": "training/job-a"},
        {"affected_workload_ids": ["training/other"]},
        {"nested": [{"source_attempt_id": "different"}]},
    ],
)
def test_nested_scope_fields_are_bound(parameters: dict[str, Any]) -> None:
    bound, _, _, store, watchdog, _, workflow = armed_submission()
    step = workflow.official_steps[0].model_copy(update={"parameters": parameters})
    store.save_workflow(
        workflow.model_copy(update={"official_steps": [step]}), expected=workflow
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "PARAMETER_SCOPE"
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None


def test_matching_nested_scope_is_accepted_but_bounded() -> None:
    bound = plan()
    core.bind_parameters(
        bound,
        {
            "nodes": [
                {"target_node_id": "spare-a", "node_ids": ["fault-a", "spare-a"]}
            ],
            "workload_ids": ["training/job-a"],
            "ignored_numeric": 3,
        },
    )
    for value in [
        {str(index): None for index in range(257)},
        [None] * 257,
        {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": None}}}}}}}}},
    ]:
        with pytest.raises(wire.ProbeError, match="PARAMETER_SIZE"):
            core.bind_parameters(bound, value)


@pytest.mark.parametrize(
    "change",
    [
        {"cluster_id": "wrong"},
        {"workflow_request_id": "wrong"},
        {"incident_id": "wrong"},
        {"fencing_token": 2},
    ],
)
def test_command_identity_refusal_never_cancels_another_scope(
    change: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    remote = command(store, incident, workflow, **change)
    # A controlled faulty filter must not widen cancellation authority.
    monkeypatch.setattr(store, "list_remote_commands", lambda **_: [remote])
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "COMMAND_SCOPE"
    assert (
        store.get_remote_command(remote.command_id).status
        is RemoteCommandStatus.PENDING
    )
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None


@pytest.mark.parametrize(
    "kind",
    [
        "incident_id",
        "incident_event",
        "embedded_workflow",
        "step",
        "batch",
        "authorization",
    ],
)
def test_embedded_command_and_batched_targets_are_checked(kind: str) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    changes: dict[str, Any] = {}
    expected = "COMMAND_SCOPE"
    if kind == "incident_id":
        changes["incident"] = incident.model_copy(update={"incident_id": "wrong"})
    elif kind == "incident_event":
        changes["incident"] = incident.model_copy(update={"event_id": "wrong"})
    elif kind == "embedded_workflow":
        changes["workflow"] = workflow.model_copy(update={"request_id": "wrong"})
    elif kind == "step":
        changes["step"] = workflow.official_steps[0].model_copy(
            update={"node_ids": ["wrong"]}
        )
        expected = "STEP_SCOPE"
    elif kind == "batch":
        changes["batched_steps"] = [
            BatchedStep(
                step_index=1,
                idempotency_key="batch",
                step=workflow.official_steps[0].model_copy(
                    update={"workload_ids": ["wrong"]}
                ),
            )
        ]
        expected = "STEP_SCOPE"
    else:
        changes["restart_authorization"] = {
            "cluster_id": "wrong",
            "job_id": "job-a",
            "source_attempt_id": "attempt-a",
            "source_gpu_count": 1,
            "restart_budget": 1,
            "restart_count": 1,
            "reservation_id": "reservation-a",
        }
    remote = command(store, incident, workflow, **changes)
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == expected
    assert (
        store.get_remote_command(remote.command_id).status
        is RemoteCommandStatus.PENDING
    )


@pytest.mark.parametrize(
    "status", [RemoteCommandStatus.PENDING, RemoteCommandStatus.WAITING]
)
def test_unleased_status_with_a_live_lease_is_not_force_cleared(
    status: RemoteCommandStatus,
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    remote = command(
        store, incident, workflow, status=status, lease_owner="still-physical"
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "COMMAND_LEASE_SHAPE"
    assert store.get_remote_command(remote.command_id) == remote


def test_leased_status_with_missing_lease_identity_is_refused() -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    remote = command(
        store, incident, workflow, status=RemoteCommandStatus.LEASED, lease_token=None
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "COMMAND_LEASE_SHAPE"
    assert store.get_remote_command(remote.command_id) == remote


@pytest.mark.parametrize(
    "changes",
    [
        {"lease_owner": "physical"},
        {"lease_token": "physical"},
        {"lease_expires_at": STAMP},
        {"result_details": {"outcome_unknown": True}},
        {"result_details": {"manual_confirmation_required": True}},
        {"result_details": {"node_action_interrupted": True}},
        {"result_details": {"stale_fence_swept": True}},
        {"result_details": {"post_cancellation_status": "WAITING"}},
        {"result_details": {"post_stale_fence_status": "WAITING"}},
        {
            "result_details": {
                "batched_results": {"0": {"details": {"outcome_unknown": True}}}
            }
        },
        {"result_details": {"node_results": [{"outcome_unknown": True}]}},
    ],
)
def test_terminal_command_needs_lease_free_known_outcome(
    changes: dict[str, Any],
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    remote = command(
        store, incident, workflow, status=RemoteCommandStatus.FAILED, **changes
    )
    assert watchdog.tick(bound.deadline_at).commands_active == 1
    result = watchdog.tick(bound.deadline_at + wire.DRAIN_SECONDS)
    assert result.state == "FAILED" and result.commands_active == 1
    assert (
        result.error_code == "DRAIN_UNRESOLVED" and not result.fence_release_authorized
    )
    assert store.get_remote_command(remote.command_id) == remote


@pytest.mark.parametrize("kind", ["owner", "expiry", "waiting", "unknown"])
def test_terminal_workflow_needs_fully_drained_execution(kind: str) -> None:
    bound, _, _, store, watchdog, _, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    changes: dict[str, Any]
    if kind == "owner":
        changes = {"execution_owner_id": "worker"}
    elif kind == "expiry":
        changes = {"execution_lease_expires_at": STAMP}
    else:
        changes = {
            "step_executions": [
                WorkflowStepExecution(
                    step_index=0,
                    operation=workflow.official_steps[0].operation,
                    status=WorkflowStepStatus.WAITING
                    if kind == "waiting"
                    else WorkflowStepStatus.FAILED,
                    details={}
                    if kind == "waiting"
                    else {"manual_confirmation_required": True},
                )
            ]
        }
    store.save_workflow(workflow.model_copy(update=changes), expected=workflow)
    assert watchdog.tick(bound.deadline_at).workflows_active == 1
    result = watchdog.tick(bound.deadline_at + wire.DRAIN_SECONDS)
    assert result.state == "FAILED" and result.workflows_active == 1


def test_unhandled_failure_is_pending_future_creation_authority() -> None:
    bound, _, port, store, watchdog = setup()
    watchdog.tick(NOW)
    claim(bound, port)
    _, workflow = seed(
        store, bound, status=WorkflowStatus.FAILED, failure_handled=False
    )
    acknowledge(bound, port)
    first = watchdog.tick(bound.deadline_at)
    assert first.pending_creation is True and first.state == "REVOKED"
    failed = watchdog.tick(bound.deadline_at + wire.DRAIN_SECONDS)
    assert failed.state == "FAILED" and failed.pending_creation is True
    child_event = f"support-after-{workflow.request_id}"
    child, child_workflow = seed(
        store,
        bound,
        status=WorkflowStatus.SUCCEEDED,
        incident_id=f"inc-{child_event}",
        event_id=child_event,
        workflow_id=f"workflow-{child_event}",
    )
    current = store.get_workflow(workflow.request_id)
    store.save_workflow(
        current.model_copy(update={"failure_handled_at": STAMP}), expected=current
    )
    result = watchdog.tick(bound.deadline_at + wire.DRAIN_SECONDS + 1)
    assert result.state == "FAILED" and result.error_code == "DRAIN_UNRESOLVED"
    assert (
        result.pending_creation is False
        and result.commands_active == result.workflows_active == 0
    )
    assert result.monitoring, "new drain proof must still complete its quiet interval"
    assert set(result.workflow_ids) == {workflow.request_id, child_workflow.request_id}
    assert (
        store.get_workflow(child.workflow_request_id).workload_withdrawn_at is not None
    )
    assert (
        watchdog.tick(bound.deadline_at + wire.DRAIN_SECONDS + 6).state == "QUIESCENT"
    )


def test_terminal_predecessor_descendant_is_not_missed_by_active_query() -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    child = workflow.model_copy(
        update={
            "request_id": "workflow-child",
            "predecessor_workflow_id": workflow.request_id,
            "status": WorkflowStatus.BLOCKED,
        }
    )
    store.save_workflow(child)
    store.save_incident(
        incident.model_copy(update={"workflow_request_id": child.request_id}),
        expected=incident,
    )
    assert (
        store.list_job_recovery_workflow_incidents(
            bound.cluster_id, bound.job_id, bound.attempt_id
        )
        == []
    )
    result = watchdog.tick(bound.deadline_at)
    assert set(result.workflow_ids) == {workflow.request_id, child.request_id}
    assert store.get_workflow(child.request_id).workload_withdrawn_at is not None
    assert watchdog.tick(bound.deadline_at + 5).state == "QUIESCENT"


def test_same_job_attempt_without_source_ancestry_is_not_owned() -> None:
    bound, _, _, store, watchdog, _, workflow = armed_submission()
    _, foreign = seed(
        store,
        bound,
        incident_id="foreign",
        event_id="foreign-event",
        workflow_id="foreign-workflow",
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "UNRELATED_JOB_RECOVERY"
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None
    assert store.get_workflow(foreign.request_id).workload_withdrawn_at is None


def test_processor_authority_blocks_quiescence_without_consuming_or_completing_it() -> (
    None
):
    bound, _, _, store, watchdog, _, _ = armed_submission(status=WorkflowStatus.FAILED)
    request = ProcessorRequest.from_http(
        method="POST",
        path="/v1/attempts/failure-detected",
        query="",
        body=wire.encode(
            {
                "cluster_id": bound.cluster_id,
                "job_id": bound.job_id,
                "attempt_id": bound.attempt_id,
                "node_id": bound.fault_node,
            }
        ).encode(),
        content_type="application/json",
        cluster_id=bound.cluster_id,
    )
    accepted, reason = store.try_enqueue_processor_request(
        request, max_depth=20, max_cluster_depth=10
    )
    assert accepted is not None and reason is None
    assert watchdog.tick(bound.deadline_at).pending_creation is True
    failed = watchdog.tick(bound.deadline_at + wire.DRAIN_SECONDS)
    assert failed.state == "FAILED" and failed.pending_creation
    assert store.has_incomplete_processor_requests_for_scopes(
        bound.cluster_id, set(request.correlation_scope_keys)
    ), "watchdog expiry must not consume processor creation authority"


@pytest.mark.parametrize("status", list(PlanStatus))
def test_source_recovery_plan_creation_authority_is_read_not_rewritten(
    status: PlanStatus,
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    recovery = RecoveryPlan(
        plan_id="recovery-a",
        incident_id=incident.incident_id,
        attempt_id=bound.attempt_id,
        trigger="local-test",
        runtime_profile_version=bound.runtime_profile_version,
        workflow_request_id=workflow.request_id,
        steps=[],
        status=status,
    )
    store.save_plan(recovery)
    store.save_workflow(
        workflow.model_copy(update={"source_plan_id": recovery.plan_id}),
        expected=workflow,
    )
    first = watchdog.tick(bound.deadline_at)
    assert first.pending_creation is (
        status in {PlanStatus.PENDING, PlanStatus.RUNNING}
    )
    assert store.get_plan(recovery.plan_id) == recovery


@pytest.mark.parametrize(
    ("reader", "response", "code"),
    [
        (
            "list_job_recovery_workflow_incidents",
            lambda incident, workflow: [(incident, workflow)]
            * (wire.MAX_OWNED_RECORDS + 1),
            "INVENTORY_SIZE",
        ),
        (
            "list_job_recovery_workflow_incidents",
            lambda incident, workflow: [(incident, workflow)] * 2,
            "DUPLICATE_WORKFLOW",
        ),
        (
            "list_job_recovery_workflow_incidents",
            lambda incident, workflow: [
                (incident, workflow.model_copy(update={"status": "UNKNOWN"}))
            ],
            "STORE_SHAPE",
        ),
        (
            "list_job_recovery_workflow_incidents",
            lambda incident, workflow: [],
            "INVENTORY_CHANGED",
        ),
        (
            "has_incomplete_processor_requests_for_scopes",
            lambda incident, workflow: None,
            "STORE_SHAPE",
        ),
        (
            "list_remote_commands",
            lambda incident, workflow: [None] * (wire.MAX_OWNED_RECORDS + 1),
            "INVENTORY_SIZE",
        ),
        ("list_remote_commands", lambda incident, workflow: [None], "STORE_SHAPE"),
    ],
)
def test_unknown_or_oversized_inventories_fail_closed(
    reader: str, response, code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission()
    monkeypatch.setattr(
        store, reader, lambda *args, **kwargs: response(incident, workflow)
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == code
    assert store.get_workflow(workflow.request_id).workload_withdrawn_at is None


def test_disappeared_observed_command_is_unresolved_not_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    command(store, incident, workflow)
    watchdog.tick(bound.deadline_at)
    monkeypatch.setattr(store, "list_remote_commands", lambda **_: [])
    result = watchdog.tick(bound.deadline_at + 5)
    assert result.state == "FAILED" and result.error_code == "RECORD_MISSING"
    assert result.command_ids == ["command-a"] and result.monitoring
    assert result.commands_active is None, (
        "absence is not a count of physically stopped commands"
    )


def test_missing_root_and_store_notfound_are_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, _, _, store, watchdog, _, _ = armed_submission(status=WorkflowStatus.FAILED)
    watchdog.tick(bound.deadline_at)
    monkeypatch.setattr(store, "get_incident_by_event", lambda _: None)
    result = watchdog.tick(bound.deadline_at + 1)
    assert result.state == "FAILED" and result.error_code == "ROOT_MISSING"
    assert result.monitoring and result.root is not None
    monkeypatch.setattr(
        store,
        "list_job_recovery_workflow_incidents",
        lambda *args, **kwargs: (_ for _ in ()).throw(NotFoundError("private-model")),
    )
    result = watchdog.tick(bound.deadline_at + 2)
    assert result.state == "FAILED" and result.error_code == "RECORD_MISSING"
    assert "private-model" not in wire.encode(result)


def test_wrong_event_link_and_unclaimed_root_are_refused() -> None:
    bound, _, port, store, watchdog = setup()
    watchdog.tick(NOW)
    incident, _ = seed(store, bound)
    assert watchdog.tick(bound.deadline_at).error_code == "UNCLAIMED_EVENT"
    bound, _, port, store, watchdog = setup()
    watchdog.tick(NOW)
    claim(bound, port)
    incident, _ = seed(store, bound, event_id="foreign-event")
    store.link_event_to_incident(bound.event_id, incident.incident_id)
    assert watchdog.tick(bound.deadline_at).error_code == "EVENT_SOURCE"


def test_explicit_negative_close_rejects_a_successful_root() -> None:
    bound, _, port, _, watchdog, _, _ = armed_submission(
        status=WorkflowStatus.SUCCEEDED
    )
    parent_write(port, wire.request_close(bound, port.read().control, now=NOW + 3))
    result = watchdog.tick(NOW + 3)
    assert result.state == "FAILED" and result.error_code == "NOT_NEGATIVE_TERMINAL"
    assert result.producer_revoked and not result.fence_release_authorized


def test_unexpected_restart_attempt_is_never_accepted_as_this_attempt() -> None:
    bound, _, _, store, watchdog, _, workflow = armed_submission()
    execution = WorkflowStepExecution(
        step_index=0,
        operation=workflow.official_steps[0].operation,
        status=WorkflowStepStatus.SUCCEEDED,
        details={"restart_attempt_id": "attempt-b"},
    )
    store.save_workflow(
        workflow.model_copy(update={"step_executions": [execution]}), expected=workflow
    )
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "FAILED" and result.error_code == "UNEXPECTED_RESTART"


def test_record_and_inventory_helpers_reject_unknown_shapes() -> None:
    with pytest.raises(wire.ProbeError, match="INVENTORY_SIZE"):
        core.bounded(None)
    bound, _, _, store, _, _, workflow = armed_submission()
    huge = workflow.model_copy(
        update={"blocked_reasons": ["x" * core.MAX_RECORD_BYTES]}
    )
    with pytest.raises(wire.ProbeError, match="RECORD_SIZE"):
        core.checked(WorkflowRequest, huge)
    with pytest.raises(wire.ProbeError, match="REVOCATION_REQUIRED"):
        core.withdraw(
            store,
            bound,
            wire.initial_control(bound),
            core.Inventory(None, {}, {}, False, False),
            now=NOW,
        )
    assert core.outcome_unknown({"known": [False, {"outcome_unknown": False}]}) is False
