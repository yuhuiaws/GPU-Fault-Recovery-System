from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.models import WorkflowStatus
from gpu_fault.orphaned_commands import DISPATCHER_ACTOR, cancel_orphaned_commands
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests._builders import workflow_request
from tests.execution.test_orphaned_commands_sweep import (
    COMMAND,
    WORKFLOW,
    orphan,
    reconcile_events,
)

OPEN_STATUSES = (
    RemoteCommandStatus.PENDING,
    RemoteCommandStatus.WAITING,
    RemoteCommandStatus.LEASED,
)
DRIFT = (
    ("status", WorkflowStatus.RUNNING),
    ("status", WorkflowStatus.SUCCEEDED),
    ("merge_revision", 5),
    ("execution_epoch", 5),
    ("fencing_token", 5),
    ("incident_id", "incident-moved"),
)


def assert_audit_rollback(store, monkeypatch, status, *, after_write=False):
    orphan(store, command_status=status)
    previous = store.get_remote_command(COMMAND).model_copy(deep=True)
    workflow = store.get_workflow(WORKFLOW).model_copy(deep=True)
    amend = store.amend_workflow

    def failed_audit(*args, **kwargs):
        if after_write:
            amend(*args, **kwargs)
        raise RuntimeError("synthetic audit write failure")

    monkeypatch.setattr(store, "amend_workflow", failed_audit)
    first = cancel_orphaned_commands(store, now=datetime.now(timezone.utc))
    assert first == {}, "the failed transaction cannot claim a completed cleanup"
    assert store.get_remote_command(COMMAND) == previous, (
        "an audit-write failure must leave cancellation retryable and unchanged"
    )
    assert store.get_workflow(WORKFLOW) == workflow, (
        "failed cancellation must roll back the audit and merge revision as well"
    )
    assert reconcile_events(store) == [], "a failed cleanup has no completion audit"

    monkeypatch.setattr(store, "amend_workflow", amend)
    second = cancel_orphaned_commands(store, now=datetime.now(timezone.utc))
    assert set(second) == {WORKFLOW}, "the next sweep retries the entire transition"
    events = reconcile_events(store)
    assert len(events) == 1, "exactly one audit event accompanies the committed cancel"
    assert events[0].details["command_ids"] == [COMMAND], (
        "the audit is bound to the command that was actually changed"
    )
    assert cancel_orphaned_commands(store, now=datetime.now(timezone.utc)) == {}, (
        "a committed cancel is idempotent on subsequent sweeps"
    )
    assert reconcile_events(store) == events, "retries must not duplicate the audit"


def assert_stale_workflow_is_untouched(store, field, value):
    _, expected = orphan(store)
    command = store.get_remote_command(COMMAND).model_copy(deep=True)
    changed = expected.model_copy(update={field: value})
    store.save_workflow(changed, expected=expected)
    result = store.cancel_orphaned_remote_commands(
        expected, now=datetime.now(timezone.utc), actor=DISPATCHER_ACTOR
    )
    assert result.command_ids == (), "a stale workflow snapshot cannot authorize cancel"
    assert result.counters == {"cancelled": 0, "cancellation_requested": 0}, (
        "refused cancellation reports no changed commands"
    )
    assert store.get_remote_command(COMMAND) == command, (
        "the command must remain available to its current workflow generation"
    )
    assert store.get_workflow(WORKFLOW) == changed, (
        "the current workflow must not be overwritten by stale audit state"
    )


def assert_missing_workflow_is_untouched(store):
    expected = workflow_request(
        "missing-workflow", "missing-incident", status=WorkflowStatus.FAILED
    )
    result = store.cancel_orphaned_remote_commands(
        expected, now=datetime.now(timezone.utc), actor=DISPATCHER_ACTOR
    )
    assert result.command_ids == (), "a nonexistent workflow cannot authorize cleanup"
    assert store.list_workflows(None) == [], "cleanup cannot create an audit-only row"


def assert_exact_changed_ids(store):
    _, expected = orphan(store, command_status=RemoteCommandStatus.LEASED)
    original = store.get_remote_command(COMMAND)
    already_requested = store.cancel_remote_commands_for_workflow(
        WORKFLOW, reason="prior cancellation request"
    )
    assert already_requested["cancellation_requested"] == 1, (
        "the fixture must contain a preexisting cancellation request"
    )
    for command_id, status in (
        ("z-new-waiting", RemoteCommandStatus.WAITING),
        ("a-new-lease", RemoteCommandStatus.LEASED),
        ("finished-command", RemoteCommandStatus.SUCCEEDED),
        ("unrelated-command", RemoteCommandStatus.PENDING),
    ):
        store.ensure_remote_command(
            original.model_copy(
                update={
                    "command_id": command_id,
                    "idempotency_key": command_id,
                    "status": status,
                    "workflow_request_id": (
                        "unrelated-workflow"
                        if command_id == "unrelated-command"
                        else WORKFLOW
                    ),
                }
            )
        )
    before = {
        command.command_id: command.model_copy(deep=True)
        for command in store.list_remote_commands()
    }
    result = store.cancel_orphaned_remote_commands(
        expected, now=datetime.now(timezone.utc), actor=DISPATCHER_ACTOR
    )
    assert result.command_ids == ("a-new-lease", "z-new-waiting"), (
        "audit IDs must be exactly the fresh transitions, ordered by command ID"
    )
    assert result.counters == {"cancelled": 1, "cancellation_requested": 1}, (
        "in-flight work must be requested to cancel, not reported as stopped"
    )
    (event,) = reconcile_events(store)
    assert event.details["command_ids"] == list(result.command_ids), (
        "the committed audit and returned IDs must agree"
    )
    assert event.details["cancelled_remote_commands"] == result.counters, (
        "the event must report the same committed counters"
    )
    for command_id in (COMMAND, "finished-command", "unrelated-command"):
        assert store.get_remote_command(command_id) == before[command_id], (
            f"cleanup changed ineligible command {command_id}"
        )


def assert_workflow_failure_is_isolated(store, monkeypatch):
    orphan(store)
    orphan(
        store,
        request_id="other-workflow",
        incident_id="other-incident",
        command_id="other-command",
    )
    amend = store.amend_workflow

    def fail_one(request_id, *args, **kwargs):
        if request_id == WORKFLOW:
            raise RuntimeError("synthetic one-workflow audit failure")
        return amend(request_id, *args, **kwargs)

    monkeypatch.setattr(store, "amend_workflow", fail_one)
    result = cancel_orphaned_commands(store, now=datetime.now(timezone.utc))
    assert set(result) == {"other-workflow"}, (
        "one failed audit must not roll back an independent workflow"
    )
    assert store.get_remote_command(COMMAND).status is RemoteCommandStatus.WAITING, (
        "the failed workflow must remain retryable"
    )
    assert reconcile_events(store) == [], "the failed workflow has no committed audit"
    assert (
        store.get_remote_command("other-command").status is RemoteCommandStatus.FAILED
    ), "the independent workflow must still cancel its waiting command"
    assert len(reconcile_events(store, "other-workflow")) == 1, (
        "the successful workflow must commit its cancellation audit"
    )
