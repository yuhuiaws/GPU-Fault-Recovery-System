from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.regional import RemoteActionCommand, RemoteCommandResult
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import NotFoundError, WorkflowLeaseError
from tests.store._cov95_compat_remote import (
    claim_command,
    command_in_state,
    complete_command,
    save_command,
)
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("cluster_id", "other-cluster"),
        ("workflow_request_id", "other-workflow"),
        ("step_index", 1),
        ("fencing_token", 4),
        ("idempotency_key", "other-command-key"),
        ("operation", WorkflowOperation.VALIDATE_GPU),
        ("execution_owner", "other-owner"),
        ("node_ids", ["other-node"]),
        ("gpu_uuids", ["GPU-local"]),
        ("workload_ids", ["other-workload"]),
    ],
)
def test_command_replay_cannot_change_its_execution_identity(
    compat_store, field, changed
):
    original = save_command(compat_store, "identity")
    payload = original.model_dump()
    if field in {
        "operation",
        "execution_owner",
        "node_ids",
        "gpu_uuids",
        "workload_ids",
    }:
        payload["step"] = {**payload["step"], field: changed}
    else:
        payload[field] = changed
    conflicting = RemoteActionCommand.model_validate(payload)

    with pytest.raises(ValueError, match="identity conflict"):
        compat_store.ensure_remote_command(conflicting)
    assert compat_store.get_remote_command(original.command_id) == original, (
        f"a rejected {field} change must not overwrite the existing command"
    )
    replay = original.model_copy(
        update={"result_details": {"untrusted": "replacement"}}
    )
    assert compat_store.ensure_remote_command(replay) == original, (
        "replaying an existing identity must not overwrite execution evidence"
    )


@pytest.mark.parametrize("status", list(RemoteCommandStatus))
def test_workflow_cancellation_distinguishes_queued_from_inflight_commands(
    compat_store, status
):
    current = command_in_state(compat_store, "cancel-target", status)
    unrelated = save_command(compat_store, "unrelated", cluster_id="foreign-cluster")

    counts = compat_store.cancel_remote_commands_for_workflow(
        current.workflow_request_id, reason="operator withdrew workflow"
    )

    expected = {"cancelled": 0, "cancellation_requested": 0}
    stored = compat_store.get_remote_command(current.command_id)
    if status in {RemoteCommandStatus.PENDING, RemoteCommandStatus.WAITING}:
        expected["cancelled"] = 1
        assert stored.status is RemoteCommandStatus.FAILED, (
            "queued work can terminate immediately because no executor owns it"
        )
        assert stored.status_source == "workflow-timeout", (
            "the stored result must distinguish workflow cancellation from execution"
        )
        assert stored.error == "operator withdrew workflow", (
            "the terminal command must retain the cancellation reason"
        )
        assert stored.lease_owner is None and stored.lease_token is None, (
            "cancelled queued work cannot retain an executor lease"
        )
    elif status is RemoteCommandStatus.LEASED:
        expected["cancellation_requested"] = 1
        assert stored.status is RemoteCommandStatus.LEASED, (
            "requesting cancellation is not proof that the executor has stopped"
        )
        assert stored.cancellation_requested_at is not None, (
            "an in-flight executor needs a durable cancellation signal"
        )
        assert stored.cancellation_reason == "operator withdrew workflow", (
            "the executor must receive the workflow cancellation reason"
        )
        assert (
            compat_store.claim_remote_commands(
                current.cluster_id, "replacement", limit=1, lease_seconds=60
            )
            == []
        ), "a cancellation request must not be handed to another executor"
        settled = complete_command(
            compat_store,
            stored,
            RemoteCommandStatus.SUCCEEDED,
            details={"observed": True},
        )
        assert settled.status is RemoteCommandStatus.FAILED, (
            "late success evidence cannot undo the cancelled workflow's outcome"
        )
        assert settled.result_details["post_cancellation_status"] == "SUCCEEDED", (
            "late executor evidence must be retained without being treated as success"
        )
        assert settled.result_details["observed"] is True, (
            "cancellation must preserve what the executor actually reported"
        )
    else:
        assert stored == current, (
            "cancellation must not rewrite an existing terminal result"
        )
    assert counts == expected, (
        "the returned counters must describe only actual transitions"
    )
    assert compat_store.get_remote_command(unrelated.command_id) == unrelated, (
        "cancelling one workflow must not affect another cluster's command"
    )
    assert compat_store.cancel_remote_commands_for_workflow(
        current.workflow_request_id, reason="retry of withdrawal"
    ) == {"cancelled": 0, "cancellation_requested": 0}, (
        "repeating the withdrawal must not count already cancelled commands again"
    )


@pytest.mark.parametrize("status", list(RemoteCommandStatus))
def test_single_command_preemption_refuses_inflight_and_terminal_work(
    compat_store, status
):
    original = command_in_state(compat_store, "preempt", status)
    cancellable = status in {RemoteCommandStatus.PENDING, RemoteCommandStatus.WAITING}
    assert compat_store.cancel_remote_command("missing", reason="preempted") is False, (
        "preemption must not create a missing command"
    )

    assert (
        compat_store.cancel_remote_command(
            original.command_id, reason="higher-priority workflow"
        )
        is cancellable
    ), "only unleased pending or waiting work is preemptible"
    current = compat_store.get_remote_command(original.command_id)
    if cancellable:
        assert current.status is RemoteCommandStatus.FAILED, (
            "successful preemption must leave a terminal row"
        )
        assert current.status_source == "workflow-preempted", (
            "preemption must remain distinguishable from timeout and execution failure"
        )
        assert current.error == "higher-priority workflow", (
            "the stored command must retain the preemption reason"
        )
    else:
        assert current == original, (
            "refused preemption must preserve all command evidence"
        )
    assert (
        compat_store.cancel_remote_command(
            original.command_id, reason="repeated preemption"
        )
        is False
    ), "preemption must be idempotent"


@pytest.mark.parametrize("method", ["renew", "complete"])
@pytest.mark.parametrize(
    "missing", [False, True], ids=["foreign-cluster", "missing-id"]
)
def test_command_mutations_hide_foreign_or_missing_rows(compat_store, method, missing):
    command = claim_command(compat_store, save_command(compat_store, "owned"))
    cluster_id = command.cluster_id if missing else "foreign-cluster"
    command_id = "absent" if missing else command.command_id

    with pytest.raises(NotFoundError):
        if method == "renew":
            compat_store.renew_remote_command_lease(
                cluster_id,
                command_id,
                "executor",
                command.lease_token,
                lease_seconds=60,
            )
        else:
            compat_store.complete_remote_command(
                cluster_id,
                command_id,
                RemoteCommandResult(
                    lease_token=command.lease_token,
                    status=RemoteCommandStatus.SUCCEEDED,
                ),
            )
    assert compat_store.get_remote_command(command.command_id) == command, (
        "a refused lookup must not mutate the real command"
    )
    with pytest.raises(NotFoundError, match="absent"):
        compat_store.get_remote_command("absent")


@pytest.mark.parametrize("invalid", ["owner", "token", "expired", "completed"])
def test_remote_lease_renewal_refusals_preserve_current_evidence(compat_store, invalid):
    command = claim_command(
        compat_store,
        save_command(compat_store, "lease-refusal"),
        lease_seconds=-1 if invalid == "expired" else 600,
    )
    if invalid == "completed":
        complete_command(compat_store, command, RemoteCommandStatus.SUCCEEDED)
    before = compat_store.get_remote_command(command.command_id)
    owner = "other-executor" if invalid == "owner" else "executor"
    token = "not-the-issued-lease" if invalid == "token" else command.lease_token

    with pytest.raises(WorkflowLeaseError, match="lease"):
        compat_store.renew_remote_command_lease(
            command.cluster_id, command.command_id, owner, token, lease_seconds=60
        )
    assert compat_store.get_remote_command(command.command_id) == before, (
        f"refusing the {invalid} lease must leave all result and cancellation state intact"
    )


def test_unclaimed_expiry_is_bounded_and_counts_only_pending_work(compat_store):
    old = datetime.now(UTC) - timedelta(hours=2)
    cutoff = old + timedelta(minutes=1)
    for name in ("a-old", "b-old"):
        save_command(compat_store, name, at=old, result_details={"retained": name})
    untouched = [
        save_command(compat_store, "recent", at=cutoff + timedelta(microseconds=1)),
        save_command(
            compat_store, "waiting", at=old, status=RemoteCommandStatus.WAITING
        ),
        save_command(compat_store, "leased", at=old, status=RemoteCommandStatus.LEASED),
        save_command(
            compat_store, "done", at=old, status=RemoteCommandStatus.SUCCEEDED
        ),
        save_command(compat_store, "future", cluster_id="future-cluster", at=NOW),
    ]

    assert (
        compat_store.expire_unclaimed_remote_commands(older_than=cutoff, limit=1) == 1
    ), "the expiry budget must apply to eligible pending rows in stable order"
    first = compat_store.get_remote_command("a-old")
    assert first.status is RemoteCommandStatus.FAILED, (
        "unclaimed work must become terminal"
    )
    assert first.status_source == "unclaimed-deadline-exceeded", (
        "expiry must not masquerade as executor failure"
    )
    assert first.result_details["retained"] == "a-old", (
        "dead-lettering must preserve already recorded command details"
    )
    assert first.result_details["unclaimed_age_seconds"] >= 7200, (
        "the stored diagnostic must report the command's real unclaimed age"
    )
    assert (
        compat_store.get_remote_command("b-old").status is RemoteCommandStatus.PENDING
    ), "the next pending command must remain available after a one-row expiry budget"
    assert (
        compat_store.expire_unclaimed_remote_commands(older_than=cutoff, limit=20) == 1
    ), "the next sweep must finish the remaining stale pending command"
    for command in untouched:
        assert compat_store.get_remote_command(command.command_id) == command, (
            f"unclaimed expiry changed ineligible command {command.command_id}"
        )
    stats = compat_store.remote_command_stats(now=cutoff + timedelta(seconds=10))
    assert stats["unclaimed_expired_total"] == 2, (
        "the expiry counter must count both rows"
    )
    assert stats["open_by_cluster"] == {"cluster-local": 3, "future-cluster": 1}, (
        "open queue counts must exclude expired and completed commands"
    )
    assert stats["oldest_unclaimed_age_seconds_by_cluster"]["future-cluster"] == 0.0, (
        "clock skew must not produce a negative unclaimed age"
    )
    assert compat_store.list_remote_commands(workflow_request_ids=[]) == [], (
        "an empty workflow selection must never expand to a fleet-wide read"
    )
    assert [
        item.command_id
        for item in compat_store.list_remote_commands(
            workflow_request_ids=iter(
                ["workflow/b-old", "workflow/a-old", "workflow/a-old"]
            )
        )
    ] == ["a-old", "b-old"], (
        "workflow filters must deduplicate input and preserve row order"
    )


def test_stale_fence_sweep_preserves_live_and_current_generation_leases(compat_store):
    old = datetime.now(UTC) - timedelta(hours=1)
    commands = {}
    for name in ("a-stale", "b-stale", "current", "live-stale"):
        saved = save_command(compat_store, name, cluster_id=f"cluster/{name}", at=old)
        commands[name] = claim_command(
            compat_store, saved, lease_seconds=600 if name == "live-stale" else -10
        )
        if name != "current":
            workflow = compat_store.get_workflow(saved.workflow_request_id)
            compat_store.save_workflow(
                workflow.model_copy(
                    update={"fencing_token": workflow.fencing_token + 1}
                ),
                expected=workflow,
            )
    cutoff = datetime.now(UTC) - timedelta(seconds=1)

    assert (
        compat_store.expire_stale_fenced_remote_commands(
            lease_expired_before=cutoff, limit=1
        )
        == 1
    ), "the fence sweep must stop after its first eligible expired generation"
    first = compat_store.get_remote_command("a-stale")
    assert first.status is RemoteCommandStatus.FAILED, (
        "an abandoned stale lease must settle"
    )
    assert first.status_source == "stale-fence", (
        "the outcome must retain its fencing cause"
    )
    assert first.result_details["stale_fence_swept"] is True, (
        "swept abandonment must remain distinguishable from executor completion"
    )
    assert first.last_lease_owner == "executor" and first.lease_owner is None, (
        "sweeping must preserve attribution while relinquishing the executor lease"
    )
    assert compat_store.get_remote_command("b-stale") == commands["b-stale"], (
        "the one-row budget must leave the next stale command unchanged"
    )
    assert (
        compat_store.expire_stale_fenced_remote_commands(
            lease_expired_before=cutoff, limit=20
        )
        == 1
    ), "only the second abandoned stale generation remains eligible"
    for name in ("current", "live-stale"):
        assert compat_store.get_remote_command(name) == commands[name], (
            f"the sweep must not close the {name} lease"
        )
    assert (
        compat_store.expire_stale_fenced_remote_commands(
            lease_expired_before=cutoff, limit=20
        )
        == 0
    ), "completed sweep work must not be counted again"


def test_terminal_command_retention_uses_update_time_and_stable_limits(compat_store):
    cutoff = NOW - timedelta(days=1)
    for name, status in (
        ("a-failed", RemoteCommandStatus.FAILED),
        ("b-succeeded", RemoteCommandStatus.SUCCEEDED),
    ):
        save_command(compat_store, name, at=cutoff, status=status)
    retained = [
        save_command(
            compat_store,
            "recent-terminal",
            at=cutoff + timedelta(microseconds=1),
            status=RemoteCommandStatus.SUCCEEDED,
        ),
        save_command(compat_store, "old-open", at=cutoff - timedelta(days=1)),
    ]

    assert (
        compat_store.cleanup_terminal_remote_commands(older_than=cutoff, limit=1) == 1
    ), "retention must apply its limit after selecting old terminal rows"
    with pytest.raises(NotFoundError, match="a-failed"):
        compat_store.get_remote_command("a-failed")
    assert (
        compat_store.cleanup_terminal_remote_commands(older_than=cutoff, limit=20) == 1
    ), "the next terminal row at the inclusive cutoff must be removed"
    assert compat_store.list_remote_commands() == retained[::-1], (
        "old unfinished and recent terminal commands must survive in creation order"
    )
    assert (
        compat_store.cleanup_terminal_remote_commands(older_than=cutoff, limit=20) == 0
    ), "an exhausted retention sweep must report no additional work"
