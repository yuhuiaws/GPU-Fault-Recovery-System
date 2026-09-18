from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import closing
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest

from gpu_fault.models import WorkflowStatus
from gpu_fault.orphaned_commands import DISPATCHER_ACTOR, cancel_orphaned_commands
from gpu_fault.regional import RemoteCommandResult
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.state_table_migrate import state_table_maintenance
from gpu_fault.store import NotFoundError
from tests.execution.test_orphaned_commands_sweep import (
    COMMAND,
    WORKFLOW,
    orphan,
    reconcile_events,
)
from tests.store import _cov95_runtime_orphan_audit as contract
from tests.store import _cov95_runtime_postgres as postgres
from tests.store._postgres_processor_claim_support import postgres_store_instance
from tests.store.test_postgres_workflow_state_tables import select_mode


@pytest.fixture(params=["legacy", "dual", "dedicated"])
def store(request):
    url = postgres.validated_url()
    with closing(postgres_store_instance()) as instances:
        instance = next(instances)
        with state_table_maintenance(url) as connection:
            select_mode(connection, "workflow", request.param)
            select_mode(connection, "remote_command", request.param)
        yield instance


@pytest.mark.parametrize("status", contract.OPEN_STATUSES)
@pytest.mark.parametrize(
    "after_write", [False, True], ids=["before-audit", "after-audit"]
)
def test_orphan_cancellation_cannot_commit_without_its_workflow_audit(
    store, monkeypatch, status, after_write
) -> None:
    contract.assert_audit_rollback(store, monkeypatch, status, after_write=after_write)


@pytest.mark.parametrize(("field", "value"), contract.DRIFT)
def test_orphan_cancellation_rechecks_the_current_workflow(store, field, value) -> None:
    contract.assert_stale_workflow_is_untouched(store, field, value)


def test_orphan_cancellation_requires_an_existing_workflow(store) -> None:
    contract.assert_missing_workflow_is_untouched(store)


def test_orphan_cancellation_audits_only_commands_it_changed(store) -> None:
    contract.assert_exact_changed_ids(store)


def test_orphan_audit_failure_is_isolated_per_workflow(store, monkeypatch) -> None:
    contract.assert_workflow_failure_is_isolated(store, monkeypatch)


@pytest.mark.parametrize("audit_fails", [False, True], ids=["commit", "rollback"])
def test_executor_waits_for_the_atomic_audit_and_cancel(
    store, monkeypatch, audit_fails
) -> None:
    orphan(store, command_status=RemoteCommandStatus.PENDING)
    (leased,) = store.claim_remote_commands(
        "cluster-a", "executor-example", limit=1, lease_seconds=60
    )
    entered = Event()
    release = Event()
    completion_started = Event()
    amend = store.amend_workflow

    def paused_audit(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test audit barrier timed out")
        value = amend(*args, **kwargs)
        if audit_fails:
            raise RuntimeError("synthetic failure after audit write")
        return value

    monkeypatch.setattr(store, "amend_workflow", paused_audit)
    with (
        postgres.peer_store("legacy") as peer,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        cancelled = pool.submit(
            cancel_orphaned_commands, store, now=datetime.now(timezone.utc)
        )
        try:
            assert entered.wait(5), "the cancellation never reached its audit boundary"
            assert peer.get_remote_command(COMMAND) == leased, (
                "another connection observed an unaudited cancellation"
            )
            assert reconcile_events(peer) == [], "the in-flight audit is not committed"

            def complete():
                completion_started.set()
                return peer.complete_remote_command(
                    "cluster-a",
                    COMMAND,
                    RemoteCommandResult(
                        lease_token=leased.lease_token,
                        status=RemoteCommandStatus.SUCCEEDED,
                        details={"evidence": "executor-example"},
                    ),
                )

            completed = pool.submit(complete)
            assert completion_started.wait(5), "the executor completion did not start"
            with pytest.raises(TimeoutError):
                completed.result(timeout=0.1)
        finally:
            release.set()
        outcome = cancelled.result(timeout=5)
        command = completed.result(timeout=5)
        if audit_fails:
            assert outcome == {}, "the failed audit must roll back cancellation"
            assert command.status is RemoteCommandStatus.SUCCEEDED, (
                "executor completion must see the command's uncancelled state"
            )
            assert reconcile_events(peer) == [], "a rolled-back audit cannot remain"
        else:
            assert outcome == {
                WORKFLOW: {"cancelled": 0, "cancellation_requested": 1}
            }, "the committed transaction requests cancellation exactly once"
            assert command.status is RemoteCommandStatus.FAILED, (
                "late completion must preserve the committed cancellation"
            )
            assert command.status_source == "completed-after-cancellation", (
                "the executor result must retain the cancellation provenance"
            )
            (audit,) = reconcile_events(peer)
            assert audit.details["command_ids"] == [COMMAND], (
                "the persisted audit must bind the in-flight command"
            )


def test_concurrent_workflow_writer_is_rechecked_after_its_commit(store) -> None:
    _, expected = orphan(store)
    command = store.get_remote_command(COMMAND)
    started = Event()
    with (
        postgres.peer_store("legacy") as peer,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        with peer.completion_transaction("orphan-workflow-writer"):
            changed = peer.amend_workflow(WORKFLOW, {"status": WorkflowStatus.RUNNING})

            def cancel():
                started.set()
                return store.cancel_orphaned_remote_commands(
                    expected, now=datetime.now(timezone.utc), actor=DISPATCHER_ACTOR
                )

            future = pool.submit(cancel)
            assert started.wait(5), "the concurrent cancellation did not start"
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
        result = future.result(timeout=5)
        assert result.command_ids == (), "the stale terminal snapshot must be refused"
        assert peer.get_workflow(WORKFLOW) == changed, (
            "the audit cannot overwrite the resumed workflow"
        )
        assert peer.get_remote_command(COMMAND) == command, (
            "current live work must remain uncancelled"
        )


@pytest.mark.parametrize("deleted", [False, True], ids=["completed", "deleted"])
def test_command_is_rechecked_after_candidate_enumeration(
    store, monkeypatch, deleted
) -> None:
    _, expected = orphan(store)
    candidates = store.list_remote_commands
    with postgres.peer_store("legacy") as peer:

        def raced_candidates(*args, **kwargs):
            rows = candidates(*args, **kwargs)
            assert peer.cancel_remote_command(COMMAND, reason="another reconciler"), (
                "the competing reconciler must win before this transaction locks"
            )
            if deleted:
                assert (
                    peer.cleanup_terminal_remote_commands(
                        older_than=datetime.now(timezone.utc) + timedelta(seconds=1),
                        limit=1,
                    )
                    == 1
                ), "the terminal command must disappear before its locked read"
            return rows

        monkeypatch.setattr(store, "list_remote_commands", raced_candidates)
        result = store.cancel_orphaned_remote_commands(
            expected, now=datetime.now(timezone.utc), actor=DISPATCHER_ACTOR
        )
        assert result.command_ids == (), "a competing cleanup is not this sweep's work"
        assert reconcile_events(peer) == [], (
            "no audit may claim another writer's changes"
        )
        if deleted:
            with pytest.raises(NotFoundError):
                peer.get_remote_command(COMMAND)
        else:
            assert peer.get_remote_command(COMMAND).error == "another reconciler", (
                "the locked recheck must preserve the prior writer's reason"
            )
