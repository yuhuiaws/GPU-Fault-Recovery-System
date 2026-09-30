from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.cluster_executor import ClusterActionExecutor, ClusterExecutorError
from gpu_fault.cluster_executor.lease import CommandLeaseWatch, LeaseAuthorityError
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from gpu_fault.store import InMemoryStore
from tests._builders import fault_incident, workflow_request, workflow_step


class MemoryClient:
    cluster_id = "cluster-a"

    def __init__(self) -> None:
        self.store = InMemoryStore()
        self.renewals = []
        self.completed = []
        self.error = None
        self.response_update = {}
        step = workflow_step(WorkflowOperation.VALIDATE_HOST, "unit-owner")
        workflow = workflow_request("wf", "incident", official_steps=[step])
        incident = fault_incident(
            "incident", "event", fencing_token=workflow.fencing_token
        )
        self.store.ensure_remote_command(
            RemoteActionCommand(
                command_id="command",
                cluster_id=self.cluster_id,
                workflow_request_id=workflow.request_id,
                incident_id=incident.incident_id,
                step_index=0,
                fencing_token=workflow.fencing_token,
                idempotency_key="unit-command",
                step=step,
                workflow=workflow,
                incident=incident,
            )
        )

    def claim(
        self, executor_id, *, execution_owners, max_commands, lease_seconds, **kwargs
    ):
        return self.store.claim_remote_commands(
            self.cluster_id,
            executor_id,
            execution_owners=set(execution_owners),
            limit=max_commands,
            lease_seconds=lease_seconds,
        )

    def renew(self, command, executor_id, lease_seconds):
        self.renewals.append(command.command_id)
        if self.error:
            raise self.error
        renewed = self.store.renew_remote_command_lease(
            self.cluster_id,
            command.command_id,
            executor_id=executor_id,
            lease_token=command.lease_token,
            lease_seconds=lease_seconds,
        )
        return renewed.model_copy(update=self.response_update)

    def complete(self, command, result):
        self.completed.append(command.command_id)
        return self.store.complete_remote_command(
            self.cluster_id, command.command_id, result
        )


class Adapter:
    owner = "unit-owner"

    def __init__(self) -> None:
        self.calls = []

    def supports(self, step):
        return True

    def execute(self, context):
        self.calls.append(context.idempotency_key)
        return WorkflowStepOutcome.succeeded()


def executor(tmp_path, client, adapter, **kwargs):
    return ClusterActionExecutor(
        client,
        [adapter],
        executor_id="unit-executor",
        allowed_namespaces=set(),
        claim_state_path=str(tmp_path / "claim.json"),
        liveness_state_path=str(tmp_path / "loop.json"),
        **kwargs,
    )


def test_preexecution_renews_the_authoritative_lease_before_adapter_entry(
    tmp_path,
) -> None:
    client, adapter = MemoryClient(), Adapter()
    value = executor(tmp_path, client, adapter)
    assert value.run_once() == 1
    assert client.renewals == ["command"]
    assert adapter.calls == ["unit-command"]
    assert client.completed == ["command"]


@pytest.mark.parametrize("code", [403, 409, 503, None])
def test_unconfirmed_preexecution_renewal_never_starts_or_reports(
    tmp_path, code
) -> None:
    client, adapter = MemoryClient(), Adapter()
    client.error = ClusterExecutorError("isolated refusal", status_code=code)
    value = executor(tmp_path, client, adapter)
    assert value.run_once() == 1
    assert adapter.calls == [] and client.completed == []
    assert value.lease_renewal_failures == 1
    assert value.results_withheld_total == 1


@pytest.mark.parametrize(
    "update",
    [
        {"lease_owner": "other"},
        {"lease_token": "different"},
        {"command_id": "different"},
        {"cluster_id": "different"},
        {"status": RemoteCommandStatus.PENDING},
        {"lease_expires_at": None},
        {"lease_expires_at": datetime(2020, 1, 1, tzinfo=timezone.utc)},
        {"lease_expires_at": datetime(2020, 1, 1)},
        {"cancellation_requested_at": datetime(2020, 1, 1, tzinfo=timezone.utc)},
    ],
    ids=[
        "owner",
        "token",
        "command",
        "cluster",
        "status",
        "no-expiry",
        "expired",
        "naive",
        "cancelled",
    ],
)
def test_invalid_renewal_identity_or_window_never_admits(
    tmp_path, update: dict[str, Any]
) -> None:
    client, adapter = MemoryClient(), Adapter()
    client.response_update = update
    value = executor(tmp_path, client, adapter)
    assert value.run_once() == 1
    assert adapter.calls == [], "an invalid or cancelled renewal never admits"
    if "cancellation_requested_at" in update:
        # A cancellation carried by an otherwise authoritative renewal is the
        # one refusal that is answered: nothing ran, and the row must not stay
        # LEASED with a cancellation nobody settles (live 2026-09-28).
        assert client.completed == ["command"], "the no-start is reported"
    else:
        assert client.completed == [], "an unproven lease reports nothing"


def test_queue_wait_cannot_rebase_the_original_deadline(tmp_path) -> None:
    client, adapter = MemoryClient(), Adapter()
    now = [1000.0]
    value = executor(tmp_path, client, adapter, clock=lambda: now[0], lease_seconds=10)
    command = client.claim(
        value.executor_id,
        execution_owners=value.execution_owners,
        max_commands=1,
        lease_seconds=10,
    )[0]
    watch = value.lifecycle.watch_claim(command)
    expiry = watch.expires_at
    now[0] += 11
    outcome = value.lifecycle.run(command, watch)
    assert outcome.status is RemoteCommandStatus.WAITING
    assert watch.expires_at == expiry
    assert client.renewals == [] and adapter.calls == [] and client.completed == []


def test_expired_or_missing_claim_expiry_cannot_be_repaired_by_renewal(
    tmp_path,
) -> None:
    for expiry in (None, datetime.now(timezone.utc) - timedelta(seconds=1)):
        client, adapter = MemoryClient(), Adapter()
        value = executor(tmp_path, client, adapter)
        command = client.claim(
            value.executor_id,
            execution_owners=value.execution_owners,
            max_commands=1,
            lease_seconds=10,
        )[0].model_copy(update={"lease_expires_at": expiry})
        value.lifecycle.run(command)
        assert client.renewals == [] and adapter.calls == []


def test_authority_refusal_is_immediate_and_cannot_reset_a_previous_loss() -> None:
    watch = CommandLeaseWatch(
        lease_seconds=120, failure_limit=3, clock=lambda: 0, expires_at=120
    )
    assert (
        watch.renewal_failed(ClusterExecutorError("temporary", status_code=503))
        is False
    )
    assert watch.lost() is False
    assert watch.renewal_failed(LeaseAuthorityError("mismatched receipt")) is True
    reason = watch.hold_reason()
    assert (
        watch.renewal_failed(ClusterExecutorError("temporary", status_code=503))
        is False
    )
    watch.invalidate("later invalidation")
    assert watch.hold_reason() == reason


def test_already_cancelled_claim_is_not_renewed_or_executed(tmp_path) -> None:
    client, adapter = MemoryClient(), Adapter()
    value = executor(tmp_path, client, adapter)
    command = client.claim(
        value.executor_id,
        execution_owners=value.execution_owners,
        max_commands=1,
        lease_seconds=10,
    )[0].model_copy(
        update={
            "cancellation_requested_at": datetime.now(timezone.utc),
            "cancellation_reason": "cancelled before queue admission",
        }
    )
    result = value.lifecycle.run(command)
    assert result.status is RemoteCommandStatus.WAITING
    assert client.renewals == adapter.calls == [], "nothing is renewed or executed"
    # Live 2026-09-28: the cancelled claim is answered as a no-start, so the row
    # does not stay LEASED with a cancellation nobody settles.
    assert client.completed == ["command"], "the no-start is reported"
    # The server row here was never cancelled (only the claimed copy carried the
    # flag), so the post lands as a plain WAITING hand-back: harmless re-queue.
    settled = client.store.get_remote_command("command")
    assert settled.status is RemoteCommandStatus.WAITING
    assert settled.status_source == "executor-cancelled-before-start"
    assert settled.result_details["node_action_not_started"] is True


def test_unhandled_dispatch_failure_does_not_fabricate_an_action_result(
    tmp_path, monkeypatch
) -> None:
    client, adapter = MemoryClient(), Adapter()
    value = executor(tmp_path, client, adapter)

    def broken_dispatch(command):
        raise RuntimeError("isolated dispatch failure")

    monkeypatch.setattr(value.dispatch, "execute", broken_dispatch)
    assert value.run_once() == 1
    assert value.unexpected_failures == 1
    assert client.completed == [] and adapter.calls == []
