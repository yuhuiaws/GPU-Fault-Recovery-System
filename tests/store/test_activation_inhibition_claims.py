from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from gpu_fault.regional_compatibility import (
    CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
    REMOTE_STEP_BATCHING_PROTOCOL_VERSION,
    command_protocol_eligible,
)
from gpu_fault.remote_command_models import BatchedStep
from gpu_fault.store import InMemoryStore, SqliteStore
from tests.regional._regional_support import workflow_state

ABSENT = object()
NOW = datetime.now(timezone.utc)


def command(
    name: str, marker: Any = ABSENT, *, batched: bool = False, **updates: Any
) -> RemoteActionCommand:
    context = workflow_state()
    parameters = {} if marker is ABSENT else {"activation_forbidden": marker}
    step = context.step.model_copy(
        update={
            "operation": WorkflowOperation.REPLACE_NODE,
            "parameters": {} if batched else parameters,
        }
    )
    tail = (
        [
            BatchedStep(
                step_index=1,
                step=step.model_copy(update={"parameters": parameters}),
                idempotency_key=f"{name}/tail",
            )
        ]
        if batched
        else []
    )
    value = RemoteActionCommand(
        command_id=name,
        cluster_id="cluster-a",
        workflow_request_id=f"workflow-{name}",
        incident_id=f"incident-{name}",
        step_index=0,
        fencing_token=context.workflow.fencing_token,
        idempotency_key=f"{name}/head",
        step=step,
        batched_steps=tail,
        workflow=context.workflow.model_copy(update={"request_id": f"workflow-{name}"}),
        incident=context.incident.model_copy(
            update={"incident_id": f"incident-{name}"}
        ),
        created_at=NOW,
        updated_at=NOW,
    )
    return value.model_copy(update=updates)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    value = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "claims.db"))
    )
    yield value
    if request.param == "sqlite":
        value.close()


@pytest.mark.parametrize("protocol", [1, 2, 3])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("marker", [True, False, None, 0, "true", {}])
def test_old_protocol_cannot_claim_marker_but_does_not_starve_ordinary_work(
    store, protocol: int, batched: bool, marker: Any
) -> None:
    guarded = command("a-inhibited", marker, batched=batched)
    ordinary = command("b-ordinary")
    store.ensure_remote_command(guarded)
    store.ensure_remote_command(ordinary)
    claimed = store.claim_remote_commands(
        "cluster-a",
        "old",
        limit=1,
        lease_seconds=60,
        executor_protocol_version=protocol,
    )
    assert [item.command_id for item in claimed] == ["b-ordinary"]
    assert (
        store.get_remote_command(guarded.command_id).status
        is RemoteCommandStatus.PENDING
    )
    capable = store.claim_remote_commands(
        "cluster-a", "capable", limit=1, lease_seconds=60, executor_protocol_version=4
    )
    assert [item.command_id for item in capable] == ["a-inhibited"]


@pytest.mark.parametrize("batched", [False, True])
def test_legacy_default_is_safe_without_changing_ordinary_batch_acceptance(
    store, batched: bool
) -> None:
    store.ensure_remote_command(command("a-guarded", True, batched=batched))
    store.ensure_remote_command(command("b-ordinary", batched=batched))
    claimed = store.claim_remote_commands(
        "cluster-a", "legacy-default", limit=10, lease_seconds=60
    )
    assert [item.command_id for item in claimed] == ["b-ordinary"]


def test_batching_still_requires_its_original_separate_acceptance(store) -> None:
    store.ensure_remote_command(command("batch", True, batched=True))
    assert (
        store.claim_remote_commands(
            "cluster-a",
            "four-no-batch",
            limit=5,
            lease_seconds=60,
            executor_protocol_version=4,
            accept_batched_steps=False,
        )
        == []
    )
    assert REMOTE_STEP_BATCHING_PROTOCOL_VERSION == 3
    assert CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION == 4


@pytest.mark.parametrize(
    "status", [RemoteCommandStatus.WAITING, RemoteCommandStatus.LEASED]
)
def test_old_executor_cannot_reclaim_waiting_or_expired_inhibited_work(
    store, status
) -> None:
    guarded = command(
        "reclaim",
        True,
        status=status,
        lease_owner="previous" if status is RemoteCommandStatus.LEASED else None,
        lease_token="owned-test-lease"
        if status is RemoteCommandStatus.LEASED
        else None,
        lease_expires_at=NOW - timedelta(seconds=1)
        if status is RemoteCommandStatus.LEASED
        else None,
    )
    store.ensure_remote_command(guarded)
    assert (
        store.claim_remote_commands(
            "cluster-a", "old", limit=1, lease_seconds=60, executor_protocol_version=3
        )
        == []
    )
    claimed = store.claim_remote_commands(
        "cluster-a", "current", limit=1, lease_seconds=60, executor_protocol_version=4
    )
    assert [item.command_id for item in claimed] == ["reclaim"]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "before,after",
    [(True, ABSENT), (ABSENT, True), (True, False), (True, 1), (None, ABSENT)],
)
def test_command_identity_cannot_gain_or_lose_inhibition_on_replay(
    store, batched, before, after
) -> None:
    store.ensure_remote_command(command("immutable", before, batched=batched))
    with pytest.raises(ValueError, match="identity conflict"):
        store.ensure_remote_command(command("immutable", after, batched=batched))


@pytest.mark.parametrize("protocol", [0, True, None, "4"])
def test_unknown_advertised_protocol_cannot_interpret_inhibition(protocol: Any) -> None:
    assert command_protocol_eligible(command("guarded", True), protocol) is False
    assert command_protocol_eligible(command("ordinary"), protocol) is True


def test_unchanged_inhibition_replay_is_idempotent(store) -> None:
    original = store.ensure_remote_command(command("same", True, batched=True))
    again = store.ensure_remote_command(command("same", True, batched=True))
    assert again.command_id == original.command_id
    assert again.batched_steps[0].step.parameters == {"activation_forbidden": True}
