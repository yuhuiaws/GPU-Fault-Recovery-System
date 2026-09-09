"""Store half of the compound remote command (性能 C).

Progress is a lease-fenced, atomic read-modify-write of one command row; the
covering lookup finds the compound command a step belongs to; the claim keeps a
compound command away from an executor whose protocol predates it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.regional import (
    RegionalRemoteWorkflowAdapter,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY
from gpu_fault.store import NotFoundError, SqliteStore, WorkflowLeaseError
from tests._builders import build_store, copy_model
from tests.regional._batching_support import (
    ACTIVE_POLICY,
    ALL_OWNERS,
    chain_state,
    reset_chain,
    step_context,
)
from tests.regional._regional_support import TOKEN_A, registration


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    instance = (
        build_store()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "progress.db"))
    )
    instance.save_regional_cluster(registration("cluster-a", TOKEN_A))
    try:
        yield instance
    finally:
        if request.param == "sqlite":
            instance.close()


def _compound_and_single(store):
    """One compound command (steps 1..4) and one single command (step 0)."""

    adapter = RegionalRemoteWorkflowAdapter(
        store, owners=ALL_OWNERS, step_batching=ACTIVE_POLICY
    )
    incident, workflow = chain_state(store, reset_chain())
    single = adapter.execute(step_context(workflow, incident, 0))
    compound = adapter.execute(step_context(workflow, incident, 1))
    return (
        store.get_remote_command(compound.details["remote_command_id"]),
        store.get_remote_command(single.details["remote_command_id"]),
        workflow,
    )


def _claim(store, *, accept_batched_steps=True, limit=5):
    return store.claim_remote_commands(
        "cluster-a",
        "executor-a",
        limit=limit,
        lease_seconds=60,
        execution_owners=ALL_OWNERS,
        accept_batched_steps=accept_batched_steps,
    )


def _entry(status: str, **details) -> dict:
    return {"status": status, "details": details, "error": None}


def test_progress_is_lease_fenced(store) -> None:
    compound, _, _ = _compound_and_single(store)
    with pytest.raises(WorkflowLeaseError):
        store.record_remote_command_progress(
            "cluster-a",
            compound.command_id,
            "executor-a",
            "nobody-issued-this",
            batched_results={"1": _entry("SUCCEEDED")},
        )
    leased = next(
        item for item in _claim(store) if item.command_id == compound.command_id
    )

    with pytest.raises(WorkflowLeaseError, match="lease is stale"):
        store.record_remote_command_progress(
            "cluster-a",
            compound.command_id,
            "executor-a",
            "wrong-token",
            batched_results={"1": _entry("SUCCEEDED")},
        )
    with pytest.raises(WorkflowLeaseError):
        store.record_remote_command_progress(
            "cluster-a",
            compound.command_id,
            "executor-b",
            leased.lease_token,
            batched_results={"1": _entry("SUCCEEDED")},
        )
    with pytest.raises(NotFoundError):
        store.record_remote_command_progress(
            "cluster-b",
            compound.command_id,
            "executor-a",
            leased.lease_token,
            batched_results={"1": _entry("SUCCEEDED")},
        )
    assert store.get_remote_command(compound.command_id).result_details == {}, (
        "a refused progress post writes nothing"
    )


def test_progress_merges_per_step_entries_and_the_terminal_result_keeps_them(
    store,
) -> None:
    compound, _, _ = _compound_and_single(store)
    leased = next(
        item for item in _claim(store) if item.command_id == compound.command_id
    )

    first = store.record_remote_command_progress(
        "cluster-a",
        compound.command_id,
        "executor-a",
        leased.lease_token,
        batched_results={"1": _entry("SUCCEEDED", agent_generations={"node-a": 5})},
    )
    second = store.record_remote_command_progress(
        "cluster-a",
        compound.command_id,
        "executor-a",
        leased.lease_token,
        batched_results={"2": _entry("WAITING", gpu_client_quiesce_attempt=1)},
    )

    assert first.status is RemoteCommandStatus.LEASED
    assert first.lease_token == leased.lease_token, "progress does not touch the lease"
    assert first.lease_expires_at == leased.lease_expires_at
    assert set(second.result_details[BATCHED_RESULTS_KEY]) == {"1", "2"}
    assert second.result_details[BATCHED_RESULTS_KEY]["1"]["details"] == {
        "agent_generations": {"node-a": 5}
    }
    stored = store.get_remote_command(compound.command_id)
    assert stored.result_details == second.result_details

    final = {**stored.result_details[BATCHED_RESULTS_KEY], "2": _entry("SUCCEEDED")}
    done = store.complete_remote_command(
        "cluster-a",
        compound.command_id,
        RemoteCommandResult(
            lease_token=leased.lease_token,
            status=RemoteCommandStatus.SUCCEEDED,
            details={BATCHED_RESULTS_KEY: final, "batched_step_indexes": [1, 2, 3, 4]},
        ),
    )

    assert done.status is RemoteCommandStatus.SUCCEEDED
    assert done.result_details[BATCHED_RESULTS_KEY]["2"]["status"] == "SUCCEEDED"
    with pytest.raises(WorkflowLeaseError):
        store.record_remote_command_progress(
            "cluster-a",
            compound.command_id,
            "executor-a",
            leased.lease_token,
            batched_results={"3": _entry("SUCCEEDED")},
        )


def test_the_covering_lookup_finds_the_compound_command_for_each_covered_step(
    store,
) -> None:
    compound, single, workflow = _compound_and_single(store)

    for index in (1, 2, 3, 4):
        found = store.find_remote_command_covering_step(
            workflow.request_id, index, "official", fencing_token=3
        )
        assert found is not None and found.command_id == compound.command_id, index
    assert (
        store.find_remote_command_covering_step(
            workflow.request_id, 0, "official", fencing_token=3
        )
        is None
    ), "a single command is not a covering command"
    assert (
        store.find_remote_command_covering_step(
            workflow.request_id, 5, "official", fencing_token=3
        )
        is None
    )
    assert (
        store.find_remote_command_covering_step(
            workflow.request_id, 2, "official", fencing_token=4
        )
        is None
    ), "another generation's command does not cover this one's step"
    assert (
        store.find_remote_command_covering_step(
            workflow.request_id, 2, "safety", fencing_token=3
        )
        is None
    ), "the safety step space shares indexes but not commands"


def test_the_covering_lookup_prefers_an_open_command_over_a_terminal_one(store) -> None:
    compound, _, workflow = _compound_and_single(store)
    leased = next(
        item for item in _claim(store) if item.command_id == compound.command_id
    )
    store.complete_remote_command(
        "cluster-a",
        compound.command_id,
        RemoteCommandResult(
            lease_token=leased.lease_token,
            status=RemoteCommandStatus.FAILED,
            error="refused",
        ),
    )
    later = copy_model(
        compound,
        command_id="remote-" + "f" * 24,
        status=RemoteCommandStatus.PENDING,
        lease_owner=None,
        lease_token=None,
        lease_expires_at=None,
        error=None,
        result_details={},
        created_at=compound.created_at - timedelta(minutes=5),
    )
    store.ensure_remote_command(later)

    found = store.find_remote_command_covering_step(
        workflow.request_id, 3, "official", fencing_token=3
    )

    assert found is not None and found.command_id == later.command_id


def test_an_executor_that_predates_batched_steps_never_claims_a_compound_command(
    store,
) -> None:
    compound, single, _ = _compound_and_single(store)

    old = _claim(store, accept_batched_steps=False)
    new = _claim(store)

    assert [item.command_id for item in old] == [single.command_id]
    assert [item.command_id for item in new] == [compound.command_id]


def test_a_compound_row_round_trips_and_a_single_row_omits_the_field(store) -> None:
    compound, single, _ = _compound_and_single(store)

    assert "batched_steps" not in single.model_dump(mode="json")
    reloaded = store.get_remote_command(compound.command_id)
    assert [item.step_index for item in reloaded.batched_steps] == [2, 3, 4]
    assert reloaded.batched_steps[1].step.operation is WorkflowOperation.RESET_GPU
    assert reloaded.created_at <= datetime.now(timezone.utc)
