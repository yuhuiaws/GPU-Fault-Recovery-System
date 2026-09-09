"""One compound remote command for a contiguous run of node-side steps (性能 C).

The control-plane half: which dispatch mints a compound command, which never
does, and how a step covered by an existing compound command reads its share
of the command instead of minting one of its own.
"""

from __future__ import annotations

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.regional import (
    RegionalRemoteWorkflowAdapter,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY
from gpu_fault.remote_step_batching import (
    REMOTE_STEP_BATCHING_ENV,
    RemoteStepBatchingPolicy,
)
from gpu_fault.store import SqliteStore
from tests._builders import build_store, copy_model, workflow_step
from tests.regional._batching_support import (
    ACTIVE_POLICY,
    ALL_OWNERS,
    NODE_OWNER,
    chain_state,
    quiesce_details,
    reset_chain,
    step_context,
)
from tests.regional._regional_support import TOKEN_A, registration


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    instance = (
        build_store()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "batching.db"))
    )
    instance.save_regional_cluster(registration("cluster-a", TOKEN_A))
    try:
        yield instance
    finally:
        if request.param == "sqlite":
            instance.close()


def _adapter(store, policy=ACTIVE_POLICY) -> RegionalRemoteWorkflowAdapter:
    return RegionalRemoteWorkflowAdapter(store, owners=ALL_OWNERS, step_batching=policy)


def _claim(store):
    claimed = store.claim_remote_commands(
        "cluster-a",
        "executor-a",
        limit=5,
        lease_seconds=60,
        execution_owners=ALL_OWNERS,
    )
    assert len(claimed) == 1, claimed
    return claimed[0]


def _entry(status: str, details=None, error=None) -> dict:
    return {"status": status, "details": details or {}, "error": error}


def test_the_reset_chain_yields_one_compound_command_for_the_node_side_run(
    store,
) -> None:
    adapter = _adapter(store)
    incident, workflow = chain_state(store, reset_chain())

    cordon = adapter.execute(step_context(workflow, incident, 0))
    head = adapter.execute(step_context(workflow, incident, 1))
    again = adapter.execute(step_context(workflow, incident, 1))

    commands = {item.step_index: item for item in store.list_remote_commands()}
    assert sorted(commands) == [0, 1], "the cordon and one compound command"
    assert commands[0].batched_steps == [], "MARK_UNSCHEDULABLE is not node-side"
    compound = commands[1]
    assert compound.step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
    assert [item.step_index for item in compound.batched_steps] == [2, 3, 4]
    assert [item.step.operation for item in compound.batched_steps] == [
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    ]
    assert [item.idempotency_key for item in compound.batched_steps] == [
        "workflow-a/2/VERIFY_NO_GPU_CLIENTS",
        "workflow-a/3/RESET_GPU",
        "workflow-a/4/RESTORE_GPU_SERVICES",
    ]
    assert compound.covered_step_indexes == (1, 2, 3, 4)
    assert cordon.status is head.status is again.status is WorkflowStepStatus.WAITING
    assert head.details["remote_command_id"] == compound.command_id
    assert head.details["batched_step_indexes"] == [1, 2, 3, 4]
    assert again.details["remote_command_id"] == compound.command_id, (
        "a redispatch of the head reads the same command"
    )
    assert (adapter.batched_commands_total, adapter.batched_steps_total) == (1, 3)


def test_validate_and_the_scheduler_release_are_never_the_head_of_a_batch(
    store,
) -> None:
    adapter = _adapter(store)
    incident, workflow = chain_state(
        store, reset_chain(), completed_step_indexes=[0, 1, 2, 3, 4]
    )

    adapter.execute(step_context(workflow, incident, 5))
    adapter.execute(step_context(workflow, incident, 6))

    assert [item.batched_steps for item in store.list_remote_commands()] == [[], []]
    assert adapter.batched_commands_total == 0


def test_a_two_node_fabric_reset_chain_is_never_batched(store) -> None:
    adapter = _adapter(store)
    nodes = ["node-a", "node-b"]
    steps = [
        workflow_step(
            WorkflowOperation.QUIESCE_GPU_SERVICES, NODE_OWNER, node_ids=nodes
        ),
        workflow_step(
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES, NODE_OWNER, node_ids=nodes
        ),
        workflow_step(
            WorkflowOperation.RESTORE_GPU_SERVICES, NODE_OWNER, node_ids=nodes
        ),
    ]
    incident, workflow = chain_state(store, steps)

    adapter.execute(step_context(workflow, incident, 0))

    (command,) = store.list_remote_commands()
    assert command.batched_steps == []


def test_a_barrier_operation_ends_the_run_even_on_one_node(store) -> None:
    adapter = _adapter(store)
    steps = [
        workflow_step(WorkflowOperation.QUIESCE_GPU_SERVICES, NODE_OWNER),
        workflow_step(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, NODE_OWNER),
        workflow_step(WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES, NODE_OWNER),
        workflow_step(WorkflowOperation.RESTORE_GPU_SERVICES, NODE_OWNER),
    ]
    incident, workflow = chain_state(store, steps)

    adapter.execute(step_context(workflow, incident, 0))

    (command,) = store.list_remote_commands()
    assert [item.step_index for item in command.batched_steps] == [1]


@pytest.mark.parametrize(
    "workflow_values",
    [
        {"dag_enabled": True},
        {"completed_step_indexes": [0, 3]},
        {"superseded_step_indexes": [3]},
    ],
    ids=["dag", "resolved-step-in-the-run", "superseded-step-in-the-run"],
)
def test_a_dag_or_a_resolved_step_bounds_the_run(store, workflow_values) -> None:
    adapter = _adapter(store)
    incident, workflow = chain_state(store, reset_chain(), **workflow_values)

    adapter.execute(step_context(workflow, incident, 1))

    (command,) = store.list_remote_commands()
    expected = [] if workflow.dag_enabled else [2]
    assert [item.step_index for item in command.batched_steps] == expected


def test_a_branch_or_dependency_on_any_step_of_the_run_ends_it(store) -> None:
    adapter = _adapter(store)
    steps = reset_chain()
    steps[3] = copy_model(steps[3], depends_on_step_indexes=[2])
    incident, workflow = chain_state(store, steps)

    adapter.execute(step_context(workflow, incident, 1))

    (command,) = store.list_remote_commands()
    assert [item.step_index for item in command.batched_steps] == [2]


@pytest.mark.parametrize(
    "policy",
    [
        None,
        RemoteStepBatchingPolicy.disabled(),
        RemoteStepBatchingPolicy(enabled=True, minimum_executor_protocol_version=2),
    ],
    ids=["no-policy", "flag-off", "an-older-executor-is-still-admitted"],
)
def test_flag_off_or_an_older_executor_keeps_one_command_per_step(
    store, policy
) -> None:
    adapter = _adapter(store, policy)
    incident, workflow = chain_state(store, reset_chain())

    adapter.execute(step_context(workflow, incident, 1))

    (command,) = store.list_remote_commands()
    assert command.batched_steps == []
    assert adapter.batched_commands_total == 0


def test_the_policy_reads_the_flag_and_the_compatibility_pins() -> None:
    narrowed = {"GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS": ""}

    assert RemoteStepBatchingPolicy.from_environment(narrowed).active is True
    assert (
        RemoteStepBatchingPolicy.from_environment(
            {**narrowed, REMOTE_STEP_BATCHING_ENV: "false"}
        ).active
        is False
    )
    # The transition deploy pins required=<previous>, compatible=<current>
    # (deploy.sh); an older executor may still claim, so no compound command.
    assert (
        RemoteStepBatchingPolicy.from_environment(
            {
                "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": "2",
                "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS": "3",
            }
        ).active
        is False
    )
    # The library default still lists the legacy version as compatible.
    assert RemoteStepBatchingPolicy.from_environment({}).active is False


def test_covered_steps_read_their_share_of_the_compound_command(store) -> None:
    adapter = _adapter(store)
    incident, workflow = chain_state(store, reset_chain())
    adapter.execute(step_context(workflow, incident, 1))
    command = _claim(store)
    store.record_remote_command_progress(
        "cluster-a",
        command.command_id,
        "executor-a",
        command.lease_token,
        batched_results={"1": _entry("SUCCEEDED", quiesce_details())},
    )

    quiesce = adapter.execute(step_context(workflow, incident, 1))
    verify = adapter.execute(step_context(workflow, incident, 2))

    assert quiesce.status is WorkflowStepStatus.SUCCEEDED
    assert quiesce.details == quiesce_details(), "the step's own details, verbatim"
    assert quiesce.adapter_operation_id == f"remote/{command.command_id}"
    assert verify.status is WorkflowStepStatus.WAITING
    assert verify.details["remote_command_id"] == command.command_id
    assert verify.details["remote_status"] == "LEASED"
    assert verify.details["batched_step_index"] == 2
    assert verify.details["mutation_submitted_by_control_plane"] is False
    assert len(store.list_remote_commands()) == 1, "no second command was minted"
    assert adapter.open_sibling_holds_total == 0


def test_a_failed_step_fails_and_the_never_reached_compensation_mints_afresh(
    store,
) -> None:
    """RESET fails inside the compound command: RESET reads the failure, and the
    RESTORE the chain dispatches as compensation gets a command of its own --
    it never started on the node, and answering "earlier batched step failed"
    would have left the node quiesced until the agent's fail-safe timer."""

    adapter = _adapter(store)
    incident, workflow = chain_state(store, reset_chain())
    adapter.execute(step_context(workflow, incident, 1))
    command = _claim(store)
    results = {
        "1": _entry("SUCCEEDED", quiesce_details()),
        "2": _entry("SUCCEEDED", {"gpu_client_quiesce_attempt": 1}),
        "3": {
            **_entry("FAILED", {"node_results": {}}, "reset refused"),
            "status_source": "node-refused",
        },
    }
    store.complete_remote_command(
        "cluster-a",
        command.command_id,
        RemoteCommandResult(
            lease_token=command.lease_token,
            status=RemoteCommandStatus.FAILED,
            error="reset refused",
            details={BATCHED_RESULTS_KEY: results, "batched_step_index": 3},
        ),
    )

    verify = adapter.execute(step_context(workflow, incident, 2))
    reset = adapter.execute(step_context(workflow, incident, 3))
    restore = adapter.execute(step_context(workflow, incident, 4))

    assert verify.status is WorkflowStepStatus.SUCCEEDED
    assert reset.status is WorkflowStepStatus.FAILED
    assert reset.error == "reset refused"
    assert reset.details["remote_status_source"] == "node-refused"
    assert restore.status is WorkflowStepStatus.WAITING
    fresh = store.get_remote_command(restore.details["remote_command_id"])
    assert fresh.command_id != command.command_id
    assert fresh.step.operation is WorkflowOperation.RESTORE_GPU_SERVICES
    assert fresh.batched_steps == [], "nothing batchable follows RESTORE"
    assert len(store.list_remote_commands()) == 2


def test_a_compound_command_cancelled_before_it_ran_fails_its_head_only(store) -> None:
    adapter = _adapter(store)
    incident, workflow = chain_state(store, reset_chain())
    head = adapter.execute(step_context(workflow, incident, 1))
    assert store.cancel_remote_command(
        head.details["remote_command_id"], reason="cancelled by a successor"
    ), "the pending compound command could not be cancelled"

    quiesce = adapter.execute(step_context(workflow, incident, 1))
    verify = adapter.execute(step_context(workflow, incident, 2))

    assert quiesce.status is WorkflowStepStatus.FAILED
    assert quiesce.error == "cancelled by a successor"
    assert quiesce.details["remote_status_source"] == "workflow-preempted"
    assert verify.status is WorkflowStepStatus.WAITING, (
        "VERIFY never reached the node, so a later dispatch may mint afresh"
    )
    assert verify.details["remote_command_id"] != head.details["remote_command_id"]


def test_a_waiting_compound_command_reports_waiting_for_its_current_step(store) -> None:
    adapter = _adapter(store)
    incident, workflow = chain_state(store, reset_chain())
    adapter.execute(step_context(workflow, incident, 1))
    command = _claim(store)
    results = {
        "1": _entry("SUCCEEDED", quiesce_details()),
        "2": _entry(
            "WAITING", {"gpu_client_quiesce_attempt": 1, "waiting_nodes": ["node-a"]}
        ),
    }
    store.complete_remote_command(
        "cluster-a",
        command.command_id,
        RemoteCommandResult(
            lease_token=command.lease_token,
            status=RemoteCommandStatus.WAITING,
            details={BATCHED_RESULTS_KEY: results},
        ),
    )

    verify = adapter.execute(step_context(workflow, incident, 2))

    assert verify.status is WorkflowStepStatus.WAITING
    assert verify.details["remote_status"] == "WAITING"
    assert verify.details["remote_command_id"] == command.command_id
    assert len(store.list_remote_commands()) == 1
