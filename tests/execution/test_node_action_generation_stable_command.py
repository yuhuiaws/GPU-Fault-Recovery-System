"""One driver remediation per step per node, whatever the agent's incarnation.

The node-action ``command_id`` is ``<idempotency_key>/<node_id>[/agent-N]``.
The ``agent-N`` suffix is right for actions the fleet *wants* re-run on a
fresh agent (a diagnostic bundle, a health snapshot): the restarted agent has
no ledger row for the new id and simply runs again. For a driver or firmware
install it is the defect: the agent that restarted mid-install closed the
attempt as INTERRUPTED in its ledger, the control plane read the new
generation, derived ``agent-N+1`` and submitted what the agent had to treat as
a brand-new command -- a second install on the same node.

These tests pin the cure: for the long-running mutating node actions the id
excludes the generation, so the retry after a generation change resolves to
the SAME command_id and the agent answers from its ledger (INTERRUPTED here)
instead of executing again. The live generation still rides in the command
body, so the agent's own generation check is untouched.
"""

from __future__ import annotations

import pytest
from tests._builders import build_store, copy_model, node_action_result

from tests.execution._support import (
    NodeActionPending,
    NodeActionStatus,
    NodeActionWorkflowAdapter,
    StubFleetRegistry,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepContext,
    WorkflowStepStatus,
    workflow_state,
)

GENERATION_STABLE = [
    WorkflowOperation.REMEDIATE_DRIVER,
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    WorkflowOperation.REMEDIATE_EFA_DRIVER,
]
KEY = "workflow/r4"


def _execute(adapter, incident, workflow, step):
    return adapter.execute(
        WorkflowStepContext(
            workflow=copy_model(workflow, official_steps=[step]),
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key=KEY,
        )
    )


def _restarting_agent(operation: WorkflowOperation):
    """A sender that accepts once, then answers every later send from a ledger.

    Attempt 1 is accepted (PENDING). Everything after that is what a restarted
    agent says for the id it already holds: the INTERRUPTED marker its startup
    sweep wrote over the IN_PROGRESS row. An id it has never seen would be
    accepted again -- that is the second execution.
    """

    seen: list = []

    def sender(_endpoint, envelope):
        command = envelope.command
        seen.append(command)
        if len(seen) == 1:
            raise NodeActionPending(command.command_id, {})
        if command.command_id != seen[0].command_id:
            raise NodeActionPending(command.command_id, {})
        return node_action_result(
            command.command_id,
            operation,
            NodeActionStatus.INTERRUPTED,
            error=(
                "agent restarted while the action was in progress; "
                "manual confirmation is required"
            ),
            retryable=False,
            attempt=1,
        )

    return sender, seen


@pytest.mark.parametrize("operation", GENERATION_STABLE, ids=lambda op: op.value)
def test_generation_change_replays_the_same_command_instead_of_reexecuting(
    operation: WorkflowOperation,
) -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [operation])
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    sender, seen = _restarting_agent(operation)
    adapter = NodeActionWorkflowAdapter({}, "s" * 32, sender=sender, registry=registry)
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )

    accepted = _execute(adapter, incident, workflow, step)
    assert accepted.status is WorkflowStepStatus.WAITING
    assert accepted.details["node_action_state"] == "PENDING"
    assert accepted.details["node_action_command_id"] == f"{KEY}/node-a", (
        "the id must not carry the agent generation"
    )
    assert seen[0].agent_generation == 7

    # The agent restarts mid-install: generation 7 -> 8.
    assert registry.restart_agent() == 8
    retry = _execute(adapter, incident, workflow, step)

    assert [command.command_id for command in seen] == [f"{KEY}/node-a"] * 2, (
        "the retry formed a new command_id; the restarted agent would run the "
        "remediation a second time"
    )
    # The body still names the live generation: the agent's own STALE /
    # UNKNOWN generation check keeps working on the new id-less contract.
    assert seen[1].agent_generation == 8
    # And the answer is the ledger's, not a fresh acceptance.
    assert retry.status is WorkflowStepStatus.FAILED
    assert retry.details["node_action_interrupted"] is True
    assert retry.details["manual_confirmation_required"] is True
    assert retry.details["failed_nodes"] == ["node-a"]


def test_diagnostic_actions_keep_the_generation_suffix() -> None:
    """The suffix stays for actions a fresh agent should simply run again."""

    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE]
    )
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    seen: list = []

    def sender(_endpoint, envelope):
        seen.append(envelope.command)
        raise NodeActionPending(envelope.command.command_id, {})

    adapter = NodeActionWorkflowAdapter({}, "s" * 32, sender=sender, registry=registry)
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )
    _execute(adapter, incident, workflow, step)
    assert registry.restart_agent() == 8
    _execute(adapter, incident, workflow, step)

    assert [command.command_id for command in seen] == [
        f"{KEY}/node-a/agent-7",
        f"{KEY}/node-a/agent-8",
    ]


@pytest.mark.parametrize("operation", GENERATION_STABLE, ids=lambda op: op.value)
def test_a_result_finished_before_the_restart_is_still_the_answer(
    operation: WorkflowOperation,
) -> None:
    """Exactly-once cuts both ways: an install that finished is not failed.

    The agent wrote SUCCEEDED, then restarted before the control plane
    polled. Under the generation-suffixed id that row was unreachable and
    the step re-ran the install; under the stable id the poll reads it.
    """

    store = build_store()
    incident, workflow = workflow_state(store, [operation])
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    seen: list = []

    def sender(_endpoint, envelope):
        command = envelope.command
        seen.append(command)
        if len(seen) == 1:
            raise NodeActionPending(command.command_id, {})
        if command.command_id != seen[0].command_id:
            raise NodeActionPending(command.command_id, {})
        return node_action_result(
            command.command_id, operation, NodeActionStatus.SUCCEEDED, attempt=1
        )

    adapter = NodeActionWorkflowAdapter({}, "s" * 32, sender=sender, registry=registry)
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )
    accepted = _execute(adapter, incident, workflow, step)
    assert accepted.status is WorkflowStepStatus.WAITING

    assert registry.restart_agent() == 8
    completed = _execute(adapter, incident, workflow, step)

    assert completed.status is WorkflowStepStatus.SUCCEEDED, completed.error
    assert {command.command_id for command in seen} == {f"{KEY}/node-a"}
