"""The real agent replays a generation-stable command across its own restart.

R4 dropped the ``agent-N`` suffix from the driver, firmware and EFA remediation
command_id so that the poll after an agent restart lands on the row the restart
closed instead of naming a command the agent has never seen. The control-plane
side is pinned with an id-equality fake sender; this file pins the agent side
with the real ``NodeActionExecutor`` and ``NodeActionLedger``: the row written
by the generation-7 incarnation answers the generation-8 retry, and the install
runner is never called a second time.

Public surface only: ``execute``, ``validate_submission``, the ledger's
``mark_in_progress`` / ``attempt_history``.
"""

from __future__ import annotations

from tests._builders import copy_model, node_action_result

from ._support import (
    CompletedProcess,
    NodeActionStatus,
    Quiesced,
    WorkflowOperation,
    command,
    envelope,
    hashlib,
    no_device_clients,
    node_action_executor,
    pytest,
)

COMMAND_ID = "workflow/3/REMEDIATE_DRIVER/node-a"
PARAMETERS = {"target_driver_branch": 575}


class InstallRunner:
    """Records every subprocess; an install that runs is visible here."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, argv, **_):
        self.commands.append(list(argv))
        stdout = "575.86.01\n" if "--query-gpu=driver_version" in argv else ""
        return CompletedProcess(argv, 0, stdout=stdout, stderr="")


def _driver_agent(tmp_path, *, agent_generation: int):
    executable = tmp_path / "driver-remediate"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    runner = InstallRunner()
    agent = node_action_executor(
        tmp_path,
        "driver-replay.db",
        allowed_operations={WorkflowOperation.REMEDIATE_DRIVER},
        reset_enabled=True,
        service_quiesce_enabled=True,
        driver_remediation_enabled=True,
        driver_remediation_command=(str(executable), "--branch", "{target}"),
        driver_remediation_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        target_driver_branch=575,
        quiesce_manager=Quiesced(),
        runner=runner,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        agent_generation=agent_generation,
        sleep=lambda _: None,
    )
    return agent, runner


def _install(agent_generation: int):
    return copy_model(
        command(
            WorkflowOperation.REMEDIATE_DRIVER,
            command_id=COMMAND_ID,
            parameters=dict(PARAMETERS),
        ),
        agent_generation=agent_generation,
    )


def test_a_row_dispatched_by_the_previous_generation_answers_the_retry(
    tmp_path,
) -> None:
    """Generation 7 started the install and died; generation 8 is asked again.

    The retry carries the same command_id (no suffix) and the new generation
    in its body. The agent must answer from its ledger -- INTERRUPTED,
    manual confirmation -- and never dispatch the install runner.
    """

    agent, runner = _driver_agent(tmp_path, agent_generation=8)
    # What the generation-7 incarnation left behind: attempt 1 dispatched,
    # no result written, process gone.
    agent.ledger.mark_in_progress(_install(7), 1)

    result = agent.execute(envelope(_install(8)))

    assert runner.commands == [], (
        "the generation-8 agent ran the install a second time for the row its "
        "predecessor left IN_PROGRESS"
    )
    assert result.status is NodeActionStatus.INTERRUPTED
    assert result.attempt == 1
    assert result.retryable is False
    history = agent.ledger.attempt_history(COMMAND_ID)
    assert [row["attempt"] for row in history] == [1], (
        "a second attempt row means the retry was treated as a new command"
    )
    assert history[0]["state"] == "INTERRUPTED"


def test_a_stale_generation_body_is_refused_before_the_ledger_is_consulted(
    tmp_path,
) -> None:
    """The body's generation is checked against the live agent, not the row.

    A generation-7 envelope reaching the generation-8 agent is a command for
    a different incarnation: refused as such (the control plane rebuilds a
    fresh envelope), with the runner untouched and no ledger row written.
    """

    agent, runner = _driver_agent(tmp_path, agent_generation=8)

    with pytest.raises(ValueError, match="different agent generation"):
        agent.validate_submission(envelope(_install(7)))
    with pytest.raises(ValueError, match="different agent generation"):
        agent.execute(envelope(_install(7)))

    assert runner.commands == []
    assert agent.ledger.attempt_history(COMMAND_ID) == []


def test_the_same_id_with_a_different_body_is_a_reuse_not_a_replay(tmp_path) -> None:
    """What the stable id gives up: a rebind that changes the body collides."""

    agent, runner = _driver_agent(tmp_path, agent_generation=8)
    agent.ledger.mark_in_progress(_install(7), 1)
    rebound = copy_model(_install(8), gpu_uuids=["GPU-a", "GPU-b"])

    with pytest.raises(ValueError, match="command_id reused"):
        agent.execute(envelope(rebound))

    assert runner.commands == []


def test_a_fresh_attempt_records_the_live_generation_on_its_row(tmp_path) -> None:
    """The row, not the command_id, now says which incarnation ran the install."""

    agent, runner = _driver_agent(tmp_path, agent_generation=8)

    result = agent.execute(envelope(_install(8)))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    installs = [argv for argv in runner.commands if argv[1:] == ["--branch", "575"]]
    assert len(installs) == 1, runner.commands
    (row,) = agent.ledger.attempt_history(COMMAND_ID)
    assert row["agent_generation"] == 8, row


def test_a_row_taken_by_another_generation_still_replays(tmp_path) -> None:
    """The generation is an audit column, not part of the command body.

    Generation 7 ran the install and failed terminally; generation 8 is asked
    the same command. The stored result answers, the runner is never called,
    and the row keeps saying 7 -- a different generation is not a reused id.
    """

    agent, runner = _driver_agent(tmp_path, agent_generation=8)
    agent.ledger.mark_in_progress(_install(7), 1, agent_generation=7)
    agent.ledger.save(
        node_action_result(
            COMMAND_ID,
            WorkflowOperation.REMEDIATE_DRIVER,
            NodeActionStatus.FAILED,
            error="RuntimeError: driver branch verification failed: 570",
            retryable=False,
            attempt=1,
        )
    )

    result = agent.execute(envelope(_install(8)))

    assert runner.commands == [], (
        "a generation change must not open a second attempt of the install"
    )
    assert result.status is NodeActionStatus.FAILED
    assert result.attempt == 1
    (row,) = agent.ledger.attempt_history(COMMAND_ID)
    assert row["agent_generation"] == 7, (
        f"the row must keep the generation that took the attempt: {row}"
    )
