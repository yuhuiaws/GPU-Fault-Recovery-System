from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from xml.sax.saxutils import escape

import pytest

from gpu_fault.node_agent import GpuServiceQuiesceManager
from gpu_fault.node_agent.late_ownership import OWNERSHIP_PROTOCOL, OwnershipRefused
from gpu_fault.node_agent.operations.drain_p2p import parse_recovery_state
from gpu_fault.orchestration.escalation import next_rung

from ._support import (
    NodeActionStatus,
    ServiceRunner,
    WorkflowOperation,
    command,
    envelope,
    no_device_clients,
    node_action_executor,
)

RECOVER = WorkflowOperation.RECOVER_DRAIN_P2P


def xml_report(actions, persistence="Disabled"):
    return (
        "<nvidia_smi_log>"
        + "".join(
            f"<gpu><uuid>{escape(uuid)}</uuid>"
            f"<persistence_mode>{persistence}</persistence_mode>"
            f"<gpu_recovery_action>{escape(action)}</gpu_recovery_action></gpu>"
            for uuid, action in actions.items()
        )
        + "</nvidia_smi_log>"
    )


class RecoveryRunner(ServiceRunner):
    def __init__(
        self, actions, *, persistence="Disabled", query_error=None, reset_error=None
    ):
        super().__init__(active=set())
        self.actions = actions
        self.persistence = persistence
        self.query_error = query_error
        self.reset_error = reset_error

    def __call__(self, argv, **kwargs):
        if argv[:4] == ["nvidia-smi", "-q", "-x", "-i"]:
            self.commands.append(argv)
            if self.query_error:
                raise self.query_error
            return CompletedProcess(
                argv, 0, xml_report(self.actions, self.persistence), ""
            )
        if argv[0] == "nvidia-smi":
            self.commands.append(argv)
            if "--gpu-reset" in argv and self.reset_error:
                raise self.reset_error
            stdout = ""
            if "--query-gpu=uuid,index" in argv:
                stdout = "GPU-a, 0\nGPU-b, 1\n"
            if "--query-gpu=uuid" in argv:
                stdout = "GPU-a\nGPU-b\n"
            return CompletedProcess(argv, 0, stdout, "")
        return super().__call__(argv, **kwargs)


def prepared_agent(
    tmp_path: Path, runner, *, quiesced=True, clients="", require_ownership=False
):
    proc = tmp_path / "proc"
    proc.mkdir()
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=("kubelet",),
        processes=(),
        containers=(),
        device_sweep_processes=(),
        proc_root=str(proc),
        settle_seconds=0,
        restore_settle_seconds=0,
        runner=runner,
        sleeper=lambda _: None,
        boot_id_reader=lambda: "test-boot",
    )
    if quiesced:
        manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")
    agent = node_action_executor(
        tmp_path,
        "recovery.db",
        allowed_operations={RECOVER},
        reset_enabled=True,
        service_quiesce_enabled=True,
        require_final_ownership=require_ownership,
        agent_generation=1,
        quiesce_manager=manager,
        proc_root=str(proc),
        runner=runner,
        device_client_finder=(
            (lambda _: [{"pid": "123", "process_name": "python", "devices": clients}])
            if clients
            else no_device_clients
        ),
        gpu_device_path_finder=lambda: {
            "GPU-a": "/dev/nvidia0",
            "GPU-b": "/dev/nvidia1",
        },
        sleep=lambda _: None,
    )
    return agent


@pytest.mark.parametrize("action", ["Reset", "Drain P2P", "Drain and Reset"])
def test_drained_recovery_resets_only_driver_requested_gpus(tmp_path, action):
    runner = RecoveryRunner({"GPU-a": action, "GPU-b": "None"})
    agent = prepared_agent(tmp_path, runner)
    signed = envelope(command(RECOVER, gpu_uuids=["GPU-a", "GPU-b"]))

    result = agent.execute(signed)
    replay = agent.execute(signed)

    assert result.status is NodeActionStatus.SUCCEEDED, result
    assert result.details["reset_completed"] == ["GPU-a"]
    assert result.details["gpu_recovery_actions"]["GPU-b"] == "None"
    assert replay == result
    assert [argv for argv in runner.commands if "--gpu-reset" in argv] == [
        ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"]
    ]
    assert (
        len(
            [argv for argv in runner.commands if argv[:3] == ["nvidia-smi", "-q", "-x"]]
        )
        == 1
    )


def test_cleared_recovery_does_not_reset(tmp_path):
    runner = RecoveryRunner({"GPU-a": "None"})
    result = prepared_agent(tmp_path, runner).execute(envelope(command(RECOVER)))

    assert result.status is NodeActionStatus.SUCCEEDED, result
    assert result.details["reset_completed"] == []
    assert result.details["persistence_disabled"] is True
    assert not any("--gpu-reset" in argv for argv in runner.commands), runner.commands


def test_reboot_required_never_resets_a_peer_first(tmp_path):
    runner = RecoveryRunner({"GPU-a": "Reset", "GPU-b": "Reboot"})
    result = prepared_agent(tmp_path, runner).execute(
        envelope(command(RECOVER, gpu_uuids=["GPU-a", "GPU-b"]))
    )

    assert result.status is NodeActionStatus.FAILED
    assert result.details["gpu_recovery_reboot_required"] is True
    assert result.details["reset_not_attempted"] == ["GPU-a", "GPU-b"]
    assert next_rung(RECOVER, set(), result.details) is WorkflowOperation.RESTART_NODE
    assert not any("--gpu-reset" in argv for argv in runner.commands), runner.commands


@pytest.mark.parametrize(
    ("actions", "persistence", "quiesced"),
    [
        ({"GPU-a": "N/A"}, "Disabled", True),
        ({"GPU-a": "new-driver-action"}, "Disabled", True),
        ({}, "Disabled", True),
        ({"GPU-a": "Reset"}, "Enabled", True),
        ({"GPU-a": "Reset"}, "N/A", True),
        ({"GPU-a": "Reset"}, "Disabled", False),
    ],
)
def test_missing_recovery_evidence_never_resets_or_requests_reboot(
    tmp_path, actions, persistence, quiesced
):
    runner = RecoveryRunner(actions, persistence=persistence)
    result = prepared_agent(tmp_path, runner, quiesced=quiesced).execute(
        envelope(command(RECOVER))
    )

    assert result.status is NodeActionStatus.FAILED
    assert next_rung(RECOVER, set(), result.details) is None
    assert not any("--gpu-reset" in argv for argv in runner.commands), runner.commands


def test_recovery_query_timeout_is_retryable_without_reset(tmp_path):
    runner = RecoveryRunner(
        {"GPU-a": "Reset"}, query_error=TimeoutExpired("nvidia-smi", 15)
    )
    result = prepared_agent(tmp_path, runner).execute(envelope(command(RECOVER)))

    assert result.status is NodeActionStatus.FAILED
    assert result.retryable, result
    assert not any("--gpu-reset" in argv for argv in runner.commands), runner.commands


def test_active_clients_prevent_recovery_query_and_reset(tmp_path):
    runner = RecoveryRunner({"GPU-a": "Reset"})
    result = prepared_agent(tmp_path, runner, clients="/dev/nvidia0").execute(
        envelope(command(RECOVER))
    )

    assert result.status is NodeActionStatus.FAILED
    assert not any("--gpu-reset" in argv for argv in runner.commands), runner.commands
    assert not any(
        argv[:3] == ["nvidia-smi", "-q", "-x"] for argv in runner.commands
    ), runner.commands


@pytest.mark.parametrize("disabled", ["reset_enabled", "service_quiesce_enabled"])
def test_disabled_reset_or_quiesce_prevents_recovery(tmp_path, disabled):
    runner = RecoveryRunner({"GPU-a": "Reset"})
    agent = prepared_agent(tmp_path, runner)
    setattr(agent, disabled, False)
    result = agent.execute(envelope(command(RECOVER)))

    assert result.status is NodeActionStatus.FAILED
    assert not any("--gpu-reset" in argv for argv in runner.commands), runner.commands


def test_missing_quiesce_manager_prevents_recovery(tmp_path):
    runner = RecoveryRunner({"GPU-a": "Reset"})
    agent = prepared_agent(tmp_path, runner)
    agent.quiesce_manager = None
    result = agent.execute(envelope(command(RECOVER)))

    assert result.status is NodeActionStatus.FAILED
    assert not any("--gpu-reset" in argv for argv in runner.commands), runner.commands


@pytest.mark.parametrize("targets", [[], ["GPU-a", "GPU-a"]])
def test_empty_or_duplicate_targets_prevent_recovery(tmp_path, targets):
    runner = RecoveryRunner({"GPU-a": "Reset"})
    result = prepared_agent(tmp_path, runner).execute(
        envelope(command(RECOVER, gpu_uuids=targets))
    )

    assert result.status is NodeActionStatus.FAILED
    assert not any("--gpu-reset" in argv for argv in runner.commands), runner.commands


@pytest.mark.parametrize(
    ("error", "unknown", "next_operation"),
    [
        (TimeoutExpired("nvidia-smi", 120), True, None),
        (RuntimeError("GPU reset failed"), False, WorkflowOperation.RESTART_NODE),
    ],
)
def test_reset_failure_preserves_requery_evidence_and_uncertainty(
    tmp_path, error, unknown, next_operation
):
    runner = RecoveryRunner({"GPU-a": "Reset"}, reset_error=error)
    agent = prepared_agent(tmp_path, runner)
    signed = envelope(command(RECOVER))
    result = agent.execute(signed)

    assert result.status is NodeActionStatus.FAILED
    assert result.details["gpu_recovery_actions"] == {"GPU-a": "Reset"}
    assert bool(result.details.get("outcome_unknown")) is unknown
    assert not result.retryable, result
    assert next_rung(RECOVER, set(), result.details) is next_operation
    assert agent.execute(signed) == result
    assert len([argv for argv in runner.commands if "--gpu-reset" in argv]) == 1


def test_recovery_requires_explicit_agent_allowlist(tmp_path):
    runner = RecoveryRunner({"GPU-a": "Reset"})
    agent = prepared_agent(tmp_path, runner)
    agent.allowed_operations = {WorkflowOperation.RESET_GPU}

    with pytest.raises(ValueError, match="RECOVER_DRAIN_P2P is not allowed"):
        agent.execute(envelope(command(RECOVER)))
    assert not any("--gpu-reset" in argv for argv in runner.commands), runner.commands


def test_final_ownership_refusal_after_query_prevents_reset(tmp_path, monkeypatch):
    runner = RecoveryRunner({"GPU-a": "Reset"})
    agent = prepared_agent(tmp_path, runner, require_ownership=True)
    checkpoints = []

    def refuse(current):
        checkpoints.append(current.operation)
        raise OwnershipRefused("OWNERSHIP_CALLER_LOST")

    monkeypatch.setattr(agent.ownership_gate, "require", refuse)
    value = command(RECOVER).model_copy(
        update={"ownership_guard": OWNERSHIP_PROTOCOL, "agent_generation": 1}
    )
    result = agent.execute(envelope(value))

    assert result.status is NodeActionStatus.FAILED
    assert checkpoints == [RECOVER]
    assert result.details["reason"] == "OWNERSHIP_CALLER_LOST"
    assert result.details["node_action_not_started"] is True
    assert result.details["gpu_recovery_actions"] == {"GPU-a": "Reset"}
    assert not any("--gpu-reset" in argv for argv in runner.commands), runner.commands


@pytest.mark.parametrize(
    "text",
    [
        "not XML",
        "<nvidia_smi_log/>",
        "<nvidia_smi_log>"
        + xml_report({"GPU-a": "Reset"})
        .removeprefix("<nvidia_smi_log>")
        .removesuffix("</nvidia_smi_log>")
        * 2
        + "</nvidia_smi_log>",
    ],
)
def test_invalid_recovery_reports_are_rejected(text):
    with pytest.raises(RuntimeError):
        parse_recovery_state(text, ["GPU-a"])


def test_recovery_report_ignores_untargeted_gpus():
    assert parse_recovery_state(
        xml_report({"GPU-a": "None", "GPU-b": "N/A"}), ["GPU-a"]
    ) == {"GPU-a": "None"}
