"""The Agent's own GPU work stays out of another incident's quiesce window.

Live (GF-REGIONAL-DESTR-001 attempt 2, plain EKS): during an XID 46 reset's
quiesce window the control plane dispatched another incident's
RUN_DCGM_DIAGNOSTIC to the same Agent, which ran ``dcgmi diag -r 1 -j`` inside
the window: VERIFY_NO_GPU_CLIENTS met the Agent's own ``dcgmi`` on the target
device (``nvvs`` beside it), and the diagnostic graded the window itself as a
GPU failure -- the escalation of that verdict re-quarantined the recovered node.
Now foreign GPU work is refused at admission while a window is open, work
already running when the window opens holds the verification and reports no
verdict, and a multi-capture bundle stops at its next spawn.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess

import pytest

from tests.node_agent._support import (
    NodeActionCommand,
    NodeActionStatus,
    ServiceRunner,
    WorkflowOperation,
    command,
    envelope,
    no_device_clients,
    node_action_executor,
    quiesce_manager,
)

QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
VERIFY = WorkflowOperation.VERIFY_NO_GPU_CLIENTS
RESET = WorkflowOperation.RESET_GPU
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
DIAGNOSTIC = WorkflowOperation.RUN_DCGM_DIAGNOSTIC
BUNDLE = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
BUSY = "GPU 00000000:59:00.0: In use by another client"
WINDOW_OPEN = "GPU maintenance window is open"
PASSING_DIAGNOSTIC = json.dumps({"tests": [{"name": "software", "status": "Pass"}]})


class WindowRunner(ServiceRunner):
    """Services as ``ServiceRunner``; records ``dcgmi`` and can hold it or the
    bundle's first capture until ``release`` is set. A reset spawned while a
    ``dcgmi`` started inside the window has not returned is refused busy, the
    way the driver refuses it."""

    def __init__(self, *, hold: tuple[str, ...] = ()) -> None:
        super().__init__(active={"nvidia-fabricmanager", "nvidia-dcgm", "kubelet"})
        self.hold = hold
        self.started = threading.Event()
        self.release = threading.Event()
        self.quiesced = False
        self.dcgmi_in_window = 0

    def __call__(self, value, **kwargs):
        if value[:1] == ["systemd-run"]:
            self.quiesced = True
        if value[:2] == ["systemctl", "stop"] and value[2].endswith(".timer"):
            self.quiesced = False
        if value[:1] == ["dcgmi"]:
            self.commands.append(value)
            if self.quiesced:
                self.dcgmi_in_window += 1
            self._hold(value)
            return CompletedProcess(value, 0, stdout=PASSING_DIAGNOSTIC, stderr="")
        if value[:2] == ["nvidia-smi", "-q"]:
            self.commands.append(value)
            self._hold(value)
            return CompletedProcess(value, 0, stdout="", stderr="")
        if "--gpu-reset" in value:
            self.commands.append(value)
            if self.dcgmi_in_window:
                raise CalledProcessError(255, value, stderr=BUSY)
            return CompletedProcess(value, 0, stdout="", stderr="")
        return super().__call__(value, **kwargs)

    def _hold(self, value: list[str]) -> None:
        if value[0] in self.hold:
            self.started.set()
            assert self.release.wait(10), "test never released the held spawn"


def _agent(tmp_path: Path, runner: WindowRunner, *, drain: float | None = None):
    # The drain bound is passed only where a test needs it short.
    extra = {} if drain is None else {"gpu_activity_drain_seconds": drain}
    return node_action_executor(
        tmp_path,
        "window.db",
        allowed_operations={QUIESCE, VERIFY, RESET, RESTORE, DIAGNOSTIC, BUNDLE},
        reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=quiesce_manager(tmp_path, runner),
        runner=runner,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        diagnostic_output_dir=str(tmp_path / "diagnostics"),
        sleep=lambda _: None,
        **extra,
    )


def _foreign(operation: WorkflowOperation, command_id: str) -> NodeActionCommand:
    """Another incident's command for the same node (the node-health diagnostic)."""

    return command(operation, command_id=command_id).model_copy(
        update={"incident_id": "incident-health", "workflow_request_id": "workflow-b"}
    )


def _quiesce(agent):
    result = agent.execute(envelope(command(QUIESCE, command_id="wf/quiesce/node-a")))
    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    return result


def _restore(agent) -> None:
    result = agent.execute(envelope(command(RESTORE, command_id="wf/restore/node-a")))
    assert result.status is NodeActionStatus.SUCCEEDED, result.error


def _dcgmi_calls(runner: WindowRunner) -> int:
    return len([item for item in runner.commands if item[:1] == ["dcgmi"]])


def test_a_foreign_diagnostic_between_verify_and_reset_is_refused(
    tmp_path: Path,
) -> None:
    runner = WindowRunner()
    agent = _agent(tmp_path, runner)
    _quiesce(agent)
    verified = agent.execute(envelope(command(VERIFY, command_id="wf/verify/node-a")))
    assert verified.status is NodeActionStatus.SUCCEEDED, verified.error

    diagnostic = _foreign(DIAGNOSTIC, "health/diag/node-a")
    with pytest.raises(ValueError, match=f"{WINDOW_OPEN}.*incident-a"):
        agent.execute(envelope(diagnostic))
    reset = agent.execute(envelope(command(RESET, command_id="wf/reset/node-a")))

    assert _dcgmi_calls(runner) == 0, "no dcgmi between the verification and the reset"
    assert agent.ledger.attempt_history(diagnostic.command_id) == [], (
        "a refusal at admission reaches no ledger row: nothing was dispatched"
    )
    assert reset.status is NodeActionStatus.SUCCEEDED, reset.error
    assert reset.details["reset_attempts"] == 1

    _restore(agent)
    after = agent.execute(envelope(diagnostic))
    assert after.status is NodeActionStatus.SUCCEEDED, after.error
    assert _dcgmi_calls(runner) == 1, "the same command runs once the window closed"


def test_the_window_owner_keeps_its_own_gpu_work(tmp_path: Path) -> None:
    runner = WindowRunner()
    agent = _agent(tmp_path, runner)
    _quiesce(agent)

    own = agent.execute(envelope(command(DIAGNOSTIC, command_id="wf/diag/node-a")))

    assert own.status is NodeActionStatus.SUCCEEDED, own.error
    assert _dcgmi_calls(runner) == 1


def test_a_diagnostic_running_when_the_window_opens_holds_the_verification(
    tmp_path: Path,
) -> None:
    runner = WindowRunner(hold=("dcgmi",))
    agent = _agent(tmp_path, runner, drain=0.05)
    diagnostic = _foreign(DIAGNOSTIC, "health/diag/node-a")
    results = {}
    worker = threading.Thread(
        target=lambda: results.setdefault(
            "diagnostic", agent.execute(envelope(diagnostic))
        )
    )
    worker.start()
    try:
        assert runner.started.wait(5), "the held spawn never started"
        quiesced = _quiesce(agent)
        held = agent.execute(envelope(command(VERIFY, command_id="wf/verify/a1")))
    finally:
        runner.release.set()
        worker.join(10)
    assert not worker.is_alive(), "the diagnostic never returned"

    overlapped = quiesced.details["agent_gpu_activities_overlapped"]
    assert [item["command_id"] for item in overlapped] == [diagnostic.command_id]
    assert [
        item["command_id"] for item in quiesced.details["agent_gpu_activities_running"]
    ] == [diagnostic.command_id], "the bounded wait ended with it still running"
    assert held.status is NodeActionStatus.FAILED
    assert "GPU agent clients are still active" in (held.error or ""), (
        "the wording the control plane's VERIFY wait matches on"
    )
    assert held.details["gpu_client_kind"] == "agent"
    assert held.details["gpu_client_holders"][0]["operation"] == DIAGNOSTIC.value

    reported = results["diagnostic"]
    assert reported.status is NodeActionStatus.FAILED
    assert reported.retryable is True, "it runs again once the window closed"
    assert "GpuActivityOverlapped" in (reported.error or "")
    assert reported.details["gpu_activity_overlapped_by"] == "incident-a"
    assert reported.details["gpu_activity_outcome"] == "SUCCEEDED"
    assert "diagnostic_outcome" not in reported.details, "no verdict of the window"

    verified = agent.execute(envelope(command(VERIFY, command_id="wf/verify/a2")))
    assert verified.status is NodeActionStatus.SUCCEEDED, verified.error
    with pytest.raises(ValueError, match=WINDOW_OPEN):
        agent.execute(envelope(diagnostic))
    _restore(agent)
    rerun = agent.execute(envelope(diagnostic))
    assert rerun.status is NodeActionStatus.SUCCEEDED, rerun.error
    assert rerun.attempt == 2


def test_the_quiesce_waits_for_a_short_diagnostic_to_end(tmp_path: Path) -> None:
    runner = WindowRunner(hold=("dcgmi",))
    agent = _agent(tmp_path, runner, drain=5.0)
    diagnostic = _foreign(DIAGNOSTIC, "health/diag/node-a")
    results = {}
    worker = threading.Thread(
        target=lambda: results.setdefault(
            "diagnostic", agent.execute(envelope(diagnostic))
        )
    )
    worker.start()
    releaser = threading.Timer(0.2, runner.release.set)
    try:
        assert runner.started.wait(5), "the held spawn never started"
        releaser.start()
        quiesced = _quiesce(agent)
    finally:
        runner.release.set()
        worker.join(10)
        releaser.cancel()

    assert quiesced.details["agent_gpu_activities_running"] == []
    assert quiesced.details["agent_gpu_activity_wait_seconds"] > 0
    assert results["diagnostic"].retryable is True
    verified = agent.execute(envelope(command(VERIFY, command_id="wf/verify/node-a")))
    assert verified.status is NodeActionStatus.SUCCEEDED, verified.error


def test_a_bundle_stops_at_its_next_capture_once_the_window_opens(
    tmp_path: Path,
) -> None:
    runner = WindowRunner(hold=("nvidia-smi",))
    agent = _agent(tmp_path, runner, drain=0.05)
    bundle = _foreign(BUNDLE, "health/bundle/node-a")
    results = {}
    worker = threading.Thread(
        target=lambda: results.setdefault("bundle", agent.execute(envelope(bundle)))
    )
    worker.start()
    try:
        assert runner.started.wait(5), "the held spawn never started"
        _quiesce(agent)
    finally:
        runner.release.set()
        worker.join(10)

    reported = results["bundle"]
    assert reported.status is NodeActionStatus.FAILED
    assert reported.retryable is True
    assert reported.details["gpu_activity_outcome"] == (
        "stopped before its next GPU subprocess"
    )
    assert _dcgmi_calls(runner) == 0, "the bundle's dcgmi capture never ran"
    assert not [
        item for item in runner.commands if item[:2] == ["nvidia-smi", "nvlink"]
    ]


def test_the_admission_refusal_is_a_retryable_conflict() -> None:
    from gpu_fault.node_agent.app import _node_action_rejection
    from gpu_fault.node_agent.gpu_activity import GpuMaintenanceWindowOpen

    refusal = GpuMaintenanceWindowOpen(
        DIAGNOSTIC, [{"incident_id": "incident-a", "phase": "QUIESCED"}]
    )

    status, detail = _node_action_rejection(refusal)

    assert status == 409
    assert detail["code"] == "GPU_MAINTENANCE_WINDOW_OPEN"
    assert detail["retryable"] is True
    assert detail["requires_new_command"] is False, "the same command, later"


def test_open_windows_are_this_boots_state_files(tmp_path: Path) -> None:
    from gpu_fault.node_agent.quiesce import open_quiesce_windows

    runner = WindowRunner()
    manager = quiesce_manager(tmp_path, runner)
    assert open_quiesce_windows(manager) == []

    manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")
    windows = open_quiesce_windows(manager)
    assert [(item["incident_id"], item["phase"]) for item in windows] == [
        ("incident-a", "QUIESCED")
    ]

    stale = manager.state_dir / "quiesce-previous-boot.json"
    stale.write_text(
        json.dumps({"incident_id": "incident-old", "boot_id": "another-boot"}),
        encoding="utf-8",
    )
    broken = manager.state_dir / "quiesce-broken.json"
    broken.write_text("[]", encoding="utf-8")
    owners = [item["incident_id"] for item in open_quiesce_windows(manager)]
    assert len(owners) == 2 and set(owners) == {None, "incident-a"}, (
        "a previous boot's file is the reconcile's; an unreadable one fails closed"
    )

    manager.restore(incident_id="incident-a")
    assert [item["incident_id"] for item in open_quiesce_windows(manager)] == [None]
