"""One start of a host DCGM engine that is down, before the quick diagnostic is graded.

The DCGM 4 package unit ``nvidia-dcgm`` is ``Restart=on-abort`` with 5 starts
per 10 s on both the plain EKS AL2023 NVIDIA AMI and the HyperPod AMI, so a
non-zero exit or a crash loop leaves a host-mode node's engine down and its next
``dcgmi diag`` is graded FAIL/``DCGM_EXECUTION_REVIEW`` -- a drain a start
would have avoided. In host mode (an ``eks-managed`` node group declaring
``gpu-fault.io/dcgm-hostengine=host``, every HyperPod node) the Agent starts an
enabled engine that is down once, waits bounded for ``dcgmi discovery -l`` and
runs the diagnostic again; an engine that still does not answer is graded as
before. It never restarts an active engine, never starts one inside a quiesce
window, never installs anything and never touches an embedded-mode node.
"""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from gpu_fault.node_agent.gpu_activity import GpuActivityGate, GpuMaintenanceWindowOpen
from gpu_fault.node_agent.operations import hostengine_recovery as recovery
from tests.node_agent._support import (
    NOW,
    NodeActionStatus,
    ServiceRunner,
    WorkflowOperation,
    command,
    envelope,
    no_device_clients,
    node_action_executor,
    quiesce_manager,
)

DIAGNOSTIC = WorkflowOperation.RUN_DCGM_DIAGNOSTIC
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
UNIT = "nvidia-dcgm"
PASSING = json.dumps({"tests": [{"name": "software", "status": "Pass"}]})
HOSTENGINE_DOWN = (
    "Error: unable to establish a connection to the specified host: localhost\n"
    "Error: Unable to connect to host engine. Host engine connection "
    "invalid/disconnected.\n"
)
DIAG = ["dcgmi", "diag", "-r", "1", "-j"]
DISCOVERY = ["dcgmi", "discovery", "-l"]


class EngineHost(ServiceRunner):
    """A node's ``nvidia-dcgm`` and the ``dcgmi`` that dials it.

    ``state`` is what ``systemctl is-active`` says while the unit is not
    running (``inactive``, ``failed``, ``activating``); the engine answers
    once ``ready_after`` discoveries have run since it came up, and a ``hung``
    engine never answers although systemd runs it. ``diag_failure`` is an
    unrelated dcgmi failure of a running engine, ``down_output`` what dcgmi
    says while no engine answers.
    """

    def __init__(
        self,
        *,
        state: str = "inactive",
        enabled: str = "enabled",
        ready_after: int = 1,
        hung: bool = False,
        diag_failure: str | None = None,
        down_output: tuple[int, str] = (226, HOSTENGINE_DOWN),
    ) -> None:
        running = {"nvidia-fabricmanager", "kubelet"}
        if state == "active":
            running.add(UNIT)
        super().__init__(active=running)
        self.state = "inactive" if state == "active" else state
        self.enabled = enabled
        self.ready_after = ready_after
        self.hung = hung
        self.diag_failure = diag_failure
        self.down_output = down_output
        self.discoveries = 0

    @property
    def up(self) -> bool:
        return UNIT in self.active

    def answers(self) -> bool:
        return self.up and not self.hung and self.discoveries >= self.ready_after

    def __call__(self, value: list[str], **kwargs: Any) -> CompletedProcess[str]:
        if value[:2] == ["systemctl", "is-active"] and value[2:] == [UNIT]:
            self.commands.append(value)
            state = "active" if self.up else self.state
            return CompletedProcess(value, 0 if self.up else 3, state + "\n", "")
        if value == ["systemctl", "is-enabled", UNIT]:
            self.commands.append(value)
            code = 0 if self.enabled == "enabled" else 1
            return CompletedProcess(value, code, self.enabled + "\n", "")
        if value == ["systemctl", "reset-failed", UNIT]:
            self.commands.append(value)
            self.state = "inactive"
            return CompletedProcess(value, 0, "", "")
        if value[:2] == ["systemctl", "stop"] and value[2:] == [UNIT]:
            self.state = "inactive"
        if value == DISCOVERY:
            self.commands.append(value)
            if self.up and not self.hung:
                self.discoveries += 1
            if self.answers():
                return CompletedProcess(value, 0, "8 GPUs found.\n", "")
            return CompletedProcess(value, 226, "", HOSTENGINE_DOWN)
        if value == DIAG:
            self.commands.append(value)
            if self.up and self.diag_failure is not None:
                return CompletedProcess(value, 1, "", self.diag_failure)
            if self.answers():
                return CompletedProcess(value, 0, PASSING, "")
            return CompletedProcess(value, self.down_output[0], "", self.down_output[1])
        return super().__call__(value, **kwargs)

    def unit_commands(self) -> list[list[str]]:
        """What touched the engine's unit, the quiesce's own calls included."""

        return [
            item
            for item in self.commands
            if item[:1] == ["systemctl"] and item[-1] == UNIT and "--quiet" not in item
        ]


def _execute(
    tmp_path: Path,
    host: EngineHost,
    *,
    provider: str | None = "eks-managed",
    mode: str = "host",
    name: str = "agent",
):
    sleeps: list[float] = []
    overrides: dict[str, Any] = {}
    if provider is not None:
        overrides.update(cluster_provider=provider, dcgm_hostengine=mode)
    agent = node_action_executor(
        tmp_path,
        f"{name}.db",
        allowed_operations={DIAGNOSTIC},
        diagnostic_output_dir=str(tmp_path / name),
        runner=host,
        sleep=sleeps.append,
        **overrides,
    )
    result = agent.execute(envelope(command(DIAGNOSTIC)))
    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    return result.details, sleeps


def _grading(details: dict[str, Any]) -> tuple[str, bool, list[str]]:
    return (
        details["diagnostic_outcome"],
        details["configuration_only_failures"],
        [item["action_code"] for item in details["recommended_actions"]],
    )


# --- through the executor -------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "mode"),
    [("eks-managed", "host"), (None, "")],
    ids=["eks-host-mode", "hyperpod"],
)
def test_a_failed_engine_is_started_once_and_the_diagnostic_graded(
    tmp_path: Path, provider: str | None, mode: str
) -> None:
    host = EngineHost(state="failed", ready_after=2)

    details, sleeps = _execute(tmp_path, host, provider=provider, mode=mode)

    assert _grading(details) == ("PASS", False, []), (
        "the engine answered again, so the diagnostic graded the GPUs"
    )
    evidence = details["hostengine_recovery"]
    assert {
        key: evidence[key]
        for key in (
            "attempted",
            "before_state",
            "unit_enabled",
            "after_state",
            "actions",
            "engine_answers",
            "discovery_polls",
            "diagnostic_rerun",
        )
    } == {
        "attempted": True,
        "before_state": "failed",
        "unit_enabled": "enabled",
        "after_state": "active",
        "actions": ["reset-failed", "start"],
        "engine_answers": True,
        "discovery_polls": 2,
        "diagnostic_rerun": True,
    }
    assert isinstance(evidence["seconds"], float), evidence
    assert evidence["first_attempt"]["returncode"] == 226
    assert (
        "Unable to connect to host engine" in evidence["first_attempt"]["output_tail"]
    )
    assert host.unit_commands() == [
        ["systemctl", "is-active", UNIT],
        ["systemctl", "is-enabled", UNIT],
        ["systemctl", "reset-failed", UNIT],
        ["systemctl", "start", UNIT],
        ["systemctl", "is-active", UNIT],
    ], "one start, never a restart"
    assert [item for item in host.commands if item[:1] == ["dcgmi"]] == [
        DIAG,
        DISCOVERY,
        DISCOVERY,
        DIAG,
    ], "the diagnostic runs once more, after the engine answered"
    assert sleeps == [recovery.DISCOVERY_INTERVAL_SECONDS]
    assert ("dcgm_hostengine" in details) is (provider == "eks-managed")


def test_an_engine_that_still_does_not_answer_is_graded_as_before(
    tmp_path: Path,
) -> None:
    host = EngineHost(state="inactive", ready_after=10**6)

    details, sleeps = _execute(tmp_path, host)

    assert _grading(details) == ("FAIL", False, ["DCGM_EXECUTION_REVIEW"]), (
        "the declared engine still does not answer: graded exactly as before"
    )
    evidence = details["hostengine_recovery"]
    assert evidence["attempted"] is True
    assert evidence["engine_answers"] is False
    assert evidence["actions"] == ["start"]
    assert evidence["after_state"] == "active", "systemd runs it, it does not answer"
    assert evidence["reason"] == (
        "nvidia-dcgm did not answer dcgmi discovery -l within 30 s"
    )
    assert "diagnostic_rerun" not in evidence
    assert host.commands.count(DIAG) == 1, "no second diagnostic without an engine"
    assert evidence["discovery_polls"] == len(sleeps) + 1 <= 16, "bounded"


def test_a_quiesce_window_keeps_the_engine_stopped(tmp_path: Path) -> None:
    """The quiesce stops an active nvidia-dcgm on purpose; its restore starts it.
    A diagnostic of the window's own incident runs inside it and must not."""

    host = EngineHost(state="active", ready_after=0)
    agent = node_action_executor(
        tmp_path,
        "window.db",
        allowed_operations={QUIESCE, DIAGNOSTIC},
        reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=quiesce_manager(tmp_path, host),
        runner=host,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        diagnostic_output_dir=str(tmp_path / "diagnostics"),
        cluster_provider="eks-managed",
        dcgm_hostengine="host",
        sleep=lambda _seconds: None,
    )
    quiesced = agent.execute(envelope(command(QUIESCE, command_id="wf/quiesce/node-a")))
    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced.error
    assert not host.up, "the quiesce stopped the engine"
    host.commands.clear()

    result = agent.execute(envelope(command(DIAGNOSTIC)))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert _grading(result.details) == ("FAIL", False, ["DCGM_EXECUTION_REVIEW"])
    evidence = result.details["hostengine_recovery"]
    assert evidence["attempted"] is False
    assert evidence["before_state"] == "inactive"
    assert evidence["reason"].startswith(
        "a quiesce window is open on this node (incident-a QUIESCED)"
    ), evidence
    assert ["systemctl", "start", UNIT] not in host.commands, (
        "the window's engine is the restore's to start"
    )
    assert not host.up, "the engine stays stopped until the restore"


@pytest.mark.parametrize(
    ("state", "enabled", "reason"),
    [
        ("inactive", "disabled", "nvidia-dcgm is disabled: an engine that is not"),
        ("inactive", "masked", "nvidia-dcgm is masked: an engine that is not"),
        ("inactive", "not-found", "nvidia-dcgm is not-found: an engine that is not"),
    ],
    ids=["disabled", "masked", "not-installed"],
)
def test_an_engine_that_is_not_enabled_is_left_alone(
    tmp_path: Path, state: str, enabled: str, reason: str
) -> None:
    host = EngineHost(state=state, enabled=enabled)

    details, _sleeps = _execute(tmp_path, host)

    assert _grading(details) == ("FAIL", False, ["DCGM_EXECUTION_REVIEW"])
    assert details["hostengine_recovery"]["attempted"] is False
    assert details["hostengine_recovery"]["reason"].startswith(reason), details
    assert not [item for item in host.commands if item[1:2] == ["start"]]


def test_an_active_engine_that_does_not_answer_is_never_restarted(
    tmp_path: Path,
) -> None:
    host = EngineHost(state="active", hung=True)

    details, sleeps = _execute(tmp_path, host)

    assert _grading(details) == ("FAIL", False, ["DCGM_EXECUTION_REVIEW"])
    evidence = details["hostengine_recovery"]
    assert (evidence["attempted"], evidence["before_state"]) == (False, "active")
    assert evidence["reason"] == (
        "nvidia-dcgm is active: an engine systemd runs or is stopping is never "
        "restarted"
    )
    assert host.unit_commands() == [
        ["systemctl", "is-active", UNIT],
        ["systemctl", "is-enabled", UNIT],
    ], "read only: no start, no restart, no stop"
    assert sleeps == []


def test_another_ungraded_failure_counts_only_when_the_engine_is_down(
    tmp_path: Path,
) -> None:
    failure = "Error: the diagnostic could not start the plugin\n"
    running = EngineHost(state="active", diag_failure=failure)
    down = EngineHost(state="failed", down_output=(1, "Error: unexpected failure\n"))

    up_details, _ = _execute(tmp_path, running, name="up")
    down_details, _ = _execute(tmp_path, down, name="down")

    assert "hostengine_recovery" not in up_details, (
        "the engine runs: an unrelated dcgmi failure is no recovery case"
    )
    assert _grading(up_details) == ("FAIL", False, ["DCGM_EXECUTION_REVIEW"])
    assert down_details["hostengine_recovery"]["attempted"] is True, (
        "a failure dcgmi does not word as a lost connection, with the unit down"
    )
    assert down_details["hostengine_recovery"]["first_attempt"]["returncode"] == 1
    assert _grading(down_details) == ("PASS", False, [])


def test_embedded_mode_never_touches_the_engine(tmp_path: Path) -> None:
    host = EngineHost(state="failed")

    details, sleeps = _execute(tmp_path, host, mode="embedded")

    assert _grading(details) == ("WARN", True, ["DCGM_HOSTENGINE_UNAVAILABLE"])
    assert "hostengine_recovery" not in details
    assert [item for item in host.commands if item[:1] == ["systemctl"]] == []
    assert host.commands == [DIAG] and sleeps == []


def test_a_quiesce_opening_during_the_recovery_stops_it_at_the_next_spawn(
    tmp_path: Path,
) -> None:
    """The window check and the start are two spawns apart: a QUIESCE of another
    incident that opens in between flags the diagnostic, which then stops before
    its next spawn -- no verdict, retryable -- and the quiesce stops whatever it
    finds active, the restore starts it again."""

    host = EngineHost(state="failed")
    agents: list[Any] = []

    def opening(value, **kwargs):
        completed = host(value, **kwargs)
        if value == ["systemctl", "start", UNIT]:
            agents[0].gpu_activity.open_window("incident-other")
        return completed

    agents.append(
        node_action_executor(
            tmp_path,
            "race.db",
            allowed_operations={DIAGNOSTIC},
            diagnostic_output_dir=str(tmp_path / "race"),
            runner=opening,
            cluster_provider="eks-managed",
            dcgm_hostengine="host",
            sleep=lambda _seconds: None,
            gpu_activity_drain_seconds=0,
        )
    )

    result = agents[0].execute(envelope(command(DIAGNOSTIC)))

    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is True, "it runs again once the window is closed"
    assert result.details["gpu_activity_overlapped_by"] == "incident-other"
    assert DISCOVERY not in host.commands, "stopped before its next spawn"


def test_a_missing_dcgmi_is_never_installed(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def missing(argv, **_kwargs):
        calls.append(list(argv))
        raise FileNotFoundError(2, "No such file or directory", argv[0])

    agent = node_action_executor(
        tmp_path,
        "missing.db",
        allowed_operations={DIAGNOSTIC},
        diagnostic_output_dir=str(tmp_path / "missing"),
        runner=missing,
        cluster_provider="eks-managed",
        dcgm_hostengine="host",
        sleep=lambda _seconds: None,
    )

    result = agent.execute(envelope(command(DIAGNOSTIC)))

    assert _grading(result.details) == ("FAIL", False, ["DCGM_EXECUTION_REVIEW"])
    assert "hostengine_recovery" not in result.details
    assert calls == [DIAG], "nothing but the diagnostic itself is run"


# --- the recovery itself --------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _recover(host: EngineHost, clock: Clock | None = None, **options: Any):
    ticking = clock or Clock()
    return recovery.recover_host_engine(
        host,
        windows=options.pop("windows", lambda: []),
        sleep=ticking.sleep,
        monotonic=ticking.monotonic,
        **options,
    )


def test_the_wait_for_the_engine_is_bounded_by_time() -> None:
    clock = Clock()
    host = EngineHost(state="inactive", ready_after=10**6)
    original = host.__call__

    def slow(value, **kwargs):
        if value == DISCOVERY:
            clock.now += 3.0  # each discovery takes 3 s to fail
        return original(value, **kwargs)

    evidence = recovery.recover_host_engine(
        slow, windows=lambda: [], sleep=clock.sleep, monotonic=clock.monotonic
    )

    assert evidence is not None and evidence["engine_answers"] is False
    assert evidence["seconds"] <= recovery.RECOVERY_BUDGET_SECONDS + 3.0
    assert evidence["discovery_polls"] < 16, "the clock, not the poll count, ended it"


def test_an_engine_systemd_is_starting_is_waited_for_without_a_start() -> None:
    host = EngineHost(state="activating")
    original = host.__call__

    def coming_up(value, **kwargs):
        if value == DISCOVERY:
            host.active.add(UNIT)  # systemd's own restart brings it up
        return original(value, **kwargs)

    evidence = recovery.recover_host_engine(
        coming_up, windows=lambda: [], sleep=Clock().sleep, monotonic=Clock().monotonic
    )

    assert evidence is not None
    assert (evidence["actions"], evidence["engine_answers"]) == (["wait"], True)
    assert not [item for item in host.commands if item[1:2] == ["start"]]


def test_a_start_systemd_refuses_is_reported_without_a_wait() -> None:
    host = EngineHost(state="inactive")
    host.fail_start = UNIT

    evidence = _recover(host)

    assert evidence is not None
    assert evidence["attempted"] is True and evidence["engine_answers"] is False
    assert (evidence["start_returncode"], evidence["start_error"]) == (
        1,
        "start failed",
    )
    assert evidence["discovery_polls"] == 0
    assert evidence["reason"] == "systemctl start nvidia-dcgm failed"


def test_an_unreadable_window_state_or_unit_state_starts_nothing() -> None:
    host = EngineHost(state="failed")

    def broken() -> list[dict[str, Any]]:
        raise OSError("state directory unreadable")

    unreadable_window = _recover(host, windows=broken)

    def no_systemctl(value, **kwargs):
        if value[:1] == ["systemctl"]:
            raise FileNotFoundError(2, "No such file or directory", "systemctl")
        return host(value, **kwargs)

    unknown_unit = recovery.recover_host_engine(
        no_systemctl, windows=lambda: [], sleep=Clock().sleep
    )

    assert unreadable_window is not None and unknown_unit is not None
    assert unreadable_window["reason"] == (
        "the quiesce window state could not be read (OSError)"
    )
    assert (unknown_unit["before_state"], unknown_unit["reason"]) == (
        "unknown",
        "the state of nvidia-dcgm could not be read",
    )
    assert not [item for item in host.commands if item[1:2] == ["start"]]


def test_a_running_engine_is_no_recovery_case_for_another_failure() -> None:
    host = EngineHost(state="active")

    assert _recover(host, only_when_down=True) is None
    assert host.commands == [["systemctl", "is-active", UNIT]]


def test_every_open_window_counts_the_callers_own_and_an_unreadable_one() -> None:
    on_disk = [
        {"incident_id": "incident-a", "phase": "QUIESCED"},
        {"incident_id": None, "phase": "UNREADABLE"},
    ]
    gate = GpuActivityGate(
        open_windows=lambda: list(on_disk), now=lambda: NOW, drain_seconds=0
    )
    gate.open_window("incident-b")  # a QUIESCE of incident-b in progress

    windows = gate.open_windows()

    assert windows == [{"incident_id": "incident-b", "phase": "OPENING"}, *on_disk]
    with pytest.raises(GpuMaintenanceWindowOpen) as refused:
        gate.refuse_if_window_open(DIAGNOSTIC, "incident-a")
    assert refused.value.windows == [
        {"incident_id": "incident-b", "phase": "OPENING"},
        {"incident_id": None, "phase": "UNREADABLE"},
    ], "the admission check still leaves the window's own incident out"
