"""QUIESCE ends persistence on its target GPUs only; the sweep is the fallback.

Live (HyperPod, DESTR-015): the two nodes of one parallel reset sat on either
side of a boot-order race -- one ran ``nvidia-persistenced`` alone, the other
also carried the legacy flag ``gpu-fault-gpu-persistence`` sets -- and the
quiesce sweep TERMed the daemon on both. Its shutdown disables persistence on
every device it holds, so on the daemon-only node every ``nvidia-smi`` of the
window re-initialised eight GPUs and its reset started 58 s after the other
node's; restore brought neither daemon back. On EKS the systemd unit was
stopped outright, with the same result on every GPU of every node.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.node_agent.late_ownership import (
    OwnershipRefused,
    command_identity,
    execute_with_final_ownership,
    final_ownership_boundary,
    sign_permit,
)
from gpu_fault.node_agent.operations.clients import ClientOperationsMixin
from gpu_fault.node_agent.operations.reset import ResetOperationsMixin
from gpu_fault.node_agent.protocol import NodeActionCommand, SignedNodeAction
from tests.regional.test_late_ownership_agent_gate import KEY, NOW
from tests.regional.test_late_ownership_agent_gate import challenge as challenge_fixture
from tests.regional.test_late_ownership_agent_gate import command as command_fixture

from ._support import (
    GpuServiceQuiesceManager,
    NodeActionStatus,
    ServiceRunner,
    WorkflowOperation,
    command,
    envelope,
    node_action_executor,
)

TARGET_BUS_ID = "00000000:53:00.0"
OTHER_BUS_ID = "00000000:64:00.0"
# Probe order is not PCI order: the target is /dev/nvidia1. The release follows
# the bus id the executor resolved, never a minor number or an index.
DEVICES = {TARGET_BUS_ID: "/dev/nvidia1", OTHER_BUS_ID: "/dev/nvidia0"}
# Above the kernel's pid_max, so never the pid of the test process itself
# (the holder scan skips its own pid).
DAEMON_PID = "7777777"
DAEMON_EXE = "/usr/bin/nvidia-persistenced"
DAEMON_START = 7351
RELEASE = ["nvidia-smi", "-i", TARGET_BUS_ID, "-pm", "0"]
SYSTEMD_CGROUP = "/system.slice/nvidia-persistenced.service"
SESSION_CGROUP = "/user.slice/user-0.slice/session-1.scope"
SERVICES = (
    "nvidia-dcgm",
    "nvidia-persistenced",
    "gpu-fault-gpu-persistence",
    "kubelet",
)
HYPERPOD_ACTIVE = {"nvidia-dcgm", "gpu-fault-gpu-persistence", "kubelet"}
EKS_ACTIVE = HYPERPOD_ACTIVE | {"nvidia-persistenced"}


def modes(**values: str) -> list[dict[str, str]]:
    return [
        {"pci_bus_id": bus_id, "persistence_mode": values[name]}
        for name, bus_id in (("target", TARGET_BUS_ID), ("other", OTHER_BUS_ID))
    ]


class PersistenceNode(ServiceRunner):
    """systemd, ``nvidia-smi`` and one persistence daemon on a fake ``/proc``.

    A GPU is persistent while the daemon holds its device node or its legacy
    flag is set. ``nvidia-smi -pm 0`` goes over the daemon's RPC when one runs
    -- the daemon closes only that device -- and clears the legacy flag
    otherwise. A TERMed daemon closes the target's device first and one more
    device per poll of the sweep (``tick``) before it exits, the way the real
    shutdown tears down device by device.
    """

    def __init__(
        self,
        tmp_path: Path,
        *,
        daemon: str | None,
        legacy: bool = False,
        active: set[str] | None = None,
        release_error: str | None = None,
        release_timeout: bool = False,
        devices: dict[str, str] | None = None,
    ) -> None:
        super().__init__(active=set(active or ()))
        self.proc = tmp_path / "proc"
        self.proc.mkdir()
        self.devices = dict(devices or DEVICES)
        self.legacy = set(self.devices) if legacy else set()
        self.held: set[str] = set()
        self.cgroup = SYSTEMD_CGROUP if daemon == "systemd" else SESSION_CGROUP
        self.release_error = release_error
        self.release_timeout = release_timeout
        self.timeouts: list[Any] = []
        self.teardown: list[str] = []
        self.ticks = 0
        if daemon is not None:
            self._start_daemon()

    @property
    def daemon_dir(self) -> Path:
        return self.proc / DAEMON_PID

    def _fd(self, bus_id: str) -> Path:
        return self.daemon_dir / "fd" / str(10 + sorted(self.devices).index(bus_id))

    def _start_daemon(self) -> None:
        process = self.daemon_dir
        (process / "fd").mkdir(parents=True)
        (process / "comm").write_text("nvidia-persiste\n")
        (process / "cgroup").write_text(f"0::{self.cgroup}\n")
        (process / "stat").write_text(
            f"{DAEMON_PID} (nvidia-persiste) S 1 {DAEMON_PID} {DAEMON_PID} 0 -1 "
            f"4194560 0 0 0 0 0 0 0 0 20 0 1 0 {DAEMON_START} 0 0\n"
        )
        (process / "exe").symlink_to(DAEMON_EXE)
        (process / "fd" / "3").symlink_to("/dev/nvidiactl")
        for bus_id in sorted(self.devices):
            self._fd(bus_id).symlink_to(self.devices[bus_id])
            self.held.add(bus_id)

    def _close(self, bus_id: str) -> None:
        self._fd(bus_id).unlink(missing_ok=True)
        self.held.discard(bus_id)

    def _exit_daemon(self) -> None:
        for bus_id in list(self.held):
            self._close(bus_id)
        self.teardown.clear()
        shutil.rmtree(self.daemon_dir)

    def running(self) -> bool:
        return self.daemon_dir.exists()

    def mode(self, bus_id: str) -> str:
        persistent = bus_id in self.held or bus_id in self.legacy
        return "Enabled" if persistent else "Disabled"

    def tick(self, _seconds: float) -> None:
        self.ticks += 1
        if self.teardown:
            self._close(self.teardown.pop(0))
            if not self.teardown:
                self._exit_daemon()

    def _nvidia_smi(self, argv: list[str], timeout: Any) -> CompletedProcess:
        if "--query-gpu=pci.bus_id,persistence_mode" in argv:
            report = "".join(
                f"{bus_id}, {self.mode(bus_id)}\n" for bus_id in sorted(self.devices)
            )
            return CompletedProcess(argv, 0, stdout=report, stderr="")
        if argv[-2:] == ["-pm", "0"]:
            if self.release_timeout:
                raise TimeoutExpired(argv, timeout)
            if self.release_error:
                return CompletedProcess(argv, 1, stdout="", stderr=self.release_error)
            targets = [argv[2]] if argv[1] == "-i" else sorted(self.devices)
            for bus_id in targets:
                if self.running():
                    self._close(bus_id)
                self.legacy.discard(bus_id)
            return CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected nvidia-smi call {argv}")

    def __call__(self, value, *, check=False, timeout=None, **kwargs):
        if value[:1] == ["nvidia-smi"]:
            self.commands.append(value)
            self.timeouts.append(timeout)
            return self._nvidia_smi(value, timeout)
        if value[:2] == ["kill", "--signal"] and value[3] == DAEMON_PID:
            if self.running() and value[2] == "KILL":
                self._exit_daemon()
            elif self.running() and not self.teardown:
                self._close(TARGET_BUS_ID)
                self.teardown = sorted(self.held)
                if not self.teardown:
                    self._exit_daemon()
        systemd_daemon = self.cgroup == SYSTEMD_CGROUP
        if value == ["systemctl", "stop", "nvidia-persistenced"] and self.running():
            if systemd_daemon:
                self._exit_daemon()
        if value == ["systemctl", "start", "nvidia-persistenced"] and systemd_daemon:
            if not self.running():
                self._start_daemon()
        restore_unit = "gpu-fault-gpu-persistence"
        if value == ["systemctl", "start", restore_unit] and restore_unit not in (
            self.active
        ):
            # nvidia-smi -pm 1: through the daemon when one runs, legacy
            # otherwise. A oneshot still active from boot does not run again.
            if self.running():
                for bus_id in sorted(set(self.devices) - self.held):
                    self._fd(bus_id).symlink_to(self.devices[bus_id])
                    self.held.add(bus_id)
            else:
                self.legacy = set(self.devices)
        return super().__call__(value, check=check, **kwargs)

    def stopped(self) -> list[str]:
        return [
            item[2]
            for item in self.commands
            if item[:2] == ["systemctl", "stop"] and not item[2].endswith(".timer")
        ]

    def releases(self) -> list[list[str]]:
        return [item for item in self.commands if "-pm" in item]


def manager(node: PersistenceNode, *, sleeper: Any = None) -> GpuServiceQuiesceManager:
    return GpuServiceQuiesceManager(
        state_dir=str(node.proc.parent / "quiesce"),
        services=SERVICES,
        processes=(),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        device_sweep_timeout_seconds=20,
        proc_root=str(node.proc),
        restore_command="/opt/gpu-fault/restore",
        runner=node,
        sleeper=sleeper or node.tick,
    )


def quiesce(
    service: GpuServiceQuiesceManager, *, whole_node: bool = False, **overrides: Any
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "incident_id": "incident-a",
        "workflow_request_id": "workflow-a",
        "target_device_paths": (
            set(DEVICES.values()) if whole_node else {DEVICES[TARGET_BUS_ID]}
        ),
        "target_pci_bus_ids": None if whole_node else [TARGET_BUS_ID],
        "whole_node": whole_node,
    }
    arguments.update(overrides)
    return service.quiesce(**arguments)


def test_daemon_regime_releases_only_the_target_through_the_daemon(tmp_path) -> None:
    node = PersistenceNode(tmp_path, daemon="hand", active=HYPERPOD_ACTIVE)
    sleeps: list[float] = []

    result = quiesce(manager(node, sleeper=sleeps.append))

    assert node.releases() == [RELEASE]
    assert node.held == {OTHER_BUS_ID}, "the daemon closed the target's device only"
    assert node.running(), "the daemon the site started is never signalled"
    assert not any(item[:1] == ["kill"] for item in node.commands), node.commands
    assert result["swept_device_holders"] == []
    assert result["persistence_mode"] == {
        "before_release": modes(target="Enabled", other="Enabled"),
        "after_release": modes(target="Disabled", other="Enabled"),
    }
    release = result["persistence_release"]
    assert release["scope"] == "targets"
    assert release["pci_bus_ids"] == [TARGET_BUS_ID]
    assert release["commands"] == [{"command": " ".join(RELEASE), "released": True}]
    assert release["complete"] is True
    assert release["daemon"]["running"] is True
    assert result["persistence_daemon"] == {
        "running": True,
        "pid": DAEMON_PID,
        "systemd_owned": False,
        "systemd_unit": None,
        "exe": DAEMON_EXE,
        "start_time_ticks": DAEMON_START,
    }
    assert "persistence_daemon_swept" not in result
    assert sleeps == [], "the release path adds no sleep"


def test_legacy_regime_clears_only_the_target_flag(tmp_path) -> None:
    node = PersistenceNode(tmp_path, daemon=None, legacy=True, active=HYPERPOD_ACTIVE)

    result = quiesce(manager(node))

    assert node.releases() == [RELEASE]
    assert node.legacy == {OTHER_BUS_ID}
    assert result["persistence_mode"]["after_release"] == modes(
        target="Disabled", other="Enabled"
    )
    assert result["persistence_release"]["complete"] is True
    assert result["persistence_release"]["daemon"]["running"] is False
    assert result["persistence_daemon"]["running"] is False
    assert result["swept_device_holders"] == []


def test_whole_node_quiesce_releases_every_gpu_in_one_call(tmp_path) -> None:
    node = PersistenceNode(tmp_path, daemon="hand", active=HYPERPOD_ACTIVE)

    result = quiesce(manager(node), whole_node=True)

    assert node.releases() == [["nvidia-smi", "-pm", "0"]]
    assert node.held == set() and node.running()
    assert result["persistence_mode"]["after_release"] == modes(
        target="Disabled", other="Disabled"
    )
    assert result["persistence_release"]["scope"] == "whole-node"
    assert result["persistence_release"]["complete"] is True
    assert result["swept_device_holders"] == []
    assert not any(item[:1] == ["kill"] for item in node.commands), node.commands


def test_systemd_daemon_unit_stays_up_after_a_release_through_it(tmp_path) -> None:
    node = PersistenceNode(tmp_path, daemon="systemd", active=EKS_ACTIVE)
    service = manager(node)

    result = quiesce(service)

    # The restore unit stops first, right before the release: whatever
    # restores the window then runs it again.
    assert node.stopped() == ["gpu-fault-gpu-persistence", "kubelet", "nvidia-dcgm"]
    assert node.commands.index(RELEASE) > node.commands.index(
        ["systemctl", "stop", "gpu-fault-gpu-persistence"]
    ), node.commands
    assert node.running() and node.held == {OTHER_BUS_ID}
    (kept,) = result["kept_services"]
    assert kept["service"] == "nvidia-persistenced"
    assert "released" in kept["reason"]
    assert result["active_services"] == [
        "nvidia-dcgm",
        "nvidia-persistenced",
        "gpu-fault-gpu-persistence",
        "kubelet",
    ]
    assert result["persistence_daemon"]["systemd_owned"] is True
    assert result["persistence_daemon"]["systemd_unit"] == "nvidia-persistenced.service"

    restored = service.restore(incident_id="incident-a")

    # gpu-fault-gpu-persistence (nvidia-smi -pm 1) re-enables the target
    # through the daemon that stayed up all along.
    assert ["systemctl", "start", "gpu-fault-gpu-persistence"] in node.commands
    assert node.held == set(DEVICES)
    assert restored["persistence_mode"] == {
        "after_restore": modes(target="Enabled", other="Enabled")
    }
    assert restored["persistence_daemon"]["running"] is True
    assert "persistence_daemon_restored" not in restored


def test_systemd_daemon_unit_is_stopped_when_the_release_fails(tmp_path) -> None:
    node = PersistenceNode(
        tmp_path,
        daemon="systemd",
        active=EKS_ACTIVE,
        release_error="Unable to set persistence mode: Insufficient Permissions",
    )
    service = manager(node)

    result = quiesce(service)

    assert "nvidia-persistenced" in node.stopped()
    assert result["kept_services"] == []
    release = result["persistence_release"]
    assert release["complete"] is False
    assert "exit status 1" in release["reason"]
    assert "Insufficient Permissions" in release["commands"][0]["error"]
    assert not node.running(), "the unit is stopped as before: the fallback"

    restored = service.restore(incident_id="incident-a")

    assert node.running(), "the unit the quiesce stopped is started again"
    assert "persistence_daemon_restored" not in restored


def test_a_stop_failing_after_the_release_still_restores_the_targets(tmp_path) -> None:
    """Started while still active, the restore unit's oneshot would not run:
    the target would stay released past the restore of a failed window."""

    node = PersistenceNode(tmp_path, daemon="hand", active=HYPERPOD_ACTIVE)
    node.fail_stop = "kubelet"
    service = manager(node)

    with pytest.raises(RuntimeError, match="stop failed"):
        quiesce(service)
    assert node.mode(TARGET_BUS_ID) == "Disabled", "released before kubelet failed"

    restored = service.restore(incident_id="incident-a")

    assert ["systemctl", "start", "gpu-fault-gpu-persistence"] in node.commands
    assert restored["persistence_mode"] == {
        "after_restore": modes(target="Enabled", other="Enabled")
    }
    assert node.running(), "the daemon was never signalled"


def test_fallback_sweep_waits_for_the_daemon_to_exit(tmp_path) -> None:
    node = PersistenceNode(
        tmp_path,
        daemon="hand",
        active=HYPERPOD_ACTIVE,
        release_error="Failed to set persistence mode",
    )

    result = quiesce(manager(node))

    assert ["kill", "--signal", "TERM", DAEMON_PID] in node.commands
    # The daemon closed the target's device at once and the other one a poll
    # later: a sweep that returned when the target's fd was gone left its
    # teardown running into VERIFY.
    assert not node.running(), "the sweep returned while the daemon was tearing down"
    assert node.ticks == 1
    assert not any(
        item[:3] == ["kill", "--signal", "KILL"] for item in node.commands
    ), "a daemon that exits within the sweep timeout is not killed"
    identity = {
        "pid": DAEMON_PID,
        "exe": DAEMON_EXE,
        "start_time_ticks": DAEMON_START,
        "systemd_owned": False,
        "systemd_unit": None,
    }
    assert result["persistence_daemon_swept"] == identity
    state = json.loads(Path(result["state_path"]).read_text(encoding="utf-8"))
    assert state["persistence_daemon_swept"] == identity
    assert result["persistence_daemon"]["running"] is False


def test_restore_reports_the_swept_daemon_and_never_relaunches_it(tmp_path) -> None:
    node = PersistenceNode(
        tmp_path,
        daemon="hand",
        active=HYPERPOD_ACTIVE,
        release_error="Failed to set persistence mode",
    )
    service = manager(node)
    quiesce(service)
    before = len(node.commands)

    restored = service.restore(incident_id="incident-a")

    assert restored["persistence_daemon_restored"] is False
    reason = restored["persistence_daemon_restore_reason"]
    assert DAEMON_PID in reason and "not relaunch" in reason
    assert restored["persistence_daemon"]["running"] is False
    assert not node.running(), "the restore never starts the daemon"
    issued = node.commands[before:]
    assert not any(item[0].endswith("nvidia-persistenced") for item in issued), issued
    assert ["systemctl", "start", "nvidia-persistenced"] not in issued
    # Persistence is back through the legacy flag only: that is the drift the
    # result names.
    assert restored["persistence_mode"] == {
        "after_restore": modes(target="Enabled", other="Enabled")
    }


def test_a_release_that_times_out_falls_back_and_spawns_nothing_more(tmp_path) -> None:
    node = PersistenceNode(
        tmp_path, daemon="hand", active=HYPERPOD_ACTIVE, release_timeout=True
    )

    result = quiesce(manager(node))

    release = result["persistence_release"]
    assert release["complete"] is False
    assert "did not answer" in release["reason"]
    smi = [item for item in node.commands if item[:1] == ["nvidia-smi"]]
    assert smi[-1] == RELEASE, (
        "nothing else is asked of a driver that stopped answering"
    )
    assert all(timeout is not None and timeout <= 30 for timeout in node.timeouts), (
        node.timeouts
    )
    assert result["persistence_mode"]["after_release"] == []
    assert "after_release" in result["persistence_mode"]["errors"]
    assert ["kill", "--signal", "TERM", DAEMON_PID] in node.commands
    assert result["quiesced"] is True


def test_unresolved_bus_ids_skip_the_release_and_keep_the_fallback(tmp_path) -> None:
    node = PersistenceNode(tmp_path, daemon="hand", active=HYPERPOD_ACTIVE)

    result = quiesce(manager(node), target_pci_bus_ids=None)

    assert node.releases() == []
    release = result["persistence_release"]
    assert release["complete"] is False
    assert "bus id" in release["reason"]
    assert ["kill", "--signal", "TERM", DAEMON_PID] in node.commands


def test_persistence_records_are_bounded(tmp_path) -> None:
    devices = {
        f"00000000:{index + 16:02X}:00.0": f"/dev/nvidia{index}" for index in range(20)
    }
    devices[TARGET_BUS_ID] = "/dev/nvidia99"
    node = PersistenceNode(
        tmp_path,
        daemon=None,
        legacy=True,
        active=HYPERPOD_ACTIVE,
        devices=devices,
        release_error="x" * 5000,
    )

    result = quiesce(manager(node), target_device_paths={"/dev/nvidia99"})

    assert len(result["persistence_mode"]["before_release"]) == 16
    assert len(result["persistence_mode"]["after_release"]) == 16
    assert len(result["persistence_release"]["commands"][0]["error"]) <= 300
    assert len(result["persistence_release"]["reason"]) <= 600
    assert len(json.dumps(result)) < 16_000


def driver_report(target_mode: str = "Disabled") -> str:
    """``nvidia-smi -q -x`` of that node: GPU-a is the target at bus 53 on
    /dev/nvidia1, released for the window; GPU-b keeps its persistence."""

    return f"""<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu id="{TARGET_BUS_ID}">
    <uuid>GPU-a</uuid>
    <minor_number>1</minor_number>
    <persistence_mode>{target_mode}</persistence_mode>
    <pci><pci_bus_id>{TARGET_BUS_ID}</pci_bus_id></pci>
  </gpu>
  <gpu id="{OTHER_BUS_ID}">
    <uuid>GPU-b</uuid>
    <minor_number>0</minor_number>
    <persistence_mode>Enabled</persistence_mode>
  </gpu>
</nvidia_smi_log>
"""


class DriverReport:
    """The node's ``nvidia-smi``: the XML report, nothing else to say.

    ``target_modes`` is the target's persistence mode in each successive
    report; the last one repeats.
    """

    def __init__(self, target_modes: tuple[str, ...] = ("Disabled",)) -> None:
        self.commands: list[list[str]] = []
        self.target_modes = target_modes

    def __call__(self, argv: list[str], **_: Any) -> CompletedProcess[str]:
        self.commands.append(list(argv))
        if argv[:3] == ["nvidia-smi", "-q", "-x"]:
            index = min(self.reports(), len(self.target_modes)) - 1
            report = driver_report(self.target_modes[index])
            return CompletedProcess(argv, 0, stdout=report, stderr="")
        return CompletedProcess(argv, 0, stdout="", stderr="")

    def reports(self) -> int:
        return sum(argv[:3] == ["nvidia-smi", "-q", "-x"] for argv in self.commands)


class CapturingManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def quiesce(self, **kwargs: Any) -> dict[str, bool]:
        self.calls.append(kwargs)
        return {"quiesced": True}


def quiesce_through_the_agent(
    tmp_path: Path, gpu_uuids: list[str]
) -> tuple[dict[str, Any], DriverReport]:
    runner = DriverReport()
    quiesce_manager = CapturingManager()
    proc = tmp_path / "agent-proc"
    proc.mkdir()
    agent = node_action_executor(
        tmp_path,
        "quiesce-bus-ids.db",
        allowed_operations={WorkflowOperation.QUIESCE_GPU_SERVICES},
        reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=quiesce_manager,
        runner=runner,
        proc_root=str(proc),
        sleep=lambda _seconds: None,
    )
    result = agent.execute(
        envelope(command(WorkflowOperation.QUIESCE_GPU_SERVICES, gpu_uuids=gpu_uuids))
    )
    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert len(quiesce_manager.calls) == 1, quiesce_manager.calls
    return quiesce_manager.calls[0], runner


def test_the_agent_releases_by_the_bus_id_the_driver_reports(tmp_path) -> None:
    call, runner = quiesce_through_the_agent(tmp_path, ["GPU-a"])

    assert call["target_device_paths"] == {"/dev/nvidia1"}, call
    assert call["target_pci_bus_ids"] == [TARGET_BUS_ID], (
        "the bus id of the target's own report entry, not its minor number"
    )
    assert call["whole_node"] is False
    assert runner.reports() == 1, "one report resolves the nodes and the bus ids"


def test_a_whole_node_quiesce_names_no_bus_ids(tmp_path) -> None:
    call, _runner = quiesce_through_the_agent(tmp_path, [])

    assert call["target_device_paths"] == {"/dev/nvidia0", "/dev/nvidia1"}, call
    assert call["target_pci_bus_ids"] is None
    assert call["whole_node"] is True


class PermitAgent(ResetOperationsMixin, ClientOperationsMixin):
    """The Agent's own device map, report and reset loop under the final boundary.

    The device map, the bus ids and the persistence modes come from the real
    mixins parsing ``DriverReport``; the permit, the fence and the holder
    samples are scripted (nothing holds a GPU).
    """

    def __init__(self, runner: DriverReport, *, refusal: str | None = None) -> None:
        self.runner = runner
        self.refusal = refusal
        self.gpu_device_path_finder = self._gpu_device_paths
        self.device_client_finder = lambda _targets: []
        self.device_client_samples = 1
        self.device_client_sample_interval_seconds = 0
        self.proc_root = None
        self.sleep = lambda _seconds: None
        self.now = lambda: NOW
        self.ownership_gate = SimpleNamespace(require=self.grant)
        self.ledger = SimpleNamespace(accept_fencing=lambda *_args: True)
        self.service_quiesce_enabled = False
        self.quiesce_manager = None
        self.calls: list[dict[str, Any]] = []

    def _run_checked(self, argv: list[str], *, timeout: int) -> Any:
        return self.runner(argv, timeout=timeout)

    def grant(self, value: NodeActionCommand) -> Any:
        if self.refusal is not None:
            raise OwnershipRefused(self.refusal)
        return sign_permit(
            challenge_fixture(command_sha256=command_identity(value)),
            KEY,
            allowed=True,
            reason="OK",
            now=NOW,
        )

    def validate_submission(self, _envelope: SignedNodeAction) -> None:
        return None

    @final_ownership_boundary
    def effect(self, _value: NodeActionCommand) -> dict[str, Any]:
        # The handler's own preflight read comes before the permit's.
        self.gpu_device_path_finder()
        self._run_reset_with_busy_retry(
            ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"],
            gpu_uuids=["GPU-a"],
            timeout=120,
            calls=self.calls,
        )
        return {"physical_calls": self.calls}


def run_reset(agent: PermitAgent) -> dict[str, Any]:
    return execute_with_final_ownership(
        agent,
        SignedNodeAction(command=command_fixture(), signature="unused"),
        agent.effect,
    )


def test_each_reset_call_records_the_target_persistence_its_permit_read() -> None:
    # Still persistent at the handler's preflight, released by the permit's read.
    runner = DriverReport(target_modes=("Enabled", "Disabled"))

    details = run_reset(PermitAgent(runner))

    assert len(details["physical_ownership_checks"]) == 1, details
    call = details["physical_calls"][0]
    assert call["outcome"] == "reset", call
    assert call["target_persistence"] == [
        {
            "gpu_uuid": "GPU-a",
            "pci_bus_id": TARGET_BUS_ID,
            "persistence_mode": "Disabled",
        }
    ], "the permit window's own read, not the preflight's"
    assert call["non_target_persistence_modes"] == {"Enabled": 1}, (
        "the other GPU kept its persistence through the window"
    )
    assert runner.reports() == 2, "the preflight's and the permit's, nothing more"


def test_a_refused_reset_call_still_records_what_its_window_read() -> None:
    runner = DriverReport(target_modes=("Enabled", "Disabled"))
    agent = PermitAgent(runner, refusal="OWNERSHIP_DRIFT")

    with pytest.raises(OwnershipRefused, match="OWNERSHIP_DRIFT"):
        run_reset(agent)

    [call] = agent.calls
    assert call["outcome"] == "refused: OWNERSHIP_DRIFT", call
    assert call["target_persistence"][0]["persistence_mode"] == "Disabled", (
        "read in the window before the refusal, not the preflight's"
    )
    assert runner.reports() == 2, "the refusal came after the window's read"
