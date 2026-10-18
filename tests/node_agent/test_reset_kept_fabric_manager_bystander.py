"""The Fabric Manager a per-GPU reset window kept no longer blocks the display-stack release.

Live (GF-REGIONAL-DESTR-001, plain EKS p5en, H200, AL2023 NVIDIA AMI, driver
580.178.04, ``nvidia_drm`` modeset=Y fbdev=Y): since batch 128 a per-GPU reset
window keeps ``nvidia-fabricmanager`` running. QUIESCE kept it and the
``nvidia-persistenced`` unit (the target released with ``nvidia-smi -pm 0``),
VERIFY_NO_GPU_CLIENTS passed, and ``nvidia-smi --gpu-reset`` was refused "In use
by another client" three times; the product escalated to RESTART_NODE. At the
last attempt no compute app ran, ``nvidia-persiste`` held ``/dev/nvidia1..7`` and
``/dev/nvidiactl``, ``nv-fabricmanage`` held ``/dev/nvidia-caps/nvidia-cap0``,
``nvidia_drm`` sat at refcount 0 and ``nvidia_modeset`` at 1 (used by
``nvidia_drm``) -- and no ``rmmod`` was attempted: the display-stack release
excused the persistence daemon only. On the same node a round earlier, with the
quiesce stopping Fabric Manager, the first reset was busy, the unload of the two
modules freed the GPU and the second reset passed.

The Fabric Manager the window kept is a bystander now, and only that one: the
window's own ``kept_services`` names it, the process runs in its unit's cgroup
and no container, and it holds nothing but the fabric's nodes. Any other holder
still blocks the unload, and the quiesce never signals Fabric Manager.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from typing import Any

import pytest

from gpu_fault.node_agent.late_ownership import classify_final_device_clients
from gpu_fault.node_agent.operations.holders import (
    display_release_bystander,
    fabric_manager_bystander,
    fabric_manager_process,
    systemd_unit_cgroup,
)
from gpu_fault.node_agent.quiesce import window_kept_services

from ._support import (
    GpuServiceQuiesceManager,
    NodeActionStatus,
    ServiceRunner,
    WorkflowOperation,
    command,
    envelope,
    no_device_clients,
    node_action_executor,
)

FM = "nvidia-fabricmanager"
PERSISTENCED = "nvidia-persistenced"
KEEP = {"fabric_manager": "keep"}
BUSY = "GPU 00000000:53:00.0: In use by another client"
TARGET = "GPU-8b50c0de-0000-4000-8000-000000000000"
GPUS = [TARGET] + [
    f"GPU-0000000{minor}-0000-4000-8000-000000000000" for minor in range(1, 8)
]
BUS_IDS = {
    uuid: f"00000000:{0x53 + 0x10 * minor:02X}:00.0" for minor, uuid in enumerate(GPUS)
}
# Above the kernel's pid_max, so never the pid of the test process itself (the
# holder scans skip their own pid).
DAEMON_PID = "7777771"
FM_PID = "7777772"
OTHER_PID = "7777773"
DAEMON_CGROUP = "/system.slice/nvidia-persistenced.service"
FM_CGROUP = "/system.slice/nvidia-fabricmanager.service"
CAP = "/dev/nvidia-caps/nvidia-cap0"
# What the live Fabric Manager holds with NVLink Inband: the fabric-management
# capability, the four NVSwitches and the NVLink node, never a GPU.
FM_DEVICES = [
    CAP,
    *(f"/dev/nvidia-nvswitch{index}" for index in range(4)),
    "/dev/nvidia-nvlink",
]
SERVICES = (
    FM,
    "nvidia-dcgm",
    PERSISTENCED,
    "gpu-fault-gpu-persistence",
    "gpu-fault-metrics-collector",
    "gpu-fault-host-collector",
    "kubelet",
)


def _device(uuid: str) -> str:
    return f"/dev/nvidia{GPUS.index(uuid)}"


class P5enNode(ServiceRunner):
    """systemd, ``nvidia-smi``, the display stack and two daemons on a fake ``/proc``.

    ``nvidia-persistenced`` (a systemd unit) holds ``/dev/nvidiactl`` and every
    GPU; ``nvidia-smi -i <bus id> -pm 0`` goes over its RPC and it closes that
    GPU alone. Fabric Manager runs in its unit's cgroup on ``fm_devices``; its
    unit's stop ends it unless ``stop_ends_fm`` is off (a process that outlives
    the stop). The reset is refused busy while ``nvidia_drm`` is loaded, or
    always with ``busy_forever``. Every call is recorded.
    """

    def __init__(
        self,
        tmp_path: Path,
        *,
        daemon: bool = True,
        fm_devices: list[str] | None = None,
        fm_cgroup: str = FM_CGROUP,
        fm_comm: str = "nv-fabricmanage",
        stop_ends_fm: bool = True,
        busy_forever: bool = False,
        active: set[str] | None = None,
    ) -> None:
        super().__init__(active=set(SERVICES) if active is None else active)
        self.proc = tmp_path / "proc"
        self.proc.mkdir()
        self.fm_devices = list(FM_DEVICES if fm_devices is None else fm_devices)
        self.fm_cgroup = fm_cgroup
        self.fm_comm = fm_comm
        self.stop_ends_fm = stop_ends_fm
        self.busy_forever = busy_forever
        self.loaded = {"nvidia_drm", "nvidia_modeset"}
        self.held: set[str] = set()
        self._write_modules()
        if daemon:
            self._start_daemon()
        self._start_fabric_manager()

    # -- /proc -------------------------------------------------------------

    def process(self, pid: str, comm: str, cgroup: str, devices: list[str]) -> Path:
        process = self.proc / pid
        (process / "fd").mkdir(parents=True)
        (process / "comm").write_text(f"{comm}\n", encoding="ascii")
        (process / "cgroup").write_text(f"0::{cgroup}\n", encoding="ascii")
        for index, device in enumerate(devices, start=3):
            (process / "fd" / str(index)).symlink_to(device)
        return process

    def _start_daemon(self) -> None:
        self.process(DAEMON_PID, "nvidia-persiste", DAEMON_CGROUP, ["/dev/nvidiactl"])
        for uuid in GPUS:
            self._hold(uuid)

    def _hold(self, uuid: str) -> None:
        fd = self.proc / DAEMON_PID / "fd" / str(20 + GPUS.index(uuid))
        if not fd.is_symlink():
            fd.symlink_to(_device(uuid))
        self.held.add(uuid)

    def _close(self, uuid: str) -> None:
        (self.proc / DAEMON_PID / "fd" / str(20 + GPUS.index(uuid))).unlink(
            missing_ok=True
        )
        self.held.discard(uuid)

    def _start_fabric_manager(self) -> None:
        if not self.fm_running():
            self.process(FM_PID, self.fm_comm, self.fm_cgroup, self.fm_devices)

    def _end(self, pid: str) -> None:
        shutil.rmtree(self.proc / pid, ignore_errors=True)
        if pid == DAEMON_PID:
            self.held.clear()

    def daemon_running(self) -> bool:
        return (self.proc / DAEMON_PID).exists()

    def fm_running(self) -> bool:
        return (self.proc / FM_PID).exists()

    def _write_modules(self) -> None:
        rows = []
        if "nvidia_drm" in self.loaded:
            rows.append("nvidia_drm 139264 0 - Live 0x0")
        if "nvidia_modeset" in self.loaded:
            used_by = "nvidia_drm," if "nvidia_drm" in self.loaded else "-"
            refcount = 1 if "nvidia_drm" in self.loaded else 0
            rows.append(f"nvidia_modeset 1503232 {refcount} {used_by} Live 0x0")
        rows.append("nvidia_uvm 4931584 0 - Live 0x0")
        rows.append("nvidia 117800960 30 nvidia_uvm,nvidia_modeset, Live 0x0")
        (self.proc / "modules").write_text("\n".join(rows) + "\n", encoding="ascii")

    # -- commands ----------------------------------------------------------

    def _report(self) -> str:
        gpus = "".join(
            f'<gpu id="{BUS_IDS[uuid]}"><uuid>{uuid}</uuid>'
            f"<minor_number>{GPUS.index(uuid)}</minor_number>"
            f"<persistence_mode>{self._mode(uuid)}</persistence_mode>"
            f"<pci><pci_bus_id>{BUS_IDS[uuid]}</pci_bus_id></pci></gpu>"
            for uuid in GPUS
        )
        return f'<?xml version="1.0" ?><nvidia_smi_log>{gpus}</nvidia_smi_log>'

    def _mode(self, uuid: str) -> str:
        return "Enabled" if uuid in self.held else "Disabled"

    def _nvidia_smi(self, argv: list[str]) -> CompletedProcess:
        stdout = ""
        if argv[1:3] == ["-q", "-x"]:
            stdout = self._report()
        elif argv[1] == "--query-gpu=pci.bus_id,persistence_mode":
            stdout = "".join(f"{BUS_IDS[uuid]}, {self._mode(uuid)}\n" for uuid in GPUS)
        elif argv[-2:] == ["-pm", "0"]:
            (uuid,) = [item for item in GPUS if BUS_IDS[item] == argv[2]]
            self._close(uuid)
        elif argv[1].startswith("--query-compute-apps="):
            stdout = ""
        elif argv[1] == "--query-gpu=uuid":
            stdout = "".join(f"{uuid}\n" for uuid in sorted(GPUS))
        elif argv[1].startswith("--query-gpu=uuid,pstate"):
            stdout = f"{argv[-1]}, P0, No, No, No\n"
        elif argv[1] == "--gpu-reset":
            if self.busy_forever or "nvidia_drm" in self.loaded:
                raise CalledProcessError(255, argv, stderr=BUSY)
        else:
            raise AssertionError(f"unexpected nvidia-smi call {argv}")
        return CompletedProcess(argv, 0, stdout=stdout, stderr="")

    def __call__(self, value, *, check=False, **kwargs):
        if value[:1] == ["nvidia-smi"]:
            self.commands.append(value)
            return self._nvidia_smi(value)
        if value[:1] in (["rmmod"], ["modprobe"]):
            self.commands.append(value)
            if value[0] == "rmmod":
                self.loaded.discard(value[1])
            else:
                self.loaded.update({"nvidia_drm", "nvidia_modeset"})
            self._write_modules()
            return CompletedProcess(value, 0, stdout="", stderr="")
        if value[:2] == ["kill", "--signal"]:
            self._end(value[3])
        if value == ["systemctl", "stop", FM] and self.stop_ends_fm:
            self._end(FM_PID)
        if value in (["systemctl", "start", FM], ["systemctl", "restart", FM]):
            self._start_fabric_manager()
        if value == ["systemctl", "stop", PERSISTENCED]:
            self._end(DAEMON_PID)
        if value == ["systemctl", "start", PERSISTENCED] and not self.daemon_running():
            self._start_daemon()
        restore_unit = "gpu-fault-gpu-persistence"
        if (
            value == ["systemctl", "start", restore_unit]
            and restore_unit not in self.active
            and self.daemon_running()
        ):
            for uuid in GPUS:
                self._hold(uuid)
        return super().__call__(value, check=check, **kwargs)

    # -- what happened -----------------------------------------------------

    def resets(self) -> list[int]:
        return [i for i, item in enumerate(self.commands) if "--gpu-reset" in item]

    def module_calls(self) -> list[list[str]]:
        return [item for item in self.commands if item[0] in {"rmmod", "modprobe"}]

    def fabric_manager_touched(self) -> list[list[str]]:
        return [
            item
            for item in self.commands
            if (item[:1] == ["systemctl"] and item[-1] == FM and item[1] != "is-active")
            or (item[:2] == ["kill", "--signal"] and item[3] == FM_PID)
        ]


def _manager(node: P5enNode, **overrides: Any) -> GpuServiceQuiesceManager:
    arguments: dict[str, Any] = {
        "state_dir": str(node.proc.parent / "quiesce"),
        "services": SERVICES,
        "processes": (),
        "failsafe_seconds": 30,
        "retry_seconds": 10,
        "settle_seconds": 0,
        "restore_settle_seconds": 0,
        # eks-managed: kubelet and containerd stay up.
        "protected_services": ("kubelet", "containerd"),
        "proc_root": str(node.proc),
        "restore_command": "/opt/gpu-fault/restore",
        "runner": node,
        "sleeper": lambda _seconds: None,
    }
    arguments.update(overrides)
    return GpuServiceQuiesceManager(**arguments)


def _agent(node: P5enNode, *, quiesce: bool = True, **overrides: Any):
    tmp_path = node.proc.parent
    if quiesce:
        overrides.setdefault("quiesce_manager", _manager(node))
    return node_action_executor(
        tmp_path,
        "p5en.db",
        allowed_operations={
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        },
        reset_enabled=True,
        service_quiesce_enabled=quiesce,
        runner=node,
        proc_root=str(node.proc),
        cluster_provider="eks-managed",
        sleep=lambda _seconds: None,
        **overrides,
    )


def _run(agent, operation: WorkflowOperation, parameters: dict | None = None):
    return agent.execute(
        envelope(
            command(
                operation,
                command_id=f"workflow/{operation.value}/node-a",
                parameters=parameters,
                gpu_uuids=[TARGET],
            )
        )
    )


def _window(agent, parameters: dict | None = KEEP):
    quiesced = _run(agent, WorkflowOperation.QUIESCE_GPU_SERVICES, parameters)
    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced.error
    verified = _run(agent, WorkflowOperation.VERIFY_NO_GPU_CLIENTS)
    assert verified.status is NodeActionStatus.SUCCEEDED, verified.error
    return quiesced


def _bystander(pid: str, comm: str, devices: str) -> dict[str, str]:
    return {"pid": pid, "process_name": comm, "devices": devices}


DAEMON_BYSTANDER = _bystander(
    DAEMON_PID,
    "nvidia-persiste",
    ",".join(
        sorted([f"/dev/nvidia{minor}" for minor in range(1, 8)] + ["/dev/nvidiactl"])
    ),
)
FM_BYSTANDER = _bystander(FM_PID, "nv-fabricmanage", CAP)


def test_the_live_window_releases_the_display_stack_beside_the_kept_fabric_manager(
    tmp_path: Path,
) -> None:
    """The live sequence end to end: FM and the daemon kept, the display stack
    idle, the first reset busy, both modules released, the retry resets."""

    node = P5enNode(tmp_path)
    agent = _agent(node)

    quiesced = _window(agent)
    held_in_window = set(node.held)
    reset = _run(agent, WorkflowOperation.RESET_GPU)
    restored = _run(agent, WorkflowOperation.RESTORE_GPU_SERVICES)

    assert [item["service"] for item in quiesced.details["kept_services"]] == [
        FM,
        PERSISTENCED,
    ], "the batch-128 window: Fabric Manager and the released daemon unit kept"
    assert held_in_window == set(GPUS) - {TARGET}, "the target alone was released"
    assert reset.status is NodeActionStatus.SUCCEEDED, reset.error
    assert reset.details["reset_attempts"] == 2
    assert reset.details["reset_busy_refusals"] == 1
    assert reset.details["released_kernel_modules"] == ["nvidia_drm", "nvidia_modeset"]
    assert reset.details["reloaded_kernel_modules"] == ["nvidia_modeset", "nvidia_drm"]
    assert reset.details["display_release_bystanders"] == [
        DAEMON_BYSTANDER,
        FM_BYSTANDER,
    ], "the record names both holders the release went ahead beside"
    assert [item.get("outcome") for item in reset.details["physical_calls"]] == [
        "busy",
        "released",
        "released",
        "reset",
    ]
    resets = node.resets()
    rmmods = [i for i, item in enumerate(node.commands) if item[0] == "rmmod"]
    assert [node.commands[i] for i in rmmods] == [
        ["rmmod", "nvidia_drm"],
        ["rmmod", "nvidia_modeset"],
    ]
    assert resets[0] < rmmods[0] < rmmods[1] < resets[1], node.commands
    assert node.module_calls()[-1] == ["modprobe", "nvidia_drm"]
    assert restored.status is NodeActionStatus.SUCCEEDED, restored.error
    assert FM not in restored.details["restored_services"]
    assert node.fabric_manager_touched() == [], "never stopped, started or signalled"
    assert node.fm_running(), "Fabric Manager runs on, untouched by the window"


def test_a_window_that_stopped_fabric_manager_releases_as_before(
    tmp_path: Path,
) -> None:
    """Round 9 on the same node: the quiesce stopped Fabric Manager, the daemon
    was the only holder, the unload freed the GPU at the second attempt."""

    node = P5enNode(tmp_path)
    agent = _agent(node)

    quiesced = _window(agent, None)
    reset = _run(agent, WorkflowOperation.RESET_GPU)
    restored = _run(agent, WorkflowOperation.RESTORE_GPU_SERVICES)

    assert FM in quiesced.details["active_services"]
    assert reset.status is NodeActionStatus.SUCCEEDED, reset.error
    assert reset.details["reset_attempts"] == 2
    assert reset.details["released_kernel_modules"] == ["nvidia_drm", "nvidia_modeset"]
    assert reset.details["display_release_bystanders"] == [DAEMON_BYSTANDER]
    assert FM in restored.details["restored_services"]
    assert node.fabric_manager_touched() == [
        ["systemctl", "stop", FM],
        ["systemctl", "start", FM],
    ]


def _assert_nothing_unloaded(node: P5enNode, reset, why: str) -> None:
    assert reset.status is NodeActionStatus.FAILED, why
    assert reset.retryable is False
    assert "In use by another client" in (reset.error or ""), reset.error
    assert len(node.resets()) == 3, "the busy retry budget is unchanged"
    assert node.module_calls() == [], why
    assert "released_kernel_modules" not in reset.details
    assert "display_release_bystanders" not in reset.details
    assert reset.details["reset_failed"] == [TARGET]


@pytest.mark.parametrize(
    ("scenario", "why"),
    [
        ("outlived-the-stop", "the window stopped its unit; the process outlived it"),
        ("not-configured", "a window that does not manage Fabric Manager kept none"),
        ("no-window", "without a quiesce window there is no record of a kept one"),
    ],
)
def test_a_fabric_manager_the_window_did_not_keep_is_not_excused(
    tmp_path: Path, scenario: str, why: str
) -> None:
    node = P5enNode(
        tmp_path,
        stop_ends_fm=scenario != "outlived-the-stop",
        daemon=scenario != "no-window",
    )
    if scenario == "no-window":
        agent = _agent(node, quiesce=False)
    else:
        services = SERVICES if scenario == "outlived-the-stop" else SERVICES[1:]
        agent = _agent(node, quiesce_manager=_manager(node, services=services))
        quiesced = _window(agent, None)
        assert FM not in window_kept_services(agent.quiesce_manager, "incident-a")
        kept = [item["service"] for item in quiesced.details["kept_services"]]
        assert kept == [PERSISTENCED], kept

    reset = _run(agent, WorkflowOperation.RESET_GPU)

    assert node.fm_running(), "the holder is still there"
    _assert_nothing_unloaded(node, reset, why)
    holders = reset.details["gpu_client_holders"]["device_holders"]
    assert FM_PID in [item["pid"] for item in holders], "it is reported, as a holder"


FM_HOLDS_A_TARGET_GPU = [CAP, "/dev/nvidia0"]


@pytest.mark.parametrize(
    ("devices", "cgroup", "comm", "why"),
    [
        (
            FM_HOLDS_A_TARGET_GPU,
            FM_CGROUP,
            "nv-fabricmanage",
            "it holds the target GPU",
        ),
        ([CAP, "/dev/nvidia3"], FM_CGROUP, "nv-fabricmanage", "it opened another GPU"),
        (
            [CAP, "/dev/nvidiactl"],
            FM_CGROUP,
            "nv-fabricmanage",
            "it talks to the driver",
        ),
        (
            [CAP, "/dev/nvidia-uvm"],
            FM_CGROUP,
            "nv-fabricmanage",
            "uvm is a compute handle",
        ),
        ([f"{CAP} (deleted)"], FM_CGROUP, "nv-fabricmanage", "a replaced device node"),
        (
            [CAP],
            "/kubepods.slice/kubepods-pod0123.slice/cri-containerd-abc.scope",
            "nv-fabricmanage",
            "a containerised Fabric Manager is not the kept unit",
        ),
        (
            [CAP],
            "/user.slice/user-0.slice/session-3.scope",
            "nv-fabricmanage",
            "a hand-started one is not the kept unit",
        ),
        ([CAP], FM_CGROUP, "nvidia-smi", "only the daemon itself, not a child"),
    ],
)
def test_only_the_kept_unit_on_the_fabric_nodes_is_a_bystander(
    tmp_path: Path, devices: list[str], cgroup: str, comm: str, why: str
) -> None:
    node = P5enNode(
        tmp_path, fm_devices=devices, fm_cgroup=cgroup, fm_comm=comm, busy_forever=True
    )
    # The sampled preflight is not what is judged here: the busy path is.
    agent = _agent(node, device_client_finder=no_device_clients)
    _window(agent)

    reset = _run(agent, WorkflowOperation.RESET_GPU)

    _assert_nothing_unloaded(node, reset, why)
    assert node.fabric_manager_touched() == [], "never signalled, whatever it holds"


def test_verify_fails_closed_on_a_fabric_manager_that_holds_the_target(
    tmp_path: Path,
) -> None:
    """VERIFY_NO_GPU_CLIENTS and the reset's preflight see a Fabric Manager on
    the target's device node like any client; the quiesce sweep leaves it
    alone (``fabric_manager_never_swept``) and nothing is reset or unloaded."""

    node = P5enNode(tmp_path, fm_devices=FM_HOLDS_A_TARGET_GPU)
    agent = _agent(node)

    quiesced = _run(agent, WorkflowOperation.QUIESCE_GPU_SERVICES, KEEP)
    verified = _run(agent, WorkflowOperation.VERIFY_NO_GPU_CLIENTS)
    reset = _run(agent, WorkflowOperation.RESET_GPU)

    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced.error
    (unswept,) = quiesced.details["unswept_device_holders"]
    assert unswept["pid"] == FM_PID
    assert unswept["reason"] == "fabric_manager_never_swept"
    assert quiesced.details["swept_device_holders"] == []
    assert verified.status is NodeActionStatus.FAILED
    assert f"{TARGET}:{FM_PID}:nv-fabricmanage" in (verified.error or "")
    assert verified.details["gpu_client_kind"] == "device"
    assert reset.status is NodeActionStatus.FAILED
    assert "GPU device clients are still active" in (reset.error or "")
    assert node.resets() == [] and node.module_calls() == []
    assert node.fabric_manager_touched() == [] and node.fm_running()


def test_an_unknown_holder_still_keeps_every_module_loaded(tmp_path: Path) -> None:
    node = P5enNode(tmp_path, busy_forever=True)
    node.process(
        OTHER_PID, "python3", "/system.slice/gpu-sampler.service", ["/dev/nvidiactl"]
    )
    agent = _agent(node)
    _window(agent)

    reset = _run(agent, WorkflowOperation.RESET_GPU)

    _assert_nothing_unloaded(node, reset, "a holder nobody can vouch for is a client")
    holders = reset.details["gpu_client_holders"]["device_holders"]
    assert sorted(item["pid"] for item in holders) == [DAEMON_PID, FM_PID, OTHER_PID]


def test_the_late_ownership_recheck_judges_target_holders_only(tmp_path: Path) -> None:
    """The final recheck samples the target's device node: the kept Fabric
    Manager on the fabric's nodes is never in it; on the target it refuses."""

    node = P5enNode(tmp_path)
    agent = _agent(node)
    _window(agent)

    assert classify_final_device_clients(agent, {TARGET}) == ([], [])

    (node.proc / FM_PID / "fd" / "30").symlink_to("/dev/nvidia0")
    blocking, query_only = classify_final_device_clients(agent, {TARGET})

    assert [(item["pid"], item["process_name"]) for item in blocking] == [
        (FM_PID, "nv-fabricmanage")
    ]
    assert query_only == []


# -- the predicates --------------------------------------------------------

LIVE_FM_HOLDER = {
    "pid": FM_PID,
    "process_name": "nv-fabricmanage",
    "devices": CAP,
    "deleted_device_node": False,
    "cgroup_paths": [FM_CGROUP],
    "kubernetes": False,
}


def test_the_live_fabric_manager_holder_is_a_bystander_of_a_window_that_kept_it() -> (
    None
):
    assert fabric_manager_bystander(LIVE_FM_HOLDER, {FM, PERSISTENCED}) is True
    assert fabric_manager_bystander(LIVE_FM_HOLDER, {PERSISTENCED}) is False
    assert fabric_manager_bystander(LIVE_FM_HOLDER, ()) is False
    every_node = {**LIVE_FM_HOLDER, "devices": ",".join(sorted(FM_DEVICES))}
    assert fabric_manager_bystander(every_node, {FM}) is True
    assert display_release_bystander(LIVE_FM_HOLDER, ["/dev/nvidia0"], {FM}) is True
    assert display_release_bystander(LIVE_FM_HOLDER, ["/dev/nvidia0"]) is False, (
        "the persistence rule alone never excuses Fabric Manager"
    )
    for change in (
        {"devices": ""},
        {"devices": f"{CAP},/dev/nvidia-modeset"},
        {"kubernetes": True},
        {"deleted_device_node": True},
        {"cgroup_paths": []},
        {"process_name": "nvidia-persiste"},
    ):
        assert fabric_manager_bystander({**LIVE_FM_HOLDER, **change}, {FM}) is False, (
            change
        )


def test_the_unit_cgroup_reads_cgroup_v1_and_v2() -> None:
    unit = "nvidia-fabricmanager"
    assert systemd_unit_cgroup([FM_CGROUP], unit) is True
    # cgroup v1: some controllers place the process in the parent slice.
    assert systemd_unit_cgroup(["/system.slice", FM_CGROUP], unit) is True
    assert systemd_unit_cgroup(["/system.slice"], unit) is False
    assert systemd_unit_cgroup([FM_CGROUP, DAEMON_CGROUP], unit) is False
    assert (
        systemd_unit_cgroup(["/user.slice/user-0.slice/session-3.scope"], unit) is False
    )
    assert systemd_unit_cgroup([f"{FM_CGROUP}.d"], unit) is False


def test_a_fabric_manager_is_recognised_by_name_or_by_its_unit() -> None:
    assert fabric_manager_process(LIVE_FM_HOLDER) is True
    assert fabric_manager_process({"process_name": "nv-fabricmanage"}) is True
    assert (
        fabric_manager_process({"process_name": "helper", "cgroup_paths": [FM_CGROUP]})
        is True
    )
    assert (
        fabric_manager_process(
            {"process_name": "nvidia-persiste", "cgroup_paths": [DAEMON_CGROUP]}
        )
        is False
    )


def test_the_window_record_is_read_without_raising(tmp_path: Path) -> None:
    node = P5enNode(tmp_path)
    manager = _manager(node)

    assert window_kept_services(manager, "incident-a") == frozenset(), "no window"
    assert window_kept_services(object(), "incident-a") == frozenset()  # type: ignore[arg-type]
    manager.quiesce(
        incident_id="incident-a",
        workflow_request_id="workflow-a",
        target_device_paths={"/dev/nvidia0"},
        keep_fabric_manager=True,
    )
    assert window_kept_services(manager, "incident-a") == frozenset({FM})
    (state_path,) = (node.proc.parent / "quiesce").glob("quiesce-*.json")
    state_path.write_text("{not json", encoding="utf-8")
    assert window_kept_services(manager, "incident-a") == frozenset(), "unreadable"


# -- the quiesce never signals Fabric Manager (or NVLSM) -------------------

KUBERNETES_CGROUP = "/kubepods.slice/kubepods-pod0123.slice/cri-containerd-abc.scope"


@pytest.mark.parametrize(
    ("route", "overrides", "workload_cgroups", "cgroup", "comm"),
    [
        ("workload cgroup", {}, {"/system.slice"}, FM_CGROUP, "nv-fabricmanage"),
        (
            "kubernetes container",
            {"sweep_kubernetes_device_holders": True},
            None,
            KUBERNETES_CGROUP,
            "nv-fabricmanage",
        ),
        # NVL5 (HGX B200/B300): the unit's start script runs NVLSM in its cgroup.
        ("nvlsm, workload cgroup", {}, {"/system.slice"}, FM_CGROUP, "nvlsm"),
        (
            "nvlsm by name, kubernetes container",
            {"sweep_kubernetes_device_holders": True},
            None,
            KUBERNETES_CGROUP,
            "nvlsm",
        ),
        (
            "nvlsm by name, in the workload's own cgroup",
            {},
            {"/kubepods.slice/kubepods-pod0123.slice"},
            KUBERNETES_CGROUP,
            "nvlsm",
        ),
        (
            "any process of the unit",
            {"sweep_kubernetes_device_holders": True},
            {"/system.slice"},
            FM_CGROUP,
            "fm-helper",
        ),
    ],
)
def test_the_device_sweep_never_signals_fabric_manager(
    tmp_path: Path,
    route: str,
    overrides: dict[str, Any],
    workload_cgroups: set[str] | None,
    cgroup: str,
    comm: str,
) -> None:
    node = P5enNode(
        tmp_path,
        daemon=False,
        fm_devices=FM_HOLDS_A_TARGET_GPU,
        fm_cgroup=cgroup,
        fm_comm=comm,
    )
    manager = _manager(node, services=("kubelet",), **overrides)

    result = manager.quiesce(
        incident_id="incident-a",
        workflow_request_id="workflow-a",
        target_device_paths={"/dev/nvidia0"},
        workload_cgroup_paths=workload_cgroups,
    )

    assert result["swept_device_holders"] == [], route
    (unswept,) = result["unswept_device_holders"]
    assert unswept["pid"] == FM_PID
    assert unswept["reason"] == "fabric_manager_never_swept"
    assert node.fabric_manager_touched() == [] and node.fm_running(), route


@pytest.mark.parametrize("name", ["nv-fabricmanage", "nvlsm"])
@pytest.mark.parametrize("setting", ["processes", "device_sweep_processes"])
def test_a_quiesce_that_would_signal_fabric_manager_by_name_is_refused(
    tmp_path: Path, name: str, setting: str
) -> None:
    """GPU_FAULT_QUIESCE_PROCESSES (``pkill --exact`` TERM) and
    GPU_FAULT_DEVICE_SWEEP_PROCESSES (the sweep whitelist) name neither Fabric
    Manager nor NVLSM: the Agent refuses the configuration at construction."""

    node = P5enNode(tmp_path)

    with pytest.raises(ValueError, match="never signals Fabric Manager or NVLSM"):
        _manager(node, **{setting: (name, "nv-hostengine")})


def test_nvlsm_is_part_of_the_fabric_by_name_or_by_its_unit() -> None:
    assert fabric_manager_process({"process_name": "nvlsm"}) is True, "by name"
    assert (
        fabric_manager_process({"process_name": "nvlsm", "cgroup_paths": [FM_CGROUP]})
        is True
    ), "in the unit"
    assert fabric_manager_process({"process_name": "nvlsm-bug-repo"}) is False, (
        "an exact comm, not a prefix"
    )
    nvlsm = {**LIVE_FM_HOLDER, "process_name": "nvlsm"}
    assert fabric_manager_bystander(nvlsm, {FM}) is False, (
        "only Fabric Manager itself is a display-release bystander"
    )
