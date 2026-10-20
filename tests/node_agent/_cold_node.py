"""An 8-GPU p5en in its quiesce window, on a simulated clock.

Live (gpu-eks-cluster, ip-10-0-45-43, p5en.48xlarge, 8x H200, AL2023 NVIDIA
AMI, GF-REGIONAL-DESTR-010): the quiesce had persistence mode off. The reads
before the busy fabric reset's display-stack release answered at once; the
first ``nvidia-smi`` after ``rmmod nvidia_drm`` outlived its 15 s, and the
``modprobe nvidia_drm`` after it logged ``Loading driver`` about 2.2 s apart
per GPU. Modelled here: the DRM device ``nvidia_drm`` registers per GPU is
what keeps a GPU initialised, and with it gone every ``nvidia-smi`` initialises
all eight GPUs one after another (tearing them down again on exit), as does
``modprobe nvidia_drm``.

``ColdNode`` is that node as the Agent's runner, ``/proc`` and
``/sys/class/drm`` see it: every call spends simulated time from those costs, a
call that outlives its ``timeout`` is killed the way ``subprocess.run`` kills
it (the GPU initialisation in flight finishes first, then ``TimeoutExpired``),
and a ``modprobe nvidia_drm`` killed mid-load leaves the GPUs whose probe had
not started without a DRM device -- ``request_irq() failed (-4)``, the module
still ``Live``. With the bounds of the release the live Agent ran, it ends as
the live node did: the reset refused ``OWNERSHIP_FINAL_SAFETY_CHANGED``, six
render nodes, ``0000:a5:00.0`` and ``0000:59:00.0`` without one. Nothing sleeps
for real.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess, TimeoutExpired
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.late_ownership import (
    OWNERSHIP_PROTOCOL,
    OwnershipChallenge,
    OwnershipRefused,
    command_identity,
    sign_permit,
)
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    SignedNodeAction,
    sign_node_action,
)
from tests.node_agent._support import SECRET, Quiesced, node_action_executor
from tests.regional.test_late_ownership_agent_gate import NOW

#: The driver's probe order (minor numbers 0..7); live the last two probed,
#: 0000:a5:00.0 and 0000:59:00.0, were the ones left without a DRM device.
GPU_BUS_IDS = (
    "0000:53:00.0",
    "0000:64:00.0",
    "0000:75:00.0",
    "0000:86:00.0",
    "0000:97:00.0",
    "0000:b8:00.0",
    "0000:a5:00.0",
    "0000:59:00.0",
)
GPU_UUIDS = [f"GPU-{index}" for index in range(len(GPU_BUS_IDS))]
#: Live: ``[nvidia-drm] Loading driver`` about 2.2 s apart per GPU.
INIT_SECONDS = 2.2
QUERY_SECONDS = 0.3
UNLOAD_SECONDS_PER_GPU = 0.4
BUSY_SECONDS = 1.0
RESET_SECONDS = 20.0
PERMIT_ROUND_TRIP_SECONDS = 0.5
#: The control plane's node-action command lifetime (``command_ttl``).
COMMAND_TTL = timedelta(minutes=2)
BUSY = "GPU 00000000:59:00.0: In use by another client"
NVIDIA_DRIVER = "nvidia"


class ColdNode:
    """``nvidia-smi``, ``rmmod`` and ``modprobe`` of a node whose GPUs are cold.

    ``hung_queries``: every read-only ``nvidia-smi`` query hangs (a wedged
    driver). ``failing_probes``: one entry per later ``modprobe``, the GPUs
    whose DRM probe fails on that load for a reason of their own.
    ``drm_users``: DRM clients holding ``nvidia_drm`` (its refcount).
    ``load_seconds``: one entry per later ``modprobe``, the seconds each GPU's
    probe takes on that load (a slower driver than the measured 2.2 s).
    ``after_unload`` / ``after_load``: what happens on the node right after an
    ``rmmod`` (with the module) or a ``modprobe`` that loaded ``nvidia_drm``.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.clock = 0.0
        self.proc = tmp_path / "proc"
        self.sys = tmp_path / "sys"
        self.drm_class = self.sys / "class" / "drm"
        self.loaded = {"nvidia_drm", "nvidia_modeset"}
        self.drm: list[str] = list(GPU_BUS_IDS)
        self.drm_users = 0
        self.compute_apps = ""
        self.hung_queries = False
        self.failing_probes: list[set[str]] = []
        self.load_seconds: list[float] = []
        self.after_unload: Callable[[str], None] | None = None
        self.after_load: Callable[[], None] | None = None
        self.calls: list[dict[str, Any]] = []
        self.resets_spawned = 0
        for bus_id in GPU_BUS_IDS:
            (self.proc / "driver" / "nvidia" / "gpus" / bus_id).mkdir(parents=True)
            device = self.sys / "devices" / "pci0000:00" / bus_id
            device.mkdir(parents=True)
            (device / "driver").symlink_to(
                self.sys / "bus" / "pci" / "drivers" / "nvidia"
            )
        framebuffer = self.sys / "devices" / "platform" / "simple-framebuffer.0"
        framebuffer.mkdir(parents=True)
        (framebuffer / "driver").symlink_to(
            self.sys / "bus" / "platform" / "drivers" / "simple-framebuffer"
        )
        self.write()

    # -- the node's state as /proc and /sys show it ---------------------------
    def write(self) -> None:
        rows = []
        if "nvidia_drm" in self.loaded:
            rows.append(f"nvidia_drm 139264 {self.drm_users} - Live 0x0")
        if "nvidia_modeset" in self.loaded:
            users = "nvidia_drm," if "nvidia_drm" in self.loaded else "-"
            count = 1 if "nvidia_drm" in self.loaded else 0
            rows.append(f"nvidia_modeset 1503232 {count} {users} Live 0x0")
        rows.append("nvidia_uvm 4931584 0 - Live 0x0")
        rows.append("nvidia 117800960 30 nvidia_uvm,nvidia_modeset, Live 0x0")
        (self.proc / "modules").write_text("\n".join(rows) + "\n", encoding="ascii")
        if self.drm_class.exists():
            shutil.rmtree(self.drm_class)
        self.drm_class.mkdir(parents=True)
        # card0 is the firmware framebuffer's (simpledrm), never the GPUs'.
        self._drm_node(
            "card0", self.sys / "devices" / "platform" / "simple-framebuffer.0"
        )
        if "nvidia_drm" in self.loaded:
            for index, bus_id in enumerate(self.drm):
                device = self.sys / "devices" / "pci0000:00" / bus_id
                self._drm_node(f"card{index + 1}", device)
                self._drm_node(f"renderD{128 + index}", device)

    def _drm_node(self, name: str, device: Path) -> None:
        node = self.drm_class / name
        node.mkdir()
        (node / "device").symlink_to(device)

    # -- what happens on the node meanwhile ----------------------------------
    def after_unload_of(self, module: str | None, action: Callable[[], None]) -> None:
        """Run ``action`` once ``module`` (any module when None) is unloaded."""

        def hook(unloaded: str) -> None:
            if module is None or unloaded == module:
                action()

        self.after_unload = hook

    def hang(self) -> None:
        """The driver stops answering read-only queries."""

        self.hung_queries = True

    def start_compute_app(self, gpu_uuid: str, pid: str = "4242") -> None:
        self.compute_apps = f"{gpu_uuid}, {pid}, python\n"

    def open_device(self, minor: int, pid: str = "4242") -> None:
        """A process that is no query opens ``/dev/nvidia<minor>``."""

        process = self.proc / pid
        (process / "fd").mkdir(parents=True)
        (process / "fd" / "3").symlink_to(f"/dev/nvidia{minor}")
        (process / "comm").write_text("python\n", encoding="ascii")

    def hold_drm(self) -> None:
        """A DRM client opens a render node: ``nvidia_drm`` is in use."""

        self.drm_users = 1
        self.write()

    def render_nodes(self) -> int:
        return sum(
            1 for item in self.drm_class.iterdir() if item.name.startswith("renderD")
        )

    def cold(self) -> int:
        """GPUs nothing keeps initialised: persistence is off in the window, so
        only a DRM device of the loaded display stack holds a GPU."""

        held = set(self.drm) if "nvidia_drm" in self.loaded else set()
        return len(GPU_BUS_IDS) - len(held)

    # -- the simulated clock --------------------------------------------------
    def now(self) -> Any:
        return NOW + timedelta(seconds=self.clock)

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def _spend(self, cmd: list[str], cost: float, timeout: float) -> None:
        if cost <= timeout:
            self.clock += cost
            return
        # subprocess.run SIGKILLs the child at its timeout and waits for it:
        # a child inside one GPU's initialisation exits once that GPU is up.
        remainder = (INIT_SECONDS - (timeout - QUERY_SECONDS) % INIT_SECONDS) % (
            INIT_SECONDS
        )
        self.clock += timeout + (remainder if self.cold() else 0.0)
        self.calls[-1]["killed"] = True
        raise TimeoutExpired(cmd, timeout)

    # -- the runner -----------------------------------------------------------
    def __call__(self, cmd: list[str], *, timeout: float, **_: Any) -> Any:
        self.calls.append(
            {"at": round(self.clock, 1), "command": " ".join(cmd), "timeout": timeout}
        )
        if cmd[0] == "rmmod":
            return self._rmmod(cmd)
        if cmd[0] == "modprobe":
            return self._modprobe(cmd, timeout)
        if "--gpu-reset" in cmd:
            return self._reset(cmd, timeout)
        return self._query(cmd, timeout)

    def _query(self, cmd: list[str], timeout: float) -> CompletedProcess[str]:
        cost = QUERY_SECONDS + INIT_SECONDS * self.cold()
        self._spend(cmd, float("inf") if self.hung_queries else cost, timeout)
        if "--query-compute-apps=gpu_uuid,pid,process_name" in cmd:
            stdout = self.compute_apps
        elif cmd[:3] == ["nvidia-smi", "-q", "-x"]:
            stdout = self._report()
        elif "--query-gpu=uuid" in cmd:
            stdout = "".join(f"{uuid}\n" for uuid in GPU_UUIDS)
        elif "--query-gpu=uuid,index" in cmd:
            stdout = "".join(f"{uuid}, {i}\n" for i, uuid in enumerate(GPU_UUIDS))
        else:
            stdout = ""
        return CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    def _report(self) -> str:
        gpus = "".join(
            f'<gpu id="0000{bus_id}"><uuid>{uuid}</uuid>'
            f"<minor_number>{minor}</minor_number>"
            f"<pci><pci_bus_id>0000{bus_id}</pci_bus_id></pci>"
            "<persistence_mode>Disabled</persistence_mode></gpu>"
            for minor, (uuid, bus_id) in enumerate(zip(GPU_UUIDS, GPU_BUS_IDS))
        )
        return f"<nvidia_smi_log>{gpus}</nvidia_smi_log>"

    def _reset(self, cmd: list[str], timeout: float) -> CompletedProcess[str]:
        if "nvidia_drm" in self.loaded and self.drm:
            self._spend(cmd, BUSY_SECONDS, timeout)
            raise CalledProcessError(255, cmd, stderr=BUSY)
        self._spend(cmd, RESET_SECONDS, timeout)
        self.resets_spawned += 1
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    def _rmmod(self, cmd: list[str]) -> CompletedProcess[str]:
        module = cmd[1]
        if module == "nvidia_drm" and self.drm_users:
            raise CalledProcessError(
                1, cmd, stderr="rmmod: ERROR: Module nvidia_drm is in use"
            )
        if module == "nvidia_drm":
            self.clock += UNLOAD_SECONDS_PER_GPU * len(self.drm)
            self.drm = []
        else:
            self.clock += 0.2
        self.loaded.discard(module)
        self.write()
        if self.after_unload is not None:
            self.after_unload(module)
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    def _modprobe(self, cmd: list[str], timeout: float) -> CompletedProcess[str]:
        if cmd != ["modprobe", "nvidia_drm"]:
            raise AssertionError(f"unexpected modprobe {cmd}")
        if "nvidia_drm" in self.loaded:
            self.clock += 0.05
            return CompletedProcess(cmd, 0, stdout="", stderr="")
        failing = self.failing_probes.pop(0) if self.failing_probes else set()
        probe = self.load_seconds.pop(0) if self.load_seconds else INIT_SECONDS
        elapsed = 0.1  # nvidia_modeset first, then one probe per GPU
        self.loaded.update({"nvidia_drm", "nvidia_modeset"})
        self.drm = []
        for bus_id in GPU_BUS_IDS:
            if elapsed >= timeout:
                # SIGKILL is pending: the probe's killable wait (kthread_create
                # behind request_irq) fails at once with -EINTR.
                continue
            elapsed += probe
            if bus_id not in failing:
                self.drm.append(bus_id)
        self.clock += elapsed
        self.write()
        if self.after_load is not None:
            self.after_load()
        if elapsed > timeout:
            self.calls[-1]["killed"] = True
            raise TimeoutExpired(cmd, timeout)
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    # -- what the tests read --------------------------------------------------
    def killed(self) -> list[str]:
        return [item["command"] for item in self.calls if item.get("killed")]

    def timeouts(self, prefix: str) -> list[float]:
        return [
            item["timeout"] for item in self.calls if item["command"].startswith(prefix)
        ]


def fabric_reset(*, guarded: bool = True) -> SignedNodeAction:
    """RESET_ALL_GPUS_NVSWITCHES as the control plane sends it: two minutes,
    behind the final ownership boundary unless ``guarded`` is off."""

    return _signed(
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES, list(GPU_UUIDS), guarded=guarded
    )


def gpu_reset(gpu_uuid: str, *, guarded: bool = False) -> SignedNodeAction:
    """RESET_GPU of one GPU, on the node's clock; unguarded unless asked."""

    return _signed(WorkflowOperation.RESET_GPU, [gpu_uuid], guarded=guarded)


def _signed(
    operation: WorkflowOperation, gpu_uuids: list[str], *, guarded: bool
) -> SignedNodeAction:
    value = NodeActionCommand(
        command_id=f"workflow/{operation.value}/node-a",
        workflow_request_id="workflow-a",
        incident_id="incident-a",
        fencing_token=1,
        operation=operation,
        node_id="node-a",
        agent_generation=1,
        gpu_uuids=gpu_uuids,
        issued_at=NOW,
        expires_at=NOW + COMMAND_TTL,
        ownership_guard=OWNERSHIP_PROTOCOL if guarded else None,
    )
    return SignedNodeAction(command=value, signature=sign_node_action(value, SECRET))


def cold_agent(
    tmp_path: Path,
    node: ColdNode,
    monkeypatch: pytest.MonkeyPatch,
    *,
    guarded: bool = True,
    operation: WorkflowOperation = WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
) -> Any:
    """The Agent with the live defaults (three holder samples 2 s apart), the
    final ownership boundary armed and a control plane that answers each
    challenge after one round trip -- and, as the real gate does, refuses one
    asked for after the command expired."""

    fabric = operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
    agent = node_action_executor(
        tmp_path,
        "cold.db",
        allowed_operations={operation},
        reset_enabled=True,
        fabric_reset_enabled=fabric,
        service_quiesce_enabled=fabric,
        quiesce_manager=Quiesced() if fabric else None,
        now=node.now,
        runner=node,
        require_final_ownership=guarded,
        agent_generation=1,
        proc_root=str(node.proc),
        sleep=node.sleep,
    )
    agent.drm_class_root = node.drm_class
    if not guarded:
        return agent
    sequence = {"value": 0}

    def require(value: NodeActionCommand, **_: Any) -> Any:
        node.clock += PERMIT_ROUND_TRIP_SECONDS
        if node.now() >= value.expires_at:
            raise OwnershipRefused("OWNERSHIP_COMMAND_EXPIRED")
        sequence["value"] += 1
        challenge = OwnershipChallenge(
            command_id=value.command_id,
            workflow_id=value.workflow_request_id,
            incident_id=value.incident_id,
            node_id=value.node_id,
            boot_id="cold-node-boot",
            agent_generation=value.agent_generation or 1,
            fencing_token=value.fencing_token,
            command_sha256=command_identity(value),
            nonce=f"{sequence['value']:064x}",
            sequence=sequence["value"],
            boundary="AGENT_PRE_SPAWN",
            expires_at=min(value.expires_at, node.now() + timedelta(seconds=90)),
        )
        return sign_permit(challenge, SECRET, allowed=True, reason="OK", now=node.now())

    monkeypatch.setattr(agent.ownership_gate, "require", require)
    return agent
