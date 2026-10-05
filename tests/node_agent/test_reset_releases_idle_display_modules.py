"""A busy reset with no visible holder unloads the idle display stack first.

Live (gpu-eks-cluster, XID 46 on ip-10-0-41-164): the quiesce had stopped
every service and container, no process held any NVIDIA device node, and the
reset still failed three times "In use by another client". The holder evidence
named the only client there was: ``nvidia_modeset refcount=1 used_by
nvidia_drm`` -- the DRM device ``nvidia_drm`` registers per GPU. The agent now
unloads ``nvidia_drm`` and ``nvidia_modeset`` on the first busy refusal when
nothing else holds the GPU, retries at once, and loads the stack again after
the reset, whatever its outcome; the result says what it unloaded and reloaded.
A visible holder (a process with a device node open) still means no module
is touched: that client is reported, not pulled from under.
"""

from __future__ import annotations

from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess

from gpu_fault.node_agent.operations.holders import idle_display_modules
from gpu_fault.node_agent.protocol import NodeActionStatus
from tests.node_agent._support import (
    WorkflowOperation,
    command,
    envelope,
    no_device_clients,
    node_action_executor,
)

BUSY = "GPU 00000000:59:00.0: In use by another client"
MODULES_WITH_IDLE_DRM = (
    "nvidia_drm 94208 0 -\n"
    "nvidia_modeset 1327104 1 nvidia_drm\n"
    "nvidia_uvm 1507328 0 -\n"
    "nvidia 8933376 2 nvidia_uvm,nvidia_modeset\n"
)


class DisplayStackRunner:
    """The reset is busy until ``nvidia_drm`` is unloaded; records every call."""

    def __init__(self, *, modules: Path, busy_forever: bool = False) -> None:
        self.modules = modules
        self.busy_forever = busy_forever
        self.commands: list[list[str]] = []
        self.loaded = {"nvidia_drm", "nvidia_modeset"}

    def __call__(self, cmd, **_):
        self.commands.append(cmd)
        if cmd[0] == "rmmod":
            self.loaded.discard(cmd[1])
            self._write()
            return CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "modprobe":
            self.loaded.update({"nvidia_drm", "nvidia_modeset"})
            self._write()
            return CompletedProcess(cmd, 0, stdout="", stderr="")
        if "--gpu-reset" in cmd and (self.busy_forever or "nvidia_drm" in self.loaded):
            raise CalledProcessError(255, cmd, stderr=BUSY)
        stdout = (
            "GPU-a\n"
            if "--query-gpu=uuid" in cmd
            else "GPU-a, 0\n"
            if "--query-gpu=uuid,index" in cmd
            else ""
        )
        return CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    def _write(self) -> None:
        rows = []
        if "nvidia_drm" in self.loaded:
            rows.append("nvidia_drm 94208 0 -")
        if "nvidia_modeset" in self.loaded:
            users = "nvidia_drm" if "nvidia_drm" in self.loaded else "-"
            rows.append(f"nvidia_modeset 1327104 {1 if users != '-' else 0} {users}")
        rows.append("nvidia 8933376 2 nvidia_uvm,nvidia_modeset")
        self.modules.write_text("\n".join(rows) + "\n", encoding="ascii")


def _proc(tmp_path: Path, *, holder: bool = False) -> Path:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "modules").write_text(MODULES_WITH_IDLE_DRM, encoding="ascii")
    if holder:
        process = proc / "4242"
        (process / "fd").mkdir(parents=True)
        (process / "fd" / "3").symlink_to("/dev/nvidiactl")
        (process / "comm").write_text("python\n", encoding="ascii")
        (process / "cgroup").write_text("0::/system.slice\n", encoding="ascii")
    return proc


def _agent(tmp_path: Path, runner, proc: Path):
    return node_action_executor(
        tmp_path,
        "display.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        runner=runner,
        proc_root=str(proc),
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        sleep=lambda _: None,
    )


def test_idle_display_modules_are_named_in_unload_order(tmp_path: Path) -> None:
    proc = _proc(tmp_path)
    assert idle_display_modules(proc) == ["nvidia_drm", "nvidia_modeset"]

    (proc / "modules").write_text(
        "nvidia_drm 94208 2 -\nnvidia_modeset 1327104 1 nvidia_drm\n", encoding="ascii"
    )
    assert idle_display_modules(proc) == [], "a DRM client holds the device: no unload"

    (proc / "modules").write_text(
        "nvidia_modeset 1327104 1 some_other_module\n", encoding="ascii"
    )
    assert idle_display_modules(proc) == [], "a module this code never unloads uses it"

    (proc / "modules").write_text("nvidia_uvm 1507328 0 -\n", encoding="ascii")
    assert idle_display_modules(proc) == [], "compute modules are never candidates"


def test_the_busy_reset_unloads_the_display_stack_retries_and_reloads(
    tmp_path: Path,
) -> None:
    proc = _proc(tmp_path)
    runner = DisplayStackRunner(modules=proc / "modules")
    agent = _agent(tmp_path, runner, proc)

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    resets = [i for i, item in enumerate(runner.commands) if "--gpu-reset" in item]
    assert len(resets) == 2, "busy once, then the first attempt that could succeed"
    rmmods = [i for i, item in enumerate(runner.commands) if item[0] == "rmmod"]
    assert [runner.commands[i] for i in rmmods] == [
        ["rmmod", "nvidia_drm"],
        ["rmmod", "nvidia_modeset"],
    ]
    assert resets[0] < rmmods[0] < rmmods[1] < resets[1], (
        "unloaded after the busy refusal and before the retry"
    )
    reloads = [i for i, item in enumerate(runner.commands) if item[0] == "modprobe"]
    assert [runner.commands[i] for i in reloads] == [["modprobe", "nvidia_drm"]], (
        "the image's display stack is loaded again after the reset"
    )
    assert reloads[0] > resets[1]
    assert result.details["released_kernel_modules"] == ["nvidia_drm", "nvidia_modeset"]
    assert result.details["reloaded_kernel_modules"] == ["nvidia_drm", "nvidia_modeset"]
    assert result.details["reset_attempts"] == 2
    assert result.details["reset_busy_refusals"] == 1


def test_a_reset_that_stays_busy_still_reloads_and_reports_the_release(
    tmp_path: Path,
) -> None:
    proc = _proc(tmp_path)
    runner = DisplayStackRunner(modules=proc / "modules", busy_forever=True)
    agent = _agent(tmp_path, runner, proc)

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is False
    assert "In use by another client" in (result.error or "")
    assert len([item for item in runner.commands if "--gpu-reset" in item]) == 3, (
        "the busy retry budget is unchanged"
    )
    assert result.details["released_kernel_modules"] == ["nvidia_drm", "nvidia_modeset"]
    assert result.details["reloaded_kernel_modules"] == ["nvidia_drm", "nvidia_modeset"]
    assert result.details["reset_failed"] == ["GPU-a"]
    assert result.details["gpu_client_holders"]["device_holders"] == []


def test_a_visible_device_holder_keeps_every_module_loaded(tmp_path: Path) -> None:
    proc = _proc(tmp_path, holder=True)
    runner = DisplayStackRunner(modules=proc / "modules", busy_forever=True)
    agent = _agent(tmp_path, runner, proc)

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.FAILED
    assert not [item for item in runner.commands if item[0] in {"rmmod", "modprobe"}], (
        "a process holds /dev/nvidiactl: the module is not pulled from under it"
    )
    assert "released_kernel_modules" not in result.details
    assert result.details["gpu_client_holders"]["device_holder_count"] == 1
