"""One RESET_GPU command for two GPUs of one node: one window, two resets, one
display-stack release (GF-REGIONAL-DESTR-026).

A merged same-node fault widens the workflow's RESET_GPU to both GPUs, and the
agent resets them one after another inside the one quiesce window
(``ResetOperationsMixin._reset_gpu``). On the p5en image the first reset is
refused busy while the idle ``nvidia_drm``/``nvidia_modeset`` stack is loaded;
the release list is shared across the loop, so the stack is unloaded once
before the first GPU's retry, the second GPU resets at once, and the stack is
loaded back once at the end. The live case grades exactly this result.
"""

from __future__ import annotations

from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from typing import Any

from gpu_fault.node_agent import NodeActionExecutor
from gpu_fault.node_agent.protocol import NodeActionResult, NodeActionStatus
from tests.node_agent._support import (
    WorkflowOperation,
    command,
    envelope,
    no_device_clients,
    node_action_executor,
)

GPUS = ("GPU-a", "GPU-b")
BUSY = "GPU 00000000:59:00.0: In use by another client"


class TwoGpuDisplayRunner:
    """Every reset is refused busy while ``nvidia_drm`` is loaded; ``rmmod``
    and ``modprobe`` rewrite ``/proc/modules`` the way the kernel would."""

    def __init__(self, modules: Path) -> None:
        self.modules = modules
        self.commands: list[list[str]] = []
        self.loaded = {"nvidia_drm", "nvidia_modeset"}
        self.write_modules()

    def __call__(self, cmd: list[str], **_: Any) -> CompletedProcess[str]:
        self.commands.append(list(cmd))
        if cmd[0] == "rmmod":
            self.loaded.discard(cmd[1])
        elif cmd[0] == "modprobe":
            self.loaded.update({"nvidia_drm", "nvidia_modeset"})
        elif "--gpu-reset" in cmd and "nvidia_drm" in self.loaded:
            raise CalledProcessError(255, cmd, stderr=BUSY)
        self.write_modules()
        stdout = ""
        if "--query-gpu=uuid" in cmd:
            stdout = "".join(f"{gpu}\n" for gpu in GPUS)
        elif "--query-gpu=uuid,index" in cmd:
            stdout = "".join(f"{gpu}, {index}\n" for index, gpu in enumerate(GPUS))
        return CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    def write_modules(self) -> None:
        rows = []
        if "nvidia_drm" in self.loaded:
            rows.append("nvidia_drm 94208 0 -")
        if "nvidia_modeset" in self.loaded:
            users = "nvidia_drm," if "nvidia_drm" in self.loaded else "-"
            rows.append(f"nvidia_modeset 1327104 {int(users != '-')} {users}")
        rows.append("nvidia 8933376 2 nvidia_uvm,nvidia_modeset")
        self.modules.write_text("\n".join(rows) + "\n", encoding="ascii")

    def resets(self) -> list[str]:
        return [item[-1] for item in self.commands if "--gpu-reset" in item]


def two_gpu_agent(
    tmp_path: Path, ledger_name: str = "two-gpu.db"
) -> tuple[NodeActionExecutor, TwoGpuDisplayRunner]:
    proc = tmp_path / "proc"
    proc.mkdir()
    runner = TwoGpuDisplayRunner(proc / "modules")
    agent = node_action_executor(
        tmp_path,
        ledger_name,
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        runner=runner,
        proc_root=str(proc),
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {
            "GPU-a": "/dev/nvidia0",
            "GPU-b": "/dev/nvidia1",
        },
        sleep=lambda _: None,
    )
    return agent, runner


def reset_both(agent: NodeActionExecutor) -> NodeActionResult:
    return agent.execute(
        envelope(command(WorkflowOperation.RESET_GPU, gpu_uuids=list(GPUS)))
    )


def test_one_command_resets_both_gpus_one_after_another(tmp_path: Path) -> None:
    agent, runner = two_gpu_agent(tmp_path)

    result = reset_both(agent)

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert runner.resets() == ["GPU-a", "GPU-a", "GPU-b"], (
        "the first GPU is retried after the release, the second resets at once"
    )
    assert result.details["reset_gpu_uuids"] == list(GPUS)
    assert result.details["reset_successes"] == 2
    assert result.details["reset_attempts"] == 3
    assert result.details["reset_busy_refusals"] == 1
    calls = [
        (call["command"].split()[-1], call["outcome"])
        for call in result.details["physical_calls"]
    ]
    assert calls == [
        ("GPU-a", "busy"),
        ("nvidia_drm", "released"),
        ("nvidia_modeset", "released"),
        ("GPU-a", "reset"),
        ("GPU-b", "reset"),
    ], "every physical call of the window, in order, answered by its outcome"


def test_the_display_stack_is_released_once_and_loaded_back_once(
    tmp_path: Path,
) -> None:
    agent, runner = two_gpu_agent(tmp_path)

    result = reset_both(agent)

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    rmmods = [item for item in runner.commands if item[0] == "rmmod"]
    assert rmmods == [["rmmod", "nvidia_drm"], ["rmmod", "nvidia_modeset"]], (
        "the second GPU finds the stack already released"
    )
    reloads = [
        index for index, item in enumerate(runner.commands) if item[0] == "modprobe"
    ]
    assert [runner.commands[index] for index in reloads] == [["modprobe", "nvidia_drm"]]
    last_reset = max(
        index for index, item in enumerate(runner.commands) if "--gpu-reset" in item
    )
    assert reloads[0] > last_reset, "loaded back after both resets, not between"
    assert result.details["released_kernel_modules"] == ["nvidia_drm", "nvidia_modeset"]
    assert result.details["reloaded_kernel_modules"] == ["nvidia_modeset", "nvidia_drm"]
    assert runner.loaded == {"nvidia_drm", "nvidia_modeset"}, (
        "the image's stack is back"
    )
