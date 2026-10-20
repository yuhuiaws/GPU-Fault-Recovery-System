"""The display stack a busy reset unloaded is retried, checked and reported.

Live (DESTR-001, plain EKS p5en, AL2023 NVIDIA AMI): the busy single-GPU reset
unloaded ``nvidia_drm`` and ``nvidia_modeset``, the retry reset the GPU and
``modprobe nvidia_drm`` loaded both back (modeset=Y fbdev=Y, ``/dev/dri/card1..8``
and ``renderD128..135`` back). Had that ``modprobe`` failed, the only trace was
``kernel_module_reload_error`` in the step details, read by nothing: the node
would have run without ``/dev/dri`` (graphics, EGL, Vulkan; CUDA unaffected)
until its next reboot. The reload is now retried once after a short pause,
``/proc/modules`` decides whether every released module is back
(``display_stack_restored``), and a stack that is not back carries the remedy
into the reset completion mail -- while the reset stays SUCCEEDED and nothing
is drained, quarantined or reset for it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from typing import Any

import pytest

from gpu_fault.node_agent.operations.display_stack import (
    DISPLAY_RELOAD_PAUSE_SECONDS,
    display_reload_timeout,
    display_stack_remedy,
    live_display_modules,
    reload_display_stack,
)
from gpu_fault.notifications.display_stack import DISPLAY_STACK_SUBJECT_MARKER
from gpu_fault.notifications.restart_guard import RestartGuardEmailBuilder
from tests.node_agent._support import (
    NodeActionStatus,
    WorkflowOperation,
    command,
    envelope,
    no_device_clients,
    node_action_executor,
)

BUSY = "GPU 00000000:53:00.0: In use by another client"
NO_DEVICE = "modprobe: ERROR: could not insert 'nvidia_drm': No such device"
DRM_FAILED = "modprobe: ERROR: could not insert 'nvidia_drm': Input/output error"
DISPLAY_KEYS = {
    "released_kernel_modules",
    "reloaded_kernel_modules",
    "display_modules_retained",
    "display_module_reload_attempts",
    "display_stack_restored",
    "display_modules_missing",
    "display_module_state_error",
    "display_stack_remedy",
    "kernel_module_reload_error",
}


class ReloadNode:
    """``nvidia-smi``, ``rmmod`` and ``modprobe`` on a fake ``/proc/modules``.

    The reset is busy while ``nvidia_drm`` is loaded. Each ``modprobe`` plays
    the next entry of ``plan``: ``ok`` loads the stack, ``fail`` loads nothing
    and exits 1, ``modeset-only`` loads ``nvidia_modeset`` and then fails on
    ``nvidia_drm``, ``noop`` exits 0 and loads nothing (an ``install`` override).
    ``modeset_user`` keeps ``nvidia_modeset`` in use, so only drm is released.
    Every call and every pause lands in ``events``.
    """

    def __init__(
        self, tmp_path: Path, plan: list[str], *, modeset_user: str | None = None
    ) -> None:
        self.proc = tmp_path / "proc"
        self.proc.mkdir()
        self.plan = list(plan)
        self.modeset_user = modeset_user
        self.loaded = {"nvidia_drm", "nvidia_modeset"}
        self.events: list[tuple[str, Any, Any]] = []
        self.write()

    def write(self) -> None:
        rows = []
        if "nvidia_drm" in self.loaded:
            rows.append("nvidia_drm 139264 0 - Live 0x0")
        if "nvidia_modeset" in self.loaded:
            users = ["nvidia_drm"] if "nvidia_drm" in self.loaded else []
            users += [self.modeset_user] if self.modeset_user else []
            used_by = "".join(f"{item}," for item in users) or "-"
            rows.append(f"nvidia_modeset 1503232 {len(users)} {used_by} Live 0x0")
        rows.append("nvidia_uvm 4931584 0 - Live 0x0")
        rows.append("nvidia 117800960 30 nvidia_uvm,nvidia_modeset, Live 0x0")
        (self.proc / "modules").write_text("\n".join(rows) + "\n", encoding="ascii")

    def sleep(self, seconds: float) -> None:
        self.events.append(("sleep", seconds, None))

    def modprobes(self) -> list[tuple[str, Any, Any]]:
        return [
            item
            for item in self.events
            if item[0] == "run" and item[1][:1] == ["modprobe"]
        ]

    def commands(self) -> list[list[str]]:
        return [item[1] for item in self.events if item[0] == "run"]

    def __call__(self, cmd, **kwargs):
        self.events.append(("run", cmd, kwargs.get("timeout")))
        if cmd[0] == "rmmod":
            self.loaded.discard(cmd[1])
            self.write()
        elif cmd[0] == "modprobe":
            return self._modprobe(cmd)
        elif "--gpu-reset" in cmd and "nvidia_drm" in self.loaded:
            raise CalledProcessError(255, cmd, stderr=BUSY)
        stdout = "GPU-a\n" if "--query-gpu=uuid" in cmd else ""
        return CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    def _modprobe(self, cmd):
        step = self.plan.pop(0) if self.plan else "ok"
        if step == "ok":
            self.loaded.update({"nvidia_drm", "nvidia_modeset"})
        elif step == "modeset-only":
            self.loaded.add("nvidia_modeset")
        self.write()
        if step in {"fail", "modeset-only"}:
            raise CalledProcessError(
                1, cmd, stderr=NO_DEVICE if step == "fail" else DRM_FAILED
            )
        return CompletedProcess(cmd, 0, stdout="", stderr="")


def _agent(tmp_path: Path, node: ReloadNode):
    return node_action_executor(
        tmp_path,
        "reload.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        runner=node,
        proc_root=str(node.proc),
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        sleep=node.sleep,
    )


def _reset(agent, command_id: str = "workflow/RESET_GPU/node-a"):
    return agent.execute(
        envelope(command(WorkflowOperation.RESET_GPU, command_id=command_id))
    )


def _mail(details: dict[str, Any]):
    return RestartGuardEmailBuilder().build_gpu_reset_completed(
        cluster_id="gpu-eks-cluster",
        incident_id="incident-a",
        workflow_id="workflow-a",
        event_id="xid-46",
        event_type="XID",
        policy_source="NVIDIA_CATALOG",
        official_action="RESET_GPU",
        reasons=["XID 46"],
        operation_id="workflow-a/4/RESET_GPU",
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
        node_results={"node-a": details},
        workload_ids=[],
    )


def _assert_nothing_escalated(result) -> None:
    """A stack that is not back is no failure and no GPU fault."""

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert result.retryable is False
    for key in ("outcome_unknown", "manual_confirmation_required", "reset_failed"):
        assert key not in result.details, key


def test_a_reload_that_works_at_once_is_recorded_as_before(tmp_path: Path) -> None:
    node = ReloadNode(tmp_path, ["ok"])
    result = _reset(_agent(tmp_path, node))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert result.details["released_kernel_modules"] == ["nvidia_drm", "nvidia_modeset"]
    assert result.details["reloaded_kernel_modules"] == ["nvidia_modeset", "nvidia_drm"]
    assert result.details["display_stack_restored"] is True
    assert result.details["display_module_reload_attempts"] == [
        {"attempt": 1, "command": "modprobe nvidia_drm", "outcome": "loaded"}
    ]
    assert [item[1] for item in node.modprobes()] == [["modprobe", "nvidia_drm"]], (
        "one modprobe, as before"
    )
    assert ("sleep", DISPLAY_RELOAD_PAUSE_SECONDS, None) not in node.events, (
        "no retry pause when the first load is good"
    )
    for key in ("display_modules_missing", "display_stack_remedy"):
        assert key not in result.details, key
    assert "kernel_module_reload_error" not in result.details
    assert DISPLAY_STACK_SUBJECT_MARKER not in _mail(result.details).subject


def test_a_failed_modprobe_is_retried_once_after_a_pause(tmp_path: Path) -> None:
    node = ReloadNode(tmp_path, ["fail", "ok"])
    result = _reset(_agent(tmp_path, node))

    _assert_nothing_escalated(result)
    modprobes = [i for i, item in enumerate(node.events) if item in node.modprobes()]
    assert len(modprobes) == 2, node.events
    assert node.events[modprobes[0] + 1] == (
        "sleep",
        DISPLAY_RELOAD_PAUSE_SECONDS,
        None,
    )
    assert modprobes[1] == modprobes[0] + 2, "the pause sits between the two loads"
    first, second = result.details["display_module_reload_attempts"]
    assert first["outcome"] == "failed"
    assert NO_DEVICE in first["error"], first
    assert first["missing"] == ["nvidia_drm", "nvidia_modeset"]
    assert second == {
        "attempt": 2,
        "command": "modprobe nvidia_drm",
        "outcome": "loaded",
    }
    assert result.details["display_stack_restored"] is True
    assert result.details["reloaded_kernel_modules"] == ["nvidia_modeset", "nvidia_drm"]
    assert "kernel_module_reload_error" not in result.details, (
        "the retry brought the stack back: nothing is left to report as an error"
    )


def test_a_stack_that_stays_unloaded_is_reported_with_its_remedy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    node = ReloadNode(tmp_path, ["fail", "fail"])
    with caplog.at_level(logging.WARNING):
        result = _reset(_agent(tmp_path, node))

    _assert_nothing_escalated(result)
    assert result.details["reset_successes"] == 1
    assert result.details["display_stack_restored"] is False
    assert result.details["display_modules_missing"] == ["nvidia_drm", "nvidia_modeset"]
    assert NO_DEVICE in result.details["kernel_module_reload_error"]
    assert result.details["display_stack_remedy"] == (
        "modprobe nvidia_drm, or reboot the node"
    )
    assert "reloaded_kernel_modules" not in result.details
    assert len(result.details["display_module_reload_attempts"]) == 2
    timeouts = [item[2] for item in node.modprobes()]
    # Bounded per load, for every GPU the load may have to initialise: this
    # fake ``/proc`` lists no GPUs, so the bound is the largest node shape's.
    assert timeouts == [display_reload_timeout(None)] * 2, timeouts
    commands = node.commands()
    assert commands[-1] == ["modprobe", "nvidia_drm"], (
        "nothing runs after the second load: no third try, no further reset"
    )
    assert "display stack not restored" in caplog.text
    assert "modprobe nvidia_drm, or reboot the node" in caplog.text


def test_a_partial_reload_names_the_module_that_is_missing(tmp_path: Path) -> None:
    node = ReloadNode(tmp_path, ["modeset-only", "modeset-only"])
    result = _reset(_agent(tmp_path, node))

    _assert_nothing_escalated(result)
    assert result.details["reloaded_kernel_modules"] == ["nvidia_modeset"]
    assert result.details["display_modules_missing"] == ["nvidia_drm"]
    assert result.details["display_stack_restored"] is False
    assert DRM_FAILED in result.details["kernel_module_reload_error"]
    assert node.loaded == {"nvidia_modeset"}


def test_a_retained_modeset_does_not_hide_a_drm_that_did_not_come_back(
    tmp_path: Path,
) -> None:
    node = ReloadNode(tmp_path, ["fail", "fail"], modeset_user="ghost_user")
    result = _reset(_agent(tmp_path, node))

    _assert_nothing_escalated(result)
    assert result.details["released_kernel_modules"] == ["nvidia_drm"]
    assert result.details["display_modules_retained"] == [
        {"module": "nvidia_modeset", "refcount": 1, "used_by": ["ghost_user"]}
    ]
    assert result.details["display_modules_missing"] == ["nvidia_drm"]
    assert result.details["display_stack_restored"] is False


def test_a_modprobe_that_loads_nothing_is_caught_by_proc_modules(
    tmp_path: Path,
) -> None:
    node = ReloadNode(tmp_path, ["noop", "noop"])
    result = _reset(_agent(tmp_path, node))

    _assert_nothing_escalated(result)
    assert result.details["display_stack_restored"] is False
    assert result.details["kernel_module_reload_error"] == (
        "modprobe nvidia_drm exited 0 but left nvidia_drm, nvidia_modeset unloaded"
    )
    attempts = result.details["display_module_reload_attempts"]
    assert [item["outcome"] for item in attempts] == ["failed", "failed"], attempts


def test_a_reset_that_released_nothing_adds_no_display_fields(tmp_path: Path) -> None:
    node = ReloadNode(tmp_path, [])
    node.loaded = set()
    node.write()
    result = _reset(_agent(tmp_path, node))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert result.details["reset_attempts"] == 1
    assert not DISPLAY_KEYS & set(result.details), sorted(result.details)
    assert not [item for item in node.commands() if item[0] in {"rmmod", "modprobe"}], (
        "no module is touched"
    )
    assert DISPLAY_STACK_SUBJECT_MARKER not in _mail(result.details).subject


def test_the_notice_clears_once_the_modules_are_back(tmp_path: Path) -> None:
    """The mail names the node while its stack is missing; once the operator's
    ``modprobe`` brought it back, the next reset's record and mail are clean."""

    node = ReloadNode(tmp_path, ["fail", "fail"])
    agent = _agent(tmp_path, node)
    first = _reset(agent, "workflow/RESET_GPU/node-a/1")
    assert first.details["display_stack_restored"] is False
    notice = _mail(first.details)
    assert DISPLAY_STACK_SUBJECT_MARKER in notice.subject, notice.subject
    assert "Remedy on node-a: modprobe nvidia_drm, or reboot the node" in (
        notice.body_text
    )

    node.loaded.update({"nvidia_drm", "nvidia_modeset"})  # the operator's modprobe
    node.write()
    assert live_display_modules(node.proc) == {"nvidia_drm", "nvidia_modeset"}
    second = _reset(agent, "workflow/RESET_GPU/node-a/2")

    assert second.status is NodeActionStatus.SUCCEEDED, second.error
    assert second.details["display_stack_restored"] is True
    assert "display_modules_missing" not in second.details
    cleared = _mail(second.details)
    assert DISPLAY_STACK_SUBJECT_MARKER not in cleared.subject, cleared.subject
    assert "Display stack" not in cleared.body_text


def test_a_module_still_loading_or_an_unreadable_table_is_not_loaded(
    tmp_path: Path,
) -> None:
    modules = tmp_path / "modules"
    modules.write_text(
        "nvidia_drm 139264 0 - Loading 0x0\nnvidia_modeset 1503232 0 - Live 0x0\n",
        encoding="ascii",
    )
    assert live_display_modules(tmp_path) == {"nvidia_modeset"}
    modules.write_text("nvidia_drm 139264 0 -\n", encoding="ascii")
    assert live_display_modules(tmp_path) == {"nvidia_drm"}, "no state column: loaded"
    modules.unlink()
    assert live_display_modules(tmp_path) is None

    calls: list[list[str]] = []
    details = reload_display_stack(
        ["nvidia_drm"],
        run_checked=lambda cmd, **_: calls.append(cmd),
        sleep=lambda _seconds: None,
        proc_root=tmp_path,
    )
    assert len(calls) == 2, "an unverifiable load is retried like a failed one"
    assert details["display_stack_restored"] is False
    assert details["display_module_state_error"] == "/proc/modules is unreadable"
    assert details["display_modules_missing"] == ["nvidia_drm"]
    assert "reloaded_kernel_modules" not in details


def test_the_remedy_loads_the_outermost_released_module() -> None:
    assert display_stack_remedy(["nvidia_drm", "nvidia_modeset"]) == (
        "modprobe nvidia_drm, or reboot the node"
    )
    assert display_stack_remedy(["nvidia_modeset"]) == (
        "modprobe nvidia_modeset, or reboot the node"
    )
    calls: list[list[str]] = []
    assert (
        reload_display_stack(
            [],
            run_checked=lambda cmd, **_: calls.append(cmd),
            sleep=lambda _seconds: None,
            proc_root=Path("/nonexistent"),
        )
        == {}
    ), "nothing released: nothing to load and no fields"
    assert calls == []
