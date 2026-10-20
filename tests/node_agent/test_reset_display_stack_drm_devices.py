"""The display stack a busy reset unloaded comes back on every GPU, or says so.

Live (gpu-eks-cluster, ip-10-0-45-43, p5en.48xlarge, 8x H200, AL2023 NVIDIA
AMI, GF-REGIONAL-DESTR-010): after the refused fabric reset the Agent ran
``modprobe nvidia_drm`` with a 12 s bound. Each cold GPU took about 2.2 s to
probe; the bound killed ``modprobe`` part way, the pending SIGKILL failed the
probes of the last two GPUs (``request_irq() failed (-4)``, "Failed to
allocate NvKmsKapiDevice"), ``nvidia_drm`` went ``Live`` anyway, and the node
was left with six render nodes for eight GPUs -- which ``/proc/modules`` alone
could not see, and which the refused reset's record did not carry at all.

The load is now bounded for every GPU of the node, the GPUs' DRM devices are
read before the release and checked after the reload, a load that is live but
short of DRM devices is unloaded and loaded again once, and what stays missing
is reported by PCI address with its remedy -- on a failed reset's record too.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.operations.display_stack import (
    display_reload_timeout,
    nvidia_drm_devices,
)
from gpu_fault.node_agent.protocol import NodeActionStatus
from tests.node_agent._cold_node import (
    GPU_BUS_IDS,
    ColdNode,
    cold_agent,
    fabric_reset,
    gpu_reset,
)
from tests.node_agent._support import node_action_executor

LAST_TWO = sorted(GPU_BUS_IDS[-2:])


def _reloads(node: ColdNode) -> list[str]:
    """The commands after the reset's last ``--gpu-reset``: the reload."""

    resets = [
        i for i, item in enumerate(node.calls) if "--gpu-reset" in item["command"]
    ]
    return [
        item["command"]
        for item in node.calls[resets[-1] + 1 :]
        if item["command"].split()[0] in {"rmmod", "modprobe"}
    ]


def test_the_reload_after_a_cold_fabric_reset_brings_every_drm_device_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = ColdNode(tmp_path)
    agent = cold_agent(tmp_path, node, monkeypatch)

    result = agent.execute(fabric_reset())

    assert sorted(node.drm) == sorted(GPU_BUS_IDS), (
        "every GPU has its DRM device back: "
        f"missing {sorted(set(GPU_BUS_IDS) - set(node.drm))}, killed {node.killed()}"
    )
    assert node.render_nodes() == len(GPU_BUS_IDS)
    assert node.timeouts("modprobe") == [display_reload_timeout(len(GPU_BUS_IDS))]
    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert result.details["display_stack_restored"] is True
    assert result.details["display_drm_devices_before"] == len(GPU_BUS_IDS)
    assert result.details["display_drm_devices_after"] == len(GPU_BUS_IDS)
    assert "display_drm_devices_missing" not in result.details


def test_a_node_left_short_by_an_earlier_load_comes_back_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node the live failure left behind: ``nvidia_drm`` loaded with six
    DRM devices for eight GPUs. Its two GPUs without one are cold before the
    release already, so every read pays their initialisation too and the busy
    reset may run out of the command's lifetime (not asserted: that is the
    command's two minutes against the model's costs) -- but whatever the
    reset's outcome, the reload brings every GPU's DRM device back, and the
    next attempt starts warm."""

    node = ColdNode(tmp_path)
    node.drm = list(GPU_BUS_IDS[:6])
    node.write()
    agent = cold_agent(tmp_path, node, monkeypatch)

    result = agent.execute(fabric_reset())

    assert sorted(node.drm) == sorted(GPU_BUS_IDS)
    assert result.details["display_stack_restored"] is True
    assert result.details["display_drm_devices_before"] == 6
    assert result.details["display_drm_devices_after"] == len(GPU_BUS_IDS)


def test_a_load_killed_part_way_is_unloaded_and_loaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A driver slower than the bound still meets the kill: the first load
    comes back live with six DRM devices, and the retry starts it over."""

    node = ColdNode(tmp_path)
    node.load_seconds = [10.0]
    agent = cold_agent(tmp_path, node, monkeypatch)

    result = agent.execute(fabric_reset())

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert sorted(node.drm) == sorted(GPU_BUS_IDS)
    assert _reloads(node) == [
        "modprobe nvidia_drm",
        "rmmod nvidia_drm",
        "modprobe nvidia_drm",
    ]
    first, second = result.details["display_module_reload_attempts"]
    assert first["outcome"] == "failed"
    assert first["error"].startswith("TimeoutExpired"), first
    assert first["drm_devices_missing"] == LAST_TWO
    assert "missing" not in first, "every module was live; only devices were short"
    assert second == {
        "attempt": 2,
        "command": "rmmod nvidia_drm && modprobe nvidia_drm",
        "outcome": "loaded",
    }
    assert result.details["display_stack_restored"] is True
    assert "kernel_module_reload_error" not in result.details


def test_a_load_short_of_drm_devices_is_retried_even_when_modprobe_says_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = ColdNode(tmp_path)
    node.failing_probes = [set(GPU_BUS_IDS[-2:])]
    agent = cold_agent(tmp_path, node, monkeypatch)

    result = agent.execute(fabric_reset())

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert sorted(node.drm) == sorted(GPU_BUS_IDS)
    attempts = result.details["display_module_reload_attempts"]
    assert [item["outcome"] for item in attempts] == ["failed", "loaded"]
    assert "error" not in attempts[0], "modprobe exited 0; /sys/class/drm said no"
    assert result.details["display_stack_restored"] is True


def test_a_stack_that_stays_short_is_reported_with_its_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    node = ColdNode(tmp_path)
    node.failing_probes = [set(GPU_BUS_IDS[-2:]), set(GPU_BUS_IDS[-2:])]
    agent = cold_agent(tmp_path, node, monkeypatch)

    with caplog.at_level(logging.WARNING):
        result = agent.execute(fabric_reset())

    assert result.status is NodeActionStatus.SUCCEEDED, (
        "a display stack that is not back never fails the reset"
    )
    assert result.retryable is False
    details = result.details
    assert details["display_stack_restored"] is False
    assert details["display_drm_devices_missing"] == LAST_TWO
    assert details["display_drm_devices_before"] == len(GPU_BUS_IDS)
    assert details["display_drm_devices_after"] == len(GPU_BUS_IDS) - 2
    assert "display_modules_missing" not in details, "both modules are live"
    assert details["kernel_module_reload_error"] == (
        "nvidia_drm is loaded without a DRM device for 2 of 8 GPUs: "
        + ", ".join(LAST_TWO)
    )
    assert details["display_stack_remedy"] == (
        "rmmod nvidia_drm && modprobe nvidia_drm, or reboot the node"
    )
    assert len(details["display_module_reload_attempts"]) == 2, "one retry, no more"
    assert "display stack not restored" in caplog.text
    assert ", ".join(LAST_TWO) in caplog.text


def test_a_drm_client_keeps_the_short_stack_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process opened a DRM device the first load created: the partial
    ``nvidia_drm`` is in use and is not unloaded from under it."""

    node = ColdNode(tmp_path)
    node.failing_probes = [set(GPU_BUS_IDS[-2:])]
    node.after_load = node.hold_drm
    agent = cold_agent(tmp_path, node, monkeypatch)

    result = agent.execute(fabric_reset())

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert _reloads(node) == ["modprobe nvidia_drm"], "no rmmod under a DRM client"
    second = result.details["display_module_reload_attempts"][1]
    assert second["error"] == (
        "nvidia_drm not unloaded for a second load: it is in use (refcount 1)"
    )
    assert result.details["display_stack_restored"] is False
    assert result.details["display_drm_devices_missing"] == LAST_TWO


def test_a_refused_fabric_reset_still_reports_its_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live record: refused between the two unloads, it named neither the
    release nor the reload. Now it says ``nvidia_drm`` was released and is back
    on every GPU."""

    node = ColdNode(tmp_path)
    agent = cold_agent(tmp_path, node, monkeypatch)
    node.after_unload_of(None, node.hang)

    result = agent.execute(fabric_reset())

    assert result.status is NodeActionStatus.FAILED
    assert result.details["reason"] == "OWNERSHIP_FINAL_SAFETY_CHANGED"
    assert result.details["released_kernel_modules"] == ["nvidia_drm"]
    assert result.details["reloaded_kernel_modules"] == ["nvidia_drm"]
    assert result.details["display_stack_restored"] is True
    assert result.details["display_drm_devices_after"] == len(GPU_BUS_IDS)
    assert sorted(node.drm) == sorted(GPU_BUS_IDS)


def test_a_gpu_reset_that_fails_without_a_verdict_still_reloads(tmp_path: Path) -> None:
    """A probe between two attempts that times out is no reset verdict and
    travels on as it is (retryable) -- with the display stack it unloaded put
    back instead of left out until the next reboot."""

    node = ColdNode(tmp_path)
    node.after_unload_of("nvidia_modeset", node.hang)
    agent = node_action_executor(
        tmp_path,
        "cold-gpu.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        now=node.now,
        runner=node,
        proc_root=str(node.proc),
        sleep=node.sleep,
        agent_generation=1,
    )
    agent.drm_class_root = node.drm_class

    result = agent.execute(gpu_reset("GPU-7"))

    assert result.status is NodeActionStatus.FAILED
    assert (result.error or "").startswith("TimeoutExpired"), result.error
    assert result.retryable is True
    assert node.loaded >= {"nvidia_drm", "nvidia_modeset"}, "the stack is back"
    assert sorted(node.drm) == sorted(GPU_BUS_IDS)


def test_drm_devices_are_the_nvidia_gpus_with_a_card_or_render_node(
    tmp_path: Path,
) -> None:
    node = ColdNode(tmp_path)
    (node.drm_class / "card1-DP-1").mkdir()  # a connector, not a device node
    (node.drm_class / "version").write_text("drm 1.1.0\n", encoding="ascii")

    assert nvidia_drm_devices(node.drm_class) == sorted(GPU_BUS_IDS), (
        "card0 is the firmware framebuffer's, connectors are not nodes"
    )
    node.drm = list(GPU_BUS_IDS[:6])
    node.write()
    assert nvidia_drm_devices(node.drm_class) == sorted(GPU_BUS_IDS[:6])
    assert nvidia_drm_devices(tmp_path / "missing") is None
    assert nvidia_drm_devices(None) is None


def test_the_reload_bound_covers_every_gpu_of_the_node() -> None:
    assert display_reload_timeout(8) == 52
    assert display_reload_timeout(1) == 17
    assert display_reload_timeout(None) == display_reload_timeout(8), (
        "a node whose GPU list cannot be read is bounded as the largest shape"
    )
    assert display_reload_timeout(8) > 2 * 8 * 2.2, "twice the measured cold load"
