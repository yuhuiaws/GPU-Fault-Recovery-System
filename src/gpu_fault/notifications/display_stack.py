"""The display-stack follow-up a reset completion mail carries.

A busy reset may unload the idle ``nvidia_drm`` / ``nvidia_modeset``; the Agent
loads them back after the reset, retries once and checks ``/proc/modules``
(``node_agent/operations/display_stack.py``). A stack that did not come back
leaves the node without ``/dev/dri`` -- graphics, EGL and Vulkan workloads lose
their DRM device; CUDA does not use it -- until someone loads it or the node
reboots. The reset itself succeeded, and nothing is drained, quarantined or
reset for a display stack, so the operator reads it where every successful
reset is reported: the reset's own completion mail (GPU_RESET_COMPLETED and
FABRIC_RESET_COMPLETED already receive each node's result). Each such node gets
a follow-up block -- what is missing, the last error, the exact remedy -- and
the subject a marker. An Agent that predates the check reports a failed reload
only as ``kernel_module_reload_error``; that reads the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DISPLAY_STACK_SUBJECT_MARKER = "[display stack not restored]"
#: The error text is the Agent's (``modprobe``'s stderr, already bounded there);
#: the mail keeps its head.
ERROR_TEXT_LIMIT = 300


def unrestored_display_stacks(
    node_results: Mapping[str, object],
) -> dict[str, Mapping[str, Any]]:
    """The nodes whose reset left the display stack unloaded, by node id."""

    unrestored: dict[str, Mapping[str, Any]] = {}
    for node_id, result in sorted(node_results.items()):
        if not isinstance(result, Mapping):
            continue
        restored = result.get("display_stack_restored")
        if restored is False or (
            restored is None and result.get("kernel_module_reload_error")
        ):
            unrestored[str(node_id)] = result
    return unrestored


def _remedy(result: Mapping[str, Any]) -> str:
    reported = result.get("display_stack_remedy")
    if isinstance(reported, str) and reported:
        return reported
    released = [str(item) for item in result.get("released_kernel_modules") or []]
    module = "nvidia_drm" if "nvidia_drm" in released or not released else released[-1]
    return f"modprobe {module}, or reboot the node"


def display_stack_follow_up(node_results: Mapping[str, object]) -> str:
    """Follow-up lines, newline-terminated; ``""`` when every stack is back."""

    lines: list[str] = []
    for node_id, result in unrestored_display_stacks(node_results).items():
        missing = result.get("display_modules_missing") or result.get(
            "released_kernel_modules"
        )
        modules = ", ".join(str(item) for item in missing or []) or "display stack"
        attempts = result.get("display_module_reload_attempts")
        count = len(attempts) if isinstance(attempts, list) and attempts else 1
        error = str(result.get("kernel_module_reload_error") or "none recorded")
        lines.extend(
            [
                f"- Display stack: NOT restored on {node_id}: {modules} not loaded "
                f"after {count} modprobe attempt(s); last error: "
                f"{error[:ERROR_TEXT_LIMIT]}",
                f"  Impact: no /dev/dri on {node_id} for graphics, EGL or Vulkan "
                "workloads; CUDA is unaffected; nothing is drained, quarantined or "
                "reset for it",
                f"  Remedy on {node_id}: {_remedy(result)}",
            ]
        )
    return "".join(f"{line}\n" for line in lines)


def display_stack_subject_marker(node_results: Mapping[str, object]) -> str:
    """The subject marker when any node's stack is not back, else ``""``."""

    return (
        DISPLAY_STACK_SUBJECT_MARKER if unrestored_display_stacks(node_results) else ""
    )
