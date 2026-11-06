"""The display-stack follow-up a reset completion mail carries.

A busy reset may unload the idle ``nvidia_drm`` / ``nvidia_modeset``; the Agent
loads them back after the reset, retries once and checks ``/proc/modules`` and
``/sys/class/drm`` (``node_agent/operations/display_stack.py``). A stack that
did not come back leaves the node, or some of its GPUs, without ``/dev/dri`` --
graphics, EGL and Vulkan workloads lose their DRM device; CUDA does not use it
-- until someone loads it or the node reboots. The reset itself succeeded, and
nothing is drained, quarantined or reset for a display stack, so the operator
reads it where every successful reset is reported: the reset's own completion
mail (GPU_RESET_COMPLETED and FABRIC_RESET_COMPLETED already receive each
node's result). Each such node gets a follow-up block and the subject a marker.
The block says, from the Agent's receipt alone, which released modules were
loaded back (``reloaded_kernel_modules``), which are still not loaded
(``display_modules_missing``) and which GPUs are still without their DRM device
(``display_drm_devices_missing``, k of the ``display_drm_devices_before`` that
had one). When ``/proc/modules`` could not be read
(``display_module_state_error``) nothing was verified: the modules' state reads
as unknown, never as not loaded. Then come the last error, the impact -- the
whole node while a module is missing, only the named GPUs once every module is
back -- and the exact remedy. An Agent that predates the check reports a failed
reload only as ``kernel_module_reload_error``; its released modules read as not
loaded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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


def _names(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


@dataclass(frozen=True)
class _Reload:
    """One node's reload as its receipt reports it."""

    reloaded: list[str]
    missing: list[str]
    unknown: list[str]
    state_error: str
    drm_missing: list[str]
    drm_before: int | None

    @classmethod
    def of(cls, result: Mapping[str, Any]) -> _Reload:
        released = _names(result.get("released_kernel_modules"))
        reloaded = _names(result.get("reloaded_kernel_modules"))
        missing = _names(result.get("display_modules_missing"))
        drm_missing = _names(result.get("display_drm_devices_missing"))
        state_error = str(result.get("display_module_state_error") or "")
        unknown: list[str] = []
        if state_error:
            # ``/proc/modules`` was unreadable: the Agent lists every released
            # module as missing, but none of them was verified either way.
            unknown, missing = missing or released, []
        elif not missing and not drm_missing:
            # An Agent that predates the check names only what it released.
            missing = [item for item in released if item not in reloaded]
        before = result.get("display_drm_devices_before")
        return cls(
            reloaded=reloaded,
            missing=missing,
            unknown=unknown,
            state_error=state_error,
            drm_missing=drm_missing,
            drm_before=(
                before
                if isinstance(before, int) and not isinstance(before, bool)
                else None
            ),
        )

    def status(self) -> str:
        """What came back, then what is still missing or unverified."""

        clauses = []
        if self.reloaded:
            clauses.append(f"{', '.join(self.reloaded)} loaded back")
        if self.missing:
            clauses.append(f"{', '.join(self.missing)} not loaded")
        if self.unknown:
            clauses.append(
                f"state of {', '.join(self.unknown)} unknown ({self.state_error})"
            )
        if self.drm_missing:
            share = (
                f"{len(self.drm_missing)} of {self.drm_before} GPUs"
                if self.drm_before
                else f"{len(self.drm_missing)} GPU(s)"
            )
            clauses.append(f"{share} without a DRM device")
        return "; ".join(clauses) or "display stack not loaded"

    def impact(self, node_id: str) -> str:
        """The whole node without a module, the named GPUs when only their
        DRM devices are missing; hedged when nothing was verified."""

        if not self.missing and self.unknown:
            return f"/dev/dri on {node_id} may be missing"
        if not self.missing and self.drm_missing:
            return f"no /dev/dri device for {', '.join(self.drm_missing)} on {node_id}"
        return f"no /dev/dri on {node_id}"


def _last_error(result: Mapping[str, Any], outcome: _Reload) -> str:
    """The Agent's last error -- with the module state unknown, the last
    ``modprobe``'s own outcome."""

    error = result.get("kernel_module_reload_error")
    attempts = result.get("display_module_reload_attempts")
    if (
        outcome.unknown
        and isinstance(attempts, list)
        and attempts
        and isinstance(attempts[-1], Mapping)
    ):
        # With ``/proc/modules`` unreadable the Agent's "exited 0 but left ...
        # unloaded" is an inference, not a reading.
        error = attempts[-1].get("error") or "none, the last modprobe exited 0"
    return str(error or "none recorded")


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
        outcome = _Reload.of(result)
        attempts = result.get("display_module_reload_attempts")
        count = len(attempts) if isinstance(attempts, list) and attempts else 1
        error = _last_error(result, outcome)
        lines.extend(
            [
                f"- Display stack: NOT restored on {node_id}: {outcome.status()} "
                f"after {count} modprobe attempt(s); last error: "
                f"{error[:ERROR_TEXT_LIMIT]}",
                f"  Impact: {outcome.impact(node_id)} for graphics, EGL or Vulkan "
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
