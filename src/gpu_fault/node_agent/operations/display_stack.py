"""Load the display stack a busy reset unloaded back, and say whether it is back.

``ResetOperationsMixin._release_idle_display_modules`` unloads the idle
``nvidia_drm`` / ``nvidia_modeset`` when they are all that keeps a busy reset
from a free GPU; after the reset, whatever its outcome, the stack is loaded
again with one ``modprobe`` of the outermost module (``nvidia_drm`` pulls
``nvidia_modeset`` in first). Live (plain EKS p5en, AL2023 NVIDIA AMI) that
brings back ``modeset=Y fbdev=Y``, ``/dev/dri/card*`` and ``renderD*``.

A reload that failed used to leave one line of step evidence
(``kernel_module_reload_error``) that nothing read: the node ran on without
``/dev/dri`` -- no DRM device for graphics, EGL or Vulkan workloads; CUDA does
not use it -- until its next reboot, and a partial reload was just as silent.
Now a reload that leaves the stack incomplete is retried once after a short
pause, ``/proc/modules`` and ``/sys/class/drm`` decide the outcome -- every
module the reset released must be live again (``display_stack_restored``,
``display_modules_missing``), and every GPU that had a DRM device before the
release must have one again (``display_drm_devices_missing``) -- and a stack
that is not back carries its remedy (``display_stack_remedy``), which the
control plane's reset completion mail puts in front of the operator
(``gpu_fault.notifications.display_stack``).

A ``modprobe`` killed by its timeout does not leave the kernel's load to
finish on its own: the pending SIGKILL fails the killable wait behind every
GPU probe that has not started yet (``request_irq() failed (-4)``, "Failed to
allocate NvKmsKapiDevice"), and ``nvidia_drm`` goes ``Live`` with only the GPUs
probed before the kill (live, gpu-eks-cluster p5en, GF-REGIONAL-DESTR-010: the
12 s bound met eight cold H200 at about 2.2 s each, six DRM devices came back,
and ``/proc/modules`` alone called the stack restored). The bound therefore
covers the initialisation of every GPU of the node, and a load that is live
but short of DRM devices is unloaded (``rmmod nvidia_drm``, never ``modprobe
-r``, which also removes ``nvidia`` itself once nothing else uses it) and
loaded again.

A stack that is not back is never a failure of the reset and never a GPU
fault: CUDA works, and draining, quarantining or resetting a node for its
display stack would be disproportionate. Nothing here raises.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpu_fault.node_agent.common import (
    COLD_GPU_INIT_BOUND_SECONDS,
    DEFAULT_NODE_GPU_COUNT,
)
from gpu_fault.node_agent.operations.holders import (
    IDLE_DISPLAY_MODULES,
    display_module_state,
)

LOGGER = logging.getLogger(__name__)

#: One load and one retry; the retry waits ``DISPLAY_RELOAD_PAUSE_SECONDS``
#: first. Each ``modprobe`` is bounded by ``display_reload_timeout``: 12 s for
#: the modules themselves plus ``COLD_GPU_INIT_BOUND_SECONDS`` for every GPU
#: ``nvidia_drm`` may have to initialise (52 s on eight GPUs; live eight cold
#: H200 took about 18 s). A retry of a partial load unloads ``nvidia_drm``
#: first, bounded like the release's own ``rmmod``.
DISPLAY_RELOAD_ATTEMPTS = 2
DISPLAY_RELOAD_TIMEOUT_SECONDS = 12
DISPLAY_RELOAD_PAUSE_SECONDS = 5.0
DISPLAY_UNLOAD_TIMEOUT_SECONDS = 30
#: The DRM device nodes of one GPU (connectors such as ``card1-DP-1`` and the
#: ``version`` file are not nodes).
DRM_NODE_PATTERN = re.compile(r"^(?:card|renderD)\d+$")
NVIDIA_PCI_DRIVER = "nvidia"


@dataclass
class DisplayRelease:
    """What a busy reset unloaded of the display stack, and what it had.

    ``modules`` in unload order; ``drm_devices`` the GPUs (PCI addresses) with
    a DRM device of the NVIDIA driver right before the first unload -- what the
    reload has to bring back -- or ``None`` when that could not be read.
    """

    modules: list[str] = field(default_factory=list)
    drm_devices: list[str] | None = None


def display_reload_timeout(gpu_count: int | None) -> int:
    """12 s plus 5 s per GPU of the node: 52 s on an eight-GPU node."""

    return DISPLAY_RELOAD_TIMEOUT_SECONDS + COLD_GPU_INIT_BOUND_SECONDS * (
        gpu_count or DEFAULT_NODE_GPU_COUNT
    )


def nvidia_drm_devices(drm_class_root: Path | str | None) -> list[str] | None:
    """The GPUs (PCI addresses) that have a DRM device, from ``/sys/class/drm``.

    A ``card<N>`` or ``renderD<N>`` node counts when its device is bound to the
    ``nvidia`` PCI driver; a firmware framebuffer's ``card0`` (simpledrm) does
    not. ``None`` when the class cannot be read.
    """

    if drm_class_root is None:
        return None
    try:
        entries = list(Path(drm_class_root).iterdir())
    except OSError:
        return None
    devices: set[str] = set()
    for entry in entries:
        if not DRM_NODE_PATTERN.fullmatch(entry.name):
            continue
        try:
            driver = os.readlink(entry / "device" / "driver")
        except OSError:
            continue
        if os.path.basename(driver) == NVIDIA_PCI_DRIVER:
            devices.add(os.path.basename(os.path.realpath(entry / "device")))
    return sorted(devices)


def live_display_modules(proc_root: Path) -> set[str] | None:
    """The display-stack modules ``/proc/modules`` lists as loaded, or None
    when it cannot be read. A module still ``Loading`` (or ``Unloading``) is
    not loaded; a row without the state column (older kernels) is."""

    try:
        rows = (proc_root / "modules").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    live: set[str] = set()
    for row in rows:
        fields = row.split()
        if fields and fields[0] in IDLE_DISPLAY_MODULES:
            if len(fields) < 5 or fields[4] == "Live":
                live.add(fields[0])
    return live


def display_stack_remedy(released: Sequence[str], *, partial: bool = False) -> str:
    """The operator's way back: ``modprobe`` of the outermost released module
    (``nvidia_drm`` pulls ``nvidia_modeset`` in), after ``rmmod nvidia_drm``
    when it is loaded without some GPUs' DRM devices -- or a reboot."""

    if partial:
        return "rmmod nvidia_drm && modprobe nvidia_drm, or reboot the node"
    return f"modprobe {_outermost(released)}, or reboot the node"


def _outermost(released: Sequence[str]) -> str:
    return "nvidia_drm" if "nvidia_drm" in released else released[-1]


def _missing(proc_root: Path, released: Sequence[str]) -> list[str] | None:
    live = live_display_modules(proc_root)
    return None if live is None else [item for item in released if item not in live]


def _absent_drm_devices(
    drm_class_root: Path | str | None, expected: Sequence[str]
) -> tuple[list[str], int]:
    """The expected GPUs without a DRM device now -- all of them when the class
    cannot be read (not verified is not restored) -- and how many GPUs have
    one."""

    if not expected:
        return [], 0
    present = nvidia_drm_devices(drm_class_root)
    if present is None:
        return list(expected), 0
    return sorted(set(expected) - set(present)), len(present)


def _drm_module_users(proc_root: Path) -> str | None:
    """Why ``nvidia_drm`` must not be unloaded for a second load, or None."""

    state = {item["module"]: item for item in display_module_state(proc_root)}
    drm = state.get("nvidia_drm")
    if drm is None:
        return "its reference count cannot be read"
    if drm["refcount"]:
        return f"it is in use (refcount {drm['refcount']})"
    return None


def reload_display_stack(
    released: Sequence[str],
    *,
    run_checked: Callable[..., Any],
    sleep: Callable[[float], None],
    proc_root: Path,
    drm_class_root: Path | str | None = None,
    drm_devices_before: Sequence[str] | None = None,
    gpu_count: int | None = None,
) -> dict[str, Any]:
    """Load ``released`` back and report it; ``{}`` when nothing was released.

    An attempt counts by what ``/proc/modules`` and ``/sys/class/drm`` show
    after it, not by the exit status alone: a ``modprobe`` that failed but
    left every module live and every GPU's DRM device back needs no retry, one
    that exited 0 and loaded nothing (an ``install`` override in
    ``modprobe.d``) or loaded ``nvidia_drm`` without some GPUs' DRM devices
    does. ``reloaded_kernel_modules`` lists, in load order, the released
    modules that are live again; ``display_drm_devices_before`` and
    ``display_drm_devices_after`` count the GPUs with a DRM device before the
    release (``drm_devices_before``) and after the last load -- a node a
    partial load left short before comes back whole.
    """

    if not released:
        return {}
    details: dict[str, Any] = {"released_kernel_modules": list(released)}
    try:
        retained = display_module_state(proc_root)
    except Exception:  # noqa: BLE001 - evidence only
        retained = []
    if retained:
        # What the release left loaded (and why: its refcount and users),
        # so a modeset that stayed behind is readable from the record.
        details["display_modules_retained"] = retained
    expected = sorted(drm_devices_before or ())
    load = ["modprobe", _outermost(released)]
    timeout = display_reload_timeout(gpu_count)
    attempts: list[dict[str, Any]] = []
    missing: list[str] | None = list(released)
    absent: list[str] = []
    present = 0
    error: str | None = None
    for attempt in range(1, DISPLAY_RELOAD_ATTEMPTS + 1):
        if attempts:
            sleep(DISPLAY_RELOAD_PAUSE_SECONDS)
        # Live but short of DRM devices: ``modprobe`` answers a loaded module
        # at once, so the partial ``nvidia_drm`` is unloaded first.
        partial = missing == [] and bool(absent)
        commands = [["rmmod", "nvidia_drm"], load] if partial else [load]
        record: dict[str, Any] = {
            "attempt": attempt,
            "command": " && ".join(" ".join(item) for item in commands),
        }
        attempts.append(record)
        error = None
        refusal = _drm_module_users(proc_root) if partial else None
        if refusal is not None:
            error = record["error"] = (
                f"nvidia_drm not unloaded for a second load: {refusal}"
            )
            commands = []
        for command in commands:
            try:
                run_checked(
                    command,
                    timeout=(
                        timeout
                        if command[0] == "modprobe"
                        else DISPLAY_UNLOAD_TIMEOUT_SECONDS
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - the error is the evidence
                error = record["error"] = f"{type(exc).__name__}: {exc}"
                break
        missing = _missing(proc_root, released)
        absent, present = _absent_drm_devices(drm_class_root, expected)
        record["outcome"] = "loaded" if missing == [] and not absent else "failed"
        if record["outcome"] == "loaded":
            break
        if missing != []:
            record["missing"] = list(released) if missing is None else missing
        if absent:
            record["drm_devices_missing"] = absent
        if refusal is not None:
            break
    restored = missing == [] and not absent
    details["display_module_reload_attempts"] = attempts
    details["display_stack_restored"] = restored
    reloaded = [item for item in reversed(released) if item not in (missing or [])]
    if missing is not None and reloaded:
        details["reloaded_kernel_modules"] = reloaded
    if expected:
        details["display_drm_devices_before"] = len(expected)
        details["display_drm_devices_after"] = present
    if restored:
        return details
    if missing == []:
        # Every module is live, some GPUs have no DRM device.
        gap = (
            f"nvidia_drm is loaded without a DRM device for {len(absent)} of "
            f"{len(expected)} GPUs: {', '.join(absent)}"
        )
        details["display_drm_devices_missing"] = absent
        details["kernel_module_reload_error"] = f"{gap}; {error}" if error else gap
        details["display_stack_remedy"] = display_stack_remedy(released, partial=True)
        what = f"DRM devices of {', '.join(absent)}"
    else:
        lost = list(released) if missing is None else missing
        details["display_modules_missing"] = lost
        if absent:
            details["display_drm_devices_missing"] = absent
        if missing is None:
            details["display_module_state_error"] = "/proc/modules is unreadable"
        details["kernel_module_reload_error"] = error or (
            f"{' '.join(load)} exited 0 but left {', '.join(lost)} unloaded"
        )
        details["display_stack_remedy"] = display_stack_remedy(released)
        what = f"{', '.join(lost)} not loaded"
    LOGGER.warning(
        "display stack not restored after the GPU reset: %s after %d reload "
        "attempt(s) (%s); /dev/dri stays incomplete for graphics, EGL and "
        "Vulkan workloads, CUDA is unaffected; remedy: %s",
        what,
        len(attempts),
        details["kernel_module_reload_error"],
        details["display_stack_remedy"],
    )
    return details
