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
pause (two attempts and the pause stay under 30 s), ``/proc/modules`` decides
the outcome -- every module the reset released must be live again
(``display_stack_restored``, ``display_modules_missing``) -- and a stack that
is not back carries its remedy (``display_stack_remedy``), which the control
plane's reset completion mail puts in front of the operator
(``gpu_fault.notifications.display_stack``).

A stack that is not back is never a failure of the reset and never a GPU
fault: CUDA works, and draining, quarantining or resetting a node for its
display stack would be disproportionate. Nothing here raises.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from gpu_fault.node_agent.operations.holders import (
    IDLE_DISPLAY_MODULES,
    display_module_state,
)

LOGGER = logging.getLogger(__name__)

#: One ``modprobe`` and one retry. Each is bounded by
#: ``DISPLAY_RELOAD_TIMEOUT_SECONDS`` and the retry waits
#: ``DISPLAY_RELOAD_PAUSE_SECONDS`` first: 12 + 5 + 12 = 29 s at most. A
#: ``modprobe`` killed by its timeout leaves the kernel's own load running;
#: the retry then answers once that load finished.
DISPLAY_RELOAD_ATTEMPTS = 2
DISPLAY_RELOAD_TIMEOUT_SECONDS = 12
DISPLAY_RELOAD_PAUSE_SECONDS = 5.0


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


def display_stack_remedy(released: Sequence[str]) -> str:
    """The operator's way back: ``modprobe`` of the outermost released module
    (``nvidia_drm`` pulls ``nvidia_modeset`` in), or a reboot."""

    return f"modprobe {_outermost(released)}, or reboot the node"


def _outermost(released: Sequence[str]) -> str:
    return "nvidia_drm" if "nvidia_drm" in released else released[-1]


def _missing(proc_root: Path, released: Sequence[str]) -> list[str] | None:
    live = live_display_modules(proc_root)
    return None if live is None else [item for item in released if item not in live]


def reload_display_stack(
    released: Sequence[str],
    *,
    run_checked: Callable[..., Any],
    sleep: Callable[[float], None],
    proc_root: Path,
) -> dict[str, Any]:
    """Load ``released`` back and report it; ``{}`` when nothing was released.

    An attempt counts by what ``/proc/modules`` shows after it, not by the
    exit status alone: a ``modprobe`` that failed but left every module live
    needs no retry, one that exited 0 and loaded nothing (an ``install``
    override in ``modprobe.d``) does. ``reloaded_kernel_modules`` lists, in
    load order, the released modules that are live again.
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
    command = ["modprobe", _outermost(released)]
    attempts: list[dict[str, Any]] = []
    missing: list[str] | None = list(released)
    error: str | None = None
    for attempt in range(1, DISPLAY_RELOAD_ATTEMPTS + 1):
        if attempts:
            sleep(DISPLAY_RELOAD_PAUSE_SECONDS)
        record: dict[str, Any] = {"attempt": attempt, "command": " ".join(command)}
        attempts.append(record)
        error = None
        try:
            run_checked(command, timeout=DISPLAY_RELOAD_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001 - the error is the evidence
            error = record["error"] = f"{type(exc).__name__}: {exc}"
        missing = _missing(proc_root, released)
        record["outcome"] = "loaded" if missing == [] else "failed"
        if missing == []:
            break
        record["missing"] = list(released) if missing is None else missing
    restored = missing == []
    details["display_module_reload_attempts"] = attempts
    details["display_stack_restored"] = restored
    reloaded = [item for item in reversed(released) if item not in (missing or [])]
    if missing is not None and reloaded:
        details["reloaded_kernel_modules"] = reloaded
    if restored:
        return details
    absent = list(released) if missing is None else missing
    details["display_modules_missing"] = absent
    if missing is None:
        details["display_module_state_error"] = "/proc/modules is unreadable"
    details["kernel_module_reload_error"] = error or (
        f"{' '.join(command)} exited 0 but left {', '.join(absent)} unloaded"
    )
    details["display_stack_remedy"] = display_stack_remedy(released)
    LOGGER.warning(
        "display stack not restored after the GPU reset: %s not loaded after %d "
        "modprobe attempt(s) (%s); /dev/dri stays absent for graphics, EGL and "
        "Vulkan workloads, CUDA is unaffected; remedy: %s",
        ", ".join(absent),
        len(attempts),
        details["kernel_module_reload_error"],
        details["display_stack_remedy"],
    )
    return details
