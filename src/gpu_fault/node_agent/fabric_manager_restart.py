"""Start Fabric Manager after a full GPU + NVSwitch reset found it down.

NVIDIA's restart procedure for H100-and-later NVSwitch systems is: stop the GPU
services and Fabric Manager, ``nvidia-smi -r``, start Fabric Manager again, so
the reset GPUs re-register with the NVLink fabric (Fabric Manager user guide,
"Initializing NVSwitch and NVLink"). The product's RESET_ALL_GPUS_NVSWITCHES
chain is that procedure, except that its restore only started what its quiesce
had stopped: a Fabric Manager that had *failed* (live, plain EKS p5en: "failed
to configure all the available GPUs or NVSwitches" at boot, the unit failed
ever since) was never stopped, so it was never started, and the reset left
every GPU waiting for a fabric manager that was not there. A Fabric Manager
restart on its own does not help either: live, all 8 GPUs stayed "In Progress"
after one, because only re-initialised GPUs register.

So after a restore of a window whose claimed reset was the full fabric reset,
a Fabric Manager unit that systemd enables but that is not running is started.
It is never a restore failure: every other service is already back, and
VALIDATE_GPU judges whether the fabric trained (``gpu_fault.gpu_fabric_manager``)
and climbs the ladder (reboot, then replacement) when it did not.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from gpu_fault.cluster_scope import (
    FABRIC_MANAGER_START_TIMEOUT_SECONDS,
    FABRIC_MANAGER_UNIT,
)

#: The scope ``_execute_reset_all`` notes on its quiesce-window claim.
FULL_FABRIC_RESET_SCOPE = "ALL_LOCAL_GPUS_AND_NVSWITCHES"
#: Fabric Manager's start returns once it has configured the fabric or given up
#: (34 s to give up, live on H200); NVIDIA's R580 unit allows
#: ``TimeoutStartSec=720`` (on NVL5 the start script polls for the InfiniBand
#: device and starts NVLSM first), so the Agent waits the unit's bound plus a
#: minute (``cluster_scope.FABRIC_MANAGER_START_TIMEOUT_SECONDS``).
START_TIMEOUT_SECONDS = FABRIC_MANAGER_START_TIMEOUT_SECONDS
QUERY_TIMEOUT_SECONDS = 15
#: A unit an operator disabled or masked stays down: that is a decision, not a
#: failure this chain may override.
STARTABLE_ENABLEMENT = frozenset({"enabled", "enabled-runtime", "static"})
TEXT_LIMIT = 400


def _answer(run: Callable[..., subprocess.CompletedProcess[str]], verb: str) -> str:
    try:
        completed = run(
            ["systemctl", verb, FABRIC_MANAGER_UNIT],
            check=False,
            capture_output=True,
            text=True,
            timeout=QUERY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unreadable: {type(exc).__name__}"
    return (completed.stdout or "").strip() or "unknown"


def start_fabric_manager_after_reset(
    run: Callable[..., subprocess.CompletedProcess[str]],
    restored_services: Sequence[str],
) -> dict[str, Any]:
    """Start a down, enabled Fabric Manager; the record of what happened.

    Empty when there is nothing to do: the restore already started the unit
    (the quiesce had stopped it), or it is running.
    """

    if FABRIC_MANAGER_UNIT in restored_services:
        return {}
    before = _answer(run, "is-active")
    if before == "active":
        return {}
    enabled = _answer(run, "is-enabled")
    record: dict[str, Any] = {
        "unit": FABRIC_MANAGER_UNIT,
        "state_before": before,
        "enabled": enabled,
        "attempted": enabled in STARTABLE_ENABLEMENT,
    }
    if not record["attempted"]:
        return {"fabric_manager_start": record}
    try:
        run(
            ["systemctl", "reset-failed", FABRIC_MANAGER_UNIT],
            check=False,
            capture_output=True,
            text=True,
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        started = run(
            ["systemctl", "start", FABRIC_MANAGER_UNIT],
            check=False,
            capture_output=True,
            text=True,
            timeout=START_TIMEOUT_SECONDS,
        )
        record["returncode"] = started.returncode
        if started.returncode:
            output = f"{started.stdout or ''}\n{started.stderr or ''}".strip()
            record["output_tail"] = output[-TEXT_LIMIT:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:TEXT_LIMIT]
    after = _answer(run, "is-active")
    record.update({"state_after": after, "active": after == "active"})
    return {"fabric_manager_start": record}
