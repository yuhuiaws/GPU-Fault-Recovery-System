#!/usr/bin/env python3
"""On-node probe for GF-REGIONAL-DESTR-017.

Two node-local jobs the control plane cannot do for the runner, both of which
must outlive the ``kubectl exec`` channel that starts them (see memory
``host-probe-cannot-stop-its-own-transport``):

* **A GPU device holder**, opened *after* the Node Agent ledger shows this
  drill's quiesce succeeded, so every ``VERIFY_NO_GPU_CLIENTS`` attempt reports
  ``GPU device clients are still active`` and the control-plane barrier keeps
  the step WAITING instead of proceeding to ``RESET_GPU``.  The holder is what
  buys the drill its waiting window; without it the reset commits in seconds
  and there is nothing to fence.  It arms on ``QUIESCE_GPU_SERVICES`` rather
  than on ``VERIFY_NO_GPU_CLIENTS``, because on a single node the verify step
  is the *first* step the holder has to break: it is granted the WAITING retry
  (``verify_max_attempts``), while a single-node ``RESET_GPU`` whose
  re-verification raises is not retryable at all.  Both operations are
  accepted so a multi-node variant can arm one step later.
* **An out-of-band reboot**, armed as a bounded ``systemd-run --on-active``
  transient timer that runs ``systemctl reboot``.  The probe cannot reboot
  synchronously: the reboot tears down the exec channel it would have to answer
  over.  It records the pre-reboot boot id in a durable marker file first, so
  ``reboot-status`` can prove after the fact that exactly one boot happened and
  which boot id preceded it.

``QUIESCE_GPU_SERVICES`` stops kubelet, so from the quiesce until the restore
no exec can reach this node -- including the moment the fence is WAITING. The
runner therefore authorizes the reboot *before* the fault is injected, as a
**conditional pre-authorization** (``pre-authorize-reboot``) bound to the run,
node, boot id, device, drill, delay and maintenance window; the timer is placed
then, and ``fire-reboot`` evaluates the barrier itself, against the ledger on
this node, every few seconds until a deadline: this run's workflow has a
succeeded ``QUIESCE_GPU_SERVICES`` row, an executed ``VERIFY_NO_GPU_CLIENTS``
row whose refusal names active clients (the holder's doing), no
``RESET_GPU``/``RESET_ALL_GPUS_NVSWITCHES``/``RESTORE_GPU_SERVICES`` row, an
unchanged boot id, the holder still alive, and a clock inside both the pinned
maintenance window and the pre-authorization's expiry. The reboot delay is the
*earliest* moment after the holder opened, never the authorization. A condition
that cannot clear, or a deadline that passes, records why, cancels the reboot
and disarms the holder.

The reboot is deliberately *not* a control-plane action: no workflow step, no
provider call, no Node Agent command.  That is the fault this case injects.

Every shell command is on an allow-list, the only units this probe may touch
are the three it names after its own run id, and the only reboot form it can
issue is a plain ``systemctl reboot``.  Nothing here executes a GPU reset,
stops a service, or touches the Node Agent.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LEDGER = Path("/var/lib/gpu-fault/node-actions.db")
STATE_DIR = Path("/var/lib/gpu-fault-acceptance")
AGENT_UNIT = "gpu-fault-node-agent.service"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SAFE_DEVICE = re.compile(r"^/dev/nvidia(?:[0-9]|1[0-5])$")
LEDGER_OPERATIONS = (
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESET_ALL_GPUS_NVSWITCHES",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
)
# The ledger rows this drill may wait on before it opens the holder. Quiesce is
# the default: the holder has to be alive before the first verify attempt.
ARM_LEDGER_OPERATIONS = {"QUIESCE_GPU_SERVICES", "VERIFY_NO_GPU_CLIENTS"}
ALLOWED_SYSTEMCTL_VERBS = {"stop", "reset-failed", "show"}
MIN_HOLD_SECONDS = 60
MAX_HOLD_SECONDS = 3600
# The reboot may fire no earlier than this long after the holder opened (the
# quiesce landed), and the delay must be short enough to land inside the
# pinned maintenance window.
MIN_REBOOT_DELAY_SECONDS = 30
MAX_REBOOT_DELAY_SECONDS = 600
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
# The conditional pre-authorization the runner delivers before the injection.
CONDITIONAL_KIND = "conditional-barrier-pre-authorization"
REBOOT_PHASE = "reboot"
LEDGER_SHAPE: dict[str, Any] = {
    "quiesce": "QUIESCE_GPU_SERVICES",
    "verify": "VERIFY_NO_GPU_CLIENTS",
    "verify_refusal": "clients are still active",
    "forbidden": ["RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTORE_GPU_SERVICES"],
}
IN_PROGRESS_STATE = "IN_PROGRESS"
MIN_MAINTENANCE_WINDOW_SECONDS = 30
MAX_MAINTENANCE_WINDOW_SECONDS = 3600
POLL_SECONDS = 5
FIRE_RUNTIME_SLACK_SECONDS = 120

# The holder body, kept a host process (Pods cannot start once GPU services are
# quiesced). It names itself, opens the device read-only, and exits on SIGTERM
# or after its bounded lifetime.
HELPER = r"""
import ctypes
import os
import signal
import sys
import time

name, device, max_hold = sys.argv[1:4]
libc = ctypes.CDLL(None)
libc.prctl(15, name.encode(), 0, 0, 0)
fd = os.open(device, os.O_RDONLY)
print("ready:" + str(os.getpid()), flush=True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
deadline = time.monotonic() + int(max_hold)
try:
    while time.monotonic() < deadline:
        time.sleep(1)
finally:
    os.close(fd)
"""


class ProbeError(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): "
            f"{shlex.join(command)}: {completed.stderr.strip()}"
        )
    return completed


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def safe_id(value: str, label: str) -> str:
    if SAFE_ID.fullmatch(value or "") is None:
        raise ProbeError(f"unsafe {label}")
    return value


def safe_device(value: str) -> str:
    if SAFE_DEVICE.fullmatch(value or "") is None:
        raise ProbeError("device is not an allow-listed /dev/nvidiaN node")
    return value


def _run_digest(run_id: str) -> str:
    return hashlib.sha256(safe_id(run_id, "run ID").encode()).hexdigest()[:16]


def holder_unit(run_id: str) -> str:
    return f"gpu-fault-destr017-holder-{_run_digest(run_id)}"


def arm_unit(run_id: str) -> str:
    return f"gpu-fault-destr017-arm-{_run_digest(run_id)}"


def reboot_unit(run_id: str) -> str:
    return f"gpu-fault-destr017-reboot-{_run_digest(run_id)}"


def _owned_units(run_id: str) -> set[str]:
    result = set()
    for name in (holder_unit(run_id), arm_unit(run_id), reboot_unit(run_id)):
        result.update({name, f"{name}.service", f"{name}.timer"})
    return result


def unit_name(value: str, run_id: str) -> str:
    if value not in _owned_units(run_id):
        raise ProbeError("unit is not in the DESTR-017 probe allow-list")
    return value


def checked_command(command: list[str], run_id: str) -> list[str]:
    """Refuse any command outside the systemctl verb/unit allow-list.

    The Node Agent unit is deliberately absent: DESTR-017 reads its state and
    never changes it, so no mutating verb may name it.
    """

    if not command or command[0] != "systemctl":
        raise ProbeError("only systemctl commands are permitted")
    if len(command) < 3 or command[1] not in ALLOWED_SYSTEMCTL_VERBS:
        raise ProbeError(f"systemctl verb is not permitted: {command[1:2]}")
    unit_name(command[2], run_id)
    for token in command[3:]:
        if not token.startswith("--property="):
            raise ProbeError(f"systemctl argument is not permitted: {token}")
    return command


def reboot_command() -> list[str]:
    """The only reboot form this probe may arm: an ordinary OS reboot.

    Not ``reboot -f``, not ``sysrq``: the case injects a system reboot that a
    site operator or an unrelated automation could plausibly issue, and a
    forced reset would also destroy the Node Agent ledger evidence the case
    reads back afterwards.
    """

    return ["/bin/systemctl", "reboot"]


def checked_max_hold(value: int) -> int:
    if not MIN_HOLD_SECONDS <= int(value) <= MAX_HOLD_SECONDS:
        raise ProbeError(
            f"max hold seconds is outside {MIN_HOLD_SECONDS}..{MAX_HOLD_SECONDS}"
        )
    return int(value)


def checked_reboot_delay(value: int) -> int:
    if not MIN_REBOOT_DELAY_SECONDS <= int(value) <= MAX_REBOOT_DELAY_SECONDS:
        raise ProbeError(
            f"reboot delay is outside {MIN_REBOOT_DELAY_SECONDS}.."
            f"{MAX_REBOOT_DELAY_SECONDS} seconds"
        )
    return int(value)


def state_path(run_id: str, *, state_dir: Path = STATE_DIR) -> Path:
    return state_dir / f"destr017-{safe_id(run_id, 'run ID')}.json"


def write_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def update_state(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    state = read_state(path)
    state.update(value)
    write_state(path, state)
    return state


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Serialize the read-modify-write cycles of the units sharing one state file."""

    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def locked_update(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    with locked(path):
        return update_state(path, value)


def ledger_rows(ledger: Path = LEDGER) -> list[dict[str, Any]]:
    if not ledger.is_file():
        return []
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT command_id, completed_at, attempt, state, operation, started_at, "
            "workflow_request_id, incident_id, agent_generation, payload "
            "FROM results WHERE operation IN "
            f"({', '.join('?' for _ in LEDGER_OPERATIONS)}) "
            "ORDER BY completed_at, command_id",
            LEDGER_OPERATIONS,
        ).fetchall()
    finally:
        connection.close()
    result = []
    for row in rows:
        try:
            error = (json.loads(row[9]) or {}).get("error") if row[9] else None
        except (TypeError, ValueError, AttributeError):
            error = None
        result.append(
            {
                "command_id": row[0],
                "completed_at": row[1],
                "attempt": row[2],
                "state": row[3],
                "operation": row[4],
                "started_at": row[5],
                "workflow_request_id": row[6],
                "incident_id": row[7],
                "agent_generation": row[8],
                "error": error,
            }
        )
    return result


def match_ledger_row(
    rows: list[dict[str, Any]],
    *,
    operation: str,
    baseline_command_ids: set[str],
    armed_at: str,
) -> dict[str, Any] | None:
    """The first succeeded ``operation`` row this drill produced after arming.

    A baseline row (present before arming) or a row completed before the arm
    timestamp belongs to an earlier drill and must not trigger the holder; an
    in-progress or failed row is not a success.
    """

    for row in rows:
        if row["operation"] != operation:
            continue
        if row["state"] != "SUCCEEDED":
            continue
        if row["command_id"] in baseline_command_ids:
            continue
        completed_at = str(row.get("completed_at") or "")
        if not completed_at or completed_at < armed_at:
            continue
        return row
    return None


def _unit_state(unit: str, *properties: str) -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            unit,
            *(f"--property={item}" for item in properties),
        ],
        check=False,
    )
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def _agent_unit_state() -> dict[str, str]:
    return _unit_state(
        AGENT_UNIT, "UnitFileState", "ActiveState", "SubState", "MainPID"
    )


def _holder_unit_state(run_id: str) -> dict[str, str]:
    return _unit_state(
        holder_unit(run_id) + ".service",
        "LoadState",
        "ActiveState",
        "SubState",
        "MainPID",
    )


def holder_active(run_id: str) -> bool:
    """Whether the GPU device holder unit is alive right now."""

    return _holder_unit_state(run_id).get("ActiveState") == "active"


def _reboot_unit_state(run_id: str) -> dict[str, dict[str, str]]:
    return {
        "timer": _unit_state(
            reboot_unit(run_id) + ".timer",
            "LoadState",
            "ActiveState",
            "SubState",
            "NextElapseUSecRealtime",
        ),
        "service": _unit_state(
            reboot_unit(run_id) + ".service",
            "LoadState",
            "ActiveState",
            "SubState",
            "Result",
        ),
    }


def boot_id() -> str:
    return BOOT_ID_PATH.read_text(encoding="utf-8").strip()


def _clear_unit(unit_with_suffix: str, run_id: str) -> None:
    run(checked_command(["systemctl", "stop", unit_with_suffix], run_id), check=False)
    run(
        checked_command(["systemctl", "reset-failed", unit_with_suffix], run_id),
        check=False,
    )


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def record_boot_observation(
    state: dict[str, Any],
    current_boot_id: str,
    *,
    observed_at: str,
) -> dict[str, Any]:
    """Append ``current_boot_id`` to the observed boot history, once per boot.

    The history is what proves *exactly one* boot happened: the runner reads
    the same durable file before arming, right after the node returns and once
    more at the end of the case, and any second reboot would add a third entry.
    """

    history = [str(item) for item in state.get("observed_boot_ids") or []]
    if not history or history[-1] != current_boot_id:
        history.append(current_boot_id)
    result = dict(state)
    result["observed_boot_ids"] = history
    result["boot_changes"] = max(0, len(history) - 1)
    result["boot_observed_at"] = observed_at
    return result


def watch_ledger(arguments: argparse.Namespace) -> None:
    """Poll the ledger for the arm row, then start the holder. Runs detached."""

    run_id = safe_id(arguments.run_id, "run ID")
    path = state_path(run_id)
    state = read_state(path)
    operation = str(state.get("after_ledger_op") or "")
    if operation not in ARM_LEDGER_OPERATIONS:
        raise ProbeError("no armed ledger operation recorded for this run")
    device = safe_device(str(state.get("device") or ""))
    max_hold = checked_max_hold(int(state.get("max_hold_seconds") or 0))
    baseline = set(state.get("baseline_command_ids") or [])
    armed_at = str(state.get("armed_at") or "")
    deadline = time.monotonic() + max_hold
    matched: dict[str, Any] | None = None
    while time.monotonic() < deadline and matched is None:
        matched = match_ledger_row(
            ledger_rows(),
            operation=operation,
            baseline_command_ids=baseline,
            armed_at=armed_at,
        )
        if matched is None:
            time.sleep(1)
    if matched is None:
        locked_update(path, {"holder_error": "arm ledger row never appeared"})
        return
    if state.get("reboot_delay_seconds") is not None:
        require_maintenance_window(str(state.get("maintenance_window_end") or ""))
    unit = holder_unit(run_id)
    _clear_unit(unit + ".service", run_id)
    started_at = datetime.now(timezone.utc).isoformat()
    run(
        [
            "systemd-run",
            "--unit",
            unit,
            f"--property=RuntimeMaxSec={max_hold}",
            "--property=KillMode=control-group",
            "/opt/gpu-fault/current/venv/bin/python",
            "-c",
            HELPER,
            f"destr017-{run_id[:8]}",
            device,
            str(max_hold),
        ]
    )
    locked_update(
        path,
        {
            "matched_row": matched,
            "hold_started_at": started_at,
            "holder_unit": unit + ".service",
        },
    )
    # The reboot timer was placed by ``pre-authorize-reboot`` and decides for
    # itself, from the ledger, whether and when the barrier holds.


def arm_holder(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    safe_id(arguments.drill_id, "drill ID")
    device = safe_device(arguments.device)
    max_hold = checked_max_hold(arguments.max_hold_seconds)
    if arguments.after_ledger_op not in ARM_LEDGER_OPERATIONS:
        raise ProbeError("after-ledger-op is not permitted for arming")
    if Path(arguments.probe_script).name != Path(__file__).name:
        raise ProbeError("probe script identity mismatch")
    baseline_ids = [
        row["command_id"]
        for row in ledger_rows()
        if row["operation"] == arguments.after_ledger_op
    ]
    reboot_delay: int | None = None
    if getattr(arguments, "reboot_delay_seconds", None) is not None:
        reboot_delay = checked_reboot_delay(arguments.reboot_delay_seconds)
        require_maintenance_window(getattr(arguments, "maintenance_window_end", ""))
    path = state_path(run_id)
    armed_at = datetime.now(timezone.utc).isoformat()
    current = boot_id()
    state = update_state(
        path,
        {
            "run_id": run_id,
            "drill_id": arguments.drill_id,
            "device": device,
            "max_hold_seconds": max_hold,
            "after_ledger_op": arguments.after_ledger_op,
            "baseline_command_ids": baseline_ids,
            "armed_at": armed_at,
            "boot_id": current,
            "reboot_delay_seconds": reboot_delay,
            "maintenance_window_end": getattr(arguments, "maintenance_window_end", ""),
        },
    )
    write_state(path, record_boot_observation(state, current, observed_at=armed_at))
    unit = arm_unit(run_id)
    _clear_unit(unit + ".service", run_id)
    run(
        [
            "systemd-run",
            "--unit",
            unit,
            f"--property=RuntimeMaxSec={max_hold + 60}",
            "/opt/gpu-fault/current/venv/bin/python",
            str(Path(arguments.probe_script)),
            "watch-ledger",
            "--run-id",
            run_id,
        ]
    )
    emit(
        {
            "run_id": run_id,
            "arm_unit": unit + ".service",
            "device": device,
            "after_ledger_op": arguments.after_ledger_op,
            "baseline_command_ids": baseline_ids,
            "armed_at": armed_at,
            "reboot_delay_seconds": reboot_delay,
        }
    )


def disarm_holder(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    _clear_unit(arm_unit(run_id) + ".service", run_id)
    _clear_unit(holder_unit(run_id) + ".service", run_id)
    path = state_path(run_id)
    if path.is_file():
        now = datetime.now(timezone.utc).isoformat()
        with locked(path):
            state = read_state(path)
            state["disarmed_at"] = state.get("disarmed_at") or now
            state["disarm_reason"] = state.get("disarm_reason") or "disarm-holder"
            write_state(path, state)
    emit(
        {
            "run_id": run_id,
            "holder_unit_state": _holder_unit_state(run_id),
        }
    )


def holder_status(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    state = read_state(state_path(run_id))
    unit_state = _holder_unit_state(run_id)
    emit(
        {
            "run_id": run_id,
            "holder_unit": holder_unit(run_id) + ".service",
            "unit_state": unit_state,
            "pid": unit_state.get("MainPID"),
            "matched_row": state.get("matched_row"),
            "hold_started_at": state.get("hold_started_at"),
            "holder_error": state.get("holder_error"),
            "after_ledger_op": state.get("after_ledger_op"),
            "pre_authorized_at": state.get("pre_authorized_at"),
            "disarmed_at": state.get("disarmed_at"),
            "disarm_reason": state.get("disarm_reason"),
        }
    )


def require_maintenance_window(value: str) -> datetime:
    try:
        deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ProbeError("reboot requires an explicit maintenance deadline") from exc
    if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline:
        raise ProbeError("maintenance window ended before reboot")
    return deadline


# --------------------------------------------------------------------------- #
# Conditional pre-authorization and the host-side barrier condition
# --------------------------------------------------------------------------- #
def check_pre_authorization(
    state: dict[str, Any],
    proof: Any,
    *,
    run_id: str,
    boot_id: str,
    now: datetime,
) -> dict[str, Any]:
    """Refuse a proof that does not bind this armed holder, now, on this boot."""

    if state.get("run_id") != run_id:
        raise ProbeError("no armed holder is recorded for this run")
    if state.get("disarmed_at"):
        raise ProbeError("the holder was already disarmed")
    if state.get("pre_authorization") or state.get("reboot_armed_at"):
        raise ProbeError("the reboot is already pre-authorized")
    if state.get("reboot_cancelled_at"):
        raise ProbeError("the reboot was cancelled")
    if (
        not isinstance(proof, dict)
        or proof.get("kind") != CONDITIONAL_KIND
        or proof.get("conditional") is not True
    ):
        raise ProbeError("authorization is not a conditional barrier pre-authorization")
    if (
        proof.get("run_id") != run_id
        or proof.get("device") != state.get("device")
        or proof.get("drill_id") != state.get("drill_id")
    ):
        raise ProbeError("pre-authorization does not bind this holder")
    if (
        not boot_id
        or proof.get("boot_id") != boot_id
        or state.get("boot_id") != boot_id
    ):
        raise ProbeError("pre-authorization belongs to another boot")
    if SAFE_ID.fullmatch(str(proof.get("node_id") or "")) is None:
        raise ProbeError("pre-authorization names no node")
    if proof.get("ledger") != LEDGER_SHAPE:
        raise ProbeError("pre-authorization ledger shape is not this probe's")
    expires = _parse_time(proof.get("expires_at"))
    window_end = _parse_time(proof.get("maintenance_window_end"))
    authorized = _parse_time(proof.get("authorized_at"))
    if expires is None or window_end is None or authorized is None:
        raise ProbeError("pre-authorization has no valid deadline")
    if now >= expires or now >= window_end or authorized > now + timedelta(seconds=90):
        raise ProbeError("pre-authorization is expired or future-dated")
    max_hold = int(state.get("max_hold_seconds") or 0)
    if expires > now + timedelta(seconds=max_hold):
        raise ProbeError("pre-authorization outlives the holder's bounded lifetime")
    window_seconds = proof.get("maintenance_window_seconds")
    if (
        type(window_seconds) is not int
        or not MIN_MAINTENANCE_WINDOW_SECONDS
        <= window_seconds
        <= MAX_MAINTENANCE_WINDOW_SECONDS
    ):
        raise ProbeError("pre-authorization agent maintenance window is out of bounds")
    delays = proof.get("not_before_seconds")
    if (
        not isinstance(delays, dict)
        or set(delays) != {REBOOT_PHASE}
        or type(delays[REBOOT_PHASE]) is not int
    ):
        raise ProbeError("pre-authorization must name exactly the reboot delay")
    return proof


def arm_reboot(run_id: str, delay: int, path: Path) -> dict[str, Any]:
    """Write the durable boot-id marker, then arm the transient reboot timer.

    Called by ``pre-authorize-reboot`` before this drill's quiesce stops
    kubelet. The marker file is written *before* the timer exists, so a reboot
    can never fire without a durable record of the boot id it replaced. The
    timer elapses after ``delay`` seconds; ``fire-reboot`` then keeps waiting
    for the barrier condition, so ``delay`` is an earliest time, never the
    authorization.
    """

    delay = checked_reboot_delay(delay)
    armed_at = datetime.now(timezone.utc)
    recorded = read_state(path)
    deadline = require_maintenance_window(
        str(recorded.get("maintenance_window_end") or "")
    )
    if armed_at + timedelta(seconds=delay) >= deadline:
        raise ProbeError("scheduled reboot would outlive the maintenance window")
    proof = recorded.get("pre_authorization")
    if (
        not isinstance(proof, dict)
        or proof.get("run_id") != run_id
        or recorded.get("reboot_armed_at")
        or recorded.get("reboot_cancelled_at")
    ):
        raise ProbeError("reboot requires a conditional pre-authorization")
    expires = _parse_time(proof.get("expires_at"))
    if expires is None or armed_at + timedelta(seconds=delay) >= expires:
        raise ProbeError("reboot would outlive the pre-authorization")
    if (proof.get("not_before_seconds") or {}).get(REBOOT_PHASE) != delay:
        raise ProbeError("reboot delay differs from the pre-authorized delay")
    current = boot_id()
    unit = reboot_unit(run_id)
    poll_starts_at = armed_at + timedelta(seconds=delay)
    state = update_state(
        path,
        {
            "run_id": run_id,
            "reboot_armed_at": armed_at.isoformat(),
            "reboot_delay_seconds": delay,
            "reboot_not_before_seconds": delay,
            "boot_id_before_reboot": current,
            "reboot_poll_starts_at": poll_starts_at.isoformat(),
            "reboot_unit": unit + ".timer",
            "reboot_command": reboot_command(),
            "reboot_cancelled_at": None,
        },
    )
    write_state(
        path,
        record_boot_observation(state, current, observed_at=armed_at.isoformat()),
    )
    _clear_unit(unit + ".timer", run_id)
    _clear_unit(unit + ".service", run_id)
    runtime = int((expires - armed_at).total_seconds()) + FIRE_RUNTIME_SLACK_SECONDS
    run(
        [
            "systemd-run",
            f"--unit={unit}",
            f"--on-active={delay}s",
            "--timer-property=AccuracySec=1s",
            "--property=Type=oneshot",
            f"--property=RuntimeMaxSec={runtime}",
            sys.executable,
            str(Path(__file__).resolve()),
            "fire-reboot",
            "--run-id",
            run_id,
        ]
    )
    return {
        "run_id": run_id,
        "reboot_unit": unit + ".timer",
        "reboot_armed_at": armed_at.isoformat(),
        "reboot_poll_starts_at": poll_starts_at.isoformat(),
        "reboot_delay_seconds": delay,
        "boot_id_before_reboot": current,
        "units": _reboot_unit_state(run_id),
    }


def pre_authorize_reboot(arguments: argparse.Namespace) -> None:
    """Record the conditional authorization and arm the reboot timer.

    Exec'd by the runner right after ``arm-holder`` and *before* the fault is
    injected: the quiesce that follows the injection takes kubelet down, so
    this is the last moment the runner can reach the node until the restore.
    """

    run_id = safe_id(arguments.run_id, "run ID")
    path = state_path(run_id)
    proof = json.loads(arguments.authorization)
    delay = checked_reboot_delay(arguments.delay_seconds)
    with locked(path):
        state = read_state(path)
        now = datetime.now(timezone.utc)
        check_pre_authorization(state, proof, run_id=run_id, boot_id=boot_id(), now=now)
        if proof["not_before_seconds"][REBOOT_PHASE] != delay:
            raise ProbeError("pre-authorization delay does not match the reboot delay")
        baseline = sorted(
            {
                str(row["workflow_request_id"])
                for row in ledger_rows()
                if row.get("workflow_request_id")
            }
        )
        state.update(
            {
                "pre_authorization": proof,
                "pre_authorized_at": now.isoformat(),
                "ledger_baseline_workflow_ids": baseline,
            }
        )
        write_state(path, state)
        record = arm_reboot(run_id, delay, path)
    emit(
        {
            **record,
            "pre_authorized_at": now.isoformat(),
            "expires_at": proof["expires_at"],
            "ledger_baseline_workflow_ids": baseline,
        }
    )


def new_workflow_ids(
    rows: list[dict[str, Any]], *, baseline: set[str], not_before: str
) -> set[str]:
    """Workflow ids with a row written since the pre-authorization."""

    result: set[str] = set()
    for row in rows:
        workflow_id = str(row.get("workflow_request_id") or "")
        if not workflow_id or workflow_id in baseline:
            continue
        stamp = str(row.get("completed_at") or row.get("started_at") or "")
        if stamp and stamp >= not_before:
            result.add(workflow_id)
    return result


def _verdict(
    reason: str,
    *,
    holds: bool = False,
    final: bool = False,
    condition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "holds": holds,
        "final": final,
        "reason": reason,
        "condition": dict(condition or {}),
    }


def _holder_gate(
    state: dict[str, Any], *, now: datetime, boot_id: str, holder_active: bool
) -> dict[str, Any] | None:
    """The non-ledger half of the condition: binding, deadlines, the holder."""

    proof = state.get("pre_authorization")
    if not isinstance(proof, dict) or proof.get("kind") != CONDITIONAL_KIND:
        return _verdict("no conditional pre-authorization is recorded", final=True)
    if state.get("disarmed_at") or state.get("reboot_cancelled_at"):
        return _verdict("the reboot was cancelled or the holder disarmed", final=True)
    if (
        not boot_id
        or boot_id != proof.get("boot_id")
        or boot_id != state.get("boot_id")
        or boot_id != state.get("boot_id_before_reboot")
    ):
        return _verdict("the boot id changed since the pre-authorization", final=True)
    if proof.get("ledger") != LEDGER_SHAPE:
        return _verdict(
            "the pre-authorization ledger shape is not this probe's", final=True
        )
    expires = _parse_time(proof.get("expires_at"))
    window_end = _parse_time(proof.get("maintenance_window_end"))
    if expires is None or window_end is None:
        return _verdict("the pre-authorization has no valid deadline", final=True)
    if now >= min(expires, window_end):
        return _verdict(
            "the pre-authorization expired before the barrier condition held",
            final=True,
        )
    delay = (proof.get("not_before_seconds") or {}).get(REBOOT_PHASE)
    if type(delay) is not int:
        return _verdict("the pre-authorization names no reboot delay", final=True)
    hold_started = _parse_time(state.get("hold_started_at"))
    if hold_started is None:
        return _verdict("the holder has not started")
    if not holder_active:
        return _verdict("the holder is no longer active", final=True)
    if now < hold_started + timedelta(seconds=delay):
        return _verdict(
            f"the reboot is not due until {delay}s after the holder started"
        )
    return None


def barrier_condition(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    now: datetime,
    boot_id: str,
    holder_active: bool,
) -> dict[str, Any]:
    """Whether the reboot may fire now, judged from this node alone.

    ``holds`` means reboot; ``final`` means stop polling because the reason can
    never clear (the caller records it, cancels and disarms). Anything else is
    a wait. The barrier is proven by the ledger: this run's workflow -- the one
    new workflow written since the pre-authorization -- has a succeeded
    quiesce, an executed client verification that refused on active clients
    (the holder's doing), and nothing beyond it.
    """

    gate = _holder_gate(state, now=now, boot_id=boot_id, holder_active=holder_active)
    if gate is not None:
        return gate
    proof = state["pre_authorization"]
    node = str(proof.get("node_id") or "")
    candidates = new_workflow_ids(
        rows,
        baseline=set(state.get("ledger_baseline_workflow_ids") or []),
        not_before=str(state.get("pre_authorized_at") or proof.get("authorized_at")),
    )
    if not candidates:
        return _verdict("no workflow of this run has reached the ledger")
    if len(candidates) > 1:
        return _verdict(
            f"more than one new workflow reached the ledger: {sorted(candidates)}",
            final=True,
        )
    workflow_id = candidates.pop()
    mine = [row for row in rows if row.get("workflow_request_id") == workflow_id]
    forbidden = sorted(
        {
            str(row.get("operation"))
            for row in mine
            if row.get("operation") in LEDGER_SHAPE["forbidden"]
        }
    )
    if forbidden:
        return _verdict(
            f"the workflow advanced beyond the barrier: {forbidden}", final=True
        )
    quiesce = [row for row in mine if row.get("operation") == LEDGER_SHAPE["quiesce"]]
    succeeded = [row for row in quiesce if row.get("state") == "SUCCEEDED"]
    if not succeeded:
        if any(row.get("state") != IN_PROGRESS_STATE for row in quiesce):
            return _verdict("the quiesce did not succeed", final=True)
        return _verdict("the quiesce has not succeeded")
    quiesce_row = succeeded[-1]
    if f"/{node}/" not in str(quiesce_row.get("command_id") or ""):
        return _verdict("the quiesce row does not name this node", final=True)
    # The control plane pins the window from the quiesce step's start; the
    # earlier of the row's two timestamps is the conservative local estimate.
    stamps = [
        stamp
        for stamp in (
            _parse_time(quiesce_row.get("started_at")),
            _parse_time(quiesce_row.get("completed_at")),
        )
        if stamp is not None
    ]
    pinned = min(stamps) if stamps else None
    if pinned is None:
        return _verdict("the pinned maintenance window cannot be derived", final=True)
    pinned_end = pinned + timedelta(seconds=int(proof["maintenance_window_seconds"]))
    if now >= pinned_end:
        return _verdict("the pinned maintenance window has expired", final=True)
    verify = [row for row in mine if row.get("operation") == LEDGER_SHAPE["verify"]]
    if any(row.get("state") == "SUCCEEDED" for row in verify):
        return _verdict(
            "the client verification succeeded; the holder did not hold", final=True
        )
    executed = [
        row
        for row in verify
        if row.get("state") != IN_PROGRESS_STATE and row.get("completed_at")
    ]
    if not executed:
        return _verdict("the client verification has not executed")
    latest = max(executed, key=lambda row: int(row.get("attempt") or 0))
    if LEDGER_SHAPE["verify_refusal"] not in str(latest.get("error") or ""):
        return _verdict("the latest client verification did not refuse on clients")
    return _verdict(
        "the barrier holds",
        holds=True,
        condition={
            "workflow_request_id": workflow_id,
            "incident_id": quiesce_row.get("incident_id"),
            "boot_id": boot_id,
            "agent_generation": quiesce_row.get("agent_generation"),
            "quiesce_command_id": quiesce_row.get("command_id"),
            "verify_command_id": latest.get("command_id"),
            "verify_attempt": latest.get("attempt"),
            "verify_completed_at": latest.get("completed_at"),
            "pinned_window_expires_at": pinned_end.isoformat(),
            "observed_at": now.isoformat(),
        },
    )


def refuse_reboot(run_id: str, path: Path, state: dict[str, Any], reason: str) -> None:
    """Record why the reboot will never fire, cancel it and disarm the holder."""

    now = datetime.now(timezone.utc).isoformat()
    state["reboot_refusal"] = {"refused_at": now, "reason": reason}
    state["reboot_cancelled_at"] = state.get("reboot_cancelled_at") or now
    state["disarmed_at"] = state.get("disarmed_at") or now
    state["disarm_reason"] = state.get("disarm_reason") or reason
    write_state(path, state)
    _clear_unit(arm_unit(run_id) + ".service", run_id)
    _clear_unit(holder_unit(run_id) + ".service", run_id)


def _reboot_intent(path: Path, run_id: str) -> dict[str, Any]:
    """Wait for the barrier condition; record the fire intent under the lock."""

    while True:
        with locked(path):
            state = read_state(path)
            if state.get("run_id") != run_id:
                raise ProbeError("reboot was not authorized for this run")
            if state.get("fire_requested_at"):
                raise ProbeError("reboot authorization was consumed")
            if state.get("reboot_cancelled_at"):
                raise ProbeError("reboot was cancelled")
            try:
                require_maintenance_window(
                    str(state.get("maintenance_window_end") or "")
                )
            except ProbeError as exc:
                refuse_reboot(run_id, path, state, str(exc))
                raise
            verdict = barrier_condition(
                state,
                ledger_rows(),
                now=datetime.now(timezone.utc),
                boot_id=boot_id(),
                holder_active=holder_active(run_id),
            )
            if verdict["holds"]:
                state["fire_requested_at"] = verdict["condition"]["observed_at"]
                state["fire_condition"] = verdict["condition"]
                write_state(path, state)
                return verdict["condition"]
            if verdict["final"]:
                refuse_reboot(run_id, path, state, verdict["reason"])
                raise ProbeError(f"reboot refused: {verdict['reason']}")
        time.sleep(POLL_SECONDS)


def fire_reboot(arguments: argparse.Namespace) -> None:
    """Run by the transient timer: reboot once the barrier holds.

    The fire intent is durable before ``systemctl reboot``, so the marker file
    the node comes back with says the reboot was requested, on which boot and
    against which ledger rows.
    """

    run_id = safe_id(arguments.run_id, "run ID")
    path = state_path(run_id)
    condition = _reboot_intent(path, run_id)
    run(reboot_command())
    emit({"run_id": run_id, "fire_requested_at": condition["observed_at"]})


def reboot_status(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    path = state_path(run_id)
    state = read_state(path)
    current = boot_id()
    observed_at = datetime.now(timezone.utc).isoformat()
    if state:
        state = record_boot_observation(state, current, observed_at=observed_at)
        write_state(path, state)
    before = str(state.get("boot_id_before_reboot") or "")
    proof = state.get("pre_authorization") or {}
    emit(
        {
            "run_id": run_id,
            "observed_at": observed_at,
            "armed": bool(state.get("reboot_armed_at")),
            "reboot_armed_at": state.get("reboot_armed_at"),
            "reboot_poll_starts_at": state.get("reboot_poll_starts_at"),
            "reboot_delay_seconds": state.get("reboot_delay_seconds"),
            "reboot_cancelled_at": state.get("reboot_cancelled_at"),
            "pre_authorized_at": state.get("pre_authorized_at"),
            "pre_authorization_expires_at": proof.get("expires_at"),
            "fire_requested_at": state.get("fire_requested_at"),
            "condition": state.get("fire_condition"),
            "refusal": state.get("reboot_refusal"),
            "boot_id_before_reboot": before or None,
            "boot_id_now": current,
            "fired": bool(state.get("fire_requested_at"))
            and bool(before)
            and before != current,
            "observed_boot_ids": state.get("observed_boot_ids") or [],
            "boot_changes": state.get("boot_changes", 0),
            "units": _reboot_unit_state(run_id),
        }
    )


def cancel_reboot(arguments: argparse.Namespace) -> None:
    """Stop the reboot timer if it has not fired. Idempotent."""

    run_id = safe_id(arguments.run_id, "run ID")
    unit = reboot_unit(run_id)
    path = state_path(run_id)
    current = boot_id()
    cancelled_at = datetime.now(timezone.utc).isoformat()
    with locked(path):
        state = read_state(path)
        if state:
            state = record_boot_observation(state, current, observed_at=cancelled_at)
            state["reboot_cancelled_at"] = state.get("reboot_cancelled_at") or (
                cancelled_at
            )
            write_state(path, state)
    before = _reboot_unit_state(run_id)
    _clear_unit(unit + ".timer", run_id)
    _clear_unit(unit + ".service", run_id)
    recorded = str(state.get("boot_id_before_reboot") or "")
    emit(
        {
            "run_id": run_id,
            "cancelled_at": cancelled_at,
            "already_fired": bool(recorded) and recorded != current,
            "units_before": before,
            "units_after": _reboot_unit_state(run_id),
        }
    )


def clear_state(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    path = state_path(run_id)
    for unit in (
        arm_unit(run_id) + ".service",
        holder_unit(run_id) + ".service",
        reboot_unit(run_id) + ".timer",
        reboot_unit(run_id) + ".service",
    ):
        state = _unit_state(unit, "LoadState", "ActiveState")
        if state.get("LoadState") != "not-found" and state.get("ActiveState") not in {
            "inactive",
            "failed",
        }:
            raise ProbeError("cannot clear state while a probe unit may still run")
    with locked(path):
        state = read_state(path)
        if state and (
            state.get("run_id") != run_id
            or not state.get("reboot_cancelled_at")
            or not state.get("disarmed_at")
        ):
            raise ProbeError("probe state has no completed cancellation proof")
        path.unlink(missing_ok=True)
    emit({"run_id": run_id, "state_cleared": True})


def snapshot(arguments: argparse.Namespace) -> None:
    run_id = arguments.run_id
    payload: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "boot_id": boot_id(),
        "agent_unit": _agent_unit_state(),
    }
    if run_id:
        safe_id(run_id, "run ID")
        payload["holder_unit"] = _holder_unit_state(run_id)
        payload["reboot_units"] = _reboot_unit_state(run_id)
        payload["state"] = read_state(state_path(run_id))
    emit(payload)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    arm = commands.add_parser("arm-holder")
    arm.add_argument("--device", required=True)
    arm.add_argument("--drill-id", required=True)
    arm.add_argument(
        "--after-ledger-op",
        default="QUIESCE_GPU_SERVICES",
        choices=sorted(ARM_LEDGER_OPERATIONS),
    )
    arm.add_argument("--max-hold-seconds", type=int, default=900)
    arm.add_argument("--run-id", required=True)
    arm.add_argument("--probe-script", required=True)
    arm.add_argument("--maintenance-window-end", default="")
    arm.add_argument(
        "--reboot-delay-seconds",
        type=int,
        default=None,
        help=(
            "if set, records the out-of-band reboot's earliest offset after the "
            "holder opens; the reboot itself is placed by pre-authorize-reboot and "
            f"the delay must be within {MIN_REBOOT_DELAY_SECONDS}.."
            f"{MAX_REBOOT_DELAY_SECONDS} seconds"
        ),
    )
    arm.set_defaults(handler=arm_holder)

    watch = commands.add_parser("watch-ledger")
    watch.add_argument("--run-id", required=True)
    watch.set_defaults(handler=watch_ledger)

    disarm = commands.add_parser("disarm-holder")
    disarm.add_argument("--run-id", required=True)
    disarm.set_defaults(handler=disarm_holder)

    status = commands.add_parser("holder-status")
    status.add_argument("--run-id", required=True)
    status.set_defaults(handler=holder_status)

    authorize = commands.add_parser("pre-authorize-reboot")
    authorize.add_argument("--run-id", required=True)
    authorize.add_argument("--authorization", required=True)
    authorize.add_argument(
        "--delay-seconds",
        type=int,
        default=MIN_REBOOT_DELAY_SECONDS,
        help=(
            "earliest seconds after the holder opens before the barrier condition "
            f"may reboot; {MIN_REBOOT_DELAY_SECONDS}..{MAX_REBOOT_DELAY_SECONDS}"
        ),
    )
    authorize.set_defaults(handler=pre_authorize_reboot)

    fire = commands.add_parser("fire-reboot")
    fire.add_argument("--run-id", required=True)
    fire.set_defaults(handler=fire_reboot)

    clear = commands.add_parser("clear-state")
    clear.add_argument("--run-id", required=True)
    clear.set_defaults(handler=clear_state)

    reboot_state = commands.add_parser("reboot-status")
    reboot_state.add_argument("--run-id", required=True)
    reboot_state.set_defaults(handler=reboot_status)

    cancel = commands.add_parser("cancel-reboot")
    cancel.add_argument("--run-id", required=True)
    cancel.set_defaults(handler=cancel_reboot)

    show = commands.add_parser("snapshot")
    show.add_argument("--run-id", default="")
    show.set_defaults(handler=snapshot)
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - the probe reports as JSON
        emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
