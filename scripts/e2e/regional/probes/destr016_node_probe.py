#!/usr/bin/env python3
"""On-node probe for GF-REGIONAL-DESTR-016.

One node-local job the control plane cannot do for the runner: hold a GPU
device open from the moment this drill's ``QUIESCE_GPU_SERVICES`` succeeds, so
the ``VERIFY_NO_GPU_CLIENTS`` step that follows it finds a live client, reports
``clients are still active`` and parks the RESET_GPU workflow at a WAITING
remote command with the quiesce already applied and not yet restored -- the
dirty boundary a stronger XID 79 must preempt and hand off.

Both the poll-and-arm and the holder run as ``systemd-run`` transient units so
they survive the ``kubectl exec`` channel the probe answers over (see memory
``host-probe-cannot-stop-its-own-transport``); the holder also has to outlive
the Pod, because no Pod can start once GPU services are quiesced.

``QUIESCE_GPU_SERVICES`` stops kubelet, and with it the exec channel every
probe answers over, so from the quiesce until ``RESTORE_GPU_SERVICES`` nothing
can reach this node. The runner therefore delivers its authorization *before*
the fault is injected, as a **conditional pre-authorization** (``pre-authorize``)
bound to the run, node, boot id, device, drill and maintenance window. The
absorb (XID 46) and escalation (XID 79) writes that must land inside the
WAITING window are handed to ``systemd-run`` timers at that moment, and each
timer evaluates the barrier itself when it fires -- against the Node Agent
ledger on this node, every few seconds until a deadline: this run's workflow
has a succeeded ``QUIESCE_GPU_SERVICES`` row, an executed ``VERIFY_NO_GPU_CLIENTS``
row whose refusal names active clients (the holder is what makes it refuse), no
``RESET_GPU``/``RESET_ALL_GPUS_NVSWITCHES``/``RESTORE_GPU_SERVICES`` row, an
unchanged boot id, the holder still alive, and a clock inside both the pinned
maintenance window and the pre-authorization's own expiry. The relative delays
are *earliest* times, never the authorization. A condition that cannot clear,
or a deadline that passes, records why and disarms the holder.

Every shell command is on an allow-list. The Node Agent unit may only be read,
never stopped, disabled or restarted: this case needs the Agent alive to take
the reboot command and to re-register after it. Nothing here resets a GPU,
reboots a node, or writes to /dev/kmsg -- the shared destructive probe owns the
XID writes; this probe only decides *when* they may happen.
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
SAFE_BDF = re.compile(r"^[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}(?:\.[0-7])?$")
SAFE_INJECT_SCRIPT = re.compile(r"^/run/gpu-fault-host-probe-[0-9a-f]{6,32}\.py$")
# Phase order is fire order: the absorbed same-rank fault first, the escalation
# after it. ``barrier_condition`` refuses to fire a phase before every earlier
# planned phase has fired.
INJECTION_PHASES: dict[str, str] = {"absorb": "write-xid46", "escalate": "write-xid79"}
MIN_INJECTION_DELAY_SECONDS = 10
LEDGER_OPERATIONS = (
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESET_ALL_GPUS_NVSWITCHES",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
)
# The only ledger operations this drill may wait on to arm the holder.
# QUIESCE_GPU_SERVICES is the one the runner uses: the holder has to exist
# *before* the verify step runs, because verify is the step that must park.
# VERIFY_NO_GPU_CLIENTS is kept for a re-arm after an already-verified step.
ARM_LEDGER_OPERATIONS = ("QUIESCE_GPU_SERVICES", "VERIFY_NO_GPU_CLIENTS")
# The step whose success means the arm race was lost: the verify already
# passed without the holder, so RESET_GPU commits and the case has no dirty
# boundary left to preempt. The runner stops instead of injecting XID 79.
RACE_OPERATION = "VERIFY_NO_GPU_CLIENTS"
READ_ONLY_SYSTEMCTL_VERBS = ("show", "is-active", "is-enabled")
OWNED_UNIT_SYSTEMCTL_VERBS = ("stop", "reset-failed", *READ_ONLY_SYSTEMCTL_VERBS)
MIN_HOLD_SECONDS = 60
MAX_HOLD_SECONDS = 3600
# The conditional pre-authorization the runner delivers before the injection.
# The ledger shape is spelled here and in the runner's proof builder; a proof
# whose shape differs is refused rather than interpreted.
CONDITIONAL_KIND = "conditional-barrier-pre-authorization"
LEDGER_SHAPE: dict[str, Any] = {
    "quiesce": "QUIESCE_GPU_SERVICES",
    "verify": "VERIFY_NO_GPU_CLIENTS",
    "verify_refusal": "clients are still active",
    "forbidden": ["RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTORE_GPU_SERVICES"],
}
IN_PROGRESS_STATE = "IN_PROGRESS"
MIN_MAINTENANCE_WINDOW_SECONDS = 30
MAX_MAINTENANCE_WINDOW_SECONDS = 3600
# How often a fired timer re-evaluates the barrier, and how much longer than
# the pre-authorization its transient unit may live.
POLL_SECONDS = 5
FIRE_RUNTIME_SLACK_SECONDS = 120

# The holder body. It names itself so `holder-status` can be read by a human,
# opens the device read-only, and exits on SIGTERM or after its bounded
# lifetime. A read-only open is enough: nvidia-smi counts it as a compute
# client, and it cannot perturb the device.
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
    return f"gpu-fault-destr016-holder-{_run_digest(run_id)}"


def arm_unit(run_id: str) -> str:
    return f"gpu-fault-destr016-arm-{_run_digest(run_id)}"


def injection_unit(run_id: str, phase: str) -> str:
    if phase not in INJECTION_PHASES:
        raise ProbeError("unknown injection phase")
    return f"gpu-fault-destr016-{phase}-{_run_digest(run_id)}"


def _owned_units(run_id: str) -> set[str]:
    owned = {
        holder_unit(run_id),
        holder_unit(run_id) + ".service",
        arm_unit(run_id),
        arm_unit(run_id) + ".service",
    }
    for phase in INJECTION_PHASES:
        unit = injection_unit(run_id, phase)
        owned.update({unit, unit + ".service", unit + ".timer"})
    return owned


def injection_plan(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    """The scheduled XID writes ``arm-holder`` was asked for, validated.

    Empty when no ``--inject-script`` was given (the holder alone). Each entry
    is stored in the run state; ``pre-authorize`` turns it into one
    ``systemd-run`` timer whose ``after_seconds`` is the *earliest* moment the
    write may happen, counted from the holder's start.
    """

    script = str(getattr(arguments, "inject_script", "") or "")
    if not script:
        return []
    if SAFE_INJECT_SCRIPT.fullmatch(script) is None:
        raise ProbeError("inject script is not an installed host probe script")
    bdf = str(getattr(arguments, "pci_bdf", "") or "")
    if SAFE_BDF.fullmatch(bdf) is None:
        raise ProbeError("unsafe PCI BDF")
    plan = []
    for phase, subcommand in INJECTION_PHASES.items():
        marker = getattr(arguments, f"{phase}_marker", "") or ""
        drill_id = getattr(arguments, f"{phase}_drill_id", "") or ""
        delay = int(getattr(arguments, f"{phase}_after_seconds", 0) or 0)
        if not marker and not drill_id:
            continue
        safe_id(marker, f"{phase} marker")
        safe_id(drill_id, f"{phase} drill ID")
        if delay < MIN_INJECTION_DELAY_SECONDS:
            raise ProbeError(f"{phase} delay is below {MIN_INJECTION_DELAY_SECONDS}s")
        plan.append(
            {
                "phase": phase,
                "subcommand": subcommand,
                "script": script,
                "marker": marker,
                "drill_id": drill_id,
                "pci_bdf": bdf,
                "after_seconds": delay,
                "maintenance_window_end": getattr(
                    arguments, "maintenance_window_end", ""
                ),
            }
        )
    if (
        plan
        and max(item["after_seconds"] for item in plan) >= arguments.max_hold_seconds
    ):
        raise ProbeError(
            "an injection is scheduled after the holder's bounded lifetime"
        )
    if plan:
        try:
            deadline = datetime.fromisoformat(
                str(plan[0]["maintenance_window_end"]).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ProbeError(
                "scheduled injections require a maintenance deadline"
            ) from exc
        if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline:
            raise ProbeError("maintenance window ended before scheduling injections")
    return plan


def injection_command(
    run_id: str,
    item: dict[str, Any],
    *,
    runtime_max_seconds: int = FIRE_RUNTIME_SLACK_SECONDS,
) -> list[str]:
    """The transient timer that evaluates and, when the barrier holds, fires.

    The timer elapses at once; ``fire-injection`` then waits for the earliest
    time and the barrier condition itself, so the write never depends on the
    exec channel that quiesce takes down.
    """

    unit = injection_unit(run_id, str(item["phase"]))
    return [
        "systemd-run",
        "--unit",
        unit,
        "--on-active=1s",
        "--timer-property=AccuracySec=1s",
        f"--property=RuntimeMaxSec={int(runtime_max_seconds)}",
        "/opt/gpu-fault/current/venv/bin/python",
        str(Path(__file__).resolve()),
        "fire-injection",
        "--run-id",
        run_id,
        "--phase",
        str(item["phase"]),
    ]


def unit_name(value: str, run_id: str) -> str:
    if value not in _owned_units(run_id) | {AGENT_UNIT}:
        raise ProbeError("unit is not in the DESTR-016 probe allow-list")
    return value


def checked_command(command: list[str], run_id: str) -> list[str]:
    """Refuse any command outside the systemctl verb/unit allow-list.

    The Node Agent unit is read-only here: this case needs the Agent to take
    the reboot command and to re-register a new boot ID afterwards, so a verb
    that could stop or disable it has no legitimate caller in this probe.
    """

    if len(command) < 3 or command[0] != "systemctl":
        raise ProbeError("only systemctl commands with a unit are permitted")
    verb, unit = command[1], unit_name(command[2], run_id)
    if verb not in OWNED_UNIT_SYSTEMCTL_VERBS:
        raise ProbeError(f"systemctl verb is not permitted: {verb}")
    if unit == AGENT_UNIT and verb not in READ_ONLY_SYSTEMCTL_VERBS:
        raise ProbeError("the Node Agent unit may only be read by this probe")
    for token in command[3:]:
        if not token.startswith("--property="):
            raise ProbeError(f"systemctl argument is not permitted: {token}")
    return command


def checked_max_hold(value: int) -> int:
    if not MIN_HOLD_SECONDS <= int(value) <= MAX_HOLD_SECONDS:
        raise ProbeError(
            f"max hold seconds is outside {MIN_HOLD_SECONDS}..{MAX_HOLD_SECONDS}"
        )
    return int(value)


def state_path(run_id: str, *, state_dir: Path = STATE_DIR) -> Path:
    return state_dir / f"destr016-{safe_id(run_id, 'run ID')}.json"


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
    """Serialize the read-modify-write cycles of the units sharing one state file.

    The arm watcher, both injection timers and the runner's exec'd commands
    all edit the same file; a lost update here could drop a fire record and let
    a write repeat.
    """

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
        placeholders = ", ".join("?" for _ in LEDGER_OPERATIONS)
        rows = connection.execute(
            "SELECT command_id, completed_at, attempt, state, operation, started_at, "
            "workflow_request_id, incident_id, agent_generation, payload "
            f"FROM results WHERE operation IN ({placeholders}) "
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
        if row.get("operation") != operation or row.get("state") != "SUCCEEDED":
            continue
        if row.get("command_id") in baseline_command_ids:
            continue
        completed_at = str(row.get("completed_at") or "")
        if not completed_at or completed_at < armed_at:
            continue
        return row
    return None


def arm_race_lost(
    rows: list[dict[str, Any]],
    *,
    baseline_command_ids: set[str],
    armed_at: str,
    hold_started_at: str,
) -> bool:
    """True when the verify step already succeeded before the holder existed.

    ``VERIFY_NO_GPU_CLIENTS`` is the step that must park WAITING. If it
    succeeded on this drill before the holder started, the reset commits and
    there is no unrestored quiesce left for XID 79 to inherit, so the case
    cannot prove what it exists to prove. The runner turns this into a stop
    condition rather than injecting the escalation anyway.
    """

    matched = match_ledger_row(
        rows,
        operation=RACE_OPERATION,
        baseline_command_ids=baseline_command_ids,
        armed_at=armed_at,
    )
    if matched is None:
        return False
    if not hold_started_at:
        return True
    return str(matched.get("completed_at") or "") < hold_started_at


def _unit_state(unit: str, run_id: str) -> dict[str, str]:
    completed = run(
        checked_command(
            [
                "systemctl",
                "show",
                unit,
                "--property=LoadState",
                "--property=UnitFileState",
                "--property=ActiveState",
                "--property=SubState",
                "--property=MainPID",
            ],
            run_id,
        ),
        check=False,
    )
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def holder_active(run_id: str) -> bool:
    """Whether the GPU device holder unit is alive right now."""

    return _unit_state(holder_unit(run_id) + ".service", run_id).get("ActiveState") == (
        "active"
    )


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


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
    for item in state.get("injections") or []:
        deadline_at = datetime.fromisoformat(
            str(item.get("maintenance_window_end") or "").replace("Z", "+00:00")
        )
        if deadline_at.tzinfo is None or datetime.now(timezone.utc) >= deadline_at:
            raise ProbeError("maintenance window ended before starting the holder")
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
            f"destr016-{run_id[:8]}",
            device,
            str(max_hold),
        ]
    )
    lost = arm_race_lost(
        ledger_rows(),
        baseline_command_ids=set(state.get("verify_baseline_ids") or []),
        armed_at=armed_at,
        hold_started_at=started_at,
    )
    locked_update(
        path,
        {
            "matched_row": matched,
            "hold_started_at": started_at,
            "holder_unit": unit + ".service",
            "arm_race_lost": lost,
        },
    )
    if lost:
        _clear_unit(unit + ".service", run_id)
    # The injection timers were placed by ``pre-authorize`` and decide for
    # themselves, from the ledger, whether and when the barrier holds.


def arm_holder(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    safe_id(arguments.drill_id, "drill ID")
    device = safe_device(arguments.device)
    max_hold = checked_max_hold(arguments.max_hold_seconds)
    if arguments.after_ledger_op not in ARM_LEDGER_OPERATIONS:
        raise ProbeError("after-ledger-op is not permitted for arming")
    if Path(arguments.probe_script).name != Path(__file__).name:
        raise ProbeError("probe script identity mismatch")
    injections = injection_plan(arguments)
    rows = ledger_rows()
    baseline_ids = [
        row["command_id"]
        for row in rows
        if row["operation"] == arguments.after_ledger_op
    ]
    verify_baseline_ids = [
        row["command_id"] for row in rows if row["operation"] == RACE_OPERATION
    ]
    path = state_path(run_id)
    write_state(
        path,
        {
            "run_id": run_id,
            "drill_id": arguments.drill_id,
            "device": device,
            "max_hold_seconds": max_hold,
            "after_ledger_op": arguments.after_ledger_op,
            "baseline_command_ids": baseline_ids,
            "verify_baseline_ids": verify_baseline_ids,
            "armed_at": datetime.now(timezone.utc).isoformat(),
            "injections": injections,
            "boot_id": _boot_id(),
            "injection_script_sha256": (
                hashlib.sha256(Path(injections[0]["script"]).read_bytes()).hexdigest()
                if injections
                else None
            ),
        },
    )
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
            "verify_baseline_ids": verify_baseline_ids,
            "injections": injections,
        }
    )


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
    if state.get("arm_race_lost"):
        raise ProbeError("the holder lost the arming race")
    if state.get("pre_authorization"):
        raise ProbeError("the holder is already pre-authorized")
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
        or not delays
        or any(type(value) is not int or value < 0 for value in delays.values())
    ):
        raise ProbeError("pre-authorization not-before delays are invalid")
    return proof


def pre_authorize(arguments: argparse.Namespace) -> None:
    """Record the conditional authorization and place the injection timers.

    Exec'd by the runner right after ``arm-holder`` and *before* the fault is
    injected: the quiesce that follows the injection takes kubelet down, so
    this is the last moment the runner can reach the node until the restore.
    """

    run_id = safe_id(arguments.run_id, "run ID")
    path = state_path(run_id)
    proof = json.loads(arguments.authorization)
    with locked(path):
        state = read_state(path)
        now = datetime.now(timezone.utc)
        check_pre_authorization(
            state, proof, run_id=run_id, boot_id=_boot_id(), now=now
        )
        planned = {str(item.get("phase")) for item in state.get("injections") or []}
        if not planned or set(proof["not_before_seconds"]) != planned:
            raise ProbeError(
                "pre-authorization phases do not match the scheduled injections"
            )
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
                "injections_fired": {},
                "injection_refusals": {},
            }
        )
        write_state(path, state)
        expires = _parse_time(proof["expires_at"]) or now
        runtime = int((expires - now).total_seconds()) + FIRE_RUNTIME_SLACK_SECONDS
        for item in state["injections"]:
            unit = injection_unit(run_id, str(item["phase"]))
            _clear_unit(unit + ".timer", run_id)
            _clear_unit(unit + ".service", run_id)
            run(injection_command(run_id, item, runtime_max_seconds=runtime))
    emit(
        {
            "run_id": run_id,
            "pre_authorized_at": now.isoformat(),
            "expires_at": proof["expires_at"],
            "not_before_seconds": proof["not_before_seconds"],
            "ledger_baseline_workflow_ids": baseline,
            "timers": {
                phase: _unit_state(injection_unit(run_id, phase) + ".timer", run_id)
                for phase in sorted(planned)
            },
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
    state: dict[str, Any],
    *,
    phase: str,
    now: datetime,
    boot_id: str,
    holder_active: bool,
) -> dict[str, Any] | None:
    """The non-ledger half of the condition: binding, deadlines, the holder."""

    proof = state.get("pre_authorization")
    if not isinstance(proof, dict) or proof.get("kind") != CONDITIONAL_KIND:
        return _verdict("no conditional pre-authorization is recorded", final=True)
    if state.get("disarmed_at"):
        return _verdict("the holder was disarmed", final=True)
    if state.get("arm_race_lost"):
        return _verdict(
            "the client verification succeeded before the holder started", final=True
        )
    if (
        not boot_id
        or boot_id != proof.get("boot_id")
        or boot_id != state.get("boot_id")
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
    delay = (proof.get("not_before_seconds") or {}).get(phase)
    if type(delay) is not int:
        return _verdict(f"the pre-authorization names no delay for {phase}", final=True)
    hold_started = _parse_time(state.get("hold_started_at"))
    if hold_started is None:
        return _verdict("the holder has not started")
    if not holder_active:
        return _verdict("the holder is no longer active", final=True)
    if now < hold_started + timedelta(seconds=delay):
        return _verdict(f"{phase} is not due until {delay}s after the holder started")
    fired = state.get("injections_fired") or {}
    for earlier in INJECTION_PHASES:
        if earlier == phase:
            break
        if earlier in proof["not_before_seconds"] and not (
            fired.get(earlier) or {}
        ).get("fired_at"):
            return _verdict(f"{earlier} has not fired yet")
    return None


def barrier_condition(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    phase: str,
    now: datetime,
    boot_id: str,
    holder_active: bool,
) -> dict[str, Any]:
    """Whether ``phase`` may fire now, judged from this node alone.

    ``holds`` means write; ``final`` means stop polling because the reason can
    never clear (the caller records it and disarms). Anything else is a wait.
    The barrier is proven by the ledger: this run's workflow -- the one new
    workflow written since the pre-authorization -- has a succeeded quiesce,
    an executed client verification that refused on active clients (the
    holder's doing), and nothing beyond it.
    """

    gate = _holder_gate(
        state, phase=phase, now=now, boot_id=boot_id, holder_active=holder_active
    )
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


def refuse_injection(
    run_id: str, path: Path, state: dict[str, Any], phase: str, reason: str
) -> None:
    """Record why ``phase`` will never fire, then disarm everything this run owns.

    A barrier that cannot be reached leaves nothing for the case to prove;
    holding the device any longer would only keep the node parked.
    """

    now = datetime.now(timezone.utc).isoformat()
    refusals = dict(state.get("injection_refusals") or {})
    refusals[phase] = {"refused_at": now, "reason": reason}
    state["injection_refusals"] = refusals
    state["disarmed_at"] = state.get("disarmed_at") or now
    state["disarm_reason"] = state.get("disarm_reason") or f"{phase}: {reason}"
    write_state(path, state)
    _clear_unit(arm_unit(run_id) + ".service", run_id)
    _clear_unit(holder_unit(run_id) + ".service", run_id)
    for other in INJECTION_PHASES:
        if other != phase:
            _clear_unit(injection_unit(run_id, other) + ".timer", run_id)
            _clear_unit(injection_unit(run_id, other) + ".service", run_id)


def _fire_record(path: Path, run_id: str, phase: str) -> dict[str, Any]:
    """Wait for the barrier condition; record the fire intent under the lock."""

    while True:
        with locked(path):
            state = read_state(path)
            if state.get("run_id") != run_id:
                raise ProbeError("injection was not authorized for this run")
            fired = (state.get("injections_fired") or {}).get(phase) or {}
            if fired.get("fire_requested_at"):
                raise ProbeError("injection was already consumed")
            items = [
                item
                for item in state.get("injections") or []
                if item.get("phase") == phase
            ]
            if len(items) != 1:
                raise ProbeError("injection phase is missing")
            verdict = barrier_condition(
                state,
                ledger_rows(),
                phase=phase,
                now=datetime.now(timezone.utc),
                boot_id=_boot_id(),
                holder_active=holder_active(run_id),
            )
            if verdict["holds"]:
                item = items[0]
                if hashlib.sha256(
                    Path(item["script"]).read_bytes()
                ).hexdigest() != state.get("injection_script_sha256"):
                    raise ProbeError("injection probe script changed")
                record = {
                    "fire_requested_at": verdict["condition"]["observed_at"],
                    "condition": verdict["condition"],
                    "item": item,
                }
                fired_all = dict(state.get("injections_fired") or {})
                fired_all[phase] = {
                    key: value for key, value in record.items() if key != "item"
                }
                state["injections_fired"] = fired_all
                write_state(path, state)
                return record
            if verdict["final"]:
                refuse_injection(run_id, path, state, phase, verdict["reason"])
                raise ProbeError(f"{phase} injection refused: {verdict['reason']}")
        time.sleep(POLL_SECONDS)


def fire_injection(arguments: argparse.Namespace) -> None:
    """Run by the transient timer: fire ``phase`` once the barrier holds.

    The fire intent is durable before the write, so a lost acknowledgement can
    never make the write repeat; the write itself goes through the shared
    destructive probe with the maintenance deadline the run was given.
    """

    run_id = safe_id(arguments.run_id, "run ID")
    phase = str(arguments.phase)
    if phase not in INJECTION_PHASES:
        raise ProbeError("unknown injection phase")
    path = state_path(run_id)
    record = _fire_record(path, run_id, phase)
    item = record["item"]
    run(
        [
            sys.executable,
            item["script"],
            item["subcommand"],
            "--marker",
            item["marker"],
            "--drill-id",
            item["drill_id"],
            "--pci-bdf",
            item["pci_bdf"],
            "--maintenance-window-end",
            item["maintenance_window_end"],
        ]
    )
    fired_at = datetime.now(timezone.utc).isoformat()
    with locked(path):
        state = read_state(path)
        fired_all = dict(state.get("injections_fired") or {})
        fired_all[phase] = {**(fired_all.get(phase) or {}), "fired_at": fired_at}
        state["injections_fired"] = fired_all
        write_state(path, state)
    emit(
        {
            "run_id": run_id,
            "phase": phase,
            "fire_requested_at": record["fire_requested_at"],
            "fired_at": fired_at,
            "condition": record["condition"],
        }
    )


def disarm_holder(arguments: argparse.Namespace) -> None:
    """Stop the arm watcher, the holder and the timers. Idempotent: a holder the
    reboot already took with it, or one that was never armed, is not an error."""

    run_id = safe_id(arguments.run_id, "run ID")
    path = state_path(run_id)
    if path.is_file():
        now = datetime.now(timezone.utc).isoformat()
        with locked(path):
            state = read_state(path)
            state["disarmed_at"] = state.get("disarmed_at") or now
            state["disarm_reason"] = state.get("disarm_reason") or "disarm-holder"
            write_state(path, state)
    _clear_unit(arm_unit(run_id) + ".service", run_id)
    _clear_unit(holder_unit(run_id) + ".service", run_id)
    for phase in INJECTION_PHASES:
        _clear_unit(injection_unit(run_id, phase) + ".timer", run_id)
        _clear_unit(injection_unit(run_id, phase) + ".service", run_id)
    emit(
        {
            "run_id": run_id,
            "arm_unit_state": _unit_state(arm_unit(run_id) + ".service", run_id),
            "holder_unit_state": _unit_state(holder_unit(run_id) + ".service", run_id),
        }
    )


def holder_status(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    state = read_state(state_path(run_id))
    unit_state = _unit_state(holder_unit(run_id) + ".service", run_id)
    proof = state.get("pre_authorization") or {}
    emit(
        {
            "run_id": run_id,
            "holder_unit": holder_unit(run_id) + ".service",
            "unit_state": unit_state,
            "pid": unit_state.get("MainPID"),
            "matched_row": state.get("matched_row"),
            "hold_started_at": state.get("hold_started_at"),
            "arm_race_lost": state.get("arm_race_lost"),
            "holder_error": state.get("holder_error"),
            "pre_authorized_at": state.get("pre_authorized_at"),
            "pre_authorization_expires_at": proof.get("expires_at"),
            "not_before_seconds": proof.get("not_before_seconds"),
            "injections_fired": state.get("injections_fired"),
            "injection_refusals": state.get("injection_refusals"),
            "disarmed_at": state.get("disarmed_at"),
            "disarm_reason": state.get("disarm_reason"),
            "boot_id": _boot_id(),
        }
    )


def snapshot(arguments: argparse.Namespace) -> None:
    run_id = arguments.run_id
    payload: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "boot_id": _boot_id(),
        "ledger": ledger_rows(),
    }
    if run_id:
        safe_id(run_id, "run ID")
        payload["agent_unit"] = _unit_state(AGENT_UNIT, run_id)
        payload["holder_unit"] = _unit_state(holder_unit(run_id) + ".service", run_id)
        payload["state"] = read_state(state_path(run_id))
    emit(payload)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="DESTR-016 on-node GPU client holder around the quiesce boundary."
    )
    commands = value.add_subparsers(dest="command", required=True)

    arm = commands.add_parser("arm-holder")
    arm.add_argument("--device", required=True)
    arm.add_argument("--drill-id", required=True)
    arm.add_argument(
        "--after-ledger-op",
        required=True,
        choices=sorted(ARM_LEDGER_OPERATIONS),
    )
    arm.add_argument("--max-hold-seconds", type=int, default=900)
    arm.add_argument("--run-id", required=True)
    arm.add_argument("--probe-script", required=True)
    # Optional: schedule the shared destructive probe's absorb (XID 46) and
    # escalation (XID 79) writes no earlier than N seconds after the holder
    # starts; ``pre-authorize`` places the timers, the barrier decides.
    arm.add_argument("--inject-script", default="")
    arm.add_argument("--pci-bdf", default="")
    arm.add_argument("--maintenance-window-end", default="")
    for phase in INJECTION_PHASES:
        arm.add_argument(f"--{phase}-marker", default="")
        arm.add_argument(f"--{phase}-drill-id", default="")
        arm.add_argument(f"--{phase}-after-seconds", type=int, default=0)
    arm.set_defaults(handler=arm_holder)

    watch = commands.add_parser("watch-ledger")
    watch.add_argument("--run-id", required=True)
    watch.set_defaults(handler=watch_ledger)

    authorize = commands.add_parser("pre-authorize")
    authorize.add_argument("--run-id", required=True)
    authorize.add_argument("--authorization", required=True)
    authorize.set_defaults(handler=pre_authorize)

    fire = commands.add_parser("fire-injection")
    fire.add_argument("--run-id", required=True)
    fire.add_argument("--phase", choices=sorted(INJECTION_PHASES), required=True)
    fire.set_defaults(handler=fire_injection)

    disarm = commands.add_parser("disarm-holder")
    disarm.add_argument("--run-id", required=True)
    disarm.set_defaults(handler=disarm_holder)

    status = commands.add_parser("holder-status")
    status.add_argument("--run-id", required=True)
    status.set_defaults(handler=holder_status)

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
