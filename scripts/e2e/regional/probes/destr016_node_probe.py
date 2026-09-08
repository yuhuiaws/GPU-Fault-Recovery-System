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

Every shell command is on an allow-list. The Node Agent unit may only be read,
never stopped, disabled or restarted: this case needs the Agent alive to take
the reboot command and to re-register after it. Nothing here resets a GPU,
reboots a node, or writes to /dev/kmsg -- the shared destructive probe owns the
XID writes. This probe only *schedules* two of them: ``QUIESCE_GPU_SERVICES``
stops kubelet, and with it the ``kubectl exec`` channel every probe answers
over, so the absorb (XID 46) and escalation (XID 79) writes that must land
inside the WAITING window are handed to ``systemd-run --on-active`` timers the
moment the holder starts, and the runner watches them arrive through the
control plane instead of exec'ing into a node that can no longer answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEDGER = Path("/var/lib/gpu-fault/node-actions.db")
STATE_DIR = Path("/var/lib/gpu-fault-acceptance")
AGENT_UNIT = "gpu-fault-node-agent.service"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SAFE_DEVICE = re.compile(r"^/dev/nvidia(?:[0-9]|1[0-5])$")
SAFE_BDF = re.compile(r"^[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}(?:\.[0-7])?$")
SAFE_INJECT_SCRIPT = re.compile(r"^/run/gpu-fault-host-probe-[0-9a-f]{6,32}\.py$")
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
    is stored in the run state and turned into one ``systemd-run --on-active``
    timer by ``watch-ledger`` right after the holder starts.
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
            }
        )
    if (
        plan
        and max(item["after_seconds"] for item in plan) >= arguments.max_hold_seconds
    ):
        raise ProbeError(
            "an injection is scheduled after the holder's bounded lifetime"
        )
    return plan


def injection_command(run_id: str, item: dict[str, Any]) -> list[str]:
    unit = injection_unit(run_id, str(item["phase"]))
    return [
        "systemd-run",
        "--unit",
        unit,
        f"--on-active={int(item['after_seconds'])}",
        "--timer-property=AccuracySec=1s",
        "--property=RuntimeMaxSec=120",
        "/opt/gpu-fault/current/venv/bin/python",
        str(item["script"]),
        str(item["subcommand"]),
        "--marker",
        str(item["marker"]),
        "--drill-id",
        str(item["drill_id"]),
        "--pci-bdf",
        str(item["pci_bdf"]),
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


def ledger_rows(ledger: Path = LEDGER) -> list[dict[str, Any]]:
    if not ledger.is_file():
        return []
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        placeholders = ", ".join("?" for _ in LEDGER_OPERATIONS)
        rows = connection.execute(
            "SELECT command_id, completed_at, attempt, state, operation, started_at "
            f"FROM results WHERE operation IN ({placeholders}) "
            "ORDER BY completed_at, command_id",
            LEDGER_OPERATIONS,
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "command_id": row[0],
            "completed_at": row[1],
            "attempt": row[2],
            "state": row[3],
            "operation": row[4],
            "started_at": row[5],
        }
        for row in rows
    ]


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


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def _clear_unit(unit_with_suffix: str, run_id: str) -> None:
    run(checked_command(["systemctl", "stop", unit_with_suffix], run_id), check=False)
    run(
        checked_command(["systemctl", "reset-failed", unit_with_suffix], run_id),
        check=False,
    )


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
        update_state(path, {"holder_error": "arm ledger row never appeared"})
        return
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
    scheduled = []
    for item in state.get("injections") or []:
        injection = injection_unit(run_id, str(item["phase"]))
        _clear_unit(injection + ".timer", run_id)
        _clear_unit(injection + ".service", run_id)
        run(injection_command(run_id, item))
        scheduled.append(
            {
                **item,
                "unit": injection + ".timer",
                "scheduled_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    update_state(
        path,
        {
            "matched_row": matched,
            "hold_started_at": started_at,
            "holder_unit": unit + ".service",
            "scheduled_injections": scheduled,
        },
    )
    update_state(
        path,
        {
            "arm_race_lost": arm_race_lost(
                ledger_rows(),
                baseline_command_ids=set(state.get("verify_baseline_ids") or []),
                armed_at=armed_at,
                hold_started_at=started_at,
            )
        },
    )


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


def disarm_holder(arguments: argparse.Namespace) -> None:
    """Stop the arm watcher and the holder. Idempotent: a holder the reboot
    already took with it, or one that was never armed, is not an error."""

    run_id = safe_id(arguments.run_id, "run ID")
    _clear_unit(arm_unit(run_id) + ".service", run_id)
    _clear_unit(holder_unit(run_id) + ".service", run_id)
    for phase in INJECTION_PHASES:
        _clear_unit(injection_unit(run_id, phase) + ".timer", run_id)
        _clear_unit(injection_unit(run_id, phase) + ".service", run_id)
    path = state_path(run_id)
    if path.is_file():
        update_state(path, {"disarmed_at": datetime.now(timezone.utc).isoformat()})
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
            "scheduled_injections": state.get("scheduled_injections"),
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
    # escalation (XID 79) writes N seconds after the holder starts.
    arm.add_argument("--inject-script", default="")
    arm.add_argument("--pci-bdf", default="")
    for phase in INJECTION_PHASES:
        arm.add_argument(f"--{phase}-marker", default="")
        arm.add_argument(f"--{phase}-drill-id", default="")
        arm.add_argument(f"--{phase}-after-seconds", type=int, default=0)
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
