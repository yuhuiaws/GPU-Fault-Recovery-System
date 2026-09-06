#!/usr/bin/env python3
"""On-node probe for GF-REGIONAL-DESTR-014.

Two node-local jobs the control plane cannot do for the runner:

* On the fault node (node-b), open a long-lived GPU device holder *after* the
  Node Agent ledger shows this drill's ``VERIFY_NO_GPU_CLIENTS`` succeeded, so
  ``reset.py``'s re-verification of clients inside ``RESET_GPU`` throws
  ``clients are still active`` and the control-plane barrier retries WAITING
  until it FAILs -- a reset failure at the reset step, not before it. Both the
  poll-and-arm and the holder itself run as ``systemd-run`` transient units so
  they survive the ``kubectl exec`` channel (see memory
  ``host-probe-cannot-stop-its-own-transport``).
* On the sibling node (node-c), ``systemctl disable`` the Node Agent unit
  *without* ``--now`` before injection, recording the prior enabled/active
  state, so the real reboot returns a node whose Agent never re-registers a new
  boot id -- RESTART_NODE stays WAITING to its managed-recovery timeout.

Every shell command is on an allow-list; unknown units and devices are refused.
Nothing here executes a GPU reset, a reboot, or touches any service other than
the Node Agent unit.
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
LEDGER_OPERATIONS = {
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESET_ALL_GPUS_NVSWITCHES",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
}
# The only ledger operation this drill may wait on to arm the holder: it is
# the last check before the reset commit the holder must break.
ARM_LEDGER_OPERATIONS = {"VERIFY_NO_GPU_CLIENTS"}
ALLOWED_SYSTEMCTL_VERBS = {
    "disable",
    "enable",
    "start",
    "stop",
    "is-enabled",
    "is-active",
    "show",
    "reset-failed",
}
MIN_HOLD_SECONDS = 60
MAX_HOLD_SECONDS = 3600

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
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError(f"unsafe {label}")
    return value


def safe_device(value: str) -> str:
    if SAFE_DEVICE.fullmatch(value or "") is None:
        raise ProbeError("device is not an allow-listed /dev/nvidiaN node")
    return value


def _run_digest(run_id: str) -> str:
    return hashlib.sha256(safe_id(run_id, "run ID").encode()).hexdigest()[:16]


def holder_unit(run_id: str) -> str:
    return f"gpu-fault-destr014-holder-{_run_digest(run_id)}"


def arm_unit(run_id: str) -> str:
    return f"gpu-fault-destr014-arm-{_run_digest(run_id)}"


def _owned_units(run_id: str) -> set[str]:
    return {
        AGENT_UNIT,
        holder_unit(run_id),
        holder_unit(run_id) + ".service",
        arm_unit(run_id),
        arm_unit(run_id) + ".service",
    }


def unit_name(value: str, run_id: str) -> str:
    if value not in _owned_units(run_id):
        raise ProbeError("unit is not in the DESTR-014 probe allow-list")
    return value


def checked_command(command: list[str], run_id: str) -> list[str]:
    """Refuse any command outside the systemctl verb/unit allow-list."""

    if not command or command[0] != "systemctl":
        raise ProbeError("only systemctl commands are permitted")
    if command[1] not in ALLOWED_SYSTEMCTL_VERBS:
        raise ProbeError(f"systemctl verb is not permitted: {command[1]}")
    unit_name(command[2], run_id)
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
    return state_dir / f"destr014-{safe_id(run_id, 'run ID')}.json"


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
        rows = connection.execute(
            """
            SELECT command_id, completed_at, attempt, state, operation, started_at
            FROM results
            WHERE operation IN (
                'QUIESCE_GPU_SERVICES', 'VERIFY_NO_GPU_CLIENTS', 'RESET_GPU',
                'RESET_ALL_GPUS_NVSWITCHES', 'RESTORE_GPU_SERVICES', 'VALIDATE_GPU'
            )
            ORDER BY completed_at, command_id
            """
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


def agent_baseline_record(*, enabled_state: str, active_state: str) -> dict[str, str]:
    return {
        "agent_enabled_baseline": enabled_state,
        "agent_active_baseline": active_state,
    }


def restore_actions(baseline: dict[str, Any]) -> list[str]:
    """The systemctl verbs that put the Node Agent back to its baseline."""

    if "agent_enabled_baseline" not in baseline:
        raise ProbeError("agent baseline record is missing the enabled state")
    actions: list[str] = []
    if str(baseline.get("agent_enabled_baseline")).startswith("enabled"):
        actions.append("enable")
    if baseline.get("agent_active_baseline") == "active":
        actions.append("start")
    return actions


def _agent_unit_state() -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            AGENT_UNIT,
            "--property=UnitFileState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
        ],
        check=False,
    )
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def _holder_unit_state(run_id: str) -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            holder_unit(run_id) + ".service",
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
        ],
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


def _clear_unit(unit_with_suffix: str) -> None:
    run(["systemctl", "stop", unit_with_suffix], check=False)
    run(["systemctl", "reset-failed", unit_with_suffix], check=False)


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
    _clear_unit(unit + ".service")
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
            f"destr014-{run_id[:8]}",
            device,
            str(max_hold),
        ]
    )
    update_state(
        path,
        {
            "matched_row": matched,
            "hold_started_at": started_at,
            "holder_unit": unit + ".service",
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
    baseline_ids = [
        row["command_id"]
        for row in ledger_rows()
        if row["operation"] == arguments.after_ledger_op
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
            "armed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    unit = arm_unit(run_id)
    _clear_unit(unit + ".service")
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
        }
    )


def disarm_holder(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    _clear_unit(arm_unit(run_id) + ".service")
    _clear_unit(holder_unit(run_id) + ".service")
    path = state_path(run_id)
    state = read_state(path)
    state["disarmed_at"] = datetime.now(timezone.utc).isoformat()
    if path.is_file():
        write_state(path, state)
    emit(
        {
            "run_id": run_id,
            "arm_unit_state": _holder_unit_state(run_id),
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
            "hold_ended_at": state.get("hold_ended_at"),
            "holder_error": state.get("holder_error"),
        }
    )


def disable_agent_restart(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    before = _agent_unit_state()
    baseline = agent_baseline_record(
        enabled_state=before.get("UnitFileState", ""),
        active_state=before.get("ActiveState", ""),
    )
    path = state_path(run_id)
    update_state(path, {"agent_baseline": baseline})
    # Disable *without* --now: the running Agent keeps answering until the
    # reboot, and does not come back after it.
    run(checked_command(["systemctl", "disable", AGENT_UNIT], run_id))
    after = _agent_unit_state()
    emit(
        {
            "run_id": run_id,
            "agent_baseline": baseline,
            "agent_state_after_disable": after,
        }
    )


def restore_agent(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    state = read_state(state_path(run_id))
    baseline = state.get("agent_baseline")
    if not isinstance(baseline, dict):
        raise ProbeError("no recorded Node Agent baseline for this run")
    for action in restore_actions(baseline):
        run(checked_command(["systemctl", action, AGENT_UNIT], run_id), timeout=120)
    after = _agent_unit_state()
    if "start" in restore_actions(baseline) and after.get("ActiveState") != "active":
        raise ProbeError("Node Agent did not return active after restore")
    emit(
        {
            "run_id": run_id,
            "agent_baseline": baseline,
            "agent_state_after_restore": after,
        }
    )


def snapshot(arguments: argparse.Namespace) -> None:
    run_id = arguments.run_id
    payload: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "boot_id": _boot_id(),
        "agent_unit": _agent_unit_state(),
    }
    if run_id:
        safe_id(run_id, "run ID")
        payload["holder_unit"] = _holder_unit_state(run_id)
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
        required=True,
        choices=sorted(ARM_LEDGER_OPERATIONS),
    )
    arm.add_argument("--max-hold-seconds", type=int, default=900)
    arm.add_argument("--run-id", required=True)
    arm.add_argument("--probe-script", required=True)
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

    disable = commands.add_parser("disable-agent-restart")
    disable.add_argument("--run-id", required=True)
    disable.set_defaults(handler=disable_agent_restart)

    restore = commands.add_parser("restore-agent")
    restore.add_argument("--run-id", required=True)
    restore.set_defaults(handler=restore_agent)

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
