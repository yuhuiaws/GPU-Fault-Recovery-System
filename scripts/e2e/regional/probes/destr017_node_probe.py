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

The reboot is deliberately *not* a control-plane action: no workflow step, no
provider call, no Node Agent command.  That is the fault this case injects.

Every shell command is on an allow-list, the only units this probe may touch
are the two it names after its own run id, and the only reboot form it can
issue is a plain ``systemctl reboot``.  Nothing here executes a GPU reset,
stops a service, or touches the Node Agent.
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
# The reboot must be far enough out that the arming exec returns first, and
# near enough that it lands inside the pinned quiesce maintenance window.
MIN_REBOOT_DELAY_SECONDS = 30
MAX_REBOOT_DELAY_SECONDS = 600
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")

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


def ledger_rows(ledger: Path = LEDGER) -> list[dict[str, Any]]:
    if not ledger.is_file():
        return []
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT command_id, completed_at, attempt, state, operation, started_at "
            "FROM results WHERE operation IN "
            f"({', '.join('?' for _ in LEDGER_OPERATIONS)}) "
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
            f"destr017-{run_id[:8]}",
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
    armed_at = datetime.now(timezone.utc).isoformat()
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
        },
    )
    write_state(path, record_boot_observation(state, boot_id(), observed_at=armed_at))
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
        }
    )


def disarm_holder(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    _clear_unit(arm_unit(run_id) + ".service", run_id)
    _clear_unit(holder_unit(run_id) + ".service", run_id)
    path = state_path(run_id)
    if path.is_file():
        update_state(path, {"disarmed_at": datetime.now(timezone.utc).isoformat()})
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
        }
    )


def arm_reboot(arguments: argparse.Namespace) -> None:
    """Arm a bounded transient timer that runs ``systemctl reboot``.

    The marker file is written *before* the timer exists, so a reboot can never
    fire without a durable record of the boot id it replaced.
    """

    run_id = safe_id(arguments.run_id, "run ID")
    delay = checked_reboot_delay(arguments.delay_seconds)
    path = state_path(run_id)
    armed_at = datetime.now(timezone.utc)
    current = boot_id()
    unit = reboot_unit(run_id)
    state = update_state(
        path,
        {
            "run_id": run_id,
            "reboot_armed_at": armed_at.isoformat(),
            "reboot_delay_seconds": delay,
            "boot_id_before_reboot": current,
            "reboot_fire_at": (armed_at + timedelta(seconds=delay)).isoformat(),
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
    run(
        [
            "systemd-run",
            f"--unit={unit}",
            f"--on-active={delay}s",
            "--timer-property=AccuracySec=1s",
            "--property=Type=oneshot",
            *reboot_command(),
        ]
    )
    emit(
        {
            "run_id": run_id,
            "reboot_unit": unit + ".timer",
            "reboot_armed_at": armed_at.isoformat(),
            "reboot_fire_at": (armed_at + timedelta(seconds=delay)).isoformat(),
            "reboot_delay_seconds": delay,
            "boot_id_before_reboot": current,
            "units": _reboot_unit_state(run_id),
        }
    )


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
    emit(
        {
            "run_id": run_id,
            "observed_at": observed_at,
            "armed": bool(state.get("reboot_armed_at")),
            "reboot_armed_at": state.get("reboot_armed_at"),
            "reboot_fire_at": state.get("reboot_fire_at"),
            "reboot_delay_seconds": state.get("reboot_delay_seconds"),
            "reboot_cancelled_at": state.get("reboot_cancelled_at"),
            "boot_id_before_reboot": before or None,
            "boot_id_now": current,
            "fired": bool(before) and before != current,
            "observed_boot_ids": state.get("observed_boot_ids") or [],
            "boot_changes": state.get("boot_changes", 0),
            "units": _reboot_unit_state(run_id),
        }
    )


def cancel_reboot(arguments: argparse.Namespace) -> None:
    """Stop the reboot timer if it has not fired. Idempotent."""

    run_id = safe_id(arguments.run_id, "run ID")
    unit = reboot_unit(run_id)
    before = _reboot_unit_state(run_id)
    _clear_unit(unit + ".timer", run_id)
    _clear_unit(unit + ".service", run_id)
    path = state_path(run_id)
    current = boot_id()
    cancelled_at = datetime.now(timezone.utc).isoformat()
    state = read_state(path)
    if state:
        state = record_boot_observation(state, current, observed_at=cancelled_at)
        state["reboot_cancelled_at"] = cancelled_at
        write_state(path, state)
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

    reboot = commands.add_parser("arm-reboot")
    reboot.add_argument("--run-id", required=True)
    reboot.add_argument(
        "--delay-seconds",
        type=int,
        default=MIN_REBOOT_DELAY_SECONDS,
        help=(
            "seconds until the transient timer runs systemctl reboot; "
            f"{MIN_REBOOT_DELAY_SECONDS}..{MAX_REBOOT_DELAY_SECONDS}"
        ),
    )
    reboot.set_defaults(handler=arm_reboot)

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
