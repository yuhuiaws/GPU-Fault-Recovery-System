#!/usr/bin/env python3
"""On-node probe for GF-REGIONAL-DESTR-018.

One node-local job the control plane cannot do for the runner: hold a
``/dev/nvidiaN`` open for the whole drill so every
``VERIFY_NO_GPU_CLIENTS`` attempt of the RESET_GPU workflow raises
``GPU device clients are still active`` and the step stays WAITING until the
workflow's hard lifetime passes. The holder is a ``systemd-run`` transient unit,
not a Pod and not a child of the ``kubectl exec`` channel, for two reasons: a
Pod cannot start once ``QUIESCE_GPU_SERVICES`` has stopped the GPU services, and
the exec channel is not a supervisor (see memory
``host-probe-cannot-stop-its-own-transport``).

Why the holder is armed *before* injection by default, rather than after a
ledger row:

* ``node_agent/operations/clients.py::_persistent_device_clients`` only counts a
  holder that survives every sample, so the holder has to already be there when
  the step runs -- and it must be there for the *first* attempt.
* If a first ``VERIFY_NO_GPU_CLIENTS`` succeeds because the holder was not up
  yet, the workflow advances to ``RESET_GPU``, whose own re-verification of
  clients raises the same error -- but ``RESET_GPU`` has no WAITING branch
  (``adapters/node_action/step_execution.py`` gates that on
  ``VERIFY_NO_GPU_CLIENTS``) and ``NodeActionExecutor._retryable_action_error``
  calls a ``RuntimeError`` non-retryable. The step FAILs and the workflow
  escalates to ``RESTART_NODE``, i.e. a real reboot this case never authorized.
  Arming ahead of the injection removes that race instead of narrowing it.
* ``QUIESCE_GPU_SERVICES`` cannot sweep the holder away: the sweep in
  ``node_agent/quiesce.py::_sweep_device_holders`` signals only a holder whose
  process name is in ``DEFAULT_DEVICE_SWEEP_PROCESSES``
  (``nvidia-persiste``/``nv-hostengine``) or whose cgroup is the affected
  workload's. This holder is neither, so it is recorded as ``skipped`` with
  ``outside_workload_cgroup_and_process_whitelist`` -- which is the documented
  fail-closed behaviour, and is itself evidence.

``--after-ledger-op QUIESCE_GPU_SERVICES`` keeps the DESTR-014 poll-and-arm
shape for an operator who wants the containment steps to run against a clean
node first. ``VERIFY_NO_GPU_CLIENTS`` is deliberately *not* an accepted arm
trigger, for the reboot reason above.

Every shell command is on an allow-list; unknown units and devices are refused.
Nothing here executes a GPU reset, a reboot, or touches any service unit other
than the two transient units this probe itself created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
PROC = Path("/proc")
VENV_PYTHON = "/opt/gpu-fault/current/venv/bin/python"
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
# The only ledger operation this drill may wait on to arm the holder. Arming
# after VERIFY_NO_GPU_CLIENTS is what DESTR-014 does and is forbidden here: it
# would break RESET_GPU instead of holding VERIFY, and a failed RESET_GPU
# escalates to a node reboot.
ARM_LEDGER_OPERATIONS = ("QUIESCE_GPU_SERVICES",)
ALLOWED_SYSTEMCTL_VERBS = ("stop", "reset-failed", "show", "is-active")
MIN_HOLD_SECONDS = 60
MAX_HOLD_SECONDS = 3600

# The holder body, kept a host process (Pods cannot start once GPU services are
# quiesced). It names itself, opens the device read-only, and exits on SIGTERM
# or after its bounded lifetime. Read-only and no CUDA context, so it is a
# device client and never a compute client -- exactly the holder the fail-closed
# verification refuses to reset around.
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
    return f"gpu-fault-destr018-holder-{_run_digest(run_id)}"


def arm_unit(run_id: str) -> str:
    return f"gpu-fault-destr018-arm-{_run_digest(run_id)}"


def owned_units(run_id: str) -> set[str]:
    return {
        holder_unit(run_id),
        holder_unit(run_id) + ".service",
        arm_unit(run_id),
        arm_unit(run_id) + ".service",
    }


def unit_name(value: str, run_id: str) -> str:
    """Refuse any unit this probe did not itself create.

    Unlike DESTR-014 this case never touches ``gpu-fault-node-agent.service``:
    the Agent must keep answering for the whole drill, because the WAITING loop
    the case measures is a chain of real node round trips.
    """

    if value not in owned_units(run_id):
        raise ProbeError("unit is not in the DESTR-018 probe allow-list")
    return value


def checked_command(command: list[str], run_id: str) -> list[str]:
    """Refuse any command outside the systemctl verb/unit allow-list."""

    if len(command) < 3 or command[0] != "systemctl":
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


def arm_mode(after_ledger_op: str | None) -> str:
    """``immediate`` when no trigger is named, ``ledger`` when one is.

    An unrecognised trigger is refused rather than downgraded to immediate: a
    typo that silently changed *when* the holder appears would change which
    workflow step the case actually breaks.
    """

    if not after_ledger_op:
        return "immediate"
    if after_ledger_op not in ARM_LEDGER_OPERATIONS:
        raise ProbeError("after-ledger-op is not permitted for arming")
    return "ledger"


def state_path(run_id: str, *, state_dir: Path = STATE_DIR) -> Path:
    return state_dir / f"destr018-{safe_id(run_id, 'run ID')}.json"


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
    placeholders = ", ".join(f"'{name}'" for name in LEDGER_OPERATIONS)
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT command_id, completed_at, attempt, state, operation, started_at "
            "FROM results "
            f"WHERE operation IN ({placeholders}) "
            "ORDER BY completed_at, command_id"
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


def device_clients(device: str, *, proc_root: Path = PROC) -> list[dict[str, str]]:
    """Processes holding ``device`` open, read straight out of ``/proc``.

    The same walk ``ClientOperationsMixin._device_clients`` does, so the probe
    reports the holder set the Node Agent's fail-closed verification will see
    rather than a proxy for it.
    """

    safe_device(device)
    result: list[dict[str, str]] = []
    try:
        entries = sorted(proc_root.iterdir())
    except OSError:
        return result
    for process in entries:
        if not process.name.isdigit():
            continue
        try:
            descriptors = list((process / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target != device:
                continue
            try:
                name = (process / "comm").read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                name = "unknown"
            result.append({"pid": process.name, "process_name": name, "device": device})
            break
    return sorted(result, key=lambda item: int(item["pid"]))


def _unit_state(unit_with_suffix: str) -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            unit_with_suffix,
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


def _clear_unit(unit_with_suffix: str, run_id: str) -> None:
    run(checked_command(["systemctl", "stop", unit_with_suffix], run_id), check=False)
    run(
        checked_command(["systemctl", "reset-failed", unit_with_suffix], run_id),
        check=False,
    )


def _start_holder(run_id: str, device: str, max_hold: int) -> str:
    unit = holder_unit(run_id)
    _clear_unit(unit + ".service", run_id)
    run(
        [
            "systemd-run",
            "--unit",
            unit,
            f"--property=RuntimeMaxSec={max_hold}",
            "--property=KillMode=control-group",
            VENV_PYTHON,
            "-c",
            HELPER,
            f"destr018-{_run_digest(run_id)[:6]}",
            device,
            str(max_hold),
        ]
    )
    return unit + ".service"


def watch_ledger(arguments: argparse.Namespace) -> None:
    """Poll the ledger for the arm row, then start the holder. Runs detached."""

    run_id = safe_id(arguments.run_id, "run ID")
    path = state_path(run_id)
    state = read_state(path)
    operation = str(state.get("after_ledger_op") or "")
    if arm_mode(operation) != "ledger":
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
    started_at = datetime.now(timezone.utc).isoformat()
    unit = _start_holder(run_id, device, max_hold)
    update_state(
        path,
        {
            "matched_row": matched,
            "hold_started_at": started_at,
            "holder_unit": unit,
        },
    )


def checked_probe_script(value: str) -> str:
    """The path the watcher unit will run must be *this* file, on the host.

    Compared as a resolved full path, not a basename: a Pod-side
    ``/host/run/...`` path has the right basename while the systemd unit,
    which runs on the host, cannot find it.
    """

    if Path(value).resolve() != Path(__file__).resolve():
        raise ProbeError("probe script identity mismatch")
    return value


def arm_holder(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    safe_id(arguments.drill_id, "drill ID")
    device = safe_device(arguments.device)
    max_hold = checked_max_hold(arguments.max_hold_seconds)
    mode = arm_mode(arguments.after_ledger_op)
    checked_probe_script(arguments.probe_script)
    baseline_ids = [
        row["command_id"]
        for row in ledger_rows()
        if row["operation"] == arguments.after_ledger_op
    ]
    path = state_path(run_id)
    armed_at = datetime.now(timezone.utc).isoformat()
    write_state(
        path,
        {
            "run_id": run_id,
            "drill_id": arguments.drill_id,
            "device": device,
            "max_hold_seconds": max_hold,
            "arm_mode": mode,
            "after_ledger_op": arguments.after_ledger_op or "",
            "baseline_command_ids": baseline_ids,
            "armed_at": armed_at,
        },
    )
    payload: dict[str, Any] = {
        "run_id": run_id,
        "arm_mode": mode,
        "device": device,
        "after_ledger_op": arguments.after_ledger_op or "",
        "baseline_command_ids": baseline_ids,
        "armed_at": armed_at,
    }
    if mode == "immediate":
        unit = _start_holder(run_id, device, max_hold)
        update_state(path, {"hold_started_at": armed_at, "holder_unit": unit})
        payload["holder_unit"] = unit
        payload["unit_state"] = _unit_state(unit)
        payload["device_clients"] = device_clients(device)
    else:
        unit = arm_unit(run_id)
        _clear_unit(unit + ".service", run_id)
        run(
            [
                "systemd-run",
                "--unit",
                unit,
                f"--property=RuntimeMaxSec={max_hold + 60}",
                VENV_PYTHON,
                str(Path(arguments.probe_script)),
                "watch-ledger",
                "--run-id",
                run_id,
            ]
        )
        payload["arm_unit"] = unit + ".service"
    emit(payload)


def disarm_holder(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    _clear_unit(arm_unit(run_id) + ".service", run_id)
    _clear_unit(holder_unit(run_id) + ".service", run_id)
    path = state_path(run_id)
    state = read_state(path)
    state["disarmed_at"] = datetime.now(timezone.utc).isoformat()
    if path.is_file():
        write_state(path, state)
    device = str(state.get("device") or "")
    remaining = device_clients(device) if SAFE_DEVICE.fullmatch(device) else []
    payload = {
        "run_id": run_id,
        "arm_unit_state": _unit_state(arm_unit(run_id) + ".service"),
        "holder_unit_state": _unit_state(holder_unit(run_id) + ".service"),
        "device_clients": remaining,
    }
    emit(payload)
    if any(item["pid"] == str(state.get("holder_pid") or "") for item in remaining):
        raise ProbeError("the holder process still holds the device after disarm")


def holder_status(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    state = read_state(state_path(run_id))
    unit_state = _unit_state(holder_unit(run_id) + ".service")
    device = str(state.get("device") or "")
    emit(
        {
            "run_id": run_id,
            "holder_unit": holder_unit(run_id) + ".service",
            "unit_state": unit_state,
            "pid": unit_state.get("MainPID"),
            "arm_mode": state.get("arm_mode"),
            "matched_row": state.get("matched_row"),
            "hold_started_at": state.get("hold_started_at"),
            "holder_error": state.get("holder_error"),
            "device_clients": (
                device_clients(device) if SAFE_DEVICE.fullmatch(device) else []
            ),
        }
    )


def snapshot(arguments: argparse.Namespace) -> None:
    run_id = arguments.run_id
    payload: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "boot_id": _boot_id(),
    }
    if arguments.device:
        payload["device_clients"] = device_clients(safe_device(arguments.device))
    if run_id:
        safe_id(run_id, "run ID")
        payload["holder_unit"] = _unit_state(holder_unit(run_id) + ".service")
        payload["arm_unit"] = _unit_state(arm_unit(run_id) + ".service")
        payload["state"] = read_state(state_path(run_id))
    emit(payload)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "GF-REGIONAL-DESTR-018 on-node holder: keep one GPU device open so "
            "VERIFY_NO_GPU_CLIENTS stays WAITING until the workflow lifetime."
        )
    )
    commands = value.add_subparsers(dest="command", required=True)

    arm = commands.add_parser("arm-holder")
    arm.add_argument("--device", required=True)
    arm.add_argument("--drill-id", required=True)
    arm.add_argument(
        "--after-ledger-op",
        default="",
        choices=("", *ARM_LEDGER_OPERATIONS),
        help="arm only after this ledger operation succeeds; default arms now",
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

    show = commands.add_parser("snapshot")
    show.add_argument("--run-id", default="")
    show.add_argument("--device", default="")
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
