#!/usr/bin/env python3
"""Detached, read-only reset-interval witness for GF-REGIONAL-DESTR-015.

``QUIESCE_GPU_SERVICES`` stops kubelet on the node it quiesces, and with it
every ``kubectl exec`` channel into that node, until ``RESTORE_GPU_SERVICES``
brings kubelet back minutes later. DESTR-015 quiesces both of its nodes at
once, so a witness that streams its observations to the runner over an exec
session dies on both nodes in the middle of the reset it exists to observe.
This probe therefore runs the strace-based witness as a ``systemd-run``
transient unit on the host and keeps its observations in durable records under
``/var/lib/gpu-fault-acceptance/destr015/<run>/``. The runner talks to it only
while kubelet is up: one ``arm`` exec before the injection, one ``collect``
exec after the node is Ready again, and ``disarm``/``status`` for cleanup.

Requests arrive as one JSON line on stdin (the pinned bundle loader delivers
the program itself); the unit is started with ``witness --run-id`` and runs the
same program from the durable copy ``arm`` made. Every record carries the
host's monotonic and realtime clocks and its boot id, so the verdict can bind
the records to the arm..collect window on one boot and align them to the
runner's clock through the two bounded exchanges. The witness never signals,
stops or restarts the Node Agent; it attaches a read-only exec trace to it,
exactly as the streaming probe did, and stops itself at its deadline. Nothing
outside the run's own state directory and transient unit is ever touched, and
no record carries raw argv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.destr015_physical_evidence import (  # noqa: E402
    ResetIntervalScope,
    evidence_digest,
)
from scripts.e2e.regional.late_ownership_barrier import (  # noqa: E402
    BoundaryDenied,
    process_identity,
)
from scripts.e2e.regional.late_ownership_trace import (  # noqa: E402
    AttachedExecWitness,
)
from scripts.e2e.regional.probes.destr015_physical_probe import (  # noqa: E402
    ClockEnvelope,
    reset_events,
)

CASE_ID = "GF-REGIONAL-DESTR-015"
ROLE = "reset-interval-detached"
STATE_ROOT = Path("/var/lib/gpu-fault-acceptance/destr015")
LEDGER = Path("/var/lib/gpu-fault/node-actions.db")
HOST_PYTHON = "/opt/gpu-fault/current/venv/bin/python"
AGENT_UNIT = "gpu-fault-node-agent.service"
UNIT_PREFIX = "gpu-fault-destr015-witness-"
ENTRY = "scripts/e2e/regional/probes/destr015_witness_probe.py"
# The pinned sources the unit runs from its durable copy. Kept identical to the
# bundle's ``reset-interval-detached`` role (a unit test pins the two together)
# so the copy's digest equals the digest the runner measured on delivery.
PROGRAM_FILES = (
    "scripts/e2e/regional/late_ownership_contract.py",
    "scripts/e2e/regional/late_ownership_barrier.py",
    ENTRY,
    "scripts/e2e/regional/late_ownership_trace.py",
    "scripts/e2e/regional/probes/late_ownership_tracer_child.py",
    "scripts/e2e/regional/destr015_physical_evidence.py",
    "scripts/e2e/regional/probes/destr015_physical_probe.py",
)
PACKAGES = (
    "scripts",
    "scripts/e2e",
    "scripts/e2e/regional",
    "scripts/e2e/regional/probes",
)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
MAX_MESSAGE_BYTES = 65536
MIN_LIFETIME_SECONDS = 600
MAX_LIFETIME_SECONDS = 7200
# How much longer than its own deadline the unit may live before systemd kills
# it, and how long the request handlers wait for the unit to report.
UNIT_SLACK_SECONDS = 120
ARM_WAIT_SECONDS = 45
FINISH_WAIT_SECONDS = 45
POLL_SECONDS = 0.2
START_KEYS = (
    "scope_sha256",
    "witness_id",
    "tracee",
    "producer",
    "executable_path",
    "executable_sha256",
)
RESPONSES = {
    "arm": "armed",
    "collect": "collected",
    "disarm": "disarmed",
    "status": "status",
}
ACTIVE_STATES = frozenset({"active", "activating", "reloading", "deactivating"})


class ProbeError(RuntimeError):
    """A refusal this probe can name without leaking anything."""


# --------------------------------------------------------------------------- #
# Host helpers
# --------------------------------------------------------------------------- #
def run(
    command: list[str], *, check: bool = True, timeout: int = 60
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"},
    )
    if check and completed.returncode:
        raise ProbeError(f"host command failed ({completed.returncode}): {command[0]}")
    return completed


def safe_run_id(value: Any) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def run_digest(run_id: str) -> str:
    return hashlib.sha256(safe_run_id(run_id).encode()).hexdigest()[:16]


def unit_name(run_id: str) -> str:
    return UNIT_PREFIX + run_digest(run_id)


def run_directory(run_id: str, *, state_root: Path | None = None) -> Path:
    return (state_root or STATE_ROOT) / safe_run_id(run_id)


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def host_clock() -> dict[str, int]:
    return {"monotonic_ns": time.monotonic_ns(), "realtime_ns": time.time_ns()}


def write_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProbeError("witness record is malformed")
    return value


def unit_state(unit: str) -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            unit,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--property=InvocationID",
        ],
        check=False,
        timeout=20,
    )
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def unit_active(state: dict[str, str]) -> bool:
    return state.get("ActiveState") in ACTIVE_STATES


def clear_unit(unit: str) -> None:
    run(["systemctl", "stop", unit], check=False, timeout=30)
    run(["systemctl", "reset-failed", unit], check=False, timeout=20)


def agent_tracee() -> Any:
    pid_text = run(
        ["systemctl", "show", "--property=MainPID", "--value", AGENT_UNIT],
        timeout=15,
    ).stdout.strip()
    if not pid_text.isdecimal() or int(pid_text) <= 0:
        raise ProbeError("no Node Agent process identity")
    return process_identity(int(pid_text))


def resolve_executable() -> Path:
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise ProbeError("physical reset executable is unavailable")
    return Path(executable).resolve()


def gpu_present(gpu_uuid: str) -> bool:
    completed = run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        check=False,
        timeout=30,
    )
    if completed.returncode:
        return False
    return gpu_uuid in {line.strip() for line in completed.stdout.splitlines()}


def reset_ledger_rows(ledger: Path = LEDGER) -> list[dict[str, Any]]:
    """The Node Agent's RESET_GPU rows, read-only; unreadable means none seen."""

    if not ledger.is_file():
        return []
    try:
        connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT command_id, attempt, state, operation, started_at, "
                "completed_at, workflow_request_id, incident_id, fencing_token, "
                "agent_generation, gpu_uuids FROM results "
                "WHERE operation = 'RESET_GPU' ORDER BY completed_at, command_id"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return []
    return [
        {
            "command_id": row[0],
            "attempt": row[1],
            "state": row[2],
            "operation": row[3],
            "started_at": row[4],
            "completed_at": row[5],
            "workflow_request_id": row[6],
            "incident_id": row[7],
            "fencing_token": row[8],
            "agent_generation": row[9],
            "gpu_uuids": json.loads(row[10]) if row[10] else [],
        }
        for row in rows
    ]


def program_digest(sources: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_program_sources(root: Path = ROOT) -> dict[str, str]:
    return {name: (root / name).read_text(encoding="utf-8") for name in PROGRAM_FILES}


def install_program(directory: Path, sources: dict[str, str]) -> str:
    """Copy the pinned sources the unit runs into the run's private directory."""

    if set(sources) != set(PROGRAM_FILES):
        raise ProbeError("program sources do not match the pinned file set")
    if directory.exists() or directory.is_symlink():
        shutil.rmtree(directory)
    directory.mkdir(mode=0o700, parents=True)
    for name, text in sources.items():
        target = directory / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
    for package in PACKAGES:
        marker = directory / package / "__init__.py"
        if not marker.exists():
            marker.write_text("", encoding="ascii")
            marker.chmod(0o600)
    return program_digest(sources)


def unit_command(
    run_id: str, *, program_root: Path, lifetime_seconds: int, state_root: Path
) -> list[str]:
    """The transient unit: bounded by ``RuntimeMaxSec`` as a failsafe kill."""

    return [
        "systemd-run",
        "--unit",
        unit_name(run_id),
        f"--property=RuntimeMaxSec={int(lifetime_seconds) + UNIT_SLACK_SECONDS}",
        "--property=KillMode=control-group",
        sys.executable or HOST_PYTHON,
        "-I",
        "-u",
        str(program_root / ENTRY),
        "witness",
        "--run-id",
        run_id,
        "--state-root",
        str(state_root),
    ]


def reset_records(directory: Path) -> None:
    for name in ("armed.json", "final.json", "finish.request"):
        (directory / name).unlink(missing_ok=True)
    trace = directory / "trace"
    if trace.exists():
        shutil.rmtree(trace)


# --------------------------------------------------------------------------- #
# Request handlers (exec'd by the runner while kubelet answers)
# --------------------------------------------------------------------------- #
def parse_scope(value: Any) -> ResetIntervalScope:
    """The scope travels as JSON; the strict model is validated in JSON mode."""

    if not isinstance(value, dict):
        raise ProbeError("witness request carries no scope")
    return ResetIntervalScope.model_validate_json(json.dumps(value, allow_nan=False))


def check_scope(value: Any, scope_sha256: str) -> ResetIntervalScope:
    scope = parse_scope(value)
    if scope.digest() != scope_sha256:
        raise ProbeError("scope digest does not match the request")
    if scope.maintenance_end.tzinfo is None:
        raise ProbeError("scope maintenance end is naive")
    return scope


def await_armed(directory: Path, unit: str, *, seconds: float) -> dict[str, Any]:
    """Wait for the unit's armed record; a refusal or silence stops the unit."""

    deadline = time.monotonic() + seconds
    while True:
        armed = read_json(directory / "armed.json")
        if armed is not None:
            return armed
        final = read_json(directory / "final.json")
        if final is not None:
            refusal = str(final.get("refusal") or "witness ended before it was armed")
            break
        if time.monotonic() >= deadline:
            refusal = f"detached witness did not report armed within {int(seconds)}s"
            break
        time.sleep(POLL_SECONDS)
    clear_unit(unit)
    state = read_json(directory / "state.json") or {}
    state.update(phase="REFUSED", refusal=refusal)
    write_private_json(directory / "state.json", state)
    raise ProbeError(refusal)


def arm(
    payload: dict[str, Any], scope_sha256: str, *, state_root: Path | None = None
) -> dict[str, Any]:
    received = host_clock()
    scope = check_scope(payload.get("scope"), scope_sha256)
    run_id = safe_run_id(scope.run_id)
    lifetime = payload.get("lifetime_seconds")
    if (
        type(lifetime) is not int
        or not MIN_LIFETIME_SECONDS <= lifetime <= MAX_LIFETIME_SECONDS
    ):
        raise ProbeError("witness lifetime is outside its bounds")
    now = datetime.now(timezone.utc)
    if (scope.maintenance_end - now).total_seconds() < lifetime:
        raise ProbeError("witness lifetime does not fit the maintenance window")
    current_boot = boot_id()
    if current_boot != scope.boot_id:
        raise ProbeError("host boot id differs from the scope")
    tracee = agent_tracee()
    if tracee.boot_id != scope.boot_id:
        raise ProbeError("Node Agent boot id differs from the scope")
    executable = resolve_executable()
    if not gpu_present(scope.gpu_uuid):
        raise ProbeError("approved GPU is not present on this host")
    directory = run_directory(run_id, state_root=state_root)
    previous = read_json(directory / "state.json")
    if (
        previous is not None
        and previous.get("phase") == "ARMED"
        and not previous.get("disarmed_at")
    ):
        raise ProbeError("a witness is already armed for this run; disarm it first")
    unit = unit_name(run_id) + ".service"
    clear_unit(unit)
    reset_records(directory)
    program_root = directory / "program"
    program_sha256 = install_program(program_root, read_program_sources())
    state = {
        "case_id": CASE_ID,
        "run_id": run_id,
        "node": scope.node,
        "phase": "ARMED",
        "scope": scope.model_dump(mode="json"),
        "scope_sha256": scope_sha256,
        "boot_id": current_boot,
        "lifetime_seconds": lifetime,
        "armed_at": now.isoformat(),
        "unit": unit,
        "program_sha256": program_sha256,
        "tracee_pid": tracee.pid,
        "executable_path": str(executable),
        "runner_clock": payload.get("runner_clock"),
        "disarmed_at": None,
        "refusal": None,
    }
    write_private_json(directory / "state.json", state)
    run(
        unit_command(
            run_id,
            program_root=program_root,
            lifetime_seconds=lifetime,
            state_root=state_root or STATE_ROOT,
        ),
        timeout=60,
    )
    armed = await_armed(directory, unit, seconds=ARM_WAIT_SECONDS)
    return {
        "run_id": run_id,
        "node": scope.node,
        "unit": unit,
        "state_dir": str(directory),
        "lifetime_seconds": lifetime,
        "boot_id": current_boot,
        "program_sha256": program_sha256,
        "armed": armed,
        "host_clock_start": received,
        "host_clock": host_clock(),
        "runner_clock": payload.get("runner_clock"),
        "unit_state": unit_state(unit),
    }


def _bound_state(
    payload: dict[str, Any], scope_sha256: str, *, state_root: Path | None
) -> tuple[str, Path, dict[str, Any] | None]:
    run_id = safe_run_id(payload.get("run_id"))
    directory = run_directory(run_id, state_root=state_root)
    state = read_json(directory / "state.json")
    if state is not None and state.get("scope_sha256") != scope_sha256:
        raise ProbeError("witness state belongs to another scope")
    return run_id, directory, state


def collect(
    payload: dict[str, Any], scope_sha256: str, *, state_root: Path | None = None
) -> dict[str, Any]:
    received = host_clock()
    run_id, directory, state = _bound_state(
        payload, scope_sha256, state_root=state_root
    )
    if state is None or state.get("run_id") != run_id:
        raise ProbeError("no armed witness state exists for this run")
    current = boot_id()
    if current != state.get("boot_id"):
        raise ProbeError(
            "host boot id changed since the witness was armed; its records are "
            "not this boot's"
        )
    unit = str(state["unit"])
    final = read_json(directory / "final.json")
    requested: dict[str, int] | None = None
    if final is None:
        requested = host_clock()
        write_private_json(
            directory / "finish.request", {"requested_at": requested, "run_id": run_id}
        )
        deadline = time.monotonic() + FINISH_WAIT_SECONDS
        while final is None and time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            final = read_json(directory / "final.json")
    return {
        "run_id": run_id,
        "node": state.get("node"),
        "unit": unit,
        "state_dir": str(directory),
        "boot_id": current,
        "state": state,
        "armed": read_json(directory / "armed.json"),
        "final": final,
        "finish_requested": requested,
        "unit_state": unit_state(unit),
        "host_clock_start": received,
        "host_clock": host_clock(),
    }


def disarm(
    payload: dict[str, Any], scope_sha256: str, *, state_root: Path | None = None
) -> dict[str, Any]:
    """Stop the unit and drop the program copy. Idempotent: a unit that already
    ended, or a run that was never armed here, is not an error."""

    run_id, directory, state = _bound_state(
        payload, scope_sha256, state_root=state_root
    )
    unit = unit_name(run_id) + ".service"
    before = unit_state(unit)
    clear_unit(unit)
    now = datetime.now(timezone.utc).isoformat()
    if state is not None:
        state["disarmed_at"] = state.get("disarmed_at") or now
        if state.get("phase") == "ARMED":
            state["phase"] = "DISARMED"
        write_private_json(directory / "state.json", state)
    for name in ("program", "trace"):
        target = directory / name
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
    (directory / "finish.request").unlink(missing_ok=True)
    return {
        "run_id": run_id,
        "unit": unit,
        "unit_state_before": before,
        "unit_state": unit_state(unit),
        "state_present": state is not None,
        "disarmed_at": (state or {}).get("disarmed_at") or now,
        "boot_id": boot_id(),
        "host_clock": host_clock(),
    }


def status(
    payload: dict[str, Any], scope_sha256: str, *, state_root: Path | None = None
) -> dict[str, Any]:
    """Read-only: what the host recorded, including a refusal, and the unit."""

    run_id, directory, state = _bound_state(
        payload, scope_sha256, state_root=state_root
    )
    unit = unit_name(run_id) + ".service"
    return {
        "run_id": run_id,
        "unit": unit,
        "unit_state": unit_state(unit),
        "state": state,
        "armed": read_json(directory / "armed.json"),
        "final": read_json(directory / "final.json"),
        "boot_id": boot_id(),
        "host_clock": host_clock(),
    }


HANDLERS = {"arm": arm, "collect": collect, "disarm": disarm, "status": status}


def refused(scope_sha256: Any, reason: str) -> dict[str, Any]:
    return {
        "kind": "refused",
        "scope_sha256": scope_sha256 if isinstance(scope_sha256, str) else None,
        "payload": {"reason": reason},
    }


def handle_request(request: Any, *, state_root: Path | None = None) -> dict[str, Any]:
    if (
        not isinstance(request, dict)
        or set(request) != {"kind", "scope_sha256", "payload"}
        or request["kind"] not in HANDLERS
        or not isinstance(request["scope_sha256"], str)
        or not isinstance(request["payload"], dict)
    ):
        return refused(
            request.get("scope_sha256") if isinstance(request, dict) else None,
            "witness request is unbound or unsupported",
        )
    kind, scope_sha256 = request["kind"], request["scope_sha256"]
    try:
        payload = HANDLERS[kind](
            request["payload"], scope_sha256, state_root=state_root
        )
    except (ProbeError, BoundaryDenied) as exc:
        return refused(scope_sha256, str(exc))
    except Exception as exc:  # noqa: BLE001 - only the type may leave the host
        return refused(scope_sha256, type(exc).__name__)
    return {"kind": RESPONSES[kind], "scope_sha256": scope_sha256, "payload": payload}


def emit(response: dict[str, Any]) -> None:
    text = json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text.encode()) > MAX_MESSAGE_BYTES:
        text = json.dumps(
            refused(response.get("scope_sha256"), "witness response is oversized"),
            sort_keys=True,
            separators=(",", ":"),
        )
    print(text, flush=True)


# --------------------------------------------------------------------------- #
# The detached unit
# --------------------------------------------------------------------------- #
def attach(
    directory: Path, scope: ResetIntervalScope, *, deadline: float
) -> tuple[AttachedExecWitness, dict[str, Any]]:
    if boot_id() != scope.boot_id:
        raise ProbeError("host boot id differs from the scope")
    executable = resolve_executable()
    tracee = agent_tracee()
    if tracee.boot_id != scope.boot_id:
        raise ProbeError("Node Agent boot id differs from the scope")
    trace = directory / "trace"
    if trace.exists() or trace.is_symlink():
        shutil.rmtree(trace)
    trace.mkdir(mode=0o700)
    os.chmod(trace, 0o700)
    witness = AttachedExecWitness(
        trace,
        tracee,
        deadline=deadline + UNIT_SLACK_SECONDS // 2,
        executable=executable,
    )
    try:
        witness.start()
    except BaseException:
        # Never leave a half-attached tracer behind, whatever the tracer's own
        # failure path did.
        witness.close()
        raise
    start = {
        "scope_sha256": scope.digest(),
        "witness_id": secrets.token_hex(24),
        "tracee": tracee.model_dump(mode="json"),
        "producer": process_identity(os.getpid()).model_dump(mode="json"),
        "executable_path": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }
    return witness, start


def supervise(
    directory: Path,
    witness: AttachedExecWitness,
    clock: ClockEnvelope,
    *,
    deadline: float,
) -> str:
    """Keep the tracer supervised until collect asks to finish or the deadline."""

    while True:
        if (directory / "finish.request").exists():
            return "finish-request"
        if time.monotonic() >= deadline:
            return "deadline"
        clock.sample()
        witness.check()
        time.sleep(POLL_SECONDS)


def final_record(
    base: dict[str, Any],
    start: dict[str, Any],
    *,
    witness: AttachedExecWitness,
    clock: ClockEnvelope,
    scope: ResetIntervalScope,
    reason: str,
    baseline_ids: set[str],
) -> dict[str, Any]:
    raw = witness.snapshot()
    executable = Path(start["executable_path"])
    refusal: str | None = None
    try:
        actions, calibration = reset_events(
            raw, executable=executable, gpu_uuid=scope.gpu_uuid
        )
    except BoundaryDenied as exc:
        actions, calibration, refusal = [], 0, str(exc)
    if (
        hashlib.sha256(executable.read_bytes()).hexdigest()
        != start["executable_sha256"]
    ):
        refusal = "physical reset executable changed"
    stamp = clock.sample()
    witness.close()
    return {
        **base,
        "record": "final",
        "witness_id": start["witness_id"],
        "boot_id": boot_id(),
        **stamp,
        "realtime_ns": time.time_ns(),
        "reason": reason,
        "start_sha256": evidence_digest(start),
        "trace_complete": True,
        "closed": True,
        "lost_events": 0,
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "trace_bytes": len(raw),
        "calibration_execs": calibration,
        "actions": actions,
        "wall_minus_monotonic_min_ns": clock.minimum,
        "wall_minus_monotonic_max_ns": clock.maximum,
        "refusal": refusal,
        "ledger_reset_rows": [
            row for row in reset_ledger_rows() if row["command_id"] not in baseline_ids
        ],
    }


def failure_record(
    base: dict[str, Any], start: dict[str, Any] | None, *, refusal: str, closed: bool
) -> dict[str, Any]:
    return {
        **base,
        "record": "final",
        "witness_id": (start or {}).get("witness_id"),
        "boot_id": boot_id(),
        "monotonic_ns": time.monotonic_ns(),
        "realtime_ns": time.time_ns(),
        "reason": "failure",
        "start_sha256": evidence_digest(start) if start else None,
        "trace_complete": False,
        "closed": closed,
        "lost_events": None,
        "trace_sha256": None,
        "trace_bytes": 0,
        "calibration_execs": 0,
        "actions": [],
        "wall_minus_monotonic_min_ns": None,
        "wall_minus_monotonic_max_ns": None,
        "refusal": refusal,
        "ledger_reset_rows": [],
    }


def run_witness(run_id: str, *, state_root: Path | None = None) -> int:
    """The unit body: attach, record ``armed``, supervise, record ``final``."""

    directory = run_directory(run_id, state_root=state_root)
    state = read_json(directory / "state.json")
    base: dict[str, Any] = {
        "case_id": CASE_ID,
        "run_id": run_id,
        "unit": unit_name(run_id) + ".service",
        "invocation_id": os.environ.get("INVOCATION_ID"),
    }
    if state is None or state.get("run_id") != run_id or state.get("phase") != "ARMED":
        write_private_json(
            directory / "final.json",
            failure_record(
                base, None, refusal="unit started without an armed state", closed=True
            ),
        )
        return 1
    base.update(node=state.get("node"), scope_sha256=state.get("scope_sha256"))
    scope = parse_scope(state.get("scope"))
    deadline = time.monotonic() + int(state["lifetime_seconds"])
    clock = ClockEnvelope()
    witness: AttachedExecWitness | None = None
    start: dict[str, Any] | None = None
    try:
        witness, start = attach(directory, scope, deadline=deadline)
        baseline_ids = {str(row["command_id"]) for row in reset_ledger_rows()}
        write_private_json(
            directory / "armed.json",
            {
                **base,
                "record": "armed",
                "witness_id": start["witness_id"],
                "boot_id": boot_id(),
                **clock.sample(),
                "realtime_ns": time.time_ns(),
                "start": start,
                "ledger_baseline_command_ids": sorted(baseline_ids),
            },
        )
        reason = supervise(directory, witness, clock, deadline=deadline)
        final = final_record(
            base,
            start,
            witness=witness,
            clock=clock,
            scope=scope,
            reason=reason,
            baseline_ids=baseline_ids,
        )
    except Exception as exc:  # noqa: BLE001 - the record is the only report
        # A tracer that never attached (``attach`` closes its own failure) or
        # one detached here leaves nothing behind; only a failed detach does.
        closed = True
        if witness is not None:
            try:
                witness.close()
            except Exception:  # noqa: BLE001 - a failed detach stays recorded
                closed = False
        refusal = (
            str(exc)
            if isinstance(exc, (ProbeError, BoundaryDenied))
            else type(exc).__name__
        )
        write_private_json(
            directory / "final.json",
            failure_record(base, start, refusal=refusal, closed=closed),
        )
        return 1
    write_private_json(directory / "final.json", final)
    return 0


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        parser = argparse.ArgumentParser(
            description="DESTR-015 detached reset-interval witness unit."
        )
        commands = parser.add_subparsers(dest="command", required=True)
        witness = commands.add_parser("witness")
        witness.add_argument("--run-id", required=True)
        witness.add_argument("--state-root", default=str(STATE_ROOT))
        parsed = parser.parse_args(arguments)
        return run_witness(parsed.run_id, state_root=Path(parsed.state_root))
    line = sys.stdin.readline(MAX_MESSAGE_BYTES + 1)
    if len(line.encode()) > MAX_MESSAGE_BYTES:
        emit(refused(None, "witness request exceeds its bound"))
        return 1
    request = json.loads(line)
    emit(handle_request(request))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
