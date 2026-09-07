#!/usr/bin/env python3
"""On-node probe for GF-REGIONAL-DESTR-019.

Four node-local jobs the control plane cannot do for the runner, all read-only
except the one restart the case exists to perform:

* **restart-agent** -- ``systemctl restart gpu-fault-node-agent.service`` on an
  idle node, with a bounded ``systemd-run --on-active`` fail-safe that starts
  the unit again if this probe dies between stop and start. Restarting the
  Agent does not kill the ``kubectl exec`` channel (that is kubelet's), so the
  restart itself is synchronous and its before/after ``MainPID`` and
  ``InvocationID`` are reported in one answer.
* **agent-health** -- ``GET /healthz`` on the node's own Agent, over the TLS
  certificate the Agent advertises, reporting the ARCH-C5 payload
  (``ledger.writable``, ``heartbeat``, ``counters``) and the HTTP status.
* **journal** -- the Agent's ARCH-C3 structured lines from journald
  (``SyslogIdentifier=gpu-fault-node-agent``), parsed into phase plus
  ``key=value`` fields for one ``command_id``.
* **ledger-audit** -- the ARCH-C4 audit columns of the rows one command left
  in ``/var/lib/gpu-fault/node-actions.db`` plus ``PRAGMA user_version``,
  read through a read-only URI.
* **migration-drill** -- builds a *scratch* ledger under
  ``/var/lib/gpu-fault-acceptance`` from the pre-ARCH-C4 ``CREATE``/``ALTER``
  statements (``git show 84ce59f:src/gpu_fault/node_agent/ledger.py``), seeds
  legacy rows, then opens it with the **deployed** ``NodeActionLedger`` (this
  probe runs under ``/opt/gpu-fault/current/venv``) and reports what the
  in-place migration did. It never touches the real ledger.

The only mutating verbs are ``restart`` on the Agent unit and ``stop``/
``reset-failed`` on the fail-safe unit this probe named after its own run id.
Nothing here writes ``/dev/kmsg``, resets a GPU, or touches any other unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sqlite3
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LEDGER = Path("/var/lib/gpu-fault/node-actions.db")
STATE_DIR = Path("/var/lib/gpu-fault-acceptance")
AGENT_ENV = Path("/etc/gpu-fault/node-agent.env")
AGENT_UNIT = "gpu-fault-node-agent.service"
SYSLOG_IDENTIFIER = "gpu-fault-node-agent"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SAFE_COMMAND_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
# node-agent.env keys the probe may read: how to reach the local Agent. The
# shared secret and the token are deliberately absent.
AGENT_ENV_KEYS = (
    "GPU_FAULT_NODE_AGENT_HOST",
    "GPU_FAULT_NODE_AGENT_PORT",
    "GPU_FAULT_NODE_AGENT_TLS_CERT",
    "GPU_FAULT_NODE_ADVERTISE_URL",
    "GPU_FAULT_NODE_ACTION_RETENTION_SECONDS",
)
ALLOWED_SYSTEMCTL_VERBS = (
    "restart",
    "start",
    "stop",
    "reset-failed",
    "show",
    "is-active",
)
MIN_RESTORE_SECONDS = 60
MAX_RESTORE_SECONDS = 600
AGENT_ACTIVE_TIMEOUT_SECONDS = 90
# The four phases ARCH-C3 logs for a command, in order, plus the rejection.
LOG_PHASES = ("accepted", "started", "completed", "failed", "rejected")
LOG_LINE = re.compile(
    r"node action (?P<phase>accepted|started|completed|failed|rejected) (?P<fields>.*)$"
)
FIELD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\S*)")
# ARCH-C4 audit columns, as the shipped ledger declares them.
AUDIT_COLUMNS = (
    "command_id",
    "attempt",
    "state",
    "operation",
    "started_at",
    "completed_at",
    "incident_id",
    "workflow_request_id",
    "fencing_token",
    "gpu_uuids",
    "parameters_digest",
    "signature_digest",
    "exit_code",
)
# The pre-ARCH-C4 schema, statement for statement
# (git show 84ce59f:src/gpu_fault/node_agent/ledger.py). The migration drill
# rebuilds a ledger exactly the way an un-upgraded agent left it.
LEGACY_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS results (
        command_id TEXT PRIMARY KEY,
        payload TEXT NOT NULL,
        completed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fencing (
        incident_id TEXT PRIMARY KEY,
        token INTEGER NOT NULL,
        updated_at TEXT
    )
    """,
    "ALTER TABLE results ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE results ADD COLUMN state TEXT NOT NULL DEFAULT 'COMPLETED'",
    "ALTER TABLE results ADD COLUMN operation TEXT",
    "ALTER TABLE results ADD COLUMN started_at TEXT",
)
LEGACY_ROW_COUNT = 3


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
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


def safe_id(value: str, label: str) -> str:
    if SAFE_ID.fullmatch(value or "") is None:
        raise ProbeError(f"unsafe {label}")
    return value


def safe_command_id(value: str) -> str:
    if SAFE_COMMAND_ID.fullmatch(value or "") is None:
        raise ProbeError("unsafe command ID")
    return value


def _run_digest(run_id: str) -> str:
    return hashlib.sha256(safe_id(run_id, "run ID").encode()).hexdigest()[:16]


def restore_unit(run_id: str) -> str:
    return f"gpu-fault-destr019-restore-{_run_digest(run_id)}"


def owned_units(run_id: str) -> set[str]:
    unit = restore_unit(run_id)
    return {unit, f"{unit}.service", f"{unit}.timer"}


def checked_command(command: list[str], run_id: str) -> list[str]:
    """Refuse any systemctl verb/unit pair outside this probe's allow-list.

    The Agent unit accepts ``restart``, ``start`` and the read-only verbs; the
    fail-safe units this run named accept ``stop``/``reset-failed`` and reads.
    No verb may stop the Agent: an Agent left down is exactly the residue the
    fail-safe exists to prevent.
    """

    if len(command) < 3 or command[0] != "systemctl":
        raise ProbeError("only systemctl commands are permitted")
    verb, unit = command[1], command[2]
    if verb not in ALLOWED_SYSTEMCTL_VERBS:
        raise ProbeError(f"systemctl verb is not permitted: {verb}")
    if unit == AGENT_UNIT:
        if verb == "stop":
            raise ProbeError("this probe may not stop the Node Agent")
    elif unit in owned_units(run_id):
        if verb in {"restart", "start"}:
            raise ProbeError("fail-safe units are started only by systemd-run")
    else:
        raise ProbeError("unit is not in the DESTR-019 probe allow-list")
    for token in command[3:]:
        if not token.startswith("--property="):
            raise ProbeError(f"systemctl argument is not permitted: {token}")
    return command


def checked_restore_seconds(value: int) -> int:
    if not MIN_RESTORE_SECONDS <= int(value) <= MAX_RESTORE_SECONDS:
        raise ProbeError(
            f"restore seconds is outside {MIN_RESTORE_SECONDS}..{MAX_RESTORE_SECONDS}"
        )
    return int(value)


def state_path(run_id: str, *, state_dir: Path = STATE_DIR) -> Path:
    return state_dir / f"destr019-{safe_id(run_id, 'run ID')}.json"


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


# --------------------------------------------------------------------------- #
# systemd
# --------------------------------------------------------------------------- #
def unit_state(unit: str, run_id: str) -> dict[str, str]:
    completed = run(
        checked_command(
            [
                "systemctl",
                "show",
                unit,
                "--property=LoadState",
                "--property=ActiveState",
                "--property=SubState",
                "--property=MainPID",
                "--property=InvocationID",
                "--property=ExecMainStartTimestamp",
                "--property=NRestarts",
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


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def agent_env() -> dict[str, str]:
    """Only the addressing keys of node-agent.env; never the secret."""

    if not AGENT_ENV.is_file():
        return {}
    values: dict[str, str] = {}
    for line in AGENT_ENV.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key in AGENT_ENV_KEYS:
            values[key] = value.strip().strip("'\"")
    return values


def gpu_fault_timers() -> list[str]:
    completed = run(
        ["systemctl", "list-timers", "--all", "--no-legend", "--no-pager"],
        check=False,
    )
    return sorted(
        {
            match.group(0)
            for line in completed.stdout.splitlines()
            for match in re.finditer(r"gpu-fault-[A-Za-z0-9_.@-]+\.timer", line)
        }
    )


def _clear_restore(run_id: str) -> None:
    unit = restore_unit(run_id)
    for suffix in (".timer", ".service"):
        run(checked_command(["systemctl", "stop", unit + suffix], run_id), check=False)
        run(
            checked_command(["systemctl", "reset-failed", unit + suffix], run_id),
            check=False,
        )


def _wait_agent_active(run_id: str, timeout_seconds: int) -> dict[str, str]:
    deadline = time.monotonic() + timeout_seconds
    state = unit_state(AGENT_UNIT, run_id)
    while state.get("ActiveState") != "active" and time.monotonic() < deadline:
        time.sleep(1)
        state = unit_state(AGENT_UNIT, run_id)
    return state


def restart_agent(arguments: argparse.Namespace) -> None:
    """Restart the Agent once, behind a bounded fail-safe start.

    The fail-safe is armed *before* the restart so a probe that dies between
    systemd's stop and start still leaves an Agent on the node. ``systemd-run
    --on-active`` starts the unit; ``systemctl start`` on an already-active
    unit is a no-op, so a healthy restart makes the fail-safe harmless and the
    runner disarms it in cleanup.
    """

    run_id = safe_id(arguments.run_id, "run ID")
    restore_seconds = checked_restore_seconds(arguments.restore_seconds)
    before = unit_state(AGENT_UNIT, run_id)
    if before.get("ActiveState") != "active":
        raise ProbeError("the Node Agent is not active at baseline")
    path = state_path(run_id)
    armed_at = datetime.now(timezone.utc).isoformat()
    _clear_restore(run_id)
    run(
        [
            "systemd-run",
            f"--unit={restore_unit(run_id)}",
            f"--on-active={restore_seconds}s",
            "--timer-property=AccuracySec=1s",
            "/bin/systemctl",
            "start",
            AGENT_UNIT,
        ]
    )
    write_state(
        path,
        {
            "run_id": run_id,
            "boot_id": boot_id(),
            "restore_unit": restore_unit(run_id) + ".timer",
            "restore_seconds": restore_seconds,
            "armed_at": armed_at,
            "before": before,
        },
    )
    restarted_at = datetime.now(timezone.utc).isoformat()
    run(checked_command(["systemctl", "restart", AGENT_UNIT], run_id), timeout=120)
    after = _wait_agent_active(run_id, AGENT_ACTIVE_TIMEOUT_SECONDS)
    state = read_state(path)
    state.update({"restarted_at": restarted_at, "after": after})
    write_state(path, state)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"the Node Agent did not return to active: {after}")
    if after.get("MainPID") == before.get("MainPID"):
        raise ProbeError("the Node Agent MainPID did not change; nothing restarted")
    emit(
        {
            "run_id": run_id,
            "restore_unit": restore_unit(run_id) + ".timer",
            "restore_seconds": restore_seconds,
            "armed_at": armed_at,
            "restarted_at": restarted_at,
            "before": before,
            "after": after,
            "boot_id": boot_id(),
        }
    )


def disarm_restore(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    _clear_restore(run_id)
    path = state_path(run_id)
    state = read_state(path)
    if state:
        state["disarmed_at"] = datetime.now(timezone.utc).isoformat()
        write_state(path, state)
    emit(
        {
            "run_id": run_id,
            "restore_timer": unit_state(restore_unit(run_id) + ".timer", run_id),
            "restore_service": unit_state(restore_unit(run_id) + ".service", run_id),
            "agent": unit_state(AGENT_UNIT, run_id),
            "gpu_fault_timers": gpu_fault_timers(),
        }
    )


def ensure_agent_active(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    before = unit_state(AGENT_UNIT, run_id)
    started = False
    if before.get("ActiveState") != "active":
        run(checked_command(["systemctl", "start", AGENT_UNIT], run_id), timeout=120)
        started = True
    after = _wait_agent_active(run_id, AGENT_ACTIVE_TIMEOUT_SECONDS)
    if after.get("ActiveState") != "active":
        raise ProbeError("the Node Agent is not active after recovery")
    emit({"started": started, "before": before, "after": after})


# --------------------------------------------------------------------------- #
# /healthz
# --------------------------------------------------------------------------- #
def health_url(env: dict[str, str]) -> str:
    advertised = env.get("GPU_FAULT_NODE_ADVERTISE_URL", "").strip()
    if advertised:
        return advertised.rstrip("/") + "/healthz"
    host = env.get("GPU_FAULT_NODE_AGENT_HOST", "").strip() or "127.0.0.1"
    port = env.get("GPU_FAULT_NODE_AGENT_PORT", "").strip() or "9099"
    scheme = "https" if env.get("GPU_FAULT_NODE_AGENT_TLS_CERT", "").strip() else "http"
    return f"{scheme}://{host}:{port}/healthz"


def _ssl_context(env: dict[str, str]) -> ssl.SSLContext | None:
    certificate = env.get("GPU_FAULT_NODE_AGENT_TLS_CERT", "").strip()
    if not certificate:
        return None
    # The Agent's certificate is self-issued and the control plane trusts it by
    # value, not by hostname; do the same here rather than skipping verification.
    context = ssl.create_default_context(cafile=certificate)
    context.check_hostname = False
    return context


def agent_health(_arguments: argparse.Namespace) -> None:
    env = agent_env()
    url = health_url(env)
    request = urllib.request.Request(url, method="GET")
    observed_at = datetime.now(timezone.utc).isoformat()
    try:
        with urllib.request.urlopen(
            request, timeout=10, context=_ssl_context(env)
        ) as r:
            status = int(r.status)
            body = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        body = exc.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = {"raw": body[:500]}
    emit(
        {
            "observed_at": observed_at,
            "url": url,
            "http_status": status,
            "payload": payload,
        }
    )


# --------------------------------------------------------------------------- #
# journald
# --------------------------------------------------------------------------- #
def parse_log_line(message: str) -> dict[str, Any] | None:
    """One ARCH-C3 line -> ``{"phase": ..., "fields": {...}}``, else None."""

    match = LOG_LINE.search(message)
    if match is None:
        return None
    fields = dict(FIELD.findall(match.group("fields")))
    return {"phase": match.group("phase"), "fields": fields}


def journal_lines(
    since_epoch: float, *, command_id: str | None = None
) -> list[dict[str, Any]]:
    completed = run(
        [
            "journalctl",
            "-t",
            SYSLOG_IDENTIFIER,
            "--since",
            f"@{since_epoch:.6f}",
            "--output=json",
            "--no-pager",
        ],
        check=False,
    )
    result: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        parsed = parse_log_line(str(value.get("MESSAGE") or ""))
        if parsed is None:
            continue
        if command_id is not None and parsed["fields"].get("command_id") != command_id:
            continue
        parsed["realtime_timestamp"] = value.get("__REALTIME_TIMESTAMP")
        parsed["invocation_id"] = value.get("_SYSTEMD_INVOCATION_ID")
        parsed["pid"] = value.get("_PID")
        result.append(parsed)
    return result


def journal(arguments: argparse.Namespace) -> None:
    command_id = safe_command_id(arguments.command_id) if arguments.command_id else None
    lines = journal_lines(float(arguments.since_epoch), command_id=command_id)
    emit(
        {
            "since_epoch": float(arguments.since_epoch),
            "command_id": command_id,
            "line_count": len(lines),
            "lines": lines,
        }
    )


# --------------------------------------------------------------------------- #
# Ledger audit
# --------------------------------------------------------------------------- #
def ledger_columns(connection: sqlite3.Connection) -> list[str]:
    return [row[1] for row in connection.execute("PRAGMA table_info(results)")]


def ledger_primary_key(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute("PRAGMA table_info(results)").fetchall()
    keyed = sorted((row[5], row[1]) for row in rows if row[5])
    return [name for _, name in keyed]


def ledger_audit_rows(
    ledger: Path, *, command_id: str | None = None, since: str | None = None
) -> dict[str, Any]:
    if not ledger.is_file():
        return {"present": False, "user_version": None, "columns": [], "rows": []}
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        columns = ledger_columns(connection)
        selectable = [name for name in AUDIT_COLUMNS if name in columns]
        clauses = []
        parameters: list[Any] = []
        if command_id is not None:
            clauses.append("command_id=?")
            parameters.append(command_id)
        if since is not None:
            clauses.append("(started_at >= ? OR completed_at >= ?)")
            parameters.extend([since, since])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = connection.execute(
            f"SELECT {', '.join(selectable)} FROM results{where} "
            "ORDER BY command_id, attempt",
            parameters,
        ).fetchall()
        interrupted = int(
            connection.execute(
                "SELECT COUNT(*) FROM results WHERE state='INTERRUPTED'"
            ).fetchone()[0]
        )
        total = int(connection.execute("SELECT COUNT(*) FROM results").fetchone()[0])
        primary_key = ledger_primary_key(connection)
    finally:
        connection.close()
    entries = []
    for row in rows:
        entry = dict(zip(selectable, row))
        raw_uuids = entry.pop("gpu_uuids", None)
        try:
            uuids = json.loads(raw_uuids) if isinstance(raw_uuids, str) else None
        except json.JSONDecodeError:
            uuids = None
        # Counts, not the UUID list: the case's evidence keeps identifiers
        # to the ones the control plane already has.
        entry["gpu_uuid_count"] = len(uuids) if isinstance(uuids, list) else None
        entry["gpu_uuids_present"] = raw_uuids is not None
        entries.append(entry)
    return {
        "present": True,
        "user_version": version,
        "columns": columns,
        "primary_key": primary_key,
        "row_count": total,
        "interrupted_count": interrupted,
        "rows": entries,
    }


def ledger_audit(arguments: argparse.Namespace) -> None:
    command_id = safe_command_id(arguments.command_id) if arguments.command_id else None
    emit(
        ledger_audit_rows(LEDGER, command_id=command_id, since=arguments.since or None)
    )


# --------------------------------------------------------------------------- #
# Migration drill (scratch ledger, deployed NodeActionLedger)
# --------------------------------------------------------------------------- #
def build_legacy_ledger(path: Path, *, now: datetime) -> list[str]:
    """A ledger exactly as the pre-C4 agent would have left it."""

    from gpu_fault.models import WorkflowOperation
    from gpu_fault.node_agent.protocol import NodeActionResult, NodeActionStatus

    if path.exists():
        path.unlink()
    connection = sqlite3.connect(str(path), isolation_level=None)
    try:
        for statement in LEGACY_STATEMENTS:
            connection.execute(statement)
        command_ids = []
        for index in range(LEGACY_ROW_COUNT):
            command_id = f"legacy-workflow/{index}/VERIFY_NO_GPU_CLIENTS/node/attempt-1"
            completed = now - timedelta(minutes=LEGACY_ROW_COUNT - index)
            result = NodeActionResult(
                command_id=command_id,
                operation=WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                status=NodeActionStatus.SUCCEEDED,
                attempt=1,
                completed_at=completed,
            )
            connection.execute(
                "INSERT INTO results(command_id, payload, completed_at, attempt, "
                "state, operation, started_at) VALUES (?, ?, ?, 1, 'SUCCEEDED', ?, ?)",
                (
                    command_id,
                    result.model_dump_json(),
                    completed.isoformat(),
                    "VERIFY_NO_GPU_CLIENTS",
                    (completed - timedelta(seconds=5)).isoformat(),
                ),
            )
            command_ids.append(command_id)
        connection.execute(
            "INSERT INTO fencing(incident_id, token, updated_at) VALUES (?, ?, ?)",
            ("legacy-incident", 3, now.isoformat()),
        )
    finally:
        connection.close()
    return command_ids


def migration_drill_report(
    scratch: Path, *, now: datetime | None = None
) -> dict[str, Any]:
    """Build, migrate and re-read one scratch ledger; return the evidence.

    Importable by the unit tests, which run it against a temporary directory
    with the worktree's ledger implementation; on the node it runs against the
    deployed wheel. Both must produce the same shape.
    """

    from gpu_fault.models import WorkflowOperation
    from gpu_fault.node_agent.ledger import LEDGER_SCHEMA_VERSION, NodeActionLedger
    from gpu_fault.node_agent.protocol import (
        NodeActionCommand,
        NodeActionResult,
        NodeActionStatus,
    )

    moment = now or datetime.now(timezone.utc)
    scratch.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    command_ids = build_legacy_ledger(scratch, now=moment)
    before = ledger_audit_rows(scratch)
    ledger = NodeActionLedger(str(scratch))
    try:
        after_open = ledger_audit_rows(scratch)
        history = ledger.attempt_history(command_ids[0])
        readable = ledger.get(command_ids[0])
        fencing_kept = ledger.accept_fencing("legacy-incident", 3)
        stale_fencing_refused = not ledger.accept_fencing("legacy-incident", 2)
        # Append attempt 2 to a legacy command the way the executor does.
        command = NodeActionCommand(
            command_id=command_ids[0],
            workflow_request_id="legacy-workflow",
            incident_id="legacy-incident",
            fencing_token=3,
            operation=WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            node_id="drill-node",
            gpu_uuids=["GPU-drill"],
            parameters={"compute_clients_only": False},
            issued_at=moment,
            expires_at=moment + timedelta(minutes=5),
        )
        ledger.mark_in_progress(command, 2, signature="drill-signature")
        ledger.save(
            NodeActionResult(
                command_id=command_ids[0],
                operation=WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                status=NodeActionStatus.FAILED,
                error="drill",
                retryable=True,
                attempt=2,
                completed_at=moment,
            ),
            exit_code=7,
        )
        appended = ledger.attempt_history(command_ids[0])
        latest = ledger.get(command_ids[0])
    finally:
        ledger.close()
    after = ledger_audit_rows(scratch)
    connection = sqlite3.connect(f"file:{scratch}?mode=ro", uri=True)
    try:
        legacy_table_present = bool(
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='results_legacy'"
            ).fetchone()
        )
    finally:
        connection.close()
    for suffix in ("", "-wal", "-shm"):
        Path(str(scratch) + suffix).unlink(missing_ok=True)
    return {
        "scratch_path": str(scratch),
        "expected_schema_version": LEDGER_SCHEMA_VERSION,
        "legacy_command_ids": command_ids,
        "before": {
            "user_version": before["user_version"],
            "columns": before["columns"],
            "primary_key": before["primary_key"],
            "row_count": before["row_count"],
        },
        "after_open": {
            "user_version": after_open["user_version"],
            "columns": after_open["columns"],
            "primary_key": after_open["primary_key"],
            "row_count": after_open["row_count"],
        },
        "legacy_table_present": legacy_table_present,
        "history_before_append": history,
        "readable_legacy_result": (
            readable.model_dump(mode="json") if readable is not None else None
        ),
        "fencing_kept": fencing_kept,
        "stale_fencing_refused": stale_fencing_refused,
        "history_after_append": appended,
        "latest_after_append": (
            latest.model_dump(mode="json") if latest is not None else None
        ),
        "final_row_count": after["row_count"],
        "scratch_removed": not scratch.exists(),
    }


def migration_drill(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    scratch = STATE_DIR / f"destr019-{run_id}-legacy-ledger.db"
    emit(migration_drill_report(scratch))


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #
def snapshot(arguments: argparse.Namespace) -> None:
    run_id = arguments.run_id or "destr019-snapshot"
    safe_id(run_id, "run ID")
    ledger = ledger_audit_rows(LEDGER, since="9999")  # header only, no rows
    emit(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "boot_id": boot_id(),
            "agent": unit_state(AGENT_UNIT, run_id),
            "agent_env": agent_env(),
            "ledger": {
                "present": ledger["present"],
                "user_version": ledger.get("user_version"),
                "columns": ledger.get("columns"),
                "primary_key": ledger.get("primary_key"),
                "row_count": ledger.get("row_count"),
                "interrupted_count": ledger.get("interrupted_count"),
            },
            "gpu_fault_timers": gpu_fault_timers(),
            "restore_timer": (
                unit_state(restore_unit(run_id) + ".timer", run_id)
                if arguments.run_id
                else None
            ),
            "state": read_state(state_path(run_id)) if arguments.run_id else {},
        }
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "GF-REGIONAL-DESTR-019 on-node probe: restart the Node Agent behind a "
            "fail-safe, read /healthz, journald and the ledger audit columns, and "
            "drill the ledger migration on a scratch copy."
        )
    )
    commands = value.add_subparsers(dest="command", required=True)

    restart = commands.add_parser("restart-agent")
    restart.add_argument("--run-id", required=True)
    restart.add_argument("--restore-seconds", type=int, default=180)
    restart.set_defaults(handler=restart_agent)

    disarm = commands.add_parser("disarm-restore")
    disarm.add_argument("--run-id", required=True)
    disarm.set_defaults(handler=disarm_restore)

    ensure = commands.add_parser("ensure-agent-active")
    ensure.add_argument("--run-id", required=True)
    ensure.set_defaults(handler=ensure_agent_active)

    health = commands.add_parser("agent-health")
    health.set_defaults(handler=agent_health)

    log = commands.add_parser("journal")
    log.add_argument("--since-epoch", type=float, required=True)
    log.add_argument("--command-id", default="")
    log.set_defaults(handler=journal)

    audit = commands.add_parser("ledger-audit")
    audit.add_argument("--command-id", default="")
    audit.add_argument("--since", default="")
    audit.set_defaults(handler=ledger_audit)

    drill = commands.add_parser("migration-drill")
    drill.add_argument("--run-id", required=True)
    drill.set_defaults(handler=migration_drill)

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
