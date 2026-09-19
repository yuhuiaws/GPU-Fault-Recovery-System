"""Cross-process deployment API admission and credential-free accounting.

The CLI shim deliberately uses only the standard library: it also runs from a
source checkout with isolated Python, without inheriting a workspace PYTHONPATH.
"""

from __future__ import annotations

import configparser
import contextlib
import contextvars
import hashlib
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

if not __package__:
    # Isolated shims load only their own release, never a caller's PYTHONPATH.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gpu_fault.admin.deadlines import (
    DeploymentDeadlineExceeded,
    cleanup_deadline,
    current_deadline,
    remaining_timeout,
)

ROOT_ENV = "GPU_FAULT_DEPLOY_API_BUDGET_DIR"
PHASE_ENV = "GPU_FAULT_DEPLOY_API_PHASE"
PARENT_ENV = "GPU_FAULT_DEPLOY_API_PARENT"
# Bump for incompatible ledger schema, admission policy or handoff semantics.
BUDGET_PROTOCOL_VERSION = 4
_PHASE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "deploy_api_phase", default=""
)
LIMITS = {"aws": 8, "kubectl": 8, "http": 4}
START_INTERVAL = {"aws": 0.125, "kubectl": 0.05, "http": 0.1}
TRANSFER_CONCURRENCY = 4
MAX_COMMAND_DEPTH = 8
MAX_NESTED_OUTPUT_BYTES = 8 * 1024 * 1024
_LABEL = re.compile(r"^[a-zA-Z0-9_./:-]{1,128}$")


class ApiBudgetError(RuntimeError):
    pass


class ApiBudgetProtocolError(ApiBudgetError):
    pass


def budget_root() -> Path | None:
    value = os.environ.get(ROOT_ENV)
    if value is None:
        return None
    root = Path(value)
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.stat().st_uid != os.geteuid()
        or root.stat().st_mode & 0o077
    ):
        raise ApiBudgetError("deployment API budget directory is not private and owned")
    _validate_protocol(root)
    return root


@contextlib.contextmanager
def _database(root: Path, *, timeout: float = 10.0) -> Iterator[sqlite3.Connection]:
    path = root / "budget.sqlite3"
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_uid != os.geteuid()
        or path.stat().st_mode & 0o077
    ):
        raise ApiBudgetError("deployment API budget database identity differs")
    database = sqlite3.connect(path, timeout=timeout)
    database.execute(f"PRAGMA busy_timeout={max(1, math.ceil(timeout * 1000))}")
    try:
        yield database
    finally:
        database.close()


def _validate_protocol(root: Path) -> None:
    message = (
        "incompatible inherited deployment API budget protocol; "
        "use matching deploy-host and source drivers"
    )
    try:
        with _database(root, timeout=remaining_timeout(10)) as database:
            database.execute("BEGIN")
            version = database.execute("PRAGMA user_version").fetchone()[0]
            if version != BUDGET_PROTOCOL_VERSION:
                raise ApiBudgetProtocolError(message)
            capacity = database.execute(
                "SELECT backend,max_weight,next_start,peak FROM capacity"
            ).fetchall()
            if (
                len(capacity) != len(LIMITS)
                or {backend: limit for backend, limit, _next, _peak in capacity}
                != LIMITS
            ):
                raise ApiBudgetProtocolError(message)
            database.execute(
                "SELECT id,backend,weight,pid,pid_start,command_pid,command_start,"
                "parent_id,depth,state,lent_to FROM leases LIMIT 0"
            )
            if database.execute(
                "SELECT 1 FROM leases WHERE state IS NULL OR state NOT IN "
                "('waiting','active','stopping','parked','resuming','killing','cancelled')"
            ).fetchone():
                raise ApiBudgetProtocolError(message)
            database.execute(
                "SELECT id,phase,backend,wait_seconds,finished,sdk_calls,sdk_attempts,"
                "attempt_records,sdk_retries,telemetry_expected FROM calls LIMIT 0"
            )
            remaining_timeout(10)
    except sqlite3.Error as error:
        if getattr(error, "sqlite_errorcode", None) in {
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
        }:
            remaining_timeout(10)
            raise
        raise ApiBudgetProtocolError(message) from None


def _read_process_stat(path: Path) -> tuple[str, int, str]:
    # comm may itself contain spaces or parentheses; the final ')' precedes
    # the fixed stat fields. Field 22 is the process start time.
    fields = path.read_text().rsplit(")", 1)[1].split()
    return fields[0], int(fields[1]), fields[19]


def _process_stat(path: Path) -> tuple[str, int, str]:
    try:
        return _read_process_stat(path)
    except (OSError, IndexError, ValueError):
        return "", 0, ""


_EXIT_STATES = frozenset({"Z", "X"})


def _pid_identity(pid: int) -> str:
    state, _parent, start = _process_stat(Path(f"/proc/{pid}/stat"))
    return start if state not in {"", *_EXIT_STATES} else ""


# A zombie thread-group leader is not proof of exit. When a multi-threaded
# program execs from a non-leader thread -- the Go launchers behind the snap
# packaged aws and kubectl (`snap run`, `snap-exec`) do -- the kernel reports
# the dying leader as Z under the process's pid and start time until the
# exec'ing thread takes that pid over, a scheduler-latency-sized window. A real
# zombie stays Z until its parent reaps it and then disappears. Confirm across
# these delays before treating a zombie as an exit (live 2026-09-19: one such
# read released a live aws CLI's reservation and failed the reaping sibling's
# own command, `cannot release a running API command`).
EXIT_CONFIRMATION_DELAYS = (0.002, 0.005, 0.01, 0.02, 0.03)
_exit_confirmation = threading.Event()


def _confirmation_wait(seconds: float) -> None:
    # Not `time.sleep`: the confirmation is a probe detail, not admission timing.
    _exit_confirmation.wait(seconds)


def _live_task(pid: int) -> bool:
    """Whether a task of the thread group still runs beside a zombie leader."""
    try:
        entries = list(Path(f"/proc/{pid}/task").iterdir())
    except OSError:
        return False
    return any(
        _process_stat(entry / "stat")[0] not in {"", *_EXIT_STATES} for entry in entries
    )


def _identity_gone(pid: int, start: str, *, allow_reuse: bool = True) -> bool:
    if not start:
        return False
    for delay in (0.0, *EXIT_CONFIRMATION_DELAYS):
        if delay:
            _confirmation_wait(delay)
        try:
            state, _parent, actual = _read_process_stat(Path(f"/proc/{pid}/stat"))
        except (FileNotFoundError, ProcessLookupError):
            return True
        except (OSError, IndexError, ValueError):
            # Unknown identity is not evidence of exit. Keep capacity charged
            # when procfs is unreadable rather than treating a failed probe as
            # a dead CLI.
            return False
        if not actual:
            return False
        if actual != start:
            return allow_reuse
        if state not in _EXIT_STATES or _live_task(pid):
            return False
    return True


def _ancestors(pid: int) -> dict[int, str]:
    ancestors: dict[int, str] = {}
    for _ in range(256):
        _state, parent, start = _process_stat(Path(f"/proc/{pid}/stat"))
        if not start or pid in ancestors:
            raise ApiBudgetError("cannot establish API command ancestry")
        ancestors[pid] = start
        if not parent:
            return ancestors
        pid = parent
    raise ApiBudgetError("API command ancestry exceeds its bound")


@dataclass(frozen=True)
class _Lease:
    id: str
    backend: str
    weight: int
    pid: int
    pid_start: str
    command_pid: int | None
    command_start: str | None
    parent_id: str | None
    depth: int
    state: str
    lent_to: str | None


def _lease(database: sqlite3.Connection, identifier: str) -> _Lease | None:
    row = database.execute(
        "SELECT id,backend,weight,pid,pid_start,command_pid,command_start,"
        "parent_id,depth,state,lent_to FROM leases WHERE id=?",
        (identifier,),
    ).fetchone()
    return _Lease(*row) if row else None


def _parent_lease(
    database: sqlite3.Connection,
    identifier: str,
    pid: int,
    *,
    nearest: bool = True,
) -> _Lease:
    parent = _lease(database, identifier)
    ancestors = _ancestors(pid)
    if (
        re.fullmatch(r"[0-9a-f]{32}", identifier) is None
        or parent is None
        or parent.pid == pid
        or parent.command_pid == pid
        or ancestors.get(parent.pid) != parent.pid_start
        or _pid_identity(parent.pid) != parent.pid_start
        or parent.state not in {"active", "stopping", "parked", "resuming"}
    ):
        raise ApiBudgetError("deployment API parent identity differs")
    # The inherited ID is a hint, not a capability. It must name the nearest
    # ledger-owned CLI, not a sibling's lease or a more distant ancestor.
    if nearest:
        for ancestor, start in ancestors.items():
            if ancestor == parent.pid:
                break
            if database.execute(
                "SELECT 1 FROM leases WHERE pid=? AND pid_start=? "
                "AND command_pid IS NOT NULL",
                (ancestor, start),
            ).fetchone():
                raise ApiBudgetError("deployment API parent ancestry differs")
    if parent.command_pid is not None and (
        ancestors.get(parent.command_pid) != parent.command_start
        or _pid_identity(parent.command_pid) != parent.command_start
        or _process_stat(Path(f"/proc/{parent.command_pid}/stat"))[1] != parent.pid
    ):
        raise ApiBudgetError("deployment API parent command identity differs")
    if parent.depth + 1 >= MAX_COMMAND_DEPTH:
        raise ApiBudgetError("deployment API command nesting exceeds its bound")
    return parent


def _capacity_parent(
    database: sqlite3.Connection, parent: _Lease | None, backend: str, pid: int
) -> _Lease | None:
    for _ in range(MAX_COMMAND_DEPTH):
        if parent is None or parent.backend == backend:
            return parent
        parent = (
            _parent_lease(database, parent.parent_id, pid, nearest=False)
            if parent.parent_id is not None
            else None
        )
    raise ApiBudgetError("deployment API command nesting exceeds its bound")


def _command_descriptor(parent: _Lease, *, runnable: bool = False) -> int:
    if parent.command_pid is None or not hasattr(os, "pidfd_open"):
        raise ApiBudgetError("cannot safely hand off API command capacity")
    descriptor = os.pidfd_open(parent.command_pid)
    try:
        state, owner, start = _process_stat(Path(f"/proc/{parent.command_pid}/stat"))
        if not start or start != parent.command_start or state in {"", "Z", "X"}:
            raise ApiBudgetError("deployment API parent command identity differs")
        if runnable and (
            state in {"T", "t"}
            or owner != parent.pid
            or _pid_identity(parent.pid) != parent.pid_start
        ):
            raise ApiBudgetError("deployment API parent is not runnable")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stop_parent(parent: _Lease, descriptor: int) -> None:
    signal.pidfd_send_signal(descriptor, signal.SIGSTOP)
    deadline = time.monotonic() + min(2, _remaining())
    tasks = Path(f"/proc/{parent.command_pid}/task")
    while time.monotonic() < deadline:
        entries = list(tasks.iterdir())
        # S3 has worker threads. An idle leader alone does not prove that
        # the parent's API work stopped; every thread must acknowledge STOP.
        if (
            entries
            and parent.command_pid is not None
            and _pid_identity(parent.command_pid) == parent.command_start
            and all(_process_stat(entry / "stat")[0] in {"T", "t"} for entry in entries)
            and {entry.name for entry in entries}
            == {entry.name for entry in tasks.iterdir()}
        ):
            return
        time.sleep(0.005)
    raise ApiBudgetError("deployment API parent did not stop")


def _lend_slot(
    database: sqlite3.Connection, parent: _Lease, identifier: str, weight: int
) -> int:
    descriptor = _command_descriptor(parent, runnable=True)
    try:
        weight = max(weight, parent.weight)
        database.execute(
            "UPDATE leases SET state='stopping',lent_to=? WHERE id=?",
            (identifier, parent.id),
        )
        database.execute("UPDATE leases SET weight=? WHERE id=?", (weight, identifier))
        # Both the branch exclusion and the charged STOP intent must survive
        # helper death, including death before any thread acknowledges STOP.
        database.commit()
        database.execute("BEGIN IMMEDIATE")
        _stop_parent(parent, descriptor)
        database.execute("UPDATE leases SET state='parked' WHERE id=?", (parent.id,))
        database.commit()
        database.execute("BEGIN IMMEDIATE")
        return weight
    finally:
        os.close(descriptor)


def _release_lease(
    database: sqlite3.Connection, identifier: str, *, owner: bool = True
) -> str | None:
    child = _lease(database, identifier)
    if (
        child is not None
        and child.command_pid is not None
        and not _identity_gone(child.command_pid, child.command_start or "")
    ):
        if owner:
            raise ApiBudgetError("cannot release a running API command")
        # A reaper is a sibling, not the owner: its earlier probe saw a state
        # that did not last, the command still runs and its reservation stays
        # charged. Failing the reaper's own command would punish the wrong CLI.
        return None
    borrowed = child is not None and (
        child.state == "cancelled"
        or (child.lent_to is not None and _lease(database, child.lent_to) is not None)
    )
    database.execute("DELETE FROM leases WHERE id=?", (identifier,))
    database.execute("UPDATE calls SET finished=1 WHERE id=?", (identifier,))
    row = database.execute(
        "SELECT id FROM leases WHERE lent_to=?", (identifier,)
    ).fetchone()
    parent = _lease(database, row[0]) if row else None
    if parent is None:
        return None
    if parent.state == "stopping":
        # STOP was not durably proved, so its weight was never released.
        database.execute("UPDATE leases SET state='resuming' WHERE id=?", (parent.id,))
    elif parent.state == "parked":
        restored = _occupied(database, parent.backend) + parent.weight
        if not borrowed and restored <= LIMITS[parent.backend]:
            database.execute(
                "UPDATE leases SET state='resuming' WHERE id=?", (parent.id,)
            )
            database.execute(
                "UPDATE capacity SET peak=MAX(peak,?) WHERE backend=?",
                (restored, parent.backend),
            )
        else:
            # A waiting heavier child may have nothing to return. A dead middle
            # CLI may also still have a runnable helper. Neither permits CONT.
            database.execute(
                "UPDATE leases SET state='killing' WHERE id=?", (parent.id,)
            )
    return parent.id


def _reap_leases(database: sqlite3.Connection) -> None:
    for identifier, pid, start, command_pid, command_start in database.execute(
        "SELECT id,pid,pid_start,command_pid,command_start FROM leases ORDER BY depth DESC"
    ).fetchall():
        # A dead shim does not release a live CLI. Conversely, an exited CLI
        # cannot retain a loan just because its shim is still draining pipes.
        owner_gone = _identity_gone(pid, start)
        if (
            command_pid is not None
            and _identity_gone(command_pid, command_start, allow_reuse=owner_gone)
        ) or (command_pid is None and owner_gone):
            _release_lease(database, identifier, owner=False)


def _settle_handoffs(database: sqlite3.Connection) -> None:
    for (identifier,) in database.execute(
        "SELECT id FROM leases WHERE state IN ('resuming','killing')"
    ).fetchall():
        parent = _lease(database, identifier)
        if parent is None or parent.command_pid is None:
            raise ApiBudgetError("deployment API handoff lost its command identity")
        gone = _identity_gone(
            parent.command_pid,
            parent.command_start or "",
            allow_reuse=_identity_gone(parent.pid, parent.pid_start),
        )
        if not gone:
            try:
                descriptor = _command_descriptor(parent)
            except (OSError, ApiBudgetError):
                # Unknown identity is neither permission to signal nor proof
                # that a charged reservation can be discarded.
                continue
            try:
                resume = (
                    parent.state == "resuming"
                    and _pid_identity(parent.pid) == parent.pid_start
                    and _process_stat(Path(f"/proc/{parent.command_pid}/stat"))[1]
                    == parent.pid
                )
                signal.pidfd_send_signal(
                    descriptor, signal.SIGCONT if resume else signal.SIGKILL
                )
                if resume:
                    database.execute(
                        "UPDATE leases SET state='active',lent_to=NULL WHERE id=?",
                        (parent.id,),
                    )
                    continue
            except OSError:
                continue
            finally:
                os.close(descriptor)
            gone = _identity_gone(
                parent.command_pid,
                parent.command_start or "",
                allow_reuse=_identity_gone(parent.pid, parent.pid_start),
            )
        if gone:
            database.execute(
                "UPDATE leases SET state='cancelled',lent_to=NULL WHERE id=?",
                (parent.id,),
            )


def _recover_slots(root: Path, *, timeout: float = 10.0) -> None:
    with _database(root, timeout=timeout) as database:
        database.execute("BEGIN IMMEDIATE")
        _reap_leases(database)
        database.commit()
        database.execute("BEGIN IMMEDIATE")
        # Only committed intents are signalled. Keep the SQLite write lock
        # through acknowledgement so no sibling can start a fresh STOP first.
        _settle_handoffs(database)
        database.commit()


def _occupied(database: sqlite3.Connection, backend: str) -> int:
    return int(
        database.execute(
            "SELECT COALESCE(SUM(weight),0) FROM leases "
            "WHERE backend=? AND state IN ('active','stopping','resuming')",
            (backend,),
        ).fetchone()[0]
    )


def _finish_slot(root: Path, identifier: str) -> None:
    with _database(root) as database:
        database.execute("BEGIN IMMEDIATE")
        parent_id = _release_lease(database, identifier)
        database.commit()
    deadline = time.monotonic() + 2
    while True:
        _recover_slots(root)
        with _database(root) as database:
            parent = _lease(database, parent_id) if parent_id else None
        if parent is None or parent.lent_to != identifier:
            return
        if time.monotonic() >= deadline:
            raise ApiBudgetError("deployment API parent handoff is still pending")
        time.sleep(0.005)


def _bind_command(root: Path, identifier: str) -> None:
    pid = os.getpid()
    start = _pid_identity(pid)
    with _database(root) as database:
        database.execute("BEGIN IMMEDIATE")
        lease = _lease(database, identifier)
        if (
            lease is None
            or lease.pid != os.getppid()
            or _pid_identity(lease.pid) != lease.pid_start
            or not start
            or lease.state != "active"
            or lease.command_pid is not None
        ):
            raise ApiBudgetError("deployment API child identity differs")
        _remaining()
        database.execute(
            "UPDATE leases SET command_pid=?,command_start=? WHERE id=?",
            (pid, start, identifier),
        )
        database.commit()


def _remaining() -> float:
    try:
        return remaining_timeout(21600 if current_deadline() is not None else 3600)
    except (ValueError, DeploymentDeadlineExceeded):
        raise ApiBudgetError(
            "deployment deadline expired or invalid while waiting for API capacity"
        ) from None


def _label(value: str) -> str:
    return value if _LABEL.fullmatch(value) else "unclassified"


@contextlib.contextmanager
def api_phase(name: str) -> Iterator[None]:
    token = _PHASE.set(_label(name))
    try:
        yield
    finally:
        _PHASE.reset(token)


def api_environment(environment: Mapping[str, str] | None) -> dict[str, str] | None:
    root = budget_root()
    if root is None:
        return dict(environment) if environment is not None else None
    values = dict(os.environ if environment is None else environment)
    values[ROOT_ENV] = str(root)
    values[PHASE_ENV] = _PHASE.get() or values.get(PHASE_ENV, "deployment")
    shim = str(root / "bin")
    path = [
        part
        for part in values.get("PATH", os.defpath).split(os.pathsep)
        if part != shim
    ]
    values["PATH"] = os.pathsep.join([shim, *path])
    return values


def resolve_tool(name: str, path: str | None = None) -> str | None:
    root = budget_root()
    if root is None and path is None:
        return shutil.which(name)
    parts = (path or os.environ.get("PATH", os.defpath)).split(os.pathsep)
    if root:
        parts = [part for part in parts if part != str(root / "bin")]
    return shutil.which(name, path=os.pathsep.join(parts))


@contextlib.contextmanager
def api_slot(
    backend: str, *, weight: int = 1, parent_id: str | None = None
) -> Iterator[str | None]:
    """Admit work, optionally borrowing a verified same-backend CLI reservation.

    Only one helper branch may run per stopped parent. Each branch retains the
    maximum ancestor weight, so returning from an admitted helper never needs
    extra capacity. A heavier helper first releases its stopped parent's weight
    and competes for global capacity; cancellation kills the stopped parent if
    its reservation cannot be restored. Other backends on the ancestry chain
    stay independently charged. STOP bounds runnable CLI work, not already-issued
    HTTP requests.
    """
    _remaining()
    root = budget_root()
    if root is None:
        yield None
        return
    if backend not in LIMITS or not 1 <= weight <= LIMITS[backend]:
        raise ApiBudgetError("invalid deployment API capacity request")
    phase = _label(_PHASE.get() or os.environ.get(PHASE_ENV, "deployment"))
    identifier = uuid.uuid4().hex
    pid = os.getpid()
    pid_start = _pid_identity(pid)
    if not pid_start:
        raise ApiBudgetError(
            "deployment API admission cannot establish process identity"
        )
    started = time.monotonic()
    deadline = started + _remaining()
    registered = False
    try:
        while True:
            remaining = min(_remaining(), deadline - time.monotonic())
            if remaining <= 0:
                raise ApiBudgetError("deployment API admission deadline expired")
            _recover_slots(root, timeout=min(10, remaining))
            with _database(root, timeout=min(10, remaining)) as database:
                database.execute("BEGIN IMMEDIATE")
                parent = (
                    _parent_lease(database, parent_id, pid)
                    if parent_id is not None
                    else None
                )
                lender = _capacity_parent(database, parent, backend, pid)
                if not registered:
                    database.execute(
                        "INSERT INTO leases(id,backend,weight,pid,pid_start,"
                        "parent_id,depth,state) VALUES(?,?,?,?,?,?,?,'waiting')",
                        (
                            identifier,
                            backend,
                            weight,
                            pid,
                            pid_start,
                            parent_id,
                            parent.depth + 1 if parent else 0,
                        ),
                    )
                    registered = True
                ready = parent is None or parent.command_pid is not None
                if ready and lender is not None:
                    if lender.command_pid is None:
                        ready = False
                    elif lender.state == "active" and lender.lent_to is None:
                        weight = _lend_slot(database, lender, identifier, weight)
                    elif lender.state != "parked" or lender.lent_to != identifier:
                        ready = False
                occupied = _occupied(database, backend)
                next_start = database.execute(
                    "SELECT next_start FROM capacity WHERE backend=?", (backend,)
                ).fetchone()[0]
                now = time.monotonic()
                if now >= deadline:
                    raise ApiBudgetError("deployment API admission deadline expired")
                _remaining()
                if ready and occupied + weight <= LIMITS[backend] and now >= next_start:
                    database.execute(
                        "UPDATE leases SET state='active' WHERE id=?", (identifier,)
                    )
                    database.execute(
                        "UPDATE capacity SET next_start=?,peak=MAX(peak,?) WHERE backend=?",
                        (now + START_INTERVAL[backend], occupied + weight, backend),
                    )
                    database.execute(
                        "INSERT INTO calls(id,phase,backend,wait_seconds) VALUES(?,?,?,?)",
                        (identifier, phase, backend, now - started),
                    )
                    database.commit()
                    break
                database.commit()
            time.sleep(min(0.05, remaining))
        yield identifier
    finally:
        if registered:
            _finish_slot(root, identifier)


def statistics(phase: str | None = None) -> dict[str, object]:
    root = budget_root()
    if root is None:
        return {"enabled": False}
    where, arguments = (" WHERE phase=?", (phase,)) if phase else ("", ())
    with _database(root) as database:
        rows = database.execute(
            "SELECT backend,COUNT(*),SUM(wait_seconds),SUM(sdk_calls),SUM(sdk_attempts),"
            "SUM(attempt_records),SUM(sdk_retries),SUM(finished),SUM(telemetry_expected)"
            " FROM calls" + where + " GROUP BY backend",
            arguments,
        ).fetchall()
        peak = dict(database.execute("SELECT backend,peak FROM capacity"))
    return {
        "enabled": True,
        "limits": dict(LIMITS),
        "scope": "deployment-and-descendant-api-commands",
        "peak_admitted_weight": peak,
        "backends": {
            backend: {
                "commands": commands,
                "admission_wait_seconds": round(wait_seconds, 3),
                "sdk_calls": calls,
                "sdk_attempts": attempts,
                "sdk_attempt_records": records,
                "sdk_retries": retries,
                "finished_commands": finished,
                "telemetry_expected_commands": expected,
            }
            for backend, commands, wait_seconds, calls, attempts, records, retries, finished, expected in rows
        },
        # CSM is UDP and can lose packets. Never infer safe concurrency or
        # application health from these counters, or label silence as zero retries.
        "sdk_accounting": "observed-csm-lower-bound",
        "http_rate_limit": False,
    }


def _record_csm(root: Path, payload: bytes) -> None:
    try:
        value = json.loads(payload)
        if not isinstance(value, dict) or value.get("Version") != 1:
            return
        identifier = value.get("ClientId")
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[0-9a-f]{32}", identifier
        ):
            return
        kind = value.get("Type")
        if kind == "ApiCall":
            count = value.get("AttemptCount")
            if type(count) is not int or not 0 <= count <= 100:
                return
            changes = (1, count, 0, max(0, count - 1))
        elif kind == "ApiCallAttempt":
            changes = (0, 0, 1, 0)
        else:
            return
        # Whitelist counters only. CSM payloads may carry signing credentials;
        # none of their strings, headers, URLs or errors are persisted or logged.
        with _database(root) as database:
            database.execute(
                "UPDATE calls SET sdk_calls=sdk_calls+?,sdk_attempts=sdk_attempts+?,"
                "attempt_records=attempt_records+?,sdk_retries=sdk_retries+? WHERE id=?",
                (*changes, identifier),
            )
            database.commit()
    except (ValueError, TypeError, sqlite3.Error, OSError, ApiBudgetError):
        return


def _collect(root: Path, listener: socket.socket, stopped: threading.Event) -> None:
    while not stopped.is_set():
        try:
            payload, _peer = listener.recvfrom(65536)
        except TimeoutError:
            continue
        except OSError:
            return
        _record_csm(root, payload)


@contextlib.contextmanager
def deployment_api_budget() -> Iterator[None]:
    if budget_root() is not None:
        yield
        return
    old = {name: os.environ.get(name) for name in (ROOT_ENV, PHASE_ENV, "PATH")}
    with tempfile.TemporaryDirectory(prefix="gpu-fault-api-") as directory:
        root = Path(directory)
        path = root / "budget.sqlite3"
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        with _database(root) as database:
            database.executescript(
                "CREATE TABLE capacity(backend TEXT PRIMARY KEY,next_start REAL,"
                "peak INTEGER,max_weight INTEGER NOT NULL);"
                "CREATE TABLE leases(id TEXT PRIMARY KEY,backend TEXT,weight INTEGER,pid INTEGER,pid_start TEXT,"
                "command_pid INTEGER,command_start TEXT,parent_id TEXT,depth INTEGER DEFAULT 0,"
                "state TEXT DEFAULT 'active',lent_to TEXT);"
                "CREATE TABLE calls(id TEXT PRIMARY KEY,phase TEXT,backend TEXT,wait_seconds REAL,"
                "finished INTEGER DEFAULT 0,sdk_calls INTEGER DEFAULT 0,sdk_attempts INTEGER DEFAULT 0,"
                "attempt_records INTEGER DEFAULT 0,sdk_retries INTEGER DEFAULT 0,telemetry_expected INTEGER DEFAULT 0);"
            )
            database.executemany(
                "INSERT INTO capacity VALUES(?,0,0,?)", list(LIMITS.items())
            )
            database.execute(f"PRAGMA user_version={BUDGET_PROTOCOL_VERSION}")
            database.commit()
        shim = root / "bin"
        shim.mkdir(mode=0o700)
        for name in ("aws", "kubectl"):
            entry = shim / name
            entry.write_text(
                "#!/bin/sh\nexec "
                + shlex.quote(sys.executable)
                + " -I "
                + shlex.quote(str(Path(__file__).resolve()))
                + ' "$0" "$@"\n'
            )
            entry.chmod(0o700)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.settimeout(0.1)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            (root / "csm-port").write_text(str(listener.getsockname()[1]))
            (root / "csm-port").chmod(0o600)
            stopped = threading.Event()
            thread = threading.Thread(
                target=_collect, args=(root, listener, stopped), daemon=True
            )
            thread.start()
            entered = False
            try:
                os.environ[ROOT_ENV] = str(root)
                os.environ.update(api_environment(None) or {})
                entered = True
                yield
            finally:
                # All command owners join their children before leaving the
                # command scope. Drain telemetry briefly, then stop the listener.
                time.sleep(0.1)
                stopped.set()
                thread.join(timeout=2)
                try:
                    if entered:
                        try:
                            with cleanup_deadline("deployment API accounting", 2):
                                accounting = json.dumps(statistics(), sort_keys=True)
                        except Exception:
                            accounting = "unavailable"
                        print(
                            "deployment API accounting: " + accounting, file=sys.stderr
                        )
                finally:
                    for name, value in old.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value


def _transfer_configuration(root: Path, environment: dict[str, str]) -> None:
    source = Path(environment.get("AWS_CONFIG_FILE") or Path.home() / ".aws/config")
    content = source.read_text() if source.is_file() else ""
    parser = configparser.RawConfigParser()
    try:
        parser.read_string(content)
        if not parser.has_section("default"):
            parser.add_section("default")
        for name in parser.sections():
            if name == "default" or name.startswith("profile "):
                values = configparser.RawConfigParser()
                values.read_string("[s3]\n" + parser.get(name, "s3", fallback=""))
                values["s3"]["max_concurrent_requests"] = str(TRANSFER_CONCURRENCY)
                values["s3"]["preferred_transfer_client"] = "classic"
                parser[name]["s3"] = "\n" + "\n".join(
                    f"{key} = {value}" for key, value in values["s3"].items()
                )
    except configparser.Error:
        raise ApiBudgetError("cannot safely bound AWS transfer concurrency") from None
    target = root / ("aws-config-" + hashlib.sha256(content.encode()).hexdigest())
    temporary = root / ("aws-config-tmp-" + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        parser.write(stream)
    os.replace(temporary, target)
    environment["AWS_CONFIG_FILE"] = str(target)


@contextlib.contextmanager
def _command_output(
    root: Path, nested: bool
) -> Iterator[tuple[BinaryIO | None, BinaryIO | None]]:
    if not nested:
        yield None, None
        return
    # A stopped credential_process caller cannot drain pipes. Anonymous 0600
    # files avoid both pipe deadlock and unbounded memory without persisting CLI
    # output in the ledger or leaving named credential-bearing artifacts.
    with (
        tempfile.TemporaryFile(dir=root) as stdout,
        tempfile.TemporaryFile(dir=root) as stderr,
    ):
        yield stdout, stderr
        # This context must outlive api_slot: restore the caller before copying
        # any output back to it. No API capacity is held while output drains.
        stdout.seek(0)
        stderr.seek(0)
        shutil.copyfileobj(stdout, sys.stdout.buffer)
        shutil.copyfileobj(stderr, sys.stderr.buffer)


def _drain_command_output(
    root: Path, process: subprocess.Popen[bytes], stdout: BinaryIO, stderr: BinaryIO
) -> int:
    if process.stdout is None or process.stderr is None:
        raise ApiBudgetError("nested API command output pipes are missing")
    deadline = time.monotonic() + _remaining()
    next_recovery = time.monotonic()
    spooled = 0
    destinations = {
        process.stdout.fileno(): stdout,
        process.stderr.fileno(): stderr,
    }
    # Start draining only after Popen/preexec binding. No threads or shared
    # resource limits are needed, and the CLI cannot write the spool files.
    with selectors.DefaultSelector() as selector:
        for descriptor in destinations:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        while selector.get_map():
            remaining = min(_remaining(), deadline - time.monotonic())
            if remaining <= 0:
                raise ApiBudgetError("nested API command output deadline expired")
            if time.monotonic() >= next_recovery:
                _recover_slots(root, timeout=min(10, remaining))
                next_recovery = time.monotonic() + 0.1
            for event, _mask in selector.select(timeout=min(0.1, remaining)):
                try:
                    chunk = os.read(event.fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(event.fd)
                    continue
                if spooled + len(chunk) > MAX_NESTED_OUTPUT_BYTES:
                    raise ApiBudgetError("nested API command output exceeds its limit")
                destinations[event.fd].write(chunk)
                spooled += len(chunk)
    remaining = min(_remaining(), deadline - time.monotonic())
    if remaining <= 0:
        raise ApiBudgetError("nested API command output deadline expired")
    return process.wait(timeout=remaining)


def _wait_command(root: Path, process: subprocess.Popen[bytes]) -> int:
    while True:
        remaining = _remaining()
        try:
            return process.wait(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            # A surviving caller shim can reclaim a dead helper's loan even
            # when no other command is currently trying to acquire capacity.
            _recover_slots(root, timeout=min(10, _remaining()))


def main() -> int:
    arguments = sys.argv[1:]
    if not arguments or Path(arguments[0]).name not in {"aws", "kubectl"}:
        raise ApiBudgetError("API shim requires an AWS or kubectl command")
    backend = Path(arguments[0]).name
    root = budget_root()
    if root is None:
        raise ApiBudgetError("API shim lost its deployment scope")
    executable = resolve_tool(backend)
    if not executable:
        raise ApiBudgetError(f"deployment tool is missing: {backend}")

    def interrupted(_signum: int, _frame: object) -> None:
        raise ApiBudgetError("deployment API command was interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    environment = dict(os.environ)
    transfer = backend == "aws" and "s3" in arguments[1:]
    if transfer:
        _transfer_configuration(root, environment)
    with (
        _command_output(root, PARENT_ENV in environment) as (stdout, stderr),
        api_slot(
            backend,
            weight=TRANSFER_CONCURRENCY if transfer else 1,
            parent_id=environment.get(PARENT_ENV),
        ) as identifier,
    ):
        if identifier is None:
            raise ApiBudgetError("API shim lost its deployment scope")
        environment[PARENT_ENV] = identifier
        if backend == "aws":
            environment.update(
                AWS_CSM_ENABLED="true",
                AWS_CSM_HOST="127.0.0.1",
                AWS_CSM_PORT=(root / "csm-port").read_text().strip(),
                AWS_CSM_CLIENT_ID=str(identifier),
            )
            with _database(root) as database:
                database.execute(
                    "UPDATE calls SET telemetry_expected=? WHERE id=?",
                    (
                        int("--version" not in arguments and "help" not in arguments),
                        identifier,
                    ),
                )
                database.commit()
        command = [executable, *arguments[1:]]

        def bind_command() -> None:
            _bind_command(root, identifier)

        with subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE if stdout is not None else None,
            stderr=subprocess.PIPE if stderr is not None else None,
            start_new_session=True,
            # This isolated stdlib shim is single-threaded. Bind in the forked
            # child before exec so shim death cannot leave unaccounted CLI work.
            preexec_fn=bind_command,
        ) as process:
            try:
                if stdout is not None and stderr is not None:
                    return _drain_command_output(root, process, stdout, stderr)
                return _wait_command(root, process)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise


def run_shim() -> int:
    """The shim's process entry: ``main`` plus the exit-status contract.

    The generic line is what callers grep for; the class and text of the
    underlying error follow it, because a release that fails on this shim has
    nothing else to diagnose from (live 2026-09-18: two identity-check kubectl
    calls failed with the bare line and the cause was unrecoverable).
    """

    try:
        return main()
    except ApiBudgetProtocolError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (
        ApiBudgetError,
        OSError,
        sqlite3.Error,
        subprocess.SubprocessError,
    ) as error:
        print(
            "deployment API command failed or exceeded its bounded capacity/deadline: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(run_shim())
