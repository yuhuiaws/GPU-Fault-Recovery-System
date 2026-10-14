"""The store's activity while a capacity run's load and drain lasted.

A formal record must state the ``pg_stat_activity`` sampler summary of the
first two storm minutes (performance acceptance plan §11: active statement
samples, the ``LWLock:BufferContent`` share, peak concurrent admission
INSERTs, the autovacuum share). That sampler ran outside the repository and
only beside the integrated rounds, so the burst rounds -- where the start-burst
storm happens -- recorded none. The burst suite now runs it itself, the way it
samples the processor (``regional_capacity_timeseries``): one ``kubectl exec
-i`` session in an api-ha Pod whose program arrives length-prefixed on stdin,
the rest of stdin being its stop channel. Every ``SAMPLE_INTERVAL_SECONDS`` the
program reads the database's active client backends and autovacuum workers
through the store's own DSN and prints one compact JSON line: the active
statements by wait event, how many of them are admission INSERTs into the
processor queue, the autovacuum workers, and the most frequent statements
(literals and numbers masked, 80 characters). ``pg-activity.json`` keeps at
most ``SAMPLE_LIMIT`` samples; :func:`activity_summary` reduces them for the
whole window and for the load's first two minutes.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from . import regional_capacity_registry as registry
    from .regional_capacity_timeseries import (
        BOOTSTRAP,
        PodSession,
        open_session,
        stop_sessions,
    )
else:
    import regional_capacity_registry as registry
    from regional_capacity_timeseries import (
        BOOTSTRAP,
        PodSession,
        open_session,
        stop_sessions,
    )

SAMPLE_INTERVAL_SECONDS = 0.5
#: One hour at two samples per second: a bound on the artifact, not a window.
SAMPLE_LIMIT = 7200
#: Statements kept per sample, the most frequent first.
TOP_STATEMENTS = 5
#: The storm a formal record reads: the load's first two minutes.
STORM_SECONDS = 120.0
BUFFER_CONTENT = "LWLock:BufferContent"
#: Runs in the Pod; ``{STORE_DSN_SNIPPET}`` is substituted by ``str.replace``.
ACTIVITY_PROGRAM = r"""
import json, os, re, sys, threading, time
import psycopg
{STORE_DSN_SNIPPET}
interval = float(os.environ["SAMPLE_INTERVAL"])
limit = int(os.environ["SAMPLE_LIMIT"])
top = int(os.environ["SAMPLE_TOP"])
stop = threading.Event()
def wait_for_eof():
    try:
        while sys.stdin.buffer.read(65536):
            pass
    finally:
        stop.set()
threading.Thread(target=wait_for_eof, daemon=True).start()
QUERY = (
    "SELECT backend_type, coalesce(wait_event_type, 'CPU'), coalesce(wait_event, ''), "
    "left(regexp_replace(query, '\\s+', ' ', 'g'), 200) FROM pg_stat_activity "
    "WHERE datname = current_database() AND state = 'active' "
    "AND pid <> pg_backend_pid() "
    "AND backend_type IN ('client backend', 'autovacuum worker')"
)
ADMISSION = re.compile(r"^INSERT INTO gpu_fault_processor_queue \(", re.IGNORECASE)
def shape(statement):
    statement = re.sub(r"'[^']*'", "'?'", statement.strip())
    return re.sub(r"\b\d+\b", "N", statement)[:80]
taken = 0
with psycopg.connect(
    store_dsn(), autocommit=True, application_name="gpu-fault-perf-activity"
) as connection:
    while not stop.is_set() and taken < limit:
        started = time.time()
        try:
            rows = connection.execute(QUERY).fetchall()
        except Exception as exc:
            rows = None
            print(json.dumps({"t": round(started, 3), "error": type(exc).__name__}), flush=True)
        if rows is not None:
            waits, statements, inserts, autovacuum = {}, {}, 0, 0
            for backend, wait_type, wait, statement in rows:
                if backend == "autovacuum worker":
                    autovacuum += 1
                    continue
                key = wait_type + ":" + wait if wait else wait_type
                waits[key] = waits.get(key, 0) + 1
                text = shape(statement or "")
                statements[text] = statements.get(text, 0) + 1
                inserts += bool(ADMISSION.match(text))
            ranked = sorted(statements.items(), key=lambda item: (-item[1], item[0]))
            print(json.dumps({
                "t": round(started, 3),
                "active": sum(waits.values()),
                "waits": waits,
                "admission_inserts": inserts,
                "autovacuum": autovacuum,
                "statements": dict(ranked[:top]),
            }), flush=True)
        taken += 1
        stop.wait(max(0.0, interval - (time.time() - started)))
"""


def program() -> str:
    return ACTIVITY_PROGRAM.replace("{STORE_DSN_SNIPPET}", registry.STORE_DSN_SNIPPET)


def _reduce(rows: list[dict[str, Any]]) -> dict[str, Any]:
    waits: Counter[str] = Counter()
    statements: Counter[str] = Counter()
    for row in rows:
        waits.update(row.get("waits") or {})
        statements.update(row.get("statements") or {})
    active = sum(waits.values())
    with_autovacuum = sum(1 for row in rows if row.get("autovacuum"))
    return {
        "samples": len(rows),
        "active_statement_samples": active,
        "mean_active": round(active / len(rows), 2) if rows else None,
        "peak_active": max((int(row.get("active") or 0) for row in rows), default=None),
        "wait_shares": {
            name: round(count / active, 3) for name, count in waits.most_common(8)
        },
        "buffer_content_share": (
            round(waits[BUFFER_CONTENT] / active, 3) if active else None
        ),
        "peak_admission_inserts": max(
            (int(row.get("admission_inserts") or 0) for row in rows), default=None
        ),
        "samples_with_autovacuum": with_autovacuum,
        "autovacuum_sample_share": (
            round(with_autovacuum / len(rows), 3) if rows else None
        ),
        "top_statements": dict(statements.most_common(8)),
    }


def activity_summary(
    samples: list[dict[str, Any]],
    *,
    load_start: float | None,
    storm_seconds: float = STORM_SECONDS,
) -> dict[str, Any]:
    """The whole sampled window and the load's first ``storm_seconds``."""

    rows = [item for item in samples if "active" in item]
    storm = None
    if load_start is not None:
        storm = _reduce(
            [
                row
                for row in rows
                if load_start <= float(row["t"]) < load_start + storm_seconds
            ]
        )
    return {
        "errors": len(samples) - len(rows),
        "window": _reduce(rows),
        "first_two_minutes": storm,
    }


class PgActivity:
    """The run's sampling session in one api-ha Pod, from :meth:`start` to
    :meth:`stop`; without an api-ha Pod it records why it sampled nothing."""

    def __init__(
        self,
        pods: list[str],
        *,
        interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
        limit: int = SAMPLE_LIMIT,
        top: int = TOP_STATEMENTS,
        popen: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.pod = next((pod for pod in pods if "api-ha" in pod), None)
        self.interval_seconds = interval_seconds
        self.limit = limit
        self.top = top
        self.popen = popen
        self.session: PodSession | None = None
        self.failed = None if self.pod else "no api-ha Pod"

    def _argv(self) -> list[str]:
        return [
            "kubectl",
            "--kubeconfig",
            registry.CONTROL_KUBECONFIG,
            "-n",
            registry.CONTROL_NAMESPACE,
            "exec",
            "-i",
            str(self.pod),
            "--",
            "env",
            f"SAMPLE_INTERVAL={self.interval_seconds:g}",
            f"SAMPLE_LIMIT={self.limit}",
            f"SAMPLE_TOP={self.top}",
            "python3",
            "-c",
            BOOTSTRAP,
        ]

    def start(self) -> PgActivity:
        if self.pod is None:
            return self
        try:
            self.session = open_session(
                self.pod, self._argv(), program(), limit=self.limit, popen=self.popen
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.failed = f"{type(exc).__name__}: {exc}"
        return self

    def stop(self) -> None:
        if self.session is not None:
            stop_sessions([self.session])

    def _samples(self) -> list[dict[str, Any]]:
        return self.session.samples if self.session is not None else []

    def _error(self) -> str | None:
        return self.failed or (self.session.error if self.session else None)

    def document(self) -> dict[str, Any]:
        return {
            "pod": self.pod,
            "interval_seconds": self.interval_seconds,
            "limit": self.limit,
            "top_statements_per_sample": self.top,
            "samples": self._samples(),
            "truncated": bool(self.session and self.session.truncated),
            "error": self._error(),
        }

    def summary(self, load_start: float | None = None) -> dict[str, Any]:
        return {
            "pod": self.pod,
            "interval_seconds": self.interval_seconds,
            "error": self._error(),
            **activity_summary(self._samples(), load_start=load_start),
        }
