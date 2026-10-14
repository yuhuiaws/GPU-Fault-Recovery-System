"""Put the Aurora writer back after a case's own failover (HA-003, HA-010).

``failover-db-cluster`` leaves the writer on whichever member RDS promoted, and
nothing moves it back: live 2026-10-13 HA-010's failover left it in us-west-2b
under a control plane entirely in us-west-2a (+~1.2 ms per database round trip,
burst admission p99 x3) until an operator failed back by hand. Every case that
requests a failover restores the pre-case writer in its cleanup through
``restore_writer``: wait for RDS to settle (available, one writer), wait out the
quiet rule of the injection (no remote command open, empty fault queue;
bounded), ``failover-db-cluster --target-db-instance-identifier <pre-case
writer>``, wait for that writer, and record ``writer-restore.json``. A restore
that is refused, times out or lands on another member raises, so the case
fails; ``gpu-fault-admin aurora-failback`` is the operator's exit.

The RDS calls are HA-003's (``run_ha003_aurora_failover_reset.writer_restore_rds``
binds them to one cluster); the clocks are the caller's, so a runner's test
clock drives every wait here.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import (
    processor_queue_backlog,
    write_json_atomic,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError

EVIDENCE_FILE = "writer-restore.json"
QUIET_EVIDENCE_FILE = "store-before-restore.json"
# How long the restore waits for the case's failover to settle and for its own
# failover, and how long for a quiet control plane before requesting it.
RESTORE_TIMEOUT_SECONDS = 900
RESTORE_QUIET_WAIT_SECONDS = 300
RESTORE_QUIET_POLL_SECONDS = 15.0
RDS_POLL_SECONDS = 5.0
RESTORE_REMEDY = (
    "fail the writer back with gpu-fault-admin aurora-failback --state-dir "
    "<state-dir> --reference <change> --confirm AURORA_FAILBACK"
)


@dataclass(frozen=True)
class WriterRestoreRds:
    """The RDS operations a restore needs, bound to one cluster.

    ``snapshot`` is HA-003's ``rds_snapshot`` (``status``, ``writer``);
    ``fail_over_to`` requests ``failover-db-cluster`` with a named target;
    ``wait_writer_left`` is HA-003's ``wait_rds_failover`` from a given writer,
    returning the final snapshot and the samples it took.
    """

    snapshot: Callable[[], dict[str, Any]]
    fail_over_to: Callable[[str], dict[str, Any]]
    wait_writer_left: Callable[[str], tuple[dict[str, Any], list[dict[str, Any]]]]


def placement_restore_errors(
    record: dict[str, Any],
    *,
    original_writer: str,
) -> list[str]:
    """The writer is back on the instance that wrote before the case.

    ``record`` is the restore record: ``target``, ``action`` (``none`` when the
    writer never left, ``failover`` when it was failed back) and ``rds_after``,
    the RDS snapshot the restore ended on (``status``, ``writer``).
    """

    errors: list[str] = []
    if not original_writer:
        return ["the pre-case writer is unknown, so placement cannot be restored"]
    if record.get("target") != original_writer:
        errors.append(
            f"the restore targeted {record.get('target')!r}, not the pre-case "
            f"writer {original_writer}"
        )
    rds = record.get("rds_after") or {}
    if rds.get("status") != "available":
        errors.append(
            f"Aurora is {rds.get('status')!r} after the writer restore, not available"
        )
    if rds.get("writer") != original_writer:
        errors.append(
            f"the writer is {rds.get('writer')!r} after the restore, not the "
            f"pre-case writer {original_writer}; {RESTORE_REMEDY}"
        )
    return errors


def store_quiet_errors(state: dict[str, Any]) -> list[str]:
    """Why a store snapshot is not quiet enough for a failover, if it is not."""

    errors: list[str] = []
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote commands are open")
    if processor_queue_backlog(state.get("queue") or {}):
        errors.append("the processor fault queue holds work")
    return errors


def wait_settled(
    rds: WriterRestoreRds,
    *,
    timeout_seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    """Poll RDS until it is ``available`` with one writer."""

    deadline = monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while monotonic() < deadline:
        last = rds.snapshot()
        if last.get("status") == "available" and last.get("writer"):
            return last
        sleep(RDS_POLL_SECONDS)
    raise RegionalFixtureError(
        f"Aurora did not settle before the writer restore: {last}"
    )


def await_quiet(
    read_state: Callable[[], dict[str, Any]],
    *,
    evidence: Path,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> None:
    """The injection's quiet rule, waited out for a bounded time."""

    deadline = monotonic() + RESTORE_QUIET_WAIT_SECONDS
    while True:
        state = read_state()
        write_json_atomic(evidence, state)
        errors = store_quiet_errors(state)
        if not errors:
            return
        if monotonic() >= deadline:
            raise RegionalFixtureError(
                f"writer restore refused: {'; '.join(errors)}; {RESTORE_REMEDY} "
                "once the site is idle"
            )
        sleep(RESTORE_QUIET_POLL_SECONDS)


def restore_writer(
    rds: WriterRestoreRds,
    *,
    original_writer: str,
    case_dir: Path,
    read_state: Callable[[], dict[str, Any]],
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Fail the writer back to ``original_writer``; record it whatever happens.

    A writer that never left is recorded as ``action: none``. Raises when the
    placement was not restored.
    """

    record: dict[str, Any] = {"target": original_writer, "action": "none"}
    try:
        settled = wait_settled(
            rds,
            timeout_seconds=RESTORE_TIMEOUT_SECONDS,
            sleep=sleep,
            monotonic=monotonic,
        )
        record["writer_before_restore"] = settled.get("writer")
        if original_writer and settled.get("writer") != original_writer:
            await_quiet(
                read_state,
                evidence=case_dir / QUIET_EVIDENCE_FILE,
                sleep=sleep,
                monotonic=monotonic,
            )
            record.update(action="failover", requested_at=now().isoformat())
            record["request"] = rds.fail_over_to(original_writer)
            settled, record["samples"] = rds.wait_writer_left(
                str(settled.get("writer"))
            )
            record["restored_at"] = now().isoformat()
        record["rds_after"] = settled
        record["errors"] = placement_restore_errors(
            record, original_writer=original_writer
        )
    except Exception as exc:
        record["errors"] = [f"{type(exc).__name__}: {exc}"]
        raise
    finally:
        write_json_atomic(case_dir / EVIDENCE_FILE, record)
    if record["errors"]:
        raise RegionalFixtureError("writer restore: " + "; ".join(record["errors"]))
    return record
