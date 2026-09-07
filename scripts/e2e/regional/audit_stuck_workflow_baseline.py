"""Read-only census of stuck workflows and zombie processor leases.

Three questions, each answered by one query against the control-plane Postgres,
none of which writes anything:

* **Q-ORPHAN** -- workflows still ``PENDING`` / ``SAFETY_PENDING`` whose incident
  points at a *different* workflow and which no other workflow names as its
  predecessor. Nothing dispatches, fences or sweeps such a record; it is the
  "aggregated but never handled" orphan. The same-generation subset is the one
  the abandoned-generation sweep cannot close, so it is counted separately.
* **Q-ZOMBIE** -- processor queue rows still ``LEASED`` after their lease expired,
  i.e. work a crashed process never released.
* **Q-STUCK-PENDING** -- ``PENDING`` workflows whose ``not_before`` has passed,
  whose predecessor (if any) is terminal *the way the dispatcher reads terminal*
  -- SUCCEEDED / FAILED / SUPERSEDED, a BLOCKED row whose safety plan settled
  (``blocked_kind`` not NEEDS_OPERATOR / INTERNAL_ERROR), or a predecessor row
  that is gone -- and which nothing has touched for the idle window. Every one
  of them should have been dispatched.

The script is meant to run either where ``GPU_FAULT_STORE_URL`` is available or
piped into an API Pod (``kubectl exec -i <pod> -- python - --json``), which is why
it depends on nothing but the standard library and psycopg. Every session opens
its one transaction with ``SET TRANSACTION READ ONLY`` -- a session-level
``SET default_transaction_read_only`` issued *inside* an already-open
transaction only takes effect for the *next* one, which is why the earlier
version guarded nothing -- sets a statement timeout and ends with a rollback.
Timestamps inside ``payload`` are compared as ISO-8601 text on purpose:
the stored values come from ``isoformat()``, whose text order equals time order,
and a ``::timestamptz`` cast would bypass the expression indexes.

The store URL is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg

ORPHAN_SQL = """
SELECT count(*),
       count(*) FILTER (
           WHERE w.payload->>'fencing_token' = i.payload->>'fencing_token'
       )
FROM gpu_fault_objects w
JOIN gpu_fault_objects i
  ON i.kind = 'incident' AND i.key = w.payload->>'incident_id'
WHERE w.kind = 'workflow'
  AND w.payload->>'status' IN ('PENDING', 'SAFETY_PENDING')
  AND i.payload->>'workflow_request_id' <> w.key
  AND NOT EXISTS (
      SELECT 1 FROM gpu_fault_objects s
      WHERE s.kind = 'workflow'
        AND s.payload->>'predecessor_workflow_id' = w.key
  )
  AND w.payload->>'created_at' < %(cutoff)s
"""

ZOMBIE_SQL = """
SELECT count(*) FROM gpu_fault_processor_queue
WHERE status = 'LEASED' AND lease_expires_at < now()
"""

STUCK_PENDING_SQL = """
SELECT count(*)
FROM gpu_fault_objects w
WHERE w.kind = 'workflow'
  AND w.payload->>'status' = 'PENDING'
  AND (w.payload->>'not_before' IS NULL OR w.payload->>'not_before' < %(now)s)
  AND w.payload->>'updated_at' < %(idle_cutoff)s
  AND (
      w.payload->>'predecessor_workflow_id' IS NULL
      OR NOT EXISTS (
          SELECT 1 FROM gpu_fault_objects gone
          WHERE gone.kind = 'workflow'
            AND gone.key = w.payload->>'predecessor_workflow_id'
      )
      OR EXISTS (
          SELECT 1 FROM gpu_fault_objects p
          WHERE p.kind = 'workflow'
            AND p.key = w.payload->>'predecessor_workflow_id'
            AND p.payload->>'status' NOT IN ('PENDING', 'SAFETY_PENDING', 'RUNNING')
            AND NOT (
                p.payload->>'status' = 'BLOCKED'
                AND p.payload->>'blocked_kind' IN ('NEEDS_OPERATOR', 'INTERNAL_ERROR')
            )
      )
  )
"""

# The first statement of the audit's only transaction. ``SET TRANSACTION`` has
# to come before any query in that transaction; psycopg opens the transaction
# on the first ``execute``, so executing this first is what makes it apply.
READ_ONLY_SQL = "SET TRANSACTION READ ONLY"

DEFAULT_ORPHAN_CUTOFF_SECONDS = 600
DEFAULT_STUCK_IDLE_SECONDS = 600
DEFAULT_STATEMENT_TIMEOUT_SECONDS = 120


def isoformat(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def orphan_counts(
    cursor: psycopg.Cursor[Any], *, now: datetime, cutoff_seconds: int
) -> dict[str, int]:
    cutoff = isoformat(now - timedelta(seconds=cutoff_seconds))
    cursor.execute(ORPHAN_SQL, {"cutoff": cutoff})
    row = cursor.fetchone()
    assert row is not None
    return {"total": int(row[0]), "same_generation": int(row[1])}


def zombie_count(cursor: psycopg.Cursor[Any]) -> int:
    cursor.execute(ZOMBIE_SQL)
    row = cursor.fetchone()
    assert row is not None
    return int(row[0])


def stuck_pending_count(
    cursor: psycopg.Cursor[Any], *, now: datetime, idle_seconds: int
) -> int:
    cursor.execute(
        STUCK_PENDING_SQL,
        {
            "now": isoformat(now),
            "idle_cutoff": isoformat(now - timedelta(seconds=idle_seconds)),
        },
    )
    row = cursor.fetchone()
    assert row is not None
    return int(row[0])


def collect(
    cursor: psycopg.Cursor[Any],
    *,
    now: datetime,
    orphan_cutoff_seconds: int = DEFAULT_ORPHAN_CUTOFF_SECONDS,
    stuck_idle_seconds: int = DEFAULT_STUCK_IDLE_SECONDS,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "observed_at": isoformat(now),
        "parameters": {
            "orphan_cutoff_seconds": orphan_cutoff_seconds,
            "stuck_idle_seconds": stuck_idle_seconds,
        },
        "orphan": orphan_counts(cursor, now=now, cutoff_seconds=orphan_cutoff_seconds),
        "zombie": zombie_count(cursor),
        "stuck_pending": stuck_pending_count(
            cursor, now=now, idle_seconds=stuck_idle_seconds
        ),
    }


def run(
    store_url: str,
    *,
    now: datetime | None = None,
    orphan_cutoff_seconds: int = DEFAULT_ORPHAN_CUTOFF_SECONDS,
    stuck_idle_seconds: int = DEFAULT_STUCK_IDLE_SECONDS,
    statement_timeout_seconds: int = DEFAULT_STATEMENT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    moment = now or datetime.now(timezone.utc)
    with psycopg.connect(store_url) as connection:
        with connection.cursor() as cursor:
            # Must be the transaction's first statement (see READ_ONLY_SQL).
            cursor.execute(READ_ONLY_SQL)
            # SET takes no bind parameters; the value is an int we own.
            cursor.execute(
                f"SET statement_timeout = '{int(statement_timeout_seconds)}s'"
            )
            try:
                return collect(
                    cursor,
                    now=moment,
                    orphan_cutoff_seconds=orphan_cutoff_seconds,
                    stuck_idle_seconds=stuck_idle_seconds,
                )
            finally:
                connection.rollback()


def nonzero(result: dict[str, Any]) -> bool:
    return bool(
        result["orphan"]["total"] or result["zombie"] or result["stuck_pending"]
    )


def markdown_row(result: dict[str, Any], *, note: str = "") -> str:
    day = str(result["observed_at"])[:10]
    orphan = result["orphan"]
    return (
        f"| {day} | {orphan['total']} / {orphan['same_generation']} "
        f"| {result['zombie']} | {result['stuck_pending']} | {note} |"
    )


def summary_lines(result: dict[str, Any]) -> list[str]:
    orphan = result["orphan"]
    return [
        f"observed_at      {result['observed_at']}",
        f"Q-ORPHAN         {orphan['total']} "
        f"(same generation: {orphan['same_generation']})",
        f"Q-ZOMBIE         {result['zombie']}",
        f"Q-STUCK-PENDING  {result['stuck_pending']}",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--store-url",
        default=None,
        help="Postgres URL; defaults to $GPU_FAULT_STORE_URL. Never printed.",
    )
    parser.add_argument(
        "--orphan-cutoff-seconds",
        type=int,
        default=DEFAULT_ORPHAN_CUTOFF_SECONDS,
        help="ignore PENDING records younger than this (aggregation window)",
    )
    parser.add_argument(
        "--stuck-idle-seconds",
        type=int,
        default=DEFAULT_STUCK_IDLE_SECONDS,
        help="a dispatchable PENDING untouched for this long counts as stuck",
    )
    parser.add_argument(
        "--statement-timeout-seconds",
        type=int,
        default=DEFAULT_STATEMENT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    parser.add_argument(
        "--append-markdown",
        type=Path,
        default=None,
        help="append one table row (date and counts only) to this file",
    )
    parser.add_argument("--note", default="", help="note column for the row")
    parser.add_argument(
        "--fail-on-nonzero",
        action="store_true",
        help="exit 1 when any of the three counts is non-zero",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store_url = args.store_url or os.environ.get("GPU_FAULT_STORE_URL")
    if not store_url:
        print(
            "GPU_FAULT_STORE_URL is not set and --store-url was not given",
            file=sys.stderr,
        )
        return 2
    result = run(
        store_url,
        orphan_cutoff_seconds=args.orphan_cutoff_seconds,
        stuck_idle_seconds=args.stuck_idle_seconds,
        statement_timeout_seconds=args.statement_timeout_seconds,
    )
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("\n".join(summary_lines(result)))
    if args.append_markdown is not None:
        with args.append_markdown.open("a", encoding="utf-8") as handle:
            handle.write(markdown_row(result, note=args.note) + "\n")
    if args.fail_on_nonzero and nonzero(result):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
