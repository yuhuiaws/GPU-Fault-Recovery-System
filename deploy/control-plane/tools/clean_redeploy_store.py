#!/usr/bin/env python3
"""Aurora reads and writes of prepare-clean-redeploy.sh, run inside a Pod.

The cleanup script ships this file on stdin to a control-plane Pod
(``kubectl exec -i <pod> -- python - <subcommand> ...``), so it may import
nothing but the standard library and psycopg, which the Pod image carries
along with ``GPU_FAULT_STORE_URL``. Every result is one tab-separated line for
the script's ``read``; an empty field prints as ``-`` because a tab-separated
``read`` collapses consecutive tabs and would shift the fields after it.

Subcommands::

  snapshot --scope all|gpu [CLUSTER_ID ...]
      active_workflows  open_commands  processor_rows  spool_rows
      leased_live  leased_details
      ``--scope gpu`` restricts every counter to the named clusters.
      ``leased_live`` counts LEASED commands whose lease has not expired (an
      executor is working on them right now); ``leased_details`` names up to
      20 of them as ``key|operation|node;node``.
  fleet-agents CLUSTER_ID [CLUSTER_ID ...]
      JSON list of the agent payloads registered for those clusters.
  fail-orphaned-leases [CLUSTER_ID ...]
      Fails LEASED commands whose lease has lapsed (status_source
      clean-redeploy-orphan) and prints how many.
  abandon-unclaimable
      Fails, in one transaction, first every PENDING/RUNNING/SAFETY_PENDING
      workflow and then every PENDING/WAITING/LEASED remote command (source
      clean-redeploy-drain); prints
      ``workflows  commands  workflow_keys  command_keys`` (20 keys each).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Sequence

import psycopg

ORPHAN_SOURCE = "clean-redeploy-orphan"
DRAIN_SOURCE = "clean-redeploy-drain"
EMPTY = "-"
NAMED_LIMIT = 20

ACTIVE_WORKFLOWS_SQL = """
    SELECT count(*)
    FROM gpu_fault_objects AS workflow
    JOIN gpu_fault_objects AS incident
      ON incident.kind = 'incident'
     AND incident.key = workflow.payload->>'incident_id'
    WHERE workflow.kind = 'workflow'
      AND workflow.payload->>'status' IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')
"""
OPEN_COMMANDS_SQL = """
    SELECT count(*)
    FROM gpu_fault_objects
    WHERE kind = 'remote_command'
      AND payload->>'status' IN ('PENDING', 'WAITING', 'LEASED')
"""
PROCESSOR_ROWS_SQL = """
    SELECT count(*)
    FROM gpu_fault_processor_queue
    WHERE status IN ('PENDING', 'LEASED')
"""
SPOOL_ROWS_SQL = "SELECT count(*) FROM gpu_fault_telemetry_spool WHERE true"
LIVE_LEASES_SQL = """
    SELECT key,
           payload->'step'->>'operation',
           payload->'step'->'node_ids'
    FROM gpu_fault_objects
    WHERE kind = 'remote_command'
      AND payload->>'status' = 'LEASED'
      AND (payload->>'lease_expires_at')::timestamptz > now()
"""
FLEET_AGENTS_SQL = """
    SELECT payload
    FROM gpu_fault_objects
    WHERE kind='agent'
    ORDER BY key
"""
# LEASED commands whose lease has lapsed after every executor was scaled to
# zero have no claimant left to complete, renew or acknowledge a cancellation.
FAIL_ORPHANED_LEASES_SQL = """
    UPDATE gpu_fault_objects
       SET payload = payload || jsonb_build_object(
               'status', 'FAILED',
               'status_source', %(source)s::text,
               'error', %(reason)s::text,
               'updated_at', %(now)s::text)
     WHERE kind = 'remote_command'
       AND payload->>'status' = 'LEASED'
       AND (payload->>'lease_expires_at')::timestamptz < %(now)s::timestamptz
"""
# WorkflowRequest forbids unknown fields, so the workflow's reason rides on its
# own terminal_failure_reason; remote commands carry status_source and error
# like the orphan pass.
ABANDON_WORKFLOWS_SQL = """
    UPDATE gpu_fault_objects
       SET payload = payload || jsonb_build_object(
               'status', 'FAILED',
               'terminal_failure_reason', %(reason)s::text,
               'updated_at', %(now)s::text)
     WHERE kind = 'workflow'
       AND payload->>'status' IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')
 RETURNING key
"""
ABANDON_COMMANDS_SQL = """
    UPDATE gpu_fault_objects
       SET payload = payload || jsonb_build_object(
               'status', 'FAILED',
               'status_source', %(source)s::text,
               'error', %(reason)s::text,
               'updated_at', %(now)s::text)
     WHERE kind = 'remote_command'
       AND payload->>'status' IN ('PENDING', 'WAITING', 'LEASED')
 RETURNING key
"""


def connect(*, autocommit: bool = False) -> psycopg.Connection[Any]:
    # Reads run autocommit so a table that does not exist yet (an empty store)
    # fails only its own count instead of poisoning the transaction for the
    # counts after it; the writes keep one transaction each.
    return psycopg.connect(
        os.environ["GPU_FAULT_STORE_URL"], connect_timeout=10, autocommit=autocommit
    )


def clean(value: object) -> str:
    """One detail field: no separator of the line, the entry or the field."""

    return str(value or "").replace(",", ";").replace("|", "/").replace("\t", " ")


def named(keys: Sequence[str]) -> str:
    return ",".join(clean(key) for key in sorted(keys)[:NAMED_LIMIT]) or EMPTY


def scalar(cursor: psycopg.Cursor[Any], statement: str, params: Any = ()) -> int:
    try:
        cursor.execute(statement, params)
    except psycopg.errors.UndefinedTable:
        return 0
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def live_leases(
    cursor: psycopg.Cursor[Any], cluster_ids: list[str] | None
) -> tuple[int, str]:
    statement = LIVE_LEASES_SQL
    params: Any = ()
    if cluster_ids is not None:
        statement += " AND payload->>'cluster_id' = ANY(%s)"
        params = (cluster_ids,)
    try:
        cursor.execute(statement + " ORDER BY key", params)
    except psycopg.errors.UndefinedTable:
        return 0, ""
    rows = cursor.fetchall()
    details = ",".join(
        "|".join(
            (
                clean(key),
                clean(operation),
                ";".join(clean(node) for node in (nodes or [])),
            )
        )
        for key, operation, nodes in rows[:NAMED_LIMIT]
    )
    return len(rows), details


def snapshot(arguments: argparse.Namespace) -> None:
    cluster_ids: list[str] | None = arguments.cluster_ids
    if arguments.scope != "gpu":
        cluster_ids = None
    elif not cluster_ids:
        raise SystemExit("scoped cleanup requires cluster ids")

    def count(cursor: psycopg.Cursor[Any], base: str, cluster_clause: str) -> int:
        if cluster_ids is None:
            return scalar(cursor, base)
        return scalar(cursor, base + cluster_clause, (cluster_ids,))

    with connect(autocommit=True) as connection, connection.cursor() as cursor:
        active_workflows = count(
            cursor,
            ACTIVE_WORKFLOWS_SQL,
            " AND incident.payload->>'cluster_id' = ANY(%s)",
        )
        open_commands = count(
            cursor, OPEN_COMMANDS_SQL, " AND payload->>'cluster_id' = ANY(%s)"
        )
        processor_rows = count(cursor, PROCESSOR_ROWS_SQL, " AND cluster_id = ANY(%s)")
        spool_rows = count(cursor, SPOOL_ROWS_SQL, " AND cluster_id = ANY(%s)")
        leased_live, leased_details = live_leases(cursor, cluster_ids)
    print(
        active_workflows,
        open_commands,
        processor_rows,
        spool_rows,
        leased_live,
        leased_details or EMPTY,
        sep="\t",
    )


def fleet_agents(arguments: argparse.Namespace) -> None:
    wanted = set(arguments.cluster_ids)
    with connect(autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(FLEET_AGENTS_SQL)
        agents = [
            row[0] for row in cursor.fetchall() if row[0].get("cluster_id") in wanted
        ]
    print(json.dumps(agents, sort_keys=True))


def fail_orphaned_leases(arguments: argparse.Namespace) -> None:
    statement = FAIL_ORPHANED_LEASES_SQL
    params: dict[str, Any] = {
        "source": ORPHAN_SOURCE,
        "reason": (
            "orphaned by clean-redeploy: no executor remains to complete the lease"
        ),
        "now": datetime.now(timezone.utc).isoformat(),
    }
    if arguments.cluster_ids:
        statement += " AND payload->>'cluster_id' = ANY(%(clusters)s::text[])"
        params["clusters"] = arguments.cluster_ids
    with connect() as connection, connection.cursor() as cursor:
        try:
            cursor.execute(statement, params)
        except psycopg.errors.UndefinedTable:
            connection.rollback()
            print(0)
            return
        failed = cursor.rowcount
        connection.commit()
    print(failed)


def abandon_unclaimable(arguments: argparse.Namespace) -> None:
    params = {
        "source": DRAIN_SOURCE,
        "reason": (
            f"abandoned by {DRAIN_SOURCE}: cluster draining for uninstall, "
            "no executor may claim"
        ),
        "now": datetime.now(timezone.utc).isoformat(),
    }
    with connect() as connection, connection.cursor() as cursor:
        try:
            cursor.execute(ABANDON_WORKFLOWS_SQL, params)
            workflows = [str(row[0]) for row in cursor.fetchall()]
            cursor.execute(ABANDON_COMMANDS_SQL, params)
            commands = [str(row[0]) for row in cursor.fetchall()]
        except psycopg.errors.UndefinedTable:
            connection.rollback()
            print(0, 0, EMPTY, EMPTY, sep="\t")
            return
        connection.commit()
    print(len(workflows), len(commands), named(workflows), named(commands), sep="\t")


def main() -> int:
    parser = argparse.ArgumentParser(prog="clean_redeploy_store")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("snapshot")
    probe.add_argument("--scope", choices=("all", "gpu"), required=True)
    probe.add_argument("cluster_ids", nargs="*")
    probe.set_defaults(run=snapshot)
    agents = subparsers.add_parser("fleet-agents")
    agents.add_argument("cluster_ids", nargs="+")
    agents.set_defaults(run=fleet_agents)
    orphans = subparsers.add_parser("fail-orphaned-leases")
    orphans.add_argument("cluster_ids", nargs="*")
    orphans.set_defaults(run=fail_orphaned_leases)
    abandon = subparsers.add_parser("abandon-unclaimable")
    abandon.set_defaults(run=abandon_unclaimable)
    arguments = parser.parse_args()
    arguments.run(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
