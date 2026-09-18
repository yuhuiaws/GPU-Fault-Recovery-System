"""CAP004-only seed and terminal readback for the disposable capacity database."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.store import PostgresStore
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.store.shared.errors import WorkflowLeaseError

CLUSTER_ID = "cap-cluster-000"
OWNER = "gpu-fault-cap004-ledger"
COMMAND_COUNT = 25
CLEANUP_TIMEOUT_SECONDS = 60
TERMINAL = {
    RemoteCommandStatus.SUCCEEDED,
    RemoteCommandStatus.FAILED,
}


class Cap004Store(ControlPlaneStore, Protocol):
    """Command methods not yet declared by the shared Store read contract."""

    def ensure_remote_command(
        self, command: RemoteActionCommand
    ) -> RemoteActionCommand: ...

    def claim_remote_commands(
        self,
        cluster_id: str,
        executor_id: str,
        *,
        limit: int,
        lease_seconds: int,
        execution_owners: set[str] | None = None,
        accept_batched_steps: bool = True,
    ) -> list[RemoteActionCommand]: ...

    def complete_remote_command(
        self,
        cluster_id: str,
        command_id: str,
        result: RemoteCommandResult,
    ) -> RemoteActionCommand: ...


def commands_for_run(run_id: str) -> list[RemoteActionCommand]:
    if re.fullmatch(r"cap[a-z0-9]{6,32}", run_id) is None:
        raise ValueError("CAP004 run identity is invalid")
    now = datetime.now(timezone.utc)
    commands = []
    for index in range(COMMAND_COUNT):
        identity = f"{run_id}-cap004-{index:03d}"
        step = WorkflowStepSpec(
            operation=WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            execution_owner=OWNER,
            node_ids=[f"node-{identity}"],
            parameters={"cap004_run_id": run_id, "nonphysical": True},
        )
        workflow = WorkflowRequest(
            request_id=f"workflow-{identity}",
            incident_id=f"incident-{identity}",
            runtime_profile_version="hyperpod-v1",
            status=WorkflowStatus.PENDING,
            official_action="CAPACITY_DIAGNOSTIC",
            fencing_token=1,
            official_steps=[step],
            created_at=now,
            updated_at=now,
        )
        incident = FaultIncident(
            incident_id=workflow.incident_id,
            event_id=f"event-{identity}",
            event_type="CAPACITY_TEST",
            cluster_id=CLUSTER_ID,
            node_ids=step.node_ids,
            policy_version="capacity-v1",
            policy_source="CONTROLLED_DRILL",
            official_action=workflow.official_action,
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=workflow.request_id,
            drill_id=run_id,
            created_at=now,
            updated_at=now,
        )
        commands.append(
            RemoteActionCommand(
                command_id=f"command-{identity}",
                cluster_id=CLUSTER_ID,
                workflow_request_id=workflow.request_id,
                incident_id=incident.incident_id,
                step_index=0,
                fencing_token=1,
                idempotency_key=f"{workflow.request_id}/0/{step.operation.value}",
                step=step,
                workflow=workflow,
                incident=incident,
                created_at=now,
                updated_at=now,
            )
        )
    return commands


def require_owned(command: RemoteActionCommand, expected: RemoteActionCommand) -> None:
    if (
        command.command_id != expected.command_id
        or command.cluster_id != CLUSTER_ID
        or command.idempotency_key != expected.idempotency_key
        or command.workflow_request_id != expected.workflow_request_id
        or command.incident_id != expected.incident_id
        or command.step_index != 0
        or command.step != expected.step
        or command.batched_steps
        or command.workflow.request_id != expected.workflow_request_id
        or command.workflow.incident_id != expected.incident_id
        or command.workflow.fencing_token != 1
        or command.workflow.runtime_profile_version
        != expected.workflow.runtime_profile_version
        or command.workflow.official_steps != [expected.step]
        or command.workflow.safety_steps
        or command.incident.drill_id != expected.incident.drill_id
        or command.incident.cluster_id != CLUSTER_ID
        or command.incident.node_ids != expected.step.node_ids
        or command.incident.workflow_request_id != expected.workflow_request_id
        or command.incident.fencing_token != 1
        or command.fencing_token != 1
    ):
        raise ValueError("CAP004 command identity differs from the owned seed")


def _validate_commands(
    rows: list[RemoteActionCommand],
    expected: dict[str, RemoteActionCommand],
    *,
    allow_missing: bool,
) -> None:
    if len(rows) != len({item.command_id for item in rows}) or any(
        item.command_id not in expected for item in rows
    ):
        raise ValueError("CAP004 database contains unexpected commands")
    if not allow_missing and {item.command_id for item in rows} != set(expected):
        raise ValueError("CAP004 command inventory is incomplete")
    for item in rows:
        require_owned(item, expected[item.command_id])


def _cleanup_commands(
    store: Cap004Store,
    run_id: str,
    expected: dict[str, RemoteActionCommand],
    rows: list[RemoteActionCommand],
) -> list[RemoteActionCommand]:
    deadline = time.monotonic() + CLEANUP_TIMEOUT_SECONDS
    candidates = rows
    while any(item.status not in TERMINAL for item in rows):
        if time.monotonic() >= deadline:
            raise TimeoutError("CAP004 cleanup could not reclaim every command lease")
        reclaimed = store.claim_remote_commands(
            CLUSTER_ID,
            f"{run_id}-cleanup",
            limit=COMMAND_COUNT,
            lease_seconds=30,
            execution_owners={OWNER},
        )
        _validate_commands(reclaimed, expected, allow_missing=True)
        current = {item.command_id: item for item in [*candidates, *reclaimed]}
        for item in current.values():
            if item.status in TERMINAL:
                continue
            if not item.lease_token:
                raise ValueError("CAP004 cleanup cannot establish command ownership")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "CAP004 cleanup exceeded its lease reclamation deadline"
                )
            try:
                store.complete_remote_command(
                    CLUSTER_ID,
                    item.command_id,
                    RemoteCommandResult(
                        lease_token=item.lease_token,
                        status=RemoteCommandStatus.FAILED,
                        status_source="cap004-cleanup",
                        error="Nonphysical CAP004 executor stopped before completion",
                        details={"nonphysical": True, "cleanup": True},
                    ),
                )
            except WorkflowLeaseError:
                # The stopped fixture's lease may expire during this batch.
                # A retry must claim a new lease, not borrow another holder's token.
                continue
        candidates = []
        if store.list_agents():
            raise ValueError("CAP004 database contains Node Agent records")
        rows = store.list_remote_commands()
        _validate_commands(rows, expected, allow_missing=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("CAP004 cleanup exceeded its lease reclamation deadline")
        if any(item.status not in TERMINAL for item in rows):
            time.sleep(min(0.1, remaining))
    return rows


def command_snapshot(store: Cap004Store, run_id: str, mode: str) -> dict[str, Any]:
    if mode not in {"seed", "inspect", "cleanup"}:
        raise ValueError("CAP004 Store probe mode is invalid")
    expected = {item.command_id: item for item in commands_for_run(run_id)}
    if store.list_agents():
        raise ValueError("CAP004 database contains Node Agent records")
    rows = store.list_remote_commands()
    if mode == "seed":
        if rows:
            raise ValueError("CAP004 requires an empty isolated command database")
        for item in expected.values():
            store.ensure_remote_command(item)
        rows = store.list_remote_commands()
    _validate_commands(rows, expected, allow_missing=mode == "cleanup")
    if mode == "cleanup":
        # Only the stopped, bounded nonphysical fixture may enter this cleanup.
        rows = _cleanup_commands(store, run_id, expected, rows)
    return {
        "run_id": run_id,
        "cluster_id": CLUSTER_ID,
        "commands_created": len(rows),
        "node_agent_records": 0,
        "status_counts": dict(Counter(item.status.value for item in rows)),
        "commands": [
            {
                "command_id": item.command_id,
                "idempotency_key": item.idempotency_key,
                "status": item.status.value,
                "lease_present": any(
                    (
                        item.lease_owner,
                        item.lease_token,
                        item.lease_expires_at,
                    )
                ),
                "nonphysical": item.result_details.get("nonphysical") is True,
                "ledger_count": item.result_details.get("ledger_count"),
                "cached": item.result_details.get("cached"),
            }
            for item in sorted(rows, key=lambda row: row.command_id)
        ],
    }


def terminal_errors(snapshot: dict[str, Any], run_id: str) -> list[str]:
    expected = {
        item.command_id: item.idempotency_key for item in commands_for_run(run_id)
    }
    rows = snapshot.get("commands")
    if (
        snapshot.get("run_id") != run_id
        or snapshot.get("cluster_id") != CLUSTER_ID
        or snapshot.get("node_agent_records") != 0
        or not isinstance(rows, list)
        or len(rows) != COMMAND_COUNT
        or any(not isinstance(row, dict) for row in rows)
        or {row.get("command_id"): row.get("idempotency_key") for row in rows}
        != expected
    ):
        return ["CAP004 terminal inventory differs from the exact owned command set"]
    if any(
        row.get("status") != "SUCCEEDED"
        or row.get("lease_present") is not False
        or row.get("nonphysical") is not True
        or type(row.get("ledger_count")) is not int
        or row["ledger_count"] != 1
        for row in rows
    ):
        return ["CAP004 commands are not terminal nonphysical ledger successes"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seed", "inspect", "cleanup"))
    parser.add_argument("run_id")
    parser.add_argument("database")
    args = parser.parse_args()
    commands_for_run(args.run_id)
    if (
        args.database != f"gpu_fault_{args.run_id}_cap004"
        or Path("/work/database-name").read_text().strip() != args.database
    ):
        raise ValueError("CAP004 database identity does not match this run")
    import psycopg

    url = Path("/work/store-url").read_text(encoding="utf-8").strip()
    with psycopg.connect(url, connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            if cursor.fetchone() != (args.database,):
                raise ValueError("CAP004 connected to a different database")
    store = PostgresStore(
        url, initialize_schema=False, pool_min_size=0, pool_max_size=2
    )
    try:
        result = command_snapshot(store, args.run_id, args.mode)
    finally:
        store.close()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
