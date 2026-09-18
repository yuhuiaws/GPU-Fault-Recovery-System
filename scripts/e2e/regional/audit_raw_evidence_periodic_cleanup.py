"""Raw-evidence retention audit, run inside a control-worker Pod (PREEMPT-038).

Inserts already-expired ``raw_evidence`` rows straight into the live store and
watches the periodic ``cleanup_expired_raw_evidence`` sweep act on them:

* an **unrelated** row (no incident names its node) must be deleted;
* a row **pinned** by an open incident of the same cluster that names its node
  inside the incident's window (ARCH-D6) must survive every sweep that removes
  the unrelated one;
* once that incident is RECOVERED the pinned row must go too.

Every row it writes is keyed ``audit-`` under the synthetic cluster
``audit-cluster`` and is deleted in ``finally``; nothing of the real fleet is
read or written. The output is one JSON object naming the keys, so the runner
can look for them in the sweep's ``cleanup raw_evidence deleted ... keys[:20]``
log line (ARCH-D7).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.models import FaultIncident, IncidentState

CLUSTER_ID = "audit-cluster"
UNRELATED_NODE = "audit-node"
PINNED_NODE = "audit-pinned-node"
POLL_SECONDS = 5
SWEEP_TIMEOUT_SECONDS = 180


def _evidence_payload(key: str, node_id: str, now: datetime) -> dict[str, Any]:
    return {
        "record_id": key,
        "cluster_id": CLUSTER_ID,
        "node_id": node_id,
        "kind": "WORKLOAD_LOG",
        "observed_at": (now - timedelta(hours=2)).isoformat(),
        "ingested_at": (now - timedelta(hours=2)).isoformat(),
        "expires_at": (now - timedelta(hours=1)).isoformat(),
        "attempt_ids": ["audit-attempt"],
        "payload": {"audit": True},
    }


def _incident_payload(
    incident_id: str, now: datetime, *, state: str, node_id: str = PINNED_NODE
) -> dict[str, Any]:
    # A real FaultIncident dump, so every other reader of the incident table
    # decodes it: it names the pinned node, was created before the row was
    # observed and is updated now, so the row sits inside
    # [created_at - 1h, updated_at + 1h].
    incident = FaultIncident(
        incident_id=incident_id,
        event_id=f"{incident_id}-event",
        event_type="AUDIT_EVIDENCE_PIN",
        cluster_id=CLUSTER_ID,
        node_ids=[node_id],
        policy_version="audit",
        policy_source="audit",
        state=IncidentState(state),
        reasons=["PREEMPT-038 evidence pin audit"],
        fencing_token=1,
        created_at=now - timedelta(hours=3),
        updated_at=now,
    )
    return dict(incident.model_dump(mode="json"))


def _count(cursor: Any, key: str) -> int:
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_objects WHERE kind='raw_evidence' AND key=%s",
        (key,),
    )
    row = cursor.fetchone()
    if row is None or type(row[0]) is not int or row[0] < 0:
        raise RuntimeError("evidence count is unknown")
    return row[0]


def _wait_deleted(connection: Any, key: str) -> float | None:
    started = time.monotonic()
    while time.monotonic() - started < SWEEP_TIMEOUT_SECONDS:
        with connection.cursor() as cursor:
            if _count(cursor, key) == 0:
                return round(time.monotonic() - started, 3)
        time.sleep(POLL_SECONDS)
    return None


def store_dsn() -> str:
    configured = os.environ.get("GPU_FAULT_STORE_URL_FILE")
    path = (
        configured if configured is not None else "/etc/gpu-fault/aurora/postgres-url"
    )
    if not path:
        raise RuntimeError("configured store DSN file path is empty")
    try:
        with open(path, encoding="utf-8") as handle:
            value = handle.read().strip()
    except FileNotFoundError:
        if configured is not None:
            raise
        return os.environ["GPU_FAULT_STORE_URL"]
    if not value:
        raise RuntimeError("store DSN file is empty")
    return value


def audit_identity(run_id: str) -> dict[str, str]:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{7,47}", run_id) is None:
        raise ValueError("invalid evidence audit run identity")
    return {
        "run_id": run_id,
        "cluster_id": CLUSTER_ID,
        "unrelated_key": f"audit-expired-evidence-{run_id}",
        "pinned_key": f"audit-pinned-evidence-{run_id}",
        "incident_id": f"audit-pin-incident-{run_id}",
        "unrelated_node": f"{UNRELATED_NODE}-{run_id}",
        "pinned_node": f"{PINNED_NODE}-{run_id}",
    }


def _owned_rows(connection: Any, identity: dict[str, str]) -> list[Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT kind,key,payload FROM gpu_fault_objects WHERE "
            "(kind='raw_evidence' AND key IN (%s,%s)) OR (kind='incident' AND key=%s)",
            (
                identity["unrelated_key"],
                identity["pinned_key"],
                identity["incident_id"],
            ),
        )
        return list(cursor.fetchall())


def cleanup_rows(connection: Any, identity: dict[str, str]) -> dict[str, Any]:
    # A failed statement leaves PostgreSQL's transaction aborted. Cleanup gets
    # a fresh transaction, and the caller can repeat it in a new connection.
    connection.rollback()
    rows = _owned_rows(connection, identity)
    for kind, key, raw in rows:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict) or payload.get("cluster_id") != CLUSTER_ID:
            raise RuntimeError("audit cleanup row ownership changed")
        if kind == "incident":
            owned = (
                payload.get("incident_id") == identity["incident_id"] == key
                and payload.get("event_type") == "AUDIT_EVIDENCE_PIN"
                and payload.get("node_ids") == [identity["pinned_node"]]
            )
        else:
            node = (
                identity["pinned_node"]
                if key == identity["pinned_key"]
                else identity["unrelated_node"]
            )
            owned = (
                payload.get("record_id") == key
                and payload.get("node_id") == node
                and payload.get("payload") == {"audit": True}
            )
        if not owned:
            raise RuntimeError("audit cleanup row ownership changed")
    with connection.cursor() as cursor:
        for kind, key, payload in rows:
            cursor.execute(
                "DELETE FROM gpu_fault_objects WHERE kind=%s AND key=%s AND payload=%s::jsonb",
                (
                    kind,
                    key,
                    json.dumps(payload) if not isinstance(payload, str) else payload,
                ),
            )
    connection.commit()
    remaining = _owned_rows(connection, identity)
    connection.rollback()
    return {"residual_rows": len(remaining)}


def audit(connection: Any, run_id: str) -> dict[str, Any]:
    identity = audit_identity(run_id)
    unrelated_key, pinned_key, incident_id = (
        identity["unrelated_key"],
        identity["pinned_key"],
        identity["incident_id"],
    )
    now = datetime.now(timezone.utc)
    report: dict[str, Any] = {**identity, "verdict": "FAIL"}
    if _owned_rows(connection, identity):
        connection.rollback()
        return {
            **report,
            "error": "audit identities already exist",
            "cleanup_permitted": False,
        }
    try:
        with connection.cursor() as cursor:
            for kind, key, payload in (
                (
                    "incident",
                    incident_id,
                    _incident_payload(
                        incident_id,
                        now,
                        state="ESCALATED",
                        node_id=identity["pinned_node"],
                    ),
                ),
                (
                    "raw_evidence",
                    unrelated_key,
                    _evidence_payload(unrelated_key, identity["unrelated_node"], now),
                ),
                (
                    "raw_evidence",
                    pinned_key,
                    _evidence_payload(pinned_key, identity["pinned_node"], now),
                ),
            ):
                cursor.execute(
                    "INSERT INTO gpu_fault_objects(kind, key, payload) VALUES (%s, %s, %s::jsonb)",
                    (kind, key, json.dumps(payload)),
                )
        connection.commit()
        report["unrelated_deleted_after_seconds"] = _wait_deleted(
            connection, unrelated_key
        )
        with connection.cursor() as cursor:
            report["pinned_present_after_unrelated_deleted"] = (
                _count(cursor, pinned_key) == 1
            )
        if (
            report["unrelated_deleted_after_seconds"] is None
            or not report["pinned_present_after_unrelated_deleted"]
        ):
            raise RuntimeError(
                "unrelated deletion and pin preservation were not proven"
            )
        time.sleep(POLL_SECONDS * 3)
        with connection.cursor() as cursor:
            report["pinned_present_after_extra_wait"] = _count(cursor, pinned_key) == 1
            if not report["pinned_present_after_extra_wait"]:
                raise RuntimeError("pinned evidence disappeared before recovery")
            cursor.execute(
                "UPDATE gpu_fault_objects SET payload=%s::jsonb WHERE kind='incident' AND key=%s",
                (
                    json.dumps(
                        _incident_payload(
                            incident_id,
                            datetime.now(timezone.utc),
                            state="RECOVERED",
                            node_id=identity["pinned_node"],
                        )
                    ),
                    incident_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("audit incident recovery was not persisted")
        connection.commit()
        report["incident_recovered_at"] = datetime.now(timezone.utc).isoformat()
        report["pinned_deleted_after_recovered_seconds"] = _wait_deleted(
            connection, pinned_key
        )
    except Exception as exc:
        report["error"] = f"audit failed: {type(exc).__name__}"
    finally:
        try:
            report.update(cleanup_rows(connection, identity))
        except Exception as exc:
            report["cleanup_error"] = f"audit cleanup failed: {type(exc).__name__}"
    report["verdict"] = (
        "PASS"
        if report.get("unrelated_deleted_after_seconds") is not None
        and report.get("pinned_present_after_unrelated_deleted") is True
        and report.get("pinned_present_after_extra_wait") is True
        and report.get("pinned_deleted_after_recovered_seconds") is not None
        and report.get("residual_rows") == 0
        and "error" not in report
        and "cleanup_error" not in report
        else "FAIL"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    import psycopg

    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--cleanup-only", action="store_true")
    arguments = parser.parse_args(argv)
    identity = audit_identity(arguments.run_id)
    with psycopg.connect(store_dsn()) as connection:
        if arguments.cleanup_only:
            report = {**identity, **cleanup_rows(connection, identity)}
            report["verdict"] = "PASS" if report["residual_rows"] == 0 else "FAIL"
        else:
            report = audit(connection, arguments.run_id)
    # Always exit 0: the verdict travels in the JSON so a Pod probe wrapper that
    # retries on non-zero exit does not run the audit three times.
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
