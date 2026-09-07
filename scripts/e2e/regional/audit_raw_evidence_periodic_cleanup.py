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

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import psycopg

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


def _incident_payload(incident_id: str, now: datetime, *, state: str) -> dict[str, Any]:
    # A real FaultIncident dump, so every other reader of the incident table
    # decodes it: it names the pinned node, was created before the row was
    # observed and is updated now, so the row sits inside
    # [created_at - 1h, updated_at + 1h].
    incident = FaultIncident(
        incident_id=incident_id,
        event_id=f"{incident_id}-event",
        event_type="AUDIT_EVIDENCE_PIN",
        cluster_id=CLUSTER_ID,
        node_ids=[PINNED_NODE],
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
    return int(row[0]) if row is not None else 0


def _wait_deleted(connection: Any, key: str) -> float | None:
    started = time.monotonic()
    while time.monotonic() - started < SWEEP_TIMEOUT_SECONDS:
        with connection.cursor() as cursor:
            if _count(cursor, key) == 0:
                return round(time.monotonic() - started, 3)
        time.sleep(POLL_SECONDS)
    return None


def main() -> None:
    suffix = uuid4().hex[:12]
    unrelated_key = f"audit-expired-evidence-{suffix}"
    pinned_key = f"audit-pinned-evidence-{suffix}"
    incident_id = f"audit-pin-incident-{suffix}"
    now = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "cluster_id": CLUSTER_ID,
        "unrelated_key": unrelated_key,
        "pinned_key": pinned_key,
        "incident_id": incident_id,
    }
    with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"]) as connection:
        with connection.cursor() as cursor:
            for kind, key, payload in (
                (
                    "incident",
                    incident_id,
                    _incident_payload(incident_id, now, state="ESCALATED"),
                ),
                (
                    "raw_evidence",
                    unrelated_key,
                    _evidence_payload(unrelated_key, UNRELATED_NODE, now),
                ),
                (
                    "raw_evidence",
                    pinned_key,
                    _evidence_payload(pinned_key, PINNED_NODE, now),
                ),
            ):
                cursor.execute(
                    "INSERT INTO gpu_fault_objects(kind, key, payload) VALUES (%s, %s, %s::jsonb)",
                    (kind, key, json.dumps(payload)),
                )
        connection.commit()
        try:
            report["unrelated_deleted_after_seconds"] = _wait_deleted(
                connection, unrelated_key
            )
            with connection.cursor() as cursor:
                report["pinned_present_after_unrelated_deleted"] = (
                    _count(cursor, pinned_key) == 1
                )
            # One more full sweep interval with the incident still open.
            time.sleep(POLL_SECONDS * 3)
            with connection.cursor() as cursor:
                report["pinned_present_after_extra_wait"] = (
                    _count(cursor, pinned_key) == 1
                )
                cursor.execute(
                    "UPDATE gpu_fault_objects SET payload = %s::jsonb WHERE kind='incident' AND key=%s",
                    (
                        json.dumps(
                            _incident_payload(
                                incident_id,
                                datetime.now(timezone.utc),
                                state="RECOVERED",
                            )
                        ),
                        incident_id,
                    ),
                )
            connection.commit()
            report["incident_recovered_at"] = datetime.now(timezone.utc).isoformat()
            report["pinned_deleted_after_recovered_seconds"] = _wait_deleted(
                connection, pinned_key
            )
        finally:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM gpu_fault_objects WHERE kind='raw_evidence' AND key IN (%s, %s)",
                    (unrelated_key, pinned_key),
                )
                cursor.execute(
                    "DELETE FROM gpu_fault_objects WHERE kind='incident' AND key=%s",
                    (incident_id,),
                )
                cursor.execute(
                    "SELECT count(*) FROM gpu_fault_objects WHERE key LIKE %s",
                    (f"audit-%-{suffix}",),
                )
                row = cursor.fetchone()
                report["residual_rows"] = int(row[0]) if row is not None else -1
            connection.commit()
    report["verdict"] = (
        "PASS"
        if report.get("unrelated_deleted_after_seconds") is not None
        and report.get("pinned_present_after_unrelated_deleted") is True
        and report.get("pinned_present_after_extra_wait") is True
        and report.get("pinned_deleted_after_recovered_seconds") is not None
        and report.get("residual_rows") == 0
        else "FAIL"
    )
    # Always exit 0: the verdict travels in the JSON so a Pod probe wrapper that
    # retries on non-zero exit does not run the audit three times.
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
