from __future__ import annotations

import gzip
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from gpu_fault.models import WorkflowStatus

LOGGER = logging.getLogger(__name__)


class ArchiveSafetyError(RuntimeError):
    pass


# Archivable is a whitelist of statuses that will never execute or be acted
# on again. BLOCKED is deliberately absent: a blocked workflow is held for
# reconciliation or an operator, and archiving its incident would delete the
# record before anyone reconciled it (F-I1, F-B4).
ARCHIVABLE_WORKFLOW_STATUSES = frozenset(
    {
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.SUPERSEDED,
    }
)
_ARCHIVABLE_STATUS_VALUES = sorted(
    status.value for status in ARCHIVABLE_WORKFLOW_STATUSES
)
# A FAILED workflow is archivable only once its failure handler ran
# (``failure_handled_at``); before that an escalation may still be owed
# (F-B4 (5)). Statuses in the whitelist plus this predicate are "inactive".
_INACTIVE_WORKFLOW_SQL = """(
    payload->>'status' = ANY(%s)
    OR (
        payload->>'status' = 'FAILED'
        AND payload->>'failure_handled_at' IS NOT NULL
    )
)"""


RELATED_RECORDS_SQL = """
WITH
workflow_ids AS (
  SELECT key FROM gpu_fault_objects
  WHERE kind='workflow' AND payload->>'incident_id'=%s
),
notification_ids AS (
  SELECT key FROM gpu_fault_objects
  WHERE kind='notification' AND payload->>'incident_id'=%s
),
plan_ids AS (
  SELECT key FROM gpu_fault_objects
  WHERE kind='plan' AND payload->>'incident_id'=%s
),
decision_ids AS (
  SELECT key,payload FROM gpu_fault_objects
  WHERE kind='decision' AND payload->>'recovery_plan_id'
        IN (SELECT key FROM plan_ids)
),
diagnostic_ids AS (
  SELECT DISTINCT payload->>'diagnostic_request_id' AS key
  FROM decision_ids
  WHERE COALESCE(payload->>'diagnostic_request_id','')<>''
),
event_ids AS (
  SELECT key FROM gpu_fault_links
  WHERE kind='incident_by_event' AND value=%s
),
objects AS (
  SELECT 'gpu_fault_objects' AS table_name,kind,key,payload,NULL::text value
  FROM gpu_fault_objects
  WHERE (kind='incident' AND key=%s)
     OR (kind='workflow' AND key IN (SELECT key FROM workflow_ids))
     OR (kind IN ('notification','notification_delivery',
                  'notification_result')
         AND key IN (SELECT key FROM notification_ids))
     OR (kind='plan' AND key IN (SELECT key FROM plan_ids))
     OR (kind IN ('decision','event')
         AND key IN (SELECT key FROM decision_ids))
     OR (kind IN ('diagnostic','triage')
         AND key IN (SELECT key FROM diagnostic_ids))
     OR (kind='marker' AND payload->>'incident_id'=%s)
     OR (kind='remote_command' AND payload->>'incident_id'=%s)
     OR (kind IN ('xid_correlation_event','xid_policy_decision',
                  'xid_correlation')
         AND key IN (SELECT key FROM event_ids))
),
links AS (
  SELECT 'gpu_fault_links' AS table_name,kind,key,NULL::jsonb payload,value
  FROM gpu_fault_links
  WHERE (kind IN ('incident_by_event','replacement_fault_group',
                  'sxid_fault_group') AND value=%s)
     OR (kind='notification_dedup'
         AND value IN (SELECT key FROM notification_ids))
)
SELECT table_name,kind,key,payload,value FROM objects
UNION ALL
SELECT table_name,kind,key,payload,value FROM links
ORDER BY table_name,kind,key
"""


class ControlRecordArchiver:
    def __init__(
        self,
        dsn: str,
        archive_uri: str,
        *,
        retention: timedelta = timedelta(days=365),
        s3_client=None,
    ) -> None:
        parsed = urlparse(archive_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("archive URI must use s3://bucket/prefix")
        if retention < timedelta(days=1):
            raise ValueError("control record retention must be >=1 day")
        self.dsn = dsn
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.retention = retention
        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")
        self.s3 = s3_client
        # Refusals by reason. A retention that is permanently withheld used
        # to be invisible; this is what the metrics contributor exports.
        self.withheld_total: dict[str, int] = {}

    @staticmethod
    def _bundle(cursor, incident_id: str) -> tuple[bytes, list[tuple]]:
        cursor.execute(
            """
            SELECT count(*) FROM gpu_fault_objects
            WHERE kind='workflow' AND payload->>'incident_id'=%s
              AND NOT """
            + _INACTIVE_WORKFLOW_SQL
            + """
            """,
            (incident_id, _ARCHIVABLE_STATUS_VALUES),
        )
        if cursor.fetchone()[0]:
            raise ArchiveSafetyError("incident has non-terminal workflow")
        cursor.execute(
            """
            SELECT count(*) FROM gpu_fault_objects
            WHERE kind='remote_command' AND payload->>'incident_id'=%s
              AND payload->>'status' IN ('PENDING','WAITING','LEASED')
            """,
            (incident_id,),
        )
        if cursor.fetchone()[0]:
            raise ArchiveSafetyError("incident has open remote command")
        cursor.execute(
            """
            SELECT count(*) FROM gpu_fault_objects
            WHERE kind='workflow' AND payload->>'incident_id'<>%s
              AND payload->>'predecessor_workflow_id' IN (
                SELECT key FROM gpu_fault_objects
                WHERE kind='workflow' AND payload->>'incident_id'=%s
              )
            """,
            (incident_id, incident_id),
        )
        if cursor.fetchone()[0]:
            raise ArchiveSafetyError("incident has external successor")
        cursor.execute(RELATED_RECORDS_SQL, (incident_id,) * 8)
        rows = cursor.fetchall()
        if not any(row[1] == "incident" for row in rows):
            raise ArchiveSafetyError("incident does not exist")
        records = [
            {
                "table": row[0],
                "kind": row[1],
                "key": row[2],
                **(
                    {"payload": row[3]}
                    if row[0] == "gpu_fault_objects"
                    else {"value": row[4]}
                ),
            }
            for row in rows
        ]
        raw = (
            "\n".join(
                json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                for item in records
            )
            + "\n"
        ).encode()
        return raw, rows

    def _key(self, incident_id: str, raw: bytes) -> str:
        digest = hashlib.sha256(raw).hexdigest()
        safe = incident_id.replace("/", "_")
        name = f"{safe}-{digest}.ndjson.gz"
        return f"{self.prefix}/{name}" if self.prefix else name

    def archive_one(self, incident_id: str) -> str:
        import psycopg

        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                raw, _ = self._bundle(cursor, incident_id)
        key = self._key(incident_id, raw)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=gzip.compress(raw, mtime=0),
            ContentType="application/x-ndjson",
            ContentEncoding="gzip",
            Metadata={"sha256": hashlib.sha256(raw).hexdigest()},
        )
        with psycopg.connect(self.dsn) as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                    (f"gpu-fault-incident-archive/{incident_id}",),
                )
                current, rows = self._bundle(cursor, incident_id)
                if current != raw:
                    raise ArchiveSafetyError("incident changed after archive upload")
                object_pairs = [
                    (row[1], row[2]) for row in rows if row[0] == "gpu_fault_objects"
                ]
                link_pairs = [
                    (row[1], row[2]) for row in rows if row[0] == "gpu_fault_links"
                ]
                if link_pairs:
                    cursor.execute(
                        """
                        DELETE FROM gpu_fault_links
                        WHERE (kind,key) IN (
                          SELECT * FROM unnest(%s::text[],%s::text[])
                        )
                        """,
                        (
                            [item[0] for item in link_pairs],
                            [item[1] for item in link_pairs],
                        ),
                    )
                if object_pairs:
                    cursor.execute(
                        """
                        DELETE FROM gpu_fault_objects
                        WHERE (kind,key) IN (
                          SELECT * FROM unnest(%s::text[],%s::text[])
                        )
                        """,
                        (
                            [item[0] for item in object_pairs],
                            [item[1] for item in object_pairs],
                        ),
                    )
        return f"s3://{self.bucket}/{key}"

    def run_once(self, *, limit: int = 25) -> list[str]:
        import psycopg

        cutoff = datetime.now(timezone.utc) - self.retention
        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT i.key
                    FROM gpu_fault_objects i
                    WHERE i.kind='incident'
                      AND (i.payload->>'updated_at')::timestamptz < %s
                      AND NOT EXISTS (
                        SELECT 1 FROM gpu_fault_objects w
                        WHERE w.kind='workflow'
                          AND w.payload->>'incident_id'=i.key
                          AND NOT """
                    + _INACTIVE_WORKFLOW_SQL.replace("payload", "w.payload")
                    + """
                      )
                    ORDER BY (i.payload->>'updated_at')::timestamptz,i.key
                    LIMIT %s
                    """,
                    (cutoff, _ARCHIVABLE_STATUS_VALUES, limit),
                )
                candidates = [row[0] for row in cursor.fetchall()]
        archived = []
        for incident_id in candidates:
            try:
                archived.append(self.archive_one(incident_id))
            except ArchiveSafetyError as exc:
                reason = str(exc)
                self.withheld_total[reason] = self.withheld_total.get(reason, 0) + 1
                LOGGER.warning(
                    "control record archive withheld incident %s: %s",
                    incident_id,
                    reason,
                )
                continue
        return archived
