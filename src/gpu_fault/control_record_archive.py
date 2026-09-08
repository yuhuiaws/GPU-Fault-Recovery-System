from __future__ import annotations

import gzip
import hashlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from gpu_fault.models import WorkflowStatus
from gpu_fault.store.shared.time import utc_text as _utc_text

LOGGER = logging.getLogger(__name__)

# Incidents one ``run_once`` may archive. At the default interval (10 min)
# this is ~28 800 incidents a day against ~600 with the old 25-an-hour shape,
# which could never catch up with a year of backlog once retention was
# switched on (control-plane review 2026-09-08, F-8).
DEFAULT_ARCHIVE_BATCH_SIZE = 200

# The candidate predicate, shape-for-shape with the partial index Agent 5
# declares as ``gpu_fault_incident_archive_candidate``:
#   ON gpu_fault_objects ((payload->>'updated_at')) WHERE kind='incident'.
# ``updated_at`` is compared as text against ``utc_text`` -- the byte form the
# models serialise -- because a ``::timestamptz`` cast is not IMMUTABLE and
# cannot be indexed; the other retention sweeps compare the same way.
ARCHIVE_CANDIDATE_SQL = """
SELECT i.key
FROM gpu_fault_objects i
WHERE i.kind='incident'
  AND i.payload->>'updated_at' <= %s
  AND NOT EXISTS (
    SELECT 1 FROM gpu_fault_objects w
    WHERE w.kind='workflow'
      AND w.payload->>'incident_id'=i.key
      AND NOT {inactive}
  )
ORDER BY i.payload->>'updated_at', i.key
LIMIT %s
"""


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
        store: Any | None = None,
        batch_size: int = DEFAULT_ARCHIVE_BATCH_SIZE,
    ) -> None:
        parsed = urlparse(archive_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("archive URI must use s3://bucket/prefix")
        if retention < timedelta(days=1):
            raise ValueError("control record retention must be >=1 day")
        if batch_size < 1:
            raise ValueError("control record archive batch size must be positive")
        self.dsn = dsn
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.retention = retention
        self.batch_size = batch_size
        # With a ``PostgresStore`` the archiver borrows its connection pool:
        # two bare ``psycopg.connect`` per incident sat outside the pool and
        # its capacity estimate, and cost a TLS handshake each (F-8). The
        # DSN path stays for tools that run without a store.
        self._db = getattr(store, "_db", None) if store is not None else None
        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")
        self.s3 = s3_client
        # Refusals by reason. A retention that is permanently withheld used
        # to be invisible; this is what the metrics contributor exports.
        self.withheld_total: dict[str, int] = {}
        # Incidents archived, and per-incident failures that were neither a
        # safety refusal nor fatal to the run, by exception type. Exported as
        # gpu_fault_control_record_archive_archived_total and
        # gpu_fault_control_record_archive_errors_total{reason}.
        self.archived_total = 0
        self.errors_total: dict[str, int] = {}

    @contextmanager
    def _read_cursor(self) -> Iterator[Any]:
        """An autocommit cursor: pooled when a store was given, else a bare
        connection on the DSN."""

        if self._db is not None:
            with self._db.cursor() as cursor:
                yield cursor
            return
        import psycopg

        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                yield cursor

    @contextmanager
    def _serializable_cursor(self) -> Iterator[Any]:
        """A cursor inside one SERIALIZABLE transaction that commits on exit."""

        if self._db is not None:
            with self._db.transaction():
                with self._db.cursor() as cursor:
                    cursor.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                    yield cursor
            return
        import psycopg

        with psycopg.connect(self.dsn) as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            with connection.cursor() as cursor:
                yield cursor

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
        with self._read_cursor() as cursor:
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
        with self._serializable_cursor() as cursor:
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

    def candidates(self, *, limit: int | None = None) -> list[str]:
        """Incident ids past retention with no active workflow, oldest first."""

        cutoff = datetime.now(timezone.utc) - self.retention
        query = ARCHIVE_CANDIDATE_SQL.format(
            inactive=_INACTIVE_WORKFLOW_SQL.replace("payload", "w.payload")
        )
        with self._read_cursor() as cursor:
            cursor.execute(
                query,
                (
                    _utc_text(cutoff),
                    _ARCHIVABLE_STATUS_VALUES,
                    self.batch_size if limit is None else limit,
                ),
            )
            return [row[0] for row in cursor.fetchall()]

    def run_once(self, *, limit: int | None = None) -> list[str]:
        """Archive up to ``limit`` (default ``batch_size``) candidates.

        A safety refusal is counted by reason and skipped; any other
        per-incident failure (S3, serialization conflict, connection loss) is
        counted by exception type and skipped too, so one bad incident does
        not end the round for the rest (F-8). Both dictionaries are exported.
        """

        archived = []
        for incident_id in self.candidates(limit=limit):
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
            except Exception as exc:
                reason = type(exc).__name__
                self.errors_total[reason] = self.errors_total.get(reason, 0) + 1
                LOGGER.exception(
                    "control record archive failed for incident %s; continuing",
                    incident_id,
                )
                continue
            self.archived_total += 1
        return archived
