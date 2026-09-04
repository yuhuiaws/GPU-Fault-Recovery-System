from __future__ import annotations

import inspect
import json

from gpu_fault.processor import ProcessorRequest
from gpu_fault.store import SqliteStore
from gpu_fault.store.postgres.ddl import create_postgres_schema
from gpu_fault.store.postgres.processor_storage import PostgresProcessorStorageMixin
from tests._builders import processor_request


class RecordingCursor:
    """Collect the statements the DDL issues, whitespace-normalized."""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.parameters: list[object] = []

    def execute(self, sql: str, parameters: object = None) -> None:
        self.statements.append(" ".join(sql.split()))
        self.parameters.append(parameters)

    def fetchone(self) -> tuple[object, ...]:
        # The counter seeding asks whether it already ran; say yes so the
        # schema pass stays a pure DDL recording.
        return (True,)

    def __enter__(self) -> RecordingCursor:
        return self

    def __exit__(self, *_arguments: object) -> bool:
        return False


def test_processor_request_has_no_virtual_partition_state() -> None:
    assert "partition_id" not in ProcessorRequest.model_fields
    assert (
        "partition_count"
        not in inspect.signature(ProcessorRequest.from_http).parameters
    )


def test_postgres_schema_only_touches_partition_state_to_remove_it() -> None:
    """Run the DDL and check the statements it actually issues.

    The virtual partition column is gone from the queue, so no statement may
    recreate it, and the payload strip has to run before the column drop -- a
    surviving ``partition_id`` JSON field would make the strict
    ``ProcessorRequest`` model reject an existing queue row on read.
    """

    cursor = RecordingCursor()
    create_postgres_schema(cursor)

    strip_queue_payload = (
        "UPDATE gpu_fault_processor_queue SET payload=payload - 'partition_id' "
        "WHERE payload ? 'partition_id'"
    )
    strip_object_payload = (
        "UPDATE gpu_fault_objects SET payload=payload - 'partition_id' "
        "WHERE kind='processor_request' AND payload ? 'partition_id'"
    )
    drop_column = (
        "ALTER TABLE gpu_fault_processor_queue DROP COLUMN IF EXISTS partition_id"
    )
    removals = [
        strip_queue_payload,
        strip_object_payload,
        "DROP INDEX IF EXISTS gpu_fault_processor_queue_claim",
        "DROP INDEX IF EXISTS gpu_fault_processor_queue_partition_claim",
        drop_column,
    ]
    for statement in removals:
        assert statement in cursor.statements, (
            f"the schema no longer removes legacy partition state: {statement}"
        )
    recreated = [
        statement
        for statement in cursor.statements
        if "partition_id" in statement and statement not in removals
    ]
    assert recreated == [], "the schema reintroduced virtual partition state"
    assert cursor.statements.index(strip_queue_payload) < cursor.statements.index(
        drop_column
    ), "the queue payload strip has to run before the column is dropped"


def test_processor_queue_writes_carry_no_partition_state() -> None:
    """The queue INSERT the mixin composes must not mention a partition.

    The statement is built from a column tuple and a placeholder count at call
    time, so this checks the SQL that reaches the database rather than the way
    the module happens to be written.
    """

    class Queue(PostgresProcessorStorageMixin):
        def __init__(self, cursor: RecordingCursor) -> None:
            self._db = type("Db", (), {"cursor": lambda _self: cursor})()

    cursor = RecordingCursor()
    request = processor_request(
        "/v1/gpu-events/xid", body=b'{"node_id":"node-a","xid":79}'
    )
    Queue(cursor)._put_processor_queue(request)

    (statement,) = cursor.statements
    assert "partition" not in statement, statement
    payloads = [
        parameter
        for parameter in cursor.parameters[0]
        if isinstance(parameter, str) and parameter.startswith("{")
    ]
    assert payloads, "the queue row carried no JSON payload"
    for payload in payloads:
        assert "partition_id" not in json.loads(payload)


def test_sqlite_upgrade_removes_legacy_partition_state(tmp_path) -> None:
    path = tmp_path / "processor.db"
    request = processor_request(
        "/v1/gpu-events/xid", body=b'{"node_id":"node-a","xid":79}'
    )
    store = SqliteStore(str(path))
    payload = request.model_dump(mode="json")
    payload["partition_id"] = 0
    store._db.execute(
        """
        INSERT INTO objects(kind, key, payload)
        VALUES ('processor_request', ?, ?)
        """,
        (request.request_id, json.dumps(payload)),
    )
    store.close()

    upgraded = SqliteStore(str(path))
    try:
        assert upgraded.get_processor_request(request.request_id) == request
        stored = upgraded._db.execute(
            """
            SELECT payload
            FROM objects
            WHERE kind='processor_request' AND key=?
            """,
            (request.request_id,),
        ).fetchone()[0]
    finally:
        upgraded.close()

    assert "partition_id" not in json.loads(stored)
