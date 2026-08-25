from __future__ import annotations

import inspect
import json
from pathlib import Path

from gpu_fault.processor import ProcessorRequest
from gpu_fault.store import SqliteStore
from tests._builders import processor_request

ROOT = Path(__file__).resolve().parents[2]


def test_processor_request_has_no_virtual_partition_state() -> None:
    assert "partition_id" not in ProcessorRequest.model_fields
    assert (
        "partition_count"
        not in inspect.signature(ProcessorRequest.from_http).parameters
    )


def test_postgres_queue_only_mentions_partition_for_migration() -> None:
    storage = (ROOT / "src/gpu_fault/store/postgres/processor_storage.py").read_text()
    ddl = (ROOT / "src/gpu_fault/store/postgres/ddl.py").read_text()

    assert "partition_id" not in storage
    assert "partition_id INTEGER" not in ddl
    assert "'partition', NEW.partition_id" not in ddl
    assert "ADD COLUMN IF NOT EXISTS partition_id" not in ddl
    assert "DROP INDEX IF EXISTS gpu_fault_processor_queue_claim" in ddl
    assert "DROP COLUMN IF EXISTS partition_id" in ddl


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
