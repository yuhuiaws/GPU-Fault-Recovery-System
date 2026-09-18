"""An expiry sweep must not delete a newly refreshed evidence version."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.store import PostgresStore
from gpu_fault.telemetry import EvidenceKind, RawEvidenceRecord
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires an isolated PostgreSQL test database"
)
NOW = datetime(2026, 9, 14, tzinfo=UTC)
CLUSTER = "raw-retention-race"


def retained_records() -> dict[str, RawEvidenceRecord]:
    import psycopg

    # The public evidence listing intentionally hides expired retained rows.
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        rows = connection.execute(
            "SELECT payload FROM gpu_fault_objects "
            "WHERE kind='raw_evidence' AND payload->>'cluster_id'=%s",
            (CLUSTER,),
        ).fetchall()
    records = [RawEvidenceRecord.model_validate(row[0]) for row in rows]
    return {item.record_id: item for item in records}


@pytest.fixture
def sweeper():
    from psycopg.conninfo import make_conninfo

    with closing(postgres_store_instance()) as setup:
        next(setup)
    assert POSTGRES_URL, "this fixture requires its explicitly allocated database"
    with closing(
        PostgresStore(
            make_conninfo(POSTGRES_URL, application_name="raw-evidence-race"),
            initialize_schema=False,
            pool_min_size=1,
            pool_max_size=1,
        )
    ) as store:
        try:
            yield store
        finally:
            _truncate()


@pytest.mark.parametrize("rollback", [False, True])
def test_raw_sweep_skips_refresh_and_uses_batch_quota(sweeper, rollback: bool) -> None:
    import psycopg
    from psycopg.types.json import Jsonb

    records = []
    for index, key in enumerate(("held", "expired-a", "expired-b")):
        record = RawEvidenceRecord(
            record_id=key,
            cluster_id=CLUSTER,
            node_id="node-a",
            kind=EvidenceKind.NODE_LOGS,
            observed_at=NOW - timedelta(days=2),
            ingested_at=NOW - timedelta(days=2),
            expires_at=NOW - timedelta(hours=3 - index),
            payload={"version": "original"},
        )
        sweeper.save_raw_evidence(record, max_records_per_node=100)
        records.append(record)
    refreshed = records[0].model_copy(
        update={
            "ingested_at": NOW,
            "expires_at": NOW + timedelta(days=1),
            "payload": {"version": "refreshed"},
        }
    )
    with (
        psycopg.connect(POSTGRES_URL, autocommit=True) as writer,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        pid = writer.execute(
            "SELECT pid FROM pg_stat_activity "
            "WHERE application_name='raw-evidence-race' AND datname=current_database()"
        ).fetchone()[0]
        with writer.transaction(force_rollback=rollback):
            writer.execute(
                "UPDATE gpu_fault_objects SET payload=%s "
                "WHERE kind='raw_evidence' AND payload->>'record_id'='held'",
                (Jsonb(refreshed.model_dump(mode="json")),),
            )
            sweep = executor.submit(
                sweeper.cleanup_expired_raw_evidence, now=NOW, limit=1
            )
            deadline = time.monotonic() + 5
            while not sweep.done():
                if writer.execute(
                    "SELECT pg_backend_pid()=ANY(pg_blocking_pids(%s))", (pid,)
                ).fetchone()[0]:
                    break
                assert time.monotonic() < deadline, "sweep did not reach candidates"
                time.sleep(0.01)
        assert sweep.result(timeout=5) == 1
    remaining = retained_records()
    assert set(remaining) == {"held", "expired-b"}, (
        "cleanup removed a locked version or failed to spend its deletion quota"
    )
    assert remaining["held"] == (records[0] if rollback else refreshed)
    assert sweeper.cleanup_expired_raw_evidence(now=NOW, limit=1) == 1
    if rollback:
        assert set(retained_records()) == {"expired-b"}, (
            "a rolled-back refresh must be eligible on the next sweep"
        )
        assert sweeper.cleanup_expired_raw_evidence(now=NOW, limit=1) == 1
        assert retained_records() == {}
    else:
        assert retained_records() == {"held": refreshed}
        assert sweeper.cleanup_expired_raw_evidence(now=NOW, limit=1) == 0
