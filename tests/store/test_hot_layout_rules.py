"""Pure rules of the v23 hot-table layout (no database)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.schema_migrations import POSTGRES_SCHEMA_MIGRATIONS
from gpu_fault.store.postgres import hot_layout
from gpu_fault.store.postgres.ddl import declared_index_names
from gpu_fault.store.postgres.ddl_processor_layout import (
    LAYOUT_DETACHED,
    LAYOUT_INLINE,
    METRIC_LATEST_CLUSTERS_TABLE,
    RETIRED_QUEUE_INDEXES,
)
from gpu_fault.store.postgres.ddl_storage_parameters import HOT_TABLE_STORAGE_PARAMETERS
from gpu_fault.store.postgres.hot_table_bloat import REINDEX_TABLES
from gpu_fault.store.postgres.index_builder import declared_index_statements


def test_the_retired_queue_indexes_are_no_longer_declared() -> None:
    declared = declared_index_names()
    assert set(RETIRED_QUEUE_INDEXES).isdisjoint(declared), sorted(
        set(RETIRED_QUEUE_INDEXES) & declared
    )
    assert {
        "gpu_fault_processor_queue_claim_window",
        "gpu_fault_processor_queue_path_claim_window",
        "gpu_fault_processor_queue_lane_rows",
        "gpu_fault_processor_queue_cluster_rows",
        "gpu_fault_processor_queue_scope_keys",
    } <= declared


def test_no_queue_index_references_status_lease_columns_or_updated_at() -> None:
    """The HOT property: an UPDATE is HOT only when no indexed column (key,
    expression or partial predicate) changes, and a claim changes exactly the
    columns listed here."""

    blocking = (
        "status",
        "lease_owner",
        "leader_epoch",
        "lease_token",
        "lease_expires_at",
        "updated_at",
    )
    for name, statement in declared_index_statements().items():
        if "ON gpu_fault_processor_queue" not in statement:
            continue
        body = statement.split("ON gpu_fault_processor_queue", 1)[1]
        for column in blocking:
            assert column not in body, (name, column, body)


class _RecordingCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, query: str, params: object | None = None) -> None:
        self.statements.append(" ".join(query.split()))


def test_the_v23_migration_is_the_last_and_retires_the_v22_set() -> None:
    latest = POSTGRES_SCHEMA_MIGRATIONS[-1]
    assert latest.version == 23
    assert (
        latest.name
        == "processor-queue-detached-payloads-replays-and-cluster-partitions"
    )
    assert latest.apply is not None
    cursor = _RecordingCursor()
    latest.apply(cursor)
    dropped = [
        item for item in cursor.statements if item.startswith("DROP INDEX IF EXISTS ")
    ]
    assert [item.removeprefix("DROP INDEX IF EXISTS ") for item in dropped] == list(
        RETIRED_QUEUE_INDEXES
    )
    backfill = [item for item in cursor.statements if item.startswith("UPDATE ")]
    assert (
        len(backfill) == 1
        and "scope_keys" in backfill[0]
        and "path IS NULL" in backfill[0]
    ), "the callback backfills the new columns of rows written before the trigger"


def test_cluster_partition_names_are_identifier_safe_and_deterministic() -> None:
    name = hot_layout.cluster_partition_name("hp-cluster-a")
    assert name == hot_layout.cluster_partition_name("hp-cluster-a")
    assert name.startswith(METRIC_LATEST_CLUSTERS_TABLE + "_c_"), name
    assert len(name) <= 63 and name.replace("_", "").isalnum()
    assert name != hot_layout.cluster_partition_name("gpu-eks-cluster")


def test_replay_buckets_are_five_minutes_and_their_names_round_trip() -> None:
    at = datetime(2026, 1, 2, 3, 4, 59, tzinfo=timezone.utc)
    start = hot_layout.replay_bucket_start(at)
    assert start % hot_layout.REPLAY_BUCKET_SECONDS == 0
    assert start <= int(at.timestamp()) < start + hot_layout.REPLAY_BUCKET_SECONDS
    name = hot_layout.replay_partition_name(start)
    assert hot_layout.replay_bucket_bounds(name) == (start, start + 300)
    assert (
        hot_layout.replay_bucket_bounds("gpu_fault_processor_replays_default") is None
    )
    assert hot_layout.REPLAY_AHEAD_SECONDS >= 2 * hot_layout.REPLAY_BUCKET_SECONDS


@pytest.mark.parametrize("value", ["inline", "DETACHED", " detached "])
def test_layout_modes_normalise_and_the_rest_are_refused(value: str) -> None:
    assert hot_layout.normalize_layout_mode(value) in {LAYOUT_INLINE, LAYOUT_DETACHED}
    with pytest.raises(ValueError):
        hot_layout.normalize_layout_mode("partitioned")


def test_metric_latest_table_follows_the_mode() -> None:
    assert (
        hot_layout.metric_latest_table(LAYOUT_INLINE) == "gpu_fault_gpu_metric_latest"
    )
    assert (
        hot_layout.metric_latest_table(LAYOUT_DETACHED) == METRIC_LATEST_CLUSTERS_TABLE
    )


def test_the_payloads_table_keeps_bodies_inline_and_partitioned_parents_carry_none() -> (
    None
):
    payloads = HOT_TABLE_STORAGE_PARAMETERS["gpu_fault_processor_payloads"]
    assert payloads["toast_tuple_target"] == "8160"
    assert payloads["autovacuum_vacuum_scale_factor"] == "0"
    # A partitioned parent refuses reloptions; its partitions are created with
    # the flat table's values instead (hot_layout.ensure_cluster_partition).
    assert METRIC_LATEST_CLUSTERS_TABLE not in HOT_TABLE_STORAGE_PARAMETERS
    assert "gpu_fault_processor_replays" not in HOT_TABLE_STORAGE_PARAMETERS
    assert "fillfactor=70" in hot_layout.METRIC_LATEST_PARTITION_OPTIONS


def test_compaction_measures_the_new_tables_but_never_the_replays() -> None:
    assert "gpu_fault_processor_payloads" in REINDEX_TABLES
    assert METRIC_LATEST_CLUSTERS_TABLE in REINDEX_TABLES
    assert "gpu_fault_processor_replays" not in REINDEX_TABLES


def test_retention_cutoff_is_now_minus_retention() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert hot_layout.replay_retention_cutoff(
        now, timedelta(seconds=600)
    ) == now - timedelta(seconds=600)
