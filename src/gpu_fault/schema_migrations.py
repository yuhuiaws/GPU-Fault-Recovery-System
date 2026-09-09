from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gpu_fault.store.postgres.ddl import _create_priority_counter_function
from gpu_fault.store.postgres.ddl_processor_retry import (
    upgrade_processor_retry_schedule,
)


class MigrationCursor(Protocol):
    def execute(
        self,
        query: str,
        params: object | None = None,
    ) -> object: ...


MigrationApply = Callable[[MigrationCursor], None]
DDL_ROOT = Path(__file__).resolve().parent / "store" / "postgres"


def postgres_ddl_source_checksum() -> str:
    digest = hashlib.sha256()
    for path in sorted(DDL_ROOT.glob("ddl*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _apply_split_ddl_v4(cursor: MigrationCursor) -> None:
    cursor.execute("SELECT 1")


def _apply_remove_processor_partition_v5(
    cursor: MigrationCursor,
) -> None:
    cursor.execute("SELECT 1")


def _apply_freeze_migration_history_v6(
    cursor: MigrationCursor,
) -> None:
    cursor.execute("SELECT 1")


def _apply_processor_retry_schedule_v7(
    cursor: MigrationCursor,
) -> None:
    upgrade_processor_retry_schedule(cursor)


def _apply_claim_order_index_v10(cursor: MigrationCursor) -> None:
    # gpu_fault_processor_queue_claim_order (F-D2) follows the same three-step
    # method as v9: declared IF NOT EXISTS by the idempotent DDL, built
    # CONCURRENTLY by an operator on a live database, validated at startup.
    cursor.execute("SELECT 1")


def _drop_duplicate_claim_window_index_v11(cursor: MigrationCursor) -> None:
    # F-D2 declared ``gpu_fault_processor_queue_claim_order`` without noticing
    # that ``gpu_fault_processor_queue_priority_claim`` already had the same
    # columns and the same partial predicate. Two identical indexes cost a
    # second write amplification and let the planner pick either, so the
    # twin goes and the original stays the only declaration. Any database that
    # built the twin (staging release gates, developer databases) loses it
    # here; production never had it.
    cursor.execute("DROP INDEX IF EXISTS gpu_fault_processor_queue_claim_order")


def _apply_store_review_indexes_v12(cursor: MigrationCursor) -> None:
    # Store review 2026-09-07, items G, H1, H2, H3: seven new partial indexes
    # on gpu_fault_objects (three /metrics count covers, the all-status
    # remote-command workflow index, the workflow incident_id index, and the
    # dispatch-order index the code had assumed since F-A2a). Same three-step
    # method as v9-v11: declared IF NOT EXISTS by the idempotent DDL, built
    # CONCURRENTLY by the operator/deploy Job first, validated at startup.
    # ``gpu_fault_remote_command_workflow_all`` covers every read the open-only
    # partial ``gpu_fault_remote_command_workflow`` served, so that one goes
    # here the way v11 dropped the twin claim-window index.
    cursor.execute("DROP INDEX IF EXISTS gpu_fault_remote_command_workflow")


def _apply_control_plane_review_indexes_v13(cursor: MigrationCursor) -> None:
    # Control-plane review 2026-09-08, items G-2 / G-9 / F-9 / F-I1 / G-6:
    # seven new partial indexes on gpu_fault_objects (the all-status workflow
    # updated_at order the /metrics detail scan reads, the incident -> workflow
    # pointer and workflow -> predecessor join keys the orphan inspections
    # probe, the incident and fleet-deployment updated_at orders the archiver
    # and the retention sweep use, and the two missing marker scope GINs) plus
    # per-table autovacuum factors for the whole-row-upsert hot table. Same
    # three-step method as v9-v12: declared IF NOT EXISTS by the idempotent
    # DDL, built CONCURRENTLY by the operator/deploy Job first, validated at
    # startup -- and from this version the startup check also refuses an
    # INVALID index and a definition that drifted from the DDL.
    cursor.execute("SELECT 1")


def _apply_objects_wakeup_trigger_v14(cursor: MigrationCursor) -> None:
    # One row trigger on gpu_fault_objects publishes the two wakeup channels
    # (gpu_fault_workflow_dispatch, gpu_fault_remote_command) that let the
    # workflow dispatcher and the data-plane executor stop paying their 5 s /
    # 2 s poll on every remediation step. Same precedent as the spool trigger
    # change folded into v13 (E-8): the idempotent DDL creates the function
    # with CREATE OR REPLACE and the trigger through _ensure_trigger, so
    # --ensure-schema installs it and a current database re-runs lock-free.
    # No table or column changes; the version exists because the migration
    # registry checksums every ddl*.py file, and a wheel without the trigger
    # would poll silently against a database that has it (or the reverse).
    cursor.execute("SELECT 1")


def _apply_dispatcher_indexes_v9(cursor: MigrationCursor) -> None:
    # The three dispatcher partial indexes (F-A9) are declared IF NOT EXISTS
    # by the idempotent DDL that ``--ensure-schema`` runs before recording this
    # version; on a live database they are built CONCURRENTLY by an operator
    # first (F-J3) and the schema check refuses to start without them.
    cursor.execute("SELECT 1")


def _apply_device_event_tier_v8(cursor: MigrationCursor) -> None:
    # Device events moved from tier 0 to tier 10 (F-D1). The priority
    # counter buckets are computed inside the trigger function, so the
    # function has to be replaced on an existing schema for a tier-10 row to
    # be counted against the fault reserve; rows admitted before this
    # migration keep their tier and their bucket.
    _create_priority_counter_function(cursor)


def _apply_registry_v3(cursor: MigrationCursor) -> None:
    cursor.execute(
        """
        COMMENT ON TABLE gpu_fault_schema_migrations IS
        'Monotonic gpu-fault schema migration history'
        """
    )


@dataclass(frozen=True)
class SchemaMigration:
    version: int
    name: str
    ddl_checksum: str | None = None
    apply: MigrationApply | None = None
    legacy_checksum: str | None = None

    @property
    def checksum(self) -> str:
        if self.legacy_checksum is not None:
            return self.legacy_checksum
        apply_source = inspect.getsource(self.apply) if self.apply is not None else ""
        value = "\0".join(
            (
                str(self.version),
                self.name,
                self.ddl_checksum or "",
                apply_source,
            )
        )
        return hashlib.sha256(value.encode()).hexdigest()


POSTGRES_SCHEMA_MIGRATIONS = (
    SchemaMigration(
        version=1,
        name="baseline-idempotent-schema",
        legacy_checksum=(
            "47c2772094d164b8d018d3c8de5d8bf023c9d2b4e9a4c08c1ded51bb112c69f5"
        ),
    ),
    SchemaMigration(
        version=2,
        name="migration-history-and-priority-counters",
        legacy_checksum=(
            "39f7a4329cec008640850dcf408a88391e97a055d696156b6a8000fa4ce0f1c4"
        ),
    ),
    SchemaMigration(
        version=3,
        name="ddl-source-checksum-and-apply-callback",
        ddl_checksum=(
            "d36fd0bbd85e68ac700cb11e626511bd97e960268c86ab69088e3b724808558f"
        ),
        apply=_apply_registry_v3,
        legacy_checksum=(
            "2a82490502995bac570f3a533c53b899645630f786f260972a72c82641ac0680"
        ),
    ),
    SchemaMigration(
        version=4,
        name="split-idempotent-ddl-into-stages",
        ddl_checksum="df431a539b9c40af98effa0e327af9441f12843d5e4394e68edb7ab156a342cc",
        apply=_apply_split_ddl_v4,
        legacy_checksum=(
            "9ce8369e79e881c566bc429c72a157a0e7fe6a2557bcf2b4d050f2565db2b947"
        ),
    ),
    SchemaMigration(
        version=5,
        name="remove-obsolete-processor-partition-state",
        ddl_checksum="165e3b14bc187c877d4cede53a2c3e6c9ae27072fd9db8cea7e6553937e247de",
        apply=_apply_remove_processor_partition_v5,
        legacy_checksum=(
            "884820da9fdf5521dad40dffd8a40b1a3acf415de1871f563202eafd083ebb97"
        ),
    ),
    SchemaMigration(
        version=6,
        name="freeze-applied-history-and-refresh-current-ddl",
        ddl_checksum="165e3b14bc187c877d4cede53a2c3e6c9ae27072fd9db8cea7e6553937e247de",
        apply=_apply_freeze_migration_history_v6,
    ),
    SchemaMigration(
        version=7,
        name="processor-retry-schedule-and-lane-policy",
        ddl_checksum="4efa9f9838dff7df19006e02b371522accff8574a1e13c24531835636a971e08",
        apply=_apply_processor_retry_schedule_v7,
    ),
    SchemaMigration(
        version=8,
        name="device-event-tier-counts-in-the-fault-bucket",
        ddl_checksum="d917e42c2da8e43f75db172da9755d3512ad9c61012dc298474fc8c0e7cb6c9d",
        apply=_apply_device_event_tier_v8,
    ),
    SchemaMigration(
        version=9,
        name="dispatcher-hot-query-indexes",
        ddl_checksum="3d3485c8fcce1763a05cc951f093708084fa6e54fbbd1eb468f284114b739231",
        apply=_apply_dispatcher_indexes_v9,
    ),
    SchemaMigration(
        version=10,
        name="processor-claim-window-index",
        ddl_checksum="3cd52c8412322ecf5a41f74cfa845ee3071d7a686f98afac8fa367e5c3238237",
        apply=_apply_claim_order_index_v10,
    ),
    SchemaMigration(
        version=11,
        name="drop-duplicate-claim-window-index",
        ddl_checksum="0038a7136838d584e28786a82d82b6f6d6fb3cfa88978f6ea6e5ac224b5c9cdf",
        apply=_drop_duplicate_claim_window_index_v11,
    ),
    SchemaMigration(
        version=12,
        name="store-review-hot-query-indexes",
        ddl_checksum="37e84d0dbacd405f6eb05223c65f997e44b0be2524d7b7ed577f3ced0a222c99",
        apply=_apply_store_review_indexes_v12,
    ),
    SchemaMigration(
        version=13,
        name="control-plane-review-indexes-and-autovacuum",
        ddl_checksum="b8d1ac08876ac315c9b170691dfa87867803bf4300730018fba596277fcc2986",
        apply=_apply_control_plane_review_indexes_v13,
    ),
    SchemaMigration(
        version=14,
        name="objects-wakeup-notify-trigger",
        ddl_checksum="b87cc5f1a92677cd3e09e38d3e57659cdad0e0a2db628236778f870485382944",
        apply=_apply_objects_wakeup_trigger_v14,
    ),
)

LATEST_POSTGRES_SCHEMA_VERSION = POSTGRES_SCHEMA_MIGRATIONS[-1].version


def validate_migration_registry() -> None:
    versions = [migration.version for migration in POSTGRES_SCHEMA_MIGRATIONS]
    if versions != list(range(1, len(versions) + 1)):
        raise RuntimeError(
            "PostgreSQL schema migrations must be contiguous and start at version 1"
        )
    if len({migration.name for migration in POSTGRES_SCHEMA_MIGRATIONS}) != len(
        POSTGRES_SCHEMA_MIGRATIONS
    ):
        raise RuntimeError("PostgreSQL schema migration names must be unique")
    if any(len(migration.checksum) != 64 for migration in POSTGRES_SCHEMA_MIGRATIONS):
        raise RuntimeError("PostgreSQL schema migration checksums must be SHA-256")
    latest = POSTGRES_SCHEMA_MIGRATIONS[-1]
    actual_ddl_checksum = postgres_ddl_source_checksum()
    if latest.ddl_checksum != actual_ddl_checksum:
        raise RuntimeError(
            "PostgreSQL DDL changed without a new schema migration: "
            f"registry={latest.ddl_checksum} source={actual_ddl_checksum}"
        )


validate_migration_registry()
