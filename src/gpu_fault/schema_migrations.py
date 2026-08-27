from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from typing import Protocol

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
