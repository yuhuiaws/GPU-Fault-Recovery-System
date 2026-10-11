"""Native retained-schema proof on OID-owned, privately allocated PostgreSQL."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from gpu_fault.schema_migrations import (
    LATEST_POSTGRES_SCHEMA_VERSION,
    POSTGRES_SCHEMA_MIGRATIONS,
)
from gpu_fault.state_table_migrate import backfill_state_table, set_state_table_mode
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.state_table_payload import STATE_LAYOUTS
from gpu_fault.store.postgres.state_table_storage import put_state_record
from gpu_fault.store.shared.primitives import state_key
from gpu_fault.telemetry_models import WorkloadObservationState
from gpu_fault.watcher import WorkloadPhase
from gpu_fault_release.regional_release_store_probe import inspect_database
from tests._builders import (
    attempt_observation,
    fault_incident,
    workflow_request,
    workflow_step,
)

if TYPE_CHECKING:
    import psycopg

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("GPU_FAULT_TEST_POSTGRES_URL"),
        reason="requires an explicitly allocated, serial local PostgreSQL grant",
    ),
    # The existing fixture only inspects the granted container's identity.
    pytest.mark.allows_cluster_binaries("docker"),
]

NOW = datetime(2026, 9, 14, 12, tzinfo=UTC)
KINDS = ("workflow", "remote_command")
PHYSICAL_COPIES = {"legacy": (1, 0), "dual": (1, 1), "dedicated": (0, 1)}

# A v22 database differs from the current one in the processor queue's layout
# only: schema v23 adds the ``path`` / ``scope_keys`` columns with their BEFORE
# INSERT fill trigger, the payloads / replays / cluster-partitioned metric-latest
# tables and the layout mode row, and replaces the queue's eight
# status-predicated indexes with five that reference neither ``status`` nor the
# lease columns. The workflow / remote-command records this proof counts, their
# tables and their modes are untouched, so the retained rows below are exactly
# what v22 wrote. The fixture rebuilds the v22 queue catalog (indexes, no new
# columns, no new tables); the pre-ensure probe never reads the queue's notify
# function, whose v23 body is the one remnant the fixture leaves in place.
V22_QUEUE_INDEXES = (
    "CREATE INDEX gpu_fault_processor_queue_priority_claim ON gpu_fault_processor_queue "
    "(priority, created_at, request_id) WHERE status IN ('PENDING', 'LEASED')",
    "CREATE INDEX gpu_fault_processor_queue_cluster ON gpu_fault_processor_queue "
    "(cluster_id, status, created_at) WHERE status IN ('PENDING', 'LEASED')",
    "CREATE INDEX gpu_fault_processor_queue_path_priority_claim ON gpu_fault_processor_queue "
    "((payload->>'path'), priority, created_at, request_id) WHERE status IN ('PENDING', 'LEASED')",
    "CREATE INDEX gpu_fault_processor_queue_lane ON gpu_fault_processor_queue "
    "(ordering_key, status, lease_expires_at) WHERE status IN ('PENDING', 'LEASED')",
    "CREATE INDEX gpu_fault_processor_queue_completed ON gpu_fault_processor_queue "
    "(updated_at) WHERE status='COMPLETED'",
    "CREATE INDEX gpu_fault_processor_queue_incomplete_cluster ON gpu_fault_processor_queue "
    "(cluster_id, priority) WHERE status IN ('PENDING', 'LEASED')",
    "CREATE INDEX gpu_fault_processor_queue_correlation_scopes ON gpu_fault_processor_queue "
    "USING GIN ((payload->'correlation_scope_keys')) WHERE status IN ('PENDING', 'LEASED')",
    "CREATE INDEX gpu_fault_processor_queue_available ON gpu_fault_processor_queue "
    "(status, not_before, priority, created_at, request_id)",
)
V23_QUEUE_INDEXES = (
    "gpu_fault_processor_queue_claim_window",
    "gpu_fault_processor_queue_path_claim_window",
    "gpu_fault_processor_queue_lane_rows",
    "gpu_fault_processor_queue_cluster_rows",
    "gpu_fault_processor_queue_scope_keys",
)
V23_TABLES = (
    "gpu_fault_processor_payloads",
    "gpu_fault_processor_replays",
    "gpu_fault_gpu_metric_latest_clusters",
    "gpu_fault_hot_layout_mode",
)


@dataclass(frozen=True)
class PrivateDatabase:
    connection: psycopg.Connection = field(repr=False)
    open_store: Callable[[bool], PostgresStore] = field(repr=False)


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> Iterator[PrivateDatabase]:
    import psycopg
    from psycopg.conninfo import conninfo_to_dict

    from tests.regional import _cov95_notify008_postgres as private_postgres

    shared = conninfo_to_dict(os.environ["GPU_FAULT_TEST_POSTGRES_URL"])
    binding = {name: value for name, value in shared.items() if name != "dbname"}

    def guarded_store(dsn: str, **options: Any) -> PostgresStore:
        target = conninfo_to_dict(dsn)
        if (
            target.get("dbname") == shared.get("dbname")
            or not re.fullmatch(r"notify008_[0-9a-f]{16}", target.get("dbname", ""))
            or {name: value for name, value in target.items() if name != "dbname"}
            != binding
        ):
            pytest.fail("Store factory refused a database outside its private grant")
        return PostgresStore(dsn, **options)

    monkeypatch.setattr(private_postgres, "PostgresStore", guarded_store)
    with private_postgres.isolated_database() as factory:
        with psycopg.connect(factory.url, autocommit=True) as connection:
            connection.execute("SET statement_timeout='10s'")
            connection.execute("SET lock_timeout='3s'")

            def open_store(initialize_schema: bool) -> PostgresStore:
                return guarded_store(
                    factory.url,
                    initialize_schema=initialize_schema,
                    pool_min_size=1,
                    pool_max_size=2,
                    pool_timeout_seconds=2,
                )

            yield PrivateDatabase(connection, open_store)


def migration_history(connection: psycopg.Connection) -> list[tuple]:
    return connection.execute(
        "SELECT version,name,checksum FROM public.gpu_fault_schema_migrations "
        "ORDER BY version"
    ).fetchall()


def install_v22(connection: psycopg.Connection) -> None:
    assert LATEST_POSTGRES_SCHEMA_VERSION == 23, (
        "a new schema requires review of the historical fixture and compatibility pair"
    )
    with connection.transaction():
        for name in V23_QUEUE_INDEXES:
            connection.execute(f"DROP INDEX IF EXISTS public.{name}")
        connection.execute(
            "DROP TRIGGER IF EXISTS gpu_fault_processor_queue_fill_columns_trigger "
            "ON public.gpu_fault_processor_queue"
        )
        connection.execute(
            "DROP FUNCTION IF EXISTS public.gpu_fault_processor_queue_fill_columns()"
        )
        connection.execute(
            "ALTER TABLE public.gpu_fault_processor_queue DROP COLUMN IF EXISTS path, "
            "DROP COLUMN IF EXISTS scope_keys"
        )
        for table in V23_TABLES:
            connection.execute(f"DROP TABLE IF EXISTS public.{table} CASCADE")
        for statement in V22_QUEUE_INDEXES:
            connection.execute(statement)
        connection.execute(
            "DELETE FROM public.gpu_fault_schema_migrations WHERE version>22"
        )
        connection.execute(
            "UPDATE public.gpu_fault_schema_version SET version=22 WHERE singleton"
        )
    assert migration_history(connection) == [
        (item.version, item.name, item.checksum)
        for item in POSTGRES_SCHEMA_MIGRATIONS
        if item.version <= 22
    ], "v22 must have the complete, unchanged migration prefix"


def configure_mode(
    connection: psycopg.Connection, mode: str, *, kinds: tuple[str, ...] = KINDS
) -> None:
    if mode == "legacy":
        return
    for kind in kinds:
        set_state_table_mode(connection, kind, "dual", expected_mode="legacy")
        if mode == "dedicated":
            backfill_state_table(connection, kind)
            set_state_table_mode(
                connection,
                kind,
                "dedicated",
                expected_mode="dual",
                confirm_dedicated=True,
            )


def seed_records(
    connection: psycopg.Connection,
    *,
    workflow_status: WorkflowStatus = WorkflowStatus.SUCCEEDED,
    command_status: RemoteCommandStatus = RemoteCommandStatus.SUCCEEDED,
) -> None:
    incident = fault_incident(
        "retained-incident",
        "retained-event",
        state=IncidentState.QUARANTINED,
        created_at=NOW,
        updated_at=NOW,
    )
    workflow = workflow_request(
        "retained-workflow",
        incident.incident_id,
        status=workflow_status,
        created_at=NOW,
        updated_at=NOW,
    )
    command = RemoteActionCommand(
        command_id="retained-command",
        cluster_id="retained-test",
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        fencing_token=workflow.fencing_token,
        step_index=0,
        step=workflow_step(WorkflowOperation.FREEZE_EVIDENCE),
        idempotency_key="retained-command/0",
        workflow=workflow,
        incident=incident,
        status=command_status,
        created_at=NOW,
        updated_at=NOW,
    )
    connection.execute(
        "INSERT INTO public.gpu_fault_objects(kind,key,payload) "
        "VALUES ('incident',%s,%s::jsonb)",
        (incident.incident_id, incident.model_dump_json()),
    )
    for kind, key, record in (
        ("workflow", workflow.request_id, workflow),
        ("remote_command", command.command_id, command),
    ):
        assert put_state_record(connection, kind, key, record) is True, (
            "the production writer must create the retained fixture"
        )


def physical_counts(connection: psycopg.Connection, kind: str) -> tuple[int, int]:
    from psycopg import sql

    legacy = connection.execute(
        "SELECT count(*) FROM ONLY public.gpu_fault_objects WHERE kind=%s", (kind,)
    ).fetchone()[0]
    native = connection.execute(
        sql.SQL("SELECT count(*) FROM ONLY {}").format(
            sql.Identifier("public", STATE_LAYOUTS[kind].table)
        )
    ).fetchone()[0]
    return legacy, native


def retained_rows(connection: psycopg.Connection) -> dict[str, list]:
    from psycopg import sql

    tables = {
        "gpu_fault_objects": ("kind", "key"),
        "gpu_fault_workflows": ("request_id",),
        "gpu_fault_remote_commands": ("command_id",),
        "gpu_fault_attempt_observations": ("key",),
        "gpu_fault_control_state_modes": ("kind",),
    }
    return {
        table: connection.execute(
            sql.SQL("SELECT pg_catalog.to_jsonb(r) FROM ONLY {} r ORDER BY {}").format(
                sql.Identifier("public", table),
                sql.SQL(",").join(map(sql.Identifier, keys)),
            )
        ).fetchall()
        for table, keys in tables.items()
    }


def seed_observation(
    connection: psycopg.Connection, storage: str, phase: str | None
) -> None:
    observation = attempt_observation(
        "retained-job",
        "retained-attempt",
        NOW,
        cluster_id="retained-test",
        workload_phase=WorkloadPhase.RUNNING,
    )
    state = WorkloadObservationState(first_observed_at=NOW, observation=observation)
    payload = state.model_dump(mode="json")
    assert "phase" not in payload["observation"], (
        "the fixture must retain the actual workload_phase serialization"
    )
    payload["observation"]["workload_phase"] = phase
    key = state_key((observation.cluster_id, observation.attempt_id))
    if storage in {"legacy", "both"}:
        connection.execute(
            "INSERT INTO public.gpu_fault_objects(kind,key,payload) "
            "VALUES ('attempt_observation',%s,%s::jsonb)",
            (key, json.dumps(payload)),
        )
    if storage in {"native", "both"}:
        connection.execute(
            "INSERT INTO public.gpu_fault_attempt_observations "
            "(key,cluster_id,attempt_id,observed_at,payload) "
            "VALUES (%s,%s,%s,%s,%s::jsonb)",
            (
                key,
                observation.cluster_id,
                observation.attempt_id,
                observation.observed_at,
                json.dumps(payload),
            ),
        )


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_retained_v17_terminal_rows_are_safe_in_real_storage_modes(
    database: PrivateDatabase, mode: str
) -> None:
    from psycopg import sql

    connection = database.connection
    configure_mode(connection, mode)
    seed_records(connection)
    install_v22(connection)
    before = migration_history(connection)
    for kind in KINDS:
        assert physical_counts(connection, kind) == PHYSICAL_COPIES[mode], (
            "the fixture must exercise the selected physical layout"
        )
        layout = STATE_LAYOUTS[kind]
        if mode != "legacy":
            has_status = connection.execute(
                sql.SQL("SELECT {} ? 'status' FROM ONLY {}").format(
                    sql.Identifier(layout.payload_column),
                    sql.Identifier("public", layout.table),
                )
            ).fetchone()[0]
            assert has_status is False, (
                "native status is a column, not a shadow value in its immutable payload"
            )

    proof = inspect_database(
        connection, LATEST_POSTGRES_SCHEMA_VERSION, allow_schema_upgrade=True
    )

    assert proof == {
        "safe": True,
        "database_state": "initialized",
        "schema_version": 22,
        "schema_ensure_required": True,
        "blockers": {"workflow": 0, "remote_command": 0, "observation": 0},
    }, "terminal retained rows must not be treated as missing-status blockers"
    assert migration_history(connection) == before, "the proof must remain read-only"
    assert connection.execute(
        "SELECT payload->>'state' FROM ONLY public.gpu_fault_objects "
        "WHERE kind='incident' AND key='retained-incident'"
    ).fetchone() == ("QUARANTINED",), "the proof must preserve quarantine history"


@pytest.mark.parametrize("workflow_mode", ["legacy", "dual", "dedicated"])
@pytest.mark.parametrize("command_mode", ["legacy", "dual", "dedicated"])
def test_blockers_count_real_physical_copies_in_independent_modes(
    database: PrivateDatabase, workflow_mode: str, command_mode: str
) -> None:
    connection = database.connection
    modes = {"workflow": workflow_mode, "remote_command": command_mode}
    for kind, mode in modes.items():
        configure_mode(connection, mode, kinds=(kind,))
    seed_records(
        connection,
        workflow_status=WorkflowStatus.BLOCKED,
        command_status=RemoteCommandStatus.LEASED,
    )
    install_v22(connection)
    before = retained_rows(connection)

    proof = inspect_database(
        connection, LATEST_POSTGRES_SCHEMA_VERSION, allow_schema_upgrade=True
    )

    assert proof["safe"] is False, "open or blocked work must refuse retained startup"
    assert proof["blockers"] == {
        **{kind: sum(PHYSICAL_COPIES[mode]) for kind, mode in modes.items()},
        "observation": 0,
    }, "the proof must count physical copies, including both during dual migration"
    assert retained_rows(connection) == before, (
        "checking blockers must not retire, settle or change any retained row"
    )


@pytest.mark.parametrize("storage", ["legacy", "native", "both"])
@pytest.mark.parametrize(
    ("phase", "blocked"),
    [
        ("SUCCEEDED", False),
        ("FAILED", False),
        ("STOPPED", False),
        ("RUNNING", True),
        ("PENDING", True),
        ("UNKNOWN", True),
        (None, True),
    ],
)
def test_real_observation_workload_phase_is_checked_in_both_heaps(
    database: PrivateDatabase, storage: str, phase: str | None, blocked: bool
) -> None:
    connection = database.connection
    seed_observation(connection, storage, phase)
    install_v22(connection)
    before = retained_rows(connection)

    proof = inspect_database(
        connection, LATEST_POSTGRES_SCHEMA_VERSION, allow_schema_upgrade=True
    )

    assert proof["blockers"] == {
        "workflow": 0,
        "remote_command": 0,
        "observation": (2 if storage == "both" else 1) if blocked else 0,
    }, "the real serialized workload_phase, not an invented phase field, is decisive"
    assert proof["safe"] is not blocked, (
        "terminal observations are retained history; unknown or active ones are blockers"
    )
    assert retained_rows(connection) == before, (
        "the probe must not rewrite observations"
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("status", ["UNKNOWN", None])
def test_unknown_legacy_status_is_never_accepted_as_terminal(
    database: PrivateDatabase, kind: str, status: str | None
) -> None:
    connection = database.connection
    seed_records(connection)
    connection.execute(
        "UPDATE public.gpu_fault_objects SET payload=jsonb_set(payload,'{status}',%s::jsonb) "
        "WHERE kind=%s",
        (json.dumps(status), kind),
    )
    install_v22(connection)

    proof = inspect_database(
        connection, LATEST_POSTGRES_SCHEMA_VERSION, allow_schema_upgrade=True
    )

    assert proof["safe"] is False, "unknown and null statuses cannot prove quiescence"
    assert proof["blockers"][kind] == 1, "the malformed physical row must be counted"


@pytest.mark.parametrize("drift", ["missing", "name", "checksum", "extra"])
def test_only_the_exact_contiguous_v22_migration_prefix_is_accepted(
    database: PrivateDatabase, drift: str
) -> None:
    from psycopg import sql

    connection = database.connection
    install_v22(connection)
    if drift == "missing":
        connection.execute(
            "DELETE FROM public.gpu_fault_schema_migrations WHERE version=7"
        )
    elif drift == "extra":
        migration = POSTGRES_SCHEMA_MIGRATIONS[-1]
        connection.execute(
            "INSERT INTO public.gpu_fault_schema_migrations(version,name,checksum,applied_at) "
            "VALUES (%s,%s,%s,%s)",
            (migration.version, migration.name, migration.checksum, NOW),
        )
    else:
        connection.execute(
            sql.SQL(
                "UPDATE public.gpu_fault_schema_migrations SET {}=%s WHERE version=22"
            ).format(sql.Identifier(drift)),
            ("0" * 64 if drift == "checksum" else "unrecognized-migration",),
        )
    before = migration_history(connection)

    with pytest.raises(ValueError, match="migration history differs"):
        inspect_database(
            connection, LATEST_POSTGRES_SCHEMA_VERSION, allow_schema_upgrade=True
        )

    assert migration_history(connection) == before, "the probe must not repair drift"


@pytest.mark.parametrize(
    ("observed", "required", "allowed", "message"),
    [
        (21, 23, True, "schema version differs"),
        (24, 23, True, "schema version differs"),
        (22, 23, False, "schema version differs"),
        (22, 22, True, "probe image schema identity differs"),
    ],
)
def test_unreviewed_or_unapproved_schema_markers_fail_closed(
    database: PrivateDatabase, observed: int, required: int, allowed: bool, message: str
) -> None:
    connection = database.connection
    install_v22(connection)
    # Invalid marker cases are deliberate corruption, not claimed v16/v19 fixtures.
    connection.execute(
        "UPDATE public.gpu_fault_schema_version SET version=%s WHERE singleton",
        (observed,),
    )
    with pytest.raises(ValueError, match=message):
        inspect_database(connection, required, allow_schema_upgrade=allowed)
    assert connection.execute(
        "SELECT version FROM public.gpu_fault_schema_version WHERE singleton"
    ).fetchone() == (observed,), (
        "refusal must not silently initialize the candidate schema"
    )


@pytest.mark.parametrize(
    "table",
    [
        "gpu_fault_objects",
        "gpu_fault_workflows",
        "gpu_fault_attempt_observations",
        "gpu_fault_control_state_modes",
    ],
)
def test_retained_heap_with_row_security_is_refused(
    database: PrivateDatabase, table: str
) -> None:
    from psycopg import sql

    connection = database.connection
    install_v22(connection)
    connection.execute(
        sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(
            sql.Identifier("public", table)
        )
    )

    with pytest.raises(
        ValueError, match="unrecognized table definition or execution policy"
    ):
        inspect_database(
            connection, LATEST_POSTGRES_SCHEMA_VERSION, allow_schema_upgrade=True
        )


@pytest.mark.parametrize("drift", ["view", "inheritance", "domain", "generated"])
def test_retained_proof_requires_plain_builtin_heap_columns(
    database: PrivateDatabase, drift: str
) -> None:
    connection = database.connection
    install_v22(connection)
    if drift == "view":
        connection.execute(
            "ALTER TABLE public.gpu_fault_workflows RENAME TO gpu_fault_retained_workflows"
        )
        connection.execute(
            "CREATE VIEW public.gpu_fault_workflows AS "
            "SELECT * FROM public.gpu_fault_retained_workflows"
        )
    elif drift == "inheritance":
        connection.execute(
            "CREATE TABLE public.gpu_fault_child (extra integer) "
            "INHERITS (public.gpu_fault_objects)"
        )
    elif drift == "domain":
        connection.execute("CREATE DOMAIN public.gpu_fault_custom_text AS text")
        connection.execute(
            "ALTER TABLE public.gpu_fault_objects "
            "ADD COLUMN extra public.gpu_fault_custom_text"
        )
    else:
        connection.execute(
            "ALTER TABLE public.gpu_fault_objects "
            "ADD COLUMN extra integer GENERATED ALWAYS AS (1) STORED"
        )

    with pytest.raises(
        ValueError, match="unrecognized table definition or execution policy"
    ):
        inspect_database(
            connection, LATEST_POSTGRES_SCHEMA_VERSION, allow_schema_upgrade=True
        )


def test_missing_storage_mode_cannot_hide_a_retained_heap(
    database: PrivateDatabase,
) -> None:
    connection = database.connection
    configure_mode(connection, "dedicated")
    seed_records(connection)
    install_v22(connection)
    connection.execute(
        "DELETE FROM public.gpu_fault_control_state_modes WHERE kind='workflow'"
    )

    with pytest.raises(ValueError, match="unknown control-state storage mode"):
        inspect_database(
            connection, LATEST_POSTGRES_SCHEMA_VERSION, allow_schema_upgrade=True
        )


def test_preensure_probe_never_executes_retained_projection_functions(
    database: PrivateDatabase,
) -> None:
    import psycopg
    from psycopg import sql

    connection = database.connection
    configure_mode(connection, "dedicated")
    seed_records(connection)
    install_v22(connection)
    for kind in KINDS:
        layout = STATE_LAYOUTS[kind]
        connection.execute(
            sql.SQL(
                "CREATE OR REPLACE FUNCTION {}(value {}) RETURNS JSONB "
                "LANGUAGE plpgsql STABLE PARALLEL SAFE AS $$ BEGIN "
                "RAISE EXCEPTION 'retained projection executed'; END $$"
            ).format(
                sql.Identifier("public", f"gpu_fault_{kind}_payload"),
                sql.Identifier("public", layout.table),
            )
        )
        with pytest.raises(
            psycopg.errors.RaiseException, match="retained projection executed"
        ):
            connection.execute(
                sql.SQL("SELECT payload FROM {}").format(
                    sql.Identifier("public", f"gpu_fault_{kind}_records")
                )
            ).fetchall()

    proof = inspect_database(
        connection, LATEST_POSTGRES_SCHEMA_VERSION, allow_schema_upgrade=True
    )

    assert proof["safe"] is True, (
        "pre-ensure inspection must use real heaps, not execute retained projections"
    )
    assert proof["schema_ensure_required"] is True, (
        "physical safety is not permission to start business Pods on old functions"
    )


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_explicit_v23_ensure_precedes_current_business_proof_and_preserves_rows(
    database: PrivateDatabase, mode: str
) -> None:
    connection = database.connection
    configure_mode(connection, mode)
    seed_records(connection)
    install_v22(connection)
    before = retained_rows(connection)
    history = migration_history(connection)

    # v23 declares queue indexes a v22 catalog lacks, so the fail-closed
    # startup check now stops on the missing indexes before it reaches the
    # migration history; either refusal keeps a business Pod off retained v22.
    with pytest.raises(
        RuntimeError, match="indexes are missing|schema migration history mismatch"
    ):
        with closing(database.open_store(False)):
            pytest.fail("a business Store must not start on retained v22")
    assert retained_rows(connection) == before, (
        "refused business startup must not mutate data"
    )
    preensure = inspect_database(
        connection, LATEST_POSTGRES_SCHEMA_VERSION, allow_schema_upgrade=True
    )
    assert preensure["safe"] is True and preensure["schema_ensure_required"] is True, (
        "the retained proof must distinguish safe physical rows from candidate schema readiness"
    )
    assert migration_history(connection) == history, (
        "the retained proof is not an ensure Job"
    )

    with closing(database.open_store(True)):
        pass

    assert migration_history(connection) == [
        (item.version, item.name, item.checksum) for item in POSTGRES_SCHEMA_MIGRATIONS
    ], "ensure must append exactly the candidate v22 migration"
    assert migration_history(connection)[:-1] == history, (
        "v1-v22 history must stay unchanged"
    )
    assert retained_rows(connection) == before, (
        "ensure must preserve retained data, selected modes and migration checkpoints"
    )
    proof = inspect_database(connection, LATEST_POSTGRES_SCHEMA_VERSION)
    assert proof == {
        "safe": True,
        "database_state": "initialized",
        "schema_version": LATEST_POSTGRES_SCHEMA_VERSION,
        "schema_ensure_required": False,
        "blockers": {"workflow": 0, "remote_command": 0, "observation": 0},
    }, "candidate definition validation must succeed only after explicit ensure"
    with closing(database.open_store(False)) as business:
        assert (
            business.get_workflow("retained-workflow").status
            is WorkflowStatus.SUCCEEDED
        ), "the first candidate business read must see the retained workflow"
        assert (
            business.get_incident("retained-incident").state
            is IncidentState.QUARANTINED
        ), "reinstallation must not clear historical quarantine"


def _queue_indexes(connection: psycopg.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename='gpu_fault_processor_queue'"
        ).fetchall()
    }


def _queue_columns(connection: psycopg.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='gpu_fault_processor_queue'"
        ).fetchall()
    }


def test_fixture_has_the_real_v22_queue_catalog_and_ensure_installs_the_v23_layout(
    database: PrivateDatabase,
) -> None:
    connection = database.connection
    configure_mode(connection, "dual")
    seed_records(connection)
    install_v22(connection)
    before = retained_rows(connection)
    assert {"path", "scope_keys"}.isdisjoint(_queue_columns(connection)), (
        "the fixture must really be a v22 database (no v23 queue columns)"
    )
    v22_names = {statement.split()[2] for statement in V22_QUEUE_INDEXES}
    assert v22_names <= _queue_indexes(connection), "the v22 index set is present"
    assert set(V23_QUEUE_INDEXES).isdisjoint(_queue_indexes(connection)), (
        "a v22 catalog carries none of the v23 queue indexes"
    )
    for table in V23_TABLES:
        assert connection.execute("SELECT to_regclass(%s)", (table,)).fetchone() == (
            None,
        ), table

    with closing(database.open_store(True)):
        pass

    assert retained_rows(connection) == before, (
        "ensure must preserve retained data, selected modes and migration checkpoints"
    )
    assert {"path", "scope_keys"} <= _queue_columns(connection)
    assert set(V23_QUEUE_INDEXES) <= _queue_indexes(connection)
    assert v22_names.isdisjoint(_queue_indexes(connection)), (
        "the v23 migration callback retires the status-predicated set"
    )
    for table in V23_TABLES:
        assert connection.execute("SELECT to_regclass(%s)", (table,)).fetchone() != (
            None,
        ), table
    assert connection.execute(
        "SELECT mode FROM public.gpu_fault_hot_layout_mode WHERE singleton"
    ).fetchone() == ("inline",), (
        "an upgraded database keeps the inline layout until the operator's switch"
    )
    assert connection.execute(
        "SELECT count(*) FROM pg_trigger "
        "WHERE tgname='gpu_fault_processor_queue_fill_columns_trigger'"
    ).fetchone() == (1,)
    assert migration_history(connection)[-1][0] == LATEST_POSTGRES_SCHEMA_VERSION
    assert connection.execute(
        "SELECT version FROM public.gpu_fault_schema_version WHERE singleton"
    ).fetchone() == (LATEST_POSTGRES_SCHEMA_VERSION,)


def test_retained_probe_works_inside_a_read_only_session(
    database: PrivateDatabase,
) -> None:
    import psycopg

    connection = database.connection
    seed_records(connection)
    install_v22(connection)
    before = retained_rows(connection)
    connection.execute("SET default_transaction_read_only=on")
    try:
        proof = inspect_database(
            connection, LATEST_POSTGRES_SCHEMA_VERSION, allow_schema_upgrade=True
        )
        assert proof["safe"] is True, (
            "retained inspection must need no write privileges"
        )
        assert connection.execute("SHOW default_transaction_read_only").fetchone() == (
            "on",
        ), "the probe must not widen the session to read-write"
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            connection.execute(
                "UPDATE public.gpu_fault_schema_version SET version=18 WHERE singleton"
            )
        assert retained_rows(connection) == before, (
            "read-only proof must preserve every row"
        )
    finally:
        connection.execute("SET default_transaction_read_only=off")
