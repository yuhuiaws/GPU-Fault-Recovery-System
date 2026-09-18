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

# Exact v17 function from the pre-remediation source, also exercised by the
# v17 upgrade case in test_postgres_state_table_lock_order. v18 changes only
# this fence, not the physical layouts. Do not just relabel a v18 database.
V17_NATIVE_FENCE = """
CREATE OR REPLACE FUNCTION gpu_fault_native_control_state_fence()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE current_mode TEXT; expected_table TEXT;
BEGIN
    SELECT table_name INTO expected_table
    FROM gpu_fault_control_state_descriptor(TG_ARGV[0]);
    IF expected_table IS DISTINCT FROM TG_TABLE_NAME THEN
        RAISE EXCEPTION 'control-state trigger identity differs'
            USING ERRCODE='55000';
    END IF;
    current_mode := gpu_fault_control_state_lock_mode(TG_ARGV[0]);
    IF current_mode='legacy' OR
       (current_mode='dual' AND pg_trigger_depth()<2) THEN
        RAISE EXCEPTION 'dedicated control-state write is not authoritative'
            USING ERRCODE='55000';
    END IF;
    RETURN CASE WHEN TG_OP='DELETE' THEN OLD ELSE NEW END;
END
$$
"""


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


def install_v17(connection: psycopg.Connection) -> None:
    assert LATEST_POSTGRES_SCHEMA_VERSION == 18, (
        "a new schema requires review of the historical fixture and compatibility pair"
    )
    with connection.transaction():
        connection.execute(V17_NATIVE_FENCE)
        connection.execute(
            "DELETE FROM public.gpu_fault_schema_migrations WHERE version=18"
        )
        connection.execute(
            "UPDATE public.gpu_fault_schema_version SET version=17 WHERE singleton"
        )
    assert migration_history(connection) == [
        (item.version, item.name, item.checksum)
        for item in POSTGRES_SCHEMA_MIGRATIONS
        if item.version <= 17
    ], "v17 must have the complete, unchanged migration prefix"


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
    install_v17(connection)
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

    proof = inspect_database(connection, 18, allow_schema_upgrade=True)

    assert proof == {
        "safe": True,
        "database_state": "initialized",
        "schema_version": 17,
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
    install_v17(connection)
    before = retained_rows(connection)

    proof = inspect_database(connection, 18, allow_schema_upgrade=True)

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
    install_v17(connection)
    before = retained_rows(connection)

    proof = inspect_database(connection, 18, allow_schema_upgrade=True)

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
    install_v17(connection)

    proof = inspect_database(connection, 18, allow_schema_upgrade=True)

    assert proof["safe"] is False, "unknown and null statuses cannot prove quiescence"
    assert proof["blockers"][kind] == 1, "the malformed physical row must be counted"


@pytest.mark.parametrize("drift", ["missing", "name", "checksum", "extra"])
def test_only_the_exact_contiguous_v17_migration_prefix_is_accepted(
    database: PrivateDatabase, drift: str
) -> None:
    from psycopg import sql

    connection = database.connection
    install_v17(connection)
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
                "UPDATE public.gpu_fault_schema_migrations SET {}=%s WHERE version=17"
            ).format(sql.Identifier(drift)),
            ("0" * 64 if drift == "checksum" else "unrecognized-migration",),
        )
    before = migration_history(connection)

    with pytest.raises(ValueError, match="migration history differs"):
        inspect_database(connection, 18, allow_schema_upgrade=True)

    assert migration_history(connection) == before, "the probe must not repair drift"


@pytest.mark.parametrize(
    ("observed", "required", "allowed", "message"),
    [
        (16, 18, True, "schema version differs"),
        (19, 18, True, "schema version differs"),
        (17, 18, False, "schema version differs"),
        (17, 17, True, "probe image schema identity differs"),
    ],
)
def test_unreviewed_or_unapproved_schema_markers_fail_closed(
    database: PrivateDatabase, observed: int, required: int, allowed: bool, message: str
) -> None:
    connection = database.connection
    install_v17(connection)
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
    install_v17(connection)
    connection.execute(
        sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(
            sql.Identifier("public", table)
        )
    )

    with pytest.raises(
        ValueError, match="unrecognized table definition or execution policy"
    ):
        inspect_database(connection, 18, allow_schema_upgrade=True)


@pytest.mark.parametrize("drift", ["view", "inheritance", "domain", "generated"])
def test_retained_proof_requires_plain_builtin_heap_columns(
    database: PrivateDatabase, drift: str
) -> None:
    connection = database.connection
    install_v17(connection)
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
        inspect_database(connection, 18, allow_schema_upgrade=True)


def test_missing_storage_mode_cannot_hide_a_retained_heap(
    database: PrivateDatabase,
) -> None:
    connection = database.connection
    configure_mode(connection, "dedicated")
    seed_records(connection)
    install_v17(connection)
    connection.execute(
        "DELETE FROM public.gpu_fault_control_state_modes WHERE kind='workflow'"
    )

    with pytest.raises(ValueError, match="unknown control-state storage mode"):
        inspect_database(connection, 18, allow_schema_upgrade=True)


def test_preensure_probe_never_executes_retained_projection_functions(
    database: PrivateDatabase,
) -> None:
    import psycopg
    from psycopg import sql

    connection = database.connection
    configure_mode(connection, "dedicated")
    seed_records(connection)
    install_v17(connection)
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

    proof = inspect_database(connection, 18, allow_schema_upgrade=True)

    assert proof["safe"] is True, (
        "pre-ensure inspection must use real heaps, not execute retained projections"
    )
    assert proof["schema_ensure_required"] is True, (
        "physical safety is not permission to start business Pods on old functions"
    )


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_explicit_v18_ensure_precedes_current_business_proof_and_preserves_rows(
    database: PrivateDatabase, mode: str
) -> None:
    connection = database.connection
    configure_mode(connection, mode)
    seed_records(connection)
    install_v17(connection)
    before = retained_rows(connection)
    history = migration_history(connection)

    with pytest.raises(RuntimeError, match="schema migration history mismatch"):
        with closing(database.open_store(False)):
            pytest.fail("a business Store must not start on retained v17")
    assert retained_rows(connection) == before, (
        "refused business startup must not mutate data"
    )
    preensure = inspect_database(connection, 18, allow_schema_upgrade=True)
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
    ], "ensure must append exactly the candidate v18 migration"
    assert migration_history(connection)[:-1] == history, (
        "v1-v17 history must stay unchanged"
    )
    assert retained_rows(connection) == before, (
        "ensure must preserve retained data, selected modes and migration checkpoints"
    )
    proof = inspect_database(connection, 18)
    assert proof == {
        "safe": True,
        "database_state": "initialized",
        "schema_version": 18,
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


def test_fixture_has_real_v17_fence_and_ensure_installs_v18_reset_behavior(
    database: PrivateDatabase,
) -> None:
    import psycopg

    connection = database.connection
    configure_mode(connection, "dual")
    seed_records(connection)
    for kind in KINDS:
        set_state_table_mode(connection, kind, "legacy", expected_mode="dual")
    install_v17(connection)
    before = retained_rows(connection)
    with pytest.raises(
        psycopg.errors.ObjectNotInPrerequisiteState, match="not authoritative"
    ):
        with connection.transaction():
            connection.execute("SET LOCAL gpu_fault.control_state_reset='workflow'")
            connection.execute(
                "DELETE FROM public.gpu_fault_workflows WHERE request_id='retained-workflow'"
            )
    assert retained_rows(connection) == before, (
        "v17 must really refuse the v18-only native reset exception"
    )

    with closing(database.open_store(True)):
        pass

    assert retained_rows(connection) == before, (
        "ensure alone must not reset the stale copy"
    )
    activated = set_state_table_mode(
        connection, "workflow", "dual", expected_mode="legacy"
    )
    assert activated["dedicated_rows"] == 0, (
        "v18 must install the reset fence before explicit mode activation can delete the copy"
    )
    assert physical_counts(connection, "workflow") == (1, 0), (
        "the explicit reset must preserve authoritative legacy history"
    )


def test_retained_probe_works_inside_a_read_only_session(
    database: PrivateDatabase,
) -> None:
    import psycopg

    connection = database.connection
    seed_records(connection)
    install_v17(connection)
    before = retained_rows(connection)
    connection.execute("SET default_transaction_read_only=on")
    try:
        proof = inspect_database(connection, 18, allow_schema_upgrade=True)
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
