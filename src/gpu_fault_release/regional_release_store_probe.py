"""Standalone read-only bootstrap proof, shipped as source to a CPU Job."""

from __future__ import annotations

import ast
import json
import os
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

MAX_RELATIONS = 256
CA_PATH = "/etc/gpu-fault/rds/ca-bundle.pem"
PROBE_ERROR_TYPES = frozenset(
    {
        "ImportError",
        "ModuleNotFoundError",
        "KeyError",
        "TypeError",
        "ValueError",
        "JSONDecodeError",
        "FileNotFoundError",
        "PermissionError",
        "OSError",
        "TimeoutError",
        "RuntimeError",
        "InterfaceError",
        "OperationalError",
        "ProgrammingError",
        "DataError",
        "IntegrityError",
        "InternalError",
        "NotSupportedError",
        "DatabaseError",
    }
)


class ProbeStage(StrEnum):
    IMPORT_DRIVER = "import_driver"
    READ_IDENTITY = "read_identity"
    BIND_CONNECTION = "bind_connection"
    CONNECT = "connect"
    INSPECT_DATABASE = "inspect_database"
    CLOSE_CONNECTION = "close_connection"
    BUILD_ENVELOPE = "build_envelope"


def compatible_retained_schema(observed: int, target: int) -> bool:
    # v18 adds a write fence; v17 has the same physical record layout. Other
    # pairs need an explicit review before this pre-ensure read may accept them.
    return (observed, target) == (17, 18)


def declared_table_names() -> set[str]:
    from gpu_fault.store.postgres import ddl

    pattern = re.compile(
        r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(gpu_fault_\w+)\s*\(", re.I
    )
    result: set[str] = set()
    for path in Path(ddl.__file__).parent.glob("ddl*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                match = pattern.match(node.value.strip())
                if match:
                    result.add(match.group(1))
    return result


def validate_heap_table(cursor: Any, name: str, *, partial: bool = False) -> None:
    cursor.execute(
        """
        SELECT c.relkind, a.amname, c.relrowsecurity, c.relforcerowsecurity,
          EXISTS (SELECT 1 FROM pg_catalog.pg_policy p WHERE p.polrelid=c.oid),
          EXISTS (SELECT 1 FROM pg_catalog.pg_rewrite r WHERE r.ev_class=c.oid),
          EXISTS (SELECT 1 FROM pg_catalog.pg_inherits i WHERE i.inhrelid=c.oid OR i.inhparent=c.oid),
          EXISTS (SELECT 1 FROM pg_catalog.pg_attribute x
            JOIN pg_catalog.pg_type t ON t.oid=x.atttypid
            JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace
            WHERE x.attrelid=c.oid AND x.attnum>0 AND NOT x.attisdropped
              AND (n.nspname<>'pg_catalog' OR x.attgenerated<>'')),
          EXISTS (SELECT 1 FROM pg_catalog.pg_trigger t
            WHERE t.tgrelid=c.oid AND NOT t.tgisinternal)
        FROM pg_catalog.pg_class c JOIN pg_catalog.pg_am a ON a.oid=c.relam
        WHERE c.oid=pg_catalog.to_regclass(%s)
        """,
        (f"public.{name}",),
    )
    row = cursor.fetchone()
    if row is None or tuple(row[:8]) != (
        "r",
        "heap",
        False,
        False,
        False,
        False,
        False,
        False,
    ):
        raise ValueError("unrecognized table definition or execution policy")
    if partial and row[8]:
        raise ValueError("uninitialized table has unverified triggers")


def prove_partial_empty(cursor: Any, relations: list[Any]) -> dict[str, Any]:
    from psycopg import sql

    declared = declared_table_names()
    if any(kind != "r" or name not in declared for _schema, name, kind in relations):
        raise ValueError("uninitialized schema has unrecognized views or tables")
    for _schema, name, _kind in relations:
        validate_heap_table(cursor, name, partial=True)
    for schema, name, _kind in relations:
        cursor.execute(
            sql.SQL("SELECT EXISTS (SELECT 1 FROM ONLY {} LIMIT 1)").format(
                sql.Identifier(schema, name)
            )
        )
        if cursor.fetchone() != (False,):
            raise ValueError("incomplete schema contains rows")
    return {
        "safe": True,
        "database_state": "uninitialized_empty",
        "schema_version": 0,
        "blockers": {"workflow": 0, "remote_command": 0, "observation": 0},
    }


def inspect_database(
    connection: Any, required_schema: int, *, allow_schema_upgrade: bool = False
) -> dict[str, Any]:
    """Prove real emptiness or inspect initialized state, never initialize it."""
    from psycopg import sql

    from gpu_fault.schema_migrations import (
        LATEST_POSTGRES_SCHEMA_VERSION,
        POSTGRES_SCHEMA_MIGRATIONS,
    )
    from gpu_fault.store.postgres.state_table_definitions import (
        validate_state_definitions,
    )
    from gpu_fault.store.postgres.state_table_payload import STATE_LAYOUTS

    if required_schema != LATEST_POSTGRES_SCHEMA_VERSION:
        raise ValueError("probe image schema identity differs")
    with connection.transaction(), connection.cursor() as cursor:
        cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        # pg_catalog remains implicitly first; current_schema() must be public
        # for the shared read-only definition validator.
        cursor.execute("SET LOCAL search_path = public")
        cursor.execute("SET LOCAL row_security = off")
        cursor.execute(
            """
            SELECT n.nspname, c.relname, c.relkind
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
            WHERE c.relkind IN ('r','p','v','m','f')
              AND n.nspname NOT IN ('pg_catalog','information_schema')
              AND n.nspname NOT LIKE 'pg_toast%'
              AND n.nspname NOT LIKE 'pg_temp_%'
              AND (c.relname LIKE 'gpu_fault_%' OR NOT EXISTS (
                SELECT 1 FROM pg_catalog.pg_depend d
                WHERE d.classid='pg_catalog.pg_class'::regclass
                  AND d.objid=c.oid AND d.deptype='e'))
            ORDER BY n.nspname,c.relname LIMIT 257
            """
        )
        relations = cursor.fetchall()
        if len(relations) > MAX_RELATIONS or any(
            schema != "public"
            or not re.fullmatch(r"gpu_fault_[a-z0-9_]+", name)
            or kind == "f"
            for schema, name, kind in relations
        ):
            raise ValueError("unrecognized database schema")
        names = {name for _schema, name, _kind in relations}
        metadata = {"gpu_fault_schema_version", "gpu_fault_schema_migrations"}
        if not metadata.issubset(names):
            return prove_partial_empty(cursor, relations)
        for name in metadata:
            validate_heap_table(cursor, name)
        cursor.execute(
            "SELECT version FROM public.gpu_fault_schema_version WHERE singleton=TRUE"
        )
        versions = cursor.fetchall()
        if not versions:
            return prove_partial_empty(cursor, relations)
        if (
            len(versions) != 1
            or len(versions[0]) != 1
            or type(versions[0][0]) is not int
            or not 1 <= versions[0][0] <= required_schema
            or versions[0][0] != required_schema
            and not (
                allow_schema_upgrade
                and compatible_retained_schema(versions[0][0], required_schema)
            )
        ):
            raise ValueError("bootstrap database schema version differs")
        observed_schema = versions[0][0]
        cursor.execute(
            "SELECT version,name,checksum FROM public.gpu_fault_schema_migrations ORDER BY version"
        )
        if cursor.fetchall() != [
            (item.version, item.name, item.checksum)
            for item in POSTGRES_SCHEMA_MIGRATIONS
            if item.version <= observed_schema
        ]:
            raise ValueError("bootstrap database migration history differs")
        if not {
            "gpu_fault_control_records",
            "gpu_fault_objects",
            "gpu_fault_attempt_observations",
        }.issubset(names):
            raise ValueError("bootstrap database record schema is incomplete")
        for name in (
            "gpu_fault_objects",
            "gpu_fault_attempt_observations",
            "gpu_fault_control_state_modes",
            STATE_LAYOUTS["workflow"].table,
            STATE_LAYOUTS["remote_command"].table,
        ):
            validate_heap_table(cursor, name)
        if observed_schema == required_schema:
            validate_state_definitions(cursor)
        # Retained older schemas are read only through validated physical heaps,
        # never through views or stored functions. The candidate ensure Job
        # must still install and validate its own functions before business Pods.
        cursor.execute("SELECT kind,mode FROM public.gpu_fault_control_state_modes")
        modes = dict(cursor.fetchall())
        if set(modes) != set(STATE_LAYOUTS) or any(
            mode not in {"legacy", "dual", "dedicated"} for mode in modes.values()
        ):
            raise ValueError("unknown control-state storage mode")
        # Native status is a typed column, excluded from payload/snapshot JSON.
        cursor.execute(
            sql.SQL("""
            SELECT kind,status,count(*) FROM (
              SELECT kind,payload->>'status' AS status FROM ONLY public.gpu_fault_objects
                WHERE kind IN ('workflow','remote_command')
              UNION ALL SELECT 'workflow',status FROM ONLY {}
              UNION ALL SELECT 'remote_command',status FROM ONLY {}
            ) records GROUP BY kind,status
            """).format(
                sql.Identifier("public", STATE_LAYOUTS["workflow"].table),
                sql.Identifier("public", STATE_LAYOUTS["remote_command"].table),
            )
        )
        terminal = {
            "workflow": {"SUCCEEDED", "FAILED", "SUPERSEDED"},
            "remote_command": {"SUCCEEDED", "FAILED"},
        }
        blockers = {"workflow": 0, "remote_command": 0, "observation": 0}
        for kind, status, count in cursor.fetchall():
            if kind not in terminal or not isinstance(count, int) or count < 0:
                raise ValueError("invalid workflow evidence")
            if status not in terminal[kind]:
                blockers[kind] += count
        # Inspect both storage locations conservatively during a hot-state
        # migration. An unrecognized or unresolved observation cannot be empty.
        cursor.execute(
            """
            SELECT count(*) FROM (
              SELECT payload FROM ONLY public.gpu_fault_objects WHERE kind='attempt_observation'
              UNION ALL SELECT payload FROM ONLY public.gpu_fault_attempt_observations
            ) observations
            WHERE coalesce(payload->'observation'->>'workload_phase','') NOT IN
              ('SUCCEEDED','FAILED','STOPPED')
            """
        )
        blockers["observation"] = int(cursor.fetchone()[0])
        return {
            "safe": not any(blockers.values()),
            "database_state": "initialized",
            "schema_version": observed_schema,
            "schema_ensure_required": observed_schema != required_schema,
            "blockers": blockers,
        }


def connection_arguments(dsn: str, identity: dict[str, Any]) -> dict[str, Any]:
    from psycopg.conninfo import conninfo_to_dict

    parsed = conninfo_to_dict(dsn)
    expected = {
        "host": identity["endpoint"],
        "port": str(identity["port"]),
        "dbname": identity["database"],
        "user": identity["username"],
        "sslmode": "verify-full",
        "sslrootcert": CA_PATH,
    }
    if any(
        parsed.get(key) != value for key, value in expected.items()
    ) or not parsed.get("password"):
        raise ValueError("database Secret binding differs")
    return {
        **expected,
        "password": parsed["password"],
        "connect_timeout": 10,
        "options": (
            "-c default_transaction_read_only=on -c statement_timeout=10000 "
            "-c lock_timeout=5000 -c idle_in_transaction_session_timeout=30000"
        ),
    }


def main() -> None:
    result: dict[str, Any] = {"safe": False}
    stage = ProbeStage.IMPORT_DRIVER
    try:
        import psycopg

        stage = ProbeStage.READ_IDENTITY
        identity = json.loads(os.environ["GPU_FAULT_PROOF_DATABASE"])
        stage = ProbeStage.BIND_CONNECTION
        arguments = connection_arguments(os.environ["GPU_FAULT_STORE_URL"], identity)
        stage = ProbeStage.CONNECT
        with psycopg.connect(**arguments) as connection:
            stage = ProbeStage.INSPECT_DATABASE
            result = inspect_database(
                connection,
                int(identity["schema_version"]),
                allow_schema_upgrade=(
                    os.environ.get("GPU_FAULT_PROOF_RETAINED_SCHEMA_UPGRADE") == "true"
                ),
            )
            stage = ProbeStage.CLOSE_CONNECTION
        stage = ProbeStage.BUILD_ENVELOPE
        result.update(
            run_id=os.environ["GPU_FAULT_PROOF_RUN_ID"],
            identity_sha256=os.environ["GPU_FAULT_PROOF_IDENTITY_SHA256"],
            finished_at=datetime.now(UTC).isoformat(),
        )
    except Exception as exc:
        # Driver exceptions may embed a DSN. Never emit their message or repr.
        error_type = next(
            (
                kind.__name__
                for kind in type(exc).__mro__
                if kind.__name__ in PROBE_ERROR_TYPES
            ),
            "unknown",
        )
        result = {"safe": False, "error_type": error_type, "probe_stage": stage.value}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
