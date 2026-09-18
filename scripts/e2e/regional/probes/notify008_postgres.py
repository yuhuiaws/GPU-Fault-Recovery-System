"""Fixed, Pod-local PostgreSQL target; no inherited or operator-supplied DSN."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from gpu_fault.store import PostgresStore
from scripts.e2e.regional.probes.notify008_protocol import ProbeError, run_identity

DSN = "postgresql:///notify008?host=/socket&user=notify008&connect_timeout=5"


def reject_ambient_database_configuration() -> None:
    if any(key.startswith(("PG", "GPU_FAULT_STORE_")) for key in os.environ):
        raise ProbeError("ambient database configuration is forbidden in the sandbox")


def connect() -> Any:
    import psycopg

    reject_ambient_database_configuration()
    return psycopg.connect(
        DSN,
        connect_timeout=5,
        options="-c statement_timeout=5000 -c idle_in_transaction_session_timeout=5000",
    )


def database_identity(connection: Any) -> dict[str, Any]:
    version = connection.info.server_version
    if type(version) is not int or not 160000 <= version < 170000:
        raise ProbeError("sandbox PostgreSQL must be version 16")
    row = connection.execute(
        "SELECT current_database(), current_user, inet_server_addr(), "
        "current_setting('listen_addresses'), current_setting('data_directory')"
    ).fetchone()
    if row != ("notify008", "notify008", None, "", "/database/data"):
        raise ProbeError("database is not the fixed Unix-socket sandbox")
    aurora = connection.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_proc WHERE proname='aurora_version')"
    ).fetchone()
    if not isinstance(aurora, tuple) or len(aurora) != 1 or aurora[0] is not False:
        raise ProbeError("NOTIFY008 refuses Aurora or an unknown backend")
    return {
        "postgres_major": version // 10000,
        "database_is_unix_socket": True,
        "production_credentials_loaded": False,
    }


def verify_owner(connection: Any, run_id: str) -> None:
    rows = connection.execute(
        "SELECT run_id FROM public.notify008_owner WHERE owner_id=1"
    ).fetchall()
    if rows != [(run_identity(run_id),)]:
        raise ProbeError("sandbox database owner differs")


def prepare_schema(run_id: str) -> dict[str, Any]:
    run_identity(run_id)
    with connect() as connection:
        facts = database_identity(connection)
        tables = connection.execute(
            "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public'"
        ).fetchall()
        if tables:
            raise ProbeError("schema preparation refuses a nonempty database")
        connection.execute(
            "CREATE TABLE public.notify008_owner "
            "(owner_id integer PRIMARY KEY CHECK (owner_id=1), run_id text NOT NULL)"
        )
        connection.execute(
            "INSERT INTO public.notify008_owner(owner_id,run_id) VALUES (1,%s)",
            (run_id,),
        )
    store = PostgresStore(
        DSN,
        initialize_schema=True,
        pool_min_size=1,
        pool_max_size=2,
        pool_timeout_seconds=2,
    )
    store.close()
    with connect() as connection:
        verify_owner(connection, run_id)
    return facts


@dataclass(frozen=True)
class PostgresFactory:
    run_id: str

    def __call__(self) -> PostgresStore:
        with connect() as connection:
            database_identity(connection)
            verify_owner(connection, self.run_id)
        return PostgresStore(
            DSN,
            initialize_schema=False,
            pool_min_size=1,
            pool_max_size=2,
            pool_timeout_seconds=2,
        )
