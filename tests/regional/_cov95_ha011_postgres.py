"""Only the explicitly granted local PostgreSQL slot, with OID-owned databases."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from uuid import uuid4

from gpu_fault.store import PostgresStore
from scripts.e2e.regional.probes.ha011_processes import WorkerProcess
from scripts.e2e.regional.run_cap005_postgres_suite import database_url
from tests.regional._cov95_notify008_postgres import validated_grant


@dataclass(frozen=True)
class WorkerStore:
    url: str = field(repr=False)

    def __call__(self):
        return PostgresStore(
            self.url, initialize_schema=False, pool_min_size=1, pool_max_size=4
        )


@dataclass(frozen=True)
class GrantedWorkers:
    store_factory: WorkerStore

    def __call__(self, target, *, role, request_id):
        return WorkerProcess(
            partial(target, store_factory=self.store_factory),
            role=role,
            request_id=request_id,
        )


@contextmanager
def isolated_database():
    import psycopg
    from psycopg import sql

    base_url = validated_grant()
    name = f"ha011_{uuid4().hex}"
    oid = None
    creation_started = False
    try:
        with psycopg.connect(base_url, autocommit=True) as connection:
            assert (
                connection.execute(
                    "SELECT oid FROM pg_database WHERE datname=%s", (name,)
                ).fetchone()
                is None
            ), "HA011 must never adopt an existing database"
            creation_started = True
            connection.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name))
            )
            (oid,) = connection.execute(
                "SELECT oid FROM pg_database WHERE datname=%s", (name,)
            ).fetchone()
        url = database_url(base_url, name)
        setup = PostgresStore(
            url, initialize_schema=True, pool_min_size=1, pool_max_size=4
        )
        setup.close()
        yield WorkerStore(url)
    finally:
        assert validated_grant() == base_url, (
            "the PostgreSQL grant changed before owned cleanup"
        )
        with psycopg.connect(base_url, autocommit=True) as connection:
            current = connection.execute(
                "SELECT oid FROM pg_database WHERE datname=%s", (name,)
            ).fetchone()
            if current is not None and creation_started:
                assert oid is not None and current == (oid,), (
                    "cleanup must not drop an unacknowledged or replaced private database"
                )
                connection.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                        sql.Identifier(name)
                    )
                )
            if creation_started:
                assert (
                    connection.execute(
                        "SELECT oid FROM pg_database WHERE datname=%s", (name,)
                    ).fetchone()
                    is None
                ), "the owned HA011 PostgreSQL database must be absent"
