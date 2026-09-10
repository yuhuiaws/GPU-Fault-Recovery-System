"""One PostgreSQL database per xdist worker for the Postgres-gated tests.

Every Postgres-gated module reads ``GPU_FAULT_TEST_POSTGRES_URL`` at import and
its fixtures ``TRUNCATE`` every ``gpu_fault_*`` table before each test. That
contract is sound for one process; run under xdist against one database it is
a race between workers, so the postgres shard ran serially -- 1037 cases in
about six minutes, the longest leg of the release gates while the in-memory
suite finished four minutes earlier on 16 workers.

``worker_database_url`` gives each worker its own database on the same server
(``<dbname>_<worker>``), created on first use. ``conftest.pytest_configure``
rewrites the environment variable before any test module is imported, so the
modules keep their one-line ``os.getenv`` contract and nothing else changes.
Server-wide state stays shared on purpose: the only test that touches it
(``test_postgres_reconnect``) creates a uniquely named role per run.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

_WORKER = re.compile(r"^[A-Za-z0-9_]+$")


def worker_database_name(base_name: str, worker_id: str) -> str:
    if not _WORKER.fullmatch(worker_id):
        raise ValueError(f"unexpected xdist worker id: {worker_id!r}")
    return f"{base_name}_{worker_id.lower()}"


def worker_database_url(base_url: str, worker_id: str) -> str:
    """Return ``base_url`` pointing at this worker's database, creating it."""

    import psycopg
    from psycopg import errors, sql

    parts = urlsplit(base_url)
    base_name = parts.path.lstrip("/") or "postgres"
    name = worker_database_name(base_name, worker_id)
    with psycopg.connect(base_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
            if cursor.fetchone() is None:
                try:
                    cursor.execute(
                        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name))
                    )
                except errors.DuplicateDatabase:
                    # Two workers raced on the same first use; either copy is fine.
                    pass
    return urlunsplit(
        (parts.scheme, parts.netloc, f"/{name}", parts.query, parts.fragment)
    )
