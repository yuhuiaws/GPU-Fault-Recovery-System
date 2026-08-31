"""Run an isolated control plane against a disposable Aurora database."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql

from gpu_fault.store import PostgresStore


WORK = Path("/work")
STOP = False


def database_url(base_url: str, database: str) -> str:
    parsed = urlsplit(base_url)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"/{database}",
            parsed.query,
            parsed.fragment,
        )
    )


def drop_database(base_url: str, database: str) -> None:
    with psycopg.connect(base_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_terminate_backend(pid) "
                "FROM pg_stat_activity "
                "WHERE datname=%s AND pid<>pg_backend_pid()",
                (database,),
            )
            cursor.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(database)
                )
            )


def stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def main() -> int:
    base_url = os.environ["GPU_FAULT_BASE_STORE_URL"]
    database = os.environ["CAP_DATABASE_NAME"]
    test_url = database_url(base_url, database)
    child: subprocess.Popen[str] | None = None
    created = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        drop_database(base_url, database)
        with psycopg.connect(base_url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
                )
        created = True
        store = PostgresStore(
            test_url,
            pool_min_size=1,
            pool_max_size=2,
            pool_timeout_seconds=30,
            initialize_schema=True,
        )
        store.close()
        WORK.mkdir(parents=True, exist_ok=True)
        store_path = WORK / "store-url"
        store_path.write_text(test_url, encoding="utf-8")
        store_path.chmod(0o600)
        (WORK / "database-name").write_text(database, encoding="utf-8")

        environment = dict(os.environ)
        environment["GPU_FAULT_STORE_URL"] = test_url
        environment.pop("GPU_FAULT_BASE_STORE_URL", None)
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "cap_probe_app:create_app",
                "--factory",
                "--app-dir",
                "/opt/cap",
                "--host",
                "0.0.0.0",
                "--port",
                "18080",
                "--workers",
                "1",
                "--no-access-log",
                "--no-proxy-headers",
                "--timeout-keep-alive",
                "5",
                "--timeout-graceful-shutdown",
                "30",
            ],
            env=environment,
            text=True,
        )
        assert child is not None
        print(
            json.dumps(
                {
                    "database": database,
                    "database_isolated": True,
                    "api_pid": child.pid,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        while not STOP:
            code = child.poll()
            if code is not None:
                return code
            time.sleep(0.5)
        return 0
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=35)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        if created:
            drop_database(base_url, database)


if __name__ == "__main__":
    raise SystemExit(main())
