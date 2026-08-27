from __future__ import annotations

import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
from psycopg import sql


def _database_url(base_url: str, database: str) -> str:
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


def _junit_stats(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def _drop_database(base_url: str, database: str) -> None:
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


def main() -> None:
    base_url = os.environ["GPU_FAULT_STORE_URL"]
    production_database = urlsplit(base_url).path.lstrip("/")
    database = f"gpu_fault_cap005_{uuid4().hex[:12]}"
    test_url = _database_url(base_url, database)
    workdir = Path(os.getenv("GPU_FAULT_CAP005_WORKDIR", "/work"))
    postgres_xml = workdir / "postgres.xml"
    contract_xml = workdir / "contract.xml"
    started = time.monotonic()
    created = False
    summary: dict[str, object] = {}
    try:
        with psycopg.connect(base_url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
                )
        created = True
        environment = {
            **os.environ,
            "GPU_FAULT_TEST_POSTGRES_URL": test_url,
        }
        subprocess.run(
            ["make", "test-postgres", "PYTHON=python3"],
            cwd=workdir,
            env={
                **environment,
                "PYTEST_ADDOPTS": (f"--junitxml={postgres_xml} --durations=20"),
            },
            check=True,
        )
        subprocess.run(
            [
                "python3",
                "-m",
                "pytest",
                "-q",
                "tests/store/test_store_contracts.py",
            ],
            cwd=workdir,
            env={
                **environment,
                "PYTEST_ADDOPTS": (f"--junitxml={contract_xml} --durations=20"),
            },
            check=True,
        )
        suites = {
            "postgres": _junit_stats(postgres_xml),
            "contract": _junit_stats(contract_xml),
        }
        if any(
            stats["failures"] or stats["errors"] or stats["skipped"]
            for stats in suites.values()
        ):
            raise RuntimeError(f"PostgreSQL suite was not clean: {suites}")
        summary = {
            "database": database,
            "production_database": production_database,
            "isolated": database != production_database,
            "suites": suites,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        if created:
            _drop_database(base_url, database)
            with psycopg.connect(base_url, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT count(*) FROM pg_database WHERE datname=%s",
                        (database,),
                    )
                    if cursor.fetchone()[0]:
                        raise RuntimeError(f"test database still exists: {database}")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
