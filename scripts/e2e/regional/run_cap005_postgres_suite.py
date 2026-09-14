from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
from psycopg import sql


ROOT = Path(__file__).resolve().parents[3]


class SuiteError(RuntimeError):
    pass


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


def _database_name(url: str) -> str:
    return urlsplit(url).path.lstrip("/")


def _junit_stats(path: Path) -> dict[str, int] | None:
    """The suite counts from a JUnit file, or None when pytest left none."""

    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def _create_database(base_url: str, database: str) -> None:
    with psycopg.connect(base_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
            )


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


def _database_exists(base_url: str, database: str) -> bool:
    with psycopg.connect(base_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM pg_database WHERE datname=%s",
                (database,),
            )
            row = cursor.fetchone()
    return bool(row and row[0])


def _run_pytest(argv: list[str], *, cwd: Path, env: dict[str, str], junit: Path) -> int:
    """Run one pytest invocation; the exit code is returned, never raised.

    ``check=True`` used to raise on the first red suite before the JUnit file
    was read, so the report said only "make returned 1". The counts are what
    the operator needs to see.
    """

    completed = subprocess.run(
        argv,
        cwd=cwd,
        env={**env, "PYTEST_ADDOPTS": f"--junitxml={junit} --durations=20"},
        check=False,
    )
    return int(completed.returncode)


def clean_suites(suites: dict[str, dict[str, int] | None]) -> list[str]:
    """Why the suites are not clean; empty when every one ran green."""

    errors: list[str] = []
    for name, stats in suites.items():
        if stats is None:
            errors.append(f"{name} suite left no JUnit report")
            continue
        if stats["tests"] == 0:
            errors.append(f"{name} suite ran no tests")
        if stats["failures"] or stats["errors"] or stats["skipped"]:
            errors.append(f"{name} suite was not clean: {stats}")
    return errors


def isolation_facts(base_url: str, test_url: str, database: str) -> dict[str, Any]:
    """Whether the suite really ran against its own database.

    ``database != production_database`` was a tautology -- the test database
    name is generated, so it always differed. The check that means something
    is that the URL handed to pytest names the generated database and not
    the production one.
    """

    production_database = _database_name(base_url)
    test_database = _database_name(test_url)
    return {
        "database": database,
        "production_database": production_database,
        "test_url_database": test_database,
        "isolated": bool(
            test_database
            and test_database == database
            and test_database != production_database
        ),
    }


def run_suite(base_url: str, workdir: Path) -> dict[str, Any]:
    production_database = _database_name(base_url)
    database = f"gpu_fault_cap005_{uuid4().hex[:12]}"
    test_url = database_url(base_url, database)
    postgres_xml = workdir / "postgres.xml"
    contract_xml = workdir / "contract.xml"
    started = time.monotonic()
    created = False
    summary: dict[str, Any] = {
        "database": database,
        "production_database": production_database,
    }
    errors: list[str] = []
    try:
        isolation = isolation_facts(base_url, test_url, database)
        summary.update(isolation)
        if not isolation["isolated"]:
            raise SuiteError(f"test database URL is not isolated: {isolation}")
        _create_database(base_url, database)
        created = True
        environment = {**os.environ, "GPU_FAULT_TEST_POSTGRES_URL": test_url}
        # The interpreter running this script is the project venv, the one
        # with pytest-xdist; a bare ``python3`` resolved to the system
        # interpreter and the parallel Postgres shard died on ``-n``.
        exit_codes = {
            "postgres": _run_pytest(
                ["make", "test-postgres-stress", f"PYTHON={sys.executable}"],
                cwd=workdir,
                env=environment,
                junit=postgres_xml,
            ),
            "contract": _run_pytest(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/store/test_store_contracts.py",
                ],
                cwd=workdir,
                env=environment,
                junit=contract_xml,
            ),
        }
        suites = {
            "postgres": _junit_stats(postgres_xml),
            "contract": _junit_stats(contract_xml),
        }
        summary["suites"] = suites
        summary["exit_codes"] = exit_codes
        errors.extend(clean_suites(suites))
        errors.extend(
            f"{name} pytest exited {code}" for name, code in exit_codes.items() if code
        )
    finally:
        if created:
            try:
                _drop_database(base_url, database)
                if _database_exists(base_url, database):
                    errors.append(f"test database still exists: {database}")
                    summary["database_dropped"] = False
                else:
                    summary["database_dropped"] = True
            except Exception as exc:  # noqa: BLE001 - merged into the report
                errors.append(f"drop database: {type(exc).__name__}: {exc}")
                summary["database_dropped"] = False
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        summary["errors"] = errors
        summary["status"] = "PASS" if not errors else "FAIL"
    return summary


def store_dsn() -> str:
    path = (
        os.environ.get("GPU_FAULT_STORE_URL_FILE")
        or "/etc/gpu-fault/aurora/postgres-url"
    )
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return os.environ["GPU_FAULT_STORE_URL"]


def main() -> None:
    base_url = store_dsn()
    workdir = Path(os.getenv("GPU_FAULT_CAP005_WORKDIR", str(ROOT)))
    summary = run_suite(base_url, workdir)
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "PASS":
        raise SuiteError("; ".join(summary["errors"]))


if __name__ == "__main__":
    main()
