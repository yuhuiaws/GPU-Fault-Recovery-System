from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import shlex
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
for source in (ROOT, ROOT / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from scripts.e2e.regional.regional_commands import run_fixture_command  # noqa: E402


class SuiteError(RuntimeError):
    pass


def validate_database_url(base_url: str) -> None:
    try:
        parsed = urlsplit(base_url)
        host = parsed.hostname or ""
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
        query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
        query_keys = [key for key, _value in query]
        valid = (
            parsed.scheme in {"postgres", "postgresql"}
            and loopback
            and parsed.port is not None
            and 1 <= parsed.port <= 65535
            and parsed.path == "/postgres"
            and not parsed.fragment
            and len(query_keys) == len(set(query_keys))
            and set(query_keys) <= {"sslmode", "sslrootcert", "connect_timeout"}
        )
    except ValueError:
        valid = False
    if not valid:
        raise SuiteError(
            "CAP-005 requires an isolated loopback PostgreSQL database: "
            "explicit port, /postgres, and no connection-target overrides"
        )


def validate_server(base_url: str) -> None:
    validate_database_url(base_url)
    import psycopg

    with psycopg.connect(
        base_url,
        autocommit=True,
        connect_timeout=10,
        options="-c statement_timeout=10000",
    ) as connection:
        if not 160000 <= connection.info.server_version < 170000:
            raise SuiteError("CAP-005 requires PostgreSQL 16")
        row = connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_proc "
            "WHERE proname = 'aurora_version')"
        ).fetchone()
        if row != (False,):
            raise SuiteError("CAP-005 refuses Aurora, including a loopback tunnel")


def test_environment(test_url: str) -> dict[str, str]:
    from scripts.e2e.regional.focused_pytest import validate_isolated_postgres_url
    from tools.run_fault_test_cases import build_isolated_environment

    validate_isolated_postgres_url(test_url)
    environment = build_isolated_environment()
    environment["GPU_FAULT_TEST_POSTGRES_URL"] = test_url
    return environment


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
    import psycopg
    from psycopg import sql

    with psycopg.connect(base_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
            )


def _drop_database(base_url: str, database: str) -> None:
    import psycopg
    from psycopg import sql

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
    import psycopg

    with psycopg.connect(base_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM pg_database WHERE datname=%s",
                (database,),
            )
            row = cursor.fetchone()
    return bool(row and row[0])


def _run_pytest(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    junit: Path,
    isolated_postgres_url: str | None = None,
) -> int:
    """Run one pytest invocation; the exit code is returned, never raised.

    ``check=True`` used to raise on the first red suite before the JUnit file
    was read, so the report said only "make returned 1". The counts are what
    the operator needs to see.
    """

    from scripts.e2e.regional.focused_pytest import is_local_pytest

    options = [f"--junitxml={junit}", "--durations=20", "-n", "0"]
    if is_local_pytest(argv):
        command = [*argv, *options]
        environment = env
    else:
        command = argv
        environment = {**env, "PYTEST_ADDOPTS": shlex.join(options)}
    completed = run_fixture_command(
        command,
        cwd=cwd,
        env=environment,
        isolated_postgres_url=isolated_postgres_url,
        check=False,
        timeout=21600,
    )
    return int(completed.returncode)


def clean_suites(suites: dict[str, dict[str, int] | None]) -> list[str]:
    """Why the suites are not clean; empty when every one ran green."""

    errors: list[str] = []
    for name, stats in suites.items():
        if stats is None:
            errors.append(f"{name} suite left no JUnit report")
            continue
        if stats["tests"] <= 0:
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


def postgres_receipt_hashes(directory: Path, workers: int) -> dict[str, str]:
    from scripts.postgres_shard_receipts import read_private_json

    names = ["run.json"]
    names.extend(
        f"shard-{index}/{name}"
        for index in range(workers)
        for name in ("pytest.json", "allocation.json")
    )
    return {
        name: hashlib.sha256(
            json.dumps(
                read_private_json(directory / name),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        for name in names
    }


def parallel_postgres_child(workers: int) -> None:
    from scripts.ci_coverage_config import pytest_targets
    from scripts.run_postgres_shards import run_postgres_shards

    # The existing launcher owns allocation, supervision, receipt checks and cleanup.
    # Reserve stdout for a structured reference to its unchanged private receipts.
    with redirect_stdout(sys.stderr):
        directory = run_postgres_shards(
            sys.executable,
            workers=workers,
            durations=20,
            tests=pytest_targets(ROOT, "postgres"),
        )
    print(
        json.dumps(
            {
                "directory": str(directory),
                "sha256": postgres_receipt_hashes(directory, workers),
            },
            sort_keys=True,
        )
    )


def run_sharded_postgres(
    workdir: Path, report_dir: Path, workers: int
) -> dict[str, Any]:
    """The PostgreSQL suite on ``workers`` owned PG16 instances (1..8).

    Every worker count, including one, goes through
    ``scripts/run_postgres_shards.py``: the suite's PostgreSQL modules expect
    the grant that launcher provides (an admin URL on a PG16 instance the
    test process owns, the allocation files, a private HOME and PGPASSFILE)
    and create their own databases from it. A single generated database on the
    CI server, which the contract stage still uses, cannot serve them.
    """
    from gpu_fault.admin.release_postgres import require_postgres_cleanup
    from scripts.ci_coverage_config import pytest_targets
    from scripts.postgres_shard_receipts import (
        read_private_json,
        unique_fields,
        validate_partition_union,
        validate_shard_receipt,
        validate_targets,
    )
    from scripts.run_postgres_shards import ShardJob, validate_independent_allocations
    from tools.pytest_result_identity import source_identity
    from tools.run_fault_test_cases import build_isolated_environment

    if workdir != ROOT or type(workers) is not int or not 1 <= workers <= 8:
        raise SuiteError("sharded CAP-005 requires this checkout and 1..8 workers")
    tests = validate_targets(workdir, pytest_targets(workdir, "postgres"))
    identity = source_identity(workdir)
    home = report_dir / "postgres-shards-home"
    home.mkdir(mode=0o700)
    environment = build_isolated_environment(
        {"PATH": os.environ.get("PATH", os.defpath)}
    )
    environment.pop("GPU_FAULT_STORE_URL")
    environment.pop("GPU_FAULT_TEST_POSTGRES_URL")
    environment.update(
        HOME=str(home),
        PYTHONPATH=os.pathsep.join((str(workdir / "src"), str(workdir))),
        PYTHONDONTWRITEBYTECODE="1",
        GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS="8",
        GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS="40",
    )
    started_after = datetime.now(timezone.utc)
    completed = run_fixture_command(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys\n"
            "from scripts.e2e.regional.run_cap005_postgres_suite "
            "import parallel_postgres_child\n"
            "parallel_postgres_child(int(sys.argv[1]))\n",
            str(workers),
        ],
        cwd=workdir,
        env=environment,
        check=False,
        timeout=21600,
    )
    if completed.returncode:
        # The launcher's own progress lines name the first failing shard and
        # nodeid; everything else the child printed stays private.
        for line in (completed.stderr or "").splitlines()[-40:]:
            if line.startswith("postgres-shards:"):
                print(line, file=sys.stderr)
        raise SuiteError("sharded PostgreSQL child did not complete successfully")
    reference = json.loads(completed.stdout, object_pairs_hook=unique_fields)
    if (
        not isinstance(reference, dict)
        or set(reference) != {"directory", "sha256"}
        or not isinstance(reference["directory"], str)
        or not isinstance(reference["sha256"], dict)
    ):
        raise SuiteError("parallel PostgreSQL receipt reference is invalid")
    directory = Path(reference["directory"])
    if directory.is_relative_to(workdir):
        raise SuiteError("parallel PostgreSQL receipts must stay outside the checkout")
    run = read_private_json(directory / "run.json")
    elapsed = run.get("elapsed_seconds")
    if (
        run.get("kind") != "local-postgres-shards"
        or run.get("status") != "COMPLETE"
        or run.get("source_identity") != identity
        or type(run.get("workers")) is not int
        or run["workers"] != workers
        or run.get("tests") != list(tests)
        or not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise SuiteError("parallel PostgreSQL run differs from the requested suite")
    jobs = [
        ShardJob(index, directory / f"shard-{index}", (), {})
        for index in range(workers)
    ]
    receipts = [
        validate_shard_receipt(
            job.state / "pytest.json",
            root=workdir,
            identity=identity,
            tests=tests,
            workers=workers,
            index=job.index,
            stress=("8", "40"),
            started_after=started_after,
        )
        for job in jobs
    ]
    timings = validate_partition_union(receipts, workers=workers)
    validate_independent_allocations(jobs)
    for job in jobs:
        require_postgres_cleanup(job.state)
        if any(
            path.exists() or path.is_symlink()
            for path in (job.state / "failure.json", job.state / "cancel.json")
        ):
            raise SuiteError("parallel PostgreSQL shard failed or was cancelled")
    if (
        type(run.get("executed_tests")) is not int
        or run["executed_tests"] != len(timings)
        or postgres_receipt_hashes(directory, workers) != reference["sha256"]
        or source_identity(workdir) != identity
    ):
        raise SuiteError("parallel PostgreSQL receipts or source changed")
    return {
        "receipt_directory": str(directory),
        "receipt_sha256": reference["sha256"],
        "source_identity": identity,
        "workers": workers,
        "executed_tests": len(timings),
        "elapsed_seconds": elapsed,
        "cleanup_verified": True,
    }


def run_suite(
    base_url: str, workdir: Path, *, postgres_workers: int = 1
) -> dict[str, Any]:
    import psycopg

    if type(postgres_workers) is not int or not 1 <= postgres_workers <= 8:
        raise SuiteError("CAP-005 PostgreSQL workers must be within 1..8")
    workdir = workdir.resolve()
    validate_database_url(base_url)
    if workdir != ROOT:
        raise SuiteError("CAP-005 must use the runner's own checkout")
    validate_server(base_url)
    production_database = _database_name(base_url)
    database = f"gpu_fault_cap005_{uuid4().hex[:12]}"
    test_url = database_url(base_url, database)
    report_dir = workdir / "artifacts" / "cap005" / database
    report_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    contract_xml = report_dir / "contract.xml"
    started = time.monotonic()
    created = False
    creation_outcome = "NOT_ATTEMPTED"
    summary: dict[str, Any] = {
        "database": database,
        "production_database": production_database,
        "postgres_workers": postgres_workers,
    }
    errors: list[str] = []
    try:
        isolation = isolation_facts(base_url, test_url, database)
        summary.update(isolation)
        if not isolation["isolated"]:
            raise SuiteError(f"test database URL is not isolated: {isolation}")
        environment = test_environment(test_url)
        creation_outcome = "UNKNOWN"
        try:
            _create_database(base_url, database)
        except psycopg.errors.DuplicateDatabase:
            creation_outcome = "REFUSED"
            raise
        created = True
        creation_outcome = "ACKNOWLEDGED"
        exit_codes: dict[str, int] = {}
        suites: dict[str, dict[str, int] | None] = {}
        # Stage 1: the PostgreSQL suite on owned PG16 instances; its proof is
        # the launcher's receipts, never a JUnit rendering of them.
        summary["postgres_shards"] = run_sharded_postgres(
            workdir, report_dir, postgres_workers
        )
        # Stage 2: the complete Store contract on this run's generated database.
        exit_codes["contract"] = _run_pytest(
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
            isolated_postgres_url=test_url,
        )
        suites["contract"] = _junit_stats(contract_xml)
        summary["suites"] = suites
        summary["exit_codes"] = exit_codes
        errors.extend(clean_suites(suites))
        errors.extend(
            f"{name} pytest exited {code}" for name, code in exit_codes.items() if code
        )
    except Exception as exc:
        errors.append(f"suite: {type(exc).__name__}")
    except BaseException as exc:
        errors.append(f"suite interrupted: {type(exc).__name__}")
        raise
    finally:
        summary["database_creation"] = creation_outcome
        if created:
            try:
                _drop_database(base_url, database)
                if _database_exists(base_url, database):
                    errors.append(f"test database still exists: {database}")
                    summary["database_dropped"] = False
                else:
                    summary["database_dropped"] = True
            except Exception as exc:  # noqa: BLE001 - merged into the report
                errors.append(f"drop database: {type(exc).__name__}")
                summary["database_dropped"] = False
        elif creation_outcome in {"UNKNOWN", "REFUSED"}:
            summary["database_dropped"] = False
            summary["cleanup_preserved"] = True
            if creation_outcome == "UNKNOWN":
                errors.append(
                    "database creation outcome is unknown; ownership and cleanup are unproven"
                )
        if "postgres_shards" in summary:
            from tools.pytest_result_identity import source_identity

            try:
                if (
                    source_identity(workdir)
                    != summary["postgres_shards"]["source_identity"]
                ):
                    errors.append("CAP-005 source changed before cleanup completed")
            except Exception as exc:
                errors.append(f"source verification: {type(exc).__name__}")
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        summary["errors"] = errors
        summary["status"] = "PASS" if not errors else "FAIL"
    return summary


def store_dsn() -> str:
    configured = os.environ.get("GPU_FAULT_STORE_URL_FILE")
    path = (
        configured if configured is not None else "/etc/gpu-fault/aurora/postgres-url"
    )
    if not path:
        raise RuntimeError("configured store DSN file path is empty")
    try:
        with open(path, encoding="utf-8") as handle:
            value = handle.read().strip()
    except FileNotFoundError:
        if configured is not None:
            raise
        return os.environ["GPU_FAULT_STORE_URL"]
    if not value:
        raise RuntimeError("store DSN file is empty")
    return value


def main(arguments: Sequence[str] = ()) -> None:
    parser = argparse.ArgumentParser(description="CAP-005 isolated PostgreSQL suite")
    parser.add_argument("--postgres-workers", type=int, choices=range(1, 9), default=1)
    options = parser.parse_args(arguments)
    base_url = store_dsn()
    workdir = Path(os.getenv("GPU_FAULT_CAP005_WORKDIR", str(ROOT)))
    summary = run_suite(base_url, workdir, postgres_workers=options.postgres_workers)
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "PASS":
        raise SuiteError("; ".join(summary["errors"]))


if __name__ == "__main__":
    main(sys.argv[1:])
