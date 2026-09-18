from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import pytest

from scripts.e2e.regional import boot_guard_isolation as boot
from scripts.e2e.regional import regional_commands
from scripts.e2e.regional import run_cap005_postgres_suite as cap005
from tests.regional._cov95_focused_mock_receipts import write_focused_receipt
from tools import pytest_result_identity

URL = os.environ.get("GPU_FAULT_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not URL, reason="requires an allocated isolated PostgreSQL 16"
)


@pytest.fixture
def databases(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, str]]:
    import psycopg
    from psycopg import sql

    cap005.validate_server(URL)
    suffix = uuid4().hex[:16]
    source = f"boot_source_{suffix}"
    probe = f"boot_probe_{suffix}"
    monkeypatch.setattr(boot, "PRODUCTION_DATABASE", source)
    monkeypatch.setattr(boot, "PROBE_DATABASE", probe)
    try:
        with psycopg.connect(URL, autocommit=True) as connection:
            connection.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(source))
            )
        yield cap005.database_url(URL, source), cap005.database_url(URL, probe)
    finally:
        with psycopg.connect(URL, autocommit=True) as connection:
            for name in (probe, source):
                connection.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                        sql.Identifier(name)
                    )
                )
            assert (
                connection.execute(
                    "SELECT datname FROM pg_database WHERE datname = ANY(%s)",
                    ([source, probe],),
                ).fetchall()
                == []
            ), "every test-owned database must be absent after cleanup"


@pytest.mark.parametrize("reset", [False, True])
def test_real_boot_schema_stays_off_the_inherited_source_database(
    reset: bool,
    databases: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    source, probe = databases
    with psycopg.connect(source, autocommit=True) as connection:
        connection.execute("CREATE TABLE source_sentinel (value integer PRIMARY KEY)")
        connection.execute("INSERT INTO source_sentinel VALUES (17)")
    credential = tmp_path / "source-dsn"
    credential.write_text(source, encoding="utf-8")
    credential.chmod(0o600)
    monkeypatch.setenv(boot.STORE_URL, "postgresql://unreachable.invalid/gpu_fault")
    monkeypatch.setenv(boot.STORE_URL_FILE, str(credential))

    boot.initialize_database()
    if reset:
        with psycopg.connect(probe, autocommit=True) as connection:
            connection.execute("CREATE TABLE prior_variant (value integer)")
        boot.initialize_database(reset=True)
    with psycopg.connect(source) as connection:
        assert connection.execute("SELECT value FROM source_sentinel").fetchall() == [
            (17,)
        ], "source rows must survive schema initialization and reset"
        assert connection.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        ).fetchall() == [("source_sentinel",)], (
            "no Store DDL may touch the inherited source"
        )
    with psycopg.connect(probe) as connection:
        assert connection.execute("SELECT to_regclass('gpu_fault_objects')").fetchone()[
            0
        ], "the actual disposable database must contain the initialized Store"
        assert connection.execute("SELECT to_regclass('prior_variant')").fetchone() == (
            None,
        ), "reset must remove the previous variant's durable state"
    boot.drop_database()
    with psycopg.connect(URL) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM pg_database WHERE datname=%s", (boot.PROBE_DATABASE,)
            ).fetchone()
            is None
        ), "the helper must verify real database deletion"


@pytest.mark.parametrize("child_fails", [False, True])
def test_real_cap005_database_is_private_and_removed_without_running_suites(
    child_fails: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg

    observed: list[str] = []
    source_identity = pytest_result_identity.source_identity(cap005.ROOT)

    def fixture_source_identity(root: Path) -> str:
        assert root == tmp_path, "the mocked checkout must retain its working directory"
        return source_identity

    monkeypatch.setattr(
        pytest_result_identity, "source_identity", fixture_source_identity
    )

    def child(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = kwargs["environment"]
        target = environment["GPU_FAULT_TEST_POSTGRES_URL"]
        with psycopg.connect(target) as connection:
            name = connection.execute("SELECT current_database()").fetchone()[0]
            assert name.startswith("gpu_fault_cap005_"), (
                "the child must use its unique database"
            )
            assert name != "postgres", (
                "the allocation's admin database must not run the suite"
            )
            observed.append(name)
        if child_fails:
            raise OSError("mocked test child failed after database creation")
        options = argparse.ArgumentParser(add_help=False)
        options.add_argument("--junitxml", "--junit-xml")
        arguments, _ = options.parse_known_args(
            [*command, *shlex.split(environment.get("PYTEST_ADDOPTS", ""))]
        )
        assert arguments.junitxml, (
            "the real child argv or environment must carry its JUnit destination"
        )
        if command[1:3] == ["-m", "pytest"]:
            write_focused_receipt(
                command,
                environment=environment,
                cwd=cap005.ROOT,
                nodeids=[
                    "tests/store/test_store_contracts.py::test_processor_queue_contract[postgres]"
                ],
            )
        Path(arguments.junitxml).write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(regional_commands, "run_command", child)

    result = cap005.run_suite(URL, tmp_path)

    assert result["status"] == ("FAIL" if child_fails else "PASS"), (
        "the real database lifecycle must retain the mocked child outcome"
    )
    assert result["database_dropped"] is True, (
        "failed children still require verified cleanup"
    )
    assert observed and set(observed) == {result["database"]}, (
        "both children must share only this run's DB"
    )
    assert len(observed) == (1 if child_fails else 2), (
        "only an earlier child failure may prevent the second suite from starting"
    )
    with psycopg.connect(URL) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM pg_database WHERE datname=%s", (result["database"],)
            ).fetchone()
            is None
        ), "CAP005 must leave no database after either outcome"


def test_real_cap005_creation_refusal_preserves_the_existing_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg
    from psycopg import sql

    cap005.validate_server(URL)
    identifier = uuid4()
    database = f"gpu_fault_cap005_{identifier.hex[:12]}"
    monkeypatch.setattr(cap005, "uuid4", lambda: identifier)
    monkeypatch.setattr(
        regional_commands,
        "run_command",
        lambda *_args, **_kwargs: pytest.fail(
            "a refused database allocation must not start test children"
        ),
    )
    created = False
    try:
        with psycopg.connect(URL, autocommit=True) as connection:
            connection.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
            )
        created = True
        existing_url = cap005.database_url(URL, database)
        with psycopg.connect(existing_url, autocommit=True) as connection:
            connection.execute("CREATE TABLE preexisting_sentinel (value integer)")
            connection.execute("INSERT INTO preexisting_sentinel VALUES (23)")

        result = cap005.run_suite(URL, tmp_path)

        assert result["status"] == "FAIL" and result["cleanup_preserved"] is True, (
            "a definite creation refusal must preserve the unowned database"
        )
        assert result["database_dropped"] is False, (
            "a failed allocation cannot claim to have cleaned an owned database"
        )
        with psycopg.connect(existing_url) as connection:
            assert connection.execute(
                "SELECT value FROM preexisting_sentinel"
            ).fetchall() == [(23,)], "the preexisting database and rows must survive"
    finally:
        if created:
            with psycopg.connect(URL, autocommit=True) as connection:
                connection.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                        sql.Identifier(database)
                    )
                )
                assert (
                    connection.execute(
                        "SELECT 1 FROM pg_database WHERE datname=%s", (database,)
                    ).fetchone()
                    is None
                ), "the fixture must remove its own database"
