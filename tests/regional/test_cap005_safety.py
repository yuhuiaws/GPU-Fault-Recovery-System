from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_cap005_postgres_suite as cap005

LOCAL_URL = "postgresql://localhost:55432/postgres"


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://database.cluster.example.rds.amazonaws.com/postgres",
        "postgresql://192.0.2.1/postgres",
        "postgresql://localhost:55432/gpu_fault",
        "postgresql://localhost:55432/postgres?host=database.example",
        "postgresql://localhost:55432/postgres?hostaddr=192.0.2.1",
        "postgresql://localhost:55432/postgres?service=production",
        "postgresql://localhost:55432/postgres?host=",
        "postgresql://localhost:55432/postgres?hostaddr=",
        "postgresql://localhost:55432/postgres?sslmode",
        "postgresql://localhost:55432/postgres?sslmode=require&sslmode=disable",
        "postgresql://localhost:0/postgres",
        "postgresql://localhost:55432/postgres#ignored",
    ],
)
def test_remote_or_redirected_database_is_rejected_before_connect(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "psycopg.connect",
        lambda *_args, **_kwargs: calls.append("connect")
        or pytest.fail("an unapproved database must not be contacted"),
    )

    with pytest.raises(cap005.SuiteError, match="isolated|loopback|database"):
        cap005.run_suite(url, tmp_path)

    assert calls == []


SHARD_PROOF = {
    "receipt_directory": "/private/receipts",
    "receipt_sha256": {"run.json": "a" * 64},
    "source_identity": "a" * 64,
    "workers": 1,
    "executed_tests": 3,
    "elapsed_seconds": 1.0,
    "cleanup_verified": True,
}


def _mock_suite(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    create_error: BaseException | None = None,
    test_error: BaseException | None = None,
    shard_error: BaseException | None = None,
    commands: list[list[str]] | None = None,
    mock_database_calls: bool = True,
) -> tuple[list[str], list[dict[str, str]]]:
    calls: list[str] = []
    environments: list[dict[str, str]] = []
    monkeypatch.setattr(cap005, "ROOT", root.resolve())

    def validate(_url: str) -> None:
        calls.append("validate_server")

    def shards(workdir: Path, report_dir: Path, workers: int) -> dict[str, Any]:
        calls.append("shards")
        assert workdir == root.resolve() and workers == 1, (
            "the shard stage runs in this checkout with the requested worker count"
        )
        assert report_dir.is_dir() and report_dir.is_relative_to(workdir), (
            "shard state lives under this run's report directory"
        )
        if shard_error is not None:
            raise shard_error
        return dict(SHARD_PROOF)

    def create(_url: str, _database: str) -> None:
        calls.append("create")
        if create_error is not None:
            raise create_error

    def run(
        _argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        junit: Path,
        isolated_postgres_url: str | None = None,
    ) -> int:
        calls.append("pytest")
        if commands is not None:
            commands.append(list(_argv))
        environments.append(env)
        assert isolated_postgres_url == (
            env["GPU_FAULT_TEST_POSTGRES_URL"]
            if _argv[1:3] == ["-m", "pytest"]
            else None
        ), (
            "only the direct contract child may explicitly request its generated database"
        )
        if test_error is not None:
            raise test_error
        junit.write_text(
            '<testsuite tests="3" failures="0" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(cap005, "validate_server", validate)
    monkeypatch.setattr(cap005, "run_sharded_postgres", shards)
    monkeypatch.setattr(cap005, "_run_pytest", run)
    monkeypatch.setattr(
        "tools.pytest_result_identity.source_identity",
        lambda _root: SHARD_PROOF["source_identity"],
    )
    if mock_database_calls:
        monkeypatch.setattr(cap005, "_create_database", create)
        monkeypatch.setattr(
            cap005, "_drop_database", lambda *_args: calls.append("drop")
        )
        monkeypatch.setattr(
            cap005,
            "_database_exists",
            lambda *_args: calls.append("verify_absence") or False,
        )
    return calls, environments


def test_suite_runs_owned_shards_then_the_contract_with_current_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    calls, _ = _mock_suite(monkeypatch, tmp_path, commands=commands)
    report = cap005.run_suite(LOCAL_URL, tmp_path)
    assert report["status"] == "PASS", report
    assert calls == [
        "validate_server",
        "create",
        "shards",
        "pytest",
        "drop",
        "verify_absence",
    ], "the PostgreSQL suite runs on owned shards before the contract child"
    assert commands == [
        [sys.executable, "-m", "pytest", "-q", "tests/store/test_store_contracts.py"]
    ], "the only direct pytest child is the Store contract, on the active interpreter"
    assert report["postgres_shards"] == SHARD_PROOF and set(report["suites"]) == {
        "contract"
    }, "the shard proof is reported as receipts, never as a JUnit rendering"


def test_shard_stage_failure_stops_before_the_contract_and_still_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _mock_suite(
        monkeypatch, tmp_path, shard_error=cap005.SuiteError("shard 0 failed")
    )
    result = cap005.run_suite(LOCAL_URL, tmp_path)
    assert result["status"] == "FAIL" and result["errors"] == ["suite: SuiteError"], (
        "a red shard stage is the case's failure"
    )
    assert calls == ["validate_server", "create", "shards", "drop", "verify_absence"], (
        "the contract must not run after a failed shard stage; cleanup still does"
    )
    assert result["database_dropped"] is True and "postgres_shards" not in result, (
        "no proof is reported for a stage that did not complete"
    )


def test_another_checkout_is_refused_before_the_server_is_contacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _mock_suite(monkeypatch, tmp_path)
    with pytest.raises(cap005.SuiteError, match="own checkout"):
        cap005.run_suite(LOCAL_URL, tmp_path / "elsewhere")
    assert calls == [], "a foreign workdir must fail before validate_server"


def test_create_ack_loss_preserves_unproven_database_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _mock_suite(
        monkeypatch, tmp_path, create_error=OSError("creation ACK lost")
    )

    result = cap005.run_suite(LOCAL_URL, tmp_path)

    assert result["status"] == "FAIL"
    assert result["database_dropped"] is False, (
        "an unknown CREATE outcome cannot prove cleanup or database custody"
    )
    assert result["database_creation"] == "UNKNOWN" and result["cleanup_preserved"], (
        "the report must distinguish uncertain creation from acknowledged ownership"
    )
    assert calls == ["validate_server", "create"], (
        "an unacknowledged create must not authorize DROP or another database command"
    )
    assert result["errors"] == [
        "suite: OSError",
        "database creation outcome is unknown; ownership and cleanup are unproven",
    ], "the uncertain outcome must remain explicit alongside the original failure"


def test_duplicate_database_refusal_never_drops_the_preexisting_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from psycopg.errors import DuplicateDatabase

    calls, _ = _mock_suite(
        monkeypatch,
        tmp_path,
        create_error=DuplicateDatabase("test-only name collision"),
    )

    result = cap005.run_suite(LOCAL_URL, tmp_path)

    assert result["status"] == "FAIL" and result["database_creation"] == "REFUSED", (
        "a definite collision must not become an acknowledged creation"
    )
    assert result["database_dropped"] is False and result["cleanup_preserved"], (
        "the preexisting database must be left outside this run's cleanup"
    )
    assert calls == ["validate_server", "create"], (
        "a DuplicateDatabase response must stop before test or cleanup commands"
    )
    assert result["errors"] == ["suite: DuplicateDatabase"], (
        "a definite refusal must not be mislabeled as an uncertain acknowledgement"
    )


def test_rejected_child_database_url_fails_before_create_or_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _mock_suite(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cap005,
        "database_url",
        lambda _base, database: f"postgresql://192.0.2.1:55432/{database}",
    )

    result = cap005.run_suite(LOCAL_URL, tmp_path)

    assert (
        result["status"] == "FAIL" and result["database_creation"] == "NOT_ATTEMPTED"
    ), (
        "invalid child configuration must be rejected before attempting database creation"
    )
    assert calls == ["validate_server"], (
        "child URL refusal must not mutate or attempt cleanup of any database"
    )
    assert result["errors"] == ["suite: ValueError"], (
        "the rejected URL must not be echoed into the failure report"
    )


def test_relative_workdir_produces_absolute_child_and_junit_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_suite(monkeypatch, tmp_path / "workspace")
    monkeypatch.chdir(tmp_path)
    seen: list[tuple[Path, Path]] = []

    def run(
        _argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        junit: Path,
        isolated_postgres_url: str | None = None,
    ) -> int:
        seen.append((cwd, junit))
        assert junit.is_absolute(), (
            "changing the child cwd must not reinterpret the JUnit destination"
        )
        with monkeypatch.context() as child:
            child.chdir(cwd)
            junit.write_text(
                '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(cap005, "_run_pytest", run)
    result = cap005.run_suite(LOCAL_URL, Path("workspace"))

    assert result["status"] == "PASS" and len(seen) == 1, (
        "the contract report must be found after the child ran from the resolved workdir"
    )
    assert all(cwd == tmp_path / "workspace" for cwd, _ in seen), (
        "the child working directory must resolve once in the caller"
    )
    assert all(junit.is_relative_to(tmp_path / "workspace") for _, junit in seen), (
        "the absolute report paths must still remain within the selected workdir"
    )


def test_test_exception_is_reported_and_cleanup_is_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _mock_suite(monkeypatch, tmp_path, test_error=OSError("runner stopped"))

    result = cap005.run_suite(LOCAL_URL, tmp_path)

    assert result["status"] == "FAIL"
    assert result["errors"] == ["suite: OSError"]
    assert calls == [
        "validate_server",
        "create",
        "shards",
        "pytest",
        "drop",
        "verify_absence",
    ]


def test_interrupted_child_propagates_after_confirmed_database_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _mock_suite(monkeypatch, tmp_path, test_error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        cap005.run_suite(LOCAL_URL, tmp_path)

    assert calls == [
        "validate_server",
        "create",
        "shards",
        "pytest",
        "drop",
        "verify_absence",
    ], "an interrupt must preserve cleanup only after CREATE was acknowledged"


def test_isolation_mismatch_stops_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _mock_suite(monkeypatch, tmp_path)
    monkeypatch.setattr(cap005, "isolation_facts", lambda *_args: {"isolated": False})

    result = cap005.run_suite(LOCAL_URL, tmp_path)

    assert (
        result["status"] == "FAIL" and result["database_creation"] == "NOT_ATTEMPTED"
    ), "a failed generated-database binding must not be promoted to creation custody"
    assert calls == ["validate_server"], (
        "isolation failure must issue no database mutation"
    )


@pytest.mark.parametrize("residual", [False, True])
def test_public_suite_uses_bound_sql_and_requires_observed_database_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, residual: bool
) -> None:
    from psycopg import sql

    _mock_suite(monkeypatch, tmp_path, mock_database_calls=False)
    connections = []

    class Connection:
        def __init__(self):
            self.queries = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return self

        def execute(self, statement, parameters=None):
            text = (
                statement.as_string()
                if isinstance(statement, sql.Composable)
                else statement
            )
            self.queries.append((text, parameters))
            return self

        def fetchone(self):
            return (int(residual),)

    def connect(url, **options):
        assert url == LOCAL_URL and options["autocommit"] is True, (
            "the SQL lifecycle must use only the verified loopback administrative connection"
        )
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr("psycopg.connect", connect)

    result = cap005.run_suite(LOCAL_URL, tmp_path)

    assert result["database_creation"] == "ACKNOWLEDGED", (
        "successful SQL acknowledgement must establish this run's database custody"
    )
    assert result["status"] == ("FAIL" if residual else "PASS"), result
    assert result["database_dropped"] is not residual, (
        "DROP acknowledgement alone cannot substitute for an absence readback"
    )
    assert len(connections) == 3, (
        "the actual SQL helpers must create, drop and independently verify absence"
    )
    database = result["database"]
    assert connections[0].queries == [
        (
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)).as_string(),
            None,
        )
    ], "creation must quote only the generated database identifier"
    assert connections[1].queries == [
        (
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname=%s AND pid<>pg_backend_pid()",
            (database,),
        ),
        (
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)")
            .format(sql.Identifier(database))
            .as_string(),
            None,
        ),
    ], "cleanup must bind termination and DROP to the acknowledged generated database"
    assert connections[2].queries == [
        ("SELECT count(*) FROM pg_database WHERE datname=%s", (database,))
    ], "absence must be independently read for that same database"
    if residual:
        assert result["errors"] == [f"test database still exists: {database}"], (
            "a retained database must keep the public suite non-PASS"
        )


def test_main_returns_only_after_complete_mock_suites_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls, _ = _mock_suite(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", LOCAL_URL)
    monkeypatch.setenv("GPU_FAULT_CAP005_WORKDIR", str(tmp_path))

    assert cap005.main() is None, "the successful entry must return normally"
    assert '"status": "PASS"' in capsys.readouterr().out, (
        "the entry must publish the completed, cleaned-up outcome"
    )
    assert calls[-2:] == ["drop", "verify_absence"], (
        "successful entry output must follow verified owned cleanup"
    )


def test_script_bootstrap_resolves_checkout_imports_without_opening_a_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {str(cap005.ROOT), str(cap005.ROOT / "src")}
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path not in sources])
    monkeypatch.setattr(
        "psycopg.connect",
        lambda *_args, **_kwargs: pytest.fail(
            "script import attempted a database connection"
        ),
    )

    loaded = runpy.run_path(cap005.__file__, run_name="cap005_import_check")

    assert loaded["ROOT"] == cap005.ROOT and sources <= set(sys.path), (
        "standalone script loading must add only its actual checkout import roots"
    )
    with pytest.raises(loaded["SuiteError"]):
        loaded["validate_database_url"]("postgresql://production.invalid/postgres")


def test_child_cannot_inherit_another_store_or_cloud_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, environments = _mock_suite(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "must-not-inherit")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", "/private/production-dsn")
    monkeypatch.setenv("GPU_FAULT_ACCEPTANCE_EXECUTION_SCOPE", "formal")
    monkeypatch.setenv("AWS_PROFILE", "production")
    monkeypatch.setenv("PGSERVICE", "production")
    monkeypatch.setenv("KUBECONFIG", "/private/production-kubeconfig")

    result = cap005.run_suite(LOCAL_URL, tmp_path)

    assert result["status"] == "PASS"
    assert len(environments) == 1, "only the contract child receives this environment"
    for environment in environments:
        assert {key for key in environment if key.startswith("GPU_FAULT_")} == {
            "GPU_FAULT_TEST_POSTGRES_URL",
            "GPU_FAULT_STORE_URL",
        }
        assert environment["GPU_FAULT_STORE_URL"] == "", (
            "the standard isolated environment must explicitly disable a production Store"
        )
        assert "AWS_PROFILE" not in environment
        assert "PGSERVICE" not in environment
        assert environment["AWS_CONFIG_FILE"] == "/dev/null"
        assert environment["AWS_SHARED_CREDENTIALS_FILE"] == "/dev/null"
        assert environment["AWS_EC2_METADATA_DISABLED"] == "true"
        assert environment["KUBECONFIG"] == "/dev/null"
        assert (
            cap005.isolation_facts(
                LOCAL_URL,
                environment["GPU_FAULT_TEST_POSTGRES_URL"],
                result["database"],
            )["isolated"]
            is True
        )


def test_stale_junit_cannot_substitute_for_this_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_suite(monkeypatch, tmp_path)
    (tmp_path / "contract.xml").write_text(
        '<testsuite tests="3" failures="0" errors="0" skipped="0"/>', encoding="utf-8"
    )
    monkeypatch.setattr(cap005, "_run_pytest", lambda *_args, **_kwargs: 0)

    result = cap005.run_suite(LOCAL_URL, tmp_path)

    assert result["status"] == "FAIL"
    assert result["suites"] == {"contract": None}
    assert result["errors"] == ["contract suite left no JUnit report"]


class _Connection:
    def __init__(self, version: int, *, aurora: bool) -> None:
        self.info = type("Info", (), {"server_version": version})()
        self.aurora = aurora

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def execute(self, _statement: str) -> _Connection:
        return self

    def fetchone(self) -> tuple[bool]:
        return (self.aurora,)


@pytest.mark.parametrize(
    ("version", "aurora", "accepted"),
    [(160004, False, True), (150010, False, False), (160004, True, False)],
)
def test_server_must_be_postgres16_and_not_an_aurora_tunnel(
    version: int, aurora: bool, accepted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "psycopg.connect", lambda *_args, **_kwargs: _Connection(version, aurora=aurora)
    )
    if accepted:
        cap005.validate_server(LOCAL_URL)
    else:
        with pytest.raises(cap005.SuiteError, match="PostgreSQL 16|Aurora"):
            cap005.validate_server(LOCAL_URL)
