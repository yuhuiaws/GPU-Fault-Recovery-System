"""Only native PostgreSQL children consume the explicit private grant HOME."""

from __future__ import annotations

import io
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import postgres_grant
from gpu_fault.admin.postgres_grant import (
    ALLOCATION_ENV,
    LOCAL_DOCKER_HOST,
    OWNER_FILE,
    POSTGRES_URL_ENV,
    URL_FILE,
    PostgresGrantError,
    PostgresTestAllocation,
    postgres_test_environment,
)
from scripts import run_release_gates
from tests.test_change_impact import MODULE as impact


@pytest.fixture
def grant(tmp_path: Path) -> tuple[Path, Path, str]:
    directory = tmp_path / "allocation"
    directory.mkdir(mode=0o700)
    home = directory / "home"
    home.mkdir(mode=0o700)
    url = "postgresql://postgres@127.0.0.1:54321/postgres"
    owner = {
        "schema_version": 1,
        "owner": "example-test-owner",
        "container": "b" * 64,
        "port": 54321,
        "pgpass_home": str(home),
    }
    for path, content in (
        (directory / OWNER_FILE, json.dumps(owner)),
        (directory / URL_FILE, url),
        (home / ".pgpass", "127.0.0.1:54321:*:postgres:example-password\n"),
    ):
        path.write_text(content)
        path.chmod(0o600)
    return directory, home, url


def test_explicit_grant_changes_only_a_copy_of_the_pg_child_environment(grant) -> None:
    directory, home, url = grant
    original = {
        "HOME": "/original/build-home",
        "DOCKER_CONTEXT": "original-context",
        "DOCKER_CONFIG": "/original/docker-config",
        "PGHOSTADDR": "foreign.example",
        "PGPASSWORD": "example-ambient-password",
        "PGPASSFILE": "/original/pgpass",
        "COSIGN_PASSWORD": "example-signing-password",
        "PATH": "/original/bin",
    }
    allocation = PostgresTestAllocation(url, directory)
    build = allocation.build_environment(original)
    expected = dict(build)
    child = postgres_test_environment(build)
    assert build == expected, "PG child environment mutated its build parent"
    assert build["HOME"] == "/original/build-home"
    assert build["COSIGN_PASSWORD"] == original["COSIGN_PASSWORD"]
    assert child["HOME"] == str(home)
    assert child["PGPASSFILE"] == str(home / ".pgpass")
    assert child[POSTGRES_URL_ENV] == url
    assert child[ALLOCATION_ENV] == str(directory)
    assert child["DOCKER_HOST"] == LOCAL_DOCKER_HOST
    assert (
        set(child)
        & {
            "PGHOSTADDR",
            "PGPASSWORD",
            "DOCKER_CONTEXT",
            "DOCKER_CONFIG",
            "COSIGN_PASSWORD",
        }
        == set()
    )


def test_no_explicit_grant_never_invents_ownership_or_changes_home() -> None:
    original = {"HOME": "/original", POSTGRES_URL_ENV: "explicit-external-url"}
    assert postgres_test_environment(original) == original


@pytest.mark.parametrize(
    "defect",
    [
        "empty-reference",
        "missing-owner",
        "missing-url",
        "missing-pgpass",
        "owner-not-json",
        "owner-not-object",
        "wrong-url",
        "password-url",
        "foreign-host",
        "bad-port",
        "bool-port",
        "different-port",
        "bad-cid",
        "blank-owner",
        "relative-home",
        "public-directory",
        "public-home",
        "public-owner",
        "public-pgpass",
        "owner-symlink",
        "pgpass-symlink",
    ],
)
def test_invalid_explicit_grant_fails_closed_without_modifying_parent(
    grant, defect: str, tmp_path: Path
) -> None:
    directory, home, url = grant
    environment = {
        "HOME": "/original/build-home",
        ALLOCATION_ENV: str(directory),
        POSTGRES_URL_ENV: url,
    }
    owner_file = directory / OWNER_FILE
    owner = json.loads(owner_file.read_text())
    if defect == "empty-reference":
        environment[ALLOCATION_ENV] = " "
    elif defect == "blank-owner":
        owner["owner"] = " "
    elif defect == "bad-port":
        owner["port"] = 65536
    elif defect == "bool-port":
        owner["port"] = True
    elif defect == "different-port":
        owner["port"] = 54322
    elif defect == "bad-cid":
        owner["container"] = "short-id"
    elif defect == "relative-home":
        owner["pgpass_home"] = "relative/home"
    owner_file.write_text(json.dumps(owner))
    if defect.startswith("missing-"):
        {
            "missing-owner": owner_file,
            "missing-url": directory / URL_FILE,
            "missing-pgpass": home / ".pgpass",
        }[defect].unlink()
    elif defect == "owner-not-json":
        owner_file.write_text("invalid")
    elif defect == "owner-not-object":
        owner_file.write_text("[]")
    elif defect == "wrong-url":
        environment[POSTGRES_URL_ENV] = url + "-different"
    elif defect in {"password-url", "foreign-host"}:
        value = (
            url.replace("postgres@", "postgres:example-password@")
            if defect == "password-url"
            else url.replace("127.0.0.1", "foreign.example")
        )
        (directory / URL_FILE).write_text(value)
        environment[POSTGRES_URL_ENV] = value
    elif defect in {"public-directory", "public-home"}:
        (directory if defect == "public-directory" else home).chmod(0o755)
    elif defect in {"public-owner", "public-pgpass"}:
        (owner_file if defect == "public-owner" else home / ".pgpass").chmod(0o644)
    elif defect in {"owner-symlink", "pgpass-symlink"}:
        selected = owner_file if defect == "owner-symlink" else home / ".pgpass"
        other = tmp_path / "foreign"
        other.write_bytes(selected.read_bytes())
        other.chmod(0o600)
        selected.unlink()
        selected.symlink_to(other)
    before = dict(environment)
    with pytest.raises(PostgresGrantError, match="matching private allocation grant"):
        postgres_test_environment(environment)
    assert environment == before


@pytest.mark.parametrize("local", [False, True])
def test_full_release_gates_only_give_postgres_the_grant_home(
    grant, monkeypatch: pytest.MonkeyPatch, local: bool
) -> None:
    directory, home, url = grant
    original = "/original/build-home"
    monkeypatch.setenv("HOME", original)
    if local:
        monkeypatch.setenv(ALLOCATION_ENV, str(directory))
    else:
        monkeypatch.delenv(ALLOCATION_ENV, raising=False)
    monkeypatch.setenv(POSTGRES_URL_ENV, url)
    monkeypatch.setenv("PGPASSFILE", "/original/pgpass")
    environments: dict[str, dict[str, str]] = {}
    invocations: dict[str, list[str]] = {}

    class FakeProcess:
        pid = 2_147_483_647

        def __init__(self, command, **options):
            environments[command[1]] = dict(options["env"])
            invocations[command[1]] = list(command)
            self.stdout = io.StringIO("")

        def wait(self):
            return 0

        def terminate(self):
            pytest.fail("successful fake gate was cancelled")

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_options: subprocess.CompletedProcess(command, 0),
    )
    run_release_gates.run_release_gates("example-python")
    postgres_target = "test-postgres-stress"
    assert set(environments) == {
        "check-static",
        postgres_target,
        "test-parallel-release",
        "artifact-check",
    }
    assert ("POSTGRES_TEST_PARALLEL=1" in invocations[postgres_target]) is local
    for name, environment in environments.items():
        assert environment["HOME"] == (
            str(home) if local and name == postgres_target else original
        )
        assert environment["PGPASSFILE"] == (
            str(home / ".pgpass")
            if local and name == postgres_target
            else "/original/pgpass"
        )
    assert os.environ["HOME"] == original
    assert os.environ["PGPASSFILE"] == "/original/pgpass"


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("custom", [False, True])
def test_impact_selected_postgres_uses_the_same_child_environment(
    grant, monkeypatch: pytest.MonkeyPatch, full: bool, local: bool, custom: bool
) -> None:
    directory, home, url = grant
    monkeypatch.setenv("HOME", "/original/build-home")
    if local:
        monkeypatch.setenv(ALLOCATION_ENV, str(directory))
    else:
        monkeypatch.delenv(ALLOCATION_ENV, raising=False)
    monkeypatch.setenv(POSTGRES_URL_ENV, url)
    calls: list[tuple[list[str], dict]] = []

    def run(command, **options):
        calls.append((list(command), options))
        output = (
            json.dumps(["tests/metrics/test_closed_loop_promql.py"])
            if "promql-test-files" in command
            else ""
        )
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(
        impact, "subprocess", SimpleNamespace(run=run, PIPE=subprocess.PIPE)
    )
    plan = impact.Plan(
        changed_files=(),
        domains=(),
        pytest_targets=("tests/example.py",),
        checks=("docs-check",),
        safe_cases=(),
        approval_cases=(),
        not_selected_families=(),
        full=full,
        postgres=True,
        reasons=(),
    )
    settings = impact.load_settings()
    if custom:
        settings = replace(settings, postgres_command=("make", "custom-postgres-gate"))
    impact.execute_plan(plan, settings)
    if full and local and not custom:
        assert len(calls) == 1, "the unified full gate was followed by duplicate tests"
        assert calls[0][0] == [
            impact.sys.executable,
            str(impact.ROOT / "scripts" / "run_release_gates.py"),
            "--mode",
            "release",
            "--python",
            impact.sys.executable,
        ], "local full impact did not use the existing parallel release gate"
        assert "env" not in calls[0][1], "the full gate inherited the PG-only HOME"
        assert os.environ["HOME"] == "/original/build-home"
        return
    expected_command = (
        ("make", "test-postgres-stress", "POSTGRES_TEST_PARALLEL=1")
        if local and not custom
        else settings.postgres_command
    )
    postgres = [
        (command, options)
        for command, options in calls
        if command[: len(expected_command)] == list(expected_command)
    ]
    assert len(postgres) == 1
    assert postgres[0][1]["env"]["HOME"] == (
        str(home) if local else "/original/build-home"
    )
    assert postgres[0][1]["env"].get("PGPASSFILE") == (
        str(home / ".pgpass") if local else os.environ.get("PGPASSFILE")
    )
    assert postgres[0][1]["env"][POSTGRES_URL_ENV] == url
    metadata = [
        (command, options)
        for command, options in calls
        if "promql-test-files" in command
    ]
    assert len(metadata) == int(not full)
    for _command, options in metadata:
        assert options["env"]["HOME"] == "/original/build-home"
        assert options["env"].get("PGPASSFILE") == os.environ.get("PGPASSFILE")
        assert options["timeout"] == 30
    assert all(
        "env" not in options
        for command, options in calls
        if (command, options) not in postgres + metadata
    ), "impact planning changed HOME for a non-PostgreSQL subprocess"
    assert os.environ["HOME"] == "/original/build-home"


def test_grant_modified_during_read_is_rejected(
    grant, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, _home, url = grant
    target = directory / OWNER_FILE
    original = os.fstat
    reads = 0

    def mutate_during_read(descriptor):
        nonlocal reads
        if Path(f"/proc/self/fd/{descriptor}").resolve() == target:
            reads += 1
            if reads == 3:
                with target.open("a") as output:
                    output.write(" ")
        return original(descriptor)

    # postgres_grant.os is the global os module, which pytest's own tmp_path teardown also
    # uses; keep the fake to the call under test.
    with monkeypatch.context() as patch:
        patch.setattr(postgres_grant.os, "fstat", mutate_during_read)
        with pytest.raises(
            PostgresGrantError, match="matching private allocation grant"
        ):
            postgres_test_environment(
                {
                    ALLOCATION_ENV: str(directory),
                    POSTGRES_URL_ENV: url,
                    "HOME": "/original",
                }
            )
    assert reads == 3
