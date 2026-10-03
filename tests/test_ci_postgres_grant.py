"""Actions grant validation uses mocked Docker metadata, never a database."""

from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

from scripts import ci_postgres_grant as grant

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = "a" * 64
PASSWORD = "unit-only:password\\fixture"


@pytest.fixture
def job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    temporary = tmp_path / "runner"
    workspace = tmp_path / "checkout"
    home = tmp_path / "original-home"
    for path in (temporary, workspace, home):
        path.mkdir()
    env_file = temporary / "github-env"
    env_file.touch()
    environment = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_JOB": "postgres",
        "GITHUB_REPOSITORY_ID": "100",
        "GITHUB_RUN_ID": "200",
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_TEMP": str(temporary),
        "GITHUB_WORKSPACE": str(workspace),
        "GITHUB_ENV": str(env_file),
        "HOME": str(home),
        "GPU_FAULT_TEST_POSTGRES_URL": "postgresql://gpu_fault@127.0.0.1:5432/gpu_fault_test",
    }
    inspected = {
        "id": CONTAINER,
        "running": True,
        "labels": {grant.OWNER_LABEL: grant.job_owner(environment)},
        "ports": {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5432"}]},
        "environment": [
            "POSTGRES_USER=gpu_fault",
            "POSTGRES_DB=gpu_fault_test",
            "POSTGRES_PASSWORD=" + PASSWORD,
        ],
    }
    calls = []

    def inspect(container):
        calls.append(container)
        return copy.deepcopy(inspected)

    monkeypatch.setattr(grant, "inspect_container", inspect)
    return environment, inspected, calls


def updates(environment: dict[str, str]) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in Path(environment["GITHUB_ENV"]).read_text().splitlines()
    )


def test_prepare_writes_private_credentials_and_password_free_references(job):
    environment, _inspected, calls = job
    grant.prepare_grant(environment, CONTAINER)
    values = updates(environment)
    allocation = Path(values["PYTEST_GPU_FAULT_POSTGRES_ALLOCATION_DIR"])
    home = Path(values["HOME"])
    assert calls == [CONTAINER], "only the Actions-owned service may be inspected"
    assert allocation.is_relative_to(Path(environment["RUNNER_TEMP"])), (
        "the allocation must not enter the checkout or uploaded evidence"
    )
    assert not allocation.is_relative_to(Path(environment["GITHUB_WORKSPACE"])), (
        "credentials cannot become source or release artifacts"
    )
    for directory in (allocation, home):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700, (
            "grant directories are private"
        )
    for path in (
        allocation / grant.OWNER_FILE,
        allocation / grant.URL_FILE,
        home / ".pgpass",
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, (
            "grant records are owner-only"
        )
    metadata = json.loads((allocation / grant.OWNER_FILE).read_text())
    assert metadata["owner"] == grant.job_owner(environment), (
        "owner binds repo/run/attempt/job"
    )
    assert metadata["container"] == CONTAINER, (
        "native fixtures receive the full service ID"
    )
    assert metadata["pgpass_home"] == str(home), "the grant binds its credential home"
    assert urlsplit(values["GPU_FAULT_TEST_POSTGRES_URL"]).password is None, (
        "native grant URLs cannot contain passwords"
    )
    assert PASSWORD not in Path(environment["GITHUB_ENV"]).read_text(), (
        "Actions environment output contains references, not credentials"
    )
    assert PASSWORD not in (allocation / grant.OWNER_FILE).read_text(), (
        "ownership metadata cannot duplicate the credential"
    )
    expected = PASSWORD.replace("\\", "\\\\").replace(":", "\\:")
    assert (
        home / ".pgpass"
    ).read_text() == f"127.0.0.1:5432:*:gpu_fault:{expected}\n", (
        "only the database field varies for private child databases on this service"
    )


def test_existing_native_fixture_consumes_the_exact_grant_shape(job, monkeypatch):
    from tests.regional import _cov95_notify008_postgres as native

    environment, inspected, _calls = job
    grant.prepare_grant(environment, CONTAINER)
    for name, value in updates(environment).items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    checked = []
    monkeypatch.setattr(native, "validate_server", checked.append)
    monkeypatch.setattr(
        "psycopg.connect",
        lambda *_args, **_kwargs: pytest.fail(
            "grant shape tests must never connect to SQL"
        ),
    )

    def inspect(command, **_kwargs):
        assert command[:2] == ["docker", "inspect"] and command[-1] == CONTAINER, (
            "the unchanged native fixture must inspect the exact granted service"
        )
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(inspected))

    monkeypatch.setattr(native.subprocess, "run", inspect)
    value = native.validated_grant()
    assert value == updates(environment)["GPU_FAULT_TEST_POSTGRES_URL"], (
        "the native fixture accepts the standard grant, without a CI bypass"
    )
    assert checked == [value], "container proof still requires native server validation"


@pytest.mark.parametrize(
    "defect",
    [
        "foreign-id",
        "foreign-label",
        "stopped",
        "running-not-bool",
        "public-port",
        "ipv6",
        "wrong-port",
        "two-bindings",
        "missing-ports",
        "remote-url",
        "hostaddr",
        "user",
        "database",
        "password",
        "missing-password",
        "duplicate-password",
        "environment",
        "invalid-url",
    ],
)
def test_foreign_or_ambiguous_service_never_receives_a_grant(job, defect):
    environment, inspected, _calls = job
    if defect == "foreign-id":
        inspected["id"] = "b" * 64
    elif defect == "foreign-label":
        inspected["labels"][grant.OWNER_LABEL] = "another-job"
    elif defect in {"stopped", "running-not-bool"}:
        inspected["running"] = False if defect == "stopped" else 1
    elif defect in {"public-port", "ipv6"}:
        inspected["ports"]["5432/tcp"][0]["HostIp"] = (
            "0.0.0.0" if defect == "public-port" else "::1"
        )
    elif defect == "wrong-port":
        inspected["ports"]["5432/tcp"][0]["HostPort"] = "5433"
    elif defect == "two-bindings":
        inspected["ports"]["5432/tcp"].append({"HostIp": "0.0.0.0", "HostPort": "5432"})
    elif defect == "missing-ports":
        inspected["ports"] = None
    elif defect == "hostaddr":
        environment["GPU_FAULT_TEST_POSTGRES_URL"] += "?hostaddr=192.0.2.1"
    elif defect in {"remote-url", "user", "database", "password", "invalid-url"}:
        environment["GPU_FAULT_TEST_POSTGRES_URL"] = {
            "remote-url": "postgresql://gpu_fault@192.0.2.1:5432/gpu_fault_test",
            "user": "postgresql://foreign@127.0.0.1:5432/gpu_fault_test",
            "database": "postgresql://gpu_fault@127.0.0.1:5432/foreign",
            "password": "postgresql://gpu_fault:wrong@127.0.0.1:5432/gpu_fault_test",
            "invalid-url": "not a connection string",
        }[defect]
    elif defect == "missing-password":
        inspected["environment"].pop()
    elif defect == "duplicate-password":
        inspected["environment"].append("POSTGRES_PASSWORD=ambiguous")
    else:
        inspected["environment"] = None
    with pytest.raises(grant.PostgresGrantError):
        grant.prepare_grant(environment, CONTAINER)
    assert not list(Path(environment["RUNNER_TEMP"]).glob("gpu-fault-ci-*")), (
        "unbound services must not receive credential files"
    )
    assert Path(environment["GITHUB_ENV"]).read_text() == "", (
        "refused grants must not publish usable environment references"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "relative",
        "checkout",
        "symlink-temp",
        "foreign-env",
        "symlink-env",
        "existing",
        "short-id",
        "wrong-job",
    ],
)
def test_paths_and_job_identity_fail_closed_before_inspection(job, tmp_path, defect):
    environment, _inspected, calls = job
    container = CONTAINER
    if defect == "relative":
        environment["RUNNER_TEMP"] = "."
    elif defect == "checkout":
        environment["RUNNER_TEMP"] = environment["GITHUB_WORKSPACE"]
    elif defect == "symlink-temp":
        alias = tmp_path / "alias"
        alias.symlink_to(environment["RUNNER_TEMP"], target_is_directory=True)
        environment["RUNNER_TEMP"] = str(alias)
    elif defect == "foreign-env":
        other = tmp_path / "foreign-env"
        other.touch()
        environment["GITHUB_ENV"] = str(other)
    elif defect == "symlink-env":
        alias = Path(environment["RUNNER_TEMP"]) / "alias-env"
        alias.symlink_to(environment["GITHUB_ENV"])
        environment["GITHUB_ENV"] = str(alias)
    elif defect == "existing":
        grant.grant_paths(environment)[1].mkdir()
    elif defect == "short-id":
        container = container[:12]
    else:
        environment["GITHUB_JOB"] = "unit"
    with pytest.raises(grant.PostgresGrantError):
        grant.prepare_grant(environment, container)
    assert calls == [], "invalid paths or job context cannot inspect another service"


@pytest.mark.parametrize(
    "name",
    [
        "PGHOSTADDR",
        "PGPASSWORD",
        "PGSERVICE",
        "PGSERVICEFILE",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
    ],
)
def test_ambient_connection_overrides_cannot_redirect_the_grant(job, name):
    environment, _inspected, calls = job
    environment[name] = "untrusted override"
    with pytest.raises(grant.PostgresGrantError):
        grant.prepare_grant(environment, CONTAINER)
    assert calls == [], (
        "ambient connection overrides must fail before service inspection"
    )


def test_cleanup_removes_only_its_grant_and_restores_home(job, tmp_path):
    environment, _inspected, calls = job
    grant.prepare_grant(environment, CONTAINER)
    allocation = Path(updates(environment)["PYTEST_GPU_FAULT_POSTGRES_ALLOCATION_DIR"])
    outside = tmp_path / "unrelated"
    outside.write_text("keep", encoding="utf-8")
    (allocation / "home" / "link").symlink_to(outside)
    grant.cleanup_grant(environment, CONTAINER)
    assert not allocation.exists(), "private credentials must be removed after the job"
    assert outside.read_text() == "keep", (
        "cleanup cannot follow a symlink outside its grant"
    )
    assert updates(environment)["HOME"] == environment["HOME"], (
        "post-job hooks regain the original home"
    )
    assert updates(environment)["PGPASSFILE"] == "", (
        "expired credential references are cleared"
    )
    assert calls == [CONTAINER], (
        "cleanup never stops, relabels or inspects the container"
    )
    grant.cleanup_grant(environment, CONTAINER)


@pytest.mark.parametrize("defect", ["owner", "container", "inode", "mode", "symlink"])
def test_cleanup_refuses_foreign_or_replaced_grants(job, tmp_path, defect):
    environment, _inspected, _calls = job
    grant.prepare_grant(environment, CONTAINER)
    allocation = Path(updates(environment)["PYTEST_GPU_FAULT_POSTGRES_ALLOCATION_DIR"])
    marker = allocation / grant.OWNER_FILE
    metadata = json.loads(marker.read_text())
    if defect == "owner":
        metadata["owner"] = "another-job"
    elif defect == "container":
        metadata["container"] = "b" * 64
    elif defect == "inode":
        metadata["allocation_inode"] += 1
    elif defect == "mode":
        marker.chmod(0o644)
    else:
        preserved = tmp_path / "preserved"
        allocation.rename(preserved)
        allocation.symlink_to(preserved, target_is_directory=True)
    if defect in {"owner", "container", "inode"}:
        marker.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(grant.PostgresGrantError):
        grant.cleanup_grant(environment, CONTAINER)
    assert allocation.exists(), "a cleanup refusal must preserve the unproven target"


def test_failed_publication_cleans_its_new_private_files(job, monkeypatch):
    environment, _inspected, _calls = job
    real_open = os.open

    def open_file(path, flags, *args, **kwargs):
        if Path(path) == Path(environment["GITHUB_ENV"]):
            raise OSError("simulated publication failure")
        return real_open(path, flags, *args, **kwargs)

    # grant.os is the global os module, which pytest's own tmp_path teardown also
    # uses; keep the fake to the call under test.
    with monkeypatch.context() as patch:
        patch.setattr(grant.os, "open", open_file)
        with pytest.raises(OSError, match="publication"):
            grant.prepare_grant(environment, CONTAINER)
    assert not list(Path(environment["RUNNER_TEMP"]).glob("gpu-fault-ci-*")), (
        "a failed prepare cannot leave its database credential behind"
    )


def test_cli_never_prints_credential_material(job, monkeypatch, capsys):
    environment, _inspected, _calls = job
    # The CLI must receive the same explicit environment as the function tests.
    with monkeypatch.context() as cli_environment:
        cli_environment.setattr(grant.os, "environ", environment)
        assert grant.main(["prepare", "--container", CONTAINER]) == 0, (
            "the bound job can prepare"
        )
        assert PASSWORD not in capsys.readouterr().out, (
            "success output contains no database password"
        )
        assert grant.main(["cleanup", "--container", CONTAINER]) == 0, (
            "the owned grant can clean"
        )
        assert PASSWORD not in capsys.readouterr().out, (
            "cleanup output contains no database password"
        )


def test_cli_refuses_local_docker_host_before_inspection(job, monkeypatch, capsys):
    environment, _inspected, calls = job
    environment["DOCKER_HOST"] = "unix:///var/run/docker.sock"
    with monkeypatch.context() as cli_environment:
        cli_environment.setattr(grant.os, "environ", environment)
        result = grant.main(["prepare", "--container", CONTAINER])
    assert result == 2, "even a local Docker override must fail closed"
    captured = capsys.readouterr()
    redacted = (
        captured.out == "ci-postgres-grant: refused unsafe or incomplete job grant\n"
        and captured.err == ""
    )
    assert redacted, "a refused CLI grant must not expose credential material"
    assert calls == [], "the CLI must reject the override before service inspection"
    assert not list(Path(environment["RUNNER_TEMP"]).glob("gpu-fault-ci-*")), (
        "a refused CLI grant must not create private credential files"
    )
    assert updates(environment) == {}, "a refused CLI grant cannot publish references"


def test_actions_declares_and_cleans_the_job_bound_native_grant():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    postgres = workflow["jobs"]["postgres"]
    service = postgres["services"]["postgres"]
    assert service["ports"] == ["127.0.0.1:5432:5432"], (
        "the test service must not publish PostgreSQL on public or IPv6 interfaces"
    )
    expected_label = (
        "gpu-fault.test-owner=gpu-fault-ci-${{ github.repository_id }}-"
        "${{ github.run_id }}-${{ github.run_attempt }}-postgres"
    )
    assert expected_label in service["options"], (
        "Actions must create the service with the same repo/run/attempt/job owner"
    )
    assert (
        urlsplit(postgres["env"]["GPU_FAULT_TEST_POSTGRES_URL"].strip()).password
        is None
    ), "the job environment and step logs must not carry a password-bearing URL"
    steps = postgres["steps"]
    names = [step.get("name") for step in steps]
    prepare = names.index("Prepare private PostgreSQL process grant")
    restore = names.index("Restore signed PostgreSQL coverage shard")
    execute = names.index("Run PostgreSQL coverage and stress shard")
    upload = names.index("Upload PostgreSQL coverage shard")
    cleanup = names.index("Clean private PostgreSQL process grant")
    assert prepare < restore < execute < upload < cleanup, (
        "the stable grant environment precedes identity/reuse and survives through evidence upload"
    )
    assert steps[cleanup]["if"] == "always()", (
        "failures must still remove grant credentials"
    )
    assert "if" not in steps[prepare], (
        "both PR and main shards need a real process grant"
    )
    for index, action in ((prepare, "prepare"), (cleanup, "cleanup")):
        command = steps[index]["run"]
        assert f"scripts/ci_postgres_grant.py {action}" in command, (
            "the workflow must use the guarded native-grant entrypoint"
        )
        assert '--container "${{ job.services.postgres.id }}"' in command, (
            "the grant may bind only the service allocated to this job"
        )


@pytest.mark.parametrize("case", ["valid", "failed", "malformed", "not-object"])
def test_inspection_is_read_only_captured_and_rejects_untrusted_output(
    monkeypatch, case
):
    def run(command, **options):
        assert command[:4] == [
            "docker",
            "--host",
            "unix:///var/run/docker.sock",
            "inspect",
        ], "inspection cannot mutate the service or use a remote Docker context"
        assert command[-1] == CONTAINER, "inspection targets only the full service ID"
        assert options["capture_output"] is True, (
            "container environment must never reach logs"
        )
        output = {
            "valid": '{"id":"bound"}',
            "failed": PASSWORD,
            "malformed": "{",
            "not-object": "[]",
        }[case]
        return subprocess.CompletedProcess(
            command, int(case == "failed"), stdout=output
        )

    monkeypatch.setattr(grant.subprocess, "run", run)
    if case == "valid":
        assert grant.inspect_container(CONTAINER) == {"id": "bound"}, (
            "decoded inspection data is handed to the ownership validator"
        )
    else:
        with pytest.raises(grant.PostgresGrantError) as raised:
            grant.inspect_container(CONTAINER)
        assert PASSWORD not in str(raised.value), (
            "inspection failures cannot echo credential output"
        )
