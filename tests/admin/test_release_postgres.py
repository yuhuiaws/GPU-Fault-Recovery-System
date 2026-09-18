from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from gpu_fault.admin import execution, release_artifacts, release_postgres
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.execution import deadline_scope, recovery_active
from gpu_fault.admin.postgres_grant import (
    ALLOCATION_ENV,
    LOCAL_DOCKER_HOST,
    OWNER_FILE,
    OWNER_LABEL,
    POSTGRES_URL_ENV,
    postgres_test_environment,
)
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.admin.release_postgres import (
    PostgresCleanupError,
    isolated_postgres_allocation,
)
from tests.admin._release_postgres_support import (
    CID,
    FOREIGN_CID,
    IMAGE,
    FakePostgresDocker,
)


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakePostgresDocker:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(release_postgres.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        release_postgres.secrets, "token_urlsafe", lambda _size: "example-password"
    )
    monkeypatch.delenv(POSTGRES_URL_ENV, raising=False)
    monkeypatch.delenv(ALLOCATION_ENV, raising=False)
    return FakePostgresDocker()


def metadata(docker: FakePostgresDocker) -> dict:
    assert docker.directory is not None, "fake creation has no ownership directory"
    return json.loads((docker.directory / "ownership.json").read_text())


def assert_removed(docker: FakePostgresDocker) -> None:
    assert docker.containers == {}
    assert docker.removed == [CID]
    assert docker.directory is not None and not docker.directory.exists(), (
        "confirmed cleanup left private credentials or metadata behind"
    )


def test_allocation_is_bound_before_start_and_supplies_a_private_grant(
    docker: FakePostgresDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    before_home = "/original/build-home"
    monkeypatch.setenv("HOME", before_home)
    monkeypatch.setenv("DOCKER_CONTEXT", "build-context")
    monkeypatch.setenv("COSIGN_PASSWORD", "example-signing-value")

    def before_create(item: dict) -> None:
        record = metadata(docker)
        assert record["phase"] == "CREATE_STARTED"
        assert record["owner"] == item["labels"][OWNER_LABEL]
        assert record["image_id"] == IMAGE
        assert record["container_id"] is None

    def started(item: dict) -> None:
        record = metadata(docker)
        assert record["container_id"] == item["id"] == CID
        assert record["phase"] == "CREATED"

    docker.hooks.update({"before-create": before_create, "start": started})
    with isolated_postgres_allocation(docker) as allocation:
        assert allocation.directory is not None, "owned allocation has no grant"
        assert allocation.url == "postgresql://postgres@127.0.0.1:54321/postgres"
        owner = json.loads((allocation.directory / OWNER_FILE).read_text())
        home = Path(owner["pgpass_home"])
        assert owner["container"] == CID and owner["port"] == 54321
        assert owner["image_id"] == IMAGE
        assert home.stat().st_mode & 0o777 == 0o700
        for name in (OWNER_FILE, "cov95-postgres-url", "postgres.env"):
            assert (allocation.directory / name).stat().st_mode & 0o777 == 0o600
        assert (home / ".pgpass").stat().st_mode & 0o777 == 0o600
        build = allocation.build_environment({"HOME": before_home})
        assert build["HOME"] == before_home
        assert "PGPASSFILE" not in build, "build received PG-child-only credentials"
        child = postgres_test_environment(build)
        assert child["HOME"] == str(home)
        assert child["PGPASSFILE"] == str(home / ".pgpass")
        assert child["DOCKER_HOST"] == LOCAL_DOCKER_HOST
        assert metadata(docker)["phase"] == "READY"
    assert_removed(docker)
    for arguments, options in docker.calls:
        assert "example-password" not in repr(arguments)
        assert options["env"]["HOME"] == before_home
        assert "DOCKER_CONTEXT" not in options["env"]
        assert "COSIGN_PASSWORD" not in options["env"]
    removed = [args for args, _ in docker.calls if args[3:5] == ["container", "rm"]]
    assert removed[0][-3:] == ["--force", "--volumes", CID]


def test_external_database_is_neither_created_nor_destroyed(
    docker: FakePostgresDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "postgresql://external.example/explicit-test-database"
    monkeypatch.setenv(POSTGRES_URL_ENV, url)
    with isolated_postgres_allocation(docker) as allocation:
        assert allocation.url == url and allocation.directory is None
    assert docker.calls == []
    with release_artifacts.isolated_postgres_url(docker) as result:
        assert result == url
    assert docker.calls == []


@pytest.mark.parametrize("image_present", [False, True])
def test_missing_image_is_pulled_once_then_created_by_exact_id(
    docker: FakePostgresDocker, image_present: bool
) -> None:
    docker.images = image_present
    with isolated_postgres_allocation(docker):
        pass
    assert docker.pulls == int(not image_present)
    creation = next(args for args, _ in docker.calls if args[3] == "create")
    assert creation[-2:] == ["--", IMAGE]
    assert_removed(docker)


@pytest.mark.parametrize(
    "operation,output",
    [
        ("image-list", "short-id"),
        ("image-list", IMAGE + "\n" + IMAGE),
        ("image-inspect", "invalid"),
        ("image-inspect", json.dumps("sha256:" + "f" * 64)),
    ],
)
def test_image_uncertainty_cannot_create_a_database(
    docker: FakePostgresDocker, operation: str, output: str
) -> None:
    docker.responses[operation] = output
    with pytest.raises((BootstrapError, ValueError)):
        with isolated_postgres_allocation(docker):
            pytest.fail("invalid image authorized a database")
    assert docker.created is None
    assert docker.containers == {}


@pytest.mark.parametrize("failure", ["timeout", "interrupt", "body", "start"])
def test_failure_cleans_by_owned_id_before_propagating(
    docker: FakePostgresDocker, failure: str
) -> None:
    error = (
        KeyboardInterrupt()
        if failure == "interrupt"
        else subprocess.TimeoutExpired(["fake-docker"], 1)
        if failure == "timeout"
        else RuntimeError("example failure")
    )
    if failure == "start":
        docker.failures["start"] = error
    with pytest.raises(type(error)):
        with isolated_postgres_allocation(docker):
            raise error
    assert_removed(docker)


@pytest.mark.parametrize("has_cid", [False, True])
def test_create_ack_loss_needs_the_original_cid_before_cleanup(
    docker: FakePostgresDocker, has_cid: bool
) -> None:
    def lost_ack(_item: dict) -> None:
        assert docker.directory is not None, "fake creation has no directory"
        if not has_cid:
            (docker.directory / "container.cid").unlink()
        raise subprocess.TimeoutExpired(["fake-docker"], 1)

    docker.hooks["after-create"] = lost_ack
    expected = subprocess.TimeoutExpired if has_cid else PostgresCleanupError
    with pytest.raises(expected):
        with isolated_postgres_allocation(docker):
            pytest.fail("unacknowledged create reached the release gate")
    assert docker.starts == []
    if has_cid:
        assert_removed(docker)
    else:
        assert docker.removal_attempts == []
        assert CID in docker.containers
        assert metadata(docker)["phase"] == "UNCONFIRMED"


def test_unknown_create_without_any_cid_retains_intent(
    docker: FakePostgresDocker,
) -> None:
    docker.failures["before-create"] = OSError("unknown create outcome")
    with pytest.raises(PostgresCleanupError, match="unconfirmed"):
        with isolated_postgres_allocation(docker):
            pytest.fail("unknown creation was accepted")
    assert docker.starts == docker.removal_attempts == []
    assert metadata(docker)["create_attempted"] is True
    assert metadata(docker)["phase"] == "UNCONFIRMED"


@pytest.mark.parametrize("field", ["id", "name", "image", "reference", "owner", "port"])
def test_identity_or_port_drift_never_authorizes_removal(
    docker: FakePostgresDocker, field: str
) -> None:
    with pytest.raises(PostgresCleanupError):
        with isolated_postgres_allocation(docker):
            item = docker.containers[CID]
            if field == "owner":
                item["labels"][OWNER_LABEL] = "foreign-owner"
            elif field == "port":
                item["ports"]["5432/tcp"][0]["HostPort"] = "54322"
            else:
                item[field] = FOREIGN_CID if field == "id" else "foreign"
    assert docker.removal_attempts == []
    assert metadata(docker)["phase"] == "UNCONFIRMED"


def test_copied_name_and_labels_cannot_replace_lost_creation_identity(
    docker: FakePostgresDocker,
) -> None:
    def replace(item: dict) -> None:
        assert docker.directory is not None, "fake creation has no directory"
        (docker.directory / "container.cid").unlink()
        docker.containers.pop(CID)
        docker.containers[FOREIGN_CID] = {**item, "id": FOREIGN_CID}
        raise OSError("creation reply lost")

    docker.hooks["after-create"] = replace
    with pytest.raises(PostgresCleanupError):
        with isolated_postgres_allocation(docker):
            pytest.fail("replacement was adopted")
    assert docker.starts == docker.removal_attempts == []
    assert FOREIGN_CID in docker.containers


def test_replacement_between_inspection_and_remove_cannot_change_target(
    docker: FakePostgresDocker,
) -> None:
    def replace(item: dict) -> None:
        docker.containers.pop(CID)
        docker.containers[FOREIGN_CID] = {**item, "id": FOREIGN_CID}

    docker.hooks["before-remove"] = replace
    with pytest.raises(PostgresCleanupError):
        with isolated_postgres_allocation(docker):
            pass
    assert docker.removal_attempts == [CID]
    assert FOREIGN_CID in docker.containers


@pytest.mark.parametrize("contents", ["bad", "d" * 64, "e" * 100])
def test_invalid_or_changed_cid_file_blocks_removal(
    docker: FakePostgresDocker, contents: str
) -> None:
    with pytest.raises(PostgresCleanupError):
        with isolated_postgres_allocation(docker):
            assert docker.directory is not None, "fake creation has no directory"
            (docker.directory / "container.cid").write_text(contents)
    assert docker.removal_attempts == []


def test_cid_symlink_is_not_followed(
    docker: FakePostgresDocker, tmp_path: Path
) -> None:
    target = tmp_path / "foreign-cid"
    target.write_text(FOREIGN_CID)
    with pytest.raises(PostgresCleanupError):
        with isolated_postgres_allocation(docker):
            assert docker.directory is not None, "fake creation has no directory"
            cidfile = docker.directory / "container.cid"
            cidfile.unlink()
            cidfile.symlink_to(target)
    assert target.read_text() == FOREIGN_CID
    assert docker.removal_attempts == []


@pytest.mark.parametrize("failed_operation", ["list", "inspect"])
def test_unknown_inspection_is_not_absence(
    docker: FakePostgresDocker, failed_operation: str
) -> None:
    with pytest.raises(PostgresCleanupError):
        with isolated_postgres_allocation(docker):
            docker.failures[failed_operation] = BootstrapError("daemon unavailable")
    assert docker.removal_attempts == []
    assert CID in docker.containers


@pytest.mark.parametrize("retained", [False, True])
def test_delete_ack_loss_requires_fresh_absence(
    docker: FakePostgresDocker, retained: bool
) -> None:
    docker.keep_after_remove = retained
    docker.failures["after-remove"] = subprocess.TimeoutExpired(["fake-docker"], 1)
    if retained:
        with pytest.raises(PostgresCleanupError):
            with isolated_postgres_allocation(docker):
                pass
        assert metadata(docker)["phase"] == "UNCONFIRMED"
    else:
        with isolated_postgres_allocation(docker):
            pass
        assert_removed(docker)


def test_confirmed_external_removal_does_not_issue_a_delete(
    docker: FakePostgresDocker,
) -> None:
    with isolated_postgres_allocation(docker):
        docker.containers.pop(CID)
    assert docker.removal_attempts == []
    assert docker.directory is not None and not docker.directory.exists(), (
        "confirmed absence left the private grant behind"
    )


@pytest.mark.parametrize("during_cleanup", [False, True])
def test_supervision_loss_propagates_without_further_commands(
    docker: FakePostgresDocker, during_cleanup: bool
) -> None:
    error = ProcessSupervisionLost("example supervisor loss")
    calls_at_loss: list[int] = []

    def lost(_item: dict) -> None:
        calls_at_loss.append(len(docker.calls))
        raise error

    if during_cleanup:
        docker.hooks["before-remove"] = lost
    with pytest.raises(ProcessSupervisionLost) as caught:
        with isolated_postgres_allocation(docker):
            if not during_cleanup:
                lost({})
    assert caught.value is error
    assert len(docker.calls) == calls_at_loss[0]
    assert metadata(docker)["supervision_lost"] is True
    assert metadata(docker)["phase"] == "SUPERVISION_LOST"
    assert CID in docker.containers


@pytest.mark.parametrize("ready_code", [1, 2, 125])
def test_unready_or_failed_probe_never_reaches_gate(
    docker: FakePostgresDocker, ready_code: int
) -> None:
    docker.ready = False
    docker.ready_code = ready_code
    with pytest.raises(BootstrapError):
        with isolated_postgres_allocation(docker):
            pytest.fail("unready PostgreSQL was granted")
    assert docker.readiness_calls == (60 if ready_code in {1, 2} else 1)
    assert_removed(docker)


def test_cleanup_has_its_bounded_recovery_scope(docker: FakePostgresDocker) -> None:
    checks: list[bool] = []
    docker.hooks["before-remove"] = lambda _item: checks.append(recovery_active())
    with deadline_scope("example parent", 30):
        with isolated_postgres_allocation(docker):
            assert not recovery_active(), "startup used cleanup's recovery privilege"
    assert checks == [True]
    assert_removed(docker)


def test_unconfirmed_site_allocation_blocks_new_build_before_reuse(
    docker: FakePostgresDocker, tmp_path: Path
) -> None:
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    docker.keep_after_remove = True
    with pytest.raises(PostgresCleanupError):
        with isolated_postgres_allocation(docker, state_dir=state):
            pass
    before = len(docker.calls)
    with pytest.raises(PostgresCleanupError, match="explicit cleanup reconciliation"):
        release_artifacts.build_signed_release(
            docker,
            repository_root=tmp_path / "repository",
            state_dir=state,
            region="us-east-1",
            runtime_repository="example.invalid/runtime",
            cache_repository=None,
            runtime_profile="example",
        )
    assert len(docker.calls) == before, "unconfirmed allocation started another build"
    assert metadata(docker)["phase"] == "UNCONFIRMED"


def test_grant_cannot_be_created_inside_repository(
    docker: FakePostgresDocker, tmp_path: Path
) -> None:
    with pytest.raises(BootstrapError, match="outside the checkout"):
        with isolated_postgres_allocation(docker, repository_root=tmp_path):
            pytest.fail("credentials were put in the checkout")
    assert docker.created is None


def test_successful_grant_passes_the_unchanged_native_process_contract(
    docker: FakePostgresDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.regional import _cov95_notify008_postgres as native

    with isolated_postgres_allocation(docker) as allocation:
        parent = allocation.build_environment({"HOME": "/original/build-home"})
        environment = postgres_test_environment(parent)
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
        validations: list[str] = []

        def inspect(arguments: list[str], **_options: object):
            assert arguments[:2] == ["docker", "inspect"]
            assert arguments[-1] == CID
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps(copy.deepcopy(docker.containers[CID])), ""
            )

        with monkeypatch.context() as io:
            io.setattr(subprocess, "run", inspect)
            io.setattr(native, "validate_server", validations.append)
            assert native.validated_grant() == allocation.url
        assert validations == [allocation.url]
    assert_removed(docker)


def test_pull_without_an_observable_image_cannot_create_postgres(
    docker: FakePostgresDocker,
) -> None:
    docker.responses["image-list"] = ""
    with pytest.raises(BootstrapError, match="unavailable after pull"):
        with isolated_postgres_allocation(docker):
            pytest.fail("unobserved image created a database")
    assert docker.pulls == 1 and docker.created is None


@pytest.mark.parametrize("output", ["short", "", CID + "\n" + FOREIGN_CID])
def test_invalid_create_ack_is_not_start_authority(
    docker: FakePostgresDocker, output: str
) -> None:
    docker.responses["create"] = output
    with pytest.raises(BootstrapError, match="invalid CID"):
        with isolated_postgres_allocation(docker):
            pytest.fail("invalid acknowledgement authorized startup")
    assert docker.starts == []
    assert_removed(docker)


@pytest.mark.parametrize("missing", [False, True])
def test_valid_ack_with_missing_cid_file_cleans_but_does_not_start(
    docker: FakePostgresDocker, missing: bool
) -> None:
    def lose_file(_item: dict) -> None:
        assert docker.directory is not None, "fake creation has no directory"
        path = docker.directory / "container.cid"
        if missing:
            path.unlink()
        else:
            path.write_text("")

    docker.hooks["after-create"] = lose_file
    with pytest.raises(BootstrapError, match="matching CID file"):
        with isolated_postgres_allocation(docker):
            pytest.fail("missing creation receipt authorized startup")
    assert docker.starts == []
    assert_removed(docker)


@pytest.mark.parametrize("output", ["short-id", CID + "\n" + FOREIGN_CID])
def test_ambiguous_cleanup_inventory_cannot_authorize_delete(
    docker: FakePostgresDocker, output: str
) -> None:
    with pytest.raises(PostgresCleanupError):
        with isolated_postgres_allocation(docker):
            docker.responses["list"] = output
    assert docker.removal_attempts == []
    assert metadata(docker)["phase"] == "UNCONFIRMED"


def test_id_lookup_must_return_the_exact_full_cid(
    docker: FakePostgresDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = docker.run

    def replaced_query(arguments, **options):
        if arguments[3:5] == ["container", "ls"] and f"id={CID}" in arguments:
            return FOREIGN_CID
        return original(arguments, **options)

    with pytest.raises(PostgresCleanupError):
        with isolated_postgres_allocation(docker):
            monkeypatch.setattr(docker, "run", replaced_query)
    assert docker.removal_attempts == []
    assert CID in docker.containers


@pytest.mark.parametrize(
    "field,value",
    [
        ("bindings", None),
        ("bindings", {}),
        ("bindings", {"5432/tcp": []}),
        ("bindings", {"5432/tcp": [None]}),
        ("bindings", {"5432/tcp": [{"HostIp": "0.0.0.0", "HostPort": ""}]}),
        ("ports", None),
        ("ports", {}),
        ("ports", {"5432/tcp": []}),
        ("ports", {"5432/tcp": [None]}),
        ("ports", {"5432/tcp": [{"HostIp": "0.0.0.0", "HostPort": "54321"}]}),
        ("ports", {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": "65536"}]}),
        ("ports", {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": 54321}]}),
    ],
)
def test_uncertain_or_nonloopback_listener_prevents_cleanup_authorization(
    docker: FakePostgresDocker, field: str, value: object
) -> None:
    with pytest.raises(PostgresCleanupError):
        with isolated_postgres_allocation(docker):
            docker.containers[CID][field] = value
    assert docker.removal_attempts == []
    assert metadata(docker)["phase"] == "UNCONFIRMED"


def test_unstarted_allocation_cannot_issue_a_grant(
    docker: FakePostgresDocker, tmp_path: Path
) -> None:
    directory = tmp_path / "unstarted"
    directory.mkdir(mode=0o700)
    info = directory.stat()
    owned = release_postgres.OwnedPostgres(
        docker, directory, IMAGE, "example-owner", (info.st_dev, info.st_ino)
    )
    with pytest.raises(BootstrapError, match="no complete identity"):
        owned.grant()
    owned.cleanup()
    assert docker.calls == []
    owned.remove_directory()
    assert not directory.exists(), "unstarted private directory was not removed"


def test_replaced_private_directory_is_not_written_or_removed(
    docker: FakePostgresDocker, tmp_path: Path
) -> None:
    old = tmp_path / "original-allocation"
    with pytest.raises(PostgresCleanupError, match="ownership update also failed"):
        with isolated_postgres_allocation(docker) as allocation:
            assert allocation.directory is not None, "allocation has no directory"
            allocation.directory.rename(old)
            allocation.directory.mkdir(mode=0o700)
            (allocation.directory / "foreign").write_text("untouched")
    assert docker.removal_attempts == []
    assert (old / "ownership.json").is_file(), "original intent was erased"
    assert docker.directory is not None
    assert (docker.directory / "foreign").read_text() == "untouched"


def test_unsafe_filesystem_cleanup_retains_private_evidence(
    docker: FakePostgresDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as platform:
        with pytest.raises(PostgresCleanupError):
            with isolated_postgres_allocation(docker):
                platform.setattr(shutil.rmtree, "avoids_symlink_attacks", False)
    assert docker.containers == {}
    assert metadata(docker)["phase"] == "UNCONFIRMED"


@pytest.mark.parametrize("during_cleanup", [False, True])
def test_supervision_loss_is_not_masked_by_metadata_write_failure(
    docker: FakePostgresDocker, monkeypatch: pytest.MonkeyPatch, during_cleanup: bool
) -> None:
    error = ProcessSupervisionLost("example lost ownership")
    at_loss: list[int] = []

    def cannot_record(*_args, **_options):
        raise OSError("example metadata persistence failed")

    def lost(_item: dict) -> None:
        at_loss.append(len(docker.calls))
        monkeypatch.setattr(release_postgres, "write_json_atomic", cannot_record)
        raise error

    if during_cleanup:
        docker.hooks["before-remove"] = lost
    with pytest.raises(ProcessSupervisionLost) as caught:
        with isolated_postgres_allocation(docker):
            if not during_cleanup:
                lost({})
    assert caught.value is error
    assert len(docker.calls) == at_loss[0]
    assert any(
        "could not update private PostgreSQL intent" in note for note in error.__notes__
    ), "failed supervision-loss persistence was not reported"
    assert metadata(docker)["phase"] == "READY"


def test_real_command_runner_keeps_supervision_and_redaction(
    docker: FakePostgresDocker,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    deadlines: list[float] = []

    def supervised(arguments, **options):
        assert options["capture"] is True
        assert options["expires_at"] > 0
        deadlines.append(options["timeout"])
        output = docker.run(
            arguments,
            capture=True,
            sensitive=True,
            env=options["environment"],
            timeout_seconds=options["timeout"],
        )
        return subprocess.CompletedProcess(arguments, 0, output, "")

    monkeypatch.setattr(execution, "run_owned_command", supervised)
    with isolated_postgres_allocation(CommandRunner()):
        pass
    assert deadlines and all(0 < value <= 600 for value in deadlines), (
        "an owned PostgreSQL command escaped the checked timeout boundary"
    )
    assert_removed(docker)
    output = capsys.readouterr()
    assert "example-password" not in output.out + output.err
    assert "<sensitive command>" in output.err
