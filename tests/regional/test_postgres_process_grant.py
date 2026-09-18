"""Private grant validation uses fake Docker/server I/O, never a database."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from tests.regional import _cov95_notify008_postgres as grants

validate: Callable[[], str] = cast(Any, grants.validated_grant)
ALLOCATION_DIR_ENV = "PYTEST_GPU_FAULT_POSTGRES_ALLOCATION_DIR"


def allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    port: int,
    *,
    directory: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[object], str]:
    monkeypatch.delenv(ALLOCATION_DIR_ENV, raising=False)
    if directory is None:
        directory = tmp_path / ".codex"
    directory.mkdir(mode=0o700)
    home = directory / "pgpass"
    home.mkdir(mode=0o700)
    owner = {
        "container": f"{port:064x}",
        "owner": f"owned-test-allocation-{port}",
        "port": str(port),
        "pgpass_home": str(home),
    }
    url = f"postgresql://test@127.0.0.1:{port}/owned"
    (directory / "cov95-postgres-owner.json").write_text(json.dumps(owner))
    (directory / "cov95-postgres-url").write_text(url)
    actual: dict[str, Any] = {
        "id": owner["container"],
        "running": True,
        "labels": {"gpu-fault.test-owner": owner["owner"]},
        "ports": {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port)}]},
    }
    calls: list[object] = []

    def inspect(
        arguments: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert arguments[0:2] == ["docker", "inspect"], (
            "grant validation may only inspect the fake container"
        )
        assert arguments[-1] == owner["container"], (
            "Docker inspection must target the allocation's full container ID"
        )
        calls.append(("inspect", arguments[-1]))
        return subprocess.CompletedProcess(arguments, 0, json.dumps(actual), "")

    monkeypatch.setattr(grants, "ROOT", tmp_path)
    monkeypatch.setattr(subprocess, "run", inspect)
    monkeypatch.setattr(grants, "validate_server", lambda value: calls.append(value))
    monkeypatch.setattr(
        "psycopg.connect",
        lambda *_args, **_kwargs: pytest.fail(
            "grant unit tests must never connect to SQL"
        ),
    )
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PGPASSFILE", str(home / ".pgpass"))
    monkeypatch.setenv("GPU_FAULT_TEST_POSTGRES_URL", url)
    return owner, actual, calls, url


@pytest.mark.parametrize("port", [32871, 32881, 49152])
def test_grant_uses_the_owned_allocation_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, port: int
) -> None:
    owner, _, calls, url = allocation(tmp_path, monkeypatch, port)
    assert validate() == url, "the default grant must return its password-free URL"
    assert calls == [("inspect", owner["container"]), url], (
        "Docker ownership must be checked before validating the allocated server"
    )


def test_override_selects_two_independent_allocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_owner, _, default_calls, default_url = allocation(
        tmp_path, monkeypatch, 32871
    )
    first_directory = tmp_path / "allocation-first"
    with monkeypatch.context() as first_patch:
        first_owner, _, first_calls, first_url = allocation(
            tmp_path, first_patch, 32881, directory=first_directory
        )
        first_patch.setenv(ALLOCATION_DIR_ENV, str(first_directory))
        assert validate() == first_url, "the first process must use its own allocation"
        second_directory = tmp_path / "allocation-second"
        with first_patch.context() as second_patch:
            second_owner, _, second_calls, second_url = allocation(
                tmp_path, second_patch, 49152, directory=second_directory
            )
            second_patch.setenv(ALLOCATION_DIR_ENV, str(second_directory))
            assert validate() == second_url, (
                "the second process must use a separate container, port and pgpass home"
            )
            assert second_calls == [
                ("inspect", second_owner["container"]),
                second_url,
            ], "the second grant must inspect and validate only its own server"
        assert validate() == first_url, (
            "selecting another allocation must not cache or replace the first grant"
        )
        assert first_calls == [("inspect", first_owner["container"]), first_url] * 2, (
            "each first-allocation validation must repeat its own ownership checks"
        )
    assert default_calls == [], (
        "explicit allocations must not validate the shared default"
    )
    assert validate() == default_url, (
        "an absent override must retain the default location"
    )
    assert default_calls == [("inspect", default_owner["container"]), default_url], (
        "the default must remain independently usable after explicit allocations"
    )


def test_fake_allocation_clears_inherited_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ALLOCATION_DIR_ENV, str(tmp_path / "inherited-allocation"))
    owner, _, calls, url = allocation(tmp_path, monkeypatch, 32881)
    assert validate() == url, "fake allocations must not inherit a parent's directory"
    assert calls == [("inspect", owner["container"]), url], (
        "fake allocation setup must select only its own default grant"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "empty",
        "whitespace",
        "missing-directory",
        "not-directory",
        "missing-owner",
        "missing-url",
        "invalid-owner-json",
        "invalid-owner-encoding",
        "invalid-url-encoding",
    ],
)
def test_invalid_override_never_falls_back_to_a_valid_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    owner, _, calls, url = allocation(tmp_path, monkeypatch, 32881)
    directory = tmp_path / "invalid-allocation"
    override = str(directory)
    if defect == "empty":
        override = ""
    elif defect == "whitespace":
        override = " \t "
    elif defect == "not-directory":
        directory.write_text("not a directory")
    elif defect != "missing-directory":
        directory.mkdir(mode=0o700)
        if defect != "missing-owner":
            (directory / "cov95-postgres-owner.json").write_text(json.dumps(owner))
        if defect != "missing-url":
            (directory / "cov95-postgres-url").write_text(url)
        if defect == "invalid-owner-json":
            (directory / "cov95-postgres-owner.json").write_text("{")
        elif defect == "invalid-owner-encoding":
            (directory / "cov95-postgres-owner.json").write_bytes(b"\xff")
        elif defect == "invalid-url-encoding":
            (directory / "cov95-postgres-url").write_bytes(b"\xff")
    monkeypatch.setenv(ALLOCATION_DIR_ENV, override)
    with pytest.raises(pytest.fail.Exception, match="allocation"):
        validate()
    assert calls == [], (
        "invalid explicit allocation files must fail before Docker or server validation"
    )
    monkeypatch.delenv(ALLOCATION_DIR_ENV)
    assert validate() == url, (
        "the unused default must still be valid without an override"
    )
    assert calls == [("inspect", owner["container"]), url], (
        "the valid default must only be consulted after removing the explicit override"
    )


@pytest.mark.parametrize(
    "field", ["url-file", "owner-file", "url-env", "HOME", "PGPASSFILE"]
)
def test_override_rejects_mixed_allocation_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    default_owner, _, default_calls, default_url = allocation(
        tmp_path, monkeypatch, 32871
    )
    directory = tmp_path / "separate-allocation"
    _, _, calls, _ = allocation(tmp_path, monkeypatch, 32881, directory=directory)
    monkeypatch.setenv(ALLOCATION_DIR_ENV, str(directory))
    if field == "url-file":
        (directory / "cov95-postgres-url").write_text(default_url)
    elif field == "owner-file":
        (directory / "cov95-postgres-owner.json").write_text(json.dumps(default_owner))
    elif field == "url-env":
        monkeypatch.setenv("GPU_FAULT_TEST_POSTGRES_URL", default_url)
    elif field == "HOME":
        monkeypatch.setenv("HOME", default_owner["pgpass_home"])
    else:
        monkeypatch.setenv(
            "PGPASSFILE", str(Path(default_owner["pgpass_home"]) / ".pgpass")
        )
    with pytest.raises(pytest.fail.Exception, match="identity differs"):
        validate()
    assert calls == [], "mixed allocation identity must fail before Docker or SQL"
    assert default_calls == [], (
        "a mismatched override must not validate the shared grant"
    )


@pytest.mark.parametrize("port", [None, "", True, 0, -1, 65536, "032881", "32881.0"])
def test_invalid_allocated_port_stops_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, port: object
) -> None:
    owner, _, calls, _ = allocation(tmp_path, monkeypatch, 32881)
    owner["port"] = port
    (tmp_path / ".codex/cov95-postgres-owner.json").write_text(json.dumps(owner))
    with pytest.raises(pytest.fail.Exception, match="port is invalid"):
        validate()
    assert calls == [], "invalid allocation metadata cannot contact Docker or SQL"


@pytest.mark.parametrize("owner_name", [None, "", " "])
def test_missing_owner_cannot_match_an_unlabelled_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner_name: object
) -> None:
    owner, actual, calls, _ = allocation(tmp_path, monkeypatch, 32881)
    owner["owner"] = owner_name
    actual["labels"] = {}
    (tmp_path / ".codex/cov95-postgres-owner.json").write_text(json.dumps(owner))
    with pytest.raises(pytest.fail.Exception, match="identity differs"):
        validate()
    assert calls == [], "an unspecified owner cannot grant container authority"


@pytest.mark.parametrize(
    "defect", ["url-port", "binding", "public", "label", "id", "stopped"]
)
@pytest.mark.parametrize("use_override", [False, True])
def test_port_changes_do_not_weaken_container_and_loopback_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str, use_override: bool
) -> None:
    _, actual, calls, _ = allocation(tmp_path, monkeypatch, 32881)
    if use_override:
        monkeypatch.setenv(ALLOCATION_DIR_ENV, str(tmp_path / ".codex"))
    if defect == "url-port":
        url = "postgresql://test@127.0.0.1:32871/owned"
        (tmp_path / ".codex/cov95-postgres-url").write_text(url)
        monkeypatch.setenv("GPU_FAULT_TEST_POSTGRES_URL", url)
    elif defect == "binding":
        actual["ports"]["5432/tcp"][0]["HostPort"] = "32871"
    elif defect == "public":
        actual["ports"]["5432/tcp"][0]["HostIp"] = "0.0.0.0"
    elif defect == "label":
        actual["labels"]["gpu-fault.test-owner"] = "foreign"
    elif defect == "id":
        actual["id"] = "b" * 64
    else:
        actual["running"] = False
    with pytest.raises(pytest.fail.Exception, match="differs"):
        validate()
    assert not any(isinstance(item, str) for item in calls), (
        "an unbound container must never reach server validation"
    )


def test_grant_rejects_non_object_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, calls, _ = allocation(tmp_path, monkeypatch, 32881)
    (tmp_path / ".codex/cov95-postgres-owner.json").write_text("[]")
    with pytest.raises(pytest.fail.Exception, match="not an object"):
        validate()
    assert calls == [], "non-object metadata cannot authorize Docker or SQL access"


@pytest.mark.parametrize("defect", ["url-host", "url-password", "short-container-id"])
def test_override_retains_url_and_container_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    directory = tmp_path / "explicit-allocation"
    owner, _, calls, url = allocation(tmp_path, monkeypatch, 32881, directory=directory)
    monkeypatch.setenv(ALLOCATION_DIR_ENV, str(directory))
    if defect == "short-container-id":
        owner["container"] = owner["container"][:12]
        (directory / "cov95-postgres-owner.json").write_text(json.dumps(owner))
    else:
        if defect == "url-host":
            url = "postgresql://test@localhost:32881/owned"
        else:
            url = "postgresql://test:@127.0.0.1:32881/owned"
        (directory / "cov95-postgres-url").write_text(url)
        monkeypatch.setenv("GPU_FAULT_TEST_POSTGRES_URL", url)
    with pytest.raises(pytest.fail.Exception, match="identity differs"):
        validate()
    assert calls == [], (
        "explicit paths must not relax URL or full-container-ID validation"
    )


def test_override_still_requires_server_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "explicit-allocation"
    owner, _, calls, url = allocation(tmp_path, monkeypatch, 32881, directory=directory)
    monkeypatch.setenv(ALLOCATION_DIR_ENV, str(directory))

    def reject_server(value: str) -> None:
        calls.append(value)
        raise RuntimeError("mock server rejected")

    monkeypatch.setattr(grants, "validate_server", reject_server)
    with pytest.raises(RuntimeError, match="mock server rejected"):
        validate()
    assert calls == [("inspect", owner["container"]), url], (
        "valid file and container identity must not bypass server validation"
    )


def test_override_still_rejects_xdist_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "explicit-allocation"
    _, _, calls, _ = allocation(tmp_path, monkeypatch, 32881, directory=directory)
    monkeypatch.setenv(ALLOCATION_DIR_ENV, str(directory))
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    with pytest.raises(pytest.fail.Exception, match="explicit serial -n0"):
        validate()
    assert calls == [], "directory isolation does not authorize xdist database sharing"
