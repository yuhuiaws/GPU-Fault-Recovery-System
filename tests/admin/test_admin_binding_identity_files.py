from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin import deploy_host_binding as binding
from gpu_fault.admin.site import SiteConfigError
from tests.admin.test_admin_deploy_host_binding import bind_site


def write_binding(prefix: Path, raw: bytes) -> Path:
    prefix.mkdir(parents=True, exist_ok=True)
    path = prefix / binding.DEPLOY_HOST_STATE_BINDING
    path.write_bytes(raw)
    path.chmod(0o600)
    return path


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"null",
        b"[]",
        b'{"schema_version": 2}',
        b'{"schema_version": true, "state_dir": "/tmp/managed"}',
        b'{"schema_version": 1}',
        b'{"schema_version": 1, "state_dir": 1}',
        b'{"schema_version": 1, "state_dir": "  "}',
        b'{"schema_version": 1, "state_dir": "relative"}',
        b'{"schema_version": 1, "state_dir": "/tmp/a\\u0000b"}',
        b'{"schema_version": 1, "state_dir": "/tmp/a", "state_dir": "/tmp/b"}',
        b'{"schema_version": 1, "state_dir": "/tmp/a", "extra": true}',
        b"\xff",
        b" " * (16 * 1024 + 1),
    ],
    ids=[
        "truncated",
        "null",
        "list",
        "unknown-schema",
        "boolean-schema",
        "missing-state",
        "numeric-state",
        "blank-state",
        "relative-state",
        "nul-state",
        "duplicate-field",
        "unknown-field",
        "encoding",
        "oversized",
    ],
)
def test_invalid_binding_never_becomes_absence(tmp_path: Path, raw: bytes) -> None:
    write_binding(tmp_path, raw)
    with pytest.raises(SiteConfigError, match="binding"):
        binding.bound_deploy_host_state_dir(tmp_path)


@pytest.mark.parametrize(
    "kind", ["symlink", "dangling", "directory", "fifo", "nonprivate", "writeonly"]
)
def test_unsafe_binding_files_fail_closed(tmp_path: Path, kind: str) -> None:
    path = write_binding(
        tmp_path, json.dumps({"schema_version": 1, "state_dir": str(tmp_path)}).encode()
    )
    if kind in {"symlink", "dangling"}:
        target = tmp_path / "target.json"
        path.rename(target)
        path.symlink_to(target if kind == "symlink" else tmp_path / "missing")
    elif kind in {"directory", "fifo"}:
        path.unlink()
        if kind == "directory":
            path.mkdir()
        else:
            os.mkfifo(path, 0o600)
    else:
        path.chmod(0o644 if kind == "nonprivate" else 0o200)
    with pytest.raises(SiteConfigError, match="binding is invalid"):
        binding.bound_deploy_host_state_dir(tmp_path)


def test_unreadable_binding_uses_a_module_local_io_double(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_binding(tmp_path, b"{}")

    def denied(*_args: object) -> int:
        raise PermissionError("fixture denial")

    monkeypatch.setattr(
        binding,
        "os",
        SimpleNamespace(
            open=denied,
            O_RDONLY=os.O_RDONLY,
            O_NOFOLLOW=os.O_NOFOLLOW,
            O_NONBLOCK=os.O_NONBLOCK,
        ),
    )
    with pytest.raises(SiteConfigError, match="binding is invalid"):
        binding.bound_state_dir(tmp_path)


def test_binding_alias_and_home_path_resolve_but_a_dangling_venv_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    write_binding(prefix, b'{"schema_version": 1, "state_dir": "~/unused/../managed"}')
    alias = tmp_path / "alias"
    alias.symlink_to(prefix, target_is_directory=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert binding.bound_state_dir(alias) == tmp_path / "managed"
    missing = tmp_path / "missing"
    assert binding.bound_state_dir(missing) is None
    broken = tmp_path / "broken"
    broken.symlink_to(missing, target_is_directory=True)
    with pytest.raises(SiteConfigError, match="binding is invalid"):
        binding.bound_state_dir(broken)


@pytest.mark.parametrize("command", ["deploy", "status", "uninstall"])
@pytest.mark.parametrize("location", ["host", "site"])
def test_bad_identity_blocks_even_deploy_or_readonly_before_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, location: str
) -> None:
    state = tmp_path / "managed"
    admin = bind_site(state)
    prefix = admin.parent.parent if location == "site" else tmp_path / "host"
    write_binding(prefix, b"{")
    interpreter = prefix if location == "host" else tmp_path / "unbound"
    monkeypatch.setattr(binding, "sys", SimpleNamespace(prefix=str(interpreter)))
    monkeypatch.setenv("GPU_FAULT_ADMIN_ALLOW_UNBOUND", "1")
    arguments = argparse.Namespace(command=command, state_dir=state)
    monkeypatch.setattr(
        cli, "parser", lambda: SimpleNamespace(parse_args=lambda: arguments)
    )
    monkeypatch.setattr(
        cli, "run", lambda _arguments: pytest.fail("bad identity dispatched")
    )
    assert cli.main() == 2
    assert not (state / "logs").exists(), "invalid binding must fail before logging"


@pytest.mark.parametrize(
    "damage",
    [
        "missing-binding",
        "foreign-binding",
        "missing-cli",
        "nonexecutable",
        "broken-venv",
    ],
)
def test_partial_site_installation_is_not_an_unbound_site(
    tmp_path: Path, damage: str
) -> None:
    state = tmp_path / "managed"
    admin = bind_site(state)
    record = admin.parent.parent / binding.DEPLOY_HOST_STATE_BINDING
    if damage == "missing-binding":
        record.unlink()
    elif damage == "foreign-binding":
        write_binding(
            admin.parent.parent,
            json.dumps(
                {"schema_version": 1, "state_dir": str(tmp_path / "other")}
            ).encode(),
        )
    elif damage == "missing-cli":
        admin.unlink()
    elif damage == "nonexecutable":
        admin.chmod(0o600)
    else:
        venv = admin.parent.parent
        venv.rename(state / "saved-venv")
        venv.symlink_to(state / "missing", target_is_directory=True)
    with pytest.raises(SiteConfigError, match="deploy-host"):
        binding.site_bound_admin(state)


def test_site_and_venv_aliases_select_the_same_validated_cli(tmp_path: Path) -> None:
    state = tmp_path / "managed"
    admin = bind_site(state)
    venv = admin.parent.parent
    version = state / "venv-version"
    venv.rename(version)
    venv.symlink_to(version, target_is_directory=True)
    alias = tmp_path / "alias"
    alias.symlink_to(state, target_is_directory=True)
    assert binding.site_bound_admin(alias) == admin
    assert binding.site_bound_admin(tmp_path / "fresh") is None
