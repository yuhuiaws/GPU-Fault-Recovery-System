from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import deploy_host_binding as binding
from gpu_fault.admin.site import SiteConfigError


def test_unmanaged_prefix_has_no_state_binding(tmp_path: Path) -> None:
    assert binding.bound_deploy_host_state_dir(tmp_path) is None, (
        "an unbound development environment must remain usable"
    )


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("{", "invalid"),
        ("null", "schema"),
        ("[]", "schema"),
        ('{"schema_version": 2}', "schema"),
        ('{"schema_version": 1}', "no state directory"),
        ('{"schema_version": 1, "state_dir": 1}', "no state directory"),
        ('{"schema_version": 1, "state_dir": "  "}', "no state directory"),
    ],
)
def test_invalid_binding_does_not_become_an_unmanaged_host(
    tmp_path: Path, text: str, reason: str
) -> None:
    path = tmp_path / binding.DEPLOY_HOST_STATE_BINDING
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(SiteConfigError, match=reason):
        binding.bound_deploy_host_state_dir(tmp_path)


def test_default_prefix_and_home_path_resolve_to_canonical_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "venv"
    prefix.mkdir()
    monkeypatch.setattr(binding, "sys", SimpleNamespace(prefix=str(prefix)))
    monkeypatch.setenv("HOME", str(tmp_path))
    path = prefix / binding.DEPLOY_HOST_STATE_BINDING
    path.write_text(
        json.dumps({"schema_version": 1, "state_dir": "~/unused/../managed"}),
        encoding="utf-8",
    )
    path.chmod(0o600)
    assert binding.bound_deploy_host_state_dir() == tmp_path / "managed", (
        "the binding must resolve its managed path, not preserve relative spellings"
    )


def test_unreadable_binding_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / binding.DEPLOY_HOST_STATE_BINDING
    path.touch()
    path.chmod(0o600)

    def unreadable(*_args: object) -> int:
        raise PermissionError("unit denied read")

    monkeypatch.setattr(
        binding,
        "os",
        SimpleNamespace(
            open=unreadable,
            O_RDONLY=os.O_RDONLY,
            O_NOFOLLOW=os.O_NOFOLLOW,
            O_NONBLOCK=os.O_NONBLOCK,
        ),
    )
    with pytest.raises(SiteConfigError, match="binding is invalid"):
        binding.bound_deploy_host_state_dir(tmp_path)


def test_unmanaged_commands_do_not_require_a_managed_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(binding, "bound_deploy_host_state_dir", lambda: None)
    assert (
        binding.enforce_deploy_host_state_dir(argparse.Namespace(command="status"))
        is None
    ), "unmanaged commands must not require an installation binding"


@pytest.mark.parametrize("selector", ["state_dir", "file"])
def test_managed_commands_accept_only_the_canonical_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: str
) -> None:
    managed = tmp_path / "managed"
    monkeypatch.setattr(binding, "bound_deploy_host_state_dir", lambda: managed)
    arguments = argparse.Namespace(
        command="status",
        state_dir=managed / "unused" / ".." if selector == "state_dir" else None,
        file=managed / "site.yaml" if selector == "file" else None,
    )
    assert binding.enforce_deploy_host_state_dir(arguments) is None, (
        "canonical state-dir and site-file targets must both be accepted"
    )


@pytest.mark.parametrize("selector", ["missing", "state_dir", "file", "conflict"])
def test_managed_commands_refuse_missing_or_conflicting_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: str
) -> None:
    managed = tmp_path / "managed"
    monkeypatch.setattr(binding, "bound_deploy_host_state_dir", lambda: managed)
    arguments = argparse.Namespace(
        command="uninstall",
        state_dir=tmp_path / "other" if selector in {"state_dir", "conflict"} else None,
        file=managed / "site.yaml"
        if selector == "conflict"
        else tmp_path / "other.yaml"
        if selector == "file"
        else None,
    )
    with pytest.raises(SiteConfigError, match="bound"):
        binding.enforce_deploy_host_state_dir(arguments)
