from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import deploy_host_binding as binding
from gpu_fault.admin.site import SiteConfigError


def _bind(state_dir: Path) -> Path:
    venv = state_dir / "deployer-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "gpu-fault-admin").write_text("#!/bin/sh\n", encoding="utf-8")
    (venv / binding.DEPLOY_HOST_STATE_BINDING).write_text(
        json.dumps({"schema_version": 1, "state_dir": str(state_dir)}), encoding="utf-8"
    )
    return venv / "bin" / "gpu-fault-admin"


def _arguments(command: str, state_dir: Path | None) -> argparse.Namespace:
    return argparse.Namespace(command=command, state_dir=state_dir, file=None)


def test_an_unbound_cli_refuses_to_mutate_a_site_that_has_its_own_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkout's gpu-fault-admin runs whatever the working tree holds; once a
    site has a bound deploy-host CLI, mutating verbs from the checkout are refused
    and the refusal names the CLI to run (live 2026-09-15: a join from the
    checkout verified against the checkout's own dist/ and rolled back)."""

    monkeypatch.setattr(binding, "bound_state_dir", lambda prefix=None: None)
    monkeypatch.delenv(binding.ALLOW_UNBOUND_ENV, raising=False)
    site = tmp_path / "site"
    admin = _bind(site)

    with pytest.raises(SiteConfigError, match=f"run {admin} join-cluster"):
        binding.enforce_deploy_host_state_dir(_arguments("join-cluster", site))
    with pytest.raises(SiteConfigError, match=binding.ALLOW_UNBOUND_ENV):
        binding.enforce_deploy_host_state_dir(_arguments("uninstall", site))


def test_an_unbound_cli_still_deploys_reads_and_serves_unbound_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(binding, "bound_state_dir", lambda prefix=None: None)
    monkeypatch.delenv(binding.ALLOW_UNBOUND_ENV, raising=False)
    site = tmp_path / "site"
    _bind(site)
    fresh = tmp_path / "fresh"
    fresh.mkdir()

    # deploy prepares the release and re-execs into the bound CLI itself.
    binding.enforce_deploy_host_state_dir(_arguments("deploy", site))
    # read-only verbs stay available from the checkout.
    binding.enforce_deploy_host_state_dir(
        _arguments("status", site), readonly_commands={"status"}
    )
    # a directory without a bound CLI (before its first deploy) is not guarded.
    binding.enforce_deploy_host_state_dir(_arguments("join-cluster", fresh))
    # no --state-dir at all: nothing to guard.
    binding.enforce_deploy_host_state_dir(_arguments("status", None))


def test_the_environment_override_runs_the_checkout_on_purpose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(binding, "bound_state_dir", lambda prefix=None: None)
    monkeypatch.setenv(binding.ALLOW_UNBOUND_ENV, "1")
    site = tmp_path / "site"
    _bind(site)

    binding.enforce_deploy_host_state_dir(_arguments("remove-cluster", site))


def test_a_bound_cli_keeps_refusing_other_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "canonical"
    monkeypatch.setattr(
        binding, "bound_state_dir", lambda prefix=None: canonical.resolve()
    )

    with pytest.raises(SiteConfigError, match="installed deploy-host is bound"):
        binding.enforce_deploy_host_state_dir(_arguments("status", tmp_path / "other"))
    with pytest.raises(SiteConfigError, match="requires that managed state"):
        binding.enforce_deploy_host_state_dir(_arguments("status", None))
    binding.enforce_deploy_host_state_dir(_arguments("status", canonical))


def test_a_refused_call_opens_no_log_under_the_other_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The binding is checked before command_log opens under --state-dir, so a
    refused call leaves nothing behind in the directory it was refused for."""

    canonical = tmp_path / "canonical"
    other = tmp_path / "other"
    other.mkdir()
    arguments = argparse.Namespace(command="status", state_dir=other, file=None)

    class Parser:
        @staticmethod
        def parse_args() -> argparse.Namespace:
            return arguments

    monkeypatch.setattr(admin_cli, "parser", Parser)
    monkeypatch.setattr(
        binding, "bound_state_dir", lambda prefix=None: canonical.resolve()
    )
    monkeypatch.setattr(
        admin_cli,
        "run",
        lambda _arguments: pytest.fail("dispatch ran despite the binding"),
    )

    assert admin_cli.main() == 2
    assert "installed deploy-host is bound" in capsys.readouterr().err
    assert not (other / "logs").exists(), sorted(other.iterdir())
