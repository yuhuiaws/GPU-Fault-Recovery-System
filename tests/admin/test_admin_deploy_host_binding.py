from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin import deploy_host_binding as binding
from gpu_fault.admin.command_log import ADMIN_LOG_ENVIRONMENT
from gpu_fault.admin.site import SiteConfigError


def bind_site(state_dir: Path) -> Path:
    venv = state_dir / "deployer-venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    record = venv / binding.DEPLOY_HOST_STATE_BINDING
    record.write_text(
        json.dumps({"schema_version": 1, "state_dir": str(state_dir.resolve())}),
        encoding="utf-8",
    )
    record.chmod(0o600)
    admin = venv / "bin" / "gpu-fault-admin"
    admin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    admin.chmod(0o700)
    return admin


@pytest.fixture(autouse=True)
def unbound_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        binding, "sys", SimpleNamespace(prefix=str(tmp_path / "unbound"))
    )
    monkeypatch.delenv(ADMIN_LOG_ENVIRONMENT, raising=False)


def enforce(arguments: argparse.Namespace) -> None:
    binding.enforce_deploy_host_state_dir(
        arguments, readonly_commands=cli.READONLY_COMMANDS
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["join-cluster", "--gpu-cluster-arn", "fixture"],
        [
            "remove-cluster",
            "--gpu-cluster-arn",
            "fixture",
            "--confirm",
            "REMOVE_GPU_CLUSTER",
        ],
        ["uninstall", "--confirm", "UNINSTALL_GPU_FAULT"],
        ["config"],
        ["config", "spare", "--node", "node-a", "--declare"],
        ["config", "spare", "--node", "node-a", "--release"],
        ["workflow-reconcile"],
        ["workflow-reconcile", "--close-escalated"],
        [
            "submit-remediation",
            "--incident-id",
            "incident-a",
            "--disposition",
            "inspected",
        ],
        ["failure-domain-map", "--output", "map.json"],
        ["deploy", "--rollback"],
        ["deploy", "--rollback", "--allow-inflight-installs"],
        ["deploy", "--prepared-source-release"],
    ],
    ids=[
        "join",
        "remove",
        "uninstall",
        "config",
        "spare-declare",
        "spare-release",
        "reconcile",
        "close-incident",
        "remediation",
        "map-output",
        "rollback",
        "rollback-consent",
        "prepared-deploy",
    ],
)
def test_unbound_mutations_name_the_site_cli(tmp_path: Path, argv: list[str]) -> None:
    state = tmp_path / "managed"
    admin = bind_site(state)
    arguments = cli.parser().parse_args([*argv, "--state-dir", str(state)])
    with pytest.raises(SiteConfigError, match="has its own deploy-host CLI") as failure:
        enforce(arguments)
    assert str(admin) in str(failure.value), (
        "the refusal must name the bound executable"
    )


@pytest.mark.parametrize("action", ["stats", "list", "requeue-dead"])
def test_outbox_reads_still_require_the_bound_cli(tmp_path: Path, action: str) -> None:
    state = tmp_path / "managed"
    bind_site(state)
    arguments = cli.parser().parse_args(
        [
            "collector-outbox",
            "--state-dir",
            str(state),
            "--cluster-id",
            "cluster-a",
            "--node",
            "node-a",
            "--collector",
            "kernel",
            "--action",
            action,
            "--reference",
            "CHG-fixture",
        ]
    )
    with pytest.raises(SiteConfigError, match="has its own deploy-host CLI"):
        enforce(arguments)


@pytest.mark.parametrize(
    "argv",
    [
        ["status"],
        ["preflight"],
        ["verify"],
        ["config", "--dry-run"],
        ["config", "spare", "--node", "node-a"],
        ["workflow-reconcile", "--dry-run"],
        ["workflow-reconcile", "--close-escalated", "--dry-run"],
        [
            "submit-remediation",
            "--incident-id",
            "incident-a",
            "--disposition",
            "inspected",
            "--plan",
        ],
        ["failure-domain-map"],
    ],
    ids=[
        "status",
        "preflight",
        "verify",
        "config-plan",
        "spare-check",
        "reconcile-plan",
        "close-plan",
        "remediation-plan",
        "map",
    ],
)
def test_readonly_forms_remain_available_and_rotate_as_readonly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    state = tmp_path / "managed"
    bind_site(state)
    arguments = cli.parser().parse_args([*argv, "--state-dir", str(state)])
    enforce(arguments)
    budgets: list[bool] = []

    def budget(_name: str, *, readonly: bool) -> nullcontext[None]:
        budgets.append(readonly)
        return nullcontext()

    monkeypatch.setattr(
        cli, "parser", lambda: SimpleNamespace(parse_args=lambda: arguments)
    )
    monkeypatch.setattr(cli, "operation_budget", budget)
    monkeypatch.setattr(cli, "run", lambda _arguments: 0)
    assert cli.main() == 0
    assert budgets == [True], "budget classification must match the binding policy"
    assert len(list((state / "logs" / "readonly").glob("*.log"))) == 1
    assert not (state / "logs" / "mutating").exists(), (
        "a read-only command must not create mutating logs"
    )


def test_ordinary_deploy_and_unbound_sites_remain_available(tmp_path: Path) -> None:
    state = tmp_path / "managed"
    bind_site(state)
    enforce(cli.parser().parse_args(["deploy", "--state-dir", str(state)]))
    enforce(argparse.Namespace(command="status"))
    enforce(
        argparse.Namespace(
            command="join-cluster", state_dir=tmp_path / "fresh", file=None
        )
    )


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_ambient_unbound_override_cannot_authorize_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    state = tmp_path / "managed"
    bind_site(state)
    monkeypatch.setenv("GPU_FAULT_ADMIN_ALLOW_UNBOUND", value)
    with pytest.raises(SiteConfigError, match="has its own deploy-host CLI"):
        enforce(argparse.Namespace(command="uninstall", state_dir=state))


@pytest.mark.parametrize("command", ["future-command", "collector-outbox"])
def test_generic_dry_run_does_not_exempt_unknown_or_mutating_commands(
    tmp_path: Path, command: str
) -> None:
    state = tmp_path / "managed"
    bind_site(state)
    with pytest.raises(SiteConfigError, match="has its own deploy-host CLI"):
        enforce(argparse.Namespace(command=command, state_dir=state, dry_run=True))


def test_parent_config_dry_run_does_not_exempt_a_spare_mutation(tmp_path: Path) -> None:
    state = tmp_path / "managed"
    bind_site(state)
    arguments = cli.parser().parse_args(
        [
            "config",
            "--dry-run",
            "spare",
            "--state-dir",
            str(state),
            "--node",
            "node-a",
            "--declare",
        ]
    )
    with pytest.raises(SiteConfigError, match="has its own deploy-host CLI"):
        enforce(arguments)


def test_rollback_help_names_the_retained_install_consent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.parser().parse_args(["deploy", "--help"])
    assert exited.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "takes no other deploy option except --allow-inflight-installs" in output


@pytest.mark.parametrize("command", ["join-cluster", "deploy"])
def test_explicit_file_and_its_alias_cannot_bypass_unbound_policy(
    tmp_path: Path, command: str
) -> None:
    state = tmp_path / "managed"
    bind_site(state)
    site = state / "site.yaml"
    site.touch()
    alias = tmp_path / "alias.yaml"
    alias.symlink_to(site)
    for source in (site, alias):
        with pytest.raises(SiteConfigError, match="has its own deploy-host CLI"):
            enforce(argparse.Namespace(command=command, file=source))


def test_bound_cli_accepts_aliases_but_checks_both_selectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "managed"
    admin = bind_site(state)
    site = state / "site.yaml"
    site.touch()
    alias = tmp_path / "alias"
    alias.symlink_to(state, target_is_directory=True)
    source_alias = tmp_path / "site-alias.yaml"
    source_alias.symlink_to(site)
    monkeypatch.setattr(
        binding, "sys", SimpleNamespace(prefix=str(admin.parent.parent))
    )
    enforce(argparse.Namespace(command="uninstall", state_dir=alias, file=source_alias))
    enforce(argparse.Namespace(command="uninstall", file=source_alias))
    with pytest.raises(SiteConfigError, match="installed deploy-host is bound"):
        enforce(
            argparse.Namespace(
                command="status", state_dir=alias, file=tmp_path / "other" / "site.yaml"
            )
        )
    with pytest.raises(SiteConfigError, match="requires that managed state"):
        enforce(argparse.Namespace(command="status"))
    assert binding.bound_state_dir(admin.parent.parent) == state


def test_implicit_site_symlink_cannot_redirect_a_bound_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "managed"
    admin = bind_site(state)
    (state / "site.yaml").symlink_to(tmp_path / "other" / "site.yaml")
    monkeypatch.setattr(
        binding, "sys", SimpleNamespace(prefix=str(admin.parent.parent))
    )
    with pytest.raises(SiteConfigError, match="canonical site file conflict"):
        enforce(argparse.Namespace(command="deploy", state_dir=state))


def test_a_noncanonical_file_in_a_bound_site_is_not_a_development_site(
    tmp_path: Path,
) -> None:
    state = tmp_path / "managed"
    bind_site(state)
    with pytest.raises(SiteConfigError, match="canonical site.yaml"):
        enforce(argparse.Namespace(command="status", file=state / "candidate.yaml"))


@pytest.mark.parametrize("bound_host", [False, True], ids=["unbound", "bound"])
def test_refusal_opens_no_log_and_never_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound_host: bool
) -> None:
    state = tmp_path / "managed"
    admin = bind_site(state)
    target = state
    if bound_host:
        monkeypatch.setattr(
            binding, "sys", SimpleNamespace(prefix=str(admin.parent.parent))
        )
        target = tmp_path / "other"
    arguments = argparse.Namespace(command="uninstall", state_dir=target)
    monkeypatch.setattr(
        cli, "parser", lambda: SimpleNamespace(parse_args=lambda: arguments)
    )
    monkeypatch.setattr(
        cli, "run", lambda _arguments: pytest.fail("refusal dispatched")
    )
    assert cli.main() == 2
    assert not (target / "logs").exists(), "a refused command must leave no command log"
    assert not (target / ".administrator-operation.lock").exists(), (
        "binding refusal must precede creation of the target operation lock"
    )
