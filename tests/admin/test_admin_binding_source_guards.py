from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin import cli, source_deploy
from gpu_fault.admin import deploy_host_binding as binding
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import SiteConfigError
from tests.admin.test_admin_deploy_host_binding import bind_site
from tests.admin.test_admin_site import site_file


@pytest.fixture(autouse=True)
def unbound_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        binding, "sys", SimpleNamespace(prefix=str(tmp_path / "unbound"))
    )
    monkeypatch.setenv("GPU_FAULT_ADMIN_ALLOW_UNBOUND", "1")


def test_bound_cli_checks_the_canonical_site_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = site_file(tmp_path)
    admin = bind_site(tmp_path)
    monkeypatch.setattr(
        binding, "sys", SimpleNamespace(prefix=str(admin.parent.parent))
    )
    alias = tmp_path / "repo-alias"
    alias.symlink_to(tmp_path / "repo", target_is_directory=True)
    arguments = argparse.Namespace(command="uninstall", file=site, repo_root=alias)
    binding.enforce_deploy_host_state_dir(arguments)
    arguments.repo_root = tmp_path / "foreign"
    with pytest.raises(SiteConfigError, match="different site repository root"):
        binding.enforce_deploy_host_state_dir(arguments)


def test_prepared_bound_deploy_may_select_its_new_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admin = bind_site(tmp_path)
    monkeypatch.setattr(
        binding, "sys", SimpleNamespace(prefix=str(admin.parent.parent))
    )
    binding.enforce_deploy_host_state_dir(
        argparse.Namespace(
            command="deploy",
            state_dir=tmp_path,
            repo_root=tmp_path / "new-snapshot",
            prepared_source_release=True,
        )
    )


@pytest.mark.parametrize("form", ["rollback", "explicit"])
def test_prepared_flag_does_not_exempt_other_deploy_forms_from_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, form: str
) -> None:
    site = site_file(tmp_path)
    admin = bind_site(tmp_path)
    monkeypatch.setattr(
        binding, "sys", SimpleNamespace(prefix=str(admin.parent.parent))
    )
    arguments = argparse.Namespace(
        command="deploy",
        state_dir=tmp_path,
        file=site if form == "explicit" else None,
        rollback=form == "rollback",
        repo_root=tmp_path / "foreign",
        prepared_source_release=True,
    )
    with pytest.raises(SiteConfigError, match="different site repository root"):
        binding.enforce_deploy_host_state_dir(arguments)


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        (
            ["--cpu-cluster-arn", "arn:aws:eks:us-east-1:123456789012:cluster/other"],
            "CPU cluster identity differs",
        ),
        (
            ["--gpu-cluster-arn", "arn:aws:eks:us-east-1:123456789012:cluster/other"],
            "remove-cluster",
        ),
        (["--reference", "CHG-fixture"], "only used with --approve-profile-plan"),
        (["--approve-profile-plan", "a" * 64], "requires --reference"),
    ],
    ids=["cpu-mismatch", "gpu-subset", "orphan-reference", "missing-reference"],
)
def test_source_deploy_exemption_preserves_target_and_approval_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str], reason: str
) -> None:
    site_file(tmp_path)
    bind_site(tmp_path)
    monkeypatch.setattr(
        cli,
        "run_source_deploy",
        lambda **_kwargs: pytest.fail(
            "invalid target or approval reached source preparation"
        ),
    )
    arguments = cli.parser().parse_args(
        ["deploy", "--state-dir", str(tmp_path), *extra]
    )
    binding.enforce_deploy_host_state_dir(arguments)
    with pytest.raises(SiteConfigError, match=reason):
        cli.run(arguments)


def test_source_deploy_exemption_does_not_replace_an_invalid_initial_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind_site(tmp_path)
    receipt = tmp_path / "initial-deploy-request.json"
    receipt.write_text("{", encoding="utf-8")
    receipt.chmod(0o600)
    before = receipt.read_bytes()
    monkeypatch.setattr(
        cli,
        "run_source_deploy",
        lambda **_kwargs: pytest.fail("an invalid receipt reached source preparation"),
    )
    arguments = cli.parser().parse_args(["deploy", "--state-dir", str(tmp_path)])
    binding.enforce_deploy_host_state_dir(arguments)
    with pytest.raises(SiteConfigError, match="initial deploy request is invalid"):
        cli.run(arguments)
    assert receipt.read_bytes() == before


def test_source_deploy_exemption_does_not_ignore_a_corrupt_recorded_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = site_file(tmp_path)
    bind_site(tmp_path)
    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    document["spec"]["notifications"] = {"adminEmail": "operations@example.com"}
    site.write_text(json.dumps(document), encoding="utf-8")
    receipt = tmp_path / "source-deploy.json"
    receipt.write_text(
        json.dumps(
            {"schema_version": 1, "source_repository_root": str(tmp_path / "missing")}
        ),
        encoding="utf-8",
    )
    receipt.chmod(0o600)
    monkeypatch.setattr(
        source_deploy,
        "run_driver",
        lambda *_args, **_kwargs: pytest.fail(
            "corrupt source receipt started a driver"
        ),
    )
    arguments = cli.parser().parse_args(["deploy", "--state-dir", str(tmp_path)])
    binding.enforce_deploy_host_state_dir(arguments)
    with pytest.raises(BootstrapError, match="recorded source repository"):
        cli.run(arguments)
