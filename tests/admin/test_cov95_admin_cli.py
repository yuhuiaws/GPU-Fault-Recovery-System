from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli, release_child, release_state
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigError,
    admin_config_pending_path,
    begin_admin_config_apply,
    load_desired_admin_config,
)
from gpu_fault.admin.config_file import write_admin_config_file
from gpu_fault.admin.operation_lock import (
    SITE_OPERATION_LOCK_FD_ENV,
    inherited_site_operation_lock_fd,
)
from gpu_fault.admin.site import SiteConfigError, load_site
from tests.admin.test_admin_site import site_file


@pytest.fixture
def context(tmp_path, monkeypatch):
    path = site_file(tmp_path)
    site = load_site(path)
    state = SimpleNamespace(
        site=site,
        root=tmp_path,
        events=[],
        driver_code=0,
        driver_error=None,
        rollback_error=None,
        live={
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
            "release_lifecycle": "COMMITTED",
            "admin_config": AdminConfig().as_dict(),
            "admin_config_sha256": AdminConfig().sha256(),
            "admin_config_role_sha256": AdminConfig().role_sha256(),
        },
    )

    def driver(arguments, **options):
        descriptor = int(options["env"][SITE_OPERATION_LOCK_FD_ENV])
        assert descriptor in options["pass_fds"], (
            "the configuration release child must inherit the site lock descriptor"
        )
        with monkeypatch.context() as child:
            child.setenv(SITE_OPERATION_LOCK_FD_ENV, str(descriptor))
            assert inherited_site_operation_lock_fd(state.root) == descriptor, (
                "the release child must receive the currently held exclusive site lock"
            )
        state.events.append("driver")
        if state.driver_error is not None:
            raise state.driver_error
        return subprocess.CompletedProcess(arguments, state.driver_code, "{}", "")

    def read(arguments, **_options):
        return subprocess.CompletedProcess(
            arguments,
            0,
            json.dumps({"data": {"state.json": json.dumps(state.live)}}),
            "",
        )

    def rollback(**_options):
        state.events.append("rollback")
        if state.rollback_error is not None:
            raise state.rollback_error
        return {}

    monkeypatch.setattr(release_child, "run_driver", driver)
    monkeypatch.setattr(release_state, "run_command", read)
    monkeypatch.setattr(
        cli,
        "verify_prebuilt_release",
        lambda *_args, **_options: state.events.append("signature"),
    )
    monkeypatch.setattr(
        cli,
        "request_aurora_capacity",
        lambda **_options: {"scale_up": False, "modified": True},
    )
    monkeypatch.setattr(cli, "reconcile_aurora_capacity", rollback)
    return state


def config_arguments(context, desired):
    write_admin_config_file(context.root / "admin-config.yaml", desired, overwrite=True)
    return cli.parser().parse_args(
        ["config", "--state-dir", str(context.root), "--reference", "CHG-EXAMPLE"]
    )


def prepare_pending(context, desired):
    raw = (context.site.repository_root / "dist/current-release.json").read_bytes()
    return begin_admin_config_apply(
        context.root,
        site_identity={
            key: context.site.release_config[key]
            for key in ("site_name", "aws_region", "cpu_eks_arn")
        },
        release_identity={
            "release_id": "release-a",
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "staging_only": False,
        },
        desired=desired,
        source="example",
        approver_identity="example",
        reference="CHG-EXAMPLE",
    )


@pytest.mark.parametrize(
    "state",
    [
        {"phase": "uploaded", "transaction_committed": True},
        {"phase": "complete", "transaction_committed": False},
    ],
)
def test_config_requires_settled_live_release_before_creating_apply(context, state):
    context.live.update(state)
    arguments = config_arguments(
        context, AdminConfig().patched({"workflow": {"dispatcherWorkers": 9}})
    )
    with pytest.raises(SiteConfigError, match="not complete|not committed"):
        cli.run(arguments)
    assert context.events == ["signature"]
    assert not admin_config_pending_path(context.root).exists(), (
        "unsettled live release created an apply"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"phase": "rollback-in-progress"},
        {"rollback_result": {"status": "FAILED"}},
        {"release_diff": {"kind": "FULL"}},
        {"admin_config_sha256": "0" * 64},
        {"phase": "UNKNOWN"},
    ],
)
def test_config_resume_requires_same_pending_target_and_nonrollback_phase(
    context, change
):
    desired = AdminConfig().patched({"workflow": {"dispatcherWorkers": 9}})
    prepare_pending(context, desired)
    before = admin_config_pending_path(context.root).read_bytes()
    context.live.update(
        transaction_committed=False,
        phase="cpu-staged",
        release_diff={"kind": "CONTROL_PLANE_ONLY"},
        admin_config_sha256=desired.sha256(),
    )
    context.live.update(change)
    with pytest.raises(
        SiteConfigError, match="rolling back|not the pending|cannot be resumed"
    ):
        cli.run(config_arguments(context, desired))
    assert context.events == ["signature"]
    assert admin_config_pending_path(context.root).read_bytes() == before


@pytest.mark.parametrize("exception", [False, True])
def test_config_release_failure_without_aurora_change_restores_desired(
    context, exception
):
    desired = AdminConfig().patched({"workflow": {"dispatcherWorkers": 9}})
    if exception:
        context.driver_error = RuntimeError("example driver failure")
        with pytest.raises(RuntimeError, match="driver failure"):
            cli.run(config_arguments(context, desired))
    else:
        context.driver_code = 17
        assert cli.run(config_arguments(context, desired)) == 17
    assert load_desired_admin_config(context.root) == AdminConfig()
    assert context.events == ["signature", "driver"]
    assert admin_config_pending_path(context.root).is_file(), (
        "failed release discarded resumable configuration state"
    )


def test_config_reports_aurora_rollback_failure_and_keeps_audit(context):
    desired = AdminConfig().patched({"aurora": {"minAcu": 16.0}})
    context.driver_code = 17
    context.rollback_error = RuntimeError("example rollback failure")
    with pytest.raises(AdminConfigError, match="Aurora rollback failed"):
        cli.run(config_arguments(context, desired))
    assert context.events == ["signature", "driver", "rollback"]
    assert admin_config_pending_path(context.root).is_file(), (
        "rollback failure discarded its pending audit"
    )


@pytest.mark.parametrize(
    "change",
    [
        "not-object",
        "no-id",
        "bad-id",
        "bad-tier",
        "missing-copy",
        "copy-drift",
        "invalid-json",
    ],
)
def test_config_rechecks_manifest_drift_after_locked_site_load(
    context, monkeypatch, change
):
    original = cli.reload_site_for_mutation
    manifest = context.site.repository_root / "dist/current-release.json"
    immutable = context.site.repository_root / "dist/release-a/release.json"

    def reload(site):
        result = original(site)
        value = json.loads(manifest.read_text())
        if change == "not-object":
            value = []
        elif change == "no-id":
            value["release_id"] = ""
        elif change == "bad-id":
            value["release_id"] = "../outside"
        elif change == "bad-tier":
            value["staging_only"] = "false"
        elif change == "missing-copy":
            immutable.unlink()
        elif change == "copy-drift":
            immutable.write_text("{}")
        if change == "invalid-json":
            manifest.write_text("invalid")
        else:
            manifest.write_text(json.dumps(value))
            if change not in {"missing-copy", "copy-drift"}:
                immutable.write_bytes(manifest.read_bytes())
        return result

    monkeypatch.setattr(cli, "reload_site_for_mutation", reload)
    with pytest.raises(SiteConfigError, match="manifest|release_id|boolean"):
        cli.run(config_arguments(context, AdminConfig()))
    assert context.events == []


def test_join_cli_rejects_one_cluster_id_for_multiple_arns(context):
    arguments = cli.parser().parse_args(
        [
            "join-cluster",
            "--state-dir",
            str(context.root),
            "--cluster-id",
            "shared",
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-c",
        ]
    )
    with pytest.raises(SiteConfigError, match="cannot be shared"):
        cli.run(arguments)
    assert context.events == []


@pytest.mark.parametrize("provided", [False, True])
def test_managed_cli_requires_existing_site_before_driver(context, provided):
    arguments = ["status"]
    if provided:
        arguments.extend(["--state-dir", str(context.root / "nonexistent")])
    with pytest.raises(SiteConfigError, match="requires --state-dir|no managed site"):
        cli.run(cli.parser().parse_args(arguments))
    assert context.events == []


def test_unregistered_command_cannot_start_without_site(context):
    with pytest.raises(SiteConfigError, match="requires -f"):
        cli.run(argparse.Namespace(command="unknown"))
    assert context.events == []


def test_history_and_evidence_environment_require_managed_paths(monkeypatch):
    assert (
        cli.release_history_environment(argparse.Namespace(state_dir="not-a-path"))
        == {}
    )
    monkeypatch.delenv(cli.QUICK_VALIDATION_EVIDENCE_ENV, raising=False)
    assert (
        cli.quick_validation_evidence_environment(
            argparse.Namespace(command="deploy", state_dir=None, file=None)
        )
        == {}
    )


@pytest.mark.parametrize("payload", ["prefix\n{\ninvalid", "prefix without JSON", "[]"])
def test_status_decoder_refuses_malformed_driver_output(payload):
    assert cli.status_document(payload) is None


def test_failure_domain_dispatch_converts_domain_errors(context, monkeypatch):
    calls = []

    def failed(arguments, *, site):
        calls.append((arguments, site))
        raise BootstrapError("example domain-map failure")

    monkeypatch.setattr(cli, "run_failure_domain_map_command", failed)
    arguments = cli.parser().parse_args(
        ["failure-domain-map", "--state-dir", str(context.root)]
    )
    with pytest.raises(SiteConfigError, match="domain-map failure"):
        cli.run(arguments)
    assert len(calls) == 1
    assert calls[0][1].source == context.site.source
