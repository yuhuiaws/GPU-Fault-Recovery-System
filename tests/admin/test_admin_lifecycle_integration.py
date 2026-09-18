from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import cluster_join_context as context
from gpu_fault.admin import release_child
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_state import load_join_state
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.operation_lock import (
    SITE_OPERATION_LOCK,
    SITE_OPERATION_LOCK_FD_ENV,
    inherited_site_operation_lock_fd,
)
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_cluster_join_rollback import Commands, _target
from tests.admin.test_admin_site import site_file


@pytest.mark.parametrize("phase", ["ROLLED_BACK", "ROLLBACK_FAILED", "COMPLETED"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_eks_arn", "arn:aws:eks:us-east-1:123456789012:cluster/other-cpu"),
        ("namespace", "another-namespace"),
        ("cpu_kubeconfig", "/another/cpu.kubeconfig"),
        ("nlb", {"security_group": "sg-foreign"}),
        ("dns", {"hosted_zone_id": "ZFOREIGN"}),
    ],
)
def test_finished_join_never_exempts_recovery_destination_drift(
    tmp_path, phase, field, value
) -> None:
    site = load_site(site_file(tmp_path))
    request = join.JoinClusterRequest(site, _target().eks_arn)
    _, path, state = load_join_state(request)
    state.update(phase=phase, source_site_non_membership_sha256="0" * 64)
    path.write_text(json.dumps(state))
    site.release_config[field] = value
    with pytest.raises(BootstrapError, match="recovery site identity drifted"):
        load_join_state(request)
    assert json.loads(path.read_text())["phase"] == phase


def test_legacy_retry_without_original_scope_requires_reconciliation(tmp_path) -> None:
    request = join.JoinClusterRequest(load_site(site_file(tmp_path)), _target().eks_arn)
    _, path, state = load_join_state(request)
    state.update(phase="ROLLBACK_FAILED", source_site_non_membership_sha256="0" * 64)
    state.pop("source_site_recovery_sha256")
    path.write_text(json.dumps(state))
    with pytest.raises(BootstrapError, match="requires reconciliation"):
        load_join_state(request)


@pytest.mark.parametrize("returncode", [0, 7])
def test_release_child_preserves_supervision_and_the_actual_held_lock(
    tmp_path, monkeypatch, returncode
) -> None:
    captured = []

    def run(arguments, **kwargs):
        fd = int(kwargs["env"][SITE_OPERATION_LOCK_FD_ENV])
        assert fd in kwargs["pass_fds"]
        assert os.fstat(fd).st_ino == (tmp_path / SITE_OPERATION_LOCK).stat().st_ino
        with monkeypatch.context() as child_environment:
            child_environment.setenv(SITE_OPERATION_LOCK_FD_ENV, str(fd))
            assert inherited_site_operation_lock_fd(tmp_path) == fd
        assert "--allow-staging-release" in arguments
        captured.append(fd)
        return subprocess.CompletedProcess(arguments, returncode)

    monkeypatch.setattr(release_child, "run_driver", run)
    with administrator_operation_lock(tmp_path) as descriptor:
        result = release_child.run_automatic_release(
            repository_root=tmp_path,
            site_file=tmp_path / "site.yaml",
            state_dir=tmp_path,
            staging_only_release=True,
            lock_fd=descriptor,
        )
        assert result == returncode
        assert captured == [descriptor]
    with pytest.raises(OSError):
        os.fstat(captured[0])


def test_context_read_failure_does_not_authorize_recreation(tmp_path, monkeypatch):
    site = load_site(site_file(tmp_path))
    calls = []

    def run(arguments, **_kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 1, "", "unavailable")

    monkeypatch.setattr(join, "run_command", run)
    target = _target()
    with pytest.raises(BootstrapError, match="cannot inspect"):
        context.ensure_kube_context(
            site,
            tmp_path / "gpu.kubeconfig",
            {
                "context": target.context,
                "region": target.region,
                "eks_name": target.eks_name,
                "eks_arn": target.eks_arn,
                "hyperpod_arn": target.hyperpod_arn,
            },
        )
    assert len(calls) == 1
    assert "get-contexts" in calls[0]


@pytest.mark.parametrize("pods", [{"kind": "PodList", "items": [{"metadata": {}}]}, {}])
def test_stale_annotation_cleanup_needs_absent_reconciler_pods(
    tmp_path, monkeypatch, pods
) -> None:
    site = load_site(site_file(tmp_path))
    command = Commands(outputs=(("get pods", json.dumps(pods)),))
    command.node_annotations["gpu-fault.io/installer-state"] = "Retrying"
    monkeypatch.setattr(join, "run_command", command)
    arguments = (
        site,
        {
            "context": "gpu-b",
            "hyperpod_cluster_name": "hp-gpu-b",
            "expected_node_uids": {"node-b": "uid-node-b"},
        },
        ["node-b"],
    )
    if pods:
        context.clear_stale_installer_annotations(*arguments)
    else:
        with pytest.raises(BootstrapError, match="malformed"):
            context.clear_stale_installer_annotations(*arguments)
    assert not command.matching("patch"), (
        "present or unreadable Reconciler Pods must prevent stale annotation removal"
    )


def test_cpu_kubeconfig_is_never_a_gpu_context_recovery_target(tmp_path, monkeypatch):
    site = load_site(site_file(tmp_path))
    monkeypatch.setattr(
        join,
        "run_command",
        lambda *_args, **_kwargs: pytest.fail("CPU kubeconfig was touched"),
    )
    with pytest.raises(BootstrapError, match="cannot rewrite CPU"):
        context.ensure_kube_context(
            site, Path(site.release_config["cpu_kubeconfig"]), {}
        )
