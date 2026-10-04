"""Refusals of the join's kubeconfig recovery and stale installer cleanup.

``cluster_join_context`` recovers a GPU kubeconfig context for a rollback and
clears installer annotations a departed cluster left behind. Each guard here is
a refusal that keeps a rollback from rewriting the wrong kubeconfig, adopting a
context it cannot observe, or patching nodes whose identity it cannot prove.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import cluster_join as admin_cluster_join
from gpu_fault.admin import cluster_join_context as join_context
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join import JoinTargetIdentityError
from tests.admin.test_admin_cluster_join_rollback import Attempt, Commands

CONTEXT = "gpu-fault-gpu-2-gpu-b"


def _target(**overrides: Any) -> dict[str, Any]:
    target: dict[str, Any] = {
        "context": CONTEXT,
        "region": "us-east-1",
        "eks_name": "gpu-b",
        "eks_arn": "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        "hyperpod_arn": "arn:aws:sagemaker:us-east-1:123456789012:cluster/hp-gpu-b",
        "hyperpod_cluster_name": "hp-gpu-b",
        "expected_node_uids": {"node-b": "uid-node-b"},
    }
    target.update(overrides)
    return target


def _kubeconfig(tmp_path: Path) -> Path:
    path = tmp_path / "gpu.kubeconfig"
    path.write_text("apiVersion: v1\n", encoding="utf-8")
    path.chmod(0o644)
    return path


def test_context_recovery_refuses_the_cpu_kubeconfig(tmp_path: Path) -> None:
    attempt = Attempt(tmp_path)
    cpu = Path(attempt.site.release_config["cpu_kubeconfig"])
    with pytest.raises(JoinTargetIdentityError, match="CPU kubeconfig"):
        join_context.ensure_kube_context(attempt.site, cpu, _target())
    assert attempt.commands.calls == [], "no command may run against the CPU file"


@pytest.mark.parametrize("field", ["context", "region", "eks_name"])
def test_context_recovery_requires_a_complete_target_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    attempt = Attempt(tmp_path)
    monkeypatch.setattr(admin_cluster_join, "run_command", attempt.commands)
    target = _target(**{field: ""})
    with pytest.raises(JoinTargetIdentityError, match="lacks target identity"):
        join_context.ensure_kube_context(attempt.site, _kubeconfig(tmp_path), target)
    assert attempt.commands.calls == [], "an unbound target must not be looked up"


def test_context_recovery_fails_closed_when_update_kubeconfig_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    commands = Commands(failures=[("update-kubeconfig", 1, "AccessDenied")])
    commands.contexts.clear()
    monkeypatch.setattr(admin_cluster_join, "run_command", commands)
    with pytest.raises(BootstrapError, match="cannot restore the recorded"):
        join_context.ensure_kube_context(attempt.site, _kubeconfig(tmp_path), _target())
    assert len(commands.matching("get-contexts")) == 1, (
        "a failed update must not be re-inspected as if it had succeeded"
    )


def test_context_recovery_restores_the_context_and_tightens_the_file_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    commands = Commands()
    commands.contexts.clear()
    monkeypatch.setattr(admin_cluster_join, "run_command", commands)
    kubeconfig = _kubeconfig(tmp_path)
    join_context.ensure_kube_context(attempt.site, kubeconfig, _target())
    assert commands.contexts == {CONTEXT}
    assert kubeconfig.stat().st_mode & 0o777 == 0o600, (
        "a kubeconfig rewritten by aws must end up private"
    )
    assert len(commands.matching("get-contexts")) == 2


def test_context_recovery_requires_the_restored_context_to_be_observable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    commands = Commands(outputs=[("get-contexts", "some-other-context")])
    monkeypatch.setattr(admin_cluster_join, "run_command", commands)
    with pytest.raises(BootstrapError, match="not observable"):
        join_context.ensure_kube_context(attempt.site, _kubeconfig(tmp_path), _target())
    assert commands.matching("update-kubeconfig"), (
        "the context must have been written before its absence is reported"
    )


def test_current_context_restore_fails_closed_when_the_view_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    commands = Commands(failures=[("view", 1, "broken kubeconfig")])
    monkeypatch.setattr(admin_cluster_join, "run_command", commands)
    with pytest.raises(BootstrapError, match="cannot read GPU current-context"):
        join_context.restore_current_context(
            attempt.site, _kubeconfig(tmp_path), deleted=CONTEXT, remaining=[]
        )
    assert not commands.matching("use-context") and not commands.matching("unset")


def test_current_context_restore_reports_a_failed_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    commands = Commands(
        outputs=[("view", CONTEXT)], failures=[("unset current-context", 1, "denied")]
    )
    monkeypatch.setattr(admin_cluster_join, "run_command", commands)
    with pytest.raises(BootstrapError, match="cannot restore GPU current-context"):
        join_context.restore_current_context(
            attempt.site, _kubeconfig(tmp_path), deleted=CONTEXT, remaining=[CONTEXT]
        )
    assert commands.matching("unset current-context"), (
        "the deleted context must be unset when no managed context remains"
    )


def test_stale_installer_cleanup_is_a_no_op_without_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    monkeypatch.setattr(admin_cluster_join, "run_command", attempt.commands)
    join_context.clear_stale_installer_annotations(attempt.site, _target(), [])
    assert attempt.commands.calls == []


@pytest.mark.parametrize(
    "node_uids", [None, {"node-c": "uid-node-c"}, {"node-b": "u", "node-c": "v"}]
)
def test_stale_installer_cleanup_requires_matching_node_uids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_uids: Any
) -> None:
    attempt = Attempt(tmp_path)
    monkeypatch.setattr(admin_cluster_join, "run_command", attempt.commands)
    with pytest.raises(JoinTargetIdentityError, match="lacks Node UIDs"):
        join_context.clear_stale_installer_annotations(
            attempt.site, _target(expected_node_uids=node_uids), ["node-b"]
        )
    assert attempt.commands.calls == []


@pytest.mark.parametrize(
    "document",
    [
        {"kind": "Deployment", "metadata": {"name": "other", "uid": "u"}},
        {"kind": "Deployment", "metadata": {"name": "x", "namespace": "n"}},
        {"kind": "ReplicaSet", "metadata": {}},
        [],
    ],
)
def test_stale_installer_cleanup_rejects_a_malformed_reconciler_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: Any
) -> None:
    attempt = Attempt(tmp_path)
    commands = Commands(
        outputs=[
            ("get deployment gpu-fault-node-installer-reconciler", json.dumps(document))
        ]
    )
    monkeypatch.setattr(admin_cluster_join, "run_command", commands)
    with pytest.raises(BootstrapError, match="identity is malformed"):
        join_context.clear_stale_installer_annotations(
            attempt.site, _target(), ["node-b"]
        )
    assert not commands.matching("patch"), "an unproven owner blocks every patch"


def test_stale_installer_cleanup_fails_closed_when_pods_cannot_be_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    commands = Commands(failures=[("get pods", 1, "Forbidden")])
    monkeypatch.setattr(admin_cluster_join, "run_command", commands)
    with pytest.raises(BootstrapError, match="Pods are absent"):
        join_context.clear_stale_installer_annotations(
            attempt.site, _target(), ["node-b"]
        )
    assert not commands.matching("patch"), "unlisted Pods must block every patch"


@pytest.mark.parametrize(
    "stdout", ['{"kind":"Secret","items":[]}', '{"kind":"PodList","items":{}}', "[]"]
)
def test_stale_installer_cleanup_rejects_a_malformed_pod_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    attempt = Attempt(tmp_path)
    commands = Commands(outputs=[("get pods", stdout)])
    monkeypatch.setattr(admin_cluster_join, "run_command", commands)
    with pytest.raises(BootstrapError, match="Pod inventory is malformed"):
        join_context.clear_stale_installer_annotations(
            attempt.site, _target(), ["node-b"]
        )
    assert not commands.matching("patch"), "an unreadable inventory blocks patches"


def test_stale_installer_cleanup_keeps_annotations_while_pods_terminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pods = {"kind": "PodList", "items": [{"metadata": {"name": "reconciler-x"}}]}
    attempt = Attempt(tmp_path)
    commands = Commands(outputs=[("get pods", json.dumps(pods))])
    monkeypatch.setattr(admin_cluster_join, "run_command", commands)
    join_context.clear_stale_installer_annotations(attempt.site, _target(), ["node-b"])
    assert not commands.matching("patch"), (
        "a terminating Reconciler Pod still owns the annotations"
    )
