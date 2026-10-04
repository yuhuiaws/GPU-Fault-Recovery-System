"""rotate-token on a mixed-provider site judges every node wave by the pin of
the cluster being rotated, not the site's.

Live 2026-10-04 (AUTH-016 on gpu-eks-cluster, journal 20261004T202443Z-dc6e0d66):
the wave waited the full 900 s on a node whose installer Job had Succeeded,
because the wait expected the site's hyperpod-v1 Agent config digest while the
Reconciler had annotated the eks-v1 node with its own profile's digest.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpu_fault.admin import rotate_token as module
from gpu_fault_release.regional_release_config import ClusterTarget

SITE_DIGEST = "d" * 64
EKS_DIGEST = "f" * 64


def _target() -> ClusterTarget:
    return ClusterTarget(
        cluster_id="gpu-a",
        context="gpu-a-context",
        executor_irsa_role_arn="arn:aws:iam::123456789012:role/executor",
        region="us-west-2",
        hyperpod_cluster_name="hp-a",
        eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/gpu-a",
        token_file="/secure/gpu-a.token",
    )


def _release() -> SimpleNamespace:
    runner = SimpleNamespace(calls=[], dry_run=False)
    runner.run = lambda args, **kwargs: runner.calls.append(list(args)) or ""
    return SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            agent_config_digest=SITE_DIGEST,
            runtime_profile_version="hyperpod-v1",
            component_digests={},
            upgrade_max_unavailable=0,
        ),
        executor_wheel_cm="wheel-cm",
        bundle_cm="bundle-cm",
        node_wheel_sha="e" * 64,
        _gpu=lambda target, *args: ["kubectl", "--context", target.context, *args],
    )


def _wave_seams(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict], list[object]]:
    waits: list[dict] = []
    contexts: list[object] = []
    monkeypatch.setattr(
        module, "target_node_names", lambda release, target: ("node-1",)
    )
    monkeypatch.setattr(
        module,
        "target_node_failure_domains",
        lambda release, target, names: {"node-1": "a"},
    )
    monkeypatch.setattr(
        module,
        "node_rollout_policy",
        lambda release, domains, *, phase: module.NodeRolloutPolicy(
            max_unavailable=1,
            first_wave_max_unavailable=1,
            max_unavailable_per_failure_domain=1,
        ),
    )
    monkeypatch.setattr(module, "validate_target_node_state", lambda *args: None)
    monkeypatch.setattr(
        module,
        "reconciler_container_env",
        lambda release, target: {module.INSTALLER_WAVE_CONFIG_MAP_ENV: "wave-cm"},
    )
    monkeypatch.setattr(
        module, "reconciler_installer_identity", lambda env: ("bundle" * 8, "tmpl" * 16)
    )
    monkeypatch.setattr(
        module,
        "read_wave_config",
        lambda release, target, name: {"allowed-nodes": "*", "generation": "g"},
    )
    monkeypatch.setattr(module, "ensure_rollout_wave_safe", lambda *a, **k: None)
    monkeypatch.setattr(
        module,
        "hand_wave_to_reconciler",
        lambda release, target, context, wave: contexts.append(context)
        or context.paused_identity,
    )
    monkeypatch.setattr(
        module,
        "wait_agents",
        lambda release, target, artifact, **kwargs: waits.append(dict(kwargs)),
    )
    monkeypatch.setattr(module, "restore_wave_config", lambda *args: None)
    return waits, contexts


def test_roll_node_tokens_waits_for_a_clusters_own_runtime_profile_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits, contexts = _wave_seams(monkeypatch)
    target = replace(
        _target(),
        runtime_profile_version="eks-v1",
        agent_config_digest=EKS_DIGEST,
        overrides_runtime_profile=True,
    )

    module.roll_node_tokens(
        _release(), target, rotation_id="r1", progress={}, record=lambda: None
    )

    assert [context.config_digest for context in contexts] == [EKS_DIGEST]
    assert [context.expected_profile for context in contexts] == ["eks-v1"]
    assert [(w["config_digest"], w["runtime_profile_version"]) for w in waits] == [
        (EKS_DIGEST, "eks-v1"),
        (EKS_DIGEST, "eks-v1"),
    ], "the wave wait and the final cluster re-check judge nodes by the cluster pin"


def test_roll_node_tokens_keeps_the_site_pin_for_a_cluster_on_the_site_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits, contexts = _wave_seams(monkeypatch)

    module.roll_node_tokens(
        _release(), _target(), rotation_id="r2", progress={}, record=lambda: None
    )

    assert [context.config_digest for context in contexts] == [SITE_DIGEST]
    assert [(w["config_digest"], w["runtime_profile_version"]) for w in waits] == [
        (SITE_DIGEST, "hyperpod-v1"),
        (SITE_DIGEST, "hyperpod-v1"),
    ], "a HyperPod cluster on the site profile is judged exactly as before"
