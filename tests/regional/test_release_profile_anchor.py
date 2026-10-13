"""A shared cluster profile version is one policy, whichever member anchors it.

Two plain EKS clusters of a HyperPod site both run their own ``eks-v1``. The
control plane stores a version once, with the ``cluster_id`` of the member
that registered it; the release renders that anchor from the first override
cluster in ``spec.clusters`` order. The join appends a cluster, the batch join
sorts by cluster id, ``deploy`` re-sorts by EKS ARN and ``remove-cluster``
drops a member -- each moved the anchor, and the release refused its own,
unchanged policy after the CPU stage ("Runtime Profile eks-v1 already exists
with different content"). The policy comparison now leaves the anchor out;
stored profiles are read as they are, never rewritten. A ``join-cluster``
release registers a version the joining cluster runs under that cluster, the
one member it has just put into the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import RuntimeProfile
from gpu_fault_release import rollout as MODULE
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_profile_model import runtime_profiles
from gpu_fault_release.regional_runtime_profile import (
    ensure_runtime_profile,
    verify_runtime_profile,
)
from tests.regional._release_orchestrator_support import (
    config_file,
    ingress_pod_list_json,
)
from tests.regional.test_regional_release_eks_managed import (
    RecordingRunner,
    _eks_cluster_with_profile,
    _hyperpod_cluster,
)

ACCOUNT = "123456789012"
REGION = "us-east-1"


class StoreRunner(RecordingRunner):
    """The ingress probes against one in-memory Store shared across releases."""

    def __init__(self) -> None:
        super().__init__()
        self.dry_run = False
        self.store: dict[str, dict[str, Any]] = {}
        self.posted: list[dict[str, Any]] = []

    def run(self, args: list[str], **kwargs: Any) -> str:
        from gpu_fault_release.regional_release_probes import probe_source

        super().run(args, **kwargs)
        if "get" in args and "pod" in args:
            return ingress_pod_list_json()
        if "exec" not in args:
            return ""
        payload = json.loads(kwargs.get("input_text") or "{}")
        compiled = compile_runtime_profile(
            RuntimeProfile.model_validate(payload)
        ).model_dump(mode="json")
        if args[-1] == probe_source("runtime_profile_inspect"):
            existing = self.store.get(compiled["profile_version"])
            return json.dumps({"desired": compiled, "existing": existing})
        assert args[-1] == probe_source("runtime_profile_register"), args[-1][:40]
        self.posted.append(payload)
        self.store[compiled["profile_version"]] = compiled
        return json.dumps(compiled)


def _eks(tmp_path: Path, cluster_id: str) -> dict[str, Any]:
    value = _eks_cluster_with_profile(tmp_path)
    value.update(
        {
            "cluster_id": cluster_id,
            "context": f"{cluster_id}-context",
            "eks_cluster_arn": f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{cluster_id}",
        }
    )
    return value


def _release(
    tmp_path: Path, runner: StoreRunner, order: list[str]
) -> MODULE.RegionalRelease:
    root = tmp_path / "-".join(order)
    root.mkdir()
    clusters = {
        "gpu-a": _hyperpod_cluster(root),
        "eks-a": _eks(root, "eks-a"),
        "eks-b": _eks(root, "eks-b"),
    }
    config = MODULE.ReleaseConfig.load(
        config_file(root, clusters=[clusters[item] for item in order])
    )
    return MODULE.RegionalRelease(config, runner)


def test_a_shared_version_stays_equal_when_deploy_reorders_or_removes_its_anchor(
    tmp_path: Path,
) -> None:
    runner = StoreRunner()
    joined = _release(tmp_path, runner, ["gpu-a", "eks-b", "eks-a"])
    ensure_runtime_profile(joined)
    assert [
        (item["profile_version"], item["cluster_id"]) for item in runner.posted
    ] == [("hyperpod-v1", "gpu-a"), ("eks-v1", "eks-b")], (
        "join order: eks-v1 is first registered under the member listed first"
    )
    stored = json.loads(json.dumps(runner.store))

    for order in (["eks-a", "eks-b", "gpu-a"], ["gpu-a", "eks-a"]):
        release = _release(tmp_path, runner, order)
        anchors = {
            spec.version: spec.registration_cluster_id
            for spec in runtime_profiles(release.config)
        }
        assert anchors["eks-v1"] == "eks-a", "the rendered anchor moved with the order"
        ensure_runtime_profile(release)
        verify_runtime_profile(release)

    assert len(runner.posted) == 2, "an unchanged policy is never registered again"
    assert runner.store == stored, "the stored profiles keep their recorded anchor"


def test_a_policy_change_under_a_moved_anchor_is_still_refused(tmp_path: Path) -> None:
    runner = StoreRunner()
    ensure_runtime_profile(_release(tmp_path, runner, ["gpu-a", "eks-b", "eks-a"]))
    runner.store["eks-v1"]["capabilities"][0]["mode"] = "OBSERVE"
    release = _release(tmp_path, runner, ["eks-a", "eks-b", "gpu-a"])

    with pytest.raises(ReleaseError, match="eks-v1 already exists with different"):
        ensure_runtime_profile(release)
    with pytest.raises(ReleaseError, match="eks-v1 differs from the declared policy"):
        verify_runtime_profile(release)
    assert len(runner.posted) == 2, "a refused version is not overwritten"


def test_a_join_registers_a_new_shared_version_under_the_joining_cluster(
    tmp_path: Path,
) -> None:
    """A batch joins eks-a and eks-b; eks-b's join release may run first, while
    eks-a is not in the registry yet, so eks-v1 must not be anchored on it."""

    runner = StoreRunner()
    release = _release(tmp_path, runner, ["gpu-a", "eks-a", "eks-b"])

    ensure_runtime_profile(release, anchor="eks-b")
    assert [
        (item["profile_version"], item["cluster_id"]) for item in runner.posted
    ] == [("hyperpod-v1", "gpu-a"), ("eks-v1", "eks-b")], (
        "the version the joining cluster runs is registered under that cluster"
    )

    ensure_runtime_profile(release, anchor="eks-a")
    ensure_runtime_profile(release)
    assert len(runner.posted) == 2, "the sibling's join finds the version registered"


def test_a_joining_cluster_on_the_site_profile_moves_no_anchor(tmp_path: Path) -> None:
    release = _release(tmp_path, StoreRunner(), ["gpu-a", "eks-a", "eks-b"])

    assert [
        (spec.version, spec.registration_cluster_id)
        for spec in runtime_profiles(release.config, anchor="gpu-a")
    ] == [("hyperpod-v1", "gpu-a"), ("eks-v1", "eks-a")], (
        "the site profile keeps its configured anchor; an override version the "
        "joining cluster does not run keeps the first member running it"
    )
    assert [
        (spec.version, spec.registration_cluster_id)
        for spec in runtime_profiles(release.config, anchor="eks-b")
    ] == [("hyperpod-v1", "gpu-a"), ("eks-v1", "eks-b")]
