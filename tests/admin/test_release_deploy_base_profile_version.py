"""The site base's version name follows the registration cluster's family.

Live 2026-10-08 (eks-first site: registration cluster gpu-eks-cluster, a
HyperPod member on the hyperpod-v1 override): a mistaken upgrade had re-based
the site to a HyperPod-family derived version, and the deploy that planned the
way back offered the eks template's content under ``regional-hyperpod-<digest12>``
-- the planner hardcoded the HyperPod prefix from before plain-EKS support and
never returned to a declared name. The base now follows the override model
(``cluster_join_profile.provider_profile_version``): the declared version while
its snapshot is absent or carries this very policy, ``regional-<provider>-<digest12>``
otherwise, and a template of the other family is refused rather than named.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import profile_approval as admin_profile_approval
from scripts import release_deploy
from tests.admin.test_release_deploy import (
    _release_diff,
    _release_summary,
    _site,
    _stability_report,
    _verification_report,
)

ROOT = Path(__file__).resolve().parents[2]
EKS_TEMPLATE = "config/runtime-profile.regional-eks-safe.example.yaml"
HYPERPOD_TEMPLATE = "config/runtime-profile.regional-hyperpod-safe.example.yaml"
EKS_OWNER = "gpu-fault-eks-adapter/regional-cluster-executor"
HYPERPOD_OWNER = "gpu-fault-hyperpod-adapter/regional-cluster-executor"
#: The live site's derived HyperPod-family base, as the mistaken re-base left it.
LIVE_DERIVED_BASE = "regional-hyperpod-08d0bbcfa39d"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _document(site: Path) -> dict[str, Any]:
    return yaml.safe_load(site.read_text(encoding="utf-8"))


def _write(site: Path, document: dict[str, Any]) -> None:
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _write_profile(path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _mixed_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, registration: str
) -> Path:
    """A managed mixed site at its first-release shape: the registration
    family's declared base (its snapshot written by that release, as live), the
    other provider's member pinned to its own declared override."""

    site = _site(tmp_path, monkeypatch)
    repo = tmp_path / "repo"
    for name in (EKS_TEMPLATE, HYPERPOD_TEMPLATE):
        (repo / name).write_bytes((ROOT / name).read_bytes())
    document = _document(site)
    spec = document["spec"]
    hyperpod_member = dict(spec["clusters"][0])
    eks_member = {
        key: value
        for key, value in hyperpod_member.items()
        if key != "hyperpodClusterName"
    }
    eks_member.update(
        {
            "clusterId": "gpu-eks-cluster",
            "context": "gpu-eks-cluster",
            "provider": "eks-managed",
            "nodeSelector": "eks.amazonaws.com/nodegroup=gpu",
            "eksClusterArn": "arn:aws:eks:us-east-1:123456789012:cluster/gpu-eks-cluster",
        }
    )
    if registration == "gpu-eks-cluster":
        base_template, base_version = EKS_TEMPLATE, "eks-v1"
        override_template, override_version = HYPERPOD_TEMPLATE, "hyperpod-v1"
        anchor, member = eks_member, hyperpod_member
    else:
        base_template, base_version = HYPERPOD_TEMPLATE, "hyperpod-v1"
        override_template, override_version = EKS_TEMPLATE, "eks-v1"
        anchor, member = hyperpod_member, eks_member
    # The join writes a member's override snapshot as the raw template under
    # its version.
    override_document = yaml.safe_load(
        (repo / override_template).read_text(encoding="utf-8")
    )
    override_document["cluster_id"] = member["clusterId"]
    override_document["profile_version"] = override_version
    override_snapshot = _write_profile(
        site.parent / "profiles" / f"{override_version}.yaml", override_document
    )
    member["runtimeProfile"] = {
        "source": str(override_snapshot),
        "templateSource": override_template,
        "version": override_version,
    }
    spec["clusters"] = [anchor, member]
    spec["runtimeProfile"] = {
        "source": base_template,
        "templateSource": base_template,
        "version": base_version,
        "registrationClusterId": registration,
    }
    _write(site, document)
    # The first release writes the declared snapshot and moves the source to it.
    plan = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=_sha256(repo / base_template)
    )
    assert plan.change_kind == "UNCHANGED", plan
    release_deploy.prepare_site_release(site, profile_plan=plan)
    assert (site.parent / "profiles" / f"{base_version}.yaml").is_file(), (
        "the first release writes the declared base snapshot"
    )
    return site


def _derive_live_base(site: Path, *, template: str, version: str) -> str:
    """Put the site on a derived base of ``template``'s family, as the
    mistaken re-base left the live site; ``templateSource`` stays the
    registration family's template, as every later deploy renders it.
    Returns the live profile SHA."""

    document = _document(site)
    repo = Path(document["spec"]["repositoryRoot"])
    profile = document["spec"]["runtimeProfile"]
    base = yaml.safe_load((repo / template).read_text(encoding="utf-8"))
    base["cluster_id"] = profile["registrationClusterId"]
    base["profile_version"] = version
    snapshot = _write_profile(site.parent / "profiles" / f"{version}.yaml", base)
    profile["source"] = str(snapshot)
    profile["version"] = version
    _write(site, document)
    return _sha256(snapshot)


def _without_claim(path: Path, capability: str) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["claims"] = [
        claim for claim in document["claims"] if claim.get("capability") != capability
    ]
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _run_json(arguments: list[str], **_kwargs: Any) -> dict[str, Any]:
    if "verify" in arguments:
        return _verification_report()
    if "stability" in arguments:
        return _stability_report()
    if "release-diff" in arguments:
        return _release_diff("FULL", ["runtime_profile"])
    return _release_summary()


def test_eks_registration_site_returns_from_a_hyperpod_base_to_eks_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live state: eks registration cluster, HyperPod member on
    hyperpod-v1, a HyperPod-family derived live base. The way back is the
    declared eks-v1, its immutable snapshot reused byte for byte."""

    site = _mixed_site(tmp_path, monkeypatch, registration="gpu-eks-cluster")
    declared = site.parent / "profiles" / "eks-v1.yaml"
    before = declared.read_bytes()
    live_sha = _derive_live_base(
        site, template=HYPERPOD_TEMPLATE, version=LIVE_DERIVED_BASE
    )

    plan = release_deploy.plan_runtime_profile(site, live_profile_sha256=live_sha)

    assert (plan.current_version, plan.desired_version) == (
        LIVE_DERIVED_BASE,
        "eks-v1",
    ), "eks content under a HyperPod-derived name was the live defect"
    assert plan.snapshot_file == declared, "the return reuses the declared snapshot"
    assert plan.change_kind == "OWNER_CHANGE", plan
    assert plan.approval_required is True, "a return of the base is still reviewed"
    assert "environment: hyperpod-eks->eks" in plan.changes, plan.changes
    assert f"nodeReboot: owner/adapter {HYPERPOD_OWNER}->{EKS_OWNER}" in plan.changes

    with pytest.raises(release_deploy.ReleaseDeployError, match="approve-profile-plan"):
        release_deploy.execute_release(
            site, run_checks=False, live_state={"runtime_profile_sha256": live_sha}
        )
    pending = json.loads(
        admin_profile_approval.profile_plan_path(site.parent).read_text(
            encoding="utf-8"
        )
    )
    assert (pending["current_version"], pending["desired_version"]) == (
        LIVE_DERIVED_BASE,
        "eks-v1",
    )
    admin_profile_approval.approve_profile(
        site.parent,
        reference="CHG-36",
        expected_plan_sha256=str(pending["plan_sha256"]),
    )
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(release_deploy, "_run_json", _run_json)

    prepared = release_deploy.execute_release(
        site, run_checks=False, live_state={"runtime_profile_sha256": live_sha}
    )

    profile = _document(site)["spec"]["runtimeProfile"]
    assert (profile["version"], profile["source"]) == ("eks-v1", str(declared))
    assert profile["templateSource"] == EKS_TEMPLATE
    assert declared.read_bytes() == before, "the immutable snapshot is untouched"
    assert prepared.runtime_profile_version == "eks-v1"
    again = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=_sha256(declared)
    )
    assert (again.change_kind, again.desired_version) == ("UNCHANGED", "eks-v1"), (
        "the next deploy has nothing left to plan"
    )


def test_hyperpod_registration_site_returns_from_an_eks_base_to_hyperpod_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The symmetric shape: HyperPod registration cluster, eks member on
    eks-v1, an eks-family derived live base."""

    site = _mixed_site(tmp_path, monkeypatch, registration="gpu-a")
    live_sha = _derive_live_base(
        site, template=EKS_TEMPLATE, version="regional-eks-316d7d12910f"
    )

    plan = release_deploy.plan_runtime_profile(site, live_profile_sha256=live_sha)

    assert plan.desired_version == "hyperpod-v1", plan
    assert plan.snapshot_file == site.parent / "profiles" / "hyperpod-v1.yaml"
    # The HyperPod template adds capabilities (gpuDetection AUGMENT,
    # providerHealthIsolation OBSERVE); a widening outranks the owner change.
    assert plan.change_kind == "EXPANSIVE", plan
    assert "environment: eks->hyperpod-eks" in plan.changes, plan.changes
    assert f"nodeReboot: owner/adapter {EKS_OWNER}->{HYPERPOD_OWNER}" in plan.changes


def test_the_declared_name_is_used_only_while_its_snapshot_carries_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot never names two policies: once the example template moved on
    under eks-v1, the successor is the family's digest-named version and the
    eks-v1 snapshot stays as it is. A declared snapshot that is absent takes
    the declared name again."""

    site = _mixed_site(tmp_path, monkeypatch, registration="gpu-eks-cluster")
    declared = site.parent / "profiles" / "eks-v1.yaml"
    before = declared.read_bytes()
    repo = tmp_path / "repo"
    _without_claim(repo / EKS_TEMPLATE, "nvlinkDiagnostics")

    moved_on = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=_sha256(declared)
    )

    assert moved_on.desired_version == f"regional-eks-{moved_on.policy_digest[:12]}"
    assert moved_on.snapshot_file.name == f"{moved_on.desired_version}.yaml"
    assert moved_on.change_kind == "RESTRICTIVE", moved_on
    assert not moved_on.desired_version.startswith("regional-hyperpod-"), (
        "an eks registration site never mints a HyperPod-named base"
    )
    assert declared.read_bytes() == before, "eks-v1 keeps its snapshot untouched"

    (repo / EKS_TEMPLATE).write_bytes((ROOT / EKS_TEMPLATE).read_bytes())
    live_sha = _derive_live_base(
        site, template=HYPERPOD_TEMPLATE, version=LIVE_DERIVED_BASE
    )
    declared.unlink()

    absent = release_deploy.plan_runtime_profile(site, live_profile_sha256=live_sha)

    assert absent.desired_version == "eks-v1", absent
    assert not absent.snapshot_file.exists(), "the declared snapshot is to be written"


def test_a_custom_template_of_the_family_gets_the_digest_named_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """eks-v1 means the example template's policy; a site template with
    another eks-family policy is named by its digest, even when the eks-v1
    snapshot is present."""

    site = _mixed_site(tmp_path, monkeypatch, registration="gpu-eks-cluster")
    repo = tmp_path / "repo"
    custom = repo / "config/custom-eks.yaml"
    custom.write_bytes((repo / EKS_TEMPLATE).read_bytes())
    _without_claim(custom, "nvlinkDiagnostics")
    live_sha = _derive_live_base(
        site, template=HYPERPOD_TEMPLATE, version=LIVE_DERIVED_BASE
    )
    document = _document(site)
    document["spec"]["runtimeProfile"]["templateSource"] = "config/custom-eks.yaml"
    _write(site, document)

    plan = release_deploy.plan_runtime_profile(site, live_profile_sha256=live_sha)

    assert plan.desired_version == f"regional-eks-{plan.policy_digest[:12]}", plan
    assert plan.change_kind == "OWNER_CHANGE", plan


def test_a_template_of_the_other_family_is_refused_not_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The re-base the live site was once talked into: an eks registration
    cluster under a HyperPod-owned template. It is refused before any plan."""

    site = _mixed_site(tmp_path, monkeypatch, registration="gpu-eks-cluster")
    declared = site.parent / "profiles" / "eks-v1.yaml"
    document = _document(site)
    document["spec"]["runtimeProfile"]["templateSource"] = HYPERPOD_TEMPLATE
    _write(site, document)

    with pytest.raises(
        release_deploy.ReleaseDeployError,
        match="follows the registration cluster.*regional-eks-safe",
    ):
        release_deploy.plan_runtime_profile(site, live_profile_sha256=_sha256(declared))
    assert not admin_profile_approval.profile_plan_path(site.parent).exists(), (
        "a refused template leaves no plan to approve"
    )


def test_the_family_comes_from_the_registration_cluster_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registration cluster missing from spec.clusters cannot name the base,
    and a site without GPU cluster entries has none to register a new version
    under: it was planned as HyperPod family and then refused by the control
    plane (the anchor names a cluster it removed), so it is refused here until
    a cluster joins and takes the anchor over."""

    site = _site(tmp_path, monkeypatch)
    profile = tmp_path / "repo/config/profile.yaml"
    initial = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=_sha256(profile)
    )
    release_deploy.prepare_site_release(site, profile_plan=initial)
    live_sha = _sha256(Path(_document(site)["spec"]["runtimeProfile"]["source"]))
    value = yaml.safe_load(profile.read_text(encoding="utf-8"))
    value["claims"] = [
        {"capability": "gpuReset", "mode": "OBSERVE", "owner": "gpu-fault-node-agent"}
    ]
    profile.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    document = _document(site)
    document["spec"]["runtimeProfile"]["registrationClusterId"] = "ghost"
    _write(site, document)

    with pytest.raises(release_deploy.ReleaseDeployError, match="ghost is not among"):
        release_deploy.plan_runtime_profile(site, live_profile_sha256=live_sha)

    document["spec"]["runtimeProfile"]["registrationClusterId"] = "gpu-a"
    document["spec"]["clusters"] = []
    _write(site, document)

    with pytest.raises(release_deploy.ReleaseDeployError, match="has no GPU cluster"):
        release_deploy.plan_runtime_profile(site, live_profile_sha256=live_sha)

    value["claims"] = []
    profile.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    unchanged = release_deploy.plan_runtime_profile(site, live_profile_sha256=live_sha)
    assert unchanged.change_kind == "UNCHANGED", (
        "an emptied site whose profile stays as it is deploys as before"
    )
