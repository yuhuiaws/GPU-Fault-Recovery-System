"""A joining cluster's own Runtime Profile: the refusals around its review (cov95).

The join and deploy suites drive ``cluster_join_profile`` through whole joins.
These cases pin its own rules directly: a provider template that is missing
or is not one mapping, a plan over no cluster or over clusters needing
different versions, an approved snapshot whose policy drifted, an approval
record that lost its plan or that cannot be superseded, a pinned entry whose
template is gone or whose digest-named successor is already approved, an
unreadable site profile in the deploy (strict) and the preflight (lenient).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_join_profile as profiles
from gpu_fault.admin import profile_approval
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_profile import (
    ClusterProfileOverride,
    JoinProfileApprovalRequired,
)
from gpu_fault.admin.profile_approval import ProfileApprovalError
from gpu_fault.cluster_scope import PROVIDER_PROFILE_SOURCES
from gpu_fault_release.regional_release_profile_model import (
    runtime_profile_policy_sha256,
)

EKS = "eks-managed"
REGION = "us-east-1"
CPU_ARN = f"arn:aws:eks:{REGION}:123456789012:cluster/control"
EKS_TEMPLATE = PROVIDER_PROFILE_SOURCES[EKS]
HYPERPOD_OWNED = {
    "environment": "hyperpod-eks",
    "claims": [
        {
            "capability": "nodeReboot",
            "mode": "OWN",
            "owner": "gpu-fault-hyperpod-adapter",
        }
    ],
    "observed": [],
}


def write_yaml(path: Path, document: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def eks_policy(replace_mode: str = "OWN") -> dict[str, Any]:
    return {
        "environment": "eks",
        "claims": [
            {
                "capability": "nodeReboot",
                "mode": "OWN",
                "owner": "gpu-fault-eks-adapter",
            },
            {"capability": "nodeReplace", "mode": replace_mode, "owner": "operator"},
        ],
        "observed": [],
    }


def rendered_site(tmp_path: Path) -> SimpleNamespace:
    """The parts of a ``RenderedSite`` the profile review reads."""

    source = write_yaml(tmp_path / "state/site-profile.yaml", HYPERPOD_OWNED)
    return SimpleNamespace(
        source=tmp_path / "state/site.yaml",
        repository_root=tmp_path / "repo",
        release_config={
            "site_name": "test-site",
            "aws_region": REGION,
            "cpu_eks_arn": CPU_ARN,
            "runtime_profile": {"version": "hyperpod-v1", "source": str(source)},
        },
    )


def override(
    tmp_path: Path, cluster_id: str = "gpu-e", *, version: str = "eks-v1"
) -> ClusterProfileOverride:
    template = tmp_path / "repo" / EKS_TEMPLATE
    if not template.is_file():
        write_yaml(template, eks_policy())
    return ClusterProfileOverride(
        cluster_id=cluster_id,
        provider=EKS,
        version=version,
        template_reference=EKS_TEMPLATE,
        template_source=template,
        policy_digest=runtime_profile_policy_sha256(template),
        snapshot_file=tmp_path / "state/profiles" / f"{version}.yaml",
    )


def pending_plan(state_dir: Path) -> dict[str, Any]:
    return dict(json.loads(profile_approval.profile_plan_path(state_dir).read_text()))


def stop_for_review(site: SimpleNamespace, items: list[ClusterProfileOverride]) -> str:
    with pytest.raises(JoinProfileApprovalRequired) as stopped:
        profiles.ensure_cluster_profiles_approved(site, items, rerun_command="rerun")
    assert stopped.value.cluster_ids == tuple(sorted(i.cluster_id for i in items))
    return str(pending_plan(site.source.parent)["plan_sha256"])


def approve(state_dir: Path, plan_sha256: str) -> None:
    profile_approval.approve_profile(
        state_dir,
        reference="CHG-cov95-1",
        expected_plan_sha256=plan_sha256,
        approver_identity="tester@fixture",
    )


# --- the override a joining cluster needs ---------------------------------------


def test_a_provider_template_missing_from_the_repository_is_refused(
    tmp_path: Path,
) -> None:
    site = rendered_site(tmp_path)
    target = SimpleNamespace(provider=EKS)

    with pytest.raises(BootstrapError) as refused:
        profiles.cluster_profile_override(site, target, "gpu-e")  # type: ignore[arg-type]

    assert str(refused.value) == (
        f"provider example profile is missing: {tmp_path / 'repo' / EKS_TEMPLATE}"
    )
    write_yaml(tmp_path / "repo" / EKS_TEMPLATE, eks_policy())
    found = profiles.cluster_profile_override(site, target, "gpu-e")  # type: ignore[arg-type]
    assert found is not None and found.version == "eks-v1", found
    assert found.owner == "gpu-fault-eks-adapter"


def test_a_plan_needs_a_cluster_and_one_version_for_all_of_them(tmp_path: Path) -> None:
    site = rendered_site(tmp_path)

    with pytest.raises(BootstrapError, match="needs at least one cluster"):
        profiles.ensure_cluster_profiles_approved(site, [], rerun_command="rerun")
    with pytest.raises(BootstrapError, match="must need the same Runtime Profile"):
        profiles.ensure_cluster_profiles_approved(
            site,
            [override(tmp_path), override(tmp_path, "gpu-f", version="regional-eks-1")],
            rerun_command="rerun",
        )
    assert not profile_approval.profile_plan_path(site.source.parent).exists(), (
        "no plan is written for a refused group"
    )


def test_an_approved_snapshot_with_another_policy_is_never_reused(
    tmp_path: Path,
) -> None:
    site = rendered_site(tmp_path)
    item = override(tmp_path)
    write_yaml(item.snapshot_file, eks_policy(replace_mode="DELEGATE"))

    with pytest.raises(BootstrapError) as refused:
        profiles.ensure_cluster_profile_approved(site, item, rerun_command="rerun")

    message = str(refused.value)
    assert message.startswith(
        f"approved Runtime Profile snapshot {item.snapshot_file} has a different "
        f"policy than {EKS_TEMPLATE}"
    ), message
    assert "remove the stale snapshot after review" in message


def test_a_template_that_is_not_one_mapping_cannot_be_planned(tmp_path: Path) -> None:
    site = rendered_site(tmp_path)
    item = override(tmp_path)
    item.template_source.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(BootstrapError, match="must contain one mapping"):
        profiles.ensure_cluster_profiles_approved(site, [item], rerun_command="rerun")

    assert not item.snapshot_file.exists(), (
        "no snapshot of a template that is not one mapping"
    )


# --- the approval -------------------------------------------------------------------


def test_an_approval_record_that_lost_its_plan_is_unusable(tmp_path: Path) -> None:
    site = rendered_site(tmp_path)
    item = override(tmp_path)
    approve(site.source.parent, stop_for_review(site, [item]))
    profile_approval.profile_plan_path(site.source.parent).unlink()

    with pytest.raises(BootstrapError) as refused:
        profiles.ensure_cluster_profiles_approved(site, [item], rerun_command="rerun")

    assert str(refused.value) == (
        "Runtime Profile approval is unusable: Profile approval record exists "
        "without its pending plan"
    )
    assert not isinstance(refused.value, JoinProfileApprovalRequired), (
        "a damaged approval is an error, not a new review"
    )
    assert not item.snapshot_file.exists(), "nothing is written on an unusable record"


def test_a_stale_approval_that_cannot_be_superseded_is_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = rendered_site(tmp_path)
    approve(site.source.parent, stop_for_review(site, [override(tmp_path)]))
    attempts: list[str] = []

    def supersede(state_dir: Path, **options: Any) -> None:
        attempts.append(str(options["reason"]))
        raise ProfileApprovalError("Profile approval archive is inconsistent")

    monkeypatch.setattr(profiles, "supersede_profile_approval", supersede)
    batch = [override(tmp_path), override(tmp_path, "gpu-f")]

    with pytest.raises(BootstrapError) as refused:
        profiles.ensure_cluster_profiles_approved(site, batch, rerun_command="rerun")

    assert str(refused.value) == (
        "Runtime Profile approval is unusable: Profile approval archive is inconsistent"
    )
    assert attempts == [
        "Profile approval is stale because the pending plan or live baseline changed"
    ], "the plan for two clusters is not the one approved for one"
    assert not batch[0].snapshot_file.exists(), (
        "nothing written on an unusable approval"
    )


# --- the deploy's per-cluster pins ------------------------------------------------------


def site_document(*clusters: dict[str, Any], source: str | None = None) -> dict:
    profile = {"version": "hyperpod-v1", **({"source": source} if source else {})}
    return {
        "metadata": {"name": "test-site"},
        "spec": {
            "awsRegion": REGION,
            "cpu": {"eksArn": CPU_ARN},
            "runtimeProfile": profile,
            "clusters": list(clusters),
        },
    }


def pinned_entry(version: str) -> dict[str, Any]:
    return {
        "clusterId": "gpu-e",
        "provider": EKS,
        "eksClusterArn": f"arn:aws:eks:{REGION}:123456789012:cluster/gpu-e",
        "runtimeProfile": {
            "source": f"/old/{version}.yaml",
            "templateSource": "config/custom-eks.yaml",
            "version": version,
            "agentConfigDigest": "0" * 64,
        },
    }


@pytest.fixture
def digests(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    computed: list[str] = []

    def compute(_runner: Any, *, runtime_profile_version: str, **_options: Any) -> str:
        computed.append(runtime_profile_version)
        return f"digest-of-{runtime_profile_version}"

    monkeypatch.setattr(profiles, "compute_agent_config_digest", compute)
    return computed


def test_a_pin_whose_template_left_the_repository_keeps_its_snapshot(
    tmp_path: Path, digests: list[str]
) -> None:
    state_dir = tmp_path / "state"
    snapshot = write_yaml(state_dir / "profiles/eks-v1.yaml", eks_policy())
    document = site_document(pinned_entry("eks-v1"))

    profiles.reconcile_cluster_runtime_profiles(
        object(),  # type: ignore[arg-type]
        document,
        repository_root=tmp_path / "empty-repo",
        state_dir=state_dir,
    )

    assert document["spec"]["clusters"][0]["runtimeProfile"] == {
        "source": str(snapshot),
        "templateSource": "config/custom-eks.yaml",
        "version": "eks-v1",
        "agentConfigDigest": "digest-of-eks-v1",
    }
    assert digests == ["eks-v1"]
    assert not profile_approval.profile_plan_path(state_dir).exists(), (
        "a pin that keeps its snapshot needs no plan"
    )


def test_an_approved_digest_named_successor_is_pinned_without_a_review(
    tmp_path: Path, digests: list[str]
) -> None:
    repository = tmp_path / "repo"
    template = write_yaml(
        repository / EKS_TEMPLATE, eks_policy(replace_mode="DELEGATE")
    )
    state_dir = tmp_path / "state"
    write_yaml(state_dir / "profiles/eks-v1.yaml", eks_policy())
    digest = runtime_profile_policy_sha256(template)
    successor = f"regional-eks-{digest[:12]}"
    approved = write_yaml(
        state_dir / f"profiles/{successor}.yaml", eks_policy(replace_mode="DELEGATE")
    )
    document = site_document(pinned_entry("eks-v1"))

    profiles.reconcile_cluster_runtime_profiles(
        object(),  # type: ignore[arg-type]
        document,
        repository_root=repository,
        state_dir=state_dir,
    )

    assert document["spec"]["clusters"][0]["runtimeProfile"] == {
        "source": str(approved),
        "templateSource": EKS_TEMPLATE,
        "version": successor,
        "agentConfigDigest": f"digest-of-{successor}",
    }
    assert not profile_approval.profile_plan_path(state_dir).exists(), (
        "an approved successor needs no new plan"
    )


def test_the_deploy_refuses_a_site_profile_it_cannot_read(
    tmp_path: Path, digests: list[str]
) -> None:
    missing = tmp_path / "missing-profile.yaml"
    document = site_document(pinned_entry("eks-v1"), source=str(missing))

    with pytest.raises(BootstrapError, match="cannot read runtime profile"):
        profiles.reconcile_cluster_runtime_profiles(
            object(),  # type: ignore[arg-type]
            document,
            repository_root=tmp_path / "repo",
            state_dir=tmp_path / "state",
        )

    assert digests == [], "nothing is computed for a site the deploy cannot judge"


@pytest.mark.parametrize(
    "content", ["spec: [unclosed\n", "- a\n- list\n"], ids=["invalid", "not-a-mapping"]
)
def test_the_preflight_leaves_an_unreadable_site_to_the_finalization(
    tmp_path: Path, content: str
) -> None:
    (tmp_path / "site.yaml").write_text(content, encoding="utf-8")

    assert (
        profiles.preflight_cluster_profile_plans(tmp_path, repository_root=tmp_path)
        is None
    )
    assert not profile_approval.profile_plan_path(tmp_path).exists(), (
        "the preflight writes nothing for a site it cannot read"
    )
