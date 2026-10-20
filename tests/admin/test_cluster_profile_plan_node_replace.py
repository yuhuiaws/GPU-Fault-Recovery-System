"""The ``CLUSTER_PROFILE_NEW`` plan line states the override's real ``nodeReplace``.

The approver of a cluster's first own Runtime Profile version reads one line
per cluster. It used to end in a hard-coded ``nodeReplace OBSERVE`` while both
provider templates OWN ``nodeReplace`` as the warm-spare handover
(``warm-spare-v1``): the adapters refuse provider-side replacement, so
REPLACE_NODE only moves the workload to a declared spare. The line is now read
from the template the reviewed snapshot copies (batch 148).

The line is part of ``plan_sha256``, so a plan an older admin CLI left pending
is re-planned under a new digest: its approval is archived ``SUPERSEDED`` and
the command stops for one more review; nothing is snapshotted on the old one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_join_profile, profile_approval

ROOT = Path(__file__).resolve().parents[2]
HYPERPOD = "config/runtime-profile.regional-hyperpod-safe.example.yaml"
EKS = "config/runtime-profile.regional-eks-safe.example.yaml"
ARN = "arn:aws:eks:us-west-2:111122223333:cluster/{}"
NO_REPLACEMENT = (
    "nothing replaces the node or hands its work to a spare: a REPLACE_NODE "
    "step has no executable owner and its workflow is BLOCKED for the operator"
)
HANDOVER = (
    "as the warm-spare handover only (warm-spare-v1: REPLACE_NODE moves the "
    "workload to a declared healthy spare, fails closed without one and never "
    "calls a provider replace API)"
)
EKS_ON_HYPERPOD_SITE = (
    "cluster eks-1 (eks-managed) runs its own Runtime Profile eks-v1: nodeReboot "
    "owned by gpu-fault-eks-adapter, nodeReplace {policy}; the site profile "
    "hyperpod-v1 stays on every other cluster"
)
#: The line of a pre-batch-148 admin CLI for the same plan.
LEGACY_EKS_LINE = (
    "cluster eks-1 (eks-managed) runs its own Runtime Profile eks-v1: nodeReboot "
    "owned by gpu-fault-eks-adapter, nodeReplace OBSERVE; the site profile "
    "hyperpod-v1 stays on every other cluster"
)


def _site(tmp_path: Path, *, base: str) -> tuple[Path, Path]:
    """A repository holding both example profiles and a deployed site whose
    profile follows ``base``, with one cluster of the other provider that has
    no profile of its own yet."""

    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    for reference in (HYPERPOD, EKS):
        (repo / reference).write_text(
            (ROOT / reference).read_text(encoding="utf-8"), encoding="utf-8"
        )
    hyperpod = {
        "clusterId": "hp-1",
        "context": "ctx-hp-1",
        "hyperpodClusterName": "hp-1",
        "eksClusterArn": ARN.format("hp-1-eks"),
    }
    eks = {
        "clusterId": "eks-1",
        "context": "ctx-eks-1",
        "provider": "eks-managed",
        "eksClusterArn": ARN.format("eks-1"),
    }
    anchor, joining = (hyperpod, eks) if base == "hyperpod" else (eks, hyperpod)
    document = {
        "apiVersion": "gpu-fault.aws/v1alpha1",
        "kind": "RegionalSite",
        "metadata": {"name": "site"},
        "spec": {
            "awsRegion": "us-west-2",
            "cpu": {"eksArn": ARN.format("cpu")},
            "runtimeProfile": {
                "source": str(repo / (HYPERPOD if base == "hyperpod" else EKS)),
                "version": "hyperpod-v1" if base == "hyperpod" else "eks-v1",
                "registrationClusterId": anchor["clusterId"],
            },
            "clusters": [anchor, joining],
        },
    }
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return repo, state_dir


def _edit_eks_template(repo: Path, edit: str) -> None:
    """Give the repository's EKS template another ``nodeReplace`` policy."""

    template = repo / EKS
    document = yaml.safe_load(template.read_text(encoding="utf-8"))
    claims = document["claims"]
    claim = next(item for item in claims if item["capability"] == "nodeReplace")
    if edit == "unclaimed":
        claims.remove(claim)
    elif edit == "unobserved":
        document["observed"] = [
            item for item in document["observed"] if item["capability"] != "nodeReplace"
        ]
    else:
        claim["mode"] = edit
        if edit in {"OBSERVE", "DISABLED"}:
            claim.pop("adapter")
    template.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _plan(state_dir: Path) -> dict[str, Any]:
    path = profile_approval.profile_plan_path(state_dir)
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _stop(repo: Path, state_dir: Path) -> str:
    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired) as stop:
        cluster_join_profile.preflight_cluster_profile_plans(
            state_dir, repository_root=repo
        )
    return str(stop.value)


@pytest.mark.parametrize(
    ("base", "line"),
    [
        (
            "hyperpod",
            EKS_ON_HYPERPOD_SITE.format(
                policy=f"owned by gpu-fault-eks-adapter {HANDOVER}"
            ),
        ),
        (
            "eks-managed",
            "cluster hp-1 (hyperpod) runs its own Runtime Profile hyperpod-v1: "
            "nodeReboot owned by gpu-fault-hyperpod-adapter, nodeReplace owned by "
            f"gpu-fault-hyperpod-adapter {HANDOVER}; the site profile eks-v1 "
            "stays on every other cluster",
        ),
    ],
    ids=["eks-cluster-on-a-hyperpod-site", "hyperpod-cluster-on-an-eks-site"],
)
def test_the_plan_line_names_the_warm_spare_handover_the_template_owns(
    tmp_path: Path, base: str, line: str
) -> None:
    repo, state_dir = _site(tmp_path, base=base)

    message = _stop(repo, state_dir)

    plan = _plan(state_dir)
    assert plan["change_kind"] == "CLUSTER_PROFILE_NEW", plan
    assert plan["changes"] == [line], plan["changes"]
    assert f"    - {line}\n" in message, "the stop prints the line to the approver"
    assert "OBSERVE" not in line, "both templates OWN nodeReplace"


@pytest.mark.parametrize(
    ("edit", "policy"),
    [
        ("OBSERVE", f"OBSERVE ({NO_REPLACEMENT})"),
        ("DISABLED", f"DISABLED ({NO_REPLACEMENT})"),
        ("unclaimed", f"not claimed ({NO_REPLACEMENT})"),
        (
            "unobserved",
            "OWN by gpu-fault-eks-adapter without an available observed "
            f"implementation, so OBSERVE ({NO_REPLACEMENT})",
        ),
        ("DELEGATE", "delegated to gpu-fault-eks-adapter"),
    ],
    ids=["OBSERVE", "DISABLED", "unclaimed", "unobserved", "DELEGATE"],
)
def test_the_plan_line_follows_the_template_policy_it_reviews(
    tmp_path: Path, edit: str, policy: str
) -> None:
    """An OBSERVE ``nodeReplace`` reads as what it does: no handover, no
    provider call, a REPLACE_NODE workflow BLOCKED for the operator; an OWN
    claim without an available observed implementation is the OBSERVE the
    profile compiler reduces it to."""

    repo, state_dir = _site(tmp_path, base="hyperpod")
    _edit_eks_template(repo, edit)

    _stop(repo, state_dir)

    assert _plan(state_dir)["changes"] == [
        EKS_ON_HYPERPOD_SITE.format(policy=policy)
    ], "the line states the policy of the snapshot under review"


def test_a_plan_pending_under_the_former_wording_is_superseded_not_consumed(
    tmp_path: Path,
) -> None:
    """An operator who approved the plan an older admin CLI wrote (same
    policy, snapshot and template, ``nodeReplace OBSERVE`` in its line) gets
    one more review: the line is part of ``plan_sha256``."""

    repo, state_dir = _site(tmp_path, base="hyperpod")
    _stop(repo, state_dir)
    current = _plan(state_dir)
    legacy = {**current, "changes": [LEGACY_EKS_LINE]}
    legacy["plan_sha256"] = profile_approval.profile_plan_sha256(legacy)
    assert legacy["plan_sha256"] != current["plan_sha256"], "the line is hashed"
    profile_approval.clear_profile_plan(state_dir)
    profile_approval.write_profile_plan(state_dir, legacy)
    profile_approval.approve_profile(
        state_dir,
        reference="CHG-148",
        expected_plan_sha256=legacy["plan_sha256"],
        approver_identity="tester@fixture",
    )

    message = _stop(repo, state_dir)

    assert "archived as SUPERSEDED" in message, message
    assert f"plan_sha256: {current['plan_sha256']}" in message, message
    assert _plan(state_dir) == current, "the current wording is written for review"
    archive = profile_approval.profile_approval_archive_path(
        state_dir, legacy["plan_sha256"]
    )
    superseded = json.loads((archive / "superseded.json").read_text(encoding="utf-8"))
    assert superseded["replacement_plan_sha256"] == current["plan_sha256"], superseded
    assert not profile_approval.profile_approval_path(state_dir).exists(), (
        "no approval is left to consume"
    )
    assert not (state_dir / "profiles" / "eks-v1.yaml").exists(), (
        "nothing is snapshotted on the superseded approval"
    )
    assert {
        field: legacy[field]
        for field in ("policy_digest", "snapshot_sha256", "source_sha256")
    } == {
        field: current[field]
        for field in ("policy_digest", "snapshot_sha256", "source_sha256")
    }, "only the wording differs; the reviewed policy is the same"

    profile_approval.approve_profile(
        state_dir,
        reference="CHG-148",
        expected_plan_sha256=current["plan_sha256"],
        approver_identity="tester@fixture",
    )
    cluster_join_profile.preflight_cluster_profile_plans(
        state_dir, repository_root=repo
    )
    assert profile_approval.profile_approval_path(state_dir).is_file(), (
        "the re-approved plan is accepted and left for the step that writes it"
    )
