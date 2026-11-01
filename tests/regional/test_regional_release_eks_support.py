"""EKS end-of-support control-plane upgrades: recorded and announced, never refused.

The world modelled is the live one of 2026-11-01, sanitised: the plain EKS GPU
cluster runs Kubernetes 1.31 under the ``EXTENDED`` upgrade policy, whose
extended support ends 2026-11-26 (``describe-cluster-versions`` prints
``2026-11-25T16:00:00-08:00``); the CPU control-plane cluster's EKS cluster and
the HyperPod GPU cluster's run 1.33, supported until 2027-07-29. After the
date AWS upgrades the control plane by itself, at any time and without notice,
and the upgrade cannot be rolled back. The ``regional_contexts`` check records
every site EKS cluster's version and date; the status header names a cluster
within 90 days of its date; a read that fails is recorded and announced as
"could not read" while the check passes as before.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from gpu_fault_release import regional_release_eks_support as support
from gpu_fault_release import regional_release_preflight as preflight
from gpu_fault_release.regional_admin_commands import status_header_lines
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional.test_regional_release_automatic_updates import (
    EKS,
    EKS_ARN,
    EKS_SUPPORT_DATES,
    HP_EKS_ARN,
    HYPERPOD,
    ContextsRelease,
    ContextsRunner,
    _report,
)

NOW = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)
REGION = "us-west-2"
CPU = support.EksCluster(
    "the CPU control-plane cluster", f"arn:aws:eks:{REGION}:1:cluster/cpu", REGION
)
GPU = support.EksCluster("GPU cluster gpu-eks", EKS_ARN, REGION)
SECTION = "(administrator-operations §5.0.5)"


class Versions:
    """``read_versions``: the dates of the versions asked for, as queried."""

    def __init__(self, dates: dict[str, dict[str, str]] | None = None) -> None:
        self.dates = dates if dates is not None else EKS_SUPPORT_DATES
        self.asked: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, region: str, versions: Any) -> dict[str, Any]:
        self.asked.append((region, tuple(versions)))
        return {
            "ClusterVersions": [
                {
                    "Version": version,
                    "EndOfStandardSupportDate": self.dates[version].get(
                        "endOfStandardSupportDate"
                    ),
                    "EndOfExtendedSupportDate": self.dates[version].get(
                        "endOfExtendedSupportDate"
                    ),
                }
                for version in versions
                if version in self.dates
            ]
        }


def _clusters(**versions: dict[str, Any]):
    def read(cluster: support.EksCluster) -> dict[str, Any]:
        value = versions[cluster.name]
        if isinstance(value, Exception):
            raise value
        return value

    return read


def _block(read_cluster, read_versions=None, now=NOW) -> dict[str, Any]:
    return support.eks_support_block(
        [CPU, GPU],
        read_cluster=read_cluster,
        read_versions=read_versions or Versions(),
        now=now,
    )


# --- the facts ---------------------------------------------------------------------


def test_the_live_extended_cluster_is_24_whole_days_from_its_upgrade() -> None:
    read_versions = Versions()

    block = _block(
        _clusters(
            cpu={"Version": "1.33", "SupportType": "EXTENDED"},
            **{"gpu-eks-cluster": {"Version": "1.31", "SupportType": "EXTENDED"}},
        ),
        read_versions,
    )

    assert block == {
        "notice_days": 90,
        "clusters": [
            {
                "cluster": "the CPU control-plane cluster",
                "eks_cluster": "cpu",
                "region": REGION,
                "version": "1.33",
                "support_type": "EXTENDED",
                "end_of_support": "2027-07-29",
                "days_left": 269,
                "state": "read",
            },
            {
                "cluster": "GPU cluster gpu-eks",
                "eks_cluster": "gpu-eks-cluster",
                "region": REGION,
                "version": "1.31",
                "support_type": "EXTENDED",
                "end_of_support": "2026-11-26",
                "days_left": 24,
                "state": "read",
            },
        ],
    }, "the Pacific-time timestamp is the next UTC day; whole days left"
    assert read_versions.asked == [(REGION, ("1.31", "1.33"))], (
        "one describe-cluster-versions per region for every version in use"
    )


def test_standard_support_ends_at_the_standard_date_and_none_means_extended() -> None:
    block = _block(
        _clusters(
            cpu={"Version": "1.33", "SupportType": None},
            **{"gpu-eks-cluster": {"Version": "1.31", "SupportType": "STANDARD"}},
        )
    )

    cpu, gpu = block["clusters"]
    assert (cpu["support_type"], cpu["support_type_default"]) == ("EXTENDED", True), (
        "AWS: the upgrade policy is EXTENDED unless specified otherwise"
    )
    assert cpu["end_of_support"] == "2027-07-29"
    assert (gpu["end_of_support"], gpu["days_left"]) == ("2025-11-26", -341), (
        "STANDARD: the end of standard support, here long past"
    )


def test_a_failed_read_is_unknown_with_its_reason_and_nothing_raises() -> None:
    denied = ReleaseError(
        "aws eks describe-cluster exited 254: AccessDeniedException: eks:DescribeCluster"
    )

    block = _block(
        _clusters(cpu=denied, **{"gpu-eks-cluster": {"Version": "1.29"}}),
        read_versions=Versions({}),
    )

    cpu, gpu = block["clusters"]
    assert cpu["state"] == "unknown" and "eks:DescribeCluster" in cpu["error"]
    assert "version" not in cpu
    assert gpu["state"] == "unknown"
    assert gpu["error"] == (
        "eks describe-cluster-versions returned no end of extended support date "
        "for Kubernetes 1.29"
    )

    def broken(_region: str, _versions: Any) -> dict[str, Any]:
        raise ReleaseError("AccessDeniedException: eks:DescribeClusterVersions")

    block = _block(
        _clusters(cpu={"Version": "1.33"}, **{"gpu-eks-cluster": {"Version": "1.31"}}),
        read_versions=broken,
    )
    assert block["versions_error"] == {
        REGION: "AccessDeniedException: eks:DescribeClusterVersions"
    }
    assert {item["state"] for item in block["clusters"]} == {"unknown"}


# --- the lines ---------------------------------------------------------------------


def test_the_line_names_the_cluster_version_date_and_consequence() -> None:
    block = _block(
        _clusters(
            cpu={"Version": "1.33", "SupportType": "EXTENDED"},
            **{"gpu-eks-cluster": {"Version": "1.31", "SupportType": "EXTENDED"}},
        )
    )

    assert support.eks_support_lines(block) == [
        "advisory: EKS cluster gpu-eks-cluster (GPU cluster gpu-eks) runs Kubernetes "
        "1.31 under the EXTENDED upgrade policy, whose support ends 2026-11-26 (in 24 "
        "day(s)): after that date AWS upgrades its control plane to the next version "
        "by itself, at any time and without notice -- only the control plane: its "
        "nodes stay on 1.31 -- and a cluster upgraded that way cannot be rolled back; "
        "upgrade it yourself before then, in a maintenance window (control plane "
        f"first, then its nodes and add-ons) {SECTION}"
    ], "the 1.33 clusters are outside the 90-day notice"


def test_a_passed_date_says_ended_and_the_default_policy_says_so() -> None:
    block = _block(
        _clusters(cpu={"Version": "1.33"}, **{"gpu-eks-cluster": {"Version": "1.31"}}),
        now=datetime(2026, 12, 3, tzinfo=timezone.utc),
    )

    (line,) = support.eks_support_lines(block)

    assert line.startswith(
        "advisory: EKS cluster gpu-eks-cluster (GPU cluster gpu-eks) runs Kubernetes "
        "1.31 under the EXTENDED upgrade policy (the default), whose support ended "
        "2026-11-26 (7 day(s) ago): AWS may upgrade its control plane"
    ), line
    assert "upgrade it yourself now" in line


@pytest.mark.parametrize(
    "now,announced",
    [
        (datetime(2026, 8, 26, 23, 59, tzinfo=timezone.utc), False),
        (datetime(2026, 8, 27, 23, 59, tzinfo=timezone.utc), True),
    ],
    ids=["91-days", "90-days"],
)
def test_the_notice_starts_90_days_before_the_date(
    now: datetime, announced: bool
) -> None:
    block = _block(
        _clusters(cpu={"Version": "1.33"}, **{"gpu-eks-cluster": {"Version": "1.31"}}),
        now=now,
    )

    assert bool(support.eks_support_lines(block)) is announced, block["clusters"][1]


def test_what_could_not_be_read_is_a_visible_line() -> None:
    def broken(_region: str, _versions: Any) -> dict[str, Any]:
        raise ReleaseError("AccessDeniedException: eks:DescribeClusterVersions")

    denied = _block(
        _clusters(cpu={"Version": "1.33"}, **{"gpu-eks-cluster": {"Version": "1.31"}}),
        read_versions=broken,
    )
    no_version = _block(
        _clusters(
            cpu=ReleaseError("AccessDeniedException"),
            **{"gpu-eks-cluster": {"Version": "1.29"}},
        )
    )

    assert support.eks_support_lines(denied) == [
        f"advisory: could not read the EKS support dates in {REGION} "
        "(AccessDeniedException: eks:DescribeClusterVersions): the end-of-support "
        "check of 2 EKS cluster(s) did not run; the operator identity needs "
        f"eks:DescribeClusterVersions {SECTION}"
    ]
    assert support.eks_support_lines(no_version) == [
        "advisory: could not read the Kubernetes version of EKS cluster cpu (the CPU "
        "control-plane cluster) (AccessDeniedException): its end-of-support check did "
        f"not run {SECTION}",
        "advisory: could not read the support dates of EKS cluster gpu-eks-cluster "
        "(GPU cluster gpu-eks), Kubernetes 1.29 (eks describe-cluster-versions "
        "returned no end of extended support date for Kubernetes 1.29): its "
        f"end-of-support check did not run {SECTION}",
    ]


def test_a_malformed_block_announces_nothing() -> None:
    assert support.eks_support_lines(None) == []
    assert support.eks_support_lines({"state": "unknown", "error": "x"}) == []
    assert (
        support.eks_support_lines(
            {"clusters": ["x", {"state": "read", "days_left": "soon"}]}
        )
        == []
    ), "a status header never fails"


# --- the check and the header -----------------------------------------------------------


def _facts(
    runner: ContextsRunner, monkeypatch: pytest.MonkeyPatch, now: datetime = NOW
) -> dict[str, Any]:
    monkeypatch.setattr(support, "utc_now", lambda: now)
    release = ContextsRelease(runner, (HYPERPOD, EKS))
    facts = preflight.ensure_region_contexts(release)  # type: ignore[arg-type]
    assert sorted(release.iam) == ["gpu-eks", "gpu-hp"], "every gate ran"
    return facts


def test_contexts_record_every_site_eks_cluster_and_the_header_names_the_due_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = ContextsRunner(eks_versions={"gpu-eks-cluster": "1.31"})

    facts = _facts(runner, monkeypatch)

    block = facts[support.FIELD]
    assert [
        (item["eks_cluster"], item["version"], item["end_of_support"])
        for item in block["clusters"]
    ] == [
        ("cpu", "1.33", "2027-07-29"),
        ("hp-gpu-eks", "1.33", "2027-07-29"),
        ("gpu-eks-cluster", "1.31", "2026-11-26"),
    ], "the CPU EKS cluster, the HyperPod orchestrator and the plain EKS cluster"
    assert HP_EKS_ARN.endswith(block["clusters"][1]["eks_cluster"])
    describes = [
        call
        for call in runner.calls
        if call[:3] == ["aws", "eks", "describe-cluster"]
        and call[call.index("--query") + 1] == support.CLUSTER_QUERY
    ]
    assert len(describes) == 3
    versions = [
        call
        for call in runner.calls
        if call[:3] == ["aws", "eks", "describe-cluster-versions"]
    ]
    assert len(versions) == 1, "one versions read for the region"
    lines = status_header_lines(_report(facts), json_destination="stdout")
    assert lines[0] == "healthy: yes (full health, 1 checks)", "the verdict stays"
    (line,) = [line for line in lines if "EKS cluster gpu-eks-cluster" in line]
    assert "whose support ends 2026-11-26 (in 24 day(s))" in line


def test_a_denied_versions_read_is_a_line_and_the_check_still_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = _facts(ContextsRunner(deny_versions=True), monkeypatch)

    block = facts[support.FIELD]
    assert "eks:DescribeClusterVersions" in block["versions_error"][REGION]
    lines = status_header_lines(_report(facts), json_destination="stdout")
    assert any(
        line.startswith(f"advisory: could not read the EKS support dates in {REGION}")
        and "the end-of-support check of 3 EKS cluster(s) did not run" in line
        for line in lines
    ), lines


def test_a_broken_eks_support_advisory_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(_release: Any) -> dict[str, Any]:
        raise RuntimeError("advisory defect")

    monkeypatch.setattr(preflight, "read_eks_support", broken)

    facts = _facts(ContextsRunner(), monkeypatch)

    assert facts[support.FIELD] == {"state": "unknown", "error": "advisory defect"}
    assert support.eks_support_lines(facts[support.FIELD]) == []


# --- the evidence ---------------------------------------------------------------------


def test_the_advisory_cites_the_aws_version_lifecycle_page() -> None:
    assert support.EKS_SUPPORT_EVIDENCE == {
        "vendor": "AWS",
        "url": "https://docs.aws.amazon.com/eks/latest/userguide/kubernetes-versions.html",
        "page": "Understand the Kubernetes version lifecycle on EKS",
        "sections": ("Amazon EKS extended support FAQs", "Version rollback"),
        "quotes": (
            "Automatic updates can happen at any time after the end of extended "
            "support date. You won’t receive any notification before the update.",
            "The auto-upgrade includes only the Kubernetes control plane.",
            "Clusters that were automatically upgraded at the end of extended "
            "support cannot be rolled back.",
            "Note: If you disable extended support, your cluster will be auto "
            "upgraded at the end of standard support.",
        ),
    }
    assert support.NOTICE_DAYS == 90
