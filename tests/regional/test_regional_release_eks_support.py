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

The versions read carries one selector, ``--cluster-versions``: the service
refuses two in one request, as it refused batch 231's ``--include-all`` beside
it on every site, and the fake refuses them the same way. Only an
authorization failure names the missing permission.
"""

from __future__ import annotations

import itertools
import json
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
    VERSIONS_REFUSAL,
    VERSIONS_SELECTORS,
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


# --- the request -------------------------------------------------------------------


def _versions_requests(runner: ContextsRunner) -> list[list[str]]:
    return [
        call
        for call in runner.calls
        if call[:3] == ["aws", "eks", "describe-cluster-versions"]
    ]


def test_the_versions_read_names_the_versions_in_use_and_no_other_selector() -> None:
    runner = ContextsRunner(eks_versions={"gpu-eks-cluster": "1.31"})

    block = support.read_eks_support(ContextsRelease(runner, (HYPERPOD, EKS)), now=NOW)

    assert "versions_error" not in block, block.get("versions_error")
    assert [
        (item["eks_cluster"], item["state"], item.get("end_of_support"))
        for item in block["clusters"]
    ] == [
        ("cpu", "read", "2027-07-29"),
        ("hp-gpu-eks", "read", "2027-07-29"),
        ("gpu-eks-cluster", "read", "2026-11-26"),
    ], "the service answers the request the check sends: every date is read"
    assert _versions_requests(runner) == [
        [
            "aws",
            "eks",
            "describe-cluster-versions",
            "--region",
            REGION,
            "--cluster-versions",
            "1.31",
            "1.33",
            "--query",
            support.VERSIONS_QUERY,
            "--output",
            "json",
        ]
    ], "one request per region, with one selector: the versions in use"


def _selector(option: str) -> list[str]:
    if option == "--cluster-versions":
        return [option, "1.31"]
    if option == "--status":
        return [option, "extended-support"]
    if option == "--version-status":
        return [option, "EXTENDED_SUPPORT"]
    return [option]


@pytest.mark.parametrize(
    "pair",
    list(itertools.combinations(VERSIONS_SELECTORS, 2)),
    ids=lambda pair: "+".join(option.lstrip("-") for option in pair),
)
def test_the_fake_refuses_two_selectors_in_the_services_words(
    pair: tuple[str, str],
) -> None:
    arguments = ["aws", "eks", "describe-cluster-versions", "--region", REGION]
    for option in pair:
        arguments += _selector(option)

    with pytest.raises(ReleaseError) as raised:
        ContextsRunner().run([*arguments, "--output", "json"], capture=True)

    message = str(raised.value)
    assert message.startswith(
        f"command failed (254): aws eks describe-cluster-versions --region {REGION} "
    ), message
    assert message.endswith(f": {VERSIONS_REFUSAL}"), message


def test_batch_231s_request_is_refused_and_its_line_blames_no_permission() -> None:
    """The live 2026-11-02 advisory, its reason cut where the live one was, now
    with a remedy that does not claim a missing permission."""

    runner = ContextsRunner()

    def batch_231_request(region: str, versions: Any) -> dict[str, Any]:
        return json.loads(
            runner.run(
                [
                    "aws",
                    "eks",
                    "describe-cluster-versions",
                    "--region",
                    region,
                    "--cluster-versions",
                    *versions,
                    "--include-all",
                    "--query",
                    support.VERSIONS_QUERY,
                    "--output",
                    "json",
                ],
                capture=True,
            )
        )

    block = _block(
        _clusters(cpu={"Version": "1.33"}, **{"gpu-eks-cluster": {"Version": "1.31"}}),
        read_versions=batch_231_request,
    )

    assert support.eks_support_lines(block) == [
        f"advisory: could not read the EKS support dates in {REGION} (command failed "
        f"(254): aws eks describe-cluster-versions --region {REGION} "
        "--cluster-versions ...: aws: [ERROR]: An error occurred "
        "(InvalidParameterException) when calling the DescribeClusterVersions "
        "operation: Only one of the defaultOnly, clusterVersions, includeAll or "
        "status request parameters ...): the end-of-support check of 2 EKS "
        "cluster(s) did not run; it is not an authorization failure: read the "
        f"dates with aws eks describe-cluster-versions --region {REGION} "
        f"--cluster-versions 1.31 1.33 and run status --full again {SECTION}"
    ]


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


def _versions_failure_line(error: str) -> str:
    def failing(_region: str, _versions: Any) -> dict[str, Any]:
        raise ReleaseError(error)

    block = _block(
        _clusters(cpu={"Version": "1.33"}, **{"gpu-eks-cluster": {"Version": "1.31"}}),
        read_versions=failing,
    )
    (line,) = support.eks_support_lines(block)
    return line


@pytest.mark.parametrize(
    "error",
    [
        "aws: [ERROR]: An error occurred (AccessDeniedException) when calling the "
        "DescribeClusterVersions operation: User: "
        "arn:aws:sts::123456789012:assumed-role/operator/session is not authorized "
        "to perform: eks:DescribeClusterVersions",
        "aws: [ERROR]: An error occurred (UnauthorizedOperation) when calling the "
        "DescribeClusterVersions operation",
        "User: arn:aws:sts::123456789012:assumed-role/operator/session is not "
        "authorized to perform: eks:DescribeClusterVersions",
    ],
    ids=["access-denied", "unauthorized-operation", "not-authorized"],
)
def test_an_authorization_failure_names_the_permission(error: str) -> None:
    assert _versions_failure_line(error) == (
        f"advisory: could not read the EKS support dates in {REGION} ({error}): the "
        "end-of-support check of 2 EKS cluster(s) did not run; the operator identity "
        f"needs eks:DescribeClusterVersions {SECTION}"
    )


@pytest.mark.parametrize(
    "error",
    [
        "aws: [ERROR]: An error occurred (InvalidParameterException) when calling "
        "the DescribeClusterVersions operation: Only one of the defaultOnly, "
        "clusterVersions, includeAll or status request parameters is accepted at "
        "a time.",
        "aws: [ERROR]: An error occurred (ExpiredTokenException) when calling the "
        "DescribeClusterVersions operation: The security token included in the "
        "request is expired",
        "command exceeded its deployment time budget: aws eks "
        "describe-cluster-versions --region us-west-2 --cluster-versions ...",
    ],
    ids=["invalid-parameter", "expired-token", "time-budget"],
)
def test_any_other_failure_states_the_error_and_claims_no_permission(
    error: str,
) -> None:
    line = _versions_failure_line(error)

    assert line == (
        f"advisory: could not read the EKS support dates in {REGION} ({error}): the "
        "end-of-support check of 2 EKS cluster(s) did not run; it is not an "
        "authorization failure: read the dates with aws eks describe-cluster-versions "
        f"--region {REGION} --cluster-versions 1.31 1.33 and run status --full "
        f"again {SECTION}"
    )
    assert "eks:DescribeClusterVersions" not in line and "needs" not in line, line


def test_a_versions_error_read_back_without_its_clusters_is_still_one_line() -> None:
    assert support.eks_support_lines({"versions_error": {REGION: "boom"}}) == [
        f"advisory: could not read the EKS support dates in {REGION} (boom): the "
        "end-of-support check of 0 EKS cluster(s) did not run; it is not an "
        "authorization failure: read the dates with aws eks describe-cluster-versions "
        f"--region {REGION} --cluster-versions <version> and run status --full again "
        f"{SECTION}"
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
    assert HP_EKS_ARN.endswith(block["clusters"][1]["eks_cluster"]), (
        "the HyperPod GPU cluster is read through the EKS cluster it is orchestrated by"
    )
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


def test_the_versions_request_cites_the_api_reference_and_the_live_refusal() -> None:
    """Fetched and probed 2026-11-02: the EKS API reference's texts of the two
    selectors -- the page states no rule for sending both -- and the service's
    refusal of both in one request, which the fake answers with."""

    assert support.DESCRIBE_CLUSTER_VERSIONS_EVIDENCE == {
        "vendor": "AWS",
        "url": (
            "https://docs.aws.amazon.com/eks/latest/APIReference/"
            "API_DescribeClusterVersions.html"
        ),
        "page": "DescribeClusterVersions",
        "section": "URI Request Parameters",
        "version": "API version 2017-11-01",
        "docs_date": "Last-Modified 2026-11-02",
        "fetched": "2026-11-02",
        "quotes": (
            "List of specific cluster versions to describe.",
            "Include all available versions in the response.",
        ),
    }
    refusal = support.VERSIONS_SERVICE_REFUSAL
    assert refusal == {
        "at": "2026-11-02T14:21:23Z",
        "request": "--cluster-versions 1.31 --include-all",
        "error": "InvalidParameterException",
        "text": (
            "Only one of the defaultOnly, clusterVersions, includeAll or status "
            "request parameters is accepted at a time."
        ),
    }
    assert VERSIONS_REFUSAL.endswith(
        f"({refusal['error']}) when calling the DescribeClusterVersions "
        f"operation: {refusal['text']}"
    ), "the fake refuses in the service's words"
