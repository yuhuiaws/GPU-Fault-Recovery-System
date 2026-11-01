"""The vendor evidence behind batch 231's advice, pinned in code and in both manuals.

Project rule: advice that rests on vendor documentation carries the exact URL,
the section, the version where the vendor versions its documentation, and a
verbatim quote -- in the deciding code, in the CN and EN operations manuals,
and in a test that keeps the two in step. The quotes are checked against the
manuals with whitespace collapsed (the manuals wrap lines); the typographic
apostrophes are the vendors' own.

- HyperPod: what a platform software update does to the nodes it updates
  (``automatic_node_updates.HYPERPOD_UPDATE_EVIDENCE``);
- Kueue v0.18.0: why its Pod webhook covers the product namespace
  (``regional_release_replica_failure.KUEUE_POD_WEBHOOK_EVIDENCE``);
- Kubernetes v1.33.13: the ReplicaSet controller's backoff and its sync on
  every update (``REPLICASET_BACKOFF_EVIDENCE``);
- EKS: the control-plane upgrade at the end of a version's support
  (``regional_release_eks_support.EKS_SUPPORT_EVIDENCE``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.automatic_node_updates import HYPERPOD_UPDATE_EVIDENCE
from gpu_fault_release.regional_release_eks_support import EKS_SUPPORT_EVIDENCE
from gpu_fault_release.regional_release_replica_failure import (
    KUEUE_POD_WEBHOOK_EVIDENCE,
    REPLICASET_BACKOFF_EVIDENCE,
    RETRY_ANNOTATION,
)

ROOT = Path(__file__).resolve().parents[2]
MANUALS = ("docs/管理员日常运维.md", "docs/en/administrator-operations.md")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _needles(evidence: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("url", "page", "section", "sections", "sources", "quotes"):
        value = evidence.get(key)
        values.extend(value if isinstance(value, tuple) else [value] if value else [])
    if evidence.get("version") and evidence.get("vendor") == "Kueue":
        values.append(f"Kueue {evidence['version']}")
    return values


def test_the_hyperpod_update_evidence_is_the_aws_page_verbatim() -> None:
    assert HYPERPOD_UPDATE_EVIDENCE == {
        "vendor": "AWS",
        "url": (
            "https://docs.aws.amazon.com/sagemaker/latest/dg/"
            "sagemaker-hyperpod-eks-operate-cli-command-update-cluster-software.html"
        ),
        "page": "Updating the SageMaker HyperPod platform software",
        "section": "Important",
        "quotes": (
            "When this API is called, SageMaker HyperPod doesn’t drain or "
            "redistribute the jobs (Pods) running on the nodes.",
            "The patching process replaces the root volume with the updated AMI, "
            "which means that your previous data stored in the instance root volume "
            "will be lost.",
            "All cluster nodes experience downtime (nodes appear as `<NotReady>` in "
            "the output of `kubectl get node`) while the patching is in progress.",
        ),
    }


@pytest.mark.parametrize("manual", MANUALS)
@pytest.mark.parametrize(
    "evidence",
    [
        HYPERPOD_UPDATE_EVIDENCE,
        KUEUE_POD_WEBHOOK_EVIDENCE,
        REPLICASET_BACKOFF_EVIDENCE,
        EKS_SUPPORT_EVIDENCE,
    ],
    ids=["hyperpod-update", "kueue-pod-webhook", "replicaset-backoff", "eks-support"],
)
def test_each_manual_carries_each_citation_verbatim(
    manual: str, evidence: dict[str, Any]
) -> None:
    text = _flat((ROOT / manual).read_text(encoding="utf-8"))

    for needle in _needles(evidence):
        assert _flat(needle) in text, f"{manual} lacks {needle!r}"


@pytest.mark.parametrize("manual", MANUALS)
def test_each_manual_names_the_remedies_the_product_prints(manual: str) -> None:
    text = _flat((ROOT / manual).read_text(encoding="utf-8"))

    for needle in (
        RETRY_ANNOTATION,
        "hyperpod-scheduled-update --state-dir",
        "--rolling",
        "eks:DescribeClusterVersions",
        "additionalProperties: false",
    ):
        assert needle in text, f"{manual} lacks {needle!r}"
