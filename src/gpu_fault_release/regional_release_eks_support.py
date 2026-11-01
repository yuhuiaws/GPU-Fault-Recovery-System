"""EKS end-of-support control-plane upgrades: recorded and announced, never refused.

Every EKS cluster of a site -- the CPU control-plane cluster, the EKS cluster a
HyperPod GPU cluster is orchestrated by, a plain EKS GPU cluster -- runs a
Kubernetes version whose support ends on a date: the end of extended support
under the ``EXTENDED`` upgrade policy (AWS's default for new and existing
clusters), the end of standard support under ``STANDARD``. After that date AWS
upgrades the cluster's control plane to the next version by itself, at any
time and without notice; the upgrade covers the control plane only, and a
cluster upgraded that way cannot be rolled back (``EKS_SUPPORT_EVIDENCE``).
Live 2026-11-01: the plain EKS GPU cluster ``gpu-eks-cluster`` ran 1.31 under
``EXTENDED``, whose extended support ends 2026-11-26; the other two ran 1.33.

The ``regional_contexts`` check (``status --full``, deploy preflight) reads
each cluster's ``version`` and ``upgradePolicy.supportType`` (``eks
describe-cluster``) and the support dates of those versions (one ``eks
describe-cluster-versions`` per region), and records them under
``details.eks_support``. The status header prints an advisory for each cluster
within ``NOTICE_DAYS`` of its date, or past it. A read that fails is recorded
with its reason and announced as "could not read"; the check's verdict, the
exit code and ``healthy`` are never changed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from gpu_fault_release.regional_release_state import aws_json

__all__ = [
    "CLUSTER_QUERY",
    "EKS_SUPPORT_EVIDENCE",
    "FIELD",
    "NOTICE_DAYS",
    "VERSIONS_QUERY",
    "EksCluster",
    "eks_support_block",
    "eks_support_lines",
    "read_eks_support",
    "site_eks_clusters",
    "utc_now",
]

#: The ``regional_contexts`` facts key of the block.
FIELD = "eks_support"
#: An advisory starts this many days before a cluster's support end date.
NOTICE_DAYS = 90
UNKNOWN = "unknown"
SECTION = "administrator-operations §5.0.5"
SUPPORT_TYPES = ("STANDARD", "EXTENDED")
#: AWS: "By default, for all new and existing clusters, the upgrade policy is
#: set to EXTENDED, unless specified otherwise." (same page, extended support
#: FAQs) -- what a describe without ``upgradePolicy`` means.
DEFAULT_SUPPORT_TYPE = "EXTENDED"
#: ``describe-cluster`` reduced to the two fields read.
CLUSTER_QUERY = (
    "{Version:cluster.version,SupportType:cluster.upgradePolicy.supportType}"
)
#: ``describe-cluster-versions`` reduced to the dates; an object, so the
#: answer is still one JSON object.
VERSIONS_QUERY = (
    "{ClusterVersions:clusterVersions[].{Version:clusterVersion,"
    "EndOfStandardSupportDate:endOfStandardSupportDate,"
    "EndOfExtendedSupportDate:endOfExtendedSupportDate}}"
)
#: What AWS does at the end of a version's support, and why it matters.
EKS_SUPPORT_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": "https://docs.aws.amazon.com/eks/latest/userguide/kubernetes-versions.html",
    "page": "Understand the Kubernetes version lifecycle on EKS",
    "sections": ("Amazon EKS extended support FAQs", "Version rollback"),
    "quotes": (
        "Automatic updates can happen at any time after the end of extended "
        "support date. You won’t receive any notification before the update.",
        "The auto-upgrade includes only the Kubernetes control plane.",
        "Clusters that were automatically upgraded at the end of extended support "
        "cannot be rolled back.",
        "Note: If you disable extended support, your cluster will be auto upgraded "
        "at the end of standard support.",
    ),
}
_MAX_REASON = 300


@dataclass(frozen=True)
class EksCluster:
    """One EKS cluster of the site, as the advisory names it."""

    label: str
    arn: str
    region: str

    @property
    def name(self) -> str:
        return self.arn.rsplit("/", 1)[-1]


def site_eks_clusters(config: Any) -> list[EksCluster]:
    """The CPU control-plane cluster's EKS cluster, then each GPU cluster's."""

    return [
        EksCluster(
            "the CPU control-plane cluster", config.cpu_eks_arn, config.aws_region
        ),
        *(
            EksCluster(
                f"GPU cluster {target.cluster_id}",
                target.eks_cluster_arn,
                target.region,
            )
            for target in config.clusters
        ),
    ]


def _reason(exc: BaseException) -> str:
    text = re.sub(r"\s+", " ", str(exc)).strip() or type(exc).__name__
    return text if len(text) <= _MAX_REASON else text[: _MAX_REASON - 3] + "..."


def _instant(value: object) -> datetime | None:
    """An ISO-8601 timestamp of the CLI's output as an aware UTC datetime."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cluster_entry(
    cluster: EksCluster, read_cluster: Callable[[EksCluster], Mapping[str, Any]]
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "cluster": cluster.label,
        "eks_cluster": cluster.name,
        "region": cluster.region,
    }
    try:
        value = read_cluster(cluster)
        version = value.get("Version") if isinstance(value, Mapping) else None
        if not isinstance(version, str) or not version:
            raise ValueError("eks describe-cluster returned no version")
        support = value.get("SupportType")
        entry["version"] = version
        entry["support_type"] = (
            support if support in SUPPORT_TYPES else DEFAULT_SUPPORT_TYPE
        )
        if support not in SUPPORT_TYPES:
            entry["support_type_default"] = True
    except Exception as exc:  # noqa: BLE001 - advisory: recorded, never raised
        entry.update(state=UNKNOWN, error=_reason(exc))
    return entry


def eks_support_block(
    clusters: Sequence[EksCluster],
    *,
    read_cluster: Callable[[EksCluster], Mapping[str, Any]],
    read_versions: Callable[[str, Sequence[str]], Mapping[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    """``{"notice_days", "clusters"[, "versions_error"]}`` for ``clusters``.

    Per cluster ``version``, ``support_type`` (``support_type_default`` when
    the describe carries none), ``end_of_support`` (the UTC date the policy's
    support ends), ``days_left`` (whole days from ``now``, negative once it has
    passed) and ``state`` ``read``; ``state`` ``unknown`` with ``error`` when a
    read failed. ``versions_error`` maps a region whose
    ``describe-cluster-versions`` failed to the reason. Never raises.
    """

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(clusters)))) as executor:
        entries = list(
            executor.map(
                lambda cluster: _cluster_entry(cluster, read_cluster), clusters
            )
        )
    dates: dict[tuple[str, str], tuple[datetime | None, datetime | None]] = {}
    errors: dict[str, str] = {}
    for region in sorted({entry["region"] for entry in entries if "version" in entry}):
        versions = sorted(
            {
                entry["version"]
                for entry in entries
                if entry["region"] == region and "version" in entry
            }
        )
        try:
            response = read_versions(region, versions)
            for item in response.get("ClusterVersions") or ():
                if isinstance(item, Mapping) and isinstance(item.get("Version"), str):
                    dates[(region, item["Version"])] = (
                        _instant(item.get("EndOfStandardSupportDate")),
                        _instant(item.get("EndOfExtendedSupportDate")),
                    )
        except Exception as exc:  # noqa: BLE001 - advisory: recorded, never raised
            errors[region] = _reason(exc)
    for entry in entries:
        if "version" not in entry:
            continue
        if entry["region"] in errors:
            entry.update(state=UNKNOWN, error=errors[entry["region"]])
            continue
        standard, extended = dates.get(
            (entry["region"], entry["version"]), (None, None)
        )
        end = extended if entry["support_type"] == "EXTENDED" else standard
        if end is None:
            entry.update(
                state=UNKNOWN,
                error=(
                    "eks describe-cluster-versions returned no end of "
                    f"{entry['support_type'].lower()} support date for "
                    f"Kubernetes {entry['version']}"
                ),
            )
            continue
        entry["end_of_support"] = end.date().isoformat()
        entry["days_left"] = math.floor((end - now).total_seconds() / 86400)
        entry["state"] = "read"
    block: dict[str, Any] = {"notice_days": NOTICE_DAYS, "clusters": entries}
    if errors:
        block["versions_error"] = errors
    return block


def utc_now() -> datetime:
    """The clock the check judges ``days_left`` by (a seam for tests)."""

    return datetime.now(timezone.utc)


def read_eks_support(release: Any, *, now: datetime | None = None) -> dict[str, Any]:
    """``eks_support_block`` of the release's site, read through ``aws_json``:
    one ``describe-cluster`` per EKS cluster, one ``describe-cluster-versions``
    per region (``--include-all``: a version past its support is still
    listed)."""

    def read_cluster(cluster: EksCluster) -> Mapping[str, Any]:
        return aws_json(
            release,
            [
                "eks",
                "describe-cluster",
                "--region",
                cluster.region,
                "--name",
                cluster.name,
                "--query",
                CLUSTER_QUERY,
            ],
            region=False,
        )

    def read_versions(region: str, versions: Sequence[str]) -> Mapping[str, Any]:
        return aws_json(
            release,
            [
                "eks",
                "describe-cluster-versions",
                "--region",
                region,
                "--cluster-versions",
                *versions,
                "--include-all",
                "--query",
                VERSIONS_QUERY,
            ],
            region=False,
        )

    return eks_support_block(
        site_eks_clusters(release.config),
        read_cluster=read_cluster,
        read_versions=read_versions,
        now=now or utc_now(),
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _support_line(entry: Mapping[str, Any]) -> str:
    days = entry["days_left"]
    policy = f"the {entry.get('support_type')} upgrade policy" + (
        " (the default)" if entry.get("support_type_default") else ""
    )
    head = (
        f"advisory: EKS cluster {entry.get('eks_cluster')} ({entry.get('cluster')}) "
        f"runs Kubernetes {entry.get('version')} under {policy}, whose support "
    )
    consequence = (
        f"-- only the control plane: its nodes stay on {entry.get('version')} -- "
        "and a cluster upgraded that way cannot be rolled back"
    )
    maintenance = (
        "in a maintenance window (control plane first, then its nodes and add-ons) "
        f"({SECTION})"
    )
    if days >= 0:
        return (
            head + f"ends {entry.get('end_of_support')} (in {days} day(s)): after "
            "that date AWS upgrades its control plane to the next version by itself, "
            f"at any time and without notice {consequence}; upgrade it yourself "
            f"before then, {maintenance}"
        )
    return (
        head + f"ended {entry.get('end_of_support')} ({-days} day(s) ago): AWS may "
        "upgrade its control plane to the next version by itself at any time and "
        f"without notice {consequence}; upgrade it yourself now, {maintenance}"
    )


def eks_support_lines(block: object) -> list[str]:
    """The advisory lines of the block: a cluster within ``notice_days`` of
    its support end (or past it), a region whose support dates could not be
    read, a cluster whose version or dates could not be read."""

    if not isinstance(block, Mapping):
        return []
    entries = [
        item for item in block.get("clusters") or () if isinstance(item, Mapping)
    ]
    errors = _mapping(block.get("versions_error"))
    notice = block.get("notice_days")
    notice = notice if type(notice) is int else NOTICE_DAYS
    lines = []
    for region, error in sorted(errors.items()):
        count = sum(
            1 for item in entries if item.get("region") == region and "version" in item
        )
        lines.append(
            f"advisory: could not read the EKS support dates in {region} ({error}): "
            f"the end-of-support check of {count} EKS cluster(s) did not run; the "
            f"operator identity needs eks:DescribeClusterVersions ({SECTION})"
        )
    for entry in entries:
        name = f"EKS cluster {entry.get('eks_cluster')} ({entry.get('cluster')})"
        if entry.get("state") == "read":
            days = entry.get("days_left")
            if type(days) is int and days <= notice:
                lines.append(_support_line(entry))
        elif "version" not in entry:
            lines.append(
                f"advisory: could not read the Kubernetes version of {name} "
                f"({entry.get('error')}): its end-of-support check did not run "
                f"({SECTION})"
            )
        elif entry.get("region") not in errors:
            lines.append(
                f"advisory: could not read the support dates of {name}, Kubernetes "
                f"{entry.get('version')} ({entry.get('error')}): its end-of-support "
                f"check did not run ({SECTION})"
            )
    return lines
