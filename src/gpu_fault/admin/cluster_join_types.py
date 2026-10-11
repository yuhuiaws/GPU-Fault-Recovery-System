"""Shared join data without a dependency on the lifecycle driver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import ClusterIdentity
from gpu_fault.admin.site import RenderedSite

DEFAULT_ALLOWED_NAMESPACES = ("gpu-fault-system", "training")


def site_allowed_namespaces(site: RenderedSite) -> tuple[str, ...]:
    """The allowed workload namespaces a joining cluster inherits.

    The list every existing GPU cluster of the site shares; the historical
    default when the site has no cluster yet or its clusters disagree (then
    the operator names the list with ``--allowed-namespace``).
    """

    lists = {
        tuple(sorted(str(item) for item in cluster.get("allowed_namespaces") or ()))
        for cluster in site.release_config.get("clusters") or ()
    }
    lists.discard(())
    if len(lists) == 1:
        return next(iter(lists))
    return DEFAULT_ALLOWED_NAMESPACES


@dataclass(frozen=True)
class JoinClusterRequest:
    site: RenderedSite
    gpu_cluster_arn: str
    cluster_id: str | None = None
    allowed_namespaces: tuple[str, ...] = DEFAULT_ALLOWED_NAMESPACES
    state_dir: Path | None = None
    #: ``--reclaim-cluster-token``: re-attach a cluster whose site membership
    #: was lost by adopting its existing token file, on proof
    #: (``cluster_join_reclaim``). Without it an orphaned token is refused.
    reclaim_cluster_token: bool = False


@dataclass(frozen=True)
class JoinInputs:
    target: ClusterIdentity
    cluster_id: str
    discovery: dict[str, Any]
    local: dict[str, Any]


@dataclass(frozen=True)
class JoinExecution:
    target: ClusterIdentity
    cluster_id: str
    discovery: dict[str, Any]
    local: dict[str, Any]
    prerequisites: dict[str, Any]
    candidate: RenderedSite


@dataclass
class JoinAttempt:
    request: JoinClusterRequest
    state_dir: Path
    state_path: Path
    state: dict[str, Any]
    execution: JoinExecution | None = None
    inputs: JoinInputs | None = None
