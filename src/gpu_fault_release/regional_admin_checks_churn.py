"""What a release's own node churn excuses in its ``verify``, and nothing else.

The driver verifies a release right after its deploy and rolls back on any
FAIL, so a node the release itself saw leave or arrive must not read as a
broken release:

* a Ready node outside the node set the release captured whose installer has
  not succeeded yet -- capacity that joined while the release ran, which the
  steady-state Reconciler is installing -- is left out of the installer check,
  the Agent coverage and the fleet readiness question, and named in a ``WARN``
  (``joined_nodes_installing``);
* an Agent of a node this release proved departed (``departed_nodes`` in the
  release state) that has no Kubernetes Node any more is not coverage, not a
  Runtime Profile to compare and not a collector to wait for, until the
  departed-node cleanup retires its record.

``status`` judges the fleet as it is and gets none of this: a node still
installing is a real ``installer is not Succeeded`` there, as before. And a
newcomer the installer did finish is judged like every node -- installed with
any artifact but the release's, it is ``artifact drift``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_release_departed_nodes import recorded_departed_nodes
from gpu_fault_release.regional_release_joined_nodes import installing_newcomers

#: Per-cluster key of the control API report, written in ``verify`` only.
RELEASE_CHURN_KEY = "release_churn"


def verify_joining(
    release: Any,
    cluster_id: str,
    ready_items: Sequence[Mapping[str, Any]],
    *,
    verify: bool,
) -> dict[str, str]:
    """``node -> installer state`` of the newcomers a ``verify`` leaves out."""

    return installing_newcomers(release, cluster_id, ready_items) if verify else {}


def annotate_release_churn(
    report: dict[str, Any],
    release: Any,
    joining_by_cluster: Mapping[str, Mapping[str, str]],
) -> None:
    """Write what this release's churn excuses into the report, per cluster."""

    clusters = report.get("clusters")
    state = getattr(release, "state", None)
    for cluster_id, cluster in clusters.items() if isinstance(clusters, dict) else ():
        if not isinstance(cluster, dict):
            continue
        kubernetes = {str(item) for item in cluster.get("kubernetes_node_ids") or []}
        departed = sorted(
            set(
                recorded_departed_nodes(
                    state if isinstance(state, Mapping) else None, cluster_id
                )
            )
            - kubernetes
        )
        joining = dict(sorted((joining_by_cluster.get(cluster_id) or {}).items()))
        if departed or joining:
            cluster[RELEASE_CHURN_KEY] = {
                "departed": departed,
                "joined_installing": joining,
            }


def churn_departed(cluster: Mapping[str, Any]) -> frozenset[str]:
    churn = cluster.get(RELEASE_CHURN_KEY)
    return frozenset(
        str(node)
        for node in (churn.get("departed") if isinstance(churn, dict) else ()) or ()
    )


def churn_excused(cluster: Mapping[str, Any]) -> frozenset[str]:
    """The nodes whose Agents this verify does not count, for one cluster."""

    churn = cluster.get(RELEASE_CHURN_KEY)
    joining = churn.get("joined_installing") if isinstance(churn, dict) else None
    return churn_departed(cluster) | frozenset(
        str(node) for node in (joining if isinstance(joining, dict) else {})
    )


def with_joining(
    value: CheckValue, cluster_id: str, joining: Mapping[str, str]
) -> CheckValue:
    """The ``gpu_cluster`` verdict, a ``WARN`` naming the newcomers left out."""

    if not joining:
        return value
    details = dict(value.details or {})
    details["joined_nodes_installing"] = dict(sorted(joining.items()))
    shown = ", ".join(f"{node} ({state})" for node, state in sorted(joining.items()))
    return replace(
        value,
        summary=(
            f"{value.summary}; {len(joining)} node(s) that joined during the release "
            f"are still being installed by the steady-state Reconciler: {shown}"
        ),
        details=details,
        status="WARN",
    )


def with_churn(value: CheckValue, report: Mapping[str, Any]) -> CheckValue:
    """The ``control_api`` verdict, a ``WARN`` when this release's churn was excused."""

    clusters = report.get("clusters")
    parts = []
    for cluster_id, cluster in sorted(
        (clusters if isinstance(clusters, dict) else {}).items()
    ):
        churn = cluster.get(RELEASE_CHURN_KEY) if isinstance(cluster, dict) else None
        if not isinstance(churn, dict):
            continue
        if churn.get("departed"):
            parts.append(
                f"{cluster_id} departed (proven by this release): "
                + ",".join(churn["departed"])
            )
        if churn.get("joined_installing"):
            parts.append(
                f"{cluster_id} joined, still installing: "
                + ",".join(sorted(churn["joined_installing"]))
            )
    if not parts:
        return value
    return replace(
        value,
        summary=f"{value.summary}; not counted: " + "; ".join(parts),
        status="WARN",
    )
