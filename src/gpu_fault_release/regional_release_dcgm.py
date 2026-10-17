"""The DCGM host engine mode of the GPU clusters, as the release engine sees it.

An ``eks-managed`` cluster whose node groups declare a host DCGM
(``cluster_scope.DCGM_HOSTENGINE_LABEL=host``) runs ``nvidia-dcgm`` on every
node: the exporter dials it, the Node Agents grade ``dcgmi diag`` against it.
Every other plain EKS cluster runs the exporter's embedded hostengine, and a
HyperPod cluster is always ``host``. The mode is a site.yaml decision deploy
and join re-derive from the node groups, so the release has to notice when it
flips and roll what depends on it:

* the diff input ``dcgm_hostengines`` (``dcgm_hostengines_digest``, recorded
  as ``dcgm_hostengines_sha256``) selects the exporter, the node installer
  reconciler and the Agent waves; only clusters in host mode are digested, so
  a site that never declared one keeps the digest of no cluster at all;
* a rollback re-renders the exporter and the reconciler for the mode the
  previous release ran, read from the live DaemonSet's
  ``gpu-fault.io/dcgm-hostengine`` annotation when the previous identity was
  captured (``captured_dcgm_hostengine``);
* the node preflight classifies each eks node's DCGM state (its JSON verdict's
  ``dcgm`` object) and the release keeps the last observation per cluster
  (``node_dcgm_hostengine``) for ``status --full`` and verify.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping

from gpu_fault.cluster_scope import (
    ClusterProvider,
    DcgmHostEngine,
    cluster_provider,
    dcgm_hostengine,
    dcgm_hostengine_source,
)
from gpu_fault.node_installer_rendering import INSTALLER_DCGM_HOSTENGINE_ANNOTATION

__all__ = [
    "DCGM_HOSTENGINES_INPUT",
    "DCGM_HOSTENGINES_STATE_KEY",
    "NODE_DCGM_STATE_KEY",
    "capture_exporter_identity",
    "captured_dcgm_hostengine",
    "cluster_dcgm_hostengines",
    "daemonset_dcgm_hostengine",
    "dcgm_hostengines_digest",
    "forward_dcgm_hostengine",
    "installed_dcgm_hostengine_matches",
    "node_dcgm_details",
    "record_node_dcgm_states",
    "recorded_dcgm_hostengines_digest",
]

#: The diff input and the state key that records its value at commit.
DCGM_HOSTENGINES_INPUT = "dcgm_hostengines"
DCGM_HOSTENGINES_STATE_KEY = "dcgm_hostengines_sha256"
#: Release state key: ``{cluster_id: {mode, observed_at, nodes}}`` -- the DCGM
#: state the last node preflight observed on every node of the cluster.
NODE_DCGM_STATE_KEY = "node_dcgm_hostengine"
#: The exporter DaemonSet's mark of its eks variant (``embedded`` / ``host``);
#: the same key as the node-group label.
DAEMONSET_ANNOTATION = "gpu-fault.io/dcgm-hostengine"
#: The one node DCGM state (the preflight's ``dcgm.state``) a host-mode
#: cluster accepts on every node.
HOST_READY_STATE = "host-active"
#: ``regional_release_state.STATE_CONFIG_MAP``, spelled here because that
#: module imports this one (a test keeps the two equal).
RELEASE_STATE_CONFIG_MAP = "gpu-fault-regional-release-state"


def cluster_dcgm_hostengines(config: Any) -> dict[str, str]:
    """``{cluster_id: "host"}`` for every ``eks-managed`` cluster in host mode."""

    return {
        str(target.cluster_id): DcgmHostEngine.HOST.value
        for target in getattr(config, "clusters", ())
        if cluster_provider(getattr(target, "provider", None))
        is ClusterProvider.EKS_MANAGED
        and dcgm_hostengine(target) is DcgmHostEngine.HOST
    }


def _digest(value: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dcgm_hostengines_digest(config: Any) -> str:
    """The ``dcgm_hostengines`` release input of ``config``."""

    return _digest(cluster_dcgm_hostengines(config))


def recorded_dcgm_hostengines_digest(state: Mapping[str, Any]) -> str:
    """What a recorded state rolled: a state written before the input existed
    rolled no host-mode cluster (the mode did not exist yet)."""

    return str(state.get(DCGM_HOSTENGINES_STATE_KEY) or _digest({}))


def daemonset_dcgm_hostengine(daemonset: Mapping[str, Any] | None) -> str | None:
    """The eks variant a live exporter DaemonSet runs, or ``None`` (HyperPod
    text, or no DaemonSet)."""

    metadata = (daemonset or {}).get("metadata") or {}
    value = str((metadata.get("annotations") or {}).get(DAEMONSET_ANNOTATION) or "")
    return value if value in {item.value for item in DcgmHostEngine} else None


def capture_exporter_identity(release: Any, target: Any) -> dict[str, Any]:
    """The live exporter's image -- and on ``eks-managed`` its mode -- for the
    previous-identity snapshot a rollback restores."""

    daemonset = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "daemonset",
            "gpu-fault-dcgm-exporter",
        )
    )
    identity: dict[str, Any] = {
        "dcgm_image": (
            daemonset.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [{}])[0]
            .get("image")
        )
    }
    if cluster_provider(getattr(target, "provider", None)) is (
        ClusterProvider.EKS_MANAGED
    ):
        identity["dcgm_hostengine"] = daemonset_dcgm_hostengine(daemonset)
    return identity


def captured_dcgm_hostengine(previous: object, target: Any) -> str | None:
    """The mode the previous release ran on ``target``, when it was captured.

    ``None`` for HyperPod (always host), and for a previous identity captured
    before the mode was: such a release predates the mode altogether, its
    reconciler ignores the variable and its exporter was the embedded one.
    """

    if cluster_provider(getattr(target, "provider", None)) is not (
        ClusterProvider.EKS_MANAGED
    ):
        return None
    clusters = previous.get("clusters") if isinstance(previous, Mapping) else None
    cluster = clusters.get(target.cluster_id) if isinstance(clusters, Mapping) else None
    if not isinstance(cluster, Mapping) or "dcgm_hostengine" not in cluster:
        return None
    return str(cluster.get("dcgm_hostengine") or DcgmHostEngine.EMBEDDED.value)


def installed_dcgm_hostengine_matches(
    node: Mapping[str, Any], expected: str | None
) -> bool:
    """Whether ``node``'s Agent was installed in ``expected`` mode.

    The reconciler stamps ``INSTALLER_DCGM_HOSTENGINE_ANNOTATION=host`` on an
    eks node installed in host mode and removes it otherwise; ``None`` (a
    rollback, a HyperPod cluster) does not judge the annotation at all.
    """

    if expected is None:
        return True
    annotations = (node.get("metadata") or {}).get("annotations") or {}
    installed = str(annotations.get(INSTALLER_DCGM_HOSTENGINE_ANNOTATION) or "")
    return installed == (
        DcgmHostEngine.HOST.value if expected == DcgmHostEngine.HOST.value else ""
    )


def forward_dcgm_hostengine(target: Any, phase: str) -> str | None:
    """The DCGM mode a node-runtime wait of ``phase`` holds the Agents to.

    The cluster's own on ``eks-managed``; ``None`` for HyperPod and for a
    rollback, whose reconciler may predate the annotation altogether.
    """

    if phase == "rollback" or cluster_provider(
        getattr(target, "provider", None)
    ) is not (ClusterProvider.EKS_MANAGED):
        return None
    return dcgm_hostengine(target).value


def record_node_dcgm_states(
    release: Any,
    target: Any,
    observed: object,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Keep what the node preflight observed on ``target``'s nodes.

    ``observed`` is the preflight summary's ``dcgm_hostengine`` map
    (``{node: dcgm object}``); only each node's ``state`` and ``conflict`` are
    kept, so the release state stays small on a large fleet (the full object is
    in the preflight Job's verdict). Nothing is recorded for a cluster whose
    preflight reported none (an older script).
    """

    state = getattr(release, "state", None)
    if not isinstance(state, dict) or not isinstance(observed, Mapping):
        return None
    nodes = {
        str(name): {
            "state": str(value.get("state") or "unknown"),
            "conflict": value.get("conflict") is True,
        }
        for name, value in sorted(observed.items())
        if isinstance(value, Mapping)
    }
    if not nodes:
        return None
    record = {
        "mode": dcgm_hostengine(target).value,
        "observed_at": (now or datetime.now(UTC)).isoformat(),
        "nodes": nodes,
    }
    recorded = state.get(NODE_DCGM_STATE_KEY)
    clusters = dict(recorded) if isinstance(recorded, Mapping) else {}
    clusters[str(target.cluster_id)] = record
    state[NODE_DCGM_STATE_KEY] = clusters
    return record


def _node_names(nodes: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    items: Iterable[Any] = nodes.get("items") or []
    return sorted(
        (str((item.get("metadata") or {}).get("name") or ""), item)
        for item in items
        if isinstance(item, Mapping)
    )


def _recorded_node_states(release: Any) -> object:
    """The release state's node DCGM observations, without replacing the
    in-memory state: a verify inside a release reads its own dict, a status
    read the persisted ConfigMap (through the snapshot-cached reader)."""

    state = getattr(release, "state", None)
    if isinstance(state, dict) and NODE_DCGM_STATE_KEY in state:
        return state[NODE_DCGM_STATE_KEY]
    try:
        value = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "configmap",
                RELEASE_STATE_CONFIG_MAP,
            )
        )
        persisted = json.loads((value.get("data") or {}).get("state.json") or "{}")
    except Exception:  # noqa: BLE001 - an unreadable state shows no observation
        return None
    return persisted.get(NODE_DCGM_STATE_KEY) if isinstance(persisted, dict) else None


def node_dcgm_details(
    release: Any, target: Any, nodes: Mapping[str, Any]
) -> dict[str, Any]:
    """The ``node_scope`` DCGM view of ``status --full`` and verify.

    The cluster's mode and source; on ``eks-managed`` also the nodes whose
    Agent still runs the other mode (``dcgm_hostengine_pending_nodes``, the
    reconciler annotation). When a node preflight observed them -- on either
    provider: a HyperPod node is always held to its host engine -- every
    node's DCGM state with the time of that observation and the nodes whose
    state conflicts with the mode (a host engine next to the embedded one, or
    a host-mode node without a working ``nvidia-dcgm``).
    """

    mode = dcgm_hostengine(target)
    details: dict[str, Any] = {
        "dcgm_hostengine": mode.value,
        "dcgm_hostengine_source": dcgm_hostengine_source(target).value,
    }
    if cluster_provider(getattr(target, "provider", None)) is (
        ClusterProvider.EKS_MANAGED
    ):
        details["dcgm_hostengine_pending_nodes"] = [
            name
            for name, item in _node_names(nodes)
            if name and not installed_dcgm_hostengine_matches(item, mode.value)
        ]
    recorded = _recorded_node_states(release)
    cluster = (
        recorded.get(str(target.cluster_id)) if isinstance(recorded, Mapping) else None
    )
    if not isinstance(cluster, Mapping):
        return details
    nodes_value = cluster.get("nodes")
    observed: dict[str, Mapping[str, Any]] = {
        str(name): value if isinstance(value, Mapping) else {}
        for name, value in (
            nodes_value.items() if isinstance(nodes_value, Mapping) else ()
        )
    }
    details["dcgm_hostengine_observed_at"] = cluster.get("observed_at")
    details["dcgm_hostengine_nodes"] = {
        name: str(value.get("state") or "unknown")
        for name, value in sorted(observed.items())
    }
    details["dcgm_hostengine_conflict_nodes"] = sorted(
        name
        for name, value in observed.items()
        if value.get("conflict") is True
        or (mode is DcgmHostEngine.HOST and value.get("state") != HOST_READY_STATE)
    )
    return details
