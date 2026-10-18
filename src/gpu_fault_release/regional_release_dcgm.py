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
  (``node_dcgm_hostengine``) for ``status --full`` and verify -- also when it
  refused the release, and across the ``sync-state`` re-commit that follows
  (``carried_node_dcgm_states``);
* the diff also probes what the data plane serves (``dcgm_hostengine_drift``):
  a live exporter whose mark names another mode than site.yaml records is
  ``dcgm_hostengine_drift``, planned like a flip. A recorded digest says what
  a release committed; after a refused flip was committed by a ``sync-state``
  from the attempt's own site (live 2026-10-17: exporter embedded, record and
  digest host, ``next deploy: NOOP``) only the exporter still told the truth.

The engine's version against the exporter's DCGM client library
(``gpu_fault.dcgm_versions``): the node preflight reads each running engine's
version and the release keeps it with the node's state; in host mode it also
keeps the client version of the release's exporter (``exporter_client_version``,
the DCGM half of its image tag) and names, with a ``WARN`` line, every node
whose engine is newer than that client -- NVIDIA's connection requirement.
Advisory: no node and no release is refused for it (see
``newer_hostengine_warning``).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping

from gpu_fault.cluster_scope import (
    DCGM_HOSTENGINE_ENV,
    ClusterProvider,
    DcgmHostEngine,
    cluster_provider,
    dcgm_hostengine,
    dcgm_hostengine_source,
)
from gpu_fault.dcgm_versions import (
    NVIDIA_EXPORTER_COMPATIBILITY_URL,
    DcgmVersionRelation,
    dcgm_version,
    dcgm_version_relation,
    exporter_tag_versions,
)
from gpu_fault.node_installer_rendering import INSTALLER_DCGM_HOSTENGINE_ANNOTATION
from gpu_fault_release.regional_deployment_inventory import GPU_RECONCILER_DEPLOYMENT
from gpu_fault_release.regional_release_config import ReleaseError

__all__ = [
    "DCGM_HOSTENGINES_INPUT",
    "DCGM_HOSTENGINES_STATE_KEY",
    "DCGM_HOSTENGINE_DRIFT_INPUT",
    "NODE_DCGM_STATE_KEY",
    "capture_exporter_identity",
    "captured_dcgm_hostengine",
    "carried_node_dcgm_states",
    "check_dcgm_hostengine_agreement",
    "cluster_dcgm_hostengines",
    "daemonset_dcgm_hostengine",
    "dcgm_hostengine_drift",
    "dcgm_hostengines_digest",
    "exporter_client_version",
    "exporter_served_mode",
    "live_dcgm_hostengines_digest",
    "reconciler_served_mode",
    "served_dcgm_hostengines",
    "sync_state_dcgm",
    "forward_dcgm_hostengine",
    "installed_dcgm_hostengine_matches",
    "newer_hostengine_warning",
    "node_dcgm_details",
    "record_node_dcgm_states",
    "recorded_dcgm_hostengines_digest",
]

#: The diff input and the state key that records its value at commit.
DCGM_HOSTENGINES_INPUT = "dcgm_hostengines"
DCGM_HOSTENGINES_STATE_KEY = "dcgm_hostengines_sha256"
#: The diff input of a live exporter serving another mode than site.yaml records.
DCGM_HOSTENGINE_DRIFT_INPUT = "dcgm_hostengine_drift"
EXPORTER_DAEMONSET = "gpu-fault-dcgm-exporter"
#: ``regional_release_rendering.REMOTE_HOSTENGINE_ENV``, spelled here because
#: that module imports this one (a test keeps the two equal): only the host
#: variant of the exporter dials the node's engine.
REMOTE_HOSTENGINE_ENV = "DCGM_REMOTE_HOSTENGINE_INFO"
#: What each live mode-dependent render is called in a refusal.
SERVED_RENDERS = {
    "exporter": f"exporter DaemonSet {EXPORTER_DAEMONSET}",
    "reconciler": f"node installer reconciler {GPU_RECONCILER_DEPLOYMENT}",
}
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


def _modes() -> frozenset[str]:
    return frozenset(item.value for item in DcgmHostEngine)


def exporter_served_mode(daemonset: Mapping[str, Any] | None) -> str | None:
    """The eks variant a live exporter DaemonSet serves; ``None`` without one.

    Its mark; an exporter rendered before the mark existed is told by its
    environment -- the host variant dials ``DCGM_REMOTE_HOSTENGINE_INFO``.
    """

    if not daemonset:
        return None
    marked = daemonset_dcgm_hostengine(daemonset)
    if marked is not None:
        return marked
    template = (daemonset.get("spec") or {}).get("template") or {}
    names = {
        str(item.get("name"))
        for container in (template.get("spec") or {}).get("containers") or ()
        if isinstance(container, Mapping)
        for item in container.get("env") or ()
        if isinstance(item, Mapping)
    }
    host = REMOTE_HOSTENGINE_ENV in names
    return (DcgmHostEngine.HOST if host else DcgmHostEngine.EMBEDDED).value


def reconciler_served_mode(deployment: Mapping[str, Any] | None) -> str | None:
    """The mode a live node installer reconciler installs Agents in.

    Its ``GPU_FAULT_DCGM_HOSTENGINE``; one rendered before the variable
    existed installs embedded Agents. ``None`` without the Deployment or its
    container, or for a value that is no mode.
    """

    template = ((deployment or {}).get("spec") or {}).get("template") or {}
    container = next(
        (
            item
            for item in (template.get("spec") or {}).get("containers") or ()
            if isinstance(item, Mapping) and item.get("name") == "reconciler"
        ),
        None,
    )
    if container is None:
        return None
    for item in container.get("env") or ():
        if isinstance(item, Mapping) and item.get("name") == DCGM_HOSTENGINE_ENV:
            value = str(item.get("value") or "")
            return value if value in _modes() else None
    return DcgmHostEngine.EMBEDDED.value


def _reads_clusters(release: Any) -> bool:
    """The gate of every classification-time probe (``node_scope_drift``):
    never a dry run, a release without a delivered control plane or a bare
    test double without a cluster reader."""

    runner = getattr(release, "runner", None)
    config = getattr(release, "config", None)
    return (
        runner is not None
        and not getattr(runner, "dry_run", False)
        and bool((getattr(config, "delivery_component_digests", None) or {}).get("cpu"))
        and callable(getattr(release, "_get_json", None))
        and callable(getattr(release, "_gpu", None))
    )


def _is_eks(target: Any) -> bool:
    return cluster_provider(getattr(target, "provider", None)) is (
        ClusterProvider.EKS_MANAGED
    )


def served_dcgm_hostengines(release: Any, target: Any) -> dict[str, str]:
    """``{"exporter": mode, "reconciler": mode}``: what ``target``'s live
    mode-dependent renders serve, for those that exist."""

    def read(kind: str, name: str) -> dict[str, Any]:
        value = release._get_json(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                kind,
                name,
                "--ignore-not-found",
            )
        )
        return value if isinstance(value, dict) else {}

    served: dict[str, str] = {}
    exporter = exporter_served_mode(read("daemonset", EXPORTER_DAEMONSET))
    if exporter is not None:
        served["exporter"] = exporter
    reconciler = reconciler_served_mode(read("deployment", GPU_RECONCILER_DEPLOYMENT))
    if reconciler is not None:
        served["reconciler"] = reconciler
    return served


def dcgm_hostengine_drift(release: Any) -> dict[str, dict[str, str]]:
    """Plain EKS clusters whose live renders serve another mode than recorded.

    ``{cluster_id: {"recorded": mode, "exporter": mode, "reconciler": mode}}``
    -- the exporter DaemonSet's variant and the reconciler's environment, the
    two per-cluster renders of the mode. A classification-time probe like
    ``node_scope_drift`` (the same gate, ``_reads_clusters``); a render that
    does not exist yet (a cluster not bootstrapped) is not drift, and HyperPod
    has no mode to serve. Live 2026-10-17 a refused flip's ``sync-state``
    had recorded the host digest; the next release re-installed host Agents
    and never re-rendered the embedded exporter -- only this probe sees that.
    """

    if not _reads_clusters(release):
        return {}
    drift: dict[str, dict[str, str]] = {}
    for target in getattr(release.config, "clusters", None) or ():
        if not _is_eks(target):
            continue
        recorded = dcgm_hostengine(target).value
        served = served_dcgm_hostengines(release, target)
        if any(value != recorded for value in served.values()):
            drift[str(target.cluster_id)] = {"recorded": recorded, **served}
    return drift


def check_dcgm_hostengine_agreement(release: Any, target: Any) -> dict[str, Any]:
    """The ``gpu_cluster:<id>`` view of the served mode; refuse a disagreement.

    ``status --full`` and verify compare what the live exporter and reconciler
    serve (``dcgm_hostengine_served``) with ``node_scope.dcgm_hostengine``, the
    mode site.yaml records -- a host-mode node engine next to the exporter's
    embedded one is exactly the conflict the mode exists to prevent. A
    disagreement fails the check, naming the cluster and each render; the
    next deploy rolls the recorded mode (``dcgm_hostengine_drift``).
    """

    if not _is_eks(target) or not callable(getattr(release, "_get_json", None)):
        return {}
    recorded = dcgm_hostengine(target).value
    served = served_dcgm_hostengines(release, target)
    wrong = {name: value for name, value in served.items() if value != recorded}
    if wrong:
        raise ReleaseError(
            f"{target.cluster_id} DCGM host engine disagreement: site.yaml records "
            f"{recorded}; "
            + "; ".join(
                f"the {SERVED_RENDERS[name]} serves {value}"
                for name, value in sorted(wrong.items())
            )
            + "; gpu-fault-admin deploy re-renders it for the recorded mode"
        )
    return {"dcgm_hostengine_served": served}


def live_dcgm_hostengines_digest(release: Any) -> str:
    """The ``dcgm_hostengines`` digest of what the exporters serve (``sync-state``).

    ``sync-state`` commits the site as running; the mode is the one input a
    refused release's bootstrap may already have rewritten, so the commit
    takes it from the live exporter, and from the record only for a cluster
    whose exporter cannot be read (none yet, a dry run, a test double).
    """

    probe = _reads_clusters(release)
    modes: dict[str, str] = {}
    for target in getattr(getattr(release, "config", None), "clusters", ()) or ():
        if not _is_eks(target):
            continue
        mode = dcgm_hostengine(target).value
        if probe:
            served = served_dcgm_hostengines(release, target).get("exporter")
            mode = served or mode
        if mode == DcgmHostEngine.HOST.value:
            modes[str(target.cluster_id)] = mode
    return _digest(modes)


def sync_state_dcgm(release: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(state to start from, state updates)`` of a ``sync-state`` re-commit:
    the node DCGM observations it keeps and, when the exporters serve another
    mode than the site records, the live mode digest instead of the record's
    (every other re-commit stays byte for byte what it was)."""

    live = live_dcgm_hostengines_digest(release)
    recorded = dcgm_hostengines_digest(getattr(release, "config", None))
    updates = {} if live == recorded else {DCGM_HOSTENGINES_STATE_KEY: live}
    return carried_node_dcgm_states(release), updates


def carried_node_dcgm_states(release: Any) -> dict[str, Any]:
    """What a ``sync-state`` re-commit keeps of the state it replaces.

    The commit is written from the site afresh; the last node preflight's
    per-node DCGM states are evidence about the nodes, not something the
    commit decides, and a refused release recorded them for ``status --full``.
    Kept for the clusters the site still manages; nothing when the state
    cannot be read (a site without one yet, a test double).
    """

    try:
        state = release._load_state()
    except Exception:  # noqa: BLE001 - nothing to carry from an unreadable state
        return {}
    recorded = state.get(NODE_DCGM_STATE_KEY) if isinstance(state, Mapping) else None
    managed = {
        str(target.cluster_id)
        for target in getattr(getattr(release, "config", None), "clusters", ())
    }
    kept = {
        str(name): value
        for name, value in (recorded.items() if isinstance(recorded, Mapping) else ())
        if str(name) in managed
    }
    return {NODE_DCGM_STATE_KEY: kept} if kept else {}


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
            EXPORTER_DAEMONSET,
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
    nodes: dict[str, dict[str, Any]] = {}
    for name, value in sorted(observed.items()):
        if not isinstance(value, Mapping):
            continue
        nodes[str(name)] = {
            "state": str(value.get("state") or "unknown"),
            "conflict": value.get("conflict") is True,
        }
        version = dcgm_version(value.get("version"))
        if version is not None:
            nodes[str(name)]["version"] = version
    if not nodes:
        return None
    mode = dcgm_hostengine(target)
    record: dict[str, Any] = {
        "mode": mode.value,
        "observed_at": (now or datetime.now(UTC)).isoformat(),
        "nodes": nodes,
    }
    if mode is DcgmHostEngine.HOST and any(
        "version" in item for item in nodes.values()
    ):
        client = exporter_client_version(release)
        record["exporter_client_version"] = client or "unknown"
        newer = newer_hostengine_nodes(nodes, client)
        if newer:
            record["hostengine_newer_than_client_nodes"] = newer
            print(
                newer_hostengine_warning(target, nodes, newer, client),
                file=sys.stderr,
                flush=True,
            )
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
    versions = {
        name: str(value["version"])
        for name, value in sorted(observed.items())
        if value.get("version")
    }
    if versions:
        details["dcgm_hostengine_versions"] = versions
    if "exporter_client_version" in cluster:
        details["dcgm_exporter_client_version"] = cluster.get("exporter_client_version")
        details["dcgm_hostengine_newer_than_client_nodes"] = sorted(
            str(item)
            for item in cluster.get("hostengine_newer_than_client_nodes") or ()
        )
    return details


# --- the engine's version against the exporter's DCGM client ------------------


def exporter_client_version(release: Any) -> str | None:
    """The DCGM client library version of ``release``'s exporter, or None.

    The DCGM half of its image tag (``<dcgm>-<exporter>-<variant>``); a
    digest-pinned image (every schema v3 release) is read through the image
    lock's ``source`` tag when the digest is the locked one. An image whose tag
    does not follow NVIDIA's scheme (a private mirror by digest alone, the bare
    ``4.8.4``) names no version.
    """

    image = str(getattr(release, "dcgm_exporter_image", "") or "")
    versions = exporter_tag_versions(image)
    if versions is None and "@sha256:" in image:
        identity = getattr(
            getattr(release, "config", None), "release_delivery_identity", None
        )
        images = identity.get("images") if isinstance(identity, Mapping) else None
        locked = images.get("dcgm_exporter") if isinstance(images, Mapping) else None
        if isinstance(locked, Mapping):
            reference = str(locked.get("reference") or "")
            if reference.rpartition("@sha256:")[2] == image.rpartition("@sha256:")[2]:
                versions = exporter_tag_versions(locked.get("source"))
    return None if versions is None else versions[0]


def newer_hostengine_nodes(nodes: Mapping[str, Any], client: str | None) -> list[str]:
    """The nodes whose recorded engine version is newer than ``client``."""

    return sorted(
        str(name)
        for name, value in nodes.items()
        if isinstance(value, Mapping)
        and dcgm_version_relation(value.get("version"), client)
        is DcgmVersionRelation.HOST_NEWER
    )


def newer_hostengine_warning(
    target: Any, nodes: Mapping[str, Any], newer: Iterable[str], client: str | None
) -> str:
    """The release's ``WARN`` line for engines newer than the exporter's client.

    Advisory, never a refusal: the exporter still connects and metrics still
    flow (NVIDIA: mismatched combinations "might function"), a HyperPod AMI's
    engine is not the release's to change, and refusing would also refuse the
    release that brings a newer exporter. The remedy is the provider's.
    """

    names = ", ".join(
        f"{name} {(nodes.get(name) or {}).get('version')}" for name in newer
    )
    remedy = (
        f"install DCGM {client} on the node group "
        f"(gpu-fault-admin prepare-eks-nodegroup --dcgm-hostengine host --dcgm-version {client}) "
        "or deploy an exporter whose DCGM is at least the engine's"
        if _is_eks(target)
        else "the AMI ships the engine: deploy an exporter whose DCGM is at least the engine's"
    )
    return (
        f"WARN  DCGM host engine newer than the exporter's DCGM client {client or 'unknown'} "
        f"on {target.cluster_id}: {names}; NVIDIA requires the exporter's DCGM client "
        f"library to be at least the host engine version ({NVIDIA_EXPORTER_COMPATIBILITY_URL}); "
        f"{remedy} (administrator operations §5.0.1)"
    )
