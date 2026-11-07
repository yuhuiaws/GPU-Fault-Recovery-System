"""A GPU cluster without nodes: parked on purpose, or unavailable.

An operator may scale every GPU node group of a cluster to zero -- an EKS
managed node group to ``desiredSize`` 0, a HyperPod instance group to
``TargetCount`` 0 -- and keep the cluster in the site; capacity may also be
lost (a Spot reclaim, nodes that fail to join). Either way the cluster's
recorded node selector selects no Kubernetes node at all -- none, Ready or
not -- while its other nodes, its product Deployments and its executor stay up
and the departed-node cleanup retires the Agents of the nodes that left.
Kubernetes alone cannot tell the two apart, so the provider is asked, and the
cluster is in one of two states (``state_of``):

* *parked* when its provider declares that capacity zero:

  * ``eks-managed``: at least one managed node group the selector covers
    (``eks_node_repair.nodegroup_hosts_gpu_nodes``), every one of them
    ``ACTIVE`` with ``scalingConfig.desiredSize`` 0, and no Karpenter NodePool
    that could hand the selector a node (``nodepools_supplying``);
  * ``hyperpod``: at least one instance group the selector covers
    (``hyperpod_groups_for_selector``), every one of them ``InService`` with
    ``TargetCount`` 0 and ``CurrentCount`` 0;

* *unavailable* otherwise, the reason naming why it is not parked: a covered
  group that wants nodes (an outage, or a scale-up still joining), one that is
  not ``ACTIVE``/``InService`` (being created, updated or replaced), a
  selector that covers no group, a NodePool that could supply the scope, a
  provider read that failed.

The rule is ``gpu_fault.zero_node_capacity``; this module reads its documents
on the deploy host. The proof reads the documents ``regional_contexts`` already
reads --
``describe-nodegroup``, ``describe-cluster``, the Karpenter NodePools of the
automatic-update advisory -- so it needs no new permission; it is asked only of
a cluster whose node list is empty, at most once per report snapshot, and never
carried from one step of a release to the next.

Zero GPU nodes affect only their own cluster: never a deploy or an upgrade of
the site, never another GPU cluster or the CPU control plane. What changes for
that cluster; everything else is judged as for any cluster:

* a parked cluster is a state, not a finding. ``regional_contexts`` records
  ``providers.<id>.parked`` and skips only the Agent-endpoint CIDR coverage of
  nodes, of which there are none; ``gpu_cluster:<id>`` still judges the
  Deployments, the node scope and the executor's readiness and TLS path, and
  has no node or installer to judge; ``control_api`` reads zero Ready nodes
  against zero counted Agents as complete coverage and asks neither fleet nor
  collector readiness (``zero_node_not_awaited``);
* an unavailable cluster's zero-node findings -- no Kubernetes node, no Ready
  node, no registered Agent, and so no fleet or collector readiness to ask --
  fail ``status`` (``gating``), the operator's view of the site, which stays
  red. A deploy, its preflight and verify and every other engine mode file
  them instead (``zero_node_finding``; the check reads ``WARN``). Every other
  finding of the cluster -- its Deployments, executor TLS, IAM, context
  binding, live Agents on nodes the selector no longer selects (coverage
  drift) -- still gates;
* every report names both (``zero_node_clusters``: state, reason, groups and,
  for an unavailable cluster, the findings that did not gate), and so do the
  ``status`` header (``status_lines``) and the deploy record;
* every release kind proceeds. Its cluster-level components roll and are
  verified as usual; its node work is skipped and recorded
  (``record_node_work_skipped``, the ``zero-node-gpu-cluster`` log lines): the
  Agent snapshot holds an explicit entry with no identity and no node
  (``zero_node_identity``), and there is no node preflight batch, fleet
  rollout, wave or Agent wait. Its steady-state Reconciler rolls with the
  release's node runtime, the installer template rendered from a node of the
  declared group (``render_node``), so a node that joins later installs the
  current release. The CPU roll's heartbeat barrier, the rollback (the
  Reconciler alone, on its pinned previous template; Agent pins from the
  clusters with live Agents, or -- no node on any cluster -- the previous
  release's own ``-config-core``), the resume and ``sync-state`` all carry
  the entry;
* a release is refused at its deploy preflight (``release_refusal``) only when
  it could not finish: a Reconciler or Agent roll over a cluster whose groups
  declare no GPU instance type the installer knows (there is no node to
  render its template from). Zero nodes on every GPU cluster refuse nothing.

Capacity that returns needs no command: the group's ownership label reaches
the new nodes, the executor's node-key sync attests them, the steady-state
Reconciler installs them and their Agents register. The cluster has nodes
again the moment its selector selects one; from then on the usual join and
churn handling judge it.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from gpu_fault import zero_node_capacity as capacity
from gpu_fault.admin.automatic_node_updates import KARPENTER_NODEPOOLS
from gpu_fault.cluster_scope import (
    EKS_NODEGROUP_LABEL,
    HYPERPOD_CLUSTER_LABEL,
    HYPERPOD_INSTANCE_GROUP_LABEL,
    INSTANCE_TYPE_LABEL,
    is_eks_managed,
    node_selector,
)
from gpu_fault.collectors.gpu.discovery import expected_accelerator_counts

# The rule itself is one pure module (``gpu_fault.zero_node_capacity``); its
# names stay importable from here.
from gpu_fault.zero_node_capacity import EKS_MANAGED as EKS_MANAGED
from gpu_fault.zero_node_capacity import HYPERPOD as HYPERPOD
from gpu_fault.zero_node_capacity import PARKED as PARKED
from gpu_fault.zero_node_capacity import UNAVAILABLE as UNAVAILABLE
from gpu_fault.zero_node_capacity import NodePoolRead as NodePoolRead
from gpu_fault.zero_node_capacity import eks_parked as eks_parked
from gpu_fault.zero_node_capacity import hyperpod_parked as hyperpod_parked
from gpu_fault.zero_node_capacity import nodepools_supplying as nodepools_supplying
from gpu_fault.zero_node_capacity import state_of as state_of
from gpu_fault_release import regional_release_state as release_state
from gpu_fault_release.regional_release_config import CheckFailure, ReleaseError
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from gpu_fault_release.regional_release_narration import narrate_step

#: The key of a cluster's proof in check details (``regional_contexts``
#: ``providers.<id>``, ``control_api`` ``clusters.<id>``, ``gpu_cluster:<id>``).
PARKED_KEY = "parked"
#: Report, status and release-state key: cluster id -> what is known of a
#: cluster without nodes (``state``, ``provider``, ``reason``, ``groups``; the
#: release state adds what the release skipped there).
ZERO_NODE_KEY = "zero_node_clusters"
#: An Agent identity entry's key for a cluster without nodes (no identity).
ZERO_NODES_IDENTITY_KEY = "zero_nodes"
#: Collector readiness evidence: the clusters without nodes it did not wait for.
NOT_AWAITED_KEY = "zero_node_not_awaited"
#: A check section's findings of an unavailable cluster that did not gate it.
FINDINGS_KEY = "zero_node_findings"
#: A ``zero_node_clusters`` entry's list of them, each with its check.
NOT_GATING_KEY = "not_gating_findings"
#: The check summary's note of them (``mark_findings``).
NOT_GATING_NOTE = "; not gating (GPU cluster without nodes, unavailable): "
#: The release attribute ``gating_findings`` sets for ``status``.
_GATING = "_zero_node_findings_gate"
#: A narrated value that needs no quoting.
_PLAIN = re.compile(r'[^\s"=]*')
#: The deploy preflight check a release fails when no GPU cluster has a node.
CHECK = "zero_node_gpu_clusters"
#: The ``narrate_step`` kind of every line about a cluster without nodes.
NARRATION = "zero-node-gpu-cluster"
#: The node-level work a release skips on a cluster without nodes.
SKIPPED_NODE_WORK = (
    "agent-snapshot",
    "node-preflight-batch",
    "fleet-rollout",
    "agent-waves",
    "agent-wait",
)
#: The script switch and the node the installer template is rendered from.
ZERO_NODE_CLUSTER_ENV = "GPU_FAULT_ZERO_NODE_CLUSTER"
RENDER_NODE_ENV = "GPU_FAULT_ZERO_NODE_RENDER_NODE"
RENDER_NODE_NAME = "gpu-fault-zero-node-template"
RENDER_NODE_UID = "00000000-0000-0000-0000-000000000000"
RENDER_NODE_ADDRESS = "0.0.0.0"
_STATE_LOCK = threading.Lock()

#: ``sagemaker describe-cluster`` reduced to the cluster's binding and, per
#: instance group, what the automatic-update advisory and the parked proof
#: read: one call serves ``regional_contexts``, the advisory and the proof (the
#: CPU cluster's advisory also counts the instances one window re-images).
HYPERPOD_CONTEXT_QUERY = (
    "{EksClusterArn:Orchestrator.Eks.ClusterArn,NodeRecovery:NodeRecovery,"
    "InstanceGroups:InstanceGroups[].{InstanceGroupName:InstanceGroupName,"
    "InstanceType:InstanceType,CurrentCount:CurrentCount,TargetCount:TargetCount,"
    "Status:Status,KubernetesConfig:KubernetesConfig,"
    "ScheduledUpdateConfig:ScheduledUpdateConfig,AutoPatchConfig:AutoPatchConfig}}"
)


def node_kind(target: object) -> str:
    """``GPU`` on an eks-managed cluster, ``HyperPod`` on a HyperPod one: the
    word every per-cluster node message uses (``target_node_names``)."""

    return "GPU" if is_eks_managed(target) else "HyperPod"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


# --- the reads -------------------------------------------------------------


def read_karpenter_nodepools(release: Any, target: Any) -> dict[str, Any] | None:
    """The cluster's ``nodepools.karpenter.sh`` list; ``None`` without the CRD.

    The read the automatic-update advisory makes; any other failure raises."""

    code, output, error = release.runner.probe_output(
        release._gpu(target, "get", KARPENTER_NODEPOOLS, "-o", "json"),
        timeout_seconds=60,
    )
    if code == 0:
        value = json.loads(output) if output.strip() else {}
        return value if isinstance(value, dict) else {}
    if "doesn't have a resource type" in error:
        return None
    raise ReleaseError(error.strip() or f"kubectl exited {code}")


def _region(target: Any) -> list[str]:
    return ["--region", str(target.region)]


def _described_nodegroups(release: Any, target: Any) -> dict[str, Mapping[str, Any]]:
    eks_name = str(target.eks_cluster_arn).rsplit("/", 1)[-1]
    names = release_state.aws_json(
        release,
        ["eks", "list-nodegroups", "--cluster-name", eks_name, *_region(target)],
        region=False,
    ).get("nodegroups")
    return {
        name: release_state.aws_json(
            release,
            [
                "eks",
                "describe-nodegroup",
                "--cluster-name",
                eks_name,
                "--nodegroup-name",
                name,
                "--query",
                "nodegroup",
                *_region(target),
            ],
            region=False,
        )
        for name in sorted(str(item) for item in names or () if item)
    }


def _instance_groups(release: Any, target: Any) -> list[Any]:
    value = release_state.aws_json(
        release,
        [
            "sagemaker",
            "describe-cluster",
            "--cluster-name",
            str(target.hyperpod_cluster_name or ""),
            "--query",
            HYPERPOD_CONTEXT_QUERY,
            *_region(target),
        ],
        region=False,
    )
    return list(value.get("InstanceGroups") or ())


def provider_proof(
    release: Any,
    target: Any,
    *,
    described: Mapping[str, Mapping[str, Any]] | None = None,
    groups: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """``target``'s proof from documents already read (``described`` for an
    eks-managed cluster, ``groups`` for a HyperPod one) or read here; never
    raises: a provider that cannot be read proves nothing."""

    provider = EKS_MANAGED if is_eks_managed(target) else HYPERPOD
    try:
        selector = node_selector(target)
    except Exception as exc:  # noqa: BLE001 - recorded as the reason, never raised
        return capacity.unreadable(provider, exc)
    return capacity.capacity_proof(
        provider,
        selector=selector,
        cluster_name=str(target.hyperpod_cluster_name or ""),
        nodegroups=lambda: (
            _described_nodegroups(release, target) if described is None else described
        ),
        instance_groups=lambda: (
            _instance_groups(release, target) if groups is None else groups
        ),
        nodepools=lambda: read_karpenter_nodepools(release, target),
    )


def read_parked(release: Any, target: Any) -> dict[str, Any]:
    """``provider_proof`` read once per report snapshot, whichever check asks
    first (``status --full`` asks from three); outside a snapshot, afresh."""

    cache = getattr(release, "_json_read_cache", None)
    lock = getattr(release, "_json_read_cache_lock", None)
    if cache is None or lock is None:
        return provider_proof(release, target)
    return release_state.cached_read(
        cache,
        lock,
        ("parked-gpu-cluster", str(target.cluster_id)),
        lambda: provider_proof(release, target),
    )


def _selected_nodes(release: Any, target: Any) -> list[Mapping[str, Any]] | None:
    """The nodes the API lists for ``target``'s selector; ``None`` when unreadable.

    Taken as the API answers, without the local re-check the fleet's node names
    get (``target_node_names``): only a list the API itself returns empty is a
    cluster without nodes, and an answer without an ``items`` list is no list
    at all. A node the re-check would drop leaves the cluster to the node path,
    which refuses an empty node set -- the safe direction.
    """

    try:
        listing = release._get_json(
            release._gpu(target, "get", "nodes", "-l", node_selector(target))
        )
        items = listing.get("items")
        if not isinstance(items, list):
            return None
        return [item for item in items if isinstance(item, Mapping)]
    except Exception:  # noqa: BLE001 - an unreadable node list proves nothing
        return None


def recorded_zero_nodes(release: Any, target: Any) -> dict[str, Any] | None:
    """The proof the release snapshot recorded for ``target`` when the release
    started without a node there (``zero_node_identity``), else ``None``."""

    state = getattr(release, "state", None)
    previous = state.get("previous") if isinstance(state, Mapping) else None
    identities = _mapping(_mapping(previous).get("agent_identities"))
    identity = identities.get(str(target.cluster_id))
    if not is_zero_node_identity(identity):
        return None
    entry = _mapping(_mapping(identity).get(ZERO_NODES_IDENTITY_KEY))
    return {
        "parked": entry.get("state") == PARKED,
        "provider": entry.get("provider"),
        "reason": entry.get("reason"),
        "groups": [dict(_mapping(item)) for item in entry.get("groups") or ()],
    }


def zero_nodes_now(release: Any, target: Any) -> dict[str, Any] | None:
    """The proof a release step skips ``target``'s node work on, else ``None``.

    The proof read now (``without_nodes``), parked or unavailable. When that
    read names no group -- the provider could not be read -- and the release
    snapshot recorded the cluster's groups at its start
    (``recorded_zero_nodes``), those stand in for it: a step of a release that
    started without nodes -- its rollback above all -- is not held up by a
    provider read that failed.
    """

    proof = without_nodes(release, target)
    if proof is None or proof.get("groups"):
        return proof
    recorded = recorded_zero_nodes(release, target)
    return recorded if recorded is not None and recorded.get("groups") else proof


def without_nodes(release: Any, target: Any) -> dict[str, Any] | None:
    """The proof of ``target`` -- parked or unavailable -- when its selector
    selects no node, else ``None``, including when the node list cannot be read.

    The node list is read first and the provider documents only when it is
    empty; a node that appears later is judged by the usual join and churn
    handling, never by this answer. Inside a report snapshot the caller opened
    (``status``, a release's capture of its previous state) both reads join it.
    This never opens one: a snapshot is release-wide state -- its cache lives
    on the release object every thread shares -- and a release step runs its
    clusters in parallel threads whose node safety reads must stay fresh
    (``gpu_node_items(fresh=True)`` refuses inside a snapshot). Live, a parked
    cluster's snapshot held open for its provider reads failed the other
    cluster's fresh read in the candidate preflight.
    """

    selected = _selected_nodes(release, target)
    if selected is None or selected:
        return None
    return read_parked(release, target)


# --- what the checks make of it -------------------------------------------


def zero_nodes_failure(message: str, proof: Mapping[str, Any] | None) -> str:
    """``message``, then why the cluster is not parked when that was asked."""

    if not isinstance(proof, Mapping) or proof.get("parked") is True:
        return message
    return f"{message}; not parked: {proof.get('reason')}"


def gating(release: Any) -> bool:
    """Whether an unavailable cluster's zero-node findings fail what reads them.

    Only ``status`` (``gating_findings``): the operator's view of the site stays
    red while a cluster has lost its nodes. A deploy, its preflight and verify
    and every other engine mode file them (``zero_node_finding``)."""

    return getattr(release, _GATING, False) is True


@contextmanager
def gating_findings(release: Any) -> Iterator[None]:
    """``release`` with an unavailable cluster's zero-node findings failing."""

    previous = getattr(release, _GATING, False)
    setattr(release, _GATING, True)
    try:
        yield
    finally:
        setattr(release, _GATING, previous)


def zero_node_finding(
    release: Any,
    section: dict[str, Any],
    message: str,
    proof: Mapping[str, Any],
    *,
    check: str,
    cluster_id: str,
    details: Any,
) -> None:
    """``message``, what ``check`` finds of ``cluster_id`` -- which selects no
    node and is not parked (``proof``) -- filed in ``section`` and narrated; in
    ``status`` (``gating``) raised instead, the failure carrying ``details``."""

    finding = zero_nodes_failure(message, proof)
    if gating(release):
        raise CheckFailure(finding, details)
    with _STATE_LOCK:
        findings = section.setdefault(FINDINGS_KEY, [])
        if finding in findings:
            return
        findings.append(finding)
    narrate(cluster_id, state=UNAVAILABLE, check=check, gating="no", finding=finding)


def zero_ready_nodes(
    release: Any, target: Any, nodes: Mapping[str, Any], ready: Sequence[Any]
) -> dict[str, Any]:
    """The ``gpu_cluster:<id>`` judgement of a cluster without a Ready node.

    ``{}`` when it has one. A cluster whose selector selects no node gets its
    proof (``{"parked": proof}``); unavailable, with its finding
    (``zero_node_finding``). A cluster with nodes of which none is Ready
    refuses, naming the cluster's kind of node.
    """

    if ready:
        return {}
    message = f"{target.cluster_id} has no Ready {node_kind(target)} nodes"
    if nodes.get("items"):
        raise ReleaseError(message)
    proof = read_parked(release, target)
    section: dict[str, Any] = {PARKED_KEY: proof}
    if proof.get("parked") is not True:
        zero_node_finding(
            release,
            section,
            message,
            proof,
            check=f"gpu_cluster:{target.cluster_id}",
            cluster_id=str(target.cluster_id),
            details={PARKED_KEY: proof},
        )
    return section


def data_plane_summary(cluster_id: str, details: Mapping[str, Any]) -> str:
    """The ``gpu_cluster:<id>`` summary, naming a cluster without nodes as such."""

    proof = details.get(PARKED_KEY)
    if not isinstance(proof, Mapping):
        return f"{cluster_id} data plane is healthy"
    if proof.get("parked") is True:
        return f"{cluster_id} data plane is healthy; parked at zero GPU nodes"
    return (
        f"{cluster_id} data plane Deployments and executor are healthy; no GPU "
        "node (unavailable)"
    )


def annotate_parked(
    report: dict[str, Any],
    release: Any,
    node_ids_by_cluster: Mapping[str, Sequence[str]],
) -> None:
    """Record the proof of every cluster of ``report`` that has no node, in place."""

    clusters = report.get("clusters")
    targets = {target.cluster_id: target for target in release.config.clusters}
    for cluster_id, node_ids in node_ids_by_cluster.items():
        cluster = clusters.get(cluster_id) if isinstance(clusters, dict) else None
        if isinstance(cluster, dict) and not node_ids and cluster_id in targets:
            cluster[PARKED_KEY] = read_parked(release, targets[cluster_id])


def is_zero_node(cluster: Mapping[str, Any]) -> bool:
    """Whether a ``control_api`` cluster section is a cluster's without nodes,
    parked or unavailable: a proof, and neither a Kubernetes node nor an
    expected one."""

    return (
        isinstance(cluster.get(PARKED_KEY), Mapping)
        and not cluster.get("expected_node_ids")
        and not cluster.get("kubernetes_node_ids")
    )


def is_parked(cluster: Mapping[str, Any]) -> bool:
    """Whether a ``control_api`` cluster section is a parked cluster's."""

    return (
        is_zero_node(cluster)
        and _mapping(cluster.get(PARKED_KEY)).get("parked") is True
    )


def not_awaited(
    clusters: Mapping[str, Any], snapshots: Mapping[str, Any], *, gate: bool = False
) -> frozenset[str]:
    """The clusters without nodes whose collector readiness answer lists no
    ACTIVE Agent: nothing there will ever report, so nothing is waited for --
    a parked cluster always, an unavailable one unless its zero-node findings
    gate (``gate``, in ``status``)."""

    return frozenset(
        str(cluster_id)
        for cluster_id, cluster in clusters.items()
        if isinstance(cluster, Mapping)
        and (is_parked(cluster) or (not gate and is_zero_node(cluster)))
        and isinstance(snapshots.get(cluster_id), Mapping)
        and _mapping(snapshots.get(cluster_id)).get("nodes") == []
    )


def summary(proof: Mapping[str, Any]) -> dict[str, Any]:
    """What a report and the release state say of a cluster without nodes."""

    return {
        "state": state_of(proof),
        "provider": proof.get("provider"),
        "reason": proof.get("reason"),
        "groups": [dict(_mapping(item)) for item in proof.get("groups") or ()],
    }


def _sections(check: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """A check's per-cluster sections: where a proof and findings are kept."""

    name = str(check.get("name") or "")
    details = _mapping(check.get("details"))
    if name == "control_api":
        sections = _mapping(details.get("clusters"))
    elif name == "regional_contexts":
        sections = _mapping(details.get("providers"))
    elif name.startswith("gpu_cluster:"):
        sections = {name.partition(":")[2]: details}
    else:
        sections = {}
    return {str(key): _mapping(value) for key, value in sorted(sections.items())}


def zero_node_clusters(checks: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """Cluster id -> ``summary`` of every cluster without nodes a report's
    checks name, parked or unavailable (the reports' ``zero_node_clusters``);
    an unavailable cluster's entry lists the findings that did not gate the
    report (``not_gating_findings``: check, finding)."""

    found: dict[str, dict[str, Any]] = {}
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        for cluster_id, section in _sections(check).items():
            proof = section.get(PARKED_KEY)
            if not isinstance(proof, Mapping):
                continue
            entry = found.setdefault(cluster_id, summary(proof))
            for finding in section.get(FINDINGS_KEY) or ():
                entry.setdefault(NOT_GATING_KEY, []).append(
                    {"check": str(check.get("name")), "finding": str(finding)}
                )
    return dict(sorted(found.items()))


def mark_findings(checks: Sequence[Any]) -> list[Any]:
    """A passing check that filed an unavailable cluster's findings reads
    ``WARN``, its summary naming them: they did not gate the report."""

    marked = []
    for check in checks:
        findings = (
            [
                str(finding)
                for section in _sections(check).values()
                for finding in section.get(FINDINGS_KEY) or ()
            ]
            if isinstance(check, Mapping)
            and check.get("status") in {"PASS", "WARN"}
            and NOT_GATING_NOTE not in str(check.get("summary"))
            else []
        )
        if findings:
            check = {
                **check,
                "status": "WARN",
                "summary": f"{check.get('summary')}{NOT_GATING_NOTE}"
                + "; ".join(findings),
            }
        marked.append(check)
    return marked


def report_fields(checks: Sequence[Any]) -> dict[str, Any]:
    """A report's ``checks`` (``mark_findings``) and ``zero_node_clusters``."""

    marked = mark_findings(checks)
    return {"checks": marked, ZERO_NODE_KEY: zero_node_clusters(marked)}


def _groups_text(entry: Mapping[str, Any]) -> str:
    groups = [_mapping(item) for item in entry.get("groups") or ()]
    # An absent provider is HyperPod (cluster_scope.cluster_provider), so only
    # an eks-managed entry reads as node groups.
    if entry.get("provider") == EKS_MANAGED:
        return ", ".join(
            f"{item.get('name')}: desiredSize {item.get('desired')}, "
            f"min {item.get('min')}, max {item.get('max')}"
            for item in groups
        )
    return ", ".join(
        f"{item.get('name')}: TargetCount {item.get('target')}, CurrentCount "
        f"{item.get('current')}"
        for item in groups
    )


def status_lines(health: object) -> list[str]:
    """One ``status`` header line per cluster without nodes of a health report."""

    checks = _mapping(health).get("checks")
    entries = zero_node_clusters(checks if isinstance(checks, list) else [])
    return [
        (
            f"parked GPU cluster {cluster_id} ({entry.get('provider')}): no node "
            f"selected; {entry.get('reason')} ({_groups_text(entry)}); not a "
            "failure -- a deploy rolls its Deployments and Reconciler and skips "
            "its node work; scale a covered group up and its nodes join with the "
            "current release by themselves"
        )
        if entry.get("state") == PARKED
        else (
            f"GPU cluster {cluster_id} unavailable: no node selected; "
            f"{entry.get('reason')}; the deploy skips its node work"
        )
        for cluster_id, entry in entries.items()
    ]


# --- what a release makes of it -------------------------------------------


def zero_node_identity(proof: Mapping[str, Any]) -> dict[str, Any]:
    """The Agent identity entry a release snapshot records for a cluster
    without nodes: no node, no identity, and why."""

    return {"node_ids": [], ZERO_NODES_IDENTITY_KEY: summary(proof)}


def is_zero_node_identity(identity: object) -> bool:
    return (
        isinstance(identity, Mapping)
        and isinstance(identity.get(ZERO_NODES_IDENTITY_KEY), Mapping)
        and not identity.get("node_ids")
    )


def live_identities(identities: Mapping[str, Any]) -> dict[str, Any]:
    """``identities`` without the entries of clusters that had no node."""

    return {
        str(cluster_id): identity
        for cluster_id, identity in identities.items()
        if not is_zero_node_identity(identity)
    }


def render_node(proof: Mapping[str, Any], target: Any) -> dict[str, Any]:
    """The node document a cluster without nodes renders its installer
    template from: a node of the first covered group whose declared instance
    type the GPU inventory knows, carrying that group's declared labels.

    The Reconciler fills the name, address, UID, provider id and instance type
    of every real node into the Jobs it builds from the template, so the node
    that is rendered from is never one a Job runs for. A cluster whose groups
    declare no known GPU instance type refuses: its template cannot be rendered.
    """

    chosen = _render_group(proof)
    if chosen is None:
        raise ReleaseError(
            f"{target.cluster_id} has no node and its node groups declare no GPU "
            "instance type the installer knows ("
            + (", ".join(_declared_types(proof)) or "none declared")
            + "): its installer template cannot be rendered; scale a group up, "
            "or declare a supported instance type"
        )
    group, instance_type = chosen
    labels = {
        **{str(k): str(v) for k, v in _mapping(group.get("labels")).items()},
        INSTANCE_TYPE_LABEL: instance_type,
    }
    if is_eks_managed(target):
        labels[EKS_NODEGROUP_LABEL] = str(group.get("name") or "")
    else:
        labels[HYPERPOD_CLUSTER_LABEL] = str(target.hyperpod_cluster_name or "")
        labels[HYPERPOD_INSTANCE_GROUP_LABEL] = str(group.get("name") or "")
    return {
        "metadata": {
            "name": RENDER_NODE_NAME,
            "uid": RENDER_NODE_UID,
            "labels": dict(sorted(labels.items())),
        },
        "spec": {"providerID": ""},
        "status": {
            "addresses": [{"type": "InternalIP", "address": RENDER_NODE_ADDRESS}]
        },
    }


def _render_group(proof: Mapping[str, Any]) -> tuple[Mapping[str, Any], str] | None:
    """The first covered group and its first instance type the GPU inventory
    knows, or ``None``."""

    for group in (_mapping(item) for item in proof.get("groups") or ()):
        for instance_type in group.get("instance_types") or ():
            if expected_accelerator_counts(str(instance_type)) is not None:
                return group, str(instance_type)
    return None


def _declared_types(proof: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(item)
        for group in (_mapping(entry) for entry in proof.get("groups") or ())
        for item in group.get("instance_types") or ()
    )


def reconciler_environment(
    proof: Mapping[str, Any],
    target: Any,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment ``deploy-node-installer-reconciler.sh`` runs a cluster
    without nodes with: no node to require, key or preflight, the installer
    template rendered from ``render_node`` -- or none at all when
    ``environment`` pins a template (``GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP``:
    a rollback restores the previous one), so that path needs no declared
    group."""

    if (environment or {}).get("GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"):
        return {ZERO_NODE_CLUSTER_ENV: "true", RENDER_NODE_ENV: ""}
    return {
        ZERO_NODE_CLUSTER_ENV: "true",
        RENDER_NODE_ENV: json.dumps(
            render_node(proof, target), sort_keys=True, separators=(",", ":")
        ),
    }


def narrate(cluster_id: str, **fields: Any) -> None:
    """One operator-visible line about a cluster without nodes; a value with a
    blank, a quote or an equals sign is quoted (JSON), so every field parses."""

    narrate_step(
        NARRATION,
        cluster=cluster_id,
        **{
            name: value if _PLAIN.fullmatch(str(value)) else json.dumps(str(value))
            for name, value in fields.items()
        },
    )


def record_node_work_skipped(
    release: Any,
    target: Any,
    proof: Mapping[str, Any],
    *,
    phase: str,
    identity: tuple[str, str],
    skipped: Sequence[str] = SKIPPED_NODE_WORK,
) -> None:
    """Narrate and record (``release.state["zero_node_clusters"]``, carried by
    the next checkpoint) the node work a release skipped on ``target``."""

    entry = {
        **summary(proof),
        "release_id": str(getattr(release, "release_id", "") or ""),
        "phase": phase,
        "skipped": list(skipped),
        "reconciler": {"bundle_sha256": identity[0], "template_sha256": identity[1]},
    }
    narrate(
        target.cluster_id,
        state=entry["state"],
        phase=phase,
        skipped=",".join(skipped),
        reconciler="steady-state",
        reconciler_bundle=identity[0][:12],
        reconciler_template=identity[1][:12],
        reason=str(entry["reason"]),
    )
    state = getattr(release, "state", None)
    if isinstance(state, dict):
        with _STATE_LOCK:
            state.setdefault(ZERO_NODE_KEY, {})[str(target.cluster_id)] = entry


def zero_node_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """The live release's record of the node work it skipped (``status``)."""

    entries = state.get(ZERO_NODE_KEY)
    mine = {
        str(cluster_id): dict(entry)
        for cluster_id, entry in (entries.items() if isinstance(entries, dict) else ())
        if isinstance(entry, Mapping)
        and entry.get("release_id") == state.get("release_id")
    }
    return {ZERO_NODE_KEY: mine} if mine else {}


def release_refusal(
    checks: Iterable[Any], plan: ReleaseExecutionPlan
) -> dict[str, Any] | None:
    """The failing deploy preflight check of a release that could not finish
    over the site's clusters without nodes; ``None`` for every other release
    (``zero_node_clusters`` of the preflight's ``checks``): a Reconciler or
    Agent roll over a cluster whose node groups declare no GPU instance type
    the installer knows -- its installer template is rendered from a declared
    node (``render_node``), and there is none.

    Zero nodes on every GPU cluster refuse nothing: with no live Agent to take
    them from, a rollback stages the previous release's own Agent pins
    (``regional_release_rollback_context.rollback_controller_config_for``).
    """

    zero_nodes = zero_node_clusters(checks)
    if not zero_nodes:
        return None
    unrenderable = sorted(
        cluster_id
        for cluster_id, entry in zero_nodes.items()
        if _render_group(entry) is None
    )
    if unrenderable and (
        plan.has(ReleaseComponent.RECONCILER) or plan.has(ReleaseComponent.AGENT)
    ):
        return _refusal(
            zero_nodes,
            "GPU cluster(s) "
            + ", ".join(
                f"{cluster_id} (declared: "
                + (", ".join(_declared_types(zero_nodes[cluster_id])) or "none")
                + ")"
                for cluster_id in unrenderable
            )
            + " have zero nodes and no node group declaring a GPU instance type "
            "the installer knows: a release that rolls the Reconciler renders "
            "their installer template from a declared node, and there is none. "
            "Scale a group up and let a node join, or declare a supported "
            "instance type on the group, then deploy again. Nothing was changed.",
        )
    return None


def _refusal(zero_nodes: Mapping[str, Any], summary_text: str) -> dict[str, Any]:
    return {
        "name": CHECK,
        "status": "FAIL",
        "summary": summary_text,
        "details": {ZERO_NODE_KEY: dict(zero_nodes)},
    }
