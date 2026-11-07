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

The proof reads the documents ``regional_contexts`` already reads --
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
  clusters with live Agents), the resume and ``sync-state`` all carry the
  entry;
* a release is refused at its deploy preflight (``release_refusal``) only when
  it could not finish or roll back: Agent waves while every GPU cluster has
  zero nodes (the rollback derives the Agent pins from live Agents), or a
  Reconciler or Agent roll over a cluster whose groups declare no GPU instance
  type the installer knows (there is no node to render its template from).

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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from gpu_fault.admin.automatic_node_updates import KARPENTER_NODEPOOLS
from gpu_fault.admin.bootstrap_ownership import hyperpod_groups_for_selector
from gpu_fault.cluster_scope import (
    EKS_NODEGROUP_LABEL,
    HYPERPOD_CLUSTER_LABEL,
    HYPERPOD_INSTANCE_GROUP_LABEL,
    INSTANCE_TYPE_LABEL,
    is_eks_managed,
    node_selector,
    selector_ownership_value,
    selector_requirements,
)
from gpu_fault.collectors.gpu.discovery import expected_accelerator_counts
from gpu_fault.eks_node_repair import nodegroup_hosts_gpu_nodes, selected_nodegroup
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
#: The state of a cluster whose provider declares zero capacity.
PARKED = "parked"
#: The state of every other cluster without nodes (``state_of``).
UNAVAILABLE = "unavailable"
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
EKS_MANAGED = "eks-managed"
HYPERPOD = "hyperpod"
_REASON_LIMIT = 300
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

NodePoolRead = Callable[[], Mapping[str, Any] | None]


def node_kind(target: object) -> str:
    """``GPU`` on an eks-managed cluster, ``HyperPod`` on a HyperPod one: the
    word every per-cluster node message uses (``target_node_names``)."""

    return "GPU" if is_eks_managed(target) else "HyperPod"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _reason(exc: BaseException) -> str:
    text = re.sub(r"\s+", " ", str(exc)).strip() or type(exc).__name__
    return text if len(text) <= _REASON_LIMIT else text[: _REASON_LIMIT - 3] + "..."


def _proof(
    provider: str, parked: bool, reason: str, groups: list[dict[str, Any]], **extra: Any
) -> dict[str, Any]:
    return {
        "parked": parked,
        "provider": provider,
        "reason": reason,
        "groups": groups,
        **extra,
    }


# --- the provider's declared capacity --------------------------------------


def _eks_waiting(group: Mapping[str, Any]) -> str:
    name, status, desired = group["name"], group["status"], group["desired"]
    if status != "ACTIVE":
        return (
            f"node group {name} is {status or 'of unknown status'} "
            "(being created, updated or replaced)"
        )
    if desired is None:
        return f"node group {name} reports no scalingConfig.desiredSize"
    return (
        f"node group {name} wants {desired} node(s) (desiredSize {desired}): "
        "its nodes are lost or still joining"
    )


def eks_parked(
    described: Mapping[str, Mapping[str, Any]],
    *,
    selector: str,
    nodepools: NodePoolRead,
) -> dict[str, Any]:
    """The proof of an eks-managed cluster from its ``describe-nodegroup``
    documents (name -> ``nodegroup``) and its Karpenter NodePools."""

    groups = [
        {
            "name": str(name),
            "status": str(group.get("status") or ""),
            "desired": _count(_mapping(group.get("scalingConfig")).get("desiredSize")),
            "min": _count(_mapping(group.get("scalingConfig")).get("minSize")),
            "max": _count(_mapping(group.get("scalingConfig")).get("maxSize")),
            # What a node of the group is born with: the installer template of
            # a cluster without nodes is rendered from them (``render_node``).
            "instance_types": [
                str(item)
                for item in (
                    *(group.get("instanceTypes") or ()),
                    *(
                        (group.get("instanceType"),)
                        if group.get("instanceType")
                        else ()
                    ),
                )
                if item
            ],
            "labels": {
                str(key): str(value)
                for key, value in _mapping(group.get("labels")).items()
            },
        }
        for name, group in sorted(described.items())
        if isinstance(group, Mapping) and nodegroup_hosts_gpu_nodes(group, selector)
    ]
    if not groups:
        return _proof(
            EKS_MANAGED,
            False,
            "the node selector covers no managed node group (a selector that "
            "selects nothing, or a group not labelled for this site)",
            groups,
        )
    waiting = [
        item for item in groups if item["status"] != "ACTIVE" or item["desired"] != 0
    ]
    if waiting:
        return _proof(
            EKS_MANAGED,
            False,
            "; ".join(_eks_waiting(item) for item in waiting),
            groups,
        )
    try:
        document = nodepools()
    except Exception as exc:  # noqa: BLE001 - an unreadable list proves nothing
        return _proof(
            EKS_MANAGED,
            False,
            f"the Karpenter NodePools could not be read: {_reason(exc)}",
            groups,
        )
    suppliers = nodepools_supplying(document, selector)
    if suppliers:
        return _proof(
            EKS_MANAGED,
            False,
            "Karpenter NodePool(s) "
            + ", ".join(suppliers)
            + " can still launch nodes the node selector selects",
            groups,
            nodepools=suppliers,
        )
    return _proof(
        EKS_MANAGED,
        True,
        "every managed node group the node selector covers is ACTIVE at "
        "desiredSize 0 and no Karpenter NodePool can launch a node it selects",
        groups,
        proven_by=[
            "eks:DescribeNodegroup status and scalingConfig",
            "Karpenter NodePools (" + ("absent" if document is None else "none") + ")",
        ],
    )


def _requirement_values(
    requirements: Sequence[Mapping[str, Any]], key: str
) -> frozenset[str] | None:
    """The values a NodePool's ``requirements`` may give ``key`` on a node:
    ``None`` when none sets it, ``frozenset({"*"})`` for any value."""

    produced: set[str] = set()
    found = False
    for item in requirements:
        if str(item.get("key") or "") != key:
            continue
        operator = str(item.get("operator") or "")
        if operator == "In":
            found = True
            produced.update(str(value) for value in item.get("values") or ())
        elif operator in {"Exists", "Gt", "Lt"}:
            return frozenset({"*"})
    return frozenset(produced) if found else None


def _could_select(
    labels: Mapping[str, str],
    requirements: Sequence[Mapping[str, Any]],
    selector: str,
) -> bool:
    """Whether a node a NodePool launches could satisfy ``selector``: its
    template labels, or a requirement that puts the key on the node."""

    for key, operator, values in selector_requirements(selector):
        if key in labels:
            value = labels[key]
            if (
                (operator == "in" and value not in values)
                or (operator == "notin" and value in values)
                or operator == "absent"
            ):
                return False
            continue
        produced = _requirement_values(requirements, key)
        if operator in {"in", "exists"} and (
            produced is None
            or (operator == "in" and "*" not in produced and not produced & values)
        ):
            return False
    return True


def nodepools_supplying(document: Mapping[str, Any] | None, selector: str) -> list[str]:
    """The Karpenter NodePools that could launch a node ``selector`` selects.

    A selector that names the ownership label or a node group is judged
    against each NodePool's template labels and requirements (a requirement
    ``<key> In [...]`` or ``Exists`` puts the label on the node too). Any other
    selector names labels a node gains at runtime (``nvidia.com/gpu.present``),
    which no template predicts, so every NodePool could supply it. ``None`` is
    a cluster without the Karpenter CRD.
    """

    if document is None:
        return []
    predictable = selector_ownership_value(selector) is not None or bool(
        selected_nodegroup(selector)
    )
    found = []
    for item in document.get("items") or ():
        if not isinstance(item, Mapping):
            continue
        template = _mapping(_mapping(item.get("spec")).get("template"))
        labels = {
            str(key): str(value)
            for key, value in _mapping(
                _mapping(template.get("metadata")).get("labels")
            ).items()
        }
        requirements = [
            requirement
            for requirement in _mapping(template.get("spec")).get("requirements") or ()
            if isinstance(requirement, Mapping)
        ]
        if not predictable or _could_select(labels, requirements, selector):
            found.append(str(_mapping(item.get("metadata")).get("name") or "<unnamed>"))
    return sorted(found)


def _hyperpod_waiting(group: Mapping[str, Any]) -> str:
    name, status = group["name"], group["status"]
    target, current = group["target"], group["current"]
    if status != "InService":
        return f"instance group {name} is {status or 'of unknown status'}"
    if target is None or current is None:
        return f"instance group {name} reports no TargetCount/CurrentCount"
    if target:
        return (
            f"instance group {name} wants {target} instance(s) (TargetCount {target}, "
            f"CurrentCount {current}): its nodes are lost or still joining"
        )
    return f"instance group {name} still has {current} instance(s) at TargetCount 0"


def hyperpod_parked(
    groups: Sequence[Any], *, cluster_name: str, selector: str
) -> dict[str, Any]:
    """The proof of a HyperPod cluster from its ``describe-cluster`` instance
    groups (``HYPERPOD_CONTEXT_QUERY``)."""

    covered = set(
        hyperpod_groups_for_selector(
            groups, cluster_name=cluster_name, selector=selector
        )
    )
    entries = [
        {
            "name": str(group.get("InstanceGroupName") or ""),
            "status": str(group.get("Status") or ""),
            "target": _count(group.get("TargetCount")),
            "current": _count(group.get("CurrentCount")),
            "instance_types": (
                [str(group["InstanceType"])] if group.get("InstanceType") else []
            ),
            "labels": {
                str(key): str(value)
                for key, value in _mapping(
                    _mapping(group.get("KubernetesConfig")).get("CurrentLabels")
                ).items()
            },
        }
        for group in groups
        if isinstance(group, Mapping)
        and str(group.get("InstanceGroupName") or "") in covered
    ]
    if not entries:
        return _proof(
            HYPERPOD, False, "the node selector covers no instance group", entries
        )
    waiting = [
        item
        for item in entries
        if item["status"] != "InService" or item["target"] != 0 or item["current"] != 0
    ]
    if waiting:
        return _proof(
            HYPERPOD,
            False,
            "; ".join(_hyperpod_waiting(item) for item in waiting),
            entries,
        )
    return _proof(
        HYPERPOD,
        True,
        "every instance group the node selector covers is InService at "
        "TargetCount 0 and CurrentCount 0",
        entries,
        proven_by=["sagemaker:DescribeCluster InstanceGroups"],
    )


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

    eks = is_eks_managed(target)
    try:
        selector = node_selector(target)
        if eks:
            return eks_parked(
                _described_nodegroups(release, target)
                if described is None
                else described,
                selector=selector,
                nodepools=lambda: read_karpenter_nodepools(release, target),
            )
        return hyperpod_parked(
            _instance_groups(release, target) if groups is None else groups,
            cluster_name=str(target.hyperpod_cluster_name or ""),
            selector=selector,
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the reason, never raised
        return _proof(
            EKS_MANAGED if eks else HYPERPOD,
            False,
            f"the provider's node capacity could not be read: {_reason(exc)}",
            [],
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
    cluster without nodes. A node the re-check would drop leaves the cluster to
    the node path, which refuses an empty node set -- the safe direction.
    """

    try:
        listing = release._get_json(
            release._gpu(target, "get", "nodes", "-l", node_selector(target))
        )
        return [
            item for item in listing.get("items") or () if isinstance(item, Mapping)
        ]
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


def state_of(proof: Mapping[str, Any]) -> str:
    """``parked`` for a proof of declared zero capacity, else ``unavailable``."""

    return PARKED if proof.get("parked") is True else UNAVAILABLE


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
    release: Any, checks: Iterable[Any], plan: ReleaseExecutionPlan
) -> dict[str, Any] | None:
    """The failing deploy preflight check of a release that could not finish
    or roll back over the site's clusters without nodes; ``None`` for every
    other release (``zero_node_clusters`` of the preflight's ``checks``):

    * Agent waves while no GPU cluster has a node: a rollback restores the
      control plane's Agent pins from the live Agents the release snapshot
      recorded, and there are none;
    * a Reconciler or Agent roll over a cluster whose node groups declare no
      GPU instance type the installer knows: its installer template is
      rendered from a declared node (``render_node``), and there is none.
    """

    zero_nodes = zero_node_clusters(checks)
    if not zero_nodes:
        return None
    configured = {str(target.cluster_id) for target in release.config.clusters}
    if (
        plan.has(ReleaseComponent.AGENT)
        and configured
        and configured.issubset(zero_nodes)
    ):
        return _refusal(
            zero_nodes,
            "every GPU cluster has zero nodes ("
            + ", ".join(sorted(configured))
            + "): a release with Agent waves snapshots the live Agents its "
            "rollback restores the control plane's Agent pins from, and there is "
            "none. Scale a node group of one cluster back up and let its nodes "
            "join, then deploy again. Nothing was changed.",
        )
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
