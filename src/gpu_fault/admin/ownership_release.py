"""Releasing the ownership label when a GPU cluster is de-registered.

A site selects a GPU cluster's nodes through the node groups that carry
``cluster_scope.OWNERSHIP_LABEL=<site id>`` (``bootstrap_ownership``). The
label is declared once, up front, by the administrator -- and until this
module existed nothing ever took it back: ``remove-cluster`` and ``uninstall``
cleaned the cluster's Kubernetes objects and node annotations and left the
node groups claiming a control plane that had disowned them. A later deploy
of the same site id silently re-adopted them without a fresh declaration, and
another site was refused ("one GPU cluster belongs to one site") until an
administrator edited the label by hand.

Both commands now end with an OWNERSHIP RELEASE step, journaled like every
other step (``cluster_removal_state.OWNERSHIP_RELEASE_STEP``; the uninstall
phase of the same name), that removes exactly ``OWNERSHIP_LABEL`` from exactly
the groups carrying this site's id:

* an EKS managed node group through ``aws eks update-nodegroup-config
  --labels removeLabels=gpu-fault.io/control-plane`` (EKS propagates the
  change to the group's existing nodes);
* a HyperPod instance group through ``aws sagemaker update-cluster`` with the
  group's *full* current definition and ``KubernetesConfig.Labels`` minus the
  key -- ``InstanceCount`` is the current ``TargetCount``, ``ImageId`` and
  ``ImageReleaseVersion`` the image the group runs now (the service documents
  an omitted release version as "the latest available", i.e. a re-image), and
  the step refuses to write (nothing is sent) while the cluster or the group
  is not ``InService``, an operation (``ActiveOperations``) or a software
  update (``SoftwareUpdateStatus``) is in flight or ended ``Failed``,
  ``CurrentCount`` differs from ``TargetCount``, another label, taint, image,
  image release version, instance type or patch schedule change is pending
  (``Desired*`` differing from ``Current*``), the running image release is
  ``EndOfLife``, or the group carries a field this module does not know how to
  carry over, because ``UpdateCluster`` can scale and re-image a group;
* a Karpenter ``NodePool`` whose template labels carry the id is a customer
  provisioning CRD: it is never edited, the step completes with the outcome
  ``manual`` and records the exact ``kubectl patch``.

Every other label is left alone (never ``sagemaker.amazonaws.com/cluster-name``,
never the ``gpu-fault.io/*`` node labels the Kubernetes cleanup strips); a
group labelled for another site is not ours and is only reported. The step
waits, bounded, until the provider shows the key gone (``describe-nodegroup``
labels; HyperPod ``DesiredLabels`` then ``CurrentLabels``, applied
asynchronously) and records before/after evidence per group. A refused write
-- most often an operator identity without ``eks:UpdateNodegroupConfig`` /
``sagemaker:UpdateCluster`` -- fails closed with the manual commands; the
journal resumes at this step. ``--keep-ownership-label`` records the outcome
``retained`` instead, for an administrator who intends to re-join soon.

``stale_ownership_claims`` is the read side: the groups of a de-registered
cluster (a COMPLETED ``remove-cluster`` tombstone, a COMPLETED uninstall) that
still carry this site's id, which ``status`` and ``node-scope`` report as a
finding. Groups a current member reads (the cluster joined again, under any
cluster id) are the member's, never a stale claim.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.bootstrap_ownership import (
    GROUP_KINDS,
    NodeGroupFacts,
    eks_nodegroup_facts,
    hyperpod_instance_group_facts,
)
from gpu_fault.admin.cluster_removal_state import (
    OWNERSHIP_RELEASE_STEP,
    read_removal_state,
)
from gpu_fault.admin.process_supervisor import ensure_supervision_safe
from gpu_fault.admin.site import RenderedSite
from gpu_fault.cluster_scope import OWNERSHIP_LABEL, ClusterProvider, cluster_provider

__all__ = [
    "FOREIGN",
    "add_keep_label_flag",
    "HYPERPOD_DERIVED_FIELDS",
    "HYPERPOD_IMAGE_END_OF_LIFE",
    "HYPERPOD_REQUIRED_FIELDS",
    "HYPERPOD_SETTLED_SOFTWARE_UPDATES",
    "HYPERPOD_SPEC_FIELDS",
    "HYPERPOD_STATUS_FIELDS",
    "KARPENTER_NODEPOOL_RESOURCE",
    "KEEP_FLAG",
    "MANUAL",
    "OPERATOR_ACTIONS",
    "RELEASED",
    "RETAINED",
    "STEP",
    "UNLABELLED",
    "ClusterGroups",
    "deregistered_clusters",
    "describe_cluster_groups",
    "hyperpod_current_labels",
    "hyperpod_instance_group_specification",
    "hyperpod_write_refusal",
    "nodepool_release_edit",
    "release_cluster_ownership",
    "release_command",
    "require_hyperpod_writable",
    "stale_claim_lines",
    "stale_ownership_claims",
    "status_findings",
    "wait_for_label",
]

STEP = OWNERSHIP_RELEASE_STEP
#: Outcomes of the step (``outcome`` of the evidence and of each group).
RELEASED = "released"
RETAINED = "retained"
MANUAL = "manual"
#: Per-group outcomes for groups the step had nothing to release from.
UNLABELLED = "unlabelled"
FOREIGN = "foreign"
KEEP_FLAG = "--keep-ownership-label"
#: The write action the operator's AWS identity needs per provider; the
#: runtime roles (``bootstrap_iam_policies``) stay Describe-only.
OPERATOR_ACTIONS: dict[str, str] = {
    ClusterProvider.EKS_MANAGED.value: "eks:UpdateNodegroupConfig",
    ClusterProvider.HYPERPOD.value: "sagemaker:UpdateCluster",
}
KARPENTER_NODEPOOL_RESOURCE = "nodepools.karpenter.sh"
#: HyperPod applies ``KubernetesConfig`` labels asynchronously (about a minute
#: observed); an EKS node group update takes a similar time. Ten minutes is
#: the bound, not the expectation.
CONVERGENCE_TIMEOUT_SECONDS = 600.0
CONVERGENCE_INTERVAL_SECONDS = 10.0
#: Every ``ClusterInstanceGroupDetails`` member, classified against the
#: installed botocore service model (``ClusterInstanceGroupDetails`` against
#: ``ClusterInstanceGroupSpecification``; ``tests/admin/
#: test_admin_ownership_release_fields.py`` pins the three tables to it). A
#: details field in none of them refuses the write: this module carries a
#: definition it understands or none at all.
#:
#: Carried verbatim -- the same name and the same shape in both structures.
HYPERPOD_SPEC_FIELDS: tuple[str, ...] = (
    "InstanceGroupName",
    "InstanceType",
    "LifeCycleConfig",
    "ExecutionRole",
    "ThreadsPerCore",
    "InstanceStorageConfigs",
    "OnStartDeepHealthChecks",
    "TrainingPlanArn",
    "OverrideVpcConfig",
    "ScheduledUpdateConfig",
    "CapacityRequirements",
    "SlurmConfig",
    "NetworkInterface",
)
#: Derived -- the details field (key) yields the specification field (value)
#: from the group's *current* state: ``InstanceCount`` is ``TargetCount``,
#: ``ImageId``/``ImageReleaseVersion`` the image the group runs now (the
#: service documents an omitted ``ImageReleaseVersion`` as "the latest
#: available", i.e. a re-image, so the running one is always named), and
#: ``KubernetesConfig``/``InstanceRequirements``/``AutoPatchConfig`` take their
#: ``Current*`` members. A ``Desired*`` member differing from its ``Current*``
#: refuses the write first (``require_hyperpod_writable``).
HYPERPOD_DERIVED_FIELDS: dict[str, str] = {
    "TargetCount": "InstanceCount",
    "MinCount": "MinInstanceCount",
    "CurrentImageId": "ImageId",
    "CurrentImageReleaseVersion": "ImageReleaseVersion",
    "KubernetesConfig": "KubernetesConfig",
    "InstanceRequirements": "InstanceRequirements",
    "AutoPatchConfig": "AutoPatchConfig",
}
#: Read only -- the group's state, never part of the specification.
HYPERPOD_STATUS_FIELDS: tuple[str, ...] = (
    "CurrentCount",
    "Status",
    "ActiveOperations",
    "TargetStateCount",
    "SoftwareUpdateStatus",
    "ActiveSoftwareUpdateConfig",
    "ImageVersionStatus",
    "DesiredImageId",
    "DesiredImageReleaseVersion",
    "InstanceTypeDetails",
    "TrainingPlanStatus",
)
#: Besides these, a group needs ``InstanceType`` or ``InstanceRequirements``
#: and an integer ``TargetCount``.
HYPERPOD_REQUIRED_FIELDS: tuple[str, ...] = (
    "InstanceGroupName",
    "LifeCycleConfig",
    "ExecutionRole",
)
#: ``SoftwareUpdateStatus`` values after which the last software update has
#: settled. ``Pending``, ``InProgress`` and ``RollbackInProgress`` are in
#: flight; ``Failed`` is a rollback that did not complete, after which the
#: nodes may run mixed images -- both refuse the write.
HYPERPOD_SETTLED_SOFTWARE_UPDATES: frozenset[str] = frozenset(
    {"Succeeded", "RollbackComplete"}
)
#: The ``ImageVersionStatus`` that refuses the write: the service may no longer
#: accept the running release as ``ImageReleaseVersion`` and could substitute
#: the latest one (a re-image). ``UpdateAvailable`` and
#: ``SecurityUpdateRequired`` only say a newer release exists; naming the
#: running release keeps the group where it is, so they do not refuse.
HYPERPOD_IMAGE_END_OF_LIFE = "EndOfLife"
_NODEPOOL_LABEL_POINTER = "/spec/template/metadata/labels/" + OWNERSHIP_LABEL.replace(
    "~", "~0"
).replace("/", "~1")


@dataclass(frozen=True)
class ClusterGroups:
    """One GPU cluster's node groups as the provider describes them now."""

    provider: ClusterProvider
    #: The provider's bare cluster name: EKS cluster name or HyperPod name.
    cluster: str
    facts: tuple[NodeGroupFacts, ...]
    #: The raw provider document per group name (the HyperPod full definition).
    documents: Mapping[str, Mapping[str, Any]]
    #: HyperPod ``ClusterStatus``; empty for EKS.
    cluster_status: str = ""

    @property
    def display(self) -> str:
        kind = "HyperPod" if self.provider is ClusterProvider.HYPERPOD else "EKS"
        return f"{kind} cluster {self.cluster}"

    @property
    def group_kind(self) -> str:
        return GROUP_KINDS[self.provider.value]

    def owned_by(self, site_id: str) -> list[NodeGroupFacts]:
        return [item for item in self.facts if item.ownership == site_id]


def _text_map(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _describe_target(target: Mapping[str, Any]) -> tuple[ClusterProvider, str]:
    """``(provider, name)`` ``describe_cluster_groups`` reads ``target`` by.

    The EKS cluster name of a plain EKS cluster, the HyperPod cluster name of a
    HyperPod one; the Region is the caller's, so within one site two targets
    with the same pair read the same node groups.
    """

    provider = cluster_provider(target.get("provider"))
    if provider is ClusterProvider.EKS_MANAGED:
        return provider, str(target["eks_cluster_arn"]).rsplit("/", 1)[-1]
    return provider, str(target.get("hyperpod_cluster_name") or "")


def describe_cluster_groups(
    runner: CommandRunner, *, region: str, target: Mapping[str, Any]
) -> ClusterGroups:
    """The node groups of ``target`` (a site cluster entry) as described now."""

    provider, cluster = _describe_target(target)
    if provider is ClusterProvider.EKS_MANAGED:
        eks_name = cluster
        names = sorted(
            str(item)
            for item in runner.aws_json(
                region, "eks", "list-nodegroups", "--cluster-name", eks_name
            ).get("nodegroups", [])
            if item
        )
        documents: dict[str, Mapping[str, Any]] = {}
        for name in names:
            described = runner.aws_json(
                region,
                "eks",
                "describe-nodegroup",
                "--cluster-name",
                eks_name,
                "--nodegroup-name",
                name,
            ).get("nodegroup")
            documents[name] = described if isinstance(described, Mapping) else {}
        return ClusterGroups(
            provider=provider,
            cluster=eks_name,
            facts=tuple(eks_nodegroup_facts(documents[name]) for name in names),
            documents=documents,
        )
    hyperpod_name = cluster
    described = runner.aws_json(
        region, "sagemaker", "describe-cluster", "--cluster-name", hyperpod_name
    )
    groups = [
        group
        for group in described.get("InstanceGroups") or ()
        if isinstance(group, Mapping)
    ]
    return ClusterGroups(
        provider=provider,
        cluster=hyperpod_name,
        facts=tuple(
            hyperpod_instance_group_facts(group, cluster_name=hyperpod_name)
            for group in groups
        ),
        documents={
            str(group.get("InstanceGroupName") or ""): group for group in groups
        },
        cluster_status=str(described.get("ClusterStatus") or ""),
    )


def add_keep_label_flag(parser: Any, *, scope: str) -> None:
    """``--keep-ownership-label`` on ``remove-cluster`` and ``uninstall``.

    ``scope`` names what the command would release: the node groups of the
    removed cluster, or every GPU cluster's node groups for an uninstall.
    """

    parser.add_argument(
        KEEP_FLAG,
        action="store_true",
        help=(
            f"keep {OWNERSHIP_LABEL}=<site id> on {scope} (recorded as retained) "
            "instead of releasing it in the last step; for a re-join soon. May be "
            "added on a resume stuck at the release step"
        ),
    )


# --- the manual commands ----------------------------------------------------------


def release_command(
    provider: ClusterProvider | str, *, cluster: str, group: str
) -> str:
    """The provider command that removes exactly the ownership label from ``group``."""

    if cluster_provider(provider) is ClusterProvider.HYPERPOD:
        return (
            f"aws sagemaker update-cluster --cluster-name {cluster} "
            f"--instance-groups '<the full current definition of {group} with "
            f"KubernetesConfig.Labels minus {OWNERSHIP_LABEL}>'"
        )
    return (
        f"aws eks update-nodegroup-config --cluster-name {cluster} "
        f"--nodegroup-name {group} --labels removeLabels={OWNERSHIP_LABEL}"
    )


def nodepool_release_edit(kubectl: Sequence[str], name: str) -> str:
    """The ``kubectl patch`` that drops the label from a Karpenter NodePool template."""

    patch = json.dumps(
        [{"op": "remove", "path": _NODEPOOL_LABEL_POINTER}], separators=(",", ":")
    )
    return " ".join(
        [*kubectl, "patch", "nodepool", name, "--type=json", "-p", f"'{patch}'"]
    )


# --- HyperPod full definitions ----------------------------------------------------


def hyperpod_instance_group_specification(
    details: Mapping[str, Any], *, labels: Mapping[str, str]
) -> dict[str, Any]:
    """The ``ClusterInstanceGroupSpecification`` that keeps ``details`` as it is.

    ``HYPERPOD_SPEC_FIELDS`` are copied verbatim. ``InstanceCount`` is the
    current ``TargetCount`` (never the live ``CurrentCount``: a group mid-scale
    would be pinned at the wrong size) and ``MinInstanceCount`` the
    ``MinCount``. ``ImageId`` and ``ImageReleaseVersion`` are the image the
    group runs now -- ``CurrentImageId`` (``default`` stays ``default``, an AMI
    id stays that AMI) and ``CurrentImageReleaseVersion`` -- whenever the
    description reports them: the service documents an omitted release version
    as "the latest available", which is a re-image, so the running one is
    named explicitly. ``InstanceRequirements`` takes the
    ``CurrentInstanceTypes``, ``AutoPatchConfig`` the ``CurrentPatchSchedule``,
    ``KubernetesConfig.Taints`` the current taints and
    ``KubernetesConfig.Labels`` exactly ``labels``. A missing required field or
    a field outside the known tables raises: a definition this module cannot
    carry over in full must not be sent at all.
    """

    name = str(details.get("InstanceGroupName") or "<instance group>")
    missing = [key for key in HYPERPOD_REQUIRED_FIELDS if not details.get(key)]
    if not details.get("InstanceType") and not details.get("InstanceRequirements"):
        missing.append("InstanceType (or InstanceRequirements)")
    if missing:
        raise BootstrapError(
            f"instance group {name} lacks {', '.join(missing)}; update-cluster "
            "needs the full definition"
        )
    unknown = sorted(
        key
        for key in details
        if key not in HYPERPOD_SPEC_FIELDS
        and key not in HYPERPOD_DERIVED_FIELDS
        and key not in HYPERPOD_STATUS_FIELDS
    )
    if unknown:
        raise BootstrapError(
            f"instance group {name} carries {', '.join(unknown)}, which this "
            "release does not know how to carry into update-cluster"
        )
    target_count = details.get("TargetCount")
    if type(target_count) is not int or target_count < 0:
        raise BootstrapError(f"instance group {name} has no integer TargetCount")
    specification: dict[str, Any] = {
        key: details[key]
        for key in HYPERPOD_SPEC_FIELDS
        if details.get(key) is not None
    }
    specification["InstanceCount"] = target_count
    if details.get("MinCount") is not None:
        specification["MinInstanceCount"] = details["MinCount"]
    for source, field in (
        ("CurrentImageId", "ImageId"),
        ("CurrentImageReleaseVersion", "ImageReleaseVersion"),
    ):
        if details.get(source):
            specification[field] = details[source]
    requirements = details.get("InstanceRequirements")
    if isinstance(requirements, Mapping) and (
        requirements.get("CurrentInstanceTypes") is not None
    ):
        specification["InstanceRequirements"] = {
            "InstanceTypes": requirements["CurrentInstanceTypes"]
        }
    patching = details.get("AutoPatchConfig")
    if isinstance(patching, Mapping):
        specification["AutoPatchConfig"] = _auto_patch_specification(patching)
    kubernetes: dict[str, Any] = {"Labels": dict(labels)}
    current = details.get("KubernetesConfig")
    if isinstance(current, Mapping) and current.get("CurrentTaints") is not None:
        kubernetes["Taints"] = current["CurrentTaints"]
    specification["KubernetesConfig"] = kubernetes
    return specification


def _auto_patch_specification(details: Mapping[str, Any]) -> dict[str, Any]:
    """``ClusterAutoPatchConfig`` from its details: the *current* schedule."""

    specification = {
        key: details[key]
        for key in ("PatchingStrategy", "DeploymentConfig")
        if details.get(key) is not None
    }
    if details.get("CurrentPatchSchedule") is not None:
        specification["PatchSchedule"] = details["CurrentPatchSchedule"]
    return specification


def hyperpod_current_labels(details: Mapping[str, Any]) -> dict[str, str]:
    """The labels an instance group's nodes carry now (``CurrentLabels``)."""

    return _hyperpod_labels(details)[0]


def _hyperpod_labels(
    details: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str] | None]:
    """``(CurrentLabels, DesiredLabels or None)`` of one instance group."""

    kubernetes = details.get("KubernetesConfig")
    kubernetes = kubernetes if isinstance(kubernetes, Mapping) else {}
    current = _text_map(kubernetes.get("CurrentLabels"))
    if not current and "CurrentLabels" not in kubernetes:
        current = _text_map(kubernetes.get("Labels"))
    desired = (
        _text_map(kubernetes.get("DesiredLabels"))
        if "DesiredLabels" in kubernetes
        else None
    )
    return current, desired


def _refuse(groups: ClusterGroups, group: str, reason: str) -> BootstrapError:
    return BootstrapError(
        f"cannot release {OWNERSHIP_LABEL} from {groups.group_kind} {group} of "
        f"{groups.display}: {reason}; nothing was written. Release it by hand with "
        + release_command(groups.provider, cluster=groups.cluster, group=group)
        + f" once the update has finished, or rerun with {KEEP_FLAG}"
    )


def require_hyperpod_writable(groups: ClusterGroups, names: Sequence[str]) -> None:
    """Refuse while UpdateCluster could do more than drop one label.

    The cluster and the group must be ``InService`` with nothing in flight
    (``ActiveOperations`` empty, the last software update settled) and nothing
    pending (``CurrentCount`` equal to ``TargetCount``, every ``Desired*`` equal
    to its ``Current*``: labels, taints, image, image release version,
    instance types, patch schedule), and the running image release must not be
    ``EndOfLife``. ``TargetStateCount`` (the nodes on the desired image) is
    informational: recorded per group in the evidence, never a refusal.
    """

    for name in names:
        reason = hyperpod_write_refusal(
            groups.cluster_status, groups.documents.get(name) or {}
        )
        if reason is not None:
            raise _refuse(groups, name, reason)


def hyperpod_write_refusal(
    cluster_status: str,
    details: Mapping[str, Any],
    *,
    allow_label_release: bool = True,
) -> str | None:
    """Why ``UpdateCluster`` with the group's full definition could do more than
    the one change it carries, or ``None``: what ``require_hyperpod_writable``
    refuses on. ``allow_label_release=False`` (a definition that keeps the
    ``CurrentLabels``, ``hyperpod_scheduled_update``) also refuses while this
    module's own release of the ownership label is still converging -- the
    definition would put the label back."""

    if cluster_status != "InService":
        return (
            f"the cluster is {cluster_status or 'in an unknown state'} (an "
            "update is in progress; UpdateCluster can scale or re-image a group)"
        )
    return _hyperpod_unsettled(details) or _hyperpod_change_pending(
        details, allow_label_release=allow_label_release
    )


def _hyperpod_unsettled(details: Mapping[str, Any]) -> str | None:
    """Why the group has an update or operation in flight, or ``None``."""

    status = str(details.get("Status") or "InService")
    if status != "InService":
        return f"the instance group is {status}"
    operations = details.get("ActiveOperations")
    if operations:
        return (
            f"ActiveOperations {json.dumps(operations, sort_keys=True)} (an "
            "operation is in flight)"
        )
    software = details.get("SoftwareUpdateStatus")
    if software and software not in HYPERPOD_SETTLED_SOFTWARE_UPDATES:
        why = (
            "its rollback did not complete; the nodes may run mixed images"
            if software == "Failed"
            else "a software update is in flight"
        )
        return f"the last software update is {software} ({why})"
    current_count, target_count = (
        details.get("CurrentCount"),
        details.get("TargetCount"),
    )
    if current_count != target_count:
        return (
            f"CurrentCount {current_count} differs from TargetCount {target_count} "
            "(the group is scaling)"
        )
    return None


def _hyperpod_change_pending(
    details: Mapping[str, Any], *, allow_label_release: bool = True
) -> str | None:
    """Why a change other than this release is pending, or ``None``.

    Every ``Desired*`` the description carries must equal its ``Current*``
    (the release's own pending label change excepted, unless
    ``allow_label_release`` is false), and the running image release must be
    one the service still offers, or ``UpdateCluster`` could apply more than
    the label.
    """

    current, desired = _hyperpod_labels(details)
    if desired is not None and desired != current:
        pending_release = dict(current)
        pending_release.pop(OWNERSHIP_LABEL, None)
        if not allow_label_release or desired != pending_release:
            return (
                f"DesiredLabels {json.dumps(desired, sort_keys=True)} differ from "
                f"CurrentLabels {json.dumps(current, sort_keys=True)} (a label "
                "change is still being applied)"
            )
    kubernetes = details.get("KubernetesConfig")
    kubernetes = kubernetes if isinstance(kubernetes, Mapping) else {}
    if "DesiredTaints" in kubernetes and kubernetes.get("DesiredTaints") != (
        kubernetes.get("CurrentTaints")
    ):
        return "DesiredTaints differ from CurrentTaints"
    if details.get("DesiredImageId") and details.get("DesiredImageId") != (
        details.get("CurrentImageId")
    ):
        return "a group image change is pending"
    desired_release = details.get("DesiredImageReleaseVersion")
    running_release = details.get("CurrentImageReleaseVersion")
    if desired_release and desired_release != running_release:
        return (
            f"DesiredImageReleaseVersion {desired_release} differs from "
            f"CurrentImageReleaseVersion {running_release} (an image release "
            "change is pending)"
        )
    if details.get("ImageVersionStatus") == HYPERPOD_IMAGE_END_OF_LIFE:
        return (
            f"the running image release {running_release} is "
            f"{HYPERPOD_IMAGE_END_OF_LIFE} (the service may not accept it as "
            "ImageReleaseVersion and could substitute the latest release, a "
            "re-image); update the cluster software first"
        )
    for field, current_key, desired_key in (
        ("InstanceRequirements", "CurrentInstanceTypes", "DesiredInstanceTypes"),
        ("AutoPatchConfig", "CurrentPatchSchedule", "DesiredPatchSchedule"),
    ):
        value = details.get(field)
        if (
            isinstance(value, Mapping)
            and desired_key in value
            and (value.get(desired_key) != value.get(current_key))
        ):
            return f"{field}.{desired_key} differs from {current_key}"
    return None


def _hyperpod_release_specifications(
    groups: ClusterGroups, names: Sequence[str]
) -> list[dict[str, Any]]:
    specifications = []
    for name in names:
        details = groups.documents.get(name) or {}
        current, _desired = _hyperpod_labels(details)
        labels = {
            key: value for key, value in current.items() if key != OWNERSHIP_LABEL
        }
        try:
            specifications.append(
                hyperpod_instance_group_specification(details, labels=labels)
            )
        except BootstrapError as exc:
            raise _refuse(groups, name, str(exc)) from exc
    return specifications


# --- Karpenter NodePools ----------------------------------------------------------


def _labelled_nodepools(
    runner: CommandRunner, kubectl: Sequence[str], *, site_id: str
) -> list[dict[str, Any]]:
    """Karpenter NodePools whose template labels carry this site's id.

    A cluster without the Karpenter CRD answers "the server doesn't have a
    resource type"; anything else is an error of the read and fails closed.
    """

    try:
        raw = runner.run(
            [*kubectl, "get", KARPENTER_NODEPOOL_RESOURCE, "-o", "json"],
            capture=True,
            timeout_seconds=60,
        )
    except BootstrapError as exc:
        if "doesn't have a resource type" in str(exc):
            return []
        raise
    try:
        items = json.loads(raw).get("items") if raw.strip() else []
    except (json.JSONDecodeError, AttributeError) as exc:
        raise BootstrapError("the Karpenter NodePool list is unreadable") from exc
    found = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, Mapping):
            continue
        name = str(((item.get("metadata") or {}).get("name")) or "")
        template = ((item.get("spec") or {}).get("template") or {}).get(
            "metadata"
        ) or {}
        labels = _text_map(template.get("labels"))
        if name and labels.get(OWNERSHIP_LABEL) == site_id:
            found.append(
                {
                    "name": name,
                    "before": site_id,
                    "outcome": MANUAL,
                    "edit": nodepool_release_edit(kubectl, name),
                }
            )
    return found


# --- convergence ----------------------------------------------------------------


def wait_for_label(
    runner: CommandRunner,
    *,
    region: str,
    target: Mapping[str, Any],
    names: Sequence[str],
    value: str | None,
    description: str,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    timeout_seconds: float = CONVERGENCE_TIMEOUT_SECONDS,
    interval_seconds: float = CONVERGENCE_INTERVAL_SECONDS,
) -> tuple[float, ClusterGroups]:
    """Wait, bounded, until every group in ``names`` shows the label as ``value``.

    ``None`` means absent. An EKS group converges when ``describe-nodegroup``
    shows the labels and the group is ``ACTIVE`` again; a HyperPod group when
    both ``DesiredLabels`` and ``CurrentLabels`` agree. Returns the seconds
    waited and the last description.
    """

    started = clock()
    while True:
        ensure_supervision_safe()
        groups = describe_cluster_groups(runner, region=region, target=target)
        if all(_label_converged(groups, name, value) for name in names):
            return clock() - started, groups
        elapsed = clock() - started
        if elapsed >= timeout_seconds:
            pending = [
                name for name in names if not _label_converged(groups, name, value)
            ]
            raise BootstrapError(
                f"{description} did not converge within {timeout_seconds:.0f}s on "
                f"{groups.group_kind}(s) {', '.join(pending)} of {groups.display}; "
                "the provider is still applying it -- rerun the command, the "
                "journal resumes at this step"
            )
        sleep(min(interval_seconds, max(timeout_seconds - elapsed, 0.0)))


def _label_converged(groups: ClusterGroups, name: str, value: str | None) -> bool:
    details = groups.documents.get(name)
    if details is None:
        # A group that disappeared carries no label any more.
        return value is None
    if groups.provider is ClusterProvider.EKS_MANAGED:
        labels = _text_map(details.get("labels"))
        return labels.get(OWNERSHIP_LABEL) == value and (
            str(details.get("status") or "ACTIVE") == "ACTIVE"
        )
    current, desired = _hyperpod_labels(details)
    return current.get(OWNERSHIP_LABEL) == value and (
        desired is None or desired.get(OWNERSHIP_LABEL) == value
    )


# --- the step -------------------------------------------------------------------


def _group_record(
    groups: ClusterGroups, item: NodeGroupFacts, site_id: str
) -> dict[str, Any]:
    if item.ownership == site_id:
        outcome = RELEASED
    elif item.ownership is None:
        outcome = UNLABELLED
    else:
        outcome = FOREIGN
    record: dict[str, Any] = {
        "name": item.name,
        "kind": groups.group_kind,
        "before": item.ownership,
        "after": item.ownership,
        "outcome": outcome,
    }
    details = groups.documents.get(item.name) or {}
    if "TargetStateCount" in details:
        # Informational: the nodes running the desired image (HyperPod); the
        # operator reads it from the evidence, the step never refuses on it.
        record["target_state_count"] = details["TargetStateCount"]
    return record


def release_cluster_ownership(
    runner: CommandRunner,
    *,
    region: str,
    site_id: str,
    target: Mapping[str, Any],
    keep: bool = False,
    kubectl: Sequence[str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    timeout_seconds: float = CONVERGENCE_TIMEOUT_SECONDS,
    interval_seconds: float = CONVERGENCE_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Run the OWNERSHIP RELEASE step for one cluster; return its evidence.

    Idempotent: a group that no longer carries the label needs no write, a
    HyperPod group whose ``DesiredLabels`` already lack the key (a previous
    attempt's release still converging) is only waited for. ``kubectl`` is
    the cluster's own context prefix, used for the Karpenter NodePool read of
    a plain EKS cluster; a HyperPod cluster's node groups are its instance
    groups and no NodePool can carry its selector's cross-check label.
    """

    groups = describe_cluster_groups(runner, region=region, target=target)
    owned = groups.owned_by(site_id)
    nodepools: list[dict[str, Any]] = []
    if kubectl is not None and groups.provider is ClusterProvider.EKS_MANAGED:
        nodepools = _labelled_nodepools(runner, kubectl, site_id=site_id)
    records = {item.name: _group_record(groups, item, site_id) for item in groups.facts}
    evidence: dict[str, Any] = {
        "label": OWNERSHIP_LABEL,
        "site_id": site_id,
        "provider": groups.provider.value,
        "cluster": groups.cluster,
        "operator_action": OPERATOR_ACTIONS[groups.provider.value],
        "groups": list(records.values()),
        "nodepools": nodepools,
    }
    if keep:
        for item in owned:
            records[item.name]["outcome"] = RETAINED
        for pool in nodepools:
            pool["outcome"] = RETAINED
            pool.pop("edit", None)
        evidence["outcome"] = RETAINED if owned or nodepools else RELEASED
        evidence["reason"] = (
            f"{KEEP_FLAG}: the administrator keeps the label for a re-join"
            if owned or nodepools
            else "no node group carried the label"
        )
        return evidence
    names = [item.name for item in owned]
    if names:
        for name in names:
            records[name]["command"] = release_command(
                groups.provider, cluster=groups.cluster, group=name
            )
        to_write = _write_release(runner, groups, names, region=region)
        waited, _after = wait_for_label(
            runner,
            region=region,
            target=target,
            names=names,
            value=None,
            description=f"the release of {OWNERSHIP_LABEL}",
            sleep=sleep,
            clock=clock,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
        )
        for name in names:
            records[name]["after"] = None
            records[name]["written"] = name in to_write
        evidence["converged_after_seconds"] = round(waited, 3)
    evidence["outcome"] = MANUAL if nodepools else RELEASED
    return evidence


def _write_release(
    runner: CommandRunner, groups: ClusterGroups, names: Sequence[str], *, region: str
) -> list[str]:
    """Send the provider writes for ``names``; return the groups actually written."""

    action = OPERATOR_ACTIONS[groups.provider.value]
    if groups.provider is ClusterProvider.EKS_MANAGED:
        for name in names:
            status = str((groups.documents.get(name) or {}).get("status") or "ACTIVE")
            if status != "ACTIVE":
                raise _refuse(groups, name, f"the node group is {status}")
        written = []
        for name in names:
            _aws_write(
                runner,
                groups,
                name,
                region=region,
                action=action,
                arguments=(
                    "eks",
                    "update-nodegroup-config",
                    "--cluster-name",
                    groups.cluster,
                    "--nodegroup-name",
                    name,
                    "--labels",
                    f"removeLabels={OWNERSHIP_LABEL}",
                ),
            )
            written.append(name)
        return written
    require_hyperpod_writable(groups, names)
    pending = []
    for name in names:
        _current, desired = _hyperpod_labels(groups.documents.get(name) or {})
        if desired is None or OWNERSHIP_LABEL in desired:
            pending.append(name)
    if not pending:
        return []
    specifications = _hyperpod_release_specifications(groups, pending)
    _aws_write(
        runner,
        groups,
        ", ".join(pending),
        region=region,
        action=action,
        arguments=(
            "sagemaker",
            "update-cluster",
            "--cluster-name",
            groups.cluster,
            "--instance-groups",
            json.dumps(specifications, separators=(",", ":")),
        ),
    )
    return pending


def _aws_write(
    runner: CommandRunner,
    groups: ClusterGroups,
    group: str,
    *,
    region: str,
    action: str,
    arguments: Sequence[str],
) -> None:
    try:
        runner.aws_json(region, *arguments, mutate=True)
    except BootstrapError as exc:
        commands = "; ".join(
            release_command(groups.provider, cluster=groups.cluster, group=name)
            for name in group.split(", ")
        )
        raise BootstrapError(
            f"could not release {OWNERSHIP_LABEL} from {groups.group_kind} {group} of "
            f"{groups.display}: {exc}. The operator's AWS identity needs {action} on "
            f"the cluster; release the label by hand with {commands} and rerun, or "
            f"rerun with {KEEP_FLAG} to record it as retained"
        ) from exc


# --- stale claims -----------------------------------------------------------------


def _cluster_entry(target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cluster_id": str(target.get("cluster_id") or ""),
        "provider": cluster_provider(target.get("provider")).value,
        "eks_cluster_arn": str(target.get("eks_cluster_arn") or ""),
        "hyperpod_cluster_name": str(target.get("hyperpod_cluster_name") or ""),
        "target": dict(target),
    }


def deregistered_clusters(site: RenderedSite) -> list[dict[str, Any]]:
    """The clusters this site de-registered and what their journals recorded.

    COMPLETED ``remove-cluster`` tombstones (one per cluster id) and, after a
    COMPLETED uninstall, every cluster the site still lists. A tombstone whose
    describe target (``_describe_target``: the EKS cluster of a plain EKS
    cluster, the HyperPod cluster name of a HyperPod one, compared normalized
    -- a HyperPod entry carries no ``provider``, a journal may record
    ``None``) is a current member's is superseded, whatever cluster id either
    was registered under: the groups it would read carry the member's label,
    which its release command would strip. A tombstone of another provider
    cluster stays reported, also when the site reuses its cluster id. The
    next ``deploy`` archives a COMPLETED uninstall together with ``site.yaml``
    and the tombstones (``deploy_command.RETIRED_AFTER_UNINSTALL``), so the
    uninstall branch never reads a re-deployed site. A journal that cannot be
    read is reported with its error rather than skipped.
    """

    root = site.source.parent
    members = {
        _describe_target(_cluster_entry(item))
        for item in site.release_config["clusters"]
    }
    items: list[dict[str, Any]] = []
    for path in sorted((root / "remove-cluster").glob("*/state.json")):
        try:
            state = read_removal_state(path)
        except BootstrapError as exc:
            items.append(
                {
                    "cluster_id": path.parent.name,
                    "deregistered_by": "remove-cluster",
                    "error": str(exc),
                }
            )
            continue
        if state.get("phase") != "COMPLETED":
            continue
        entry = _cluster_entry(state["target"])
        described = _describe_target(entry)
        if described[1] and described in members:
            continue
        recorded = (state.get("evidence") or {}).get(STEP)
        items.append(
            {
                **entry,
                "deregistered_by": "remove-cluster",
                "recorded_outcome": (
                    recorded.get("outcome") if isinstance(recorded, Mapping) else None
                ),
            }
        )
    uninstall = root / "uninstall" / "state.json"
    if uninstall.is_file():
        try:
            value = json.loads(uninstall.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return [*items, {"deregistered_by": "uninstall", "error": str(exc)}]
        if isinstance(value, dict) and value.get("phase") == "COMPLETED":
            recorded = value.get("ownership_release")
            per_cluster = {
                str(item.get("cluster_id") or ""): item.get("outcome")
                for item in (
                    recorded.get("clusters") if isinstance(recorded, Mapping) else []
                )
                or []
                if isinstance(item, Mapping)
            }
            for target in site.release_config["clusters"]:
                entry = _cluster_entry(target)
                items.append(
                    {
                        **entry,
                        "deregistered_by": "uninstall",
                        "recorded_outcome": per_cluster.get(entry["cluster_id"]),
                    }
                )
    return items


def stale_ownership_claims(
    site: RenderedSite, runner: CommandRunner
) -> list[dict[str, Any]]:
    """De-registered clusters whose node groups still carry this site's id."""

    site_id = str(site.release_config["site_name"])
    region = str(site.release_config["aws_region"])
    findings: list[dict[str, Any]] = []
    for item in deregistered_clusters(site):
        if "error" in item:
            findings.append(
                {**item, "site_id": site_id, "groups": [], "release_commands": []}
            )
            continue
        try:
            groups = describe_cluster_groups(
                runner, region=region, target=item["target"]
            )
        except BootstrapError as exc:
            findings.append(
                {
                    **{key: value for key, value in item.items() if key != "target"},
                    "site_id": site_id,
                    "error": str(exc),
                    "groups": [],
                    "release_commands": [],
                }
            )
            continue
        stale = [facts.name for facts in groups.owned_by(site_id)]
        if not stale:
            continue
        findings.append(
            {
                **{key: value for key, value in item.items() if key != "target"},
                "site_id": site_id,
                "cluster": groups.cluster,
                "group_kind": groups.group_kind,
                "groups": stale,
                "release_commands": [
                    release_command(groups.provider, cluster=groups.cluster, group=name)
                    for name in stale
                ],
            }
        )
    return findings


def stale_claim_lines(findings: Sequence[Mapping[str, Any]]) -> list[str]:
    """One readable line per finding, for ``status`` and ``node-scope``."""

    lines = []
    for finding in findings:
        if finding.get("error"):
            lines.append(
                "stale ownership claim check: "
                f"{finding.get('cluster_id') or finding.get('deregistered_by')} could "
                f"not be read: {finding['error']}"
            )
            continue
        lines.append(
            f"stale ownership claim: {finding['deregistered_by']} de-registered "
            f"{finding['cluster_id']} ({finding['provider']} {finding['cluster']}) but "
            f"{finding['group_kind']}(s) {', '.join(finding['groups'])} still carry "
            f"{OWNERSHIP_LABEL}={finding['site_id']}; release with "
            + "; ".join(finding["release_commands"])
        )
    return lines


def status_findings(
    site: RenderedSite, runner: CommandRunner | None = None
) -> list[str]:
    """The stale-claim lines for ``status``; a failed read is itself a line."""

    try:
        return stale_claim_lines(
            stale_ownership_claims(site, runner or CommandRunner())
        )
    except (BootstrapError, OSError, ValueError) as exc:
        return [f"stale ownership claim check failed: {exc}"]
