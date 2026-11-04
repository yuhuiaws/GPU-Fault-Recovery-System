from __future__ import annotations

import base64
import ipaddress
import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease, Runner

from gpu_fault.admin.automatic_node_updates import (
    ASG_QUERY,
    CONTROL_PLANE,
    FIELD as AUTOMATIC_NODE_UPDATES,
    KARPENTER_NODEPOOLS,
    control_plane_updates,
    eks_node_group_updates,
    guarded,
    hyperpod_updates,
    karpenter_updates,
)
from gpu_fault.admin.bootstrap_ownership import hyperpod_groups_for_selector
from gpu_fault.admin.eks_nodegroup_update_strategy import update_strategy
from gpu_fault.cluster_scope import is_eks_managed, node_selector
from gpu_fault.eks_node_repair import (
    hosts_gpu_instance_types,
    node_repair_error,
    node_repair_state,
    nodegroup_hosts_gpu_nodes,
)
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_monitoring_safety as monitoring_safety
from gpu_fault_release.regional_observability_rollback import (
    AMP_FAILED_STATUSES,
    AMP_SETTLING_STATUSES,
    AmpDefinition,
    amp_definition_status,
    describe_amp_definition,
)
from gpu_fault_release.regional_release_config import (
    EKS_ARN_PATTERN,
    ClusterTarget,
    ReleaseError,
)
from gpu_fault_release.regional_release_eks_support import (
    FIELD as EKS_SUPPORT,
    read_eks_support,
)
from gpu_fault_release.regional_release_state import aws_json
from gpu_fault_release.regional_verify_scope import gates, record_not_gating


def require_cpu_namespace_anchor(
    release: RegionalRelease, *, bootstrap: bool = False
) -> None:
    """Absence is useful only in a proved CPU namespace, never on a read error."""
    namespace = release.config.namespace
    anchor = release._get_json(
        release._cpu("get", "namespace", namespace)
        if bootstrap
        else release._cpu("-n", namespace, "get", "deployment", "gpu-fault-api-ha")
    )
    metadata = anchor.get("metadata") or {}
    if (
        anchor.get("kind") != ("Namespace" if bootstrap else "Deployment")
        or metadata.get("name") != (namespace if bootstrap else "gpu-fault-api-ha")
        or (not bootstrap and metadata.get("namespace") != namespace)
        or metadata.get("deletionTimestamp")
    ):
        raise ReleaseError("missing resource has no verified CPU namespace anchor")


def monitoring_repair_preflight(
    release: RegionalRelease, *, bootstrap: bool = False
) -> dict[str, Any]:
    """Verify infrastructure and readable versioned definitions before repair."""
    health = release.config.health
    if not health.amp_workspace_id or not health.sns_topic_arn:
        raise ReleaseError("AMP workspace or SNS topic is not configured")
    workspace = (
        aws_json(
            release,
            ["amp", "describe-workspace", "--workspace-id", health.amp_workspace_id],
        ).get("workspace")
        or {}
    )
    if workspace.get("workspaceId") != health.amp_workspace_id:
        raise ReleaseError("AMP workspace identity differs")
    if (workspace.get("status") or {}).get("statusCode") != "ACTIVE":
        raise ReleaseError("AMP workspace must be ACTIVE before release repair")
    missing: list[str] = []
    for operation, root, suffix in (
        (
            "describe-rule-groups-namespace",
            "ruleGroupsNamespace",
            ("--name", health.amp_rule_namespace),
        ),
        ("describe-alert-manager-definition", "alertManagerDefinition", ()),
    ):
        definition = describe_amp_definition(
            release,
            AmpDefinition(
                describe=(
                    "aws",
                    "amp",
                    operation,
                    "--workspace-id",
                    health.amp_workspace_id,
                    "--region",
                    release.config.aws_region,
                    *suffix,
                ),
                root=root,
                label=root,
            ),
        )
        if definition is None:
            missing.append(root)
            continue
        if amp_definition_status(definition) not in {
            "ACTIVE",
            *AMP_FAILED_STATUSES,
            *AMP_SETTLING_STATUSES,
        }:
            raise ReleaseError("cannot validate an unknown AMP definition status")
        if (
            root == "ruleGroupsNamespace"
            and definition.get("name") != health.amp_rule_namespace
        ):
            raise ReleaseError("AMP rule namespace identity differs")
        try:
            text = base64.b64decode(definition["data"], validate=True).decode("utf-8")
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise ReleaseError("cannot read the AMP definition data") from exc
        if root == "alertManagerDefinition" and health.sns_topic_arn not in text:
            raise ReleaseError(
                "AMP Alertmanager does not reference the configured SNS topic"
            )
    if missing:
        require_cpu_namespace_anchor(release, bootstrap=bootstrap)
    subscriptions = (
        aws_json(
            release,
            ["sns", "list-subscriptions-by-topic", "--topic-arn", health.sns_topic_arn],
        ).get("Subscriptions")
        or []
    )
    email_summary = monitoring_safety.email_subscription_summary(
        subscriptions, release.config.notifications.admin_email
    )
    return {
        "workspace_id": health.amp_workspace_id,
        "rule_namespace": health.amp_rule_namespace,
        "missing": missing,
        "email_subscription": email_summary,
    }


def preflight_retired_collectors(
    release: RegionalRelease, target: ClusterTarget
) -> None:
    names = frozenset(
        resource["name"]
        for resource in inventory.GPU_RESOURCES
        if resource.get("retired")
        and resource["kind"] == "deployment"
        and resource["phase"] == "producer"
    )
    if not names:
        return
    listing = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "deployments",
        )
    )
    items = listing.get("items") if isinstance(listing, dict) else None
    if not isinstance(items, list):
        raise ReleaseError("cannot establish whether retired collectors remain")
    present: set[str] = set()
    for item in items:
        metadata = item.get("metadata") if isinstance(item, dict) else None
        name = metadata.get("name") if isinstance(metadata, dict) else None
        if not isinstance(name, str) or not name or name != name.strip():
            raise ReleaseError("cannot establish whether retired collectors remain")
        present.add(name)
    remaining = sorted(names & present)
    if remaining:
        raise ReleaseError(
            f"{target.cluster_id}: retired HMA collectors remain. Stop upstream "
            "forwarding, drain queued events with the previous release and retire "
            f"{', '.join(remaining)} before deploying; no resources were deleted."
        )


def _context_eks_arn(runner: Runner, kubectl: list[str], *, label: str) -> str:
    cluster = runner.run(
        kubectl
        + [
            "config",
            "view",
            "--minify",
            "-o",
            "jsonpath={.contexts[0].context.cluster}",
        ],
        capture=True,
    )
    if EKS_ARN_PATTERN.fullmatch(cluster) is None:
        raise ReleaseError(
            f"{label} kubeconfig cluster identity is not an EKS ARN; "
            "regenerate it with aws eks update-kubeconfig"
        )
    return cluster


def validate_eks_managed_cluster(
    runner: Runner,
    *,
    region: str,
    expected_eks_arn: str,
    node_selector: str = "",
    described: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """A plain EKS GPU cluster: the cluster and its managed node groups are ACTIVE
    and no node group hosting GPU nodes repairs them itself.

    There is no HyperPod to describe; the counterpart of its ``NodeRecovery``
    gate is EKS node auto repair (``nodeRepairConfig.enabled``), which would
    race this solution's recovery. A GPU node group is one the cluster's
    recorded ``node_selector`` covers (``eks_node_repair.nodegroup_hosts_gpu_nodes``:
    the ownership label, a named node group, or -- for a legacy selector --
    an accelerator instance type). Returns the per-group facts (``status``,
    ``node_repair``, ``update_strategy``, ``gpu``) so the ``regional_contexts``
    check records them; ``unset`` is the opt-in field never written, which EKS
    treats as off. ``update_strategy`` (``DEFAULT`` when absent) is recorded
    only: a GPU group not on ``MINIMAL`` gets an advisory in ``status --full``,
    never a refusal (``eks_nodegroup_update_strategy``). ``described`` collects
    each ``describe-nodegroup`` document, for the advisory reads that follow
    (``eks_automatic_updates``).
    """

    eks_name = expected_eks_arn.rsplit("/", 1)[-1]
    raw = runner.run(
        [
            "aws",
            "eks",
            "describe-cluster",
            "--region",
            region,
            "--name",
            eks_name,
            "--query",
            "{Arn:cluster.arn,Status:cluster.status}",
            "--output",
            "json",
        ],
        capture=True,
    )
    value = json.loads(raw)
    if value.get("Arn") != expected_eks_arn:
        raise ReleaseError(
            f"EKS cluster {eks_name} does not match configured EKS ARN {expected_eks_arn}"
        )
    if value.get("Status") != "ACTIVE":
        raise ReleaseError(f"EKS cluster {eks_name} is not ACTIVE")
    nodegroups = (
        json.loads(
            runner.run(
                [
                    "aws",
                    "eks",
                    "list-nodegroups",
                    "--region",
                    region,
                    "--cluster-name",
                    eks_name,
                    "--output",
                    "json",
                ],
                capture=True,
            )
        ).get("nodegroups")
        or []
    )
    facts: dict[str, dict[str, Any]] = {}
    for nodegroup in sorted(str(item) for item in nodegroups):
        document = json.loads(
            runner.run(
                [
                    "aws",
                    "eks",
                    "describe-nodegroup",
                    "--region",
                    region,
                    "--cluster-name",
                    eks_name,
                    "--nodegroup-name",
                    nodegroup,
                    "--query",
                    "nodegroup",
                    "--output",
                    "json",
                ],
                capture=True,
            )
        )
        group = document if isinstance(document, dict) else {}
        if described is not None:
            described[nodegroup] = group
        status = group.get("status")
        if status != "ACTIVE":
            raise ReleaseError(
                f"EKS node group {nodegroup} of {eks_name} is not ACTIVE ({status})"
            )
        repair = node_repair_state(group)
        gpu = (
            nodegroup_hosts_gpu_nodes(group, node_selector)
            if node_selector
            else hosts_gpu_instance_types(group)
        )
        facts[nodegroup] = {
            "status": status,
            "node_repair": repair,
            "update_strategy": update_strategy(group),
            "gpu": gpu,
        }
        if gpu:
            reason = node_repair_error(nodegroup, eks_name, repair)
            if reason is not None:
                raise ReleaseError(reason)
    return facts


def _validate_hyperpod_cluster(
    runner: Runner,
    *,
    region: str,
    cluster_name: str,
    expected_eks_arn: str,
    require_node_recovery_none: bool,
) -> list[dict[str, Any]]:
    """Bind the HyperPod cluster to its EKS ARN and NodeRecovery; return the
    instance groups the same describe reads for ``hyperpod_automatic_updates``."""

    raw = runner.run(
        [
            "aws",
            "sagemaker",
            "describe-cluster",
            "--region",
            region,
            "--cluster-name",
            cluster_name,
            "--query",
            HYPERPOD_CONTEXT_QUERY,
            "--output",
            "json",
        ],
        capture=True,
    )
    value = json.loads(raw)
    if value.get("EksClusterArn") != expected_eks_arn:
        raise ReleaseError(
            f"HyperPod cluster {cluster_name} does not target "
            f"configured EKS ARN {expected_eks_arn}"
        )
    if require_node_recovery_none and value.get("NodeRecovery") != "None":
        raise ReleaseError(
            f"HyperPod cluster {cluster_name} must set NodeRecovery=None"
        )
    groups = value.get("InstanceGroups")
    return [item for item in groups or () if isinstance(item, dict)]


#: The context describe of a HyperPod cluster: its binding, and per instance
#: group what the automatic-update advisory needs (one read serves both; the
#: CPU cluster's advisory also counts the instances one window re-images).
HYPERPOD_CONTEXT_QUERY = (
    "{EksClusterArn:Orchestrator.Eks.ClusterArn,NodeRecovery:NodeRecovery,"
    "InstanceGroups:InstanceGroups[].{InstanceGroupName:InstanceGroupName,"
    "InstanceType:InstanceType,CurrentCount:CurrentCount,"
    "KubernetesConfig:KubernetesConfig,ScheduledUpdateConfig:ScheduledUpdateConfig,"
    "AutoPatchConfig:AutoPatchConfig}}"
)


def hyperpod_automatic_updates(
    groups: list[dict[str, Any]], target: ClusterTarget
) -> dict[str, Any]:
    """The advisory block of a HyperPod cluster: the instance groups its
    recorded selector covers (``automatic_node_updates``); no extra read."""

    selected = hyperpod_groups_for_selector(
        groups,
        cluster_name=target.hyperpod_cluster_name,
        selector=node_selector(target),
    )
    return hyperpod_updates(
        groups, selected=selected, gpu_cluster_arn=target.eks_cluster_arn
    )


def eks_automatic_updates(
    release: RegionalRelease,
    target: ClusterTarget,
    described: dict[str, dict[str, Any]],
    facts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """The advisory block of an eks-managed cluster (``automatic_node_updates``).

    Two reads, both advisory -- a failure records ``unknown``, never fails the
    check: the Auto Scaling groups of the node groups that host GPU nodes
    (``autoscaling:DescribeAutoScalingGroups``) and the site's Karpenter
    NodePools (a cluster without the CRD is ``absent``).
    """

    def read_asgs(names: Any) -> dict[str, Any]:
        return aws_json(
            release,
            [
                "autoscaling",
                "describe-auto-scaling-groups",
                "--region",
                target.region,
                "--auto-scaling-group-names",
                *names,
                "--query",
                ASG_QUERY,
            ],
            region=False,
        )

    def read_nodepools() -> dict[str, Any] | None:
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

    return {
        "node_groups": eks_node_group_updates(
            described,
            selected=[name for name, item in facts.items() if item.get("gpu")],
            read_asgs=read_asgs,
        ),
        "karpenter": karpenter_updates(
            read_nodepools, site_id=release.config.site_name
        ),
    }


def control_plane_automatic_updates(
    release: RegionalRelease, groups: list[dict[str, Any]]
) -> dict[str, Any]:
    """The advisory block of the CPU control-plane cluster (``automatic_node_updates``).

    Every instance group of its HyperPod cluster, from the context describe
    already made, and ``eks_node_groups``: the managed node groups its EKS
    cluster may also have -- any CPU node hosts control-plane Pods -- with
    their Auto Scaling groups, read the way the GPU path reads them (one
    ``list-nodegroups``, a describe per group, one Auto Scaling read). Those
    reads are advisory: a failure records ``state: unknown``. A rolling
    group's rollback alarms are judged by the ``cpu_rolling_update`` check
    (``regional_cpu_rolling_update``).
    """

    config = release.config
    block = control_plane_updates(
        groups,
        hyperpod_cluster_name=config.cpu_hyperpod_cluster_name,
        cpu_eks_arn=config.cpu_eks_arn,
    )
    eks_name = config.cpu_eks_arn.rsplit("/", 1)[-1]

    def node_groups() -> dict[str, Any]:
        names = aws_json(
            release, ["eks", "list-nodegroups", "--cluster-name", eks_name]
        ).get("nodegroups")
        described = {
            str(name): aws_json(
                release,
                [
                    "eks",
                    "describe-nodegroup",
                    "--cluster-name",
                    eks_name,
                    "--nodegroup-name",
                    str(name),
                    "--query",
                    "nodegroup",
                ],
            )
            for name in sorted(str(item) for item in names or () if item)
        }
        return {
            "state": "read",
            "node_groups": eks_node_group_updates(
                described,
                selected=described,
                read_asgs=lambda asgs: aws_json(
                    release,
                    [
                        "autoscaling",
                        "describe-auto-scaling-groups",
                        "--auto-scaling-group-names",
                        *asgs,
                        "--query",
                        ASG_QUERY,
                    ],
                ),
            ),
        }

    block["eks_node_groups"] = guarded(node_groups)
    return block


def _validate_agent_endpoint_cidrs(
    runner: Runner,
    kubectl: list[str],
    *,
    cluster_id: str,
    node_selector: str,
    configured_cidrs: tuple[str, ...],
) -> None:
    networks = tuple(
        ipaddress.ip_network(value, strict=False) for value in configured_cidrs
    )
    if not networks:
        raise ReleaseError(f"{cluster_id} requires Agent endpoint CIDRs")
    raw = runner.run(
        kubectl
        + [
            "get",
            "nodes",
            "-l",
            node_selector,
            "-o",
            "json",
        ],
        capture=True,
    )
    nodes = json.loads(raw).get("items") or []
    if not nodes:
        raise ReleaseError(f"{cluster_id} has no Kubernetes nodes")
    uncovered: list[str] = []
    for node in nodes:
        addresses = [
            item.get("address")
            for item in node.get("status", {}).get("addresses", [])
            if item.get("type") == "InternalIP"
        ]
        if not addresses or any(
            not any(ipaddress.ip_address(value) in network for network in networks)
            for value in addresses
        ):
            uncovered.append(str(node.get("metadata", {}).get("name") or "unknown"))
    if uncovered:
        raise ReleaseError(
            f"{cluster_id} Agent endpoint CIDRs do not cover nodes: "
            + ", ".join(sorted(uncovered))
        )


def ensure_region_contexts(release: RegionalRelease) -> dict[str, Any]:
    """Validate every context; return what the ``regional_contexts`` check records
    (the cluster ids and, per eks-managed cluster, its node groups' facts; per
    GPU cluster and for the CPU control-plane cluster, ``control_plane``, the
    ``automatic_node_updates`` advisory block; ``eks_support``, every site EKS
    cluster's version and its support end date)."""

    config = release.config
    runner = release.runner
    providers: dict[str, dict[str, Any]] = {}
    cpu_command = release._cpu
    gpu_command = release._gpu
    cpu_eks_arn = _context_eks_arn(runner, cpu_command(), label="CPU")
    if cpu_eks_arn != config.cpu_eks_arn:
        raise ReleaseError(
            "CPU kubeconfig EKS ARN does not match configured cpu_eks_arn"
        )
    cpu_groups = _validate_hyperpod_cluster(
        runner,
        region=config.aws_region,
        cluster_name=config.cpu_hyperpod_cluster_name,
        expected_eks_arn=config.cpu_eks_arn,
        require_node_recovery_none=False,
    )
    runner.run(cpu_command("get", "--raw=/readyz"), capture=True)
    control_plane: dict[str, Any] = {
        "provider": "hyperpod",
        "hyperpod_cluster_name": config.cpu_hyperpod_cluster_name,
    }

    eks_support: dict[str, Any] = {}

    def control_plane_advisory() -> None:
        # Advisory only, after the CPU gates above; never a failure of the check.
        control_plane[AUTOMATIC_NODE_UPDATES] = guarded(
            lambda: control_plane_automatic_updates(release, cpu_groups)
        )

    def eks_support_advisory() -> None:
        # Advisory only (regional_release_eks_support): AWS upgrades a control
        # plane by itself once its version's support ends.
        eks_support.update(guarded(lambda: read_eks_support(release)))

    def validate_cluster(target: ClusterTarget) -> None:
        gpu_eks_arn = _context_eks_arn(
            runner,
            gpu_command(target),
            label=f"GPU cluster {target.cluster_id}",
        )
        if gpu_eks_arn != target.eks_cluster_arn:
            raise ReleaseError(
                f"GPU context for {target.cluster_id} does not match "
                "configured eks_cluster_arn"
            )
        described: dict[str, dict[str, Any]] = {}
        groups: list[dict[str, Any]] = []
        if is_eks_managed(target):
            providers[target.cluster_id] = {
                "provider": "eks-managed",
                "node_groups": validate_eks_managed_cluster(
                    runner,
                    region=target.region,
                    expected_eks_arn=target.eks_cluster_arn,
                    node_selector=node_selector(target),
                    described=described,
                ),
            }
        else:
            providers[target.cluster_id] = {"provider": "hyperpod"}
            groups = _validate_hyperpod_cluster(
                runner,
                region=target.region,
                cluster_name=target.hyperpod_cluster_name,
                expected_eks_arn=target.eks_cluster_arn,
                require_node_recovery_none=True,
            )
        runner.run(gpu_command(target, "get", "--raw=/readyz"), capture=True)
        _validate_agent_endpoint_cidrs(
            runner,
            gpu_command(target),
            cluster_id=target.cluster_id,
            node_selector=node_selector(target),
            configured_cidrs=target.agent_endpoint_allowed_cidrs,
        )
        release._validate_executor_iam_role(target)
        preflight_retired_collectors(release, target)
        # Advisory only (automatic_node_updates): recorded beside the facts it
        # extends, after every gate above, and never a failure of this check.
        entry = providers[target.cluster_id]
        entry[AUTOMATIC_NODE_UPDATES] = guarded(
            lambda: (
                eks_automatic_updates(release, target, described, entry["node_groups"])
                if is_eks_managed(target)
                else hyperpod_automatic_updates(groups, target)
            )
        )

    facts = {
        "clusters": [item.cluster_id for item in config.clusters],
        CONTROL_PLANE: control_plane,
        EKS_SUPPORT: eks_support,
    }
    if not config.clusters:
        control_plane_advisory()
        eks_support_advisory()
        return facts
    # The CPU checks above gate everything, so they stay ahead of this; the
    # clusters themselves are independent, and each one is a handful of reads
    # against a different API server plus a whole IAM role expansion. Every
    # future is resolved in configuration order, so a fleet where two clusters
    # are both wrong reports the same one every time instead of whichever
    # thread lost the race. The advisory reads run beside them.
    with ThreadPoolExecutor(max_workers=min(8, len(config.clusters) + 2)) as executor:
        advisories = [
            executor.submit(control_plane_advisory),
            executor.submit(eks_support_advisory),
        ]
        futures = [
            executor.submit(validate_cluster, target) for target in config.clusters
        ]
    not_gating: dict[str, Any] = {}
    for target, future in zip(config.clusters, futures, strict=True):
        try:
            future.result()
        except Exception as error:
            # Another cluster's own context is a finding of a verify scoped to
            # a membership verb's target (regional_verify_scope).
            if gates(release, target.cluster_id):
                raise
            record_not_gating(not_gating, target.cluster_id, str(error))
    for advisory in advisories:
        advisory.result()
    return {**facts, "providers": providers, **not_gating}
