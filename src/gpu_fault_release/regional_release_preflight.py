from __future__ import annotations

import base64
import ipaddress
import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease, Runner

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
from gpu_fault_release.regional_release_state import aws_json


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


def _validate_hyperpod_cluster(
    runner: Runner,
    *,
    region: str,
    cluster_name: str,
    expected_eks_arn: str,
    require_node_recovery_none: bool,
) -> None:
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
            "{EksClusterArn:Orchestrator.Eks.ClusterArn,NodeRecovery:NodeRecovery}",
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


def _validate_agent_endpoint_cidrs(
    runner: Runner,
    kubectl: list[str],
    *,
    cluster_id: str,
    hyperpod_cluster_name: str,
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
            f"sagemaker.amazonaws.com/cluster-name={hyperpod_cluster_name}",
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


def ensure_region_contexts(release: RegionalRelease) -> None:
    config = release.config
    runner = release.runner
    cpu_command = release._cpu
    gpu_command = release._gpu
    cpu_eks_arn = _context_eks_arn(runner, cpu_command(), label="CPU")
    if cpu_eks_arn != config.cpu_eks_arn:
        raise ReleaseError(
            "CPU kubeconfig EKS ARN does not match configured cpu_eks_arn"
        )
    _validate_hyperpod_cluster(
        runner,
        region=config.aws_region,
        cluster_name=config.cpu_hyperpod_cluster_name,
        expected_eks_arn=config.cpu_eks_arn,
        require_node_recovery_none=False,
    )
    runner.run(cpu_command("get", "--raw=/readyz"), capture=True)

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
        _validate_hyperpod_cluster(
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
            hyperpod_cluster_name=target.hyperpod_cluster_name,
            configured_cidrs=target.agent_endpoint_allowed_cidrs,
        )
        release._validate_executor_iam_role(target)
        preflight_retired_collectors(release, target)

    if not config.clusters:
        return
    # The CPU checks above gate everything, so they stay ahead of this; the
    # clusters themselves are independent, and each one is a handful of reads
    # against a different API server plus a whole IAM role expansion. Every
    # future is resolved in configuration order, so a fleet where two clusters
    # are both wrong reports the same one every time instead of whichever
    # thread lost the race.
    with ThreadPoolExecutor(max_workers=min(8, len(config.clusters))) as executor:
        futures = [
            executor.submit(validate_cluster, target) for target in config.clusters
        ]
    for future in futures:
        future.result()
