"""The reboot scope of ``GF-REGIONAL-COLLECT-004``/``COLLECT-015``, per provider.

``capture_reboot_scope`` reads every identity the reboot proof later binds --
cluster, node, EC2 instance, executor population -- without mutating anything.
On HyperPod the provider facts come from SageMaker (``describe-cluster``,
``list-cluster-nodes``, ``describe-cluster-node``); on a plain EKS cluster
(``eks-managed``) there is no SageMaker: the cluster fact is ``eks
describe-cluster``, the node's managed node group and EC2 instance are read
directly, the logical node id is the Kubernetes Node name (what the EKS
provider's submission records) and the provider inventory is the GPU nodes as
Kubernetes reports them. Live 2026-10-05 the HyperPod-only read failed the
COLLECT-004 plan on gpu-eks-cluster with "read-only AWS scope query failed"
(``ResourceNotFound`` from SageMaker).

The validation helpers stay in ``collector_reboot_evidence``; this module is
the capture only, so that module keeps its size and its proof code unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from scripts.e2e.regional import collector_reboot_evidence as evidence

if TYPE_CHECKING:
    from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture


def capture_reboot_scope(
    regional: RegionalLiveFixture,
    *,
    node: str,
    hyperpod_cluster: str,
    executor_role_arn: str,
    provider: str | None = None,
) -> evidence.RebootScope:
    """Capture actual read-only identities; raise safely on missing/drifted facts.

    Call before injection and after recovery. Node UID/instance/logical identity
    must stay fixed; a new boot ID is an expected *separate* reboot observation.
    These sequential reads do not lock external actors or promise atomicity.
    ``hyperpod_cluster`` is the provider cluster scope (the HyperPod name, or on
    ``eks-managed`` the cluster id); the provider defaults to the Node's labels.
    """
    try:
        evidence._text(node, "target Node")
        evidence._text(hyperpod_cluster, "provider cluster scope")
        evidence._text(regional.settings.cluster_id, "managed cluster ID")
        role = evidence._arn(executor_role_arn, "iam")
        node_document = evidence._mapping(
            evidence._kube(regional, "get", "node", node, "-o", "json"), "Node"
        )
        labels = evidence._mapping(node_document.get("metadata"), "Node metadata").get(
            "labels"
        )
        if provider is None:
            provider = (
                "eks-managed"
                if isinstance(labels, Mapping)
                and labels.get(evidence.EKS_NODEGROUP_LABEL)
                and not labels.get("sagemaker.amazonaws.com/cluster-name")
                else "hyperpod"
            )
        if provider == "eks-managed":
            return _capture_eks_reboot_scope(
                regional,
                node=node,
                cluster_name=hyperpod_cluster,
                executor_role_arn=executor_role_arn,
                role=role,
                node_document=node_document,
            )
        cluster = evidence._aws(
            regional,
            "sagemaker",
            "describe-cluster",
            "--cluster-name",
            hyperpod_cluster,
        )
        hp_arn = evidence._text(cluster.get("ClusterArn"), "HyperPod ARN")
        hp = evidence._arn(hp_arn, "sagemaker")
        evidence._require(
            cluster.get("ClusterName") == hyperpod_cluster
            and hp[0] == role[0]
            and hp[2] == role[2]
            and not role[1]
            and hp[1] == regional.settings.region
            and cluster.get("NodeRecovery") == "None"
            and cluster.get("ClusterStatus") == "InService",
            "actual HyperPod name/ARN/Region/account or recovery state differs",
        )
        orchestrator = evidence._mapping(
            cluster.get("Orchestrator"), "HyperPod orchestrator"
        )
        evidence._require(
            set(orchestrator) == {"Eks"}, "HyperPod orchestrator is not exclusively EKS"
        )
        eks = evidence._mapping(orchestrator["Eks"], "HyperPod EKS orchestrator")
        eks_arn = evidence._text(eks.get("ClusterArn"), "orchestrator EKS ARN")
        eks_facts = evidence._eks_identity(regional, eks_arn, hp[2])
        before = evidence._node_identity(
            node_document,
            node,
            regional.settings.region,
            hyperpod_cluster_arn=hp_arn,
            hyperpod_cluster_name=hyperpod_cluster,
        )
        inventory = evidence._aws(
            regional,
            "sagemaker",
            "list-cluster-nodes",
            "--cluster-name",
            hp_arn,
            "--include-node-logical-ids",
        )
        evidence._require(
            not inventory.get("NextToken"), "HyperPod inventory is incomplete"
        )
        rows = evidence._rows(
            inventory.get("ClusterNodeSummaries"), "HyperPod node inventory"
        )
        logical_ids = [
            evidence._text(row.get("NodeLogicalId"), "provider logical node ID")
            for row in rows
        ]
        evidence._require(
            len(set(logical_ids)) == len(logical_ids),
            "HyperPod inventory repeats a logical node",
        )
        matches = [
            row for row in rows if row.get("InstanceId") == before["instance_id"]
        ]
        evidence._require(
            len(matches) == 1, "Node providerID has no unique HyperPod node mapping"
        )
        target = matches[0]
        evidence._require(
            evidence._mapping(target.get("InstanceStatus"), "HyperPod node status").get(
                "Status"
            )
            == "Running",
            "HyperPod target instance is not Running",
        )
        deployment_uid, template_digest, executors = evidence._executor_identity(
            regional, hyperpod_cluster, executor_role_arn
        )
        after = evidence._node_identity(
            evidence._mapping(
                evidence._kube(regional, "get", "node", node, "-o", "json"), "Node"
            ),
            node,
            regional.settings.region,
            hyperpod_cluster_arn=hp_arn,
            hyperpod_cluster_name=hyperpod_cluster,
        )
        evidence._require(after == before, "Node identity changed during scope capture")
        detail = evidence._aws(
            regional,
            "sagemaker",
            "describe-cluster-node",
            "--cluster-name",
            hp_arn,
            "--node-logical-id",
            target["NodeLogicalId"],
        )
        details = evidence._mapping(detail.get("NodeDetails"), "HyperPod node detail")
        evidence._require(
            details.get("NodeLogicalId") == target["NodeLogicalId"]
            and details.get("InstanceId") == before["instance_id"],
            "HyperPod node mapping changed during scope capture",
        )
        cluster_after = evidence._aws(
            regional, "sagemaker", "describe-cluster", "--cluster-name", hp_arn
        )
        evidence._require(
            all(
                cluster_after.get(key) == cluster.get(key)
                for key in (
                    "ClusterArn",
                    "ClusterName",
                    "NodeRecovery",
                    "ClusterStatus",
                    "Orchestrator",
                )
            ),
            "HyperPod identity or recovery settings changed during scope capture",
        )
        result: evidence.RebootScope = {
            "provider": "hyperpod",
            "cluster_id": regional.settings.cluster_id,
            "node_id": node,
            "node_uid": before["node_uid"],
            "node_provider_id": before["provider_id"],
            "boot_id": before["boot_id"],
            "instance_id": before["instance_id"],
            "node_logical_id": target["NodeLogicalId"],
            "cluster_name": hyperpod_cluster,
            "cluster_arn": hp_arn,
            "eks_cluster_arn": eks_arn,
            "region": regional.settings.region,
            "account_id": hp[2],
            "executor_role_arn": executor_role_arn,
            "gpu_context": regional.settings.gpu_context,
            "provider_inventory": [
                {
                    "node_logical_id": row["NodeLogicalId"],
                    "instance_id": evidence._optional_text(row.get("InstanceId")),
                }
                for row in sorted(rows, key=lambda item: str(item["NodeLogicalId"]))
            ],
            "eks_namespace_uid": eks_facts["eks_namespace_uid"],
            "eks_endpoint_sha256": eks_facts["eks_endpoint_sha256"],
            "eks_ca_sha256": eks_facts["eks_ca_sha256"],
            "executor_deployment_uid": deployment_uid,
            "executor_template_sha256": template_digest,
            "executor_pods": executors,
            "node_recovery": "None",
            "cluster_status": "InService",
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        evidence._scope_identity(result)
        return result
    except evidence.RebootEvidenceError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError):
        raise evidence.RebootEvidenceError(
            "reboot scope identity evidence is malformed"
        ) from None


def _capture_eks_reboot_scope(
    regional: RegionalLiveFixture,
    *,
    node: str,
    cluster_name: str,
    executor_role_arn: str,
    role: tuple[str, str, str, str],
    node_document: Mapping[str, Any],
) -> evidence.RebootScope:
    """The plain-EKS scope: EKS cluster and node group, EC2 instance, GPU nodes.

    No SageMaker exists here. The cluster fact is ``eks describe-cluster``
    (``ACTIVE``), the node group the Node's label names must be ``ACTIVE``, the
    EC2 instance behind the Node's providerID must be ``running`` and tagged for
    this cluster, the logical node id is the Kubernetes Node name (what the EKS
    provider's submission records), and the provider inventory is the GPU
    nodes as Kubernetes reports them.
    """

    description = evidence._mapping(
        evidence._aws(regional, "eks", "describe-cluster", "--name", cluster_name).get(
            "cluster"
        ),
        "EKS cluster",
    )
    eks_arn = evidence._text(description.get("arn"), "EKS ARN")
    eks = evidence._arn(eks_arn, "eks")
    evidence._require(
        description.get("name") == cluster_name
        and description.get("status") == "ACTIVE"
        and eks[0] == role[0]
        and eks[2] == role[2]
        and not role[1]
        and eks[1] == regional.settings.region,
        "actual EKS name/ARN/Region/account or status differs",
    )
    eks_facts = evidence._eks_identity(regional, eks_arn, eks[2])
    before = evidence._node_identity(
        node_document,
        node,
        regional.settings.region,
        hyperpod_cluster_arn=eks_arn,
        hyperpod_cluster_name=cluster_name,
    )
    labels = evidence._mapping(
        evidence._mapping(node_document.get("metadata"), "Node metadata").get("labels"),
        "Node labels",
    )
    nodegroup = evidence._text(
        labels.get(evidence.EKS_NODEGROUP_LABEL), "EKS node group label"
    )
    group = evidence._mapping(
        evidence._aws(
            regional,
            "eks",
            "describe-nodegroup",
            "--cluster-name",
            cluster_name,
            "--nodegroup-name",
            nodegroup,
        ).get("nodegroup"),
        "EKS node group",
    )
    evidence._require(
        group.get("nodegroupName") == nodegroup
        and group.get("clusterName") == cluster_name
        and group.get("status") == "ACTIVE",
        "EKS node group is not the Node's ACTIVE managed node group",
    )
    reservations = evidence._rows(
        evidence._aws(
            regional,
            "ec2",
            "describe-instances",
            "--instance-ids",
            before["instance_id"],
        ).get("Reservations"),
        "EC2 reservations",
    )
    instances = [
        row
        for reservation in reservations
        for row in evidence._rows(
            evidence._mapping(reservation, "EC2 reservation").get("Instances"),
            "EC2 instances",
        )
    ]
    evidence._require(len(instances) == 1, "EC2 instance lookup is not a singleton")
    instance = evidence._mapping(instances[0], "EC2 instance")
    tags = {
        str(tag.get("Key")): tag.get("Value")
        for tag in evidence._rows(instance.get("Tags") or [], "EC2 tags")
        if isinstance(tag, Mapping)
    }
    evidence._require(
        instance.get("InstanceId") == before["instance_id"]
        and evidence._mapping(instance.get("State"), "EC2 state").get("Name")
        == "running"
        and tags.get("eks:cluster-name", cluster_name) == cluster_name
        and tags.get("eks:nodegroup-name", nodegroup) == nodegroup,
        "EC2 instance is not the running member of this cluster's node group",
    )
    nodes = evidence._rows(
        evidence._mapping(
            evidence._kube(regional, "get", "node", "-o", "json"), "Nodes"
        ).get("items"),
        "Nodes",
    )
    rows = []
    for item in nodes:
        status = evidence._mapping(item.get("status"), "Node status")
        if not evidence._mapping(
            status.get("allocatable") or {}, "Node allocatable"
        ).get("nvidia.com/gpu"):
            continue
        metadata = evidence._mapping(item.get("metadata"), "Node metadata")
        provider_id = str(
            evidence._mapping(item.get("spec"), "Node spec").get("providerID") or ""
        )
        rows.append(
            {
                "node_logical_id": evidence._text(
                    metadata.get("name"), "GPU Node name"
                ),
                "instance_id": provider_id.rsplit("/", 1)[-1] or None,
            }
        )
    rows.sort(key=lambda row: str(row["node_logical_id"]))
    evidence._require(
        [
            row["node_logical_id"]
            for row in rows
            if row["instance_id"] == before["instance_id"]
        ]
        == [node],
        "Node providerID has no unique GPU node mapping",
    )
    deployment_uid, template_digest, executors = evidence._executor_identity(
        regional, cluster_name, executor_role_arn
    )
    after = evidence._node_identity(
        evidence._mapping(
            evidence._kube(regional, "get", "node", node, "-o", "json"), "Node"
        ),
        node,
        regional.settings.region,
        hyperpod_cluster_arn=eks_arn,
        hyperpod_cluster_name=cluster_name,
    )
    evidence._require(after == before, "Node identity changed during scope capture")
    result: evidence.RebootScope = {
        "provider": "eks-managed",
        "cluster_id": regional.settings.cluster_id,
        "node_id": node,
        "node_uid": before["node_uid"],
        "node_provider_id": before["provider_id"],
        "boot_id": before["boot_id"],
        "instance_id": before["instance_id"],
        "node_logical_id": node,
        "cluster_name": cluster_name,
        "cluster_arn": eks_arn,
        "eks_cluster_arn": eks_arn,
        "region": regional.settings.region,
        "account_id": eks[2],
        "executor_role_arn": executor_role_arn,
        "gpu_context": regional.settings.gpu_context,
        "provider_inventory": rows,
        "eks_namespace_uid": eks_facts["eks_namespace_uid"],
        "eks_endpoint_sha256": eks_facts["eks_endpoint_sha256"],
        "eks_ca_sha256": eks_facts["eks_ca_sha256"],
        "executor_deployment_uid": deployment_uid,
        "executor_template_sha256": template_digest,
        "executor_pods": executors,
        "node_recovery": "not-applicable",
        "cluster_status": "ACTIVE",
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
    evidence._scope_identity(result)
    return result
