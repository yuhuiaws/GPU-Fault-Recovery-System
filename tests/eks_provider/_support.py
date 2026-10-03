"""Fakes for the plain-EKS provider: Kubernetes nodes, EC2 and EKS clients.

The cluster mirrors the design target: two p5en.48xlarge SPOT nodes in one
managed node group with FQDN node names and ``aws:///<zone>/i-...`` provider
IDs, plus a CPU node the GPU selector must never match.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.eks_provider import EksManagedNodeProvider, EksProviderConfig

CLUSTER_ID = "gpu-a"
EKS_CLUSTER = "gpu-eks-cluster"
NODEGROUP = "gpu-p5en-efa-spot-node-group"
CPU_NODEGROUP = "cpu-node-group-1"
SELECTOR = "nvidia.com/gpu.present=true"
REGION = "us-west-2"
NODE_A = "ip-10-0-33-205.us-west-2.compute.internal"
NODE_B = "ip-10-0-40-17.us-west-2.compute.internal"
CPU_NODE = "ip-10-0-1-9.us-west-2.compute.internal"
INSTANCE_A = "i-0a000001"
INSTANCE_B = "i-0a000002"
CPU_INSTANCE = "i-0a000009"


def gpu_node(
    name: str,
    instance_id: str,
    *,
    zone: str = "us-west-2a",
    ready: bool = True,
    nodegroup: str | None = NODEGROUP,
    labels: dict[str, str] | None = None,
    provider_id: str | None = "default",
) -> dict[str, Any]:
    node_labels = {
        "nvidia.com/gpu.present": "true",
        "vpc.amazonaws.com/efa.present": "true",
        "node-type": "gpu",
        "node.kubernetes.io/instance-type": "p5en.48xlarge",
        "topology.kubernetes.io/zone": zone,
        "eks.amazonaws.com/capacityType": "SPOT",
        **(labels or {}),
    }
    if nodegroup:
        node_labels["eks.amazonaws.com/nodegroup"] = nodegroup
    spec: dict[str, Any] = {"unschedulable": False, "taints": []}
    if provider_id == "default":
        spec["providerID"] = f"aws:///{zone}/{instance_id}"
    elif provider_id is not None:
        spec["providerID"] = provider_id
    return {
        "metadata": {
            "name": name,
            "uid": f"{name}-uid",
            "labels": node_labels,
            "annotations": {},
            "resourceVersion": "7",
        },
        "spec": spec,
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "nodeInfo": {"bootID": f"{name}-boot"},
            "addresses": [{"type": "InternalIP", "address": "10.0.33.205"}],
        },
    }


def cpu_node() -> dict[str, Any]:
    node = gpu_node(CPU_NODE, CPU_INSTANCE, nodegroup=CPU_NODEGROUP)
    node["metadata"]["labels"] = {
        "node-type": "cpu",
        "eks.amazonaws.com/nodegroup": CPU_NODEGROUP,
        "node.kubernetes.io/instance-type": "m7i-flex.xlarge",
    }
    return node


def _matches(node: dict[str, Any], selector: str | None) -> bool:
    if not selector:
        return True
    labels = node["metadata"].get("labels") or {}
    for clause in selector.split(","):
        clause = clause.strip()
        if "=" in clause:
            key, value = clause.split("=", 1)
            if labels.get(key.strip()) != value.strip():
                return False
        elif clause not in labels:
            return False
    return True


class FakeCore:
    """``list_node``/``read_node``/``patch_node`` over dict-shaped nodes."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self.nodes = {node["metadata"]["name"]: node for node in nodes}
        self.list_selectors: list[str | None] = []
        self.reads: list[str] = []

    def list_node(self, label_selector: str | None = None, **_: Any) -> dict[str, Any]:
        self.list_selectors.append(label_selector)
        return {
            "items": [
                node for node in self.nodes.values() if _matches(node, label_selector)
            ]
        }

    def read_node(self, name: str, **_: Any) -> dict[str, Any]:
        self.reads.append(name)
        try:
            return self.nodes[name]
        except KeyError:
            error = KeyError(name)
            error.status = 404  # type: ignore[attr-defined]
            raise error

    def patch_node(self, name: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        node = self.nodes[name]
        node["metadata"].setdefault("annotations", {}).update(
            {
                key: value
                for key, value in (body.get("metadata") or {})
                .get("annotations", {})
                .items()
                if value is not None
            }
        )
        node["spec"].update(body.get("spec") or {})
        return node


def ec2_instance(
    instance_id: str,
    *,
    state: str = "running",
    cluster: str | None = EKS_CLUSTER,
    nodegroup: str | None = NODEGROUP,
    zone: str = "us-west-2a",
    private_dns: str | None = None,
) -> dict[str, Any]:
    tags = []
    if cluster:
        tags.append({"Key": "eks:cluster-name", "Value": cluster})
    if nodegroup:
        tags.append({"Key": "eks:nodegroup-name", "Value": nodegroup})
    return {
        "InstanceId": instance_id,
        "InstanceType": "p5en.48xlarge",
        "InstanceLifecycle": "spot",
        "State": {"Name": state, "Code": 16},
        "StateReason": {"Message": f"state {state}"},
        "Placement": {"AvailabilityZone": zone},
        "PrivateDnsName": private_dns or "",
        "PrivateIpAddress": "10.0.33.205",
        "Tags": tags,
    }


class FakeEc2:
    def __init__(self, instances: list[dict[str, Any]]) -> None:
        self.instances = {item["InstanceId"]: item for item in instances}
        self.describe_calls: list[dict[str, Any]] = []
        self.reboots: list[list[str]] = []
        self.reboot_error: Exception | None = None

    def describe_instances(self, **kwargs: Any) -> dict[str, Any]:
        self.describe_calls.append(kwargs)
        wanted: set[str] = set()
        for item in kwargs.get("Filters") or []:
            if item["Name"] == "instance-id":
                wanted.update(item["Values"])
        found = [self.instances[key] for key in sorted(wanted) if key in self.instances]
        return {"Reservations": [{"Instances": found}] if found else []}

    def reboot_instances(self, **kwargs: Any) -> dict[str, Any]:
        self.reboots.append(list(kwargs["InstanceIds"]))
        if self.reboot_error is not None:
            raise self.reboot_error
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}


class FakeEks:
    def __init__(
        self, *, status: str = "ACTIVE", nodegroups: dict[str, str] | None = None
    ) -> None:
        self.status = status
        self.nodegroups = (
            nodegroups if nodegroups is not None else {NODEGROUP: "ACTIVE"}
        )
        self.described_clusters: list[str] = []
        self.described_nodegroups: list[tuple[str, str]] = []

    def describe_cluster(self, **kwargs: Any) -> dict[str, Any]:
        name = kwargs["name"]
        self.described_clusters.append(name)
        return {
            "cluster": {
                "name": name,
                "arn": f"arn:aws:eks:{REGION}:123456789012:cluster/{name}",
                "status": self.status,
            }
        }

    def describe_nodegroup(self, **kwargs: Any) -> dict[str, Any]:
        key = (kwargs["clusterName"], kwargs["nodegroupName"])
        self.described_nodegroups.append(key)
        return {
            "nodegroup": {
                "nodegroupName": key[1],
                "status": self.nodegroups.get(key[1], "ACTIVE"),
            }
        }


def config(**overrides: Any) -> EksProviderConfig:
    values: dict[str, Any] = {
        "cluster_id": CLUSTER_ID,
        "node_selector": SELECTOR,
        "region_name": REGION,
        "reboot_enabled": True,
    }
    values.update(overrides)
    return EksProviderConfig(**values)


def default_nodes() -> list[dict[str, Any]]:
    return [
        gpu_node(NODE_A, INSTANCE_A),
        gpu_node(NODE_B, INSTANCE_B, zone="us-west-2b"),
        cpu_node(),
    ]


def default_instances() -> list[dict[str, Any]]:
    return [
        ec2_instance(INSTANCE_A, private_dns=NODE_A),
        ec2_instance(INSTANCE_B, zone="us-west-2b", private_dns=NODE_B),
        ec2_instance(CPU_INSTANCE, nodegroup=CPU_NODEGROUP, private_dns=CPU_NODE),
    ]


def provider(
    *,
    core: FakeCore | None = None,
    ec2: FakeEc2 | None = None,
    eks: FakeEks | None = None,
    store: Any | None = None,
    **config_overrides: Any,
) -> tuple[EksManagedNodeProvider, FakeCore, FakeEc2, FakeEks]:
    core = core if core is not None else FakeCore(default_nodes())
    ec2 = ec2 if ec2 is not None else FakeEc2(default_instances())
    eks = eks if eks is not None else FakeEks()
    adapter = EksManagedNodeProvider(
        config(**config_overrides), core, ec2_client=ec2, eks_client=eks, store=store
    )
    return adapter, core, ec2, eks
