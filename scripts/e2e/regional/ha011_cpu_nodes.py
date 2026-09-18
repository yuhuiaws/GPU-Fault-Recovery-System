"""Conservative CPU-only EC2 node inventory, refreshed before placement and arm."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from scripts.e2e.regional.ha011_contracts import ProofError, digest

# Unknown families/providers require a reviewed inventory adapter, not a guess.
CPU_FAMILIES = frozenset(
    {
        "c5",
        "c5a",
        "c5n",
        "c6i",
        "c6a",
        "c6g",
        "c6in",
        "c7i",
        "c7a",
        "c7g",
        "c8g",
        "m5",
        "m5a",
        "m5n",
        "m6i",
        "m6a",
        "m6g",
        "m7i",
        "m7a",
        "m7g",
        "m8g",
        "r5",
        "r6i",
        "r6a",
        "r6g",
        "r7i",
        "r7a",
        "r7g",
        "r8g",
        "t3",
        "t3a",
        "t4g",
    }
)


def cpu_node(kubernetes: Any, name: str) -> dict[str, Any]:
    node = kubernetes.read("Node", name)
    metadata, spec, status = (
        node["metadata"],
        node.get("spec", {}),
        node.get("status", {}),
    )
    labels = metadata.get("labels", {})
    instance_type = labels.get("node.kubernetes.io/instance-type", "")
    provider = spec.get("providerID", "")
    if (
        not metadata.get("uid")
        or metadata.get("deletionTimestamp")
        or labels.get("kubernetes.io/os") != "linux"
        or not labels.get("kubernetes.io/hostname")
        or instance_type.removeprefix("ml.").partition(".")[0] not in CPU_FAMILIES
        or labels.get("topology.kubernetes.io/region") != kubernetes.settings.region
        or not provider.startswith("aws:///")
        or spec.get("unschedulable", False)
        or any(
            item.get("effect") in {"NoSchedule", "NoExecute"}
            for item in spec.get("taints", [])
        )
        or not any(
            item.get("type") == "Ready" and item.get("status") == "True"
            for item in status.get("conditions", [])
        )
    ):
        raise ProofError(
            "placement requires a Ready, schedulable, explicitly CPU-only node"
        )
    for inventory in ("capacity", "allocatable"):
        resources = status.get(inventory, {})
        try:
            capacity = str(resources["cpu"])
            cpus = (
                Decimal(capacity[:-1]) / 1000
                if capacity.endswith("m")
                else Decimal(capacity)
            )
            if not cpus.is_finite() or cpus < 4:
                raise ProofError(
                    "CPU probe requires at least four allocatable CPU cores"
                )
            for key, value in resources.items():
                if any(
                    term in key.lower() for term in ("gpu", "mig-", "neuron", "fpga")
                ):
                    if Decimal(str(value)) != 0:
                        raise ProofError(
                            "accelerator inventory forbids CPU acceptance placement"
                        )
        except (KeyError, InvalidOperation):
            raise ProofError(
                "CPU node resource inventory is missing or malformed"
            ) from None
    if any(
        key.startswith("nvidia.com/gpu") and str(value).lower() not in {"false", "0"}
        for key, value in labels.items()
    ):
        raise ProofError("GPU feature labels forbid CPU acceptance placement")
    lease = kubernetes.read("Lease", name, namespace="kube-node-lease")
    renewed = datetime.fromisoformat(lease["spec"]["renewTime"].replace("Z", "+00:00"))
    if (
        renewed.tzinfo is None
        or not 0 <= (datetime.now(timezone.utc) - renewed).total_seconds() < 60
        or lease["spec"].get("holderIdentity") != name
        or not any(
            owner.get("uid") == metadata["uid"]
            for owner in lease["metadata"].get("ownerReferences", [])
        )
    ):
        raise ProofError("CPU node lease is stale or not bound to the current node UID")
    return {
        "name": name,
        "uid": metadata["uid"],
        "hostname": labels["kubernetes.io/hostname"],
        "instance_type": instance_type,
        "region": labels["topology.kubernetes.io/region"],
        "inventory_sha256": digest(
            {
                "labels": labels,
                "provider": provider,
                "capacity": status["capacity"],
                "allocatable": status["allocatable"],
            }
        ),
    }
