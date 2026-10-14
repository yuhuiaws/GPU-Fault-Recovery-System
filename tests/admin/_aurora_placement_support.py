"""The CPU cluster and Aurora as the placement reads see them, recorded.

``World.run`` answers the four reads ``aurora_placement`` makes (the Running
control-plane Pods, their nodes, ``describe-db-clusters`` and the filtered
``describe-db-instances``) and refuses anything else, mutations included.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError

CLUSTER_ID = "aurora-a"
REGION = "us-west-2"


class World:
    def __init__(
        self,
        *,
        writer: str = "aurora-a-reader",
        zones: dict[str, str] | None = None,
        tiers: dict[str, int | None] | None = None,
        statuses: dict[str, str] | None = None,
        nodes: dict[str, str] | None = None,
        cluster_status: str = "available",
    ) -> None:
        self.writer = writer
        self.zones = zones or {
            "aurora-a-writer": "us-west-2a",
            "aurora-a-reader": "us-west-2b",
        }
        self.tiers: dict[str, int | None] = tiers or {
            "aurora-a-writer": 0,
            "aurora-a-reader": 1,
        }
        self.statuses = statuses or {name: "available" for name in self.zones}
        self.nodes = nodes or {"cpu-1": "us-west-2a", "cpu-2": "us-west-2a"}
        self.pods = [
            ("gpu-fault-api-ha-0", "cpu-1"),
            ("gpu-fault-api-ha-1", "cpu-2"),
            ("gpu-fault-control-worker-0", "cpu-1"),
        ]
        self.cluster_status = cluster_status
        # The identifier RDS answers with; ``None`` echoes the one asked for.
        self.answered_cluster: str | None = None
        self.calls: list[list[str]] = []
        self.timeouts: list[float | None] = []
        self.failure: str | None = None

    def run(self, arguments: Any, **kwargs: Any) -> str:
        argv = [str(item) for item in arguments]
        self.calls.append(argv)
        self.timeouts.append(kwargs.get("timeout_seconds"))
        if kwargs.get("mutate"):
            raise AssertionError(f"a placement read mutated: {argv}")
        if self.failure is not None and self.failure in " ".join(argv):
            raise BootstrapError(f"command failed (1): {argv[0]}: unit outage")
        if argv[0] == "kubectl" and "pod" in argv:
            return json.dumps(
                {
                    "items": [
                        {"metadata": {"name": pod}, "spec": {"nodeName": node}}
                        for pod, node in self.pods
                    ]
                }
            )
        if argv[0] == "kubectl" and "node" in argv:
            names = argv[argv.index("node") + 1 : argv.index("-o")]
            items = [
                {
                    "kind": "Node",
                    "metadata": {
                        "name": name,
                        "labels": {"topology.kubernetes.io/zone": self.nodes[name]},
                    },
                }
                for name in names
            ]
            # kubectl answers one name with the object, several with a List.
            return json.dumps(
                items[0] if len(items) == 1 else {"kind": "List", "items": items}
            )
        if argv[:3] == ["aws", "rds", "describe-db-clusters"]:
            asked = argv[argv.index("--db-cluster-identifier") + 1]
            return json.dumps(
                {
                    "DBClusters": [
                        {
                            "DBClusterIdentifier": self.answered_cluster or asked,
                            "Status": self.cluster_status,
                            "DBClusterMembers": [
                                {
                                    "DBInstanceIdentifier": name,
                                    "IsClusterWriter": name == self.writer,
                                    "PromotionTier": self.tiers[name],
                                }
                                for name in self.zones
                            ],
                        }
                    ]
                }
            )
        if argv[:3] == ["aws", "rds", "describe-db-instances"]:
            return json.dumps(
                {
                    "DBInstances": [
                        {
                            "DBInstanceIdentifier": name,
                            "AvailabilityZone": zone,
                            "DBInstanceStatus": self.statuses[name],
                        }
                        for name, zone in self.zones.items()
                    ]
                }
            )
        raise AssertionError(f"unexpected command: {argv}")


def site(path: Path) -> SimpleNamespace:
    (path / "site.yaml").write_text("name: staging\n", encoding="utf-8")
    return SimpleNamespace(
        source=path / "site.yaml",
        repository_root=path,
        environment={},
        source_sha256="c" * 64,
        release_config={
            "site_name": "staging",
            "aws_region": REGION,
            "cpu_kubeconfig": str(path / "cpu.kubeconfig"),
            "namespace": "gpu-fault-system",
            "health": {"aurora_cluster_id": CLUSTER_ID},
            "clusters": [],
        },
    )
