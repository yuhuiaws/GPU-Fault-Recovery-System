from __future__ import annotations

from typing import Any

from kubernetes import client

from tests.hyperpod.test_hyperpod_spares import FakeCore


class TypedCore(FakeCore):
    """SDK-shaped reads over the existing in-memory Kubernetes fake."""

    def read_node(self, node_id: str) -> Any:
        value = super().read_node(node_id)
        metadata = value["metadata"]
        return client.V1Node(
            metadata=client.V1ObjectMeta(
                name=metadata.get("name", node_id),
                resource_version=metadata["resourceVersion"],
                annotations=metadata["annotations"],
                labels=metadata["labels"],
            ),
            spec=client.V1NodeSpec(
                provider_id=value["spec"].get("providerID"),
                unschedulable=value["spec"].get("unschedulable", False),
            ),
            status=client.V1NodeStatus(
                conditions=[
                    client.V1NodeCondition(type=row["type"], status=row["status"])
                    for row in value["status"]["conditions"]
                ]
            ),
        )

    def list_node(self, label_selector: str | None = None) -> Any:
        return client.V1NodeList(items=[self.read_node(name) for name in self.nodes])
