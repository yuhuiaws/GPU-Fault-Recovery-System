"""Bind native node cleanup to the GPU fleet approved before teardown."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from gpu_fault.installation_inventory import InstalledUnitInventory
from scripts.e2e.regional.boot032_contract import (
    Settings,
    UninstallCaseError,
    cluster_specs,
    mapping,
    require,
)
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture


def agent_identity(agent: dict[str, Any], cluster_id: str, name: str) -> dict[str, Any]:
    require(
        agent.get("cluster_id") == cluster_id
        and agent.get("node_id") == name
        and agent.get("lifecycle_state") == "ACTIVE"
        and isinstance(agent.get("node_instance_id"), str)
        and agent["node_instance_id"],
        "fleet node ownership or instance identity is unproved",
    )
    inventory = InstalledUnitInventory.model_validate(
        agent.get("installed_unit_inventory")
    )
    require(inventory.units, "fleet node has no installed unit inventory")
    return {
        "cluster_id": cluster_id,
        "node_id": name,
        "node_instance_id": agent["node_instance_id"],
        "lifecycle_state": "ACTIVE",
        "installed_unit_inventory": inventory.model_dump(mode="json"),
    }


def node_inventory(fixture: RegionalLiveFixture) -> dict[str, Any]:
    nodes = fixture.gpu_nodes()
    require(nodes, "GPU node inventory is empty")
    result = {}
    for node in nodes:
        name, uid = node.get("name"), node.get("uid")
        if not isinstance(name, str) or not name or not isinstance(uid, str) or not uid:
            raise UninstallCaseError("GPU node name or UID is missing")
        require(name not in result, "GPU node identity is duplicated")
        require(
            node.get("ready") == "True" and node.get("unschedulable") is False,
            "GPU node is not ready for isolated maintenance",
        )
        snapshot = fixture.store_snapshot(node=name, queue_attempts=1)
        agent = mapping(snapshot.get("agent"), "GPU node fleet identity is missing")
        result[name] = {
            "uid": uid,
            "agent": agent_identity(agent, fixture.settings.cluster_id, name),
        }
    require(
        len({item["uid"] for item in result.values()}) == len(result),
        "GPU node UIDs alias",
    )
    return result


def distinct_fleets(target: dict[str, Any], protected: dict[str, Any]) -> None:
    identities = [
        [
            (row["uid"], row["agent"]["node_instance_id"])
            for nodes in site["nodes"].values()
            for row in nodes.values()
        ]
        for site in (target, protected)
    ]
    for index in (0, 1):
        left, right = ([row[index] for row in values] for values in identities)
        require(
            len(set(left)) == len(left)
            and len(set(right)) == len(right)
            and not set(left) & set(right),
            "GPU node or instance identity aliases another approved cluster",
        )


def verify_fleet_receipt(
    settings: Settings,
    binding: dict[str, Any],
    document: dict[str, Any],
) -> None:
    rows = document.get("fleet_snapshot")
    if not isinstance(rows, list):
        raise UninstallCaseError("native fleet unit inventory is missing")
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        document.get("fleet_snapshot_sha256") == digest,
        "native fleet snapshot digest changed",
    )
    approved = binding["target"]["nodes"]
    expected = {
        (cluster_id, name): item["agent"]
        for cluster_id, nodes in approved.items()
        for name, item in nodes.items()
    }
    actual = {}
    for value in rows:
        agent = mapping(value, "native fleet identity is malformed")
        key = agent.get("cluster_id"), agent.get("node_id")
        require(
            key in expected and key not in actual,
            "native fleet targets differ from approval",
        )
        actual[key] = agent_identity(agent, str(key[0]), str(key[1]))
    require(actual == expected, "native fleet unit inventory differs from approval")
    targets = {
        "gpu:" + spec["context"]: {
            name: item["uid"] for name, item in approved[spec["cluster_id"]].items()
        }
        for spec in cluster_specs(settings.target)[1:]
    }
    require(
        document.get("node_targets") == targets,
        "native node targets differ from approval",
    )
