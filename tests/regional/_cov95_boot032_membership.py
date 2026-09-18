from __future__ import annotations

import copy
from dataclasses import replace

import yaml

from gpu_fault.admin.bootstrap_common import Arn
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import InstallationResourceSnapshot
from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_lifecycle as lifecycle
from scripts.e2e.regional import regional_case_contract as cases
from tests.regional._cov95_boot032_support import CA, arn, resource_snapshot
from tests.regional._cov95_boot032_world import World


def two_cluster_world(root, monkeypatch):
    world = World(root, monkeypatch)
    name = "accepted-z-primary"
    source = world.protected.source
    document = yaml.safe_load(source.read_text())
    added = copy.deepcopy(document["spec"]["clusters"][0])
    added.update(
        clusterId=name,
        context=name,
        eksClusterArn=arn("eks", name),
        hyperpodClusterName=name,
    )
    for field in ("tokenFile", "fleetMasterFile"):
        path = source.parent / "secure" / (name + "-" + field)
        path.write_text("unit-fixture-" + field + "-material")
        path.chmod(0o600)
        added[field] = str(path)
    document["spec"]["clusters"].append(added)
    source.write_text(yaml.safe_dump(document))
    kube = contract.kubeconfig(world.protected, "gpu")
    config = yaml.safe_load(kube.read_text())
    config["contexts"].append(
        {"name": name, "context": {"cluster": name, "user": "fixture"}}
    )
    config["clusters"].append(
        {
            "name": name,
            "cluster": {
                "server": f"https://{name}.example.invalid",
                "certificate-authority-data": CA,
            },
        }
    )
    kube.write_text(yaml.safe_dump(config))
    world.protected = load_site(source)
    world.settings = replace(
        world.settings, protected=world.protected, protected_cluster_id=name
    )
    snapshot = world.snapshots[world.protected.metadata_name]
    added_rows = [
        row.model_copy(
            update={
                "resource_key": f"cluster/{name}/"
                + row.resource_type.removeprefix("gpu_"),
                "resource_id": name,
                "resource_arn": arn("eks", name)
                if row.resource_type == "gpu_eks"
                else None,
            }
        )
        for row in snapshot.resources
        if row.resource_type in {"gpu_eks", "gpu_hyperpod"}
    ]
    value = InstallationResourceSnapshot(
        site_id=snapshot.site_id, resources=[*snapshot.resources, *added_rows]
    )
    world.snapshots[value.site_id] = value.model_copy(
        update={"source_sha256": value.digest()}
    )
    world.add_site(world.protected)
    order = root / "formal-order.yaml"
    order.write_text(
        "phases:\n- sequence: 17\n  entries:\n  - case: GF-REGIONAL-COLLECT-015\n"
        f"- sequence: 18\n  entries:\n  - case: {contract.CASE_ID}\n"
        "    predecessor: GF-REGIONAL-COLLECT-015\n"
    )
    monkeypatch.setattr(cases, "ORDER_PATH", order)
    monkeypatch.setattr(lifecycle, "predecessor_path", cases.predecessor_path)
    cases.expanded_order.cache_clear()
    return world


def cross_region_world(root, monkeypatch):
    world = World(root, monkeypatch)
    source = world.protected.source

    def regional(value):
        if isinstance(value, dict):
            return {key: regional(item) for key, item in value.items()}
        if isinstance(value, list):
            return [regional(item) for item in value]
        if isinstance(value, str) and value.startswith("arn:"):
            parsed = Arn.parse(value)
            if parsed.region:
                return f"arn:{parsed.partition}:{parsed.service}:us-west-2:{parsed.account}:{parsed.resource}"
        return value

    document = regional(yaml.safe_load(source.read_text()))
    document["spec"]["awsRegion"] = "us-west-2"
    source.write_text(yaml.safe_dump(document))
    world.protected = load_site(source)
    world.settings = replace(world.settings, protected=world.protected)
    world.snapshots[world.protected.metadata_name] = resource_snapshot(world.protected)
    world.add_site(world.protected)
    return world
