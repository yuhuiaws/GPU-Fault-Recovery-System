from __future__ import annotations

import pytest

from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_fleet as fleet
from scripts.e2e.regional import boot032_journal as journal
from scripts.e2e.regional import boot032_native as adapter
from scripts.e2e.regional import boot032_observe as observe
from tests.regional._cov95_boot032_approval import approve, execute
from tests.regional._cov95_boot032_native import NativeHarness
from tests.regional._cov95_boot032_world import World


def test_node_replacement_after_plan_is_refused_before_native_cleanup(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    _settings, _plan, deadline = approve(world)
    spec = contract.cluster_specs(world.target)[1]
    key = (world.target.metadata_name, spec["context"], "node", "node-a", "")
    world.uids[key] = "replacement-node-uid"
    with pytest.raises(contract.UninstallCaseError, match="node"):
        execute(world, deadline)
    assert harness.events == [], (
        "unapproved node incarnation must stop before uninstalling its units"
    )


def test_pause_requires_sealed_native_fleet_unit_inventory(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    NativeHarness(world, monkeypatch)
    settings, plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "capture real cleanup checkpoint"
    )
    path = settings.native_dir / "kubernetes-cleanup.json"
    document = journal.CLEANUP.read_state(path)
    document["fleet_snapshot"] = None
    journal.CLEANUP.atomic_write(path, document)
    with pytest.raises(contract.UninstallCaseError, match="fleet"):
        journal.cleanup_complete(settings, plan["details"]["binding"])


@pytest.mark.parametrize("identity", ["uid", "instance"])
def test_sacrificial_gpu_nodes_cannot_alias_accepted_hosts(
    tmp_path, monkeypatch, identity
):
    world = World(tmp_path, monkeypatch)
    target, protected = (
        contract.cluster_specs(site)[1] for site in (world.target, world.protected)
    )
    if identity == "uid":
        world.uids[
            (world.target.metadata_name, target["context"], "node", "node-a", "")
        ] = world.uids[
            (world.protected.metadata_name, protected["context"], "node", "node-a", "")
        ]
    else:
        world.agents[(world.target.metadata_name, target["context"])][
            "node_instance_id"
        ] = world.agents[(world.protected.metadata_name, protected["context"])][
            "node_instance_id"
        ]
    with pytest.raises(contract.UninstallCaseError, match="aliases"):
        adapter.NativeBackend(world.settings).initial()
    assert not world.settings.native_dir.exists(), (
        "a shared GPU host cannot be a disposable target"
    )


@pytest.mark.parametrize(
    "node",
    [
        {"name": "", "uid": "uid"},
        {"name": "node-a", "uid": ""},
        {"name": "node-a", "uid": "uid", "ready": "Unknown", "unschedulable": False},
    ],
)
def test_unknown_gpu_node_identity_is_not_approved(tmp_path, monkeypatch, node):
    world = World(tmp_path, monkeypatch)
    fixture = observe.regional_fixture(
        world.target, contract.cluster_specs(world.target)[1]
    )
    fixture.gpu_nodes = lambda: [node]
    with pytest.raises(contract.UninstallCaseError, match="node"):
        fleet.node_inventory(fixture)


@pytest.mark.parametrize(
    "change",
    [
        {"node_id": "another-node"},
        {"cluster_id": "another-cluster"},
        {"lifecycle_state": "RETIRED"},
        {"node_instance_id": None},
        {"installed_unit_inventory": None},
    ],
)
def test_plan_requires_original_active_fleet_and_full_unit_inventory(
    tmp_path, monkeypatch, change
):
    world = World(tmp_path, monkeypatch)
    spec = contract.cluster_specs(world.target)[1]
    world.agents[(world.target.metadata_name, spec["context"])].update(change)
    with pytest.raises((contract.UninstallCaseError, ValueError)):
        adapter.NativeBackend(world.settings).initial()
    assert not world.settings.native_dir.exists(), (
        "missing fleet ownership must block every uninstall phase"
    )
