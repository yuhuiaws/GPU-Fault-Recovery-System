from __future__ import annotations

import pytest

from gpu_fault.admin import node_key_custody_activation_io as module
from gpu_fault.admin.node_key_custody_models import CustodyError
from tests.admin._security_activation_io_world import ActivationWorld


@pytest.fixture
def world(tmp_path, monkeypatch):
    return ActivationWorld(tmp_path, monkeypatch)


def test_real_activation_adapter_orders_all_consumers_and_preserves_siblings(world):
    io = world.io
    before_peer = world.store.get_agent("cluster-a", "node-b")
    snapshot = io.capture()
    io.verify(snapshot)
    io.guard(snapshot)
    assert (
        world.store.get_agent("cluster-a", "node-a").lifecycle_state.value == "DRAINING"
    )
    io.fence(snapshot)
    world.key_values["node-a"] = "new-fixture-" + "z" * 40
    world.write_keys()
    world.bind_provisioned()
    io.refresh_executor(snapshot)
    io.install(snapshot)
    io.refresh_cpu(snapshot)
    observed = io.observe(snapshot)
    assert observed["independent_witness"] == "NOT_PROVED"
    assert world.store.get_agent("cluster-a", "node-b") == before_peer
    assert io.unfence(snapshot)["restored"] is True
    assert world.records["gpu", "configmap", "wave"]["data"] == snapshot["wave"]["data"]
    consumers = [name for plane, kind, name in world.patches if kind == "deployment"]
    assert consumers[0] == module.GPU_EXECUTOR_DEPLOYMENT
    assert consumers[-1] == module.CPU_INGRESS_DEPLOYMENT
    assert world.patches.count(("gpu", "node", "node-a")) == 1
    assert all(
        not (kind == "node" and name == "node-b") for _, kind, name in world.patches
    ), "single-node key activation must not mutate its sibling Node"
    assert io.current(snapshot)["agent_incarnation_id"] == "new-agent"
    commands = [call for call in world.calls if call[0] == "kubectl"]
    assert commands, "the activation control must exercise actual command construction"
    assert all(
        command[:3]
        == (
            "kubectl",
            "--kubeconfig",
            str(
                world.context.gpu_kubeconfig
                if "gpu-context" in command
                else world.context.cpu_kubeconfig
            ),
        )
        for command in commands
    ), "every activation command must retain its bound CPU or GPU kubeconfig"


@pytest.mark.parametrize(
    "defect",
    ["node-uid", "sibling-key", "deployment-uid", "deployment-spec", "foreign-restart"],
)
def test_activation_rejects_drift_before_mutation(world, defect):
    snapshot = world.io.capture()
    if defect == "node-uid":
        world.records["gpu", "node", "node-a"]["metadata"]["uid"] = "replacement"
    elif defect == "sibling-key":
        world.key_values["node-b"] = "foreign-fixture-" + "w" * 40
        world.write_keys()
    else:
        value = world.records["gpu", "deployment", module.GPU_EXECUTOR_DEPLOYMENT]
        if defect == "deployment-uid":
            value["metadata"]["uid"] = "replacement"
        elif defect == "deployment-spec":
            value["spec"]["replicas"] = 2
        else:
            value["spec"]["template"]["metadata"]["annotations"][
                module.ACTIVATION_ANNOTATION
            ] = "9" * 64
    with pytest.raises(CustodyError, match="changed"):
        world.io.verify(snapshot)
    assert world.patches == []


def test_projection_updated_after_startup_cannot_certify_cached_signers(world):
    snapshot = world.io.capture()
    world.io.guard(snapshot)
    world.io.fence(snapshot)
    world.bind_provisioned()
    world.projection = "late"
    with pytest.raises(CustodyError, match="projection did not precede"):
        world.io.refresh_executor(snapshot)
    assert (
        world.patches.count(("gpu", "deployment", module.GPU_EXECUTOR_DEPLOYMENT))
        == module.MAX_CONSUMER_RELOADS
    )
    assert (
        world.store.get_agent("cluster-a", "node-a").lifecycle_state.value == "DRAINING"
    )


@pytest.mark.parametrize("defect", ["namespace", "wave-data", "wave-uid"])
def test_activation_does_not_take_over_a_foreign_wave(world, defect):
    snapshot = world.io.capture()
    value = world.records["gpu", "configmap", "wave"]
    if defect == "wave-uid":
        value["metadata"]["uid"] = "replacement"
    elif defect == "wave-data":
        value["data"]["generation"] = "foreign-writer"
    else:
        value["data"]["allowed-nodes"] = "node-b"
    with pytest.raises(CustodyError, match="ownership changed"):
        world.io.fence(snapshot)
    assert world.patches == []


def test_late_sibling_runtime_change_prevents_activation_completion(world):
    snapshot = world.io.capture()
    peer = world.store.get_agent("cluster-a", "node-b")
    world.store.save_agent(peer.model_copy(update={"generation": peer.generation + 1}))
    with pytest.raises(CustodyError, match="sibling identity"):
        world.io.observe(snapshot)


def test_projection_probe_reads_real_file_metadata_without_printing_material(
    tmp_path, monkeypatch, capsys
):
    import io
    import json
    import sys

    material = "private-unit-key-" + "k" * 40
    (tmp_path / "node-a").write_text(material)
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_KEYS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"node_id":"node-a"}'))
    exec(compile(module.PROJECTED_KEY_PROBE, "<projection-probe>", "exec"), {})
    printed = capsys.readouterr().out
    result = json.loads(printed)
    assert len(result["sha256"]) == 64 and result["mtime_ns"] > 0
    assert material not in printed
