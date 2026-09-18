from __future__ import annotations

import copy

import pytest

from gpu_fault.admin import node_key_custody_activation as engine
from gpu_fault.admin.node_key_custody_models import Authorization, CustodyError, Signed
from tests.admin._security_activation_io_world import ActivationWorld


@pytest.fixture
def fenced(tmp_path, monkeypatch):
    world = ActivationWorld(tmp_path, monkeypatch)
    approval = Signed[Authorization](
        statement=world.authorization,
        signer_sha256="a" * 64,
        signature="local-transport-only-" + "x" * 64,
    )
    handle = engine.CustodyActivation(world.io, world.context, approval)
    handle.prepare()
    world.bind_provisioned()
    return world, approval, handle


def lose_wave(world, drift):
    current = world.records["gpu", "configmap", "wave"]
    if drift == "uid":
        current["metadata"]["uid"] = "recreated-wave"
    elif drift == "generation":
        current["data"]["generation"] = "another-owner"
    else:
        current["data"] = {
            "allowed-nodes": "*",
            "max-unavailable": "2",
            "generation": "steady",
        }


@pytest.mark.parametrize("drift", ["uid", "generation", "early-restore"])
def test_resume_rejects_an_already_lost_wave_before_any_later_mutation(fenced, drift):
    world, approval, _handle = fenced
    before = copy.deepcopy(world.patches)
    lose_wave(world, drift)
    resumed = engine.CustodyActivation(world.io, world.context, approval)
    with pytest.raises(CustodyError, match="owned installer wave"):
        resumed.prepare()
    assert world.patches == before, "resume must not mutate after losing its saved wave"


@pytest.mark.parametrize("operation", ["refresh_executor", "install", "refresh_cpu"])
@pytest.mark.parametrize("drift", ["uid", "generation", "early-restore"])
def test_each_later_io_operation_revalidates_the_owned_wave(fenced, operation, drift):
    world, _, handle = fenced
    before = copy.deepcopy(world.patches)
    lose_wave(world, drift)
    with pytest.raises(CustodyError, match="owned installer wave"):
        getattr(world.io, operation)(handle.state.snapshot)
    assert world.patches == before, (
        "a later operation must not use another owner's wave"
    )


@pytest.mark.parametrize("operation", ["refresh_executor", "install"])
def test_wave_loss_after_operation_reads_is_rechecked_at_the_actual_patch(
    fenced, monkeypatch, operation
):
    world, _, handle = fenced
    original = world.io.execute
    before = copy.deepcopy(world.patches)

    def lost_before_patch(command, **kwargs):
        lose_wave(world, "generation")
        return original(command, **kwargs)

    monkeypatch.setattr(world.io, "execute", lost_before_patch)
    with pytest.raises(CustodyError, match="owned installer wave"):
        getattr(world.io, operation)(handle.state.snapshot)
    assert world.patches == before, (
        "the final patch barrier must recheck wave ownership"
    )


def test_valid_owned_wave_can_resume_the_complete_activation(fenced):
    world, approval, _handle = fenced
    resumed = engine.CustodyActivation(world.io, world.context, approval)
    resumed.prepare()
    assert resumed.finish()["runtime_activation"] == "DEPLOYED_NOT_WITNESSED", (
        "an intact owned wave must support full activation without forging a witness"
    )
    assert world.records["gpu", "configmap", "wave"]["data"]["allowed-nodes"] == "*", (
        "completed activation must restore the original installer scope"
    )
    assert world.patches.count(("gpu", "node", "node-a")) == 1, (
        "resuming an intact activation must install its target exactly once"
    )


def test_unfence_ack_loss_only_resumes_cleanup_not_earlier_mutations(
    fenced, monkeypatch
):
    world, approval, handle = fenced
    original = world.io.unfence

    def lost_ack(snapshot):
        original(snapshot)
        raise RuntimeError("local acknowledgement lost")

    monkeypatch.setattr(world.io, "unfence", lost_ack)
    with pytest.raises(RuntimeError, match="acknowledgement"):
        handle.finish()
    assert "UNFENCED" in handle.load().started, (
        "the cleanup intent must precede mutation"
    )
    assert "UNFENCED" not in handle.load().completed, (
        "lost ACK cannot be recorded as completion"
    )
    before = copy.deepcopy(world.patches)
    monkeypatch.setattr(world.io, "unfence", original)
    resumed = engine.CustodyActivation(world.io, world.context, approval)
    resumed.prepare()
    assert resumed.finish()["runtime_activation"] == "DEPLOYED_NOT_WITNESSED", (
        "an acknowledged original wave must permit cleanup-only recovery"
    )
    assert world.patches == before, "cleanup recovery must not replay earlier mutations"
