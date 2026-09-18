from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.admin import deadlines, node_key_custody, node_key_custody_chain
from gpu_fault.admin import node_key_custody_activation as engine
from gpu_fault.admin.execution import current_deadline, deadline_scope
from gpu_fault.admin.node_key_custody_admin import CustodyReconciliationRequired
from gpu_fault.admin.node_key_custody_crypto import parse
from gpu_fault.admin.node_key_custody_models import Authorization, Signed, canonical
from tests.admin import test_node_key_custody_admin_rotation as rotation_cases
from tests.admin.test_node_key_custody_admin import provision

installed = rotation_cases.installed


def writer_setup(installed, monkeypatch, *, remaining=1):
    world, _, request_path = installed
    request = json.loads(request_path.read_text())
    approval_path = Path(request["authorization"])
    envelope = parse(Signed[Authorization], approval_path.read_bytes())
    approved = envelope.statement.model_copy(
        update={"expires_at": datetime.now(timezone.utc) + timedelta(hours=1)}
    )
    envelope = world.authorities.envelope(approved, "approval")
    approval_path.write_bytes(canonical(envelope))
    world.activation_io.authorization = approved
    world.configure()
    handle = engine.CustodyActivation(world.activation_io, world.context(), envelope)
    handle.prepare()
    saved = handle.load()
    wall = [saved.deadline - timedelta(seconds=remaining)]
    monotonic = [time.monotonic()]

    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return wall[0] if tz is None else wall[0].astimezone(tz)

    for module in (engine, node_key_custody, node_key_custody_chain):
        monkeypatch.setattr(module, "datetime", ClockDateTime)
    monkeypatch.setattr(deadlines.time, "monotonic", lambda: monotonic[0])
    return world, handle, wall, monotonic


def test_actual_key_helper_inherits_the_fixed_activation_deadline(
    installed, monkeypatch
):
    world, handle, wall, _mono = writer_setup(installed, monkeypatch, remaining=10)
    original = world.run
    budgets = []

    def run(arguments, **kwargs):
        if str(arguments[0]).endswith("provision-node-action-keys.sh"):
            budgets.append(current_deadline())
            assert "--custody-activation-state" in arguments, (
                "the real writer must receive its saved activation state"
            )
            assert "--custody-activation-sha256" in arguments, (
                "the writer state must be content-bound"
            )
        return original(arguments, **kwargs)

    monkeypatch.setattr(world, "run", run)
    with deadline_scope("long parent bootstrap", 7200):
        result = provision(world)
    assert result["runtime_activation"] == "DEPLOYED_NOT_WITNESSED", (
        "a bounded real key writer must support successful activation"
    )
    assert (
        len(budgets) == 1
        and budgets[0].remaining() <= (handle.load().deadline - wall[0]).total_seconds()
    ), (
        "the helper must inherit the activation deadline, not the longer bootstrap budget"
    )


@pytest.mark.parametrize("clock", ["both", "wall", "monotonic"])
def test_resume_near_deadline_cannot_write_keys_after_it(installed, monkeypatch, clock):
    world, handle, wall, monotonic = writer_setup(installed, monkeypatch)
    original = world.run
    before = copy.deepcopy(world.api.state["secrets"])

    def run(arguments, **kwargs):
        if str(arguments[0]).endswith("provision-node-action-keys.sh"):
            if clock in {"wall", "both"}:
                wall[0] += timedelta(seconds=2)
            if clock in {"monotonic", "both"}:
                monotonic[0] += 2
        return original(arguments, **kwargs)

    monkeypatch.setattr(world, "run", run)
    with deadline_scope("long parent bootstrap", 7200):
        with pytest.raises(CustodyReconciliationRequired):
            provision(world)
    assert world.api.state["secrets"] == before, (
        "no key map may change after the fixed deadline"
    )
    state = handle.load()
    assert "KEYS_PROVISIONED" in state.started, (
        "failed writer admission must retain its intent"
    )
    assert "KEYS_PROVISIONED" not in state.completed, (
        "expired writes cannot gain completion evidence"
    )


def test_writer_finishing_inside_the_original_window_can_activate(
    installed, monkeypatch
):
    world, handle, wall, monotonic = writer_setup(installed, monkeypatch, remaining=5)
    original = world.run
    before = copy.deepcopy(world.api.state["secrets"])

    def run(arguments, **kwargs):
        if str(arguments[0]).endswith("provision-node-action-keys.sh"):
            wall[0] += timedelta(seconds=1)
            monotonic[0] += 1
        return original(arguments, **kwargs)

    monkeypatch.setattr(world, "run", run)
    with deadline_scope("long parent bootstrap", 7200):
        assert provision(world)["runtime_activation"] == "DEPLOYED_NOT_WITNESSED", (
            "work completed inside the original window must be allowed to activate"
        )
    assert (
        world.api.state["secrets"]["gpu"]["data"]["node-a"]
        != before["gpu"]["data"]["node-a"]
    ), "the authorized node key must change during successful provisioning"
    assert (
        world.api.state["secrets"]["gpu"]["data"]["node-b"]
        == before["gpu"]["data"]["node-b"]
    ), "successful provisioning must preserve the sibling node key"
    assert handle.load().deadline == wall[0] + timedelta(seconds=4), (
        "successful resume must not reset or extend the saved deadline"
    )


@pytest.mark.parametrize("drift", ["uid", "generation", "early-restore"])
def test_resumed_key_writer_does_not_reacquire_a_lost_owned_wave(
    installed, monkeypatch, drift
):
    world, handle, _wall, _mono = writer_setup(installed, monkeypatch, remaining=30)
    before = copy.deepcopy(world.api.state["secrets"])
    count = world.helper_calls
    if drift == "uid":
        world.activation_io.wave["metadata"]["uid"] = "replacement"
    elif drift == "generation":
        world.activation_io.wave["data"]["generation"] = "foreign"
    else:
        world.activation_io.wave["data"] = copy.deepcopy(
            handle.state.snapshot["wave"]["data"]
        )
    with pytest.raises(CustodyReconciliationRequired):
        provision(world)
    assert world.helper_calls == count, (
        "lost ownership must reject before starting the key helper"
    )
    assert world.api.state["secrets"] == before, (
        "lost wave ownership cannot authorize key replacement"
    )


def test_real_helper_rechecks_wave_after_transfer_validation_before_sending_write(
    installed, monkeypatch
):
    world, _handle, _wall, _mono = writer_setup(installed, monkeypatch, remaining=30)
    original = node_key_custody.ProvisionCustody.transfer
    before = copy.deepcopy(world.api.state["secrets"])

    def transfer(custody, document, *, gpu):
        original(custody, document, gpu=gpu)
        world.activation_io.wave["data"]["generation"] = "other-owner"

    monkeypatch.setattr(node_key_custody.ProvisionCustody, "transfer", transfer)
    with pytest.raises(CustodyReconciliationRequired):
        provision(world)
    assert world.api.state["secrets"] == before, (
        "the real write barrier must catch post-transfer wave drift"
    )


def test_expiry_between_secret_writes_preserves_partial_intent_without_completion(
    installed, monkeypatch
):
    world, handle, wall, monotonic = writer_setup(installed, monkeypatch)
    original = world.private_run
    before = copy.deepcopy(world.api.state["secrets"])

    def command(arguments, **kwargs):
        result = original(arguments, **kwargs)
        if "replace" in arguments and "gpu-context" in arguments:
            wall[0] += timedelta(seconds=2)
            monotonic[0] += 2
        return result

    monkeypatch.setattr(world, "private_run", command)
    with pytest.raises(CustodyReconciliationRequired):
        provision(world)
    assert world.api.state["secrets"]["gpu"]["data"] != before["gpu"]["data"], (
        "the partial-write control must actually write GPU keys before expiry"
    )
    assert world.api.state["secrets"]["cpu"] == before["cpu"], (
        "the second key write must not start after the deadline"
    )
    assert "KEYS_PROVISIONED" not in handle.load().completed, (
        "a partial key write must remain incomplete for reconciliation"
    )


def test_real_writer_rejects_a_wall_clock_regression_before_the_key_mutation(
    installed, monkeypatch
):
    world, _handle, wall, monotonic = writer_setup(installed, monkeypatch, remaining=30)
    before = copy.deepcopy(world.api.state["secrets"])
    original = node_key_custody.ProvisionCustody.transfer

    def transfer(custody, document, *, gpu):
        original(custody, document, gpu=gpu)
        wall[0] -= timedelta(seconds=1)
        monotonic[0] += 1

    monkeypatch.setattr(node_key_custody.ProvisionCustody, "transfer", transfer)
    with pytest.raises(CustodyReconciliationRequired):
        provision(world)
    assert world.api.state["secrets"] == before, (
        "clock regression must stop actual key writes, not only their completion record"
    )
