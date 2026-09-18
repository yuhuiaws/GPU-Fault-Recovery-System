from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from gpu_fault.admin import node_key_custody_activation as activation
from gpu_fault.admin.bootstrap_common import BootstrapMutationRequired
from gpu_fault.admin.node_key_custody_crypto import parse
from gpu_fault.admin.node_key_custody_models import Authorization, CustodyError, Signed
from tests.admin import test_node_key_custody_admin_rotation as rotation_cases
from tests.admin.test_node_key_custody_admin import provision

installed = rotation_cases.installed


def selected(world, request_path):
    request = json.loads(request_path.read_text())
    approval = parse(Signed[Authorization], Path(request["authorization"]).read_bytes())
    return activation.CustodyActivation(world.activation_io, world.context(), approval)


@pytest.mark.parametrize(
    "phase", ["guard", "fence", "executor", "install", "cpu", "observe", "unfence"]
)
def test_activation_intent_survives_each_failed_phase_and_resumes_without_rekeying(
    installed, phase
):
    world, _, request_path = installed
    world.configure()
    world.activation_io.failure = phase
    with pytest.raises(RuntimeError):
        provision(world)
    handle = selected(world, request_path)
    pending = handle.load()
    assert pending is not None and pending.started
    with pytest.raises(BootstrapMutationRequired, match="activation is pending"):
        handle.probe()
    writes = world.helper_calls
    world.activation_io.failure = None
    result = provision(world)
    assert result["runtime_activation"] == "DEPLOYED_NOT_WITNESSED"
    assert world.helper_calls == (writes + 1 if phase in {"guard", "fence"} else writes)
    assert set(handle.load().completed) == set(activation.ACTIVATION_STEPS)
    assert world.activation_io.events.count("capture") == 1
    assert provision(world, probe=True) == result


def test_completed_activation_probe_cannot_restart_any_consumer(installed):
    world, _, request_path = installed
    world.configure()
    provision(world)
    events = list(world.activation_io.events)
    handle = selected(world, request_path)
    assert handle.probe()["runtime_activation"] == "DEPLOYED_NOT_WITNESSED"
    assert world.activation_io.events == events
    assert not any("sign" in event for event in world.activation_io.events), (
        "a completed activation probe must never sign new authority or evidence"
    )


@pytest.mark.parametrize(
    "defect", ["foreign-approval", "deadline", "phase-hole", "duplicate-start"]
)
def test_activation_rejects_unbound_or_impossible_progress(installed, defect):
    world, _, request_path = installed
    world.configure()
    handle = selected(world, request_path)
    handle.prepare()
    document = json.loads(handle.path.read_text())
    if defect == "foreign-approval":
        document["authorization_sha256"] = "f" * 64
    elif defect == "deadline":
        document["deadline"] = (
            handle.authorization.statement.expires_at + timedelta(seconds=1)
        ).isoformat()
    elif defect == "phase-hole":
        document["completed"]["NODE_INSTALLED"] = {}
    else:
        document["started"].append(document["started"][0])
    handle.path.write_text(json.dumps(document))
    with pytest.raises(CustodyError, match="inconsistent|foreign"):
        handle.load()


def test_activation_expiry_never_extends_its_authorization_or_restores_old_key(
    installed,
):
    world, _, request_path = installed
    world.configure()
    handle = selected(world, request_path)
    handle.prepare()
    original = handle.path.read_bytes()
    handle.now = lambda: handle.authorization.statement.expires_at + timedelta(
        seconds=1
    )
    with pytest.raises(CustodyError, match="approved window"):
        handle.finish()
    assert handle.path.read_bytes() == original
    assert world.activation_io.events == ["capture", "guard", "fence"]


def test_step_returning_after_deadline_cannot_record_success(installed):
    world, _, request_path = installed
    world.configure()
    handle = selected(world, request_path)
    handle.prepare()
    original = handle.state

    def late_return(_snapshot):
        handle.now = lambda: original.deadline
        return {"looks_successful": True}

    with pytest.raises(CustodyError, match="deadline|approved window"):
        handle.step("KEYS_PROVISIONED", late_return)
    saved = handle.load()
    assert "KEYS_PROVISIONED" in saved.started
    assert "KEYS_PROVISIONED" not in saved.completed


def test_completed_journal_must_prove_completion_inside_the_fixed_window(installed):
    world, _, request_path = installed
    world.configure()
    provision(world)
    handle = selected(world, request_path)
    document = json.loads(handle.path.read_text())
    document["completed"]["UNFENCED"]["completed_at"] = document["deadline"]
    handle.path.write_text(json.dumps(document))
    with pytest.raises(CustodyError, match="completion timing"):
        handle.probe()


def test_completed_readonly_probe_does_not_reopen_the_expired_mutation_window(
    installed,
):
    world, _, request_path = installed
    world.configure()
    provision(world)
    handle = selected(world, request_path)
    calls = list(world.activation_io.events)
    handle.now = lambda: handle.authorization.statement.expires_at + timedelta(days=1)
    assert handle.probe()["runtime_activation"] == "DEPLOYED_NOT_WITNESSED"
    assert world.activation_io.events == calls
