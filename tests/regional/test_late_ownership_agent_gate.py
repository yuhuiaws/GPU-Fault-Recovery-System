from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.late_ownership import (
    OWNERSHIP_PROTOCOL,
    NodeOwnershipGate,
    OwnershipChallenge,
    OwnershipPermit,
    OwnershipRefused,
    command_identity,
    current_ownership_challenge,
    execute_with_final_ownership,
    final_ownership_boundary,
    ownership_recheck_scope,
    ownership_required,
    permit_signature,
    physical_ownership_scope,
    require_physical_ownership,
    sign_permit,
    supports_final_ownership,
)
from gpu_fault.node_agent.protocol import NodeActionCommand, SignedNodeAction

KEY = "local-only-node-key-" + "x" * 32
NOW = datetime.now(timezone.utc)


def command(**changes):
    value = NodeActionCommand(
        command_id="owned-command",
        workflow_request_id="owned-workflow",
        incident_id="owned-incident",
        fencing_token=3,
        operation=WorkflowOperation.RESET_GPU,
        node_id="node-a",
        agent_generation=7,
        gpu_uuids=["GPU-a"],
        issued_at=NOW,
        expires_at=NOW + timedelta(seconds=120),
        ownership_guard=OWNERSHIP_PROTOCOL,
    )
    return value.model_copy(update=changes)


def challenge(**changes):
    value = OwnershipChallenge(
        command_id="owned-command",
        workflow_id="owned-workflow",
        incident_id="owned-incident",
        node_id="node-a",
        boot_id="boot",
        agent_generation=7,
        fencing_token=3,
        command_sha256=command_identity(command()),
        nonce="a" * 64,
        sequence=1,
        expires_at=NOW + timedelta(seconds=90),
        boundary="AGENT_PRE_SPAWN",
    )
    return value.model_copy(update=changes)


class Clock:
    def __init__(self):
        self.now = NOW
        self.elapsed = 0.0
        self.ready = Event()

    def monotonic(self):
        self.ready.set()
        return self.elapsed


@pytest.fixture
def pending_gate():
    clock = Clock()
    gate = NodeOwnershipGate(
        secret=KEY, boot_id="boot", now=lambda: clock.now, monotonic=clock.monotonic
    )
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(gate.require, command())
    try:
        assert clock.ready.wait(2), "the owned Agent did not publish a challenge"
        current = gate.challenge(command().command_id)
        assert current is not None
        yield gate, clock, future, current
    finally:
        gate.cancel_all()
        pool.shutdown(wait=True)
        assert future.done(), "the owned Agent gate worker was not drained"


def test_native_queue_gate_cannot_run_without_a_fresh_signed_permit(pending_gate):
    gate, clock, future, current = pending_gate
    assert not future.done(), (
        "the native action must remain pending before permit authorization"
    )
    permit = sign_permit(current, KEY, allowed=True, reason="OK", now=clock.now)
    gate.authorize(permit)
    assert future.result(2) == permit
    assert gate.challenge(current.command_id) is None
    with pytest.raises(OwnershipRefused, match="STALE"):
        gate.authorize(permit)


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_id", "wrong-node"),
        ("boot_id", "wrong-boot"),
        ("workflow_id", "wrong-workflow"),
        ("incident_id", "wrong-incident"),
        ("command_id", "wrong-command"),
        ("command_sha256", "b" * 64),
        ("nonce", "b" * 64),
        ("sequence", 2),
        ("agent_generation", 8),
        ("fencing_token", 4),
    ],
)
def test_signed_wrong_identity_cannot_release_the_native_gate(
    pending_gate, field, value
):
    gate, clock, future, current = pending_gate
    permit = sign_permit(
        current.model_copy(update={field: value}),
        KEY,
        allowed=True,
        reason="OK",
        now=clock.now,
    )
    with pytest.raises(OwnershipRefused, match="STALE"):
        gate.authorize(permit)
    assert not future.done(), (
        "a permit for another identity must not release the pending action"
    )
    assert gate.challenge(current.command_id) == current


@pytest.mark.parametrize(
    "defect", ["signature", "expired", "future", "unbounded", "challenge-expired"]
)
def test_freshness_and_authentication_are_mandatory(pending_gate, defect):
    gate, clock, future, current = pending_gate
    permit = sign_permit(current, KEY, allowed=True, reason="OK", now=clock.now)
    if defect == "signature":
        permit = permit.model_copy(update={"signature": "f" * 64})
    elif defect == "expired":
        clock.now = permit.expires_at
    elif defect == "future":
        permit = sign_permit(
            current,
            KEY,
            allowed=True,
            reason="OK",
            now=clock.now + timedelta(seconds=1),
        )
    elif defect == "unbounded":
        permit = permit.model_copy(
            update={"expires_at": clock.now + timedelta(seconds=4)}
        )
        permit = permit.model_copy(update={"signature": permit_signature(permit, KEY)})
    else:
        permit = sign_permit(
            current.model_copy(update={"expires_at": clock.now}),
            KEY,
            allowed=True,
            reason="OK",
            now=clock.now,
        )
    with pytest.raises(OwnershipRefused, match="INVALID"):
        gate.authorize(permit)
    assert not future.done(), "an invalid or stale permit must leave the action pending"


def test_signed_denial_is_terminal_and_cannot_turn_into_a_hardware_failure(
    pending_gate,
):
    gate, clock, future, current = pending_gate
    gate.authorize(
        sign_permit(
            current, KEY, allowed=False, reason="STOP_OWNERSHIP_DRIFT", now=clock.now
        )
    )
    with pytest.raises(OwnershipRefused, match="STOP_OWNERSHIP_DRIFT") as caught:
        future.result(2)
    assert caught.value.action_details["manual_confirmation_required"] is True
    assert caught.value.action_details["safety_rejection"] is True
    assert caught.value.action_details["agent_queue_ownership_checked"] is True


def test_parent_loss_revokes_the_pending_callback_and_future_callbacks(pending_gate):
    gate, _clock, future, current = pending_gate
    gate.cancel_all()
    with pytest.raises(OwnershipRefused):
        future.result(2)
    assert gate.challenge(current.command_id) is None
    with pytest.raises(OwnershipRefused, match="CALLER_LOST"):
        gate.require(command(command_id="next"))


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"agent_generation": None}, "PROTOCOL_REQUIRED"),
        ({"ownership_guard": None}, "PROTOCOL_REQUIRED"),
        ({"expires_at": NOW}, "COMMAND_EXPIRED"),
        ({"expires_at": NOW.replace(tzinfo=None)}, "COMMAND_EXPIRED"),
    ],
)
def test_legacy_or_expired_commands_cannot_park_as_valid_work(changes, reason):
    gate = NodeOwnershipGate(secret=KEY, boot_id="boot", now=lambda: NOW)
    with pytest.raises(OwnershipRefused, match=reason):
        gate.require(command(**changes))
    assert gate.challenge("owned-command") is None


def test_duplicate_boundary_cannot_create_a_second_pending_action(pending_gate):
    gate, _clock, future, _current = pending_gate
    with pytest.raises(OwnershipRefused, match="ALREADY_PENDING"):
        gate.require(command())
    assert not future.done(), (
        "a duplicate boundary must not complete the original pending action"
    )


@pytest.mark.parametrize("boot,key", [("", KEY), ("boot", "short")])
def test_gate_requires_node_identity_and_key(boot, key):
    with pytest.raises(ValueError, match="identity"):
        NodeOwnershipGate(secret=key, boot_id=boot)


@pytest.mark.parametrize("value", [0, True, None, "bad", "2026-09-13T00:00:00"])
def test_wire_does_not_coerce_unknown_or_naive_timestamps(value):
    data = challenge().model_dump()
    with pytest.raises(ValidationError):
        OwnershipChallenge.model_validate(data | {"expires_at": value})


def test_iso_wire_roundtrip_preserves_signature_and_rejects_contradictory_decision():
    permit = sign_permit(challenge(), KEY, allowed=True, reason="OK", now=NOW)
    decoded = OwnershipPermit.model_validate_json(permit.model_dump_json())
    assert decoded == permit
    assert permit_signature(decoded, KEY) == permit.signature
    with pytest.raises(ValidationError, match="contradicts"):
        sign_permit(challenge(), KEY, allowed=False, reason="OK", now=NOW)


def test_semantic_binding_excludes_only_transport_timestamps():
    baseline = command_identity(command())
    assert (
        command_identity(
            command(
                issued_at=NOW + timedelta(seconds=1),
                expires_at=NOW + timedelta(seconds=121),
            )
        )
        == baseline
    )
    for changes in (
        {"fencing_token": 4},
        {"node_id": "other"},
        {"parameters": {"drift": True}},
    ):
        assert command_identity(command(**changes)) != baseline


def test_recheck_context_does_not_escape_its_owned_call():
    assert current_ownership_challenge() is None
    value = challenge()
    with ownership_recheck_scope(value):
        assert current_ownership_challenge() == value
    assert current_ownership_challenge() is None
    calls = []
    with physical_ownership_scope(lambda: calls.append("checked")):
        require_physical_ownership()
    require_physical_ownership()
    assert calls == ["checked"]


def test_unmarked_mutating_handler_is_refused_before_running():
    effects = []
    agent = SimpleNamespace(ownership_gate=object())
    signed = SignedNodeAction(command=command(), signature="unused")
    with pytest.raises(OwnershipRefused, match="UNSUPPORTED"):
        execute_with_final_ownership(agent, signed, lambda value: effects.append(value))
    assert effects == []
    assert not supports_final_ownership(lambda: None), (
        "unmarked handlers must not advertise final-ownership support"
    )
    assert ownership_required(WorkflowOperation.RESET_GPU), (
        "GPU reset must require final ownership authorization"
    )
    assert not ownership_required(WorkflowOperation.RESTORE_GPU_SERVICES), (
        "service restoration must remain available as compensation"
    )
    assert not ownership_required(WorkflowOperation.VERIFY_NO_GPU_CLIENTS), (
        "read-only GPU-client verification must not require an ownership permit"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "fence",
        "window",
        "signature",
        "expired",
        "new-compute-client",
        "new-device-client",
        "unresolvable",
        "io",
    ],
)
def test_last_local_fences_are_checked_after_the_permit_before_effect(defect):
    permit = sign_permit(challenge(), KEY, allowed=True, reason="OK", now=NOW)
    effects = []

    @final_ownership_boundary
    def handler(value):
        require_physical_ownership()
        effects.append(value.command_id)
        return {"reset": True}

    def validate(envelope):
        if defect == "signature":
            raise ValueError("generation changed")
        return envelope.command

    def quiesced(**kwargs):
        if defect == "window":
            raise RuntimeError("quiesce no longer exists")

    checked = []

    def compute(*args, **kwargs):
        checked.append("compute")
        assert kwargs == {"include_device_clients": False}
        if defect == "new-compute-client":
            raise RuntimeError("new CUDA client")
        if defect == "io":
            raise OSError("unreadable final client state")

    def devices(*args):
        checked.append("devices")
        if defect == "unresolvable":
            raise RuntimeError("GPU identity changed")

    agent = SimpleNamespace(
        ownership_gate=SimpleNamespace(require=lambda value: permit),
        validate_submission=validate,
        ledger=SimpleNamespace(accept_fencing=lambda *args: defect != "fence"),
        service_quiesce_enabled=True,
        quiesce_manager=SimpleNamespace(assert_quiesced=quiesced),
        _verify_no_clients=compute,
        _device_path_cache_window=nullcontext,
        _require_resolvable_targets=devices,
        device_client_finder=lambda targets: [{"pid": "new"}]
        if defect == "new-device-client"
        else [],
        now=lambda: permit.expires_at if defect == "expired" else NOW,
    )
    signed = SignedNodeAction(command=command(), signature="unused")
    if defect == "none":
        result = execute_with_final_ownership(agent, signed, handler)
        assert effects == ["owned-command"]
        assert checked == ["compute", "devices"]
        assert (
            result["physical_ownership_checks"][0]["challenge_nonce"]
            == challenge().nonce
        )
    else:
        with pytest.raises(OwnershipRefused) as refused:
            execute_with_final_ownership(agent, signed, handler)
        assert effects == []
        # A refusal names the fence that moved; only a foreign failure of the
        # final local reads is reported as the opaque safety code, and then
        # with its cause class so an operator can tell a bug from a client.
        expected = {
            "fence": "OWNERSHIP_FINAL_FENCE_CHANGED",
            "expired": "OWNERSHIP_PERMIT_EXPIRED",
            "new-device-client": "OWNERSHIP_FINAL_CLIENTS_CHANGED",
        }.get(defect, "OWNERSHIP_FINAL_SAFETY_CHANGED")
        assert refused.value.action_details["reason"] == expected
        if expected == "OWNERSHIP_FINAL_SAFETY_CHANGED":
            assert refused.value.action_details["refusal_cause"] in {
                "RuntimeError",
                "ValueError",
                "OSError",
            }


@pytest.mark.parametrize("seen_in", ["first", "second"])
def test_a_transient_device_holder_does_not_abort_the_final_recheck(seen_in):
    """A monitoring ``nvidia-smi`` query holds every /dev/nvidia* node for a
    few hundred milliseconds. The sampled preflight already classifies a holder
    seen in a single sample as transient; the final recheck must apply the same
    rule (live 2026-09-17: a 0.25 s acceptance sampler aborted the GPU reset as
    OWNERSHIP_FINAL_SAFETY_CHANGED and the node was quarantined).
    """
    permit = sign_permit(challenge(), KEY, allowed=True, reason="OK", now=NOW)
    effects = []
    calls = []

    @final_ownership_boundary
    def handler(value):
        require_physical_ownership()
        effects.append(value.command_id)
        return {"reset": True}

    def finder(targets):
        calls.append(sorted(targets))
        hit = {"gpu_uuid": "GPU-a", "pid": "4242", "device": "/dev/nvidia0"}
        return [hit] if (len(calls) == 1) == (seen_in == "first") else []

    slept = []
    agent = SimpleNamespace(
        ownership_gate=SimpleNamespace(require=lambda value: permit),
        validate_submission=lambda envelope: envelope.command,
        ledger=SimpleNamespace(accept_fencing=lambda *args: True),
        service_quiesce_enabled=False,
        quiesce_manager=None,
        _verify_no_clients=lambda *args, **kwargs: None,
        _device_path_cache_window=nullcontext,
        _require_resolvable_targets=lambda targets: None,
        device_client_finder=finder,
        sleep=slept.append,
        now=lambda: NOW,
    )
    signed = SignedNodeAction(command=command(), signature="unused")
    result = execute_with_final_ownership(agent, signed, handler)
    assert effects == ["owned-command"]
    # An empty sample already proves there is no persistent holder, so a
    # holder seen only in the first sample costs one more sample; a clean
    # first sample ends the recheck without sleeping.
    assert calls == ([["GPU-a"], ["GPU-a"]] if seen_in == "first" else [["GPU-a"]])
    assert slept == ([pytest.approx(0.5)] if seen_in == "first" else [])
    assert result["physical_ownership_checks"]


def test_a_persistent_device_holder_is_refused_by_its_own_code():
    permit = sign_permit(challenge(), KEY, allowed=True, reason="OK", now=NOW)
    effects = []
    calls = []

    @final_ownership_boundary
    def handler(value):
        require_physical_ownership()
        effects.append(value.command_id)
        return {"reset": True}

    def finder(targets):
        calls.append(1)
        return [{"gpu_uuid": "GPU-a", "pid": "4242", "device": "/dev/nvidia0"}]

    agent = SimpleNamespace(
        ownership_gate=SimpleNamespace(require=lambda value: permit),
        validate_submission=lambda envelope: envelope.command,
        ledger=SimpleNamespace(accept_fencing=lambda *args: True),
        service_quiesce_enabled=False,
        quiesce_manager=None,
        _verify_no_clients=lambda *args, **kwargs: None,
        _device_path_cache_window=nullcontext,
        _require_resolvable_targets=lambda targets: None,
        device_client_finder=finder,
        sleep=lambda seconds: None,
        now=lambda: NOW,
    )
    signed = SignedNodeAction(command=command(), signature="unused")
    with pytest.raises(OwnershipRefused) as refused:
        execute_with_final_ownership(agent, signed, handler)
    assert effects == []
    assert len(calls) == 2
    assert refused.value.action_details["reason"] == "OWNERSHIP_FINAL_CLIENTS_CHANGED"
    assert "refusal_cause" not in refused.value.action_details
