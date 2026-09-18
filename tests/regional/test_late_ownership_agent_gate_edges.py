"""Hermetic permit races and the native Agent's final ownership checkpoint."""

from __future__ import annotations

import socket
import subprocess
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, Literal, TypeVar, cast

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.late_ownership import (
    NodeOwnershipGate,
    OwnershipChallenge,
    OwnershipPermit,
    OwnershipRefused,
    command_identity,
    current_ownership_challenge,
    execute_with_final_ownership,
    final_ownership_boundary,
    ownership_recheck_scope,
    permit_signature,
    physical_ownership_scope,
    require_physical_ownership,
    sign_permit,
    supports_final_ownership,
)
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionResult,
    NodeActionStatus,
    SignedNodeAction,
    sign_node_action,
)
from tests.node_agent._support import SECRET, FakeRunner, node_action_executor
from tests.regional.test_late_ownership_agent_gate import KEY, NOW, Clock
from tests.regional.test_late_ownership_agent_gate import challenge as challenge_fixture
from tests.regional.test_late_ownership_agent_gate import command as command_fixture
from tests.regional.test_late_ownership_dispatch import (
    assert_refusal_does_not_escalate as escalation_fixture,
)

command: Callable[..., NodeActionCommand] = command_fixture
challenge: Callable[..., OwnershipChallenge] = challenge_fixture
make_clock: Callable[[], Clock] = Clock
assert_refusal_does_not_escalate: Callable[[WorkflowStepOutcome], None] = (
    escalation_fixture
)
Callback = TypeVar("Callback", bound=Callable[..., Any])


def boundary_handler(handler: Callback) -> Callback:
    return cast(Callback, final_ownership_boundary(handler))


@pytest.fixture(autouse=True)
def owned_agent_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """May be loaded as a plugin to isolate the existing Agent fixtures too."""
    boot = tmp_path / "owned-boot-id"
    boot.write_text("owned-local-boot\n", encoding="ascii")
    read_text = Path.read_text

    def owned_read(path: Path, *args: Any, **kwargs: Any) -> str:
        selected = boot if path == Path("/proc/sys/kernel/random/boot_id") else path
        return read_text(selected, *args, **kwargs)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("late-ownership tests must use owned fake I/O")

    monkeypatch.setattr(Path, "read_text", owned_read)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


class ParkedClock(Clock):
    def __init__(self) -> None:
        initialize: Callable[[], None] = super().__init__
        initialize()
        self.parked = Event()
        self.release = Event()
        self.calls = 0

    def monotonic(self) -> float:
        self.calls += 1
        if self.calls == 2:
            self.parked.set()
            if not self.release.wait(3):
                raise AssertionError("owned gate consumer was not released")
        return float(self.elapsed)


@contextmanager
def parked_gate(
    value: NodeActionCommand | None = None,
    *,
    boundary: Literal["AGENT_PRE_SPAWN", "AGENT_HANDLER_ENTRY"] = "AGENT_PRE_SPAWN",
) -> Iterator[tuple[NodeOwnershipGate, ParkedClock, Any, OwnershipChallenge]]:
    clock = ParkedClock()
    gate = NodeOwnershipGate(
        secret=KEY,
        boot_id="owned-boot",
        now=lambda: clock.now,
        monotonic=clock.monotonic,
    )
    value = value or command()
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(gate.require, value, boundary=boundary)
    try:
        assert clock.parked.wait(3), "the gate did not publish its owned challenge"
        current = gate.challenge(value.command_id)
        assert current is not None
        yield gate, clock, future, current
    finally:
        gate.cancel_all()
        clock.release.set()
        try:
            future.result(timeout=3)
        except OwnershipRefused:
            pass
        finally:
            pool.shutdown(wait=True)
        assert future.done(), "owned gate worker must finish before fixture teardown"


@pytest.mark.parametrize("allowed", [True, False])
def test_duplicate_pending_decision_is_idempotent_without_extending_its_ttl(
    allowed: bool,
) -> None:
    with parked_gate() as (gate, clock, future, current):
        reason = "OK" if allowed else "STOP_OWNERSHIP_DRIFT"
        first = sign_permit(current, KEY, allowed=allowed, reason=reason, now=clock.now)
        gate.authorize(first)
        assert gate.challenge(current.command_id) is None
        clock.now += timedelta(seconds=1)
        duplicate = sign_permit(
            current, KEY, allowed=allowed, reason=reason, now=clock.now
        )
        assert duplicate.expires_at > first.expires_at
        gate.authorize(duplicate)
        clock.now = first.expires_at
        clock.release.set()
        with pytest.raises(OwnershipRefused, match="PERMIT_EXPIRED"):
            future.result(timeout=3)
        with pytest.raises(OwnershipRefused, match="STALE"):
            gate.authorize(duplicate)


@pytest.mark.parametrize("replacement", ["allow", "different-denial", "same-denial"])
def test_a_latched_denial_cannot_be_replaced_by_a_newer_decision(
    replacement: str,
) -> None:
    with parked_gate() as (gate, clock, future, current):
        denied = sign_permit(
            current, KEY, allowed=False, reason="STOP_OWNERSHIP_DRIFT", now=clock.now
        )
        gate.authorize(denied)
        incoming = sign_permit(
            current,
            KEY,
            allowed=replacement == "allow",
            reason=(
                "OK"
                if replacement == "allow"
                else "STOP_PARTICIPANTS_CHANGED"
                if replacement == "different-denial"
                else "STOP_OWNERSHIP_DRIFT"
            ),
            now=clock.now,
        )
        if replacement == "same-denial":
            gate.authorize(incoming)
        else:
            with pytest.raises(OwnershipRefused, match="STALE"):
                gate.authorize(incoming)
        clock.release.set()
        with pytest.raises(OwnershipRefused, match="STOP_OWNERSHIP_DRIFT") as caught:
            future.result(timeout=3)
        assert caught.value.action_details["safety_rejection"] is True


def test_denial_supersedes_an_allowance_before_the_owned_callback_consumes_it() -> None:
    with parked_gate() as (gate, clock, future, current):
        gate.authorize(
            sign_permit(current, KEY, allowed=True, reason="OK", now=clock.now)
        )
        gate.authorize(
            sign_permit(
                current,
                KEY,
                allowed=False,
                reason="STOP_PARTICIPANTS_CHANGED",
                now=clock.now,
            )
        )
        clock.release.set()
        with pytest.raises(OwnershipRefused, match="STOP_PARTICIPANTS_CHANGED"):
            future.result(timeout=3)
        assert gate.challenge(current.command_id) is None


@pytest.mark.parametrize("field", ["checked_at", "expires_at", "challenge.expires_at"])
def test_native_permit_objects_with_naive_timestamps_are_refused(field: str) -> None:
    with parked_gate() as (gate, clock, future, current):
        permit = sign_permit(current, KEY, allowed=True, reason="OK", now=clock.now)
        if field == "challenge.expires_at":
            permit = permit.model_copy(
                update={
                    "challenge": current.model_copy(
                        update={"expires_at": current.expires_at.replace(tzinfo=None)}
                    )
                }
            )
        else:
            stamp = getattr(permit, field).replace(tzinfo=None)
            permit = permit.model_copy(update={field: stamp})
        permit = permit.model_copy(update={"signature": permit_signature(permit, KEY)})
        with pytest.raises(OwnershipRefused, match="INVALID"):
            gate.authorize(permit)
        assert not future.done(), (
            "naive permit timestamps must not release the pending callback"
        )
        assert gate.challenge(current.command_id) == current


@pytest.mark.parametrize("field", ["node_uid", "source_sha256", "generation"])
def test_permits_for_another_semantic_source_cannot_release_the_gate(
    field: str,
) -> None:
    original = command(
        parameters={
            "node_uid": "owned-node-uid",
            "source_sha256": "a" * 64,
            "generation": 7,
        }
    )
    with parked_gate(original) as (gate, clock, future, current):
        parameters = {**original.parameters, field: "foreign-source"}
        foreign = original.model_copy(update={"parameters": parameters})
        permit = sign_permit(
            current.model_copy(update={"command_sha256": command_identity(foreign)}),
            KEY,
            allowed=True,
            reason="OK",
            now=clock.now,
        )
        with pytest.raises(OwnershipRefused, match="STALE"):
            gate.authorize(permit)
        assert not future.done(), (
            "foreign source identity must not release the pending callback"
        )
        assert gate.challenge(current.command_id) == current


def test_a_sibling_node_signing_key_never_releases_the_callback() -> None:
    with parked_gate() as (gate, clock, future, current):
        permit = sign_permit(
            current,
            "other-owned-test-key-" + "y" * 32,
            allowed=True,
            reason="OK",
            now=clock.now,
        )
        with pytest.raises(OwnershipRefused, match="INVALID"):
            gate.authorize(permit)
        assert not future.done(), (
            "sibling node key must not release the pending callback"
        )


def test_monotonic_deadline_can_expire_even_when_wall_time_does_not_advance() -> None:
    elapsed = iter([0.0, 91.0])
    gate = NodeOwnershipGate(
        secret=KEY, boot_id="owned", now=lambda: NOW, monotonic=lambda: next(elapsed)
    )
    with pytest.raises(OwnershipRefused, match="RECHECK_TIMEOUT"):
        gate.require(command())
    assert gate.challenge("owned-command") is None


def test_a_pending_permit_cannot_outlive_the_challenge_monotonic_deadline() -> None:
    with parked_gate() as (gate, clock, future, current):
        gate.authorize(
            sign_permit(current, KEY, allowed=True, reason="OK", now=clock.now)
        )
        clock.elapsed = 91.0
        clock.release.set()
        with pytest.raises(OwnershipRefused, match="PERMIT_EXPIRED"):
            future.result(timeout=3)


def test_latched_permit_uses_its_original_monotonic_deadline() -> None:
    with parked_gate() as (gate, clock, future, current):
        gate.authorize(
            sign_permit(current, KEY, allowed=True, reason="OK", now=clock.now)
        )
        clock.elapsed = 3.0
        clock.release.set()
        with pytest.raises(OwnershipRefused, match="PERMIT_EXPIRED"):
            future.result(timeout=3)
        assert clock.now == NOW


def test_delivery_check_is_one_use_and_does_not_change_the_signed_wire() -> None:
    with parked_gate() as (gate, clock, future, current):
        permit = sign_permit(current, KEY, allowed=True, reason="OK", now=clock.now)
        wire = permit.model_dump_json()
        signature = permit_signature(permit, KEY)
        gate.authorize(permit)
        clock.release.set()
        delivered = future.result(timeout=3)
        assert delivered.model_dump_json() == wire
        assert permit_signature(delivered, KEY) == signature
        delivered.validate_delivery()
        with pytest.raises(OwnershipRefused, match="PERMIT_USED"):
            delivered.validate_delivery()
        with pytest.raises(OwnershipRefused, match="STALE"):
            gate.authorize(permit)


@pytest.mark.parametrize("change", ["cancel", "monotonic-expiry", "clock-backwards"])
def test_delivered_permit_never_recovers_cancelled_or_expired_authority(
    change: str,
) -> None:
    with parked_gate() as (gate, clock, future, current):
        gate.authorize(
            sign_permit(current, KEY, allowed=True, reason="OK", now=clock.now)
        )
        clock.release.set()
        permit = future.result(timeout=3)
        if change == "cancel":
            gate.cancel_all()
        elif change == "monotonic-expiry":
            clock.elapsed = 3.0
        else:
            clock.now = NOW - timedelta(seconds=1)
        with pytest.raises(OwnershipRefused):
            permit.validate_delivery()


def test_cancellation_revokes_an_already_latched_but_unconsumed_permit() -> None:
    with parked_gate() as (gate, clock, future, current):
        permit = sign_permit(current, KEY, allowed=True, reason="OK", now=clock.now)
        gate.authorize(permit)
        gate.cancel_all()
        clock.release.set()
        with pytest.raises(OwnershipRefused, match="PERMIT_EXPIRED"):
            future.result(timeout=3)
        with pytest.raises(OwnershipRefused, match="STALE"):
            gate.authorize(permit)


def test_nested_contexts_restore_the_previous_callback_after_refusal() -> None:
    calls: list[str] = []
    outer, inner = challenge(), challenge(nonce="b" * 64)

    def denied() -> None:
        calls.append("inner")
        raise OwnershipRefused("OWNED_TEST_DENIAL")

    with (
        ownership_recheck_scope(outer),
        physical_ownership_scope(lambda: calls.append("outer")),
    ):
        with pytest.raises(OwnershipRefused):
            with ownership_recheck_scope(inner), physical_ownership_scope(denied):
                assert current_ownership_challenge() == inner
                require_physical_ownership()
        assert current_ownership_challenge() == outer
        require_physical_ownership()
    require_physical_ownership()
    assert current_ownership_challenge() is None
    assert calls == ["inner", "outer"]


class FakeNodeAgent:
    def __init__(self, *, guarded: bool = True) -> None:
        self.calls: list[str] = []
        self.ownership_gate = SimpleNamespace(require=self.grant) if guarded else None
        self.ledger = SimpleNamespace(accept_fencing=lambda *args: True)
        self.service_quiesce_enabled = False
        self.quiesce_manager: Any = None
        self.now = lambda: NOW

    def grant(self, value: NodeActionCommand) -> OwnershipPermit:
        self.calls.append("permit")
        return sign_permit(
            challenge(command_sha256=command_identity(value)),
            KEY,
            allowed=True,
            reason="OK",
            now=NOW,
        )

    def validate_submission(self, envelope: SignedNodeAction) -> None:
        self.calls.append("validate")

    def _verify_no_clients(self, *args: Any, **kwargs: Any) -> None:
        assert kwargs == {"include_device_clients": False}
        self.calls.append("compute")

    @contextmanager
    def _device_path_cache_window(self) -> Iterator[None]:
        self.calls.append("cache")
        yield

    def _require_resolvable_targets(self, targets: list[str]) -> None:
        self.calls.append("targets")

    def device_client_finder(self, targets: set[str]) -> list[Any]:
        self.calls.append("devices")
        return []

    @boundary_handler
    def effect(self, value: NodeActionCommand) -> dict[str, Any]:
        require_physical_ownership()
        self.calls.append("effect")
        return {"effect": True}


@pytest.mark.parametrize(
    ("operation", "checks"),
    [
        (WorkflowOperation.RESET_GPU, ["compute", "cache", "targets", "devices"]),
        (WorkflowOperation.RESTART_FABRIC_MANAGER, ["compute"]),
        (WorkflowOperation.REMEDIATE_EFA_DRIVER, []),
    ],
)
@pytest.mark.parametrize("quiesce", ["disabled", "enabled"])
def test_final_check_routes_only_the_required_local_guards(
    operation: WorkflowOperation, checks: list[str], quiesce: str
) -> None:
    agent = FakeNodeAgent()
    agent.service_quiesce_enabled = quiesce != "disabled"

    def stopped(**kwargs: Any) -> None:
        assert kwargs == {"incident_id": "owned-incident", "for_reset": False}
        agent.calls.append("quiesce")

    if quiesce == "enabled":
        agent.quiesce_manager = SimpleNamespace(assert_quiesced=stopped)
    result = execute_with_final_ownership(
        agent,
        SignedNodeAction(command=command(operation=operation), signature="local"),
        agent.effect,
    )
    quiesce_calls = (
        ["quiesce"]
        if quiesce == "enabled" and operation is WorkflowOperation.RESET_GPU
        else []
    )
    assert agent.calls == [
        "permit",
        "validate",
        *quiesce_calls,
        *checks,
        "validate",
        *quiesce_calls,
        "effect",
    ]
    assert result["physical_ownership_checks"][0]["sequence"] == 1
    assert supports_final_ownership(agent.effect), (
        "decorated Agent handler must expose its final ownership boundary"
    )


@pytest.mark.parametrize(
    ("guarded", "operation"),
    [
        (False, WorkflowOperation.RESET_GPU),
        (True, WorkflowOperation.RESTORE_GPU_SERVICES),
        (True, WorkflowOperation.VERIFY_NO_GPU_CLIENTS),
    ],
)
def test_legacy_and_nonmutating_callbacks_do_not_inherit_an_outer_gate(
    guarded: bool, operation: WorkflowOperation
) -> None:
    agent = FakeNodeAgent(guarded=guarded)
    outer: list[str] = []
    with physical_ownership_scope(lambda: outer.append("outer")):
        result = execute_with_final_ownership(
            agent,
            SignedNodeAction(command=command(operation=operation), signature="local"),
            agent.effect,
        )
        require_physical_ownership()
    assert agent.calls == ["effect"]
    assert outer == ["outer"]
    assert "physical_ownership_checks" not in result
    assert ("node_action_command_id" in result) is guarded


def test_callback_failure_restores_the_previous_physical_scope() -> None:
    agent = FakeNodeAgent()
    outer: list[str] = []

    @boundary_handler
    def failed(value: NodeActionCommand) -> dict[str, Any]:
        require_physical_ownership()
        raise RuntimeError("owned test callback failed")

    with physical_ownership_scope(lambda: outer.append("outer")):
        with pytest.raises(RuntimeError, match="owned test callback"):
            execute_with_final_ownership(
                agent, SignedNodeAction(command=command(), signature="local"), failed
            )
        require_physical_ownership()
    assert outer == ["outer"]
    assert "effect" not in agent.calls


class NativeHarness:
    """Real executor/ledger and fake reads at the final physical boundary."""

    def __init__(self, tmp_path: Path) -> None:
        self.clock = make_clock()
        self.granted = False
        self.reads: list[str] = []
        self.on_permit: Callable[[], None] = lambda: None
        self.on_compute: Callable[[], None] = lambda: None
        self.on_device: Callable[[], None] = lambda: None
        box = self

        class Runner(FakeRunner):
            def __call__(self, command: Any, **kwargs: Any) -> Any:
                if (
                    box.granted
                    and "--query-compute-apps=gpu_uuid,pid,process_name" in command
                ):
                    box.reads.append("compute")
                    box.on_compute()
                run: Callable[..., Any] = super().__call__
                return run(command, **kwargs)

        def devices(targets: set[str]) -> list[dict[str, str]]:
            if box.granted:
                box.reads.append("device")
                box.on_device()
            return []

        self.hardware = Runner()
        self.agent = node_action_executor(
            tmp_path,
            "owned-native-actions.db",
            allowed_operations={WorkflowOperation.RESET_GPU},
            reset_enabled=True,
            runner=self.hardware,
            agent_generation=7,
            now=lambda: self.clock.now,
            sleep=lambda _: None,
            device_client_finder=devices,
            gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
            device_client_samples=1,
        )
        self.gate = NodeOwnershipGate(
            secret=SECRET,
            boot_id="owned-native-boot",
            now=lambda: self.clock.now,
            monotonic=self.clock.monotonic,
        )
        self.agent.ownership_gate = self.gate
        value = command()
        self.signed = SignedNodeAction(
            command=value, signature=sign_node_action(value, SECRET)
        )
        self.permit = sign_permit(
            challenge(
                boot_id=self.gate.boot_id, command_sha256=command_identity(value)
            ),
            SECRET,
            allowed=True,
            reason="OK",
            now=self.clock.now,
        )

    def execute(self) -> NodeActionResult:
        self.clock.ready.clear()
        with ThreadPoolExecutor(max_workers=1) as pool:
            execute: Callable[[SignedNodeAction], NodeActionResult] = self.agent.execute
            pending = pool.submit(execute, self.signed)
            try:
                assert self.clock.ready.wait(3), (
                    "the native Agent did not publish its physical-boundary challenge"
                )
                current = self.gate.challenge(self.signed.command.command_id)
                assert current is not None
                self.permit = sign_permit(
                    current, SECRET, allowed=True, reason="OK", now=self.clock.now
                )
                self.granted = True
                self.on_permit()
                self.gate.authorize(self.permit)
                return pending.result(timeout=3)
            finally:
                self.gate.cancel_all()
                pending.result(timeout=3)

    def close(self) -> None:
        self.gate.cancel_all()
        self.agent.ledger.close()


@pytest.fixture
def native(tmp_path: Path) -> Iterator[NativeHarness]:
    harness = NativeHarness(tmp_path)
    try:
        yield harness
    finally:
        harness.close()


def assert_native_refusal(harness: NativeHarness, result: Any) -> None:
    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is False
    assert result.details["safety_rejection"] is True
    assert result.details["manual_confirmation_required"] is True
    assert result.details["node_action_not_started"] is True
    assert not any("--gpu-reset" in args for args in harness.hardware.commands), (
        "ownership refusal must prevent GPU reset dispatch"
    )
    assert_refusal_does_not_escalate(
        WorkflowStepOutcome.failed("owned final-check refusal", details=result.details)
    )


@pytest.mark.parametrize("read", ["compute", "device"])
@pytest.mark.parametrize("elapsed", [3, 4])
def test_permit_expiry_after_client_read_never_spawns_or_escalates(
    native: NativeHarness, read: str, elapsed: int
) -> None:
    def expire() -> None:
        native.clock.now = NOW + timedelta(seconds=elapsed)
        native.clock.elapsed = float(elapsed)

    if read == "compute":
        native.on_compute = expire
    else:
        native.on_device = expire
    result = native.execute()
    assert result.details["reason"] == "OWNERSHIP_PERMIT_EXPIRED"
    assert read in native.reads
    assert_native_refusal(native, result)


@pytest.mark.parametrize("drift", ["generation", "node", "key-source", "fence"])
def test_native_identity_recheck_rejects_drift_after_the_signed_permit(
    native: NativeHarness, drift: str
) -> None:
    def change() -> None:
        if drift == "generation":
            native.agent.set_agent_generation(8)
        elif drift == "node":
            native.agent.node_ids = {"node-b"}
        elif drift == "key-source":
            native.agent.secret = "different-owned-node-key-" + "z" * 32
        else:
            assert native.agent.ledger.accept_fencing("owned-incident", 4), (
                "test must advance the incident fence before final validation"
            )

    native.on_permit = change
    result = native.execute()
    # A moved fence names itself; identity drift surfaces through the signed
    # submission check as the opaque safety code with its cause class.
    expected = (
        "OWNERSHIP_FINAL_FENCE_CHANGED"
        if drift == "fence"
        else "OWNERSHIP_FINAL_SAFETY_CHANGED"
    )
    assert result.details["reason"] == expected
    assert_native_refusal(native, result)


@pytest.mark.parametrize(
    "drift", ["generation", "node", "key-source", "fence", "cancel", "clock-backwards"]
)
def test_drift_during_the_last_client_read_cannot_be_used_as_permission(
    native: NativeHarness, drift: str
) -> None:
    def change() -> None:
        if drift == "generation":
            native.agent.set_agent_generation(8)
        elif drift == "node":
            native.agent.node_ids = {"node-b"}
        elif drift == "key-source":
            native.agent.secret = "different-owned-node-key-" + "z" * 32
        elif drift == "fence":
            assert native.agent.ledger.accept_fencing("owned-incident", 4), (
                "test must advance the incident fence during the final device read"
            )
        elif drift == "cancel":
            native.gate.cancel_all()
        else:
            native.clock.now = NOW - timedelta(seconds=1)

    native.on_device = change
    result = native.execute()
    assert_native_refusal(native, result)


def test_late_client_read_cannot_refresh_the_original_monotonic_permit_budget(
    native: NativeHarness,
) -> None:
    def spend_budget() -> None:
        native.clock.elapsed = 3.0

    native.on_device = spend_budget
    result = native.execute()
    assert native.clock.now == NOW
    assert result.details["reason"] == "OWNERSHIP_PERMIT_EXPIRED"
    assert_native_refusal(native, result)


def test_native_approved_action_records_its_boundary_and_replay_never_resets_twice(
    native: NativeHarness,
) -> None:
    first = native.execute()
    assert first.status is NodeActionStatus.SUCCEEDED
    receipts = first.details["physical_ownership_checks"]
    assert len(receipts) == 1
    assert receipts[0]["challenge_nonce"] == native.permit.challenge.nonce
    assert first.details["node_action_command_id"] == native.signed.command.command_id
    before = list(native.hardware.commands)
    assert native.agent.execute(native.signed) == first
    assert native.hardware.commands == before
    assert sum("--gpu-reset" in args for args in before) == 1
