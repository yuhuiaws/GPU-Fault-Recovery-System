"""Bounded-activation guards of the custody engine against a recording I/O.

The engine drives one authorized node-key activation through a journal. These
tests pin the refusals that keep the journal honest: no step or save without a
prepared state, no step out of order, no key writer before the wave is owned or
after the keys are already provisioned, and no progress once the wall clock
regresses or passes the fixed deadline, including during preparation itself.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import node_key_custody_activation as engine
from gpu_fault.admin.node_key_custody_models import (
    Authorization,
    Completed,
    CustodyError,
    Signed,
)
from tests.admin._security_activation_io_world import ActivationWorld


class RecordingIO:
    """Answers every activation step with inert evidence and records the order."""

    def __init__(self, clock: Clock | None = None) -> None:
        self.calls: list[str] = []
        self.clock = clock
        self.capture_advance = timedelta(0)

    def bind_completed(self, completed: Completed) -> None:
        self.calls.append("bind_completed")

    def capture(self) -> dict[str, Any]:
        self.calls.append("capture")
        if self.clock is not None:
            self.clock.advance(self.capture_advance)
        return {"agents": {"node-a": {"generation": 1}}}

    def verify(self, snapshot: dict[str, Any]) -> None:
        self.calls.append("verify")

    def require_owned_wave(self, snapshot: dict[str, Any]) -> None:
        self.calls.append("require_owned_wave")

    def __getattr__(self, name: str) -> Any:
        if name in engine.ACTIVATION_STEPS or name in {
            "guard",
            "fence",
            "keys_provisioned",
            "refresh_executor",
            "install",
            "refresh_cpu",
            "observe",
            "unfence",
            "current",
        }:

            def step(snapshot: dict[str, Any]) -> dict[str, Any]:
                self.calls.append(name)
                return {"step": name}

            return step
        raise AttributeError(name)


class Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now = self.now + delta


def _signed(authorization: Authorization) -> Signed[Authorization]:
    return Signed[Authorization](
        statement=authorization, signer_sha256="5" * 64, signature="a" * 64
    )


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ActivationWorld:
    return ActivationWorld(tmp_path, monkeypatch)


def _handle(
    world: ActivationWorld, *, expires_in: timedelta = timedelta(hours=2)
) -> tuple[engine.CustodyActivation, RecordingIO, Clock]:
    start = datetime.now(timezone.utc)
    clock = Clock(start)
    io = RecordingIO(clock)
    authorization = world.authorization.model_copy(
        update={
            "not_before": start - timedelta(hours=1),
            "expires_at": start + expires_in,
        }
    )
    handle = engine.CustodyActivation(
        io, world.context, _signed(authorization), now=clock
    )
    return handle, io, clock


def test_saving_and_stepping_require_a_prepared_state(world: ActivationWorld) -> None:
    handle, io, _clock = _handle(world)
    with pytest.raises(CustodyError, match="no durable intent"):
        handle.save()
    with pytest.raises(CustodyError, match="no prepared state"):
        handle.step("GUARDED", io.guard)
    assert not handle.path.exists(), "nothing may be journaled without preparation"
    assert io.calls == []


def test_steps_must_follow_the_journal_order(world: ActivationWorld) -> None:
    handle, io, _clock = _handle(world)
    handle.prepare()
    assert set(handle.load().completed) == {"GUARDED", "FENCED"}
    with pytest.raises(CustodyError, match="out of order"):
        handle.step("OBSERVED", io.observe)
    assert "observe" not in io.calls
    assert set(handle.load().completed) == {"GUARDED", "FENCED"}, (
        "a refused step must not alter the journal"
    )


def test_the_key_writer_requires_a_durable_owned_wave(world: ActivationWorld) -> None:
    handle, _io, _clock = _handle(world)
    with pytest.raises(CustodyError, match="requires a durable owned wave"):
        with handle.key_writer():
            pass


def test_the_key_writer_refuses_to_run_twice(world: ActivationWorld) -> None:
    handle, io, _clock = _handle(world)
    handle.prepare()
    handle.step("KEYS_PROVISIONED", io.keys_provisioned)
    with pytest.raises(CustodyError, match="already completed"):
        with handle.key_writer():
            pass


def test_a_resumed_key_writer_keeps_its_recorded_intent(world: ActivationWorld) -> None:
    handle, io, _clock = _handle(world)
    handle.prepare()
    with pytest.raises(RuntimeError, match="writer interrupted"):
        with handle.key_writer():
            raise RuntimeError("writer interrupted")
    assert handle.load().started.count("KEYS_PROVISIONED") == 1
    with handle.key_writer() as (path, digest):
        assert path == handle.path
        assert len(digest) == 64
    assert handle.load().started.count("KEYS_PROVISIONED") == 1, (
        "resuming the writer must not record the intent a second time"
    )
    assert io.calls[-1] == "require_owned_wave"


def test_a_regressed_clock_stops_the_activation(world: ActivationWorld) -> None:
    handle, io, clock = _handle(world)
    handle.prepare()
    clock.advance(-timedelta(minutes=5))
    with pytest.raises(CustodyError, match="clock regressed"):
        handle.step("KEYS_PROVISIONED", io.keys_provisioned)
    assert "keys_provisioned" not in io.calls
    assert "KEYS_PROVISIONED" not in handle.load().started


def test_the_fixed_deadline_stops_the_activation(world: ActivationWorld) -> None:
    handle, io, clock = _handle(world)
    handle.prepare()
    clock.advance(timedelta(seconds=engine.ACTIVATION_TIMEOUT_SECONDS + 1))
    with pytest.raises(CustodyError, match="deadline expired; retain state"):
        handle.step("KEYS_PROVISIONED", io.keys_provisioned)
    assert handle.load() is not None, "the journal must survive for reconciliation"
    assert "keys_provisioned" not in io.calls


def test_a_slow_capture_cannot_start_an_activation_past_its_deadline(
    world: ActivationWorld,
) -> None:
    handle, io, _clock = _handle(world)
    io.capture_advance = timedelta(seconds=engine.ACTIVATION_TIMEOUT_SECONDS + 1)
    with pytest.raises(CustodyError, match="preparation deadline expired"):
        handle.prepare()
    assert io.calls == ["capture"]
    assert not handle.path.exists(), "an expired preparation must leave no journal"
