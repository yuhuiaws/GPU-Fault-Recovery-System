"""Bind complete readiness to the original deadline before transferring ownership."""

from __future__ import annotations

import signal
from types import SimpleNamespace

import pytest

from tests.regional import _cov95_probe_gap_holders as support

PROBE = support.PROBE
holders = support.holders


def receipt_clock(
    holders: support.HolderHarness, monkeypatch: pytest.MonkeyPatch, completed_at: float
) -> tuple[SimpleNamespace, bytearray, list[float]]:
    holders.deadman_seconds = 4
    clock = SimpleNamespace(now=0.0)
    received = bytearray()
    observed: list[float] = []
    read = PROBE.os.read

    def controlled_read(descriptor: int, size: int) -> bytes:
        process = holders.processes[0]
        assert process.stdout is not None and process.stderr is not None, (
            "the owned holder must have both startup pipes"
        )
        assert descriptor in {process.stdout.fileno(), process.stderr.fileno()}, (
            "the clock fixture may read only its owned child's startup pipes"
        )
        chunk = read(descriptor, size)
        if descriptor == process.stdout.fileno():
            received.extend(chunk)
            expected = f"ready:{process.pid}:{holders.target}\n".encode()
            if bytes(received) == expected:
                clock.now = completed_at
                observed.append(completed_at)
        return chunk

    monkeypatch.setattr(PROBE, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(PROBE, "os", SimpleNamespace(read=controlled_read))
    return clock, received, observed


@pytest.mark.parametrize("completed_at", [5.0, 5.25])
def test_complete_exact_receipt_at_or_after_deadline_is_refused_and_cleaned(
    holders: support.HolderHarness, monkeypatch: pytest.MonkeyPatch, completed_at: float
) -> None:
    assert PROBE.STARTUP_TIMEOUT_SECONDS == 5.0, "preserve the five-second policy"
    clock, received, observed = receipt_clock(holders, monkeypatch, completed_at)
    with pytest.raises(RuntimeError, match="readiness timed out"):
        PROBE.start_holder("unrelated-gpu", holders.target)

    assert len(holders.processes) == 1, "only one fake-device holder may be created"
    process = holders.processes[0]
    assert bytes(received) == f"ready:{process.pid}:{holders.target}\n".encode(), (
        "timeout must concern a complete exact receipt, not malformed or partial I/O"
    )
    assert observed == [completed_at] and clock.now >= PROBE.STARTUP_TIMEOUT_SECONDS, (
        "the exact receipt must be observed at or after the absolute deadline"
    )
    assert process.returncode == -signal.SIGKILL, (
        "start_holder must kill and reap the late child before fixture teardown"
    )
    assert process.stdout is not None and process.stdout.closed, (
        "late startup must close stdout before fixture teardown"
    )
    assert process.stderr is not None and process.stderr.closed, (
        "late startup must close stderr before fixture teardown"
    )
    assert holders.deadman_fired == [], "the fixture watchdog must not provide cleanup"
    assert holders.stopped == [], "no explicit test stop may hide startup cleanup"


def test_complete_exact_receipt_before_deadline_is_accepted(
    holders: support.HolderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, received, observed = receipt_clock(holders, monkeypatch, 4.75)
    process = PROBE.start_holder("unrelated-gpu", holders.target)
    assert bytes(received) == f"ready:{process.pid}:{holders.target}\n".encode(), (
        "the positive control must deliver the same complete exact readiness form"
    )
    assert observed == [4.75] and clock.now < PROBE.STARTUP_TIMEOUT_SECONDS, (
        "the positive receipt must be observed before the absolute deadline"
    )
    assert process.poll() is None, "timely readiness must leave the owned holder alive"
    PROBE.stop(process)
    assert process.returncode is not None, (
        "positive-control cleanup must reap its child"
    )
    assert process.stdout is not None and process.stdout.closed, (
        "positive-control cleanup must close stdout"
    )
    assert process.stderr is not None and process.stderr.closed, (
        "positive-control cleanup must close stderr"
    )
    assert holders.deadman_fired == [], "the fixture watchdog must not provide cleanup"
