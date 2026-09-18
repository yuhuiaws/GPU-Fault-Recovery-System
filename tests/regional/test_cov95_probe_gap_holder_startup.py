"""Bound startup I/O and prove failed holders cannot escape local ownership."""

from __future__ import annotations

import errno
import signal
import subprocess
import time
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from tests.regional import _cov95_probe_gap_holders as holder_support
from tests.regional._cov95_probe_gap_holders import PROBE, HolderHarness

holders = holder_support.holders


@pytest.fixture
def startup(holders: HolderHarness, monkeypatch: pytest.MonkeyPatch) -> HolderHarness:
    holders.deadman_seconds = 4
    monkeypatch.setattr(PROBE, "STARTUP_TIMEOUT_SECONDS", 1.0)
    return holders


def program(body: str, *, ignore_term: bool = True, linger: bool = True) -> str:
    code = (
        "import os, signal, sys, time\n"
        "name, device = sys.argv[1:3]\n"
        "descriptor = os.open(device, os.O_RDONLY)\n"
    )
    if ignore_term:
        code += "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    code += body + "\n"
    if linger:
        code += "time.sleep(60)\n"
    return code


def assert_failed_process_cleaned(startup: HolderHarness, returncode: int) -> None:
    process = startup.processes[-1]
    assert process.returncode == returncode, "start_holder itself must reap its child"
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
    assert startup.deadman_fired == [], "cleanup must precede the fixture watchdog"


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("pass", "readiness timed out"),
        (
            "os.write(1, f'ready:{os.getpid()}:{device}'.encode())",
            "readiness timed out",
        ),
        ("os.close(2)\nos.write(1, b'ready:')", "readiness timed out"),
        ("os.close(1)", "stdout ended before readiness"),
        ("os.write(1, b'x' * 4097)", "startup output limit exceeded"),
        ("os.write(2, b'e' * 4097)", "startup output limit exceeded"),
        (
            "print(f'ready:{os.getpid()}:{device}.foreign', flush=True)",
            "readiness identity mismatch",
        ),
        ("print('ready:', flush=True)", "readiness identity mismatch"),
        (
            "os.write(1, f'ready:{os.getpid()}:{device}\\nextra\\n'.encode())",
            "readiness identity mismatch",
        ),
    ],
)
def test_startup_refuses_stalls_unbounded_output_and_malformed_receipts(
    startup: HolderHarness, monkeypatch: pytest.MonkeyPatch, body: str, reason: str
) -> None:
    monkeypatch.setattr(PROBE, "HELPER", program(body))
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=reason):
        PROBE.start_holder("unrelated-gpu", startup.target)
    assert time.monotonic() - started < 3
    assert len(startup.processes) == 1
    assert_failed_process_cleaned(startup, -signal.SIGKILL)
    assert startup.sweeps == []


def test_readiness_cannot_claim_another_owned_pid_or_kill_that_peer(
    startup: HolderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    peer = PROBE.start_holder("nvidia-persiste", startup.other)
    monkeypatch.setattr(
        PROBE, "HELPER", program(f"print(f'ready:{peer.pid}:{{device}}', flush=True)")
    )
    with pytest.raises(RuntimeError, match="readiness identity mismatch"):
        PROBE.start_holder("unrelated-gpu", startup.target)
    assert_failed_process_cleaned(startup, -signal.SIGKILL)
    assert peer.poll() is None
    assert peer.stdout is not None and not peer.stdout.closed
    assert peer.stderr is not None and not peer.stderr.closed
    PROBE.stop(peer)
    assert peer.returncode is not None
    assert peer.stdout.closed and peer.stderr.closed


def test_already_exited_child_is_not_accepted_even_with_an_exact_receipt(
    startup: HolderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup.wait_before_return = True
    monkeypatch.setattr(
        PROBE,
        "HELPER",
        program(
            "print(f'ready:{os.getpid()}:{device}', flush=True)",
            ignore_term=False,
            linger=False,
        ),
    )
    with pytest.raises(RuntimeError, match="holder already exited"):
        PROBE.start_holder("unrelated-gpu", startup.target)
    assert_failed_process_cleaned(startup, 0)


def test_fragmented_receipt_and_closed_stderr_are_accepted_within_the_deadline(
    startup: HolderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        PROBE,
        "HELPER",
        program(
            "os.close(2)\n"
            "os.write(1, b'ready:')\n"
            "time.sleep(0.02)\n"
            "os.write(1, str(os.getpid()).encode() + b':')\n"
            "time.sleep(0.02)\n"
            "os.write(1, device.encode() + b'\\n')",
            ignore_term=False,
        ),
    )
    process = PROBE.start_holder("unrelated-gpu", startup.target)
    assert process.poll() is None
    PROBE.stop(process)
    assert_failed_process_cleaned(startup, -signal.SIGTERM)


@pytest.mark.parametrize("overflow", [False, True])
def test_stdout_and_stderr_share_one_inclusive_output_budget(
    startup: HolderHarness, monkeypatch: pytest.MonkeyPatch, overflow: bool
) -> None:
    limit = 256
    monkeypatch.setattr(PROBE, "STARTUP_MAX_BYTES", limit)
    monkeypatch.setattr(
        PROBE,
        "HELPER",
        program(
            "receipt = f'ready:{os.getpid()}:{device}\\n'.encode()\n"
            f"os.write(2, b'e' * ({limit} - len(receipt) + {int(overflow)}))\n"
            "os.write(1, receipt)",
            ignore_term=False,
        ),
    )
    if overflow:
        with pytest.raises(RuntimeError, match="startup output limit exceeded"):
            PROBE.start_holder("unrelated-gpu", startup.target)
        assert_failed_process_cleaned(startup, -signal.SIGKILL)
    else:
        process = PROBE.start_holder("unrelated-gpu", startup.target)
        assert process.poll() is None
        PROBE.stop(process)
        assert_failed_process_cleaned(startup, -signal.SIGTERM)


def test_read_progress_does_not_reset_the_absolute_startup_deadline(
    startup: HolderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(PROBE, "HELPER", program("os.write(1, b'ready:')"))
    clock = iter([0.0, 0.25, 1.25])
    reads: list[bytes] = []
    read = PROBE.os.read

    def record_read(descriptor: int, size: int) -> bytes:
        chunk: bytes = read(descriptor, size)
        reads.append(chunk)
        return chunk

    monkeypatch.setattr(PROBE, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(PROBE, "os", SimpleNamespace(read=record_read))
    with pytest.raises(RuntimeError, match="readiness timed out"):
        PROBE.start_holder("unrelated-gpu", startup.target)
    assert reads == [b"ready:"]
    assert_failed_process_cleaned(startup, -signal.SIGKILL)


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.EIO, "fixture pipe read failed"),
        KeyboardInterrupt("fixture cancel"),
    ],
)
def test_startup_io_failure_or_cancellation_preserves_error_and_cleans_child(
    startup: HolderHarness, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    reads: list[int] = []

    def failed_read(descriptor: int, size: int) -> NoReturn:
        process = startup.processes[0]
        assert process.stdout is not None and process.stderr is not None
        assert descriptor in {process.stdout.fileno(), process.stderr.fileno()}
        assert 0 < size <= PROBE.STARTUP_MAX_BYTES + 1
        reads.append(descriptor)
        raise error

    monkeypatch.setattr(PROBE, "os", SimpleNamespace(read=failed_read))
    with pytest.raises(type(error)) as refused:
        PROBE.start_holder("unrelated-gpu", startup.target)
    assert refused.value is error
    assert len(reads) == 1
    assert_failed_process_cleaned(startup, -signal.SIGKILL)


def test_failed_startup_still_reaps_and_closes_when_kill_reports_an_error(
    startup: HolderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(PROBE, "HELPER", program("print('wrong', flush=True)"))
    popen = startup.popen
    kills: list[int] = []

    def launch(command: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        process = popen(command, **kwargs)
        kill = process.kill

        def failed_kill_ack() -> NoReturn:
            kills.append(process.pid)
            kill()
            raise OSError("fixture kill acknowledgement failed")

        monkeypatch.setattr(process, "kill", failed_kill_ack)
        return process

    monkeypatch.setattr(PROBE.subprocess, "Popen", launch)
    with pytest.raises(OSError, match="fixture kill acknowledgement failed") as failed:
        PROBE.start_holder("unrelated-gpu", startup.target)
    assert isinstance(failed.value.__context__, RuntimeError), (
        "cleanup failure must retain the original startup refusal as its context"
    )
    assert "readiness identity mismatch" in str(failed.value.__context__)
    assert kills == [startup.processes[0].pid]
    assert_failed_process_cleaned(startup, -signal.SIGKILL)


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_stop_closes_the_other_pipe_even_when_one_close_reports_failure(
    startup: HolderHarness, monkeypatch: pytest.MonkeyPatch, stream_name: str
) -> None:
    process = PROBE.start_holder("unrelated-gpu", startup.target)
    stream = getattr(process, stream_name)
    assert stream is not None
    close = stream.close

    def failed_close_ack() -> NoReturn:
        close()
        raise OSError("fixture pipe close acknowledgement failed")

    with monkeypatch.context() as guarded:
        guarded.setattr(stream, "close", failed_close_ack)
        with pytest.raises(OSError, match="fixture pipe close acknowledgement failed"):
            PROBE.stop(process)
    assert process.returncode is not None
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed


def test_stop_closes_streams_and_reports_a_signal_failure_without_claiming_exit(
    startup: HolderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = PROBE.start_holder("unrelated-gpu", startup.target)
    signals: list[int] = []

    def refused_signal(sig: int) -> NoReturn:
        signals.append(sig)
        raise PermissionError("fixture signal refused")

    with monkeypatch.context() as guarded:
        guarded.setattr(process, "send_signal", refused_signal)
        with pytest.raises(PermissionError, match="fixture signal refused"):
            PROBE.stop(process)
    assert signals == [signal.SIGTERM]
    assert process.poll() is None
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
    process.kill()
    assert process.wait(timeout=5) == -signal.SIGKILL
