from __future__ import annotations

import io
import os
import runpy
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import late_ownership_trace as trace
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.probes import late_ownership_tracer_child as child_entry
from tests.regional._late_ownership_support import evidence, process
from tests.regional.test_late_ownership_trace import (
    CALIBRATION,
    SAFE_EXECUTABLE,
    closed_trace,
    exec_line,
)


class FakeTracer:
    pid = 751

    def __init__(self):
        self.stderr = io.BytesIO()
        self.returncode = None
        self.waits = []
        self.timeout_once = False

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.timeout_once:
            self.timeout_once = False
            raise subprocess.TimeoutExpired("owned-tracer", timeout)
        self.returncode = 0
        return 0


@pytest.fixture
def fake_witness(tmp_path, monkeypatch):
    proof = evidence()
    start = proof.witness_starts[0]
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    witness = trace.AttachedExecWitness(
        directory, start.tracee, deadline=time.monotonic() + 20
    )
    child = FakeTracer()
    witness.child = child
    witness.tracer = process(child.pid, start.node.boot_id)
    witness.raw = tempfile.TemporaryFile(mode="w+b", dir=tmp_path)
    witness.raw.write(closed_trace())
    witness.raw.flush()
    witness.pidfd = os.open("/dev/null", os.O_RDONLY)
    witness.start_released = True
    witness.armed = True
    monkeypatch.setattr(
        trace,
        "process_identity",
        lambda pid: witness.tracer if pid == child.pid else start.producer,
    )
    monkeypatch.setattr(trace, "tracer_pids", lambda identity: {child.pid})
    signals = []
    monkeypatch.setattr(
        trace.signal,
        "pidfd_send_signal",
        lambda descriptor, number: signals.append((descriptor, number)),
    )
    try:
        yield witness, child, proof, signals
    finally:
        child.returncode = 0
        witness.close()


@pytest.mark.parametrize("defect", ["unarmed", "closed", "no-child", "no-file"])
def test_inactive_witness_cannot_claim_continuity(fake_witness, defect):
    witness, child, _proof, _signals = fake_witness
    if defect == "unarmed":
        witness.armed = False
    elif defect == "closed":
        witness.closed = True
    elif defect == "no-child":
        witness.child = None
    else:
        witness.raw.close()
        witness.raw = None
    with pytest.raises(BoundaryDenied, match="not armed"):
        witness.snapshot()
    witness.child = child


@pytest.mark.parametrize(
    "defect",
    ["deadline", "tracer-exited", "tracer-replaced", "attachment-lost", "overflow"],
)
def test_every_loss_of_physical_trace_continuity_is_a_refusal(
    fake_witness, monkeypatch, defect
):
    witness, child, _proof, _signals = fake_witness
    if defect == "deadline":
        witness.deadline = time.monotonic() - 1
    elif defect == "tracer-exited":
        child.returncode = 0
    elif defect == "tracer-replaced":
        monkeypatch.setattr(trace, "process_identity", lambda pid: process(pid + 1))
    elif defect == "attachment-lost":
        monkeypatch.setattr(trace, "tracer_pids", lambda identity: {0})
    else:
        witness.raw.truncate(trace.TRACE_LIMIT + 1)
    with pytest.raises(BoundaryDenied, match="continuity"):
        witness.check()


def test_short_physical_read_is_not_no_action(fake_witness, monkeypatch):
    witness, _child, _proof, _signals = fake_witness
    monkeypatch.setattr(trace.os, "pread", lambda *args: b"")
    with pytest.raises(BoundaryDenied, match="incomplete"):
        witness.snapshot()


def test_cleanup_racing_a_snapshot_cannot_invent_an_empty_trace(
    fake_witness, monkeypatch
):
    witness, _child, _proof, _signals = fake_witness

    def lose_handle():
        witness.raw.close()
        witness.raw = None

    monkeypatch.setattr(witness, "check", lose_handle)
    with pytest.raises(BoundaryDenied, match="handle is unavailable"):
        witness.snapshot()


@pytest.mark.parametrize("defect", ["action-before-stop", "wrong-boot", "foreign-node"])
def test_calibration_refuses_early_action_and_wrong_physical_target(
    fake_witness, defect
):
    witness, _child, proof, _signals = fake_witness
    node = proof.scope.nodes[0]
    if defect == "action-before-stop":
        witness.raw.write(
            closed_trace((str(SAFE_EXECUTABLE), "--gpu-reset")).replace(
                b"101 ", b"102 "
            )
        )
        witness.raw.flush()
    elif defect == "wrong-boot":
        witness.tracee = witness.tracee.model_copy(update={"boot_id": "wrong"})
    else:
        node = node.model_copy(update={"uid": "wrong"})
    with pytest.raises(BoundaryDenied, match="preceded|incarnation|escaped"):
        witness.start_receipt(
            proof.scope,
            node,
            witness_id="witness",
            nvidia_smi=SAFE_EXECUTABLE,
            calibration_argv=CALIBRATION,
        )
    assert witness.baseline is None


@pytest.mark.parametrize(
    "defect",
    [
        "no-baseline",
        "wrong-producer",
        "wrong-tracee",
        "scope",
        "executor",
        "commands",
        "callbacks",
        "terminal",
        "revoked",
        "prefix",
    ],
)
def test_trace_end_requires_exact_quiescence_and_unchanged_capture(
    fake_witness, defect
):
    witness, _child, proof, _signals = fake_witness
    start = proof.witness_starts[0]
    quiet = proof.quiescence
    witness.baseline = closed_trace()
    if defect == "no-baseline":
        witness.baseline = None
    elif defect == "wrong-producer":
        start = start.model_copy(update={"producer": process()})
    elif defect == "wrong-tracee":
        start = start.model_copy(update={"tracee": process()})
    elif defect == "prefix":
        witness.raw.seek(0)
        witness.raw.write(b"x")
        witness.raw.flush()
    else:
        field, value = {
            "scope": ("scope_sha256", "0" * 64),
            "executor": ("executor_uid", "wrong"),
            "commands": ("open_commands", 1),
            "callbacks": ("pending_callbacks", 1),
            "terminal": ("workflow_terminal", False),
            "revoked": ("gate_revoked", False),
        }[defect]
        quiet = quiet.model_copy(update={field: value})
    with pytest.raises(BoundaryDenied, match="quiescence|prefix changed"):
        witness.finish_receipt(
            start, quiet, nvidia_smi=SAFE_EXECUTABLE, calibration_argv=CALIBRATION
        )
    assert not witness.closed, (
        "failed closure must preserve the owned observer for cleanup"
    )


@pytest.mark.parametrize("defect", ["replaced", "no-pidfd", "timeout"])
def test_tracer_cleanup_never_signals_an_unpinned_or_replaced_process(
    fake_witness, monkeypatch, defect
):
    witness, child, _proof, signals = fake_witness
    original_fd = witness.pidfd
    if defect == "replaced":
        monkeypatch.setattr(trace, "process_identity", lambda pid: process(pid + 1))
    elif defect == "no-pidfd":
        os.close(witness.pidfd)
        witness.pidfd = None
    else:
        child.timeout_once = True
    with pytest.raises(BoundaryDenied, match="replaced|pidfd|did not finish"):
        witness.close()
    assert witness.closed and witness.raw is None and witness.pidfd is None
    assert signals == (
        [(original_fd, signal.SIGTERM), (original_fd, signal.SIGKILL)]
        if defect == "timeout"
        else []
    )
    assert child.waits == ([10, 3] if defect == "timeout" else [])


def test_empty_and_unreleased_witness_cleanup_reap_without_attachment(tmp_path):
    witness = trace.AttachedExecWitness(
        tmp_path, process(), deadline=time.monotonic() + 20
    )
    witness.close()
    assert witness.closed, "closing an unstarted witness must mark it closed"
    child = FakeTracer()
    witness.child = child
    witness.close()
    assert child.waits == [3]
    assert child.stderr.closed, (
        "witness cleanup must close the tracer diagnostic stream"
    )
    child.returncode = None
    child.timeout_once = True
    with pytest.raises(BoundaryDenied, match="unreleased"):
        witness.close()


@pytest.mark.parametrize(
    "defect",
    [
        "reused",
        "expired",
        "mode",
        "not-directory",
        "symlink",
        "owner",
        "already-traced",
    ],
)
def test_start_refuses_unowned_inputs_before_spawning_a_tracer(
    tmp_path, monkeypatch, defect
):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    witness = trace.AttachedExecWitness(
        directory, process(), deadline=time.monotonic() + 20
    )
    monkeypatch.setattr(trace, "tracer_pids", lambda identity: {0})
    spawned = []
    monkeypatch.setattr(
        trace.subprocess, "Popen", lambda *args, **kwargs: spawned.append(args)
    )
    if defect == "reused":
        witness.closed = True
    elif defect == "expired":
        witness.deadline = float("nan")
    elif defect == "mode":
        directory.chmod(0o755)
    elif defect == "not-directory":
        target = tmp_path / "regular"
        target.write_text("")
        witness.directory = target
    elif defect == "symlink":
        target = tmp_path / "link"
        target.symlink_to(directory)
        witness.directory = target
    elif defect == "owner":
        monkeypatch.setattr(trace.os, "geteuid", lambda: directory.stat().st_uid + 1)
    else:
        monkeypatch.setattr(trace, "tracer_pids", lambda identity: {200})
    with pytest.raises(BoundaryDenied):
        witness.start()
    assert spawned == []


@pytest.mark.parametrize("defect", ["spawn", "pipe"])
def test_startup_failure_closes_anonymous_file_without_leaving_trace_child(
    tmp_path, monkeypatch, defect
):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    witness = trace.AttachedExecWitness(
        directory, process(), deadline=time.monotonic() + 20
    )
    monkeypatch.setattr(trace, "tracer_pids", lambda identity: {0})

    def fail(*args, **kwargs):
        raise OSError("local startup failure")

    if defect == "spawn":
        monkeypatch.setattr(trace.subprocess, "Popen", fail)
    else:
        monkeypatch.setattr(trace.os, "pipe2", fail)
    with pytest.raises(OSError, match="startup"):
        witness.start()
    assert witness.child is None
    assert witness.raw is None
    assert witness.closed, "startup failure must leave the witness closed"


def test_changed_agent_or_absent_thread_inventory_is_not_trace_coverage(monkeypatch):
    identity = process()
    monkeypatch.setattr(trace, "process_identity", lambda pid: process(pid + 1))
    with pytest.raises(BoundaryDenied, match="replaced"):
        trace.tracer_pids(identity)
    monkeypatch.setattr(trace, "process_identity", lambda pid: identity)
    monkeypatch.setattr(Path, "iterdir", lambda path: iter(()))
    with pytest.raises(BoundaryDenied, match="threads"):
        trace.tracer_pids(identity)


def test_nul_in_exec_identity_is_not_a_literal_safe_argument():
    with pytest.raises(BoundaryDenied, match="invalid argument"):
        trace.parse_exec_trace(exec_line(executable="/usr/bin/tr\x00ue").encode())


@pytest.fixture
def child_io(monkeypatch):
    calls = []
    monkeypatch.setattr(
        child_entry.ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(prctl=lambda *args: 0),
    )
    monkeypatch.setattr(child_entry.os, "getppid", lambda: 123)
    monkeypatch.setattr(
        child_entry.resource, "setrlimit", lambda *args: calls.append(("limit", *args))
    )
    monkeypatch.setattr(child_entry.os, "read", lambda *args: b"G")
    monkeypatch.setattr(child_entry.os, "close", lambda fd: calls.append(("close", fd)))
    monkeypatch.setattr(
        child_entry.os, "execvp", lambda *args: calls.append(("exec", *args))
    )
    return calls


def test_child_starts_only_the_tracer_after_pinning_parent_and_limits(child_io):
    arguments = ["123", "8", "4096", "strace", "-f", "-p", "321"]
    assert child_entry.run(arguments) == 125, (
        "an exec that returned must never look successful"
    )
    assert child_io == [
        ("limit", child_entry.resource.RLIMIT_CORE, (0, 0)),
        ("limit", child_entry.resource.RLIMIT_FSIZE, (4096, 4096)),
        ("close", 8),
        ("exec", "strace", ["strace", "-f", "-p", "321"]),
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["x", "8", "4096", "strace"],
        ["0", "8", "4096", "strace"],
        ["123", "-1", "4096", "strace"],
        ["123", "8", "0", "strace"],
        ["123", "8", str(child_entry.MAX_TRACE_BYTES + 1), "strace"],
        ["123", "8", "4096"],
        ["123", "8", "4096", "other-command"],
    ],
)
def test_child_refuses_unbound_or_expanded_startup(child_io, arguments):
    assert child_entry.run(arguments) == 125
    assert child_io == []


@pytest.mark.parametrize(
    "defect", ["prctl", "parent", "eof", "late-parent", "read-error", "exec-error"]
)
def test_child_denies_parent_loss_or_startup_error_without_an_unowned_exec(
    child_io, monkeypatch, defect
):
    if defect == "prctl":
        monkeypatch.setattr(
            child_entry.ctypes,
            "CDLL",
            lambda *args, **kwargs: SimpleNamespace(prctl=lambda *args: -1),
        )
    elif defect == "parent":
        monkeypatch.setattr(child_entry.os, "getppid", lambda: 999)
    elif defect == "eof":
        monkeypatch.setattr(child_entry.os, "read", lambda *args: b"")
    elif defect == "late-parent":
        parents = iter([123, 999])
        monkeypatch.setattr(child_entry.os, "getppid", lambda: next(parents))
    else:

        def fail(*args):
            raise OSError("local child failure")

        monkeypatch.setattr(
            child_entry.os, "read" if defect == "read-error" else "execvp", fail
        )
    assert child_entry.run(["123", "8", "4096", "strace"]) == 125
    assert not any(call[0] == "exec" for call in child_io), (
        "parent loss or startup failure must prevent an unowned tracer exec"
    )
    if defect in {"read-error", "eof", "late-parent", "exec-error"}:
        assert ("close", 8) in child_io


def test_child_main_rejects_invalid_invocation_without_any_side_effect(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(trace.TRACER_CHILD)])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(trace.TRACER_CHILD), run_name="__main__")
    assert raised.value.code == 125


class FakeDiagnostic(io.BytesIO):
    def fileno(self):
        return 501


@pytest.mark.parametrize(
    "defect",
    [
        "missing-stream",
        "timeout",
        "deadline",
        "exited",
        "oversized",
        "prefix",
        "not-attached",
        "partial-threads",
        "pidfd",
        "identity",
    ],
)
def test_attachment_failure_never_mints_an_armed_witness(tmp_path, monkeypatch, defect):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    witness = trace.AttachedExecWitness(
        directory, process(), deadline=time.monotonic() + 20
    )
    child = FakeTracer()
    child.stderr = None if defect == "missing-stream" else FakeDiagnostic()
    child.returncode = 1 if defect == "exited" else None
    monkeypatch.setattr(trace.subprocess, "Popen", lambda *args, **kwargs: child)
    signals = []
    monkeypatch.setattr(
        trace.signal, "pidfd_send_signal", lambda *args: signals.append(args)
    )

    def pidfd(pid):
        if defect == "pidfd":
            raise OSError("pidfd unavailable")
        return os.open("/dev/null", os.O_RDONLY)

    def identity(pid):
        if defect == "identity":
            raise BoundaryDenied("process identity is unavailable")
        return process(pid)

    checks = iter([{0}, {0, child.pid} if defect == "partial-threads" else {child.pid}])
    monkeypatch.setattr(trace.os, "pidfd_open", pidfd)
    monkeypatch.setattr(trace, "process_identity", identity)
    monkeypatch.setattr(trace, "tracer_pids", lambda tracee: next(checks))
    diagnostic = {
        "oversized": b"x" * 4096,
        "prefix": b"attachment unavailable",
        "not-attached": b"strace: Process 200 waiting\n",
    }.get(defect, b"strace: Process 200 attached\n")
    monkeypatch.setattr(trace.os, "read", lambda *args: diagnostic)
    monkeypatch.setattr(
        trace.select,
        "select",
        lambda *args: ([] if defect == "timeout" else [child.stderr], [], []),
    )
    if defect == "deadline":
        witness.deadline = -1
        monkeypatch.setattr(trace, "math_is_valid_deadline", lambda value: True)
    with pytest.raises((BoundaryDenied, OSError)):
        witness.start()
    assert not witness.armed, "failed attachment must not arm the witness"
    assert witness.closed, "failed attachment must close the witness"
    assert witness.raw is None and witness.pidfd is None
    assert list(directory.iterdir()) == []
    if defect in {"pidfd", "identity"}:
        assert not witness.start_released, (
            "identity failure must not release tracer startup"
        )
        assert signals == []
        assert child.waits == [3]
    else:
        assert witness.start_released, (
            "post-release failure must retain proof that tracer startup was released"
        )
        assert child.waits == [10]
