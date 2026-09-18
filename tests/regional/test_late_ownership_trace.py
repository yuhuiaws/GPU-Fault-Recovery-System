from __future__ import annotations

import ctypes
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import pytest

from scripts.e2e.regional import late_ownership_trace as trace
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied, process_identity
from tests.regional._late_ownership_support import evidence

SAFE_EXECUTABLE = Path("/usr/bin/true")
CALIBRATION = (
    str(SAFE_EXECUTABLE),
    "--query-gpu=uuid",
    "--format=csv,noheader,nounits",
)


def hex_string(value: str) -> str:
    return '"' + "".join(f"\\x{byte:02x}" for byte in value.encode()) + '"'


def exec_line(
    pid: int = 101,
    arguments: tuple[str, ...] = CALIBRATION,
    *,
    executable: str = str(SAFE_EXECUTABLE),
    result: str = "0",
) -> str:
    encoded = ", ".join(hex_string(item) for item in arguments)
    return f"{pid}  123.000001 execve({hex_string(executable)}, [{encoded}], 0xff /* 0 vars */) = {result}\n"


def closed_trace(arguments: tuple[str, ...] = CALIBRATION, *, code: int = 0) -> bytes:
    return (
        exec_line(arguments=arguments) + f"101  123.000002 +++ exited with {code} +++\n"
    ).encode()


def test_trace_parser_recovers_actual_exec_identity_and_exit():
    events = trace.parse_exec_trace(closed_trace())
    assert len(events) == 1
    assert events[0] == trace.ExecEvent(
        pid=101,
        started_ns=123_000_001_000,
        ended_ns=123_000_002_000,
        executable=str(SAFE_EXECUTABLE),
        argv=CALIBRATION,
        returncode=0,
    )
    assert (
        trace.physical_actions(
            events, nvidia_smi=SAFE_EXECUTABLE, calibration_argv=CALIBRATION
        )
        == ()
    )


@pytest.mark.parametrize("full", [False, True])
def test_physical_reset_is_counted_from_exec_even_when_model_says_none(full):
    args = (
        (str(SAFE_EXECUTABLE), "--gpu-reset")
        if full
        else (str(SAFE_EXECUTABLE), "--gpu-reset", "-i", "GPU-local")
    )
    data = closed_trace() + closed_trace(args).replace(b"101 ", b"102 ")
    events = trace.parse_exec_trace(data)
    actions = trace.physical_actions(
        events, nvidia_smi=SAFE_EXECUTABLE, calibration_argv=CALIBRATION
    )
    assert len(actions) == 1
    assert actions[0].operation == (
        "RESET_ALL_GPUS_NVSWITCHES" if full else "RESET_GPU"
    )
    assert actions[0].pid == 102
    assert actions[0].returncode == 0
    assert len(actions[0].argv_sha256) == 64


def test_exec_failed_and_process_killed_are_not_zero_actions():
    failed = exec_line(
        102,
        (str(SAFE_EXECUTABLE), "--gpu-reset"),
        result="-1 EACCES (Permission denied)",
    )
    killed = (
        exec_line(103, (str(SAFE_EXECUTABLE), "--gpu-reset"))
        + "103  123.000003 +++ killed by SIGKILL +++\n"
    )
    events = trace.parse_exec_trace(closed_trace() + (failed + killed).encode())
    actions = trace.physical_actions(
        events, nvidia_smi=SAFE_EXECUTABLE, calibration_argv=CALIBRATION
    )
    assert [item.pid for item in actions] == [102, 103]
    assert [item.returncode for item in actions] == [-1, -1]


def test_unfinished_exec_is_joined_only_to_its_own_resumption():
    line = exec_line().strip()
    before, after = line.rsplit(" = ", 1)
    data = (
        before
        + " <unfinished ...>\n"
        + f"101  123.000002 <... execve resumed> = {after}\n"
        + "101  123.000003 +++ exited with 0 +++\n"
    ).encode()
    event = trace.parse_exec_trace(data)[0]
    assert event.started_ns == 123_000_001_000
    assert event.ended_ns == 123_000_003_000


@pytest.mark.parametrize(
    "data,error",
    [
        (b"x" * (trace.TRACE_LIMIT + 1), "oversized"),
        (b"not-terminated", "incomplete"),
        (b"\xff\n", "pinned format"),
        (b"unstructured output\n", "unaccounted"),
        (b"0  123.000001 +++ exited with 0 +++\n", "identity changed"),
        (b'101  123.000001 execveat(AT_FDCWD, "x", [], NULL, 0) = 0\n', "unsupported"),
        (b"101  123.000001 <... execve resumed>) = 0\n", "unknown syscall"),
        (b"101  123.000001 execve( <unfinished ...>\n", "drained"),
        (b"101  123.000001 open( <unfinished ...>\n", "ambiguous"),
        (
            b"101  123.000001 execve( <unfinished ...>\n101  123.000002 execve( <unfinished ...>\n",
            "ambiguous",
        ),
        (
            b"101  123.000001 execve( <unfinished ...>\n101  123.000002 +++ exited with 0 +++\n",
            "unfinished exec",
        ),
        (exec_line().encode(), "drained"),
        ((exec_line() + exec_line()).encode(), "exec chain"),
        ((exec_line() + "101  122.999999 +++ exited with 0 +++\n").encode(), "clock"),
        (exec_line(executable="relative").encode(), "absolute"),
        (exec_line(arguments=()).encode(), "absolute"),
    ],
)
def test_incomplete_or_unsupported_trace_is_not_no_action(data, error):
    with pytest.raises(BoundaryDenied, match=error):
        trace.parse_exec_trace(data)


def test_unexecuted_attached_parent_exit_is_accounted_without_inventing_exec():
    assert trace.parse_exec_trace(b"101  123.000001 +++ exited with 0 +++\n") == ()
    assert trace.parse_exec_trace(b"101  123.000001 +++ killed by SIGTERM +++\n") == ()


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "failed",
        "unknown-tool",
        "unknown-nvidia",
        "bad-target",
        "no-args",
    ],
)
def test_zero_action_requires_exact_successful_calibration_and_recognized_tools(defect):
    events = trace.parse_exec_trace(closed_trace())
    executable = SAFE_EXECUTABLE
    calibration = CALIBRATION
    if defect == "missing":
        events = ()
    elif defect == "duplicate":
        events = events * 2
    elif defect == "failed":
        events = trace.parse_exec_trace(closed_trace(code=2))
    elif defect == "unknown-tool":
        executable = Path("/usr/bin/other")
    elif defect == "unknown-nvidia":
        events += trace.parse_exec_trace(closed_trace((str(SAFE_EXECUTABLE), "-r")))
    elif defect == "bad-target":
        executable = Path("true")
    else:
        calibration = ()
    with pytest.raises(BoundaryDenied):
        trace.physical_actions(
            events, nvidia_smi=executable, calibration_argv=calibration
        )


@pytest.mark.parametrize(
    "args",
    [
        (
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ),
        ("--query-gpu=uuid,index", "--format=csv,noheader,nounits"),
        ("-q", "-x"),
    ],
)
def test_known_readonly_node_client_checks_do_not_count_as_resets(args):
    events = trace.parse_exec_trace(
        closed_trace()
        + closed_trace((str(SAFE_EXECUTABLE), *args)).replace(b"101 ", b"102 ")
    )
    assert (
        trace.physical_actions(
            events, nvidia_smi=SAFE_EXECUTABLE, calibration_argv=CALIBRATION
        )
        == ()
    )


def trace_target(connection):
    # Grant ptrace only to this test's parent and its children, not arbitrary
    # same-UID processes. The target runs only the fixed harmless executable.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(0x59616D61, os.getppid(), 0, 0, 0) != 0:
        connection.send("ptrace-refused")
        return
    connection.send("ready")
    while True:
        command = connection.recv()
        if command == "exit":
            return
        args = (
            CALIBRATION
            if command == "calibrate"
            else (str(SAFE_EXECUTABLE), "--gpu-reset", "-i", "GPU-local")
        )
        completed = subprocess.run(
            args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        connection.send(completed.returncode)


@pytest.fixture
def owned_tracee():
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    worker = context.Process(target=trace_target, args=(child,))
    worker.start()
    child.close()
    try:
        assert parent.poll(10), "owned trace target did not become ready"
        assert parent.recv() == "ready", "owned trace target could not admit its tracer"
        yield worker, parent
    finally:
        if worker.is_alive():
            parent.send("exit")
        worker.join(10)
        if worker.is_alive():
            worker.terminate()
            worker.join(10)
        parent.close()
        assert not worker.is_alive(), "owned trace target was not reaped"
        worker.close()


@pytest.mark.parametrize("execute_action", [False, True])
def test_real_owned_child_exec_witness_cannot_be_replaced_with_model_counters(
    tmp_path, owned_tracee, execute_action
):
    worker, peer = owned_tracee
    directory = tmp_path / "witness"
    directory.mkdir(mode=0o700)
    identity = process_identity(worker.pid)
    witness = trace.AttachedExecWitness(
        directory, identity, deadline=time.monotonic() + 20
    )
    try:
        witness.start()
        peer.send("calibrate")
        assert peer.poll(10), "physical calibration was not acknowledged"
        assert peer.recv() == 0
        before = witness.snapshot()
        assert (
            trace.physical_actions(
                trace.parse_exec_trace(before),
                nvidia_smi=SAFE_EXECUTABLE,
                calibration_argv=CALIBRATION,
            )
            == ()
        )
        if execute_action:
            peer.send("deliberately-cross-boundary")
            assert peer.poll(10), "owned physical action did not complete"
            assert peer.recv() == 0
        actions = trace.physical_actions(
            trace.parse_exec_trace(witness.snapshot()),
            nvidia_smi=SAFE_EXECUTABLE,
            calibration_argv=CALIBRATION,
        )
        assert len(actions) == int(execute_action), (
            "model zero counters cannot erase a kernel exec"
        )
        assert list(directory.iterdir()) == [], "raw trace must remain anonymous"
    finally:
        witness.close()
    assert worker.is_alive(), "witness cleanup must not signal the observed process"


def test_trace_receipts_are_minted_only_from_real_calibrated_capture(
    tmp_path, owned_tracee
):
    worker, peer = owned_tracee
    directory = tmp_path / "witness"
    directory.mkdir(mode=0o700)
    identity = process_identity(worker.pid)
    witness = trace.AttachedExecWitness(
        directory, identity, deadline=time.monotonic() + 20
    )
    proof = evidence()
    node = proof.scope.nodes[0].model_copy(update={"boot_id": identity.boot_id})
    binding = proof.scope.model_copy(update={"nodes": (node, proof.scope.nodes[1])})
    try:
        witness.start()
        with pytest.raises(BoundaryDenied, match="calibration"):
            witness.start_receipt(
                binding,
                node,
                witness_id="witness-local",
                nvidia_smi=SAFE_EXECUTABLE,
                calibration_argv=CALIBRATION,
            )
        peer.send("calibrate")
        assert peer.poll(10), "physical calibration was not acknowledged"
        assert peer.recv() == 0
        start = witness.start_receipt(
            binding,
            node,
            witness_id="witness-local",
            nvidia_smi=SAFE_EXECUTABLE,
            calibration_argv=CALIBRATION,
        )
        with pytest.raises(BoundaryDenied, match="reused"):
            witness.start_receipt(
                binding,
                node,
                witness_id="witness-local",
                nvidia_smi=SAFE_EXECUTABLE,
                calibration_argv=CALIBRATION,
            )
        quiet = proof.quiescence.model_copy(update={"scope_sha256": binding.digest()})
        end = witness.finish_receipt(
            start, quiet, nvidia_smi=SAFE_EXECUTABLE, calibration_argv=CALIBRATION
        )
        assert end.actions == ()
        assert end.trace_bytes > 0
        assert end.exec_events == 1
        assert end.start_sha256 == start.digest()
        assert end.quiescence_sha256 == quiet.digest()
        assert witness.closed, "finishing the calibrated trace must close the witness"
    finally:
        witness.close()
