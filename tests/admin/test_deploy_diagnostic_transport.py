from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time

from gpu_fault.admin.execution import run_driver
from gpu_fault.admin.process_supervisor import write_diagnostic


def fill_pipe(descriptor: int) -> None:
    os.set_blocking(descriptor, False)
    try:
        while True:
            os.write(descriptor, b"x" * 4096)
    except BlockingIOError:
        pass
    finally:
        os.set_blocking(descriptor, True)


def test_full_pipe_drops_progress_without_changing_other_writers():
    read_fd, write_fd = os.pipe()
    try:
        fill_pipe(write_fd)
        with os.fdopen(write_fd, "w") as stream:
            started = time.monotonic()
            assert not write_diagnostic("progress\n", stream=stream), (
                "full pipe accepted diagnostic output"
            )
            assert time.monotonic() - started < 1, "full console blocked progress"
            assert os.get_blocking(stream.fileno()), (
                "diagnostics changed the shared console open-file flags"
            )
            os.read(read_fd, 4096)
            assert write_diagnostic("progress\n", stream=stream), (
                "drained pipe rejected diagnostic output"
            )
    finally:
        os.close(read_fd)


def test_closed_pipe_is_not_a_command_failure():
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    with os.fdopen(write_fd, "w") as stream:
        assert not write_diagnostic("progress\n", stream=stream), (
            "closed pipe accepted diagnostic output"
        )
        assert os.get_blocking(stream.fileno()), "diagnostics changed shared pipe flags"


def test_regular_file_progress_is_deferred_and_final_output_keeps_offset(tmp_path):
    path = tmp_path / "diagnostic"
    with path.open("w") as stream:
        stream.write("before\n")
        stream.flush()
        assert not write_diagnostic("deferred\n", stream=stream), (
            "live progress wrote to a regular file"
        )
        assert write_diagnostic("final\n", stream=stream, final=True), (
            "final file diagnostic was lost"
        )
        stream.write("after\n")
    assert path.read_text() == "before\nfinal\nafter\n"


def test_custom_memory_writer_is_not_invoked_from_the_diagnostic_thread():
    class BlockingWriter(io.StringIO):
        def write(self, text):
            raise AssertionError("arbitrary writer callback must not be invoked")

    assert not write_diagnostic("progress\n", stream=BlockingWriter()), (
        "arbitrary stream callback was accepted"
    )


def test_json_driver_finishes_even_when_console_pipe_is_full(monkeypatch):
    read_fd, write_fd = os.pipe()
    try:
        fill_pipe(write_fd)
        with os.fdopen(write_fd, "w") as stream:
            child = (
                "import sys; "
                "print('release-phase 2026-09-11T00:00:00Z "
                "schema-ready elapsed=1.0s total=1.0s',file=sys.stderr,flush=True); "
                "print('{\"healthy\":true}')"
            )
            with monkeypatch.context() as context:
                context.setattr(sys, "stderr", stream)
                started = time.monotonic()
                result = run_driver(
                    [sys.executable, "-c", child], stdout=subprocess.PIPE
                )
            assert time.monotonic() - started < 3, "console blocked driver cleanup"
            assert result.returncode == 0
            assert json.loads(result.stdout) == {"healthy": True}
    finally:
        os.close(read_fd)
