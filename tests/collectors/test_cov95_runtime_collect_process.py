"""Bounded command execution with fake children and local in-memory pipes."""

from __future__ import annotations

import io
import subprocess
from types import SimpleNamespace

import pytest

from gpu_fault.collectors import process
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


@pytest.mark.parametrize("check", [False, True])
def test_default_runner_preserves_exit_status_and_output_using_fake_popen(
    monkeypatch, check
):
    calls = []
    child = SimpleNamespace(returncode=3, communicate=lambda **kwargs: ("out", "err"))

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return child

    monkeypatch.setattr(process.subprocess, "Popen", popen)
    runner = process.BoundedProcessRunner()
    if check:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            runner(
                ["private-command"],
                capture_output=True,
                text=True,
                timeout=7,
                check=True,
            )
        assert (caught.value.output, caught.value.stderr) == ("out", "err")
    else:
        result = runner(["private-command"], capture_output=True, text=True, timeout=7)
        assert (result.returncode, result.stdout, result.stderr) == (3, "out", "err")
    assert calls == [
        (
            ["private-command"],
            {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True},
        )
    ]


@pytest.mark.parametrize("reap", [False, True])
@pytest.mark.parametrize("empty_argv", [False, True])
def test_timeout_bounds_kill_wait_and_closes_only_fake_child_streams(
    monkeypatch, reap, empty_argv
):
    calls = []
    output, error = io.StringIO(), io.StringIO()
    argv = [] if empty_argv else ["private-command"]

    def communicate(**kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    def kill():
        calls.append("kill")
        raise OSError("fake child already exited")

    def wait(timeout=None):
        calls.append(("wait", timeout))
        if reap and timeout is not None:
            raise subprocess.TimeoutExpired(argv, timeout)
        if reap:
            raise OSError("fake reaper wait failed")
        return 0

    child = SimpleNamespace(
        pid=23,
        returncode=-9,
        stdin=None,
        stdout=output,
        stderr=error,
        communicate=communicate,
        kill=kill,
        wait=wait,
    )
    monkeypatch.setattr(
        process,
        "threading",
        SimpleNamespace(
            Thread=lambda *, target, args, **kwargs: SimpleNamespace(
                start=lambda: target(*args)
            )
        ),
    )
    runner = process.BoundedProcessRunner(
        popen=lambda *args, **kwargs: child, kill_grace_seconds=0.2
    )
    with pytest.raises(subprocess.TimeoutExpired):
        runner(argv, timeout=0.5)
    assert output.closed and error.closed, (
        "timeout cleanup must close both fake output streams"
    )
    assert calls == ["kill", ("wait", 0.2)] + ([("wait", None)] if reap else [])
