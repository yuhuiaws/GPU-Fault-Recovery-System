"""Check holder ownership verdicts and cleanup using only owned local children."""

from __future__ import annotations

import json
import runpy
import subprocess
from pathlib import Path
from typing import Any, NoReturn

import pytest

import gpu_fault.node_agent as node_agent
from tests.regional import _cov95_probe_gap_holders as holder_support
from tests.regional._cov95_probe_gap_holders import (
    MANAGER_OPTIONS,
    PROBE,
    WORKLOAD_PATH,
    HolderHarness,
)

holders = holder_support.holders


def test_holder_opens_only_the_fixture_file_and_is_reaped_by_stop(
    holders: HolderHarness,
) -> None:
    process = PROBE.start_holder("nvidia-persiste", holders.target)
    assert process is holders.processes[0]
    assert PROBE.alive(process) is True
    assert PROBE.settled(process, timeout=0.01) is False

    PROBE.stop(process)

    assert process.poll() is not None
    assert PROBE.settled(process, timeout=0.01) is True
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
    assert holders.stopped == [process.pid]
    assert holders.sweeps == []


def test_holder_refuses_failed_device_open_instead_of_returning_a_process(
    holders: HolderHarness,
) -> None:
    Path(holders.other).unlink()
    holders.wait_before_return = True
    with pytest.raises(
        RuntimeError, match="holder unrelated-gpu failed to start"
    ) as exc:
        PROBE.start_holder("unrelated-gpu", holders.other)

    assert len(holders.processes) == 1
    process = holders.processes[0]
    assert process.wait(timeout=5) == 1
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
    assert "FileNotFoundError" in str(exc.value)
    assert holders.other in str(exc.value)
    assert holders.sweeps == []


def test_cgroup_evidence_filters_roots_empty_and_malformed_rows(
    holders: HolderHarness,
) -> None:
    cgroup = holders.root / "cgroup-sample"
    cgroup.write_text(
        "malformed\n1:cpu\n0::/\n2:cpu:\n3:memory: \n"
        "4:devices:/fixture/owner/\n5:cpu:/fixture/owner\n"
        "6:memory:/fixture/peer\n7:memory:/fixture/with:colon/\n",
        encoding="utf-8",
    )
    holders.proc_files["/proc/73/cgroup"] = cgroup

    assert PROBE.cgroup_paths(73) == {
        "/fixture/owner",
        "/fixture/peer",
        "/fixture/with:colon",
    }
    assert holders.proc_reads == ["/proc/73/cgroup"]
    assert holders.processes == []


def test_main_binds_workload_scope_and_cleans_every_holder_on_successful_return(
    holders: HolderHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    PROBE.main()

    whitelist, same, other = holders.processes
    assert holders.manager_options == [MANAGER_OPTIONS]
    assert [command[3:] for command in holders.launches] == [
        ["nvidia-persiste", holders.target],
        ["unrelated-gpu", holders.target],
        ["unrelated-gpu", holders.other],
    ]
    assert holders.sweeps == [
        {"target_device_paths": {holders.target}, "workload_cgroup_paths": set()},
        {
            "target_device_paths": {holders.target},
            "workload_cgroup_paths": {WORKLOAD_PATH},
        },
    ]
    assert holders.proc_reads == [f"/proc/{same.pid}/cgroup"]
    assert json.loads(capsys.readouterr().out) == {
        "first": {
            "swept_pids": [str(whitelist.pid)],
            "skipped_pids": [str(same.pid)],
            "whitelist_alive": False,
            "same_gpu_alive": True,
            "other_gpu_alive": True,
        },
        "second": {
            "swept_pids": [str(same.pid)],
            "skipped_pids": [],
            "same_gpu_alive": False,
            "other_gpu_alive": True,
            "workload_cgroup_paths": [WORKLOAD_PATH],
        },
    }
    assert holders.stopped == [other.pid, same.pid, whitelist.pid]
    assert all(process.poll() is not None for process in holders.processes), (
        "successful scope checks must not leave a fixture holder running"
    )


@pytest.mark.parametrize(
    ("outcome", "phase", "field", "value"),
    [
        ("missing-whitelist", 1, "swept_pids", []),
        ("missing-skip", 1, "skipped_pids", []),
        ("whitelist-survives", 1, "whitelist_alive", True),
        ("same-stopped-first", 1, "same_gpu_alive", False),
        ("other-stopped-first", 1, "other_gpu_alive", False),
        ("missing-second-sweep", 2, "swept_pids", []),
        ("same-survives-second", 2, "same_gpu_alive", True),
        ("other-stopped-second", 2, "other_gpu_alive", False),
    ],
)
def test_main_refuses_missing_receipts_or_broken_holder_scope_and_cleans_all_children(
    holders: HolderHarness,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
    phase: int,
    field: str,
    value: Any,
) -> None:
    holders.outcome = outcome

    with pytest.raises(AssertionError) as refused:
        PROBE.main()

    assert len(holders.sweeps) == phase
    assert isinstance(refused.value.args[0], dict), (
        "scope refusal must retain the failed phase's structured observations"
    )
    evidence = refused.value.args[0]
    assert evidence[field] == value
    assert ("workload_cgroup_paths" in evidence) is (phase == 2)
    assert holders.proc_reads == (
        [f"/proc/{holders.processes[1].pid}/cgroup"] if phase == 2 else []
    )
    assert capsys.readouterr().out == "", "failed ownership must not emit success"
    assert holders.stopped == [process.pid for process in reversed(holders.processes)]
    assert len(holders.processes) == 3
    assert all(process.poll() is not None for process in holders.processes), (
        "scope refusal must stop every owned holder before fixture teardown"
    )


@pytest.mark.parametrize(
    ("outcome", "phase", "error"),
    [
        ("first-io-error", 1, "fixture first sweep failed"),
        ("second-io-error", 2, "fixture second sweep failed"),
        ("missing-cgroup", 1, "No such file or directory"),
    ],
)
def test_main_preserves_io_failure_and_cleans_owned_children(
    holders: HolderHarness,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
    phase: int,
    error: str,
) -> None:
    holders.outcome = outcome
    with pytest.raises(OSError, match=error):
        PROBE.main()

    assert len(holders.sweeps) == phase
    assert capsys.readouterr().out == ""
    assert holders.stopped == [process.pid for process in reversed(holders.processes)]
    assert len(holders.processes) == 3
    assert all(process.poll() is not None for process in holders.processes), (
        "probe I/O failure must stop every owned holder before fixture teardown"
    )


@pytest.mark.parametrize("failed_start", [2, 3])
def test_startup_failure_must_stop_previously_owned_holders(
    holders: HolderHarness, capsys: pytest.CaptureFixture[str], failed_start: int
) -> None:
    holders.spawn_failure_at = failed_start
    with pytest.raises(OSError, match="fixture holder startup refused"):
        PROBE.main()
    assert len(holders.processes) == failed_start - 1
    assert holders.sweeps == []
    assert all(process.poll() is not None for process in holders.processes), (
        "previously started fixture holders survived main's startup failure"
    )
    assert holders.stopped == [process.pid for process in reversed(holders.processes)]
    assert all(
        stream is not None and stream.closed
        for process in holders.processes
        for stream in (process.stdout, process.stderr)
    ), "startup failure must close the previously started holders' pipes"
    assert capsys.readouterr().out == ""


def test_success_is_published_only_after_all_children_and_streams_are_closed(
    holders: HolderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    published: list[dict[str, Any]] = []

    def publish(payload: str) -> None:
        assert len(holders.stopped) == 3
        assert all(process.poll() is not None for process in holders.processes), (
            "success must not be published while a fixture holder is running"
        )
        assert all(
            stream is not None and stream.closed
            for process in holders.processes
            for stream in (process.stdout, process.stderr)
        ), "success must not be published before every holder pipe is closed"
        published.append(json.loads(payload))

    monkeypatch.setattr(PROBE, "print", publish, raising=False)
    PROBE.main()
    assert len(published) == 1
    assert set(published[0]) == {"first", "second"}


@pytest.mark.parametrize("failed_child", [0, 1, 2])
def test_cleanup_attempts_every_owned_child_and_does_not_publish_after_stop_error(
    holders: HolderHarness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed_child: int,
) -> None:
    stop = PROBE.stop

    def failing_stop(process: subprocess.Popen[str]) -> None:
        stop(process)
        if process is holders.processes[failed_child]:
            raise OSError("fixture stop verification failed")

    monkeypatch.setattr(PROBE, "stop", failing_stop)
    with pytest.raises(OSError, match="fixture stop verification failed"):
        PROBE.main()
    assert holders.stopped == [process.pid for process in reversed(holders.processes)]
    assert all(process.poll() is not None for process in holders.processes), (
        "one cleanup failure must not leave another owned holder running"
    )
    assert all(
        stream is not None and stream.closed
        for process in holders.processes
        for stream in (process.stdout, process.stderr)
    ), "one cleanup failure must not prevent closing other holders' pipes"
    assert capsys.readouterr().out == ""


def test_script_entry_propagates_manager_refusal_without_starting_holders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options: list[dict[str, Any]] = []
    launches: list[str] = []

    def manager(**kwargs: Any) -> NoReturn:
        options.append(kwargs)
        raise OSError("fixture manager initialization refused")

    def forbidden_process(*args: Any, **kwargs: Any) -> NoReturn:
        launches.append("unexpected process")
        raise AssertionError("manager refusal must precede process creation")

    monkeypatch.setattr(node_agent, "GpuServiceQuiesceManager", manager)
    with monkeypatch.context() as guarded:
        guarded.setattr(subprocess, "Popen", forbidden_process)
        with pytest.raises(OSError, match="fixture manager initialization refused"):
            runpy.run_path(str(PROBE.path), run_name="__main__")
    assert options == [MANAGER_OPTIONS]
    assert launches == []
