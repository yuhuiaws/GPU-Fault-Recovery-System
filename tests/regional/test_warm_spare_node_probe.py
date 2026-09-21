"""Behavioral regressions for the now owner-bound warm-spare service probe."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest

from scripts.e2e.regional.probes import warm_spare_node_probe as PROBE
from tests.regional._destr008_service_window import Host


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    return Host(tmp_path, monkeypatch)


def test_a_delayed_stop_answers_before_the_service_goes_down(host: Host) -> None:
    host.arm()
    report = host.window.schedule_stop()
    assert report["phase"] == "SCHEDULED" and report["stop_requested"] is False
    assert host.units["kubelet.service"]["ActiveState"] == "active"
    assert report["restore_at"] == int(host.now) + 195
    mutations = host.mutations()
    arm = (
        "start",
        "--no-block",
        "--job-mode=fail",
        host.window.units["restore-service"],
    )
    schedule = (
        "start",
        "--no-block",
        "--job-mode=fail",
        host.window.units["stop-timer"],
    )
    assert mutations.index(arm) < mutations.index(schedule)
    assert ("stop", "--no-block", "--job-mode=fail", "kubelet.service") not in mutations
    host.fire_stop()
    assert host.units["kubelet.service"]["ActiveState"] == "inactive"


def test_an_inline_stop_still_verifies_the_service_actually_stopped(host: Host) -> None:
    # Zero delay also uses the owned service; controller transport never carries
    # an inline stop, and the actual stopped state is separately observed.
    service = "gpu-fault-node-agent.service"
    host.binding = host.make_binding(service, delay=0)
    host.window = PROBE.ServiceWindow(host.binding)
    report = host.stopped()
    assert report["stop_requested"] is True
    assert PROBE.service_snapshot(service)["ActiveState"] == "inactive"
    assert host.read()["stop_observation"]["Job"] == "0"
    assert host.window.restore()["after"]["ActiveState"] == "active"


def test_an_inline_stop_that_left_the_service_active_is_an_error(host: Host) -> None:
    host.binding = host.make_binding("gpu-fault-node-agent.service", delay=0)
    host.window = PROBE.ServiceWindow(host.binding)
    host.arm()
    host.window.schedule_stop()
    service = host.binding["service"]
    host.fail.add(("stop", "--no-block", "--job-mode=fail", service))
    with pytest.raises(PROBE.ProbeError, match="systemctl failed"):
        host.fire_stop()
    assert host.read()["stop_requested"] is True
    assert host.window.targets["restore-service"].exists(), host.window.targets
    assert host.units[service]["ActiveState"] == "active"


def test_restore_disarms_the_pending_stop_before_starting_the_service(
    host: Host,
) -> None:
    host.stopped()
    # Timer cancellation alone is insufficient if the stop helper is still live.
    name = host.window.units["stop-service"]
    host.units[name].update(ActiveState="active", SubState="running", MainPID="55")
    report = host.window.restore()
    mutations = host.mutations()
    start = mutations.index(
        ("start", "--no-block", "--job-mode=fail", "kubelet.service")
    )
    assert mutations.index(("stop", host.window.units["stop-timer"])) < start
    assert mutations.index(("stop", name)) < start
    assert report["stop_quiescence"]["timer"]["job_id"] == 0
    assert report["stop_quiescence"]["service"]["main_pid"] == 0


def test_the_delay_is_bounded(host: Host) -> None:
    before = list(host.calls)
    assert (
        PROBE.main(
            [
                "stop-with-failsafe",
                "--service",
                "kubelet.service",
                "--run-id",
                "test-run",
                "--stop-delay-seconds",
                "600",
            ]
        )
        == 1
    )
    assert host.calls == before


def test_failed_restore_keeps_the_independent_start_timer(host: Host) -> None:
    host.stopped()
    host.pending_start = True
    with pytest.raises(PROBE.ProbeError, match="quiescent"):
        host.window.restore()
    assert host.read()["phase"] == "RESTORING"
    assert ("stop", host.window.units["restore-service"]) not in host.mutations()
    assert host.window.targets["restore-service"].exists(), host.window.targets


def test_file_identity_accepts_files_writable_only_by_the_callers_own_group(
    tmp_path: Path,
) -> None:
    # This EKS AMI ships /usr/bin/kubelet as 0775 root:root (live 2026-09-19):
    # group-writable by the caller's own group is still the caller's privilege,
    # while a world-writable file lets any principal change the service.
    import os

    own_group = tmp_path / "kubelet"
    own_group.write_text("#!/bin/sh\n")
    own_group.chmod(0o775)
    os.chown(own_group, os.geteuid(), os.getegid())
    identity = PROBE.file_identity(own_group)
    assert identity["mode"] == 0o775 and len(identity["sha256"]) == 64, identity
    world = tmp_path / "environment"
    world.write_text("KEY=value\n")
    world.chmod(0o666)
    with pytest.raises(PROBE.ProbeError, match="protected regular file"):
        PROBE.file_identity(world)
    foreign_groups = [gid for gid in os.getgroups() if gid != os.getegid()]
    if foreign_groups:
        foreign = tmp_path / "foreign"
        foreign.write_text("x\n")
        foreign.chmod(0o775)
        os.chown(foreign, os.geteuid(), foreign_groups[0])
        with pytest.raises(PROBE.ProbeError, match="protected regular file"):
            PROBE.file_identity(foreign)


def _hold_lock() -> int:
    holder = os.open(PROBE.ROOT / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)
    return holder


def test_a_request_landing_inside_a_watcher_tick_waits_for_the_lock(
    host: Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The independent watcher holds the journal lock for every tick; a controller
    # request that lands inside one must wait for it, not fail the window.
    host.arm()
    holder = _hold_lock()
    polls = 0
    original = host.clock.sleep

    def sleep(seconds: float) -> None:
        nonlocal polls
        polls += 1
        original(seconds)
        if polls == 3:
            fcntl.flock(holder, fcntl.LOCK_UN)

    monkeypatch.setattr(host.clock, "sleep", sleep)
    try:
        assert host.window.status()["phase"] == "ARMED"
    finally:
        os.close(holder)
    assert polls == 3


def test_a_lock_held_past_the_bound_still_fails_closed(host: Host) -> None:
    host.arm()
    holder = _hold_lock()
    try:
        with pytest.raises(BlockingIOError):
            with host.window.locked():
                pass
    finally:
        os.close(holder)
