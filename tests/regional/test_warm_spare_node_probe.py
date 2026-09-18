"""Behavioral regressions for the now owner-bound warm-spare service probe."""

from __future__ import annotations

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
