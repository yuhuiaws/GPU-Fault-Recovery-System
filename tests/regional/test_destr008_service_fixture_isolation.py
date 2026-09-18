"""A modeled host clock must not replace shared process-wide timing or I/O."""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts.e2e.regional import destr008_service_window as controller
from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from tests.regional._destr008_service_window import Host, Transport
from tests.regional.test_destr008_service_controller import build


def test_modeled_host_preserves_process_wide_stdlib_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clocks = {
        name: getattr(time, name)
        for name in ("time", "monotonic", "clock_gettime", "sleep")
    }
    run = subprocess.run

    Host(tmp_path, monkeypatch)

    for name, original in clocks.items():
        assert getattr(time, name) is original, (
            f"the host fixture replaced process-wide time.{name}"
        )
    assert subprocess.run is run, (
        "unrelated subprocess clients must not enter the modeled systemctl boundary"
    )


def test_controller_and_probe_share_only_the_modeled_host_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = Host(tmp_path, monkeypatch)

    assert controller.time is probe.time and probe.time is not time, (
        "the modeled service participants need their own shared clock"
    )
    probe.time.sleep(2)
    controller.time.sleep(3)
    assert host.now == 2_000_000_005 and host.elapsed == 1005, (
        "modeled sleeps must still advance wall and boot time together"
    )
    assert controller.time.time() == host.now, "controller lost the modeled wall clock"
    assert probe.time.clock_gettime(probe.time.CLOCK_BOOTTIME) == host.elapsed, (
        "independent recovery lost the modeled boot clock"
    )


def test_unrelated_thread_sleep_cannot_expire_a_modeled_service_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    host = Host(tmp_path, monkeypatch)
    window = build(Transport(host, tmp_path / "controller"))
    try:
        window.create()
        before = (host.now, host.elapsed)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(time.sleep, 0.001).result(timeout=5)
        assert (host.now, host.elapsed) == before, (
            "an unrelated sleeper advanced the service fixture's maintenance clock"
        )
        assert (
            window.stop("kubelet.service", delay_seconds=15)["phase"] == "SCHEDULED"
        ), "the original approved service window must still be available"
        assert window.binding["restore_at"] == 2_000_000_195, (
            "isolation must not widen or recompute the approved recovery duration"
        )
        host.fire_stop()
        assert window.restore()["phase"] == "RESTORED", (
            "the modeled original service still needs its bound restoration proof"
        )
        assert all(value is False for value in window.cleanup().values()), (
            "the regression must finish its modeled service and transport cleanup"
        )
    finally:
        window.close()
