"""GF-REGIONAL-DESTR-017 runner: the waits that run out of budget instead of
finding what they wait for, the host inventory that names no device for the
holder, a reboot that no longer fits the pinned window, and a cleanup that
finds residual probe resources. Each is driven through ``execute_case`` with
the fence harness and graded on the recorded error and the calls that did or
did not follow."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr017_out_of_band_reboot_fence as case
from tests.regional._cov95_destr_fence import FenceHarness, data
from tests.regional._cov95_destr_warm import NOW


def test_window_remaining_treats_a_naive_expiry_as_utc() -> None:
    now = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
    naive = {"window_expires_at": "2026-09-06T10:05:00"}
    aware = {"window_expires_at": "2026-09-06T10:05:00+00:00"}
    assert case.window_remaining_seconds(naive, now=now) == 300.0
    assert case.window_remaining_seconds(aware, now=now) == 300.0
    assert case.window_remaining_seconds({}, now=now) is None


def test_a_quiesce_that_never_pins_a_window_fails_within_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.pin_delays = 10**6
    h.advance_at["waiting.read"] = case.QUIESCE_PIN_BUDGET_SECONDS / 3 + 1
    code, report = h.execute(tmp_path, seconds=20000)
    assert code == 1, report
    assert "never pinned a maintenance generation" in report["error"], report
    assert "last state: RUNNING" in report["error"], report
    assert h.waiting_reads == 3, "the wait stops at its budget, not on a verdict"
    names = h.names()
    assert "fence.cancel-reboot" in names, names
    assert "aftermath" not in names, names


def test_a_verify_that_never_waits_fails_within_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.verify_delays = 10**6
    h.advance_at["waiting.read"] = case.WAITING_BUDGET_SECONDS / 2 + 1
    code, report = h.execute(tmp_path, seconds=20000)
    assert code == 1, report
    assert "never reached WAITING" in report["error"], report
    assert (tmp_path / "cases" / case.CASE_ID / "maintenance-pin.json").is_file(), (
        "the pin that was found is recorded"
    )
    assert not (tmp_path / "cases" / case.CASE_ID / "waiting-verify.json").exists(), (
        "no barrier observation is recorded when none was seen"
    )
    assert "fence.cancel-reboot" in h.names(), h.names()


def test_observation_stops_at_its_budget_when_the_incident_never_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.terminal_delays = 10**6
    store_snapshot = h.regional.store_snapshot

    def slow_store_after_reboot(**kwargs: Any) -> dict[str, Any]:
        # Only the post-reboot reads burn the clock: the arming and the barrier
        # wait before it must still happen inside the pre-authorization's life.
        if h.rebooted:
            h.clock.sleep(case.OBSERVATION_BUDGET_SECONDS / 2 + 1)
        return store_snapshot(**kwargs)

    monkeypatch.setattr(h.regional, "store_snapshot", slow_store_after_reboot)
    code, report = h.execute(tmp_path, seconds=40000)
    assert h.clock.elapsed >= case.OBSERVATION_BUDGET_SECONDS, h.clock.elapsed
    reads_after_reboot = h.names()[h.names().index("node.ready") :].count("store")
    assert reads_after_reboot <= 4, "the observation stops polling at its budget"
    state = json.loads(
        (tmp_path / "cases" / case.CASE_ID / "workflow-state.json").read_text()
    )
    assert state["incident"]["state"] == "ACTION_PENDING", (
        "the recorded state is the unsettled one the budget ran out on"
    )
    names = h.names()
    assert "aftermath" in names, "the aftermath is still read after the budget"
    assert names.index("aftermath") > names.index("node.ready"), names
    assert report["case_id"] == case.CASE_ID and code in {0, 1}, report


def test_the_holder_needs_a_device_index_from_the_nodes_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    inventory = h.baseline["gpu_inventory"]
    # The node still reports every GPU and the approved device is present; it
    # merely carries no index, and it is not the first entry the lookup walks.
    h.settings = replace(h.settings, pci_bdf=inventory[2]["pci_bdf"])
    inventory[2]["index"] = None
    h.plan(tmp_path)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert "does not name a device index" in report["error"], report
    assert inventory[2]["pci_bdf"] in report["error"], report
    names = h.names()
    assert "fence.cancel-reboot" in names, names
    assert "aftermath" not in names, names


def test_a_reboot_that_outlives_the_pinned_window_is_refused_before_firing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.quiesce["details"]["maintenance_window_expires_at"] = (
        NOW + timedelta(seconds=h.settings.reboot_delay_seconds - 1)
    ).isoformat()
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert "fires after the maintenance window ends" in report["error"], report
    record = (tmp_path / "cases" / case.CASE_ID / "reboot-window.json").read_text()
    assert "window_remaining_seconds" in record
    names = h.names()
    assert "aftermath" not in names, "nothing is graded once the window is lost"
    assert "fence.cancel-reboot" in names, names


def test_cleanup_refuses_residual_probe_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    for probe in (h.host, h.fence):

        def cleanup(label: str = probe.label) -> dict[str, bool]:
            h.call(f"{label}.cleanup")
            # Clean while the preflight looks; a Pod left behind once injected.
            return {"pod": h.injected}

        monkeypatch.setattr(probe, "cleanup", cleanup)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    residual = [
        item for item in report["cleanup"]["errors"] if "residual resources" in item
    ]
    assert len(residual) == 2, report["cleanup"]
    assert all("'pod': True" in item for item in residual), residual
    assert report["verdict"] == "FAIL", report
    assert report["node"] == data.NODE


def test_the_window_is_rechecked_right_before_arming_and_before_injecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    execute = h.host.execute
    snapshots = {"seen": 0}

    def slow_baseline(action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        # The plan's preflight and the run's own preflight each snapshot the
        # host; the third snapshot is the run's baseline, right before arming.
        if action == "snapshot":
            snapshots["seen"] += 1
            if snapshots["seen"] == 3:
                h.clock.sleep(61)
        return execute(action, *args, **kwargs)

    monkeypatch.setattr(h.host, "execute", slow_baseline)
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1, report
    assert "maintenance window ended before arming" in report["error"], report
    assert "fence.arm-holder" not in h.names(), h.names()
    assert h.injected is False


def test_the_window_is_rechecked_after_the_pre_authorization_before_injecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.advance_at["fence.pre-authorize-reboot"] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1, report
    assert "maintenance window ended before injection" in report["error"], report
    assert h.injected is False, h.names()
    assert "fence.cancel-reboot" in h.names(), "the armed timer is still cancelled"
