"""Real probe classification, restore wrapper and provider proof share one guard."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts.e2e.regional import collector_inventory_reboot as runner
from scripts.e2e.regional import host_probe_fixture as host_module
from scripts.e2e.regional.collector_acceptance_fixture import CollectorAcceptanceFixture
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture
from tests.regional._collector_inventory_reboot_support import APPLIED, BASELINE, INTENT
from tests.regional._cov95_collect_net import Clock
from tests.regional._host_probe_support import ProbeApi, host_probe
from tests.regional.test_collector_reboot_evidence import START, inputs


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize("ready", ["True", "False", "Unknown"])
def test_empty_response_and_released_reboot_wait_without_recreating_offline_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, returncode: int, ready: str
) -> None:
    value = inputs()
    scope = value["scope"]
    scope["node_id"] = "node-a"
    workflow = value["workflow"]
    workflow.update(
        status="RUNNING",
        completed_step_indexes=[],
        updated_at=(START + timedelta(seconds=6)).isoformat(),
    )
    workflow["official_steps"][0]["node_ids"] = ["node-a"]
    workflow["step_executions"][0].update(
        status="WAITING", updated_at=workflow["updated_at"]
    )
    command = value["commands"][0]
    command.update(
        status="LEASED",
        step=deepcopy(workflow["official_steps"][0]),
        updated_at=(START + timedelta(seconds=20)).isoformat(),
    )
    value["submission"]["requested_node_identifiers"] = ["node-a"]
    observed = START + timedelta(seconds=21)
    state = {
        "event_id": "owned-event",
        "incidents": [{"incident_id": workflow["incident_id"]}],
        "workflows": [workflow],
        "node_workflow_ids": [workflow["request_id"]],
        "commands": value["commands"],
        "submissions": [value["submission"]],
        "captured_at": observed.isoformat(),
    }
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    monkeypatch.setattr(
        runner, "datetime", SimpleNamespace(now=lambda tz=None: observed)
    )
    node = {
        "name": "node-a",
        "uid": scope["node_uid"],
        "boot_id": scope["boot_id"],
        "ready": ready,
    }
    reads: list[str] = []

    def cpu(script: str, *args: str) -> dict[str, Any]:
        assert script == runner.INVENTORY_RECOVERY_STATE
        reads.append("submission")
        return deepcopy(state)

    regional = SimpleNamespace(
        cpu_python=cpu, node_snapshot=lambda *args: deepcopy(node)
    )
    settings = SimpleNamespace(
        node="node-a",
        hyperpod_cluster=scope["cluster_name"],
        regional=SimpleNamespace(cluster_id=scope["cluster_id"]),
    )
    recovery = runner.InventoryRecovery(
        cast(RegionalLiveFixture, regional),
        settings,
        scope,
        "run-a",
        "d" * 32,
        START,
        START,
        9,
        BASELINE,
        tmp_path,
        1,
        event_id="owned-event",
    )
    recovery.accept_arming(
        {
            "run_id": "run-a",
            "cluster_id": scope["cluster_id"],
            "node_id": "node-a",
            "boot_id": scope["boot_id"],
            "baseline_sha256": BASELINE,
            "applied_sha256": APPLIED,
            "intent_sha256": INTENT,
            "mutation_started": True,
            "timer_armed": True,
            "boot_restore_armed": True,
        }
    )
    api = ProbeApi()
    api.node_uid = scope["node_uid"]
    monkeypatch.setattr(host_module, "run_fixture_command", api.run)
    host = host_probe(tmp_path, case_id="GF-REGIONAL-COLLECT-004")
    host.create()
    creations_before = len([args for args, _ in api.calls if args[0] == "create"])
    api.probe_stdout, api.probe_returncode = "", returncode
    collector = cast(CollectorAcceptanceFixture, SimpleNamespace(execute=host.execute))

    deferred = recovery.restore(collector, allow_defer=True)

    assert deferred["deferred"] is True and deferred["restored"] is False
    assert deferred["error_type"] == "HostProbeMissingResponseError"
    assert recovery.environment_restored is False
    assert reads == ["submission", "submission"]
    assert (
        len([args for args, _ in api.calls if args[0] == "create"]) == creations_before
    )
    assert len([args for args, _ in api.calls if args[0] == "exec"]) == 1
    with pytest.raises(RegionalFixtureError, match="operator hold"):
        recovery.read_for_cleanup()

    original_deadline = recovery.reboot_deadline_monotonic
    transitions = iter(
        [
            {**node, "ready": "True"},
            {**node, "ready": "Unknown"},
            {**node, "ready": "True", "boot_id": "new-boot"},
        ]
    )
    regional.node_snapshot = lambda *args: next(transitions)
    assert recovery.wait_reboot()["boot_id"] == "new-boot"
    assert clock.sleeps == [5, 5]
    assert recovery.reboot_deadline_monotonic == original_deadline
    host.cleanup()
    host.create()
    api.probe_returncode = 0
    api.probe_stdout = json.dumps(
        {
            "run_id": "run-a",
            "state": "CLEANED",
            "restored": True,
            "cleanup_verified": True,
            "timer_disarmed": True,
            "baseline_sha256": BASELINE,
            "intent_sha256": INTENT,
        }
    )
    assert recovery.restore(collector)["state"] == "CLEANED"
    assert recovery.environment_restored is True
    assert not any(host.cleanup().values()), (
        "test_empty_response_and_released_reboot_wait_without_recreating_offline_probe: expected no any(host.cleanup().values())"
    )


@pytest.mark.parametrize("ready", ["True", "Unknown"])
def test_reboot_wait_never_renews_its_deadline_without_a_new_boot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ready: str
) -> None:
    from tests.regional._collector_inventory_reboot_support import InventoryHarness
    from tests.regional.test_collector_inventory_reboot_runner import recovery

    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    subject.reboot_deadline_monotonic = harness.clock.now + 10
    original = subject.reboot_deadline_monotonic
    harness.regional.node_snapshot = lambda *args: {
        "name": "node-a",
        "uid": "node-uid-a",
        "ready": ready,
        "boot_id": "boot-a",
    }
    with pytest.raises(RegionalFixtureError, match="before deadline"):
        subject.wait_reboot()
    assert subject.reboot_deadline_monotonic == original
    assert harness.clock.sleeps == [5, 5]
    with pytest.raises(RegionalFixtureError, match="operator hold"):
        subject.read_for_cleanup()
