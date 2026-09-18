from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from scripts.e2e.regional import collector_acceptance_fixture as fixtures
from scripts.e2e.regional import run_collect016_training_recovery as training
from scripts.e2e.regional import run_collect017_efa_plugin as efa
from scripts.e2e.regional import run_collector_acceptance as acceptance
from scripts.e2e.regional import run_collector_destructive as destructive
from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional.test_collector_power_safety import PowerHost
from tests.regional.test_collector_power_safety import power_host as power_host_fixture

power_host = power_host_fixture


class Abort(BaseException):
    pass


@pytest.mark.parametrize("failure", [RuntimeError("partial throttle"), Abort()])
def test_partial_throttle_always_attempts_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    now = datetime.now(timezone.utc)
    stamps = iter([(now - timedelta(seconds=60)).isoformat(), now.isoformat()])
    monkeypatch.setattr(acceptance, "gpu_metrics_stamp", lambda _: next(stamps))
    monkeypatch.setattr(acceptance, "seconds_until_next_summary", lambda *a, **k: 0)
    calls: list[str] = []

    def execute(command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(command)
        if command == "throttle-gpu":
            raise failure
        return {
            "restored": True,
            "load_stopped": True,
            "timer_disarmed": True,
            "cleanup_verified": True,
        }

    fixture = SimpleNamespace(
        snapshot=lambda: {
            "collector_env": {
                "GPU_FAULT_METRICS_INTERVAL_SECONDS": "5",
                "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "60",
                "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "2",
            },
            "gpu_power": [{"index": 0}],
        },
        execute=execute,
    )
    with pytest.raises(type(failure)) as caught:
        acceptance.run_collect002(fixture, tmp_path, 1)
    assert caught.value is failure
    assert calls == ["throttle-gpu", "restore-gpu-power-limit"]


@pytest.mark.parametrize("failure", [RuntimeError("sampling failed"), Abort()])
def test_failed_inventory_sampling_has_nothing_to_restore_or_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    from tests.regional._collector_inventory_reboot_support import InventoryHarness

    harness = InventoryHarness(monkeypatch, tmp_path)
    harness.failure = failure
    if isinstance(failure, Exception):
        result = harness.run()
        assert result["verdict"] == "FAIL"
        assert result["errors"] == [
            f"inventory reboot failed: {type(failure).__name__}: {failure}"
        ]
    else:
        with pytest.raises(type(failure)) as caught:
            harness.run()
        assert caught.value is failure
    result = json.loads((tmp_path / "inventory-reboot-progress-a1.json").read_text())
    assert result["verdict"] == "FAIL"
    assert type(failure).__name__ in result["errors"][0]
    assert result["cleanup_errors"] == []
    assert result["production_configuration_modified"] is False
    assert result["publication_started"] is False
    assert harness.calls[-1] == "sample-gpu-inventory"
    assert "override-expected-gpu-count" not in harness.calls
    assert "restore-collector-env" not in harness.calls
    assert "publish-gpu-inventory" not in harness.calls
    assert harness.restores == 0
    operation = json.loads((tmp_path / "inventory-operation-a1.json").read_text())
    assert operation["run_id"] == harness.run_id == result["run_id"]
    assert len(operation["owner_nonce"]) == 32
    assert len(harness.cleanup.state_readers) == 0
    assert harness.cleanup.finish(profile_version="v1", reason="test")["errors"] == []
    assert "incident-cleanup" not in harness.calls


@pytest.mark.parametrize("failure", [RuntimeError("partial unbind"), Abort()])
def test_partial_efa_unbind_is_restored(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    monkeypatch.setattr(efa, "_base_settings", lambda *args: None)
    calls: list[str] = []

    def execute(command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(command)
        if command == "unbind-efa":
            raise failure
        return {"bound": True}

    collector = SimpleNamespace(
        snapshot=lambda: {
            "efa_inventory": {"devices": [{"pci_bdf": "0000:01:00.0", "driver": "efa"}]}
        },
        execute=execute,
    )
    with pytest.raises(type(failure)) as caught:
        efa.run_efa_unbind(SimpleNamespace(node="node-a"), object(), collector, 1)
    assert caught.value is failure
    assert calls == ["unbind-efa", "restore-efa"]


@pytest.mark.parametrize("module", [acceptance, destructive])
def test_failed_final_blast_read_cannot_leave_a_pass(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = (
        "GF-REGIONAL-COLLECT-003" if module is acceptance else "GF-REGIONAL-COLLECT-004"
    )
    regional = SimpleNamespace(
        evidence_identity=lambda: {
            "release_id": "release-a",
            "cluster_id": "cluster-a",
        },
        cpu_blast_snapshot=Mock(side_effect=RuntimeError("blast read failed")),
    )
    probe = SimpleNamespace(node="node-a", create=Mock(), cleanup=Mock(return_value={}))
    monkeypatch.setattr(module, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(module, "CollectorAcceptanceFixture", lambda *a, **k: probe)
    monkeypatch.setattr(
        module,
        "read_only_preflight",
        lambda *a, **k: {
            "errors": [],
            "stores": [{"profile": {"profile_version": "profile-a"}}],
            "store": {"profile": {"profile_version": "profile-a"}},
            "cpu_blast": {},
            "reboot_scope": {},
        },
    )
    if module is destructive:
        monkeypatch.setattr(module, "reset_fixture", lambda *a, **k: probe)
    monkeypatch.setattr(
        module,
        "run_collect003" if module is acceptance else "run_collect004",
        lambda *a, **k: {"verdict": "PASS", "errors": []},
    )
    settings = SimpleNamespace(
        regional=object(),
        case_id=case_id,
        node="node-a",
        nodes=("node-a",),
        host_probe_image="image",
    )
    status = module.execute_case(
        settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(minutes=5)
    )
    result = json.loads((tmp_path / "cases" / case_id / f"{case_id}.json").read_text())
    assert status == 1
    assert result["verdict"] == "FAIL"
    assert "blast read failed" in result["error"]
    probe.cleanup.assert_called_once()


def test_training_does_not_reset_after_a_failed_restart_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = SimpleNamespace(evidence_identity=lambda: {})
    prewarm = SimpleNamespace(create=Mock(), cleanup=Mock(return_value={}))
    monkeypatch.setattr(training, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(training, "ImagePrewarmFixture", lambda *a, **k: prewarm)
    monkeypatch.setattr(
        training,
        "read_only_preflight",
        lambda *a: {"errors": [], "candidate_nodes": [{"name": "node-a"}]},
    )
    monkeypatch.setattr(
        training,
        "run_restart_budget_sections",
        lambda *a, **k: {"errors": ["wrong attempt"], "a": {}, "b": {}},
    )
    reset = Mock(side_effect=AssertionError("reset must not start"))
    monkeypatch.setattr(training, "run_reset_section", reset)
    assert (
        training.execute_case(
            SimpleNamespace(regional=object()),
            tmp_path,
            1,
            datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        == 1
    )
    reset.assert_not_called()
    prewarm.cleanup.assert_called_once()


@pytest.mark.parametrize("failed_phase", ["a", "b"])
def test_plugin_case_stops_after_a_failed_phase(
    failed_phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = SimpleNamespace(evidence_identity=lambda: {})
    collector = SimpleNamespace(create=Mock(), cleanup=Mock(return_value={}))
    monkeypatch.setattr(efa, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(efa, "CollectorAcceptanceFixture", lambda *a, **k: collector)
    monkeypatch.setattr(efa, "read_only_preflight", lambda *a: {"errors": []})
    a = Mock(return_value={"errors": ["unbind failed"] if failed_phase == "a" else []})
    b = Mock(return_value={"errors": ["plugin failed"]})
    c = Mock(side_effect=AssertionError("training mutation must not start"))
    monkeypatch.setattr(efa, "run_efa_unbind", a)
    monkeypatch.setattr(efa, "run_gpu_plugin", b)
    monkeypatch.setattr(efa, "run_training_plugin", c)
    assert (
        efa.execute_case(
            SimpleNamespace(regional=object(), node="node-a", host_probe_image="image"),
            tmp_path,
            1,
            datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        == 1
    )
    assert b.call_count == (0 if failed_phase == "a" else 1)
    c.assert_not_called()
    collector.cleanup.assert_called_once()


def test_failed_annotation_delete_stays_registered() -> None:
    def kubectl(*args: str, check: bool = True, **kwargs: Any) -> str:
        if check:
            raise RuntimeError("delete refused")
        return ""

    fixture = SimpleNamespace(node="node-a", regional=SimpleNamespace(kubectl=kubectl))
    cleanup = acceptance.CaseCleanup()
    cleanup.register_annotation(fixture, "gpu-fault.io/mechanical-inspection-complete")
    result = cleanup.finish(profile_version="profile-a", reason="test cleanup")
    assert len(result["errors"]) == 1
    assert "delete refused" in result["errors"][0]
    assert len(cleanup.annotations) == 1


def test_restore_rejects_a_node_left_cordoned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fixtures, "WarmSpareLiveFixture", lambda *a: object())
    fixture = SimpleNamespace(
        regional=object(),
        node="node-a",
        residual_isolation=lambda: {
            "ownership_annotations": {},
            "taints": [],
            "unschedulable": True,
        },
    )
    with pytest.raises(fixtures.RegionalFixtureError, match="still isolated"):
        fixtures.CollectorAcceptanceFixture.restore_incidents(
            fixture,
            {"incidents": [], "workflows": [], "commands": []},
            profile_version="profile-a",
            reason="test",
        )


@pytest.mark.parametrize("failure_at", ["query", "set", "readback"])
def test_power_restore_keeps_deadman_until_all_gpus_are_verified(
    failure_at: str, power_host: PowerHost
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    host.calls.clear()
    if failure_at == "query":
        host.power_returncode = 1
    elif failure_at == "set":

        def refuse(command: list[str]) -> None:
            if "-pl" in command:
                raise probe.ProbeError("power restore refused")

        host.before_call = refuse
    else:
        host.ignore_writes = True
    with pytest.raises(probe.ProbeError):
        probe.restore_gpu_power_limit(host.arguments())
    assert not any(
        item[:2] == ["systemctl", "disable"] and item[-1].endswith(".timer")
        for item in host.calls
    ), f"power restore failure at {failure_at} disarmed its recovery timer"
    assert host.units()[0] in host.active


def test_power_throttle_refuses_a_custom_baseline_before_mutation(
    power_host: PowerHost,
) -> None:
    power_host.gpus[0]["power_limit_w"] = 500
    with pytest.raises(probe.ProbeError, match="nondefault"):
        probe.throttle_gpu(power_host.arguments())
    assert not power_host.power_writes(), (
        "test_power_throttle_refuses_a_custom_baseline_before_mutation: expected no power_host.power_writes()"
    )
    assert not (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists(), (
        'test_power_throttle_refuses_a_custom_baseline_before_mutation: expected no (probe.ACCEPTANCE_STATE / "gpu-power-owner.json").exists()'
    )


@pytest.mark.parametrize(
    "case_id", ["GF-REGIONAL-COLLECT-002", "GF-REGIONAL-COLLECT-012"]
)
def test_mutating_acceptance_preflight_rejects_a_workload(
    case_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = SimpleNamespace(
        node_snapshot=lambda node: {
            "name": node,
            "ready": "True",
            "ownership_annotations": {},
            "unschedulable": False,
            "taints": [],
        },
        store_snapshot=lambda node: {"agent": {"lifecycle_state": "ACTIVE"}},
        evidence_identity=lambda: {
            "release_id": "release-a",
            "cluster_id": "cluster-a",
        },
        business_workloads=lambda node: [{"name": "active-workload"}],
        cpu_blast_snapshot=lambda: {},
    )
    monkeypatch.setattr(acceptance, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(
        acceptance, "predecessor_evidence", lambda *a, **k: {"valid": True}
    )
    monkeypatch.setattr(acceptance, "focused_tests", lambda *a, **k: {"passed": True})
    preflight = acceptance.read_only_preflight(
        SimpleNamespace(
            regional=object(),
            nodes=("node-a",),
            case_id=case_id,
            predecessor_path=tmp_path / "unused.json",
        ),
        tmp_path,
    )
    assert preflight["errors"] == ["node-a has a non-system workload"]


def test_collect009_configure_uses_the_formal_non_pytest_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acceptance, "settings_from_arguments", lambda _: object())
    arguments = acceptance.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--case",
            "GF-REGIONAL-COLLECT-009",
            "--node",
            "node-a",
            "--host-probe-image",
            "test-image",
        ]
    )
    settings = acceptance.configure(arguments)
    assert settings.predecessor_path == (
        tmp_path / "cases/GF-REGIONAL-COLLECT-005/GF-REGIONAL-COLLECT-005.json"
    )
