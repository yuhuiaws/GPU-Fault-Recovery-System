from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from scripts.e2e.regional import collector_action_guard as guard
from scripts.e2e.regional import collector_negative_evidence as negative
from scripts.e2e.regional import run_collector_acceptance as acceptance
from scripts.e2e.regional.collector_acceptance_fixture import (
    CollectorAcceptanceFixture,
    collector_setting,
)
from scripts.e2e.regional.collector_window_fixture import parse_metric_samples
from scripts.e2e.regional.probes import collector_node_probe as probe
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), 0, -1, True]
)
def test_action_time_refuses_nonfinite_and_nonpositive_bounds(value: float) -> None:
    with pytest.raises(RegionalFixtureError):
        guard.require_action_time(value)


def test_expired_monotonic_window_refuses_mutation_but_allows_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [10.0]
    monkeypatch.setattr(guard.time, "monotonic", lambda: clock[0])
    kubeconfig = tmp_path / "unit-kubeconfig"
    kubeconfig.touch()
    fixture = CollectorAcceptanceFixture(
        SimpleNamespace(
            settings=SimpleNamespace(
                gpu_kubeconfig=kubeconfig,
                gpu_context="gpu",
                namespace="test",
                cluster_id="cluster-a",
            )
        ),
        node="node-a",
        image="image",
        case_id="GF-REGIONAL-COLLECT-002",
        run_id="guard",
        case_dir=tmp_path,
    )
    host = Mock()
    fixture.host = host
    with guard.action_window(datetime.now(timezone.utc) + timedelta(minutes=5)):
        clock[0] += 301
        with pytest.raises(RegionalFixtureError, match="next action"):
            fixture.execute("throttle-gpu")
        fixture.execute("restore-gpu-power-limit", timeout=60)
    host.execute.assert_called_once_with("restore-gpu-power-limit", timeout=60)


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1", "1.5"])
def test_collector_intervals_cannot_disable_or_unbound_the_wait(value: str) -> None:
    with pytest.raises(RegionalFixtureError):
        collector_setting({"INTERVAL": value}, "INTERVAL")


@pytest.mark.parametrize("failure", [RuntimeError("lost ACK"), KeyboardInterrupt()])
def test_seed_is_owned_before_write_ack_and_cleanup_rediscovers_it(
    tmp_path: Path, failure: BaseException
) -> None:
    cleanup = acceptance.CaseCleanup()
    restored = []

    def execute(*args: str, **kwargs: Any) -> None:
        assert cleanup.seed_markers, (
            f"injection {args!r} began before its marker was registered for cleanup"
        )
        raise failure

    fixture = SimpleNamespace(
        node="node-a",
        snapshot=lambda: {"gpu_inventory": [{"pci_bdf": "0000:01:00.0"}]},
        execute=execute,
        store_snapshot=lambda marker: {
            "seed_marker": marker,
            "incidents": [{"incident_id": "incident-a"}],
            "workflows": [],
            "commands": [],
        },
        restore_incidents=lambda state, **kwargs: restored.append(state) or [],
    )
    with pytest.raises(type(failure)) as caught:
        acceptance.run_collect009(fixture, tmp_path, 1, cleanup=cleanup)
    assert caught.value is failure
    result = cleanup.finish(profile_version="profile-a", reason="unit cleanup")
    assert result["errors"] == []
    assert restored[0]["incidents"] == [{"incident_id": "incident-a"}]
    assert cleanup.seed_markers == []


def test_unresolved_seed_lookup_cannot_claim_cleanup() -> None:
    cleanup = acceptance.CaseCleanup()
    fixture = SimpleNamespace(node="node-a", store_snapshot=lambda _: {})
    cleanup.register_seed(fixture, "marker-a")
    result = cleanup.finish(profile_version="profile-a", reason="unit cleanup")
    assert len(result["errors"]) == 1
    assert cleanup.seed_markers == [(fixture, "marker-a")]


def firmware_proof() -> dict[str, Any]:
    fields = {
        "target_present": False,
        "update_command_present": False,
        "verify_command_present": False,
    }
    return {
        "cpu_replicas": [dict(fields)],
        "node_agent": {**fields, "allow_disabled": True},
    }


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("node_agent", "allow_disabled", False),
        ("node_agent", "allow_disabled", "false"),
        ("node_agent", "target_present", True),
        ("node_agent", "update_command_present", None),
        ("node_agent", "verify_command_present", True),
        ("cpu_replicas", "target_present", True),
        ("cpu_replicas", "update_command_present", "false"),
        ("cpu_replicas", "verify_command_present", None),
    ],
)
def test_firmware_negative_refuses_enabled_or_unknown_projection(
    section: str, key: str, value: Any
) -> None:
    proof = firmware_proof()
    target = proof[section][0] if section == "cpu_replicas" else proof[section]
    target[key] = value
    assert negative.firmware_premise_errors(proof), (
        f"firmware-negative premise accepted {section}.{key}={value!r}"
    )


def test_firmware_negative_accepts_only_the_exact_missing_target_owner_block() -> None:
    assert negative.firmware_premise_errors(firmware_proof()) == []
    state = {
        "decisions": [{"official_action": "UPDATE_SWFW", "workflow_request_id": "w"}],
        "workflows": [
            {
                "request_id": "w",
                "status": "BLOCKED",
                "blocked_reasons": sorted(negative.FIRMWARE_REASONS),
            }
        ],
    }
    assert negative.firmware_negative_errors(state) == []
    # The policy engine's own catalog-binding reason rides along with every
    # XID 78 block (live 2026-09-17, catalog 610); it is part of the exact
    # firmware gate, not an unrelated block.
    state["workflows"][0]["blocked_reasons"].append(
        "exact NVIDIA Catalog 610 workflow for XID 78: UPDATE_SWFW"
    )
    assert negative.firmware_negative_errors(state) == []
    state["workflows"][0]["blocked_reasons"].append("node workload state is UNKNOWN")
    assert negative.firmware_negative_errors(state), (
        "an additional UNKNOWN-workload block was accepted as the exact firmware gate"
    )
    only_catalog = {
        "decisions": state["decisions"],
        "workflows": [
            {
                "request_id": "w",
                "status": "BLOCKED",
                "blocked_reasons": [
                    "exact NVIDIA Catalog 610 workflow for XID 78: UPDATE_SWFW"
                ],
            }
        ],
    }
    assert negative.firmware_negative_errors(only_catalog), (
        "the catalog reason alone is not the missing target/owner gate"
    )


def test_firmware_projection_never_exposes_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    process = tmp_path / "123"
    process.mkdir()
    (process / "stat").write_text("123 (node-agent) " + " ".join(["0"] * 20))
    (process / "environ").write_bytes(
        b"GPU_FAULT_NODE_ALLOW_FIRMWARE_UPDATE=false\0"
        b"GPU_FAULT_CONTROL_PLANE_TOKEN=unit-test-only-do-not-emit\0"
    )
    monkeypatch.setattr(probe, "PROC_ROOT", tmp_path)
    monkeypatch.setattr(
        probe,
        "service_snapshot",
        lambda: {
            "gpu-fault-node-agent.service": {
                "MainPID": "123",
                "ActiveState": "active",
                "InvocationID": "invocation-a",
            }
        },
    )
    probe.firmware_premise(SimpleNamespace())
    value = json.loads(capsys.readouterr().out)
    assert value == {
        "pid": 123,
        "start_ticks": "0",
        "invocation_id": "invocation-a",
        "allow_disabled": True,
        "target_present": False,
        "update_command_present": False,
        "verify_command_present": False,
    }


def test_failed_firmware_premise_never_submits_xid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = SimpleNamespace(
        snapshot=lambda: {},
        regional=object(),
        node="node-a",
        execute=Mock(return_value={}),
    )
    monkeypatch.setattr(
        acceptance,
        "firmware_premise",
        Mock(side_effect=RegionalFixtureError("firmware configured")),
    )
    with pytest.raises(RegionalFixtureError, match="configured"):
        acceptance.run_collect010(fixture, tmp_path, 1, "profile-a")
    fixture.execute.assert_called_once_with("firmware-premise")


@pytest.mark.parametrize("sample", ["NaN", "+Inf", "-Inf"])
def test_nonfinite_metrics_cannot_satisfy_a_collector_verdict(sample: str) -> None:
    with pytest.raises(RegionalFixtureError, match="finite"):
        parse_metric_samples(f"collector_errors {sample}", "collector_errors")


class FirmwareApi:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.missing_replica = False
        self.mode = "OBSERVE"
        self.changed = False

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "profile": {
                "profile_version": "profile-a",
                "warnings": [],
                "capabilities": [
                    {"capability": "softwareFirmwareUpdate", "mode": self.mode}
                ],
            }
        }

    def ready_pods(self, plane: str, app: str) -> list[dict[str, str]]:
        assert plane == "cpu"
        return [
            {"name": f"{app}-{index}", "uid": f"{app}-uid-{index}"}
            for index in range(1 if self.missing_replica else 2)
        ]

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "cpu"
        if args[0] == "exec":
            assert (
                args[args.index("--") + 1] == "/opt/gpu-fault/control-plane/bin/python"
            )
            self.calls.append(args[2])
            return json.dumps(firmware_proof()["cpu_replicas"][0])
        assert args[:2] == ("get", "deployment")
        return json.dumps(
            {
                "metadata": {
                    "uid": "deployment-a",
                    "generation": 2 if self.changed and self.calls else 1,
                },
                "spec": {"replicas": 2},
            }
        )


def test_firmware_premise_checks_every_ready_replica() -> None:
    api = FirmwareApi()
    result = negative.firmware_premise(
        api, firmware_proof()["node_agent"], node="node-a"
    )
    assert len(result["cpu_replicas"]) == 4
    assert len(api.calls) == 4
    assert result["firmware_mode"] == "OBSERVE"


@pytest.mark.parametrize("change", ["missing_replica", "changed", "mode"])
def test_firmware_population_or_profile_uncertainty_refuses(change: str) -> None:
    api = FirmwareApi()
    setattr(api, change, "OWN" if change == "mode" else True)
    with pytest.raises(RegionalFixtureError):
        negative.firmware_premise(api, firmware_proof()["node_agent"], node="node-a")
