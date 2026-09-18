"""Recovery ordering and provider evidence using fake host and store transports."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import collector_inventory_reboot as inventory_reboot
from scripts.e2e.regional import run_collector_destructive as runner
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeError,
    HostProbeTransportError,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._cov95_collect_net import Clock, no_external_effects  # noqa: F401

ORIGIN = datetime(2026, 9, 1, tzinfo=timezone.utc)
ROLE = "arn:aws:iam::123456789012:role/Executor"


def settings(case_id: str = "GF-REGIONAL-COLLECT-004") -> Any:
    return SimpleNamespace(
        node="node-a",
        case_id=case_id,
        regional=SimpleNamespace(cluster_id="cluster-a"),
        debounce_tolerance=0.5,
        executor_role_arn=ROLE,
        hyperpod_cluster="fixture",
    )


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "same-boot",
        "extra-workflow",
        "missing-restart",
        "failed-workflow",
        "digest",
        "reboot-count",
        "actor",
        "publication",
        "response-lost",
        "publication-unresolved",
    ],
)
def test_isolated_inventory_reboot_keeps_strict_provider_proof_without_env_restore(
    monkeypatch: Any, tmp_path: Path, problem: str | None
) -> None:
    from scripts.e2e.regional import collector_inventory_reboot as inventory
    from scripts.e2e.regional import collector_reboot_evidence as provider
    from tests.regional._collector_inventory_reboot_support import (
        ORIGIN as inventory_origin,
    )
    from tests.regional._collector_inventory_reboot_support import InventoryHarness
    from tests.regional.test_collector_reboot_evidence import inputs as provider_inputs

    harness = InventoryHarness(monkeypatch, tmp_path)
    harness.problem = problem if problem in {"same-boot", "digest"} else None
    bound = provider_inputs()
    harness.scope.update(
        {
            field: bound["scope"][field]
            for field in (
                "eks_cluster_arn",
                "region",
                "account_id",
                "node_logical_id",
                "provider_inventory",
                "node_recovery",
                "cluster_status",
            )
        }
    )
    harness.scope["cluster_id"] = harness.settings.regional.cluster_id

    def stamp(seconds: float) -> str:
        return (inventory_origin + timedelta(seconds=seconds)).isoformat()

    workflow = bound["workflow"]
    workflow.update(
        incident_id="incident-a", created_at=stamp(31), updated_at=stamp(60)
    )
    workflow["official_steps"][0]["node_ids"] = [harness.settings.node]
    workflow["step_executions"][0]["updated_at"] = stamp(60)
    command = bound["commands"][0]
    command.update(
        cluster_id=harness.settings.regional.cluster_id,
        incident_id="incident-a",
        created_at=stamp(32),
        updated_at=stamp(60),
    )
    command["step"] = deepcopy(workflow["official_steps"][0])
    submission = bound["submission"]
    submission.update(
        cluster_name=harness.settings.hyperpod_cluster,
        requested_node_identifiers=[harness.settings.node],
        created_at=stamp(33.1),
        updated_at=stamp(34.2),
    )
    submission["result"]["cluster_name"] = harness.settings.hyperpod_cluster
    event = bound["events"][0]
    event.update(
        event_time=stamp(34),
        session_issuer_role_name="Other" if problem == "actor" else "Executor",
        session_issuer_arn=ROLE.replace("Executor", "Other")
        if problem == "actor"
        else ROLE,
        identity_arn="arn:aws:sts::123456789012:assumed-role/Executor/fixture",
        request_cluster_name=harness.settings.hyperpod_cluster,
    )
    original_state = harness.state
    original_execute = harness.execute
    original_cpu = harness.cpu
    original_node_snapshot = harness.regional.node_snapshot
    monkeypatch.setattr(
        inventory,
        "datetime",
        SimpleNamespace(
            now=lambda tz=None: inventory_origin
            + timedelta(seconds=harness.clock.now - 1000)
        ),
    )

    def state() -> dict[str, Any]:
        value = original_state()
        current = deepcopy(workflow)
        commands = deepcopy(bound["commands"])
        if not harness.rebooted:
            current["status"] = "RUNNING"
            current["updated_at"] = stamp(35)
            current["completed_step_indexes"] = []
            current["step_executions"][0].update(status="WAITING", updated_at=stamp(35))
            commands[0].update(status="WAITING", updated_at=stamp(35))
        if harness.rebooted and problem == "failed-workflow":
            current["status"] = "FAILED"
        if harness.rebooted and problem == "missing-restart":
            current["official_steps"] = []
        value.update(
            workflows=[current], commands=commands, submissions=[deepcopy(submission)]
        )
        if harness.rebooted and problem == "extra-workflow":
            value["workflows"].append({**current, "request_id": "workflow-b"})
        value["node_workflow_ids"] = [row["request_id"] for row in value["workflows"]]
        return value

    def host(verb: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        if verb == "publish-gpu-inventory" and problem == "publication":
            raise HostProbeError("inventory publication guard rejected")
        receipt = original_execute(verb, *args, **kwargs)
        if verb == "publish-gpu-inventory" and problem in {
            "response-lost",
            "publication-unresolved",
        }:
            raise HostProbeTransportError("owned probe unavailable")
        return receipt

    def cpu(script: str, *args: str) -> dict[str, Any]:
        if (
            problem == "publication-unresolved"
            and script == inventory.INVENTORY_RECOVERY_STATE
        ):
            raise RegionalFixtureError("publication outcome remains unknown")
        value = original_cpu(script, *args)
        if (
            problem == "publication-unresolved"
            and script == inventory.HOST_INVENTORY_EVIDENCE
        ):
            value["records"] = []
        return value

    def events(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        harness.calls.append("provider-events")
        return [] if problem == "reboot-count" else [deepcopy(event)]

    def node_snapshot(node: str) -> dict[str, Any]:
        assert node == harness.settings.node
        current: dict[str, Any] = original_node_snapshot(node)
        return current

    monkeypatch.setattr(harness, "state", state)
    monkeypatch.setattr(harness.collector, "execute", host)
    monkeypatch.setattr(harness.regional, "cpu_python", cpu)
    monkeypatch.setattr(harness.regional, "provider_events", events)
    monkeypatch.setattr(harness.regional, "node_snapshot", node_snapshot)
    monkeypatch.setattr(inventory, "prove_reboot_scope", provider.prove_reboot_scope)
    monkeypatch.setattr(
        inventory, "submitted_reboot_errors", provider.submitted_reboot_errors
    )
    result = harness.run()
    calls = harness.calls
    assert result["verdict"] == (
        "PASS" if problem in {None, "response-lost"} else "FAIL"
    ), result
    if problem in {"same-boot", "publication", "publication-unresolved"}:
        assert "recreate" not in calls, "hard guards stop before post-reboot host reads"
    else:
        assert "recreate" in calls, "host reads after reboot must use a recreated probe"
    if problem == "publication":
        assert "node-ready" not in calls
        assert any(
            "publication guard rejected" in error for error in result["errors"]
        ), (
            'test_isolated_inventory_reboot_keeps_strict_provider_proof_without_env_restore: expected any("publication guard rejected" in error for er...'
        )
    else:
        assert calls.count("publish-gpu-inventory") == 1
    assert "restore-collector-env" not in calls
    assert "override-expected-gpu-count" not in calls
    if problem in {"response-lost", "publication-unresolved"}:
        assert result["publication"]["response_unknown"] is True
        assert result["publication"]["replayed"] is False
    if problem in {"actor", "reboot-count"}:
        message = (
            "exact approved executor role"
            if problem == "actor"
            else "exactly one provider mutation"
        )
        assert any(message in error for error in result["provider_proof"]["errors"]), (
            'test_isolated_inventory_reboot_keeps_strict_provider_proof_without_env_restore: expected any(message in error for error in result["provid...'
        )
    if problem in {None, "response-lost"}:
        assert result["provider_proof"]["valid"] is True
        assert result["production_configuration_modified"] is False
        assert result["isolated_sampling"]["live_delivery_proven"] is True
    stored = json.loads((tmp_path / "inventory-reboot-progress-a1.json").read_text())
    assert stored["verdict"] == result["verdict"]
    assert stored["errors"] == result["errors"]
    assert stored["cleanup_errors"] == result["cleanup_errors"]
    if problem == "publication-unresolved":
        released = harness.cleanup.finish(profile_version="v1", reason="test")
        assert any(
            "outcome remains unknown" in error for error in released["errors"]
        ), (
            'test_isolated_inventory_reboot_keeps_strict_provider_proof_without_env_restore: expected any("outcome remains unknown" in error for error...'
        )
        assert "incident-cleanup" not in calls


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "same-boot",
        "provider-identity",
        "reboot-count",
        "actor",
        "replacement",
        "missing-workflow",
        "failed-workflow",
        "missing-submission",
    ],
)
def test_fatal_sxid_reboot_verifies_provider_identity_actor_and_no_replace(
    monkeypatch: Any, tmp_path: Path, problem: str | None
) -> None:
    calls = []
    workflow = {
        "request_id": "workflow-a",
        "status": "FAILED" if problem == "failed-workflow" else "SUCCEEDED",
        "official_steps": [{"operation": "RESTART_NODE"}],
        "step_executions": [{"operation": "RESTART_NODE", "status": "SUCCEEDED"}],
    }

    class Provider:
        reads = 0

        def __init__(self, regional: Any, hyperpod_cluster: str) -> None:
            assert hyperpod_cluster == "fixture"

        def provider_inventory(self) -> dict[str, str]:
            self.reads += 1
            return {
                "node-a": "other"
                if problem == "provider-identity" and self.reads > 1
                else "instance-a"
            }

        def cluster_recovery(self) -> dict[str, Any]:
            return {"node_recovery": "None"}

    monkeypatch.setattr(runner, "WarmSpareLiveFixture", Provider)
    regional = SimpleNamespace(
        node_snapshot=lambda node: {"boot_id": "boot-a", "unschedulable": False},
        wait_node_ready=lambda *a, **k: {
            "boot_id": "boot-a" if problem == "same-boot" else "boot-b"
        },
        cpu_python=lambda *a: {
            "submission": None
            if problem == "missing-submission"
            else {"state": "SUBMITTED"}
        },
        executor_python=lambda *a: {"GPU_FAULT_ALLOW_HYPERPOD_REPLACE": "false"},
        wait_provider_events=lambda *a, **k: []
        if problem == "reboot-count"
        else [
            {
                "event_name": "BatchRebootClusterNodes",
                "session_issuer_role_name": "Other"
                if problem == "actor"
                else "Executor",
            }
        ],
        provider_events=lambda *a: [{"event_name": "ReplaceClusterNodes"}]
        if problem == "replacement"
        else [],
        provider_events_provisional=lambda ended: True,
    )
    collector = SimpleNamespace(
        snapshot=lambda: {"gpu_inventory": [{"pci_bdf": "0000:af:00.0"}]},
        execute=lambda verb, *args, **kwargs: calls.append(verb) or {},
        wait_marker=lambda *a, **k: {
            "workflows": [] if problem == "missing-workflow" else [workflow]
        },
    )
    cleanup = SimpleNamespace(register_seed=lambda *a: calls.append("seed"))
    result = runner.run_collect015(
        settings("GF-REGIONAL-COLLECT-015"),
        regional,
        collector,
        tmp_path,
        1,
        cleanup=cleanup,
    )
    assert result["verdict"] == ("PASS" if problem is None else "FAIL")
    assert calls == ["seed", "append-sxid"], "record seed ownership before injection"
    assert result["cluster_recovery"]["node_recovery"] == "None"
    assert result["executor_flags"]["GPU_FAULT_ALLOW_HYPERPOD_REPLACE"] == "false"


@pytest.mark.parametrize("second_fails", [False, True])
def test_restore_defers_until_an_explicit_post_reboot_retry(second_fails: bool) -> None:
    calls = []
    owner_nonce = "b" * 32
    arguments = (
        "restore-collector-env",
        "--run-id",
        "run-a",
        "--owner-nonce",
        owner_nonce,
    )

    def execute(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert args == arguments
        assert kwargs == {"timeout": 300}
        calls.append("execute")
        if calls.count("execute") == 1:
            raise HostProbeTransportError("original probe unavailable")
        if second_fails:
            raise HostProbeTransportError("replacement probe unavailable")
        return {"restored": True}

    def reboot_transition() -> bool:
        calls.append("reboot-binding")
        return True

    collector = SimpleNamespace(
        execute=execute, recreate=lambda: calls.append("recreate")
    )
    result = runner.restore_collector_env(
        collector, "run-a", owner_nonce=owner_nonce, reboot_transition=reboot_transition
    )
    assert result["restored"] is False and result["deferred"] is True
    assert result["error_type"] == "HostProbeTransportError"
    assert "original probe unavailable" not in json.dumps(result)
    assert owner_nonce not in json.dumps(result)
    assert calls == ["execute", "reboot-binding"], "do not recreate an offline probe"
    collector.recreate()
    if second_fails:
        with pytest.raises(HostProbeTransportError):
            runner.restore_collector_env(collector, "run-a", owner_nonce=owner_nonce)
    else:
        assert runner.restore_collector_env(
            collector, "run-a", owner_nonce=owner_nonce
        ) == {"restored": True}
    assert calls == ["execute", "reboot-binding", "recreate", "execute"]


@pytest.mark.parametrize(
    "incident_id,owned", [("", False), ("incident-a", False), ("incident-a", True)]
)
def test_incident_restore_requires_explicit_incident_and_current_ownership(
    monkeypatch: Any, incident_id: str, owned: bool
) -> None:
    calls = []
    regional = SimpleNamespace(
        node_snapshot=lambda node: {
            "ownership_annotations": {"owner": "incident-a"} if owned else {}
        }
    )

    class Restore:
        def __init__(self, instance: Any, cluster: str) -> None:
            assert instance is regional

        def create_restore_workflow(self, **kwargs: Any) -> dict[str, str]:
            calls.append(kwargs)
            return {"workflow_request_id": "restore-a"}

        def wait_workflow_id(self, request_id: str) -> dict[str, str]:
            return {"status": "SUCCEEDED", "request_id": request_id}

    monkeypatch.setattr(runner, "WarmSpareLiveFixture", Restore)
    result = runner.restore_incident(
        regional,
        settings(),
        incident_id=incident_id,
        profile_version="profile-a",
        reason="fixture",
    )
    assert result == (
        {"status": "SUCCEEDED", "request_id": "restore-a"}
        if incident_id and owned
        else None
    )
    assert len(calls) == int(bool(incident_id) and owned)


@pytest.mark.parametrize(
    "planned,converges", [(False, False), (False, True), (True, False), (True, True)]
)
def test_workflow_pollers_keep_waiting_until_the_requested_proof_or_timeout(
    monkeypatch: Any, planned: bool, converges: bool
) -> None:
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    reads = []

    def cpu(*args: Any) -> dict[str, Any]:
        reads.append(args)
        if len(reads) == 1:
            return {
                "matches": [{"workflow": {"status": "RUNNING", "official_steps": []}}]
            }
        if not converges:
            return {"matches": []}
        return {
            "matches": [
                {
                    "workflow": {
                        "request_id": "workflow-a",
                        "status": "SUCCEEDED",
                        "official_steps": [{"operation": "RESTART_NODE"}],
                    }
                }
            ]
        }

    regional = SimpleNamespace(cpu_python=cpu)

    def invoke() -> dict[str, Any]:
        if planned:
            return runner.wait_planned_workflow(
                regional,
                settings(),
                observed_after=ORIGIN,
                timeout_seconds=10,
                operation="RESTART_NODE",
            )
        return runner.latest_node_workflow(
            regional, settings(), observed_after=ORIGIN, timeout_seconds=10
        )

    if converges:
        result = invoke()
        assert (result if planned else result["workflow"])["request_id"] == "workflow-a"
    else:
        with pytest.raises(runner.RegionalFixtureError):
            invoke()
    assert len(reads) == 2


def test_mismatch_poll_ignores_other_samples_and_checks_declared_confirmation_count(
    monkeypatch: Any,
) -> None:
    clock = Clock()
    monkeypatch.setattr(inventory_reboot, "time", clock)
    finding = {
        "observed_at": (ORIGIN + timedelta(seconds=30)).isoformat(),
        "samples": [
            {"name": "other", "value": 1},
            {"name": "gpu_inventory_mismatch", "value": 0},
            {
                "name": "gpu_inventory_mismatch",
                "value": 1,
                "labels": {
                    "consecutive_mismatch_samples": "2",
                    "required_consecutive_samples": "3",
                    "expected_count": "9",
                    "observed_count": "8",
                },
            },
        ],
    }
    reads = iter([{"records": []}, {"records": [finding]}])
    regional = SimpleNamespace(cpu_python=lambda *a: deepcopy(next(reads)))
    assert runner.wait_mismatch_finding(
        regional, settings(), observed_after=ORIGIN, timeout_seconds=10
    ) == [finding]
    assert clock.sleeps == [5]
    errors = runner.debounce_errors(
        [finding], interval=15, required_samples=2, started_at=ORIGIN, tolerance=0.5
    )
    assert any("collector reports" in error for error in errors), (
        "conflicting confirmation count must fail"
    )
