from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import warm_spare_fixture as warm
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings
from tests.regional.test_warm_spare_fixture_safety import NodeApi


class Transport:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.settings = regional_settings(tmp_path)
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.responses: dict[str, list[Any]] = {}
        self.errors: set[str] = set()
        self.node = {
            "metadata": {
                "name": "spare-a",
                "uid": "uid-a",
                "resourceVersion": "1",
                "labels": {warm.SPARE_LABEL: "true"},
                "annotations": {},
            },
            "spec": {"unschedulable": True},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]},
        }
        self.environment: Any = {
            "spare_failover": "true",
            "remote_state": "true",
            "allow_replace": "false",
            "spare_label": None,
        }
        self.gate: Any = True
        self.pod_present = False
        monkeypatch.setattr(warm, "time", self.clock)
        monkeypatch.setattr(warm, "datetime", self.clock)

    def cpu_python(self, script: str, *args: str, **kwargs: Any) -> Any:
        self.calls.append(("cpu", {"script": script, "args": args, **kwargs}))
        values = self.responses.get(script, [{}])
        value = values.pop(0) if len(values) > 1 else values[0]
        if isinstance(value, BaseException):
            raise value
        return deepcopy(value)

    def executor_python(self, script: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("executor", {"script": script, "args": args, **kwargs}))
        return {"released": True}

    def ready_pods(self, plane: str, app: str) -> list[dict[str, str]]:
        return [{"name": f"{plane}-pod"}]

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.calls.append(("kubectl", {"plane": plane, "args": args, **kwargs}))
        key = " ".join(args[:2])
        if key in self.errors:
            if kwargs.get("check", True):
                raise RegionalFixtureError(f"fake {key} failed")
            return ""
        if args[:2] == ("get", "node"):
            if "-l" in args:
                return json.dumps({"items": [self.node]})
            return json.dumps(self.node)
        if args[0] == "exec":
            return json.dumps(
                self.environment if plane == "gpu" else {"enabled": self.gate}
            )
        if args[0] == "apply":
            self.pod_present = True
            return ""
        if args[0] == "wait":
            return ""
        if args[:2] == ("delete", "pod"):
            self.pod_present = False
            return ""
        if args[:2] == ("get", "pod"):
            return "pod/owned" if self.pod_present else ""
        raise AssertionError(f"unexpected fake request: {args}")

    def run(self, command: list[str], **kwargs: Any) -> Any:
        self.calls.append(("provider", {"command": command, **kwargs}))
        if "list-cluster-nodes" in command:
            value = {
                "ClusterNodeSummaries": [
                    {"NodeLogicalId": "b", "InstanceId": "i-b"},
                    {"NodeLogicalId": "a", "InstanceId": "i-a"},
                ]
            }
        else:
            value = {
                "ClusterName": "fake-hp",
                "ClusterStatus": "InService",
                "NodeRecovery": "None",
            }
        return SimpleNamespace(stdout=json.dumps(value))


@pytest.mark.parametrize("failure", ["read-error", "missing-condition"])
def test_negative_node_readiness_never_invents_unknown_from_failed_or_missing_read(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(tmp_path, monkeypatch)
    fixture = warm.WarmSpareLiveFixture(transport, "fake-hp")
    if failure == "read-error":
        transport.errors.add("get node")
    else:
        transport.node["status"]["conditions"] = []
    with pytest.raises(RegionalFixtureError, match="did not reach"):
        fixture.wait_node_ready("spare-a", ready=False, timeout_seconds=10)
    assert transport.clock.elapsed == 10, transport.clock.elapsed


@pytest.mark.parametrize("value", [{}, {"ready": None}, {"ready": 0}])
def test_negative_fleet_readiness_requires_an_explicit_boolean(
    value: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(tmp_path, monkeypatch)
    transport.responses[warm.FLEET_READINESS] = [value]
    fixture = warm.WarmSpareLiveFixture(transport, "fake-hp")
    with pytest.raises(RegionalFixtureError, match="fleet readiness"):
        fixture.wait_fleet_readiness("spare-a", ready=False, timeout_seconds=10)


def test_agent_wait_refuses_ambiguous_duplicate_active_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(tmp_path, monkeypatch)
    transport.responses[warm.STORE_PROBE] = [
        {
            "agents": [
                {"node_id": "spare-a", "lifecycle_state": "ACTIVE"},
                {"node_id": "spare-a", "lifecycle_state": "ACTIVE"},
            ]
        }
    ]
    fixture = warm.WarmSpareLiveFixture(transport, "fake-hp")
    with pytest.raises(RegionalFixtureError, match="agent did not return ACTIVE"):
        fixture.wait_agent_active("spare-a", timeout_seconds=10)


@pytest.mark.parametrize("failed", ["get pod", "delete pod"])
def test_holder_cleanup_requires_successful_delete_and_absence_read(
    failed: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(tmp_path, monkeypatch)
    transport.errors.add(failed)
    holder = warm.GpuHolderFixture(
        warm.WarmSpareLiveFixture(transport, "fake-hp"), node="spare-a", run_id="owned"
    )
    with pytest.raises(RegionalFixtureError, match="fake"):
        holder.cleanup()


def test_warm_fixture_reads_scoped_inventory_and_environment_through_fake_transports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(tmp_path, monkeypatch)
    fixture = warm.WarmSpareLiveFixture(transport, "fake-hp")
    inventory = fixture.provider_inventory()
    assert [node["node_logical_id"] for node in inventory["nodes"]] == ["a", "b"], (
        inventory
    )
    assert inventory["count"] == 2 and len(inventory["sha256"]) == 64, inventory
    recovery = fixture.cluster_recovery()
    assert recovery == {
        "cluster_name": "fake-hp",
        "status": "InService",
        "node_recovery": "None",
    }, recovery
    spares = fixture.spare_nodes()
    assert spares == ["spare-a"], spares
    environment = fixture.executor_environment()
    assert (
        environment[0]["allow_replace"] == "false"
        and environment[0]["spare_label"] is None
    ), environment
    gates = fixture.synthetic_replacement_gates()
    assert gates == [{"pod": "cpu-pod", "enabled": "True"}], gates
    transport.gate = None
    gates = fixture.synthetic_replacement_gates()
    assert gates[0]["enabled"] is None, gates
    transport.environment = []
    with pytest.raises(RegionalFixtureError, match="JSON object"):
        fixture.executor_environment()


def test_warm_fixture_binds_write_cluster_and_uses_one_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(tmp_path, monkeypatch)
    fixture = warm.WarmSpareLiveFixture(transport, "fake-hp")
    with pytest.raises(RegionalFixtureError, match="cluster ID"):
        fixture.post_synthetic_replacement({"cluster_id": "foreign"})
    assert transport.calls == [], transport.calls
    fixture.post_synthetic_replacement({"cluster_id": "cluster-a"})
    fixture.release_spares(["spare-a"], "incident-owned")
    fixture.reactivate_agent("spare-a")
    fixture.create_restore_workflow(
        incident_id="incident-owned",
        node="spare-a",
        profile_version="v1",
        reason="unit",
    )
    fixture.close_incident("incident-owned", reason="unit", operator="unit")
    assert all(call["attempts"] == 1 for _, call in transport.calls), transport.calls
    assert json.loads(transport.calls[0][1]["args"][0])["cluster_id"] == "cluster-a", (
        transport.calls
    )


@pytest.mark.parametrize("terminal", [False, True])
def test_warm_workflow_and_agent_waiters_poll_until_proven_state_or_timeout(
    terminal: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(tmp_path, monkeypatch)
    fixture = warm.WarmSpareLiveFixture(transport, "fake-hp")
    transport.responses[warm.STORE_PROBE] = [
        {"workflow": {"status": "RUNNING"}, "commands": [{"status": "WAITING"}]},
        {"workflow": {"status": "SUCCEEDED" if terminal else "RUNNING"}},
    ]
    arguments = dict(
        event_id="e", job_id="j", attempt_id="a", case_dir=tmp_path, timeout_seconds=10
    )
    if terminal:
        workflow = fixture.wait_for_workflow(**arguments)
        assert workflow["workflow"]["status"] == "SUCCEEDED", workflow
    else:
        with pytest.raises(RegionalFixtureError, match="terminal state"):
            fixture.wait_for_workflow(**arguments)
    transport.responses[warm.WORKFLOW_BY_ID] = [
        {"status": "RUNNING"},
        {"status": "FAILED" if terminal else "RUNNING"},
    ]
    if terminal:
        restored = fixture.wait_workflow_id("restore", timeout_seconds=10)
        assert restored["status"] == "FAILED", restored
    else:
        with pytest.raises(RegionalFixtureError, match="terminal state"):
            fixture.wait_workflow_id("restore", timeout_seconds=10)
    transport.responses[warm.STORE_PROBE] = [
        {"agents": []},
        {
            "agents": [
                {
                    "node_id": "spare-a",
                    "lifecycle_state": "ACTIVE" if terminal else "DRAINING",
                }
            ]
        },
    ]
    if terminal:
        agent = fixture.wait_agent_active("spare-a", timeout_seconds=10)
        assert agent["lifecycle_state"] == "ACTIVE", agent
    else:
        with pytest.raises(RegionalFixtureError, match="ACTIVE"):
            fixture.wait_agent_active("spare-a", timeout_seconds=10)


@pytest.mark.parametrize("ready", [False, True])
def test_readiness_waits_accept_observed_unknown_only_as_not_ready(
    ready: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(tmp_path, monkeypatch)
    fixture = warm.WarmSpareLiveFixture(transport, "fake-hp")
    transport.node["status"]["conditions"][0]["status"] = "True" if ready else "Unknown"
    node = fixture.wait_node_ready("spare-a", ready=ready, timeout_seconds=10)
    assert node["ready"] == ("True" if ready else "Unknown"), node
    transport.responses[warm.FLEET_READINESS] = [{"ready": not ready}, {"ready": ready}]
    fleet = fixture.wait_fleet_readiness("spare-a", ready=ready, timeout_seconds=10)
    assert fleet["ready"] is ready, fleet


def cleanup_state(**updates: Any) -> dict[str, Any]:
    return {
        "incident": {"incident_id": "incident-owned", "cluster_id": "cluster-a"},
        "workflow_status": "FAILED",
        "active_workflows": [],
        "open_commands": [],
        **updates,
    }


@pytest.mark.parametrize("failure", ["identity", "open", "flapping"])
def test_incident_quiescence_requires_bound_identity_and_continuous_quiet_window(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(tmp_path, monkeypatch)
    fixture = warm.WarmSpareLiveFixture(transport, "fake-hp")
    if failure == "identity":
        transport.responses[warm.INCIDENT_CLEANUP_STATE] = [
            cleanup_state(incident={"incident_id": "other"})
        ]
        with pytest.raises(RegionalFixtureError, match="incomplete"):
            fixture.wait_incident_idle("incident-owned", timeout_seconds=10)
    elif failure == "open":
        transport.responses[warm.INCIDENT_CLEANUP_STATE] = [
            cleanup_state(open_commands=[{"status": "LEASED"}])
        ]
        with pytest.raises(RegionalFixtureError, match="quiescent"):
            fixture.wait_incident_idle("incident-owned", timeout_seconds=10)
    else:
        transport.responses[warm.INCIDENT_CLEANUP_STATE] = [
            cleanup_state(),
            cleanup_state(active_workflows=[{"status": "RUNNING"}]),
            cleanup_state(),
        ]
        result = fixture.wait_incident_idle(
            "incident-owned", quiet_seconds=10, timeout_seconds=30
        )
        assert (
            result["incident_id"] == "incident-owned" and transport.clock.elapsed == 20
        ), result


def test_open_workflow_and_budget_reads_preserve_scope_and_reject_ambiguous_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(tmp_path, monkeypatch)
    transport.responses[warm.OPEN_WORKFLOWS] = [{"workflows": [{"request_id": "one"}]}]
    transport.responses[warm.ACTIVE_BUDGET_CLAIMS] = [
        {"workflows": [{"request_id": "budget-one"}]}
    ]
    fixture = warm.WarmSpareLiveFixture(transport, "fake-hp")
    workflows = fixture.open_workflows(job_id="job", nodes=["a", "b"])
    assert workflows == [{"request_id": "one"}], workflows
    assert transport.calls[-1][1]["args"] == ("cluster-a", "job", "a,b"), (
        transport.calls
    )
    claims = fixture.active_budget_claims()
    assert claims == [{"request_id": "budget-one"}], claims
    for node in ("", "a,b"):
        with pytest.raises(ValueError, match="non-empty"):
            fixture.open_workflows(nodes=[node])
    transport.responses[warm.INCIDENT_BY_ID] = [{"state": "RECOVERED"}]
    incident = fixture.incident_by_id("owned")
    assert incident == {"state": "RECOVERED"}, incident


def test_holder_and_service_fixture_keep_deadline_and_private_probe_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(tmp_path, monkeypatch)
    fixture = warm.WarmSpareLiveFixture(transport, "fake-hp")
    holder = warm.GpuHolderFixture(
        fixture, node="spare-a", run_id="owned", hold_seconds=60
    )
    holder.create()
    assert holder.deadline_at == NOW + timedelta(seconds=60), holder.deadline_at
    manifest = json.loads(
        next(
            value["input_text"]
            for name, value in transport.calls
            if name == "kubectl" and value["args"][0] == "apply"
        )
    )
    assert (
        manifest["spec"]["nodeName"] == "spare-a"
        and manifest["spec"]["activeDeadlineSeconds"] == 120
    ), manifest
    assert holder.cleanup() is False, transport.calls
    with pytest.raises(ValueError, match="positive"):
        warm.GpuHolderFixture(fixture, node="spare-a", run_id="owned", hold_seconds=0)
    from datetime import datetime, timezone

    from tests.regional._destr008_service_window import CASE, PINS, Host
    from tests.regional._destr008_service_window import Transport as ServiceTransport

    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    service_host = Host(tmp_path, monkeypatch)
    service_transport = ServiceTransport(service_host, tmp_path / "host-probes")
    settings = service_transport.settings
    service_warm = SimpleNamespace(
        regional=SimpleNamespace(
            settings=SimpleNamespace(
                gpu_kubeconfig=settings.kubeconfig,
                gpu_context=settings.context,
                namespace=settings.namespace,
                cluster_id=PINS["cluster_id"],
            )
        ),
        node_snapshot=lambda _node: {"uid": PINS["node_uid"]},
    )

    def build(settings: Any) -> Any:
        assert settings.state_directory == tmp_path / "host-probes", settings
        return service_transport

    monkeypatch.setattr(warm, "HostProbeFixture", build)
    service = warm.WarmSpareServiceFixture(
        service_warm,
        node=PINS["node"],
        image=settings.image,
        case_id=CASE,
        run_id=settings.run_id,
        state_directory=tmp_path / "host-probes",
        node_uid=PINS["node_uid"],
        plan_sha256="d" * 64,
        release_id="test-release",
        maintenance_expires_at=datetime.fromtimestamp(2_000_001_200, timezone.utc),
    )
    with pytest.raises(RegionalFixtureError, match="controller journal"):
        service.restore()
    service.create()
    stopped = service.stop("kubelet.service", restore_seconds=180, delay_seconds=15)
    assert stopped["phase"] == "SCHEDULED", stopped
    assert service.failsafe_at is not None
    assert service.failsafe_at.timestamp() == service_host.binding["restore_at"]
    service_host.fire_stop()
    restored = service.restore()
    residuals = service.cleanup()
    assert restored["phase"] == "RESTORED" and not any(residuals.values())
    assert service_transport.commands == [
        "snapshot",
        "prepare",
        "status",
        "stop-with-failsafe",
        "restore-service",
        "cleanup",
    ]
    service.close()


def test_node_mutation_tracks_cordon_and_rejects_unobserved_or_foreign_changes() -> (
    None
):
    api = NodeApi()
    fixture = warm.WarmSpareLiveFixture(api, "fake-hp")
    mutation = warm.NodeMutationFixture(fixture, "spare-a", track_unschedulable=True)
    mutation.apply(warm.NodePatch(labels={}, annotations={}, unschedulable=False))
    assert api.node["spec"]["unschedulable"] is False, api.node
    restored = mutation.restore()
    assert restored["unschedulable"] is True, restored
    assert api.patches[-1]["spec"] == {"unschedulable": True}, api.patches
