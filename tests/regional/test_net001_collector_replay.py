"""NET-001 runner contracts: cleanup reach, kubectl deadlines, convergence."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import regional_commands
from scripts.e2e.regional import regional_live_fixture as live
from scripts.e2e.regional import run_net001_collector_replay as net001

SERVICES_OK = {
    service: {"ActiveState": "active", "NRestarts": 0} for service in net001.SERVICES
}


def _runner(
    tmp_path: Path, *, window_minutes: int = 60, namespace: str = "gpu-fault-system"
) -> net001.Runner:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("{}", encoding="utf-8")
    gpu.write_text("{}", encoding="utf-8")
    settings = net001.Settings(
        cpu_kubeconfig=cpu,
        cpu_context="",
        gpu_kubeconfig=gpu,
        gpu_context="gpu",
        namespace=namespace,
        cluster_id="cluster-a",
        region="us-west-2",
        target_node="node-a",
        endpoint_host="control.example",
        host_probe_image="probe@sha256:" + "0" * 64,
        predecessor={"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"},
    )
    deadline = datetime.now(timezone.utc) + timedelta(minutes=window_minutes)
    return net001.Runner(tmp_path / "run", settings, 1, deadline)


def _snapshot(*, replayable: int = 0, matching: list[dict[str, Any]] | None = None):
    outbox = {"replayable_count": replayable, "line_count": 0, "matching": []}
    kernel = dict(outbox, matching=matching or [])
    return {
        "services": SERVICES_OK,
        "outboxes": {
            "kernel": kernel,
            "dcgm": dict(outbox),
            "host": dict(outbox),
            "fabric-manager": dict(outbox),
        },
        "rules": [],
    }


def _store(evidence: int = 3, incidents: list[dict[str, Any]] | None = None):
    return {
        "evidence": [{"record_id": f"r{index}"} for index in range(evidence)],
        "incidents": incidents or [],
        "notifications": [],
    }


# --------------------------------------------------------------------------- #
# kubectl deadlines
# --------------------------------------------------------------------------- #
def test_every_kubectl_verb_gets_a_deadline() -> None:
    assert net001.kubectl_timeout(["kubectl", "exec", "pod", "--", "true"]) == 120
    assert net001.kubectl_timeout(["kubectl", "delete", "pod", "x"]) == 300
    assert net001.kubectl_timeout(["kubectl", "wait", "--for=condition=Ready"]) == 240
    assert net001.kubectl_timeout(["kubectl", "get", "pods"]) == 120


def test_a_hung_kubectl_becomes_a_case_error(monkeypatch) -> None:
    def hang(argv: list[str], **kwargs: Any) -> None:
        assert kwargs["timeout_seconds"] == 120
        raise subprocess.TimeoutExpired(argv, kwargs["timeout_seconds"])

    monkeypatch.setattr(regional_commands, "run_command", hang)

    with pytest.raises(net001.CaseError, match="timed out after 120s"):
        net001.command(["kubectl", "exec", "pod", "--", "sleep"])


# --------------------------------------------------------------------------- #
# Firewall tags reach cleanup as soon as they are armed
# --------------------------------------------------------------------------- #
def test_a_tag_is_active_the_moment_its_timer_is_armed(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner(tmp_path)
    runner.ips = ["192.0.2.10"]

    def host_tool(verb: str, *_args: str) -> dict[str, Any]:
        if verb == "arm":
            return {"armed": True}
        raise net001.CaseError("block failed after the timer was armed")

    monkeypatch.setattr(runner, "host_tool", host_tool)

    with pytest.raises(net001.CaseError, match="block failed"):
        runner.arm_and_block(runner.tag, 720)

    assert runner.blocked is True, "cleanup would skip the armed tag"
    assert runner.active_tags == [runner.tag]


def test_connectivity_assertion_failure_keeps_the_tag_active(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner(tmp_path)
    runner.ips = ["192.0.2.10"]
    monkeypatch.setattr(
        runner,
        "host_tool",
        lambda verb, *_a: {"connectivity": {"192.0.2.10": True}, "armed": True},
    )

    with pytest.raises(net001.CaseError, match="remains reachable"):
        runner.arm_and_block(runner.reconnect_tag, 60)

    assert runner.reconnect_blocked is True


def test_cleanup_tag_clears_the_active_tag(tmp_path: Path, monkeypatch) -> None:
    runner = _runner(tmp_path)
    runner.ips = ["192.0.2.10"]
    runner.active_tags.append(runner.tag)
    monkeypatch.setattr(
        runner,
        "host_tool",
        lambda *_a: {"rules": [], "connectivity": {"192.0.2.10": True}},
    )

    runner.cleanup_tag(runner.tag)

    assert runner.blocked is False


# --------------------------------------------------------------------------- #
# Convergence and non-blocking checks
# --------------------------------------------------------------------------- #
def test_replay_convergence_needs_only_the_records_and_empty_outboxes() -> None:
    assert net001.Runner.replay_converged(_snapshot(), _store(evidence=3)) is True
    assert net001.Runner.replay_converged(_snapshot(), _store(evidence=2)) is False
    assert net001.Runner.replay_converged(_snapshot(replayable=1), _store()) is False


def test_incident_checks_are_never_vacuously_true() -> None:
    empty = net001.Runner.incident_checks({})
    assert all(value is False for value in empty.values()), empty


def test_result_checks_are_false_without_a_final_store(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    checks = runner.result_checks({}, {})
    assert checks["no_mutation_workflow"] is False
    assert checks["drill_notifications_suppressed"] is False
    assert checks["three_unique_records_after_recovery"] is False


def test_validate_final_blocks_on_a_workflow_but_not_on_incident_state(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner(tmp_path)
    runner.baseline = {"services": SERVICES_OK}
    monkeypatch.setattr(net001, "write_json", lambda _p, _v: None)
    unrecovered = {
        "state": "ACTIVE",
        "workflow_request_id": None,
        "decision_disposition": "MONITOR_ONLY",
        "decision_action": "NO_ACTION",
        "official_action": "IGNORE",
    }

    soft = runner.validate_final(_snapshot(), _store(incidents=[unrecovered] * 3))
    assert soft["incidents_recovered"] is False
    assert soft["incidents_monitor_only"] is True

    with pytest.raises(net001.CaseError, match="created a workflow"):
        runner.validate_final(
            _snapshot(),
            _store(incidents=[dict(unrecovered, workflow_request_id="wf-1")]),
        )


def test_reconnect_once_returns_the_documents_it_already_read(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner(tmp_path)
    reads: list[str] = []
    before = _store(evidence=3)
    snapshot, after = _snapshot(), _store(evidence=3)

    def store_probe() -> dict[str, Any]:
        reads.append("store")
        return before

    monkeypatch.setattr(runner, "store_probe", store_probe)
    monkeypatch.setattr(runner, "arm_and_block", lambda _tag, _ttl: None)
    monkeypatch.setattr(runner, "cleanup_tag", lambda _tag: {})
    monkeypatch.setattr(runner, "wait_for_replay", lambda: (snapshot, after))
    monkeypatch.setattr(net001.time, "sleep", lambda _s: None)
    monkeypatch.setattr(net001, "write_json", lambda _p, _v: None)

    returned = runner.reconnect_once()

    assert returned == (snapshot, after)
    assert returned[0] is snapshot and returned[1] is after
    assert reads == ["store"], "only the pre-reconnect store read is needed"


# --------------------------------------------------------------------------- #
# Maintenance window and evidence identity
# --------------------------------------------------------------------------- #
def test_run_refuses_a_window_under_twenty_five_minutes(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner(tmp_path, window_minutes=20)
    monkeypatch.setattr(
        runner, "cleanup_resources", lambda: {"pod": False, "configmap": False}
    )
    monkeypatch.setattr(runner, "read_release_id", lambda: "release-1")

    assert runner.run() == 1

    document = json.loads(
        (runner.case_dir / f"{net001.CASE_ID}.json").read_text(encoding="utf-8")
    )
    assert "25 minutes" in document["error"]
    assert document["verdict"] == "FAIL"
    assert document["checks"]["no_mutation_workflow"] is False


def test_result_carries_the_release_identity(tmp_path: Path, monkeypatch) -> None:
    runner = _runner(tmp_path, window_minutes=60)
    monkeypatch.setattr(
        runner, "cleanup_resources", lambda: {"pod": False, "configmap": False}
    )
    monkeypatch.setattr(runner, "read_release_id", lambda: "release-1")
    monkeypatch.setattr(
        runner,
        "create_resources",
        lambda: (_ for _ in ()).throw(net001.CaseError("stop here")),
    )

    runner.run()

    document = json.loads(
        (runner.case_dir / f"{net001.CASE_ID}.json").read_text(encoding="utf-8")
    )
    assert document["release_id"] == "release-1"
    assert document["cluster_id"] == "cluster-a"


def test_resource_cleanup_failure_is_recorded_not_raised(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _runner(tmp_path, window_minutes=20)

    def failing_cleanup() -> dict[str, bool]:
        raise net001.CaseError("command timed out after 300s: kubectl delete")

    monkeypatch.setattr(runner, "cleanup_resources", failing_cleanup)

    assert runner.run() == 1

    document = json.loads(
        (runner.case_dir / f"{net001.CASE_ID}.json").read_text(encoding="utf-8")
    )
    assert "resource cleanup failed" in document["error"]


def _pod(
    namespace: str, name: str, *, gpus: int = 0, resource_kind: str = "requests"
) -> dict[str, Any]:
    resources = {resource_kind: {"nvidia.com/gpu": str(gpus)}} if gpus else {}
    return {
        "metadata": {"namespace": namespace, "name": name},
        "spec": {"containers": [{"resources": resources}]},
    }


@pytest.mark.parametrize("namespace", ["gpu-fault-system", "acceptance-system"])
@pytest.mark.parametrize("resource_kind", ["requests", "limits"])
def test_business_workloads_share_the_fleet_wide_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, namespace: str, resource_kind: str
) -> None:
    runner = _runner(tmp_path, namespace=namespace)
    pods = [
        _pod("kube-system", "coredns"),
        _pod("cert-manager", "cainjector"),
        _pod("hyperpod-inference-system", "router"),
        _pod("kubeflow", "training-operator"),
        _pod("aws-hyperpod", "controller"),
        _pod(namespace, "executor"),
        _pod(namespace, "acceptance-worker", gpus=8, resource_kind=resource_kind),
        _pod("training", "training-worker", gpus=8, resource_kind=resource_kind),
        _pod("training", "cpu-business-worker"),
    ]
    payload = json.dumps({"items": pods})

    def gpu(*args: str) -> subprocess.CompletedProcess[str]:
        assert "spec.nodeName=node-a,status.phase=Running" in args
        assert "-A" in args
        return subprocess.CompletedProcess(args, 0, payload, "")

    monkeypatch.setattr(runner, "gpu", gpu)
    regional = live.RegionalLiveFixture(
        live.RegionalLiveSettings(
            cpu_kubeconfig=runner.settings.cpu_kubeconfig,
            gpu_kubeconfig=runner.settings.gpu_kubeconfig,
            gpu_context=runner.settings.gpu_context,
            namespace=namespace,
            cluster_id=runner.settings.cluster_id,
            region=runner.settings.region,
        )
    )

    def kubectl(*args: str, **kwargs: Any) -> str:
        assert args[:3] == ("gpu", "get", "pod")
        assert "spec.nodeName=node-a,status.phase=Running" in args
        assert kwargs["all_namespaces"] is True
        return payload

    monkeypatch.setattr(regional, "kubectl", kubectl)
    expected = [
        {"namespace": namespace, "name": "acceptance-worker"},
        {"namespace": "training", "name": "training-worker"},
        {"namespace": "training", "name": "cpu-business-worker"},
    ]
    assert runner.active_business_workloads() == expected
    assert regional.business_workloads("node-a") == expected
