from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.cluster_executor import bootstrap
from scripts.e2e.regional import run_workload_acceptance as runner
from scripts.e2e.regional.probes import e2e_executor_readiness as executor_probe
from tests.regional._cov95_identity_support import Clock
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_cov95_identity_common import identity_site


@pytest.mark.parametrize("fallback", [False, True])
def test_workload_site_reuses_real_site_target_and_pair_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fallback: bool
) -> None:
    identity = identity_site(monkeypatch, tmp_path, count=2, fallback=fallback)
    monkeypatch.setattr(runner, "load_site", lambda *args, **kwargs: identity.site)
    site = runner.WorkloadSite(tmp_path / "site")
    with pytest.raises(runner.WorkloadAcceptanceError, match="required"):
        site.target("")
    with pytest.raises(runner.WorkloadAcceptanceError, match="absent"):
        site.target("foreign")
    a, b = site.target("a"), site.target("b")
    pair = site.multi(a, b)
    assert pair.cluster_a.cluster_id == "a" and pair.cluster_b.cluster_id == "b"
    constructed = []
    monkeypatch.setattr(
        runner,
        "RegionalLiveFixture",
        lambda settings: constructed.append(settings) or settings,
    )
    assert site.regional(b).gpu_context == "context-b"
    assert constructed[0].gpu_kubeconfig == tmp_path / "gpu"
    site.targets = {"a": a}
    assert site.target("") is a
    identity.config.pop("gpu_kubeconfig", None)
    identity.site.environment = {}
    with pytest.raises(runner.WorkloadAcceptanceError, match="GPU kubeconfig"):
        runner.WorkloadSite(tmp_path / "site")


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "nodes",
        "readiness",
        "collector",
        "agent",
        "pin",
        "artifact",
        "executor",
        "empty-logs",
        "suspicious",
        "log-read",
        "probe-read",
        "pod-replaced",
        "pod-missing",
        "container-replaced",
        "container-restarted",
        "incomplete-executor",
        "unready-executor",
        "pod-uid-missing",
        "pod-uid-duplicate",
        "inventory-read",
        "claim-stale",
        "claim-future",
        "claim-naive",
        "claim-before-container",
        "claim-malformed",
        "claim-missing",
        "claim-foreign",
        "claim-owners",
        "readiness-failed",
    ],
)
def test_e2e_preflight_proves_all_nodes_collectors_agents_and_executor_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    nodes = [
        {
            "name": f"node-{index}",
            "uid": f"uid-{index}",
            "ready": "True",
            "unschedulable": False,
            "labels": {},
        }
        for index in range(3)
    ]
    state = {
        "collector_statuses": [
            {"node_id": node["name"], "collector": kind}
            for node in nodes
            for kind in ("NVIDIA_KERNEL", "GPU_METRICS", "HOST_TELEMETRY")
        ],
        "agents": [
            {
                "node_id": node["name"],
                "lifecycle_state": "ACTIVE",
                "lease_expires_at": (
                    datetime.now(timezone.utc) + timedelta(minutes=10)
                ).isoformat(),
                "artifact_sha256": "a" * 64,
            }
            for node in nodes
        ],
    }
    metadata = {"required-agent-artifact-sha256": "a" * 64}
    if defect == "nodes":
        nodes = []
    elif defect == "readiness":
        nodes[0]["ready"] = "False"
    elif defect == "collector":
        state["collector_statuses"].pop()
    elif defect == "agent":
        state["agents"][0]["lease_expires_at"] = "2000-01-01T00:00:00Z"
    elif defect == "pin":
        metadata = {}
    elif defect == "artifact":
        state["agents"][0]["artifact_sha256"] = "b" * 64
    started_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    inventory = {
        "items": [
            {
                "metadata": {
                    "name": name,
                    "uid": f"uid-{name}",
                    "namespace": "gpu-fault-system",
                },
                "spec": {"nodeName": "node-0", "containers": [{"name": "executor"}]},
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [
                        {
                            "name": "executor",
                            "ready": True,
                            "containerID": f"containerd://{name}",
                            "imageID": "image@sha256:" + "a" * 64,
                            "restartCount": 0,
                            "state": {"running": {"startedAt": started_at}},
                        }
                    ],
                },
            }
            for name in ("executor-a", "executor-b")
        ]
    }
    if defect == "executor":
        inventory["items"] = []
    elif defect == "incomplete-executor":
        inventory["items"][1]["status"]["containerStatuses"][0].pop("containerID")
    elif defect == "unready-executor":
        inventory["items"][1]["status"]["containerStatuses"][0]["ready"] = False
    elif defect == "pod-uid-missing":
        inventory["items"][1]["metadata"].pop("uid")
    elif defect == "pod-uid-duplicate":
        inventory["items"][1]["metadata"]["uid"] = "uid-executor-a"
    inventory_reads = 0
    readiness_calls = []
    monkeypatch.setattr(
        bootstrap,
        "readiness_probe",
        lambda: readiness_calls.append(True) or int(defect == "readiness-failed"),
    )
    monkeypatch.delenv("GPU_FAULT_CLUSTER_EXECUTOR_ID", raising=False)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "a")
    calls = []

    def kubectl(*args: str, **kwargs: Any) -> str:
        nonlocal inventory_reads
        calls.append(args)
        if args[0] == "cpu":
            return json.dumps({"data": metadata})
        if args[1] == "get":
            inventory_reads += 1
            value = deepcopy(inventory)
            if inventory_reads > 1:
                if defect == "inventory-read":
                    raise runner.RegionalFixtureError("Pod inventory read failed")
                elif defect == "pod-replaced":
                    value["items"][0]["metadata"]["uid"] = "replacement"
                elif defect == "pod-missing":
                    value["items"].pop()
                elif defect == "container-replaced":
                    value["items"][0]["status"]["containerStatuses"][0][
                        "containerID"
                    ] = "containerd://replacement"
                elif defect == "container-restarted":
                    value["items"][0]["status"]["containerStatuses"][0][
                        "restartCount"
                    ] = 1
            return json.dumps(value)
        assert kwargs["check"] is True
        if args[1] == "logs":
            if defect == "log-read":
                raise runner.RegionalFixtureError("log read failed")
            return (
                ""
                if defect == "empty-logs" or defect.startswith("claim-")
                else "ERROR auth failed"
                if defect == "suspicious"
                else "worker healthy\nerror_count=0\n"
            )
        assert args[1] == "exec"
        assert args[4:9] == (
            "-c",
            "executor",
            "--",
            runner.component_python("gpu"),
            "-",
        )
        if defect == "probe-read":
            raise runner.RegionalFixtureError("probe read failed")
        pod_name = args[3]
        claim: dict[str, Any] = {
            "executor_id": f"a/{pod_name}",
            "execution_owners": ["gpu-fault-kubernetes-adapter"],
            "last_successful_claim_at": datetime.now(timezone.utc).isoformat(),
            "unrelated": "not-for-evidence",
        }
        if defect == "claim-stale":
            claim["last_successful_claim_at"] = (
                datetime.now(timezone.utc) - timedelta(seconds=301)
            ).isoformat()
        elif defect == "claim-future":
            claim["last_successful_claim_at"] = (
                datetime.now(timezone.utc) + timedelta(seconds=60)
            ).isoformat()
        elif defect == "claim-naive":
            claim["last_successful_claim_at"] = datetime.now().isoformat()
        elif defect == "claim-before-container":
            claim["last_successful_claim_at"] = (
                datetime.fromisoformat(started_at) - timedelta(seconds=1)
            ).isoformat()
        elif defect == "claim-missing":
            claim["last_successful_claim_at"] = None
        elif defect == "claim-foreign":
            claim["executor_id"] = "another-executor"
        elif defect == "claim-owners":
            claim["execution_owners"] = []
        path = tmp_path / f"{pod_name}-claim.json"
        path.write_text("{" if defect == "claim-malformed" else json.dumps(claim))
        path.chmod(0o600)
        monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(path))
        monkeypatch.setattr(executor_probe.socket, "gethostname", lambda: pod_name)
        try:
            return json.dumps(executor_probe.readiness_snapshot(*args[-3:]))
        except (ValueError, OSError) as exc:
            raise runner.RegionalFixtureError(type(exc).__name__) from None

    region = SimpleNamespace(
        settings=SimpleNamespace(cluster_id="a", namespace="gpu-fault-system"),
        cpu_python=lambda *args: state,
        kubectl=kubectl,
        gpu_nodes=lambda: nodes,
    )
    rejected = {
        "log-read",
        "probe-read",
        "pod-replaced",
        "pod-missing",
        "container-replaced",
        "container-restarted",
        "incomplete-executor",
        "unready-executor",
        "pod-uid-missing",
        "pod-uid-duplicate",
        "inventory-read",
        "readiness-failed",
    }
    if defect in rejected or defect.startswith("claim-"):
        with pytest.raises(runner.RegionalFixtureError):
            runner.e2e_preflight(region)
        return
    result = runner.e2e_preflight(region)
    assert bool(result["errors"]) is (defect not in {"none", "empty-logs"})
    if defect in {"none", "empty-logs"}:
        assert len(result["active_agents"]) == 3
        assert result["suspicious_executor_logs"] == []
        assert len(result["executor_logs"]) == 2
        assert len(readiness_calls) == 2
        assert inventory_reads == 2
        for log in result["executor_logs"]:
            assert log["read_succeeded"] is True
            assert log["uid"] == f"uid-{log['pod']}"
            assert log["lines"] == (0 if defect == "empty-logs" else 2)
            if defect == "empty-logs":
                assert log["sha256"] == hashlib.sha256(b"").hexdigest()
            assert log["readiness"]["authenticated_ready"] is True
            assert "not-for-evidence" not in json.dumps(log)
    assert all(args[1] in {"get", "logs", "exec"} for args in calls), (
        "E2E preflight must remain read-only"
    )


@pytest.mark.parametrize("change", ["none", "regression", "owners", "malformed"])
def test_e2e_executor_probe_entrypoint_preserves_only_valid_claim_evidence(
    change: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    claimed_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    state = {
        "executor_id": "cluster/executor",
        "execution_owners": ["owner"],
        "last_successful_claim_at": claimed_at.isoformat(),
        "unrelated": "not-for-evidence",
    }
    path = tmp_path / "claim.json"
    path.write_text(json.dumps(state))
    path.chmod(0o600)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster")
    monkeypatch.delenv("GPU_FAULT_CLUSTER_EXECUTOR_ID", raising=False)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(path))
    monkeypatch.setattr(executor_probe.socket, "gethostname", lambda: "executor")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e2e_executor_readiness.py",
            "cluster",
            "executor",
            (claimed_at - timedelta(minutes=1)).isoformat(),
        ],
    )
    calls = []

    def readiness() -> int:
        calls.append(True)
        if change == "regression":
            state["last_successful_claim_at"] = (
                claimed_at - timedelta(seconds=1)
            ).isoformat()
        elif change == "owners":
            state["execution_owners"] = ["other-owner"]
        path.write_text("{" if change == "malformed" else json.dumps(state))
        return 0

    monkeypatch.setattr(bootstrap, "readiness_probe", readiness)
    if change != "none":
        with pytest.raises(SystemExit, match="executor readiness evidence failed"):
            executor_probe.main()
        assert capsys.readouterr().out == ""
    else:
        executor_probe.main()
        text = capsys.readouterr().out
        result = json.loads(text)
        assert result["authenticated_ready"] is True
        assert result["claim_before"]["executor_id"] == "cluster/executor"
        assert result["claim_after"]["last_successful_claim_at"] == (
            claimed_at.isoformat()
        )
        assert "not-for-evidence" not in text
    assert calls == [True]


@pytest.mark.parametrize("read_fails", [False, True])
def test_iso_executor_log_negative_proof_still_rejects_empty_or_failed_reads(
    read_fails: bool,
) -> None:
    def kubectl(*args: str, **kwargs: Any) -> str:
        if read_fails:
            raise runner.RegionalFixtureError("log read failed")
        return ""

    regional = SimpleNamespace(
        ready_pods=lambda *args: [{"name": "executor"}], kubectl=kubectl
    )
    with pytest.raises(runner.RegionalFixtureError):
        runner.executor_logs_since(regional, since=datetime.now(timezone.utc))


@pytest.mark.parametrize("terminal", [False, True])
def test_terminal_waiter_requires_three_successful_container_terminations(
    monkeypatch: pytest.MonkeyPatch, terminal: bool
) -> None:
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    calls = []

    def store(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(None)
        return {
            "observations": []
            if len(calls) == 1
            else [
                {
                    "workload_phase": "SUCCEEDED",
                    "containers": [
                        {"terminated": terminal and len(calls) > 2, "exit_code": 0}
                        for _ in range(3)
                    ],
                }
            ]
        }

    monkeypatch.setattr(runner, "workload_store", store)
    if terminal:
        result = runner.wait_terminal_observation(
            None, job_id="job", attempt_id="attempt", timeout_seconds=30
        )
        assert all(
            item["terminated"] for item in result["observations"][0]["containers"]
        ), "terminal convergence must prove all three containers terminated"
        assert len(calls) == 3
    else:
        with pytest.raises(runner.RegionalFixtureError, match="did not converge"):
            runner.wait_terminal_observation(
                None, job_id="job", attempt_id="attempt", timeout_seconds=20
            )


@pytest.mark.parametrize("healthy", [False, True])
def test_finite_waiter_does_not_pass_empty_logs_or_a_read_failure(
    monkeypatch: pytest.MonkeyPatch, healthy: bool
) -> None:
    monkeypatch.setattr(runner, "time", Clock())
    calls = []

    def snapshot() -> dict[str, Any]:
        calls.append(None)
        if len(calls) == 1:
            raise runner.RegionalFixtureError("temporary read failure")
        return {
            "pods": [
                {"name": f"pod-{index}", "node": f"node-{index}", "phase": "Succeeded"}
                for index in range(3)
            ],
            "heartbeat_logs": {
                f"pod-{index}": "SUCCESS all_reduce=300.0"
                if healthy and len(calls) > 2
                else ""
                for index in range(3)
            },
        }

    if healthy:
        assert (
            len(
                runner.wait_finite_workload(
                    SimpleNamespace(snapshot=snapshot), timeout_seconds=30
                )["pods"]
            )
            == 3
        )
        assert len(calls) == 3
    else:
        with pytest.raises(runner.RegionalFixtureError, match="did not succeed"):
            runner.wait_finite_workload(
                SimpleNamespace(snapshot=snapshot), timeout_seconds=20
            )


@pytest.mark.parametrize("interval", [None, "17"])
def test_watcher_poll_interval_comes_from_the_selected_deployment(
    interval: str | None,
) -> None:
    env = [{"name": "IGNORED", "valueFrom": {}}]
    if interval is not None:
        env.append(
            {"name": "GPU_FAULT_COMPLETION_WATCH_INTERVAL_SECONDS", "value": interval}
        )
    region = SimpleNamespace(
        kubectl=lambda *args: json.dumps(
            {"spec": {"template": {"spec": {"containers": [{"env": env}]}}}}
        )
    )
    assert runner.watcher_interval_seconds(region) == (17 if interval else 30)


def test_admin_status_wrapper_only_records_digests_of_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    monkeypatch.setattr(
        runner,
        "run_fixture_command",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or subprocess.CompletedProcess(args[0], 1, "example-output", "example-error"),
    )
    result = runner.admin_status(tmp_path)
    assert result["returncode"] == 1
    assert len(result["stdout_sha256"]) == len(result["stderr_sha256"]) == 64
    assert "example-output" not in json.dumps(result)
    assert calls[0][1]["timeout"] == 1800
    assert calls[0][0][0][-2:] == ["--state-dir", str(tmp_path)]


def test_managed_fixture_factory_passes_complete_workload_expectations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, site = tmp_path / "manifest", tmp_path / "site"
    manifest.write_text("synthetic", encoding="ascii")
    site.write_text("synthetic", encoding="ascii")
    calls = []
    monkeypatch.setattr(
        runner,
        "ManagedWorkloadFixture",
        lambda regional, settings: calls.append(settings) or settings,
    )
    value = runner.managed_fixture(
        None, manifest=manifest, site_file=site, job_id="job", attempt_id="attempt"
    )
    assert value.expected_pods == 3 and value.expected_gpu_count == 24
    assert value.restart_budget == 1
    with pytest.raises(runner.RegionalFixtureError, match="at least three"):
        runner.prewarm_nodes(SimpleNamespace(gpu_nodes=lambda: []))


@pytest.mark.parametrize("injection", [False, True])
def test_cleanup_is_deferred_for_unproved_control_terminal_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, injection: bool
) -> None:
    deletions = []
    monkeypatch.setattr(
        runner,
        "workload_store",
        lambda *args, **kwargs: {"commands": [], "workflows": [{"status": "UNKNOWN"}]},
    )
    monkeypatch.setattr(
        runner.workload_case,
        "wait_for_cleanup_quiescence",
        lambda **kwargs: {"safe_to_delete": False},
    )
    outcome: dict[str, Any] = {}
    errors = runner.cleanup_workload(
        regional=None,
        workload=SimpleNamespace(delete=lambda: deletions.append("delete")),
        case_dir=tmp_path,
        job_id="job",
        attempt_id="attempt",
        injection={
            "node": "node",
            "marker": "unit",
            "observed_after": datetime.now(timezone.utc),
        }
        if injection
        else None,
        result=outcome,
        label="a",
    )
    assert errors and outcome["workload_cleanup_deferred"] is True
    assert deletions == []


def test_uid_recheck_after_window_cannot_return_a_recreated_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "time", Clock())
    fixture = SimpleNamespace(
        pods=lambda: [{"uid": "old"}], snapshot=lambda: {"pods": [{"uid": "new"}]}
    )
    with pytest.raises(runner.RegionalFixtureError, match="UIDs changed"):
        runner.wait_pod_uids_unchanged(fixture, {"old"}, timeout_seconds=3)


@pytest.mark.parametrize("document", [{}, [], {"items": "unknown"}])
def test_workload_cleanup_inventory_requires_a_resource_list(document: Any) -> None:
    regional = SimpleNamespace(kubectl=lambda *args: json.dumps(document))
    with pytest.raises(runner.RegionalFixtureError, match="not a resource list"):
        runner.workload_residuals(regional, SimpleNamespace(resource="job"), "job")


@pytest.mark.parametrize("case_id", runner.CASE_IDS)
@pytest.mark.parametrize("defect", ["none", "handler", "confirmation", "predecessor"])
def test_workload_cli_authorizes_and_binds_dispatch_before_publishing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case_id: str, defect: str
) -> None:
    events = []
    targets = {
        name: SimpleNamespace(cluster_id=name, context="context-" + name)
        for name in ("a", "b")
    }
    for name in ("site", "cpu.kubeconfig", "gpu.kubeconfig"):
        (tmp_path / name).write_text("unit fixture\n")
    site = SimpleNamespace(
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        target=lambda name: targets[name],
        regional=lambda target: SimpleNamespace(
            evidence_identity=lambda: {
                "release_id": "unit-release",
                "cluster_id": target.cluster_id,
            }
        ),
    )
    monkeypatch.setattr(runner, "WorkloadSite", lambda path: site)
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner.os, "umask", lambda mode: None)
    monkeypatch.setattr(
        runner, "predecessor_path", lambda *args: ("previous", tmp_path / "previous")
    )
    monkeypatch.setattr(
        runner,
        "predecessor_evidence",
        lambda *args, **kwargs: {"valid": defect != "predecessor"},
    )
    deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
    monkeypatch.setattr(
        runner,
        "authorize_execution",
        lambda *args, **kwargs: events.append("authorize") or deadline,
    )
    confirmation = case_id.removeprefix("GF-REGIONAL-").replace("-", "") + "_EXECUTE"
    argv = [
        "workload",
        "--run-dir",
        str(tmp_path),
        "--site",
        str(tmp_path / "site"),
        "--case",
        case_id,
        "--cluster-id",
        "a",
        "--secondary-cluster-id",
        "b",
        "--state-dir",
        str(tmp_path),
        "--host-probe-image",
        "unit@sha256:" + "a" * 64,
        "--execute",
        "--confirm",
        "WRONG" if defect == "confirmation" else confirmation,
    ]
    monkeypatch.setattr(sys, "argv", argv)

    def handler(**kwargs: Any) -> dict[str, Any]:
        events.append("handler")
        assert events[0] == "authorize"
        if case_id in {"GF-REGIONAL-E2E-001", "GF-REGIONAL-ISO-001"}:
            assert kwargs["maintenance_window_end"] == deadline
        if defect == "handler":
            raise runner.WorkloadCaseError(
                RuntimeError("synthetic failure"),
                {"cleanup_errors": ["unproven"], "verdict": "PASS"},
            )
        return {"verdict": "PASS", "checks": {"offline": True}}

    for name in ("run_workload_baseline", "run_iso001", "run_e2e001"):
        monkeypatch.setattr(runner, name, handler)
    path = tmp_path / "cases" / case_id / f"{case_id}.json"
    if defect in {"confirmation", "predecessor"}:
        with pytest.raises(
            runner.WorkloadAcceptanceError, match="confirmation|predecessor"
        ):
            runner.main()
        assert "handler" not in events and not path.exists()
        return
    assert runner.main() == int(defect == "handler")
    result = json.loads(path.read_text())
    assert result["verdict"] == ("FAIL" if defect == "handler" else "PASS")
    assert result["cluster_id"] == "a" and result["release_id"] == "unit-release"
    if defect == "handler":
        assert result["cleanup_errors"] == ["unproven"]


@pytest.mark.parametrize("changed", ["cpu", "gpu", "site", "release"])
def test_workload_plan_refuses_changed_live_connection_or_site(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: str
) -> None:
    from scripts.e2e.regional import live_driver_guard

    files = {name: tmp_path / f"{name}.yaml" for name in ("cpu", "gpu", "site")}
    for path in files.values():
        path.write_text("original unit fixture\n")
    identity = {"release_id": "unit-release", "cluster_id": "a"}
    target = SimpleNamespace(cluster_id="a", context="context-a")
    site = SimpleNamespace(
        cpu_kubeconfig=files["cpu"],
        gpu_kubeconfig=files["gpu"],
        target=lambda name: target,
        regional=lambda selected: SimpleNamespace(
            evidence_identity=lambda: dict(identity)
        ),
    )
    monkeypatch.setattr(runner, "WorkloadSite", lambda path: site)
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner.os, "umask", lambda mode: None)
    monkeypatch.setattr(runner, "predecessor_path", lambda *args: (None, None))
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "a" * 64)
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_EXECUTION_SCOPE", raising=False)
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_SELECTION_REFERENCE", raising=False)
    common = [
        "workload",
        "--run-dir",
        str(tmp_path),
        "--site",
        str(files["site"]),
        "--case",
        "GF-REGIONAL-WORKLOAD-001",
        "--cluster-id",
        "a",
    ]
    monkeypatch.setattr(sys, "argv", [*common, "--plan"])
    assert runner.main() == 0
    plan = json.loads(
        (tmp_path / "cases/GF-REGIONAL-WORKLOAD-001/plan.json").read_text()
    )
    assert len(plan["connections"]) == 2
    assert plan["details"]["release_id"] == "unit-release"
    if changed == "release":
        identity["release_id"] = "another-release"
    else:
        files[changed].write_text("changed unit fixture\n")
    called = []
    monkeypatch.setattr(
        runner, "run_workload_baseline", lambda **kwargs: called.append(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *common,
            "--execute",
            "--confirm",
            "WORKLOAD001_EXECUTE",
            "--maintenance-window-end",
            (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        ],
    )
    with pytest.raises(RuntimeError, match="drifted at connections|drifted at details"):
        runner.main()
    assert called == [], "changed deployment inputs must be refused before submission"
