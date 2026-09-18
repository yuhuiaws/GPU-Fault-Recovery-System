from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_e2e002_multicluster_fault as e2e
from scripts.e2e.regional import run_iso006_cluster_offline as offline
from scripts.e2e.regional import run_workload_acceptance as workload
from scripts.e2e.regional.multi_cluster_fixture import REJECTION_COUNTERS
from tests.regional._identity_lifecycle_support import (
    host_type,
    regional_type,
    workload_types,
)
from tests.regional._identity_lifecycle_support import recovery_state as recovery_state


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "event",
        "attempt",
        "incident",
        "command",
        "step",
        "duplicate",
        "safety",
        "budget",
    ],
)
def test_successful_statuses_do_not_replace_causal_identity(defect: str) -> None:
    observed = datetime.now(timezone.utc) - timedelta(seconds=1)
    state = recovery_state("a", "node-a", "marker-test")
    if defect == "event":
        state["decision"]["event_id"] = "foreign"
    elif defect == "attempt":
        state["event"]["attempt_id"] = "old"
    elif defect == "incident":
        state["incident"]["event_id"] = "foreign"
    elif defect == "command":
        state["commands"][0]["workflow_request_id"] = "foreign"
    elif defect == "step":
        state["workflow"]["step_executions"][0]["adapter_operation_id"] = "remote/other"
    elif defect == "duplicate":
        state["commands"].append(copy.deepcopy(state["commands"][0]))
    elif defect == "safety":
        state["workflow"]["safety_steps"] = [{"operation": "RESET_GPU"}]
    elif defect == "budget":
        state["restart_budget"]["cluster_id"] = "b"
    errors = workload.recovery_identity_errors(
        state,
        cluster_id="a",
        job_id="job-test",
        attempt_id="attempt-test",
        node="node-a",
        marker="marker-test",
        observed_after=observed,
    )
    assert bool(errors) is (defect != "none"), errors


def lifecycle_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, defect: str = ""
) -> tuple[Any, list[Any], list[str]]:
    events: list[str] = []
    regions: list[Any] = []
    Managed, Prewarm = workload_types(events, defect)
    Region = regional_type(tmp_path, defect, regions, events)
    Host = host_type(tmp_path, events)

    a, b = Region("a"), Region("b")
    targets = [SimpleNamespace(cluster_id=region.cluster) for region in regions]
    multi = SimpleNamespace(
        cluster_a=targets[0],
        cluster_b=targets[1],
        regional=lambda target: a if target.cluster_id == "a" else b,
    )
    site = SimpleNamespace(
        site_file=tmp_path / "site.yaml",
        regional=multi.regional,
        multi=lambda *args: multi,
    )
    site.site_file.write_text("synthetic test fixture\n", encoding="ascii")

    def observation(regional: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        pods = regional.workload.source["pods"]
        return {
            "cluster_id": regional.cluster,
            "job_id": "job-test",
            "attempt_id": "attempt-test",
            "restart_budget": regional.workload.restart_budget,
            "node_id": pods[0]["node"],
            "workload_phase": "RUNNING",
            "runtime_profile_version": "profile-test",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "workload_ids": ["gpu-system/pytorchjob/test"],
            "containers": [
                {
                    "pod_name": pod["name"],
                    "pod_uid": pod["uid"],
                    "node_id": pod["node"],
                    "gpu_uuids": [
                        f"GPU-{regional.cluster}-{index * 8 + slot}"
                        for slot in range(8)
                    ],
                }
                for index, pod in enumerate(pods)
            ],
        }

    def store(regional: Any, **kwargs: Any) -> dict[str, Any]:
        state = {
            "cluster_id": regional.cluster,
            "restart_budget": None,
            "decision": None,
            "observations": [],
            "incidents": [],
            "workflows": [],
            "commands": [],
        }
        if defect == "dirty" and not regional.submitted:
            state["workflows"] = [{"request_id": "old"}]
        if regional.observation is not None:
            state["observations"] = [copy.deepcopy(regional.observation)]
            if defect == "virtual" and regional.cluster == "b":
                state["observations"][0]["containers"][0]["pod_uid"] = "foreign"
        return state

    def post(regional: Any, value: dict[str, Any]) -> dict[str, bool]:
        regional.observation = copy.deepcopy(value)
        events.append("observation-" + regional.cluster + "-" + value["node_id"])
        return {"accepted": True}

    monkeypatch.setattr(workload, "managed_fixture", Managed)
    monkeypatch.setattr(e2e, "ManagedWorkloadFixture", Managed)
    monkeypatch.setattr(workload, "ImagePrewarmFixture", Prewarm)
    monkeypatch.setattr(e2e, "ImagePrewarmFixture", Prewarm)
    monkeypatch.setattr(workload, "HostProbeFixture", Host)
    monkeypatch.setattr(
        workload, "HostProbeSettings", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        workload,
        "e2e_preflight",
        lambda regional: {"errors": [], "nodes": regional.gpu_nodes()},
    )
    monkeypatch.setattr(workload, "workload_store", store)
    monkeypatch.setattr(workload, "post_observation", post)
    monkeypatch.setattr(workload, "workload_residuals", lambda *args: [])
    monkeypatch.setattr(
        workload,
        "workload_case_settings",
        lambda **kwargs: SimpleNamespace(regional=kwargs["regional"].settings),
    )
    monkeypatch.setattr(workload.workload_case, "wait_observation", observation)
    monkeypatch.setattr(
        workload.workload_case,
        "wait_for_cleanup_quiescence",
        lambda **kwargs: {"safe_to_delete": True},
    )
    monkeypatch.setattr(
        workload,
        "wait_pod_uids_unchanged",
        lambda managed, *args, **kwargs: managed.snapshot(),
    )
    registrations = [
        {
            "cluster_id": region.cluster,
            "enabled": True,
            "synthetic": False,
            "eks_cluster_arn": f"arn:aws:eks:us-west-2:000000000000:cluster/{region.cluster}",
            "hyperpod_cluster_name": "hp-" + region.cluster,
        }
        for region in regions
    ]
    monkeypatch.setattr(workload, "registration_snapshot", lambda *args: registrations)
    containers = {
        "api": {
            "uid": "api-uid",
            "phase": "Running",
            "ready": True,
            "containers": {"api": {"restart_count": 0}},
        }
    }
    monkeypatch.setattr(
        e2e,
        "read_only_preflight",
        lambda *args, **kwargs: {
            "errors": [],
            "cpu_containers": containers,
            "registrations": registrations,
            "nodes_a": a.gpu_nodes(),
            "nodes_b": b.gpu_nodes(),
            "executor_identities": {
                region.cluster: ["executor-" + region.cluster] for region in regions
            },
        },
    )
    monkeypatch.setattr(
        e2e, "control_plane_container_statuses", lambda *args: containers
    )
    return site, targets, events


@pytest.mark.parametrize("defect", ["none", "binding", "cleanup", "expired"])
def test_e2e001_handoff_is_written_after_cleanup_and_never_hides_failure(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path, defect=defect)
    deadline = datetime.now(timezone.utc) + timedelta(
        minutes=-1 if defect == "expired" else 60
    )
    kwargs = {
        "site": site,
        "target": targets[0],
        "case_dir": tmp_path,
        "job_id": "job-test",
        "attempt_id": "attempt-test",
        "host_probe_image": "image@sha256:" + "a" * 64,
        "attempt": 1,
        "maintenance_window_end": deadline,
    }
    if defect == "expired":
        with pytest.raises(workload.WorkloadCaseError):
            workload.run_e2e001(**kwargs)
        assert "kmsg-inject" not in events
        assert not (tmp_path / "execution-card.json").exists(), (
            "an expired window must not publish a BLAST producer handoff"
        )
    else:
        result = workload.run_e2e001(**kwargs)
        card = json.loads((tmp_path / "execution-card.json").read_text())
        assert (
            result["verdict"]
            == card["verdict"]
            == ("PASS" if defect == "none" else "FAIL")
        )
        assert card["cleanup_complete"] is (defect != "cleanup")
        assert len(card["baseline_sha256"]) == len(card["state_sha256"]) == 64
        assert (
            events.index("authorize-a")
            < events.index("wait-restarted-a")
            < events.index("delete-a")
        )
    assert "host-cleanup" in events and "delete-a" in events


@pytest.mark.parametrize("defect", ["none", "dirty", "virtual", "binding"])
def test_iso001_refuses_bad_baselines_and_restores_virtual_observations(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path, defect=defect)
    kwargs = {
        "site": site,
        "primary": targets[0],
        "secondary": targets[1],
        "case_dir": tmp_path,
        "job_id": "job-test",
        "attempt_id": "attempt-test",
        "attempt": 1,
        "maintenance_window_end": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    if defect in {"dirty", "virtual"}:
        with pytest.raises(workload.WorkloadCaseError):
            workload.run_iso001(**kwargs)
        assert "inject-a" not in events
        if defect == "dirty":
            assert not any(
                event.startswith(("delete-", "submit-", "prewarm-a", "prewarm-b"))
                for event in events
            ), "a dirty identity baseline must stop before workload mutation"
        else:
            assert all(
                site.regional(target).observation["node_id"]
                == "node-" + target.cluster_id + "-0"
                for target in targets
            ), "both physical Observations must be restored after virtual proof failure"
    else:
        result = workload.run_iso001(**kwargs)
        assert result["verdict"] == ("PASS" if defect == "none" else "FAIL"), result
        assert "delete-a" in events and "delete-b" in events
        assert events.index("authorize-a") < events.index("wait-restarted-a")
        assert "authorize-b" not in events and "wait-restarted-b" not in events


@pytest.mark.parametrize("defect", ["none", "ack", "branch", "dirty", "binding"])
def test_e2e002_observes_both_injected_branches_before_scoped_cleanup(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path, defect=defect)
    settings = e2e.Settings(
        multi=site.multi(*targets),
        site_file=site.site_file,
        job_id="job-test",
        attempt_id="attempt-test",
        predecessor_path=tmp_path / "previous.json",
    )
    result = e2e.execute_case(
        settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    assert result == (0 if defect == "none" else 1)
    if defect == "dirty":
        assert not any(event.startswith(("submit-", "delete-")) for event in events), (
            "dirty two-cluster state must not authorize workload submission or deletion"
        )
    else:
        assert "settle-a" in events and "settle-b" in events
        assert events.index("delete-a") > max(
            events.index("settle-a"), events.index("settle-b")
        )
        assert "delete-b" in events
        for cluster in ("a", "b"):
            if defect == "none":
                assert f"wait-restarted-{cluster}" in events
            if f"wait-restarted-{cluster}" in events:
                assert (
                    events.index(f"settle-{cluster}")
                    < events.index(f"authorize-{cluster}")
                    < events.index(f"wait-restarted-{cluster}")
                    < events.index(f"delete-{cluster}")
                )


@pytest.mark.parametrize(
    "defect",
    ["none", "ack", "residual", "expired", "prepare", "a-recovery", "a-cleanup"],
)
def test_iso006_recovers_a_during_b_cut_and_cleans_both_on_failure(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    clock = [0.0]
    blocked = [False]
    nodes = [{"name": "node-b", "uid": "node-b-uid", "ready": "True"}]
    containers = {
        "api": {
            "uid": "api-uid",
            "phase": "Running",
            "ready": True,
            "containers": {"api": {"restart_count": 0}},
        }
    }
    a = SimpleNamespace(
        cluster="a",
        evidence_identity=lambda: {"release_id": "release-test", "cluster_id": "a"},
        ready_pods=lambda *args: [{"name": "executor-a"}],
    )
    b = SimpleNamespace(cluster="b", gpu_nodes=lambda: nodes)
    multi = SimpleNamespace(
        cluster_a=SimpleNamespace(cluster_id="a"),
        cluster_b=SimpleNamespace(
            cluster_id="b", gpu_kubeconfig=tmp_path / "gpu-b", gpu_context="context-b"
        ),
        namespace="gpu-system",
        regional=lambda target: a if target.cluster_id == "a" else b,
    )
    settings = offline.Settings(
        multi=multi,
        host_probe_image="image@sha256:" + "a" * 64,
        control_plane_cidrs=("10.0.0.0/24",),
        duration_seconds=60,
        predecessor_path=tmp_path / "prior",
        site_file=tmp_path / "site.yaml",
    )
    expected_directory = tmp_path / "cases" / offline.CASE_ID / "host-probes"

    class Primary:
        def __init__(self, regional: Any, **kwargs: Any) -> None:
            assert regional is a
            assert kwargs["case_dir"] == expected_directory.parent / "cluster-a"

        def prepare(self) -> None:
            events.append("prepare-a")
            if defect == "prepare":
                raise offline.RegionalFixtureError("A preparation failed")

        def recover(self, deadline: float, *, cut_is_active: Any) -> dict[str, Any]:
            assert blocked[0] and cut_is_active()
            events.append("recover-a")
            if defect == "a-recovery":
                raise offline.RegionalFixtureError("A recovery failed")
            return {"state": {"workflow": {"status": "SUCCEEDED"}}}

        def cleanup(self, result: dict[str, Any]) -> list[str]:
            assert blocked[0] is False
            events.append("cleanup-a")
            return ["A workload remains"] if defect == "a-cleanup" else []

    class Host:
        def __init__(self, settings: Any) -> None:
            assert settings.state_directory == expected_directory

        def create(self) -> None:
            events.append("create")

        def execute(self, action: str, *args: str) -> dict[str, Any]:
            events.append(action)
            if action == "block":
                blocked[0] = True
                if defect == "ack":
                    raise offline.RegionalFixtureError("lost block ACK")
                return {"blocked": True}
            blocked[0] = False
            return {
                "blocked": False,
                "residual": False,
                "chain_present": False,
                "jumps": {"OUTPUT": False, "FORWARD": False},
            }

        def cleanup(self) -> dict[str, bool]:
            events.append("cleanup")
            return {"pod": defect == "residual", "configmap": False}

    monkeypatch.setattr(
        offline, "HostProbeSettings", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(offline, "HostProbeFixture", Host)
    monkeypatch.setattr(offline, "PrimaryRecovery", Primary)
    monkeypatch.setattr(
        offline,
        "read_only_preflight",
        lambda *args, **kwargs: {
            "errors": [],
            "nodes_b": nodes,
            "cpu_containers": containers,
            "cpu_pods": {"api": "api-uid"},
        },
    )
    monkeypatch.setattr(
        offline,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )
    monkeypatch.setattr(offline, "cpu_pod_snapshot", lambda *args: {"api": "api-uid"})
    monkeypatch.setattr(
        offline, "control_plane_container_statuses", lambda *args: containers
    )
    monkeypatch.setattr(
        offline,
        "control_plane_pressure",
        lambda *args: {
            "cluster_queue_depth": 0,
            "rejections": {name: 0 for name in REJECTION_COUNTERS},
        },
    )
    monkeypatch.setattr(
        offline,
        "claim_sample",
        lambda region: {
            "status": None,
            "transport_error": "ConnectionRefusedError",
            "transport_failure_kind": "network",
        }
        if region is b and blocked[0]
        else {"status": 200, "latency_seconds": 0.1},
    )
    deadline = datetime.now(timezone.utc) + timedelta(
        seconds=5 if defect == "expired" else 3600
    )
    assert offline.execute_case(settings, tmp_path, 1, deadline) == (
        0 if defect == "none" else 1
    )
    result = json.loads(
        (expected_directory.parent / f"{offline.CASE_ID}.json").read_text()
    )
    if defect == "none":
        assert result["claim_path_component_verdict"] == "PASS"
        assert result["a_recovery_exercised"] is True
        assert (
            events.index("block") < events.index("recover-a") < events.index("unblock")
        )
    if defect not in {"expired", "prepare"}:
        assert events.index("unblock") > events.index("block")
        assert blocked[0] is False
    else:
        assert "block" not in events
    assert events[-1] == "cleanup-a"
