"""Regression tests for the 2026-09-07 regional e2e review fixes.

One test per finding, each written to fail on the pre-review runner and pass on
the fixed one: DESTR-001/002/009/010/012, the managed workload fixture and
``restore_validated_quarantine``.

The runner-internal claims (what ``read_only_preflight``, ``plan_details`` and
``execute_case`` call, and with which arguments) are driven behaviourally in
``test_destructive_preflight_identity.py`` and the per-runner
``test_destr00N_review_fixes.py`` files rather than by reading source text.
"""

from __future__ import annotations

import json
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    EffectiveCapability,
    EffectiveRuntimeProfile,
    Environment,
    FaultIncident,
    IncidentState,
    WorkflowOperation,
)
from gpu_fault.orchestration import OPERATION_CAPABILITY
from gpu_fault.orchestration.workflow_builder import WorkflowBuilder
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import restore_validated_quarantine as restore
from scripts.e2e.regional import run_destr001_gpu_reset as destr001
from scripts.e2e.regional import run_destr002_hyperpod_reboot as destr002
from scripts.e2e.regional import run_destr009_workload_restart as destr009
from scripts.e2e.regional import run_destr010_fabric_manager_restart as destr010
from scripts.e2e.regional import run_destr012_managed_recovery_guard as destr012
from scripts.e2e.regional.managed_workload_fixture import (
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
    heartbeat_healthy,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
)

ROOT = Path(__file__).resolve().parents[2]
PROBE_IMAGE = "registry.example/probe@sha256:" + "a" * 64


def _regional(tmp_path: Path) -> RegionalLiveFixture:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        )
    )


def _workload_settings(
    tmp_path: Path, *, pods: int, gpus: int
) -> ManagedWorkloadSettings:
    manifest = tmp_path / "workload.yaml"
    manifest.write_text(
        "apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: training-a\n",
        encoding="utf-8",
    )
    site = tmp_path / "site.yaml"
    site.write_text("clusters: []\n", encoding="utf-8")
    return ManagedWorkloadSettings(
        manifest=manifest,
        site_file=site,
        job_id="job-a",
        attempt_id="job-a-a001",
        restart_budget=1,
        expected_pods=pods,
        expected_gpu_count=gpus,
    )


# --- item 7: every runner binds evidence identity, reuses focused tests --------
# The preflight/plan_details/execute_case bindings are exercised in
# test_destructive_preflight_identity.py and the per-runner review files.


def test_focused_tests_reuse_the_plan_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = {"passed": True, "returncode": 0, "command": ["pytest"]}
    for module in (destr001, destr002, destr009, destr010, destr012):
        monkeypatch.setattr(module, "reusable_focused_tests", lambda _path: recorded)
        monkeypatch.setattr(
            module.RegionalLiveFixture,
            "run",
            lambda *_a, **_k: pytest.fail("focused pytest must not re-run"),
        )
        result = module.focused_tests(tmp_path, reuse=True)
        assert result["focused_tests_reused"] is True, module.__name__
        assert result["passed"] is True, module.__name__


# --- item 1: DESTR-002 ---------------------------------------------------------


def test_destr002_submission_must_name_exactly_the_target() -> None:
    aliases = {"i-0123", "ip-10-0-0-1.ec2.internal"}
    check = destr002.submission_target_errors

    assert (
        check(
            {"requested_node_identifiers": ["i-0123"]},
            target_aliases=aliases,
            node="node-a",
        )
        == []
    )
    assert (
        check(
            {"requested_node_identifiers": ["node-a"]},
            target_aliases=aliases,
            node="node-a",
        )
        == []
    )
    assert check({}, target_aliases=aliases, node="node-a") == [
        "submission record names no node"
    ]
    # The old overlap test let a record that named the target *and* another
    # node through; one identifier is the contract.
    assert check(
        {"requested_node_identifiers": ["i-0123", "i-9999"]},
        target_aliases=aliases,
        node="node-a",
    ) == ["submission record names 2 nodes, not exactly the target"]
    assert check(
        {"requested_node_identifiers": ["i-9999"]},
        target_aliases=aliases,
        node="node-a",
    ) == ["submission record does not identify the target node"]


def test_destr002_executor_restart_refuses_an_unmatched_lease_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    pods = [
        {"name": "executor-a", "uid": "uid-a", "node": "node-a"},
        {"name": "executor-b", "uid": "uid-b", "node": "node-b"},
    ]
    deletes: list[tuple[Any, ...]] = []

    def kubectl(*args: Any, **_kwargs: Any) -> str:
        deletes.append(args)
        return ""

    monkeypatch.setattr(regional, "ready_pods", lambda *_args: list(pods))
    monkeypatch.setattr(regional, "kubectl", kubectl)

    with pytest.raises(RegionalFixtureError, match="cannot identify the submitting"):
        destr002.restart_executor(
            regional, {"result_details": {"executor_id": "lease/executor-zzz"}}
        )
    with pytest.raises(RegionalFixtureError, match="cannot identify the submitting"):
        destr002.restart_executor(regional, {"result_details": {}})

    assert deletes == [], "no Pod may be deleted on a failed match"


def test_destr002_executor_restart_records_the_match_basis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    snapshots = iter(
        (
            [
                {"name": "executor-a", "uid": "uid-a", "node": "node-a"},
                {"name": "executor-b", "uid": "uid-b", "node": "node-b"},
            ],
            [
                {"name": "executor-b", "uid": "uid-b", "node": "node-b"},
                {"name": "executor-c", "uid": "uid-c", "node": "node-c"},
            ],
        )
    )
    monkeypatch.setattr(regional, "ready_pods", lambda *_args: next(snapshots))
    monkeypatch.setattr(regional, "kubectl", lambda *_args, **_kwargs: "")

    result = destr002.restart_executor(regional, {"last_lease_owner": "executor-a/7"})

    assert result["deleted"]["name"] == "executor-a"
    assert result["match_basis"]["field"] == "last_lease_owner"
    assert result["match_basis"]["executor_id"] == "executor-a/7"


def test_destr002_duplicate_replay_gate_refuses_without_recorded_key() -> None:
    submitted = {"state": "SUBMITTED", "idempotency_key": "key-a"}

    missing = destr002.duplicate_replay_gate({"result_details": {}}, submitted)
    assert missing is not None and missing["status"] == "NOT_APPLIED"
    assert "does not recompute" in missing["reason"]

    not_submitted = destr002.duplicate_replay_gate(
        {"result_details": {"submission_idempotency_key": "key-a"}},
        {"state": "INTENDED"},
    )
    assert not_submitted is not None and "not SUBMITTED" in not_submitted["reason"]

    other_key = destr002.duplicate_replay_gate(
        {"result_details": {"submission_idempotency_key": "key-b"}}, submitted
    )
    assert other_key is not None and "differs" in other_key["reason"]

    assert (
        destr002.duplicate_replay_gate(
            {"result_details": {"submission_idempotency_key": "key-a"}}, submitted
        )
        is None
    )


# --- item 2: DESTR-001 ---------------------------------------------------------


class _Log:
    def __init__(self) -> None:
        self.entries: list[str] = []

    def index(self, entry: str) -> int:
        return self.entries.index(entry)


def _destr001_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    log: _Log,
    sampler_start_fails: bool,
) -> destr001.Settings:
    regional = _regional(tmp_path)
    baseline = {
        "gpu_inventory": [{"pci_bdf": "0000:01:00.0"}],
        "compute_clients": [],
        "quiesce_states": [],
        "kmsg_writable": True,
        "ledger": [],
        "gpu_fault_timers": [],
        "services": {},
    }

    class FakeHost:
        def __init__(self, _settings: Any) -> None:
            self.host_script = "/run/probe.py"

        def create(self) -> None:
            log.entries.append("host:create")

        def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
            command = arguments[0]
            log.entries.append(f"host:{command}")
            if command == "start-reset-sampler" and sampler_start_fails:
                raise RegionalFixtureError("exec lost after the unit was created")
            if command == "snapshot":
                return dict(baseline)
            return {"command": command}

        def cleanup(self) -> dict[str, bool]:
            log.entries.append("host:cleanup")
            return {}

    class FakeRegional:
        def __init__(self, _settings: Any) -> None:
            pass

        def node_snapshot(self, _node: str) -> dict[str, Any]:
            return {
                "ready": "True",
                "unschedulable": False,
                "ownership_annotations": {},
            }

        def wait_for_workflow(self, **_kwargs: Any) -> dict[str, Any]:
            log.entries.append("regional:wait_for_workflow")
            return {"incident": {"incident_id": "inc-1"}, "workflow": {}}

        def provider_events(self, *_args: Any) -> list[dict[str, str]]:
            log.entries.append("regional:provider_events")
            return []

        @staticmethod
        def provider_events_provisional(_ended_at: datetime) -> bool:
            return True

        def cpu_blast_snapshot(self) -> dict[str, Any]:
            return {}

    preflight = {
        "errors": [],
        "release_id": "release-a",
        "evidence_identity": {"release_id": "release-a", "cluster_id": "cluster-a"},
        "node": {"gpu_allocatable": "1", "uid": "uid-a"},
        "store": {},
        "cpu_blast": {},
        "focused_tests": {"passed": True, "focused_tests_reused": True},
    }
    monkeypatch.setattr(destr001, "read_only_preflight", lambda *_a, **_k: preflight)
    monkeypatch.setattr(destr001, "verify_plan_identity", lambda *_a, **_k: None)
    monkeypatch.setattr(destr001, "RegionalLiveFixture", FakeRegional)
    monkeypatch.setattr(destr001, "HostProbeFixture", FakeHost)
    return destr001.Settings(
        regional=regional.settings,
        node="node-a",
        host_probe_image=PROBE_IMAGE,
        predecessor_path=tmp_path / "predecessor.json",
    )


def test_destr001_stops_a_sampler_whose_start_did_not_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    settings = _destr001_fakes(tmp_path, monkeypatch, log=log, sampler_start_fails=True)
    run_dir = tmp_path / "run-a"

    exit_code = destr001.execute_case(
        settings, run_dir, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )

    assert exit_code == 1
    # The start raised, so the unit may exist on the host: the stop must run.
    assert "host:stop-reset-sampler" in log.entries, log.entries
    result = json.loads(
        (run_dir / "cases" / destr001.CASE_ID / f"{destr001.CASE_ID}.json").read_text()
    )
    assert "sampler_cleanup_error" not in result


def test_destr001_stops_the_sampler_after_the_post_reset_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    settings = _destr001_fakes(
        tmp_path, monkeypatch, log=log, sampler_start_fails=False
    )
    run_dir = tmp_path / "run-b"

    destr001.execute_case(
        settings, run_dir, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )

    stops = [entry for entry in log.entries if entry == "host:stop-reset-sampler"]
    assert len(stops) == 1, log.entries
    snapshots = [i for i, entry in enumerate(log.entries) if entry == "host:snapshot"]
    assert snapshots[-1] < log.index("host:stop-reset-sampler")
    assert log.index("host:stop-reset-sampler") < log.index("regional:provider_events")
    assert log.index("host:stop-reset-sampler") < log.index("host:restore-quiesce")
    result = json.loads(
        (run_dir / "cases" / destr001.CASE_ID / f"{destr001.CASE_ID}.json").read_text()
    )
    assert result["release_id"] == "release-a"
    assert result["cluster_id"] == "cluster-a"
    assert result["provider_events_provisional"] is True
    assert result["focused_tests_reused"] is True


# --- item 4: DESTR-009 and the managed workload fixture -------------------------


@pytest.mark.parametrize(
    ("text", "world_size", "expected"),
    [
        ("HEARTBEAT rank=0 step=3 all_reduce=300.0 elapsed=6.1", 24, True),
        ("HEARTBEAT rank=0 step=3 all_reduce=3.0 elapsed=6.1", 24, False),
        ("HEARTBEAT rank=0 step=3 all_reduce=garbage", 24, False),
        ("SUCCESS rank=5/24 all_reduce=300.0", 24, True),
        ("SUCCESS rank=0/2", 24, False),
        ("HEARTBEAT step=1 all_reduce=1.0", 1, True),
    ],
)
def test_heartbeat_health_parses_the_all_reduce_value(
    text: str, world_size: int, expected: bool
) -> None:
    assert heartbeat_healthy(text, world_size=world_size) is expected


def test_heartbeat_logs_keep_loss_lines_without_crowding_out_heartbeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    fixture = ManagedWorkloadFixture(
        regional, _workload_settings(tmp_path, pods=1, gpus=1)
    )
    lines = ["HEARTBEAT step=0 all_reduce=1.0"]
    lines += [f"rank=0 step={i} loss=0.{i:03d}" for i in range(40)]
    lines += ["noise line", "SUCCESS rank=0/1 all_reduce=1.0"]
    monkeypatch.setattr(regional, "kubectl", lambda *_a, **_k: "\n".join(lines))

    logs = fixture.heartbeat_logs([{"name": "pod-a"}])["pod-a"]

    assert "HEARTBEAT step=0" in logs
    assert "SUCCESS rank=0/1" in logs
    assert "loss=0.039" in logs
    assert "noise line" not in logs


def test_wait_restarted_reads_logs_once_after_the_pods_are_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    old_pod = {
        "name": "p-old",
        "uid": "old",
        "node": "n1",
        "phase": "Running",
        "ready": True,
    }
    new_pod = {
        "name": "p-new",
        "uid": "new",
        "node": "n1",
        "phase": "Running",
        "ready": True,
    }
    calls = {"pods": 0, "logs": 0}

    class Fixture(ManagedWorkloadFixture):
        def pods(self) -> list[dict[str, Any]]:
            calls["pods"] += 1
            return [old_pod] if calls["pods"] <= 3 else [new_pod]

        def workload(self) -> dict[str, Any]:
            return {
                "metadata": {"uid": "w", "annotations": {}, "labels": {}},
                "spec": {},
            }

        def heartbeat_logs(self, pods: list[dict[str, Any]]) -> dict[str, str]:
            calls["logs"] += 1
            return {str(p["name"]): "HEARTBEAT step=1 all_reduce=1.0" for p in pods}

    fixture = Fixture(regional, _workload_settings(tmp_path, pods=1, gpus=1))
    monkeypatch.setattr(time_module, "sleep", lambda _s: None)

    result = fixture.wait_restarted({"old"}, timeout_seconds=30, poll_seconds=0)

    assert result["pods"] == [new_pod]
    assert calls["pods"] >= 4
    assert calls["logs"] == 1, "logs are read once the UIDs changed, not per poll"


def test_image_prewarm_skips_nodes_that_already_hold_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    prewarm = ImagePrewarmFixture(regional, case_id="CASE", run_id="run-a")
    digest = prewarm.image.rsplit("@", 1)[-1]
    applied: list[str] = []
    waited: list[str] = []

    def kubectl(_plane: str, *arguments: str, **kwargs: Any) -> str:
        if arguments[:2] == ("get", "node"):
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "node-a"},
                            "status": {"images": [{"names": [f"repo@{digest}"]}]},
                        },
                        {"metadata": {"name": "node-b"}, "status": {"images": []}},
                    ]
                }
            )
        if arguments[0] == "apply":
            applied.append(json.loads(kwargs["input_text"])["spec"]["nodeName"])
            return ""
        if arguments[0] == "wait":
            waited.append(arguments[2])
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(regional, "kubectl", kubectl)

    outcome = prewarm.create(["node-a", "node-b"])

    assert outcome == {"skipped": ["node-a"], "created": ["node-b"]}
    assert applied == ["node-b"]
    assert len(waited) == 1


def test_destr009_wait_loops_take_the_single_queue_sample(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    class Regional:
        def __init__(self, phase: str) -> None:
            self.phase = phase

        def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs)
            return {
                "observations": [
                    {
                        "workload_phase": self.phase,
                        "containers": [{"gpu_count": 1, "gpu_uuids": ["GPU-1"]}],
                    }
                ],
                "event": {"event_id": "e"},
                "workflow": {"request_id": "wf-a", "status": "SUCCEEDED"},
                "commands": [],
            }

    settings = cast(
        Any, type("S", (), {"job_id": "job-a", "attempt_id": "job-a-a001"})()
    )
    destr009.wait_observation(
        cast(Any, Regional("RUNNING")), settings, node="node-a", expected_gpu_count=1
    )
    destr009.wait_for_cleanup_quiescence(
        regional=cast(Any, Regional("FAILED")),
        node="node-a",
        marker="m",
        observed_after=datetime.now(timezone.utc),
        job_id="job-a",
        attempt_id="job-a-a001",
        case_dir=tmp_path,
        timeout_seconds=1,
        quiet_seconds=0,
        poll_seconds=0,
        workflow_request_ids=["wf-a"],
    )

    assert all(item["queue_attempts"] == 1 for item in seen), seen
    assert seen[-1]["workflow_request_ids"] == ["wf-a"]


def test_log_write_snapshot_is_inconclusive_without_lines(tmp_path: Path) -> None:
    outputs = {"pod-a": "", "pod-b": "2026-09-07 INFO gpu_fault.api served /healthz"}

    class Regional:
        def ready_pods(self, _plane: str, app: str) -> list[dict[str, Any]]:
            return [{"name": name} for name in outputs]

        def kubectl(self, _plane: str, *arguments: str, **_kwargs: Any) -> str:
            return outputs[arguments[1]]

    logs = destr009.log_write_snapshot(
        cast(Any, Regional()),
        plane="cpu",
        apps=("gpu-fault-api-ha",),
        since=datetime.now(timezone.utc),
        workload_name="training-a",
    )
    assert logs["verdict"] == "INCONCLUSIVE"
    assert logs["inconclusive"] == ["gpu-fault-api-ha/pod-a"]
    assert destr009.log_write_errors(logs, "control-plane") == [
        "control-plane logs are INCONCLUSIVE: no lines from gpu-fault-api-ha/pod-a"
    ]

    outputs["pod-a"] = '{"verb":"patch","resource":"pytorchjobs","name":"training-a"}'
    logs = destr009.log_write_snapshot(
        cast(Any, Regional()),
        plane="cpu",
        apps=("gpu-fault-api-ha",),
        since=datetime.now(timezone.utc),
        workload_name="training-a",
    )
    assert logs["verdict"] == "SUSPICIOUS"
    assert destr009.log_write_errors(logs, "control-plane") == [
        "control-plane logs show a Kubernetes workload write"
    ]

    outputs["pod-a"] = "2026-09-07 INFO gpu_fault.processor accepted event"
    logs = destr009.log_write_snapshot(
        cast(Any, Regional()),
        plane="cpu",
        apps=("gpu-fault-api-ha",),
        since=datetime.now(timezone.utc),
        workload_name="training-a",
    )
    assert logs["verdict"] == "CLEAN"
    assert destr009.log_write_errors(logs, "control-plane") == []


def test_destr009_evidence_records_the_step_owners() -> None:
    state = {
        "workflow": {
            "official_steps": [
                {
                    "operation": "FREEZE_EVIDENCE",
                    "execution_owner": "gpu-fault-control-plane",
                },
                {
                    "operation": "STOP_WORKLOADS",
                    "execution_owner": "gpu-fault-kubernetes-adapter",
                    "node_ids": ["n"],
                },
            ]
        }
    }
    assert destr009.workflow_official_steps(state) == [
        {"operation": "FREEZE_EVIDENCE", "execution_owner": "gpu-fault-control-plane"},
        {
            "operation": "STOP_WORKLOADS",
            "execution_owner": "gpu-fault-kubernetes-adapter",
        },
    ]


# --- item 3: DESTR-012 ---------------------------------------------------------


def test_destr012_group_a_reads_the_owners_from_destr009_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "GF-REGIONAL-DESTR-009.json"
    predecessor = {"evidence_valid": True, "verdict": "PASS"}
    steps = [
        {"operation": "FREEZE_EVIDENCE", "execution_owner": "gpu-fault-control-plane"},
        {
            "operation": "STOP_WORKLOADS",
            "execution_owner": "gpu-fault-kubernetes-adapter",
        },
        {
            "operation": "RESTART_WORKLOAD",
            "execution_owner": "gpu-fault-kubernetes-adapter",
        },
    ]
    path.write_text(
        json.dumps(
            {
                "case_id": "GF-REGIONAL-DESTR-009",
                "verdict": "PASS",
                "release_id": "release-a",
                "workflow_request_id": "wf-9",
                "workflow_official_steps": steps,
            }
        ),
        encoding="utf-8",
    )

    group_a = destr012.group_a_from_evidence(path, predecessor)
    assert group_a["verdict"] == "PASS", group_a
    assert group_a["group_a_source"] == "GF-REGIONAL-DESTR-009 evidence"
    assert group_a["workflow_request_id"] == "wf-9"

    steps[2]["execution_owner"] = "gpu-fault-hyperpod-adapter"
    path.write_text(
        json.dumps({"verdict": "PASS", "workflow_official_steps": steps}),
        encoding="utf-8",
    )
    assert destr012.group_a_from_evidence(path, predecessor)["errors"] == [
        "group A RESTART_WORKLOAD owner is not Kubernetes adapter"
    ]

    path.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
    stale = destr012.group_a_from_evidence(path, predecessor)
    assert stale["verdict"] == "FAIL"
    assert "lacks workflow_official_steps" in stale["error"]
    assert "--rerun-group-a-workload" in stale["error"]

    skipped = destr012.group_a_from_evidence(
        path, {"evidence_valid": False, "verdict": "SKIPPED_BY_OPERATOR"}
    )
    assert skipped["verdict"] == "FAIL"
    assert "SKIPPED_BY_OPERATOR" in skipped["error"]


def test_destr012_rerun_flag_keeps_the_workload_group_a(tmp_path: Path) -> None:
    arguments = destr012.parser().parse_args(
        ["--run-dir", str(tmp_path), "--rerun-group-a-workload"]
    )
    assert arguments.rerun_group_a_workload is True
    assert (
        destr012.parser()
        .parse_args(["--run-dir", str(tmp_path)])
        .rerun_group_a_workload
        is False
    )


def test_destr012_groups_not_run_is_derived_from_the_verdicts() -> None:
    groups = {
        "B_before": {"verdict": "PASS"},
        "A": {"verdict": "PASS"},
        "D": {"verdict": "PASS"},
        "C": {"verdict": "NOT_RUN"},
        "B_after": {"verdict": "PASS"},
    }
    assert destr012.groups_not_run(groups) == ["C"]
    groups["D"]["verdict"] = "NOT_RUN"
    assert destr012.groups_not_run(groups) == ["D", "C"]


def test_destr012_group_workloads_are_deleted_only_after_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deleted: list[str] = []
    quiescence_calls: list[dict[str, Any]] = []

    class Workload:
        def delete(self) -> None:
            deleted.append("delete")

    case_settings = cast(
        Any, type("S", (), {"job_id": "job-d", "attempt_id": "job-d-a001"})()
    )

    def failing(**kwargs: Any) -> dict[str, Any]:
        quiescence_calls.append(kwargs)
        raise RegionalFixtureError("RESTART_WORKLOAD still PENDING")

    monkeypatch.setattr(destr009, "wait_for_cleanup_quiescence", failing)
    result: dict[str, Any] = {"verdict": "PASS"}
    destr012.delete_after_quiescence(
        cast(Any, object()),
        cast(Any, Workload()),
        result,
        case_settings=case_settings,
        case_dir=tmp_path,
        node="node-a",
        marker="m",
        observed_after=datetime.now(timezone.utc),
        workflow_request_ids=["wf-d"],
    )
    assert deleted == []
    assert result["workload_cleanup_deferred"] is True
    assert result["verdict"] == "FAIL"
    assert quiescence_calls[0]["workflow_request_ids"] == ["wf-d"]
    assert quiescence_calls[0]["job_id"] == "job-d"

    monkeypatch.setattr(
        destr009, "wait_for_cleanup_quiescence", lambda **_k: {"safe_to_delete": True}
    )
    result = {"verdict": "PASS"}
    destr012.delete_after_quiescence(
        cast(Any, object()),
        cast(Any, Workload()),
        result,
        case_settings=case_settings,
        case_dir=tmp_path,
        node="node-a",
        marker="m",
        observed_after=datetime.now(timezone.utc),
        workflow_request_ids=[],
    )
    assert deleted == ["delete"]
    assert result["verdict"] == "PASS"


# --- item 5: DESTR-010 ---------------------------------------------------------


def test_destr010_runs_on_the_shared_fixture() -> None:
    for name in (
        "run",
        "kubectl",
        "ready_pod",
        "cpu_python",
        "STORE_PROBE",
        "CaseError",
    ):
        assert not hasattr(destr010, name), name
    assert not hasattr(destr010, "node_snapshot"), (
        "node_snapshot moved to the shared fixture"
    )
    assert not hasattr(destr010, "business_workloads"), (
        "business_workloads moved to the shared fixture"
    )
    assert not hasattr(destr010, "provider_events"), (
        "provider_events moved to the shared fixture"
    )
    assert "regional" in destr010.Settings.__dataclass_fields__
    # The wait loop never scans raw evidence; only the terminal probe does.
    assert "include_evidence" in destr010.FABRIC_PROBE


def test_destr010_node_state_is_compared_on_scheduling_fields_only() -> None:
    baseline = {
        "uid": "u",
        "ready": "True",
        "boot_id": "b1",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }
    flicker = {**baseline, "ready": "False", "boot_id": "b2"}
    cordoned = {**baseline, "unschedulable": True}

    assert destr010.node_state_errors(baseline, dict(baseline), stage="final") == []
    assert destr010.node_state_errors(baseline, flicker, stage="final") == [
        "final: target node is not Ready"
    ]
    assert destr010.node_state_errors(baseline, cordoned, stage="final") == [
        "final: target node unschedulable differs from baseline"
    ]


def test_destr010_workflow_errors_accept_the_shared_notification_shape() -> None:
    state: dict[str, Any] = {
        "event": {"xid": 45, "evidence_ref": "kmsg://node/boot/1"},
        "decision": {"official_action": "RESTART_FM", "disposition": "EXECUTABLE"},
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [
                {
                    "operation": "FREEZE_EVIDENCE",
                    "execution_owner": "gpu-fault-control-plane",
                },
                {
                    "operation": "RESTART_FABRIC_MANAGER",
                    "execution_owner": "gpu-fault-node-agent",
                },
            ],
            "completed_operations": ["FREEZE_EVIDENCE", "RESTART_FABRIC_MANAGER"],
        },
        "commands": [{"status": "SUCCEEDED", "lease_token_sha256": "a" * 64}],
        "evidence": [{"record_id": "evidence-a"}],
        "notifications": [
            {
                "notification": {
                    "notification_id": "n-a",
                    "category": "ACTION_COMPLETED",
                    "subject": "[GPU Fabric Manager restarted]",
                },
                "result": {"status": "SENT", "provider_message_id": "m-1"},
            }
        ],
    }
    assert destr010.workflow_errors(state) == []
    normalized = destr010.normalize_notifications(state["notifications"])
    assert normalized[0]["status"] == "SENT"
    assert normalized[0]["provider_message_id_present"] is True


def test_destr010_replay_rebuilds_the_command_without_redaction_fields() -> None:
    command = {
        "command_id": "c",
        "lease_token_sha256": "a" * 64,
        "lease_token_length": 27,
        "result_details": {},
    }
    rebuilt = destr010.replayable_command(command)
    assert "lease_token_sha256" not in rebuilt
    assert "lease_token_length" not in rebuilt
    assert rebuilt["lease_token"] is None
    assert rebuilt["command_id"] == "c"


def test_destr010_preflight_reads_the_shared_profile_shape() -> None:
    state = {
        "agent": {
            "lifecycle_state": "ACTIVE",
            "allowed_operations": ["RESTART_FABRIC_MANAGER"],
            "generation": 3,
        },
        "profile": {
            "profile_version": "profile-a",
            "warnings": [],
            "capabilities": [
                {
                    "capability": "fabricManagerRestart",
                    "mode": "OWN",
                    "owner": "gpu-fault-node-agent",
                    "adapter": "node-action",
                }
            ],
        },
        "queue": {"depth": 0},
        "remote_commands": {"open_by_cluster": {}},
    }
    fabric: dict[str, Any] = {"active_workflow_incidents": [], "recent_xid_events": []}
    node = {"ready": "True", "unschedulable": False, "taints": []}

    assert destr010.validate_preflight(state, fabric, node, []) == []
    fabric["active_workflow_incidents"] = [{"incident_id": "i"}]
    assert destr010.validate_preflight(state, fabric, node, []) == [
        "target node already has an active workflow"
    ]


# --- item 6: restore_validated_quarantine --------------------------------------


def _profile(cluster_id: str = "cluster-a") -> EffectiveRuntimeProfile:
    return EffectiveRuntimeProfile(
        cluster_id=cluster_id,
        environment=Environment.HYPERPOD_EKS,
        profile_version="profile-a",
        capabilities=[
            EffectiveCapability(
                capability=CapabilityName.DEEP_DIAGNOSTICS,
                mode=CapabilityMode.DELEGATE,
                owner="site-validation-adapter",
                adapter="regional-cluster-executor",
            ),
            EffectiveCapability(
                capability=CapabilityName.SCHEDULER_DRAIN,
                mode=CapabilityMode.OWN,
                owner="gpu-fault-kubernetes-adapter",
                adapter="regional-cluster-executor",
            ),
            EffectiveCapability(
                capability=CapabilityName.GPU_RESET,
                mode=CapabilityMode.DISABLED,
                owner="nobody",
            ),
        ],
    )


def _quarantined_incident() -> FaultIncident:
    return FaultIncident(
        incident_id="inc-a",
        event_id="xid-a",
        event_type="XID",
        cluster_id="cluster-a",
        node_ids=["node-a"],
        policy_version="610",
        policy_source="NVIDIA",
        state=IncidentState.QUARANTINED,
        fencing_token=4,
    )


def test_restore_workflow_resolves_owners_from_the_stored_profile() -> None:
    store = InMemoryStore()
    profile = _profile()
    store.save_profile(profile)
    store.save_incident(_quarantined_incident())

    incident, workflow = restore.build_restore_workflow(
        store,
        incident_id="inc-a",
        node_id="node-a",
        reason="validated restore",
        runtime_profile_version="profile-a",
    )

    owners = [step.execution_owner for step in workflow.official_steps]
    assert owners == [
        "site-validation-adapter",
        "site-validation-adapter",
        "site-validation-adapter",
        "gpu-fault-kubernetes-adapter",
    ]
    assert [step.operation for step in workflow.official_steps] == list(
        restore.RESTORE_OPERATIONS
    )
    assert workflow.fencing_token == 4
    assert incident.state is IncidentState.ACTION_PENDING
    assert incident.workflow_request_id == workflow.request_id
    assert incident.reasons[-1] == "validated restore"
    # The same answer the compiler gives for the same profile.
    for operation in restore.RESTORE_OPERATIONS:
        assert restore.resolve_execution_owner(
            profile, operation
        ) == WorkflowBuilder.owner(
            cast(Any, None), profile, OPERATION_CAPABILITY[operation]
        )


def test_restore_workflow_validates_the_profile_version() -> None:
    store = InMemoryStore()
    store.save_incident(_quarantined_incident())
    store.save_profile(
        _profile(cluster_id="cluster-b").model_copy(
            update={"profile_version": "foreign"}
        )
    )

    with pytest.raises(ValueError, match="is not stored"):
        restore.build_restore_workflow(
            store,
            incident_id="inc-a",
            node_id="node-a",
            reason="r",
            runtime_profile_version="missing",
        )
    with pytest.raises(ValueError, match="belongs to cluster cluster-b"):
        restore.build_restore_workflow(
            store,
            incident_id="inc-a",
            node_id="node-a",
            reason="r",
            runtime_profile_version="foreign",
        )


def test_restore_workflow_refuses_a_profile_without_an_executable_owner() -> None:
    profile = _profile().model_copy(
        update={
            "capabilities": [
                EffectiveCapability(
                    capability=CapabilityName.DEEP_DIAGNOSTICS,
                    mode=CapabilityMode.OBSERVE,
                    owner="watcher",
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="no executable owner for deepDiagnostics"):
        restore.resolve_execution_owner(profile, WorkflowOperation.VALIDATE_GPU)
