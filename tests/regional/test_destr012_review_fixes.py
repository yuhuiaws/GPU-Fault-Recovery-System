"""Behavioural DESTR-012 runner contracts (replacing source-text checks).

Each test drives ``execute_case``, ``run_group_a``/``run_group_d`` or
``executor_log_snapshot`` with fakes and asserts what the runner did, not how
it is spelled.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.e2e.regional import run_destr009_workload_restart as destr009
from scripts.e2e.regional import run_destr012_managed_recovery_guard as destr012
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)

RUNTIME_IDENTITY = {"cpu": {"gpu-fault-api-ha": {"generation": 1}}}
CANDIDATES = [
    {"name": f"node-{index}", "uid": f"uid-{index}"} for index in ("a", "b", "c")
]
PREFLIGHT_ERRORS_FREE_GROUP_B = {
    "profile": {"profile_sha256s": ["p" * 64]},
    "workloads": {},
    "errors": [],
}


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


def _settings(
    tmp_path: Path, *, rerun_group_a_workload: bool = False
) -> destr012.Settings:
    site = tmp_path / "site.yaml"
    site.write_text("clusters: []\n", encoding="utf-8")
    manifests = {}
    for group, kind in (("a", "PyTorchJob"), ("d", "Job")):
        path = tmp_path / f"{group}.yaml"
        path.write_text(
            f"apiVersion: batch/v1\nkind: {kind}\nmetadata:\n  name: training-{group}\n",
            encoding="utf-8",
        )
        manifests[group] = path
    return destr012.Settings(
        regional=_regional(tmp_path).settings,
        site_file=site,
        a_manifest=manifests["a"],
        d_manifest=manifests["d"],
        a_job_id="job-a",
        a_attempt_id="job-a-a001",
        d_job_id="job-d",
        d_attempt_id="job-d-a001",
        predecessor_path=tmp_path / "GF-REGIONAL-DESTR-009.json",
        rerun_group_a_workload=rerun_group_a_workload,
    )


class _Log:
    """Call log shared by every fake of one test."""

    def __init__(self) -> None:
        self.entries: list[str] = []


class FakeWorkload:
    """A ``ManagedWorkloadFixture`` stand-in that records deletes."""

    def __init__(self, log: _Log, name: str = "training-x") -> None:
        self.log = log
        self.name = name
        self.resource = "job"

    def delete(self) -> None:
        self.log.entries.append(f"workload:{self.name}:delete")

    def submit(self) -> dict[str, Any]:
        self.log.entries.append(f"workload:{self.name}:submit")
        return {"submitted": True}

    def wait_running(self, timeout_seconds: int = 900) -> dict[str, Any]:
        return {
            "pods": [{"uid": "uid-old", "node": "node-a"}],
            "workload": {"annotations": {}, "suspend": False},
        }

    def annotate_auto_resume(self, value: str | None) -> None:
        self.log.entries.append(f"workload:{self.name}:annotate={value}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "pods": [{"uid": "uid-old"}],
            "workload": {"annotations": {}, "suspend": False},
        }

    def wait_restarted(
        self, source_uids: set[str], timeout_seconds: int = 900
    ) -> dict[str, Any]:
        return {"pods": [{"uid": "uid-new"}]}


class FakeRegional:
    """The slice of ``RegionalLiveFixture`` the DESTR-012 groups touch."""

    def __init__(
        self,
        log: _Log,
        *,
        workflow_states: list[dict[str, Any]] | None = None,
        executor_logs: str = "",
    ) -> None:
        self.log = log
        self.workflow_states = list(workflow_states or [])
        self.executor_logs = executor_logs
        self.provider_window_ends: list[datetime] = []

    # --- execute_case -----------------------------------------------------
    def verify_runtime_identity(self, _expected: Any, **kwargs: Any) -> None:
        self.log.entries.append(f"regional:verify:{kwargs['stage']}")

    def provider_events(self, *_args: Any) -> list[dict[str, Any]]:
        self.log.entries.append("regional:provider_events")
        return []

    def provider_events_provisional(self, window_end: datetime) -> bool:
        self.provider_window_ends.append(window_end)
        return True

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {"nodes": ["cpu-1"]}

    def kubectl(self, _plane: str, *arguments: str, **_kwargs: Any) -> str:
        self.log.entries.append("regional:kubectl:" + " ".join(arguments[:3]))
        if arguments[0] == "logs":
            return self.executor_logs
        return ""

    # --- groups -----------------------------------------------------------
    def node_metadata(self, _node: str) -> dict[str, Any]:
        return {"product": "NVIDIA H100"}

    def post_xid_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.log.entries.append(f"regional:post_xid:{payload['record_id']}")
        return {"receipt": {"status": 200}}

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.log.entries.append(f"regional:wait_for_workflow:{kwargs['marker']}")
        return self.workflow_states.pop(0)

    def ready_pods(self, _plane: str, app: str) -> list[dict[str, Any]]:
        return [{"name": f"{app}-pod-a"}]


class FakePrewarm:
    def __init__(self, log: _Log, **_kwargs: Any) -> None:
        self.log = log

    def create(self, nodes: list[str]) -> dict[str, list[str]]:
        return {"skipped": [], "created": list(nodes)}

    def cached_nodes(self) -> list[str]:
        return [item["name"] for item in CANDIDATES]

    def cleanup(self) -> dict[str, bool]:
        self.log.entries.append("prewarm:cleanup")
        return {}


def _preflight(*, reused: bool = True) -> dict[str, Any]:
    return {
        "errors": [],
        "release_id": "release-a",
        "evidence_identity": {"release_id": "release-a", "cluster_id": "cluster-a"},
        "gpu_nodes": CANDIDATES,
        "candidate_nodes": CANDIDATES,
        "gpu_workloads": [],
        "store": {"profile": {"profile_version": "profile-a"}},
        "runtime_identity": RUNTIME_IDENTITY,
        "group_b": PREFLIGHT_ERRORS_FREE_GROUP_B,
        "focused_tests": {"passed": True, "focused_tests_reused": reused},
        "cpu_blast": {"nodes": ["cpu-1"]},
        "predecessor": {"valid": True, "evidence_valid": True, "verdict": "PASS"},
    }


def _write_plan(case_dir: Path, preflight: dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "plan.json").write_text(
        json.dumps(
            {
                "details": {
                    "preflight_identity": {
                        "release_id": preflight["release_id"],
                        "runtime_profile_version": "profile-a",
                        "profile_sha256s": preflight["group_b"]["profile"][
                            "profile_sha256s"
                        ],
                        "candidate_node_uids": sorted(
                            item["uid"] for item in CANDIDATES
                        ),
                        "runtime_identity": preflight["runtime_identity"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _execute_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    log: _Log,
    rerun_group_a_workload: bool = False,
    group_a_verdict: str = "PASS",
    group_d_verdict: str = "PASS",
) -> tuple[destr012.Settings, Path, dict[str, Any]]:
    settings = _settings(tmp_path, rerun_group_a_workload=rerun_group_a_workload)
    run_dir = tmp_path / "run-1"
    case_dir = run_dir / "cases" / destr012.CASE_ID
    preflight = _preflight()
    _write_plan(case_dir, preflight)
    calls: dict[str, Any] = {"preflight": [], "group_b_audit": 0}

    def read_only_preflight(
        _settings: Any, _case_dir: Path, **kwargs: Any
    ) -> dict[str, Any]:
        calls["preflight"].append(kwargs)
        return preflight

    def run_group_a(*_args: Any) -> dict[str, Any]:
        log.entries.append("group_a:workload")
        return {"group": "A", "verdict": group_a_verdict, "group_a_source": "rerun"}

    def group_a_from_evidence(
        path: Path, predecessor: dict[str, Any]
    ) -> dict[str, Any]:
        log.entries.append("group_a:evidence")
        calls["group_a_from_evidence"] = (path, predecessor)
        return {
            "group": "A",
            "verdict": group_a_verdict,
            "group_a_source": destr012.GROUP_A_EVIDENCE_SOURCE,
        }

    def run_group_d(*_args: Any) -> dict[str, Any]:
        log.entries.append("group_d:workload")
        return {"group": "D", "verdict": group_d_verdict}

    def group_b_audit(_regional: Any, *, profile_version: str) -> dict[str, Any]:
        calls["group_b_audit"] += 1
        return {"profile": {}, "workloads": {}, "errors": []}

    regional = FakeRegional(log)
    monkeypatch.setattr(destr012, "read_only_preflight", read_only_preflight)
    monkeypatch.setattr(destr012, "RegionalLiveFixture", lambda _s: regional)
    monkeypatch.setattr(
        destr012, "ImagePrewarmFixture", lambda _r, **kw: FakePrewarm(log, **kw)
    )
    monkeypatch.setattr(
        destr012, "ManagedWorkloadFixture", lambda _r, s: FakeWorkload(log, s.job_id)
    )
    monkeypatch.setattr(destr012, "run_group_a", run_group_a)
    monkeypatch.setattr(destr012, "group_a_from_evidence", group_a_from_evidence)
    monkeypatch.setattr(destr012, "run_group_d", run_group_d)
    monkeypatch.setattr(destr012, "group_b_audit", group_b_audit)
    return settings, run_dir, calls


def _evidence(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "cases" / destr012.CASE_ID / f"{destr012.CASE_ID}.json"
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _window_end() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=1)


# --- execute_case ---------------------------------------------------------------


def test_execute_reads_group_a_from_evidence_unless_the_rerun_flag_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    settings, run_dir, calls = _execute_fakes(tmp_path, monkeypatch, log=log)

    exit_code = destr012.execute_case(settings, run_dir, 1, _window_end())

    assert exit_code == 0, _evidence(run_dir)
    assert "group_a:evidence" in log.entries, log.entries
    assert "group_a:workload" not in log.entries, log.entries
    path, predecessor = calls["group_a_from_evidence"]
    assert path == settings.predecessor_path
    assert predecessor == _preflight()["predecessor"]
    assert _evidence(run_dir)["group_a_source"] == destr012.GROUP_A_EVIDENCE_SOURCE


def test_execute_rerun_flag_restarts_the_group_a_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    settings, run_dir, _calls = _execute_fakes(
        tmp_path, monkeypatch, log=log, rerun_group_a_workload=True
    )

    exit_code = destr012.execute_case(settings, run_dir, 1, _window_end())

    assert exit_code == 0, _evidence(run_dir)
    assert "group_a:workload" in log.entries, log.entries
    assert "group_a:evidence" not in log.entries, log.entries
    assert _evidence(run_dir)["group_a_source"] == "rerun"


def test_execute_records_groups_not_run_from_the_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    settings, run_dir, _calls = _execute_fakes(tmp_path, monkeypatch, log=log)

    destr012.execute_case(settings, run_dir, 1, _window_end())

    evidence = _evidence(run_dir)
    assert set(evidence["groups"]) == {"B_before", "A", "D", "C", "B_after"}
    assert evidence["groups_not_run"] == destr012.groups_not_run(evidence["groups"])
    # Group C is the only planned NOT_RUN on a passing run; A, D and both B
    # audits carry real verdicts.
    assert evidence["groups_not_run"] == ["C"]
    assert evidence["groups"]["C"]["verdict"] == "NOT_RUN"


def test_execute_binds_identity_and_the_reused_focused_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    settings, run_dir, calls = _execute_fakes(tmp_path, monkeypatch, log=log)

    destr012.execute_case(settings, run_dir, 1, _window_end())

    assert calls["preflight"] == [{"reuse_focused_tests": True}]
    evidence = _evidence(run_dir)
    assert evidence["verdict"] == "PASS", evidence
    assert evidence["release_id"] == "release-a"
    assert evidence["cluster_id"] == "cluster-a"
    assert evidence["focused_tests_reused"] is True
    # No provider mutation inside the CloudTrail delivery window is provisional.
    assert evidence["provider_events"] == []
    assert evidence["provider_events_provisional"] is True


def test_execute_finally_checks_residuals_instead_of_deleting_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    settings, run_dir, calls = _execute_fakes(
        tmp_path, monkeypatch, log=log, rerun_group_a_workload=True
    )

    destr012.execute_case(settings, run_dir, 1, _window_end())

    deletes = [entry for entry in log.entries if entry.endswith(":delete")]
    assert deletes == [], "execute_case must not delete a group workload itself"
    residual_reads = [
        entry for entry in log.entries if entry.startswith("regional:kubectl:get")
    ]
    assert len(residual_reads) == 2, log.entries
    evidence = _evidence(run_dir)
    assert evidence["workload_residuals"] == {"A": False, "D": False}
    # The execute preflight's group B audit is B_before; only B_after re-reads.
    assert calls["group_b_audit"] == 1
    assert evidence["groups"]["B_before"]["reused_from"] == "execute preflight"
    assert evidence["groups"]["B_before"]["errors"] == []


def test_execute_residual_check_skips_group_a_when_it_was_read_from_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    settings, run_dir, _calls = _execute_fakes(tmp_path, monkeypatch, log=log)

    destr012.execute_case(settings, run_dir, 1, _window_end())

    assert _evidence(run_dir)["workload_residuals"] == {"D": False}


# --- run_group_a / run_group_d ----------------------------------------------------


def _group_fakes(
    monkeypatch: pytest.MonkeyPatch, *, log: _Log, workload: FakeWorkload
) -> list[dict[str, Any]]:
    """Spy ``delete_after_quiescence`` and stub the DESTR-009 helpers."""

    quiescence: list[dict[str, Any]] = []

    def delete_after_quiescence(
        _regional: Any, target: Any, result: dict[str, Any], **kwargs: Any
    ) -> None:
        log.entries.append("delete_after_quiescence")
        quiescence.append({"workload": target, "result": result, **kwargs})

    monkeypatch.setattr(destr012, "delete_after_quiescence", delete_after_quiescence)
    monkeypatch.setattr(destr012, "ManagedWorkloadFixture", lambda _r, _s: workload)
    monkeypatch.setattr(
        destr009,
        "wait_observation",
        lambda *_a, **_k: {
            "workload_ids": ["wl-1"],
            "runtime_profile_version": "profile-a",
        },
    )
    monkeypatch.setattr(destr009, "workflow_errors", lambda *_a, **_k: [])
    return quiescence


def _blocked_state() -> dict[str, Any]:
    return {
        "workflow": {
            "status": "FAILED",
            "request_id": "wf-blocked",
            "step_executions": [
                {
                    "operation": "STOP_WORKLOADS",
                    "status": "FAILED",
                    "error": "enable-job-auto-resume is enabled on training-d",
                    "details": {
                        "managed_job_recovery_workloads": ["wl-1"],
                        "required_annotation": destr012.AUTO_RESUME_ANNOTATION,
                        "required_annotation_value": "absent or false",
                        "remediation_commands": [
                            f"kubectl annotate job training-d "
                            f"{destr012.AUTO_RESUME_ANNOTATION}-"
                        ],
                    },
                }
            ],
        }
    }


def test_group_a_deletes_its_workload_only_through_the_quiescence_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    workload = FakeWorkload(log, "training-a")
    quiescence = _group_fakes(monkeypatch, log=log, workload=workload)
    state = {
        "workflow": {
            "request_id": "wf-a",
            "official_steps": [
                {
                    "operation": "STOP_WORKLOADS",
                    "execution_owner": destr012.MANAGED_WORKLOAD_OWNER,
                },
                {
                    "operation": "RESTART_WORKLOAD",
                    "execution_owner": destr012.MANAGED_WORKLOAD_OWNER,
                },
            ],
        }
    }
    regional = FakeRegional(log, workflow_states=[state])

    result = destr012.run_group_a(
        _settings(tmp_path), cast(Any, regional), tmp_path / "case", _window_end()
    )

    assert result["verdict"] == "PASS", result
    assert log.entries[-1] == "delete_after_quiescence", log.entries
    assert "workload:training-a:delete" not in log.entries, log.entries
    assert len(quiescence) == 1, quiescence
    assert quiescence[0]["workload"] is workload
    assert quiescence[0]["result"] is result
    assert quiescence[0]["workflow_request_ids"] == ["wf-a"]
    assert quiescence[0]["node"] == "node-a"
    assert quiescence[0]["marker"] == result["marker"]


def test_group_a_still_takes_the_quiescence_gate_when_it_fails_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    workload = FakeWorkload(log, "training-a")
    quiescence = _group_fakes(monkeypatch, log=log, workload=workload)

    def failing_submit() -> dict[str, Any]:
        raise RuntimeError("apply refused")

    monkeypatch.setattr(workload, "submit", failing_submit)

    result = destr012.run_group_a(
        _settings(tmp_path),
        cast(Any, FakeRegional(log)),
        tmp_path / "case",
        _window_end(),
    )

    assert result["verdict"] == "FAIL"
    assert result["error"] == "RuntimeError: apply refused"
    assert "workload:training-a:delete" not in log.entries, log.entries
    assert [item["workload"] for item in quiescence] == [workload]
    # Nothing was injected, so the gate is told there is no workflow to wait on.
    assert quiescence[0]["node"] is None
    assert quiescence[0]["workflow_request_ids"] == []


def test_group_d_deletes_its_workload_only_through_the_quiescence_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    workload = FakeWorkload(log, "training-d")
    quiescence = _group_fakes(monkeypatch, log=log, workload=workload)
    remediated = {"workflow": {"request_id": "wf-retry", "status": "SUCCEEDED"}}
    regional = FakeRegional(
        log,
        workflow_states=[_blocked_state(), remediated],
        executor_logs="2026-09-07 INFO executor idle\n",
    )

    result = destr012.run_group_d(
        _settings(tmp_path), cast(Any, regional), tmp_path / "case", _window_end()
    )

    assert result["verdict"] == "PASS", result
    assert log.entries[-1] == "delete_after_quiescence", log.entries
    assert "workload:training-d:delete" not in log.entries, log.entries
    assert [item["workload"] for item in quiescence] == [workload]
    # The gate follows the most recent injection: the remediated retry.
    assert quiescence[0]["marker"] == result["remediated_marker"]
    assert quiescence[0]["workflow_request_ids"] == ["wf-blocked", "wf-retry"]
    # The annotation is always removed again before the delete.
    assert log.entries.index("workload:training-d:annotate=None") < log.entries.index(
        "delete_after_quiescence"
    )


# --- executor logs share the DESTR-009 inconclusive rule ----------------------------


def test_executor_log_snapshot_is_inconclusive_without_lines(tmp_path: Path) -> None:
    log = _Log()
    regional = FakeRegional(log, executor_logs="")

    logs = destr012.executor_log_snapshot(
        cast(Any, regional), datetime.now(timezone.utc), "training-d"
    )

    assert logs["verdict"] == "INCONCLUSIVE"
    assert logs["inconclusive"] == [
        "gpu-fault-cluster-executor/gpu-fault-cluster-executor-pod-a"
    ]
    assert logs["suspicious"] == []
    # Only the GPU-plane executor Pods are read, nothing on the CPU plane. An
    # empty window read is followed by the silence proof (describe the Pod, then
    # its log history); this fake answers the describe with nothing, so the Pod
    # stays inconclusive and the history is never asked for.
    assert log.entries == [
        "regional:kubectl:logs gpu-fault-cluster-executor-pod-a --since-time",
        "regional:kubectl:get pod gpu-fault-cluster-executor-pod-a",
    ]


def test_executor_log_snapshot_flags_a_workload_write(tmp_path: Path) -> None:
    regional = FakeRegional(
        _Log(), executor_logs='{"verb":"patch","resource":"jobs","name":"training-d"}\n'
    )

    logs = destr012.executor_log_snapshot(
        cast(Any, regional), datetime.now(timezone.utc), "training-d"
    )

    assert logs["verdict"] == "SUSPICIOUS"
    assert destr009.log_write_errors(logs, "group D executor") == [
        "group D executor logs show a Kubernetes workload write"
    ]


def test_group_d_fails_on_inconclusive_executor_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    workload = FakeWorkload(log, "training-d")
    _group_fakes(monkeypatch, log=log, workload=workload)
    remediated = {"workflow": {"request_id": "wf-retry", "status": "SUCCEEDED"}}
    regional = FakeRegional(
        log, workflow_states=[_blocked_state(), remediated], executor_logs=""
    )

    result = destr012.run_group_d(
        _settings(tmp_path), cast(Any, regional), tmp_path / "case", _window_end()
    )

    assert result["verdict"] == "FAIL"
    assert result["errors"] == [
        "group D executor logs are INCONCLUSIVE: no lines from "
        "gpu-fault-cluster-executor/gpu-fault-cluster-executor-pod-a"
    ]
    written = json.loads(
        (tmp_path / "case" / "group-d-executor-logs.json").read_text(encoding="utf-8")
    )
    assert written["verdict"] == "INCONCLUSIVE"
