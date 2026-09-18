from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.training_submit_cli import inject_metadata
from scripts.e2e.regional import run_workload_acceptance as runner
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError


def training_logs() -> dict[str, str]:
    return {
        f"pod-{pod}": "\n".join(
            [
                *(
                    f"rank={rank} step={step} loss={1.0 - step * 0.1}"
                    for rank in range(pod * 8, (pod + 1) * 8)
                    for step in range(3)
                ),
                *(
                    f"SUCCESS rank={rank}/24 all_reduce=300.0"
                    for rank in range(pod * 8, (pod + 1) * 8)
                ),
            ]
        )
        for pod in range(3)
    }


@pytest.mark.parametrize(
    "defect",
    ["missing-rank", "missing-step", "duplicate-step", "flat", "nan", "infinity"],
)
def test_training_requires_every_finite_decreasing_rank_series(defect: str) -> None:
    logs = training_logs()
    assert runner.loss_errors(logs) == []
    if defect == "missing-rank":
        logs["pod-0"] = "\n".join(
            line
            for line in logs["pod-0"].splitlines()
            if not line.startswith("rank=0 ")
        )
    elif defect == "missing-step":
        logs["pod-0"] = logs["pod-0"].replace("rank=0 step=1 loss=0.9\n", "")
    elif defect == "duplicate-step":
        logs["pod-0"] += "\nrank=0 step=1 loss=0.9"
    elif defect == "flat":
        logs["pod-0"] = logs["pod-0"].replace(
            "rank=0 step=2 loss=0.8", "rank=0 step=2 loss=1.0"
        )
    else:
        value = "nan" if defect == "nan" else "1e999"
        logs["pod-0"] = logs["pod-0"].replace(
            "rank=0 step=1 loss=0.9", f"rank=0 step=1 loss={value}"
        )
    assert runner.loss_errors(logs), "incomplete or invalid training evidence passed"


@pytest.mark.parametrize("defect", ["missing", "duplicate", "world", "sum"])
def test_collective_proof_requires_all_twenty_four_ranks(defect: str) -> None:
    logs = training_logs()
    assert runner.collective_errors(logs) == []
    original = "SUCCESS rank=0/24 all_reduce=300.0"
    replacement = {
        "missing": "",
        "duplicate": original + "\n" + original,
        "world": "SUCCESS rank=0/8 all_reduce=300.0",
        "sum": "SUCCESS rank=0/24 all_reduce=300.01",
    }[defect]
    logs["pod-0"] = logs["pod-0"].replace(original, replacement)
    assert runner.collective_errors(logs), "incomplete collective output passed"


@pytest.mark.parametrize("separator", ["", " ", "\n"])
def test_shared_stdout_keeps_all_complete_adjacent_rank_records(separator: str) -> None:
    logs = {
        pod: separator.join(text.splitlines()) for pod, text in training_logs().items()
    }
    assert runner.loss_errors(logs) == []
    assert runner.collective_errors(logs) == []


@pytest.mark.parametrize(
    "defect",
    [
        "missing-loss",
        "duplicate-loss",
        "bad-loss-suffix",
        "nan-loss",
        "empty-loss",
        "incomplete-loss",
        "missing-success",
        "duplicate-success",
        "wrong-sum",
        "incomplete-success",
    ],
)
def test_adjacent_records_never_hide_missing_duplicate_or_corrupt_proof(
    defect: str,
) -> None:
    logs = training_logs()
    lines = logs["pod-0"].splitlines()
    loss = "rank=0 step=1 loss=0.9"
    success = "SUCCESS rank=0/24 all_reduce=300.0"
    if defect == "missing-loss":
        lines.remove(loss)
    elif defect == "duplicate-loss":
        lines.append(loss)
    elif defect == "bad-loss-suffix":
        lines[lines.index(loss)] = loss + "rank=broken"
    elif defect == "nan-loss":
        lines[lines.index(loss)] = "rank=0 step=1 loss=NaN"
    elif defect == "empty-loss":
        lines[lines.index(loss)] = "rank=0 step=1 loss="
    elif defect == "incomplete-loss":
        lines.append("rank=0 step=1 loss=")
    elif defect == "missing-success":
        lines.remove(success)
    elif defect == "duplicate-success":
        lines.append(success)
    elif defect == "wrong-sum":
        lines[lines.index(success)] = success.replace("300.0", "299.0")
    else:
        lines.append("SUCCESS rank=0/24 ")
    logs["pod-0"] = "".join(lines)
    errors = runner.loss_errors(logs) + runner.collective_errors(logs)
    assert errors, "joining stdout records must not weaken the evidence contract"


def baseline() -> dict[str, Any]:
    return {
        "cluster_id": "cluster-a",
        "restart_budget": None,
        "decision": None,
        "observations": [],
        "incidents": [],
        "workflows": [],
        "commands": [],
    }


def baseline_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, defect: str = ""
) -> tuple[Any, Any, list[str]]:
    events: list[str] = []
    logs = training_logs()
    metadata = inject_metadata(
        yaml.safe_load(runner.BASELINE_MANIFEST.read_text()),
        job_id="job-test",
        attempt_id="attempt-test",
        runtime_profile="profile-test",
        restart_budget=1,
    )
    if defect == "metadata":
        del metadata["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["metadata"][
            "labels"
        ]["gpu-fault.io/role"]
    if defect == "loss":
        logs["pod-0"] = logs["pod-0"].replace("rank=0 step=1 loss=0.9\n", "")
    pods = [
        {
            "name": f"pod-{index}",
            "uid": f"uid-{index}",
            "node": f"node-{index}",
            "phase": "Succeeded",
        }
        for index in range(3)
    ]
    containers = [
        {"rank": index, "pod_uid": pod["uid"], "terminated": True, "exit_code": 0}
        for index, pod in enumerate(pods)
    ]
    terminal = {
        **baseline(),
        "observations": [
            {
                "workload_phase": "SUCCEEDED",
                "containers": containers,
                "expected_critical_ranks": 3,
            }
        ],
        "decision": {"event_key": "cluster-a/attempt-test/TrainingAttemptTerminal"},
    }
    after = copy.deepcopy(terminal)
    if defect == "containers":
        after["observations"][0]["containers"].pop()
    if defect == "ranks":
        terminal["observations"][0]["expected_critical_ranks"] = 2

    class Workload:
        resource = "pytorchjob"
        submitted = False
        deleted = False

        def submit(self) -> dict[str, Any]:
            self.submitted = True
            events.append("submit")
            return {}

        def submission_arguments(self) -> list[str]:
            return [str(runner.BASELINE_MANIFEST)]

        def apply_rendered(self, path: Path) -> dict[str, Any]:
            assert path.read_text() == "managed\n"
            completed = regional.run(
                ["kubectl", "apply", "--dry-run=server", "-f", str(path)], check=False
            )
            if completed.returncode:
                raise RegionalFixtureError("server-side dry-run rejected manifest")
            self.submitted = True
            events.append("apply")
            return {
                "create_only_identity": True,
                "server_dry_run_returncode": 0,
                "apply_returncode": 0,
                "returncode": 0,
            }

        def delete(self) -> None:
            if not self.submitted:
                return
            self.deleted = True
            events.append("delete")

        def snapshot(self) -> dict[str, Any]:
            return {"pods": pods, "heartbeat_logs": logs}

        def workload(self) -> dict[str, Any]:
            return metadata

    workload = Workload()

    class Prewarm:
        def create(self, nodes: list[str]) -> None:
            events.append("prewarm")

        def cleanup(self) -> dict[str, bool]:
            events.append("prewarm-cleanup")
            return {"prewarm": False}

    def kubectl(*args: str, **kwargs: Any) -> str:
        if args[:2] == ("gpu", "logs"):
            return logs[args[2]]
        if args[:3] == ("gpu", "get", "pod,pytorchjob"):
            items = (
                [{"kind": "Pod", "metadata": {"name": "remaining", "uid": "uid-r"}}]
                if defect == "residual"
                else []
            )
            return json.dumps({"items": items})
        raise AssertionError(f"unexpected fake kubectl request: {args}")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if Path(command[0]).name == "gpu-fault-workload-annotate":
            events.append("annotate")
            Path(command[command.index("--output") + 1]).write_text("managed\n")
            return subprocess.CompletedProcess(command, 0, "", "")
        if Path(command[0]).name == "gpu-training-submit":
            events.append("submit-dry-run")
            assert "--dry-run" in command, (
                "independent apply called the live submit path"
            )
            return subprocess.CompletedProcess(
                command, 0, "mismatch\n" if defect == "render" else "managed\n", ""
            )
        if "--dry-run=server" in command:
            events.append("server-dry-run")
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected local command: {command}")

    regional = SimpleNamespace(
        settings=SimpleNamespace(
            cluster_id="cluster-a",
            namespace="gpu-system",
            gpu_context="context-a",
            gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        ),
        gpu_workloads=lambda: [],
        gpu_nodes=lambda: [
            {"name": f"node-{index}", "ready": "True", "unschedulable": False}
            for index in range(3)
        ],
        kubectl=kubectl,
        run=run,
    )
    site = SimpleNamespace(
        regional=lambda target: regional,
        namespace="gpu-system",
        site_file=tmp_path / "site.yaml",
        config={"runtime_profile": {"version": "profile-test"}},
    )

    def store(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if not workload.submitted:
            value = baseline()
            if defect == "dirty":
                value["decision"] = {"event_key": "previous"}
            return value
        return copy.deepcopy(after if workload.deleted else terminal)

    monkeypatch.setattr(runner, "readme_fixture", lambda *args, **kwargs: workload)
    monkeypatch.setattr(runner, "training_cli", lambda name: name)
    monkeypatch.setattr(
        runner, "ImagePrewarmFixture", lambda *args, **kwargs: Prewarm()
    )
    monkeypatch.setattr(runner, "workload_store", store)
    monkeypatch.setattr(runner, "watcher_interval_seconds", lambda regional: 1)
    monkeypatch.setattr(runner, "admin_status", lambda state: {"returncode": 0})
    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    return site, SimpleNamespace(cluster_id="cluster-a"), events


@pytest.mark.parametrize(
    "case", ["GF-REGIONAL-WORKLOAD-001", "GF-REGIONAL-WORKLOAD-002"]
)
def test_finite_workload_lifecycle_uses_the_selected_submission_path(
    case: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, events = baseline_harness(monkeypatch, tmp_path)
    result = runner.run_workload_baseline(
        case_id=case,
        site=site,
        target=target,
        case_dir=tmp_path,
        job_id="job-test",
        attempt_id="attempt-test",
        state_dir=tmp_path,
        attempt=1,
    )
    assert result["verdict"] == "PASS", result
    assert events[-1] == "prewarm-cleanup"
    if case.endswith("001"):
        assert events == ["prewarm", "submit", "delete", "prewarm-cleanup"]
    else:
        assert events == [
            "prewarm",
            "annotate",
            "submit-dry-run",
            "server-dry-run",
            "apply",
            "delete",
            "prewarm-cleanup",
        ]


@pytest.mark.parametrize("defect", ["metadata", "loss", "containers", "ranks"])
def test_finite_workload_rejects_incomplete_terminal_or_training_proof(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, events = baseline_harness(monkeypatch, tmp_path, defect=defect)
    result = runner.run_workload_baseline(
        case_id="GF-REGIONAL-WORKLOAD-001",
        site=site,
        target=target,
        case_dir=tmp_path,
        job_id="job-test",
        attempt_id="attempt-test",
        state_dir=None,
        attempt=1,
    )
    assert result["verdict"] == "FAIL"
    assert events[-1] == "prewarm-cleanup"


def test_render_mismatch_never_reaches_live_workload_apply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, events = baseline_harness(monkeypatch, tmp_path, defect="render")
    with pytest.raises(runner.WorkloadCaseError, match="dry-run differ"):
        runner.run_workload_baseline(
            case_id="GF-REGIONAL-WORKLOAD-002",
            site=site,
            target=target,
            case_dir=tmp_path,
            job_id="job-test",
            attempt_id="attempt-test",
            state_dir=tmp_path,
            attempt=1,
        )
    assert events == ["prewarm", "annotate", "submit-dry-run", "prewarm-cleanup"]


def test_dirty_baseline_refuses_before_any_workload_or_prewarm_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, events = baseline_harness(monkeypatch, tmp_path, defect="dirty")
    with pytest.raises(RegionalFixtureError, match="already has control-plane state"):
        runner.run_workload_baseline(
            case_id="GF-REGIONAL-WORKLOAD-001",
            site=site,
            target=target,
            case_dir=tmp_path,
            job_id="job-test",
            attempt_id="attempt-test",
            state_dir=None,
            attempt=1,
        )
    assert events == []


def test_workload_delete_ack_does_not_prove_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, events = baseline_harness(monkeypatch, tmp_path, defect="residual")
    with pytest.raises(runner.WorkloadCaseError) as caught:
        runner.run_workload_baseline(
            case_id="GF-REGIONAL-WORKLOAD-001",
            site=site,
            target=target,
            case_dir=tmp_path,
            job_id="job-test",
            attempt_id="attempt-test",
            state_dir=None,
            attempt=1,
        )
    assert caught.value.outcome["cleanup_errors"], "residual resources were ignored"
    assert events[-1] == "prewarm-cleanup"


@pytest.mark.parametrize("log_result", ["no-pods", "empty", "error"])
def test_unavailable_executor_logs_are_not_a_clean_isolation_window(
    log_result: str,
) -> None:
    def kubectl(*args: str, **kwargs: Any) -> str:
        assert kwargs.get("check", True) is True
        if log_result == "error":
            raise RegionalFixtureError("logs unavailable")
        return ""

    regional = SimpleNamespace(
        ready_pods=lambda *args: []
        if log_result == "no-pods"
        else [{"name": "executor"}],
        kubectl=kubectl,
    )
    with pytest.raises(RegionalFixtureError):
        runner.executor_logs_since(
            regional, since=runner.datetime.now(runner.timezone.utc)
        )


def test_empty_node_and_blast_snapshots_do_not_prove_cleanup() -> None:
    assert runner.gpu_nodes_clean([]) is False
    assert runner.control_plane_blast_errors({}, {}) == [
        "control-plane blast snapshot is incomplete"
    ]
    assert runner.identity_baseline_errors({}) == [
        "control-plane identity baseline is incomplete"
    ]
