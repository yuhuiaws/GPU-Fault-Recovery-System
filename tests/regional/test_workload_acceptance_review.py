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


def training_logs(pods: int = 3) -> dict[str, str]:
    """The fixture job's logs for ``pods`` nodes of eight ranks each."""

    world = pods * 8
    marker = world * (world + 1) / 2
    return {
        f"pod-{pod}": "\n".join(
            [
                *(
                    f"rank={rank} step={step} loss={1.0 - step * 0.1}"
                    for rank in range(pod * 8, (pod + 1) * 8)
                    for step in range(3)
                ),
                *(
                    f"SUCCESS rank={rank}/{world} all_reduce={marker}"
                    for rank in range(pod * 8, (pod + 1) * 8)
                ),
            ]
        )
        for pod in range(pods)
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


def _site_namespace(
    tmp_path: Path, regional: Any, cluster_profile: str | None
) -> SimpleNamespace:
    """The site as the runner sees it: base profile ``profile-test``; the target
    cluster pinned to ``cluster_profile`` when given (the submitted job then
    carries that version unless the harness' ``stamped_profile`` says otherwise).
    ``readme_requests`` records what the runner asked the fixture builder."""

    cluster_entry: dict[str, Any] = {"cluster_id": "cluster-a"}
    if cluster_profile:
        cluster_entry["runtime_profile"] = {"version": cluster_profile}
    return SimpleNamespace(
        regional=lambda target: regional,
        namespace="gpu-system",
        site_file=tmp_path / "site.yaml",
        config={
            "runtime_profile": {"version": "profile-test"},
            "clusters": [cluster_entry],
        },
        readme_requests=[],
    )


def baseline_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    defect: str = "",
    nodes: int = 3,
    cluster_profile: str | None = None,
    stamped_profile: str | None = None,
) -> tuple[Any, Any, list[str]]:
    """A cluster of ``nodes`` schedulable GPU nodes running the fixture job.

    ``defect`` ``missing-rank`` drops the last rank's lines from the last Pod
    (the collective marker stays, so the waiter passes and the checks judge);
    ``native-ranks`` renders the manifest for the fixtures' three nodes
    whatever ``nodes`` says, so its expected-critical-ranks disagree.
    ``cluster_profile``/``stamped_profile``: see ``_site_namespace``.
    """

    events: list[str] = []
    effective_profile = cluster_profile or "profile-test"
    logs = training_logs(nodes)
    if defect == "missing-rank":
        last = f"pod-{nodes - 1}"
        dropped = f"rank={nodes * 8 - 1}"
        logs[last] = "\n".join(
            line
            for line in logs[last].splitlines()
            if not (line.startswith(dropped + " ") or f"SUCCESS {dropped}/" in line)
        )
    manifest = yaml.safe_load(runner.BASELINE_MANIFEST.read_text())
    manifest["spec"]["pytorchReplicaSpecs"]["Worker"]["replicas"] = (
        2 if defect == "native-ranks" else nodes - 1
    )
    metadata = inject_metadata(
        manifest,
        job_id="job-test",
        attempt_id="attempt-test",
        runtime_profile=stamped_profile or effective_profile,
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
        for index in range(nodes)
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
                "expected_critical_ranks": nodes,
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
            for index in range(nodes)
        ],
        kubectl=kubectl,
        run=run,
    )
    site = _site_namespace(tmp_path, regional, cluster_profile)

    def store(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if not workload.submitted:
            value = baseline()
            if defect == "dirty":
                value["decision"] = {"event_key": "previous"}
            return value
        return copy.deepcopy(after if workload.deleted else terminal)

    monkeypatch.setattr(
        runner,
        "readme_fixture",
        lambda *args, **kwargs: site.readme_requests.append(kwargs) or workload,
    )
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


def test_a_two_node_cluster_is_judged_and_recorded_by_its_own_topology(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Live 2026-10-05 on the two-node EKS cluster: the job ran 16 ranks while
    the checks still expected the fixtures' 24 and the evidence had no
    training_topology. Every check now reads the same derived topology."""

    site, target, events = baseline_harness(monkeypatch, tmp_path, nodes=2)

    result = runner.run_workload_baseline(
        case_id="GF-REGIONAL-WORKLOAD-001",
        site=site,
        target=target,
        case_dir=tmp_path,
        job_id="job-test",
        attempt_id="attempt-test",
        state_dir=tmp_path,
        attempt=1,
    )

    assert result["verdict"] == "PASS", result
    assert result["collective_errors"] == [] and result["loss_errors"] == []
    assert result["metadata_errors"] == [], "expected-critical-ranks is the node count"
    assert result["training_topology"]["node_count"] == 2
    assert result["training_topology"]["world_size"] == 16
    assert result["checks"]["three_pods_on_three_nodes"] is True, (
        "the check keeps its name; its expectation is the derived node count"
    )
    assert events[-1] == "prewarm-cleanup"


@pytest.mark.parametrize("defect", ["missing-rank", "native-ranks"])
def test_two_node_proof_defects_are_judged_against_the_two_node_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    site, target, _events = baseline_harness(
        monkeypatch, tmp_path, nodes=2, defect=defect
    )

    result = runner.run_workload_baseline(
        case_id="GF-REGIONAL-WORKLOAD-001",
        site=site,
        target=target,
        case_dir=tmp_path,
        job_id="job-test",
        attempt_id="attempt-test",
        state_dir=tmp_path,
        attempt=1,
    )

    assert result["verdict"] == "FAIL"
    assert result["training_topology"]["world_size"] == 16
    if defect == "missing-rank":
        assert (
            "collective ranks do not cover exactly 0..15"
            in (result["collective_errors"])
        )
        assert result["loss_errors"] == ["loss ranks do not cover exactly 0..15"]
    else:
        assert result["metadata_errors"] == [
            "Master expected critical ranks differs",
            "Worker expected critical ranks differs",
        ], "the manifest still carries the fixtures' three; the cluster has two"


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


# -- mixed-provider site: the job carries the target cluster's own profile ---------


def _baseline(site: Any, target: Any, tmp_path: Path) -> dict[str, Any]:
    return runner.run_workload_baseline(
        case_id="GF-REGIONAL-WORKLOAD-001",
        site=site,
        target=target,
        case_dir=tmp_path,
        job_id="job-test",
        attempt_id="attempt-test",
        state_dir=tmp_path,
        attempt=1,
    )


def test_a_cluster_pinned_to_its_own_profile_is_stamped_and_judged_by_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Live (round 8, HyperPod lane of the mixed S5 site): WORKLOAD-001 read the
    site base (eks-v1) as "the" profile, so the job on the hyperpod-v1 cluster
    was stamped eks-v1, its terminal observation carried eks-v1, and the check
    agreed with itself. The runner now stamps and judges by the version the
    target cluster runs and records both versions."""

    site, target, _events = baseline_harness(
        monkeypatch, tmp_path, cluster_profile="hyperpod-v1"
    )

    result = _baseline(site, target, tmp_path)

    assert result["verdict"] == "PASS", result
    (request,) = site.readme_requests
    assert request["runtime_profile_version"] == "hyperpod-v1", (
        "gpu-training-submit must be told the cluster's version explicitly"
    )
    assert request["topology"].node_count == 3
    assert result["metadata_errors"] == []
    assert result["runtime_profile"] == {
        "cluster_id": "cluster-a",
        "site_version": "profile-test",
        "effective_version": "hyperpod-v1",
        "source": "cluster",
    }


def test_a_cluster_on_the_site_profile_is_stamped_with_the_site_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, _events = baseline_harness(monkeypatch, tmp_path)

    result = _baseline(site, target, tmp_path)

    assert result["verdict"] == "PASS", result
    assert site.readme_requests[0]["runtime_profile_version"] == "profile-test"
    assert result["runtime_profile"]["effective_version"] == "profile-test"
    assert result["runtime_profile"]["source"] == "site"


def test_a_job_stamped_with_the_site_base_on_a_pinned_cluster_fails_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, _events = baseline_harness(
        monkeypatch,
        tmp_path,
        cluster_profile="hyperpod-v1",
        stamped_profile="profile-test",
    )

    result = _baseline(site, target, tmp_path)

    assert result["verdict"] == "FAIL"
    assert result["metadata_errors"] == [
        "Master runtime profile differs",
        "Worker runtime profile differs",
    ], "the live r8 shape -- a HyperPod job carrying the plain-EKS version -- fails"
