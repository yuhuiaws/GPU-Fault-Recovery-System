"""Unit contracts for run_workload_acceptance (WORKLOAD-001/002, ISO-001, E2E-001)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr009_workload_restart as destr009
from scripts.e2e.regional import run_workload_acceptance as workload
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError

ROOT = Path(__file__).resolve().parents[2]


# --- 4a: executor log markers on word boundaries -----------------------------


def test_suspicious_log_lines_match_whole_words_and_record_the_line() -> None:
    text = "\n".join(
        [
            "INFO claim ok error_count=0 request=req-4011 bytes=4032",
            "WARN retrying after HTTP 403 from control plane",
            "ERROR lease lost",
            "Traceback (most recent call last):",
            "INFO CERTIFICATE_VERIFY_FAILED_COUNT=0",
        ]
    )

    hits = workload.suspicious_log_lines(text)

    assert [item["marker"] for item in hits] == ["403", "ERROR", "Traceback"]
    assert hits[0]["line"] == 2 and "HTTP 403" in hits[0]["text"]
    assert workload.suspicious_log_lines("x" * 10, limit=1) == []
    assert len(workload.suspicious_log_lines("ERROR\nERROR\nERROR", limit=2)) == 2


# --- 4b: empty identity baseline before submitting --------------------------


def test_identity_baseline_must_be_empty_or_the_operator_changes_the_identity() -> None:
    clean = {
        "cluster_id": "cluster-a",
        "restart_budget": None,
        "decision": None,
        "observations": [],
        "incidents": [],
        "workflows": [],
    }
    assert workload.identity_baseline_errors(clean) == []
    workload.assert_clean_identity_baseline([clean], job_id="j", attempt_id="j-a001")

    dirty = {
        **clean,
        "cluster_id": "cluster-b",
        "restart_budget": {"restart_count": 1},
        "observations": [{"workload_phase": "RUNNING"}],
    }
    with pytest.raises(RegionalFixtureError, match="change --attempt/--job-id") as info:
        workload.assert_clean_identity_baseline(
            [clean, dirty], job_id="j", attempt_id="j-a001"
        )
    assert "cluster-b: restart budget already exists" in str(info.value)
    assert "1 attempt observations exist" in str(info.value)


# --- 4c: the virtual node stays cluster-scoped -----------------------------


def _observation(
    cluster_id: str, pods: list[dict[str, Any]], *, gpu_prefix: str, node: str
) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "containers": [
            {
                "pod_name": pod["name"],
                "pod_uid": pod["uid"],
                "node_id": node,
                "gpu_uuids": [f"GPU-{gpu_prefix}-{index}"],
                "gpu_count": 1,
            }
            for index, pod in enumerate(pods)
        ],
    }


def test_virtual_isolation_compares_each_clusters_own_pods_and_gpus() -> None:
    pods_a = [{"name": "a-0", "uid": "uid-a0"}, {"name": "a-1", "uid": "uid-a1"}]
    pods_b = [{"name": "b-0", "uid": "uid-b0"}, {"name": "b-1", "uid": "uid-b1"}]
    sources = [{"pods": pods_a}, {"pods": pods_b}]
    good = [
        {
            "cluster_id": "cluster-a",
            "decision": None,
            "restart_budget": None,
            "observations": [
                _observation("cluster-a", pods_a, gpu_prefix="a", node="shared")
            ],
        },
        {
            "cluster_id": "cluster-b",
            "decision": None,
            "restart_budget": None,
            "observations": [
                _observation("cluster-b", pods_b, gpu_prefix="b", node="shared")
            ],
        },
    ]

    assert (
        workload.virtual_isolation_errors(
            good,
            sources=sources,
            cluster_ids=["cluster-a", "cluster-b"],
            shared_node="shared",
        )
        == []
    )

    # B's scope overwritten by A's post: same node name, so the old node-id-only
    # check passed. The Pods and GPUs give it away.
    overwritten = [
        good[0],
        {
            **good[1],
            "observations": [
                _observation("cluster-a", pods_a, gpu_prefix="a", node="shared")
            ],
            "restart_budget": {"restart_count": 0},
        },
    ]
    errors = workload.virtual_isolation_errors(
        overwritten,
        sources=sources,
        cluster_ids=["cluster-a", "cluster-b"],
        shared_node="shared",
    )
    assert "cluster-b: observation cluster_id is 'cluster-a'" in errors
    assert "cluster-b: observation Pod UIDs are not its own" in errors
    assert "cluster-b: pre-fault restart budget is not 404" in errors
    assert "GPU UUIDs are shared between the two cluster scopes" in errors


def test_command_ids_in_logs_reports_only_the_leaked_ids() -> None:
    logs = {"exec-0": "claimed cmd-1\nresult cmd-1", "exec-1": "idle"}

    assert workload.command_ids_in_logs(["cmd-1", "cmd-2", ""], logs) == ["cmd-1"]
    assert workload.command_ids_in_logs(["cmd-2"], logs) == []


def test_workload_store_probe_reads_commands_by_workflow_not_the_whole_table() -> None:
    probe = workload.WORKLOAD_STORE_PROBE

    assert "list_remote_commands()" not in probe
    assert "list_remote_commands(workflow_request_ids=request_ids)" in probe
    assert "list_active_workflow_incidents(" in probe
    assert "list_job_recovery_workflow_incidents(" in probe
    for key in ('"cluster_id"', '"incidents"', '"workflows"'):
        assert key in probe


def test_workload_store_passes_explicit_workflow_ids_as_the_fourth_argument() -> None:
    calls: list[tuple[str, ...]] = []

    class Regional:
        settings = type("S", (), {"cluster_id": "cluster-a"})()

        def cpu_python(self, script: str, *arguments: str) -> dict[str, Any]:
            calls.append(arguments)
            return {"observations": []}

    workload.workload_store(Regional(), job_id="j", attempt_id="a")  # type: ignore[arg-type]
    workload.workload_store(
        Regional(),  # type: ignore[arg-type]
        job_id="j",
        attempt_id="a",
        workflow_request_ids=["wf-1", "wf-2"],
    )

    assert calls == [("cluster-a", "j", "a"), ("cluster-a", "j", "a", "wf-1,wf-2")]
    with pytest.raises(ValueError):
        workload.workload_store(
            Regional(),  # type: ignore[arg-type]
            job_id="j",
            attempt_id="a",
            workflow_request_ids=["a,b"],
        )


# --- 4d: never delete a workload under a RUNNING workflow -------------------


class FakeWorkload:
    def __init__(self) -> None:
        self.deleted = 0

    def delete(self) -> None:
        self.deleted += 1


def test_cleanup_workload_defers_the_delete_when_quiescence_is_unproven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_quiescent(**kwargs: Any) -> dict[str, Any]:
        raise RegionalFixtureError("commands still RUNNING")

    monkeypatch.setattr(destr009, "wait_for_cleanup_quiescence", not_quiescent)
    fake = FakeWorkload()
    result: dict[str, Any] = {}

    errors = workload.cleanup_workload(
        regional=object(),  # type: ignore[arg-type]
        workload=fake,  # type: ignore[arg-type]
        case_dir=tmp_path,
        job_id="j",
        attempt_id="a",
        injection={
            "node": "n",
            "marker": "m",
            "observed_after": datetime.now(timezone.utc),
        },
        result=result,
        label="workload",
    )

    assert fake.deleted == 0
    assert result["workload_cleanup_deferred"] is True
    assert result["workload_cleanup_quiescence_error"].startswith(
        "RegionalFixtureError"
    ), result["workload_cleanup_quiescence_error"]
    assert errors == ["workload: workflow not quiescent; workload left in place"]


def test_cleanup_workload_deletes_after_quiescence_or_without_an_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    def quiescent(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return {"safe_to_delete": True}

    monkeypatch.setattr(destr009, "wait_for_cleanup_quiescence", quiescent)
    fake = FakeWorkload()
    result: dict[str, Any] = {}

    assert (
        workload.cleanup_workload(
            regional=object(),  # type: ignore[arg-type]
            workload=fake,  # type: ignore[arg-type]
            case_dir=tmp_path,
            job_id="j",
            attempt_id="a",
            injection={
                "node": "n",
                "marker": "m",
                "observed_after": datetime.now(timezone.utc),
                "workflow_request_ids": ["wf-1"],
            },
            result=result,
            label="cluster_a",
        )
        == []
    )
    assert fake.deleted == 1
    assert seen[0]["workflow_request_ids"] == ["wf-1"]
    assert result["cluster_a_cleanup_quiescence"] == {"safe_to_delete": True}

    # Cluster B in ISO-001 never had a workflow: straight delete, no wait.
    other = FakeWorkload()
    workload.cleanup_workload(
        regional=object(),  # type: ignore[arg-type]
        workload=other,  # type: ignore[arg-type]
        case_dir=tmp_path,
        job_id="j",
        attempt_id="a",
        injection=None,
        result={},
        label="cluster_b",
    )
    assert other.deleted == 1 and len(seen) == 1


# --- 4e: a failing body still reports what cleanup did ----------------------


def test_failure_outcome_carries_the_partial_result_of_a_case_error() -> None:
    partial = {
        "verdict": "FAIL",
        "cleanup_errors": ["prewarm resources remain"],
        "prewarm_residuals": {"pod-0": True},
        "limitations": ["from the case"],
    }
    error = workload.WorkloadCaseError(TimeoutError("kubectl delete"), partial)

    outcome = workload.failure_outcome(error)

    assert outcome["verdict"] == "FAIL"
    assert outcome["error"] == "WorkloadCaseError: TimeoutError: kubectl delete"
    assert outcome["cleanup_errors"] == ["prewarm resources remain"]
    assert outcome["prewarm_residuals"] == {"pod-0": True}
    assert outcome["limitations"][-1] == "from the case"

    plain = workload.failure_outcome(RuntimeError("boom"))
    assert plain["error"] == "RuntimeError: boom" and "cleanup_errors" not in plain


def test_raise_with_outcome_wraps_once_and_merges_the_result() -> None:
    workload.raise_with_outcome(None, {"ignored": True})

    with pytest.raises(workload.WorkloadCaseError) as info:
        workload.raise_with_outcome(ValueError("first"), {"cleanup_errors": ["x"]})
    assert info.value.outcome == {"cleanup_errors": ["x"]}
    assert isinstance(info.value.cause, ValueError), (
        f"the original exception must ride along as cause: {info.value.cause!r}"
    )

    with pytest.raises(workload.WorkloadCaseError) as again:
        workload.raise_with_outcome(info.value, {"prewarm_residuals": {}})
    assert again.value is info.value
    assert again.value.outcome == {"cleanup_errors": ["x"], "prewarm_residuals": {}}


# --- 4f/4g: WORKLOAD-002 pre-apply delete, real checks ----------------------


def test_workload002_deletes_the_same_named_job_before_apply() -> None:
    source = Path(workload.__file__).read_text(encoding="utf-8")
    body = source[source.index("def run_workload_baseline") :]
    body = body[
        : body.index("def OBSERVATION_POST_PROBE")
        if "def OBSERVATION_POST_PROBE" in body
        else len(body)
    ]

    delete_index = body.index("fixture.delete()")
    apply_index = body.index('"apply",\n                    "-f",')
    assert delete_index < apply_index, "spec step 1: delete before kubectl apply"
    assert '"pre_apply_delete": True' in body
    # The two tautologies are gone.
    assert "render_equivalence = True" not in body
    assert '"submission_succeeded": bool(submission)' not in body
    assert 'int(submission["returncode"]) == 0' in body


def test_loss_errors_require_non_increasing_loss_per_rank() -> None:
    good = {
        "pod-0": "rank=0 step=0 loss=0.250000\nrank=0 step=1 loss=0.240000\n"
        "rank=1 step=0 loss=0.300000\nrank=1 step=1 loss=0.300000\nSUCCESS rank=0/2",
        "pod-1": "HEARTBEAT x\nrank=8 step=1 loss=0.10\nrank=8 step=0 loss=0.20",
    }
    assert workload.loss_errors(good) == []

    bad = {
        "pod-0": "rank=0 step=0 loss=0.25\nrank=0 step=1 loss=0.26",
        "pod-1": "SUCCESS rank=8/16 all_reduce=136.0",
    }
    errors = workload.loss_errors(bad)
    assert errors == [
        "pod-0: rank 0 loss increased: [0.25, 0.26]",
        "pod-1: no loss lines",
    ]


def test_observation_ranks_must_be_exactly_the_expected_set() -> None:
    def observation(ranks: list[int | None]) -> dict[str, Any]:
        return {"containers": [{"rank": rank} for rank in ranks]}

    assert (
        workload.observation_rank_errors(
            observation([2, 0, 1]), expected_ranks={0, 1, 2}
        )
        == []
    )
    assert workload.observation_rank_errors(
        observation([0, 1]), expected_ranks={0, 1, 2}
    ) == ["container ranks [0, 1] != [0, 1, 2]"]
    assert workload.observation_rank_errors(
        observation([0, 0, 1]), expected_ranks={0, 1, 2}
    ) == ["duplicate container ranks: [0, 0, 1]"]
    assert workload.observation_rank_errors(
        observation([0, None]), expected_ranks={0, 1}
    ) == ["an observed container has no rank"]


# --- 4h: E2E-001 control-plane identity without the unbounded event set -----


def test_control_plane_blast_errors_flag_new_evictions_but_not_aged_out_ones() -> None:
    before = {
        "nodes": {"cpu-1": {"taints": [], "unschedulable": False}},
        "gpu_fault_jobs": [["ns", "job-a"]],
        "eviction_events": [
            ["uid-old", "Evicted", "pod-old"],
            ["uid-gone", "Evicted", "p"],
        ],
    }
    after_identical = {
        **before,
        # `uid-gone` aged out of the API server between the snapshots; the
        # old whole-set equality called that CHANGED.
        "eviction_events": [["uid-old", "Evicted", "pod-old"]],
    }
    assert workload.control_plane_blast_errors(before, after_identical) == []

    after_changed = {
        "nodes": {"cpu-1": {"taints": [["gpu-fault.io/x", None, "NoSchedule"]]}},
        "gpu_fault_jobs": [["ns", "job-a"], ["ns", "job-b"]],
        "eviction_events": [["uid-new", "TaintManagerEviction", "pod-new"]],
    }
    errors = workload.control_plane_blast_errors(before, after_changed)
    assert errors == [
        "CPU node taints/cordon/gpu-fault metadata changed",
        "gpu-fault labelled Jobs on the control plane changed",
        "1 new eviction events in the window",
    ]


def test_gpu_nodes_clean_tolerates_only_the_cordoned_spare() -> None:
    nodes = [
        {"ready": "True", "unschedulable": False, "taints": [], "labels": {}},
        {
            "ready": "True",
            "unschedulable": True,
            "taints": [],
            "labels": {"gpu-fault.io/spare": "true"},
        },
    ]
    assert workload.gpu_nodes_clean(nodes) is True, (
        "schedulable, untainted GPU nodes are clean"
    )
    assert not workload.gpu_nodes_clean(
        [{**nodes[0], "taints": [{"key": "gpu-fault.io/quarantined"}]}]
    ), "a quarantine taint is a residual"
    assert not workload.gpu_nodes_clean([{**nodes[0], "unschedulable": True}]), (
        "a cordoned node is a residual"
    )


def test_e2e001_source_checks_attempt_id_scope_and_nodes_after_cleanup() -> None:
    source = Path(workload.__file__).read_text(encoding="utf-8")
    body = source[source.index("def run_e2e001") : source.index("def gpu_nodes_clean")]

    assert '"workload_attempt_id_changed"' in body
    assert '"incident_and_plan_cluster_scoped"' in body
    assert '"gpu_nodes_restored_after_cleanup"' in body
    assert "control_plane_blast_errors(blast_before, blast_after)" in body
    assert "blast_before == blast_after" not in body


# --- 4i: admin status once per group ----------------------------------------


def test_admin_status_runs_only_on_the_group_closing_case_unless_skipped() -> None:
    assert workload.admin_status_policy("GF-REGIONAL-WORKLOAD-002", skip=False) == (
        True,
        "GF-REGIONAL-WORKLOAD-002 closes the WORKLOAD group",
    )
    run, reason = workload.admin_status_policy("GF-REGIONAL-WORKLOAD-001", skip=False)
    assert run is False and "runs once per group" in reason
    assert workload.admin_status_policy("GF-REGIONAL-WORKLOAD-002", skip=True) == (
        False,
        "skipped by --skip-admin-status",
    )

    arguments = workload.parser().parse_args(
        [
            "--case",
            "GF-REGIONAL-WORKLOAD-002",
            "--run-dir",
            "/tmp/run",
            "--site",
            "/tmp/site.yaml",
            "--skip-admin-status",
        ]
    )
    assert arguments.skip_admin_status is True
    assert arguments.state_dir is None


# --- 4j: the two ISO-001 clusters are prepared concurrently ----------------


def test_iso001_prepares_both_clusters_in_a_thread_pool() -> None:
    source = Path(workload.__file__).read_text(encoding="utf-8")
    body = source[source.index("def run_iso001") : source.index("E2E_PREFLIGHT_PROBE")]

    assert "ThreadPoolExecutor(max_workers=2)" in body
    assert "pool.map(prepare" in body
    assert "assert_clean_identity_baseline(" in body
    assert "executor_logs_since(regional_b" in body
    assert '"secondary_executor_logs_free_of_primary_command_ids"' in body


# --- 4k: UID polling without log pulls --------------------------------------


def test_wait_pod_uids_unchanged_polls_pods_and_snapshots_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Fixture:
        def __init__(self) -> None:
            self.pod_calls = 0
            self.snapshot_calls = 0

        def pods(self) -> list[dict[str, Any]]:
            self.pod_calls += 1
            return [{"uid": "u1"}, {"uid": "u2"}]

        def snapshot(self) -> dict[str, Any]:
            self.snapshot_calls += 1
            return {"pods": self.pods(), "heartbeat_logs": {}}

    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(time, "sleep", sleep)
    fixture = Fixture()

    result = workload.wait_pod_uids_unchanged(
        fixture,  # type: ignore[arg-type]
        {"u1", "u2"},
        timeout_seconds=20,
        poll_seconds=5,
    )

    assert {item["uid"] for item in result["pods"]} == {"u1", "u2"}
    assert fixture.snapshot_calls == 1
    assert fixture.pod_calls >= 4

    changed = Fixture()
    changed.pods = lambda: [{"uid": "other"}]  # type: ignore[method-assign]
    with pytest.raises(RegionalFixtureError, match="Pod UIDs changed"):
        workload.wait_pod_uids_unchanged(changed, {"u1"}, timeout_seconds=5)  # type: ignore[arg-type]


# --- mutation probes are single-attempt -------------------------------------


def test_post_observation_never_retries() -> None:
    seen: dict[str, Any] = {}

    class Regional:
        def executor_python(
            self, script: str, *arguments: str, **kwargs: Any
        ) -> dict[str, Any]:
            seen.update(kwargs)
            return {"status": 202}

    workload.post_observation(Regional(), {"job_id": "j"})  # type: ignore[arg-type]

    assert seen == {"attempts": 1}


def test_main_binds_results_and_predecessors_to_the_deployment_identity() -> None:
    source = Path(workload.__file__).read_text(encoding="utf-8")
    body = source[source.index("def main()") :]

    assert "identity = site.regional(primary).evidence_identity()" in body
    assert "predecessor_evidence(path, predecessor_id, **identity)" in body
    assert "**identity," in body
    assert "outcome = failure_outcome(exc)" in body
