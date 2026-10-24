import json
from datetime import timedelta
from subprocess import CompletedProcess

import pytest

from gpu_fault.execution.hung_classification import classify_hung_signals
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowStatus
from gpu_fault.node_agent import NodeActionStatus
from gpu_fault.node_agent.operations import flight_recorder, hung_triage
from tests.execution._cov95_runtime_triage import triage_harness
from tests.execution._process_evidence_support import stack_json
from tests.node_agent._cov95_runtime_hung import COMPUTE_QUERY, process
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import NOW, command, envelope


def proc_stat(pid, ticks, state="R", start=1000):
    fields = [state] + ["0"] * 19
    fields[11] = str(ticks)
    fields[19] = str(start)
    return f"{pid} (python) " + " ".join(fields)


def collect(
    node_factory,
    tmp_path,
    monkeypatch,
    *,
    exit_code=0,
    malformed=False,
    state="R",
    gpu="20",
    shared=False,
    context=False,
):
    directory = process(tmp_path / "proc", 100)
    task = directory / "task/100"
    task.mkdir(parents=True)
    (directory / "environ").write_bytes(
        b"GPU_FAULT_RANK_PROGRESS_FILE=/tmp/context.json\0" if context else b""
    )
    (directory / "io").write_text("read_bytes: 0\nwrite_bytes: 0\nwchar: 0\n")
    for parent in (directory, task):
        (parent / "stat").write_text(proc_stat(100, 0, state))
        (parent / "wchan").write_text("io_schedule" if state == "D" else "0")
    if context:
        root = directory / "root/tmp"
        root.mkdir(parents=True)
        (root / "context.json").write_text(
            json.dumps(
                {
                    "job_id": "job-a",
                    "attempt_id": "attempt-a",
                    "rank": 0,
                    "world_size": 1,
                    "process_start_ticks": 1000,
                    "process_pid": 100,
                    "observed_at": NOW.isoformat(),
                    "diagnostic_context": {
                        "group_name": "tp",
                        "members": [0],
                        "role": "trainer",
                        "pipeline_stage": "0",
                        "expert_role": "none",
                        "hardware_class": "p5",
                        "code_version": "test",
                        "thread_roles": {"100": "TRAINING"},
                    },
                }
            )
        )

    class Clock:
        value = 0.0

        def monotonic(self):
            return self.value

        def sleep(self, seconds):
            self.value += seconds
            (directory / "stat").write_text(
                proc_stat(100, int(self.value * 100), state)
            )

    clock = Clock()
    monkeypatch.setattr(hung_triage, "time", clock)
    monkeypatch.setattr(flight_recorder, "time", clock)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if COMPUTE_QUERY in argv:
            return CompletedProcess(
                argv,
                0,
                "100, GPU-a, python\n" + ("200, GPU-a, other\n" if shared else ""),
                "",
            )
        if argv[0] == "/unit/py-spy":
            return CompletedProcess(
                argv,
                exit_code,
                "Permission denied" if malformed else stack_json(100),
                "warning",
            )
        return CompletedProcess(argv, 0, f"GPU-a, {gpu}, 0, 0x0, H100\n", "")

    agent = node_factory(
        allowed_operations={Op.COLLECT_HUNG_TRIAGE},
        runner=runner,
        python_stack_tool="/unit/py-spy",
        sleep=clock.sleep,
        now=lambda: NOW + timedelta(seconds=clock.value),
    )
    result = agent.execute(
        envelope(
            command(
                Op.COLLECT_HUNG_TRIAGE,
                parameters={
                    "require_workload_identity": True,
                    "workload_cgroup_paths_by_node": {"node-a": ["/training/job"]},
                    "job_id": "job-a",
                    "attempt_id": "attempt-a",
                },
            )
        )
    )
    return result, calls


@pytest.mark.parametrize("exit_code,malformed", [(1, False), (0, True), (0, False)])
def test_real_capture_checks_exit_code_json_shape_and_does_not_parse_error_text(
    node_factory, tmp_path, monkeypatch, exit_code, malformed
):
    result, calls = collect(
        node_factory, tmp_path, monkeypatch, exit_code=exit_code, malformed=malformed
    )
    assert result.status is NodeActionStatus.SUCCEEDED, result
    [row] = result.details["ranks"]
    samples = row["stack_series"]["samples"]
    assert len(samples) == 3, samples
    assert all(item["status"] == "COLLECTED" for item in samples) is (
        exit_code == 0 and not malformed
    ), samples
    if exit_code or malformed:
        assert not any(item.get("signature") for item in samples), samples
    assert all(
        "--json" in argv and "--nonblocking" in argv
        for argv in calls
        if argv[0] == "/unit/py-spy"
    ), calls
    assert [item["cpu_cores"] for item in row["process_series"]["windows"]] == [1, 1], (
        row
    )


def test_unknown_rank_storage_wait_reaches_workflow_without_strong_attribution(
    node_factory, tmp_path, monkeypatch
):
    result, _ = collect(node_factory, tmp_path, monkeypatch, state="D")
    harness = triage_harness(
        {"node_results": {"node-a": result.details}}, observe_job=None
    )
    outcome = harness.execute()
    assert outcome.status is WorkflowStatus.RUNNING, outcome
    saved = harness.store.get_workflow(harness.workflow.request_id)
    bundle = saved.official_steps[1]
    [candidate] = bundle.parameters["hung_triage_decision"]["candidate_processes"]
    assert (
        candidate["node_id"] == "node-a"
        and candidate["pid"] == 100
        and candidate["rank"] is None
    ), candidate
    assert "REPEATED_D_STATE" in candidate["reasons"], candidate
    assert bundle.parameters["capture_process_state"] is False, bundle
    assert bundle.parameters["hung_triage_decision"]["root_cause_confirmed"] is False, (
        bundle
    )


@pytest.mark.parametrize("gpu,shared", [("NaN", False), ("101", False), ("0", True)])
def test_gpu_invalid_or_shared_values_do_not_replace_process_evidence(
    node_factory, tmp_path, monkeypatch, gpu, shared
):
    result, _ = collect(
        node_factory, tmp_path, monkeypatch, gpu=gpu, shared=shared, state="D"
    )
    decision = classify_hung_signals(result.details["ranks"], undetermined_nodes=[])
    report = decision["evidence_branches"]["process_observations"]["processes"][0]
    assert report["behavior"]["cpu_cores"] == 1, report
    assert "REPEATED_D_STATE" in decision["candidate_processes"][0]["reasons"], decision
    if shared:
        assert report["behavior"]["gpu_shared"] is True, report
    else:
        assert "GPU_MAPPING_OR_SAMPLE_COVERAGE" in report["behavior"]["missing"], report


def test_optional_role_file_is_process_bound_and_does_not_require_rank_env_or_train_step(
    node_factory, tmp_path, monkeypatch
):
    result, _ = collect(node_factory, tmp_path, monkeypatch, context=True)
    [row] = result.details["ranks"]
    assert row["rank"] is None, row
    assert row["comparison_context"]["verified"] is True, row
    assert row["stack_series"]["samples"][0]["threads"][0]["role"] == "TRAINING", row
    assert "members" not in row["comparison_context"], (
        "large membership is represented once by a digest"
    )
