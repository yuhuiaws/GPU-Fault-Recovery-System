import pytest

from gpu_fault.execution.hung_classification import classify_hung_signals
from gpu_fault.node_agent.operations.process_samples import process_window
from tests.execution._flight_series_support import entry, temporal_signal
from tests.execution._process_evidence_support import process_signal


def diagnose(rows):
    return classify_hung_signals(rows, undetermined_nodes=[])


def test_unknown_rank_retains_identity_bound_repeated_d_state_without_gpu():
    row = process_signal(state="D", wchan="io_schedule", cpu=0)
    row["gpu_series"] = {}
    result = diagnose([row])
    [candidate] = result["candidate_processes"]
    assert candidate["pid"] == 100 and candidate["rank"] is None, result
    assert "REPEATED_D_STATE" in candidate["reasons"], result
    assert result["targeted_capture_eligible"] is False, result
    report = result["evidence_branches"]["process_observations"]["processes"][0]
    assert "GPU_MAPPING_OR_SAMPLE_COVERAGE" in report["behavior"]["missing"], report


def test_zero_wchan_is_not_a_wait_location():
    result = diagnose([process_signal(state="S", wchan="0", cpu=0)])
    assert result["candidate_processes"] == [], result
    observations = result["evidence_branches"]["process_observations"]["processes"][0][
        "observations"
    ]
    assert not any(item["code"] == "REPEATED_THREAD_WAIT" for item in observations), (
        observations
    )


def test_cpu_ticks_are_normalized_by_actual_elapsed_time():
    short = process_signal(0, context=True, cpu=2, interval=1)
    long = process_signal(1, context=True, cpu=2, interval=4)
    third = process_signal(2, context=True, cpu=2, interval=2)
    result = diagnose([short, long, third])
    reports = result["evidence_branches"]["process_observations"]["processes"]
    assert [item["behavior"]["cpu_cores"] for item in reports] == [2, 2, 2], reports
    assert result["candidate_processes"] == [], result


@pytest.mark.parametrize(
    "defect",
    [
        "no-context",
        "pp-stage",
        "expert-role",
        "hardware",
        "image",
        "nonoverlap",
        "shared-gpu",
    ],
)
def test_legitimate_role_or_window_differences_are_not_peer_outliers(defect):
    rows = [
        process_signal(
            index, context=True, function="other" if index == 0 else "compute"
        )
        for index in range(5)
    ]
    context = rows[0]["comparison_context"]
    if defect == "no-context":
        for row in rows:
            row["comparison_context"] = {}
    elif defect == "nonoverlap":
        for sample in rows[0]["process_series"]["samples"]:
            sample["observed_at"] = sample["observed_at"].replace("00:00:0", "00:00:1")
    elif defect == "shared-gpu":
        rows[0]["gpu_series"]["shared_with_other_processes"] = True
    else:
        field = {
            "pp-stage": "pipeline_stage",
            "expert-role": "expert_role",
            "hardware": "hardware_class",
            "image": "code_version",
        }[defect]
        context[field] = "different"
    result = diagnose(rows)
    assert result["candidate_processes"] == [], result


def test_repeated_same_role_thread_difference_is_only_a_weak_pid_lead():
    rows = [
        process_signal(
            index, context=True, function="other" if index == 0 else "compute"
        )
        for index in range(5)
    ]
    result = diagnose(rows)
    [candidate] = result["candidate_processes"]
    assert candidate["pid"] == 100 and candidate["rank"] is None, result
    assert candidate["reasons"] == ["ROLE_COHORT_STACK_DIFFERENCE"], result
    assert result["classification"] == "WEAK" and result["culprit_ranks"] == [], result
    assert result["root_cause_confirmed"] is False, result


def test_fr_completion_progress_does_not_silence_process_io_evidence():
    row = process_signal(rank=0, state="D", wchan="io_schedule", cpu=0)
    fr = temporal_signal(0, [entry(100)], [entry(100, "completed")])
    row.update(
        {key: value for key, value in fr.items() if key.startswith("flight_recorder")}
    )
    result = diagnose([row])
    evidence = result["evidence_branches"]
    assert evidence["fr_temporal"]["process_groups"][0][
        "completion_advanced_ranks"
    ] == [0], evidence
    assert "REPEATED_D_STATE" in result["candidate_processes"][0]["reasons"], result
    assert result["culprit_ranks"] == [] and result["root_cause_confirmed"] is False, (
        result
    )


@pytest.mark.parametrize(
    "defect", ["transient", "thread-reused", "missing-stack", "bad-identity"]
)
def test_transient_or_untrusted_samples_do_not_become_persistent_process_leads(defect):
    row = process_signal(state="D", wchan="io_schedule", cpu=0, function="all_reduce")
    if defect == "bad-identity":
        row["process_identity"]["verified"] = False
    elif defect == "missing-stack":
        row["stack_series"]["samples"][1] = {
            "status": "FAILED",
            "returncode": 1,
            "error": "py-spy failed",
        }
        for sample in row["process_series"]["samples"]:
            sample["threads"]["100"].update(state="R", wchan="0")
    else:
        middle = row["process_series"]["samples"][1]["threads"]["100"]
        if defect == "thread-reused":
            middle["start_ticks"] = 7777
        else:
            middle.update(state="R", wchan="0")
        row["stack_series"]["samples"][1] = {"status": "FAILED"}
    samples = row["process_series"]["samples"]
    row["process_series"]["windows"] = [
        process_window(left, right, 100) for left, right in zip(samples, samples[1:])
    ]
    result = diagnose([row])
    assert result["candidate_processes"] == [], result


def test_single_legacy_stack_or_raw_ticks_never_authorizes_cross_role_inference():
    rows = [
        {
            "rank": index,
            "python_stack": {
                "signature": "different" if index == 0 else "same",
                "collective_frames": True,
            },
            "proc": {"cpu_ticks_delta": 0 if index == 0 else 100},
            "gpu": {"utilization_gpu_percent": 0},
        }
        for index in range(5)
    ]
    result = diagnose(rows)
    assert result["culprit_ranks"] == result["candidate_processes"] == [], result


def test_missing_cpu_counter_does_not_discard_repeated_io_wait():
    row = process_signal(state="D", wchan="io_schedule", cpu=0)
    for sample in row["process_series"]["samples"]:
        sample.pop("cpu_ticks")
    samples = row["process_series"]["samples"]
    row["process_series"]["windows"] = [
        process_window(left, right, 100) for left, right in zip(samples, samples[1:])
    ]
    result = diagnose([row])
    assert "REPEATED_D_STATE" in result["candidate_processes"][0]["reasons"], result
    report = result["evidence_branches"]["process_observations"]["processes"][0]
    assert report["behavior"]["cpu_cores"] is None, report
    assert "CPU_WINDOW_COVERAGE" in report["behavior"]["missing"], report
