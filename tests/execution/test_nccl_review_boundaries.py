from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.communication_window import common_window
from gpu_fault.execution.flight_semantics import flight_semantics
from gpu_fault.fabric_scope import validate_task_fabric
from gpu_fault.flight_progress import apply_pg_status, group_progress
from gpu_fault.hung_summary_codec import decode_summary
from gpu_fault.hung_survey_series import recheck_details
from gpu_fault.nccl_events import event_kind
from tests.execution._flight_series_support import NOW, entry, temporal_signal
from tests.execution._hung_survey_support import passive_document
from tools.analyze_nccl_flights import analyze


@pytest.mark.parametrize("defect", ["missing-id", "bad-counters", "new-version", None])
def test_pg_status_is_versioned_and_never_indexed_by_group_name(defect):
    payload = {
        "version": "2.10",
        "pg_status": {
            "42": {
                "last_enqueued_collective": "100",
                "last_started_collective": "100",
                "last_completed_collective": "99",
            }
        },
    }
    if defect == "bad-counters":
        payload["pg_status"]["42"]["last_completed_collective"] = "101"
    if defect == "new-version":
        payload["version"] = "3.0"
    result = apply_pg_status(
        group_progress([entry(100)]), payload, None if defect == "missing-id" else 42
    )
    assert (result["watermark_source"] == "PG_STATUS") is (defect is None), result
    if defect == "bad-counters":
        assert result["comparison_supported"] is False, result
    if defect is None:
        assert result["last_completed"] == 99 and result["history_complete"] is False, (
            result
        )


def semantic_signal(rank, operation, **fields):
    op = entry(
        100,
        profiling_name=operation,
        input_sizes=[[8]],
        output_sizes=[[8]],
        input_dtypes=["Float"],
        output_dtypes=["Float"],
        **fields,
    )
    signal = temporal_signal(rank, [op], members=[0, 1])
    for point in ("first", "second"):
        group = signal["flight_recorder_series"][point]["groups"][0]
        group["semantic_entries"] = [deepcopy(op)]
        group["pg_rank_order"] = [0, 1]
    return signal


@pytest.mark.parametrize("change", ["operation", "shape", "dtype"])
def test_persistent_collective_semantics_are_not_just_sequence_numbers(change):
    signals = [semantic_signal(rank, "nccl:all_reduce") for rank in (0, 1)]
    for point in ("first", "second"):
        op = signals[1]["flight_recorder_series"][point]["groups"][0][
            "semantic_entries"
        ][0]
        op[
            {
                "operation": "profiling_name",
                "shape": "input_sizes",
                "dtype": "input_dtypes",
            }[change]
        ] = {"operation": "nccl:broadcast", "shape": [[16]], "dtype": ["Half"]}[change]
    report = flight_semantics(signals)
    assert len(report["findings"]) == 1, report
    assert report["findings"][0]["issues"], report
    assert report["root_cause_confirmed"] is False, report


@pytest.mark.parametrize("defect", [None, "missing-peer", "size", "ambiguous", "time"])
def test_p2p_evidence_does_not_trigger_topology_guessing_or_rank_attribution(defect):
    signals = [
        semantic_signal(0, "nccl:send 0->1", is_p2p=True, p2p_seq_id=7),
        semantic_signal(1, "nccl:recv 1<-0", is_p2p=True, p2p_seq_id=91),
    ]
    if defect == "missing-peer":
        signals.pop()
    elif defect == "time":
        signals[1]["flight_recorder_series"]["clock_uncertainty_seconds"] = None
    else:
        for point in ("first", "second"):
            ops = signals[1]["flight_recorder_series"][point]["groups"][0][
                "semantic_entries"
            ]
            if defect == "size":
                ops[0]["output_sizes"] = [[16]]
            elif defect == "ambiguous":
                ops.append({**deepcopy(ops[0]), "record_id": 222})
    report = flight_semantics(signals)
    assert report["findings"] == [], report
    assert report["unsupported_p2p_entries"] > 0, report
    assert report["p2p_analysis"] == "OUT_OF_SCOPE", report


def test_two_sweep_barrier_repairs_disjoint_batches_without_removing_time_checks():
    baseline, recheck = {}, {}
    for index in range(128):
        node = f"node-{index}"
        left = passive_document(node, [index], 128)
        left["groups"] = {"0": {"group_key": "world"}}
        left["ranks"][0]["communication"] = [
            {
                "group": "0",
                "before": [10, 10, 9],
                "after": [10, 10, 9],
                "first_observed_at": (NOW + timedelta(seconds=index)).isoformat(),
                "second_observed_at": (NOW + timedelta(seconds=index + 1)).isoformat(),
                "clock_uncertainty_seconds": 0.1,
            }
        ]
        right = deepcopy(left)
        for field in ("first_observed_at", "second_observed_at"):
            right["ranks"][0]["communication"][0][field] = (
                NOW + timedelta(seconds=132 + index + (field == "second_observed_at"))
            ).isoformat()
        baseline[node], recheck[node] = left, right
    merged = recheck_details({"node_results": baseline}, {"node_results": recheck})
    rows = [
        {**decode_summary(value)["ranks"][0]["communication"][0], "node_id": node}
        for node, value in merged["node_results"].items()
    ]
    assert common_window(rows)["time_comparable"] is True, merged
    rows[-1]["first_observed_at"] = rows[0]["second_observed_at"]
    assert common_window(rows)["time_comparable"] is False, (
        "disjoint windows remain inadmissible"
    )


@pytest.mark.parametrize("missing", [False, True])
def test_training_nic_down_or_missing_cannot_be_hidden_by_management_link(missing):
    samples = [SimpleNamespace(name="network_link_up", device="eth0", value=1)]
    samples.extend(
        SimpleNamespace(name=name, device="eth1", value=0)
        for name in ("network_errors_delta", "network_drops_delta")
    )
    if not missing:
        samples.append(SimpleNamespace(name="network_link_up", device="eth1", value=0))
    pending, failures = validate_task_fabric(
        samples,
        {
            "basis": "TASK_DEVICE_BINDING",
            "network_interfaces": ["eth1"],
            "rdma_ports": [],
        },
    )
    assert bool(pending) is missing, pending
    assert bool(failures) is not missing, failures


def test_offline_official_analyzer_missing_dependency_is_not_a_success(
    tmp_path, monkeypatch
):
    import importlib.metadata

    def absent(_):
        raise importlib.metadata.PackageNotFoundError("torch")

    monkeypatch.setattr(importlib.metadata, "version", absent)
    assert analyze({"records": []}, tmp_path)["status"] == "UNAVAILABLE"


@pytest.mark.parametrize(
    "text,kind",
    [
        ("Watchdog caught collective operation timeout", "COLLECTIVE_TIMEOUT"),
        ("NCCL WARN unhandled system error", "EXPLICIT_NCCL_ERROR"),
        ("disk timeout", None),
    ],
)
def test_error_classification(text, kind):
    assert event_kind(text) == kind, text
