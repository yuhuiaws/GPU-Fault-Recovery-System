from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.node_agent.communication_history import CommunicationHistory
from gpu_fault.node_agent.observation_supervisor import ObservationSupervisor
from gpu_fault.node_agent.operations.communication_cache import (
    cached_communication,
    cached_flight_pairs,
)
from gpu_fault.ras_survey import ras_survey
from gpu_fault.models import WorkflowOperation as Op
from tests.execution._flight_series_support import NOW
from tests.node_agent._support import command
from tests.node_agent.test_communication_monitor import ras_document
from gpu_fault.ras_observation import parse_ras
from gpu_fault.ras_progress import ras_progress
from gpu_fault.node_agent.observation_worker import ObservationAgent
from gpu_fault.node_agent.passive_communication import listening_owner
from tests.node_agent._cov95_runtime_hung import process
import json


@pytest.mark.parametrize(
    "address,expected", [("0100007F", True), ("00000000", True), ("0100000A", False)]
)
def test_ras_socket_must_be_owned_and_bound_to_the_queried_loopback(
    tmp_path, address, expected
):
    directory = process(tmp_path / "proc", 100)
    (directory / "fd").mkdir()
    (directory / "fd/7").symlink_to("socket:[123]")
    (directory / "net").mkdir()
    (directory / "net/tcp").write_text(
        "header\n" + f"0: {address}:{28028:04X} 00000000:0000 0A 0 0 0 0 0 123\n"
    )
    agent = ObservationAgent(
        {
            "proc_root": str(tmp_path / "proc"),
            "infiniband_root": str(tmp_path / "ib"),
            "diagnostic_output_dir": str(tmp_path / "evidence"),
        }
    )
    rows = [
        {"pid": 100, "gpu_uuids": ["GPU-a"], "process_identity": {"verified": True}}
    ]
    assert (listening_owner(agent, rows) is not None) is expected, address


def test_delayed_survey_reads_the_requested_common_epochs_not_latest_cache(tmp_path):
    root = tmp_path / "proc/sys/kernel/random"
    root.mkdir(parents=True)
    (root / "boot_id").write_text("boot")
    history = CommunicationHistory(tmp_path / "evidence/communication-history.db")
    epoch = int(NOW.timestamp() // 30)
    for offset in range(10):
        stamp = NOW + timedelta(seconds=30 * offset)
        history.append(
            history.scope("job", "attempt", "boot"),
            epoch + offset,
            stamp.timestamp(),
            {
                "observed_at": stamp.isoformat(),
                "node_addresses": ["10.0.0.1"],
                "processes": [],
            },
        )
    agent = SimpleNamespace(
        proc_root=tmp_path / "proc",
        diagnostic_output_dir=tmp_path / "evidence",
        now=lambda: NOW + timedelta(seconds=300),
    )
    request = SimpleNamespace(
        parameters={
            "job_id": "job",
            "attempt_id": "attempt",
            "communication_history_epochs": [epoch + 1, epoch + 2],
        }
    )
    result = cached_communication(agent, request)
    assert result["communication_history"]["first_epoch"] == epoch + 1, result
    assert result["communication_history"]["second_epoch"] == epoch + 2, result
    assert result["communication_history"]["clock_uncertainty_seconds"] is None, result
    request.parameters["attempt_id"] = "other"
    assert (
        cached_communication(agent, request)["communication_history"]["status"]
        == "INSUFFICIENT_SAMPLES"
    )


@pytest.mark.parametrize("defect", ["same-file", "pid-reuse", "parse-error"])
def test_cached_flights_require_new_file_and_same_process_generation(defect):
    first = {
        "processes": [
            {
                "pid": 100,
                "rank": None,
                "process_identity": {"verified": True, "start_time_ticks": 1000},
                "flight_recorder": {
                    "status": "dumped",
                    "source_mtime_ns": 10,
                    "last_entries_by_pg": [],
                },
            }
        ]
    }
    second = deepcopy(first)
    second["processes"][0]["flight_recorder"]["source_mtime_ns"] = 20
    if defect == "same-file":
        second["processes"][0]["flight_recorder"]["source_mtime_ns"] = 10
    elif defect == "pid-reuse":
        second["processes"][0]["process_identity"]["start_time_ticks"] = 2000
    else:
        second["processes"][0]["flight_recorder"]["parse_error"] = "bad data"
    assert cached_flight_pairs(first, second) == [], defect


def test_ras_local_rank_is_joined_to_verified_pid_without_global_rank():
    snapshot = {**parse_ras(json.dumps(ras_document()).encode()), "status": "COLLECTED"}
    report = ras_progress(snapshot, snapshot)
    nodes = {
        "node-a": {
            "node_addresses": ["10.0.0.1"],
            "cached_ras": report,
            "ranks": [
                {
                    "pid": 1000,
                    "rank": None,
                    "gpu_uuids": ["GPU-a"],
                    "process_identity": {
                        "verified": True,
                        "start_time_ticks": 7,
                        "namespace_pids": [1000, 100],
                    },
                }
            ],
        }
    }
    result = ras_survey(nodes)
    [lead] = result["ras_process_leads"]
    assert (
        lead["pid"] == 1000 and lead["rank"] is None and lead["communicator_rank"] == 0
    ), result
    assert result["ras_completion_progress_available"] is False, result
    nodes["node-a"]["ranks"][0]["process_identity"]["verified"] = False
    assert ras_survey(nodes)["ras_process_leads"] == [], (
        "PID generation must be verified"
    )


@pytest.mark.parametrize(
    "operation,name",
    [
        (Op.COLLECT_HUNG_SUMMARY, "summary"),
        (Op.COLLECT_HUNG_TRIAGE, "triage"),
        (Op.COLLECT_DIAGNOSTIC_BUNDLE, "bundle"),
    ],
)
def test_real_readonly_worker_handles_all_three_evidence_operations(
    tmp_path, monkeypatch, operation, name
):
    # Empty owned procfs and no tool PATH exercise missing-data paths, never host GPUs.
    root = tmp_path / "proc"
    root.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path))
    request = command(
        operation,
        parameters={
            "job_id": "job",
            "attempt_id": "attempt",
            "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
            "snapshot_epoch": "snapshot",
            "workload_cgroup_paths_by_node": {"node-a": ["/training/job"]},
            "capture_process_state": False,
        },
    )
    result = ObservationSupervisor().run(
        {
            "operation": name,
            "command": request.model_dump(mode="json"),
            "settings": {
                "proc_root": str(root),
                "infiniband_root": str(tmp_path / "ib"),
                "diagnostic_output_dir": str(tmp_path / "evidence"),
                "python_stack_tool": "",
            },
        },
        15,
    )
    assert result.get("status") not in {
        "UNAVAILABLE",
        "WORKER_EXITED",
        "PARTIAL_TIMEOUT",
    }, result
    assert result.get("evidence_complete") is not False, result
