import json
from datetime import timedelta
from subprocess import CompletedProcess

import pytest

from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent import NodeActionStatus
from gpu_fault.rank_progress import RankProgressSnapshot, write_rank_progress
from tests.node_agent._cov95_runtime_hung import process
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import NOW, command, envelope
from tests.node_agent.test_hung_collection_hardening import stat_text


def progress(rank, world, **updates):
    return {
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "rank": rank,
        "world_size": world,
        "process_start_ticks": 1000,
        "process_pid": 100,
        "observed_at": NOW.isoformat(),
        "step": 10,
        "phase": "TRAINING",
        "process_groups": [
            {
                "name": "dp",
                "members": list(range(world)),
                "enqueued": 10,
                "started": 10,
                "completed": 9,
            }
        ],
        **updates,
    }


def summary_command():
    return command(
        Op.COLLECT_HUNG_SUMMARY,
        parameters={
            "job_id": "job-a",
            "attempt_id": "attempt-a",
            "snapshot_epoch": "epoch",
            "workload_cgroup_paths_by_node": {"node-a": ["/training/job"]},
        },
    )


def test_summary_visits_all_scoped_ranks_without_fr_or_stack_tools(
    node_factory, tmp_path
):
    for rank in range(40):
        directory = process(tmp_path / "proc", 100 + rank)
        (directory / "stat").write_text(stat_text(100 + rank, 1000))
        (directory / "environ").write_bytes(f"RANK={rank}\0WORLD_SIZE=40\0".encode())
    outsider = process(tmp_path / "proc", 999, cgroup="/other-job")
    (outsider / "environ").write_bytes(b"RANK=99\0WORLD_SIZE=100\0")
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return CompletedProcess(argv, 0, "", "")

    agent = node_factory(allowed_operations={Op.COLLECT_HUNG_SUMMARY}, runner=runner)
    result = agent.execute(envelope(summary_command()))
    assert result.status is NodeActionStatus.SUCCEEDED, result
    assert {row["rank"] for row in result.details["ranks"]} == set(range(40))
    assert result.details["attachments"] == 0
    assert result.details["scan_complete"] is True
    assert len(calls) == 2 and all(argv[0] == "nvidia-smi" for argv in calls), calls


@pytest.mark.parametrize(
    "defect", [None, "old-attempt", "stale", "reused-pid", "missing-file"]
)
def test_cooperative_progress_is_bound_to_attempt_process_and_time(
    node_factory, tmp_path, defect
):
    directory = process(tmp_path / "proc", 100)
    (directory / "stat").write_text(stat_text(100, 1000))
    (directory / "environ").write_bytes(
        b"RANK=0\0WORLD_SIZE=3\0GPU_FAULT_RANK_PROGRESS_FILE=/tmp/progress.json\0"
    )
    namespace = directory / "root" / "tmp"
    namespace.mkdir(parents=True)
    document = progress(0, 3)
    if defect == "old-attempt":
        document["attempt_id"] = "old-attempt"
    if defect == "stale":
        document["observed_at"] = (NOW - timedelta(seconds=60)).isoformat()
    if defect != "missing-file":
        (namespace / "progress.json").write_text(json.dumps(document))

    def sleep(_):
        if defect == "reused-pid":
            (directory / "stat").write_text(stat_text(100, 9999))

    agent = node_factory(allowed_operations={Op.COLLECT_HUNG_SUMMARY}, sleep=sleep)
    result = agent.execute(envelope(summary_command()))
    assert result.status is NodeActionStatus.SUCCEEDED, result
    [row] = result.details["ranks"]
    assert row["communication_status"] == (
        "AVAILABLE" if defect is None else "UNAVAILABLE"
    ), row
    if defect is None:
        assert len(result.details["groups"]) == len(row["communication"]) == 1
    else:
        assert row["communication"] == [], row


@pytest.mark.parametrize("environment", [b"", b"RANK=bad\0", b"WORLD_SIZE=8\0"])
def test_summary_keeps_unranked_workload_processes(node_factory, tmp_path, environment):
    directory = process(tmp_path / "proc", 100)
    (directory / "stat").write_text(stat_text(100, 1000))
    (directory / "environ").write_bytes(environment)
    agent = node_factory(allowed_operations={Op.COLLECT_HUNG_SUMMARY})
    result = agent.execute(envelope(summary_command()))
    [row] = result.details["ranks"]
    assert row["pid"] == 100 and row["rank"] is None, row
    assert row["process_identity"]["verified"] is True, row
    assert result.details["unranked_processes"] == 1, result


def test_communication_counters_ignore_invalid_training_step_and_phase(
    node_factory, tmp_path
):
    directory = process(tmp_path / "proc", 100)
    (directory / "stat").write_text(stat_text(100, 1000))
    (directory / "environ").write_bytes(
        b"RANK=0\0WORLD_SIZE=3\0GPU_FAULT_RANK_PROGRESS_FILE=/tmp/progress.json\0"
    )
    namespace = directory / "root" / "tmp"
    namespace.mkdir(parents=True)
    (namespace / "progress.json").write_text(
        json.dumps(progress(0, 3, step="not-a-step", phase="not-a-phase"))
    )
    agent = node_factory(allowed_operations={Op.COLLECT_HUNG_SUMMARY})
    result = agent.execute(envelope(summary_command()))
    [row] = result.details["ranks"]
    assert len(row["communication"]) == 1, row
    assert "application" not in row and "step" not in row, row


def test_missing_or_reset_rdma_pairs_are_unknown_not_zero(node_factory, tmp_path):
    directory = tmp_path / "infiniband" / "efa0" / "ports" / "1" / "hw_counters"
    directory.mkdir(parents=True)
    (directory / "rx_bytes").write_text("100")
    (directory / "tx_bytes").write_text("100")

    def reset(_):
        (directory / "tx_bytes").write_text("1")

    agent = node_factory(allowed_operations={Op.COLLECT_HUNG_SUMMARY}, sleep=reset)
    result = agent.execute(envelope(summary_command()))
    pair = result.details["rdma"]["efa0/1"]
    assert pair["valid"] is False, pair
    assert pair["rx_bytes_delta"] is pair["tx_bytes_delta"] is None


def test_application_writer_produces_atomic_process_bound_snapshot(
    tmp_path, monkeypatch
):
    path = tmp_path / "progress.json"
    monkeypatch.setenv("GPU_FAULT_RANK_PROGRESS_FILE", str(path))
    write_rank_progress(
        job_id="job-a", attempt_id="attempt-a", rank=0, world_size=2, step=10
    )
    value = RankProgressSnapshot.model_validate_json(path.read_text())
    assert value.rank == 0 and value.step == 10 and value.process_start_ticks > 0, value
    assert list(tmp_path.iterdir()) == [path], "no temporary snapshots left behind"


def test_gpu_and_bidirectional_rdma_values_are_device_scoped(node_factory, tmp_path):
    port = tmp_path / "infiniband" / "efa0" / "ports" / "1" / "hw_counters"
    port.mkdir(parents=True)
    (port / "rx_bytes").write_text("100")
    (port / "tx_bytes").write_text("100")

    def runner(argv, **kwargs):
        text = (
            "GPU-bad, N/A, N/A\nGPU-a, 50, 20\nGPU-other, 90, 90\n"
            if "--query-gpu=uuid,utilization.gpu,utilization.memory" in argv
            else ""
        )
        return CompletedProcess(argv, 0, text, "")

    def advance(_):
        (port / "rx_bytes").write_text("120")
        (port / "tx_bytes").write_text("130")

    agent = node_factory(
        allowed_operations={Op.COLLECT_HUNG_SUMMARY}, runner=runner, sleep=advance
    )
    result = agent.execute(envelope(summary_command()))
    assert result.details["gpu"] == {
        "GPU-a": {"utilization_gpu_percent": 50, "utilization_memory_percent": 20}
    }
    assert result.details["rdma"]["efa0/1"]["rx_bytes_delta"] == 20
    assert result.details["rdma"]["efa0/1"]["tx_bytes_delta"] == 30


@pytest.mark.parametrize("state", ["Z", "X"])
def test_exited_gpu_process_is_not_a_live_communication_participant(
    node_factory, tmp_path, state
):
    from gpu_fault.hung_participants import participant
    from tests.node_agent.test_process_diagnostics_e2e import proc_stat

    directory = process(tmp_path / "proc", 100)
    (directory / "stat").write_text(proc_stat(100, 0, state))
    (directory / "environ").write_bytes(b"RANK=0\0WORLD_SIZE=1\0")

    def runner(argv, **kwargs):
        return CompletedProcess(
            argv,
            0,
            "100,GPU-a\n" if "--query-compute-apps=pid,gpu_uuid" in argv else "",
            "",
        )

    agent = node_factory(allowed_operations={Op.COLLECT_HUNG_SUMMARY}, runner=runner)
    result = agent.execute(envelope(summary_command()))
    [row] = result.details["ranks"]
    assert participant(row) is False and row["process_identity"]["verified"] is False, (
        row
    )
