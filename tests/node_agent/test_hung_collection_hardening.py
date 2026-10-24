import json
import os
from subprocess import CompletedProcess
from threading import Thread

import pytest

from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent import NodeActionStatus
from tests.node_agent._cov95_runtime_hung import (
    COMPUTE_QUERY,
    BundleRunner,
    capture_bundle,
    process,
)
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import command, envelope


def stat_text(pid, start):
    fields = ["S"] + ["0"] * 19
    fields[19] = str(start)
    return f"{pid} (python) " + " ".join(fields)


@pytest.mark.parametrize("flight", [False, True, "large-group"])
def test_scoped_triage_collects_real_fifo_dump_or_explicit_unavailability(
    node_factory, tmp_path, flight
):
    owned = process(tmp_path / "proc", 100)
    foreign = process(tmp_path / "proc", 200, cgroup="/other-job")
    for directory, pid in [(owned, 100), (foreign, 200)]:
        (directory / "stat").write_text(stat_text(pid, 1000))
        (directory / "environ").write_bytes(f"RANK={pid // 100 - 1}\0".encode())
    worker = None
    if flight:
        namespace = owned / "root" / "tmp"
        namespace.mkdir(parents=True)
        pipe = namespace / "trace0.pipe"
        os.mkfifo(pipe)
        # Keep a local reader open before the agent's nonblocking FIFO write.
        descriptor = os.open(pipe, os.O_RDONLY | os.O_NONBLOCK)
        (owned / "environ").write_bytes(
            b"RANK=0\0TORCH_NCCL_DEBUG_INFO_PIPE_FILE=/tmp/trace\0TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/dump\0"
        )

        def dump():
            import select

            assert select.select([descriptor], [], [], 3)[0], (
                "agent must trigger the configured pipe"
            )
            os.read(descriptor, 100)
            (namespace / "dump0").write_text(
                json.dumps(
                    {
                        "pg_config": {
                            "0": {
                                "ranks": json.dumps(list(range(10000)))
                                if flight == "large-group"
                                else "[0, 1, 2]"
                            },
                            "1": {"ranks": [0, 3, 4]},
                        },
                        "entries": [
                            {
                                "process_group": ["0", "dp"],
                                "collective_seq_id": 10,
                                "state": "started",
                            },
                            {
                                "process_group": ["1", "tp"],
                                "collective_seq_id": 900,
                                "state": "completed",
                            },
                        ],
                    }
                )
            )

        worker = Thread(target=dump)
        worker.start()
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return CompletedProcess(
            argv,
            0,
            "100, GPU-a, python\n200, GPU-a, python\n" if COMPUTE_QUERY in argv else "",
            "",
        )

    agent = node_factory(
        allowed_operations={Op.COLLECT_HUNG_TRIAGE}, runner=runner, python_stack_tool=""
    )
    try:
        result = agent.execute(
            envelope(
                command(
                    Op.COLLECT_HUNG_TRIAGE,
                    parameters={
                        "require_workload_identity": True,
                        "workload_cgroup_paths_by_node": {"node-a": ["/training/job"]},
                    },
                )
            )
        )
        assert result.status is NodeActionStatus.SUCCEEDED, result
        [row] = result.details["ranks"]
        assert row["pid"] == 100, result.details
        assert row["process_identity"] == {"verified": True, "start_time_ticks": 1000}
        recorder = row["flight_recorder"]
        assert recorder["status"] == ("dumped" if flight else "not_configured"), (
            recorder
        )
        if flight:
            assert {
                entry["pg_name"]: entry["collective_seq_id"]
                for entry in recorder["last_entries_by_pg"]
            } == {"0:dp": 10, "1:tp": 900}
            first = recorder["last_entries_by_pg"][0]
            if flight == "large-group":
                assert first["pg_member_count"] == 10000
                assert "pg_members" not in first, (
                    "large membership must not be repeated per rank"
                )
                assert len(json.dumps(recorder)) < 10000, recorder
            else:
                assert first["pg_members"] == [0, 1, 2]
        assert not any(
            "strace" in arg or "dcgmi" in arg for argv in calls for arg in argv
        ), calls
    finally:
        if worker is not None:
            worker.join(timeout=4)
            os.close(descriptor)
            assert not worker.is_alive(), "dump writer must terminate"


@pytest.mark.parametrize(
    "reason", ["EFA_TRAFFIC_HUNG_SUSPECTED", "TRAINING_HUNG", "APPLICATION_NCCL_ERROR"]
)
def test_hung_bundle_omits_intrusive_diagnostics_and_reused_pid(
    node_factory, tmp_path, reason
):
    directory = process(tmp_path / "proc", 100)
    (directory / "stat").write_text(stat_text(100, 9999))
    runner = BundleRunner("100, GPU-a, python\n")
    agent = node_factory(
        allowed_operations={Op.COLLECT_DIAGNOSTIC_BUNDLE},
        runner=runner,
        python_stack_tool="",
    )
    manifest, contents = capture_bundle(
        agent,
        diagnostic_reason=reason,
        target_pids_by_node={"node-a": [100]},
        target_start_ticks_by_pid_by_node={"node-a": {"100": 1000}},
        require_process_start_identity=True,
    )
    assert manifest["gpu_processes"] == [], manifest
    assert (
        manifest["skipped_processes"][0]["reason"]
        == "process_identity_changed_or_missing"
    ), manifest
    assert "dcgm-diag.json" not in contents
    assert "nvidia-bug-report.log.gz" not in contents
    assert not any(
        argv[0] in {"dcgmi", "/usr/bin/nvidia-bug-report.sh", "timeout", "py-spy"}
        for argv in runner.calls
    ), runner.calls
