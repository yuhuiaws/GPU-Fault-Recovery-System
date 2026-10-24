import json
import socket
import threading
import time
from datetime import timedelta
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

import gpu_fault
from gpu_fault.communication_monitoring import observation_plan
from gpu_fault.node_agent.communication_history import CommunicationHistory
from gpu_fault.node_agent.observation_limits import service_limits
from gpu_fault.node_agent.observation_supervisor import ObservationSupervisor
from gpu_fault.node_agent.operations.network_bindings import task_rdma_bindings
from gpu_fault.ras_observation import parse_ras, query_ras
from gpu_fault.ras_progress import ras_progress
from tests._builders import attempt_observation, container_observation
from tests.execution._flight_series_support import NOW
from tests.node_agent._cov95_runtime_hung import process
from tests.node_agent.test_process_diagnostics_e2e import proc_stat


def use_imported_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the supervisor's child processes import the gpu_fault this test imported.

    The deploy's release gate runs pytest with the deploy-host venv, whose installed gpu_fault has
    no node_agent; the test reaches the source tree through sys.path, which a child does not inherit.
    """
    monkeypatch.setenv("PYTHONPATH", str(Path(gpu_fault.__file__).resolve().parents[1]))


def ras_document():
    return {
        "nccl_version": "2.28.9",
        "communicators_count": 1,
        "communicators": [
            {
                "hash": "0xab",
                "secondary_hash": "0xcd:0xef",
                "size": 3,
                "ranks_count": 3,
                "missing_ranks_count": 0,
                "missing_ranks": [],
                "ranks": [
                    {
                        "rank": rank,
                        "host": f"10.0.0.{rank + 1}",
                        "pid": 100 + rank,
                        "status": {
                            "init_state": 0,
                            "async_error": 0,
                            "finalize_called": False,
                            "destroy_flag": False,
                            "abort_flag": False,
                        },
                        "collective_counts": {
                            "AllReduce": 9 if rank == 0 else 10,
                            "SendRecv": rank,
                        },
                    }
                    for rank in range(3)
                ],
            }
        ],
        "ras": {"timeouts_count": 0, "collection_time_sec": 0.1},
    }


def test_real_local_ras_protocol_two_and_json_response():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    errors = []

    def server():
        try:
            with listener.accept()[0] as client, client.makefile("rb") as stream:
                for request, response in [
                    (b"CLIENT PROTOCOL 2\n", b"SERVER PROTOCOL 2\n"),
                    (b"TIMEOUT 2\n", b"OK\n"),
                    (b"SET FORMAT json\n", b"OK\n"),
                    (b"VERBOSE STATUS\n", json.dumps(ras_document()).encode()),
                ]:
                    assert stream.readline() == request
                    client.sendall(response)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=server)
    worker.start()
    try:
        result = query_ras("127.0.0.1", listener.getsockname()[1])
        assert result["groups"][0]["rank_namespace"] == "COMMUNICATOR_LOCAL", result
        assert result["completion_progress_available"] is False, result
    finally:
        listener.close()
        worker.join(timeout=3)
    assert not worker.is_alive() and not errors, errors


@pytest.mark.parametrize("defect", ["duplicate", "count", "version", "negative"])
def test_ras_rejects_invalid_or_ambiguous_evidence(defect):
    payload = ras_document()
    if defect == "duplicate":
        payload["communicators"][0]["ranks"][1]["rank"] = 0
    elif defect == "count":
        payload["communicators_count"] = 2
    elif defect == "version":
        payload["nccl_version"] = "2.26.0"
    else:
        payload["communicators"][0]["ranks"][0]["collective_counts"]["AllReduce"] = -1
    with pytest.raises(ValueError):
        parse_ras(json.dumps(payload).encode())


def test_ras_lag_is_a_submission_lead_and_not_a_completed_collective():
    snapshot = {**parse_ras(json.dumps(ras_document()).encode()), "status": "COLLECTED"}
    result = ras_progress(snapshot, snapshot)
    [candidate] = result["groups"][0]["candidate_members"]
    assert candidate["communicator_rank"] == 0 and candidate["confidence"] == "WEAK", (
        result
    )
    assert result["completion_progress_available"] is False, result
    assert result["root_cause_confirmed"] is False, result


def test_history_is_bounded_and_does_not_cross_attempt_or_boot(tmp_path):
    history = CommunicationHistory(tmp_path / "history.db", max_bytes=4096, max_age=60)
    scope = history.scope("job", "attempt", "boot")
    for epoch in range(80):
        history.append(scope, epoch, epoch, {"pid": epoch, "value": "x" * 500})
    rows = history.read(scope, since=0, until=100)
    assert rows and rows[0]["observed_at_epoch"] >= 19, rows
    assert history.read(history.scope("job", "other", "boot"), since=0, until=100) == []
    assert (
        history.read(history.scope("job", "attempt", "new-boot"), since=0, until=100)
        == []
    )
    with history.connect() as db:
        assert db.execute("SELECT SUM(size) FROM sample").fetchone()[0] <= 4096


def test_live_qps_bind_task_nic_and_missing_participants_remain_unknown(tmp_path):
    directory = process(tmp_path / "proc", 100)
    (directory / "stat").write_text(proc_stat(100, 0))
    (tmp_path / "ib/efa0/ports/1").mkdir(parents=True)
    rows = [
        {
            "pid": 100,
            "gpu_uuids": ["GPU-a"],
            "process_identity": {"verified": True, "start_time_ticks": 1000},
        }
    ]
    agent = SimpleNamespace(
        proc_root=tmp_path / "proc",
        infiniband_root=tmp_path / "ib",
        runner=lambda argv, **kwargs: CompletedProcess(
            argv,
            0,
            json.dumps(
                [
                    {"pid": 100, "ifname": "efa0", "port": 1, "state": "RTS"},
                    {"pid": 900, "ifname": "management", "port": 1, "state": "RTS"},
                ]
            ),
            "",
        ),
    )
    result = task_rdma_bindings(agent, rows, ["/training/job"])
    assert result["basis"] == "TASK_DEVICE_BINDING" and result["rdma_ports"] == [
        "efa0/1"
    ], result
    (directory / "stat").write_text(proc_stat(100, 0, start=9999))
    assert task_rdma_bindings(agent, rows, ["/training/job"])["basis"] == "UNKNOWN"


def _qp(pid, lqpn, ifname="rdmap16s27"):
    """One ``rdma -j resource show qp`` row (iproute2 ``res_qp_line`` keys)."""

    return {
        "ifindex": 2,
        "ifname": ifname,
        "port": 1,
        "lqpn": lqpn,
        "type": "DRIVER",
        "state": "RTS",
        "pid": pid,
        "comm": "pt_nccl_proxy",
    }


def _thread_agent(tmp_path, resources):
    rank = process(tmp_path / "proc", 100)
    (rank / "stat").write_text(proc_stat(100, 0))
    for thread in (100, 131):
        (rank / "task" / str(thread)).mkdir(parents=True)
    other = process(tmp_path / "proc", 900, cgroup="/other-job")
    (other / "stat").write_text(proc_stat(900, 0))
    (other / "task/905").mkdir(parents=True)
    (tmp_path / "ib/rdmap16s27/ports/1").mkdir(parents=True)
    return SimpleNamespace(
        proc_root=tmp_path / "proc",
        infiniband_root=tmp_path / "ib",
        runner=lambda argv, **kwargs: CompletedProcess(
            argv, 0, json.dumps(resources), ""
        ),
    )


RANK_ROWS = [
    {
        "pid": 100,
        "gpu_uuids": ["GPU-a"],
        "process_identity": {"verified": True, "start_time_ticks": 1000},
    }
]


def test_a_qp_created_by_a_rank_thread_binds_that_rank(tmp_path):
    """The kernel names the thread that created a QP, never its process.

    ``rdma resource show qp`` prints ``task_pid_vnr(res->task)`` and a user QP's
    restrack task is ``current`` at ``ibv_create_qp`` time; NCCL runs the net
    plugin's listen/connect -- where aws-ofi-nccl creates its per-thread
    endpoints and libfabric's EFA QPs -- on the rank's proxy thread. Live
    (HyperPod, GF-GENERAL-LIVE-011): every QP named a proxy TID, no rank PID
    matched, and the hung VALIDATE_FABRIC waited for a scope that never came.
    """

    agent = _thread_agent(
        tmp_path,
        [
            _qp(131, 7),
            # A thread of another job's process on the same port is not ours.
            _qp(905, 8),
            {**_qp(None, 1), "type": "GSI", "comm": "[ib_core]"},
        ],
    )
    result = task_rdma_bindings(agent, RANK_ROWS, ["/training/job"])
    assert result["basis"] == "TASK_DEVICE_BINDING", result
    assert result["rdma_ports"] == ["rdmap16s27/1"], result
    assert result["covered_pids"] == [100] and result["coverage_complete"], result


def test_only_a_foreign_thread_on_the_port_leaves_the_rank_unknown(tmp_path):
    agent = _thread_agent(tmp_path, [_qp(905, 8), _qp(4242, 9)])
    result = task_rdma_bindings(agent, RANK_ROWS, ["/training/job"])
    assert result["basis"] == "UNKNOWN", result
    assert result["reason"] == "TASK_RDMA_COVERAGE_INCOMPLETE", result
    assert result["rdma_ports"] == [] and result["covered_pids"] == [], result


def test_service_resource_envelope_requires_real_finite_limits(tmp_path):
    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/self/cgroup").write_text("0::/agent\n")
    group = tmp_path / "cgroup/agent"
    group.mkdir(parents=True)
    (group / "cpu.max").write_text("200000 100000")
    (group / "memory.max").write_text(str(1024**3))
    assert service_limits(tmp_path / "proc", tmp_path / "cgroup")["verified"] is True
    (group / "cpu.max").write_text("max 100000")
    assert service_limits(tmp_path / "proc", tmp_path / "cgroup")["verified"] is False


def test_target_projection_is_scoped_cached_and_elects_one_ras_node():
    class Store:
        calls = 0

        def list_attempt_observations(self, cluster):
            self.calls += 1
            return [
                attempt_observation(
                    "job",
                    "attempt",
                    NOW,
                    containers=[
                        container_observation(
                            f"pod-{rank}",
                            "trainer",
                            rank,
                            f"node-{rank}",
                            cgroup_path="/job",
                        )
                        for rank in range(128)
                    ],
                )
            ]

    store = Store()
    from gpu_fault.models import WorkflowOperation as Op

    plans = []
    for rank in range(128):
        record = SimpleNamespace(
            node_id=f"node-{rank}",
            cluster_id="cluster-a",
            boot_id="boot",
            generation=1,
            allowed_operations=[Op.COLLECT_HUNG_SUMMARY],
            lifecycle_state=SimpleNamespace(value="ACTIVE"),
            endpoint=f"https://10.0.0.{rank + 1}:9443",
        )
        plans.append(observation_plan(store, record, NOW))
    assert store.calls == 1, "large tasks must not be reread from the DB once per node"
    assert sum(plan.targets[0].ras_coordinator for plan in plans) == 1
    assert all(
        plan.expires_at - plan.issued_at == timedelta(seconds=90) for plan in plans
    ), "every task scope must expire within its bounded lease"


@pytest.mark.parametrize("path", [None, "/", "/job/../other"])
def test_missing_task_scope_remains_in_coverage_denominator(path):
    class Store:
        def list_attempt_observations(self, cluster):
            return [
                attempt_observation(
                    "job",
                    "attempt",
                    NOW,
                    containers=[
                        container_observation(
                            "pod", "trainer", 0, "node-a", cgroup_path=path
                        )
                    ],
                )
            ]

    from gpu_fault.models import WorkflowOperation as Op

    record = SimpleNamespace(
        node_id="node-a",
        cluster_id="cluster-a",
        boot_id="boot",
        generation=1,
        allowed_operations=[Op.COLLECT_HUNG_SUMMARY],
        lifecycle_state=SimpleNamespace(value="ACTIVE"),
        endpoint="https://node-a:9443",
    )
    plan = observation_plan(Store(), record, NOW)
    assert plan.expected_target_count == 1 and plan.targets == [], plan


def test_owned_worker_returns_partial_at_deadline(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    root.mkdir()
    import os
    import sys

    directory = process(root, 100)
    (directory / "stat").write_text(proc_stat(100, 0))
    fifo, marker = tmp_path / "blocked", tmp_path / "started"
    os.mkfifo(fifo)
    tool = tmp_path / "nvidia-smi"
    tool.write_text(
        f"#!{sys.executable}\nimport os\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        f"os.open({str(fifo)!r}, os.O_RDONLY)\n"
    )
    tool.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    use_imported_package(monkeypatch)
    supervisor = ObservationSupervisor()
    released = threading.Event()
    started = time.monotonic()
    result = supervisor.run(
        {
            "operation": "sample",
            "settings": {
                "proc_root": str(root),
                "infiniband_root": str(tmp_path / "ib"),
                "diagnostic_output_dir": str(tmp_path / "diagnostics"),
                "python_stack_tool": "",
            },
            "target": {
                "job_id": "job",
                "attempt_id": "attempt",
                "cgroup_paths": ["/training/job"],
            },
            "plan": {
                "cluster_id": "cluster-a",
                "node_id": "node-a",
                "boot_id": "boot",
                "generation": 1,
                "issued_at": NOW.isoformat(),
                "expires_at": (NOW + timedelta(seconds=90)).isoformat(),
                "expected_target_count": 1,
                "targets": [],
            },
        },
        1.5,
        release=released.set,
    )
    assert time.monotonic() - started < 4, result
    assert result["evidence_complete"] is False, result
    assert marker.exists(), "the real owned probe must reach the blocking read"
    assert released.wait(3), (
        "owned probe subtree must be reaped before releasing admission"
    )
    assert not (Path("/proc") / marker.read_text()).exists(), result
