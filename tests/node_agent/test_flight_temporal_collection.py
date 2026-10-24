import json
import os
import select
import time
from datetime import timedelta
from subprocess import CompletedProcess
from threading import Event, Thread

import pytest

from gpu_fault.execution.hung_classification import classify_hung_signals
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowStatus
from gpu_fault.node_agent import NodeActionStatus
from gpu_fault.node_agent.operations import flight_recorder, hung_triage
from tests.execution._cov95_runtime_triage import triage_harness
from tests.execution._flight_series_support import entry
from tests.node_agent._cov95_runtime_hung import COMPUTE_QUERY, process
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import NOW, command, envelope
from tests.node_agent.test_hung_collection_hardening import stat_text


def collect_pair(node_factory, tmp_path, monkeypatch, mode="stalled"):
    class Clock:
        value = 0.0

        def monotonic(self):
            return self.value

        def sleep(self, seconds):
            self.value += seconds
            if mode == "reused-pid" and seconds == 2:
                (tmp_path / "proc/100/stat").write_text(stat_text(100, 9999))
            time.sleep(0.01)

    clock = Clock()
    monkeypatch.setattr(hung_triage, "time", clock)
    monkeypatch.setattr(flight_recorder, "time", clock)
    stop = Event()
    workers, descriptors, requests, failures = [], [], {}, []
    for rank in range(3):
        pid = 100 + rank
        directory = process(tmp_path / "proc", pid)
        (directory / "stat").write_text(stat_text(pid, 1000))
        (directory / "environ").write_bytes(
            f"RANK={rank}\0TORCH_NCCL_DEBUG_INFO_PIPE_FILE=/tmp/trace\0TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/dump\0".encode()
        )
        root = directory / "root/tmp"
        root.mkdir(parents=True)
        pipe = root / f"trace{rank}.pipe"
        os.mkfifo(pipe)
        fd = os.open(pipe, os.O_RDONLY | os.O_NONBLOCK)
        descriptors.append(fd)
        requests[rank] = 0

        def writer(rank=rank, root=root, fd=fd):
            try:
                while not stop.is_set():
                    if not select.select([fd], [], [], 0.05)[0]:
                        continue
                    payload = os.read(fd, 100)
                    if not payload:
                        stop.wait(0.005)
                        continue
                    requests[rank] += payload.count(b"1\n")
                    if mode == "stale-second" and requests[rank] >= 2:
                        continue
                    completed = rank != 0 or mode == "advancing" and requests[rank] >= 2
                    entries = [
                        entry(99, "completed"),
                        entry(100, "completed" if completed else "started"),
                    ]
                    document = {
                        "pg_config": {"0": {"ranks": [0, 1, 2]}},
                        "entries": [
                            {**item, "process_group": ["0", "dp"]} for item in entries
                        ],
                    }
                    target = root / f"dump{rank}"
                    if mode == "partial-second" and requests[rank] >= 2:
                        target.write_text("{")
                        time.sleep(0.015)
                    temporary = root / "pending"
                    temporary.write_text(json.dumps(document))
                    temporary.replace(target)
            except Exception as exc:
                failures.append(exc)

        worker = Thread(target=writer)
        worker.start()
        workers.append(worker)

    def runner(argv, **kwargs):
        text = (
            "\n".join(f"{100 + rank}, GPU-a, python" for rank in range(3))
            if COMPUTE_QUERY in argv
            else ""
        )
        return CompletedProcess(argv, 0, text, "")

    agent = node_factory(
        allowed_operations={Op.COLLECT_HUNG_TRIAGE},
        runner=runner,
        python_stack_tool="",
        sleep=clock.sleep,
        now=lambda: NOW + timedelta(seconds=clock.value),
    )
    try:
        result = agent.execute(
            envelope(
                command(
                    Op.COLLECT_HUNG_TRIAGE,
                    parameters={
                        "require_workload_identity": True,
                        "workload_cgroup_paths_by_node": {"node-a": ["/training/job"]},
                        "triage_timeout_seconds": 20,
                    },
                )
            )
        )
        return result, requests
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=2)
            assert not worker.is_alive(), "owned FIFO responder must terminate"
        for fd in descriptors:
            os.close(fd)
        assert failures == [], failures


@pytest.mark.parametrize(
    "mode", ["stalled", "advancing", "stale-second", "partial-second", "reused-pid"]
)
def test_two_real_fifo_requests_are_fresh_process_bound_and_time_bounded(
    node_factory, tmp_path, monkeypatch, mode
):
    result, requests = collect_pair(node_factory, tmp_path, monkeypatch, mode)
    assert result.status is NodeActionStatus.SUCCEEDED, result
    assert requests == (
        {0: 1, 1: 2, 2: 2} if mode == "reused-pid" else {0: 2, 1: 2, 2: 2}
    ), requests
    assert result.details["elapsed_seconds"] <= 20.5, result
    rows = result.details["ranks"]
    decision = classify_hung_signals(rows, undetermined_nodes=[])
    assert decision["targeted_capture_eligible"] is (
        mode in {"stalled", "partial-second"}
    ), decision
    assert decision["root_cause_confirmed"] is False, decision
    if mode == "stale-second":
        assert all(
            row["flight_recorder_series"]["status"] == "SECOND_UNAVAILABLE"
            for row in rows
        ), rows
    if mode == "advancing":
        assert decision["culprit_ranks"] == [], decision
    if mode in {"stalled", "partial-second"}:
        assert decision["culprit_ranks"] == [0], decision
        assert rows[0]["flight_recorder_series"]["interval_seconds"] >= 2, rows


def test_real_fifo_temporal_evidence_drives_persisted_targeted_bundle(
    node_factory, tmp_path, monkeypatch
):
    result, requests = collect_pair(node_factory, tmp_path, monkeypatch)
    harness = triage_harness({"node_results": {"node-a": result.details}})
    outcome = harness.execute()
    assert outcome.status is WorkflowStatus.RUNNING, outcome
    bundle = harness.store.get_workflow(harness.workflow.request_id).official_steps[1]
    assert bundle.node_ids == ["node-a"], bundle
    assert bundle.parameters["target_pids_by_node"] == {"node-a": [100, 101]}, bundle
    assert bundle.parameters["hung_triage_decision"]["temporal_verified"] is True, (
        bundle
    )
    assert bundle.parameters["hung_triage_decision"]["root_cause_confirmed"] is False, (
        bundle
    )
    assert requests == {0: 2, 1: 2, 2: 2}, requests
