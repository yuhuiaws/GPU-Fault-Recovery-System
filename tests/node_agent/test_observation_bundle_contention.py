"""A hung diagnostic bundle on a shared node is waited for, then handed back partial.

Live (GF-GENERAL-LIVE-011 ``fr_second_missing``, HyperPod, 2026-10-26
04:46:00-04:46:13Z, release 37db33f254a1): triage sampled every rank of both
nodes, but step 5 COLLECT_DIAGNOSTIC_BUNDLE ended FAILED with ``node_failures``
= ``ObservationIncomplete: WORKER_STILL_RUNNING`` on ``hyperpod-i-08d8…`` alone
-- the node shared with the other lane's prewarm/training Pods, whose passive
communication sampler held the single observation worker slot when the bundle
arrived; the other node finished and the workflow escalated. The bundle takes
the same slot-wait and preemption as the triage, and a bundle the worker runs
but cannot finish in its budget hands back the manifest it captured, the
captures still pending and the elapsed budget, not a bare status.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent.communication_runtime import CommunicationRuntime
from gpu_fault.node_agent.observation_supervisor import (
    SLOT_WAIT_MAX_SECONDS,
    ObservationSlotBusy,
    ObservationSupervisor,
)
from tests.node_agent._cov95_runtime_hung import process
from tests.node_agent._observation_deadline import DeadlineClock
from tests.node_agent._support import command
from tests.node_agent.test_communication_monitor import (
    REAL_WORKER_BUDGET_SECONDS,
    use_imported_package,
)
from tests.node_agent.test_hung_triage_result_budget import NODE_ID, _Activity
from tests.node_agent.test_observation_slot_contention import (
    blocking_nvidia_smi,
    run_in_thread,
    sample_request,
    wait_for,
)
from tests.node_agent.test_process_diagnostics_e2e import proc_stat


def bundle_command(**extra: Any):
    return command(
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
        command_id="workflow-a/5/COLLECT_DIAGNOSTIC_BUNDLE/node-a/agent-7",
        gpu_uuids=[],
        parameters={
            "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
            "job_id": "gl11-frsm",
            "attempt_id": "gl11-frsm-a001",
            "disable_strace": True,
            "capture_process_state": False,
            "bundle_scope": "bounded_triage_nodes",
            "raw_flights_by_node": {NODE_ID: []},
            "target_pids_by_node": {},
            **extra,
        },
    )


def _runtime(tmp_path: Path, monkeypatch) -> CommunicationRuntime:
    monkeypatch.setattr(
        "gpu_fault.node_agent.communication_runtime.service_limits",
        lambda: {"verified": True, "cpu_cores": 2, "memory_bytes": 1 << 30},
    )
    boot = tmp_path / "proc" / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True, exist_ok=True)
    boot.write_text("boot")
    return CommunicationRuntime(
        SimpleNamespace(
            node_ids={NODE_ID},
            agent_generation=1,
            proc_root=tmp_path / "proc",
            infiniband_root=tmp_path / "ib",
            diagnostic_output_dir=tmp_path / "diagnostics",
            python_stack_tool="",
            diagnostic_s3_uri=None,
            diagnostic_retention_seconds=600,
            diagnostic_max_archives=10,
            gpu_activity=_Activity(),
        )
    )


def test_a_bundle_preempts_the_passive_sample_holding_the_shared_slot(
    tmp_path, monkeypatch
):
    """The live cause: the sampler holds 08d8's slot; the bundle waits and runs.

    The sample's own deadline is one it never reaches
    (``REAL_WORKER_BUDGET_SECONDS``), not 12 s of real time that a CPU-starved
    host spent starting its worker: only the preemption frees the slot within
    the bundle's 10 s wait.
    """

    root = tmp_path / "proc"
    directory = process(root, 100)
    (directory / "stat").write_text(proc_stat(100, 0))
    fifo, marker = tmp_path / "blocked", tmp_path / "started"
    os.mkfifo(fifo)
    blocking_nvidia_smi(tmp_path / "slow", fifo, marker)
    use_imported_package(monkeypatch)
    monkeypatch.setenv("PATH", str(tmp_path / "slow"))
    runtime = _runtime(tmp_path, monkeypatch)
    passive, box = run_in_thread(
        runtime.supervisor,
        {**sample_request(root, tmp_path), "settings": runtime.settings},
        REAL_WORKER_BUDGET_SECONDS,
        preemptible=True,
    )
    wait_for(marker, REAL_WORKER_BUDGET_SECONDS)

    fast = tmp_path / "fast"
    fast.mkdir()
    (fast / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (fast / "nvidia-smi").chmod(0o700)
    monkeypatch.setenv("PATH", str(fast))
    result = runtime.run_command("bundle", bundle_command())
    assert result["manifest_summary"]["capture_count"] >= 0, result
    assert 0 < result["slot_wait_seconds"] < SLOT_WAIT_MAX_SECONDS, (
        "the bundle ran once the preempted sample left the slot",
        result,
    )
    passive.join(timeout=REAL_WORKER_BUDGET_SECONDS)
    assert box["result"]["status"] == "PREEMPTED", box["result"]
    assert box["result"]["preempted_by"] == "bundle", box["result"]


def test_a_bundle_refused_for_the_whole_wait_is_retryable(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch)
    refusal = {
        "status": "WORKER_STILL_RUNNING",
        "evidence_complete": False,
        "ranks": [],
        "error": "observation worker slot held by sample for 11.0 s; waited 10.0 s",
        "failed_component": "observation_supervisor",
        "failure_stage": "WORKER_NOT_STARTED",
        "worker_started": False,
        "slot_wait_seconds": 10.0,
    }
    seen: list[float] = []

    def run(request, timeout, *, release, preemptible=False, slot_wait=0.0):
        seen.append(slot_wait)
        release()
        return dict(refusal)

    monkeypatch.setattr(runtime.supervisor, "run", run)
    try:
        runtime.run_command("bundle", bundle_command())
    except ObservationSlotBusy as exc:
        assert exc.action_details["worker_started"] is False
    else:
        raise AssertionError("a refused bundle must raise the retryable slot refusal")
    # The bundle's 120 s timeout caps its slot wait at SLOT_WAIT_MAX_SECONDS.
    assert seen == [10.0], seen


def test_the_bundle_deadline_hands_back_the_captured_manifest(tmp_path, monkeypatch):
    """A bundle the worker ran but could not finish returns what it captured.

    The deadline falls once the supervisor has read the stand-in's last
    checkpoint (``checkpointed``, ``DeadlineClock``), not after 1.5 s of real
    time, in which a CPU-starved host had not yet started the stand-in (b210).
    """

    from gpu_fault.node_agent import observation_supervisor

    use_imported_package(monkeypatch)
    monkeypatch.setattr(
        observation_supervisor,
        "observation_guard",
        SimpleNamespace(__name__="tests.node_agent._stalled_bundle_worker"),
    )
    checkpointed = tmp_path / "checkpointed"
    clock = DeadlineClock(REAL_WORKER_BUDGET_SECONDS, when=checkpointed.exists)
    clock.install(monkeypatch)
    supervisor = ObservationSupervisor()
    result = supervisor.run(
        {"operation": "bundle", "settings": {}, "checkpointed": str(checkpointed)},
        REAL_WORKER_BUDGET_SECONDS,
    )
    returned = time.monotonic()
    assert clock.expired_at is not None, ("the stand-in never checkpointed", result)
    assert returned - clock.expired_at < 2.5, "the supervisor returned at its deadline"
    assert result["status"] == "PARTIAL_TIMEOUT", result
    assert result["evidence_complete"] is False, result
    assert result["collection_stage"] == "BUNDLE_CAPTURING", result
    assert result["captures_done"] == ["nvidia-smi.txt", "dmesg.txt"], result
    assert result["captures_pending"] == ["ps.txt"], result
    assert result["manifest"]["node_id"] == "node-a", result
    assert result["supervisor_timeout_seconds"] == REAL_WORKER_BUDGET_SECONDS, result
