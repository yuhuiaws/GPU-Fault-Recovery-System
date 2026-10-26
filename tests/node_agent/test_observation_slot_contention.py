"""A command that finds the node's observation worker slot busy is not refused.

Live (GF-GENERAL-LIVE-011 ``fr_temporal_completion``, HyperPod, 2026-10-26
04:21:31Z, release 37db33f254a1): COLLECT_HUNG_TRIAGE reached both nodes 1.45 s
after a 30-second wall-clock epoch began -- the moment every Agent's passive
communication sampler starts its 12-second worker on the same single slot. The
supervisor's ``slot.acquire(blocking=False)`` failed on both nodes and each
answered the 70-byte ``{"status":"WORKER_STILL_RUNNING","evidence_complete":
false,"ranks":[]}`` within 1.2 s: no worker ran, the test dump responders'
ledgers show 0 FIFO requests on all 16 ranks, the step recorded ``node_failures``
for both nodes, the control plane did not ask again and the decision saw no
process. Siblings of the same release landed 8 s (``different_process_groups``)
and 18 s (``fr_enabled``) into their epochs and sampled every rank of a node in
~5 s; ``ambiguous_candidates`` (0.14 s in) lost one node the same way. These
tests run the real supervisor and worker.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent import NodeActionStatus, observation_worker
from gpu_fault.node_agent.communication_runtime import CommunicationRuntime
from gpu_fault.node_agent.observation_limits import (
    WORKER_BUDGET_MARGIN_SECONDS,
    WORKER_NOT_STARTED,
)
from gpu_fault.node_agent.observation_supervisor import (
    ObservationSlotBusy,
    ObservationSupervisor,
)
from gpu_fault.node_agent.operations.triage_budget import (
    TRIAGE_RESULT_BUDGET_BYTES,
    encoded_size,
)
from tests.execution._flight_series_support import NOW
from tests.node_agent._cov95_runtime_hung import process
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import command, envelope, node_action_executor
from tests.node_agent.test_communication_monitor import use_imported_package
from tests.node_agent.test_hung_triage_result_budget import (
    NODE_ID,
    DumpResponder,
    FakeRecorderNode,
    _Activity,
)
from tests.node_agent.test_process_diagnostics_e2e import proc_stat

#: The passive sampler's own GPU query (``hung_summary.gpu_sample``); the triage
#: asks for ``clocks_throttle_reasons.active,name`` too, so a fake nvidia-smi
#: can hold the one and answer the other.
PASSIVE_GPU_QUERY = "--query-gpu=uuid,utilization.gpu,utilization.memory"


def sample_request(root: Path, tmp_path: Path) -> dict[str, Any]:
    return {
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
    }


def blocking_nvidia_smi(directory: Path, fifo: Path, marker: Path, *, only: str = ""):
    """An nvidia-smi that blocks on ``fifo`` (and notes its pid in ``marker``).

    With ``only`` set, it blocks for that query alone and answers everything
    else with one idle GPU and no compute app, which is what the triage's own
    queries need to proceed.
    """

    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / "nvidia-smi"
    block = (
        f"    Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        f"    os.open({str(fifo)!r}, os.O_RDONLY)\n"
    )
    if only:
        body = (
            f"if {only!r} in sys.argv:\n{block}"
            "elif '--query-compute-apps=pid,gpu_uuid,process_name' in sys.argv:\n"
            f"    sys.stdout.write(Path({str(directory / 'compute-rows')!r}).read_text())\n"
            "else:\n"
            "    sys.stdout.write("
            "'GPU-00000000-836d-eec4-44cf-48502ab91a20, 0, 0, 0x0, H200\\n')\n"
        )
    else:
        body = f"if True:\n{block}"
    tool.write_text(
        f"#!{sys.executable}\nimport os, sys\nfrom pathlib import Path\n{body}"
    )
    tool.chmod(0o700)
    return tool


def wait_for(path: Path, seconds: float = 8) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.02)


def run_in_thread(supervisor: ObservationSupervisor, *args: Any, **kwargs: Any):
    box: dict[str, Any] = {}

    def target() -> None:
        box["result"] = supervisor.run(*args, **kwargs)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, box


# --------------------------------------------------------------------------- #
# the supervisor's slot
# --------------------------------------------------------------------------- #
def test_a_command_preempts_the_passive_sample_and_runs(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    (root / "sys/kernel/random").mkdir(parents=True)
    directory = process(root, 100)
    (directory / "stat").write_text(proc_stat(100, 0))
    fifo, marker = tmp_path / "blocked", tmp_path / "started"
    os.mkfifo(fifo)
    blocking_nvidia_smi(tmp_path / "slow", fifo, marker)
    use_imported_package(monkeypatch)
    monkeypatch.setenv("PATH", str(tmp_path / "slow"))
    supervisor = ObservationSupervisor()
    released = threading.Event()
    passive, box = run_in_thread(
        supervisor,
        sample_request(root, tmp_path),
        12,
        release=released.set,
        preemptible=True,
    )
    wait_for(marker)

    # A fast tool for the command: an empty scope answers without nvidia-smi.
    fast = tmp_path / "fast"
    fast.mkdir()
    (fast / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (fast / "nvidia-smi").chmod(0o700)
    monkeypatch.setenv("PATH", str(fast))
    empty_root = tmp_path / "empty-proc"
    empty_root.mkdir()
    started = time.monotonic()
    result = supervisor.run(
        {**sample_request(empty_root, tmp_path), "operation": "sample"}, 12, slot_wait=8
    )
    elapsed = time.monotonic() - started

    assert result.get("status") not in {"WORKER_STILL_RUNNING", None}, result
    assert result.get("worker_started", True) is not False, result
    assert 0 < result["slot_wait_seconds"] < 4, result
    assert elapsed < 8, elapsed
    passive.join(timeout=5)
    assert box["result"]["status"] == "PREEMPTED", box["result"]
    assert box["result"]["preempted_by"] == "sample", box["result"]
    assert box["result"]["evidence_complete"] is False, box["result"]
    assert released.wait(3), "the preempted sample's admission is released"
    assert not (Path("/proc") / marker.read_text()).exists(), (
        "the preempted probe subtree is reaped"
    )


def test_a_holder_that_cannot_be_preempted_is_waited_for_then_named(
    tmp_path, monkeypatch
):
    root = tmp_path / "proc"
    directory = process(root, 100)
    (directory / "stat").write_text(proc_stat(100, 0))
    fifo, marker = tmp_path / "blocked", tmp_path / "started"
    os.mkfifo(fifo)
    blocking_nvidia_smi(tmp_path / "slow", fifo, marker)
    use_imported_package(monkeypatch)
    monkeypatch.setenv("PATH", str(tmp_path / "slow"))
    supervisor = ObservationSupervisor()
    released = threading.Event()
    # A command's worker (not preemptible) that will hit its own 3 s deadline.
    holder, box = run_in_thread(
        supervisor, sample_request(root, tmp_path), 3, release=released.set
    )
    wait_for(marker)

    started = time.monotonic()
    refused = supervisor.run(sample_request(root, tmp_path), 12, slot_wait=0.5)
    waited = time.monotonic() - started
    assert refused["status"] == "WORKER_STILL_RUNNING", refused
    assert refused["evidence_complete"] is False and refused["ranks"] == [], refused
    assert refused["worker_started"] is False, refused
    assert refused["failure_stage"] == WORKER_NOT_STARTED, refused
    assert refused["failed_component"] == "observation_supervisor", refused
    assert 0.4 <= refused["slot_wait_seconds"] <= waited < 2, (refused, waited)
    assert refused["error"].startswith("observation worker slot held by sample for"), (
        refused["error"]
    )
    assert "; waited 0.5 s" in refused["error"], refused["error"]
    assert refused["slot_holder"]["operation"] == "sample", refused
    assert refused["slot_holder"]["preempted"] is False, refused
    assert refused["slot_holder"]["held_seconds"] >= 0.4, refused

    # The same command asked again with a longer wait runs once the holder
    # reaches its deadline and leaves the slot.
    fast = tmp_path / "fast"
    fast.mkdir()
    (fast / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (fast / "nvidia-smi").chmod(0o700)
    monkeypatch.setenv("PATH", str(fast))
    empty_root = tmp_path / "empty-proc"
    empty_root.mkdir()
    result = supervisor.run(sample_request(empty_root, tmp_path), 12, slot_wait=8)
    assert result.get("status") != "WORKER_STILL_RUNNING", result
    assert 0 < result["slot_wait_seconds"] < 6, result
    holder.join(timeout=5)
    assert box["result"]["status"] == "PARTIAL_TIMEOUT", box["result"]
    assert "preempted_by" not in box["result"], box["result"]
    assert released.wait(3), "the holder's admission is released once it exits"


def test_a_refusal_names_the_slot_wait_the_caller_allowed(tmp_path, monkeypatch):
    """Without a wait (the passive sampler's own call) the refusal is immediate."""

    root = tmp_path / "proc"
    directory = process(root, 100)
    (directory / "stat").write_text(proc_stat(100, 0))
    fifo, marker = tmp_path / "blocked", tmp_path / "started"
    os.mkfifo(fifo)
    blocking_nvidia_smi(tmp_path / "slow", fifo, marker)
    use_imported_package(monkeypatch)
    monkeypatch.setenv("PATH", str(tmp_path / "slow"))
    supervisor = ObservationSupervisor()
    holder, box = run_in_thread(supervisor, sample_request(root, tmp_path), 2)
    wait_for(marker)
    started = time.monotonic()
    refused = supervisor.run(sample_request(root, tmp_path), 12, preemptible=True)
    assert time.monotonic() - started < 0.5, "no wait was allowed"
    assert refused["status"] == "WORKER_STILL_RUNNING", refused
    assert refused["slot_wait_seconds"] == 0, refused
    assert refused["worker_started"] is False, refused
    holder.join(timeout=5)
    assert box["result"]["status"] == "PARTIAL_TIMEOUT", box["result"]


# --------------------------------------------------------------------------- #
# the live shape through the runtime, the supervisor and the worker
# --------------------------------------------------------------------------- #
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


def triage_command(interval: float = 2):
    return command(
        Op.COLLECT_HUNG_TRIAGE,
        command_id="workflow-a/3/COLLECT_HUNG_TRIAGE/node-a/agent-7",
        gpu_uuids=[],
        parameters={
            "job_id": "gl11-frtc",
            "attempt_id": "gl11-frtc-a001",
            "require_workload_identity": True,
            "workload_cgroup_paths_by_node": {NODE_ID: ["/training/job"]},
            "triage_timeout_seconds": 20,
            "triage_fr_sample_interval_seconds": interval,
        },
    )


def test_a_triage_arriving_during_the_passive_sample_pairs_every_rank(
    tmp_path, monkeypatch
):
    """The live shape: eight ranks a node, two dumps each, the sampler mid-flight."""

    node = FakeRecorderNode(tmp_path / "proc", ranks=8)
    for pipe in node.pipes:
        os.mkfifo(pipe)
    fifo, marker = tmp_path / "blocked", tmp_path / "started"
    os.mkfifo(fifo)
    tools = tmp_path / "bin"
    blocking_nvidia_smi(tools, fifo, marker, only=PASSIVE_GPU_QUERY)
    (tools / "compute-rows").write_text(node.compute_rows())
    monkeypatch.setenv("PATH", str(tools))
    use_imported_package(monkeypatch)
    runtime = _runtime(tmp_path, monkeypatch)
    passive, box = run_in_thread(
        runtime.supervisor,
        {**sample_request(tmp_path / "proc", tmp_path), "settings": runtime.settings},
        12,
        preemptible=True,
    )
    wait_for(marker)
    responder = DumpResponder(node)
    responder.start()
    started = time.monotonic()
    try:
        result = runtime.run_command("triage", triage_command())
    finally:
        responder.close()
    elapsed = time.monotonic() - started

    assert [row["rank"] for row in result["ranks"]] == list(range(8)), result.get(
        "errors"
    )
    assert {row["flight_recorder_series"]["status"] for row in result["ranks"]} == {
        "PAIRED"
    }, [row["flight_recorder_series"].get("status") for row in result["ranks"]]
    assert sorted(node.writes) == sorted([*range(8), *range(8)]), node.writes
    assert 0 < result["slot_wait_seconds"] < 4, result["slot_wait_seconds"]
    assert result["triage_budget_seconds"] == 20, result["triage_budget_seconds"]
    assert result["elapsed_seconds"] < 15, result["elapsed_seconds"]
    assert elapsed < 20, elapsed
    passive.join(timeout=5)
    assert box["result"]["status"] == "PREEMPTED", box["result"]
    assert box["result"]["preempted_by"] == "triage", box["result"]
    # What the report quotes: the measured fit of the live shape.
    print(
        "LIVE-SHAPE 8 ranks: slot_wait=%.2fs triage elapsed=%.2fs fr_wait=%.2fs"
        " wall=%.2fs"
        % (
            result["slot_wait_seconds"],
            result["elapsed_seconds"],
            result["flight_recorder_wait_seconds"],
            elapsed,
        )
    )


def test_a_refusal_after_the_wait_is_a_retryable_failure_of_the_command(
    tmp_path, monkeypatch
):
    """``run_command`` raises the retryable refusal; the executor records it so."""

    runtime = _runtime(tmp_path, monkeypatch)
    refusal = {
        "status": "WORKER_STILL_RUNNING",
        "evidence_complete": False,
        "ranks": [],
        "error": "observation worker slot held by sample for 9.1 s; waited 8.4 s",
        "failed_component": "observation_supervisor",
        "failure_stage": WORKER_NOT_STARTED,
        "worker_started": False,
        "slot_wait_seconds": 8.4,
    }
    seen: list[dict[str, Any]] = []

    def run(request, timeout, *, release, preemptible=False, slot_wait=0.0):
        seen.append({"timeout": timeout, "slot_wait": slot_wait, "pre": preemptible})
        release()
        return dict(refusal)

    monkeypatch.setattr(runtime.supervisor, "run", run)
    try:
        runtime.run_command("triage", triage_command())
    except ObservationSlotBusy as exc:
        assert str(exc) == (
            "WORKER_STILL_RUNNING: observation worker slot held by sample for 9.1 s;"
            " waited 8.4 s"
        )
        assert exc.action_details == refusal
    else:
        raise AssertionError("a refused command must raise")
    assert seen == [{"timeout": 28, "slot_wait": 8.4, "pre": False}], seen


def test_a_slot_refusal_is_retryable_and_a_resubmission_runs_the_next_attempt(tmp_path):
    refusal = {
        "status": "WORKER_STILL_RUNNING",
        "evidence_complete": False,
        "ranks": [],
        "error": "observation worker slot held by sample for 3.0 s; waited 8.4 s",
        "failed_component": "observation_supervisor",
        "failure_stage": WORKER_NOT_STARTED,
        "worker_started": False,
        "slot_wait_seconds": 8.4,
    }
    calls: list[str] = []

    def run_command(operation: str, cmd: Any) -> dict[str, Any]:
        calls.append(operation)
        if len(calls) == 1:
            raise ObservationSlotBusy(dict(refusal))
        return {"node_id": NODE_ID, "ranks": [{"pid": 3000, "rank": 0}]}

    agent = node_action_executor(
        tmp_path,
        "ledger.db",
        allowed_operations={Op.COLLECT_HUNG_TRIAGE},
        proc_root=str(tmp_path / "proc"),
        infiniband_root=str(tmp_path / "ib"),
        diagnostic_output_dir=str(tmp_path / "diagnostics"),
    )
    agent.observation_runtime = SimpleNamespace(run_command=run_command)
    signed = envelope(triage_command())
    first = agent.execute(signed)
    assert first.status is NodeActionStatus.FAILED, first
    assert first.retryable is True, first
    assert first.error == (
        "ObservationSlotBusy: WORKER_STILL_RUNNING: observation worker slot held by"
        " sample for 3.0 s; waited 8.4 s"
    ), first.error
    assert first.details["worker_started"] is False, first.details
    second = agent.execute(signed)
    assert second.status is NodeActionStatus.SUCCEEDED, second
    assert second.attempt == 2 and second.details["ranks"][0]["rank"] == 0, second
    assert calls == ["triage", "triage"], calls


# --------------------------------------------------------------------------- #
# the worker's budget and its partial checkpoints
# --------------------------------------------------------------------------- #
def test_the_worker_plans_the_triage_within_the_budget_it_was_handed(
    tmp_path, monkeypatch
):
    seen: list[float | None] = []

    def collect(self, cmd):
        seen.append(cmd.parameters.get("triage_timeout_seconds"))
        return {"ranks": []}

    monkeypatch.setattr(
        observation_worker.ObservationAgent, "_collect_hung_triage", collect
    )
    settings = {
        "proc_root": str(tmp_path / "proc"),
        "infiniband_root": str(tmp_path / "ib"),
        "diagnostic_output_dir": str(tmp_path / "diagnostics"),
        "python_stack_tool": "",
    }
    request = {
        "operation": "triage",
        "settings": settings,
        "command": triage_command().model_dump(mode="json"),
    }
    observation_worker.execute(request)
    observation_worker.execute(request, budget_seconds=9.5)
    observation_worker.execute(request, budget_seconds=3.0)
    observation_worker.execute(request, budget_seconds=float("inf"))
    assert seen == [20, 9.5 - WORKER_BUDGET_MARGIN_SECONDS, 2.0, 20], seen


def test_the_triage_publishes_its_dumps_before_the_second_interval(
    node_factory, tmp_path, monkeypatch
):
    """The checkpoints a deadline can hand back, in order, each naming its stage."""

    from subprocess import CompletedProcess

    from gpu_fault.node_agent.operations import flight_recorder, hung_triage
    from tests.node_agent._cov95_runtime_hung import COMPUTE_QUERY
    from tests.regional._cov95_runtime_support import Clock

    node = FakeRecorderNode(tmp_path / "proc", ranks=16)
    for pipe in node.pipes:
        os.mkfifo(pipe)
    clock = Clock(step=0.05)
    monkeypatch.setattr(flight_recorder, "time", clock)
    monkeypatch.setattr(hung_triage, "time", clock)
    real_open, real_write, real_close = os.open, os.write, os.close
    descriptors: dict[int, Path] = {}

    def open_pipe(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(path) in node.pipes:
            descriptor = 900_000 + len(descriptors)
            descriptors[descriptor] = Path(path)
            return descriptor
        return real_open(path, flags, *args, **kwargs)

    def write_pipe(descriptor: int, data: bytes) -> int:
        if descriptor not in descriptors:
            return real_write(descriptor, data)
        node.dump(descriptors[descriptor])
        return len(data)

    def runner(argv: list[str], **kwargs: Any) -> CompletedProcess:
        text = node.compute_rows() if COMPUTE_QUERY in argv else ""
        return CompletedProcess(argv, 0, stdout=text, stderr="")

    partials: list[dict[str, Any]] = []
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", open_pipe)
        patch.setattr(os, "write", write_pipe)
        patch.setattr(
            os, "close", lambda fd: None if fd in descriptors else real_close(fd)
        )
        agent = node_factory(
            allowed_operations={Op.COLLECT_HUNG_TRIAGE},
            runner=runner,
            python_stack_tool="",
            sleep=clock.sleep,
        )
        agent.partial_evidence_sink = partials.append
        result = agent.execute(envelope(triage_command()))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert [item["collection_stage"] for item in partials] == [
        "RANKS_DISCOVERED",
        "PROCESS_SERIES_SAMPLED",
        "FIRST_FLIGHT_DUMPS_READ",
    ], [item.get("collection_stage") for item in partials]
    for item in partials:
        assert item["status"] == "COLLECTING" and item["evidence_complete"] is False
        assert item["node_id"] == NODE_ID and item["triage_budget_seconds"] == 20
        assert item["elapsed_seconds"] >= 0, item
    assert [row["rank"] for row in partials[0]["ranks"]] == list(range(16)), partials[0]
    dumps = partials[2]
    assert {row["flight_recorder"]["status"] for row in dumps["ranks"]} == {"dumped"}
    # The first-dump checkpoint is a bounded copy: its rows carry the first
    # dump only, and trimming it never touched the live rows the triage went
    # on to pair (``deepcopy`` in ``hung_triage.checkpoint``).
    assert all(
        (row.get("flight_recorder_series") or {}).get("status") != "PAIRED"
        for row in dumps["ranks"]
    ), "the second dump is not paired at the first-dump checkpoint"
    assert encoded_size(dumps) <= TRIAGE_RESULT_BUDGET_BYTES, encoded_size(dumps)
    note = dumps.get("evidence_trimmed")
    assert note is None or note["within_budget"] is True, note
    assert result.details["triage_budget_seconds"] == 20, result.details.keys()
    # The final result still pairs every rank: the bounded checkpoint's copy
    # did not shorten the evidence the triage was assembling.
    assert {
        row["flight_recorder_series"]["status"] for row in result.details["ranks"]
    } == {"PAIRED"}, [
        row["flight_recorder_series"].get("status") for row in result.details["ranks"]
    ]


def test_the_deadline_hands_back_the_last_checkpoint(tmp_path, monkeypatch):
    """A worker past the supervisor's deadline answers with what it had read."""

    from gpu_fault.node_agent import observation_supervisor

    use_imported_package(monkeypatch)
    monkeypatch.setattr(
        observation_supervisor,
        "observation_guard",
        SimpleNamespace(__name__="tests.node_agent._stalled_observation_worker"),
    )
    supervisor = ObservationSupervisor()
    started = time.monotonic()
    result = supervisor.run({"operation": "triage", "settings": {}}, 1.5)
    assert time.monotonic() - started < 4
    assert result["status"] == "PARTIAL_TIMEOUT", result
    assert result["evidence_complete"] is False, result
    assert result["collection_stage"] == "FIRST_FLIGHT_DUMPS_READ", result
    assert [row["rank"] for row in result["ranks"]] == [0, 1], result
    assert result["ranks"][0]["flight_recorder"]["status"] == "dumped", result
    assert result["supervisor_timeout_seconds"] == 1.5, result
    assert result["worker_returncode"] is None or result["worker_returncode"] != 0, (
        result
    )
