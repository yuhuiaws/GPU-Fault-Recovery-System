"""The read-only worker's request budget against a hung bundle's context.

Live (GF-GENERAL-LIVE-011 ``tid_namespace``), the control plane bound the
bundle step to a ``hung_triage_decision`` of 259 KB -- sixteen processes' per-
thread stacks and wait observations, a few rows more per process than the
``role_mismatch`` variant because a churn thread kept starting short threads --
and the observation request weighed 264 KB against the worker's 256 KiB budget.
The supervisor refused it 5 ms after the action started, before any worker ran,
and the node reported ``{"status": "UNAVAILABLE", "ranks": [], "error":
"ValueError"}``: a whole bundle lost over a field the node never reads, with a
bare class name for a cause. These tests run the real supervisor and worker.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gpu_fault
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent.communication_runtime import (
    CONTEXT_ONLY_PARAMETERS,
    REQUEST_BUDGET_OMISSION,
    CommunicationRuntime,
)
from gpu_fault.node_agent.observation_limits import OBSERVATION_REQUEST_BUDGET_BYTES
from gpu_fault.node_agent.observation_supervisor import (
    WORKER_NOT_STARTED,
    ObservationIncomplete,
    ObservationSupervisor,
)
from gpu_fault.node_agent.protocol import NodeActionStatus
from tests.node_agent._support import (
    FakeRunner,
    command,
    envelope,
    node_action_executor,
)

PROCESSES = 16
ROLES = ("TRAINING", "COMMUNICATION", "BACKGROUND", "BACKGROUND", "UNKNOWN")


def _frames(depth: int) -> list[dict[str, Any]]:
    return [
        {
            "file": "/usr/local/lib/python3.12/threading.py"
            if index % 2
            else "/workspace/train.py",
            "function": "wait" if index % 2 else f"step_{index}",
            "line": 300 + index,
        }
        for index in range(depth)
    ]


def live_shaped_decision(
    processes: int = PROCESSES, churn_rows: int = 12
) -> dict[str, Any]:
    """A triage decision shaped like the live record: verdict fields, candidate
    processes and ``evidence_branches.process_observations.processes`` with
    one stack and one wait observation per thread; ``churn_rows`` extra thread
    rows per process are what the churn thread added live."""

    observed = []
    for index in range(processes):
        node = f"ip-10-0-{32 + index % 2}-162.us-west-2.compute.internal"
        pid = 1399522 + index * 7
        threads = [
            {
                "active": False,
                "distributed_call_context": False,
                "frames": _frames(5 + thread % 3),
                "frames_truncated": False,
                "host_thread_id": pid + thread,
                "name": "MainThread" if thread == 0 else f"gfrs-helper-{thread}",
                "observed_span_seconds": 2.282074,
                "os_thread_id": 897 + thread,
                "owns_gil": thread == 0,
                "role": ROLES[thread % len(ROLES)],
                "role_source": "declared",
                "signature": hashlib.sha256(f"{pid}/{thread}".encode()).hexdigest(),
                "thread_id": pid + thread,
                "thread_key": f"{pid}:{pid + thread}:11918786",
                "thread_start_ticks": 11918786 + thread,
            }
            for thread in range(6)
        ]
        waits = [
            {
                "candidate": False,
                "code": "REPEATED_THREAD_WAIT",
                "observed_span_seconds": 2.281995,
                "qualification": "Repeated observations, not continuous blocking "
                "time or NCCL root cause",
                "role": ROLES[thread % len(ROLES)],
                "thread_id": pid + thread,
                "thread_start_ticks": 11918786 + thread,
                "wchan": "do_poll.constprop.0" if thread % 2 else "futex_wait_queue",
            }
            for thread in range(10 + churn_rows)
        ]
        observed.append(
            {
                "behavior": {
                    "cpu_cores": 0.02,
                    "gpu_mapping_verified": True,
                    "gpu_scope": "DEVICE_LEVEL",
                    "gpu_shared": False,
                    "io_windows": [
                        {
                            "elapsed_seconds": 1.14,
                            "read_bytes_delta": 0,
                            "wchar_delta": 0,
                            "write_bytes_delta": 0,
                        }
                    ],
                    "observations": waits,
                },
                "cross_evidence": {"fr_pending_groups": []},
                "fr_completion_advanced_groups": [],
                "fr_persistent_pending_groups": [],
                "node_id": node,
                "observations": [
                    {
                        k: v
                        for k, v in row.items()
                        if k not in {"thread_start_ticks", "wchan"}
                    }
                    for row in waits
                ],
                "peer_comparison": {"basis": None},
                "pid": pid,
                "rank": index,
                "root_cause_confirmed": False,
                "stack": {
                    "errors": [],
                    "missing_samples": 0,
                    "requested_samples": 1,
                    "stable_threads": threads,
                    "thread_coverage_complete": True,
                },
                "start_time_ticks": 11918700 + index,
            }
        )
    candidates = [
        {
            "confidence": "WEAK",
            "diagnostic_only": True,
            "node_id": item["node_id"],
            "pid": item["pid"],
            "rank": item["rank"],
            "reasons": ["REPEATED_THREAD_WAIT"],
            "start_time_ticks": item["start_time_ticks"],
        }
        for item in observed[:2]
    ]
    return {
        "classification": "WEAK",
        "confidence": "WEAK",
        "culprit_ranks": [],
        "control_ranks": [],
        "reason": "no communication root cause identified; inspect process "
        "observations and coverage",
        "scope": "collected_processes_only",
        "root_cause_confirmed": False,
        "targeted_capture_eligible": False,
        "candidate_processes": candidates,
        "candidate_process_count": len(candidates),
        "evidence_branches": {
            "fr_temporal": None,
            "fr_snapshot": None,
            "fr_semantics": {"status": "NO_DUMPS"},
            "snapshot_authoritative": True,
            "process_observations": {
                "processes": observed,
                "candidate_processes": candidates,
                "candidate_process_count": len(candidates),
                "process_details_truncated": False,
            },
        },
        "coverage": {
            "requested_node_count": 2,
            "responding_node_count": 2,
            "not_sampled_node_count": 0,
            "rank_inventory_complete": False,
        },
    }


def bundle_command(decision: dict[str, Any], **extra: Any):
    return command(
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
        command_id="workflow-a/5/COLLECT_DIAGNOSTIC_BUNDLE/node-a/agent-5",
        parameters={
            "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
            "job_id": "gl11-tid",
            "attempt_id": "gl11-tid-a001",
            "hung_triage_decision": decision,
            "raw_flights_by_node": {"node-a": []},
            "target_pids_by_node": {},
            "capture_process_state": False,
            "disable_strace": True,
            "bundle_scope": "bounded_triage_nodes",
            **extra,
        },
    )


class _Activity:
    def __init__(self) -> None:
        self.released = threading.Event()

    def admit(self, operation: Any, key: str, incident: str) -> str:
        return key

    def release(self, activity: str) -> None:
        self.released.set()


def _runtime(tmp_path: Path, monkeypatch) -> CommunicationRuntime:
    root = tmp_path / "proc"
    boot = root / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text("boot")
    monkeypatch.setattr(
        "gpu_fault.node_agent.communication_runtime.service_limits",
        lambda: {"verified": True, "cpu_cores": 2, "memory_bytes": 1 << 30},
    )
    # The supervisor's child processes import the gpu_fault this test imported.
    monkeypatch.setenv("PYTHONPATH", str(Path(gpu_fault.__file__).resolve().parents[1]))
    return CommunicationRuntime(
        SimpleNamespace(
            node_ids={"node-a"},
            agent_generation=1,
            proc_root=root,
            infiniband_root=tmp_path / "ib",
            diagnostic_output_dir=tmp_path / "diagnostics",
            python_stack_tool="",
            diagnostic_s3_uri=None,
            diagnostic_retention_seconds=600,
            diagnostic_max_archives=10,
            gpu_activity=_Activity(),
        )
    )


def _request_bytes(runtime: CommunicationRuntime, value: Any) -> int:
    return len(
        json.dumps(
            {
                "operation": "bundle",
                "settings": runtime.settings,
                "command": value.model_dump(mode="json"),
            }
        )
    )


def _manifest(result: dict[str, Any]) -> dict[str, Any]:
    path = Path(result["evidence_ref"].removeprefix("file://"))
    with tarfile.open(path) as archive:
        stream = archive.extractfile("diagnostics/manifest.json")
        assert stream is not None, archive.getnames()
        return json.loads(stream.read())


def test_a_decision_over_the_request_budget_is_trimmed_and_the_bundle_still_collects(
    tmp_path, monkeypatch
):
    # No diagnostic tools on PATH: every capture fails fast and is recorded,
    # the archive still forms -- what matters here is that a worker ran at all.
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    runtime = _runtime(tmp_path, monkeypatch)
    decision = live_shaped_decision()
    action = bundle_command(decision)
    assert _request_bytes(runtime, action) > OBSERVATION_REQUEST_BUDGET_BYTES, (
        "the stimulus must exceed the budget the way the live record did"
    )
    encoded = json.dumps(decision, sort_keys=True, separators=(",", ":")).encode()

    result = runtime.run_command("bundle", action)

    assert result["evidence_ref"].startswith("file://"), result
    assert result["manifest_summary"]["capture_count"] == 7, result
    assert result["manifest_summary"]["diagnostic_reason"] == (
        "EFA_TRAFFIC_HUNG_SUSPECTED"
    ), result
    stand_in = {
        "omitted": REQUEST_BUDGET_OMISSION,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "recorded_in": "workflow step parameters and COLLECT_HUNG_TRIAGE details",
    }
    assert result["observation_request_trimmed"] == {
        "hung_triage_decision": stand_in
    }, result
    context = _manifest(result)["fault_context"]
    assert context["hung_triage_decision"] == stand_in, context
    assert context["observation_request_trimmed"] == ["hung_triage_decision"], context
    assert context["raw_flights_by_node"] == {"node-a": []}, context
    assert context["diagnostic_reason"] == "EFA_TRAFFIC_HUNG_SUSPECTED", context
    assert runtime.agent.gpu_activity.released.wait(5), (
        "the worker's GPU activity is released once the bundle is done"
    )


def test_a_decision_within_the_budget_reaches_the_worker_untouched(
    tmp_path, monkeypatch
):
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    runtime = _runtime(tmp_path, monkeypatch)
    # Sixteen processes without the churn rows: the shape of the sibling
    # ``role_mismatch`` run that fit the budget and collected seven captures.
    decision = live_shaped_decision(processes=16, churn_rows=0)
    action = bundle_command(decision)
    assert _request_bytes(runtime, action) <= OBSERVATION_REQUEST_BUDGET_BYTES, (
        "a sixteen-process decision without churn rows fits the budget"
    )

    result = runtime.run_command("bundle", action)

    assert "observation_request_trimmed" not in result, result
    context = _manifest(result)["fault_context"]
    assert context["hung_triage_decision"] == decision, context
    assert "observation_request_trimmed" not in context, context


def test_a_refused_request_names_its_size_the_budget_and_the_heaviest_parameters():
    supervisor = ObservationSupervisor()
    released = threading.Event()
    request = {
        "operation": "bundle",
        "settings": {},
        "command": {
            "parameters": {
                "raw_flights_by_node": {"node-a": [{"path": "x" * 300000}]},
                "hung_triage_decision": {"classification": "WEAK"},
                "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
            }
        },
    }
    size = len(json.dumps(request))

    result = supervisor.run(request, 1, release=released.set)

    assert result["status"] == "UNAVAILABLE", result
    assert result["evidence_complete"] is False, result
    weights = {
        key: len(json.dumps(value))
        for key, value in request["command"]["parameters"].items()
    }
    assert result["error"] == (
        "ValueError: observation input exceeds budget: "
        f"{size} > {OBSERVATION_REQUEST_BUDGET_BYTES} bytes; largest parameters: "
        f"raw_flights_by_node={weights['raw_flights_by_node']} bytes, "
        f"diagnostic_reason={weights['diagnostic_reason']} bytes, "
        f"hung_triage_decision={weights['hung_triage_decision']} bytes"
    ), result
    assert result["failed_component"] == "observation_supervisor", result
    assert result["failure_stage"] == WORKER_NOT_STARTED, result
    assert result["worker_started"] is False, result
    assert released.is_set(), "a refused request releases its admission"
    raised = ObservationIncomplete(result)
    assert str(raised) == f"UNAVAILABLE: {result['error']}", str(raised)
    assert raised.action_details is result, raised.action_details


def test_parameters_the_worker_reads_are_never_trimmed(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch)
    assert CONTEXT_ONLY_PARAMETERS == ("hung_triage_decision",), CONTEXT_ONLY_PARAMETERS
    # The raw flight recorder references steer the bundle; a request they
    # alone push over the budget is refused, and the refusal says so.
    flights = [
        {"path": f"/var/lib/gpu-fault/diagnostics/nccl-flight-{index}.dump"}
        for index in range(4000)
    ]
    action = bundle_command(
        live_shaped_decision(processes=1, churn_rows=0),
        raw_flights_by_node={"node-a": flights},
    )

    with pytest.raises(ObservationIncomplete) as raised:
        runtime.run_command("bundle", action)

    details, message = raised.value.action_details, str(raised.value)
    assert message.startswith(
        "UNAVAILABLE: ValueError: observation input exceeds budget: "
    ), message
    assert "largest parameters: raw_flights_by_node=" in message, message
    assert details["failed_component"] == "observation_supervisor", details
    assert details["failure_stage"] == WORKER_NOT_STARTED, details
    assert sorted(details["observation_request_trimmed"]) == ["hung_triage_decision"], (
        details
    )
    assert runtime.agent.gpu_activity.released.is_set(), (
        "a refused request releases its admission"
    )


def test_the_node_action_result_carries_the_refusal_message_and_component(tmp_path):
    agent = node_action_executor(
        tmp_path,
        "actions.db",
        allowed_operations={Op.COLLECT_DIAGNOSTIC_BUNDLE},
        runner=FakeRunner(),
        diagnostic_output_dir=str(tmp_path / "diagnostics"),
    )
    refusal = {
        "status": "UNAVAILABLE",
        "ranks": [],
        "evidence_complete": False,
        "error": "ValueError: observation input exceeds budget: 264750 > "
        f"{OBSERVATION_REQUEST_BUDGET_BYTES} bytes; largest parameters: "
        "hung_triage_decision=259304 bytes",
        "failed_component": "observation_supervisor",
        "failure_stage": WORKER_NOT_STARTED,
        "worker_started": False,
    }

    def run_command(operation: str, value: Any) -> dict[str, Any]:
        raise ObservationIncomplete(refusal)

    agent.observation_runtime = SimpleNamespace(run_command=run_command)

    result = agent.execute(
        envelope(bundle_command(live_shaped_decision(processes=1, churn_rows=0)))
    )

    assert result.status is NodeActionStatus.FAILED, result
    assert result.error == f"ObservationIncomplete: UNAVAILABLE: {refusal['error']}", (
        result.error
    )
    assert result.details == refusal, result.details
    assert result.retryable is False, result
