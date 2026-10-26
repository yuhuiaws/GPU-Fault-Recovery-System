"""A hung triage over the worker's result line keeps its ranks.

Live (GF-GENERAL-LIVE-011 ``fr_enabled``, HyperPod, 2026-10-25 23:05:48-23:06:01Z,
release d5d7610460e0): the triage found the eight ranks of each node (the
summary before it had ``rank_inventory_complete=true`` over sixteen ranks),
wrote into each rank's ``TORCH_NCCL_DEBUG_INFO_PIPE_FILE`` twice (PyTorch logged
32 "preparing to dump debug info" lines; the sixteen 29.5 KB dumps under
``/tmp/nccl_trace_rank_*`` are dated 23:05:49-54, inside the step) and the step
still recorded ``node_failures: {<both nodes>: "ObservationIncomplete:
RESULT_SIZE_LIMIT"}`` with ``node_results: {}``. The document exceeded the
worker's 1 MiB result line and was replaced by ``{"status": "RESULT_SIZE_LIMIT",
"ranks": []}``: with the recorder off the same eight-rank triage weighed
290,643 bytes a node (plain EKS ``role_mismatch``, 20:58Z); each rank's two
dumps added ~118 KB, 64 operation summaries serialized four times. These tests
build that shape with the product's own reader, through the triage in process
and through the real supervisor and worker.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import pickle
import select
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from typing import Any

import pytest

import gpu_fault
from gpu_fault.execution.hung_classification import classify_hung_signals
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent import (
    NodeActionExecutor,
    NodeActionStatus,
    observation_worker,
)
from gpu_fault.node_agent.communication_runtime import CommunicationRuntime
from gpu_fault.node_agent.observation_limits import (
    OBSERVATION_RESULT_BUDGET_BYTES,
    WORKER_OUTPUT,
)
from gpu_fault.node_agent.observation_supervisor import ObservationIncomplete
from gpu_fault.node_agent.operations import flight_recorder, hung_triage
from gpu_fault.node_agent.operations.flight_recorder import summarize_flight_dump
from gpu_fault.node_agent.operations.flight_series import snapshot
from gpu_fault.node_agent.operations.stack_samples import stack_summary
from gpu_fault.node_agent.operations.triage_budget import (
    TRIAGE_RESULT_BUDGET_BYTES,
    bounded_triage_result,
    encoded_size,
)
from tests.node_agent._cov95_runtime_hung import COMPUTE_QUERY, process
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import command, envelope
from tests.node_agent.test_hung_collection_hardening import stat_text
from tests.regional._cov95_runtime_support import Clock

#: The live job's recorder environment (gl11-fron-a5 source manifest).
RECORDER_ENVIRONMENT = (
    "TORCH_NCCL_TRACE_BUFFER_SIZE=20000\0"
    "TORCH_NCCL_DEBUG_INFO_PIPE_FILE=/tmp/torch_nccl_debug_pipe_\0"
    "TORCH_FR_DUMP_TEMP_FILE=/tmp/nccl_trace_rank_\0"
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_trace_rank_\0"
    "TORCH_NCCL_DUMP_ON_TIMEOUT=1\0"
)
NODE_ID = "node-a"


def torch29_dump(entries: int, *, pending_from: int, world: int = 16) -> dict:
    """What PyTorch 2.9's ``_dump_nccl_trace`` pickles: one default group, every
    collective before ``pending_from`` completed, the rest scheduled."""

    rows = []
    for seq in range(1, entries + 1):
        done = seq < pending_from
        created = 1_792_969_500_000_000_000 + seq * 2_000_000_000
        rows.append(
            {
                "record_id": seq - 1,
                "pg_id": 0,
                "process_group": ("0", "default_pg"),
                "collective_seq_id": seq,
                "p2p_seq_id": 0,
                "op_id": seq,
                "profiling_name": "nccl:all_reduce",
                "time_created_ns": created,
                "input_sizes": [[393216]],
                "input_dtypes": ["Float"],
                "output_sizes": [[393216]],
                "output_dtypes": ["Float"],
                "state": "completed" if done else "scheduled",
                "time_discovered_started_ns": created + 1000 if done else None,
                "time_discovered_completed_ns": created + 900_000 if done else None,
                "retired": done,
                "timeout_ms": 600_000,
                "is_p2p": False,
                "frames": [
                    {
                        "name": f"fn{i}",
                        "filename": "/workspace/train.py",
                        "line": 100 + i,
                    }
                    for i in range(8)
                ],
            }
        )
    return {
        "version": "2.9",
        "pg_config": {
            "0": {
                "name": "0",
                "desc": "default_pg",
                "ranks": json.dumps(list(range(world))),
            }
        },
        "pg_status": {
            "0": {
                "last_enqueued_collective": entries,
                "last_started_collective": pending_from - 1,
                "last_completed_collective": pending_from - 1,
            }
        },
        "entries": rows,
    }


def pyspy_json(pid: int, threads: int = 12) -> str:
    """A py-spy 0.4.1 dump of ``threads`` threads, each five frames deep."""

    return json.dumps(
        [
            {
                "pid": pid,
                "thread_id": pid * 100 + index,
                "os_thread_id": pid + index,
                "thread_name": "MainThread" if index == 0 else f"pt_thread_{index}",
                "active": index == 0,
                "owns_gil": index == 0,
                "frames": [
                    {
                        "name": "all_reduce"
                        if depth == 0 and index == 0
                        else f"step_{depth}",
                        "filename": "/usr/local/lib/python3.12/site-packages/torch/"
                        "distributed/distributed_c10d.py"
                        if depth == 0
                        else "/workspace/train.py",
                        "line": 2000 + depth,
                    }
                    for depth in range(5)
                ],
            }
            for index in range(threads)
        ]
    )


def process_sample(pid: int, moment: float, threads: int = 24) -> dict[str, Any]:
    return {
        "state": "S",
        "cpu_ticks": 100_000 + int(moment * 100),
        "start_ticks": 1000,
        "read_bytes": 0,
        "write_bytes": 0,
        "wchar": 4096,
        "voluntary_ctxt_switches": 500 + int(moment),
        "wchan": "futex_wait_queue",
        "sample_monotonic": moment,
        "threads": {
            str(pid + index): {
                "start_ticks": 1000 + index,
                "state": "S",
                "wchan": "futex_wait_queue" if index % 3 else "do_poll.constprop.0",
                "namespace_ids": [pid + index, 7 + index],
            }
            for index in range(threads)
        },
        "scanned_thread_count": threads,
        "thread_scan_complete": True,
        "workload_scope_verified": True,
        "observed_at": f"2026-10-25T23:05:{49 + int(moment):02d}+00:00",
    }


def recorder_summary(rank: int, sample: str, *, entries: int = 64) -> dict[str, Any]:
    """What ``_finish_flight_recorder`` reports for one real 2.9 dump."""

    raw = pickle.dumps(torch29_dump(entries, pending_from=entries), protocol=2)
    summary = summarize_flight_dump(raw)
    return {
        "status": "dumped",
        "pipe": f"/proc/{3000 + rank}/root/tmp/torch_nccl_debug_pipe_{rank}.pipe",
        "dump_file": f"/proc/{3000 + rank}/root/tmp/nccl_trace_rank_{rank}",
        "dump_size_bytes": len(raw),
        "dump_source": "TORCH_FR_DUMP_TEMP_FILE",
        **summary,
        "raw_evidence": {
            "status": "PRESERVED",
            "path": f"/var/lib/gpu-fault/diagnostics/nccl-flight-a5-{rank}-{sample}.dump",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        },
    }


def live_shaped_rank(rank: int, *, recorder: bool = True) -> dict[str, Any]:
    """One rank row as the live triage returns it: three process samples with
    their windows, three bounded stack samples, the GPU series and -- with the
    recorder on -- the second dump's summary and a PAIRED two-dump series."""

    pid = 3000 + rank
    stacks = [
        {
            **stack_summary(pyspy_json(pid), pid),
            "observed_at": f"2026-10-25T23:05:{49 + index}+00:00",
            "finished_at": f"2026-10-25T23:05:{50 + index}+00:00",
            "returncode": 0,
        }
        for index in range(3)
    ]
    samples = [process_sample(pid, 1.0 + index * 1.15) for index in range(3)]
    windows = [
        {
            "status": "COLLECTED",
            "elapsed_seconds": 1.15,
            "clock_ticks_per_second": 100,
            "cpu_ticks_delta": 115,
            "wchar_delta": 0,
            "cpu_cores": 1.0,
            "wait_threads": [
                {
                    "tid": pid + index,
                    "start_ticks": 1000 + index,
                    "state_before": "S",
                    "state_after": "S",
                    "wchan": "futex_wait_queue",
                    "repeated_d": False,
                    "same_wait": True,
                    "observed_span_seconds": 1.15,
                }
                for index in range(16)
            ],
            "thread_scan_complete": True,
            "wchan_before": "futex_wait_queue",
            "wchan_after": "futex_wait_queue",
            "wchan_unchanged": True,
        }
        for _ in range(2)
    ]
    row: dict[str, Any] = {
        "node_id": NODE_ID,
        "pid": pid,
        "rank": rank,
        "gpu_uuid": f"GPU-{rank:08x}-836d-eec4-44cf-48502ab91a20",
        "gpu_uuids": [f"GPU-{rank:08x}-836d-eec4-44cf-48502ab91a20"],
        "local_rank": rank % 8,
        "process_name": "/usr/bin/python3.12",
        "process_identity": {"verified": True, "start_time_ticks": 1000},
        "comparison_context": {"verified": False, "host_thread_roles": {}},
        "process_series": {
            "version": "process-series/v1",
            "requested_samples": 3,
            "samples": samples,
            "windows": windows,
        },
        "stack_series": {
            "version": "stack-series/v1",
            "requested_samples": 3,
            "samples": stacks,
        },
        "gpu_series": {
            "scope": "DEVICE_NOT_PID",
            "mapping_verified": True,
            "shared_with_other_processes": False,
            "samples": [
                {
                    "observed_at": f"2026-10-25T23:05:{49 + index}+00:00",
                    "devices": {
                        row_gpu: {
                            "utilization_gpu_percent": 0.0,
                            "utilization_memory_percent": 0.0,
                            "clocks_throttle_reasons_active": "0x0000000000000001",
                            "device_model": "NVIDIA H200",
                        }
                        for row_gpu in [f"GPU-{rank:08x}-836d-eec4-44cf-48502ab91a20"]
                    },
                }
                for index in range(3)
            ],
        },
        "python_stack": stacks[-1],
        "proc": windows[-1],
        "gpu": {"utilization_gpu_percent": 0.0, "utilization_memory_percent": 0.0},
        "flight_recorder": {"status": "not_configured"},
    }
    if not recorder:
        return row
    first, second = recorder_summary(rank, "first"), recorder_summary(rank, "second")
    row["flight_recorder"] = second
    row["flight_recorder_series"] = {
        "version": "fr-series/v1",
        "status": "PAIRED",
        "first_observed_at": "2026-10-25T23:05:48.900000+00:00",
        "process_start_ticks": 1000,
        "clock_uncertainty_seconds": None,
        "first": snapshot(first),
        "second_requested_at": "2026-10-25T23:05:52.000000+00:00",
        "interval_seconds": 2.0,
        "second": snapshot(second),
        "second_observed_at": "2026-10-25T23:05:54.000000+00:00",
    }
    return row


def live_shaped_result(ranks: int, *, recorder: bool = True) -> dict[str, Any]:
    return {
        "triage_version": "nccl-hung-triage/v4",
        "node_id": NODE_ID,
        "ranks": [live_shaped_rank(rank, recorder=recorder) for rank in range(ranks)],
        "efa": {"counter_deltas": {}, "total_delta": 0},
        "errors": [],
        "elapsed_seconds": 12.6,
        "flight_recorder_wait_seconds": 3.1,
        "read_only": True,
        "workload_scope": {"basis": "RUNTIME_CONTAINER_ID", "complete": True},
        "process_coverage": {
            "eligible": ranks,
            "collected": ranks,
            "truncated": False,
            "rank_inventory_complete": False,
        },
    }


def semantic_entries(row: dict[str, Any], which: str) -> list[dict[str, Any]]:
    [group] = row["flight_recorder_series"][which]["groups"]
    return group["semantic_entries"]


def test_the_reader_lists_a_groups_operations_once() -> None:
    """``last_entry`` repeated the newest group's operation list: 28 KB of a
    live 29 KB dump, serialized a second time for every rank."""

    summary = recorder_summary(0, "first")
    [group] = summary["last_entries_by_pg"]
    assert len(group["semantic_entries"]) == 64, summary["entry_count"]
    assert "semantic_entries" not in summary["last_entry"], sorted(
        summary["last_entry"]
    )
    assert summary["last_entry"]["collective_seq_id"] == group["collective_seq_id"]
    assert summary["last_entry"]["progress"] == group["progress"]


def test_eight_ranks_with_two_dumps_each_fit_after_one_lossless_step() -> None:
    """The live HyperPod shape: eight ranks, 64 operations a dump, two dumps."""

    result = live_shaped_result(8)
    full = copy.deepcopy(result)
    before = encoded_size(result)
    assert before > TRIAGE_RESULT_BUDGET_BYTES, before
    without = encoded_size(live_shaped_result(8, recorder=False))
    assert without <= TRIAGE_RESULT_BUDGET_BYTES // 2, (
        f"the recorder is what pushes an eight-rank triage over the line: {without}"
    )

    bounded = bounded_triage_result(result)

    after = encoded_size(bounded)
    assert after <= TRIAGE_RESULT_BUDGET_BYTES, after
    assert bounded["evidence_trimmed"] == {
        "budget_bytes": TRIAGE_RESULT_BUDGET_BYTES,
        "bytes_before": before,
        "bytes_after": after,
        "within_budget": True,
        "steps": [
            {
                "step": "flight_recorder.semantic_entries_in_series",
                "ranks": 8,
                "bytes": bounded["evidence_trimmed"]["steps"][0]["bytes"],
            }
        ],
    }, bounded["evidence_trimmed"]
    for full_row, row in zip(full["ranks"], bounded["ranks"]):
        assert row["pid"] == full_row["pid"] and row["rank"] == full_row["rank"]
        assert row["process_identity"] == full_row["process_identity"]
        assert row["flight_recorder_series"] == full_row["flight_recorder_series"], (
            "the series -- what the control plane compares -- is untouched"
        )
        [group] = row["flight_recorder"]["last_entries_by_pg"]
        assert "semantic_entries" not in group, sorted(group)
        assert group["semantic_entries_in"] == "flight_recorder_series.second"
        assert (
            group["progress"]
            == full_row["flight_recorder"]["last_entries_by_pg"][0]["progress"]
        )
        assert (
            row["flight_recorder"]["raw_evidence"]
            == full_row["flight_recorder"]["raw_evidence"]
        )
        assert row["stack_series"] == full_row["stack_series"]
        assert row["process_series"] == full_row["process_series"]
    assert classify_hung_signals(
        bounded["ranks"], undetermined_nodes=[]
    ) == classify_hung_signals(full["ranks"], undetermined_nodes=[]), (
        "the verdict over the trimmed rows is the verdict over the full rows"
    )


def test_a_result_within_the_budget_is_returned_as_it_is() -> None:
    result = live_shaped_result(8, recorder=False)
    assert bounded_triage_result(result) is result
    assert "evidence_trimmed" not in result


def test_sixteen_ranks_shorten_the_operation_lists_before_any_process_evidence():
    result = live_shaped_result(16)
    full = copy.deepcopy(result)

    bounded = bounded_triage_result(result)

    note = bounded["evidence_trimmed"]
    assert note["within_budget"] is True, note
    assert note["bytes_after"] <= TRIAGE_RESULT_BUDGET_BYTES, note
    names = [step["step"] for step in note["steps"]]
    assert names[:2] == [
        "flight_recorder.semantic_entries_in_series",
        "semantic_entries:32",
    ], note
    assert all(name.startswith("semantic_entries:") for name in names[1:]), (
        f"only the operation lists shrink, never the process evidence: {names}"
    )
    assert all(step["ranks"] == 16 for step in note["steps"]), note
    kept_count = int(names[-1].rsplit(":", 1)[1])
    for full_row, row in zip(full["ranks"], bounded["ranks"]):
        for which in ("first", "second"):
            kept, original = (
                semantic_entries(row, which),
                semantic_entries(full_row, which),
            )
            assert kept == original[:kept_count], (
                "the head of the list -- pending first"
            )
            [group] = row["flight_recorder_series"][which]["groups"]
            assert group["semantic_entries_total"] == 64
            assert group["semantic_entries_truncated"] is True
            assert (
                group["progress"]
                == full_row["flight_recorder_series"][which]["groups"][0]["progress"]
            )
        assert row["stack_series"] == full_row["stack_series"]
        assert row["process_series"] == full_row["process_series"]
    assert classify_hung_signals(
        bounded["ranks"], undetermined_nodes=[]
    ) == classify_hung_signals(full["ranks"], undetermined_nodes=[])


def test_process_evidence_is_the_last_to_go_before_the_floor() -> None:
    """Above the skeleton floor the process evidence is cut to its last sample,
    never dropped whole; the budget here is what sixteen ranks weigh at that
    step (measured on a copy), so the ladder stops exactly there."""

    result = live_shaped_result(16)
    probe = bounded_triage_result(copy.deepcopy(result), budget=1)
    last_sample = next(
        step["bytes"]
        for step in probe["evidence_trimmed"]["steps"]
        if step["step"] == "process_series.samples:last"
    )
    budget = last_sample + 1_000

    bounded = bounded_triage_result(result, budget=budget)

    note = bounded["evidence_trimmed"]
    assert [step["step"] for step in note["steps"]] == [
        "flight_recorder.semantic_entries_in_series",
        "semantic_entries:32",
        "semantic_entries:16",
        "semantic_entries:8",
        "semantic_entries",
        "stack_series.samples:last",
        "process_series.samples:last",
    ], note
    assert note["within_budget"] is True, note
    assert note["bytes_after"] == encoded_size(bounded)
    for row in bounded["ranks"]:
        assert len(row["stack_series"]["samples"]) == 1
        assert row["stack_series"]["samples_trimmed"] == 2
        assert len(row["process_series"]["samples"]) == 1
        assert len(row["process_series"]["windows"]) == 2, "windows stay"
        assert row["python_stack"]["status"] == "COLLECTED"
        for group in row["flight_recorder_series"]["second"]["groups"]:
            assert "semantic_entries" not in group
            assert group["semantic_entries_trimmed"] is True
            assert group["semantic_entries_total"] == 64
            assert group["progress"]["last_enqueued"] == 64
        assert row["flight_recorder"]["raw_evidence"]["status"] == "PRESERVED"
        assert row["flight_recorder"]["last_entry"]["collective_seq_id"] == 64


def test_a_single_dump_series_drops_the_recorders_copy_against_its_first_snapshot():
    row = live_shaped_rank(0)
    first = recorder_summary(0, "first")
    row["flight_recorder"] = first
    row["flight_recorder_series"]["status"] = "SECOND_UNAVAILABLE"
    row["flight_recorder_series"]["first"] = snapshot(first)
    row["flight_recorder_series"]["second"] = {"status": "dump_missing"}
    result = {"ranks": [row]}
    # One byte over: the lossless step alone brings the document under.
    budget = encoded_size(result) - 1

    bounded_triage_result(result, budget=budget)

    [group] = row["flight_recorder"]["last_entries_by_pg"]
    assert group["semantic_entries_in"] == "flight_recorder_series.first"
    assert "semantic_entries" not in group
    assert [step["step"] for step in result["evidence_trimmed"]["steps"]] == [
        "flight_recorder.semantic_entries_in_series"
    ]


def published_line(document: dict[str, Any], *, final: bool = True) -> dict[str, Any]:
    stream = io.BytesIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "stdout", SimpleNamespace(buffer=stream, flush=lambda: None))
        observation_worker.publish(document, final=final)
    line = stream.getvalue()
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert len(line) <= OBSERVATION_RESULT_BUDGET_BYTES, len(line)
    return json.loads(line)


def test_the_worker_publishes_a_bounded_triage_whole() -> None:
    document = bounded_triage_result(live_shaped_result(8))

    message = published_line(document)

    assert message == {"final": True, "result": document}


def test_the_worker_names_a_result_it_cannot_send() -> None:
    document = {
        **live_shaped_result(8, recorder=False),
        "ranks": [live_shaped_rank(rank, recorder=False) for rank in range(32)],
    }
    size = encoded_size({"final": True, "result": document}) + 1
    assert size > OBSERVATION_RESULT_BUDGET_BYTES, size

    message = published_line(document)

    result = message["result"]
    assert result["status"] == "RESULT_SIZE_LIMIT", result
    assert result["ranks"] == [] and result["evidence_complete"] is False, result
    heaviest = sorted(
        ((encoded_size(value), key) for key, value in document.items()), reverse=True
    )[:3]
    assert [key for _, key in heaviest] == [
        "ranks",
        "process_coverage",
        "workload_scope",
    ]
    assert result["error"] == (
        "observation result exceeds budget: "
        f"{size} > {OBSERVATION_RESULT_BUDGET_BYTES} bytes; largest fields: "
        + ", ".join(f"{key}={weight} bytes" for weight, key in heaviest)
    ), result["error"]
    assert result["failed_component"] == "observation_worker", result
    assert result["failure_stage"] == WORKER_OUTPUT, result
    assert result["worker_started"] is True and result["result_bytes"] == size, result
    raised = ObservationIncomplete(result)
    assert str(raised) == f"RESULT_SIZE_LIMIT: {result['error']}", str(raised)


def test_a_line_exactly_at_the_budget_is_still_too_long_for_the_supervisor() -> None:
    """The supervisor counts the newline it buffers; the worker counts it too."""

    document = {"status": "COLLECTED", "ranks": [], "pad": ""}
    overhead = encoded_size({"final": True, "result": document}) + 1
    document["pad"] = "x" * (OBSERVATION_RESULT_BUDGET_BYTES - overhead)
    assert published_line(document)["result"] == document
    document["pad"] += "x"
    assert published_line(document)["result"]["status"] == "RESULT_SIZE_LIMIT"


# --------------------------------------------------------------------------- #
# through the triage operation
# --------------------------------------------------------------------------- #
class FakeRecorderNode:
    """Sixteen rank processes of one training container, each with the live
    job's recorder environment, a debug FIFO and a dump that appears when the
    Agent writes into the FIFO (PyTorch's monitor thread, in effect)."""

    def __init__(self, root: Path, *, ranks: int, entries: int = 64) -> None:
        self.root = root
        self.ranks = ranks
        self.entries = entries
        self.pipes: dict[Path, int] = {}
        self.writes: list[int] = []
        self.mtime = 1_792_969_549_000_000_000
        for rank in range(ranks):
            pid = self.pid(rank)
            directory = process(root, pid)
            (directory / "stat").write_text(stat_text(pid, 1000))
            (directory / "environ").write_bytes(
                f"RANK={rank}\0LOCAL_RANK={rank % 8}\0WORLD_SIZE=16\0{RECORDER_ENVIRONMENT}".encode()
            )
            namespace = directory / "root" / "tmp"
            namespace.mkdir(parents=True)
            self.pipes[namespace / f"torch_nccl_debug_pipe_{rank}.pipe"] = rank

    @staticmethod
    def pid(rank: int) -> int:
        return 3000 + rank

    def compute_rows(self) -> str:
        return "".join(
            f"{self.pid(rank)}, GPU-{rank:08x}-836d-eec4-44cf-48502ab91a20, python\n"
            for rank in range(self.ranks)
        )

    def dump(self, pipe: Path) -> None:
        rank = self.pipes[pipe]
        self.writes.append(rank)
        target = pipe.parent / f"nccl_trace_rank_{rank}"
        target.write_bytes(
            pickle.dumps(
                torch29_dump(self.entries, pending_from=self.entries), protocol=2
            )
        )
        self.mtime += 1_000_000
        os.utime(target, ns=(self.mtime, self.mtime))


def test_sixteen_paired_ranks_come_back_within_the_budget_through_the_triage(
    node_factory: Callable[..., NodeActionExecutor], tmp_path: Path, monkeypatch
) -> None:
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
        assert data == b"1\n", data
        node.dump(descriptors[descriptor])
        return len(data)

    def runner(argv: list[str], **kwargs: Any) -> CompletedProcess:
        text = node.compute_rows() if COMPUTE_QUERY in argv else ""
        return CompletedProcess(argv, 0, stdout=text, stderr="")

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
        result = agent.execute(
            envelope(
                command(
                    Op.COLLECT_HUNG_TRIAGE,
                    # No GPU filter: the live command names the node's GPUs
                    # under gpu_uuids_by_node, not on the command itself.
                    gpu_uuids=[],
                    parameters={
                        "require_workload_identity": True,
                        "workload_cgroup_paths_by_node": {NODE_ID: ["/training/job"]},
                        "triage_timeout_seconds": 20,
                        "triage_fr_sample_interval_seconds": 2,
                    },
                )
            )
        )

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    details = result.details
    assert sorted(node.writes) == sorted([*range(16), *range(16)]), (
        "two dump requests per rank"
    )
    assert [row["rank"] for row in details["ranks"]] == list(range(16)), details[
        "process_coverage"
    ]
    assert {row["flight_recorder_series"]["status"] for row in details["ranks"]} == {
        "PAIRED"
    }
    assert encoded_size(details) <= TRIAGE_RESULT_BUDGET_BYTES, encoded_size(details)
    note = details["evidence_trimmed"]
    assert note["within_budget"] is True, note
    assert note["bytes_before"] > TRIAGE_RESULT_BUDGET_BYTES, note
    assert note["steps"][0]["step"] == "flight_recorder.semantic_entries_in_series"
    for row in details["ranks"]:
        assert row["flight_recorder"]["status"] == "dumped", row["flight_recorder"]
        assert row["flight_recorder"]["raw_evidence"]["status"] == "PRESERVED"
        [group] = row["flight_recorder"]["last_entries_by_pg"]
        assert group["semantic_entries_in"] == "flight_recorder_series.second"
        assert group["progress"]["last_enqueued"] == 64


# --------------------------------------------------------------------------- #
# through the real supervisor and worker
# --------------------------------------------------------------------------- #
class _Activity:
    def __init__(self) -> None:
        self.released = threading.Event()

    def admit(self, operation: Any, key: str, incident: str) -> str:
        return key

    def release(self, activity: str) -> None:
        self.released.set()


class DumpResponder(threading.Thread):
    """Answers the Agent's FIFO writes from this process while the worker runs
    in its own: PyTorch's monitor thread, writing the dump after the request."""

    def __init__(self, node: FakeRecorderNode) -> None:
        super().__init__(name="fr-dump-responder", daemon=True)
        self.node = node
        self.stop = threading.Event()
        # O_RDWR keeps a writer on every FIFO, so a closed Agent write end is
        # not an endless EOF to select on; the Agent's own O_WRONLY|O_NONBLOCK
        # open still succeeds because a reader exists.
        self.descriptors = {
            os.open(pipe, os.O_RDWR | os.O_NONBLOCK): pipe for pipe in node.pipes
        }

    def run(self) -> None:
        while not self.stop.is_set():
            ready, _, _ = select.select(list(self.descriptors), [], [], 0.05)
            for descriptor in ready:
                try:
                    data = os.read(descriptor, 64)
                except BlockingIOError:
                    continue
                if b"1" in data:
                    self.node.dump(self.descriptors[descriptor])

    def close(self) -> None:
        self.stop.set()
        self.join(timeout=5)
        for descriptor in self.descriptors:
            os.close(descriptor)


def fake_nvidia_smi(directory: Path, rows: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "nvidia-smi"
    script.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f"  *query-compute-apps*) printf '%s' '{rows}' ;;\n"
        "  *query-gpu*) printf 'GPU-00000000-836d-eec4-44cf-48502ab91a20, 0, 0, 0x0, H200\\n' ;;\n"
        "esac\n"
    )
    script.chmod(0o755)


def test_the_real_worker_returns_sixteen_paired_ranks_instead_of_result_size_limit(
    tmp_path: Path, monkeypatch
) -> None:
    node = FakeRecorderNode(tmp_path / "proc", ranks=16)
    for pipe in node.pipes:
        os.mkfifo(pipe)
    fake_nvidia_smi(tmp_path / "bin", node.compute_rows())
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv("PYTHONPATH", str(Path(gpu_fault.__file__).resolve().parents[1]))
    monkeypatch.setattr(
        "gpu_fault.node_agent.communication_runtime.service_limits",
        lambda: {"verified": True, "cpu_cores": 2, "memory_bytes": 1 << 30},
    )
    boot = tmp_path / "proc" / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text("boot")
    runtime = CommunicationRuntime(
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
    responder = DumpResponder(node)
    responder.start()
    try:
        result = runtime.run_command(
            "triage",
            command(
                Op.COLLECT_HUNG_TRIAGE,
                command_id="workflow-a/3/COLLECT_HUNG_TRIAGE/node-a/agent-7",
                gpu_uuids=[],
                parameters={
                    "job_id": "gl11-fron",
                    "attempt_id": "gl11-fron-a001",
                    "require_workload_identity": True,
                    "workload_cgroup_paths_by_node": {NODE_ID: ["/training/job"]},
                    "triage_timeout_seconds": 20,
                    "triage_fr_sample_interval_seconds": 1,
                },
            ),
        )
    finally:
        responder.close()

    assert result.get("status") != "RESULT_SIZE_LIMIT", result.get("error")
    assert [row["rank"] for row in result["ranks"]] == list(range(16)), result.get(
        "errors"
    )
    assert {row["flight_recorder_series"]["status"] for row in result["ranks"]} == {
        "PAIRED"
    }, [row["flight_recorder_series"].get("status") for row in result["ranks"]]
    assert sorted(node.writes) == sorted([*range(16), *range(16)])
    assert encoded_size(result) <= TRIAGE_RESULT_BUDGET_BYTES, encoded_size(result)
    assert result["evidence_trimmed"]["within_budget"] is True, result[
        "evidence_trimmed"
    ]
    assert result["evidence_trimmed"]["bytes_before"] > TRIAGE_RESULT_BUDGET_BYTES
    assert runtime.agent.gpu_activity.released.wait(5), (
        "the worker's GPU activity is released once the triage is done"
    )
