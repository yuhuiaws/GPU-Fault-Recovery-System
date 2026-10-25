"""A hung triage decision shaped like the live GF-GENERAL-LIVE-011 record.

Sixteen processes, one stack and one wait observation per thread, roughly
15 KB a process: the ``tid_namespace`` run's decision weighed 259 KB and pushed
the node's observation request over its 256 KiB budget; the ``role_mismatch``
run's (no churn rows) weighed 223 KB and fit.
"""

from __future__ import annotations

import hashlib
from typing import Any

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
