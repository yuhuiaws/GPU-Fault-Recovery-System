"""Verify the inherited service cgroup envelope instead of claiming RLIMIT_RSS."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def service_limits(
    proc_root: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")
) -> dict[str, Any]:
    try:
        lines = (proc_root / "self/cgroup").read_text().splitlines()
        relative = next(line[3:] for line in lines if line.startswith("0::"))
        if relative == "/" or ".." in relative.split("/"):
            raise ValueError("no delegated service cgroup")
        directory = cgroup_root / relative.lstrip("/")
        quota, period = (directory / "cpu.max").read_text().split()
        memory = int((directory / "memory.max").read_text())
        cores = int(quota) / int(period)
        return {
            "verified": 0 < cores <= 2 and 0 < memory <= 1024 * 1024 * 1024,
            "cpu_cores": cores,
            "memory_bytes": memory,
            "scope": "AGENT_AND_ALL_DESCENDANTS",
        }
    except (OSError, ValueError, StopIteration, ZeroDivisionError):
        return {"verified": False, "reason": "BOUNDED_CGROUP_V2_ENVELOPE_UNAVAILABLE"}


#: What one observation request may weigh on the read-only worker's stdin. The
#: supervisor refuses a larger request before it starts the worker, and the
#: worker refuses it again on its own side of the pipe.
OBSERVATION_REQUEST_BUDGET_BYTES = 262144
#: What one result line may weigh on the worker's stdout, newline included: the
#: supervisor drops the connection on a longer line, so the worker never sends
#: one -- a document over it goes out as a ``RESULT_SIZE_LIMIT`` stand-in that
#: names the size (``observation_worker.oversized_result``). The hung triage
#: trims its own evidence under this line first
#: (``operations.triage_budget.bounded_triage_result``), down to a skeleton of
#: every rank's identity and verdict, so only a document whose skeleton alone
#: is over the line reaches the stand-in.
OBSERVATION_RESULT_BUDGET_BYTES = 1_048_576
#: What the stand-in spends on naming the ranks of the result it replaces
#: (``ranks_summary``: pid, rank and status per rank, in the result's own
#: order, as many as fit). The operator's rule for the last resort: never drop
#: everything and just report an error.
RESULT_STAND_IN_RANKS_BUDGET_BYTES = 16 * 1024
#: ``failure_stage`` of a result the supervisor could not obtain: the worker was
#: never started (the request itself was refused, or the process could not be
#: spawned), or it ran and its output could not be used. Defined here, where
#: both sides of the pipe import from, so the worker can stamp its own
#: oversized-output stand-in without importing the supervisor.
WORKER_NOT_STARTED = "WORKER_NOT_STARTED"
WORKER_OUTPUT = "WORKER_OUTPUT"
#: The environment variable through which the supervisor tells the worker how
#: many seconds of its deadline are left once the worker starts -- the wait for
#: the node's single worker slot is inside that deadline. The worker plans a
#: triage within it less ``WORKER_BUDGET_MARGIN_SECONDS``, the time it takes to
#: trim and write a result of the live size, so a command that waited for the
#: slot finishes what fits and says so instead of being cut at its deadline.
OBSERVATION_BUDGET_VARIABLE = "OBSERVATION_BUDGET_SECONDS"
WORKER_BUDGET_MARGIN_SECONDS = 2.5
