"""A stand-in observation worker that checkpoints and then never finishes.

Spawned by the real ``ObservationSupervisor`` in place of ``observation_guard``
(``-m tests.node_agent._stalled_observation_worker``): it publishes the triage's
partial documents through the product's own ``publish`` and then stalls for
longer than any supervisor deadline, the way a worker stuck past its budget
would. What the supervisor hands back at its deadline is then the last
checkpoint, not a bare status.

Once the supervisor has read every byte of its checkpoints it creates the file
the request names under ``checkpointed``: the test's clock passes the
supervisor's deadline only then (``_observation_deadline.DeadlineClock``), so
the deadline falls after the checkpoint however slowly this process started.
"""

from __future__ import annotations

import array
import fcntl
import json
import sys
import termios
import time
from pathlib import Path
from typing import Any

from gpu_fault.node_agent.observation_worker import publish


def announce_checkpointed(request: dict[str, Any], *, patience: float = 120) -> None:
    """Create ``request["checkpointed"]`` once the supervisor has read stdout dry.

    ``FIONREAD`` on a pipe's write end counts the bytes still in the pipe; at 0
    the supervisor has read every checkpoint line, and it parses what it read
    before it looks at its clock again.
    """

    path = request.get("checkpointed")
    if not path:
        return
    unread = array.array("i", [0])
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline:
        fcntl.ioctl(sys.stdout.fileno(), termios.FIONREAD, unread, True)
        if unread[0] == 0:
            Path(path).touch()
            return
        time.sleep(0.01)


def main() -> None:
    request = json.loads(sys.stdin.buffer.read() or b"{}")
    publish({"status": "COLLECTING", "ranks": [], "read_only": True}, final=False)
    ranks = [{"pid": 3000 + rank, "rank": rank} for rank in range(2)]
    publish(
        {
            "status": "COLLECTING",
            "collection_stage": "RANKS_DISCOVERED",
            "node_id": "node-a",
            "ranks": ranks,
            "elapsed_seconds": 0.2,
            "triage_budget_seconds": 20,
            "evidence_complete": False,
        },
        final=False,
    )
    publish(
        {
            "status": "COLLECTING",
            "collection_stage": "FIRST_FLIGHT_DUMPS_READ",
            "node_id": "node-a",
            "ranks": [
                {**row, "flight_recorder": {"status": "dumped"}} for row in ranks
            ],
            "elapsed_seconds": 0.6,
            "triage_budget_seconds": 20,
            "evidence_complete": False,
        },
        final=False,
    )
    announce_checkpointed(request)
    time.sleep(60)


if __name__ == "__main__":
    main()
