"""A stand-in observation worker that checkpoints and then never finishes.

Spawned by the real ``ObservationSupervisor`` in place of ``observation_guard``
(``-m tests.node_agent._stalled_observation_worker``): it publishes the triage's
partial documents through the product's own ``publish`` and then stalls for
longer than any supervisor deadline, the way a worker stuck past its budget
would. What the supervisor hands back at its deadline is then the last
checkpoint, not a bare status.
"""

from __future__ import annotations

import sys
import time

from gpu_fault.node_agent.observation_worker import publish


def main() -> None:
    sys.stdin.buffer.read()
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
    time.sleep(60)


if __name__ == "__main__":
    main()
