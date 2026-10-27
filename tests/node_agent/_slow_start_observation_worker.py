"""The real read-only worker behind a slow start, recording the triage it plans.

Spawned by the real ``ObservationSupervisor`` in place of ``observation_guard``
(``-m tests.node_agent._slow_start_observation_worker``). It first spends
``START_SECONDS`` the way a guard and a worker spend their start on a node whose
CPU a training job saturates (an interpreter each, the package's imports: 3.3 s
from spawn to the worker's first line with eight busy loops on its CPU), then
runs the real ``observation_worker.main`` -- the request off stdin, the deadline
off the environment, the plan -- with the triage itself replaced by a record
of what was planned: the budget, when, and the deadline it was planned against.
"""

from __future__ import annotations

import os
import time
from typing import Any

#: The slow start, on top of this process's own.
START_SECONDS = 3.0


def main() -> None:
    time.sleep(START_SECONDS)
    from gpu_fault.node_agent import observation_worker

    def planned(_agent: Any, command: Any) -> dict[str, Any]:
        return {
            "status": "PLANNED",
            "planned_seconds": command.parameters["triage_timeout_seconds"],
            "planned_at": time.monotonic(),
            "deadline": observation_worker.handed_deadline(os.environ),
            "evidence_complete": True,
        }

    observation_worker.ObservationAgent._collect_hung_triage = planned  # type: ignore[method-assign]
    observation_worker.main()


if __name__ == "__main__":
    main()
