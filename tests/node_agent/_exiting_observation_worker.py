"""A worker tree that has answered and is on its way out.

Spawned by the real ``ObservationSupervisor`` in place of ``observation_guard``
(``-m tests.node_agent._exiting_observation_worker``): it publishes a final
result through the product's own ``publish``, closes its end of the pipe -- the
supervisor reads the end of the output -- and exits ``exit_after`` seconds
later, or never without it. That is the window between the guard closing its
stdout and being reaped, which a CPU-starved host stretches past 0.1 s. With
``checkpointed`` in the request, that file appears once the supervisor has read
the result, before the pipe is closed.
"""

from __future__ import annotations

import json
import os
import sys
import time

from gpu_fault.node_agent.observation_worker import publish
from tests.node_agent._stalled_observation_worker import announce_checkpointed

#: What the worker answers.
RESULT = {"status": "COMPLETE", "ranks": [], "read_only": True}


def main() -> None:
    request = json.loads(sys.stdin.buffer.read() or b"{}")
    publish(RESULT, final=True)
    announce_checkpointed(request)
    # The pipe's last write end goes; fd 1 stays valid for the exit's flush.
    os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    time.sleep(float(request.get("exit_after") or 3600))


if __name__ == "__main__":
    main()
