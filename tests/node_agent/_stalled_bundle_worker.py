"""A stand-in bundle worker that checkpoints its captures and then never finishes.

Spawned by the real ``ObservationSupervisor`` in place of ``observation_guard``;
publishes the bundle's partial manifest through the product's own ``publish``
(captures done, captures pending, elapsed) and then stalls past any supervisor
deadline, the way a bundle on an overloaded shared node would; like the triage
stand-in it announces when the supervisor has read its checkpoints.
"""

from __future__ import annotations

import json
import sys
import time

from gpu_fault.node_agent.observation_worker import publish
from tests.node_agent._stalled_observation_worker import announce_checkpointed


def partial(done: list[str], pending: list[str]) -> dict:
    return {
        "status": "COLLECTING",
        "collection_stage": "BUNDLE_CAPTURING",
        "manifest": {
            "node_id": "node-a",
            "captures": [{"file": name} for name in done],
        },
        "captures_done": done,
        "captures_pending": pending,
        "elapsed_seconds": 0.3 * len(done),
        "evidence_complete": False,
    }


def main() -> None:
    request = json.loads(sys.stdin.buffer.read() or b"{}")
    publish({"status": "COLLECTING", "ranks": [], "read_only": True}, final=False)
    publish(partial(["nvidia-smi.txt"], ["dmesg.txt", "ps.txt"]), final=False)
    publish(partial(["nvidia-smi.txt", "dmesg.txt"], ["ps.txt"]), final=False)
    announce_checkpointed(request)
    time.sleep(60)


if __name__ == "__main__":
    main()
