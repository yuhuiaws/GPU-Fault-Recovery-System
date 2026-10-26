"""A stand-in bundle worker that checkpoints its captures and then never finishes.

Spawned by the real ``ObservationSupervisor`` in place of ``observation_guard``;
publishes the bundle's partial manifest through the product's own ``publish``
(captures done, captures pending, elapsed) and then stalls past any supervisor
deadline, the way a bundle on an overloaded shared node would.
"""

from __future__ import annotations

import sys
import time

from gpu_fault.node_agent.observation_worker import publish


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
    sys.stdin.buffer.read()
    publish({"status": "COLLECTING", "ranks": [], "read_only": True}, final=False)
    publish(partial(["nvidia-smi.txt"], ["dmesg.txt", "ps.txt"]), final=False)
    publish(partial(["nvidia-smi.txt", "dmesg.txt"], ["ps.txt"]), final=False)
    time.sleep(60)


if __name__ == "__main__":
    main()
