"""The bare worker's dispatcher threads: ``run_forever`` and its wakeup listeners.

The processor shape gets its dispatch loop and listeners inside
``start_processor_threads``; this is the same set for the worker that has no
processor, started and joined as one unit by the lifespan.
"""

from __future__ import annotations

from threading import Event, Thread
from typing import Any

from gpu_fault.app.lifespan_workers import start_dispatcher_wakeup_threads


def start_dispatcher_threads(context: Any, stop: Event) -> list[Thread]:
    """Start the scanner thread and its wakeup listener threads.

    Returned in join order: the scanner first, so ``dispatcher.stop()`` is
    observed before the listeners (already released by ``stop``) are joined.
    """

    worker = Thread(
        target=context.dispatcher.run_forever,
        name="gpu-fault-workflow-dispatcher",
        daemon=True,
    )
    worker.start()
    return [worker, *start_dispatcher_wakeup_threads(context, stop)]
