"""The workflow dispatcher's side of the store wakeup channels (性能 A).

``WorkflowDispatcher.wake()`` is process-local: an ingress route calling it in
the split deployment shortens nothing, because the worker process is the one
that scans. Cross-process wakeups arrive instead through two listener threads
(``run_wakeups``, one per ``WakeupChannel``), which the store feeds from its
NOTIFYs and which call ``wake()`` in the worker process. Bursts coalesce: the
wake event is either set or not, so any number of wakeups between two scans
buy exactly one extra scan, and a wake that lands during a scan buys exactly
one more (``run_forever`` clears before scanning, F-A8). The poll interval
stays the fallback for a wakeup that was never delivered.

The functions here take the dispatcher itself: the counters they keep live on
it, next to the other per-process series ``/metrics`` reads, and the tests and
``lifespan_workers`` address them there (``dispatcher.wakeups_total``,
``dispatcher.run_wakeups``). Each counter key is written by exactly one
listener thread, so no lock.
"""

from __future__ import annotations

import logging
import time
from threading import Event
from typing import TYPE_CHECKING, TypeVar

from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store.contracts import WakeupChannel

if TYPE_CHECKING:
    from gpu_fault.execution.dispatcher import WorkflowDispatcher

# A child of the dispatcher's own logger: the listener failure below is a
# dispatcher event to whoever filters on ``gpu_fault.execution.dispatcher``.
LOGGER = logging.getLogger("gpu_fault.execution.dispatcher.wakeups")
_T = TypeVar("_T")

# Remote-command transitions that let a WAITING step take its next step.
# PENDING is the row the dispatcher's own dispatch wrote, LEASED and WAITING
# are the executor claiming and running it; a scan on any of them would find
# the same WAITING step still waiting.
WAKING_REMOTE_COMMAND_STATUSES: frozenset[str] = frozenset(
    {RemoteCommandStatus.SUCCEEDED.value, RemoteCommandStatus.FAILED.value}
)


def wakeup_key(channel: WakeupChannel) -> str:
    """Counter and label key for a channel: ``workflow_dispatch`` /
    ``remote_command`` rather than the ``gpu_fault_``-prefixed NOTIFY name."""

    return WakeupChannel(channel).name.lower()


def per_channel(value: _T) -> dict[str, _T]:
    """Initial value of a per-channel counter or gauge, keyed by ``wakeup_key``."""

    return {wakeup_key(channel): value for channel in WakeupChannel}


def notify_wakeup(
    dispatcher: WorkflowDispatcher, channel: WakeupChannel, payload: dict[str, object]
) -> bool:
    """Translate one store wakeup into ``dispatcher.wake()``; True when it woke.

    A workflow wakeup always wakes -- including one whose ``not_before`` is in
    the future, because ``_eligible`` already holds such a row and the poll
    dispatches it later; deciding here would duplicate that rule. A
    remote-command wakeup wakes only on ``WAKING_REMOTE_COMMAND_STATUSES``;
    the rest are the executor's side of that channel and are counted as
    ignored.
    """
    if (
        channel is WakeupChannel.REMOTE_COMMAND
        and payload.get("status") not in WAKING_REMOTE_COMMAND_STATUSES
    ):
        dispatcher.wakeups_ignored_total += 1
        return False
    key = wakeup_key(channel)
    dispatcher.wakeups_total[key] = dispatcher.wakeups_total.get(key, 0) + 1
    dispatcher.wakeup_last_seen_timestamp_seconds = time.time()
    dispatcher.wake()
    return True


def run_wakeups(
    dispatcher: WorkflowDispatcher, channel: WakeupChannel, stop: Event | None
) -> None:
    """Listener thread body: forward ``channel`` wakeups into ``wake()`` until
    ``stop`` (the lifespan's shared stop event; the dispatcher's own when None).

    Same shape as the processor's ``run_queue_notifications``: the store
    listener reconnects on its own and reports its state through ``on_state``;
    an exception that escapes it is logged, never raised, because a dead
    listener costs one poll interval per step, not correctness -- the poll is
    still turning. A store without the listener (test doubles) simply gets no
    wakeups.
    """
    listener = getattr(dispatcher.store, "run_wakeup_listener", None)
    if listener is None or not dispatcher.config.enabled:
        return
    try:
        listener(
            channel,
            stop if stop is not None else dispatcher._stop,
            lambda payload: notify_wakeup(dispatcher, channel, payload),
            on_state=lambda connected: dispatcher._set_wakeup_listener_state(
                channel, connected
            ),
        )
    except Exception:
        dispatcher._set_wakeup_listener_state(channel, False)
        LOGGER.exception(
            "workflow dispatch wakeup listener failed channel=%s; polling only",
            WakeupChannel(channel).value,
        )


def set_listener_state(
    dispatcher: WorkflowDispatcher, channel: WakeupChannel, connected: bool
) -> None:
    """Record whether ``channel``'s LISTEN thread is connected; False means
    the dispatcher is back to polling only for that channel. Logs transitions."""

    key = wakeup_key(channel)
    if dispatcher.wakeup_listener_connected.get(key) != connected:
        LOGGER.info(
            "workflow dispatch wakeup listener %s channel=%s",
            "connected" if connected else "disconnected",
            WakeupChannel(channel).value,
        )
    dispatcher.wakeup_listener_connected[key] = connected
