"""In-process stand-in for the two wakeup NOTIFY channels.

On PostgreSQL a row trigger on ``gpu_fault_objects`` (``ddl_wakeups.py``)
publishes ``gpu_fault_workflow_dispatch`` and ``gpu_fault_remote_command``. The
memory and SQLite stores have no trigger, so their write paths publish into a
:class:`WakeupHub` under the condition the trigger evaluates -- the rules live
here, in one place, so the trigger and the hub cannot drift apart (the
``test_wakeup_listener`` module pins the SQL literals to these constants).

Wakeups are hints, not state. A subscriber that falls behind loses the oldest
entries, a payload published with nobody listening is gone, and PostgreSQL
drops NOTIFY across a reconnect; every consumer therefore keeps its polling
fallback and treats a payload as "scan now".
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Callable

from gpu_fault.models import EXECUTABLE_WORKFLOW_STATUSES
from gpu_fault.store.contracts import WakeupChannel

# Workflow fields whose change re-fires the dispatch wakeup on an update of an
# executable row. The executor writes a WAITING row back on every dispatch
# (D-7) and renews its lease every few seconds; each is an UPDATE of a RUNNING
# row, and waking the dispatcher on those would re-dispatch the row it had just
# written, in a loop, until the remote command completed. Only what the scan
# orders or filters on re-fires: a status change into the executable set, a
# moved ``not_before``, a merge that appended steps (``merge_revision``), a
# lease handover (``execution_owner_id``) or a new generation
# (``fencing_token``).
WORKFLOW_WAKEUP_FIELDS: tuple[str, ...] = (
    "not_before",
    "merge_revision",
    "execution_owner_id",
    "fencing_token",
)
# ``cluster_id`` is published for symmetry with the remote-command payload and
# with the trigger's ``NEW.payload->>'cluster_id'``; ``WorkflowRequest`` does not
# carry the field today, so it is null on every backend until it does. A
# consumer that wants the cluster reads the incident.
WORKFLOW_WAKEUP_PAYLOAD_FIELDS: tuple[str, ...] = (
    "request_id",
    "cluster_id",
    "status",
    "not_before",
)
REMOTE_COMMAND_WAKEUP_PAYLOAD_FIELDS: tuple[str, ...] = (
    "command_id",
    "cluster_id",
    "workflow_request_id",
    "status",
)

_EXECUTABLE_STATUS_VALUES = frozenset(
    status.value for status in EXECUTABLE_WORKFLOW_STATUSES
)

# How many undelivered payloads one subscriber may hold before the oldest is
# dropped. The bound exists so a wedged consumer cannot grow memory without
# limit; losing a hint costs one poll interval, which is the pre-wakeup
# behaviour.
SUBSCRIBER_BOUND = 4096


def workflow_wakeup_fields(workflow: Any) -> dict[str, Any]:
    """The JSON-mode values the wakeup rule compares and publishes.

    JSON mode so a previous row read back from SQLite (``json_extract``) and a
    previous model held by the memory store compare equal to the same text the
    PostgreSQL trigger compares (``payload->>'field'``).
    """

    return dict(
        workflow.model_dump(
            mode="json",
            include={*WORKFLOW_WAKEUP_FIELDS, *WORKFLOW_WAKEUP_PAYLOAD_FIELDS},
        )
    )


def workflow_wakeup(
    previous: Mapping[str, Any] | None, workflow: Any
) -> dict[str, Any] | None:
    """The ``WORKFLOW_DISPATCH`` payload a write of ``workflow`` publishes, or
    None when the trigger would stay quiet.

    ``previous`` is the stored row's :func:`workflow_wakeup_fields` before the
    write (None for an insert).
    """

    fields = workflow_wakeup_fields(workflow)
    if fields["status"] not in _EXECUTABLE_STATUS_VALUES:
        return None
    if previous is not None and all(
        previous.get(name) == fields[name]
        for name in (*WORKFLOW_WAKEUP_FIELDS, "status")
    ):
        return None
    return {name: fields.get(name) for name in WORKFLOW_WAKEUP_PAYLOAD_FIELDS}


def remote_command_wakeup(
    previous_status: str | None, command: Any
) -> dict[str, Any] | None:
    """The ``REMOTE_COMMAND`` payload a write of ``command`` publishes, or None
    when its status did not change (a lease renewal, a cancellation request,
    a result detail)."""

    fields = command.model_dump(
        mode="json", include=set(REMOTE_COMMAND_WAKEUP_PAYLOAD_FIELDS)
    )
    if previous_status is not None and previous_status == fields["status"]:
        return None
    return {name: fields.get(name) for name in REMOTE_COMMAND_WAKEUP_PAYLOAD_FIELDS}


class _Subscription:
    def __init__(self) -> None:
        self._pending: deque[dict[str, Any]] = deque(maxlen=SUBSCRIBER_BOUND)
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def push(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._pending.append(payload)
        self._ready.set()

    def wait(self, timeout: float) -> None:
        self._ready.wait(timeout)

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            self._ready.clear()
            drained = list(self._pending)
            self._pending.clear()
        return drained


class WakeupHub:
    """Per-channel fan-out to every subscribed listener in this process.

    Every listener receives every payload -- there is no shard lock, unlike
    the processor queue listener, because every dispatcher replica and every
    executor wants every wakeup and the claim that follows is what serialises
    them.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[WakeupChannel, list[_Subscription]] = {
            channel: [] for channel in WakeupChannel
        }

    def publish(self, channel: WakeupChannel, payload: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers[WakeupChannel(channel)])
        for subscription in subscribers:
            subscription.push(payload)

    @contextmanager
    def subscribe(self, channel: WakeupChannel) -> Iterator[_Subscription]:
        subscription = _Subscription()
        key = WakeupChannel(channel)
        with self._lock:
            self._subscribers[key].append(subscription)
        try:
            yield subscription
        finally:
            with self._lock:
                self._subscribers[key].remove(subscription)


class ObservedRows(dict[str, Any]):
    """A dict whose ``__setitem__`` is the memory store's row trigger.

    The memory store writes workflows and remote commands as plain dict
    assignments from some twenty methods; hooking the table rather than each
    method is what guarantees the hub sees every path the PostgreSQL trigger
    would. ``rule(previous, value)`` returns the payload to publish or None.
    """

    def __init__(
        self,
        hub: WakeupHub,
        channel: WakeupChannel,
        rule: Callable[[Any, Any], dict[str, Any] | None],
    ) -> None:
        super().__init__()
        self._hub = hub
        self._channel = channel
        self._rule = rule

    def __setitem__(self, key: str, value: Any) -> None:
        previous = dict.get(self, key)
        super().__setitem__(key, value)
        payload = self._rule(previous, value)
        if payload is not None:
            self._hub.publish(self._channel, payload)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return dict.__getitem__(self, key)

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value


def observed_workflow_rows(hub: WakeupHub) -> ObservedRows:
    return ObservedRows(
        hub,
        WakeupChannel.WORKFLOW_DISPATCH,
        lambda previous, workflow: workflow_wakeup(
            None if previous is None else workflow_wakeup_fields(previous), workflow
        ),
    )


def observed_remote_command_rows(hub: WakeupHub) -> ObservedRows:
    return ObservedRows(
        hub,
        WakeupChannel.REMOTE_COMMAND,
        lambda previous, command: remote_command_wakeup(
            None if previous is None else previous.status.value, command
        ),
    )


class InProcessWakeupMixin:
    """``run_wakeup_listener`` over a :class:`WakeupHub` (memory, SQLite)."""

    # Attributes supplied by the composed concrete implementation.
    _wakeup_hub: WakeupHub

    def run_wakeup_listener(
        self,
        channel: WakeupChannel,
        stop_event: threading.Event,
        on_notification: Callable[[dict[str, Any]], None],
        *,
        timeout_seconds: float = 1.0,
        on_state: Callable[[bool], None] | None = None,
    ) -> None:
        """See ``WakeupStore.run_wakeup_listener``.

        ``timeout_seconds`` bounds how long a stop request waits, exactly as
        the ``notifies`` timeout does on PostgreSQL.
        """

        if timeout_seconds <= 0:
            raise ValueError("wakeup listener timeout must be positive")
        with self._wakeup_hub.subscribe(channel) as subscription:
            if on_state is not None:
                on_state(True)
            while not stop_event.is_set():
                subscription.wait(timeout_seconds)
                seen: set[tuple[tuple[str, Any], ...]] = set()
                for payload in subscription.drain():
                    key = tuple(sorted(payload.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                    on_notification(payload)
                    if stop_event.is_set():
                        break
        if on_state is not None:
            on_state(False)
