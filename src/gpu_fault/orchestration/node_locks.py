"""Re-entrant locks keyed by ``(cluster_id, node_id)`` for the ingest paths.

The orchestrator used to hold one process-wide ``RLock`` across every
ingest, several store round trips included, which serialised the ingestion
of *all* nodes in *all* clusters inside a worker process (性能 1). The
processor lanes already serialise per node, and the store's
``merge_attempt_fault_workflow`` / ``create_incident_workflow_if_absent``
take their own advisory locks for the cross-node merges, so the only thing
the in-process lock still has to guarantee is that two ingests for the
*same* node do not interleave. That is what :class:`NodeLocks` keys on.

Lives in its own module so ``coordinator.py`` carries only the wiring; the
primitive is independent of the orchestrator and tested on its own.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock, RLock, local
from typing import Iterable, Iterator, cast

NodeLockKey = tuple[str, str]


class NodeLocks:
    """Per-node re-entrant locks that also stand in for one plain ``RLock``.

    ``held(keys)`` acquires the locks for a set of keys in sorted order --
    one global order, so a caller that needs several nodes (a distributed
    XID batch, a placement hold over an attempt's nodes, ``simulate`` over
    a workflow's nodes) cannot deadlock against another such caller. The
    locks stay ``RLock`` because the ingest paths re-enter: the drain and
    grouped-health families call back into ``ingest_node_health`` for the
    finding they are already holding.

    The object is also a context manager itself, standing in for the plain
    ``RLock`` the family services (``NodeHealthIngestionService``,
    ``ResetOperationService``) were built with. Those services take a lock
    at construction and ``with self.lock:`` per call, so they cannot name
    the node they are about to ingest; entering this object instead
    re-acquires whatever keys ``held`` recorded on the calling thread. The
    coordinator therefore wraps every call into a family in ``held(...)``
    and the family's own ``with`` is a free re-entry. Entered on a thread
    with no recorded scope -- a caller that bypassed the coordinator -- it
    falls back to one process-wide ``RLock``, i.e. the previous behaviour,
    rather than running unserialised. Alternative rejected: giving the
    families a lock *factory* keyed by the finding, which changes three
    modules' constructors for the same effect.

    The dictionary of locks only grows (one entry per node ever ingested
    by the process); that is bounded by fleet size and cheaper than the
    refcounting an eviction scheme would need.
    """

    def __init__(self) -> None:
        self._guard = Lock()
        self._locks: dict[NodeLockKey, RLock] = {}
        self._fallback = RLock()
        # Per-thread: the keys the innermost ``held`` recorded, and the
        # stack of lock lists ``__enter__`` acquired (so ``__exit__``
        # releases exactly what its own ``__enter__`` took).
        self._thread = local()

    def _lock_for(self, key: NodeLockKey) -> RLock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = RLock()
            return lock

    def _locks_for(self, keys: Iterable[NodeLockKey] | None) -> list[RLock]:
        ordered = sorted(set(keys)) if keys is not None else []
        if not ordered:
            return [self._fallback]
        return [self._lock_for(key) for key in ordered]

    @contextmanager
    def held(self, keys: Iterable[NodeLockKey]) -> Iterator[None]:
        """Hold the locks for ``keys`` (sorted, so one global order) and
        record them as the thread's current scope for ``__enter__``."""

        ordered = sorted(set(keys))
        locks = self._locks_for(ordered)
        for lock in locks:
            lock.acquire()
        previous = getattr(self._thread, "scope", None)
        self._thread.scope = ordered
        try:
            yield
        finally:
            self._thread.scope = previous
            for lock in reversed(locks):
                lock.release()

    def __enter__(self) -> "NodeLocks":
        locks = self._locks_for(getattr(self._thread, "scope", None))
        for lock in locks:
            lock.acquire()
        stack: list[list[RLock]] = getattr(self._thread, "stack", None) or []
        stack.append(locks)
        self._thread.stack = stack
        return self

    def __exit__(self, *exc_info: object) -> None:
        locks = self._thread.stack.pop()
        for lock in reversed(locks):
            lock.release()

    def as_rlock(self) -> RLock:
        """This object, typed as the ``RLock`` the family constructors
        declare. They only ever ``with`` it, which is the interface kept."""

        return cast(RLock, self)
