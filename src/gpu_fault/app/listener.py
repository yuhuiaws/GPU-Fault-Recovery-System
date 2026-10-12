"""What this process knows about the service socket it accepts on.

``gpu_fault.app.serve`` binds one listening socket per worker process with
``SO_REUSEPORT`` and registers it here; ``gpu_fault.app.process_health``
renders the facts on ``/metrics``: whether the socket is the worker's own,
how many connections it has accepted and how many it holds open. The two
modules never import each other, so the launcher can stay a thin shell over
uvicorn and the metrics render can stay free of launcher imports.

A process started by the plain ``uvicorn`` command line registers nothing -
it inherited the supervisor's shared socket and cannot count its own accepts
- and the families stay absent from its render.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ServiceListener:
    """The service socket of this process.

    ``accepted_total`` is bumped from the event loop thread on every
    ``connection_made``; the GIL makes the increment atomic and the metrics
    render only reads it, so no lock is taken on the accept path.
    ``open_connections`` is supplied by the launcher and reads uvicorn's live
    connection set, which is the number of clients pinned to this worker -
    the figure that says whether long-lived connections have left one worker
    with more than its share.
    """

    port: int
    reuse_port: bool
    worker_index: int | None = None
    accepted_total: int = 0
    open_connections: Callable[[], int] | None = None

    def connection_accepted(self) -> None:
        self.accepted_total += 1

    def snapshot(self) -> dict[str, Any]:
        open_count = 0
        if self.open_connections is not None:
            try:
                open_count = int(self.open_connections())
            except Exception:  # noqa: BLE001 - a metrics read must never raise
                open_count = 0
        return {
            "port": self.port,
            "reuse_port": self.reuse_port,
            "worker_index": self.worker_index,
            "accepted_total": self.accepted_total,
            "open_connections": open_count,
        }


_LISTENER: ServiceListener | None = None


def register_service_listener(listener: ServiceListener) -> ServiceListener:
    """Make ``listener`` the process's service socket of record."""

    global _LISTENER
    _LISTENER = listener
    return listener


def service_listener() -> ServiceListener | None:
    """The registered service socket, or ``None`` under the plain uvicorn CLI."""

    return _LISTENER


def reset_service_listener() -> None:
    """Forget the registration (tests that build several listeners)."""

    global _LISTENER
    _LISTENER = None
