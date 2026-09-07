"""Per-command lease guard shared between the regional executor and the adapter.

The regional executor holds a lease on the command it is executing and renews
it on a side thread. When renewal fails repeatedly, or the local view of the
lease has expired, or the control plane reports a cancellation, the executor
must stop *starting* node actions for that command -- but the adapter API is
``execute(context)`` and the context model is owned elsewhere. A context
variable carries the executor's verdict into the adapter without widening
that API: the executor sets it around ``adapter.execute`` on the thread that
runs the command, and ``_send_action`` consults it before every send.

The guard never interrupts an in-flight node-side operation; the agent ledger
owns that. It only refuses to begin another one.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Callable

active_lease_guard: ContextVar[Callable[[], str | None] | None] = ContextVar(
    "gpu_fault_node_action_lease_guard",
    default=None,
)


def lease_hold_reason() -> str | None:
    """Why no new node action may start for the current command, or None."""

    guard = active_lease_guard.get()
    if guard is None:
        return None
    return guard()
