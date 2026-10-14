"""The Agent's own GPU work against an open quiesce window on its node.

Live (GF-REGIONAL-DESTR-001, plain EKS, XID 46 reset): while the reset's
quiesce window was open, the control plane dispatched another incident's
``RUN_DCGM_DIAGNOSTIC`` to the same Agent, and the Agent ran
``dcgmi diag -r 1 -j`` inside the window -- ``VERIFY_NO_GPU_CLIENTS`` met the
Agent's own ``dcgmi`` (cgroup ``gpu-fault-node-agent.service``) on the target
device and ``nvvs`` beside it. The diagnostic graded the window itself (Fabric
Manager stopped, the target's persistence released) as a GPU failure, and the
escalation of that verdict drained a node the reset chain had just recovered.
The Agent had no notion of the window outside the reset's own claim.

The rules here make the window exclusive against the Agent's own GPU work:

- An operation in ``GPU_ACTIVITY_OPERATIONS`` of an incident other than the
  window's is refused at admission while a window is open on the node (a
  quiesce state file of this boot, or a QUIESCE still running): retryable,
  nothing reaches the ledger, and the control plane resubmits it after
  RESTORE_GPU_SERVICES closed the window. The window's own incident keeps its
  operations (the NVLink field diagnostic runs inside its own window).
- Every admitted such operation is registered for its whole run, and it is
  registered *before* the window check: a QUIESCE that opens after the check
  finds the activity in flight.
- QUIESCE marks the window open before it stops anything, flags each foreign
  activity in flight as overlapped and waits a bounded time for them to end.
- ``VERIFY_NO_GPU_CLIENTS`` and the reset's own re-checks fail closed while
  any registered activity is in flight: its subprocesses come and go between
  samples, so the sampled ``/proc`` scan alone cannot see it.
- An overlapped activity never reports a verdict: it stops at its next
  subprocess spawn, and its result is a retryable failure that runs again once
  the window is closed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from gpu_fault.models import WorkflowOperation

#: The Agent operations that open GPU clients of their own (``dcgmi`` and its
#: ``nvvs``, ``nvidia-smi``, the field diagnostic binary, the bug report) or
#: restart the collectors a quiesce stops. None of them is node-exclusive on
#: the control plane, so nothing there keeps them out of another workflow's
#: window.
GPU_ACTIVITY_OPERATIONS = frozenset(
    {
        WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.COLLECT_HUNG_TRIAGE,
        WorkflowOperation.RUN_FIELD_DIAGNOSTIC,
        WorkflowOperation.RUN_NVLINK74_WORKFLOW,
        WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
    }
)
#: How long a QUIESCE waits for foreign activities in flight to end before it
#: stops services under them. A ``dcgmi diag -r 1`` or a collector restart ends
#: well inside it; a longer bundle stops at its next spawn once flagged, and the
#: verification keeps the reset waiting until it has.
DEFAULT_WINDOW_DRAIN_SECONDS = 30.0
#: The admission refusal's wording; the HTTP layer maps it to a retryable 409.
WINDOW_OPEN_REFUSAL = "GPU maintenance window is open"
OUTCOME_LIMIT = 400


class GpuMaintenanceWindowOpen(ValueError):
    """Admission refusal: another incident's quiesce window is open here."""

    def __init__(
        self, operation: WorkflowOperation, windows: list[dict[str, Any]]
    ) -> None:
        owners = sorted({str(item.get("incident_id") or "unknown") for item in windows})
        super().__init__(
            f"{WINDOW_OPEN_REFUSAL} on this node (quiesce of {', '.join(owners)}); "
            f"{operation.value} runs after RESTORE_GPU_SERVICES closes it"
        )
        self.windows = windows


@dataclass
class GpuActivity:
    token: int
    operation: str
    command_id: str
    incident_id: str
    started_at: str
    overlapped_by: str | None = None

    def evidence(self) -> dict[str, str]:
        record = {
            "process_name": "gpu-fault-node-agent",
            "operation": self.operation,
            "command_id": self.command_id,
            "incident_id": self.incident_id,
            "started_at": self.started_at,
        }
        if self.overlapped_by is not None:
            record["overlapped_by"] = self.overlapped_by
        return record


class GpuActivityOverlapped(RuntimeError):
    """A quiesce window opened under this activity; its outcome is not a verdict."""

    def __init__(self, activity: GpuActivity, *, outcome: str) -> None:
        super().__init__(
            f"{activity.operation} overlapped the quiesce window of "
            f"{activity.overlapped_by}: its outcome describes the window, not "
            "the GPU, and it runs again once the window is closed"
        )
        self.action_details: dict[str, Any] = {
            "gpu_activity_overlapped_by": activity.overlapped_by,
            "gpu_activity_started_at": activity.started_at,
            "gpu_activity_outcome": outcome[:OUTCOME_LIMIT],
        }


class GpuActivityGate:
    """In-flight Agent GPU work and the quiesce windows it must stay out of.

    ``open_windows`` reads the windows on disk (quiesce state files of this
    boot); QUIESCE calls in progress are held in memory, so a window is closed
    to foreign work from the moment its QUIESCE starts.
    """

    def __init__(
        self,
        *,
        open_windows: Callable[[], list[dict[str, Any]]],
        now: Callable[[], datetime],
        drain_seconds: float = DEFAULT_WINDOW_DRAIN_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 <= drain_seconds <= 120:
            raise ValueError("GPU activity drain must be between 0 and 120 seconds")
        self.drain_seconds = drain_seconds
        self._open_windows = open_windows
        self._now = now
        self._monotonic = monotonic
        self._condition = threading.Condition()
        self._activities: dict[int, GpuActivity] = {}
        self._opening: dict[str, int] = {}
        self._tokens = 0
        self._local = threading.local()

    def _foreign_windows(self, incident_id: str) -> list[dict[str, Any]]:
        with self._condition:
            opening = [
                {"incident_id": owner, "phase": "OPENING"} for owner in self._opening
            ]
        return [
            window
            for window in [*opening, *self._open_windows()]
            if window.get("incident_id") != incident_id
        ]

    def refuse_if_window_open(
        self, operation: WorkflowOperation, incident_id: str
    ) -> None:
        if operation not in GPU_ACTIVITY_OPERATIONS:
            return
        windows = self._foreign_windows(incident_id)
        if windows:
            raise GpuMaintenanceWindowOpen(operation, windows)

    def admit(
        self, operation: WorkflowOperation, command_id: str, incident_id: str
    ) -> GpuActivity | None:
        """Register first, then look for a window: the order closes the race."""

        if operation not in GPU_ACTIVITY_OPERATIONS:
            return None
        with self._condition:
            self._tokens += 1
            activity = GpuActivity(
                token=self._tokens,
                operation=operation.value,
                command_id=command_id,
                incident_id=incident_id,
                started_at=self._now().isoformat(),
            )
            self._activities[activity.token] = activity
        try:
            self.refuse_if_window_open(operation, incident_id)
        except BaseException:
            self.release(activity)
            raise
        return activity

    def release(self, activity: GpuActivity | None) -> None:
        if activity is None:
            return
        with self._condition:
            self._activities.pop(activity.token, None)
            self._condition.notify_all()

    @contextmanager
    def running(self, activity: GpuActivity | None) -> Iterator[None]:
        """Bind ``activity`` to this thread for ``checkpoint`` and ``in_flight``."""

        previous = getattr(self._local, "activity", None)
        self._local.activity = activity
        try:
            yield
        finally:
            self._local.activity = previous

    def checkpoint(self) -> None:
        """Stop this thread's activity at its next spawn once it is overlapped."""

        activity = getattr(self._local, "activity", None)
        if activity is not None and activity.overlapped_by is not None:
            raise GpuActivityOverlapped(
                activity, outcome="stopped before its next GPU subprocess"
            )

    def outcome_failure(
        self, activity: GpuActivity | None, error: Exception | None
    ) -> Exception | None:
        """The failure to report for a finished activity, ``None`` for success."""

        if activity is None or activity.overlapped_by is None:
            return error
        if isinstance(error, GpuActivityOverlapped):
            return error
        outcome = (
            "SUCCEEDED" if error is None else f"FAILED: {type(error).__name__}: {error}"
        )
        return GpuActivityOverlapped(activity, outcome=outcome)

    def in_flight(self) -> list[dict[str, str]]:
        """Every registered activity but the calling thread's own."""

        current = getattr(self._local, "activity", None)
        with self._condition:
            return [
                activity.evidence()
                for token, activity in sorted(self._activities.items())
                if activity is not current
            ]

    def open_window(self, incident_id: str) -> dict[str, Any]:
        """Open ``incident_id``'s window to the Agent; flag and drain foreign work.

        Returns the evidence for the QUIESCE receipt: the foreign activities that
        were in flight and the ones still running when the wait ended (the
        verification refuses while those are in flight).
        """

        started = self._monotonic()
        with self._condition:
            self._opening[incident_id] = self._opening.get(incident_id, 0) + 1
            flagged = []
            for activity in self._activities.values():
                if activity.incident_id == incident_id:
                    continue
                if activity.overlapped_by is None:
                    activity.overlapped_by = incident_id
                flagged.append(activity.evidence())
            while True:
                remaining = [
                    activity
                    for activity in self._activities.values()
                    if activity.incident_id != incident_id
                ]
                left = started + self.drain_seconds - self._monotonic()
                if not remaining or left <= 0:
                    break
                self._condition.wait(left)
            still_running = [activity.evidence() for activity in remaining]
        if not flagged:
            return {}
        return {
            "agent_gpu_activities_overlapped": flagged,
            "agent_gpu_activities_running": still_running,
            "agent_gpu_activity_wait_seconds": round(self._monotonic() - started, 3),
        }

    def close_opening(self, incident_id: str) -> None:
        """The QUIESCE call ended; its state file is the window from here on."""

        with self._condition:
            count = self._opening.get(incident_id, 0) - 1
            if count > 0:
                self._opening[incident_id] = count
            else:
                self._opening.pop(incident_id, None)


class CheckpointedRunner:
    """The executor's subprocess runner, checkpointing the bound activity first.

    Every spawn an Agent operation makes goes through the executor's runner, so
    this is the one place an overlapped activity can be stopped before its next
    GPU client without threading a checkpoint through every handler. Attribute
    reads fall through to the wrapped runner (tests read its call log).
    """

    def __init__(self, runner: Callable[..., Any], gate: GpuActivityGate) -> None:
        self._runner = runner
        self._gate = gate

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._gate.checkpoint()
        return self._runner(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runner, name)
