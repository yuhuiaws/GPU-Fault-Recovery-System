"""Mark a predecessor as about to be preempted, in the merge transaction (F-C1).

A merge that creates a stronger successor used to persist two rows -- the
incident and the successor -- and leave the predecessor untouched until an
executor happened to claim it and discover the successor. In the meantime
the dispatcher could start the predecessor. The stamp below is written in
the same transaction so the dispatcher holds the predecessor at once.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.models import EXECUTABLE_WORKFLOW_STATUSES, WorkflowRequest


def preemption_pending_update(
    successor: WorkflowRequest,
    predecessor: WorkflowRequest | None,
) -> WorkflowRequest | None:
    """The predecessor row to write alongside ``successor``, or ``None``."""

    if (
        not successor.preempt_predecessor
        or predecessor is None
        or predecessor.request_id == successor.request_id
        or predecessor.status not in EXECUTABLE_WORKFLOW_STATUSES
        or predecessor.preemption_pending_by_workflow_id == successor.request_id
    ):
        return None
    return predecessor.model_copy(
        update={
            "preemption_pending_by_workflow_id": successor.request_id,
            "updated_at": datetime.now(timezone.utc),
        }
    )
