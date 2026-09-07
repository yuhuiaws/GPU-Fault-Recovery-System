from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from gpu_fault.models import FaultIncident, RecoveryPlan, WorkflowRequest
from gpu_fault.store.shared.errors import NotFoundError, StaleWriteError
from gpu_fault.retired_generation import retired_generation_records
from gpu_fault.workflow_resolution import reconciled_restore_records
from gpu_fault.store.shared.preemption import preemption_pending_update

if TYPE_CHECKING:
    from gpu_fault.regional import RemoteActionCommand

LOGGER = logging.getLogger(__name__)


# The three fields ``save_workflow`` guards when the caller passes no
# ``expected`` copy, each with the writer family that moves it (store review
# 2026-09-07, item B). Shared by the three backends so the refusal reads the
# same everywhere.
WORKFLOW_VERSION_FIELDS: tuple[tuple[str, str], ...] = (
    ("merge_revision", "a merge widened the record"),
    ("execution_epoch", "an executor re-leased it"),
    ("fencing_token", "the incident moved to a new generation"),
)


def stale_workflow_versions(
    stored: WorkflowRequest, workflow: WorkflowRequest
) -> StaleWriteError | None:
    """The refusal a version-guarded ``save_workflow`` raises, or None.

    Names every one of the three fields that moved and what moving it means,
    so the caller can tell a merge from a re-lease from a generation change
    without reading the row again first.
    """

    moved = [
        f"{name} ({why}: stored {getattr(stored, name)}, "
        f"caller {getattr(workflow, name)})"
        for name, why in WORKFLOW_VERSION_FIELDS
        if getattr(stored, name) != getattr(workflow, name)
    ]
    if not moved:
        return None
    return StaleWriteError(
        f"workflow/{workflow.request_id} moved since it was read -- "
        + "; ".join(moved)
        + " -- re-read the row and reapply the change"
    )


def workflow_matches_expected(
    current: WorkflowRequest | None, expected: WorkflowRequest
) -> bool:
    """Whole-payload equality for the ``expected=`` compare-and-set form of
    ``save_workflow`` on the backends without a SQL-level CAS."""

    return current is not None and current.model_dump() == expected.model_dump()


def lease_extension_due(
    workflow: WorkflowRequest, *, renewed_at: datetime, lease_duration: timedelta
) -> bool:
    """Whether ``renew_workflow_lease`` should write, once the lease validated.

    Only when less than half of ``lease_duration`` remains. The executor renews
    before and after every step, seconds into a 3-minute lease, and each write
    re-serialized the whole record into a new heap tuple, its TOAST chunks and
    every partial index entry (store review 2026-09-07, item F1).
    """

    expires_at = workflow.execution_lease_expires_at
    assert expires_at is not None  # validated by the caller
    return expires_at - renewed_at <= lease_duration / 2


class TransactionalWorkflowMixin:
    """Multi-row incident/workflow/plan transactions shared by SQLite and PostgreSQL.

    Lock order rule (F-B2, extended by store review 2026-09-07, item C). Every
    transaction here that touches more than one row takes its locks in one
    order:

    1. advisory locks first, sorted by key (``_state_transaction`` takes one;
       ``claim_workflow`` sorts its budget scopes);
    2. the incident row;
    3. workflow rows, in ascending ``request_id``;
    4. the plan row.

    The families use different advisory keys (``incident_workflow/``,
    ``replacement_fault_group/``, ``sxid_fault_group/``,
    ``workflow_reconcile/``, ``retired_generation/``) and the executor's leased
    writers take none, so it is the row locks that serialize them -- and two
    families taking the same two rows in opposite orders were a deadlock pair.
    ``save_incident_and_workflow`` and ``save_workflow_and_incident_if_leased``
    went workflow -> incident while every merge went incident -> workflow: the
    executor held W and waited for I, the merge held I and waited for W, and
    PostgreSQL aborted one side after ``deadlock_timeout`` with 40P01. That is
    classified retryable, so it surfaced as one-second stalls and 503s rather
    than BLOCKED rows. The reconcile paths read the workflow *unlocked* only to
    learn its incident, lock the incident, then the workflows in key order,
    then the plan, and re-check the workflow's incident pointer under its lock
    (a mismatch is a ``StaleWriteError``). ``_duplicate_event_records`` follows
    the same order: the linked incident, then its workflow.

    The one sanctioned exception is the preemption stamp
    (``_stamp_preemption_pending``): after the successor it hops to the
    predecessor workflow, whatever its key. Predecessors are strictly older
    rows and the hop never returns to the incident, so within one incident
    every rule-following transaction is still waiting at the incident row and
    the hop cannot close a cycle; across incidents every family here locks only
    workflows of the incident it locked first, so no transaction that holds the
    predecessor can be waiting on a row the stamping transaction holds.
    """

    # Attributes supplied by the composed concrete implementation.
    _get: Callable[..., Any]
    _get_for_update: Callable[..., Any]
    _get_link: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _list: Callable[..., Any]
    _link: Callable[..., Any]
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]
    list_remote_commands: Callable[..., list[RemoteActionCommand]]

    # How many ``incident_by_event`` links pointed at a record that was not
    # there when a duplicate event arrived, and were repaired by rebuilding
    # (F-B7). Per store instance; read by the metrics family. A plain class
    # default (not an annotated contract attribute): it is state this mixin
    # owns, not something the composed store has to supply.
    stale_event_link_repairs = 0

    def _duplicate_event_records(
        self, event_id: str
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        """The incident and workflow an already-ingested event resolves to.

        ``None`` means "treat this event as new". That covers the plain case
        of no link, and the dirty case: a link whose incident is gone, whose
        incident never got a workflow pointer, or whose workflow row is
        missing. The fast path used an unguarded ``_get`` there, so a dangling
        pointer made every re-post of the event fail inside the transaction --
        and a re-post is exactly the retry path, so the event became a poison
        message (P1-57F, P1-69D). Falling through is the idempotent repair:
        the caller rebuilds from the group state and re-links this
        ``event_id``, and ``_link`` is an upsert, so the dirty link is
        overwritten rather than left for the next retry to trip on.

        The reads take the row lock (``_locked_optional``): a duplicate return
        is read-only, but a fall-through writes, and the group incident it will
        read next may be the same row. Incident first, then its workflow -- the
        lock order rule (class docstring).
        """

        incident_id = self._get_link("incident_by_event", event_id)
        if incident_id is None:
            return None
        incident = self._locked_optional("incident", incident_id)
        workflow = (
            self._locked_optional("workflow", incident.workflow_request_id)
            if incident is not None and incident.workflow_request_id
            else None
        )
        if incident is not None and workflow is not None:
            return incident, workflow
        self.stale_event_link_repairs += 1
        LOGGER.warning(
            "incident_by_event link for %s points at incident %s whose %s is "
            "missing; rebuilding the event instead of failing the re-post",
            event_id,
            incident_id,
            (
                "record"
                if incident is None
                else "workflow pointer"
                if not incident.workflow_request_id
                else f"workflow {incident.workflow_request_id}"
            ),
        )
        return None

    def _locked_optional(self, kind: str, key: str) -> Any:
        """Read a record for update inside the current transaction, or None.

        The merge paths used ``_get_optional`` -- no row lock -- so an executor
        holding the row could commit between the merge's read and its blind
        write, and one of the two writers lost (P0-78A). Backends without row
        locks (sqlite, memory) serialize on their process-wide lock instead.
        """

        try:
            return self._locked(kind, key)
        except NotFoundError:
            return None

    def _locked(self, kind: str, key: str) -> Any:
        """Read a record for update inside the current transaction (raises
        ``NotFoundError``); the plain read on backends without row locks."""

        getter: Callable[..., Any] = getattr(self, "_get_for_update", self._get)
        return getter(kind, key)

    def _locked_workflows(self, request_ids: Iterable[str]) -> dict[str, Any]:
        """Lock workflow rows in ascending ``request_id`` (lock order rule,
        step 3), whatever order the caller named them in."""

        return {
            request_id: self._locked("workflow", request_id)
            for request_id in sorted(set(request_ids))
        }

    @staticmethod
    def _require_incident_pointer(
        workflow: WorkflowRequest, incident: FaultIncident
    ) -> None:
        """The incident was locked from an unlocked read of the workflow; a
        pointer that moved in between means the row was re-parented under
        another generation while we waited, and the caller must re-read."""

        if workflow.incident_id != incident.incident_id:
            raise StaleWriteError(
                f"workflow/{workflow.request_id} moved from incident "
                f"{incident.incident_id} to {workflow.incident_id} while its "
                "incident was being locked -- re-read and retry"
            )

    @staticmethod
    def _merged(
        existing: WorkflowRequest | None, workflow: WorkflowRequest
    ) -> WorkflowRequest:
        """Stamp a merge into an existing row so leased writers can see it (F-B1)."""

        if existing is None or existing.request_id != workflow.request_id:
            return workflow
        return workflow.model_copy(
            update={"merge_revision": existing.merge_revision + 1}
        )

    def amend_workflow(
        self,
        request_id: str,
        updates: Mapping[str, object],
    ) -> WorkflowRequest:
        with self._state_transaction(f"workflow/{request_id}"):
            current = self._locked_optional("workflow", request_id)
            if current is None:
                raise NotFoundError(request_id)
            amended: WorkflowRequest = current.model_copy(
                update={
                    **dict(updates),
                    "merge_revision": current.merge_revision + 1,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._put("workflow", request_id, amended)
            return amended

    def save_incident_and_workflow(
        self,
        incident: FaultIncident,
        workflow: WorkflowRequest,
    ) -> None:
        if incident.workflow_request_id != workflow.request_id:
            raise ValueError("incident workflow pointer does not match workflow")
        if workflow.incident_id != incident.incident_id:
            raise ValueError("workflow incident pointer does not match incident")
        with self._state_transaction(f"incident_workflow/{incident.incident_id}"):
            # Lock order rule: the incident row first (None for a new
            # incident), then the workflow. Taking the workflow first made this
            # writer the W->I half of a deadlock pair with every merge.
            self._locked_optional("incident", incident.incident_id)
            existing = self._locked_optional("workflow", workflow.request_id)
            workflow = self._merged(existing, workflow)
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._stamp_preemption_pending(workflow)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )

    def reconcile_restored_workflow(
        self,
        workflow_request_id: str,
        successor_workflow_id: str,
        *,
        expected_fencing_token: int,
        expected_execution_epoch: int,
        expected_workflow_updated_at: datetime | None = None,
        reference: str,
        reconciled_at: datetime,
    ) -> tuple[WorkflowRequest, FaultIncident, RecoveryPlan]:
        with self._state_transaction(f"workflow_reconcile/{workflow_request_id}"):
            # Lock order rule: an unlocked read only to learn the incident,
            # then incident -> workflows (key order) -> plan; the pointer is
            # re-checked once the workflow is actually locked.
            incident_id = self._get("workflow", workflow_request_id).incident_id
            incident = self._locked("incident", incident_id)
            rows = self._locked_workflows((workflow_request_id, successor_workflow_id))
            workflow = rows[workflow_request_id]
            successor = rows[successor_workflow_id]
            self._require_incident_pointer(workflow, incident)
            if not workflow.source_plan_id:
                raise ValueError("workflow has no source recovery plan")
            source_plan = self._locked("plan", workflow.source_plan_id)
            updated_workflow, updated_incident, updated_plan = (
                reconciled_restore_records(
                    workflow,
                    incident,
                    successor,
                    source_plan,
                    # Only this workflow's commands (F-J5): the full-table load
                    # ran inside the lock-holding transaction (P1-78D).
                    self.list_remote_commands(
                        workflow_request_ids=[workflow_request_id]
                    ),
                    expected_fencing_token=expected_fencing_token,
                    expected_execution_epoch=expected_execution_epoch,
                    expected_workflow_updated_at=expected_workflow_updated_at,
                    reference=reference,
                    reconciled_at=reconciled_at,
                )
            )
            self._put("workflow", workflow_request_id, updated_workflow)
            self._put("incident", incident.incident_id, updated_incident)
            self._put("plan", source_plan.plan_id, updated_plan)
            return updated_workflow, updated_incident, updated_plan

    def reconcile_retired_generation_workflow(
        self,
        workflow_request_id: str,
        successor_workflow_id: str,
        *,
        expected_fencing_token: int,
        reference: str | None,
        reconciled_at: datetime,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        """Terminalize a workflow whose incident has moved to a later generation.

        Unlike ``reconcile_restored_workflow`` this takes no expected
        ``updated_at``. A retired generation is still being dispatched, and each
        dispatch renews its execution lease and stamps the row, so an
        ``updated_at`` precondition would never hold. The compare-and-set is on
        ``fencing_token`` -- the value that decides whether the record is behind
        its incident -- and every other condition is re-derived inside this
        transaction by ``retired_generation_records``.

        The write releases the execution lease rather than requiring it. That is
        the point of a revocation: the lease holder is the dispatcher that keeps
        the retired record alive, and its next ``save_workflow_if_leased`` is
        meant to fail.
        """

        with self._state_transaction(f"retired_generation/{workflow_request_id}"):
            # Lock order rule: see ``reconcile_restored_workflow``.
            incident_id = self._get("workflow", workflow_request_id).incident_id
            incident = self._locked("incident", incident_id)
            rows = self._locked_workflows((workflow_request_id, successor_workflow_id))
            workflow = rows[workflow_request_id]
            successor = rows[successor_workflow_id]
            self._require_incident_pointer(workflow, incident)
            updated_workflow, updated_incident = retired_generation_records(
                workflow,
                incident,
                successor,
                self.list_remote_commands(workflow_request_ids=[workflow_request_id]),
                expected_fencing_token=expected_fencing_token,
                reference=reference,
                reconciled_at=reconciled_at,
            )
            self._put("workflow", workflow_request_id, updated_workflow)
            self._put("incident", incident.incident_id, updated_incident)
            return updated_workflow, updated_incident

    def create_incident_workflow_if_absent(
        self,
        event_id: str,
        builder: Callable[[], tuple[FaultIncident, WorkflowRequest]],
        *,
        serialization_key: str | None = None,
    ) -> tuple[FaultIncident, WorkflowRequest, bool]:
        with self._state_transaction(
            "incident_workflow/" + (serialization_key or event_id)
        ):
            duplicate = self._duplicate_event_records(event_id)
            if duplicate is not None:
                return duplicate[0], duplicate[1], False
            incident, workflow = builder()
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._stamp_preemption_pending(workflow)
            self._link(
                "incident_by_event",
                event_id,
                incident.incident_id,
            )
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )
            return incident, workflow, True

    def _stamp_preemption_pending(self, successor: WorkflowRequest) -> None:
        """In the merge transaction, hold the predecessor a successor preempts (F-C1).

        The sanctioned exception to the lock order rule (class docstring): the
        hop from the successor to its strictly older predecessor.
        """

        if not (successor.preempt_predecessor and successor.predecessor_workflow_id):
            return
        predecessor = self._locked_optional(
            "workflow", successor.predecessor_workflow_id
        )
        stamped = preemption_pending_update(successor, predecessor)
        if stamped is not None:
            self._put("workflow", stamped.request_id, stamped)

    def merge_replacement_workflow(
        self,
        group_key: str,
        event_id: str,
        builder: Callable[
            [FaultIncident | None, WorkflowRequest | None],
            tuple[FaultIncident, WorkflowRequest],
        ],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        with self._state_transaction(f"replacement_fault_group/{group_key}"):
            duplicate = self._duplicate_event_records(event_id)
            if duplicate is not None:
                return duplicate
            incident_id = self._get_link("replacement_fault_group", group_key)
            existing_incident = (
                self._locked_optional("incident", incident_id)
                if incident_id is not None
                else None
            )
            existing_workflow = (
                self._locked_optional(
                    "workflow",
                    existing_incident.workflow_request_id,
                )
                if existing_incident is not None
                and existing_incident.workflow_request_id
                else None
            )
            incident, workflow = builder(existing_incident, existing_workflow)
            workflow = self._merged(existing_workflow, workflow)
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._stamp_preemption_pending(workflow)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )
            self._link(
                "incident_by_event",
                event_id,
                incident.incident_id,
            )
            self._link(
                "replacement_fault_group",
                group_key,
                incident.incident_id,
            )
            return incident, workflow

    def merge_attempt_fault_workflow(
        self,
        group_key: str,
        event_id: str,
        builder: Callable[
            [FaultIncident | None, WorkflowRequest | None],
            tuple[FaultIncident, WorkflowRequest],
        ],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        with self._state_transaction(f"sxid_fault_group/{group_key}"):
            duplicate = self._duplicate_event_records(event_id)
            if duplicate is not None:
                return duplicate
            incident_id = self._get_link("sxid_fault_group", group_key)
            existing_incident = (
                self._locked_optional("incident", incident_id)
                if incident_id is not None
                else None
            )
            existing_workflow = (
                self._locked_optional(
                    "workflow",
                    existing_incident.workflow_request_id,
                )
                if existing_incident is not None
                and existing_incident.workflow_request_id
                else None
            )
            incident, workflow = builder(existing_incident, existing_workflow)
            workflow = self._merged(existing_workflow, workflow)
            self._put("incident", incident.incident_id, incident)
            self._put("workflow", workflow.request_id, workflow)
            self._stamp_preemption_pending(workflow)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )
            self._link(
                "incident_by_event",
                event_id,
                incident.incident_id,
            )
            self._link(
                "sxid_fault_group",
                group_key,
                incident.incident_id,
            )
            return incident, workflow

    def merge_sxid_workflow(
        self,
        group_key: str,
        event_id: str,
        builder: Callable[
            [FaultIncident | None, WorkflowRequest | None],
            tuple[FaultIncident, WorkflowRequest],
        ],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        return self.merge_attempt_fault_workflow(group_key, event_id, builder)
