"""Workflows an incident re-planned away from, and how they stop blocking.

When a family re-plans an incident it normally links the new workflow as a
preempting successor, and the executor supersedes the predecessor at a step
boundary. ``families/health.py`` links one only in its "serialized behind a
node-exclusive incumbent" branch, so an incident that escalates for an unrelated
reason -- an idle-utilization signal, say -- bumps its generation, points
``workflow_request_id`` at a new workflow, and leaves the old record carrying
every field a live workflow has.

Nothing closed that record. ``_validate_fencing`` only rejects a stale token
while the incident still names *this* workflow; once it names another, the
mismatch is read as a preemption for the (absent) successor link to resolve. On
2026-09-04 a ``RESTART_APP`` workflow at generation 1 was therefore dispatched
every 80 seconds for four and a half hours, and did two kinds of damage: it
blocked every release through the ``workflow_safety`` gate -- including the
release carrying this fix -- and, because ``run_once`` admits one workflow per
incident, it starved the successor its incident was actually waiting on, which
sat ``ACTION_PENDING`` the entire time.

These cases pin the terminalization, the un-starving, the reservation it must
not leak, and the conditions under which the sweep must keep its hands off.

This sweep only ever closes the cheap shape: a record no adapter was handed and
nobody is driving. The record that actually cost a fleet was ``RUNNING`` with an
unsettled destructive remote command, and closing that one needs a
remote-command pass first -- ``test_retired_generation.py`` covers it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.workflow_resolution import abandoned_generation_successor
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome

NOW = datetime(2026, 9, 4, 11, 17, tzinfo=timezone.utc)
INCIDENT = "inc-a"
ABANDONED = "workflow-abandoned"
CURRENT = "workflow-current"
CLUSTER = "cluster-a"
JOB = "training-a"

# The shape of the live record: RESTART_APP, whose STOP_WORKLOADS and
# RESTART_WORKLOAD are what made it a release blocker in the first place.
ABANDONED_STEPS = [
    workflow_step(WorkflowOperation.FREEZE_EVIDENCE),
    workflow_step(WorkflowOperation.STOP_WORKLOADS),
    workflow_step(
        WorkflowOperation.RESTART_WORKLOAD,
        parameters={"cluster_id": CLUSTER, "job_id": JOB, "restart_budget": 1},
    ),
]


def scenario(
    store,
    *,
    abandoned_status=WorkflowStatus.PENDING,
    abandoned_token: int = 1,
    incident_token: int = 4,
    incident_points_at: str = CURRENT,
    successor_incident_id: str = INCIDENT,
    successor_token: int | None = None,
    successor_predecessor: str | None = None,
    steps=None,
    **abandoned_values,
):
    """The live 2026-09-04 shape: one stale record, one current successor.

    Defaults are the real generations -- the abandoned workflow at 1, the
    incident and its successor at 4 -- so each case below overrides exactly the
    one field whose effect it is testing.
    """

    incident = fault_incident(
        INCIDENT,
        "event-a",
        cluster_id=CLUSTER,
        state=IncidentState.ACTION_PENDING,
        fencing_token=incident_token,
        workflow_request_id=incident_points_at,
    )
    store.save_incident(incident)
    abandoned = workflow_request(
        ABANDONED,
        INCIDENT,
        status=abandoned_status,
        fencing_token=abandoned_token,
        official_action="RESTART_APP",
        official_steps=ABANDONED_STEPS if steps is None else steps,
        **abandoned_values,
    )
    store.save_workflow(abandoned)
    if successor_incident_id != INCIDENT:
        # The successor still gets dispatched on this pass, so its own incident
        # has to exist -- otherwise the case fails on a missing record instead
        # of on the behaviour it is describing.
        store.save_incident(
            fault_incident(
                successor_incident_id,
                "event-b",
                cluster_id=CLUSTER,
                state=IncidentState.ACTION_PENDING,
                fencing_token=incident_token,
                workflow_request_id=CURRENT,
            )
        )
    current = workflow_request(
        CURRENT,
        successor_incident_id,
        fencing_token=(incident_token if successor_token is None else successor_token),
        official_action="RUN_DIAGNOSTICS",
        official_steps=[workflow_step(WorkflowOperation.VALIDATE_GPU)],
        predecessor_workflow_id=successor_predecessor,
    )
    store.save_workflow(current)
    return incident, abandoned, current


def dispatcher(store, adapters=(), operations=frozenset()):
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, adapters, operations),
        WorkflowDispatcherConfig(enabled=True, batch_size=100),
    )


def test_the_abandoned_generation_is_terminalized_with_its_successor_named() -> None:
    """SUPERSEDED, and pointing at what replaced it.

    The status alone would stop the blocking; ``preempted_by_workflow_id`` is
    what lets an operator reading the record afterwards see that the incident
    moved on rather than that something silently gave up on a remediation.
    """

    store = build_store()
    scenario(store)

    dispatcher(store).run_once()

    resolved = store.get_workflow(ABANDONED)
    assert resolved.status is WorkflowStatus.SUPERSEDED
    assert resolved.preempted_by_workflow_id == CURRENT
    assert resolved.superseded_at is not None
    assert "advanced to generation 4" in (resolved.preemption_reason or "")
    # The audit trail lives in ``preemption_reason``; ``blocked_reasons`` is the
    # executor's safety-step switch and must not carry prose (P0-61A).
    assert resolved.blocked_reasons == []
    assert "generation 4" in (resolved.preemption_reason or ""), (
        "the audit trail has to survive on the record, not only in a log line"
    )


def test_terminalizing_it_unstarves_the_workflow_the_incident_is_waiting_on() -> None:
    """The half of the damage that is invisible in the release gate.

    ``run_once`` admits one workflow per incident, and the abandoned record won
    that slot on every pass. So the incident's *current* workflow never ran
    either -- the incident sat ACTION_PENDING while the thing meant to diagnose
    it was never dispatched. Superseding the predecessor has to hand the slot
    over on the same pass, otherwise the fix trades a wedged release for a
    permanently undiagnosed incident.
    """

    store = build_store()
    scenario(store)
    adapter = FakeAdapter(
        {WorkflowOperation.VALIDATE_GPU: WorkflowStepOutcome.succeeded()}
    )

    dispatcher(store, [adapter], [WorkflowOperation.VALIDATE_GPU]).run_once()

    assert store.get_workflow(ABANDONED).status is WorkflowStatus.SUPERSEDED
    assert store.get_workflow(CURRENT).status is WorkflowStatus.SUCCEEDED, (
        "the successor has to get the incident's admission slot on the same pass"
    )


def test_a_restart_reservation_no_adapter_attempted_is_released() -> None:
    """Otherwise the cleanup spends the job's restart budget.

    Planning reserves against the job's restart budget for a RESTART_WORKLOAD
    step. Terminalizing a workflow that never restarted anything, without
    releasing that reservation, silently charges the next real restart of the
    same job -- the failure mode is a legitimate remediation later refused for
    lack of budget, with nothing pointing back to here.
    """

    store = build_store()
    _, abandoned, _ = scenario(store)
    reservation = f"{ABANDONED}/2/{WorkflowOperation.RESTART_WORKLOAD.value}"
    store.reserve_job_restart(CLUSTER, JOB, 1, reservation)

    dispatcher(store).run_once()

    assert store.get_workflow(ABANDONED).status is WorkflowStatus.SUPERSEDED
    store.reserve_job_restart(CLUSTER, JOB, 1, "a-later-real-restart")


def test_the_sweep_is_not_reported_as_a_dispatch_failure() -> None:
    """Cleanup must not look like a workflow that failed.

    Folding it into ``failed`` would put a routine supersession into the
    dispatcher's failure count, where it becomes an alert about a remediation
    that never happened.
    """

    store = build_store()
    scenario(store)
    adapter = FakeAdapter(
        {WorkflowOperation.VALIDATE_GPU: WorkflowStepOutcome.succeeded()}
    )

    report = dispatcher(store, [adapter], [WorkflowOperation.VALIDATE_GPU]).run_once()

    assert store.get_workflow(ABANDONED).status is WorkflowStatus.SUPERSEDED
    assert report.failed == 0
    assert report.failures == []


STARTED_SHAPES = [
    # Past PENDING, or holding something a live workflow holds. The abandoned
    # predicate refuses every one of them: it only accepts a record no adapter
    # has ever been handed and nobody is driving. What closes these instead is
    # the retired-generation revocation, which cancels remote commands under the
    # dispatcher's own lease -- see ``test_retired_generation.py``.
    ("a step completed", {"completed_step_indexes": [0]}),
    (
        "an operation completed",
        {"completed_operations": [WorkflowOperation.FREEZE_EVIDENCE]},
    ),
    (
        "it holds a remediation budget claim",
        {"remediation_budget_claims": ["cluster-a/restart"]},
    ),
    ("somebody is executing it", {"execution_owner_id": "executor-b"}),
    (
        "its lease has not expired",
        {"execution_lease_expires_at": NOW + timedelta(days=3650)},
    ),
    ("it is RUNNING", {"abandoned_status": WorkflowStatus.RUNNING}),
    ("it is SAFETY_PENDING", {"abandoned_status": WorkflowStatus.SAFETY_PENDING}),
]


@pytest.mark.parametrize(
    "name,overrides",
    [
        # Each of these leaves some reading under which the workflow is still
        # live, so neither sweep may touch it and the record goes on blocking.
        # They are the fail-closed half of both predicates.
        (
            "a step was handed to a local adapter",
            {
                "step_executions": [
                    workflow_step_execution(
                        0,
                        WorkflowOperation.FREEZE_EVIDENCE,
                        status=WorkflowStepStatus.WAITING,
                    )
                ]
            },
        ),
        # The generation comparison is strict, and the successor has to really
        # be the incident's current workflow for this incident. An equal
        # generation is left alone only while something links the pair: an
        # *unlinked* same-generation twin is closed too (P1-80A, pinned in
        # tests/store/test_merge_executor_isolation.py).
        (
            "its generation is not behind and the successor links back to it",
            {"abandoned_token": 4, "successor_predecessor": ABANDONED},
        ),
        ("its generation is ahead", {"abandoned_token": 5}),
        ("the incident still names it", {"incident_points_at": ABANDONED}),
        (
            "the successor belongs to another incident",
            {"successor_incident_id": "inc-b"},
        ),
        ("the successor is not at the incident's generation", {"successor_token": 3}),
        # Somebody already owns this record: with ``preempt_predecessor`` the
        # executor supersedes it at a step boundary -- and that is the only path
        # that compensates an unrestored quiesce -- while without it the
        # successor is queued behind it and it is meant to run to completion.
        ("the successor names it as predecessor", {"successor_predecessor": ABANDONED}),
    ],
)
def test_anything_unproven_keeps_blocking(name: str, overrides: dict) -> None:
    store = build_store()
    status_override = overrides.pop("abandoned_status", WorkflowStatus.PENDING)
    scenario(store, abandoned_status=status_override, **overrides)

    dispatcher(store).run_once()

    assert store.get_workflow(ABANDONED).status is not WorkflowStatus.SUPERSEDED, (
        f"the sweep must not terminalize a workflow when {name}"
    )


@pytest.mark.parametrize("name,overrides", STARTED_SHAPES)
def test_the_abandoned_predicate_refuses_a_workflow_that_started(
    name: str, overrides: dict
) -> None:
    """The cheap predicate stays cheap.

    Each shape here is now closed by the retired-generation revocation instead,
    which is allowed to be more invasive because it cancels remote commands
    first. This predicate must not follow it there: it terminalizes without a
    remote-command pass, so it may only ever accept a record that never ran.
    """

    store = build_store()
    status_override = overrides.pop("abandoned_status", WorkflowStatus.PENDING)
    _, abandoned, _ = scenario(store, abandoned_status=status_override, **overrides)

    assert abandoned_generation_successor(store, abandoned, now=NOW) is None, (
        f"the abandoned predicate must refuse a workflow when {name}"
    )


def test_a_missing_incident_leaves_the_workflow_alone() -> None:
    """A Store read that cannot prove the workflow dead is not permission.

    The record keeps blocking, which is the same posture the fleet fence takes
    when it loses its supersession evidence.
    """

    store = build_store()
    abandoned = workflow_request(
        ABANDONED, "inc-missing", fencing_token=1, official_steps=ABANDONED_STEPS
    )
    store.save_workflow(abandoned)

    assert abandoned_generation_successor(store, abandoned, now=NOW) is None


def test_a_deliberately_aged_pending_workflow_is_not_swept() -> None:
    """The predicate is not keyed on age, and must not become so.

    A workflow can legitimately sit PENDING for hours behind an aggregation
    deadline or the fleet rollout fence. A staleness threshold would eventually
    terminalize one of those -- a remediation quietly dropped instead of run --
    which is why the only evidence accepted is the incident having moved to a
    strictly higher generation.
    """

    store = build_store()
    incident = fault_incident(
        INCIDENT,
        "event-a",
        cluster_id=CLUSTER,
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
        workflow_request_id=ABANDONED,
    )
    store.save_incident(incident)
    held = workflow_request(
        ABANDONED,
        INCIDENT,
        fencing_token=1,
        official_steps=ABANDONED_STEPS,
        created_at=NOW - timedelta(days=3),
        updated_at=NOW - timedelta(days=3),
    )
    store.save_workflow(held)

    assert abandoned_generation_successor(store, held, now=NOW) is None
