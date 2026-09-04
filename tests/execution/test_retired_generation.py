"""Closing a retired generation that had already started running.

``test_abandoned_generation.py`` covers the cheap half of this: a workflow its
incident re-planned away from before any adapter touched it. The record that
actually cost a fleet on 2026-09-04 was the other half. A ``RESTART_APP``
workflow sat at generation 1, ``RUNNING``, holding an execution owner, a lease
the dispatcher renewed every eighty seconds, six remediation budget claims and an
unsettled ``STOP_WORKLOADS`` remote command against three nodes of a live 24-GPU
job -- while its incident had recovered at generation 4 four and a half hours
earlier and pointed at a different workflow. The fleet rollout fence was the only
thing between that command and the job.

Nothing could close it. The abandoned sweep refuses anything past ``PENDING``,
``_validate_fencing`` read the pointer mismatch as a preemption for a successor
link that was never written, and the ``workflow_safety`` release gate counted the
record as a blocker -- including for the release that would have fixed it.

The order these cases pin is the whole design. A remote command outlives the
workflow row, so the commands are cancelled first and the row is closed only once
the Store shows them settled; a record that already *completed* something
destructive is never closed here at all, because there is real fleet state to
compensate for and that needs an operator.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution import restart_budget_preflight
from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.retired_generation import (
    apply_retired_generation_plan,
    build_retired_generation_plan,
)
from gpu_fault.workflow_resolution import retired_generation_successor
from tests._builders import (
    active_workflow_executor,
    build_store,
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome

INCIDENT = "inc-a"
RETIRED = "workflow-retired"
CURRENT = "workflow-current"
COMMAND = "command-stop-workloads"
CLUSTER = "cluster-a"
JOB = "training-a"
NODES = ["node-a", "node-b", "node-c"]

# The live shape. STOP_WORKLOADS is the step whose remote command was in flight,
# and RESTART_WORKLOAD is what held the job's restart reservation.
RETIRED_STEPS = [
    workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=NODES),
    workflow_step(WorkflowOperation.STOP_WORKLOADS, node_ids=NODES),
    workflow_step(
        WorkflowOperation.RESTART_WORKLOAD,
        node_ids=NODES,
        parameters={"cluster_id": CLUSTER, "job_id": JOB, "restart_budget": 1},
    ),
]


def scenario(
    store,
    *,
    retired_status=WorkflowStatus.RUNNING,
    retired_token: int = 1,
    incident_token: int = 4,
    successor_predecessor: str | None = None,
    command_status: RemoteCommandStatus | None = RemoteCommandStatus.WAITING,
    steps=None,
    **retired_values,
):
    """One retired generation mid-flight, one successor its incident waits on.

    Defaults are the live 2026-09-04 record: ``RUNNING`` at generation 1 with an
    owner, a live lease, budget claims and one open remote command, under an
    incident that has moved to generation 4. Each case overrides only the field
    whose effect it is describing.
    """

    now = datetime.now(timezone.utc)
    incident = fault_incident(
        INCIDENT,
        "event-a",
        cluster_id=CLUSTER,
        node_ids=list(NODES),
        state=IncidentState.ACTION_PENDING,
        fencing_token=incident_token,
        workflow_request_id=CURRENT,
    )
    store.save_incident(incident)
    values: dict = {
        "execution_owner_id": "executor-b",
        "execution_lease_expires_at": now + timedelta(minutes=3),
        "remediation_budget_claims": [f"{CLUSTER}/stop-workloads"],
        "completed_step_indexes": [0],
        "completed_operations": [WorkflowOperation.FREEZE_EVIDENCE],
    }
    values.update(retired_values)
    retired = workflow_request(
        RETIRED,
        INCIDENT,
        status=retired_status,
        fencing_token=retired_token,
        official_action="RESTART_APP",
        official_steps=RETIRED_STEPS if steps is None else steps,
        **values,
    )
    store.save_workflow(retired)
    current = workflow_request(
        CURRENT,
        INCIDENT,
        fencing_token=incident_token,
        official_action="RUN_DIAGNOSTICS",
        official_steps=[workflow_step(WorkflowOperation.VALIDATE_GPU)],
        predecessor_workflow_id=successor_predecessor,
    )
    store.save_workflow(current)
    if command_status is not None:
        store.ensure_remote_command(
            RemoteActionCommand(
                command_id=COMMAND,
                cluster_id=CLUSTER,
                workflow_request_id=RETIRED,
                incident_id=INCIDENT,
                step_index=1,
                fencing_token=retired_token,
                idempotency_key=f"{RETIRED}/1/STOP_WORKLOADS",
                step=RETIRED_STEPS[1],
                workflow=retired,
                incident=incident,
                status=command_status,
                lease_owner=(
                    "regional-executor-a"
                    if command_status is RemoteCommandStatus.LEASED
                    else None
                ),
                lease_expires_at=(
                    now + timedelta(minutes=1)
                    if command_status is RemoteCommandStatus.LEASED
                    else None
                ),
            )
        )
    return incident, retired, current


def dispatcher(store, adapters=(), operations=frozenset()):
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, adapters, operations),
        WorkflowDispatcherConfig(enabled=True, batch_size=100),
    )


def diagnostic_dispatcher(store):
    adapter = FakeAdapter(
        {WorkflowOperation.VALIDATE_GPU: WorkflowStepOutcome.succeeded()}
    )
    return dispatcher(store, [adapter], [WorkflowOperation.VALIDATE_GPU])


def test_the_open_command_is_cancelled_before_the_workflow_is_closed() -> None:
    """The ordering that keeps a destructive command from outliving its workflow.

    A remote command is a row of its own: closing the workflow around an open one
    leaves a ``STOP_WORKLOADS`` that the next fence evaluation happily releases
    onto three nodes, with no workflow left to attribute it to. So the first tick
    only cancels, and the record stays open and keeps blocking releases until the
    Store itself shows the command settled.
    """

    store = build_store()
    scenario(store)

    dispatcher(store).run_once()

    assert store.get_remote_command(COMMAND).status is RemoteCommandStatus.FAILED
    assert store.get_workflow(RETIRED).status is WorkflowStatus.RUNNING, (
        "the workflow must stay open until the Store shows its command settled"
    )


def test_the_second_tick_revokes_the_settled_record() -> None:
    store = build_store()
    scenario(store)
    sweep = dispatcher(store)

    sweep.run_once()
    sweep.run_once()

    revoked = store.get_workflow(RETIRED)
    assert revoked.status is WorkflowStatus.SUPERSEDED
    assert revoked.preempted_by_workflow_id == CURRENT
    assert revoked.superseded_at is not None
    assert "advanced to generation 4" in (revoked.preemption_reason or "")
    assert revoked.execution_owner_id is None, (
        "revocation has to break the lease the dispatcher keeps renewing"
    )
    assert revoked.execution_lease_expires_at is None
    assert any(
        "revoked retired generation" in reason
        for reason in store.get_incident(INCIDENT).reasons
    ), "the incident is where an operator looks for what happened to the record"


def test_blocked_reasons_are_left_alone_so_the_step_set_does_not_change() -> None:
    """An audit line here would silently repoint every later reader.

    ``blocked_reasons`` is not a log: a non-empty value means later readers --
    including the restart-reservation release below -- switch from
    ``official_steps`` to ``safety_steps``. Writing the audit there would release
    reservations for the wrong step set, so the audit goes on the incident and in
    ``preemption_reason`` instead.
    """

    store = build_store()
    scenario(store)
    sweep = dispatcher(store)

    sweep.run_once()
    sweep.run_once()

    assert store.get_workflow(RETIRED).blocked_reasons == []


def test_withholding_the_retired_record_unstarves_its_successor_at_once() -> None:
    """The half of the damage the release gate never showed.

    ``run_once`` admits one workflow per incident and the retired record won that
    slot on every pass, so the workflow the incident was actually waiting on never
    ran either -- four and a half hours ACTION_PENDING with an undiagnosed fault.
    The successor must get the slot on the *first* tick, while the retired record
    is still waiting for its command to settle.
    """

    store = build_store()
    scenario(store)

    diagnostic_dispatcher(store).run_once()

    assert store.get_workflow(RETIRED).status is WorkflowStatus.RUNNING
    assert store.get_workflow(CURRENT).status is WorkflowStatus.SUCCEEDED, (
        "the successor cannot wait for the retired record to finish settling"
    )


def test_the_sweep_is_not_reported_as_a_dispatch_failure() -> None:
    """Cleanup must not look like a remediation that failed.

    Neither the deferring tick nor the revoking one is an execution, and folding
    either into ``failed`` turns routine cleanup into an alert about a workflow
    that never happened.
    """

    store = build_store()
    scenario(store)
    sweep = diagnostic_dispatcher(store)

    deferred = sweep.run_once()
    revoked = sweep.run_once()

    assert store.get_workflow(RETIRED).status is WorkflowStatus.SUPERSEDED
    assert (deferred.failed, deferred.failures) == (0, [])
    assert (revoked.failed, revoked.failures) == (0, [])


def test_a_restart_reservation_no_adapter_attempted_is_released() -> None:
    """Otherwise the revocation spends the job's restart budget.

    Planning reserves against the job's budget for the ``RESTART_WORKLOAD`` step.
    Terminalizing without releasing it charges the next real restart of the same
    job, and the failure mode is a legitimate remediation refused for lack of
    budget with nothing pointing back to here. Terminal status alone does not do
    it: budget *claims* are only counted for a leased RUNNING workflow, but a
    restart reservation is a durable row.
    """

    store = build_store()
    scenario(store)
    store.reserve_job_restart(
        CLUSTER, JOB, 1, f"{RETIRED}/2/{WorkflowOperation.RESTART_WORKLOAD.value}"
    )
    sweep = dispatcher(store)

    sweep.run_once()
    sweep.run_once()

    assert store.get_workflow(RETIRED).status is WorkflowStatus.SUPERSEDED
    store.reserve_job_restart(CLUSTER, JOB, 1, "a-later-real-restart")


def test_a_completed_destructive_operation_is_refused_before_anything_is_cancelled() -> (
    None
):
    """The load-bearing refusal, and it comes first.

    ``completed_operations`` is what the workflow actually did to the fleet. A
    record that already stopped workloads has left state a later workflow must
    compensate for, and supersession compensates nothing -- so it is not
    revocable at any point, and the release gate goes on reporting it until an
    operator resolves it. The refusal is evaluated against the record alone,
    before the remote-command pass, so a record nobody may close does not get its
    commands cancelled as a side effect.
    """

    store = build_store()
    scenario(
        store,
        completed_step_indexes=[0, 1],
        completed_operations=[
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.STOP_WORKLOADS,
        ],
    )
    sweep = dispatcher(store)

    sweep.run_once()
    sweep.run_once()

    assert store.get_workflow(RETIRED).status is WorkflowStatus.RUNNING
    assert store.get_remote_command(COMMAND).status is RemoteCommandStatus.WAITING, (
        "a record that needs an operator must not have its commands cancelled"
    )


def test_a_leased_command_defers_revocation_until_its_executor_reports() -> None:
    """A cancellation *request* is not a settlement.

    Only the executor holding the lease can report a ``LEASED`` command, and it
    may already be talking to the nodes. Cancelling records the request -- which
    permanently bars re-claiming and forces whatever comes back to ``FAILED`` --
    but the workflow stays open, because until the command lands the effect is
    still in flight. The record keeps blocking releases, which is the correct
    posture for something only an operator can finish.
    """

    store = build_store()
    scenario(store, command_status=RemoteCommandStatus.LEASED)
    sweep = dispatcher(store)

    sweep.run_once()
    sweep.run_once()

    command = store.get_remote_command(COMMAND)
    assert command.status is RemoteCommandStatus.LEASED
    assert command.cancellation_requested_at is not None
    assert store.get_workflow(RETIRED).status is WorkflowStatus.RUNNING


def test_a_step_waiting_on_a_local_adapter_needs_an_operator() -> None:
    """In-flight work no Store row accounts for.

    A ``WAITING`` step execution is the only shape in-flight work takes --
    ``WorkflowStepStatus`` has no "running". When it is remote-backed the
    command's own status is the authority and it can be cancelled. When it is not,
    nothing anywhere says whether the adapter's action landed, so there is nothing
    to cancel and no proof to revoke on.
    """

    store = build_store()
    scenario(
        store,
        command_status=None,
        step_executions=[
            workflow_step_execution(
                1, WorkflowOperation.STOP_WORKLOADS, status=WorkflowStepStatus.WAITING
            )
        ],
    )
    sweep = dispatcher(store)

    sweep.run_once()
    sweep.run_once()

    assert store.get_workflow(RETIRED).status is WorkflowStatus.RUNNING


def test_a_remote_backed_waiting_step_does_not_block_after_its_command_settles() -> (
    None
):
    """The counterpart: a remote ``WAITING`` step is the command's business.

    The live record's ``STOP_WORKLOADS`` step stayed ``WAITING`` for hours with
    ``adapter_operation_id`` naming its command. Reading that as unaccounted work
    would make the record unrevocable forever -- the step execution is never
    rewritten once the command is cancelled -- so the command status has to be the
    only authority on it.
    """

    store = build_store()
    scenario(
        store,
        step_executions=[
            workflow_step_execution(
                1,
                WorkflowOperation.STOP_WORKLOADS,
                status=WorkflowStepStatus.WAITING,
                adapter_operation_id=f"remote/{COMMAND}",
            )
        ],
    )
    sweep = dispatcher(store)

    sweep.run_once()
    sweep.run_once()

    assert store.get_workflow(RETIRED).status is WorkflowStatus.SUPERSEDED


def test_a_successor_that_names_it_as_predecessor_is_left_to_the_executor() -> None:
    """Somebody already owns this record.

    With ``preempt_predecessor`` the executor supersedes it at a step boundary,
    and that is the only path that compensates an unrestored
    ``QUIESCE_GPU_SERVICES``; without it the successor is queued behind this
    record and it is meant to run to completion. Revoking either shape drops real
    work, and withholding either from dispatch deadlocks the pair -- the
    successor waits for a predecessor that is never dispatched again.
    """

    store = build_store()
    _, retired, _ = scenario(store, successor_predecessor=RETIRED)

    assert retired_generation_successor(store, retired) is None

    dispatcher(store).run_once()

    assert store.get_workflow(RETIRED).status is WorkflowStatus.RUNNING
    assert store.get_remote_command(COMMAND).status is RemoteCommandStatus.WAITING


def test_a_retired_generation_is_refused_at_the_fence() -> None:
    """The belt, for the tick before the sweep gets to the record.

    The sweep runs once per dispatch loop and cannot be the only guard: an
    executor that reached ``execute`` with a retired generation would otherwise
    hand its next destructive step to an adapter. Reading a pointer mismatch as
    "a preemption the successor link resolves" is exactly what kept the live
    record dispatchable, so the fence now checks whether that link exists.
    """

    store = build_store()
    scenario(store, execution_owner_id=None, execution_lease_expires_at=None)
    executor = active_workflow_executor(store, [], [])

    with pytest.raises(Exception, match="retired by " + CURRENT):
        execute_workflow(executor, RETIRED, expected_fencing_token=1)


def test_the_fence_still_dispatches_a_workflow_its_successor_names() -> None:
    """The same discrimination, from the dispatch side.

    A linked predecessor has to keep reaching ``execute``: that is where
    ``_supersede_if_safe`` closes it, and where an unrestored quiesce gets
    compensated. Refusing it at the fence would leave the compensation unrun.

    The step set is trimmed to the one step this case is about, so that a
    ``RESTART_WORKLOAD`` refused for its own unrelated reason -- missing restart
    safety context -- cannot be mistaken for the fence refusing.
    """

    store = build_store()
    scenario(
        store,
        successor_predecessor=RETIRED,
        execution_owner_id=None,
        execution_lease_expires_at=None,
        command_status=None,
        steps=[workflow_step(WorkflowOperation.STOP_WORKLOADS, node_ids=NODES)],
        completed_step_indexes=[],
        completed_operations=[],
    )
    adapter = FakeAdapter(
        {WorkflowOperation.STOP_WORKLOADS: WorkflowStepOutcome.succeeded()}
    )
    executor = active_workflow_executor(
        store, [adapter], [WorkflowOperation.STOP_WORKLOADS]
    )

    result = execute_workflow(executor, RETIRED, expected_fencing_token=1)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls, "the linked predecessor still gets to run its next step"


def test_the_store_write_refuses_a_generation_that_moved_under_it() -> None:
    """The compare-and-set, and why it is on the token rather than ``updated_at``.

    A retired generation is still being dispatched, and every dispatch renews its
    lease and stamps the row, so an ``updated_at`` precondition would never hold.
    The token is the value that decides whether the record is behind its incident,
    and every other condition is re-derived inside the transaction -- which is
    what makes a lease-breaking write safe.
    """

    store = build_store()
    scenario(store, command_status=None)

    with pytest.raises(ValueError, match="fencing token changed"):
        store.reconcile_retired_generation_workflow(
            RETIRED,
            CURRENT,
            expected_fencing_token=2,
            reference="operator-reference",
            reconciled_at=datetime.now(timezone.utc),
        )

    assert store.get_workflow(RETIRED).status is WorkflowStatus.RUNNING


def test_the_operator_apply_cancels_then_revokes_in_one_pass() -> None:
    """The operator does not get asked to run it twice.

    The dispatcher can afford a deferring tick because it comes back in eighty
    seconds. The operator is running this to clear a release blocker, so the two
    halves happen in one command -- cancelling a ``PENDING`` or ``WAITING``
    command settles it in the same Store call, and the re-plan in between is what
    proves it settled rather than assuming it.
    """

    store = build_store()
    scenario(store)
    plan = build_retired_generation_plan(store, [RETIRED])
    assert plan["items"][0]["eligible"] is False, (
        "an open command is the blocker this apply is expected to clear itself"
    )
    assert plan["items"][0]["cancellable"] is True

    result = apply_retired_generation_plan(
        store,
        workflow_ids=[RETIRED],
        expected_plan_sha256=plan["plan_sha256"],
        reference="pre-deploy-6459c07ea279",
    )

    assert result["applied_workflow_ids"] == [RETIRED]
    assert result["records_deleted"] == 0
    assert result["cancelled_remote_commands"][RETIRED]["cancelled"] == 1
    assert result["restart_reservation_warnings"] == []
    assert store.get_workflow(RETIRED).status is WorkflowStatus.SUPERSEDED
    assert store.get_remote_command(COMMAND).status is RemoteCommandStatus.FAILED


def test_the_operator_apply_refuses_a_digest_that_no_longer_describes_the_store() -> (
    None
):
    """The review has to bind the write, not merely precede it.

    The record is being dispatched every eighty seconds while the operator reads
    the plan, so between plan and apply the incident can advance again or a step
    can be handed to an adapter. Re-deriving the plan and comparing digests is
    what turns "I looked at this" into "this is what I looked at".
    """

    store = build_store()
    scenario(store)

    with pytest.raises(ValueError, match="plan changed before apply"):
        apply_retired_generation_plan(
            store,
            workflow_ids=[RETIRED],
            expected_plan_sha256="f" * 64,
            reference="pre-deploy-6459c07ea279",
        )

    assert store.get_workflow(RETIRED).status is WorkflowStatus.RUNNING
    assert store.get_remote_command(COMMAND).status is RemoteCommandStatus.WAITING, (
        "a refused apply must not have cancelled anything"
    )


def test_a_reservation_the_deployed_image_cannot_release_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-deploy run happens against an image this file cannot assume.

    Releasing the restart reservation reaches into ``gpu_fault.execution``, and
    this apply exists precisely to run against the image deployed *before* the
    release that carries it. If that reach fails it fails after the workflow row
    is already terminalized, so raising would hand the operator a traceback for a
    write that succeeded -- at the one moment they have no other way to clear the
    release blocker and no way to tell whether to run the command again.

    So the shortfall is named in the result instead. What it costs is bounded: the
    job's restart budget stays charged, and the operator can release it with the
    release that carries the fix rather than meet it later as an unexplained
    refusal.
    """

    store = build_store()
    scenario(store)
    store.reserve_job_restart(
        CLUSTER, JOB, 1, f"{RETIRED}/2/{WorkflowOperation.RESTART_WORKLOAD.value}"
    )
    monkeypatch.delattr(
        restart_budget_preflight, "release_unattempted_restart_reservations"
    )
    plan = build_retired_generation_plan(store, [RETIRED])

    result = apply_retired_generation_plan(
        store,
        workflow_ids=[RETIRED],
        expected_plan_sha256=plan["plan_sha256"],
        reference="pre-deploy-6459c07ea279",
    )

    assert store.get_workflow(RETIRED).status is WorkflowStatus.SUPERSEDED
    assert result["applied_workflow_ids"] == [RETIRED]
    warnings = result["restart_reservation_warnings"]
    assert len(warnings) == 1
    assert RETIRED in warnings[0]
    assert "restart reservations were not released" in warnings[0]


def test_the_operator_write_records_its_reference() -> None:
    store = build_store()
    scenario(store, command_status=None)

    revoked, incident = store.reconcile_retired_generation_workflow(
        RETIRED,
        CURRENT,
        expected_fencing_token=1,
        reference="pre-deploy-6459c07ea279",
        reconciled_at=datetime.now(timezone.utc),
    )

    assert revoked.status is WorkflowStatus.SUPERSEDED
    assert "operator reconciliation pre-deploy-6459c07ea279" in (
        revoked.preemption_reason or ""
    )
    assert any("pre-deploy-6459c07ea279" in reason for reason in incident.reasons), (
        "the incident is where the next reader learns the record was closed on "
        f"purpose; reasons were {incident.reasons}"
    )
