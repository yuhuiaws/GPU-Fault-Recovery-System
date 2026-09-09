"""Control-plane review 2026-09-08: the fast paths that were still poison.

C-04: ``coordinator.ingest`` and ``escalation.escalate`` read the workflow
behind an unchecked ``incident.workflow_request_id``; a dangling pointer made
every re-post of the event (the retry path) raise ``NotFoundError`` out of the
handler. Both now fall through to the build path -- the same idempotent repair
the store's ``_duplicate_event_records`` performs -- and count the repair.

C-09: the no-baseline generation fence wrote ``save_incident`` and then
``link_event_to_incident`` as two autocommit statements; ``save_incident``
already links ``incident.event_id`` in the same transaction, and the fenced
incident *is* built from the event, so the second write was a second
transaction for nothing.

C-10: ``ingest_distributed_xids`` read the runtime profile unguarded;
``NotFoundError`` left the route as a 500 and the data plane isolated the
batch. A missing profile is a BLOCKED(NEEDS_OPERATOR) plan, like every other
family compiles it.

C-12: a node-health trial (``_persist=False``) inside the attempt-group lock
must not write; the inventory-coverage branch linked the event anyway.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.watcher import WorkloadPhase
from tests._builders import (
    asgi_client,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.orchestration._cross_fault_support import NOW as FENCE_NOW
from tests.orchestration._cross_fault_support import (
    observation,
    post_faults,
    xid_payload,
)
from tests.orchestration.test_ingest_fast_path_guards import (
    _distributed_batch,
    _running_reboot,
)

from ._support import (
    WorkloadState,
    _inventory_mismatch_finding,
    _save_attempt,
    event,
    ingest,
)

# ------------------------------------------------------------------ C-04


def test_a_re_posted_xid_whose_pointer_dangles_is_rebuilt_not_raised(context):
    first = event(48, event_id="xid48-dangling")
    _, incident, workflow = ingest(context, first)
    assert workflow is not None
    context.store.save_incident(
        incident.model_copy(update={"workflow_request_id": "workflow-vanished"})
    )
    repairs_before = context.store.stale_event_link_repairs

    _, again_incident, again_workflow = ingest(context, first)

    assert again_workflow is not None
    assert again_incident.workflow_request_id == again_workflow.request_id
    assert context.store.get_workflow(again_workflow.request_id) == again_workflow
    assert context.store.get_incident_by_event(first.event_id).incident_id == (
        again_incident.incident_id
    )
    assert context.store.stale_event_link_repairs > repairs_before


def _failed_escalatable(store):
    """A FAILED isolation the escalation family answers with a support plan."""

    from gpu_fault.app import default_simulated_profile

    store.save_profile(default_simulated_profile())
    alias = "acceptance-alias-0123456789ab"
    incident = fault_incident(
        "inc-src",
        "event-src",
        state=IncidentState.ESCALATED,
        effective_action=RecoveryAction.ESCALATE_OPERATOR,
        workflow_request_id="wf-src",
        node_ids=[alias],
        policy_source="SITE_SAFETY",
        fencing_token=1,
    )
    executions = [
        workflow_step_execution(0, WorkflowOperation.FREEZE_EVIDENCE),
        workflow_step_execution(
            1,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowStepStatus.FAILED,
            error=f"node {alias} is absent; isolation cannot be observed",
            details={"safety_rejection": True, "node_id": alias, "absent": True},
        ),
    ]
    workflow = workflow_request(
        "wf-src",
        "inc-src",
        status=WorkflowStatus.FAILED,
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_steps=[
            workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=[alias]),
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=[alias]),
            workflow_step(WorkflowOperation.QUARANTINE, node_ids=[alias]),
        ],
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.FREEZE_EVIDENCE],
        step_executions=executions,
    )
    store.save_incident_and_workflow(incident, workflow)
    return workflow


class _StubBuilder:
    def compile_steps(
        self, operations, profile, node_ids, gpu_uuids, workload_ids=None
    ):
        return (
            [
                workflow_step(
                    operation,
                    node_ids=list(node_ids),
                    gpu_uuids=list(gpu_uuids),
                    workload_ids=list(workload_ids or []),
                )
                for operation in operations
            ],
            [],
        )


def test_an_escalation_whose_pointer_dangles_is_emitted_again_not_raised():
    store = build_store()
    workflow = _failed_escalatable(store)
    service = HardwareEscalationService(store, _StubBuilder())
    first = service.escalate(workflow)
    assert first is not None
    escalated, _ = first
    store.save_incident(
        escalated.model_copy(update={"workflow_request_id": "workflow-vanished"})
    )

    second = service.escalate(workflow)

    assert second is not None
    again_incident, support = second
    assert again_incident.incident_id == escalated.incident_id
    assert again_incident.workflow_request_id == support.request_id
    assert store.get_workflow(support.request_id) == support
    assert store.stale_event_link_repairs >= 1
    assert len(store.list_workflows()) == 2, "no third record"


# ------------------------------------------------------------------ C-09


def test_the_no_baseline_fence_writes_its_incident_and_link_in_one_call(monkeypatch):
    from tests._builders import build_context

    context = build_context()
    old = observation(started_at=FENCE_NOW - timedelta(minutes=1))
    context.store.save_attempt_observation(
        old.model_copy(
            update={
                "workload_phase": WorkloadPhase.STOPPED,
                "observed_at": FENCE_NOW + timedelta(seconds=10),
            }
        )
    )
    context.store.save_attempt_observation(
        observation(
            attempt_id="train-a002",
            started_at=FENCE_NOW + timedelta(seconds=60),
            observed_at=FENCE_NOW + timedelta(seconds=70),
        )
    )
    calls: list[str] = []
    original_save = context.store.save_incident
    original_link = context.store.link_event_to_incident

    def save_incident(*args, **kwargs):
        calls.append("save_incident")
        return original_save(*args, **kwargs)

    def link_event_to_incident(*args, **kwargs):
        calls.append("link_event_to_incident")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(context.store, "save_incident", save_incident)
    monkeypatch.setattr(context.store, "link_event_to_incident", link_event_to_incident)
    payload = xid_payload(11, "stale-fence-one-write")
    payload.update(
        {
            "observed_at": (FENCE_NOW + timedelta(seconds=40)).isoformat(),
            "collected_at": (FENCE_NOW + timedelta(seconds=69)).isoformat(),
        }
    )

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))[0]

    incident = context.store.get_incident(result["incident_id"])
    assert incident.state is IncidentState.RECOVERED
    assert context.store.get_incident_by_event("stale-fence-one-write") == incident
    assert calls == ["save_incident"], calls


# ------------------------------------------------------------------ C-10


def test_a_distributed_batch_with_an_unreleased_profile_is_blocked_not_500():
    context = ApplicationContext()
    batch = _distributed_batch()
    for item in batch["events"]:
        item["runtime_profile_version"] = "not-released-v9"

    async def scenario() -> dict:
        async with asgi_client(context) as client:
            response = await client.post("/v1/gpu-events/xid/distributed", json=batch)
        assert response.status_code == 200, response.text
        return response.json()

    result = asyncio.run(scenario())

    workflow = context.store.get_workflow(result["workflow"]["request_id"])
    assert workflow.status is WorkflowStatus.BLOCKED
    assert workflow.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert any("runtime profile does not exist" in r for r in workflow.blocked_reasons)
    incident = context.store.get_incident(result["incident"]["incident_id"])
    assert incident.state is IncidentState.ESCALATED
    assert incident.workflow_request_id == workflow.request_id
    for item in batch["events"]:
        linked = context.store.get_incident_by_event(item["event_id"])
        assert linked is not None and linked.incident_id == incident.incident_id


# ------------------------------------------------------------------ C-12


def test_a_node_health_trial_run_writes_no_event_link(context):
    _running_reboot(context.store)
    finding = _inventory_mismatch_finding(event_id="inventory-trial").model_copy(
        update={"node_id": "node-a"}
    )

    incident, workflow = context.orchestrator.ingest_node_health(
        finding, _persist=False
    )

    assert workflow is not None and workflow.request_id == "wf-run", (
        "the trial still reports the covering incumbent"
    )
    assert context.store.get_incident_by_event("inventory-trial") is None, (
        "a trial (_persist=False) must not write"
    )


def test_the_attempt_group_trial_skips_the_node_resource_merge(context, monkeypatch):
    recorded: list[dict] = []
    original = context.orchestrator.ingest_node_health

    def recording(finding, **kwargs):
        recorded.append(dict(kwargs))
        return original(finding, **kwargs)

    monkeypatch.setattr(context.orchestrator, "ingest_node_health", recording)
    _save_attempt(context, ("node-a",))
    finding = _inventory_mismatch_finding(
        event_id="efa-grouped-trial",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    ).model_copy(
        update={
            "metric_name": "efa_inventory_mismatch",
            "recommended_action": RecoveryAction.REBOOT_NODE,
            "diagnostic_parameters": {"expected_count": "16"},
        }
    )

    _, workflow = context.orchestrator.ingest_node_health(finding)

    assert workflow is not None
    trials = [call for call in recorded if call.get("_persist") is False]
    assert trials, recorded
    assert all(call.get("_skip_node_resource_merge") is True for call in trials), trials
