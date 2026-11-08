"""A re-armed finding the operator hand-off absorbs sends no advisory of its
own (b277).

Live 2026-11-08 on R28 (HyperPod, GF-GENERAL-LIVE-013
post_repair_isolation_lifted): the EFA driver chain's VALIDATE_FABRIC failed,
its reboot rung was refused for want of observed isolation and the refusal
was handed over (support-after at ``containment_or_release``: the isolation
put back, one ticket). The chain's TRIGGER_HEALTH_SNAPSHOT had restarted the
host collector, whose first tick read 0 and the next 1: a re-arm under a new
event id. b275 (``families/health.py`` ``_absorb_into_open_handoff``) linked
it to the ESCALATED hand-off that holds the node -- one incident lineage, one
ticket -- but ingestion still mailed the finding's own
``hardware-inventory-mismatch-zh-v1`` advisory: ``app/ingest/node_health.py``
mailed every inventory finding whichever incident it landed on, and the
template deduplicates per event id, so every re-arm was one more mail naming
the hand-off's incident and its ESCALATE_SUPPORT plan, about the fault that
ticket already holds. Nothing had been decided and nothing was to be done.

Now the absorbed finding's advisory folds into the hand-off: no mail, and the
hand-off's record carries the back-reference (the event link, the finding's
marker and b275's bounded reason naming the event). A genuinely new fault --
another signal, failure mode or node -- and a re-arm once the hand-off no
longer holds the node (released, or closed by the operator) still mail.

Everything goes through the product's ingestion: host telemetry posted to the
control-plane API over an in-memory store, as the host collector posts it.
The only scripted part is the executor's progress through the chain (as in
``test_handoff_signal_absorb``); the escalations are the product's own.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import (
    AdvisoryNotification,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.notifications.common import HARDWARE_INVENTORY_TEMPLATE_VERSION
from gpu_fault.orchestration.failure_outcome import (
    ISOLATION_NOT_ISOLATED,
    LIFECYCLE_ISOLATION_DETAIL,
)
from gpu_fault.orchestration.families.health import handoff_holds_node
from tests._builders import copy_model, host_telemetry_batch, workflow_step_execution
from tests.host_health._support import RecordingNotifier
from tests.host_health.test_grouped_node_health_incidents import HOST_TELEMETRY, post

OP = WorkflowOperation
NODE = "hyperpod-node-a"
#: R28's live host batches (the collector's ``host-<node>-<observed us>``) of
#: the first finding and its re-arm, 33 s apart; the clear between them is the
#: tick 15 s before the re-arm (its batch id is not in the case evidence).
FIRST_BATCH = 1794149968100446
CLEAR_BATCH = 1794149986312201
REARM_BATCH = 1794150001399506
NOW = datetime.now(timezone.utc)
EFA_LABELS = {
    "resource": "EFA",
    "expected_count": "16",
    "observed_count": "15",
    "discovered_count": "16",
    "driver_bound_count": "15",
    "missing_count": "1",
    "excess_count": "0",
    "consecutive_mismatch_samples": "2",
    "required_consecutive_samples": "2",
    "node_instance_type": "ml.p5en.48xlarge",
    "expected_gpu_count": "8",
    "expected_efa_device_count": "16",
    "failure_mode": "DRIVER_UNBOUND",
}
PCI_MISSING = {
    **EFA_LABELS,
    "discovered_count": "15",
    "failure_mode": "PCI_DEVICE_MISSING",
}
GPU_LOST = {
    **EFA_LABELS,
    "resource": "GPU",
    "expected_count": "8",
    "observed_count": "7",
    "discovered_count": "7",
    "failure_mode": "PCI_DEVICE_MISSING",
}


def reading(
    batch: int,
    seconds: float,
    value: int,
    *,
    node: str = NODE,
    metric: str = "efa_inventory_mismatch",
    labels: dict[str, str] = EFA_LABELS,
) -> HostTelemetryBatch:
    return host_telemetry_batch(
        f"host-{node}-{batch}",
        NOW + timedelta(seconds=seconds),
        [HostMetricSample(name=metric, value=value, labels=dict(labels))],
        node_id=node,
        workload_state="IDLE",
        runtime_profile_version="simulated-v1",
    )


def ingest(context: ApplicationContext, batch: HostTelemetryBatch) -> dict:
    status, body = post(context, HOST_TELEMETRY, batch.model_dump(mode="json"))
    assert status == 200, body
    return body


def event_id(batch: HostTelemetryBatch) -> str:
    return f"{batch.batch_id}-{batch.samples[0].name}-node"


def inventory_mails(
    notifier: RecordingNotifier, event: str
) -> list[AdvisoryNotification]:
    """The ``hardware-inventory-mismatch-zh-v1`` mails sent for ``event``."""

    key = f"{event}/hardware-inventory/{HARDWARE_INVENTORY_TEMPLATE_VERSION}"
    return [item for item in notifier.notifications if item.deduplication_key == key]


def advisories(context: ApplicationContext, event: str) -> list[AdvisoryNotification]:
    """Every outbox row that names ``event``: none for a folded finding."""

    return [
        item
        for item in context.store.list_notifications()
        if event in item.deduplication_key or f"Event ID: {event}\n" in item.body_text
    ]


def _ran(
    workflow: WorkflowRequest,
    failed: OP | None = None,
    *,
    error: str = "",
    details: dict | None = None,
) -> list:
    executions = []
    for index, step in enumerate(workflow.official_steps):
        if step.operation is failed:
            executions.append(
                workflow_step_execution(
                    index,
                    step.operation,
                    WorkflowStepStatus.FAILED,
                    error=error,
                    details=details or {},
                )
            )
            break
        executions.append(workflow_step_execution(index, step.operation))
    return executions


def _finish(
    context: ApplicationContext,
    workflow: WorkflowRequest,
    executions: list,
    status: WorkflowStatus,
    state: IncidentState,
) -> WorkflowRequest:
    """The record as the executor terminalizes it, its incident in ``state``."""

    stored = context.store.get_workflow(workflow.request_id)
    succeeded = [
        item for item in executions if item.status is WorkflowStepStatus.SUCCEEDED
    ]
    context.store.save_workflow(
        copy_model(
            stored,
            status=status,
            step_executions=executions,
            completed_step_indexes=[item.step_index for item in succeeded],
            completed_operations=[item.operation for item in succeeded],
        ),
        expected=stored,
    )
    incident = context.store.get_incident(stored.incident_id)
    context.store.save_incident(copy_model(incident, state=state), expected=incident)
    return context.store.get_workflow(workflow.request_id)


def handed_over(context: ApplicationContext) -> WorkflowRequest:
    """R28's chain up to its one hand-off: the first finding's EFA driver
    chain fails VALIDATE_FABRIC, its reboot rung is refused for want of
    isolation and the refusal's support-after holds the node with a ticket."""

    first = ingest(context, reading(FIRST_BATCH, 0, 1))
    [request_id] = first["workflow_request_ids"]
    chain = context.store.get_workflow(request_id)
    assert OP.REMEDIATE_EFA_DRIVER in {step.operation for step in chain.official_steps}
    failed = _finish(
        context,
        chain,
        _ran(
            chain,
            OP.VALIDATE_FABRIC,
            error=f"validation failed on nodes: {NODE} "
            "(efa_inventory_active_count=15,expected=16)",
            details={
                "failed_nodes": [NODE],
                "node_failures": {NODE: ["efa_inventory_active_count=15,expected=16"]},
                "validation": "VALIDATE_FABRIC",
            },
        ),
        WorkflowStatus.FAILED,
        IncidentState.QUARANTINED,
    )
    rung = context.orchestrator.escalate_failed_hardware_remediation(failed)
    assert rung is not None
    refused = _finish(
        context,
        rung[1],
        _ran(
            rung[1],
            OP.RESTART_NODE,
            error=f"node {NODE} is not isolated: spec.unschedulable is not true",
            details={
                "safety_rejection": True,
                LIFECYCLE_ISOLATION_DETAIL: ISOLATION_NOT_ISOLATED,
                "node_failures": {
                    NODE: [
                        "RESTART_NODE not submitted, isolation not observed: "
                        "spec.unschedulable is not true"
                    ]
                },
            },
        ),
        WorkflowStatus.FAILED,
        IncidentState.QUARANTINED,
    )
    hand_off = context.orchestrator.escalate_failed_hardware_remediation(refused)
    assert hand_off is not None
    support = hand_off[1]
    assert support.request_id == (
        f"workflow-support-after-workflow-reboot-after-{request_id}"
    )
    assert [step.operation for step in support.official_steps] == [
        OP.FREEZE_EVIDENCE,
        OP.MARK_UNSCHEDULABLE,
        OP.QUARANTINE,
        OP.ESCALATE_SUPPORT,
    ]
    support = _finish(
        context,
        support,
        _ran(support),
        WorkflowStatus.SUCCEEDED,
        IncidentState.ESCALATED,
    )
    # The collector restart's first tick: the signal clears.
    assert ingest(context, reading(CLEAR_BATCH, 18, 0))["findings"] == []
    return support


@pytest.fixture
def notifier() -> RecordingNotifier:
    return RecordingNotifier()


@pytest.fixture
def context(notifier: RecordingNotifier) -> ApplicationContext:
    return ApplicationContext(notification_notifier=notifier)


def test_the_absorbed_re_arm_is_recorded_on_the_hand_off_and_mails_nothing(
    context: ApplicationContext, notifier: RecordingNotifier
) -> None:
    support = handed_over(context)
    first = event_id(reading(FIRST_BATCH, 0, 1))
    assert len(inventory_mails(notifier, first)) == 1, "the fault's own advisory"
    rearm_batch = reading(REARM_BATCH, 33, 1)
    rearm = event_id(rearm_batch)
    mails = len(notifier.notifications)

    body = ingest(context, rearm_batch)
    replay = ingest(context, rearm_batch)

    assert [item["event_id"] for item in body["findings"]] == [rearm]
    assert body["incident_ids"] == [support.incident_id], "b275: one lineage"
    assert body["notification_ids"] == [] and replay["notification_ids"] == []
    assert inventory_mails(notifier, rearm) == [], "no advisory per re-arm"
    assert advisories(context, rearm) == []
    assert len(notifier.notifications) == mails, "nothing else was mailed either"
    # The back-reference the folded advisory leaves on the hand-off's record.
    handoff = context.store.get_incident(support.incident_id)
    assert handoff.state is IncidentState.ESCALATED, "the operator still owns it"
    assert [line for line in handoff.reasons if rearm in line] == [
        f"absorbed efa_inventory_mismatch finding {rearm} on node: "
        "recorded only, the operator hand-off holds the node for this signal"
    ]
    linked = context.store.get_incident_by_event(rearm)
    assert linked is not None and linked.incident_id == support.incident_id
    assert [
        marker.incident_id
        for marker in context.store.list_markers_for_incident(support.incident_id)
        if marker.marker_id == f"marker-{rearm}"
    ] == [support.incident_id]
    assert handoff_holds_node(context.store.get_workflow(support.request_id), NODE), (
        "the hand-off keeps the node cordoned and quarantined"
    )


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param(reading(REARM_BATCH, 33, 1, node="hyperpod-node-b"), id="node"),
        pytest.param(reading(REARM_BATCH, 33, 1, labels=PCI_MISSING), id="mode"),
        pytest.param(
            reading(
                REARM_BATCH, 33, 1, metric="gpu_inventory_mismatch", labels=GPU_LOST
            ),
            id="signal",
        ),
    ],
)
def test_a_genuinely_new_fault_still_mails(
    context: ApplicationContext, notifier: RecordingNotifier, fault: HostTelemetryBatch
) -> None:
    support = handed_over(context)
    event = event_id(fault)

    body = ingest(context, fault)

    assert [item["event_id"] for item in body["findings"]] == [event]
    assert body["incident_ids"] != [support.incident_id]
    [mail] = inventory_mails(notifier, event)
    assert mail.notification_id in body["notification_ids"]
    assert not any(
        event in line
        for line in context.store.get_incident(support.incident_id).reasons
    ), "the hand-off records nothing of another fault"


def _release(context: ApplicationContext, support: WorkflowRequest) -> None:
    """The hand-off's RESTORE_SCHEDULING completed: it no longer holds the node."""

    stored = context.store.get_workflow(support.request_id)
    restore = copy_model(stored.official_steps[1], operation=OP.RESTORE_SCHEDULING)
    context.store.save_workflow(
        copy_model(
            stored,
            official_steps=[*stored.official_steps, restore],
            step_executions=[
                *stored.step_executions,
                workflow_step_execution(4, OP.RESTORE_SCHEDULING),
            ],
        ),
        expected=stored,
    )
    assert not handoff_holds_node(
        context.store.get_workflow(support.request_id), NODE
    ), "a completed RESTORE_SCHEDULING released the node"


def _close(context: ApplicationContext, support: WorkflowRequest) -> None:
    """The operator closed the hand-off."""

    handed = context.store.get_incident(support.incident_id)
    context.store.save_incident(
        copy_model(handed, state=IncidentState.RECOVERED), expected=handed
    )


@pytest.mark.parametrize("ended", [_release, _close], ids=["released", "closed"])
def test_a_re_arm_after_the_hand_off_stopped_holding_the_node_still_mails(
    context: ApplicationContext, notifier: RecordingNotifier, ended
) -> None:
    support = handed_over(context)
    ended(context, support)
    rearm_batch = reading(REARM_BATCH, 33, 1)
    rearm = event_id(rearm_batch)

    body = ingest(context, rearm_batch)

    assert body["incident_ids"] == [f"inc-{rearm}"], "the re-arm opens anew"
    [mail] = inventory_mails(notifier, rearm)
    assert mail.incident_id == f"inc-{rearm}"
    assert mail.notification_id in body["notification_ids"]
