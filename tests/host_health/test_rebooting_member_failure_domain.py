"""A shared-storage finding on a node the product is rebooting (b215).

While a workflow of the product reboots a node (RESTART_NODE dispatched, the
RESTORE_SCHEDULING behind it not yet done), every finding of that node is
absorbed into the reboot's record (``orchestration/reboot_window.py``): the
event is linked and the marker re-pointed, the record is not re-labelled. A
reboot of a lost GPU is a ``NODE_HEALTH`` record (``gpu_inventory_mismatch`` ->
REBOOT_NODE), an XID's is ``GPU_FAULT_GROUP`` or, outside the grouping, the
event family's own label (``XID``). The shared-storage failure domain counted
such a node for the node's own look only: a peer losing the same export after
it looked for its peers' per-node incidents among records minted by the
activation and grouped records, and the reboot's ``NODE_HEALTH``/``XID``
record is neither -- a two-node outage of one export formed no record when the
rebooting node reported first. The back-reference a member's record gets said
"the diagnostic-only handling of this node is unchanged" on every record,
the reboot's or an attempt recovery's included.

Everything goes through the product's ingestion (host telemetry and XIDs posted
to the control-plane API over an in-memory store); the executor's progress on
the reboot is the one scripted step.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
    NodeHealthFinding,
)
from gpu_fault.models import FaultIncident, WorkflowOperation, WorkflowRequest
from gpu_fault.shared_storage_failure_domain import (
    SHARED_STORAGE_FAILURE_DOMAIN_NOTIFICATION_CATEGORY,
)
from gpu_fault.store import InMemoryStore
from tests._builders import copy_model, host_telemetry_batch
from tests.host_health._support import RecordingNotifier
from tests.host_health.test_grouped_node_health_incidents import (
    EXPORT,
    HOST_TELEMETRY,
    assert_member,
    at,
    domains,
    executed,
    ingest,
    post,
    routed,
    running_attempt,
    shared,
    xid_recovery,
)
from tests.host_health.test_shared_storage_failure_domain import InFlightPeerStore

DIAGNOSTIC_ONLY = "the diagnostic-only handling of this node is unchanged"
REBOOT_PREFIX = (
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.MARK_UNSCHEDULABLE,
    WorkflowOperation.QUARANTINE,
    WorkflowOperation.RESTART_NODE,
)


def gpu_lost(batch_id: str, node: str, seconds: float) -> HostTelemetryBatch:
    """The host collector's GPU inventory reading: one of eight GPUs gone."""

    return host_telemetry_batch(
        batch_id,
        at(seconds),
        [
            HostMetricSample(
                name="gpu_inventory_mismatch",
                value=1,
                labels={
                    "resource": "GPU",
                    "expected_count": "8",
                    "observed_count": "7",
                    "discovered_count": "7",
                    "missing_count": "1",
                    "excess_count": "0",
                    "consecutive_mismatch_samples": "2",
                    "required_consecutive_samples": "2",
                    "node_instance_type": "ml.p5en.48xlarge",
                    "expected_gpu_count": "8",
                    "expected_efa_device_count": "16",
                    "failure_mode": "PCI_DEVICE_MISSING",
                },
            )
        ],
        node_id=node,
        runtime_profile_version="simulated-v1",
    )


def rebooting(context: ApplicationContext, node: str = "node-a") -> FaultIncident:
    """The product's own reboot of ``node``: a lost GPU's REBOOT_NODE record
    (``NODE_HEALTH``) whose RESTART_NODE the executor has run, the validation
    tail and RESTORE_SCHEDULING still ahead."""

    [event_id] = ingest(context, gpu_lost("gpu-lost", node, -120))
    record = routed(context, event_id)
    assert (record.event_type, record.effective_action.value) == (
        "NODE_HEALTH",
        "REBOOT_NODE",
    ), record
    workflow = context.store.get_workflow(record.workflow_request_id)
    assert [step.operation for step in workflow.official_steps][:4] == list(
        REBOOT_PREFIX
    ), workflow.official_steps
    executed(context, record, *REBOOT_PREFIX)
    return record


def relabelled(context: ApplicationContext, record: FaultIncident, label: str) -> None:
    """The same reboot under another record label: the label a fault family
    gives a record of its own (``XID`` outside the attempt grouping) or a
    grouping's. A later route keeps the record's id and event id."""

    if label != record.event_type:
        stored = context.store.get_incident(record.incident_id)
        context.store.save_incident(copy_model(stored, event_type=label))


def plan(context: ApplicationContext, record: FaultIncident) -> dict[str, Any]:
    """What the executor runs for ``record``: steps, progress, status."""

    workflow: WorkflowRequest = context.store.get_workflow(record.workflow_request_id)
    return workflow.model_dump(
        include={
            "request_id",
            "status",
            "official_steps",
            "completed_step_indexes",
            "step_executions",
            "fencing_token",
            "not_before",
        }
    )


def back_references(
    context: ApplicationContext, incident_id: str, domain: FaultIncident
) -> list[str]:
    return [
        reason
        for reason in context.store.get_incident(incident_id).reasons
        if reason.startswith(f"common failure domain {domain.incident_id}:")
    ]


def domain_mails(notifier: RecordingNotifier) -> list[Any]:
    return [
        item
        for item in notifier.notifications
        if item.category == SHARED_STORAGE_FAILURE_DOMAIN_NOTIFICATION_CATEGORY
    ]


# --------------------------------------------------------------------------- #
# the rebooting node is a member for the peer that reports after it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", ["NODE_HEALTH", "XID", "GPU_FAULT_GROUP"])
def test_a_node_the_product_reboots_is_a_member_for_the_peer_after_it(
    label: str,
) -> None:
    # node-a lost a GPU and the product is rebooting it; the export goes away
    # under node-a first (its finding joins the reboot) and under idle node-b
    # a tick later.
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    reboot = rebooting(context)
    relabelled(context, reboot, label)
    before = plan(context, reboot)

    [absorbed_event] = ingest(context, shared("a-1", "node-a", 1))
    assert routed(context, absorbed_event).incident_id == reboot.incident_id, (
        "node-a's finding is the reboot's evidence: no record of its own"
    )
    assert domains(context) == [], "node-a alone is no failure domain"
    [peer_event] = ingest(context, shared("b-1", "node-b", 2))

    [domain] = domains(context)
    assert domain.node_ids == ["node-a", "node-b"]
    assert_member(context, domain, "node-a", reboot.incident_id)
    assert_member(context, domain, "node-b", routed(context, peer_event).incident_id)
    assert (
        len([item for item in domain.reasons if item.startswith("member node-a:")]) == 1
    ), "node-a is one member, named once"
    assert len(back_references(context, reboot.incident_id, domain)) == 1
    [mail] = domain_mails(notifier)
    assert mail.incident_id == domain.incident_id
    assert f"node-a: {reboot.incident_id}" in mail.body_text
    # The correlation is a record: the reboot keeps its plan, its progress and
    # its place; node-a gets no chain of its own beside it.
    assert plan(context, reboot) == before, "the reboot is not touched"
    assert {
        workflow.incident_id
        for workflow in context.store.list_workflows()
        if "node-a"
        in {node for step in workflow.official_steps for node in step.node_ids}
    } == {reboot.incident_id}, "no second workflow for the rebooting node"


def test_the_look_after_the_commit_meets_a_rebooting_member_in_flight() -> None:
    # b207's two looks: node-b's look inside its own transaction cannot see
    # node-a's absorbed activation (READ COMMITTED), the look after its commit
    # does -- through the reboot record's marker.
    notifier = RecordingNotifier()
    store = InFlightPeerStore(InMemoryStore(), hidden="node-a")
    context = ApplicationContext(notification_notifier=notifier, store=store)
    reboot = rebooting(context)
    [absorbed_event] = ingest(context, shared("a-1", "node-a", 1))
    assert routed(context, absorbed_event).incident_id == reboot.incident_id

    status, second = post(
        context, HOST_TELEMETRY, shared("b-1", "node-b", 3).model_dump(mode="json")
    )

    assert status == 200, second
    [domain] = domains(context)
    assert domain.node_ids == ["node-a", "node-b"]
    assert_member(context, domain, "node-a", reboot.incident_id)
    assert len(second["notification_ids"]) == 1, "the later commit's look mails once"


def test_the_safety_net_names_the_reboot_a_member_activation_sits_in() -> None:
    # node-a's activation carried no export identity and joined the reboot;
    # node-b's carries one but finds no peer. node-a's next failing sample
    # carries the identity: the per-batch safety net makes node-a a member --
    # of the reboot's record, not "none open".
    context = ApplicationContext(notification_notifier=RecordingNotifier())
    reboot = rebooting(context)
    [absorbed_event] = ingest(context, shared("a-1", "node-a", 1, labels={}))
    assert routed(context, absorbed_event).incident_id == reboot.incident_id
    [peer_event] = ingest(context, shared("b-1", "node-b", 2))
    assert domains(context) == [], "an activation without identity correlates nothing"

    assert ingest(context, shared("a-2", "node-a", 17)) == [], "no new activation"

    [domain] = domains(context)
    assert domain.node_ids == ["node-a", "node-b"]
    assert_member(context, domain, "node-a", reboot.incident_id)
    assert_member(context, domain, "node-b", routed(context, peer_event).incident_id)
    assert not any("none open" in reason for reason in domain.reasons), domain.reasons


def test_the_rebooting_nodes_own_late_look_adds_no_record_and_no_mail() -> None:
    # The look after node-a's commit runs again once the record exists (a
    # replay, the after-commit look of a batch in flight): the same member set
    # derives the same record, no second record, no second mail, no second
    # back-reference on the reboot.
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    reboot = rebooting(context)
    batch = shared("a-1", "node-a", 1)
    status, first = post(context, HOST_TELEMETRY, batch.model_dump(mode="json"))
    assert status == 200, first
    ingest(context, shared("b-1", "node-b", 2))
    [domain] = domains(context)

    late = context.shared_storage_domains.observe_committed(
        batch, [NodeHealthFinding.model_validate(item) for item in first["findings"]]
    )

    assert late == []
    assert domains(context) == [domain]
    assert len(back_references(context, reboot.incident_id, domain)) == 1
    assert len(domain_mails(notifier)) == 1


# --------------------------------------------------------------------------- #
# the back-reference says what the record it sits on does
# --------------------------------------------------------------------------- #
def test_the_back_reference_on_a_reboot_names_the_reboot_not_a_diagnostic() -> None:
    context = ApplicationContext(notification_notifier=RecordingNotifier())
    reboot = rebooting(context)
    ingest(context, shared("a-1", "node-a", 1))
    [peer_event] = ingest(context, shared("b-1", "node-b", 2))
    [domain] = domains(context)

    [on_reboot] = back_references(context, reboot.incident_id, domain)
    [on_peer] = back_references(
        context, routed(context, peer_event).incident_id, domain
    )

    assert EXPORT in on_reboot and "node-b" in on_reboot
    assert "diagnostic-only" not in on_reboot, on_reboot
    assert on_reboot.endswith(
        "this incident's REBOOT_NODE handling of this node is unchanged"
    ), on_reboot
    assert on_peer.endswith(DIAGNOSTIC_ONLY), "node-b's own chain is diagnostic-only"


@pytest.mark.parametrize(
    ("xid", "handling"),
    [(13, "RESTART_WORKLOAD"), (48, "RESET_GPU"), (79, "REBOOT_NODE")],
)
def test_the_back_reference_on_an_attempt_recovery_names_that_recovery(
    xid: int, handling: str
) -> None:
    # node-a trains while an XID recovery of its attempt is pending; its
    # shared-storage finding is folded into that recovery (b213).
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    running_attempt(context, "node-a")
    recovery = xid_recovery(context, xid)
    [folded_event] = ingest(context, shared("a-1", "node-a", 1))
    assert routed(context, folded_event).incident_id == recovery.incident_id
    [peer_event] = ingest(context, shared("b-1", "node-b", 2))
    [domain] = domains(context)

    [on_recovery] = back_references(context, recovery.incident_id, domain)
    [on_peer] = back_references(
        context, routed(context, peer_event).incident_id, domain
    )

    assert on_recovery.endswith(
        f"this incident's {handling} handling of this node is unchanged"
    ), on_recovery
    assert on_peer.endswith(DIAGNOSTIC_ONLY), on_peer
    # The record and its mail say the same for the member that is not on a
    # diagnostic of its own, and claim no diagnostic-only chain for every node.
    [line] = [
        reason for reason in domain.reasons if reason.startswith("member node-a:")
    ]
    assert line.endswith(f"; that incident's handling: {handling})"), line
    [peer_line] = [
        reason for reason in domain.reasons if reason.startswith("member node-b:")
    ]
    assert "handling" not in peer_line, peer_line
    assert "diagnostic-only chain" not in domain.reasons[0], domain.reasons[0]
    [mail] = domain_mails(notifier)
    assert f"node-a: {recovery.incident_id} (mount /mnt/shared; {handling})" in (
        mail.body_text
    ), mail.body_text
    assert "unchanged and diagnostic-only" not in mail.body_text, mail.body_text
    assert "not suspected hardware faults" not in mail.support_case_draft


def test_an_all_diagnostic_domain_keeps_its_diagnostic_wording() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    ingest(context, shared("a-1", "node-a", 1))
    ingest(context, shared("b-1", "node-b", 2))
    [domain] = domains(context)

    assert all(
        "handling" not in reason
        for reason in domain.reasons
        if reason.startswith("member ")
    ), domain.reasons
    [mail] = domain_mails(notifier)
    assert "Node handling: unchanged and diagnostic-only" in mail.body_text
    assert "not suspected hardware faults" in mail.support_case_draft
