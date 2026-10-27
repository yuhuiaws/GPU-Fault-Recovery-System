"""Findings the product folds into another finding's record (b213).

A node-health finding that arrives while a managed attempt's recovery is in
flight is grouped into that attempt's record: ``GPU_FAULT_GROUP``, under the
event id, policy source and node scope of the fault that minted it (here an
XID's, ``NVIDIA_CATALOG``), and a record-only fold does not even add the
finding's node. Two readers took only ``NODE_HEALTH`` records the finding
minted itself:

* the shared-storage failure domain looked for its peers' per-node incidents
  that way, so a node whose ``shared_filesystem_unavailable`` joined an
  attempt's recovery was invisible to the peer that activated after it -- no
  record for a two-node outage, or one naming "none open" for the node;
* the EFA traffic admin action refused a SPIKE grouped with the attempt's
  recovery: 409 "event_id is not the requested node's EFA traffic event".

Everything goes through the product's own ingestion: host telemetry and XIDs
are posted to the control-plane API over an in-memory store, as the collectors
post them. The only scripted step is the executor's progress on a recovery
(completed steps, a restart's new attempt) where a case needs it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration.families.incident_labels import (
    NODE_HEALTH_INCIDENT_EVENT_TYPES,
)
from gpu_fault.shared_storage_failure_domain import (
    SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE,
    SHARED_STORAGE_FAILURE_DOMAIN_NOTIFICATION_CATEGORY,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.watcher import AttemptObservation, WorkloadPhase
from tests._builders import (
    asgi_client,
    attempt_observation,
    container_observation,
    copy_model,
    host_telemetry_batch,
    workflow_step_execution,
)
from tests.host_health._support import RecordingNotifier
from tests.host_health.test_shared_storage_failure_domain import InFlightPeerStore

NOW = datetime.now(timezone.utc)
EXPORT = "10.0.0.5:/export"
LABELS = {"source": EXPORT, "fstype": "nfs4"}
TOKEN = "a" * 32
WORKLOAD = "training/job/job-a"
HOST_TELEMETRY = "/v1/collector-events/host-telemetry"
ADMIN_ACTIONS = "/v1/efa-traffic/admin-actions"


def at(seconds: float) -> datetime:
    return NOW + timedelta(seconds=seconds)


def running_attempt(
    context: ApplicationContext,
    *nodes: str,
    attempt_id: str = "attempt-a",
    observed_at: datetime = NOW,
    started_at: datetime | None = None,
) -> AttemptObservation:
    """``job-a``/``attempt_id`` running one rank per node, ``GPU-<rank>`` each."""

    observation = attempt_observation(
        "job-a",
        attempt_id,
        observed_at,
        started_at=started_at,
        expected_critical_ranks=len(nodes),
        workload_ids=[WORKLOAD],
        containers=[
            container_observation(
                f"pod-{attempt_id}-{rank}",
                f"worker-{rank}",
                rank,
                node,
                gpu_uuids=[f"GPU-{rank}"],
            )
            for rank, node in enumerate(nodes)
        ],
    )
    context.store.save_attempt_observation(observation)
    return observation


def post(
    context: ApplicationContext,
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    async def scenario() -> tuple[int, dict[str, Any]]:
        async with asgi_client(context) as client:
            response = await client.post(path, json=payload, headers=headers)
            return response.status_code, dict(response.json())

    return asyncio.run(scenario())


def ingest(context: ApplicationContext, batch: HostTelemetryBatch) -> list[str]:
    """Post one host batch as the host collector does: its findings' event ids."""

    status, body = post(context, HOST_TELEMETRY, batch.model_dump(mode="json"))
    assert status == 200, body
    return [str(item["event_id"]) for item in body["findings"]]


def shared(
    batch_id: str, node: str, seconds: float, labels: dict[str, str] = LABELS
) -> HostTelemetryBatch:
    return host_telemetry_batch(
        batch_id,
        at(seconds),
        [
            HostMetricSample(
                name="shared_filesystem_unavailable",
                value=1,
                device="/mnt/shared",
                labels=dict(labels),
            )
        ],
        node_id=node,
        runtime_profile_version="simulated-v1",
    )


def efa(batch_id: str, value: float, seconds: float) -> HostTelemetryBatch:
    return host_telemetry_batch(
        batch_id,
        at(seconds),
        [
            HostMetricSample(
                name="efa_traffic_bytes_per_second", value=value, unit="bytes/second"
            )
        ],
        workload_state="ACTIVE",
        affected_workload_ids=[WORKLOAD],
        runtime_profile_version="simulated-v1",
    )


def xid_recovery(
    context: ApplicationContext,
    xid: int,
    node: str = "node-a",
    gpu: str = "GPU-0",
    observed_at: datetime = NOW,
) -> FaultIncident:
    """The attempt's recovery an XID opens: XID 13 stops and restarts the job,
    XID 48 resets the GPU in between (both ``GPU_FAULT_GROUP``)."""

    status, body = post(
        context,
        "/v1/gpu-events/xid",
        {
            "event_id": f"xid-{xid}",
            "cluster_id": "cluster-a",
            "node_id": node,
            "observed_at": observed_at.isoformat(),
            "ingested_at": observed_at.isoformat(),
            "xid": xid,
            "gpu_uuid": gpu,
            "product": "H100",
            "driver_branch": 575,
            "cuda_version": "12.9",
            "runtime_profile_version": "simulated-v1",
        },
    )
    assert status == 200, body
    incident: FaultIncident = context.store.get_incident(body["incident_id"])
    assert incident.event_type == "GPU_FAULT_GROUP", incident
    return incident


def executed(
    context: ApplicationContext, incident: FaultIncident, *done: WorkflowOperation
) -> None:
    """The executor's progress: ``done`` completed, the workflow RUNNING."""

    workflow = context.store.get_workflow(incident.workflow_request_id)
    indexes = [
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation in done
    ]
    context.store.save_workflow(
        copy_model(
            workflow,
            status=WorkflowStatus.RUNNING,
            execution_owner_id="executor-a",
            not_before=None,
            aggregation_max_deadline=NOW - timedelta(seconds=1),
            completed_step_indexes=indexes,
            completed_operations=[
                workflow.official_steps[i].operation for i in indexes
            ],
            step_executions=[
                workflow_step_execution(i, workflow.official_steps[i].operation)
                for i in indexes
            ],
        )
    )


def routed(context: ApplicationContext, event_id: str) -> FaultIncident:
    """The incident ingestion routed a finding to (its event link)."""

    incident = context.store.get_incident_by_event(event_id)
    assert incident is not None, f"{event_id} reached no incident"
    return incident


def domains(context: ApplicationContext) -> list[FaultIncident]:
    return [
        incident
        for incident in context.store.list_incidents_by_state(
            "cluster-a", set(IncidentState)
        )
        if incident.event_type == SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE
    ]


def assert_member(
    context: ApplicationContext, domain: FaultIncident, node: str, incident_id: str
) -> None:
    """The record names ``node``'s incident and that incident points back."""

    assert any(
        reason.startswith(f"member {node}: incident {incident_id} ")
        for reason in domain.reasons
    ), f"the record must name {incident_id} for {node}: {domain.reasons}"
    reasons = context.store.get_incident(incident_id).reasons
    assert any(
        reason.startswith(f"common failure domain {domain.incident_id}:")
        for reason in reasons
    ), f"{incident_id} must point back at the record: {reasons}"


# --------------------------------------------------------------------------- #
# shared-storage failure domain
# --------------------------------------------------------------------------- #
def test_a_node_folded_into_an_attempt_recovery_is_a_member_for_the_peer_after_it() -> (
    None
):
    # node-a trains while an XID 13 recovery of its attempt is pending; the
    # export goes away under node-a first and under idle node-b a tick later.
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    running_attempt(context, "node-a")
    recovery = xid_recovery(context, 13)

    [folded_event] = ingest(context, shared("a-1", "node-a", 1))
    folded = routed(context, folded_event)
    assert (folded.incident_id, folded.event_id, folded.policy_source) == (
        recovery.incident_id,
        "xid-13",
        "NVIDIA_CATALOG",
    ), "node-a's finding joins the attempt's recovery under the XID's identity"
    assert domains(context) == [], "node-a alone is no failure domain"
    plan = context.store.get_workflow(recovery.workflow_request_id).official_steps
    [peer_event] = ingest(context, shared("b-1", "node-b", 2))

    [domain] = domains(context)
    assert domain.node_ids == ["node-a", "node-b"]
    assert_member(context, domain, "node-a", recovery.incident_id)
    assert_member(context, domain, "node-b", routed(context, peer_event).incident_id)
    [mail] = [
        item
        for item in notifier.notifications
        if item.category == SHARED_STORAGE_FAILURE_DOMAIN_NOTIFICATION_CATEGORY
    ]
    assert mail.incident_id == domain.incident_id
    assert f"node-a: {recovery.incident_id}" in mail.body_text
    after = context.store.get_workflow(recovery.workflow_request_id).official_steps
    assert after == plan, "the correlation changes no plan"


def test_the_look_after_the_commit_meets_a_folded_activation_in_flight() -> None:
    # b207's two looks with node-a's activation folded: node-b's look inside
    # its own transaction cannot see node-a's (READ COMMITTED), the look after
    # its commit does -- through the folded record's marker.
    notifier = RecordingNotifier()
    store = InFlightPeerStore(InMemoryStore(), hidden="node-a")
    context = ApplicationContext(notification_notifier=notifier, store=store)
    running_attempt(context, "node-a")
    recovery = xid_recovery(context, 13)
    [folded_event] = ingest(context, shared("a-1", "node-a", 1))
    assert routed(context, folded_event).incident_id == recovery.incident_id

    status, second = post(
        context, HOST_TELEMETRY, shared("b-1", "node-b", 3).model_dump(mode="json")
    )

    assert status == 200, second
    [domain] = domains(context)
    assert domain.node_ids == ["node-a", "node-b"]
    assert_member(context, domain, "node-a", recovery.incident_id)
    assert len(second["notification_ids"]) == 1, "the later commit's look mails once"


def test_a_record_only_fold_that_does_not_name_the_node_is_found_by_its_marker() -> (
    None
):
    # A two-node attempt whose XID 48 reset on node-a has stopped the job: the
    # shared-storage finding of the peer rank on node-b is a read-only look the
    # pending restart already covers, recorded on the reset's record without
    # adding node-b to it. Only the finding's marker ties node-b to it.
    context = ApplicationContext(notification_notifier=RecordingNotifier())
    running_attempt(context, "node-a", "node-b")
    recovery = xid_recovery(context, 48)
    executed(
        context,
        recovery,
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.STOP_WORKLOADS,
    )

    [folded_event] = ingest(context, shared("b-1", "node-b", 1))
    folded = routed(context, folded_event)
    assert (folded.incident_id, folded.node_ids) == (recovery.incident_id, ["node-a"])
    [peer_event] = ingest(context, shared("c-1", "node-c", 2))

    [domain] = domains(context)
    assert domain.node_ids == ["node-b", "node-c"]
    assert_member(context, domain, "node-b", recovery.incident_id)
    assert_member(context, domain, "node-c", routed(context, peer_event).incident_id)


def test_the_safety_net_names_the_record_a_folded_activation_sits_in() -> None:
    # node-b's activation carried no export identity and joined its attempt's
    # recovery; its next failing sample carries the identity, and the
    # per-batch safety net makes it a member -- of the record that holds it.
    context = ApplicationContext(notification_notifier=RecordingNotifier())
    running_attempt(context, "node-b")
    recovery = xid_recovery(context, 13, node="node-b")
    [idle_event] = ingest(context, shared("a-1", "node-a", 1))
    [folded_event] = ingest(context, shared("b-1", "node-b", 2, labels={}))
    assert routed(context, folded_event).incident_id == recovery.incident_id
    assert domains(context) == [], "an activation without identity correlates nothing"

    assert ingest(context, shared("b-2", "node-b", 17)) == [], "no new activation"

    [domain] = domains(context)
    assert domain.node_ids == ["node-a", "node-b"]
    assert_member(context, domain, "node-a", routed(context, idle_event).incident_id)
    assert_member(context, domain, "node-b", recovery.incident_id)


@pytest.mark.parametrize("label", sorted(NODE_HEALTH_INCIDENT_EVENT_TYPES))
def test_a_peer_record_its_activation_minted_counts_under_every_label(
    label: str,
) -> None:
    # The record node-a's activation minted, under each label the product's
    # grouping gives such a record (re-labelled here by hand: a later route
    # keeps the record's id and event id).
    context = ApplicationContext(notification_notifier=RecordingNotifier())
    [minted_event] = ingest(context, shared("a-1", "node-a", 1))
    minted = routed(context, minted_event)
    context.store.save_incident(copy_model(minted, event_type=label))
    [peer_event] = ingest(context, shared("b-1", "node-b", 2))

    [domain] = domains(context)
    assert domain.node_ids == ["node-a", "node-b"]
    assert_member(context, domain, "node-a", minted.incident_id)
    assert_member(context, domain, "node-b", routed(context, peer_event).incident_id)


# --------------------------------------------------------------------------- #
# EFA traffic admin actions
# --------------------------------------------------------------------------- #
def admin(
    context: ApplicationContext,
    event_id: str,
    action: str = "ACKNOWLEDGE_TRANSIENT",
    *,
    node: str = "node-a",
    cluster: str = "cluster-a",
    attempt_id: str = "attempt-a",
) -> tuple[int, dict[str, Any]]:
    return post(
        context,
        ADMIN_ACTIONS,
        {
            "cluster_id": cluster,
            "node_id": node,
            "job_id": "job-a",
            "attempt_id": attempt_id,
            "event_id": event_id,
            "action": action,
            "operator": "admin@example.com",
            "reason": "checkpoint traffic burst",
        },
        headers={"X-GPU-Fault-Execution-Token": TOKEN},
    )


@pytest.fixture
def efa_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "100")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO", "2")


@pytest.mark.usefixtures("efa_thresholds")
def test_admin_acknowledges_a_spike_folded_into_the_attempts_recovery() -> None:
    context = ApplicationContext(execution_token=TOKEN)
    running_attempt(context, "node-a")
    assert ingest(context, efa("baseline", 1_000, 0)) == []
    recovery = xid_recovery(context, 13)
    [spike] = ingest(context, efa("spike", 3_000, 15))
    folded = routed(context, spike)
    assert (folded.incident_id, folded.policy_source) == (
        recovery.incident_id,
        "NVIDIA_CATALOG",
    ), "the SPIKE joins the attempt's recovery under the XID's identity"

    status, decision = admin(context, spike)

    assert status == 200, decision
    assert decision["resulting_signal"] == "SPIKE"
    state = context.store.get_efa_traffic_state("cluster-a/node-a")
    assert state.spike_acknowledged_by == "admin@example.com"
    evidence = context.store.list_raw_evidence(
        "cluster-a", node_id="node-a", attempt_id="attempt-a"
    )
    assert any(item.kind.value == "ADMIN_ACTION" for item in evidence), (
        "the acknowledgement leaves its ADMIN_ACTION evidence"
    )


def restarted_attempt(context: ApplicationContext) -> FaultIncident:
    """The XID 13 recovery restarted the job into ``attempt-b``, its record
    (``attempt-a``'s) still open: findings of ``attempt-b`` are attempt-grouped
    but open a ``GPU_FAULT_GROUP`` record of their own."""

    old = running_attempt(context, "node-a", observed_at=at(-300), started_at=at(-600))
    recovery = xid_recovery(context, 13, observed_at=at(-300))
    workflow = context.store.get_workflow(recovery.workflow_request_id)
    [restart] = [
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    ]
    context.store.save_workflow(
        copy_model(
            workflow,
            status=WorkflowStatus.RUNNING,
            execution_owner_id="executor-a",
            step_executions=[
                workflow_step_execution(
                    restart,
                    WorkflowOperation.RESTART_WORKLOAD,
                    details={"restart_attempt_id": "attempt-b"},
                )
            ],
        )
    )
    context.store.save_attempt_observation(
        copy_model(old, workload_phase=WorkloadPhase.STOPPED, observed_at=at(-60))
    )
    running_attempt(context, "node-a", attempt_id="attempt-b", started_at=at(-30))
    return recovery


@pytest.mark.usefixtures("efa_thresholds")
def test_admin_actions_take_a_spike_whose_own_record_is_grouped() -> None:
    context = ApplicationContext(execution_token=TOKEN)
    recovery = restarted_attempt(context)
    assert ingest(context, efa("baseline", 1_000, 0)) == []
    [spike] = ingest(context, efa("spike", 3_000, 15))
    own = routed(context, spike)
    assert (own.incident_id, own.event_type, own.policy_source) == (
        f"inc-{spike}",
        "GPU_FAULT_GROUP",
        "SITE_EFA_TRAFFIC",
    ), "the SPIKE opens its own grouped record, beside the restart's"
    assert own.incident_id != recovery.incident_id

    acknowledged, decision = admin(context, spike, attempt_id="attempt-b")
    assert acknowledged == 200, decision

    # A shared-storage finding of the attempt folds into the SPIKE's record
    # and wins it: the record's policy source is no longer the SPIKE's.
    [folded_event] = ingest(context, shared("a-1", "node-a", 16))
    assert routed(context, folded_event).incident_id == own.incident_id
    assert context.store.get_incident(own.incident_id).policy_source != (
        "SITE_EFA_TRAFFIC"
    )
    accepted, decision = admin(
        context, spike, "ACCEPT_NEW_BASELINE", attempt_id="attempt-b"
    )

    assert accepted == 200, decision
    assert decision["resulting_signal"] == "NORMAL"
    assert decision["resulting_baseline_bytes_per_second"] == 3_000


@pytest.mark.usefixtures("efa_thresholds")
def test_admin_actions_still_refuse_an_event_that_is_not_the_nodes_efa_traffic() -> (
    None
):
    context = ApplicationContext(execution_token=TOKEN)
    running_attempt(context, "node-a")
    ingest(context, efa("baseline", 1_000, 0))
    recovery = xid_recovery(context, 13)
    [spike] = ingest(context, efa("spike", 3_000, 15))

    refusals = {
        "the XID's own event": admin(context, recovery.event_id),
        "another node's request": admin(context, spike, node="node-b"),
        "another cluster's request": admin(context, spike, cluster="cluster-b"),
    }

    for case, (status, body) in refusals.items():
        assert (status, body.get("detail")) == (
            409,
            "event_id is not the requested node's EFA traffic event",
        ), case
    state = context.store.get_efa_traffic_state("cluster-a/node-a")
    assert state.spike_acknowledged_by is None, "nothing was acknowledged"
