"""GF-GENERAL-LIVE-005 ``shared_failure_domain``: the peer reports after the
first node's own chain has already ended (b216).

A shared-storage finding mints a diagnostic-only chain (FREEZE_EVIDENCE,
VALIDATE_HOST). VALIDATE_HOST reads the gauge, finds the export still
unavailable and fails at once, and a failed chain that never touched the node
closes its incident RECOVERED "diagnostic inconclusive" -- live 1-2 s after the
incident was created, while the export was still down. Each node finds the
outage on its own collector tick, so a peer's activation lands up to one tick
(15 s) later: plain EKS, R13h, 2026-10-28 03:13Z, incidents 2.5 s apart, the
first closed 0.8 s before the second was created, no record (the same on
HyperPod at 11:45Z the day before, 3.4 s apart). The peer look took only open
per-node incidents as candidates, so the node that reported first was never a
peer. Its latest reading of the export -- the actual evidence -- still said
"unavailable".

These tests run the product's own executor on the first node's chain before
the second node reports, which is the live order.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.adapters import ControlPlaneEvidenceAdapter
from gpu_fault.adapters.gpu_validation import GpuValidationAdapter
from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.execution import ProductionExecutorConfig
from gpu_fault.gpu_metrics import GpuMetricsService
from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
    NodeHealthFinding,
)
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.shared_storage_failure_domain import (
    SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE,
    SHARED_STORAGE_FAILURE_DOMAIN_NOTIFICATION_CATEGORY,
)
from gpu_fault.store import InMemoryStore
from tests._builders import asgi_client, host_telemetry_batch
from tests.host_health._support import RecordingNotifier

EXPORT = "10.0.0.7:/a"
MOUNT = "/mnt/gfrs-live/0123456789ab/shared"
LABELS = {"source": EXPORT, "fstype": "nfs4"}
CLUSTER = "cluster-a"
DIAGNOSTIC_ONLY = [WorkflowOperation.FREEZE_EVIDENCE, WorkflowOperation.VALIDATE_HOST]
OWNER = "simulated-runtime"


def _context(
    notifier: RecordingNotifier, store: Any | None = None
) -> ApplicationContext:
    """The control plane with the product's FREEZE_EVIDENCE and VALIDATE_HOST
    adapters, so a node's chain ends the way it ends live."""

    backing = store if store is not None else InMemoryStore()
    context = ApplicationContext(
        notification_notifier=notifier,
        store=backing,
        production_executor_config=ProductionExecutorConfig(
            enabled=True,
            executor_id="b216-closed-member",
            allowed_operations=frozenset(DIAGNOSTIC_ONLY),
        ),
        production_adapters=[
            ControlPlaneEvidenceAdapter(owner=OWNER),
            GpuValidationAdapter(
                GpuMetricsService(store=backing), store=backing, owner=OWNER
            ),
        ],
        execution_token="b216-closed-member-" + "x" * 32,
    )
    backing.save_profile(default_simulated_profile())
    return context


def _batch(
    batch_id: str,
    node_id: str,
    value: float,
    observed_at: datetime,
    *,
    labels: dict[str, str] | None = None,
) -> HostTelemetryBatch:
    """One host batch: the shared mount's gauge plus the host metrics
    VALIDATE_HOST needs to judge at all (healthy)."""

    return host_telemetry_batch(
        batch_id,
        observed_at,
        [
            HostMetricSample(
                name="shared_filesystem_unavailable",
                value=value,
                device=MOUNT,
                labels=dict(LABELS if labels is None else labels),
            ),
            HostMetricSample(name="load1_per_cpu", value=0.1),
            HostMetricSample(name="memory_used_percent", value=20),
            HostMetricSample(name="filesystem_used_percent", value=20, device="/"),
        ],
        node_id=node_id,
        runtime_profile_version="simulated-v1",
    )


def _ingest(context: ApplicationContext, *batches: HostTelemetryBatch) -> list[dict]:
    async def scenario() -> list[dict]:
        results = []
        async with asgi_client(context) as client:
            for batch in batches:
                response = await client.post(
                    "/v1/collector-events/host-telemetry",
                    json=batch.model_dump(mode="json"),
                )
                assert response.status_code < 300, response.text
                results.append(response.json())
        return results

    return asyncio.run(scenario())


def _run_chain(context: ApplicationContext, response: dict) -> FaultIncident:
    """Run the node's chain to its end: VALIDATE_HOST fails on the unavailable
    export and the incident closes RECOVERED "diagnostic inconclusive"."""

    (request_id,) = response["workflow_request_ids"]
    result = context.workflow_executor.execute(
        request_id, WorkflowExecutionRequest(expected_fencing_token=1)
    )
    assert result.status is WorkflowStatus.FAILED, result
    workflow = context.store.get_workflow(request_id)
    assert [step.operation for step in workflow.official_steps] == DIAGNOSTIC_ONLY
    incident = context.store.get_incident(workflow.incident_id)
    assert incident.state is IncidentState.RECOVERED, incident.state
    assert any(
        reason.startswith("diagnostic inconclusive") for reason in incident.reasons
    ), incident.reasons
    return incident


def _domain_records(context: ApplicationContext) -> list[FaultIncident]:
    return [
        incident
        for incident in context.store.list_incidents_by_state(
            CLUSTER, set(IncidentState)
        )
        if incident.event_type == SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE
    ]


def _node_incident(context: ApplicationContext, node_id: str) -> FaultIncident:
    (incident,) = [
        incident
        for incident in context.store.list_incidents_by_state(
            CLUSTER, set(IncidentState), node_ids={node_id}
        )
        if incident.event_type == "NODE_HEALTH"
    ]
    return incident


def _domain_mails(notifier: RecordingNotifier) -> list[Any]:
    return [
        item
        for item in notifier.notifications
        if item.category == SHARED_STORAGE_FAILURE_DOMAIN_NOTIFICATION_CATEGORY
    ]


def _assert_one_record_for_both(
    context: ApplicationContext, notifier: RecordingNotifier
) -> FaultIncident:
    """The record the harness oracle reads: both nodes, a member line with each
    node's incident, a back-reference on each incident, one mail."""

    (domain,) = _domain_records(context)
    assert domain.node_ids == ["node-a", "node-b"], domain.node_ids
    assert domain.state is IncidentState.DETECTED, domain.state
    assert domain.policy_reference == f"shared-filesystem:{EXPORT}"
    assert domain.effective_action is None, "the domain record drives no action"
    assert domain.workflow_request_id is None, "the domain record has no workflow"
    assert EXPORT in domain.reasons[0], domain.reasons[0]
    for node_id in ("node-a", "node-b"):
        incident = _node_incident(context, node_id)
        assert any(
            f"member {node_id}: incident {incident.incident_id}" in reason
            for reason in domain.reasons
        ), f"the record must name {node_id}'s incident: {domain.reasons}"
        assert any(
            reason.startswith(f"common failure domain {domain.incident_id}:")
            and "diagnostic-only handling of this node is unchanged" in reason
            for reason in incident.reasons
        ), f"{node_id}'s incident must point back at the record: {incident.reasons}"
        # The reference is prose on the record only: its chain and verdict stay.
        assert incident.effective_action is RecoveryAction.RUN_DIAGNOSTICS
    (mail,) = _domain_mails(notifier)
    assert mail.incident_id == domain.incident_id, mail.incident_id
    for node_id in ("node-a", "node-b"):
        incident_id = _node_incident(context, node_id).incident_id
        assert incident_id in mail.evidence_refs, mail.evidence_refs
    return domain


def test_a_peer_reporting_after_the_first_nodes_chain_closed_forms_the_record() -> None:
    # The live order on EKS: node-a's activation, node-a's chain runs and
    # closes RECOVERED while the export is still down, then node-b's
    # activation 2.5 s (node clock) later.
    notifier = RecordingNotifier()
    context = _context(notifier)
    start = datetime.now(timezone.utc) - timedelta(seconds=40)

    (first,) = _ingest(context, _batch("a-1", "node-a", 1, start))
    closed = _run_chain(context, first)
    (second,) = _ingest(
        context, _batch("b-1", "node-b", 1, start + timedelta(seconds=2.5))
    )

    domain = _assert_one_record_for_both(context, notifier)
    assert first["notification_ids"] == [], "one node alone is not a domain"
    assert len(second["notification_ids"]) == 1, second["notification_ids"]
    reread = context.store.get_incident(closed.incident_id)
    assert reread.state is IncidentState.RECOVERED, (
        "the back-reference does not reopen the first node's closed chain"
    )
    # node-b's chain ends the same way; the record does not follow it: it
    # stays open while a member still reports the export unavailable.
    _run_chain(context, second)
    assert context.store.get_incident(domain.incident_id).state is (
        IncidentState.DETECTED
    ), "a member's closed chain is not the export coming back"

    _ingest(
        context,
        _batch("a-2", "node-a", 0, start + timedelta(seconds=20)),
        _batch("b-2", "node-b", 0, start + timedelta(seconds=22)),
    )
    cleared = context.store.get_incident(domain.incident_id)
    assert cleared.state is IncidentState.RECOVERED, cleared.reasons
    assert len(_domain_mails(notifier)) == 1, "clearing mails nothing new"


def test_a_closed_node_whose_export_answers_again_is_no_peer() -> None:
    # The candidate is only a candidate: the node's latest reading decides. A
    # node that reported the export available after its chain closed is not
    # part of the next node's outage.
    notifier = RecordingNotifier()
    context = _context(notifier)
    start = datetime.now(timezone.utc) - timedelta(seconds=30)

    (first,) = _ingest(context, _batch("a-1", "node-a", 1, start))
    _run_chain(context, first)
    _ingest(
        context,
        _batch("a-2", "node-a", 0, start + timedelta(seconds=5)),
        _batch("b-1", "node-b", 1, start + timedelta(seconds=8)),
    )

    assert _domain_records(context) == [], "node-a reads the export available"
    assert _domain_mails(notifier) == [], "no domain, no domain mail"


def test_the_safety_net_names_a_members_closed_chain() -> None:
    # node-b's activation carried no export identity, so it correlated with
    # nothing; both chains closed RECOVERED. node-b's next failing sample --
    # delivered with its health summary -- names the export: the safety net
    # forms the record with both closed chains as the members' incidents.
    notifier = RecordingNotifier()
    context = _context(notifier)
    start = datetime.now(timezone.utc) - timedelta(seconds=40)

    (first,) = _ingest(context, _batch("a-1", "node-a", 1, start))
    _run_chain(context, first)
    (second,) = _ingest(
        context, _batch("b-1", "node-b", 1, start + timedelta(seconds=3), labels={})
    )
    _run_chain(context, second)
    assert _domain_records(context) == [], "no identity, nothing to correlate on"

    (summary,) = _ingest(
        context, _batch("b-2", "node-b", 1, start + timedelta(seconds=20))
    )

    _assert_one_record_for_both(context, notifier)
    assert len(summary["notification_ids"]) == 1, summary["notification_ids"]


class BlindLookStore:
    """Two activations in flight at once, and a finish half that runs late.

    Under READ COMMITTED the look inside node-b's ingestion transaction does
    not see a node-a whose own transaction has not committed yet (the
    ``InFlightPeerStore`` shape of ``test_shared_storage_failure_domain``).
    ``observe_committed`` then runs on the replica that took node-b's POST
    after node-b's commit -- and a dispatcher on another replica may claim and
    finish node-b's chain before it does (a processor group finishes after
    its whole group committed). In process the finish half runs at once, so
    this wrapper keeps ``hidden`` out of every incident read for the whole
    request while ``blind`` is set; the test ends node-b's chain and then
    looks the way the late finish half would.
    """

    def __init__(self, store: Any, hidden: str) -> None:
        self._store = store
        self._hidden = hidden
        self.blind = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def list_incidents_by_state(
        self, cluster_id: str, states: Any, **kwargs: Any
    ) -> list[FaultIncident]:
        incidents: list[FaultIncident] = self._store.list_incidents_by_state(
            cluster_id, states, **kwargs
        )
        if not self.blind:
            return incidents
        return [item for item in incidents if self._hidden not in item.node_ids]

    @contextmanager
    def blinded(self) -> Iterator[None]:
        self.blind = True
        try:
            yield
        finally:
            self.blind = False


def test_a_late_look_whose_own_chain_already_closed_still_counts_it() -> None:
    notifier = RecordingNotifier()
    store = BlindLookStore(InMemoryStore(), hidden="node-a")
    context = _context(notifier, store)
    start = datetime.now(timezone.utc) - timedelta(seconds=30)
    batch_b = _batch("b-1", "node-b", 1, start + timedelta(seconds=2.5))

    (first,) = _ingest(context, _batch("a-1", "node-a", 1, start))
    with store.blinded():
        (second,) = _ingest(context, batch_b)
    assert _domain_records(context) == [], "both looks of the request were blind"
    # Both chains end before node-b's late finish half looks.
    _run_chain(context, second)
    _run_chain(context, first)

    findings = [NodeHealthFinding.model_validate(item) for item in second["findings"]]
    late = context.shared_storage_domains.observe_committed(batch_b, findings)

    _assert_one_record_for_both(context, notifier)
    assert len(late) == 1, late
