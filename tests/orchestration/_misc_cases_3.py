"""Ingest-path cases: the boot-id-over-age fence, per-node ingest locking and
marker correlation without incident reads."""

from __future__ import annotations

import threading
from datetime import timedelta

from gpu_fault.models import IncidentState, Severity
from gpu_fault.policy import ActionDisposition
from tests._builders import (
    build_context,
    copy_model,
    fault_incident,
    node_health_finding,
)

from ._support import (
    NOW,
    ApplicationContext,
    NodeHealthCategory,
    NodeHealthFinding,
    RecoveryAction,
    WorkflowStatus,
    WorkloadState,
    XidEvent,
    _active_agent,
    event,
)

TWENTY_MINUTES = timedelta(minutes=20)


def _aged_event(event_id: str, *, source_boot_id: str | None) -> XidEvent:
    return copy_model(
        event(48, event_id=event_id),
        source_boot_id=source_boot_id,
        source_event_time=NOW,
    )


def _agent_fresh_at(boot_id: str, now):
    """``_active_agent`` leases until NOW+1m; the fence runs 20 minutes later,
    so the Agent has to have heartbeaten since or it reads as stale."""

    return copy_model(
        _active_agent(boot_id=boot_id),
        last_seen_at=now,
        lease_expires_at=now + timedelta(minutes=1),
    )


# --- Item 1: a matching boot id outranks the 900 s age limit -----------------


def test_fault_action_generation_fence_trusts_matching_boot_over_age() -> None:
    """A 15-minute processor backlog must not turn every reset into a
    quarantine when the event's boot id proves it is this incarnation."""

    context = build_context()
    context.store.save_agent(_agent_fresh_at("boot-current", NOW + TWENTY_MINUTES))
    xid_event = _aged_event("xid-backlog-same-boot", source_boot_id="boot-current")
    decision = context.policy.evaluate_xid(xid_event)
    assert decision.disposition is ActionDisposition.EXECUTABLE
    assert decision.action is RecoveryAction.RESET_GPU

    fenced = context.orchestrator.apply_fault_action_generation_fence(
        xid_event, decision, now=NOW + TWENTY_MINUTES
    )

    assert fenced.disposition is ActionDisposition.EXECUTABLE
    assert fenced.action is RecoveryAction.RESET_GPU
    assert fenced.marker.active, "expected fenced.marker.active to be truthy"
    assert not fenced.requires_operator, "expected fenced.requires_operator to be falsy"
    # The age is still recorded for the audit trail; it just does not block.
    assert any(
        "exceeds automatic action limit" in reason for reason in fenced.reasons
    ), "expected the source event age to be annotated on the decision"
    assert not any(
        reason.startswith("STALE_FAULT_GENERATION") for reason in fenced.reasons
    ), "a boot-confirmed event must not carry a STALE_FAULT_GENERATION reason"

    incident, workflow = context.orchestrator.ingest(xid_event, fenced)
    assert workflow is not None
    assert workflow.status is WorkflowStatus.PENDING
    assert incident.state is IncidentState.ACTION_PENDING


def test_fault_action_generation_fence_age_limit_holds_without_boot_id() -> None:
    """Without a boot id on either side the age limit is the only proof."""

    context = build_context()
    context.store.save_agent(_agent_fresh_at("boot-current", NOW + TWENTY_MINUTES))
    xid_event = _aged_event("xid-backlog-no-boot", source_boot_id=None)
    decision = context.policy.evaluate_xid(xid_event)

    fenced = context.orchestrator.apply_fault_action_generation_fence(
        xid_event, decision, now=NOW + TWENTY_MINUTES
    )

    assert fenced.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert fenced.action is None
    assert any(
        reason.startswith("STALE_FAULT_GENERATION")
        and "exceeds automatic action limit" in reason
        for reason in fenced.reasons
    ), "expected the age limit to block an event without a boot id"

    # The Agent side can be the one missing the boot id too: still blocked.
    context.store.save_agent(_agent_fresh_at("", NOW + TWENTY_MINUTES))
    with_boot = _aged_event("xid-backlog-agent-no-boot", source_boot_id="boot-x")
    fenced = context.orchestrator.apply_fault_action_generation_fence(
        with_boot, context.policy.evaluate_xid(with_boot), now=NOW + TWENTY_MINUTES
    )
    assert fenced.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert any("did not report a boot ID" in reason for reason in fenced.reasons), (
        "expected the missing Agent boot id to be reported"
    )
    assert any(
        "exceeds automatic action limit" in reason for reason in fenced.reasons
    ), "expected the age limit to keep applying when the Agent has no boot id"


def test_fault_action_generation_fence_mismatched_boot_still_blocks() -> None:
    context = build_context()
    context.store.save_agent(_agent_fresh_at("boot-current", NOW + TWENTY_MINUTES))
    xid_event = _aged_event("xid-backlog-old-boot", source_boot_id="boot-previous")
    decision = context.policy.evaluate_xid(xid_event)

    fenced = context.orchestrator.apply_fault_action_generation_fence(
        xid_event, decision, now=NOW + TWENTY_MINUTES
    )

    assert fenced.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert fenced.action is None
    assert not fenced.marker.active, "expected fenced.marker.active to be falsy"
    assert any(
        "does not match current Agent boot ID" in reason for reason in fenced.reasons
    ), "expected the boot id mismatch to be reported"
    assert any(
        "exceeds automatic action limit" in reason for reason in fenced.reasons
    ), "expected the age limit to be reported alongside the mismatch"


# --- Item 2: ingest serialises per (cluster_id, node_id), not per process ----


class _Gate:
    """Blocks one node's ingest inside a store call, outside the store's own
    lock, so the test can observe what the orchestrator lock lets through."""

    def __init__(self, store, method: str, *, block_when) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self._block_when = block_when
        original = getattr(store, method)

        def gated(*args, **kwargs):
            if self._block_when(*args, **kwargs):
                self.entered.set()
                assert self.release.wait(30), "gate was never released"
            return original(*args, **kwargs)

        setattr(store, method, gated)


def _node_event(node_id: str, event_id: str) -> XidEvent:
    return copy_model(event(48, event_id=event_id), node_id=node_id)


def _run(target) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_ingest_of_different_nodes_is_not_serialised() -> None:
    context = build_context()
    gate = _Gate(
        context.store,
        "get_incident_by_event",
        block_when=lambda event_id: event_id == "xid-node-a",
    )
    results: dict[str, object] = {}

    def ingest(node_id: str, event_id: str) -> None:
        xid_event = _node_event(node_id, event_id)
        decision = context.policy.evaluate_xid(xid_event)
        results[node_id] = context.orchestrator.ingest(xid_event, decision)

    blocked = _run(lambda: ingest("node-a", "xid-node-a"))
    assert gate.entered.wait(5), "node-a ingest never reached the store"

    other = _run(lambda: ingest("node-b", "xid-node-b"))
    other.join(5)
    assert not other.is_alive(), (
        "node-b ingest waited on node-a's ingest; the lock is still process-wide"
    )
    assert "node-b" in results, "expected node-b ingest to finish while node-a held"

    gate.release.set()
    blocked.join(5)
    assert not blocked.is_alive(), "node-a ingest did not finish after release"
    assert "node-a" in results, "expected node-a ingest to finish after release"


def test_ingest_of_the_same_node_stays_serialised() -> None:
    context = build_context()
    gate = _Gate(
        context.store,
        "get_incident_by_event",
        block_when=lambda event_id: event_id == "xid-node-a-first",
    )
    order: list[str] = []

    def ingest(event_id: str) -> None:
        xid_event = _node_event("node-a", event_id)
        decision = context.policy.evaluate_xid(xid_event)
        context.orchestrator.ingest(xid_event, decision)
        order.append(event_id)

    first = _run(lambda: ingest("xid-node-a-first"))
    assert gate.entered.wait(5), "first ingest never reached the store"

    second = _run(lambda: ingest("xid-node-a-second"))
    second.join(0.5)
    assert second.is_alive(), (
        "a second ingest for the same node ran while the first was mid-flight"
    )
    assert order == []

    gate.release.set()
    first.join(5)
    second.join(5)
    assert order == ["xid-node-a-first", "xid-node-a-second"]


def _quarantine_finding(node_id: str, finding_id: str) -> NodeHealthFinding:
    return node_health_finding(
        finding_id,
        finding_id,
        node_id=node_id,
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="retired pages pending",
        recommended_action=RecoveryAction.RESET_GPU,
        gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )


def test_node_health_ingest_of_different_nodes_is_not_serialised() -> None:
    """The health family holds the lock the coordinator hands it; that lock
    must scope to the finding's node, not the process."""

    context = build_context()
    gate = _Gate(
        context.store,
        "create_incident_workflow_if_absent",
        block_when=lambda event_id, *args, **kwargs: event_id == "health-node-a",
    )
    results: dict[str, object] = {}

    def ingest(node_id: str, finding_id: str) -> None:
        results[node_id] = context.orchestrator.ingest_node_health(
            _quarantine_finding(node_id, finding_id)
        )

    blocked = _run(lambda: ingest("node-a", "health-node-a"))
    assert gate.entered.wait(5), "node-a health ingest never reached the store"

    other = _run(lambda: ingest("node-b", "health-node-b"))
    other.join(5)
    assert not other.is_alive(), (
        "node-b health ingest waited on node-a; the health lock is process-wide"
    )

    gate.release.set()
    blocked.join(5)
    assert not blocked.is_alive(), "node-a health ingest did not finish"
    assert set(results) == {"node-a", "node-b"}


def test_simulate_waits_for_the_workflow_nodes_ingest() -> None:
    """``simulate`` only has a workflow id; it must still queue behind an
    ingest on the workflow's node rather than run beside it."""

    context = build_context()
    xid_event = _node_event("node-a", "xid-node-a-seed")
    _, workflow = context.orchestrator.ingest(
        xid_event, context.policy.evaluate_xid(xid_event)
    )
    assert workflow is not None

    gate = _Gate(
        context.store,
        "get_incident_by_event",
        block_when=lambda event_id: event_id == "xid-node-a-hold",
    )

    def ingest() -> None:
        held = _node_event("node-a", "xid-node-a-hold")
        context.orchestrator.ingest(held, context.policy.evaluate_xid(held))

    holder = _run(ingest)
    assert gate.entered.wait(5), "holding ingest never reached the store"

    simulated: list[object] = []
    simulator = _run(
        lambda: simulated.append(
            context.orchestrator.simulate(workflow.request_id, workflow.fencing_token)
        )
    )
    simulator.join(0.5)
    assert simulator.is_alive(), "simulate ran beside an ingest on its own node"

    gate.release.set()
    holder.join(5)
    simulator.join(5)
    assert len(simulated) == 1
    assert simulated[0].status is WorkflowStatus.SUCCEEDED


# --- Item 3: marker correlation reads cluster_id off the marker --------------


def _save_finding_marker(
    context: ApplicationContext,
    finding: NodeHealthFinding,
    *,
    cluster_id: str | None,
    incident_cluster_id: str = "cluster-a",
) -> str:
    marker = finding.marker().model_copy(update={"cluster_id": cluster_id})
    context.completion.add_marker(marker)
    context.store.save_incident(
        fault_incident(
            marker.incident_id,
            finding.event_id,
            "NODE_HEALTH",
            incident_cluster_id,
            node_ids=[finding.node_id],
            gpu_uuids=finding.gpu_uuids,
            policy_version=finding.policy_version,
            policy_source=finding.policy_source,
            effective_action=finding.recommended_action,
            state=IncidentState.DETECTED,
            reasons=[finding.reason],
            created_at=finding.observed_at,
            updated_at=finding.observed_at,
        )
    )
    return marker.incident_id


def _memory_finding(finding_id: str) -> NodeHealthFinding:
    return node_health_finding(
        finding_id,
        finding_id,
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity=Severity.CRITICAL,
        reason="DCGM field 319 increased by one",
        metric_name="dcgm_ecc_dbe_delta",
        gpu_uuids=["GPU-a"],
        pci_bdf="0000:59:00.0",
        recommended_action=RecoveryAction.RESET_GPU,
        runtime_profile_version="simulated-v1",
    )


def _kernel_xid48() -> XidEvent:
    return XidEvent(
        event_id="kernel-xid48",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW + timedelta(seconds=10),
        event_source="NVIDIA_KERNEL_LOG",
        xid=48,
        gpu_uuid="GPU-a",
        pci_bdf="0000:59:00.0",
        product="H100",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )


def _count_incident_reads(store) -> list[str]:
    reads: list[str] = []
    original = store.get_incident

    def counting(incident_id: str):
        reads.append(incident_id)
        return original(incident_id)

    store.get_incident = counting
    return reads


def test_marker_correlation_does_not_read_incidents_for_scoped_markers() -> None:
    context = build_context()
    incident_ids = [
        _save_finding_marker(
            context, _memory_finding(f"dcgm-memory-{index}"), cluster_id="cluster-a"
        )
        for index in range(5)
    ]
    decision = context.policy.evaluate_xid(_kernel_xid48())
    reads = _count_incident_reads(context.store)

    correlated = context.orchestrator.correlate_provider_event(
        decision, cluster_id="cluster-a"
    )

    assert correlated.duplicate, "expected correlated.duplicate to be truthy"
    assert correlated.marker.incident_id in incident_ids
    assert reads == [], "scoped markers must not cost one incident read each"


def test_marker_correlation_skips_scoped_markers_of_other_clusters() -> None:
    context = build_context()
    _save_finding_marker(
        context,
        _memory_finding("dcgm-memory-other"),
        cluster_id="cluster-b",
        incident_cluster_id="cluster-b",
    )
    decision = context.policy.evaluate_xid(_kernel_xid48())
    reads = _count_incident_reads(context.store)

    correlated = context.orchestrator.correlate_provider_event(
        decision, cluster_id="cluster-a"
    )

    assert not correlated.duplicate, "expected correlated.duplicate to be falsy"
    assert reads == []


def test_marker_correlation_falls_back_to_the_incident_for_legacy_markers() -> None:
    context = build_context()
    local = _save_finding_marker(
        context, _memory_finding("dcgm-memory-legacy-local"), cluster_id=None
    )
    decision = context.policy.evaluate_xid(_kernel_xid48())
    reads = _count_incident_reads(context.store)

    correlated = context.orchestrator.correlate_provider_event(
        decision, cluster_id="cluster-a"
    )

    assert correlated.duplicate, "expected correlated.duplicate to be truthy"
    assert correlated.marker.incident_id == local
    assert reads == [local], "a legacy marker must still be scoped by its incident"


def test_marker_correlation_reads_legacy_markers_of_other_clusters_out() -> None:
    context = build_context()
    foreign = _save_finding_marker(
        context,
        _memory_finding("dcgm-memory-legacy-foreign"),
        cluster_id=None,
        incident_cluster_id="cluster-b",
    )
    decision = context.policy.evaluate_xid(_kernel_xid48())
    reads = _count_incident_reads(context.store)

    correlated = context.orchestrator.correlate_provider_event(
        decision, cluster_id="cluster-a"
    )

    assert not correlated.duplicate, "expected correlated.duplicate to be falsy"
    assert reads == [foreign]
