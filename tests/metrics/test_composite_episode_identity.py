"""One composite finding id -- one incident -- per sustained episode.

GF-GENERAL-LIVE-024 ``independent_xid`` failed live (2026-10-26 07:52-08:07Z,
plain EKS ip-10-0-43-5, R13g) with two CORRECTABLE_MEMORY_DEGRADATION incidents
for one held pair: the composite's finding id carried the batch, so when the
third consecutive breach re-graded it CRITICAL the store called the severity
change an activation and the ingest minted ``inc-gpu-<new finding id>`` beside
the open WARNING incident. The contract (024 ``reused_components``,
``fresh_components``): a sustained pair keeps one composite incident per
episode, a re-grade updates that incident, a fresh pair after the episode
cleared starts a new one -- and, the device-signal rule, a composite state its
GPU stopped reporting must not lend its identity to the next episode.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.gpu_metric_models import (
    GpuFindingState,
    GpuHealthFinding,
    GpuHealthSeverity,
    gpu_finding_transition,
)
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
)
from gpu_fault.models import IncidentState, RecoveryAction, WorkloadState
from gpu_fault.store import InMemoryStore, SqliteStore
from tests._builders import asgi_client, build_context, gpu_metric_batch

NOW = datetime(2026, 10, 26, 7, 54, 9, tzinfo=timezone.utc)
CLUSTER = "gpu-eks-cluster"
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
GPU = "GPU-053ab270-714e-0f16-46ad-f96f00134d8a"
RULE = "CORRECTABLE_MEMORY_DEGRADATION"
PAIR = ["ecc_sbe_volatile_total", "row_remap_correctable_total"]


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Any]:
    value = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "composite-episode.db"))
    )
    yield value
    if isinstance(value, SqliteStore):
        value.close()


def sample(name: str, value: float) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=f"DCGM_FI_DEV_{name.upper()}",
        canonical_name=name,
        value=value,
        gpu_index="1",
        gpu_uuid=GPU,
        pci_bdf="0000:01:00.0",
    )


def batch(
    batch_id: str, seconds: int, samples: list[GpuMetricSample]
) -> GpuMetricBatch:
    return gpu_metric_batch(
        batch_id,
        NOW + timedelta(seconds=seconds),
        GpuMetricSource.DCGM_EXPORTER,
        samples,
        cluster_id=CLUSTER,
        node_id=NODE,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
        evidence_ref=f"prometheus://{NODE}:9400/metrics/{batch_id}",
    )


def pair(batch_id: str, seconds: int, value: float) -> GpuMetricBatch:
    return batch(batch_id, seconds, [sample(name, value) for name in PAIR])


def composites(
    service: GpuMetricsService, *, active_only: bool
) -> list[GpuHealthFinding]:
    return [
        item
        for item in service.findings(CLUSTER, NODE, active_only=active_only)
        if item.correlation_rule_id == RULE
    ]


def test_a_sustained_pair_keeps_one_identity_and_reports_the_regrade() -> None:
    """Live shape: both counters +1 on every 15 s scrape, uninterrupted."""

    service = GpuMetricsService()
    outcomes = [service.ingest(pair(f"b{tick}", 15 * tick, tick)) for tick in range(6)]
    activations = [
        [(item.correlation_rule_id, item.severity.value, item.finding_id)]
        if item_list
        else []
        for item_list in (outcome.new_composite_findings for outcome in outcomes)
        for item in (item_list or [None])
    ]
    first = outcomes[1].new_composite_findings[0]
    assert activations == [[], [(RULE, "WARNING", first.finding_id)], [], [], [], []], (
        "one activation per episode"
    )
    assert first.finding_id == f"b1-{GPU}-composite-{RULE}"
    assert [outcome.updated_composite_findings for outcome in outcomes[:3]] == [
        [],
        [],
        [],
    ]
    regrade = outcomes[3].updated_composite_findings
    assert [(item.finding_id, item.severity.value) for item in regrade] == [
        (first.finding_id, "CRITICAL")
    ]
    assert regrade[0].automatic_action == RecoveryAction.RUN_DIAGNOSTICS.value
    assert "3 consecutive samples" in regrade[0].reason
    assert sorted(regrade[0].component_finding_ids) == sorted(
        f"b3-{GPU}-{name}" for name in PAIR
    ), "component finding ids are this batch's evidence"
    assert outcomes[3].new_findings == [], "a re-grade mints nothing"
    assert [outcome.updated_composite_findings for outcome in outcomes[4:]] == [
        [],
        [],
    ], "the grade held: nothing to report"
    assert [item.finding_id for item in composites(service, active_only=True)] == [
        first.finding_id
    ]
    assert composites(service, active_only=True)[0].severity == "CRITICAL"
    assert [item.severity for item in composites(service, active_only=False)] == [
        "CRITICAL"
    ], "the activation history keeps the episode once, at its latest grade"


def test_a_fresh_pair_after_the_episode_cleared_opens_a_new_identity() -> None:
    service = GpuMetricsService()
    service.ingest(pair("baseline", 0, 0))
    first = service.ingest(pair("episode-1", 15, 1)).new_composite_findings[0]
    cleared = service.ingest(pair("flat", 30, 1))
    assert cleared.composite_findings == [] and cleared.new_findings == []
    assert composites(service, active_only=True) == []
    again = service.ingest(pair("episode-2", 45, 2))
    (second,) = again.new_composite_findings
    assert second.finding_id == f"episode-2-{GPU}-composite-{RULE}"
    assert second.finding_id != first.finding_id
    assert second.severity == "WARNING"
    assert "1 consecutive samples" in second.reason, "the count restarts"
    assert again.updated_composite_findings == []
    assert sorted(
        item.finding_id for item in composites(service, active_only=False)
    ) == sorted([first.finding_id, second.finding_id]), (
        "two episodes, two activation history entries"
    )


def test_a_state_its_gpu_stopped_reporting_does_not_lend_its_identity() -> None:
    """The device-signal rule (b183) for composites: a composite state the
    correlation window does not reach -- the GPU stopped reporting mid-episode
    -- is the previous episode's. The breach observed when the GPU reports
    again opens a new episode at the same grade, with its own identity and a
    count of one, instead of continuing the stale one."""

    service = GpuMetricsService()
    service.ingest(pair("baseline", 0, 0))
    first = service.ingest(pair("episode-1", 15, 1)).new_composite_findings[0]
    assert "1 consecutive samples" in first.reason
    # Nothing from this GPU for 120 s (window 45 s), then both counters moved.
    returned = service.ingest(pair("returned", 15 + 120, 2))
    (second,) = returned.new_composite_findings
    assert second.finding_id == f"returned-{GPU}-composite-{RULE}"
    assert second.severity == "WARNING"
    assert "1 consecutive samples" in second.reason
    assert returned.updated_composite_findings == []
    state = service.store.get_gpu_finding_state(
        (CLUSTER, NODE, GPU, f"composite:{RULE}")
    )
    assert state is not None and state.consecutive_breaches == 1
    # The episode continues from there: the third sample of *this* episode
    # escalates, not the second.
    third = service.ingest(pair("returned-2", 15 + 135, 3))
    assert third.updated_composite_findings == [] and third.new_composite_findings == []
    fourth = service.ingest(pair("returned-3", 15 + 150, 4))
    assert [item.severity.value for item in fourth.updated_composite_findings] == [
        "CRITICAL"
    ]
    assert fourth.updated_composite_findings[0].finding_id == second.finding_id


def test_a_changed_action_is_a_new_identity() -> None:
    """A re-grade that changes the automatic action is a new recovery
    decision -- a DRAIN plan supersedes the diagnostic -- and keeps its own
    finding id (and incident, serialized behind the diagnostic as before)."""

    service = GpuMetricsService()
    limits = [
        sample("gpu_slowdown_temperature_c", 92),
        sample("gpu_shutdown_temperature_c", 95),
        sample("gpu_max_operating_temperature_c", 87),
    ]
    warning = service.ingest(
        batch(
            "thermal-warning",
            0,
            [
                *limits,
                sample("gpu_temperature_c", 88),
                sample("clock_throttle_reasons", 0x20),
            ],
        )
    )
    (first,) = warning.new_composite_findings
    assert (
        first.correlation_rule_id,
        first.severity.value,
        first.automatic_action,
    ) == ("THERMAL_STRESS", "WARNING", "RUN_DIAGNOSTICS")
    critical = service.ingest(
        batch(
            "thermal-shutdown",
            15,
            [
                *limits,
                sample("gpu_temperature_c", 96),
                sample("clock_throttle_reasons", 0x20),
            ],
        )
    )
    (second,) = critical.new_composite_findings
    assert (second.severity.value, second.automatic_action) == ("CRITICAL", "DRAIN")
    assert second.finding_id == f"thermal-shutdown-{GPU}-composite-THERMAL_STRESS"
    assert critical.updated_composite_findings == []
    held = service.ingest(
        batch(
            "thermal-shutdown-held",
            30,
            [
                *limits,
                sample("gpu_temperature_c", 96),
                sample("clock_throttle_reasons", 0x20),
            ],
        )
    )
    assert held.new_composite_findings == [] and held.updated_composite_findings == []
    assert held.composite_findings[0].finding_id == second.finding_id


def _composite(
    finding_id: str, severity: GpuHealthSeverity, seconds: int
) -> GpuHealthFinding:
    return GpuHealthFinding(
        finding_id=finding_id,
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW + timedelta(seconds=seconds),
        severity=severity,
        reason="r",
        canonical_name=f"composite:{RULE}",
        value=1,
        gpu_uuid=GPU,
        automatic_action="RUN_DIAGNOSTICS",
        finding_kind="COMPOSITE",
        correlation_rule_id=RULE,
    )


def _metric(
    finding_id: str, severity: GpuHealthSeverity, seconds: int
) -> GpuHealthFinding:
    return _composite(finding_id, severity, seconds).model_copy(
        update={
            "canonical_name": "ecc_sbe_volatile_total",
            "finding_kind": "METRIC",
            "correlation_rule_id": None,
        }
    )


def test_finding_transition_rule_shared_by_every_store() -> None:
    """The activation rule the memory, SQLite and PostgreSQL stores apply."""

    warning = _composite("e1", GpuHealthSeverity.WARNING, 15)
    state, activated = gpu_finding_transition(None, warning, warning.observed_at)
    assert (activated, state.consecutive_breaches) == (True, 1)
    held = warning.model_copy(update={"observed_at": NOW + timedelta(seconds=30)})
    state, activated = gpu_finding_transition(state, held, held.observed_at)
    assert (activated, state.consecutive_breaches) == (False, 2)
    critical = _composite("e1", GpuHealthSeverity.CRITICAL, 45)
    state, activated = gpu_finding_transition(state, critical, critical.observed_at)
    assert (activated, state.consecutive_breaches) == (True, 3), (
        "a re-grade of the open episode activates (the caller tells it apart by the id)"
    )
    fresh = _composite("e2", GpuHealthSeverity.CRITICAL, 180)
    state, activated = gpu_finding_transition(state, fresh, fresh.observed_at)
    assert (activated, state.consecutive_breaches) == (True, 1), (
        "a new composite identity is a new episode even at the same grade"
    )
    cleared, activated = gpu_finding_transition(
        state, None, NOW + timedelta(seconds=195)
    )
    assert (activated, cleared.finding, cleared.consecutive_breaches) == (
        False,
        None,
        0,
    )
    state, activated = gpu_finding_transition(
        cleared, fresh, NOW + timedelta(seconds=210)
    )
    assert (activated, state.consecutive_breaches) == (True, 1)
    # METRIC findings are stamped per batch: their id says nothing about identity.
    metric = _metric("m1", GpuHealthSeverity.WARNING, 15)
    state, activated = gpu_finding_transition(None, metric, metric.observed_at)
    assert (activated, state.consecutive_breaches) == (True, 1)
    renamed = _metric("m2", GpuHealthSeverity.WARNING, 30)
    state, activated = gpu_finding_transition(state, renamed, renamed.observed_at)
    assert (activated, state.consecutive_breaches) == (False, 2)
    graded = _metric("m3", GpuHealthSeverity.CRITICAL, 45)
    state, activated = gpu_finding_transition(state, graded, graded.observed_at)
    assert (activated, state.consecutive_breaches) == (True, 3)


def test_stores_apply_the_transition_rule(store: Any) -> None:
    key = (CLUSTER, NODE, GPU, f"composite:{RULE}")
    warning = _composite("e1", GpuHealthSeverity.WARNING, 15)
    assert store.update_gpu_findings([(key, warning, warning.observed_at)]) == [True]
    critical = _composite("e1", GpuHealthSeverity.CRITICAL, 45)
    assert store.update_gpu_findings([(key, critical, critical.observed_at)]) == [True]
    state: GpuFindingState = store.get_gpu_finding_state(key)
    assert state.consecutive_breaches == 2 and state.finding.severity == "CRITICAL"
    assert store.update_gpu_findings([(key, critical, critical.observed_at)]) == [
        False
    ], "a stale or repeated observation is not an activation"
    fresh = _composite("e2", GpuHealthSeverity.CRITICAL, 180)
    assert store.update_gpu_findings([(key, fresh, fresh.observed_at)]) == [True]
    state = store.get_gpu_finding_state(key)
    assert state.consecutive_breaches == 1
    history = [
        item.finding_id
        for item in store.list_gpu_findings(CLUSTER, NODE, active_only=False)
    ]
    assert sorted(history) == ["e1", "e2"]
    assert [
        item.severity
        for item in store.list_gpu_findings(CLUSTER, NODE, active_only=False)
        if item.finding_id == "e1"
    ] == ["CRITICAL"]


def _post(context: Any, payload: GpuMetricBatch) -> dict[str, Any]:
    async def request() -> dict[str, Any]:
        async with asgi_client(context) as client:
            response = await client.post(
                GPU_METRICS_PATH, json=payload.model_dump(mode="json")
            )
            assert response.status_code == 200, response.text
            return dict(response.json())

    return asyncio.run(request())


def _composite_incidents(store: Any) -> list[Any]:
    return sorted(
        (
            incident
            for incident in store.list_incidents_by_state(CLUSTER, set(IncidentState))
            if incident.event_id.endswith(f"-composite-{RULE}")
        ),
        key=lambda incident: incident.created_at,
    )


def _marker(store: Any, event_id: str) -> Any:
    (marker,) = [
        item for item in store.list_markers() if item.marker_id == f"marker-{event_id}"
    ]
    return marker


def test_ingest_keeps_one_incident_per_episode_and_records_the_regrade(
    store: Any,
) -> None:
    context = build_context(store=store)
    _post(context, pair("b0", 0, 0))
    first = _post(context, pair("b1", 15, 1))
    (activated,) = first["new_composite_findings"]
    event_id = f"gpu-{activated['finding_id']}"
    incident = store.get_incident_by_event(event_id)
    assert incident is not None
    assert incident.effective_action is RecoveryAction.RUN_DIAGNOSTICS
    assert _marker(store, event_id).severity.value == "warning"
    assert _post(context, pair("b2", 30, 2))["new_findings"] == []

    third = _post(context, pair("b3", 45, 3))
    assert third["new_findings"] == [] and third["new_composite_findings"] == []
    assert [item["severity"] for item in third["updated_composite_findings"]] == [
        "CRITICAL"
    ]
    incidents = _composite_incidents(store)
    assert [item.incident_id for item in incidents] == [incident.incident_id], (
        "the re-grade updated the open incident instead of minting a second one"
    )
    updated = incidents[0]
    assert updated.state is IncidentState.ACTION_PENDING
    assert updated.effective_action is RecoveryAction.RUN_DIAGNOSTICS
    assert updated.workflow_request_id == incident.workflow_request_id
    escalations = [
        reason
        for reason in updated.reasons
        if reason.startswith(f"composite:{RULE} escalated to CRITICAL: ")
    ]
    assert len(escalations) == 1 and "3 consecutive samples" in escalations[0]
    assert updated.updated_at >= incident.updated_at
    assert _marker(store, event_id).severity.value == "critical"
    assert _marker(store, event_id).incident_id == incident.incident_id

    # A replay of the batch that re-graded reaches the same record once.
    replay = _post(context, pair("b3", 45, 3))
    assert replay["duplicate"] is True
    assert _composite_incidents(store)[0].reasons == updated.reasons

    # The grade holds: nothing is re-recorded on later scrapes.
    _post(context, pair("b4", 60, 4))
    assert _composite_incidents(store)[0].reasons == updated.reasons

    # The pair clears, then breaches again: a new episode, a new incident.
    cleared = _post(context, pair("flat", 75, 4))
    assert cleared["composite_findings"] == []
    again = _post(context, pair("b6", 90, 5))
    (second,) = again["new_composite_findings"]
    assert second["finding_id"] != activated["finding_id"]
    incidents = _composite_incidents(store)
    assert [item.event_id for item in incidents] == [
        event_id,
        f"gpu-{second['finding_id']}",
    ]
    assert incidents[1].reasons[0].startswith("multiple correctable memory"), (
        "the new episode's incident opens with its own composite reason"
    )


@pytest.mark.parametrize("kernel_first", [True, False])
def test_a_repeated_kernel_xid_continues_the_open_pcie_episode(
    store: Any, kernel_first: bool
) -> None:
    """The kernel-event path shares the identity rule: a second XID 32/79 on
    the GPU while its PCIE_XID_LINK_FAILURE composite holds re-confirms the
    open episode (same finding id, the new event kept as evidence) and is not a
    second composite or incident; the next scrape continues it as well."""

    from tests.metrics import test_kernel_pcie_composite as pcie

    service = GpuMetricsService(store=store)
    event = pcie.kernel()
    (first,) = pcie.correlate(service, event, kernel_first=kernel_first)
    opened_by = "breach" if kernel_first else f"kernel-xid-{event.event_id}"
    assert first.finding_id == f"{opened_by}-GPU-a-composite-{pcie.RULE}"
    key = ("cluster-a", "node-a", "GPU-a", f"composite:{pcie.RULE}")

    again = pcie.kernel("kernel-again", 30)
    pcie.persist_xid(store, again)
    assert service.correlate_kernel_xid(again) == [], (
        "a repeated XID while the composite holds is not a new activation"
    )
    state = store.get_gpu_finding_state(key)
    assert state is not None and state.finding is not None
    assert state.finding.finding_id == first.finding_id
    assert again.evidence_ref in state.finding.component_evidence_refs
    assert state.consecutive_breaches == 2

    held = service.ingest(pcie.batch("held", 45, 1010))
    assert held.new_composite_findings == [] and held.updated_composite_findings == []
    assert [item.finding_id for item in held.composite_findings] == [first.finding_id]
    assert [item.finding_id for item in pcie.composites(service)] == [first.finding_id]
    # The opening event's own replay keeps its idempotent answer: nothing for
    # an episode a batch opened, the stored finding (its id) for one it opened.
    replay = service.correlate_kernel_xid(event)
    assert [item.finding_id for item in replay] == (
        [] if kernel_first else [first.finding_id]
    )
