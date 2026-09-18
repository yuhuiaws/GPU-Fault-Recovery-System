from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.fleet import AgentRecord
from gpu_fault.gpu_metrics import (
    GpuHealthFinding,
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
)
from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.models import RecoveryAction, WorkloadState
from gpu_fault.nvidia_logs import NvidiaKernelLogEvent, NvidiaLogNormalizer
from gpu_fault.policy import FaultPolicyDecision, GpuFaultPolicyEngine, XidEvent
from gpu_fault.store import InMemoryStore, NotFoundError, SqliteStore
from gpu_fault.telemetry import EvidenceKind
from gpu_fault.xid_correlation import XidCorrelationCoordinator
from tests._builders import asgi_client, build_context

NOW = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
RULE = "PCIE_XID_LINK_FAILURE"


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Any]:
    value = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "pcie.db"))
    )
    yield value
    if isinstance(value, SqliteStore):
        value.close()


def metric(
    value: float,
    *,
    name: str = "pcie_replay_total",
    gpu_uuid: str = "GPU-a",
    pci_bdf: str = "0000:01:00.0",
    boot_id: str | None = None,
) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=f"DCGM_FI_DEV_{name.upper()}",
        canonical_name=name,
        value=value,
        gpu_uuid=gpu_uuid,
        pci_bdf=pci_bdf,
        labels={"source_boot_id": boot_id} if boot_id else {},
    )


def batch(
    batch_id: str,
    seconds: int,
    value: float = 1005,
    *,
    samples: list[GpuMetricSample] | None = None,
    **updates: Any,
) -> GpuMetricBatch:
    return GpuMetricBatch(
        batch_id=batch_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW + timedelta(seconds=seconds),
        source=GpuMetricSource.DCGM_EXPORTER,
        samples=samples if samples is not None else [metric(value)],
        product="H100",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training-a"],
        evidence_ref=f"prometheus://node-a/metrics/{batch_id}",
    ).model_copy(update=updates)


def kernel_record(
    record_id: str = "kernel-a", seconds: int = 20, xid: int = 79
) -> NvidiaKernelLogEvent:
    return NvidiaKernelLogEvent(
        record_id=record_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW + timedelta(seconds=seconds),
        collected_at=NOW + timedelta(seconds=seconds),
        source_boot_id="boot-a",
        source_monotonic_us=1_000_000 + seconds * 1_000,
        message=f"NVRM: Xid (PCI:0000:01:00): {xid}, PCIe link failure",
        product="H100",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training-a"],
        evidence_ref=f"kmsg://node-a/boot-a/{record_id}",
    )


def kernel(
    record_id: str = "kernel-a", seconds: int = 20, xid: int = 79, **updates: Any
) -> XidEvent:
    event = (
        NvidiaLogNormalizer()
        .normalize_kernel(kernel_record(record_id, seconds, xid))
        .xid_events[0]
    )
    return event.model_copy(update={"gpu_uuid": "GPU-a", **updates})


def persist_xid(store: Any, event: XidEvent) -> FaultPolicyDecision:
    return XidCorrelationCoordinator(
        store,
        GpuFaultPolicyEngine(),
        lambda _event, decision: decision,
        now=lambda: NOW,
    ).ingest(event)


def composites(service: GpuMetricsService) -> list[GpuHealthFinding]:
    return [
        finding
        for finding in service.findings("cluster-a", "node-a")
        if finding.correlation_rule_id == RULE
    ]


def correlate(
    service: GpuMetricsService, event: XidEvent, *, kernel_first: bool
) -> list[GpuHealthFinding]:
    service.ingest(batch("baseline", 0, 1000))
    if kernel_first:
        persist_xid(service.store, event)
        assert service.correlate_kernel_xid(event) == [], (
            "a counter baseline is not PCIe threshold evidence"
        )
        return service.ingest(batch("breach", 15)).new_composite_findings
    service.ingest(batch("breach", 15))
    persist_xid(service.store, event)
    return service.correlate_kernel_xid(event)


@pytest.mark.parametrize("kernel_first", [True, False])
@pytest.mark.parametrize("seconds", [10, 20])
@pytest.mark.parametrize("xid", [32, 79])
@pytest.mark.parametrize("gpu_uuid", ["GPU-a", None])
def test_both_arrival_orders_use_normalized_kernel_evidence(
    store: Any, kernel_first: bool, seconds: int, xid: int, gpu_uuid: str | None
) -> None:
    service = GpuMetricsService(store=store)
    event = kernel(seconds=seconds, xid=xid, gpu_uuid=gpu_uuid)

    findings = correlate(service, event, kernel_first=kernel_first)

    assert len(findings) == 1, (
        "both evidence arrival orders must activate one composite"
    )
    finding = findings[0]
    assert finding.correlation_rule_id == RULE
    assert finding.automatic_action == RecoveryAction.DRAIN
    assert finding.component_metrics == ["pcie_replay_total", "kernel_xid"]
    assert event.evidence_ref in finding.component_evidence_refs
    assert finding.gpu_uuid == "GPU-a"
    assert store.get_xid_event(event.event_id).event_source == "KERNEL_LOG"
    decision = store.get_xid_policy_decision(event.event_id)
    assert decision.action == (
        RecoveryAction.RESTART_WORKLOAD if xid == 32 else RecoveryAction.REBOOT_NODE
    ), "the XID's independent NVIDIA policy must be retained"
    assert all(
        item.sample.canonical_name != "xid_last_error"
        for item in service.latest("cluster-a", "node-a")
    ), "kernel evidence must not become a DCGM gauge"


@pytest.mark.parametrize("kernel_first", [True, False])
@pytest.mark.parametrize(
    "updates",
    [
        {"gpu_uuid": "GPU-other"},
        {"pci_bdf": "0000:02:00.0"},
        {"pci_bdf": "0001:01:00.0"},
        {"pci_bdf": "0000:01:00.1"},
        {"pci_bdf": "not-a-bdf"},
        {"cluster_id": "cluster-other"},
        {"node_id": "node-other"},
        {"event_source": "training-log"},
        {"event_source": None},
        {"synthetic": True},
        {"xid": 31},
        {"product": "unknown"},
        {"runtime_profile_version": "simulated-other"},
        {"gpu_uuid": None, "pci_bdf": None},
    ],
    ids=[
        "gpu",
        "pci-slot",
        "pci-domain",
        "pci-function",
        "invalid-pci",
        "cluster",
        "node",
        "untrusted-source",
        "unknown-source",
        "synthetic",
        "unrelated-xid",
        "unknown-policy-product",
        "profile-drift",
        "unknown-device",
    ],
)
def test_mismatched_or_untrusted_evidence_does_not_correlate(
    store: Any, kernel_first: bool, updates: dict[str, Any]
) -> None:
    service = GpuMetricsService(store=store)

    result = correlate(service, kernel(**updates), kernel_first=kernel_first)

    assert result == [], (
        "untrusted or mismatched evidence cannot activate the composite"
    )
    assert composites(service) == []


@pytest.mark.parametrize("kernel_first", [True, False])
@pytest.mark.parametrize("seconds", [-31, 61])
def test_kernel_evidence_outside_correlation_window_is_rejected(
    store: Any, kernel_first: bool, seconds: int
) -> None:
    service = GpuMetricsService(store=store)

    result = correlate(service, kernel(seconds=seconds), kernel_first=kernel_first)

    assert result == [], "the 45-second window applies in both time directions"
    assert composites(service) == []


@pytest.mark.parametrize("seconds", [-30, 60])
def test_correlation_window_includes_its_exact_boundary(
    store: Any, seconds: int
) -> None:
    service = GpuMetricsService(store=store)

    result = correlate(service, kernel(seconds=seconds), kernel_first=False)

    assert len(result) == 1, "evidence exactly 45 seconds apart remains in the window"


@pytest.mark.parametrize("kernel_first", [True, False])
def test_source_timestamp_cannot_hide_stale_evidence(
    store: Any, kernel_first: bool
) -> None:
    service = GpuMetricsService(store=store)

    result = correlate(
        service,
        kernel(source_event_time=NOW - timedelta(minutes=5)),
        kernel_first=kernel_first,
    )

    assert result == [], "a current envelope cannot make old source evidence fresh"


@pytest.mark.parametrize("kernel_first", [True, False])
def test_unknown_source_timezone_rejects_correlation_without_losing_xid_policy(
    store: Any, kernel_first: bool
) -> None:
    service = GpuMetricsService(store=store)
    event = kernel(source_event_time=NOW.replace(tzinfo=None))

    result = correlate(service, event, kernel_first=kernel_first)

    assert result == [], "a source timestamp without a timezone cannot prove freshness"
    assert (
        store.get_xid_policy_decision(event.event_id).action
        is RecoveryAction.REBOOT_NODE
    )


@pytest.mark.parametrize("kernel_first", [True, False])
@pytest.mark.parametrize(
    "marker_updates",
    [
        {"trusted": False},
        {"cluster_id": "cluster-other"},
        {"source_boot_id": "boot-other"},
        {"event_source": "DCGM_EXPORTER"},
    ],
    ids=["untrusted", "cluster", "boot", "source"],
)
def test_kernel_composite_requires_the_matching_trusted_policy_marker(
    store: Any, kernel_first: bool, marker_updates: dict[str, Any]
) -> None:
    service = GpuMetricsService(store=store)
    event = kernel()
    service.ingest(batch("baseline", 0, 1000))
    if not kernel_first:
        service.ingest(batch("breach", 15))
    decision = persist_xid(store, event)
    store.save_xid_policy_decision(
        decision.model_copy(
            update={"marker": decision.marker.model_copy(update=marker_updates)}
        )
    )

    findings = (
        service.ingest(batch("breach", 15)).new_composite_findings
        if kernel_first
        else service.correlate_kernel_xid(event)
    )

    assert findings == [], "a source label alone is not a trusted policy decision"


def test_unclassified_persisted_event_is_not_composite_evidence(store: Any) -> None:
    service = GpuMetricsService(store=store)
    event = kernel()
    store.save_xid_event_if_absent(event)
    service.ingest(batch("baseline", 0, 1000))

    result = service.ingest(batch("breach", 15))

    assert result.composite_findings == [], "the saved event needs a policy decision"
    assert service.correlate_kernel_xid(event) == []


@pytest.mark.parametrize("kernel_first", [True, False])
@pytest.mark.parametrize("boot_source", ["sample", "inventory"])
def test_kernel_and_pcie_must_not_cross_known_boots(
    store: Any, kernel_first: bool, boot_source: str
) -> None:
    service = GpuMetricsService(store=store)
    event = kernel()
    service.ingest(batch("baseline", 0, 1000))
    if boot_source == "inventory":
        store.save_gpu_inventory_snapshot(
            GpuInventorySnapshot(
                cluster_id="cluster-a",
                node_id="node-a",
                observed_at=NOW,
                source=GpuMetricSource.NVIDIA_SMI,
                source_boot_id="boot-other",
                devices=[
                    GpuInventoryDevice(
                        gpu_index=0, gpu_uuid="GPU-a", pci_bdf="0000:01:00.0"
                    )
                ],
            )
        )
    breach = batch(
        "breach",
        15,
        samples=[
            metric(1005, boot_id="boot-other" if boot_source == "sample" else None)
        ],
    )
    if kernel_first:
        persist_xid(store, event)
        result = service.ingest(breach).new_composite_findings
    else:
        service.ingest(breach)
        persist_xid(store, event)
        result = service.correlate_kernel_xid(event)

    assert result == [], "known boot disagreement must reject cross-source correlation"


def test_pci_only_kernel_evidence_must_identify_one_gpu(store: Any) -> None:
    service = GpuMetricsService(store=store)
    event = kernel(gpu_uuid=None)
    persist_xid(store, event)
    service.ingest(
        batch(
            "baseline",
            0,
            samples=[
                metric(1000),
                metric(1000, gpu_uuid="GPU-b", pci_bdf="0000:01:00.1"),
            ],
        )
    )

    result = service.ingest(
        batch(
            "breach",
            15,
            samples=[
                metric(1005),
                metric(1005, gpu_uuid="GPU-b", pci_bdf="0000:01:00.1"),
            ],
        )
    )

    assert result.composite_findings == [], "a slot without a function may be ambiguous"
    assert service.correlate_kernel_xid(event) == []


def test_current_inventory_disagreement_rejects_old_device_evidence(store: Any) -> None:
    service = GpuMetricsService(store=store)
    store.save_gpu_inventory_snapshot(
        GpuInventorySnapshot(
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=NOW,
            source=GpuMetricSource.NVIDIA_SMI,
            source_boot_id="boot-a",
            devices=[
                GpuInventoryDevice(
                    gpu_index=0, gpu_uuid="GPU-replacement", pci_bdf="0000:01:00.0"
                )
            ],
        )
    )

    findings = correlate(service, kernel(), kernel_first=False)

    assert findings == [], "the current inventory must not identify a different GPU"


@pytest.mark.parametrize("boot_id", ["boot-a", "boot-other", None])
def test_kernel_correlation_checks_available_agent_boot(
    store: Any, boot_id: str | None
) -> None:
    service = GpuMetricsService(store=store)
    store.save_agent(
        AgentRecord(
            cluster_id="cluster-a",
            node_id="node-a",
            endpoint="https://node-a:8443",
            agent_version="test",
            artifact_sha256="a" * 64,
            policy_version="610",
            runtime_profile_version="simulated-v1",
            config_digest="test-config",
            allowed_operations=[],
            boot_id=boot_id,
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
    )

    findings = correlate(service, kernel(), kernel_first=False)

    assert len(findings) == (0 if boot_id == "boot-other" else 1), (
        "available agent boot evidence must agree with the kernel and PCIe scope"
    )


def test_gpu_index_alone_cannot_bind_a_kernel_xid(store: Any) -> None:
    service = GpuMetricsService(store=store)
    event = kernel()
    persist_xid(store, event)
    unbound = GpuMetricSample(
        metric_name="DCGM_FI_DEV_PCIE_REPLAY_COUNTER",
        canonical_name="pcie_replay_total",
        gpu_index="0",
        value=1000,
    )
    service.ingest(batch("baseline", 0, samples=[unbound]))
    result = service.ingest(
        batch(
            "unbound",
            15,
            samples=[
                unbound.model_copy(update={"value": 1005}),
                GpuMetricSample(metric_name="node", canonical_name="node", value=1),
            ],
        )
    )

    assert result.composite_findings == [], (
        "a GPU index is not a physical kernel identity"
    )
    assert service.correlate_kernel_xid(event) == []


def observe_xid_reads(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[tuple[str, str, datetime, datetime]], list[str]]:
    history_reads = []
    policy_reads = []
    list_events = store.list_xid_events
    get_policy = store.get_xid_policy_decision

    def history(
        cluster_id: str,
        node_id: str,
        *,
        observed_after: datetime,
        observed_before: datetime,
    ) -> list[XidEvent]:
        history_reads.append((cluster_id, node_id, observed_after, observed_before))
        return list_events(
            cluster_id,
            node_id,
            observed_after=observed_after,
            observed_before=observed_before,
        )

    def policy(event_id: str) -> FaultPolicyDecision | None:
        policy_reads.append(event_id)
        return get_policy(event_id)

    monkeypatch.setattr(store, "list_xid_events", history)
    monkeypatch.setattr(store, "get_xid_policy_decision", policy)
    return history_reads, policy_reads


def test_healthy_scrapes_do_not_query_historical_xid_evidence(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = GpuMetricsService(store=store)
    persist_xid(store, kernel())
    history, policies = observe_xid_reads(store, monkeypatch)

    service.ingest(batch("baseline", 0, 1000))
    service.ingest(batch("healthy-pcie", 15, 1002))
    service.ingest(
        batch("healthy-gpu", 20, samples=[metric(40, name="gpu_temperature_c")])
    )

    assert history == [], "healthy scrapes and counter baselines need no XID history"
    assert policies == [], "healthy scrapes must not load per-event XID policies"


def test_kernel_without_pcie_breach_does_not_reload_xid_history(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = GpuMetricsService(store=store)
    event = kernel()
    persist_xid(store, event)
    event_reads = []
    read_event = store.get_xid_event

    def record(event_id: str) -> XidEvent:
        event_reads.append(event_id)
        return read_event(event_id)

    monkeypatch.setattr(store, "get_xid_event", record)
    history, policies = observe_xid_reads(store, monkeypatch)
    assert service.correlate_kernel_xid(event) == []
    service.ingest(batch("baseline", 0, 1000))
    assert service.correlate_kernel_xid(event) == []

    assert event_reads == [], "no PCIe breach means no redundant persisted-XID read"
    assert history == []
    assert policies == []


def test_other_gpu_scrapes_do_not_query_an_untouched_pcie_breach(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = GpuMetricsService(store=store)
    correlate(service, kernel(), kernel_first=True)
    history, policies = observe_xid_reads(store, monkeypatch)

    service.ingest(
        batch(
            "other-gpu",
            20,
            samples=[
                metric(1000, gpu_uuid="GPU-b", pci_bdf="0000:02:00.0"),
                metric(
                    40,
                    name="gpu_temperature_c",
                    gpu_uuid="GPU-b",
                    pci_bdf="0000:02:00.0",
                ),
            ],
        )
    )

    assert history == [], "only breached GPUs touched by this batch need correlation"
    assert policies == []


def test_multiple_pcie_breaches_share_one_scoped_window_query(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = GpuMetricsService(store=store)
    relevant = [
        kernel(),
        kernel("kernel-b", 10, gpu_uuid="GPU-b", pci_bdf="0000:02:00.0"),
    ]
    for event in [
        *relevant,
        kernel("kernel-unrelated", 10, gpu_uuid="GPU-c", pci_bdf="0000:03:00.0"),
        kernel("kernel-old", -31),
        kernel("kernel-future", 61),
        kernel("kernel-other-node", 10, node_id="node-other"),
        kernel("kernel-other-cluster", 10, cluster_id="cluster-other"),
    ]:
        persist_xid(store, event)
    service.ingest(
        batch(
            "baseline",
            0,
            samples=[
                metric(1000),
                metric(1000, gpu_uuid="GPU-b", pci_bdf="0000:02:00.0"),
            ],
        )
    )
    history, policies = observe_xid_reads(store, monkeypatch)

    result = service.ingest(
        batch(
            "breaches",
            15,
            samples=[
                metric(1005),
                metric(1005, gpu_uuid="GPU-b", pci_bdf="0000:02:00.0"),
            ],
        )
    )

    assert len(result.new_composite_findings) == 2
    assert history == [
        (
            "cluster-a",
            "node-a",
            NOW - timedelta(seconds=30),
            NOW + timedelta(seconds=60),
        )
    ], "all breached GPUs share one bounded same-node history query"
    assert sorted(policies) == sorted(event.event_id for event in relevant), (
        "time, node, cluster and device filtering precede per-event policy reads"
    )


def test_pcie_counter_delta_must_not_span_known_boots(store: Any) -> None:
    service = GpuMetricsService(store=store)
    event = kernel()
    persist_xid(store, event)
    service.ingest(batch("old-boot", 0, samples=[metric(1000, boot_id="boot-old")]))

    result = service.ingest(
        batch("new-boot", 15, samples=[metric(1005, boot_id="boot-a")])
    )

    assert result.findings == [], "a new boot establishes a new PCIe counter baseline"
    assert result.new_composite_findings == []
    assert service.correlate_kernel_xid(event) == []


@pytest.mark.parametrize("kernel_first", [True, False])
def test_newer_healthy_pcie_sample_clears_even_when_kernel_clock_is_ahead(
    store: Any, kernel_first: bool
) -> None:
    service = GpuMetricsService(store=store)
    event = kernel(seconds=20)
    correlate(service, event, kernel_first=kernel_first)

    service.ingest(batch("healthy", 18))

    assert composites(service) == [], (
        "healthy PCIe evidence newer than the breach clears an ahead-of-scrape XID join"
    )
    assert service.correlate_kernel_xid(event) == [], (
        "replaying the ahead-of-scrape XID cannot restore cleared threshold evidence"
    )


def test_older_unrelated_metric_cannot_clear_a_newer_kernel_composite(
    store: Any,
) -> None:
    service = GpuMetricsService(store=store)
    correlate(service, kernel(seconds=20), kernel_first=False)

    service.ingest(
        batch("unrelated-old", -30, samples=[metric(40, name="gpu_temperature_c")])
    )

    assert len(composites(service)) == 1, (
        "an unrelated older metric is not a newer healthy PCIe observation"
    )


def test_unrelated_scrape_cannot_fence_delayed_kernel_evidence(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = GpuMetricsService(store=store)
    correlate(service, kernel("earlier-kernel", 35), kernel_first=False)
    service.ingest(batch("later-pcie-breach", 75, 1025))
    before = composites(service)
    history, policies = observe_xid_reads(store, monkeypatch)

    service.ingest(
        batch("temperature-only", 82, samples=[metric(40, name="gpu_temperature_c")])
    )

    assert composites(service) == before, (
        "an unrelated scrape cannot close an active PCIe episode or advance its fence"
    )
    assert history == [] and policies == [], (
        "a batch with no PCIe/XID samples must not query historical XID evidence"
    )
    delayed = kernel("delayed-kernel", 76)
    persist_xid(store, delayed)
    service.correlate_kernel_xid(delayed)
    findings = composites(service)
    assert len(findings) == 1, "delayed valid kernel evidence must retain the PCIe join"
    assert delayed.evidence_ref in findings[0].component_evidence_refs, (
        "an unrelated later scrape must not reject valid in-window kernel evidence"
    )


@pytest.mark.parametrize("value", [1000, 1002, 5])
def test_kernel_xid_without_pcie_threshold_evidence_does_not_create_composite(
    store: Any, value: float
) -> None:
    service = GpuMetricsService(store=store)
    event = kernel()
    persist_xid(store, event)

    assert service.correlate_kernel_xid(event) == [], (
        "kernel XID alone is not PCIe evidence"
    )
    baseline = service.ingest(batch("baseline", 0, 1000))
    result = service.ingest(batch("not-a-breach", 15, value))

    assert baseline.new_composite_findings == [], (
        "an initial counter is only a baseline"
    )
    assert result.new_composite_findings == [], "no rate breach means no composite"
    assert service.correlate_kernel_xid(event) == []


@pytest.mark.parametrize("xid", [32, 79])
def test_dcgm_gauge_only_composite_remains_supported(store: Any, xid: int) -> None:
    service = GpuMetricsService(store=store)
    service.ingest(
        batch("baseline", 0, samples=[metric(1000), metric(0, name="xid_last_error")])
    )

    result = service.ingest(
        batch("breach", 15, samples=[metric(1005), metric(xid, name="xid_last_error")])
    )

    assert len(result.new_composite_findings) == 1
    assert result.new_composite_findings[0].component_metrics == [
        "pcie_replay_total",
        "xid_last_error",
    ]
    assert result.xid_events[0].xid == xid


def test_duplicate_kernel_replays_cannot_rearm_a_cleared_composite(store: Any) -> None:
    service = GpuMetricsService(store=store)
    event = kernel()
    first = correlate(service, event, kernel_first=False)[0]

    replayed = service.correlate_kernel_xid(event)
    forged_replay = event.model_copy(
        update={"observed_at": NOW + timedelta(seconds=50), "gpu_uuid": "GPU-other"}
    )
    assert replayed == [first], "an exact retry reuses the original finding identity"
    assert service.correlate_kernel_xid(forged_replay) == [first], (
        "changed replay fields must not replace the persisted evidence"
    )
    assert service.ingest(batch("breach", 15)).duplicate, (
        "an exact scrape replay must use the persisted batch result"
    )

    service.ingest(batch("clear", 30))
    assert composites(service) == [], "healthy PCIe metrics clear the composite"
    assert service.correlate_kernel_xid(event) == []
    rearmed_pcie = service.ingest(batch("new-pcie-breach", 45, 1010))
    assert rearmed_pcie.new_composite_findings == [], (
        "a pre-clear kernel event cannot activate the next PCIe episode"
    )
    assert service.correlate_kernel_xid(forged_replay) == []

    fresh = kernel("kernel-fresh", 50)
    persist_xid(store, fresh)
    new_findings = service.correlate_kernel_xid(fresh)
    assert len(new_findings) == 1, "fresh XID plus the new PCIe breach rearms normally"
    assert new_findings[0].finding_id != first.finding_id
    history = [
        item
        for item in service.findings("cluster-a", "node-a", active_only=False)
        if item.correlation_rule_id == RULE
    ]
    assert len(history) == 2, "duplicate replays must not add activation history"


def test_kernel_correlation_uses_persisted_state_after_sqlite_restart(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "restart.db")
    event = kernel()
    first = SqliteStore(path)
    service = GpuMetricsService(store=first)
    service.ingest(batch("baseline", 0, 1000))
    persist_xid(first, event)
    first.close()
    reopened = SqliteStore(path)
    try:
        resumed = GpuMetricsService(store=reopened)
        result = resumed.ingest(batch("breach", 15))
        assert len(result.new_composite_findings) == 1, (
            "kernel evidence must survive process restart before the PCIe sample"
        )
        assert resumed.correlate_kernel_xid(event) == [], (
            "the second arrival cannot activate a persisted composite twice"
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("kernel_first", [True, False])
def test_http_arrival_orders_keep_xid_policy_and_use_node_health_arbitration(
    store: Any, monkeypatch: pytest.MonkeyPatch, kernel_first: bool
) -> None:
    context = build_context(store=store)
    calls: list[NodeHealthFinding] = []
    ingest_node_health = context.orchestrator.ingest_node_health

    def record(finding: NodeHealthFinding) -> Any:
        calls.append(finding)
        return ingest_node_health(finding)

    monkeypatch.setattr(context.orchestrator, "ingest_node_health", record)
    raw = kernel_record()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            baseline = await client.post(
                GPU_METRICS_PATH,
                json=batch("baseline", 0, 1000).model_dump(mode="json"),
            )
            assert baseline.status_code == 200
            requests = [
                (NVIDIA_KERNEL_PATH, raw.model_dump(mode="json")),
                (GPU_METRICS_PATH, batch("breach", 15).model_dump(mode="json")),
            ]
            if not kernel_first:
                requests.reverse()
            for path, payload in requests:
                response = await client.post(path, json=payload)
                assert response.status_code == 200, (
                    "both real ingestion routes must accept"
                )
                if path == NVIDIA_KERNEL_PATH:
                    assert response.json()["decisions"][0]["action"] == "REBOOT_NODE"
            retry = await client.post(
                NVIDIA_KERNEL_PATH, json=raw.model_dump(mode="json")
            )
            assert retry.status_code == 200
            assert retry.json()["decisions"][0]["duplicate"] is True

    asyncio.run(scenario())

    findings = composites(context.gpu_metrics)
    assert len(findings) == 1
    composite = findings[0]
    matching_calls = [item for item in calls if item.metric_name == f"composite:{RULE}"]
    assert matching_calls, (
        "an existing XID marker must not bypass composite arbitration"
    )
    assert all(
        item.recommended_action is RecoveryAction.DRAIN for item in matching_calls
    ), "both arrival paths must retain the composite's DRAIN action"
    assert len({item.event_id for item in matching_calls}) == 1
    assert store.get_incident_by_event(f"gpu-{composite.finding_id}") is not None
    assert (
        raw.evidence_ref
        in matching_calls[0].diagnostic_parameters["component_evidence_refs"]
    )
    records = store.list_raw_evidence("cluster-a", node_id="node-a", limit=20)
    assert sum(item.kind is EvidenceKind.GPU_METRICS for item in records) == 2, (
        "only the two actual scrape batches may be GPU_METRICS evidence"
    )
    assert sum(item.kind is EvidenceKind.NVIDIA_KERNEL for item in records) == 1


def test_kernel_composite_activation_rolls_back_with_incident_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteStore(str(tmp_path / "atomic.db"))
    context = build_context(store=store)
    raw = kernel_record()
    event = NvidiaLogNormalizer().normalize_kernel(raw).xid_events[0]
    original = context.orchestrator.ingest_node_health

    def fail_composite(finding: NodeHealthFinding) -> Any:
        if finding.metric_name == f"composite:{RULE}":
            raise RuntimeError("composite incident write failed")
        return original(finding)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            for item in [batch("baseline", 0, 1000), batch("breach", 15)]:
                response = await client.post(
                    GPU_METRICS_PATH, json=item.model_dump(mode="json")
                )
                assert response.status_code == 200
            monkeypatch.setattr(
                context.orchestrator, "ingest_node_health", fail_composite
            )
            with pytest.raises(RuntimeError, match="composite incident write failed"):
                await client.post(NVIDIA_KERNEL_PATH, json=raw.model_dump(mode="json"))
            assert composites(context.gpu_metrics) == [], (
                "a failed incident write must not leave the activation latched"
            )
            with pytest.raises(NotFoundError):
                store.get_xid_event(event.event_id)
            monkeypatch.setattr(context.orchestrator, "ingest_node_health", original)
            response = await client.post(
                NVIDIA_KERNEL_PATH, json=raw.model_dump(mode="json")
            )
            assert response.status_code == 200
            assert len(composites(context.gpu_metrics)) == 1

    try:
        asyncio.run(scenario())
    finally:
        store.close()
