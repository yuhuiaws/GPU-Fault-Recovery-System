"""PCIe/kernel correlation and atomicity on an explicitly allocated PostgreSQL."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Event
from typing import Any

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.gpu_metrics import GpuMetricsService
from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.nvidia_logs import NvidiaLogNormalizer
from gpu_fault.store import NotFoundError, PostgresStore
from tests._builders import asgi_client, build_context
from tests.metrics import test_kernel_pcie_composite as cases
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires an isolated PostgreSQL test database"
)


@pytest.fixture(params=["legacy", "dedicated"])
def native_store(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[PostgresStore]:
    with monkeypatch.context() as setup_environment:
        setup_environment.setenv("GPU_FAULT_POSTGRES_HOT_STATE_MODE", "legacy")
        with closing(postgres_store_instance()) as prepared:
            next(prepared)
    monkeypatch.setenv("GPU_FAULT_POSTGRES_HOT_STATE_MODE", request.param)
    if POSTGRES_URL is None:
        raise AssertionError("the native test requires its owned PostgreSQL URL")
    with closing(PostgresStore(POSTGRES_URL, initialize_schema=False)) as store:
        yield store


def post(context: Any, path: str, payload: Any) -> dict[str, Any]:
    async def request() -> dict[str, Any]:
        async with asgi_client(context) as client:
            response = await client.post(path, json=payload.model_dump(mode="json"))
        assert response.status_code == 200, "the public ingestion request must succeed"
        return dict(response.json())

    return asyncio.run(request())


@pytest.mark.parametrize("kernel_first", [True, False])
@pytest.mark.parametrize("xid", [32, 79])
def test_postgres_correlates_both_orders_and_rejects_cross_scope_replays(
    native_store: PostgresStore, kernel_first: bool, xid: int
) -> None:
    service = GpuMetricsService(store=native_store)
    event = cases.kernel(xid=xid)
    finding = cases.correlate(service, event, kernel_first=kernel_first)[0]
    assert finding.correlation_rule_id == cases.RULE
    service.correlate_kernel_xid(event)
    service.ingest(cases.batch("clear", 30))
    assert cases.composites(service) == [], "healthy PCIe evidence clears the composite"
    rearmed = service.ingest(cases.batch("rearm", 45, 1010))
    assert rearmed.new_composite_findings == [], "pre-clear XID evidence cannot rearm"
    assert service.correlate_kernel_xid(event) == []
    fresh = cases.kernel("kernel-fresh", 50, xid=xid)
    cases.persist_xid(native_store, fresh)
    assert len(service.correlate_kernel_xid(fresh)) == 1
    wrong = cases.kernel("wrong-scope", 52, cluster_id="other-cluster")
    cases.persist_xid(native_store, wrong)
    assert service.correlate_kernel_xid(wrong) == [], (
        "stored XID evidence must never cross a PostgreSQL cluster scope"
    )


@pytest.mark.parametrize("kernel_first", [True, False])
@pytest.mark.parametrize(
    "changes",
    [
        {"cluster_id": "other-cluster"},
        {"node_id": "other-node"},
        {"gpu_uuid": "GPU-other"},
        {"pci_bdf": "0001:01:00.0"},
        {"runtime_profile_version": "other-profile"},
        {"event_source": "training-log"},
        {"product": "unknown"},
        {"synthetic": True},
    ],
    ids=[
        "cluster",
        "node",
        "gpu",
        "pci-domain",
        "profile",
        "source",
        "policy",
        "synthetic",
    ],
)
def test_postgres_kernel_composite_refuses_mismatched_evidence(
    native_store: PostgresStore, kernel_first: bool, changes: dict[str, object]
) -> None:
    service = GpuMetricsService(store=native_store)

    findings = cases.correlate(
        service, cases.kernel(**changes), kernel_first=kernel_first
    )

    assert findings == [], "untrusted or mismatched evidence must not activate a join"
    assert cases.composites(service) == [], (
        "the PostgreSQL composite state must remain inactive after refusal"
    )


def test_postgres_unrelated_scrape_cannot_fence_delayed_kernel_evidence(
    native_store: PostgresStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases.test_unrelated_scrape_cannot_fence_delayed_kernel_evidence(
        native_store, monkeypatch
    )


@pytest.mark.parametrize("kernel_first", [True, False])
def test_postgres_composite_and_incident_share_rollback_boundary(
    native_store: PostgresStore, monkeypatch: pytest.MonkeyPatch, kernel_first: bool
) -> None:
    context = build_context(store=native_store)
    raw = cases.kernel_record()
    breach = cases.batch("breach", 15)
    post(context, GPU_METRICS_PATH, cases.batch("baseline", 0, 1000))
    if kernel_first:
        post(context, NVIDIA_KERNEL_PATH, raw)
        path, payload = GPU_METRICS_PATH, breach
    else:
        post(context, GPU_METRICS_PATH, breach)
        path, payload = NVIDIA_KERNEL_PATH, raw
    original = context.orchestrator.ingest_node_health

    def fail(finding: NodeHealthFinding) -> Any:
        if finding.metric_name == f"composite:{cases.RULE}":
            raise RuntimeError("composite incident write failed")
        return original(finding)

    monkeypatch.setattr(context.orchestrator, "ingest_node_health", fail)
    with pytest.raises(RuntimeError, match="composite incident write failed"):
        post(context, path, payload)
    assert cases.composites(context.gpu_metrics) == [], (
        "a failed incident transaction must not persist a composite activation"
    )
    if kernel_first:
        assert (
            native_store.get_gpu_metrics_batch(("cluster-a", "node-a", "breach"))
            is None
        )
    else:
        event = NvidiaLogNormalizer().normalize_kernel(raw).xid_events[0]
        with pytest.raises(NotFoundError):
            native_store.get_xid_event(event.event_id)
    monkeypatch.setattr(context.orchestrator, "ingest_node_health", original)
    post(context, path, payload)
    findings = cases.composites(context.gpu_metrics)
    assert len(findings) == 1
    assert (
        native_store.get_incident_by_event(f"gpu-{findings[0].finding_id}") is not None
    )


@pytest.mark.parametrize("first_path", [NVIDIA_KERNEL_PATH, GPU_METRICS_PATH])
def test_postgres_two_replicas_serialize_pcie_and_kernel_evidence(
    native_store: PostgresStore, monkeypatch: pytest.MonkeyPatch, first_path: str
) -> None:
    first = build_context(store=native_store)
    post(first, GPU_METRICS_PATH, cases.batch("baseline", 0, 1000))
    first_entered, second_entered, release_first = Event(), Event(), Event()
    methods = {
        NVIDIA_KERNEL_PATH: "save_xid_event_if_absent",
        GPU_METRICS_PATH: "observe_gpu_metrics",
    }
    payloads = {
        NVIDIA_KERNEL_PATH: cases.kernel_record(),
        GPU_METRICS_PATH: cases.batch("breach", 15),
    }
    second_path = (
        GPU_METRICS_PATH if first_path == NVIDIA_KERNEL_PATH else NVIDIA_KERNEL_PATH
    )
    if POSTGRES_URL is None:
        raise AssertionError("the native test requires its owned PostgreSQL URL")
    with closing(PostgresStore(POSTGRES_URL, initialize_schema=False)) as second_store:
        second = build_context(store=second_store)
        first_original = getattr(native_store, methods[first_path])
        second_original = getattr(second_store, methods[second_path])

        def hold_first(*args: Any, **kwargs: Any) -> Any:
            first_entered.set()
            assert release_first.wait(20), (
                "the owning test must release its first writer"
            )
            return first_original(*args, **kwargs)

        def observe_second(*args: Any, **kwargs: Any) -> Any:
            second_entered.set()
            return second_original(*args, **kwargs)

        monkeypatch.setattr(native_store, methods[first_path], hold_first)
        monkeypatch.setattr(second_store, methods[second_path], observe_second)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_result = pool.submit(post, first, first_path, payloads[first_path])
            try:
                assert first_entered.wait(20), (
                    "the first replica must reach persistence"
                )
                second_result = pool.submit(
                    post, second, second_path, payloads[second_path]
                )
                assert not second_entered.wait(0.25), (
                    "the second replica must wait for the node's composite transaction"
                )
            finally:
                release_first.set()
            first_result.result(timeout=30)
            second_result.result(timeout=30)
        assert second_entered.is_set(), "the second writer must resume after commit"
        findings = cases.composites(second.gpu_metrics)
        assert len(findings) == 1
        assert (
            second_store.get_incident_by_event(f"gpu-{findings[0].finding_id}")
            is not None
        )
        history = [
            item
            for item in second.gpu_metrics.findings(
                "cluster-a", "node-a", active_only=False
            )
            if item.correlation_rule_id == cases.RULE
        ]
        assert len(history) == 1, "two replicas must activate only one composite"
        replay = post(second, NVIDIA_KERNEL_PATH, cases.kernel_record())
        assert replay["decisions"][0]["duplicate"] is True
