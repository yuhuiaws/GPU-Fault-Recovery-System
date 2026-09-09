"""The pool-capacity guard counts every connection consumer of the role it runs
in, and says so on /metrics rather than once in a startup log.

FINAL-建议汇总 F-E5 (P1-78F, P2-16B/C/D). ``_warn_pool_capacity`` counted only
the Store I/O threads: 8 in the worker role against a pool of 8, so ``8 < 8``
never warned -- in the one role where 24 processor threads, 8 dispatcher
threads, the periodic runner and an unpooled LISTEN connection actually compete
for checkouts.

A-5 / G-7 (cp-review-20260908). The estimate then read
``GPU_FAULT_PROCESSOR_WORKERS`` (24) where the coordinator actually starts the
explicit pool split (4+2+4+4 = 14), and left out five thread families that do
check connections out: the processor claim loop, xid correlation, the
notification dispatcher, the collector-metrics snapshot, the diagnostics
publisher, and -- in every regional role -- the registry refresh whose failure
is what turns ``/healthz`` red. The ingress estimate came out exactly equal to
``POOL_MAX`` and so never warned; the baseline is now two spare connections.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from gpu_fault.app.admission_runtime import AdmissionRuntimeFactory
from gpu_fault.app.builtin_metric_contributors import postgres_pool_metric_lines


@pytest.fixture
def worker_environment(monkeypatch):
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://control-plane/gpu_fault")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "worker")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "8")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_WORKERS", "8")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_WORKERS", "24")
    monkeypatch.setenv("GPU_FAULT_WORKFLOW_DISPATCHER_WORKERS", "8")
    monkeypatch.delenv("GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER", raising=False)
    monkeypatch.delenv("GPU_FAULT_TELEMETRY_SPOOL", raising=False)


@pytest.mark.parametrize("token", ["1", "yes", "on"])
def test_dispatcher_switch_accepts_every_enabled_token(
    worker_environment, monkeypatch, token: str
):
    """A default-on switch spelled ``=1`` used to count as switched off."""

    monkeypatch.setenv("GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER", token)

    estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})

    assert estimate is not None
    assert estimate.demand_by_consumer["workflow_dispatcher"] == 8


def test_the_worker_role_counts_its_processor_and_dispatcher_threads(
    worker_environment, caplog
):
    caplog.set_level(logging.WARNING, logger="gpu_fault.app.admission_runtime")

    estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})

    assert estimate is not None
    assert estimate.pool_max == 8
    # No explicit pool split in this environment, so the budget falls into
    # four pools of six: 24 threads that really exist.
    assert estimate.demand_by_consumer == {
        "store_io_general": 8,
        "processor_claim_loop": 1,
        "processor_workers": 24,
        "workflow_dispatcher": 8,
        "periodic_services": 1,
        "xid_correlation": 1,
        "notification_dispatcher": 1,
        "processor_diagnostics": 1,
        "collector_metrics_snapshot": 1,
    }
    assert estimate.demand == 46
    assert estimate.oversubscription_ratio == pytest.approx(46 / 8)
    assert estimate.headroom == 8 - 46
    # The LISTEN connection is real but bypasses the pool: reported, not
    # folded into the pool ratio.
    assert estimate.unpooled_connections == 1
    assert any(
        "46" in record.getMessage() and "processor_workers" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_the_worker_role_counts_the_explicit_pool_split_not_the_budget(
    worker_environment, monkeypatch
):
    """Production sets 4+2+4+4; the coordinator starts exactly those threads."""

    monkeypatch.setenv("GPU_FAULT_PROCESSOR_FAULT_WORKERS", "4")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_OBSERVATION_WORKERS", "2")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS", "4")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_HOST_TELEMETRY_WORKERS", "4")

    estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})

    assert estimate is not None
    assert estimate.demand_by_consumer["processor_workers"] == 14


def test_regional_mode_counts_the_registry_refresh_in_every_role(
    worker_environment, monkeypatch
):
    monkeypatch.setenv("GPU_FAULT_DEPLOYMENT_MODE", "regional")
    monkeypatch.setenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "64")

    for role in ("ingress", "worker", "spool-worker"):
        monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", role)
        estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})
        assert estimate is not None, role
        assert estimate.demand_by_consumer["regional_registry"] == 1, role
        assert estimate.demand_by_consumer["collector_metrics_snapshot"] == 1, role


def test_the_estimate_wants_two_spare_connections(
    worker_environment, monkeypatch, caplog
):
    """``demand > pool_max`` let an ingress sized exactly to its threads pass.

    The registry refresh and the scrape then queued behind saturated store
    threads; the estimate now insists on ``demand + 2 <= pool_max``.
    """

    caplog.set_level(logging.WARNING, logger="gpu_fault.app.admission_runtime")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_WORKERS", "28")
    monkeypatch.setenv("GPU_FAULT_FAULT_STORE_IO_WORKERS", "8")
    monkeypatch.setenv("GPU_FAULT_EVIDENCE_STORE_IO_WORKERS", "4")
    monkeypatch.setenv("GPU_FAULT_DEPLOYMENT_MODE", "regional")
    # 28 + 8 + 4 + registry 1 + snapshot 1 = 42 against the production 40.
    monkeypatch.setenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "40")

    estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})

    assert estimate is not None
    assert estimate.demand == 42
    assert estimate.headroom == -2
    assert not estimate.has_headroom, "demand 42 against max 40 leaves no headroom"
    assert any("42" in record.getMessage() for record in caplog.records), (
        "the shortfall log must name the demand"
    )

    caplog.clear()
    monkeypatch.setenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "43")
    estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})
    assert estimate is not None
    assert estimate.headroom == 1
    assert not estimate.has_headroom, "one spare connection is not enough headroom"
    assert caplog.records, "one spare connection is not enough headroom"

    caplog.clear()
    monkeypatch.setenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "44")
    estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})
    assert estimate is not None
    assert estimate.has_headroom, "two spare connections clear the headroom floor"
    assert not caplog.records, "sufficient headroom must not log a warning"


def test_the_ingress_role_does_not_count_threads_it_never_starts(
    worker_environment, monkeypatch
):
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "40")

    monkeypatch.delenv("GPU_FAULT_DEPLOYMENT_MODE", raising=False)

    estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})

    assert estimate is not None
    assert set(estimate.demand_by_consumer) == {
        "store_io_general",
        "store_io_fault",
        "store_io_evidence",
        "collector_metrics_snapshot",
    }
    assert estimate.unpooled_connections == 0
    assert estimate.oversubscription_ratio == pytest.approx(21 / 40)


def test_a_non_postgres_store_has_no_pool_to_estimate(monkeypatch):
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "sqlite:///tmp/x.db")

    assert AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False}) is None


def test_the_ratio_reaches_metrics(worker_environment):
    estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})
    runtime = SimpleNamespace(context=SimpleNamespace(postgres_pool_capacity=estimate))

    lines = postgres_pool_metric_lines(runtime)

    assert "gpu_fault_postgres_pool_oversubscription_ratio 5.75" in lines
    assert "gpu_fault_postgres_pool_max_size 8" in lines
    assert (
        'gpu_fault_postgres_pool_demand_connections{consumer="processor_workers"} 24'
        in lines
    )
    assert "gpu_fault_postgres_unpooled_connections 1" in lines


def test_metrics_stay_quiet_without_an_estimate():
    runtime = SimpleNamespace(context=SimpleNamespace())

    assert postgres_pool_metric_lines(runtime) == []


def test_fault_reserve_must_hold_one_whole_cluster_fault_wave(monkeypatch) -> None:
    """A correlated whole-cluster fault is one fault-priority request per node.

    The renderer derives the reserve from the declared largest cluster; this is
    the worker defending itself against an environment assembled some other
    way (perf plan section 13.4 measured the 1000-node cluster overflowing a
    reserve sized as an eighth of a 1024 depth).
    """

    monkeypatch.delenv("GPU_FAULT_TELEMETRY_SPOOL", raising=False)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_QUEUE_DEPTH", "65536")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH", "4096")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH", "128")
    monkeypatch.setenv("GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT", "512")
    monkeypatch.setenv("GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT", "512")
    factory = AdmissionRuntimeFactory(SimpleNamespace(), None)

    with pytest.raises(RuntimeError, match="whole-cluster fault wave"):
        factory.limits()

    monkeypatch.setenv("GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH", "512")
    limits = factory.limits()
    assert limits["fault_reserved_cluster_depth"] == 512

    # An unknown node count (0, the default) cannot be defended and must not
    # block a start-up that predates the declaration.
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH", "128")
    monkeypatch.delenv("GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT")
    assert factory.limits()["fault_reserved_cluster_depth"] == 128
