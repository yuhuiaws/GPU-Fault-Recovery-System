"""The pool-capacity guard counts every connection consumer of the role it runs
in, and says so on /metrics rather than once in a startup log.

FINAL-建议汇总 F-E5 (P1-78F, P2-16B/C/D). ``_warn_pool_capacity`` counted only
the Store I/O threads: 8 in the worker role against a pool of 8, so ``8 < 8``
never warned -- in the one role where 24 processor threads, 8 dispatcher
threads, the periodic runner and an unpooled LISTEN connection actually compete
for checkouts.
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


def test_the_worker_role_counts_its_processor_and_dispatcher_threads(
    worker_environment, caplog
):
    caplog.set_level(logging.WARNING, logger="gpu_fault.app.admission_runtime")

    estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})

    assert estimate is not None
    assert estimate.pool_max == 8
    assert estimate.demand_by_consumer == {
        "store_io_general": 8,
        "processor_workers": 24,
        "workflow_dispatcher": 8,
        "periodic_services": 1,
    }
    assert estimate.demand == 41
    assert estimate.oversubscription_ratio == pytest.approx(41 / 8)
    # The LISTEN connection is real but bypasses the pool: reported, not
    # folded into the pool ratio.
    assert estimate.unpooled_connections == 1
    assert any(
        "41" in record.getMessage() and "processor_workers" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_the_ingress_role_does_not_count_threads_it_never_starts(
    worker_environment, monkeypatch
):
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "40")

    estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})

    assert estimate is not None
    assert set(estimate.demand_by_consumer) == {
        "store_io_general",
        "store_io_fault",
        "store_io_evidence",
    }
    assert estimate.unpooled_connections == 0
    assert estimate.oversubscription_ratio == pytest.approx(20 / 40)


def test_a_non_postgres_store_has_no_pool_to_estimate(monkeypatch):
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "sqlite:///tmp/x.db")

    assert AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False}) is None


def test_the_ratio_reaches_metrics(worker_environment):
    estimate = AdmissionRuntimeFactory.pool_capacity({"spool_enabled": False})
    runtime = SimpleNamespace(context=SimpleNamespace(postgres_pool_capacity=estimate))

    lines = postgres_pool_metric_lines(runtime)

    assert "gpu_fault_postgres_pool_oversubscription_ratio 5.125" in lines
    assert "gpu_fault_postgres_pool_max_size 8" in lines
    assert (
        'gpu_fault_postgres_pool_demand_connections{consumer="processor_workers"} 24'
        in lines
    )
    assert "gpu_fault_postgres_unpooled_connections 1" in lines


def test_metrics_stay_quiet_without_an_estimate():
    runtime = SimpleNamespace(context=SimpleNamespace())

    assert postgres_pool_metric_lines(runtime) == []
