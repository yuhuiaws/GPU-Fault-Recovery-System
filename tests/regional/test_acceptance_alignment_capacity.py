from __future__ import annotations

import itertools
import json
import time
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.response import addinfourl

import pytest

from gpu_fault.cluster_executor import ClusterExecutorError
from scripts.e2e.regional import capacity_acceptance_cases as cases
from scripts.e2e.regional.capacity_acceptance_base import CapError
from scripts.e2e.regional.capacity_acceptance_retry import run_claim_retry_proof
from scripts.e2e.regional.capacity_connection_budget import unpooled_connection_budget


@pytest.mark.parametrize(
    "role,consumers",
    [
        ("gpu-fault-api-ha", {"remote_command_claim": 1}),
        (
            "gpu-fault-control-worker",
            {"remote_command_claim": 1, "processor_queue": 1, "workflow_dispatch": 2},
        ),
        (
            "gpu-fault-telemetry-spool-worker",
            {"remote_command_claim": 1, "telemetry_spool": 1},
        ),
    ],
)
def test_pool_budget_includes_every_installed_listener(role, consumers) -> None:
    assert unpooled_connection_budget(role) == consumers
    with pytest.raises(CapError):
        unpooled_connection_budget("unknown")


@pytest.mark.parametrize("failure", [503, 403])
def test_capacity_retry_uses_the_real_executor_loop_and_http_client(
    monkeypatch, tmp_path, failure
) -> None:
    requests, released = [], []

    def urlopen(request, **kwargs):
        body = json.loads(request.data)
        requests.append((body, time.monotonic()))
        if len(requests) == 1:
            raise HTTPError(
                request.full_url,
                failure,
                "isolated status",
                {"Retry-After": "2"},
                BytesIO(b"{}"),
            )
        assert released, "capacity must be released before the production retry"
        return addinfourl(BytesIO(b'{"commands": []}'), {}, request.full_url, 200)

    monkeypatch.setattr("gpu_fault.cluster_executor.regional_client.urlopen", urlopen)
    result = run_claim_retry_proof(
        "http://127.0.0.1:18623",
        "fixture",
        tmp_path,
        lambda: released.append(True),
        poll_seconds=0.001,
    )
    assert result["passed"] is (failure == 503)
    assert result["adapter_count"] == 0 and result["commands_executed"] == 0
    assert requests[0][0]["executor_id"] == "cap002-real-executor"
    if failure == 503:
        assert len(requests) == 2 and result["retry_delays_seconds"][0] >= 0.001
    else:
        assert len(requests) == 1 and released == []


@pytest.mark.parametrize(
    "defect", ["none", "latency", "http", "listener", "disconnect", "nonempty"]
)
def test_production_long_poll_is_separate_from_the_short_poll_latency_gate(
    monkeypatch, tmp_path, defect
) -> None:
    calls = []
    ticks = itertools.count(0, 21.1 if defect == "latency" else 20)
    monkeypatch.setattr(
        cases, "time", SimpleNamespace(perf_counter=lambda: next(ticks))
    )
    listener_reads = []

    pins = {"executor_artifact_sha256": "e" * 64}

    class Client:
        def __init__(self, url, cluster, token, **kwargs):
            assert (
                kwargs.get("executor_artifact_sha256")
                == pins["executor_artifact_sha256"]
            ), "long-poll executors must present the release's executor pins"
            self.cluster = cluster

        def claim(self, owner, **kwargs):
            calls.append((self.cluster, owner, kwargs))
            if owner != "cap003-warmup":
                if defect == "http":
                    raise ClusterExecutorError("isolated503", status_code=503)
                if defect == "nonempty":
                    return [object()]
            return []

    monkeypatch.setattr(cases, "write_json", lambda *args: None)

    # The private probe answers the same runtime wait budget on every read; a
    # 30 s request budget minus the 3 s claim reserve leaves the full 20 s hold.
    wait_budget = {
        "requested_wait_seconds": 20,
        "request_budget_seconds": 30,
        "claim_wait_reserve_seconds": 3,
        "server_max_wait_seconds": 30,
        "effective_wait_seconds": 20,
    }

    def listener(*args):
        listener_reads.append(args)
        assert args[1:] == (
            "/__cap__/claim-wakeups",
            {"tag": "cap003", "wait_seconds": 20},
        ), (
            f"the long-poll load asked the probe for something other than the "
            f"cap003 wait budget: {args[1:]}"
        )
        return {
            **wait_budget,
            "listener_alive": True,
            "listener_connected": defect != "listener",
            "disconnects_total": len(listener_reads) if defect == "disconnect" else 0,
        }

    harness = SimpleNamespace(
        tokens=["fixture"] * 20,
        probe_control=listener,
        pod_cpu_usage_usec=lambda _pod: 0,
        isolated_db_connections=lambda _pod: 2,
        executor_client=lambda url, index: Client(
            url, f"cap-cluster-{index:03d}", "fixture", **pins
        ),
    )
    # The method is exercised on its public harness contract, not an alternate load loop.
    method = cases.CapacityAcceptanceCases.cap003_long_poll_samples
    if defect == "nonempty":
        with pytest.raises(CapError, match="unexpected commands"):
            method(
                harness,
                SimpleNamespace(url="http://127.0.0.1:18623", pod="isolated"),
                tmp_path,
            )
        return
    rows = method(
        harness, SimpleNamespace(url="http://127.0.0.1:18623", pod="isolated"), tmp_path
    )
    assert [row["cluster_count"] for row in rows] == [1, 5, 10, 20]
    assert all(
        row["wait_seconds"] == 20 and row["executor_replicas_per_cluster"] == 2
        for row in rows
    ), "long-poll load must use the production default wait and replica count"
    assert all(
        row["effective_wait_seconds"] == 20
        and row["wait_configuration"] == {k: float(v) for k, v in wait_budget.items()}
        for row in rows
    ), "each scale point must record the probe-reported effective wait budget"
    assert all(row["passed"] is (defect == "none") for row in rows), (
        "each scale point must reflect the injected contract defect"
    )
    assert len(calls) == 217 and all(call[2]["wait_seconds"] == 20 for call in calls)
