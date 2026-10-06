"""ISO-006 probes the deliberately offline cluster in a Running executor Pod.

Live (mixed site, attempt 5): the block landed, ``cut_proof`` failed at the
transport as required, and from the next readiness tick both cluster-B executor
Pods were NotReady -- their readiness probe is ``gpu-fault-cluster-executor-
readiness``, an authenticated POST to the control plane the block rejects. The
in-cluster claim probe insisted on a Ready Pod, found none for the rest of the
block and the runner aborted with ``no Ready gpu Pod for
app=gpu-fault-cluster-executor``; both Pods were Running with zero restarts and
Ready again after the unblock. Readiness stays the control plane's signal: the
runner requires it of cluster A and probes B's Running Pods.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_iso006_cluster_offline as runner


class FakeClock:
    """``time`` for the observation loop: sleeping advances the clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(float(seconds), 0.001)


def test_the_observation_window_probes_cluster_b_in_a_running_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def claim_sample(fixture: Any, *, offline: bool = False) -> dict[str, Any]:
        calls.append((fixture.name, offline))
        if fixture.name == "b":
            return {
                "status": None,
                "transport_error": "URLError",
                "transport_failure_kind": "network",
            }
        return {"status": 200, "latency_seconds": 0.01}

    class Primary:
        def recover(self, deadline: float, *, cut_is_active: Any) -> dict[str, Any]:
            assert cut_is_active() is True, "the cut is judged from a B claim"
            return {"recovered": True}

    clock = FakeClock()
    monkeypatch.setattr(runner, "time", clock)
    monkeypatch.setattr(runner, "claim_sample", claim_sample)
    monkeypatch.setattr(
        runner, "control_plane_pressure", lambda _f, _c: {"cluster_queue_depth": 0.0}
    )
    monkeypatch.setattr(runner, "cpu_pod_snapshot", lambda _f: [])
    monkeypatch.setattr(runner, "control_plane_container_statuses", lambda _f: {})
    monkeypatch.setattr(runner, "container_status_errors", lambda _b, _a: [])
    monkeypatch.setattr(runner, "latency_errors", lambda _b, _s: [])
    monkeypatch.setattr(runner, "pressure_errors", lambda _b, _a: [])
    monkeypatch.setattr(runner, "latency_p95", lambda _s: 0.0)
    a = SimpleNamespace(name="a", ready_pods=lambda _plane, _app: [{"name": "exec-a"}])
    b = SimpleNamespace(name="b")
    settings = SimpleNamespace(
        multi=SimpleNamespace(cluster_b=SimpleNamespace(cluster_id="gpu-b")),
        duration_seconds=2 * runner.SAMPLE_INTERVAL_SECONDS,
    )
    result: dict[str, Any] = {"samples": [], "baseline_samples": []}

    errors = runner.observe_recovery_during_cut(
        a=a,
        b=b,
        primary=Primary(),
        settings=settings,
        preflight={"cpu_containers": {}, "cpu_pods": []},
        pressure_before={},
        result=result,
        maintenance_window_end=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    assert errors == []
    assert len(result["samples"]) == 2, result["samples"]
    b_calls = [offline for name, offline in calls if name == "b"]
    a_calls = [offline for name, offline in calls if name == "a"]
    assert b_calls and all(b_calls), (
        "every cluster-B claim during the block runs in a Running Pod"
    )
    assert a_calls and not any(a_calls), "cluster A's claims still require Ready"
    assert result["a_recovery"] == {"recovered": True}
    assert result["a_recovery_exercised"] is True


def test_the_limitation_names_why_readiness_is_not_required_of_cluster_b() -> None:
    assert any(
        "NotReady" in item and "Running, not Ready" in item
        for item in runner.VALIDATION_LIMITATIONS
    ), runner.VALIDATION_LIMITATIONS
