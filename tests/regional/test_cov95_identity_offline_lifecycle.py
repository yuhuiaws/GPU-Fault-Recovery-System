from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_iso006_cluster_offline as runner
from scripts.e2e.regional.multi_cluster_fixture import REJECTION_COUNTERS
from tests.regional._cov95_identity_isolation_support import preflight_environment
from tests.regional._cov95_identity_support import Clock
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "preflight",
        "expired",
        "baseline",
        "create",
        "nodes-before",
        "receipt",
        "slow-block",
        "cut-http",
        "a-ready",
        "cpu-readiness",
        "unblock",
        "b-timeout",
        "b-delayed",
        "nodes-after",
        "cleanup-unknown",
        "cleanup-error",
        "a-cleanup-error",
    ],
)
def test_iso006_lifecycle_fails_closed_and_cleans_every_started_resource(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    settings, regions, _registrations, _reads = preflight_environment(
        monkeypatch, tmp_path, runner
    )
    settings = replace(settings, duration_seconds=60)
    a, b = regions
    preflight = copy.deepcopy(runner.read_only_preflight(settings, tmp_path))
    monkeypatch.setattr(
        runner, "read_only_preflight", lambda *args, **kwargs: preflight
    )
    if defect == "preflight":
        preflight["errors"] = ["synthetic preflight refusal"]
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    events = []
    probes = []
    after_unblock = []
    cpu_pods = preflight["cpu_pods"]
    containers = preflight["cpu_containers"]
    pressure = {
        "cluster_queue_depth": 0,
        "rejections": {name: 0 for name in REJECTION_COUNTERS},
    }
    monkeypatch.setattr(
        runner, "control_plane_pressure", lambda *args: copy.deepcopy(pressure)
    )
    monkeypatch.setattr(
        runner, "control_plane_container_statuses", lambda *args: containers
    )

    class Primary:
        def __init__(self, region: Any, **kwargs: Any) -> None:
            assert region is a

        def prepare(self) -> None:
            events.append("prepare-a")

        def recover(self, deadline: float, *, cut_is_active: Any) -> dict[str, Any]:
            assert cut_is_active() is True
            events.append("recover-a")
            return {"state": {"workflow": {"status": "SUCCEEDED"}}}

        def cleanup(self, result: dict[str, Any]) -> list[str]:
            events.append("cleanup-a")
            if defect == "a-cleanup-error":
                raise RuntimeError("synthetic A cleanup failure")
            return []

    class Host:
        def __init__(self, settings: Any) -> None:
            self.settings = settings
            self.blocked = False
            probes.append(self)

        def create(self) -> None:
            events.append("create-" + self.settings.node)
            if defect == "create":
                raise RuntimeError("synthetic create failure")
            if defect == "nodes-before":
                b.nodes[0]["uid"] = "replacement-before"

        def execute(self, action: str, *args: str) -> dict[str, Any]:
            events.append(action + "-" + self.settings.node)
            if action == "block":
                self.blocked = True
                if defect == "slow-block":
                    clock.sleep(runner.MAX_BLOCK_SPAN_SECONDS + 1)
                return {"blocked": defect != "receipt"}
            assert action == "unblock"
            if defect == "unblock":
                return {"blocked": False}
            self.blocked = False
            if defect == "nodes-after":
                b.nodes[0]["uid"] = "replacement-after"
            return {
                "blocked": False,
                "residual": False,
                "chain_present": False,
                "jumps": {"OUTPUT": False, "FORWARD": False},
            }

        def cleanup(self) -> dict[str, bool]:
            events.append("cleanup-" + self.settings.node)
            if defect == "cleanup-error":
                raise RuntimeError("synthetic host cleanup failure")
            return (
                {}
                if defect == "cleanup-unknown"
                else {"pod": False, "configmap": False}
            )

    def is_blocked() -> bool:
        return any(probe.blocked for probe in probes)

    def claim(region: Any) -> dict[str, Any]:
        if region is a:
            return {
                "status": 503 if defect == "baseline" else 200,
                "latency_seconds": 0.1,
            }
        if is_blocked():
            return (
                {"status": 403}
                if defect == "cut-http"
                else {
                    "status": None,
                    "transport_error": "ConnectionRefusedError",
                    "transport_failure_kind": "network",
                }
            )
        after_unblock.append(None)
        return {
            "status": 503
            if defect == "b-timeout" or defect == "b-delayed" and len(after_unblock) < 3
            else 200,
            "latency_seconds": 0.1,
        }

    monkeypatch.setattr(runner, "PrimaryRecovery", Primary)
    monkeypatch.setattr(
        runner, "HostProbeSettings", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(runner, "HostProbeFixture", Host)
    monkeypatch.setattr(runner, "claim_sample", claim)
    monkeypatch.setattr(
        runner,
        "cpu_pod_snapshot",
        lambda *args: [] if defect == "cpu-readiness" and is_blocked() else cpu_pods,
    )
    original_ready = a.ready_pods
    monkeypatch.setattr(
        a,
        "ready_pods",
        lambda *args: []
        if defect == "a-ready" and is_blocked()
        else original_ready(*args),
    )
    deadline = datetime.now(timezone.utc) + timedelta(
        hours=-1 if defect == "expired" else 1
    )
    if defect in {"preflight", "expired"}:
        with pytest.raises(
            runner.RegionalFixtureError, match="preflight failed|window has ended"
        ):
            runner.execute_case(settings, tmp_path, 1, deadline)
        assert events == [] and probes == []
        return
    assert runner.execute_case(settings, tmp_path, 1, deadline) == int(
        defect not in {"none", "b-delayed"}
    )
    result = json.loads(
        (tmp_path / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json").read_text()
    )
    assert result["verdict"] == ("PASS" if defect in {"none", "b-delayed"} else "FAIL")
    assert events[-1] == "cleanup-a"
    assert len([event for event in events if event.startswith("create-")]) == len(
        [event for event in events if event.startswith("cleanup-node-")]
    )
    block_events = [event for event in events if event.startswith("block-")]
    unblock_events = [event for event in events if event.startswith("unblock-")]
    if block_events:
        assert unblock_events, "every started B cut must attempt scoped restoration"
    if defect != "unblock":
        assert not is_blocked(), "successful cleanup left a fake B node blocked"
    if defect in {"baseline", "create", "nodes-before"}:
        assert block_events == []
    if defect in {"none", "b-delayed"}:
        assert result["a_recovery_exercised"] is True
        assert events.index("recover-a") < next(
            index for index, event in enumerate(events) if event.startswith("unblock-")
        )
    if defect == "b-delayed":
        assert len(after_unblock) == 3
    if defect in {"unblock", "cleanup-unknown", "cleanup-error", "a-cleanup-error"}:
        assert result["cleanup_errors"]


@pytest.mark.parametrize("value", [None, True, -1, float("nan"), float("inf")])
def test_iso006_rejects_unknown_latency_even_with_a_200_response(value: Any) -> None:
    baseline = [{"status": 200, "latency_seconds": 0.1}]
    errors = runner.latency_errors(
        baseline, [{"status": 200, "latency_seconds": value}]
    )
    assert any("latency is invalid" in error for error in errors), errors


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        ({}, {}, "not measured"),
        ({"x": 0}, {"y": 0}, "not measured"),
        ({"x": 1}, {"x": 0}, "reset"),
        ({"x": 0}, {"x": None}, "not a valid"),
        ({"x": 0}, {"x": 1}, "increased"),
    ],
)
def test_iso006_pressure_requires_matching_monotonic_finite_counters(
    before: dict[str, Any], after: dict[str, Any], message: str
) -> None:
    errors = runner.pressure_errors({"rejections": before}, {"rejections": after})
    assert len(errors) == 1
    assert message in errors[0]
