"""GF-REGIONAL-DESTR-018 runner: the pure gates on shapes the happy fixtures
never carry (a profile whose first capability is another one, a deployment
that vanished, an open remote command queue, a predecessor that is not PASS),
the absorb wait that runs out of budget, an env window that drifts the runtime
identity, and the cleanup paths for an isolated node nobody owns and a refused
evidence close."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr018_lifetime_deadline as case
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_lifetime import LifetimeHarness, data, window


def test_capability_skips_other_and_malformed_entries() -> None:
    profile = {
        "capabilities": [
            {"capability": "workloadStop", "mode": "OWN"},
            "not-a-mapping",
            {"capability": "gpuReset", "mode": "OWN", "owner": "gpu-fault-node-agent"},
        ]
    }
    assert case.capability(profile, "gpuReset") == profile["capabilities"][2]
    assert case.capability(profile, "nodeReboot") is None
    assert case.capability(None, "gpuReset") is None


def test_identity_errors_name_a_deployment_that_disappeared() -> None:
    before = ready_runtime()
    after = deepcopy(before)
    vanished = next(
        name for name in after["deployments"]["cpu"] if name != window.DEPLOYMENT
    )
    del after["deployments"]["cpu"][vanished]
    errors = case.identity_errors(before, after, worker_generation_delta=0)
    assert errors == [
        f"cpu deployment {vanished} disappeared",
        f"cpu/{vanished} runtime deployment is unavailable",
    ]
    assert case.identity_errors(before, before, worker_generation_delta=0) == []


def test_preflight_refuses_an_open_remote_command_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    state = deepcopy(h.store)
    state["remote_commands"] = {"open_by_cluster": {"cluster-a": 2}}
    errors = case.preflight_errors(
        h.settings, state, h.node, [], {"passed": True}, h.survey_data
    )
    assert errors == ["remote command queue is not empty"]
    clean = case.preflight_errors(
        h.settings, h.store, h.node, [], {"passed": True}, h.survey_data
    )
    assert clean == []


def test_preflight_refuses_a_predecessor_that_did_not_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.predecessor_valid = False
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [
        f"{case.PREDECESSOR_CASE_ID} predecessor evidence is not PASS"
    ]
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        h.execute(tmp_path)
    assert "window.open" not in [name for name, _ in h.calls], h.calls


def test_the_absorb_wait_stops_at_its_budget_without_an_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.absorb_delays = 10**6
    store_snapshot = h.regional.store_snapshot

    def slow_absorb_reads(**kwargs: Any) -> dict[str, Any]:
        if str(kwargs.get("marker") or "").endswith("-absorb"):
            h.clock.sleep(case.ABSORB_TIMEOUT_SECONDS / 2 + 1)
        return store_snapshot(**kwargs)

    monkeypatch.setattr(h.regional, "store_snapshot", slow_absorb_reads)
    code, report = h.execute(tmp_path, seconds=20000)
    assert code == 1, report
    absorb_reads = [
        detail
        for name, detail in h.calls
        if name == "store" and str(detail.get("marker") or "").endswith("-absorb")
    ]
    assert len(absorb_reads) == 2, "the wait stops once its budget is spent"
    recorded = json.loads(
        (tmp_path / "cases" / case.CASE_ID / "absorb-state.json").read_text()
    )
    assert recorded == {}, "an absorb that never landed is recorded as empty"
    assert "window.close" in [name for name, _ in h.calls], h.calls


def test_an_env_window_that_drifts_the_release_identity_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    open_window = h.window.open_window

    def drifting_open(*args: Any) -> dict[str, Any]:
        opened = open_window(*args)
        h.runtime["release_state"] = "another-release"
        return opened

    monkeypatch.setattr(h.window, "open_window", drifting_open)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert "changed the control-worker in a way beyond" in report["error"], report
    assert "release state identity drifted" in report["error"], report
    names = [name for name, _ in h.calls]
    assert "window.close" in names, "the window is still closed on the way out"
    assert not h.injected, "nothing is injected into a drifted control plane"


def test_restore_refuses_an_isolated_node_with_no_known_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["workflow.wait"] = RuntimeError("store unreachable")
    node_snapshot = h.regional.node_snapshot

    def isolated_without_owner(node: str) -> dict[str, Any]:
        snapshot = node_snapshot(node)
        if h.injected:
            snapshot.update(
                unschedulable=True,
                taints=[{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}],
                ownership_annotations={},
            )
        return snapshot

    monkeypatch.setattr(h.regional, "node_snapshot", isolated_without_owner)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert "store unreachable" in report["error"], report
    assert any(
        "isolated but no incident is known" in item
        for item in report["cleanup"]["errors"]
    ), report["cleanup"]
    assert "restore.create" not in [name for name, _ in h.calls], (
        "no restore is created for an owner nobody can name"
    )


def test_a_refused_evidence_close_of_the_reset_incident_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    create_restore = h.warm.create_restore_workflow

    def refuse_reset_restore(**kwargs: Any) -> dict[str, Any]:
        if kwargs["incident_id"] == data.SUPPORT_INCIDENT_ID:
            return create_restore(**kwargs)
        h.call("restore.create", kwargs)
        raise RegionalFixtureError("operator hold retained")

    closes: list[dict[str, Any]] = []

    def refused_close(incident_id: str, **kwargs: Any) -> dict[str, Any]:
        closes.append({"incident_id": incident_id, **kwargs})
        return {"refusal": "incident is not quarantined"}

    monkeypatch.setattr(h.warm, "create_restore_workflow", refuse_reset_restore)
    monkeypatch.setattr(
        h.warm, "close_incident_with_evidence", refused_close, raising=False
    )
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert any(
        "evidence close of inc-1 was refused: incident is not quarantined" in item
        for item in report["cleanup"]["errors"]
    ), report["cleanup"]
    assert [item["incident_id"] for item in closes] == ["inc-1"]
    assert closes[0]["operator"] == "acceptance-fixture"
    assert report["cleanup"]["restore_isolated_node"]["incident_id"] == (
        data.SUPPORT_INCIDENT_ID
    ), report["cleanup"]
