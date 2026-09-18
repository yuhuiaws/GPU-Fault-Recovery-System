"""Collector fixture orchestration through explicit fake host/CPU transports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import collector_acceptance_fixture as acceptance
from scripts.e2e.regional import collector_action_guard as guard
from scripts.e2e.regional import collector_case_cleanup as cleanup
from scripts.e2e.regional import collector_window_fixture as window
from tests.regional._cov95_collect_net import Clock, no_external_effects  # noqa: F401


@pytest.fixture
def fixture_host(tmp_path: Path, monkeypatch: Any) -> Any:
    kubeconfig = tmp_path / "fixture-kubeconfig"
    kubeconfig.touch()
    calls = []
    state = SimpleNamespace(host_result={}, cpu_result={}, pod_count=1, calls=calls)

    class Host:
        def __init__(self, settings: Any) -> None:
            calls.append(("host-settings", settings))

        def create(self) -> None:
            calls.append(("create",))

        def cleanup(self) -> dict[str, bool]:
            calls.append(("cleanup",))
            return {"pod": False}

        def execute(self, *args: str, **kwargs: Any) -> Any:
            calls.append(("host", args, kwargs))
            return state.host_result

    def kubectl(plane: str, *args: str, **kwargs: Any) -> str:
        calls.append(("kubectl", plane, args, kwargs))
        return 'notice\n{"metrics":"metric 1"}' if args[0] == "exec" else "fixture-log"

    def cpu_python(*args: str) -> Any:
        calls.append(("cpu", args))
        return state.cpu_result

    regional = SimpleNamespace(
        settings=SimpleNamespace(
            gpu_kubeconfig=kubeconfig,
            gpu_context="gpu",
            namespace="test",
            cluster_id="cluster-a",
        ),
        cpu_python=cpu_python,
        kubectl=kubectl,
        ready_pods=lambda plane, app: [{"name": app + "-a"}] * state.pod_count,
    )
    for module in (window, acceptance):
        monkeypatch.setattr(module, "HostProbeFixture", Host)
    state.regional = regional
    state.window = window.CollectorWindowFixture(
        regional,
        node="node-a",
        image="example",
        case_id="GF-REGIONAL-COLLECT-019",
        run_id="run-a",
        case_dir=tmp_path,
    )
    state.acceptance = acceptance.CollectorAcceptanceFixture(
        regional,
        node="node-a",
        image="example",
        case_id="GF-REGIONAL-COLLECT-017",
        run_id="run-a",
        case_dir=tmp_path,
    )
    state.root = tmp_path
    return state


def test_host_lifecycle_and_probe_wrappers_preserve_local_scope(
    fixture_host: Any,
) -> None:
    host = fixture_host
    host.acceptance.create()
    host.acceptance.recreate()
    host.acceptance.snapshot()
    host.host_result = {"efa_inventory": {"discovered_count": 2}}
    assert host.acceptance.efa_inventory() == {"discovered_count": 2}
    host.host_result = {}
    with pytest.raises(acceptance.RegionalFixtureError, match="no inventory"):
        host.acceptance.efa_inventory()
    assert host.acceptance.cleanup() == {"pod": False}
    host.window.create()
    host.window.snapshot()
    host.window.snapshot("marker-a")
    assert host.window.cleanup() == {"pod": False}
    assert (
        "host",
        ("snapshot", "--marker", "marker-a"),
        {"timeout": 240},
    ) in host.calls
    assert [call[0] for call in host.calls].count("create") == 3
    assert all(
        call[1].state_directory == host.root / "host-probes"
        for call in host.calls
        if call[0] == "host-settings"
    ), "host ownership journals must stay under the case directory"


def test_store_and_status_projections_forward_exact_node_and_time(
    fixture_host: Any,
) -> None:
    host = fixture_host
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    host.acceptance.store_snapshot("marker-a", observed_after=now, scan_evidence=False)
    assert host.calls[-1][1][-5:] == (
        "cluster-a",
        "node-a",
        "marker-a",
        (now - timedelta(seconds=30)).isoformat(),
        "",
    )
    host.acceptance.store_snapshot("marker-a")
    assert host.calls[-1][1][-1] == "1"
    host.cpu_result = {"records": [{"collector": "kernel"}]}
    assert host.window.collector_statuses() == [{"collector": "kernel"}]
    assert (
        host.window.node_activity(now, evidence_kind="GPU_INVENTORY") is host.cpu_result
    )
    assert host.calls[-1][1][-6:] == (
        "cluster-a",
        "node-a",
        now.isoformat(),
        "GPU_INVENTORY",
        now.isoformat(),
        "",
    )
    host.cpu_result = {}
    assert host.window.collector_statuses() == []


@pytest.mark.parametrize("populated", [False, True])
def test_metrics_and_logs_visit_each_ready_role_with_component_interpreter(
    fixture_host: Any, populated: bool
) -> None:
    host = fixture_host
    host.pod_count = int(populated)
    assert host.window.control_plane_metrics() == (
        ["metric 1", "metric 1"] if populated else []
    )
    assert host.window.control_plane_logs(30) == (
        "fixture-log\nfixture-log" if populated else ""
    )
    if populated:
        execs = [
            call for call in host.calls if call[0] == "kubectl" and call[2][0] == "exec"
        ]
        assert [call[2][-1] for call in execs] == ["8081", "8080"]
        assert all(
            "/opt/gpu-fault/control-plane/bin/python" in call[2] for call in execs
        ), "CPU readers must not use system Python"


@pytest.mark.parametrize("converges", [False, True])
def test_window_wait_records_poll_results_and_respects_deadline(
    fixture_host: Any, monkeypatch: Any, converges: bool
) -> None:
    clock = Clock()
    monkeypatch.setattr(window, "time", clock)
    result = fixture_host.window.wait_until(
        lambda: {"done": True} if converges and clock.now >= 1005 else None,
        timeout_seconds=10,
        poll_seconds=5,
        case_dir=fixture_host.root,
        name="sample",
    )
    assert result == ({"done": True} if converges else None)
    timeline = json.loads((fixture_host.root / "sample-timeline.json").read_text())[
        "entries"
    ]
    assert len(timeline) == (2 if converges else 3)
    assert timeline[-1]["accepted"] is converges


@pytest.mark.parametrize("mode", ["success", "timeout", "terminal-race"])
def test_acceptance_poll_rechecks_terminal_workflow_after_evidence_scan(
    fixture_host: Any, monkeypatch: Any, mode: str
) -> None:
    clock = Clock()
    monkeypatch.setattr(acceptance, "time", clock)
    calls = []

    def snapshot(marker: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["scan_evidence"])
        pending = mode == "timeout" or (mode == "terminal-race" and len(calls) < 3)
        return {
            "workflows": [
                {
                    "status": "RUNNING"
                    if pending and kwargs["scan_evidence"]
                    else "SUCCEEDED"
                }
            ],
            "evidence": [{"record_id": "record-a"}] if kwargs["scan_evidence"] else [],
        }

    monkeypatch.setattr(fixture_host.acceptance, "store_snapshot", snapshot)
    if mode == "timeout":
        with pytest.raises(acceptance.RegionalFixtureError, match="did not converge"):
            fixture_host.acceptance.wait_marker(
                "marker-a",
                case_dir=fixture_host.root,
                timeout_seconds=10,
                terminal_workflow=True,
            )
    else:
        result = fixture_host.acceptance.wait_marker(
            "marker-a",
            case_dir=fixture_host.root,
            timeout_seconds=10,
            terminal_workflow=True,
        )
        assert result["evidence"] == [{"record_id": "record-a"}]
        assert calls == (
            [False, True] if mode == "success" else [False, True, False, True]
        )


def test_setting_workflow_selection_and_metric_filter_edges() -> None:
    with pytest.raises(acceptance.RegionalFixtureError, match="absent"):
        acceptance.collector_setting({}, "INTERVAL")
    assert acceptance.collector_setting({"INTERVAL": "15"}, "INTERVAL") == 15
    assert (
        acceptance.select_workflow(
            [{"official_action": "OTHER"}], official_actions={"RESET"}
        )
        is None
    )
    assert (
        acceptance.select_workflow([{"official_steps": []}], operation="RESET_GPU")
        is None
    )
    text = '\n# HELP metric\nmalformed\nother 9\nmetric invalid\nmetric{node="a"} 2\nmetric{node="b"} 5\n'
    assert window.metric_sum([text], "metric", where={"node": "a"}) == 2
    assert window.metric_max([text], "metric", where={"node": "b"}) == 5
    assert window.metric_max([text], "metric", where={"node": "absent"}) is None


@pytest.mark.parametrize("endpoint", ["", "control.invalid"])
def test_window_configuration_and_environment_keep_explicit_scope(
    tmp_path: Path, monkeypatch: Any, endpoint: str
) -> None:
    regional = SimpleNamespace(environment=lambda: {"CLUSTER": "cluster-a"})
    monkeypatch.setattr(window, "settings_from_arguments", lambda args: regional)
    parser = argparse.ArgumentParser()
    window.add_window_arguments(parser, "CONFIRM")
    args = parser.parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--node",
            "node-a",
            "--host-probe-image",
            "image@sha256:" + "a" * 64,
            "--endpoint-host",
            endpoint,
        ]
    )
    settings = window.configure(args, "GF-REGIONAL-COLLECT-019")
    assert settings.environment()["GPU_FAULT_WINDOW_NODE"] == "node-a"
    assert ("GPU_FAULT_NET_ENDPOINT_HOST" in settings.environment()) is bool(endpoint)
    args.host_probe_image = "mutable"
    with pytest.raises(window.RegionalFixtureError, match="immutable"):
        window.configure(args, "GF-REGIONAL-COLLECT-019")


@pytest.mark.parametrize(
    "problem", [None, "predecessor", "ready", "ownership", "workload", "agent"]
)
def test_window_preflight_uses_bound_predecessor_and_refuses_each_missing_premise(
    tmp_path: Path, monkeypatch: Any, problem: str | None
) -> None:
    node = {
        "name": "node-a",
        "ready": "False" if problem == "ready" else "True",
        "ownership_annotations": {"owner": "other"} if problem == "ownership" else {},
    }
    regional = SimpleNamespace(
        node_snapshot=lambda _: node,
        store_snapshot=lambda **k: {
            "agent": {"lifecycle_state": "UNKNOWN" if problem == "agent" else "ACTIVE"}
        },
        business_workloads=lambda _: ["work"] if problem == "workload" else [],
        evidence_identity=lambda: {
            "release_id": "release-a",
            "cluster_id": "cluster-a",
        },
        cpu_blast_snapshot=lambda: {},
    )
    seen = []
    monkeypatch.setattr(window, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(
        window,
        "predecessor_evidence",
        lambda *a, **k: seen.append(k) or {"valid": problem != "predecessor"},
    )
    settings = window.WindowSettings(
        object(),
        "GF-REGIONAL-COLLECT-019",
        "node-a",
        "image",
        tmp_path / "predecessor" if problem else None,
        "previous" if problem else None,
    )
    result = window.read_only_preflight(settings, tmp_path)
    assert len(result["errors"]) == (0 if problem is None else 1)
    assert seen == (
        [{"release_id": "release-a", "cluster_id": "cluster-a"}] if problem else []
    )


def test_nested_action_window_never_extends_outer_authorization(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(guard.RegionalFixtureError, match="timezone"):
        with guard.action_window(now.replace(tzinfo=None)):
            pytest.fail("naive window admitted")
    with guard.action_window(now + timedelta(seconds=120)):
        with guard.action_window(now + timedelta(hours=1)):
            with pytest.raises(guard.RegionalFixtureError, match="next action"):
                guard.require_action_time(180)
        with guard.action_window(now + timedelta(seconds=30)):
            guard.require_action_time(1)
    guard.require_action_time(180)
    calls = []

    @guard.bounded_window_case
    def execute(
        settings: Any, fixture: Any, case_dir: Path, attempt: int, deadline: datetime
    ) -> dict[str, Any]:
        calls.append((case_dir, attempt))
        guard.require_action_time(1)
        return {"verdict": "PASS"}

    assert execute(None, None, tmp_path, 1, now + timedelta(seconds=120)) == {
        "verdict": "PASS"
    }
    assert calls == [(tmp_path, 1)]


@pytest.mark.parametrize("error", [None, "missing-unit", "rollback"])
def test_open_failure_preserves_original_error_and_records_rollback_failure(
    error: str | None,
) -> None:
    initial = RuntimeError("open failed")
    calls = []

    def execute(*args: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(args)
        if args[0] == "open-window" or error == "rollback":
            raise initial
        return {}

    args = (
        ()
        if error == "missing-unit"
        else ("--unit", "gpu-fault-host-collector.service")
    )
    with pytest.raises(RuntimeError) as caught:
        window.open_window_or_rollback(SimpleNamespace(execute=execute), "run-a", *args)
    assert caught.value is initial
    if error:
        assert len(caught.value.__notes__) == 1
    else:
        assert calls[-1][0] == "close-window"


def test_cleanup_tracks_unique_seeds_and_only_the_matching_quiesce_host() -> None:
    tracker = cleanup.CaseCleanup()
    calls = []
    snapshot = {
        "seed_marker": "marker-a",
        "incidents": [{"incident_id": "incident-a"}],
        "workflows": [],
        "commands": [],
    }
    fixture = SimpleNamespace(
        restore_incidents=lambda *a, **k: [{"restored": True}],
        store_snapshot=lambda marker: snapshot,
    )
    other = SimpleNamespace()
    host = SimpleNamespace(
        execute=lambda *a, **k: calls.append(a) or {"quiesce_states": []}
    )
    with pytest.raises(cleanup.RegionalFixtureError, match="empty"):
        tracker.register_seed(fixture, "")
    tracker.register_seed(other, "other", quiesce_host=host)
    tracker.register_seed(fixture, "marker-a", quiesce_host=host)
    tracker.register_seed(fixture, "marker-a", quiesce_host=host)
    assert len(tracker.seed_markers) == 2
    assert len(tracker.quiesce_hosts) == 2
    with pytest.raises(cleanup.RegionalFixtureError, match="not terminal"):
        tracker.restore(
            fixture,
            {"commands": [{"status": "RUNNING"}], "workflows": []},
            profile_version="profile",
            reason="fixture",
        )
    with pytest.raises(cleanup.RegionalFixtureError, match="exact incident"):
        tracker.restore(
            fixture,
            {"incidents": [{}], "workflows": [], "commands": []},
            profile_version="profile",
            reason="fixture",
        )
    tracker.restore(fixture, snapshot, profile_version="profile", reason="fixture")
    assert calls == [("snapshot",)], "host checks must never restore quiesce state"
    assert tracker.seed_markers == [(other, "other")]
