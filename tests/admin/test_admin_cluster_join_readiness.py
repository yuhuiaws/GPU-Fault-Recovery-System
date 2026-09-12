"""The join's COLLECTORS_READY gate and the commit path's scoped engine steps."""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cluster_join_engine as engine
from gpu_fault.admin import cluster_join_readiness as readiness
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.collector_registry import CollectorKind
from gpu_fault_release import regional_admin_checks as engine_checks
from gpu_fault_release.regional_release_config import ReleaseError

NODES = ("node-1", "node-2")
FAST = ("GPU_INVENTORY", "HOST_TELEMETRY")
SLOW = ("FABRIC_MANAGER_LOG", "GPU_METRICS", "NODE_LOGS", "NVIDIA_KERNEL")


def _collector(*, reported: bool, unit_state: str = "active") -> dict:
    return {
        "unit": "gpu-fault-x",
        "unit_state": unit_state,
        "unit_enabled": "enabled",
        "last_success_at": "2026-09-12T09:29:00+00:00" if reported else None,
        "age_seconds": 12.0 if reported else None,
        "ready": reported,
    }


def _report(
    *,
    fast_reported: bool = True,
    slow_reported: bool = False,
    slow_unit_state: str = "active",
    nodes: tuple[str, ...] = NODES,
) -> dict:
    items = []
    for node in nodes:
        collectors = {kind: _collector(reported=fast_reported) for kind in FAST}
        collectors.update(
            {
                kind: _collector(reported=slow_reported, unit_state=slow_unit_state)
                for kind in SLOW
            }
        )
        items.append(
            {
                "node_id": node,
                "collectors": collectors,
                "ready": all(item["ready"] for item in collectors.values()),
            }
        )
    return {
        "cluster_id": "gpu-b",
        "ready": all(item["ready"] for item in items),
        "nodes": items,
    }


def _fleet(*, ready: bool = True, nodes: tuple[str, ...] = NODES) -> dict:
    return {
        "cluster_id": "gpu-b",
        "ready": ready,
        "nodes": [
            {
                "node_id": node,
                "ready": ready,
                "reasons": [] if ready else ["agent lease is expired"],
            }
            for node in nodes
        ],
    }


def test_slow_kinds_are_derived_from_the_configured_report_period() -> None:
    """No kind is named: a period above the ceiling is what makes a kind slow."""

    periods = readiness.collector_report_periods({})
    waited, deferred = readiness.classify_collector_kinds(periods)

    assert waited == frozenset(FAST)
    assert deferred == frozenset(SLOW)
    faster = readiness.collector_report_periods(
        {"GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "30"}
    )
    assert "GPU_METRICS" in readiness.classify_collector_kinds(faster)[0], (
        "a DCGM summary configured under the ceiling must be waited for"
    )
    assert set(periods) >= {
        kind
        for kind in CollectorKind
        if kind not in {CollectorKind.HMA_NODE, CollectorKind.HMA_CLOUDWATCH}
    }, "every node collector kind needs a report period"


def test_readiness_passes_with_slow_kinds_deferred() -> None:
    """Fast kinds reported on every node, slow units active, fleet live: ready."""

    verdict = readiness.evaluate_join_readiness(
        _report(),
        expected_nodes=NODES,
        fleet=_fleet(),
        periods=readiness.collector_report_periods({}),
    )

    assert verdict.ready is True
    evidence = verdict.evidence()
    assert evidence["waited_kinds"] == sorted(FAST)
    assert sorted(evidence["deferred_kinds"]) == sorted(SLOW)
    assert evidence["deferred_kinds"]["GPU_METRICS"] == {
        "report_period_seconds": 300.0,
        "verified_as": "scheduled",
        "scheduled_nodes": 2,
        "reported_nodes": 0,
        "nodes": 2,
    }
    assert evidence["nodes"] == sorted(NODES)


@pytest.mark.parametrize(
    ("report", "fleet", "expected"),
    [
        (_report(fast_reported=False), _fleet(), "waiting for GPU_INVENTORY"),
        (_report(slow_unit_state="failed"), _fleet(), "unit not active"),
        (_report(nodes=("node-1",)), _fleet(), "no Agent row for: node-2"),
        (_report(), _fleet(ready=False), "agent lease is expired"),
        (_report(), None, "fleet readiness was not evaluated"),
    ],
)
def test_readiness_refuses_a_missing_fast_report_or_a_dead_agent(
    report: dict, fleet: dict | None, expected: str
) -> None:
    verdict = readiness.evaluate_join_readiness(
        report,
        expected_nodes=NODES,
        fleet=fleet,
        periods=readiness.collector_report_periods({}),
    )

    assert verdict.ready is False
    assert expected in verdict.describe_shortfall()


def test_the_gate_returns_once_the_fast_kinds_have_reported() -> None:
    reports = iter(
        [
            {"collectors": _report(fast_reported=False), "fleet": _fleet()},
            {"collectors": _report(), "fleet": _fleet()},
        ]
    )
    slept: list[float] = []

    evidence = readiness.wait_join_collector_readiness(
        object(),
        "gpu-b",
        expected_nodes=NODES,
        timeout_seconds=100,
        interval_seconds=10,
        report=lambda *_args, **_kwargs: next(reports),
        clock=lambda: 0.0,
        sleep=slept.append,
    )

    assert evidence["ready"] is True
    assert slept == [10], "the gate polled once more after the first miss"
    assert sorted(evidence["deferred_kinds"]) == sorted(SLOW)


def test_the_gate_fails_hard_when_a_fast_kind_never_reports() -> None:
    clock = iter([0.0, 0.0, 5.0, 5.0, 11.0, 11.0, 11.0])

    with pytest.raises(BootstrapError, match="waiting for GPU_INVENTORY"):
        readiness.wait_join_collector_readiness(
            object(),
            "gpu-b",
            expected_nodes=NODES,
            timeout_seconds=10,
            interval_seconds=0,
            report=lambda *_a, **_k: {
                "collectors": _report(fast_reported=False),
                "fleet": _fleet(),
            },
            clock=lambda: next(clock),
            sleep=lambda _seconds: None,
        )


def _probe_document(*, joined_ready: bool) -> dict:
    return {
        "healthz": {"status": "ok"},
        "clusters": {
            "gpu-a": {
                "expected_node_ids": ["a-1"],
                "fleet_readiness": {"ready": True, "nodes": []},
                "collector_readiness": {"ready": True, "nodes": []},
            },
            "gpu-b": {
                "expected_node_ids": list(NODES),
                "fleet_readiness": _fleet(),
                "collector_readiness": _report(slow_reported=joined_ready),
            },
        },
    }


def test_control_api_readiness_is_relaxed_only_for_deferred_kinds() -> None:
    """The engine's whole-set verdict is replaced by the join's rule, and only
    for a joined cluster whose gate deferred kinds; nothing else moves."""

    gate = {"deferred_kinds": {kind: {} for kind in SLOW}}
    text = json.dumps(_probe_document(joined_ready=False))

    rewritten, relaxed = engine.relax_control_api_report(text, {"gpu-b": gate})
    document = json.loads(rewritten)

    assert document["clusters"]["gpu-b"]["collector_readiness"]["ready"] is True
    assert sorted(
        document["clusters"]["gpu-b"]["collector_readiness"]["join_deferred_kinds"]
    ) == sorted(SLOW)
    assert relaxed == {"gpu-b": sorted(SLOW)}
    assert (
        document["clusters"]["gpu-a"]
        == _probe_document(joined_ready=False)["clusters"]["gpu-a"]
    ), "a cluster the join did not touch was rewritten"

    untouched, relaxed_none = engine.relax_control_api_report(
        text, {"gpu-b": {"deferred_kinds": {}}}
    )
    assert untouched == text, "a gate that deferred nothing must not relax verify"
    assert relaxed_none == {}


def test_control_api_readiness_stays_refused_when_a_fast_kind_is_stale() -> None:
    document = _probe_document(joined_ready=False)
    for item in document["clusters"]["gpu-b"]["collector_readiness"]["nodes"]:
        item["collectors"]["HOST_TELEMETRY"]["ready"] = False

    rewritten, relaxed = engine.relax_control_api_report(
        json.dumps(document), {"gpu-b": {"deferred_kinds": {"GPU_METRICS": {}}}}
    )

    assert (
        json.loads(rewritten)["clusters"]["gpu-b"]["collector_readiness"]["ready"]
        is False
    )
    assert relaxed == {}


def test_join_verify_partitions_every_engine_check() -> None:
    """Kept plus skipped is exactly the engine's verify list; a new engine check
    has to be classified here before it can be silently dropped."""

    release = SimpleNamespace(
        config=SimpleNamespace(
            site_name="test-site", clusters=[SimpleNamespace(cluster_id="gpu-a")]
        )
    )
    report = engine_checks.build_health_report(release, mode="verify")
    engine_names = {item["name"] for item in report["checks"]}
    kept = {
        name.replace("<cluster_id>", "gpu-a") for name in engine.JOIN_VERIFY_KEPT_CHECKS
    }

    assert engine_names == kept | set(engine.JOIN_VERIFY_SKIPPED_CHECKS)
    assert not kept & set(engine.JOIN_VERIFY_SKIPPED_CHECKS)


def test_join_verify_runs_the_kept_checks_with_the_relaxed_control_api(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    probe = engine.probe_source("control_api_inspect")

    class Runner:
        def run(self, args, **_kwargs):
            calls.append("probe")
            assert probe in args, "the runner view intercepted a foreign command"
            return json.dumps(_probe_document(joined_ready=False))

    release = SimpleNamespace(
        config=SimpleNamespace(
            site_name="test-site", clusters=[SimpleNamespace(cluster_id="gpu-a")]
        ),
        runner=Runner(),
        _apply_health_baseline=lambda _state: None,
        _load_state=lambda: {},
        _read_snapshot=nullcontext,
        _prime_deployment_snapshot=lambda: None,
    )
    monkeypatch.setattr(engine, "build_release", lambda _site: release)
    monkeypatch.setattr(engine, "site_process_environment", lambda _site: nullcontext())
    for name in (
        "_check_contexts",
        "check_cpu_secrets",
        "_check_cpu_workloads",
        "_verify_profile",
        "run_read_only_verifiers",
        "_check_runtime_component_identity",
    ):
        monkeypatch.setattr(
            engine_checks,
            name,
            lambda _release, _name=name: calls.append(_name)
            or engine_checks.CheckValue(_name),
        )
    monkeypatch.setattr(
        engine_checks,
        "_check_gpu_cluster",
        lambda _release, target: calls.append(f"gpu:{target.cluster_id}")
        or engine_checks.CheckValue("gpu"),
    )

    def control_api(view):
        calls.append("control_api")
        document = json.loads(view.runner.run([probe], capture=True))
        assert document["clusters"]["gpu-b"]["collector_readiness"]["ready"] is True
        return engine_checks.CheckValue("control api")

    monkeypatch.setattr(engine_checks, "_check_control_api", control_api)

    report = engine.verify_joined_clusters(
        object(), readiness={"gpu-b": {"deferred_kinds": {"GPU_METRICS": {}}}}
    )

    assert report["healthy"] is True
    assert {item["name"] for item in report["checks"]} == {
        "regional_contexts",
        "cpu_secrets",
        "cpu_workloads",
        "runtime_profile",
        "read_only_verifiers",
        "runtime_component_identity",
        "control_api",
        "gpu_cluster:gpu-a",
    }
    assert report["scope"]["relaxed_collector_readiness"] == {"gpu-b": sorted(SLOW)}
    assert set(report["scope"]["skipped_checks"]) == set(
        engine.JOIN_VERIFY_SKIPPED_CHECKS
    )
    assert "gpu:gpu-a" in calls
    assert json.loads(capsys.readouterr().out)["mode"] == "verify", (
        "the report is printed like the engine's verify"
    )
    summary = engine.verify_report_summary(report)
    assert summary["summary"]["FAIL"] == 0
    assert summary["relaxed_collector_readiness"] == {"gpu-b": sorted(SLOW)}


def test_join_verify_fails_on_a_failing_check(monkeypatch: pytest.MonkeyPatch) -> None:
    release = SimpleNamespace(
        config=SimpleNamespace(site_name="test-site", clusters=[]),
        runner=SimpleNamespace(run=lambda *_a, **_k: "{}"),
        _apply_health_baseline=lambda _state: None,
        _load_state=lambda: {},
        _read_snapshot=nullcontext,
        _prime_deployment_snapshot=lambda: None,
    )
    monkeypatch.setattr(engine, "build_release", lambda _site: release)
    monkeypatch.setattr(engine, "site_process_environment", lambda _site: nullcontext())
    for name in (
        "_check_contexts",
        "check_cpu_secrets",
        "_check_cpu_workloads",
        "_verify_profile",
        "run_read_only_verifiers",
        "_check_runtime_component_identity",
        "_check_control_api",
    ):
        monkeypatch.setattr(
            engine_checks, name, lambda _release: engine_checks.CheckValue("ok")
        )
    monkeypatch.setattr(
        engine_checks,
        "_check_cpu_workloads",
        lambda _release: (_ for _ in ()).throw(ReleaseError("adot 0/1 Ready")),
    )

    with pytest.raises(BootstrapError, match="cpu_workloads: adot 0/1 Ready"):
        engine.verify_joined_clusters(object(), readiness={"gpu-b": {}})


def _release_for_capture(
    *,
    cpu_image: str = "img@sha256:aa",
    live_state: dict | None = None,
    node_names: dict[str, list[str]] | None = None,
    saved: list[tuple[str, dict]] | None = None,
) -> SimpleNamespace:
    names = node_names or {"gpu-a": ["a-1"], "gpu-b": list(NODES)}
    clusters = [SimpleNamespace(cluster_id=name) for name in ("gpu-a", "gpu-b")]
    return SimpleNamespace(
        config=SimpleNamespace(clusters=clusters),
        _save_state=lambda phase, **updates: (
            saved.append((phase, updates)) if saved is not None else None
        ),
        node_installer_image="installer@sha256:11",
        cpu_image=cpu_image,
        _target=lambda cluster_id: next(
            item for item in clusters if item.cluster_id == cluster_id
        ),
        _read_snapshot=nullcontext,
        _prime_deployment_snapshot=lambda: None,
        _load_state=lambda: dict(
            live_state
            if live_state is not None
            else {"runtime_image": "img@sha256:aa", "node_installer_image": ""}
        ),
        _cpu=lambda *args: ["cpu", *args],
        _target_node_names=lambda target: tuple(names[target.cluster_id]),
    )


def _capture_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    joined_image: str = "img@sha256:aa",
    agents: dict[str, list[str]] | None = None,
) -> list[str]:
    captured: list[str] = []
    monkeypatch.setattr(
        engine,
        "capture_gpu_cluster_snapshot",
        lambda _release, target: (
            captured.append(target.cluster_id)
            or (
                {"wheel": "w", "dcgm_image": "dcgm"},
                {f"{target.cluster_id}/executor": joined_image},
                "installer@sha256:11",
            )
        ),
    )
    monkeypatch.setattr(
        engine,
        "deployment_image",
        lambda release, _args, deployment, **_k: release.cpu_image,
    )
    rows = agents or {"gpu-a": ["a-1"], "gpu-b": list(NODES)}
    monkeypatch.setattr(
        engine,
        "capture_agent_identities",
        lambda _release: {
            cluster: {"artifact_sha256": "s", "node_ids": list(nodes)}
            for cluster, nodes in rows.items()
        },
    )
    return captured


def test_scoped_sync_state_captures_only_the_joined_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_stubs(monkeypatch)
    release = _release_for_capture()

    document = engine.capture_joined_cluster_previous(release, "gpu-b")

    assert captured == ["gpu-b"], "an existing cluster was re-captured"
    assert document["live_runtime_image"] == "img@sha256:aa"
    assert set(document["clusters"]) == {"gpu-b"}
    assert document["agent_nodes"] == {"gpu-a": 1, "gpu-b": 2}
    assert document["capture_scope"] == "cluster:gpu-b"


def test_scoped_sync_state_keeps_the_merged_image_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The joined cluster's image is judged against the CPU images *and* the
    image the committed state records for the clusters that were not re-read."""

    _capture_stubs(monkeypatch, joined_image="img@sha256:bb")

    with pytest.raises(ReleaseError, match="runtime images are inconsistent"):
        engine.capture_joined_cluster_previous(_release_for_capture(), "gpu-b")

    _capture_stubs(monkeypatch)
    with pytest.raises(ReleaseError, match="release-state/runtime_image"):
        engine.capture_joined_cluster_previous(
            _release_for_capture(live_state={"runtime_image": "img@sha256:old"}),
            "gpu-b",
        )


def test_scoped_sync_state_keeps_the_merged_agent_set_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_stubs(monkeypatch, agents={"gpu-a": ["a-1", "a-stale"], "gpu-b": NODES})

    with pytest.raises(ReleaseError, match="gpu-a active Agent set"):
        engine.capture_joined_cluster_previous(_release_for_capture(), "gpu-b")


def test_scoped_sync_state_feeds_the_engine_through_the_capture_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real ``sync_release_state`` runs: it reads the merged capture through
    the seam and writes the committed NOOP state for every configured cluster."""

    _capture_stubs(monkeypatch)
    saved: list[tuple[str, dict]] = []
    release = _release_for_capture(saved=saved)
    monkeypatch.setattr(engine, "build_release", lambda _site: release)
    monkeypatch.setattr(engine, "site_process_environment", lambda _site: nullcontext())

    evidence = engine.sync_cluster_release_state(object(), cluster_id="gpu-b")

    assert [phase for phase, _updates in saved] == ["complete"]
    updates = saved[0][1]
    assert updates["adopted_live_runtime_image"] == "img@sha256:aa", (
        "the engine's sync-state did not read the merged capture"
    )
    assert updates["previous"] is None
    assert updates["completed_cluster_ids"] == ["gpu-a", "gpu-b"]
    assert updates["release_lifecycle"] == "COMMITTED"
    assert evidence == {
        "capture_scope": "cluster:gpu-b",
        "live_runtime_image": "img@sha256:aa",
        "agent_nodes": {"gpu-a": 1, "gpu-b": 2},
    }
