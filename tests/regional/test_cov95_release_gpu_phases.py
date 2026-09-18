from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_gpu_rollout as gpu
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseComponent as Component
from gpu_fault_release.regional_release_diff import (
    ReleaseExecutionPlan,
    diff_from_changed,
)
from tests.regional._cov95_release_support import ResourceRelease, deployment


def render_model(
    monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]
) -> tuple[ResourceRelease, list[str]]:
    release = ResourceRelease()
    release.documents[("gpu-a", "pods", "")] = {"items": []}
    events: list[str] = []

    def run(arguments: list[str], kwargs: dict[str, Any]) -> str:
        if "apply" in arguments:
            documents = list(yaml.safe_load_all(kwargs["input_text"]))
            prefix = "preflight" if "--dry-run=server" in arguments else "apply"
            events.append(f"{prefix}:{documents[0]['metadata']['name']}")
            return ""
        if "rolebindings" in arguments:
            events.append("prune-read")
            return json.dumps({"items": []})
        raise AssertionError(f"unexpected fake GPU transport: {arguments[:4]}")

    def render(_release: Any, _target: Any, wheel: str, **kwargs: Any) -> list[Any]:
        assert kwargs["deployment_names"] == frozenset(names)
        assert wheel == "candidate-wheel"
        return [(name, yaml.safe_dump(deployment(name))) for name in names]

    def wait(_release: Any, _target: Any, name: str) -> dict[str, Any]:
        events.append(f"wait:{name}")
        return {"deployment": name}

    release.runner.handler = run
    monkeypatch.setattr(gpu, "render_gpu_rollout_manifests", render)
    monkeypatch.setattr(gpu, "wait_deployment_rollout", wait)
    return release, events


@pytest.mark.parametrize(
    "name", [inventory.GPU_EXECUTOR_DEPLOYMENT, inventory.GPU_COLLECTOR_DEPLOYMENT]
)
def test_selected_gpu_component_preflights_applies_waits_then_scopes_rbac_cleanup(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    release, events = render_model(monkeypatch, (name,))
    gpu.apply_gpu_deployments(
        release,
        release.config.clusters[0],
        "candidate-wheel",
        deployment_names=frozenset({name}),
    )
    expected = [f"preflight:{name}", f"apply:{name}", f"wait:{name}"]
    if name == inventory.GPU_EXECUTOR_DEPLOYMENT:
        expected.append("prune-read")
    assert events == expected


def test_candidate_preflight_does_not_require_a_live_candidate_pin_or_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = inventory.GPU_COLLECTOR_DEPLOYMENT
    release, events = render_model(monkeypatch, (name,))
    release.metadata.clear()
    gpu.preflight_gpu_deployments(
        release,
        release.config.clusters[0],
        "candidate-wheel",
        deployment_names=frozenset({name}),
    )
    assert events == [f"preflight:{name}"]
    assert release.reads == [("gpu-a", "pods", "")]


@pytest.mark.parametrize("failure", ["preflight", "wait"])
def test_gpu_phase_failure_stops_later_mutation_and_rbac_pruning(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    executor, collector = (
        inventory.GPU_EXECUTOR_DEPLOYMENT,
        inventory.GPU_COLLECTOR_DEPLOYMENT,
    )
    release, events = render_model(monkeypatch, (executor, collector))
    original = release.runner.handler
    assert original is not None

    def run(arguments: list[str], kwargs: dict[str, Any]) -> str:
        result = original(arguments, kwargs)
        if failure == "preflight" and events[-1] == f"preflight:{collector}":
            raise ReleaseError("admission refused")
        return result

    def wait(_release: Any, _target: Any, name: str) -> dict[str, Any]:
        events.append(f"wait:{name}")
        if name == executor:
            raise ReleaseError("pods did not become Ready")
        return {"deployment": name}

    release.runner.handler = run
    monkeypatch.setattr(gpu, "wait_deployment_rollout", wait)
    with pytest.raises(
        ReleaseError,
        match="admission refused"
        if failure == "preflight"
        else f"{executor} rollout failed",
    ):
        gpu.apply_gpu_deployments(
            release,
            release.config.clusters[0],
            "candidate-wheel",
            deployment_names=frozenset({executor, collector}),
        )
    expected = [f"preflight:{executor}", f"preflight:{collector}"]
    if failure == "wait":
        expected += [f"apply:{executor}", f"apply:{collector}"]
        assert events[:4] == expected
        assert sorted(events[4:]) == sorted([f"wait:{executor}", f"wait:{collector}"])
    else:
        assert events == expected
    assert "prune-read" not in events


@pytest.mark.parametrize(
    "key,value,problem",
    [
        (
            "required-regional-executor-protocol-version",
            "bad",
            "invalid regional executor pin",
        ),
        (
            "required-regional-executor-artifact-sha256",
            "f" * 64,
            "pin preflight rejected",
        ),
    ],
)
def test_executor_pin_refuses_malformed_or_incompatible_live_metadata(
    key: str, value: str, problem: str
) -> None:
    release = ResourceRelease()
    release.metadata[key] = value
    with pytest.raises(ReleaseError, match=problem):
        gpu.require_executor_pin(
            release,
            artifact_sha=release.executor_wheel_sha,
            compatibility_digest=release.config.component_digests["executor"],
        )
    assert release.runner.calls == []
    assert release.reads == [("cpu", "configmap", "gpu-fault-release-metadata")]


@pytest.mark.parametrize("record_progress", [False, True])
def test_failed_selected_component_reports_failure_without_advancing(
    record_progress: bool,
) -> None:
    events = []
    progress = []

    def connect(_target: Any) -> None:
        events.append("connect")
        raise ReleaseError("connection update rejected")

    release = SimpleNamespace(
        _ensure_connection_secret=connect,
        _verify_gpu_control_plane_endpoint=lambda _target: events.append("verify"),
    )
    with pytest.raises(ReleaseError, match="connection update rejected"):
        gpu.upgrade_gpu_target(
            release,
            SimpleNamespace(cluster_id="gpu-a"),
            diff_from_changed({"endpoint"}),
            ReleaseExecutionPlan((Component.ENDPOINT,)),
            progress=(lambda *args: progress.append(args)) if record_progress else None,
        )
    assert events == ["connect"]
    assert progress == (
        [
            ((Component.ENDPOINT,), "STARTED", None),
            ((Component.ENDPOINT,), "FAILED", None),
        ]
        if record_progress
        else []
    )


def test_agent_component_completion_follows_the_mutation_started_callback() -> None:
    release = ResourceRelease()
    attempts: list[dict[str, Any]] = []
    progress = []

    def roll(_target: Any, **kwargs: Any) -> None:
        attempts.append(kwargs)
        assert progress == []
        kwargs["mutation_started"]()
        assert progress == [((Component.RECONCILER, Component.AGENT), "STARTED", None)]

    model = SimpleNamespace(
        config=release.config,
        executor_wheel_cm="executor-wheel",
        bundle_cm="bundle",
        node_wheel_sha=release.node_wheel_sha,
        _roll_node_runtime=roll,
    )
    gpu.upgrade_gpu_target(
        model,
        release.config.clusters[0],
        diff_from_changed({"node_wheel"}),
        ReleaseExecutionPlan((Component.RECONCILER, Component.AGENT)),
        progress=lambda *args: progress.append(args),
        candidate_preflighted=True,
    )
    assert progress[-1] == ((Component.RECONCILER, Component.AGENT), "COMPLETED", None)
    assert len(attempts) == 1
    assert attempts[0]["phase"] == "upgrade"
    assert attempts[0]["candidate_preflight_completed"] is True
    assert attempts[0]["artifact_sha"] == release.node_wheel_sha


def test_join_refuses_active_remote_commands_before_returning_a_target() -> None:
    target = SimpleNamespace(cluster_id="gpu-a")
    lookups = []
    release = SimpleNamespace(
        _target=lambda cluster_id: lookups.append(cluster_id) or target,
        _remote_commands_are_idle=lambda: False,
    )
    with pytest.raises(
        ReleaseError, match="remote commands are PENDING/LEASED/WAITING"
    ):
        gpu.join_target(release, "gpu-a")
    assert lookups == ["gpu-a"]
