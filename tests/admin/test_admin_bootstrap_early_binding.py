from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import bootstrap, notification_bootstrap
from gpu_fault.admin.bootstrap_checkpoint import bind_bootstrap_inputs
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    BootstrapState,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_task_inputs import task_input_spec
from gpu_fault.admin.bootstrap_tasks import TaskGraph, TaskSpec
from gpu_fault.admin.config import (
    LEGACY_AURORA_MAX_ACU,
    LEGACY_AURORA_MIN_ACU,
    AdminConfigError,
    AuroraCapacityConfig,
    default_admin_config,
    persist_desired_admin_config,
)
from gpu_fault.admin.notifications import NotificationRouting
from tests.admin._bootstrap_support import _cluster
from tests.admin.test_admin_config import legacy_capacity_record


@pytest.fixture
def inputs(tmp_path: Path):
    cpu = _cluster()
    gpu = replace(
        cpu,
        role="gpu",
        input_arn=cpu.input_arn.replace("/control", "/gpu-a"),
        eks_arn=cpu.eks_arn.replace("/control", "/gpu-a"),
        eks_name="gpu-a",
        hyperpod_arn=cpu.hyperpod_arn.replace("/control", "/gpu-a"),
        hyperpod_name="gpu-a",
        context="gpu-a",
        vpc_id="vpc-gpu-a",
    )
    request = BootstrapRequest(
        cpu.input_arn,
        (gpu.input_arn,),
        Path(__file__).resolve().parents[2],
        tmp_path / "state",
        alert_email="admin@example.com",
    )
    persist_desired_admin_config(
        request.state_dir, config=default_admin_config(), source="test"
    )
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({"release_id": "candidate-a"}))
    release = {
        "manifest": str(manifest),
        "release_id": "candidate-a",
        "agent_config_digest": "c" * 64,
        "images": {},
    }
    state = BootstrapState(request.state_dir / "bootstrap-state.json", site_id="test")
    return request, cpu, (gpu,), release, state


def bind(inputs, *, release: bool = False) -> None:
    request, cpu, gpu, candidate, state = inputs
    bind_bootstrap_inputs(
        state,
        request=request,
        cpu=cpu,
        gpu_clusters=gpu,
        release=candidate if release else None,
    )


def test_full_bind_retains_only_proven_early_completions_across_crash_resume(inputs):
    *_, state = inputs
    bind(inputs)
    early = dict(state.value["task_input_sha256"])
    assert "release" not in early, "an unfinished release acquired an input identity"
    for name in early:
        state.record(name, {"name": name})
        state.complete(name)
    state.record("unknown-task", {})
    state.complete("unknown-task")
    bind(inputs, release=True)
    assert state.value["completed_tasks"] == sorted(early), (
        "late binding lost proven work or retained a task with no declared identity"
    )
    assert {name: state.value["task_input_sha256"][name] for name in early} == early
    restored = BootstrapState(state.path, site_id="test")
    bind((*inputs[:-1], restored))
    assert restored.value["completed_tasks"] == sorted(early), (
        "restart erased tasks completed against the same early inputs"
    )


@pytest.mark.parametrize("task", ["nlb_network", "pki", "aurora"])
@pytest.mark.parametrize("reload", [False, True])
def test_changed_or_unproven_completion_is_not_a_current_process_exemption(
    inputs, task, reload
):
    *_, state = inputs
    state.record(task, {"previous": True})
    state.complete(task)
    bind(inputs)
    assert not state.is_complete(task), "an unproven completion became reusable"
    state.complete(task)
    if reload:
        state = BootstrapState(state.path, site_id="test")
    state.bind_inputs("changed", {task: "changed"})
    assert not state.is_complete(task), "a changed digest retained task completion"
    assert state.result(task) == {"previous": True}, "invalidation erased audit state"


@pytest.mark.parametrize("task", ["nlb_network", "pki"])
def test_stale_foundation_runs_in_the_deploy_that_changed_inputs(inputs, task):
    *_, previous = inputs
    previous.bind_inputs("old", {task: "old"})
    previous.record(task, {"generation": "previous"})
    previous.complete(task)
    state = BootstrapState(previous.path, site_id="test")
    bind((*inputs[:-1], state))
    calls = []
    graph = TaskGraph(
        {
            task: TaskSpec(
                lambda: calls.append(task) or {"generation": "current"},
                input_policy=task_input_spec(task),
            )
        }
    )
    assert graph.run(state=state) == {task: {"generation": "current"}}
    assert calls == [task], "the stale checkpoint survived until the next deploy"
    bind((*inputs[:-1], state), release=True)
    assert state.is_complete(task), "candidate binding erased proven early work"
    assert graph.run(state=BootstrapState(state.path, site_id="test")) == {
        task: {"generation": "current"}
    }
    assert calls == [task], "crash resume repeated work with unchanged inputs"


def test_a_bound_graph_missing_task_identity_refuses_before_probe_or_ensure(inputs):
    *_, state = inputs
    state.bind_inputs("whole-inputs", {})
    state.record("pki", {})
    state.complete("pki")
    calls = []
    graph = TaskGraph(
        {
            "pki": TaskSpec(
                lambda: calls.append("ensure"),
                input_policy=task_input_spec("pki"),
                probe=lambda: calls.append("probe"),
                revalidate=True,
            )
        }
    )
    with pytest.raises(BootstrapError, match="input identities are unbound: pki"):
        graph.run(state=state)
    assert calls == [], "missing identity reached a probe or an ensure"


def test_unbound_graph_does_not_reuse_or_probe_a_checkpoint(inputs):
    *_, state = inputs
    state.record("pki", {"previous": True})
    state.complete("pki")
    calls = []
    graph = TaskGraph(
        {
            "pki": TaskSpec(
                lambda: calls.append("ensure") or {"current": True},
                input_policy=task_input_spec("pki"),
                probe=lambda: calls.append("probe"),
                revalidate=True,
            )
        }
    )
    assert graph.run(state=state) == {"pki": {"current": True}}
    assert calls == ["ensure"], "an unbound checkpoint authorized a cached proof"
    bind(inputs)
    assert not state.is_complete("pki"), "late binding vouched for unproven work"


class NoCommands(CommandRunner):
    def run(self, arguments, **_kwargs):
        pytest.fail(f"early binding attempted a command: {arguments[:3]}")


class ReachedGraph(RuntimeError):
    pass


@pytest.mark.parametrize("tampered", [False, True])
def test_legacy_config_migration_precedes_binding_build_and_foundation(
    inputs, monkeypatch, tampered
):
    request, cpu, gpu, release, _state = inputs
    desired = request.state_dir / "admin-config/desired.json"
    legacy = legacy_capacity_record(default_admin_config())
    if tampered:
        legacy["config_sha256"] = "invalid"
    desired.write_text(json.dumps(legacy))
    monkeypatch.setattr(bootstrap, "validate_bootstrap_dependencies", lambda: None)
    monkeypatch.setattr(
        bootstrap, "discover_bootstrap_scope", lambda **_: (None, cpu, gpu)
    )
    monkeypatch.setattr(
        notification_bootstrap,
        "notification_routing",
        lambda *_, **__: (
            request.alert_email,
            NotificationRouting(
                sender="admin@example.com",
                recipients=("admin@example.com",),
                subject_prefix="test",
            ),
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "plan_cluster_access",
        lambda *_, **__: bootstrap.ClusterAccessPlan(
            TaskGraph({}), desired.parent, desired.parent, desired.parent
        ),
    )
    build_ready = threading.Event()
    observed: dict[str, Any] = {}

    def prepare(**kwargs):
        state = kwargs["state"]
        observed["before_build"] = json.loads(desired.read_text())
        observed["early"] = state.value["task_input_sha256"]["aurora"]
        try:
            bind_bootstrap_inputs(
                state, request=request, cpu=cpu, gpu_clusters=gpu, release=release
            )
            observed["built"] = state.value["task_input_sha256"]["aurora"]
            return release
        finally:
            build_ready.set()

    def foundation(**kwargs):
        assert build_ready.wait(5), "the test release build did not finish"
        observed["capacity"] = kwargs["aurora_capacity"]
        raise ReachedGraph

    monkeypatch.setattr(bootstrap, "prepare_signed_release", prepare)
    monkeypatch.setattr(bootstrap, "foundation_task_graph", foundation)
    expected = AdminConfigError if tampered else ReachedGraph
    with pytest.raises(expected):
        bootstrap.bootstrap_from_arns(request, runner=NoCommands())
    if tampered:
        assert observed == {}, "invalid config reached release build or graph setup"
        assert json.loads(desired.read_text()) == legacy
    else:
        migrated = observed["before_build"]
        assert "aurora" in migrated["config"], "build started before config migration"
        assert migrated == json.loads(desired.read_text()), (
            "foundation changed the config bytes after early binding"
        )
        assert observed["early"] == observed["built"], (
            "early and candidate binds hashed different Aurora inputs"
        )
        assert observed["capacity"] == AuroraCapacityConfig(
            min_acu=LEGACY_AURORA_MIN_ACU, max_acu=LEGACY_AURORA_MAX_ACU
        ), "legacy migration silently changed the established Aurora capacity"
