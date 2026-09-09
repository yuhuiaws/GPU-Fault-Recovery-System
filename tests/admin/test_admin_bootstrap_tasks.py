"""The bootstrap task graph: what each task waits for, and what re-proves it.

These builders decide three things no other module can: which tasks exist,
which of them are re-proved by a read-only probe every run instead of being
trusted from ``completed_tasks``, and which task reads another's result. All are
exercised through the real ``run_parallel`` so a rerun here means what a rerun
means in bootstrap, and the scheduler itself is pinned with tasks that signal
each other, so "started while" is observed rather than timed.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

from gpu_fault.admin import (
    bootstrap_load_balancer,
    bootstrap_services,
    bootstrap_tasks,
    notification_bootstrap,
)
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapMutationRequired,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
    ReadOnlyProbeRunner,
    run_parallel,
)
from gpu_fault.admin.notifications import NotificationRouting
from tests.admin._bootstrap_support import _cluster


class Recorder:
    """Stands in for one ``ensure`` function and remembers how it was called."""

    def __init__(self, name: str, log: list[tuple[str, bool, bool]]) -> None:
        self.name = name
        self.log = log
        self.keywords: list[dict[str, Any]] = []

    def __call__(self, runner: CommandRunner, *_args: Any, **keywords: Any) -> dict:
        read_only = isinstance(runner, ReadOnlyProbeRunner)
        self.log.append((self.name, read_only, bool(keywords.get("probe_only"))))
        self.keywords.append(keywords)
        return {"task": self.name}


def _gpu(name: str) -> ClusterIdentity:
    return replace(
        _cluster(), role="gpu", hyperpod_name=name, eks_name=name, context=name
    )


def _state(
    tmp_path: Path, *, completed: dict[str, Any] | None = None
) -> BootstrapState:
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")
    for name, value in (completed or {}).items():
        state.record(name, value)
        state.complete(name)
    return state


def _routing() -> NotificationRouting:
    return NotificationRouting(
        sender="alerts@example.com",
        recipients=("ops@example.com",),
        subject_prefix="[gpu-fault]",
    )


def test_pod_identity_agent_is_reproved_on_every_run(tmp_path: Path) -> None:
    """The agent is a cluster add-on that can be removed out of band.

    Trusting ``completed_tasks`` for it would leave every Pod Identity
    association pointing at an agent that is no longer installed, so a completed
    run still re-enters the check -- read-only unless it finds drift.
    """

    log: list[tuple[str, bool, bool]] = []
    ensure = Recorder("pod_identity_agent", log)
    runner = CommandRunner()

    bootstrap_tasks.revalidate_pod_identity_agent(
        runner, _state(tmp_path), _cluster(), "site-a", ensure
    )
    assert log == [("pod_identity_agent", False, False)]

    log.clear()
    bootstrap_tasks.revalidate_pod_identity_agent(
        runner,
        _state(tmp_path, completed={"pod_identity_agent": {"agent": "present"}}),
        _cluster(),
        "site-a",
        ensure,
    )

    assert log == [("pod_identity_agent", True, False)], (
        "a completed pod identity agent task was trusted instead of re-proved"
    )


def test_a_probe_that_needs_a_mutation_falls_back_to_ensure(tmp_path: Path) -> None:
    """Drift detected read-only must escalate to the mutating path.

    The probe is how bootstrap avoids touching healthy resources; if a probe that
    reports drift were treated as a failure, a rerun could never repair anything.
    """

    calls: list[bool] = []

    def ensure(runner: CommandRunner, *_args: Any) -> dict:
        read_only = isinstance(runner, ReadOnlyProbeRunner)
        calls.append(read_only)
        if read_only:
            raise BootstrapMutationRequired("aws")
        return {"agent": "reinstalled"}

    bootstrap_tasks.revalidate_pod_identity_agent(
        CommandRunner(),
        _state(tmp_path, completed={"pod_identity_agent": {"agent": "present"}}),
        _cluster(),
        "site-a",
        ensure,
    )

    assert calls == [True, False]


# --- the platform graph on its own ------------------------------------------------------

# What the platform tasks read from the foundation; when the platform graph runs
# alone these must already be complete in the state.
FOUNDATION_RESULTS = {
    "monitoring_resources": {"workspace_id": "ws-a", "sns_topic_arn": "arn:sns"},
    "aurora": {"cluster_id": "gpu-fault-site-a-aurora", "instance_ids": ["w", "r"]},
}


def _platform_graph(
    tmp_path: Path,
    state: BootstrapState,
    log: list[tuple[str, bool, bool]],
    *,
    ensure_aurora_ready: Callable[..., Any] | None = None,
) -> bootstrap_tasks.TaskGraph:
    return bootstrap_tasks.platform_task_graph(
        runner=CommandRunner(),
        state=state,
        repository_root=tmp_path,
        cpu=_cluster(),
        gpu_clusters=[_gpu("gpu-a"), _gpu("gpu-b")],
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        adot_image="adot:1",
        alert_email=None,
        release_manifest=tmp_path / "manifest.json",
        runtime_image="runtime:1",
        fleet_master_file=tmp_path / "master",
        ensure_aurora_ready=ensure_aurora_ready or Recorder("aurora_ready", log),
    )


def _stub_platform_services(
    monkeypatch: pytest.MonkeyPatch, log: list[tuple[str, bool, bool]]
) -> Recorder:
    """Replace the three service installers with recorders; returns the
    credential-refresh one, whose received keywords a test may inspect."""

    refresh = Recorder("install_aurora_refresh", log)
    monkeypatch.setattr(
        bootstrap_services, "install_monitoring", Recorder("install_monitoring", log)
    )
    monkeypatch.setattr(bootstrap_services, "install_aurora_refresh", refresh)
    monkeypatch.setattr(
        bootstrap_services,
        "provision_node_action_keys",
        lambda runner, **keywords: Recorder(f"node_keys:{keywords['cluster_id']}", log)(
            runner, **keywords
        ),
    )
    return refresh


def test_every_platform_prerequisite_task_is_reproved_per_gpu_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node keys are provisioned per GPU cluster, and none of the tasks is cached.

    Node membership changes without bootstrap running, and the manifests these
    tasks apply can be edited in the cluster, so a completed run re-proves all of
    them read-only. A per-cluster task name is what makes joining a second GPU
    cluster provision its own keys rather than reuse the first cluster's.
    """

    log: list[tuple[str, bool, bool]] = []
    _stub_platform_services(monkeypatch, log)

    state = _state(tmp_path, completed=FOUNDATION_RESULTS)
    _platform_graph(tmp_path, state, log).run(state=state)

    assert {name for name, _read_only, _probe in log} == {
        "install_monitoring",
        "aurora_ready",
        "install_aurora_refresh",
        "node_keys:gpu-a",
        "node_keys:gpu-b",
    }
    assert not any(read_only for _name, read_only, _probe in log), (
        "a first run reached the read-only probe instead of ensure"
    )
    assert not any(probe for _name, _read_only, probe in log), (
        "a first run ran a probe instead of the task itself"
    )

    log.clear()
    state = _state(
        tmp_path,
        completed={
            **FOUNDATION_RESULTS,
            "monitoring_install": {},
            "aurora_ready": {},
            "aurora_refresh": {},
            "node_keys:gpu-a": {},
            "node_keys:gpu-b": {},
        },
    )
    _platform_graph(tmp_path, state, log).run(state=state)

    assert len(log) == 5
    assert all(read_only for _name, read_only, _probe in log), (
        "a completed prerequisite task was re-run with the mutating runner"
    )
    assert all(probe for _name, _read_only, probe in log), (
        "a probe run did not pass probe_only"
    )


def test_the_credential_refresh_follows_readiness_and_reads_its_secret_arn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``aurora_ready`` owns the master Secret ARN now that the instance wait
    left the ``aurora`` task; the refresh CronJob renders that ARN, so it runs
    after readiness and sees the readiness result merged over ``aurora``. A
    checkpoint written before the split carries the ARN on ``aurora``, and the
    merge serves that shape too."""

    log: list[tuple[str, bool, bool]] = []
    refresh = _stub_platform_services(monkeypatch, log)

    def ready(runner: CommandRunner, **keywords: Any) -> dict[str, str]:
        assert keywords["aurora"] == FOUNDATION_RESULTS["aurora"]
        log.append(("aurora_ready", False, False))
        return {"master_secret_arn": "arn:new", "master_secret_kms_key_arn": "kms"}

    state = _state(tmp_path, completed=FOUNDATION_RESULTS)
    results = _platform_graph(tmp_path, state, log, ensure_aurora_ready=ready).run(
        state=state
    )

    names = [name for name, _read_only, _probe in log]
    assert names.index("aurora_ready") < names.index("install_aurora_refresh")
    assert refresh.keywords[0]["aurora"] == {
        **FOUNDATION_RESULTS["aurora"],
        "master_secret_arn": "arn:new",
        "master_secret_kms_key_arn": "kms",
    }
    assert results["aurora_ready"] == {
        "master_secret_arn": "arn:new",
        "master_secret_kms_key_arn": "kms",
    }

    old_shape = {**FOUNDATION_RESULTS["aurora"], "master_secret_arn": "arn:old"}
    state = _state(
        tmp_path / "old-shape", completed={**FOUNDATION_RESULTS, "aurora": old_shape}
    )
    _platform_graph(tmp_path, state, log, ensure_aurora_ready=lambda *_a, **_k: {}).run(
        state=state
    )
    assert refresh.keywords[-1]["aurora"] == old_shape


def test_a_completed_readiness_task_is_reproved_read_only_and_its_result_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[tuple[str, bool, bool]] = []
    _stub_platform_services(monkeypatch, log)
    ready = Recorder("aurora_ready", log)
    recorded = {"master_secret_arn": "arn:recorded", "master_secret_kms_key_arn": ""}
    state = _state(tmp_path, completed={**FOUNDATION_RESULTS, "aurora_ready": recorded})

    results = _platform_graph(tmp_path, state, log, ensure_aurora_ready=ready).run(
        state=state
    )

    assert [entry for entry in log if entry[0] == "aurora_ready"] == [
        ("aurora_ready", True, True)
    ], "a completed aurora_ready was not re-proved by its read-only probe"
    assert results["aurora_ready"] == recorded, (
        "the probe's return value replaced the checkpointed readiness result"
    )


# --- the foundation graph on its own ----------------------------------------------------


def _foundation_graph(
    tmp_path: Path,
    state: BootstrapState,
    log: list[tuple[str, bool, bool]],
    *,
    ensure_aurora: Callable[..., Any] | None = None,
    ensure_nlb_network: Callable[..., Any] | None = None,
) -> bootstrap_tasks.TaskGraph:
    return bootstrap_tasks.foundation_task_graph(
        runner=CommandRunner(),
        state=state,
        cpu=_cluster(),
        gpu_clusters=[_gpu("gpu-a")],
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        state_dir=tmp_path,
        admin_email="admin@example.com",
        routing=_routing(),
        aurora_capacity=None,
        ensure_nlb_network=ensure_nlb_network or Recorder("nlb_network", log),
        ensure_pki=Recorder("pki", log),
        ensure_aurora=ensure_aurora or Recorder("aurora", log),
    )


def _stub_foundation_services(
    monkeypatch: pytest.MonkeyPatch, log: list[tuple[str, bool, bool]]
) -> None:
    monkeypatch.setattr(
        bootstrap_load_balancer,
        "ensure_load_balancer_controller",
        lambda runner, **_keywords: Recorder("load_balancer_controller", log)(runner),
    )
    monkeypatch.setattr(
        bootstrap_services,
        "ensure_executor_role",
        lambda runner, **keywords: Recorder(
            f"executor_role:{keywords['cluster'].hyperpod_name}", log
        )(runner),
    )
    monkeypatch.setattr(
        notification_bootstrap,
        "notification_bootstrap_tasks",
        lambda runner, **_keywords: {
            name: (lambda name=name: Recorder(name, log)(runner))
            for name in (
                "control_plane_role",
                "email_notifications",
                "monitoring_resources",
            )
        },
    )


FOUNDATION_TASKS = {
    "nlb_network",
    "pki",
    "aurora",
    "load_balancer_controller",
    "control_plane_role",
    "email_notifications",
    "monitoring_resources",
    "executor_role:gpu-a",
}
FOUNDATION_REPROVED = {
    "load_balancer_controller",
    "control_plane_role",
    "email_notifications",
    "monitoring_resources",
    "executor_role:gpu-a",
}
# ``_platform_graph`` provisions node keys for two GPU clusters.
PLATFORM_TASKS = {
    "monitoring_install",
    "aurora_ready",
    "aurora_refresh",
    "node_keys:gpu-a",
    "node_keys:gpu-b",
}


def test_foundation_tasks_cache_the_one_time_resources_and_reprove_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only resources that cannot drift are trusted from ``completed_tasks``.

    A VPC endpoint and a CA are created once and referenced by digest, so
    re-proving them every run would cost an AWS round trip for nothing. Roles,
    the load balancer controller and the notification resources are all editable
    from the console, so those are re-proved -- including one task per GPU
    cluster, since a cluster joined later has a role of its own.
    """

    log: list[tuple[str, bool, bool]] = []
    _stub_foundation_services(monkeypatch, log)

    state = _state(tmp_path)
    results = _foundation_graph(tmp_path, state, log).run(state=state)

    assert set(results) == FOUNDATION_TASKS
    assert {name for name, _read_only, _probe in log} == set(results)

    log.clear()
    completed = {name: {"task": name} for name in results}
    state = _state(tmp_path, completed=completed)
    _foundation_graph(tmp_path, state, log).run(state=state)

    assert {name for name, _read_only, _probe in log} == FOUNDATION_REPROVED, (
        "the wrong set of foundation tasks was re-entered on a completed rerun"
    )
    assert all(read_only for _name, read_only, _probe in log), (
        "a re-proved foundation task used the mutating runner"
    )


# --- the scheduler ----------------------------------------------------------------------


def test_a_task_starts_the_moment_its_dependencies_hold_and_not_before(
    tmp_path: Path,
) -> None:
    """``b`` reads ``a``'s result and ``c`` reads nothing: ``c`` runs while ``a``
    is still running, ``b`` only after ``a`` has been recorded. ``a`` does not
    end until it has seen ``c`` start, so the overlap is observed, not timed."""

    state = _state(tmp_path)
    events: list[tuple[str, str]] = []
    lock = threading.Lock()
    a_started = threading.Event()
    c_started = threading.Event()

    def note(task: str, what: str) -> None:
        with lock:
            events.append((task, what))

    def a() -> dict[str, int]:
        note("a", "start")
        a_started.set()
        assert c_started.wait(timeout=5), "c did not start while a was running"
        note("a", "end")
        return {"a": 1}

    def b() -> dict[str, int]:
        note("b", "start")
        assert state.result("a") == {"a": 1}, "b started before a was recorded"
        note("b", "end")
        return {"b": 1}

    def c() -> dict[str, int]:
        assert a_started.wait(timeout=5), "a never started"
        note("c", "start")
        c_started.set()
        note("c", "end")
        return {}

    results = run_parallel(
        {"a": a, "b": b, "c": c}, state=state, dependencies={"b": ("a",)}
    )

    assert events.index(("a", "end")) < events.index(("b", "start"))
    assert events.index(("c", "start")) < events.index(("a", "end"))
    assert results == {"a": {"a": 1}, "b": {"b": 1}, "c": {}}
    assert state.value["completed_tasks"] == ["a", "b", "c"]


def test_a_failed_dependency_skips_its_dependents_and_reports_them(
    tmp_path: Path,
) -> None:
    """A failed task fails the run only after its independent siblings finish
    (their results are kept for the rerun); a task whose dependency failed is
    skipped and named as such, not counted as a failure of its own."""

    state = _state(tmp_path)
    ran: list[str] = []

    def fail() -> dict:
        raise BootstrapError("aurora failed")

    with pytest.raises(BootstrapError, match="aurora failed") as info:
        run_parallel(
            {
                "aurora": fail,
                "aurora_ready": lambda: ran.append("aurora_ready") or {},
                "aurora_refresh": lambda: ran.append("aurora_refresh") or {},
                "pki": lambda: ran.append("pki") or {"certificate_arn": "arn"},
            },
            state=state,
            dependencies={
                "aurora_ready": ("aurora",),
                "aurora_refresh": ("aurora_ready",),
            },
        )

    assert ran == ["pki"], "a dependent of the failed task still ran"
    note = "\n".join(info.value.__notes__)
    assert "parallel bootstrap task(s) failed: aurora" in note
    assert (
        "skipped because a dependency failed: aurora_ready (after aurora), "
        "aurora_refresh (after aurora_ready)"
    ) in note
    assert state.value["completed_tasks"] == ["pki"]


def test_a_dependency_outside_the_graph_must_already_be_complete(
    tmp_path: Path,
) -> None:
    """The platform graph run alone still names the foundation tasks it reads;
    those must hold from an earlier run, and a name nothing ever completes is a
    wiring error caught before anything starts."""

    state = _state(tmp_path)
    graph = {"aurora_ready": lambda: {"ready": True}}
    dependencies = {"aurora_ready": ("aurora",)}

    with pytest.raises(BootstrapError, match=r"depends on \['aurora'\]"):
        run_parallel(graph, state=state, dependencies=dependencies)

    state.record("aurora", {"cluster_id": "c"})
    state.complete("aurora")
    assert run_parallel(graph, state=state, dependencies=dependencies) == {
        "aurora_ready": {"ready": True}
    }


def test_tasks_that_wait_on_each_other_are_refused(tmp_path: Path) -> None:
    with pytest.raises(BootstrapError, match="cycle"):
        run_parallel(
            {"a": lambda: {}, "b": lambda: {}},
            state=_state(tmp_path),
            dependencies={"a": ("b",), "b": ("a",)},
        )


# --- the whole bootstrap graph ---------------------------------------------------------


def test_monitoring_and_node_keys_start_while_aurora_and_the_nlb_are_still_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner's question: why did Grafana/monitoring and the node keys wait
    for the whole AWS foundation? They no longer do. Here ``aurora`` and
    ``nlb_network`` refuse to finish until both have started, so a scheduler
    that still ran the two former phases back to back would time out."""

    log: list[tuple[str, bool, bool]] = []
    _stub_foundation_services(monkeypatch, log)
    _stub_platform_services(monkeypatch, log)
    started = {
        "monitoring_install": threading.Event(),
        "node_keys:gpu-a": threading.Event(),
        "node_keys:gpu-b": threading.Event(),
    }
    saw_platform_start: dict[str, bool] = {}

    def slow(name: str) -> Callable[..., dict[str, str]]:
        def ensure(_runner: CommandRunner, **_keywords: Any) -> dict[str, str]:
            saw_platform_start[name] = all(
                event.wait(timeout=5) for event in started.values()
            )
            log.append((name, False, False))
            return {"task": name}

        return ensure

    def install_monitoring(_runner: CommandRunner, **_keywords: Any) -> dict:
        started["monitoring_install"].set()
        log.append(("install_monitoring", False, False))
        return {}

    def node_keys(_runner: CommandRunner, **keywords: Any) -> dict:
        started[f"node_keys:{keywords['cluster_id']}"].set()
        log.append((f"node_keys:{keywords['cluster_id']}", False, False))
        return {}

    monkeypatch.setattr(bootstrap_services, "install_monitoring", install_monitoring)
    monkeypatch.setattr(bootstrap_services, "provision_node_action_keys", node_keys)
    state = _state(tmp_path)

    results = bootstrap_tasks.run_bootstrap_tasks(
        state=state,
        foundation=_foundation_graph(
            tmp_path,
            state,
            log,
            ensure_aurora=slow("aurora"),
            ensure_nlb_network=slow("nlb_network"),
        ),
        platform=_platform_graph(tmp_path, state, log),
    )

    assert saw_platform_start == {"aurora": True, "nlb_network": True}, (
        "monitoring_install / node_keys waited for the foundation to finish"
    )
    names = [name for name, _read_only, _probe in log]
    assert names.index("monitoring_resources") < names.index("install_monitoring"), (
        "monitoring_install started before the AMP/SNS resources it reads"
    )
    assert names.index("aurora") < names.index("aurora_ready")
    assert names.index("aurora_ready") < names.index("install_aurora_refresh")
    assert set(results) == FOUNDATION_TASKS | PLATFORM_TASKS


def _bootstrap_graphs(
    tmp_path: Path,
    state: BootstrapState,
    log: list[tuple[str, bool, bool]],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[bootstrap_tasks.TaskGraph, bootstrap_tasks.TaskGraph]:
    _stub_foundation_services(monkeypatch, log)
    _stub_platform_services(monkeypatch, log)
    return (
        _foundation_graph(tmp_path, state, log),
        _platform_graph(tmp_path, state, log),
    )


def test_the_phase_markers_follow_the_two_subsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkpoint readers and the acceptance evidence know two phase names;
    each is still recorded when its subset is complete, even though the tasks
    of the second may have started before the first was marked."""

    log: list[tuple[str, bool, bool]] = []
    state = _state(tmp_path)
    foundation, platform = _bootstrap_graphs(tmp_path, state, log, monkeypatch)
    phases: list[tuple[str, set[str]]] = []
    original = state.phase
    monkeypatch.setattr(
        state,
        "phase",
        lambda value: (
            phases.append((value, set(state.value["completed_tasks"]))),
            original(value),
        ),
    )

    bootstrap_tasks.run_bootstrap_tasks(
        state=state, foundation=foundation, platform=platform
    )

    assert [name for name, _completed in phases] == [
        "aws-infrastructure-ready",
        "platform-prerequisites-ready",
    ]
    assert FOUNDATION_TASKS <= phases[0][1], (
        "aws-infrastructure-ready was marked before every foundation task completed"
    )
    assert set(platform.tasks) <= phases[1][1]


def test_a_completed_graph_reruns_only_what_a_probe_re_proves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rerun parity with the two former phases: the one-time resources are
    trusted, everything else is probed read-only, and the probes of the second
    graph still run after the results they read exist."""

    log: list[tuple[str, bool, bool]] = []
    state = _state(tmp_path)
    foundation, platform = _bootstrap_graphs(tmp_path, state, log, monkeypatch)
    bootstrap_tasks.run_bootstrap_tasks(
        state=state, foundation=foundation, platform=platform
    )
    assert not any(read_only for _name, read_only, _probe in log), (
        "a first run reached a read-only probe"
    )

    log.clear()
    rerun = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")
    foundation, platform = _bootstrap_graphs(tmp_path, rerun, log, monkeypatch)
    results = bootstrap_tasks.run_bootstrap_tasks(
        state=rerun, foundation=foundation, platform=platform
    )

    assert {name for name, _read_only, _probe in log} == FOUNDATION_REPROVED | {
        "install_monitoring",
        "aurora_ready",
        "install_aurora_refresh",
        "node_keys:gpu-a",
        "node_keys:gpu-b",
    }
    assert all(read_only for _name, read_only, _probe in log), (
        "a re-proved task used the mutating runner"
    )
    assert results["aurora"] == {"task": "aurora"}, "a one-time task was re-entered"
