"""The bootstrap task graphs and what each one re-proves on a rerun.

These builders decide two things no other module can: which tasks exist, and
which of them are re-proved by a read-only probe every run instead of being
trusted from ``completed_tasks``. Both are exercised through the real
``run_parallel`` so a rerun here means what a rerun means in bootstrap.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import (
    bootstrap_load_balancer,
    bootstrap_services,
    bootstrap_tasks,
    notification_bootstrap,
)
from gpu_fault.admin.bootstrap_common import (
    BootstrapMutationRequired,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.notifications import NotificationRouting
from tests.admin._bootstrap_support import _cluster


class Recorder:
    """Stands in for one ``ensure`` function and remembers how it was called."""

    def __init__(self, name: str, log: list[tuple[str, bool, bool]]) -> None:
        self.name = name
        self.log = log

    def __call__(self, runner: CommandRunner, *_args: Any, **keywords: Any) -> dict:
        read_only = isinstance(runner, ReadOnlyProbeRunner)
        self.log.append((self.name, read_only, bool(keywords.get("probe_only"))))
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


def _run_prerequisites(tmp_path: Path, state: BootstrapState) -> None:
    bootstrap_tasks.run_platform_prerequisite_tasks(
        runner=CommandRunner(),
        state=state,
        repository_root=tmp_path,
        cpu=_cluster(),
        gpu_clusters=[_gpu("gpu-a"), _gpu("gpu-b")],
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        monitoring={},
        adot_image="adot:1",
        alert_email=None,
        release_manifest=tmp_path / "manifest.json",
        runtime_image="runtime:1",
        aurora={},
        fleet_master_file=tmp_path / "master",
    )


def test_every_platform_prerequisite_task_is_reproved_per_gpu_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node keys are provisioned per GPU cluster, and none of the three is cached.

    Node membership changes without bootstrap running, and the manifests these
    tasks apply can be edited in the cluster, so a completed run re-proves all of
    them read-only. A per-cluster task name is what makes joining a second GPU
    cluster provision its own keys rather than reuse the first cluster's.
    """

    log: list[tuple[str, bool, bool]] = []
    for name in ("install_monitoring", "install_aurora_refresh"):
        monkeypatch.setattr(bootstrap_services, name, Recorder(name, log))
    monkeypatch.setattr(
        bootstrap_services,
        "provision_node_action_keys",
        lambda runner, **keywords: Recorder(f"node_keys:{keywords['cluster_id']}", log)(
            runner, **keywords
        ),
    )

    _run_prerequisites(tmp_path, _state(tmp_path))

    assert {name for name, _read_only, _probe in log} == {
        "install_monitoring",
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
    _run_prerequisites(
        tmp_path,
        _state(
            tmp_path,
            completed={
                "monitoring_install": {},
                "aurora_refresh": {},
                "node_keys:gpu-a": {},
                "node_keys:gpu-b": {},
            },
        ),
    )

    assert len(log) == 4
    assert all(read_only for _name, read_only, _probe in log), (
        "a completed prerequisite task was re-run with the mutating runner"
    )
    assert all(probe for _name, _read_only, probe in log), (
        "a probe run did not pass probe_only"
    )


def _run_foundation(
    tmp_path: Path, state: BootstrapState, log: list[tuple[str, bool, bool]]
) -> dict[str, Any]:
    return bootstrap_tasks.run_foundation_tasks(
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
        ensure_nlb_network=Recorder("nlb_network", log),
        ensure_pki=Recorder("pki", log),
        ensure_aurora=Recorder("aurora", log),
    )


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

    results = _run_foundation(tmp_path, _state(tmp_path), log)

    assert set(results) == {
        "nlb_network",
        "pki",
        "aurora",
        "load_balancer_controller",
        "control_plane_role",
        "email_notifications",
        "monitoring_resources",
        "executor_role:gpu-a",
    }

    assert {name for name, _read_only, _probe in log} == set(results)

    log.clear()
    completed = {name: {"task": name} for name in results}
    _run_foundation(tmp_path, _state(tmp_path, completed=completed), log)

    assert {name for name, _read_only, _probe in log} == {
        "load_balancer_controller",
        "control_plane_role",
        "email_notifications",
        "monitoring_resources",
        "executor_role:gpu-a",
    }, "the wrong set of foundation tasks was re-entered on a completed rerun"
    assert all(read_only for _name, read_only, _probe in log), (
        "a re-proved foundation task used the mutating runner"
    )
