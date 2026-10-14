"""The background identity refresh follows the regional registry, no CPU restart.

The lifespan handed both refresh workers -- the periodic runner's ``identity``
job and the identity worker of a process without a processor -- a list it
copied at start-up. A HyperPod cluster joined online (``join-cluster`` publishes
a registry revision and rolls no CPU Pod; on a site with two HyperPod clusters
the second always joins that way) therefore had its node identities refreshed
only on demand, inside a managed-recovery observation, until every CPU Pod
restarted, and a cluster removed online stayed in the background refresh. The
workers now read the registry runtime's current snapshot on every pass.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, Callable, NoReturn

import pytest
from fastapi.testclient import TestClient

from gpu_fault.app import context as context_module
from gpu_fault.app import create_app, periodic_services
from gpu_fault.app import lifespan_workers as workers
from gpu_fault.app.context import ApplicationContext
from gpu_fault.app.periodic_services import PeriodicServiceConfig, PeriodicServiceRunner
from gpu_fault.hyperpod import HyperPodNode
from gpu_fault.regional import RegionalClusterRegistration, RegionalRegistryRevision
from gpu_fault.regional_registry_runtime import RegionalRegistryRuntime
from tests.app_services._cov95_runtime_workers import Stop, ThreadRecorder
from tests.regional._regional_support import (
    TOKEN_A,
    TOKEN_B,
    regional_environment,
    registration,
)

INTERVAL = 20.0


class FakeHyperPod:
    """Stands in for ``HyperPodLifecycleAdapter``: one node per cluster, and a
    log of the clusters an adapter was built for and listed."""

    def __init__(self) -> None:
        self.built: list[str] = []
        self.listed: list[str] = []
        self.unbuildable: set[str] = set()

    def adapter(self, config: Any, client: Any = None, *, store: Any = None) -> Any:
        del client, store
        name = config.cluster_name
        if name in self.unbuildable:
            raise RuntimeError(f"synthetic adapter build failure for {name}")
        self.built.append(name)
        listed = self.listed

        class Adapter:
            def __init__(self) -> None:
                self.config = config

            def list_nodes(self, *, enrich: bool = False) -> list[HyperPodNode]:
                del enrich
                listed.append(name)
                return [
                    HyperPodNode(
                        node_logical_id=f"{name}-node-0",
                        instance_id="i-" + sha256(name.encode()).hexdigest()[:17],
                        status="Running",
                    )
                ]

        return Adapter()

    def count(self, name: str) -> int:
        return self.listed.count(name)


def cluster_entry(cluster_id: str, token: str) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "region": "us-west-2",
        "hyperpod_cluster_name": f"hp-{cluster_id}",
        "eks_cluster_arn": f"arn:aws:eks:us-west-2:123456789012:cluster/{cluster_id}",
        "token": token,
        "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
    }


def regional_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hyperpod: FakeHyperPod,
    entries: list[dict[str, Any]],
) -> ApplicationContext:
    """An active regional control plane with the managed observer, as the
    control-worker runs it, over a SQLite store seeded from ``entries``."""

    regional_environment(monkeypatch, tmp_path)
    for name, value in {
        "GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER": "true",
        "GPU_FAULT_ENABLE_AGENT_REGISTRY": "true",
        "GPU_FAULT_AGENT_REGISTRATION_SECRET": "m" * 32,
        "GPU_FAULT_NODE_ACTION_SECRET": "m" * 32,
        "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "b" * 64,
        "GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION": "2",
        "GPU_FAULT_REGIONAL_CLUSTERS_JSON": json.dumps(entries),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(context_module, "HyperPodLifecycleAdapter", hyperpod.adapter)
    return ApplicationContext.from_environment()


def bind_runtime(context: ApplicationContext) -> RegionalRegistryRuntime:
    """What ``create_app`` does before the lifespan starts."""

    runtime = RegionalRegistryRuntime.bootstrap(
        context.store,
        member_id="unit-control-worker",
        service_role="worker",
        release_id="unit-release",
        poll_seconds=1,
        stale_seconds=5,
    )
    context.bind_regional_registry_runtime(runtime)
    context.regional_registry_runtime = runtime
    return runtime


def publish(
    context: ApplicationContext,
    registrations: list[RegionalClusterRegistration],
    reason: str,
) -> None:
    """One online registry change (join/remove), applied by this process's
    registry runtime the way its poller would within a second."""

    runtime = context.regional_registry_runtime
    assert runtime is not None, "the registry runtime is bound before publishing"
    current = runtime.snapshot()
    context.store.publish_regional_registry_revision(
        RegionalRegistryRevision.build(
            generation=current.generation + 1,
            registrations=registrations,
            previous_generation=current.generation,
            required_member_ids=[],
            reason=reason,
            created_at=datetime.now(timezone.utc),
        ),
        expected_generation=current.generation,
    )
    runtime.refresh_once(raise_on_failure=True)


def registered(context: ApplicationContext) -> list[RegionalClusterRegistration]:
    runtime = context.regional_registry_runtime
    assert runtime is not None, "the registry runtime is bound"
    return list(runtime.snapshot().registrations.values())


def without(
    context: ApplicationContext, cluster_id: str
) -> list[RegionalClusterRegistration]:
    return [item for item in registered(context) if item.cluster_id != cluster_id]


def eks_registration(cluster_id: str) -> RegionalClusterRegistration:
    return registration(cluster_id, TOKEN_B).model_copy(
        update={"provider": "eks-managed", "hyperpod_cluster_name": ""}
    )


def synthetic_registration(cluster_id: str) -> RegionalClusterRegistration:
    """A perf/acceptance placeholder: a HyperPod name no SageMaker cluster has."""

    return registration(cluster_id, TOKEN_B).model_copy(
        update={
            "hyperpod_cluster_name": cluster_id,
            "synthetic": True,
            "synthetic_run_id": "unit-perf-run",
            "synthetic_expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
        }
    )


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


def leader_runner(
    monkeypatch: pytest.MonkeyPatch, context: ApplicationContext, clock: Clock
) -> PeriodicServiceRunner:
    """The control-worker's periodic runner, holding every task lease."""

    monkeypatch.setattr(
        periodic_services,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now, time=lambda: 1_700_000_000.0 + clock.now
        ),
    )
    monkeypatch.setattr(
        periodic_services, "training_health_monitor_enabled", lambda: False
    )
    config = PeriodicServiceConfig.from_environment()
    assert config.identity_interval == INTERVAL, "the default refresh cadence"
    return PeriodicServiceRunner(
        context=context,
        processor=SimpleNamespace(
            is_healthy=lambda: True,
            is_leader=lambda: True,
            active_consumers=False,
            owner_id="unit-periodic-owner",
        ),
        stop=Event(),
        identity_registries=context.background_identity_registries(),
        ingest_node_health_findings=lambda *args: None,
        notify_silent_collectors=lambda *args, **kwargs: None,
        config=config,
    )


def identity_stamp(runner: PeriodicServiceRunner) -> float | None:
    return runner.metrics_snapshot()["job_last_run_timestamp_seconds"].get("identity")


def test_start_up_clusters_keep_their_start_up_registries_and_cadence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hyperpod = FakeHyperPod()
    context = regional_context(
        monkeypatch,
        tmp_path,
        hyperpod,
        [cluster_entry("cluster-a", TOKEN_A), cluster_entry("cluster-b", TOKEN_B)],
    )
    bind_runtime(context)
    assert hyperpod.built == ["hp-cluster-a", "hp-cluster-b"], (
        "start-up builds one observer per registered HyperPod cluster"
    )
    clock = Clock()
    runner = leader_runner(monkeypatch, context, clock)

    runner.run_all_due(clock.now)
    assert hyperpod.listed == ["hp-cluster-a", "hp-cluster-b"]
    assert hyperpod.built == ["hp-cluster-a", "hp-cluster-b"], (
        "the first pass must refresh the registries start-up built, not new ones"
    )
    assert [
        identity.node_logical_id
        for identity in context.store.list_hyperpod_node_identities("hp-cluster-b")
    ] == ["hp-cluster-b-node-0"]

    runner.run_all_due(clock.advance(INTERVAL / 2))
    assert len(hyperpod.listed) == 2, "refreshed again before its interval"
    runner.run_all_due(clock.advance(INTERVAL / 2))
    assert hyperpod.listed[2:] == ["hp-cluster-a", "hp-cluster-b"]
    context.store.close()


def test_a_cluster_joined_online_is_refreshed_within_one_interval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hyperpod = FakeHyperPod()
    context = regional_context(
        monkeypatch, tmp_path, hyperpod, [cluster_entry("cluster-a", TOKEN_A)]
    )
    bind_runtime(context)
    clock = Clock()
    runner = leader_runner(monkeypatch, context, clock)
    runner.run_all_due(clock.now)
    assert hyperpod.listed == ["hp-cluster-a"]

    publish(
        context,
        [*registered(context), registration("cluster-b", TOKEN_B)],
        "join cluster-b",
    )
    runner.run_all_due(clock.advance(INTERVAL))

    assert hyperpod.listed == ["hp-cluster-a", "hp-cluster-a", "hp-cluster-b"], (
        "the joined cluster must enter the background refresh without a restart"
    )
    assert [
        identity.node_logical_id
        for identity in context.store.list_hyperpod_node_identities("hp-cluster-b")
    ] == ["hp-cluster-b-node-0"]
    observer = context.regional_managed_observer
    assert observer is not None, "the regional managed observer is configured"
    joined = observer.observers.get("cluster-b")
    assert joined is not None, "the workflow path resolves the joined cluster"
    assert hyperpod.built == ["hp-cluster-a", "hp-cluster-b"], (
        "the refresh and the workflow path must share one observer per cluster"
    )
    assert "identity" not in runner.metrics_snapshot()["periodic_job_errors_total"]
    context.store.close()


def test_a_cluster_removed_online_leaves_the_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hyperpod = FakeHyperPod()
    context = regional_context(
        monkeypatch,
        tmp_path,
        hyperpod,
        [cluster_entry("cluster-a", TOKEN_A), cluster_entry("cluster-b", TOKEN_B)],
    )
    bind_runtime(context)
    clock = Clock()
    runner = leader_runner(monkeypatch, context, clock)
    runner.run_all_due(clock.now)
    assert hyperpod.listed == ["hp-cluster-a", "hp-cluster-b"]

    publish(context, without(context, "cluster-a"), "remove cluster-a purged")
    runner.run_all_due(clock.advance(INTERVAL))
    assert hyperpod.listed[2:] == ["hp-cluster-b"], (
        "a cluster removed online must stop being refreshed"
    )

    publish(context, without(context, "cluster-b"), "remove cluster-b purged")
    stamp = identity_stamp(runner)
    runner.run_all_due(clock.advance(INTERVAL))
    assert hyperpod.listed[3:] == []
    assert identity_stamp(runner) == stamp, (
        "with no HyperPod cluster left the identity job must not run at all"
    )
    context.store.close()


def test_placeholder_and_plain_eks_registrations_never_reach_a_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A synthetic registration (perf ``perf-cap-NNN``, AUTH-007/008's
    ``auth-logical-``) names a HyperPod cluster that does not exist; refreshing
    it only produced ResourceNotFound every interval. A plain EKS cluster has
    no HyperPod identity at all."""

    hyperpod = FakeHyperPod()
    context = regional_context(monkeypatch, tmp_path, hyperpod, [])
    bind_runtime(context)
    clock = Clock()
    runner = leader_runner(monkeypatch, context, clock)
    publish(
        context,
        [synthetic_registration("perf-cap-000"), eks_registration("cluster-e")],
        "capacity register",
    )

    runner.run_all_due(clock.now)
    runner.run_all_due(clock.advance(INTERVAL))

    assert hyperpod.built == [] and hyperpod.listed == []
    assert identity_stamp(runner) is None, (
        "nothing to refresh must not run the identity job"
    )
    context.store.close()


def test_a_cluster_whose_observer_cannot_be_built_fails_alone_and_is_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hyperpod = FakeHyperPod()
    context = regional_context(
        monkeypatch, tmp_path, hyperpod, [cluster_entry("cluster-a", TOKEN_A)]
    )
    bind_runtime(context)
    clock = Clock()
    runner = leader_runner(monkeypatch, context, clock)
    hyperpod.unbuildable.add("hp-cluster-b")
    publish(
        context,
        [registration("cluster-b", TOKEN_B), *registered(context)],
        "join cluster-b",
    )

    runner.run_all_due(clock.now)
    assert hyperpod.listed == ["hp-cluster-a"], (
        "one cluster's build failure must not starve another cluster"
    )
    errors = runner.metrics_snapshot()["periodic_job_errors_total"]
    assert errors.get("identity") == 1, "the build failure is the job's failure"

    hyperpod.unbuildable.clear()
    runner.run_all_due(clock.advance(INTERVAL))
    assert hyperpod.listed[1:] == ["hp-cluster-a", "hp-cluster-b"]
    assert hyperpod.built == ["hp-cluster-a", "hp-cluster-b"]
    context.store.close()


def test_an_unavailable_registry_snapshot_refreshes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail closed like the endpoint allow-list: no snapshot, no membership."""

    hyperpod = FakeHyperPod()
    context = regional_context(
        monkeypatch, tmp_path, hyperpod, [cluster_entry("cluster-a", TOKEN_A)]
    )
    runtime = bind_runtime(context)

    def unavailable() -> NoReturn:
        raise RuntimeError("regional registry snapshot is unavailable")

    monkeypatch.setattr(runtime, "snapshot", unavailable)
    clock = Clock()
    runner = leader_runner(monkeypatch, context, clock)
    runner.run_all_due(clock.now)

    assert hyperpod.listed == []
    assert identity_stamp(runner) is None
    context.store.close()


@pytest.mark.parametrize("hyperpod_registry", [False, True])
def test_a_process_without_a_registry_keeps_its_start_up_registries(
    hyperpod_registry: bool,
) -> None:
    context = ApplicationContext()
    registry = SimpleNamespace(refresh=lambda: [])
    if hyperpod_registry:
        context.hyperpod_identity_registry = registry

    assert context.background_identity_registries() == (
        [registry] if hyperpod_registry else []
    )


class _RemovingStop(Stop):
    """Applies one registry change at the worker's first wait."""

    def __init__(self, change: Callable[[], None]) -> None:
        super().__init__(cycles=1)
        self.change: Callable[[], None] | None = change

    def wait(self, seconds: float) -> bool:
        if self.change is not None:
            self.change()
            self.change = None
        return super().wait(seconds)


def test_the_identity_worker_without_a_processor_follows_the_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Start-up saw no HyperPod cluster (a plain EKS site); one joins, then
    the other leaves. The worker must exist and read every pass."""

    recorder = ThreadRecorder()
    monkeypatch.setattr(workers, "Thread", recorder)
    monkeypatch.setattr(workers, "training_health_monitor_enabled", lambda: False)
    hyperpod = FakeHyperPod()
    entry = cluster_entry("cluster-e", TOKEN_A) | {
        "provider": "eks-managed",
        "hyperpod_cluster_name": "",
    }
    context = regional_context(monkeypatch, tmp_path, hyperpod, [entry])
    bind_runtime(context)

    _xid, _training, _spare, identity = workers.start_nonprocessor_workers(
        context=context,
        stop=_RemovingStop(
            lambda: publish(context, without(context, "cluster-a"), "remove cluster-a")
        ),
        identity_registries=context.background_identity_registries(),
        ingest_node_health_findings=lambda *args: None,
    )
    assert identity is not None, (
        "a registry-backed refresh must start even when start-up saw no cluster"
    )
    publish(
        context,
        [
            *registered(context),
            registration("cluster-a", TOKEN_A),
            registration("cluster-b", TOKEN_B),
        ],
        "join cluster-a and cluster-b",
    )
    identity.run()

    assert hyperpod.listed == ["hp-cluster-a", "hp-cluster-b", "hp-cluster-b"]
    context.store.close()


def wait_for(predicate: Callable[[], bool], what: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


def test_the_worker_lifespan_refreshes_an_online_join_without_a_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real lifespan, as the worker role starts it: the regression was the
    start-up copy here, so this proves the wiring, not just the workers."""

    hyperpod = FakeHyperPod()
    context = regional_context(
        monkeypatch, tmp_path, hyperpod, [cluster_entry("cluster-a", TOKEN_A)]
    )
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "worker")
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_IDENTITY_REFRESH_SECONDS", "0.05")
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_MODE", raising=False)
    app = create_app(context)

    with TestClient(app):
        wait_for(lambda: hyperpod.count("hp-cluster-a") >= 1, "the start-up refresh")
        publish(
            context,
            [*registered(context), registration("cluster-b", TOKEN_B)],
            "join cluster-b",
        )
        wait_for(lambda: hyperpod.count("hp-cluster-b") >= 1, "the joined cluster")

        publish(context, without(context, "cluster-a"), "remove cluster-a purged")
        passes = hyperpod.count("hp-cluster-b")
        # One pass may have read the registry before the removal landed.
        wait_for(lambda: hyperpod.count("hp-cluster-b") >= passes + 2, "two passes")
        removed_at = hyperpod.count("hp-cluster-a")
        wait_for(lambda: hyperpod.count("hp-cluster-b") >= passes + 4, "two more")
        assert hyperpod.count("hp-cluster-a") == removed_at, (
            "a cluster removed online must stop being refreshed"
        )
    context.store.close()
