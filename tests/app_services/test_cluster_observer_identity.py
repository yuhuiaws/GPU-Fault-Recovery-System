"""A cluster re-joined as another cluster gets a new observer, with no restart.

The regional managed-recovery observers -- one per cluster, shared by the
workflow path and the background identity refresh -- were cached by cluster id
alone. A cluster removed and re-joined under the same id as a different
HyperPod (or EKS) cluster kept the observer built for the old one, so both
paths kept reading the old cluster's nodes until every CPU Pod restarted. The
map now records the provider identity each observer was built from (provider,
region, HyperPod cluster name, EKS orchestrator ARN) and checks it against the
registry on every lookup.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from gpu_fault.app.context import ApplicationContext
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalClusterRegistration,
    RegionalRegistryRevision,
    cluster_token_sha256,
)
from tests.app_services.test_identity_refresh_follows_registry import (
    INTERVAL,
    Clock,
    FakeHyperPod,
    bind_runtime,
    cluster_entry,
    leader_runner,
    publish,
    regional_context,
    registered,
    without,
)
from tests.regional._regional_support import TOKEN_A, TOKEN_B


def observers(context: ApplicationContext) -> Any:
    observer = context.regional_managed_observer
    assert observer is not None, "the regional managed observer is configured"
    return observer.observers


def lookup(context: ApplicationContext, cluster_id: str) -> Any:
    """The workflow path's lookup, exactly as the regional router makes it."""

    return observers(context).get(cluster_id)


def observe(context: ApplicationContext, cluster_id: str) -> Any:
    """One managed-recovery observation routed by cluster, as the workflow
    executor makes it. The node is unknown everywhere, so the step fails
    right after the observer refreshed its cluster's identities."""

    observer = context.regional_managed_observer
    assert observer is not None, "the regional managed observer is configured"
    return observer.observe(
        SimpleNamespace(
            incident=SimpleNamespace(cluster_id=cluster_id),
            step=SimpleNamespace(
                node_ids=["unit-unknown-node"], operation=WorkflowOperation.RESTART_NODE
            ),
            workflow=SimpleNamespace(step_executions=[]),
            step_index=0,
        )
    )


def current(
    context: ApplicationContext, cluster_id: str
) -> RegionalClusterRegistration:
    return next(item for item in registered(context) if item.cluster_id == cluster_id)


def another_cluster(
    registration: RegionalClusterRegistration, hyperpod_cluster_name: str
) -> RegionalClusterRegistration:
    """The same cluster id re-joined as another HyperPod cluster: its own
    HyperPod name and EKS orchestrator, and the new token a join mints."""

    return registration.model_copy(
        update={
            "hyperpod_cluster_name": hyperpod_cluster_name,
            "eks_cluster_arn": (
                f"arn:aws:eks:us-west-2:123456789012:cluster/{hyperpod_cluster_name}"
            ),
            "token_sha256": cluster_token_sha256(TOKEN_B),
        }
    )


def rejoin(
    context: ApplicationContext, registration: RegionalClusterRegistration
) -> None:
    """remove-cluster's purge, then join-cluster under the same cluster id."""

    cluster_id = registration.cluster_id
    publish(context, without(context, cluster_id), f"remove {cluster_id} purged")
    publish(context, [*registered(context), registration], f"join {cluster_id}")


def test_a_cluster_rejoined_as_another_hyperpod_cluster_gets_a_new_observer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hyperpod = FakeHyperPod()
    context = regional_context(
        monkeypatch, tmp_path, hyperpod, [cluster_entry("cluster-a", TOKEN_A)]
    )
    bind_runtime(context)
    clock = Clock()
    runner = leader_runner(monkeypatch, context, clock)
    old = lookup(context, "cluster-a")
    observe(context, "cluster-a")
    runner.run_all_due(clock.now)
    assert hyperpod.listed == ["hp-cluster-a", "hp-cluster-a"]

    rejoin(context, another_cluster(current(context, "cluster-a"), "hp-cluster-a-2"))
    outcome = observe(context, "cluster-a")
    runner.run_all_due(clock.advance(INTERVAL))

    new = lookup(context, "cluster-a")
    assert new is not None and new is not old, "the old cluster's observer was kept"
    assert hyperpod.built == ["hp-cluster-a", "hp-cluster-a-2"], (
        "one new observer, built once, must serve both paths"
    )
    assert hyperpod.listed[2:] == ["hp-cluster-a-2", "hp-cluster-a-2"], (
        "the workflow path and the background refresh must both read the "
        "re-joined cluster, never the old one"
    )
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "unit-unknown-node" in (outcome.error or ""), outcome.error
    assert all(item is not old for item in observers(context).values()), (
        "the old observer must be dropped from the map"
    )
    publish(context, without(context, "cluster-a"), "remove cluster-a again")
    assert lookup(context, "cluster-a") is new, "a dropped observer came back"
    context.store.close()


def test_the_same_cluster_rejoined_keeps_its_observer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hyperpod = FakeHyperPod()
    context = regional_context(
        monkeypatch, tmp_path, hyperpod, [cluster_entry("cluster-a", TOKEN_A)]
    )
    bind_runtime(context)
    clock = Clock()
    runner = leader_runner(monkeypatch, context, clock)
    old = lookup(context, "cluster-a")
    original = current(context, "cluster-a")

    publish(context, without(context, "cluster-a"), "remove cluster-a purged")
    assert lookup(context, "cluster-a") is old, (
        "absent from the registry, the map answers with what it holds"
    )
    pending = original.model_copy(
        update={
            "token_sha256": cluster_token_sha256(TOKEN_B),
            "lifecycle_state": RegionalClusterLifecycle.PENDING,
        }
    )
    publish(context, [*registered(context), pending], "join cluster-a pending")
    assert lookup(context, "cluster-a") is old
    active = pending.model_copy(
        update={"lifecycle_state": RegionalClusterLifecycle.ACTIVE}
    )
    publish(context, [*without(context, "cluster-a"), active], "join cluster-a active")
    observe(context, "cluster-a")
    runner.run_all_due(clock.now)

    assert lookup(context, "cluster-a") is old
    assert hyperpod.built == ["hp-cluster-a"], (
        "re-joining the same cluster must not build another observer"
    )
    assert hyperpod.listed == ["hp-cluster-a", "hp-cluster-a"]
    context.store.close()


class GatedHyperPod(FakeHyperPod):
    """Holds the build of one HyperPod cluster's adapter until released."""

    def __init__(self, gated: str) -> None:
        super().__init__()
        self.gated = gated
        self.entered = threading.Event()
        self.release = threading.Event()

    def adapter(self, config: Any, client: Any = None, *, store: Any = None) -> Any:
        if config.cluster_name == self.gated:
            self.entered.set()
            assert self.release.wait(10), "the test never released the build"
        return super().adapter(config, client, store=store)


def test_concurrent_lookups_during_the_switch_build_exactly_one_observer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hyperpod = GatedHyperPod("hp-cluster-a-2")
    context = regional_context(
        monkeypatch, tmp_path, hyperpod, [cluster_entry("cluster-a", TOKEN_A)]
    )
    bind_runtime(context)
    old = lookup(context, "cluster-a")
    rejoin(context, another_cluster(current(context, "cluster-a"), "hp-cluster-a-2"))
    source = context.background_identity_registries()
    assert callable(source), "a bound regional process follows its registry"
    (refresh_entry,) = source()

    workers = 8
    start = threading.Barrier(workers)
    looked_up: list[Any] = []
    refreshed: list[Any] = []
    failures: list[BaseException] = []
    guard = threading.Lock()

    def run(step: Callable[[], Any], sink: list[Any]) -> None:
        try:
            start.wait(5)
            value = step()
            with guard:
                sink.append(value)
        except BaseException as exc:  # reported below, not lost in the thread
            with guard:
                failures.append(exc)

    paths: list[tuple[Callable[[], Any], list[Any]]] = [
        (lambda: lookup(context, "cluster-a"), looked_up),
        (refresh_entry.refresh, refreshed),
    ]
    threads = [
        threading.Thread(target=run, args=paths[index % 2], daemon=True)
        for index in range(workers)
    ]
    for thread in threads:
        thread.start()
    assert hyperpod.entered.wait(5), "no lookup started the new observer's build"
    hyperpod.release.set()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive(), "a lookup never returned"

    assert failures == []
    new = lookup(context, "cluster-a")
    assert hyperpod.built == ["hp-cluster-a", "hp-cluster-a-2"], (
        "the switch must build exactly one observer"
    )
    assert len(looked_up) == len(refreshed) == workers // 2
    assert all(value is new for value in looked_up), "lookups got different observers"
    assert new is not old
    assert hyperpod.listed == ["hp-cluster-a-2"] * (workers // 2), (
        "every refresh during the switch must read the re-joined cluster"
    )
    context.store.close()


def test_a_cluster_rejoined_as_plain_eks_loses_its_hyperpod_observer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hyperpod = FakeHyperPod()
    context = regional_context(
        monkeypatch, tmp_path, hyperpod, [cluster_entry("cluster-a", TOKEN_A)]
    )
    bind_runtime(context)
    clock = Clock()
    runner = leader_runner(monkeypatch, context, clock)
    assert lookup(context, "cluster-a") is not None
    rejoin(
        context,
        current(context, "cluster-a").model_copy(
            update={
                "provider": "eks-managed",
                "hyperpod_cluster_name": "",
                "eks_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/eks-a",
                "token_sha256": cluster_token_sha256(TOKEN_B),
            }
        ),
    )

    outcome = observe(context, "cluster-a")
    runner.run_all_due(clock.now)

    assert lookup(context, "cluster-a") is None
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "no managed recovery observer registered for cluster cluster-a" in (
        outcome.error or ""
    ), outcome.error
    assert hyperpod.listed == [] and hyperpod.built == ["hp-cluster-a"]
    publish(context, without(context, "cluster-a"), "remove cluster-a again")
    assert lookup(context, "cluster-a") is None, "a dropped observer came back"
    context.store.close()


def test_a_stale_observer_stays_dropped_while_its_successor_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hyperpod = FakeHyperPod()
    context = regional_context(
        monkeypatch, tmp_path, hyperpod, [cluster_entry("cluster-a", TOKEN_A)]
    )
    bind_runtime(context)
    old = lookup(context, "cluster-a")
    moved = another_cluster(current(context, "cluster-a"), "hp-cluster-a-2")
    hyperpod.unbuildable.add("hp-cluster-a-2")
    rejoin(context, moved)

    with pytest.raises(RuntimeError, match="synthetic adapter build failure"):
        lookup(context, "cluster-a")
    publish(context, without(context, "cluster-a"), "remove cluster-a again")
    assert lookup(context, "cluster-a") is None, (
        "a failed rebuild must not leave the old observer to be served"
    )

    hyperpod.unbuildable.clear()
    publish(context, [*registered(context), moved], "join cluster-a again")
    new = lookup(context, "cluster-a")
    assert new is not None and new is not old
    assert hyperpod.built == ["hp-cluster-a", "hp-cluster-a-2"]
    context.store.close()


def test_a_start_up_observer_is_checked_against_the_registry_it_binds_to(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Start-up builds observers from the store before the runtime is bound;
    a re-join published in between (an operator racing a Pod start) must not
    leave the old cluster's observer behind."""

    hyperpod = FakeHyperPod()
    context = regional_context(
        monkeypatch, tmp_path, hyperpod, [cluster_entry("cluster-a", TOKEN_A)]
    )
    started = lookup(context, "cluster-a")
    (original,) = context.store.list_regional_clusters()
    context.store.publish_regional_registry_revision(
        RegionalRegistryRevision.build(
            generation=1,
            registrations=[another_cluster(original, "hp-cluster-a-2")],
            previous_generation=None,
            required_member_ids=[],
            reason="join cluster-a raced the start",
            created_at=datetime.now(timezone.utc),
        ),
        expected_generation=0,
    )
    bind_runtime(context)

    bound = lookup(context, "cluster-a")
    assert bound is not None and bound is not started
    assert hyperpod.built == ["hp-cluster-a", "hp-cluster-a-2"]
    context.store.close()
