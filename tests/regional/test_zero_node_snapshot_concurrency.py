"""A release step never starts a report snapshot, so a parked cluster's
zero-node proof cannot leak one into another cluster's thread.

Live (R24 attempt 4): the deploy's candidate preflight
(``regional_release_mutation_preflight.preflight_upgrade_mutations``) runs
every GPU cluster's node preflight in its own thread. The parked cluster's
zero-node proof (``regional_parked_cluster.without_nodes``) opened a read
snapshot -- ``regional_release_state.read_snapshot``, which keeps its cache on
the release object itself, shared by every thread -- for the length of its
provider reads. The HyperPod cluster's thread, which has nodes, made its fresh
node safety read in that window and refused: "fresh node safety reads cannot
use a report snapshot". The deploy failed and rolled back before any roll.

These tests run the real ``read_snapshot`` and the real candidate preflight
with the two clusters in parallel, the parked cluster's provider read held
open until the other cluster's fresh read has happened -- the live
interleaving, made deterministic.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from gpu_fault_release import regional_parked_cluster as PARKED
from gpu_fault_release import regional_release_mutation_preflight as MUTATION
from gpu_fault_release import regional_release_node_preflight as NODE_PREFLIGHT
from gpu_fault_release import regional_release_node_runtime_rollout as NODE_RUNTIME
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from tests.regional._parked_cluster_support import EKS, GROUP, HYPERPOD, ParkedSite

WAIT = 5.0
AGENT_WAVES = ReleaseExecutionPlan(
    nodes=(ReleaseComponent.RECONCILER, ReleaseComponent.AGENT)
)


class SnapshotSite(ParkedSite):
    """The parked site with the engine's own report snapshot and node
    preflight, the parked cluster's provider read held open (``hold``)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config.upgrade_max_unavailable = 1
        self.config.rollback_max_unavailable = 1
        self.bundle_cm = "candidate-bundle"
        for target in self.config.clusters:
            self.documents[(target.cluster_id, "deployments", "")] = {"items": []}
        (hyperpod_node,) = self.documents[(HYPERPOD, "nodes", self._hp_key())]["items"]
        hyperpod_node["metadata"]["uid"] = "node-a-uid"
        self.inside_provider_read = threading.Event()
        self.fresh_read_done = threading.Event()
        self.cache_seen_by_provider_read: list[bool] = []
        self.preflights: list[tuple[str, dict[str, str]]] = []
        provider_run = self.provider.run

        def run(arguments: list[str], **run_kwargs: Any) -> str:
            if str(arguments[0]).endswith("deploy-node-installer-reconciler.sh"):
                environment = dict(run_kwargs.get("env") or {})
                cluster = (
                    EKS if environment.get(PARKED.ZERO_NODE_CLUSTER_ENV) else HYPERPOD
                )
                self.preflights.append((cluster, environment))
                count = 0 if cluster == EKS else 1
                return json.dumps({"node_count": count, "status": "PASSED"})
            if arguments[:3] == ["aws", "eks", "describe-nodegroup"]:
                # The parked cluster's provider read: what the other thread
                # sees of the release while it runs.
                self.cache_seen_by_provider_read.append(
                    getattr(self, "_json_read_cache", None) is not None
                )
                self.inside_provider_read.set()
                self.fresh_read_done.wait(WAIT)
            return provider_run(arguments, **run_kwargs)

        self.provider.run = run  # type: ignore[method-assign]

    def _hp_key(self) -> str:
        from gpu_fault.cluster_scope import node_selector

        return "-l " + node_selector(self.config.clusters[0])

    def _read_snapshot(self) -> Any:
        return STATE.read_snapshot(self)

    def _preflight_node_runtime(self, target: Any, **kwargs: Any) -> None:
        NODE_RUNTIME.preflight_node_runtime(self, target, **kwargs)

    def _get_json(self, arguments: list[str]) -> dict[str, Any]:
        if "deployments" in arguments and arguments[2] == HYPERPOD:
            # The HyperPod thread starts its preflight once the parked
            # cluster's thread is inside its provider read.
            assert self.inside_provider_read.wait(WAIT), "the threads never met"
        return super()._get_json(arguments)


@pytest.fixture
def node_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """The script's environment builders stubbed; everything else is real."""

    monkeypatch.setattr(
        NODE_PREFLIGHT, "build_reconciler_environment", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        NODE_PREFLIGHT, "reconciler_template_identity", lambda *_a: "t" * 64
    )
    monkeypatch.setattr(NODE_PREFLIGHT, "pin_installer_version", lambda *_a: None)


def _release_fresh_read(monkeypatch: pytest.MonkeyPatch, site: SnapshotSite) -> None:
    """Free the parked cluster's provider read once the HyperPod thread's first
    fresh node read is over, whatever its outcome."""

    isolated = NODE_RUNTIME.product_isolated_nodes

    def product_isolated_nodes(*args: Any, **kwargs: Any) -> Any:
        try:
            return isolated(*args, **kwargs)
        finally:
            site.fresh_read_done.set()

    monkeypatch.setattr(NODE_RUNTIME, "product_isolated_nodes", product_isolated_nodes)


def test_the_candidate_preflight_of_a_parked_and_a_populated_cluster_runs_in_parallel(
    monkeypatch: pytest.MonkeyPatch, node_preflight: None
) -> None:
    site = SnapshotSite()
    _release_fresh_read(monkeypatch, site)

    MUTATION.preflight_upgrade_mutations(site, AGENT_WAVES)

    assert sorted(cluster for cluster, _env in site.preflights) == sorted(
        [EKS, HYPERPOD]
    ), "both clusters were preflighted"
    assert site.cache_seen_by_provider_read and not any(
        site.cache_seen_by_provider_read
    ), "the parked cluster's provider reads ran outside any report snapshot"
    assert getattr(site, "_json_read_cache", None) is None, "no snapshot left open"


def test_the_zero_node_proof_opens_no_snapshot_another_thread_could_see() -> None:
    site = SnapshotSite()
    site.fresh_read_done.set()  # nothing to wait for here

    proof = PARKED.without_nodes(site, site.eks)

    assert proof is not None and proof["parked"] is True, proof
    assert site.cache_seen_by_provider_read and not any(
        site.cache_seen_by_provider_read
    ), "the provider reads ran with no release-wide snapshot open"
    assert getattr(site, "_json_read_cache", None) is None


def test_inside_a_report_snapshot_the_proof_is_read_once() -> None:
    """``status`` opens the snapshot itself: there the proof joins it, and the
    checks that ask for the same cluster are answered from one read."""

    site = SnapshotSite()
    site.fresh_read_done.set()

    with STATE.read_snapshot(site):
        first = PARKED.without_nodes(site, site.eks)
        again = PARKED.read_parked(site, site.eks)

    assert first == again and first is not None
    listings = [
        call
        for call in site.provider.calls
        if call[:3] == ["aws", "eks", "list-nodegroups"]
    ]
    assert len(listings) == 1, "one provider read per snapshot"
    assert site.cache_seen_by_provider_read and all(site.cache_seen_by_provider_read), (
        "inside status's own snapshot"
    )
    assert [item["name"] for item in first["groups"]] == [GROUP]
