"""The fleet rollout fence, and how an abandoned deployment stops holding it.

The fence refuses every destructive remediation for a cluster while any fleet
deployment there is non-terminal. ``DeploymentStatus`` has no terminal
"superseded" value, so on 2026-09-03 an upgrade that wrote its deployment record
and then died fenced a live cluster for 36 hours, silently. These cases pin the
three parts of the answer: the record gets terminalized when its successor is
minted, the fence routes around one that was not, and the hold is measurable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.fleet_preflight import (
    fleet_preflight_reason,
    superseded_fence_deployments,
)
from gpu_fault.fleet import (
    FleetDeploymentRequest,
    FleetNodeReadiness,
    FleetReadinessReport,
)
from gpu_fault.fleet_deployment import (
    DeploymentNode,
    DeploymentNodeStatus,
    DeploymentStatus,
    FleetDeployment,
)
from gpu_fault.models import WorkflowOperation
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.fleet._support import ARTIFACT, CONFIG, SECRET, heartbeat, signed

NOW = datetime(2026, 9, 3, 1, 18, tzinfo=timezone.utc)
CLUSTER = "cluster-a"
# RESET_GPU is destructive and MARK_UNSCHEDULABLE is not, which is what makes the
# fence look at this plan at all.
STEPS = [
    workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE),
    workflow_step(WorkflowOperation.RESET_GPU),
]


def deployment(
    deployment_id: str,
    *,
    created_at: datetime,
    node_statuses: list[DeploymentNodeStatus],
    cluster_id: str = CLUSTER,
) -> FleetDeployment:
    """A deployment whose status is derived from its nodes, as the store's is.

    ``status`` is passed explicitly rather than defaulted because
    ``DeploymentStatus.PLANNED`` is exactly "every node is still PENDING", and
    the fence reads that equivalence -- a case that set one without the other
    would be testing a state the store cannot hold.
    """

    nodes = [
        DeploymentNode(node_id=f"node-{index}", status=status, updated_at=created_at)
        for index, status in enumerate(node_statuses)
    ]
    if all(item.status is DeploymentNodeStatus.PENDING for item in nodes):
        status = DeploymentStatus.PLANNED
    elif all(item.status is DeploymentNodeStatus.READY for item in nodes):
        status = DeploymentStatus.SUCCEEDED
    else:
        status = DeploymentStatus.IN_PROGRESS
    return FleetDeployment(
        deployment_id=deployment_id,
        cluster_id=cluster_id,
        desired_agent_version="0.10.0",
        desired_artifact_sha256=ARTIFACT,
        desired_policy_version="catalog-a",
        desired_runtime_profile_version="profile-a",
        desired_config_digest=CONFIG,
        max_unavailable=len(nodes),
        waves=[[item.node_id for item in nodes]],
        nodes=nodes,
        status=status,
        created_at=created_at,
        updated_at=created_at,
    )


class _Registry:
    """The fence only ever reads ``store`` and ``readiness`` off the registry.

    ``readiness`` answers ready, so a released fence produces ``None`` rather
    than a second reason -- otherwise a case could not tell "the fence let this
    through" from "the fence held for a different reason".
    """

    def __init__(self, store: object) -> None:
        self.store = store
        self.readiness_calls: list[tuple[str, list[str]]] = []

    def readiness(self, cluster_id: str, node_ids: list[str]) -> FleetReadinessReport:
        self.readiness_calls.append((cluster_id, list(node_ids)))
        return FleetReadinessReport(
            cluster_id=cluster_id,
            ready=True,
            evaluated_at=NOW,
            nodes=[
                FleetNodeReadiness(
                    node_id=node_id,
                    ready=True,
                    generation=7,
                    endpoint=f"https://{node_id}:9099",
                    reasons=[],
                )
                for node_id in node_ids
            ],
        )


def fence_reason(store: object) -> str | None:
    return fleet_preflight_reason(
        _Registry(store),
        workflow_request("workflow-a", "incident-a"),
        fault_incident("incident-a", "event-a", cluster_id=CLUSTER),
        STEPS,
    )


def test_the_fence_holds_for_the_newest_never_started_deployment() -> None:
    """A rollout waiting on its first wave is the case the fence exists for."""

    store = build_store()
    store.save_fleet_deployment(
        deployment(
            "release-upgrade-aaaa-1",
            created_at=NOW,
            node_statuses=[DeploymentNodeStatus.PENDING] * 4,
        )
    )

    reason = fence_reason(store)

    assert reason is not None
    assert "1 active deployment(s)" in reason


def test_the_fence_ignores_a_never_started_deployment_a_newer_one_overtook() -> None:
    """The 2026-09-03 record: PLANNED, four PENDING nodes, 19 newer siblings."""

    store = build_store()
    store.save_fleet_deployment(
        deployment(
            "release-upgrade-aaaa-1",
            created_at=NOW,
            node_statuses=[DeploymentNodeStatus.PENDING] * 4,
        )
    )
    store.save_fleet_deployment(
        deployment(
            "release-rollback-bbbb-2",
            created_at=NOW + timedelta(hours=6),
            node_statuses=[DeploymentNodeStatus.READY] * 4,
        )
    )

    assert superseded_fence_deployments(
        store, CLUSTER, store.list_active_fleet_deployments(CLUSTER)
    ) == frozenset({"release-upgrade-aaaa-1"})
    # No fence reason at all: with the stranded record dropped there is nothing
    # active left, so the workflow goes on to its normal gates.
    assert fence_reason(store) is None


def test_an_overtaken_deployment_that_started_a_wave_keeps_fencing() -> None:
    """One INSTALLING node means something is mid-install on a real machine."""

    store = build_store()
    store.save_fleet_deployment(
        deployment(
            "release-upgrade-aaaa-1",
            created_at=NOW,
            node_statuses=[
                DeploymentNodeStatus.READY,
                DeploymentNodeStatus.INSTALLING,
                DeploymentNodeStatus.PENDING,
            ],
        )
    )
    store.save_fleet_deployment(
        deployment(
            "release-rollback-bbbb-2",
            created_at=NOW + timedelta(hours=6),
            node_statuses=[DeploymentNodeStatus.READY] * 3,
        )
    )

    reason = fence_reason(store)

    assert reason is not None
    assert "1 active deployment(s)" in reason


def test_a_newer_deployment_in_another_cluster_does_not_release_the_fence() -> None:
    """Supersession is per cluster; another cluster's roll proves nothing here."""

    store = build_store()
    store.save_fleet_deployment(
        deployment(
            "release-upgrade-aaaa-1",
            created_at=NOW,
            node_statuses=[DeploymentNodeStatus.PENDING] * 2,
        )
    )
    store.save_fleet_deployment(
        deployment(
            "release-upgrade-aaaa-2",
            created_at=NOW + timedelta(hours=6),
            node_statuses=[DeploymentNodeStatus.READY] * 2,
            cluster_id="cluster-b",
        )
    )

    reason = fence_reason(store)

    assert reason is not None
    assert "1 active deployment(s)" in reason


def test_losing_the_supersession_evidence_keeps_the_fence_closed() -> None:
    """A read failure must degrade to fencing, never to letting a mutation go."""

    store = build_store()
    store.save_fleet_deployment(
        deployment(
            "release-upgrade-aaaa-1",
            created_at=NOW,
            node_statuses=[DeploymentNodeStatus.PENDING] * 2,
        )
    )
    store.save_fleet_deployment(
        deployment(
            "release-rollback-bbbb-2",
            created_at=NOW + timedelta(hours=6),
            node_statuses=[DeploymentNodeStatus.READY] * 2,
        )
    )

    def broken() -> list[FleetDeployment]:
        raise TimeoutError("statement timeout")

    store.list_fleet_deployments = broken  # type: ignore[method-assign]
    reason = fence_reason(store)

    assert reason is not None
    assert "1 active deployment(s)" in reason


def _moving_registry() -> tuple[object, object, list[datetime]]:
    from gpu_fault.fleet import FleetCompatibilityPolicy, FleetRegistry

    store = build_store()
    clock = [NOW]

    def now() -> datetime:
        return clock[0]

    fleet = FleetRegistry(
        store,
        SECRET,
        FleetCompatibilityPolicy(
            required_agent_version="0.9.0",
            required_artifact_sha256=ARTIFACT,
            required_policy_version="catalog-a",
            required_runtime_profile_version="profile-a",
            required_config_digest=CONFIG,
        ),
        now=now,
    )
    return store, fleet, clock


def _request(node_ids: list[str]) -> FleetDeploymentRequest:
    return FleetDeploymentRequest(
        cluster_id=CLUSTER,
        node_ids=node_ids,
        desired_agent_version="0.9.0",
        desired_artifact_sha256=ARTIFACT,
        desired_policy_version="catalog-a",
        desired_runtime_profile_version="profile-a",
        desired_config_digest=CONFIG,
    )


def test_minting_a_deployment_terminalizes_the_never_started_predecessor() -> None:
    """The cure: the abandoned record becomes terminal, so retention collects it.

    Routing around it in the fence alone would not last. The retention drain only
    deletes SUCCEEDED/FAILED records, so the newer siblings that prove the old one
    dead are collected first and it becomes the newest record for the cluster --
    closing the fence again with nothing left to prove otherwise.
    """

    store, fleet, clock = _moving_registry()
    abandoned = fleet.create_deployment(_request(["node-a"]))
    assert abandoned.status is DeploymentStatus.PLANNED

    clock[0] = NOW + timedelta(hours=6)
    successor = fleet.create_deployment(_request(["node-b"]))

    stranded = store.get_fleet_deployment(abandoned.deployment_id)
    assert stranded.status is DeploymentStatus.FAILED
    assert [item.status for item in stranded.nodes] == [DeploymentNodeStatus.FAILED]
    assert successor.deployment_id in (stranded.nodes[0].reason or "")
    assert [
        item.deployment_id for item in store.list_active_fleet_deployments(CLUSTER)
    ] == [successor.deployment_id]
    # And it now ages out on the normal schedule, which it never could while
    # PLANNED.
    assert (
        store.cleanup_terminal_fleet_deployments(
            older_than=NOW + timedelta(days=8), limit=100
        )
        == 1
    )


def test_minting_a_deployment_leaves_a_started_predecessor_alone() -> None:
    """A wave in flight is not superseded by a later create; it is a conflict."""

    store, fleet, clock = _moving_registry()
    fleet.register(signed(heartbeat("node-a", observed_at=NOW)))
    started = fleet.create_deployment(_request(["node-a", "node-b"]))
    assert started.status is DeploymentStatus.IN_PROGRESS

    clock[0] = NOW + timedelta(hours=6)
    fleet.create_deployment(_request(["node-c"]))

    assert (
        store.get_fleet_deployment(started.deployment_id).status
        is DeploymentStatus.IN_PROGRESS
    )


def test_the_fence_hold_is_measurable_per_cluster() -> None:
    """The hold was a WARNING on one worker and nothing else; now it is a gauge."""

    from types import SimpleNamespace

    from gpu_fault.app.builtin_metric_contributors import fleet_rollout_metric_lines

    store = build_store()
    store.save_fleet_deployment(
        deployment(
            "release-upgrade-aaaa-1",
            created_at=datetime.now(timezone.utc) - timedelta(hours=36),
            node_statuses=[DeploymentNodeStatus.PENDING] * 4,
        )
    )
    runtime = SimpleNamespace(context=SimpleNamespace(store=store, regional_mode=True))
    store.list_regional_cluster_ids = lambda: [CLUSTER]  # type: ignore[method-assign]

    lines = fleet_rollout_metric_lines(runtime)

    age = next(
        line
        for line in lines
        if line.startswith("gpu_fault_fleet_rollout_fence_age_seconds{")
    )
    assert f'cluster_id="{CLUSTER}"' in age
    assert float(age.rsplit(" ", 1)[1]) > 36 * 3600 - 60
    assert f'gpu_fault_fleet_rollout_never_started{{cluster_id="{CLUSTER}"}} 1' in lines


class _RemoteRegistry:
    """Shaped like ``RegionalFleetRegistry``: no store of its own.

    ``store = self`` is the real proxy's arrangement, so a case that omits it
    would not reproduce the failure these two cases exist for.
    """

    def __init__(self, fencing: list[str] | None, *, capable: bool = True) -> None:
        self.store = self
        self.calls = 0
        self._fencing = fencing
        if capable:
            self.fleet_rollout_fence_deployments = self._answer  # type: ignore[method-assign]

    def _answer(self, cluster_id: str) -> list[str]:
        self.calls += 1
        assert cluster_id == CLUSTER
        if self._fencing is None:
            raise RuntimeError("control plane unreachable")
        return list(self._fencing)

    def readiness(self, cluster_id: str, node_ids: list[str]) -> FleetReadinessReport:
        return FleetReadinessReport(
            cluster_id=cluster_id,
            ready=True,
            evaluated_at=NOW,
            nodes=[
                FleetNodeReadiness(
                    node_id=node_id,
                    ready=True,
                    generation=7,
                    endpoint=f"https://{node_id}:9099",
                    reasons=[],
                )
                for node_id in node_ids
            ],
        )


def registry_reason(registry: object) -> str | None:
    return fleet_preflight_reason(
        registry,
        workflow_request("workflow-a", "incident-a"),
        fault_incident("incident-a", "event-a", cluster_id=CLUSTER),
        STEPS,
    )


def test_the_data_plane_asks_the_control_plane_for_the_fence() -> None:
    """The executor that owns no store still gets a real verdict, both ways."""

    open_registry = _RemoteRegistry([])
    assert registry_reason(open_registry) is None
    assert open_registry.calls == 1

    held = _RemoteRegistry(["release-upgrade-aaaa-1"])
    reason = registry_reason(held)
    assert reason is not None
    assert "fleet rollout fence blocked destructive workflow" in reason
    assert "1 active deployment(s)" in reason


def test_an_unreachable_fence_still_fails_closed_on_the_data_plane() -> None:
    """Losing the answer must hold, or the fence buys nothing under a partition."""

    reason = registry_reason(_RemoteRegistry(None))

    assert reason is not None
    assert "fleet rollout fence is unavailable" in reason
    assert "RuntimeError" in reason


def test_a_storeless_proxy_is_not_mistaken_for_a_store() -> None:
    """The 2026-09-04 outage: AttributeError read as "fence unavailable".

    ``store = self`` made the fence call ``list_active_fleet_deployments`` on a
    proxy that has no such method. Because the fence fails closed, every
    destructive remote command on every regional cluster was held forever --
    a total loss of remediation, from a fence that was meant to hold rarely.
    A capability the registry does not have is a static fact about the
    registry, so it must not surface as a read failure at the mutation
    boundary.
    """

    incapable = _RemoteRegistry([], capable=False)

    assert registry_reason(incapable) is None


def test_the_real_regional_proxy_implements_the_fence_capability() -> None:
    """Pins the production proxy to the seam, not just the test double."""

    from gpu_fault.cluster_executor import RegionalFleetRegistry

    assert callable(
        getattr(RegionalFleetRegistry, "fleet_rollout_fence_deployments", None)
    ), "the regional proxy must answer the fence it cannot read locally"
