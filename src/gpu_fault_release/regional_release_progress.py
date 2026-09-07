from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

from gpu_fault_release.regional_release_diff import ReleaseComponent

PROGRESS_SCHEMA_VERSION = 1
PROGRESS_STARTED = "STARTED"
PROGRESS_COMPLETED = "COMPLETED"
PROGRESS_FAILED = "FAILED"
ACTIVE_PROGRESS_STATUSES = frozenset(
    {
        PROGRESS_STARTED,
        PROGRESS_COMPLETED,
        PROGRESS_FAILED,
    }
)
CLUSTER_ATTEMPT_STATES = frozenset(
    {"PENDING", "RUNNING", "CONVERGED", "PAUSED", "FAILED", "ROLLED_BACK"}
)


def cluster_attempts_with(
    state: dict[str, Any],
    cluster_id: str,
    lifecycle: str,
    *,
    details: dict[str, Any] | None = None,
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The attempt ledger with this cluster's entry advanced to `lifecycle`.

    `details` is operator-facing context and is nested; `fields` are attempt
    facts other code *reads* (`converged_at_epoch`), so they sit beside `state`
    where a reader can find them without knowing which release wrote them.
    """

    if lifecycle not in CLUSTER_ATTEMPT_STATES:
        raise ValueError(f"unsupported cluster attempt state: {lifecycle}")
    attempts = {
        str(key): dict(value)
        for key, value in (state.get("cluster_attempts") or {}).items()
        if isinstance(value, dict)
    }
    previous = dict(attempts.get(cluster_id) or {})
    generation = int(previous.get("attempt_generation") or 1)
    if lifecycle == "RUNNING" and previous.get("state") in {
        "PAUSED",
        "FAILED",
        "ROLLED_BACK",
    }:
        generation += 1
    entry = {
        **previous,
        "state": lifecycle,
        "attempt_generation": generation,
        "updated_at_epoch": time.time(),
    }
    if details:
        entry["details"] = {
            **dict(previous.get("details") or {}),
            **details,
        }
    if fields:
        reserved = {"state", "attempt_generation", "updated_at_epoch"}
        if forbidden := reserved & fields.keys():
            raise ValueError(
                f"fields must not contain reserved keys: {sorted(forbidden)}"
            )
        entry.update(fields)
    attempts[cluster_id] = entry
    return attempts


GPU_COMPONENTS = frozenset(
    {
        ReleaseComponent.ENDPOINT,
        ReleaseComponent.DCGM,
        ReleaseComponent.EXECUTOR,
        ReleaseComponent.WATCHER,
        ReleaseComponent.COLLECTOR,
        ReleaseComponent.RECONCILER,
        ReleaseComponent.AGENT,
    }
)
CPU_COMPONENTS = frozenset(
    {
        ReleaseComponent.REGISTRY,
        ReleaseComponent.CPU_STAGE,
        ReleaseComponent.RUNTIME_PROFILE,
        ReleaseComponent.CPU_FINALIZE,
    }
)
ROLLBACK_COMPONENTS = (
    GPU_COMPONENTS | CPU_COMPONENTS | frozenset({ReleaseComponent.OBSERVABILITY})
)
REPLAYED_MANIFEST_CHANGES = frozenset(
    {
        "cpu_manifests",
        "cpu_ingress_manifests",
        "cpu_worker_manifests",
        "cpu_spool_manifests",
        "schema_manifests",
        "executor_manifests",
        "watcher_manifests",
        "collector_manifests",
        "node_manifests",
        "dcgm_manifests",
    }
)
"""Manifest deliveries a rollback re-renders from the candidate checkout.

These are the ``delivery_component_digests`` entries -- digests of the manifest
*files*, not of the rendered output, so they move only when a template moves and
never when a pin moves. When one of them is in ``release_diff.changed`` the
rollback puts the previous release's pins into the *candidate's* templates,
because that is the only copy of them on disk. The result is correct for
everything the pins control and silently wrong for anything the template edit
itself changed, so it is recorded rather than left to be inferred.

``observability_manifests`` and ``endpoint_manifests`` are deliberately absent:
both are restored from a snapshot of the live objects
(``regional_observability_rollback``, ``regional_endpoint_rollback``) rather than
re-applied from the tree, so a template edit to either is genuinely undone.
"""

PHASE_COMPONENTS = {
    "registry-staged": ReleaseComponent.REGISTRY,
    "cpu-staged": ReleaseComponent.CPU_STAGE,
    "profile-ready": ReleaseComponent.RUNTIME_PROFILE,
    "endpoint-ready": ReleaseComponent.ENDPOINT,
    "observability-ready": ReleaseComponent.OBSERVABILITY,
    "cpu-finalized": ReleaseComponent.CPU_FINALIZE,
}


@dataclass(frozen=True)
class RollbackCompensationPlan:
    global_components: frozenset[ReleaseComponent]
    cluster_components: dict[str, frozenset[ReleaseComponent]]
    conservative: bool
    replayed_manifests: frozenset[str] = frozenset()

    def global_has(self, *components: ReleaseComponent) -> bool:
        return bool(self.global_components.intersection(components))

    def for_cluster(self, cluster_id: str) -> frozenset[ReleaseComponent]:
        return self.cluster_components.get(cluster_id, frozenset())

    @property
    def needs_controller(self) -> bool:
        return any(
            ReleaseComponent.AGENT in components
            for components in self.cluster_components.values()
        )

    @property
    def restores_cpu(self) -> bool:
        return bool(self.global_components.intersection(CPU_COMPONENTS)) or (
            self.needs_controller
        )

    @property
    def restores_data_plane(self) -> bool:
        return any(self.cluster_components.values())

    @property
    def restores_observability(self) -> bool:
        return ReleaseComponent.OBSERVABILITY in self.global_components

    @property
    def restores_endpoint(self) -> bool:
        """Whether the global half of the endpoint component has to be put back.

        The endpoint component does two unrelated things. Per GPU cluster it
        writes the connection Secret, which ``rollback_target`` already restores
        from the Secret backups. Once, globally, it applies the NLB Service and
        UPSERTs the Route53 record -- and that is what the ``endpoint-ready``
        phase records and what this compensates.
        """

        return ReleaseComponent.ENDPOINT in self.global_components

    @property
    def empty(self) -> bool:
        return not self.global_components and not self.restores_data_plane

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "conservative": self.conservative,
            "global_components": sorted(
                component.value for component in self.global_components
            ),
            "clusters": {
                cluster_id: sorted(component.value for component in components)
                for cluster_id, components in sorted(self.cluster_components.items())
                if components
            },
            "needs_controller": self.needs_controller,
            "restores_cpu": self.restores_cpu,
            "restores_data_plane": self.restores_data_plane,
            "restores_observability": self.restores_observability,
            "restores_endpoint": self.restores_endpoint,
            "replayed_manifests": sorted(self.replayed_manifests),
        }


def update_component_progress(
    state: dict[str, Any],
    component: ReleaseComponent,
    status: str,
    *,
    cluster_id: str | None = None,
    observed_at_epoch: float | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in ACTIVE_PROGRESS_STATUSES:
        raise ValueError(f"unsupported component progress status: {status}")
    observed = observed_at_epoch if observed_at_epoch is not None else time.time()
    raw = state.get("component_progress")
    if isinstance(raw, dict) and raw.get("schema_version") == PROGRESS_SCHEMA_VERSION:
        # Copy only the path this update touches. The caller's progress tree must
        # stay untouched, but a batch of component transitions would otherwise
        # deep-copy every recorded component of every cluster once per component.
        progress = dict(raw)
        progress["global"] = dict(progress.get("global") or {})
        progress["clusters"] = dict(progress.get("clusters") or {})
    else:
        progress = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "global": {},
            "clusters": {},
        }
    if cluster_id is None:
        scope = progress["global"]
    else:
        clusters = progress["clusters"]
        scope = dict(clusters.get(cluster_id) or {})
        clusters[cluster_id] = scope
    entry = dict(scope.get(component.value) or {})
    entry["status"] = status
    entry.setdefault("started_at_epoch", observed)
    entry["updated_at_epoch"] = observed
    if status == PROGRESS_COMPLETED:
        entry["completed_at_epoch"] = observed
        entry["duration_seconds"] = max(
            0.0,
            observed - float(entry["started_at_epoch"]),
        )
    if details:
        entry["details"] = {**dict(entry.get("details") or {}), **details}
    scope[component.value] = entry
    return progress


def update_components_progress(
    state: dict[str, Any],
    components: Iterable[ReleaseComponent],
    status: str,
    *,
    cluster_id: str | None = None,
    observed_at_epoch: float | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = tuple(dict.fromkeys(components))
    if not selected:
        raise ValueError("component progress batch cannot be empty")
    observed = observed_at_epoch if observed_at_epoch is not None else time.time()
    raw_progress = state.get("component_progress")
    progress: dict[str, Any] | None = (
        raw_progress if isinstance(raw_progress, dict) else None
    )
    working = dict(state)
    for component in selected:
        if progress is not None:
            working["component_progress"] = progress
        progress = update_component_progress(
            working,
            component,
            status,
            cluster_id=cluster_id,
            observed_at_epoch=observed,
            details=details,
        )
    if progress is None:
        raise RuntimeError("component progress batch produced no state")
    return progress


def _execution_components(
    state: dict[str, Any],
) -> tuple[frozenset[ReleaseComponent], bool]:
    raw = (state.get("execution_plan") or {}).get("nodes")
    if not isinstance(raw, list):
        return ROLLBACK_COMPONENTS, True
    try:
        components = frozenset(ReleaseComponent(str(value)) for value in raw)
    except ValueError:
        return ROLLBACK_COMPONENTS, True
    return components.intersection(ROLLBACK_COMPONENTS), False


def _progress_components(raw: object) -> frozenset[ReleaseComponent]:
    if not isinstance(raw, dict):
        return frozenset()
    selected: set[ReleaseComponent] = set()
    for name, entry in raw.items():
        if not isinstance(entry, dict) or entry.get("status") not in (
            ACTIVE_PROGRESS_STATUSES
        ):
            continue
        try:
            selected.add(ReleaseComponent(str(name)))
        except ValueError:
            continue
    return frozenset(selected).intersection(ROLLBACK_COMPONENTS)


def _legacy_gpu_may_have_started(
    execution_components: frozenset[ReleaseComponent],
    completed_phases: set[str],
) -> bool:
    if not execution_components.intersection(GPU_COMPONENTS):
        return False
    # Only the phases this release both checkpoints *and* reaches before the
    # data plane can gate the data plane. Two ways that goes wrong: a plan
    # without a schema change writes no `schema-ready`, so requiring it
    # unconditionally reads every such state as "the GPU rollout never started"
    # and compensates nothing; and `observability-ready` is now recorded *after*
    # the clusters roll (the install runs beside them and is joined before the
    # finalize), so requiring it would do the same to every state that crashed
    # mid-rollout. `uploaded` is the floor: it is the last phase before the data
    # plane may run in a plan with no control-plane work. Being wrong in this
    # direction means compensating a cluster that never started, which is a
    # replay of the previous release's manifests -- the safe direction.
    required = {"uploaded"}
    if ReleaseComponent.SCHEMA in execution_components:
        required.add("schema-ready")
    if ReleaseComponent.REGISTRY in execution_components:
        required.add("registry-staged")
    if ReleaseComponent.CPU_STAGE in execution_components:
        required.add("cpu-staged")
    if ReleaseComponent.RUNTIME_PROFILE in execution_components:
        required.add("profile-ready")
    if ReleaseComponent.ENDPOINT in execution_components:
        required.add("endpoint-ready")
    return required.issubset(completed_phases)


def build_rollback_compensation_plan(
    state: dict[str, Any],
    cluster_ids: Iterable[str],
) -> RollbackCompensationPlan:
    clusters = tuple(sorted(set(cluster_ids)))
    execution_components, unknown_execution = _execution_components(state)
    completed_phases = set(state.get("completed_phases") or [])
    completed_clusters = set(state.get("completed_cluster_ids") or [])
    raw_progress = state.get("component_progress")
    has_progress = (
        isinstance(raw_progress, dict)
        and raw_progress.get("schema_version") == PROGRESS_SCHEMA_VERSION
    )
    conservative = unknown_execution or not has_progress
    legacy_without_evidence = (
        unknown_execution and not has_progress and not completed_phases
    )

    global_components = set(
        _progress_components(
            (raw_progress or {}).get("global") if has_progress else None
        )
    )
    if legacy_without_evidence:
        global_components.update(execution_components.intersection(CPU_COMPONENTS))
    for phase, component in PHASE_COMPONENTS.items():
        if phase in completed_phases and component in execution_components:
            global_components.add(component)
    if (
        ReleaseComponent.CPU_FINALIZE in execution_components
        and "data-converged" in completed_phases
        and "cpu-finalized" not in completed_phases
        and not has_progress
    ):
        global_components.add(ReleaseComponent.CPU_FINALIZE)
    if (
        ReleaseComponent.CPU_STAGE in execution_components
        and "registry-staged" in completed_phases
        and "cpu-staged" not in completed_phases
        and not has_progress
    ):
        global_components.add(ReleaseComponent.CPU_STAGE)

    by_cluster: dict[str, frozenset[ReleaseComponent]] = {}
    progress_clusters = (raw_progress or {}).get("clusters") if has_progress else {}
    gpu_plan = execution_components.intersection(GPU_COMPONENTS)
    legacy_gpu_started = (
        _legacy_gpu_may_have_started(
            execution_components,
            completed_phases,
        )
        or legacy_without_evidence
    )
    for cluster_id in clusters:
        selected = set(
            _progress_components(
                (progress_clusters or {}).get(cluster_id)
                if isinstance(progress_clusters, dict)
                else None
            )
        )
        if cluster_id in completed_clusters:
            selected.update(gpu_plan)
        elif not has_progress and legacy_gpu_started:
            selected.update(gpu_plan)
        by_cluster[cluster_id] = frozenset(selected.intersection(gpu_plan))

    changed = (state.get("release_diff") or {}).get("changed") or []
    return RollbackCompensationPlan(
        global_components=frozenset(
            global_components.intersection(execution_components)
        ),
        cluster_components=by_cluster,
        conservative=conservative,
        replayed_manifests=REPLAYED_MANIFEST_CHANGES.intersection(
            str(entry) for entry in changed
        ),
    )
