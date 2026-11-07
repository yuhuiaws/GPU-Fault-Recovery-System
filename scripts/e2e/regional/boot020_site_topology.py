"""GF-REGIONAL-BOOT-020 judged per GPU cluster, also when a cluster has no node.

The contract was written for a site whose GPU clusters all have nodes, and its
verdicts read the site as one: a rollback plan's components were the union
over every cluster, an Agent or FULL stage passed when *some* cluster's
capture moved, and nothing read the Fleet waves. With a cluster parked at zero
GPU nodes (``gpu_fault_release.regional_parked_cluster``) that proves less than
it says -- the parked cluster's components and its Reconciler roll can stand
in for a cluster with nodes whose Agents never moved -- and nothing at all of
the parked path. Every stage is therefore judged per cluster from what the run
recorded (``check_stage``, before the stage's pass marker):

* **The site shape** (``guard_site_shape``), before anything changes the site
  and again on every resume. Each cluster has nodes, or none and its provider
  proves it *parked* (``regional_parked_cluster.without_nodes``, read fresh: a
  proof recorded by an earlier release never stands in for a provider read
  that failed). The operator names the clusters expected without nodes
  (``--zero-node-cluster``); the run refuses unless exactly those have no node
  and are parked, and when no cluster has nodes. The shape is recorded as the
  stage ``site_shape``; a later attempt that finds another shape is refused,
  and so is any full snapshot whose Agent identities disagree with it.
* **A cluster with nodes** shows the full node behaviour: a rollback plan that
  names the stage's components for it; on the Agent and FULL rollbacks Fleet
  waves that cover each of its nodes exactly once, every wave completed with
  safety start, safety end, Reconciler applied, Agents converged and done in
  that order; Agent identities unchanged by the CPU and Executor releases,
  changed by the Agent and FULL ones (node runtime artifact and installer
  bundle, and for FULL its own Runtime Profile version and Agent config
  digest), and after the restore exactly those the run started with. It never
  gains a zero-node record.
* **A cluster without nodes** shows the parked path: the explicit identity
  entry ``{node_ids: [], zero_nodes: {state, ...}}`` in every full snapshot;
  the rollback components the clusters with nodes got; no Fleet wave; per
  deploy that does GPU work the engine's record of the node work it skipped
  (``state.zero_node_clusters``: this transaction's release id, the phase --
  ``upgrade`` forward, ``rollback`` when compensating -- the skipped steps, and
  a Reconciler identity equal to the one captured after it); and a Reconciler
  that takes each node runtime bundle. A release without GPU work (NOOP, the
  CPU-only drill, the executor attempt interrupted at ``cpu-staged``) records
  no node work of its own: a new transaction starts from an empty state, a
  NOOP carries the record it found.

The FULL candidate registers its suffixed Runtime Profile versions for good
(``POST /v1/runtime-profiles`` is create-only); the stage
``full_runtime_profile_versions`` names them, and the completed evidence
repeats them as ``registered_runtime_profile_versions``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from gpu_fault_release import regional_parked_cluster as parked_cluster
from gpu_fault_release.regional_release_profile_model import (
    runtime_profiles,
    target_agent_config_digest,
    target_runtime_profile_version,
)
from scripts.e2e.regional.acceptance_runner_common import utc_now
from scripts.e2e.regional.boot020_deployment_reads import AcceptanceCheckError

Require = Callable[..., None]

SHAPE_STAGE = "site_shape"
PROFILE_STAGE = "full_runtime_profile_versions"
NODES = "nodes"
ZERO_NODES = parked_cluster.ZERO_NODES_IDENTITY_KEY
#: The deploy result keys: the engine's zero-node records after the deploy
#: (``state.zero_node_clusters``) and as the deploy found them.
RECORDS = parked_cluster.ZERO_NODE_KEY
RECORDS_BEFORE = f"{RECORDS}_before"
#: A Fleet wave's events, in the order every completed wave records them
#: (``regional_release_fleet_rollout`` -> ``record_rollback_wave_event``).
WAVE_EVENTS = (
    "safety_started",
    "safety_completed",
    "reconciler_applied",
    "agents_converged",
    "completed",
)
#: What a node runtime release changes in each Agent's identity.
NODE_RUNTIME_FIELDS = ("artifact_sha256", "installer_bundle_sha256")
#: The components each data-plane rollback restores on a cluster with nodes.
ROLLBACK_COMPONENTS = {
    "executor": frozenset({"executor"}),
    "agent": frozenset({"agent"}),
    "full": frozenset({"agent", "executor"}),
}
#: The node work the engine records as skipped on a cluster without nodes: a
#: Reconciler-only roll skips the node preflight batch alone
#: (``regional_release_gpu_rollout.deploy_reconciler_of_cluster``), a node
#: runtime roll all of its node work.
RECONCILER_ONLY_SKIPPED = ("node-preflight-batch",)
SKIPPED = {
    "executor": RECONCILER_ONLY_SKIPPED,
    "agent": tuple(parked_cluster.SKIPPED_NODE_WORK),
    "full": tuple(parked_cluster.SKIPPED_NODE_WORK),
    "restore": tuple(parked_cluster.SKIPPED_NODE_WORK),
}
#: Per stage: the deploy results and, for each, the phase its zero-node record
#: must carry (``None``: the deploy does no GPU work and leaves the record) and
#: the snapshot that captured the Reconciler the record names.
DEPLOYS: dict[str, tuple[tuple[str, str | None, str | None], ...]] = {
    "noop": (("noop_apply", None, None),),
    "control_plane": (
        ("control_plane_injected_failure_and_rollback", None, None),
        ("control_plane_apply", None, None),
    ),
    "executor": (
        (
            "executor_injected_failure_and_rollback",
            "rollback",
            "executor_rollback_snapshot",
        ),
        ("executor_interrupted_failure", None, None),
        ("executor_resumed", "upgrade", "executor_after"),
    ),
    "agent": (
        ("agent_injected_failure_and_rollback", "rollback", "agent_rollback_snapshot"),
        ("agent_apply_after_rollback", "upgrade", "agent_after"),
    ),
    "full": (
        ("full_injected_failure_and_rollback", "rollback", "full_rollback_snapshot"),
        ("full_apply_after_rollback", "upgrade", "full_after"),
    ),
    # The site's own config against candidate D: its node runtime differs by
    # construction (``boot020_release_candidates.validate_candidate_delta``).
    "restore": (("restore_apply", "upgrade", "restore_after"),),
}
SNAPSHOTS = {
    "noop": ("noop_before", "noop_after"),
    "control_plane": (
        "control_plane_before",
        "control_plane_rollback_snapshot",
        "control_plane_after",
    ),
    "executor": ("executor_before", "executor_rollback_snapshot", "executor_after"),
    "agent": ("agent_before", "agent_rollback_snapshot", "agent_after"),
    "full": ("full_before", "full_rollback_snapshot", "full_after"),
    "restore": ("restore_before", "restore_after"),
}
ROLLBACKS = {
    stage: f"{stage}_injected_failure_and_rollback"
    for stage in ("control_plane", "executor", "agent", "full")
}
#: The snapshots whose Agent identities a stage compares; the restore compares
#: its end with the run's first observation.
IDENTITY_PAIRS = {
    "control_plane": ("control_plane_before", "control_plane_after"),
    "executor": ("executor_before", "executor_after"),
    "agent": ("agent_before", "agent_after"),
    "full": ("full_before", "full_after"),
    "restore": ("noop_before", "restore_after"),
}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _brief(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)[:800]


# --- the site shape -----------------------------------------------------------


def node_shape(release: Any, target: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """``target``'s entry of the site shape: its node names, or the proof of a
    cluster that lists none (``regional_parked_cluster.without_nodes``)."""

    nodes = sorted(name for name in names if name)
    if nodes:
        return {NODES: nodes}
    proof = parked_cluster.without_nodes(release, target)
    if proof is None:
        return {
            ZERO_NODES: None,
            "unproven": (
                "the node list was empty, then its selector selected a node or "
                "the node list could not be read"
            ),
        }
    return {ZERO_NODES: parked_cluster.summary(proof)}


def live_node_safety(backend: Any, scenario: str) -> dict[str, Any]:
    """Every GPU cluster's nodes through the engine's own safety gate, and the
    site shape (``node_shape``).

    ``validate_target_node_state`` is what the data-plane wave runs right
    before mutating nodes (quarantine/health taints, cordons that are not a
    parked warm spare, unready or deleting nodes, active installers); it
    raises ``ReleaseError`` with the blockers, which are recorded here. A
    cluster without nodes has nothing to block; its shape entry says why it
    has none.
    """

    from gpu_fault_release import regional_release_node_preflight as preflight
    from gpu_fault_release.regional_release_config import ReleaseError
    from gpu_fault_release.regional_release_gpu_rollout import gpu_node_items

    release = backend._release(scenario)
    clusters: dict[str, Any] = {}
    shape: dict[str, Any] = {}
    for target in release.config.clusters:
        names = tuple(
            sorted(
                str((item.get("metadata") or {}).get("name") or "")
                for item in gpu_node_items(release, target, fresh=True)
            )
        )
        try:
            preflight.validate_target_node_state(release, target, names)
        except ReleaseError as exc:
            text = str(exc)
            blockers: Any = text
            if ": [" in text:
                try:
                    blockers = json.loads(text[text.index(": [") + 2 :])
                except ValueError:
                    blockers = text
            clusters[target.cluster_id] = blockers
        else:
            clusters[target.cluster_id] = []
        shape[target.cluster_id] = node_shape(release, target, names)
    return {
        "safe": not any(clusters.values()),
        "clusters": clusters,
        "shape": shape,
        "checked_at": utc_now(),
    }


def declared_zero_node_clusters(document: Mapping[str, Any]) -> list[str]:
    """The clusters the run's inputs declare without nodes (``--zero-node-cluster``)."""

    values = _mapping(document.get("inputs")).get("zero_node_clusters") or ()
    return sorted({str(value) for value in values})


def shape_refusal(shape: object, declared: list[str]) -> str | None:
    """Why the site shape is no place to run BOOT-020, or ``None``."""

    if not isinstance(shape, Mapping) or not shape:
        return (
            "BOOT-020 refuses before changing anything: node safety reported no "
            "site shape (which GPU clusters have nodes)"
        )
    problems: list[str] = []
    for cluster_id, value in sorted(shape.items()):
        entry = _mapping(value)
        if NODES in entry:
            if cluster_id in declared:
                problems.append(
                    f"{cluster_id} is declared with --zero-node-cluster but has "
                    f"{len(entry[NODES])} node(s)"
                )
            continue
        proof = entry.get(ZERO_NODES)
        if not isinstance(proof, Mapping):
            problems.append(
                f"{cluster_id} lists no node but its zero-node proof could not be "
                f"read ({entry.get('unproven')})"
            )
        elif cluster_id not in declared:
            problems.append(
                f"{cluster_id} has no node ({proof.get('state')}: "
                f"{proof.get('reason')}) and is not declared; pass "
                f"--zero-node-cluster {cluster_id} to run over it, or bring its "
                "nodes back"
            )
        elif proof.get("state") != parked_cluster.PARKED:
            problems.append(
                f"{cluster_id} has no node and is {proof.get('state')}, not parked "
                f"({proof.get('reason')}): its capacity may come back mid-run"
            )
    if not any(NODES in _mapping(entry) for entry in shape.values()):
        problems.append(
            "no GPU cluster has nodes: the Agent and FULL stages would prove no "
            "node behaviour"
        )
    unknown = sorted(set(declared) - {str(cluster_id) for cluster_id in shape})
    if unknown:
        problems.append(
            "--zero-node-cluster names no cluster of the site: " + ", ".join(unknown)
        )
    if not problems:
        return None
    return "BOOT-020 refuses this site shape before changing anything: " + "; ".join(
        problems
    )


def _shape_key(shape: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(cluster_id): (
            [NODES, list(_mapping(entry).get(NODES) or [])]
            if NODES in _mapping(entry)
            else [ZERO_NODES, _mapping(_mapping(entry).get(ZERO_NODES)).get("state")]
        )
        for cluster_id, entry in shape.items()
    }


def guard_site_shape(recorder: Any, safety: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a site shape BOOT-020 cannot judge, record it once, refuse drift.

    Run before the first mutation of every attempt. The first attempt records
    the shape as the stage ``site_shape``; every later one must find the same
    clusters with the same nodes, and the same clusters without nodes in the
    same state -- the evidence of the passed stages describes that shape.
    """

    declared = declared_zero_node_clusters(recorder.document)
    shape = safety.get("shape")
    refusal = shape_refusal(shape, declared)
    if refusal is not None:
        raise AcceptanceCheckError(refusal)
    observed = dict(_mapping(shape))
    recorded = recorder.stage(
        SHAPE_STAGE,
        lambda: {
            "clusters": observed,
            "declared_zero_node_clusters": declared,
            "recorded_at": utc_now(),
        },
    )
    before, now = _shape_key(_mapping(recorded.get("clusters"))), _shape_key(observed)
    if before != now:
        raise AcceptanceCheckError(
            f"the site shape changed since this run recorded it ({SHAPE_STAGE}): "
            f"recorded {_brief(before)}, now {_brief(now)}; the evidence of the "
            "passed stages describes the recorded shape, so continue only on it "
            "or start a new --run-dir"
        )
    return dict(_mapping(recorded.get("clusters")))


def recorded_shape(document: Mapping[str, Any]) -> dict[str, Any]:
    shape = _mapping(_mapping(document.get("stages")).get(SHAPE_STAGE)).get("clusters")
    if not isinstance(shape, Mapping) or not shape:
        raise AcceptanceCheckError(
            f"the run recorded no site shape ({SHAPE_STAGE}); run it again from "
            "its beginning"
        )
    return {str(key): dict(_mapping(value)) for key, value in shape.items()}


def _node_clusters(shape: Mapping[str, Any]) -> list[str]:
    return sorted(cid for cid, entry in shape.items() if NODES in _mapping(entry))


def _zero_node_clusters(shape: Mapping[str, Any]) -> list[str]:
    return sorted(cid for cid, entry in shape.items() if NODES not in _mapping(entry))


# --- the engine's records ------------------------------------------------------


def zero_node_records(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``state.zero_node_clusters``: per cluster, the node work skipped there."""

    return {
        str(cluster_id): dict(entry)
        for cluster_id, entry in _mapping(state.get(RECORDS)).items()
        if isinstance(entry, Mapping)
    }


def profile_versions_record(drill: Any, site: Any) -> dict[str, Any]:
    """Per cluster, the Runtime Profile and Agent config digest of ``drill`` and
    of the site's own release config, and the versions only ``drill`` names
    -- registered for good when it is deployed."""

    def per_cluster(config: Any) -> dict[str, dict[str, Any]]:
        return {
            str(target.cluster_id): {
                "version": target_runtime_profile_version(config, target),
                "agent_config_digest": target_agent_config_digest(config, target),
                "own_profile": bool(
                    getattr(target, "overrides_runtime_profile", False)
                ),
            }
            for target in config.clusters
        }

    existing = {spec.version for spec in runtime_profiles(site)}
    return {
        "clusters": per_cluster(drill),
        "site_clusters": per_cluster(site),
        "registered_versions": [
            spec.version
            for spec in runtime_profiles(drill)
            if spec.version not in existing
        ],
        "permanent": True,
        "why": (
            "POST /v1/runtime-profiles is create-only: a registered version is "
            "immutable and stays in the store after the restore; a later BOOT-020 "
            "with the same policy registers the same versions again idempotently"
        ),
    }


def live_profile_versions(backend: Any, scenario: str) -> dict[str, Any]:
    """``profile_versions_record`` of ``scenario``'s config against the site's."""

    load = backend.config_module.ReleaseConfig.load
    return profile_versions_record(
        load(backend.configs[scenario]), load(backend.configs["noop"])
    )


# --- the per-cluster verdicts --------------------------------------------------


def _identities(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return dict(_mapping(_mapping(snapshot.get("live")).get("agent_identities")))


def _captures(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return dict(_mapping(_mapping(snapshot.get("live")).get("clusters")))


def assert_snapshot_shape(
    snapshot: object, shape: Mapping[str, Any], *, label: str, require: Require
) -> None:
    """A full snapshot's Agent identities are the site shape, cluster by cluster."""

    require(isinstance(snapshot, Mapping), f"{label} was not recorded")
    if not isinstance(_mapping(snapshot).get("live"), Mapping):
        return  # the light form (live=False) captures no identity
    identities = _identities(_mapping(snapshot))
    require(
        set(identities) == set(shape),
        f"{label}: the captured Agent identities name other clusters than the site "
        "shape",
        {"captured": sorted(identities), "shape": sorted(shape)},
    )
    for cluster_id, entry in sorted(shape.items()):
        identity = identities.get(cluster_id)
        if NODES in entry:
            require(
                isinstance(identity, Mapping)
                and not parked_cluster.is_zero_node_identity(identity)
                and sorted(str(n) for n in identity.get("node_ids") or ())
                == list(entry[NODES]),
                f"{label}: the Agents of {cluster_id} are not exactly its nodes",
                {"nodes": entry[NODES], "identity": identity},
            )
        else:
            require(
                parked_cluster.is_zero_node_identity(identity)
                and _mapping(_mapping(identity).get(ZERO_NODES)).get("state")
                == _mapping(entry.get(ZERO_NODES)).get("state"),
                f"{label}: {cluster_id} has no nodes but its capture holds no "
                "explicit zero-node entry in that state",
                identity,
            )


def assert_waves(
    cluster_id: str,
    waves: object,
    nodes: list[str],
    *,
    label: str,
    require: Require,
) -> None:
    """Every node in exactly one completed wave, its five events in order."""

    require(
        isinstance(waves, Mapping) and bool(waves),
        f"{label}: {cluster_id} recorded no Fleet wave of its rollback",
        waves,
    )
    seen: Counter[str] = Counter()
    for key, wave in sorted(_mapping(waves).items()):
        wave = _mapping(wave)
        seen.update(str(node) for node in wave.get("nodes") or ())
        stamps = [wave.get(f"{event}_at_epoch") for event in WAVE_EVENTS]
        events = dict(zip(WAVE_EVENTS, stamps, strict=True))
        require(
            wave.get("status") == "COMPLETED",
            f"{label}: {cluster_id} wave {key} did not complete",
            wave.get("status"),
        )
        require(
            all(
                isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
                for stamp in stamps
            ),
            f"{label}: {cluster_id} wave {key} lacks one of its five timestamps",
            events,
        )
        require(
            stamps == sorted(stamps),
            f"{label}: {cluster_id} wave {key} timestamps are out of order",
            events,
        )
    require(
        sorted(seen) == sorted(nodes) and set(seen.values()) == {1},
        f"{label}: the rollback waves of {cluster_id} do not cover each of its "
        "nodes exactly once",
        {"nodes": sorted(nodes), "waves": dict(sorted(seen.items()))},
    )


def _plan_components(result: Mapping[str, Any]) -> dict[str, set[str]]:
    clusters = _mapping(_mapping(result.get("rollback_plan")).get("clusters"))
    return {
        str(cluster_id): {str(component).lower() for component in components or ()}
        for cluster_id, components in clusters.items()
    }


def assert_rollback(
    stage: str,
    result: Mapping[str, Any],
    shape: Mapping[str, Any],
    *,
    label: str,
    require: Require,
) -> None:
    """The rollback's scope and timing, cluster by cluster."""

    components = _plan_components(result)
    if stage == "control_plane":
        require(
            not any(components.get(cluster_id) for cluster_id in shape),
            f"{label}: the CPU-only rollback restored a GPU cluster",
            {key: sorted(value) for key, value in components.items()},
        )
        return
    nodes, zero = _node_clusters(shape), _zero_node_clusters(shape)
    required = ROLLBACK_COMPONENTS[stage]
    for cluster_id in nodes:
        scope = components.get(cluster_id, set())
        require(
            required <= scope,
            f"{label}: the rollback of {cluster_id} (nodes) lacks "
            f"{sorted(required - scope)}",
            sorted(scope),
        )
        if stage == "executor":
            require(
                "agent" not in scope,
                f"{label}: the Executor-only rollback reinstalled the Agents of "
                f"{cluster_id}",
                sorted(scope),
            )
    for cluster_id in zero:
        scope = components.get(cluster_id, set())
        require(
            bool(scope) and all(scope == components.get(other) for other in nodes),
            f"{label}: {cluster_id} (no nodes) was not rolled back with the "
            "components the clusters with nodes got",
            {key: sorted(value) for key, value in components.items()},
        )
    timing = _mapping(_mapping(result.get("rollback_timing")).get("clusters"))
    for cluster_id in sorted(shape):
        entry = _mapping(timing.get(cluster_id))
        require(
            entry.get("status") == "COMPLETED",
            f"{label}: {cluster_id} has no completed rollback timing",
            dict(entry),
        )
        if stage in {"agent", "full"} and cluster_id in nodes:
            assert_waves(
                cluster_id,
                entry.get("waves"),
                list(shape[cluster_id][NODES]),
                label=label,
                require=require,
            )
        else:
            require(
                not entry.get("waves"),
                f"{label}: {cluster_id} rolled Fleet waves although "
                + (
                    "it has no node"
                    if cluster_id in zero
                    else "the Executor-only rollback restores no Agent"
                ),
                sorted(_mapping(entry.get("waves"))),
            )


def assert_records(
    stage: str,
    result: Mapping[str, Any],
    shape: Mapping[str, Any],
    *,
    phase: str | None,
    snapshot: Mapping[str, Any] | None,
    components: Mapping[str, set[str]],
    label: str,
    require: Require,
) -> None:
    """The engine's zero-node records before and after one deploy.

    A new transaction starts from an empty state (``upgrade_release``), so a
    deploy that does no node work on a cluster leaves it no record at all; a
    NOOP rewrites the state it loaded and carries the earlier one unchanged.
    Either way the cluster gains no record of its own.
    """

    after, before = result.get(RECORDS), result.get(RECORDS_BEFORE)
    require(
        isinstance(after, Mapping) and isinstance(before, Mapping),
        f"{label}: the deploy result carries no zero-node records",
        {"after": after, "before": before},
    )
    for cluster_id in sorted(shape):
        now, then = _mapping(after).get(cluster_id), _mapping(before).get(cluster_id)
        expected = phase
        if stage == "executor" and "reconciler" not in components.get(cluster_id, ()):
            expected = None  # no Reconciler in the plan: nothing rolls there
        if NODES in shape[cluster_id] or expected is None:
            require(
                now is None or now == then,
                f"{label}: the engine recorded node work for {cluster_id} in a deploy "
                + (
                    "although it has nodes"
                    if NODES in shape[cluster_id]
                    else "that does no GPU work"
                ),
                {"before": then, "after": now},
            )
            continue
        entry = _mapping(now)
        require(
            isinstance(now, Mapping) and now != then,
            f"{label}: the engine recorded no skipped node work for {cluster_id} "
            "in this deploy",
            {"before": then, "after": now},
        )
        require(
            entry.get("release_id") == result.get("release_id")
            and entry.get("phase") == expected
            and entry.get("state") == parked_cluster.PARKED
            and tuple(entry.get("skipped") or ()) == SKIPPED[stage],
            f"{label}: the zero-node record of {cluster_id} is not this release's "
            f"{expected} with the skipped node work {list(SKIPPED[stage])}",
            dict(entry),
        )
        capture = _mapping(_captures(_mapping(snapshot)).get(cluster_id))
        reconciler = _mapping(entry.get("reconciler"))
        require(
            bool(capture.get("bundle_sha256"))
            and reconciler.get("bundle_sha256") == capture.get("bundle_sha256")
            and reconciler.get("template_sha256") == capture.get("template_sha256"),
            f"{label}: the Reconciler recorded for {cluster_id} is not the one "
            "captured after it",
            {"recorded": dict(reconciler), "captured": dict(capture)},
        )


def _changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    return sorted(
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    )


def assert_identities(
    stage: str,
    stages: Mapping[str, Any],
    shape: Mapping[str, Any],
    *,
    require: Require,
) -> None:
    """What each stage did to the Agents and Reconcilers, cluster by cluster."""

    pair = IDENTITY_PAIRS.get(stage)
    if pair is None:
        return
    first, last = (_mapping(stages.get(key)) for key in pair)
    before, after = _identities(first), _identities(last)
    if stage == "restore":
        require(
            after == before and _captures(last) == _captures(first),
            f"restore: the Agent identities and GPU cluster captures are not back "
            f"to what {pair[0]} observed",
            {
                cluster_id: _changed(
                    _mapping(before.get(cluster_id)), _mapping(after.get(cluster_id))
                )
                for cluster_id in sorted(shape)
            },
        )
        return
    profiles = _mapping(stages.get(PROFILE_STAGE))
    for cluster_id, entry in sorted(shape.items()):
        old, new = _mapping(before.get(cluster_id)), _mapping(after.get(cluster_id))
        if NODES not in entry:
            if stage in {"agent", "full"}:
                bundle = (
                    _mapping(_captures(first).get(cluster_id)).get("bundle_sha256"),
                    _mapping(_captures(last).get(cluster_id)).get("bundle_sha256"),
                )
                require(
                    bundle[0] != bundle[1],
                    f"{stage}: the Reconciler of {cluster_id} (no nodes) did not "
                    "take the release's node runtime bundle",
                    bundle,
                )
            continue
        moved = _changed(old, new)
        if stage in {"control_plane", "executor"}:
            require(
                not moved,
                f"{stage}: the Agents of {cluster_id} changed in a release without "
                "a node runtime",
                moved,
            )
            continue
        require(
            set(NODE_RUNTIME_FIELDS) <= set(moved),
            f"{stage}: the Agents of {cluster_id} did not take the release's node "
            "runtime",
            moved,
        )
        if stage == "full":
            drill = _mapping(_mapping(profiles.get("clusters")).get(cluster_id))
            own = _mapping(_mapping(profiles.get("site_clusters")).get(cluster_id))
            require(
                bool(drill.get("version"))
                and drill.get("version") != own.get("version"),
                f"full: the FULL candidate changes no Runtime Profile of "
                f"{cluster_id}, which has nodes, so the change reaches none of its "
                "Agents; rebuild the candidates (boot020_release_candidates configs "
                "suffixes every cluster's own profile)",
                {"full": dict(drill), "site": dict(own)},
            )
            require(
                new.get("runtime_profile_version") == drill.get("version")
                and new.get("config_digest") == drill.get("agent_config_digest"),
                f"full: the Agents of {cluster_id} do not report the FULL Runtime "
                "Profile and Agent config digest",
                {"expected": dict(drill), "reported": dict(new)},
            )


def check_stage(recorder: Any, stage: str, *, require: Require) -> None:
    """Judge ``stage`` per cluster from its recorded evidence."""

    stages = _mapping(recorder.document.get("stages"))
    shape = recorded_shape(recorder.document)
    if stage == "full":
        require(
            isinstance(stages.get(PROFILE_STAGE), Mapping),
            f"full: the FULL stage recorded no {PROFILE_STAGE}",
        )
    for key in SNAPSHOTS.get(stage, ()):
        assert_snapshot_shape(stages.get(key), shape, label=key, require=require)
    components: dict[str, set[str]] = {}
    if stage in ROLLBACKS:
        result = _mapping(stages.get(ROLLBACKS[stage]))
        components = _plan_components(result)
        assert_rollback(stage, result, shape, label=ROLLBACKS[stage], require=require)
    for key, phase, snapshot_key in DEPLOYS.get(stage, ()):
        assert_records(
            stage,
            _mapping(stages.get(key)),
            shape,
            phase=phase,
            snapshot=_mapping(stages.get(snapshot_key)) if snapshot_key else None,
            components=components,
            label=key,
            require=require,
        )
    assert_identities(stage, stages, shape, require=require)


def evidence_summary(stages: Mapping[str, Any]) -> dict[str, Any]:
    """The site shape and the permanently registered profile versions, for the
    completed evidence's top level."""

    summary: dict[str, Any] = {}
    shape = _mapping(_mapping(stages.get(SHAPE_STAGE)).get("clusters"))
    if shape:
        summary["site_shape"] = {
            cluster_id: (
                {NODES: len(_mapping(entry)[NODES])}
                if NODES in _mapping(entry)
                else {
                    ZERO_NODES: _mapping(_mapping(entry).get(ZERO_NODES)).get("state")
                }
            )
            for cluster_id, entry in sorted(shape.items())
        }
        summary["zero_node_clusters"] = _zero_node_clusters(shape)
    profiles = _mapping(stages.get(PROFILE_STAGE))
    if profiles:
        summary["registered_runtime_profile_versions"] = list(
            profiles.get("registered_versions") or []
        )
    return summary
