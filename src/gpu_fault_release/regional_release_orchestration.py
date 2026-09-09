from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_dataplane_observability import (
    restore_observability_with_dataplane,
)
from gpu_fault_release.regional_release_automatic_rollback import (
    recover_failed_upgrade,
)
from gpu_fault_release.regional_release_config import (
    ClusterLocalReleaseError,
    ClusterTarget,
    PartialClusterRolloutError,
    ReleaseError,
    canonical_sha256,
)
from gpu_fault_release.regional_release_diff import (
    GPU_CLUSTER_COMPONENTS,
    ReleaseChangeKind,
    ReleaseComponent,
    ReleaseDiff,
    ReleaseExecutionPlan,
    build_execution_plan,
    control_plane_role_targets,
)
from gpu_fault_release.regional_release_legacy import (
    rollback_controller_config,
    validate_rollback_agent_identity,
)
from gpu_fault_release.regional_release_mutation_preflight import (
    preflight_upgrade_mutations,
)
from gpu_fault_release.regional_release_narration import narrate_phase, narrate_step
from gpu_fault_release.regional_release_progress import (
    PROGRESS_COMPLETED,
    PROGRESS_FAILED,
    PROGRESS_STARTED,
    RollbackCompensationPlan,
    build_rollback_compensation_plan,
    cluster_attempts_with,
    update_component_progress,
    update_components_progress,
)
from gpu_fault_release.regional_release_rollback_context import (
    # Re-exported: `rollout_regional_release` and the command tests reach this
    # through the orchestration module, which is the entry point that owns the
    # rollback flow even though the environment assembly now lives beside the
    # other rollback identity helpers.
    build_rollback_environment as build_rollback_environment,
)
from gpu_fault_release.regional_release_rollback_context import (
    previous_container_env_snapshot,
    write_rollback_container_env,
)
from gpu_fault_release.regional_release_rollback_context import (
    rollback_identity_context as _rollback_identity_context,
)
from gpu_fault_release.regional_release_rollback_context import (
    rollback_target_arguments as _rollback_target_arguments,
)
from gpu_fault_release.regional_schema_change import (
    SCHEMA_CHANGE_ACCEPTANCE_KEY,
    ensure_schema_change_snapshot,
    recorded_acceptance,
    resolve_acceptance,
)
from gpu_fault_release.regional_release_rollout_cleanup import (
    # Re-exported for the same reason, and bound as module globals rather than
    # called through the source module so a test that patches
    # `ORCHESTRATION.cleanup_candidate_rollout_state` still intercepts the call
    # `rollback_release` makes below.
    cleanup_candidate_rollout_state as cleanup_candidate_rollout_state,
)
from gpu_fault_release.regional_release_rollout_cleanup import (
    terminalize_stranded_rollouts as terminalize_stranded_rollouts,
)
from gpu_fault_release.regional_release_state import state_transaction
from gpu_fault_release.regional_release_timing import (
    complete_timed_entry,
    initialize_rollback_timing,
    mark_rollback_safe,
    merge_rollback_wave_timings,
    start_timed_entry,
)
from gpu_fault_release.regional_release_transaction import complete_rollback
from gpu_fault_release.regional_release_validation import validate_gpu_rollback_target
from gpu_fault_release.regional_runtime_profile import ensure_runtime_profile

from gpu_fault.admin.config import AdminConfig

ROOT = Path(__file__).resolve().parents[2]
# Changes whose previous state nothing captures, so a rollback that claimed to
# restore them would be reporting a state it never put back. The ADOT manifest
# and image used to be here; the observability snapshot now carries the live
# collector objects, so they are compensated like the AMP blobs beside them
# (`regional_observability_rollback`). `endpoint` and `endpoint_manifests` have
# now left too: the endpoint snapshot carries the live NLB Service and the
# Route53 record the candidate overwrites (`regional_endpoint_rollback`).
#
# `clusters` -- the cluster-registry digest -- stays. It is not a manifest that
# can be captured and re-applied: adding or removing a GPU cluster fans out into
# the registry, the control-plane role config, and per-cluster state in a cluster
# that may no longer be reachable, and a release that removed a cluster has
# nothing left to roll back *to* in it. Compensating that is a separate design,
# not a snapshot.
NON_TRANSACTIONAL_CHANGES = frozenset({"clusters"})
# The terminal phases of a fail-forward transaction that stopped: nothing is
# still moving, nothing rolled back, the site is a mixture of the failed
# candidate and the last committed release. Only a transaction in one of these
# phases may be superseded by a different candidate (`upgrade_release` with
# `supersede=`); every other phase either resumes the same release or continues
# a rollback.
SUPERSEDABLE_PHASES = frozenset({"failed", "partial-convergence"})
# The release-state key that records the transaction a new one took over from.
SUPERSEDED_TRANSACTION_KEY = "superseded_transaction"


def _apply_rollback_cpu_environment(
    self: Any,
    environment: dict[str, str],
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="gpu-fault-rollback-role-split-"
    ) as directory:
        generated = Path(directory)
        render_environment = {
            **environment,
            "GPU_FAULT_ROLE_SPLIT_OUT_DIR": str(generated),
        }
        self.runner.run(
            [
                "bash",
                str(
                    ROOT / "deploy/control-plane/tools/"
                    "render-control-plane-role-split.sh"
                ),
            ],
            env=render_environment,
        )
        apply_environment = {
            **environment,
            "GPU_FAULT_ROLE_SPLIT_GENERATED_DIR": str(generated),
        }
        self.runner.run(
            [
                "bash",
                str(
                    ROOT / "deploy/control-plane/tools/"
                    "apply-control-plane-role-split.sh"
                ),
            ],
            env=apply_environment,
        )


def _default_release_diff() -> ReleaseDiff:
    return ReleaseDiff(
        kind=ReleaseChangeKind.FULL,
        changed=frozenset(
            {
                "control_plane_wheel",
                "executor_wheel",
                "node_runtime_wheel",
                "node_bundle",
                "database_schema",
                "agent_protocol",
                "executor_protocol",
                "agent_config",
                "runtime_profile",
                "endpoint",
                "dcgm",
            }
        ),
    )


def _validate_upgrade_transaction(
    self: Any,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
    *,
    resume: bool = False,
    supersede: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Refuse before mutation, or return the schema-change acceptance to record.

    A schema change under ``autoRollback: true`` used to be refused outright;
    the operator had to edit the site to fail-forward. It is now refused unless
    the command carried ``--accept-schema-change``, in which case the acceptance
    (mode, time, later the snapshot) is returned for the transaction state and
    the transaction runs fail-forward without touching the site
    (``regional_schema_change``).

    A transaction that supersedes a failed one inherits that transaction's
    acceptance when it covers the same schema version: the schema already moved
    and cannot be un-moved, so re-asking would only teach operators to pass the
    flag by habit. A candidate that moves the schema again is a new schema
    change and needs its own acceptance.
    """

    acceptance = resolve_acceptance(
        self,
        changed=diff.changed,
        resume=resume,
        inherited=recorded_acceptance(supersede) if supersede is not None else None,
    )
    non_transactional = diff.changed.intersection(NON_TRANSACTIONAL_CHANGES)
    if self.config.auto_rollback and non_transactional:
        raise ReleaseError(
            "automatic rollback is not yet transactional for: "
            + ", ".join(sorted(non_transactional))
        )
    return acceptance


def _require_compensable_observability_snapshot(
    self: Any,
    previous: dict[str, Any],
) -> None:
    """Refuse before mutation when the snapshot cannot restore the collector.

    A snapshot captured by a release engine that predates ADOT compensation has
    no collector objects, and a resumed transaction keeps the snapshot it was
    started with. Automatic rollback would then get as far as the observability
    restore and fail there, after it had already put other components back. The
    operator is told the same thing the other non-transactional changes tell
    them, at the same point: before anything moves.
    """

    observability = previous.get("observability")
    if not self.config.auto_rollback or not isinstance(observability, dict):
        return
    if "adot" not in observability:
        raise ReleaseError(
            "previous observability snapshot predates ADOT rollback capture; "
            "deploy this release with spec.autoRollback: false"
        )


def _require_compensable_endpoint_snapshot(
    self: Any,
    previous: dict[str, Any],
    plan: ReleaseExecutionPlan,
) -> None:
    """Refuse before mutation when the snapshot cannot restore the endpoint.

    Same failure mode as the observability guard, and the same reason to check it
    here: a snapshot taken by an engine that predates endpoint capture has no
    ``endpoint`` key at all, and a resumed transaction keeps the snapshot it was
    started with. Rollback would then reach the endpoint restore -- after the
    data plane and the control plane had already been put back -- and fail with
    the Route53 record still pointing at the candidate load balancer.

    A present ``endpoint`` of ``None`` is not that case: it is what a plan
    without the endpoint component records, and such a plan has no endpoint
    mutation to compensate.
    """

    if not self.config.auto_rollback or not plan.has(ReleaseComponent.ENDPOINT):
        return
    if "endpoint" not in previous:
        raise ReleaseError(
            "previous snapshot predates endpoint rollback capture; "
            "deploy this release with spec.autoRollback: false"
        )


def inherit_superseded_previous(
    self: Any,
    failed: dict[str, Any],
) -> dict[str, Any]:
    """The baseline a superseding transaction rolls back to: the failed one's.

    A failed fail-forward transaction left the site as a mixture of the failed
    candidate and the last committed release. Capturing a fresh ``previous``
    from that mixture would make it the rollback target of the next transaction,
    so an automatic rollback would "restore" half of a release that never
    committed. The committed release is exactly what the failed transaction
    captured as its own ``previous`` -- including the Secret backups it took,
    which are still in place because only a commit or a rollback deletes them --
    so the new transaction inherits that snapshot instead. Every check the
    inherited snapshot has to pass is checked here, before anything moves.
    """

    failed_id = str(failed.get("release_id") or "")
    phase = str(failed.get("phase") or "")
    if phase not in SUPERSEDABLE_PHASES:
        raise ReleaseError(
            f"only a transaction in {'/'.join(sorted(SUPERSEDABLE_PHASES))} can be "
            f"superseded; release {failed_id or 'unknown'} is in phase "
            f"{phase or 'unknown'}"
        )
    if failed_id == str(self.release_id):
        raise ReleaseError(
            f"release {failed_id} is the failed candidate itself; "
            "rerun deploy without the supersede flag to resume it"
        )
    previous = failed.get("previous")
    if not isinstance(previous, dict) or not previous:
        raise ReleaseError(
            f"failed release {failed_id} has no previous baseline to inherit"
        )
    expected_sha = str(failed.get("previous_snapshot_sha256") or "")
    if not expected_sha:
        raise ReleaseError("superseded previous baseline digest is missing")
    if canonical_sha256(previous) != expected_sha:
        raise ReleaseError("superseded previous baseline digest drifted")
    if not isinstance(previous.get("secret_backups"), dict):
        raise ReleaseError(
            f"failed release {failed_id} recorded no Secret backups to inherit"
        )
    expected_ids = {target.cluster_id for target in self.config.clusters}
    if set(failed.get("cluster_ids") or []) != expected_ids:
        raise ReleaseError("supersede cluster membership drifted")
    inherited: dict[str, Any] = json.loads(json.dumps(previous))
    return inherited


def superseded_transaction_record(failed: dict[str, Any]) -> dict[str, Any]:
    """What the new transaction records about the one it took over from."""

    record: dict[str, Any] = {
        "release_id": failed.get("release_id"),
        "phase": failed.get("phase"),
        "superseded_at": datetime.now(UTC).isoformat(),
    }
    for name in (
        "updated_at_epoch",
        "original_failure",
        "release_lifecycle",
        "release_diff",
        "completed_phases",
        "completed_cluster_ids",
        SCHEMA_CHANGE_ACCEPTANCE_KEY,
    ):
        if failed.get(name) is not None:
            record[name] = failed[name]
    return record


def _upgrade_context(
    self: Any,
    *,
    resume: bool,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
    supersede: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], set[str], set[str], bool]:
    loaded = self._load_state() if resume else {}
    if resume:
        if loaded.get("release_id") != self.release_id:
            raise ReleaseError("resume release_id does not match the candidate")
        if loaded.get("release_diff") != diff.as_dict():
            raise ReleaseError("resume release diff does not match the candidate")
        if loaded.get("execution_plan") != plan.as_dict():
            raise ReleaseError("resume execution plan does not match the candidate")
        self.state = dict(loaded)
        previous = loaded.get("previous")
        completed_phases = set(loaded.get("completed_phases") or [])
        completed_clusters = set(loaded.get("completed_cluster_ids") or [])
        registry_staged = bool(loaded.get("registry_staged", False))
    else:
        if supersede is not None:
            previous = inherit_superseded_previous(self, supersede)
        else:
            previous = self._capture_previous(plan=plan)
            previous["secret_backups"] = self._backup_release_secrets()
        completed_phases = set()
        completed_clusters = set()
        registry_staged = False
    if not isinstance(previous, dict) or not previous:
        raise ReleaseError("previous release state is unavailable")
    _require_compensable_observability_snapshot(self, previous)
    _require_compensable_endpoint_snapshot(self, previous, plan)
    if resume:
        expected_previous_sha = str(loaded.get("previous_snapshot_sha256") or "")
        if not expected_previous_sha:
            raise ReleaseError("resume previous baseline digest is missing")
        if canonical_sha256(previous) != expected_previous_sha:
            raise ReleaseError("resume previous baseline digest drifted")
        self._validate_resume_checkpoint(
            loaded=loaded,
            previous=previous,
            plan=plan,
        )
    return (
        previous,
        completed_phases,
        completed_clusters,
        registry_staged,
    )


def upgrade_gpu_clusters(
    self: Any,
    *,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
    previous: dict[str, Any],
    completed_phases: set[str],
    completed_clusters: set[str],
    registry_staged: bool,
) -> None:
    if not plan.has(*GPU_CLUSTER_COMPONENTS):
        return
    pending_targets = [
        target
        for target in self.config.clusters
        if target.cluster_id not in completed_clusters
    ]
    if not pending_targets:
        return

    def record_progress(
        cluster_id: str,
        selection: ReleaseComponent | tuple[ReleaseComponent, ...],
        status: str,
        details: dict[str, Any] | None,
    ) -> None:
        components = selection if isinstance(selection, tuple) else (selection,)
        # Deriving the new progress tree from `self.state` and checkpointing it
        # is one read-modify-write: clusters rolling in parallel each contribute
        # their own subtree and must not observe a half-applied one.
        with state_transaction(self):
            progress = update_components_progress(
                self.state,
                components,
                status,
                cluster_id=cluster_id,
                details=details,
            )
            self.state["component_progress"] = progress
            if status != PROGRESS_COMPLETED:
                self._save_state(
                    str(self.state.get("phase") or "preflight"),
                    component_progress=progress,
                )

    def cluster_checkpoint(
        phase: str,
        cluster_id: str,
        lifecycle: str,
        *,
        fields: dict[str, Any] | None = None,
    ) -> None:
        with state_transaction(self):
            self._save_state(
                phase,
                previous=previous,
                release_diff=diff.as_dict(),
                execution_plan=plan.as_dict(),
                completed_phases=sorted(completed_phases),
                completed_cluster_ids=sorted(completed_clusters),
                cluster_attempts=cluster_attempts_with(
                    self.state,
                    cluster_id,
                    lifecycle,
                    fields=fields,
                ),
                release_lifecycle="ROLLING_CLUSTERS",
                registry_staged=registry_staged,
            )

    started: set[str] = set()
    failures: dict[str, Exception] = {}
    # A failure stops new clusters from starting, but a cluster already rolling
    # its node fleet must be allowed to finish its wave sequence: killing it
    # mid-wave would leave the fleet in a state no checkpoint describes.
    abort = threading.Event()

    def roll_cluster(target: ClusterTarget) -> None:
        cluster_id = target.cluster_id
        with state_transaction(self):
            if abort.is_set():
                return
            started.add(cluster_id)
            cluster_checkpoint("data-plane-progress", cluster_id, "RUNNING")
        try:
            self._upgrade_gpu_target(
                target,
                diff,
                plan,
                progress=(
                    lambda component,
                    status,
                    details,
                    active_cluster_id=cluster_id: record_progress(
                        active_cluster_id,
                        component,
                        status,
                        details,
                    )
                ),
                candidate_preflighted=True,
            )
        except Exception as exc:
            abort.set()
            with state_transaction(self):
                failures[cluster_id] = exc
            return
        with state_transaction(self):
            completed_clusters.add(cluster_id)
            cluster_checkpoint(
                "data-plane-progress",
                cluster_id,
                "CONVERGED",
                # When this cluster stopped being restarted, for the readers that
                # have to excuse restart-shaped alerts for a bounded window.
                # An observability-only plan (OBSERVABILITY is a GPU-cluster
                # component since F10) lands here too, so its clusters read
                # CONVERGED with a fresh `converged_at_epoch` although no Agent
                # or Deployment rolled: the grace window it opens excuses
                # nothing that happened, and the stamp is not evidence of a
                # data-plane roll; the cluster's progress record carries only
                # the OBSERVABILITY component for such a release, which is the
                # honest reading. Recorded, not changed (F10 fix 1, F5).
                # `identity_verified` stays false on purpose: convergence here
                # compares the artifact and config digests, not the protocol
                # version or the compatibility digest, so nothing yet proves the
                # full candidate pin -- only the pre-finalize heartbeat barrier
                # does, and it is still the gate that must run.
                fields={
                    "converged_at_epoch": time.time(),
                    "identity_verified": False,
                },
            )

    workers = min(
        self.config.upgrade_max_parallel_clusters,
        len(pending_targets),
    )
    if workers > 1:
        # Canary first, in site order. A failure the engine cannot attribute to
        # one cluster is a *global* failure, and rolling that back reverts every
        # cluster that already converged -- so the first cluster proves the
        # candidate alone, and only then does the fleet overlap. The canary is
        # the same cluster a serial rollout would have done first, so this costs
        # nothing a serial release did not already pay.
        canary, remaining = pending_targets[0], pending_targets[1:]
        roll_cluster(canary)
        if remaining and canary.cluster_id in completed_clusters:
            with ThreadPoolExecutor(
                max_workers=min(workers, len(remaining))
            ) as executor:
                for future in as_completed(
                    [executor.submit(roll_cluster, target) for target in remaining]
                ):
                    future.result()
    else:
        for target in pending_targets:
            roll_cluster(target)
    if not failures:
        return
    local = all(isinstance(exc, ClusterLocalReleaseError) for exc in failures.values())
    failed_cluster_ids = [
        target.cluster_id for target in pending_targets if target.cluster_id in failures
    ]
    not_started_cluster_ids = [
        target.cluster_id
        for target in pending_targets
        if target.cluster_id not in started
    ]
    with state_transaction(self):
        for failed_cluster_id in failed_cluster_ids:
            failure = failures[failed_cluster_id]
            self.state["cluster_attempts"] = cluster_attempts_with(
                self.state,
                failed_cluster_id,
                "FAILED",
                details={"error": f"{type(failure).__name__}: {failure}"},
            )
        attempts = self.state["cluster_attempts"]
        self._save_state(
            "data-plane-paused",
            previous=previous,
            release_diff=diff.as_dict(),
            execution_plan=plan.as_dict(),
            completed_phases=sorted(completed_phases),
            completed_cluster_ids=sorted(completed_clusters),
            failed_cluster_ids=failed_cluster_ids,
            paused_cluster_ids=[],
            not_started_cluster_ids=not_started_cluster_ids,
            failure_scope="cluster-local" if local else "global",
            cluster_attempts=attempts,
            release_lifecycle="PAUSED" if local else "FAILED",
            registry_staged=registry_staged,
        )
    cluster_id = failed_cluster_ids[0]
    error = failures[cluster_id]
    message = (
        f"{cluster_id} rollout failed; not_started={not_started_cluster_ids}: {error}"
    )
    if len(failed_cluster_ids) > 1:
        message += "; also failed: " + ", ".join(failed_cluster_ids[1:])
    error_type = PartialClusterRolloutError if local else ReleaseError
    raise error_type(message) from error


def bootstrap_gpu_target(self: Any, target: ClusterTarget) -> None:
    self._ensure_gpu_namespace(target)
    self._ensure_connection_secret(target)
    self._quiesce_gpu_executor(target)
    self._verify_gpu_control_plane_endpoint(target)
    self._apply_gpu_dcgm_exporter(target)
    self._apply_gpu_adot_collector(target)
    self._apply_gpu_deployments(target, self.executor_wheel_cm)
    self._roll_node_runtime(
        target,
        phase="bootstrap",
        wheel_cm=self.executor_wheel_cm,
        bundle_cm=self.bundle_cm,
        artifact_sha=self.node_wheel_sha,
        config_digest=self.config.agent_config_digest,
    )


def bootstrap_gpu_clusters(
    self: Any,
    completed_cluster_ids: set[str],
    *,
    bootstrap: Callable[[Any, ClusterTarget], None] = bootstrap_gpu_target,
) -> None:
    pending = [
        target
        for target in self.config.clusters
        if target.cluster_id not in completed_cluster_ids
    ]
    if not pending:
        return
    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=min(4, len(pending))) as executor:
        futures = {
            executor.submit(bootstrap, self, target): target.cluster_id
            for target in pending
        }
        for future in as_completed(futures):
            cluster_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((cluster_id, exc))
                continue
            completed_cluster_ids.add(cluster_id)
            self._save_state(
                "bootstrap-data-plane-progress",
                previous=None,
                completed_cluster_ids=sorted(completed_cluster_ids),
            )
    if failures:
        names = ", ".join(sorted(cluster_id for cluster_id, _exc in failures))
        first_cluster, first_error = failures[0]
        raise ReleaseError(
            f"GPU bootstrap failed for {names}; "
            f"first observed cluster {first_cluster}: {first_error}"
        ) from first_error


# The order the upgrade completes its phases in, used to stamp a state write
# that carries more than one completed phase with the furthest one it proves.
# The stamp is what `regional_admin_commands.RESUMABLE_PHASES` reads, so it has
# to name the phase an operator would resume from, not the first phase that
# happened to finish.
UPGRADE_PHASE_ORDER = (
    "uploaded",
    "candidate-preflight-ready",
    "schema-ready",
    "registry-staged",
    "cpu-staged",
    "profile-ready",
    "endpoint-ready",
    "observability-ready",
    "data-converged",
    "cpu-finalized",
    "verified",
    "complete",
)
# One worker per phase that may run beside the others: the candidate node
# preflight, the control-plane endpoint, and the observability install. Each
# touches a cluster or an AWS resource none of the others do, and each is joined
# at the phase that genuinely depends on it. The pool is sized so no submit can
# ever queue behind a peer -- a queued future joined before its queue-mate
# finishes would be a deadlock that only shows up under a slow phase.
UPGRADE_PHASE_WORKERS = 3
# The component whose work a phase checkpoint attests to. A release whose plan
# omits the component has nothing to do for the phase and writes no checkpoint
# for it, so every downstream gate has to read "not planned" as satisfied
# (`phase_done`) rather than waiting for a stamp that will never arrive.
PHASE_COMPONENT_GATES = {
    "schema-ready": ReleaseComponent.SCHEMA,
    "registry-staged": ReleaseComponent.REGISTRY,
    "cpu-staged": ReleaseComponent.CPU_STAGE,
    "profile-ready": ReleaseComponent.RUNTIME_PROFILE,
    "endpoint-ready": ReleaseComponent.ENDPOINT,
    "observability-ready": ReleaseComponent.OBSERVABILITY,
    "cpu-finalized": ReleaseComponent.CPU_FINALIZE,
}


# The names the background phases are tracked and narrated under. They are not
# checkpoint phases: `candidate-preflight-ready` is written when the future is
# joined, not when it is submitted.
PHASE_CANDIDATE_PREFLIGHT = "candidate-preflight"
PHASE_ENDPOINT = "endpoint"
PHASE_OBSERVABILITY = "observability"


def _annotate_with_background_failures(
    error: BaseException,
    notes: list[str],
) -> None:
    """Fold background-phase failures into the error the caller will report.

    Appended to the same exception object rather than raised in its place:
    `upgrade_release` decides between a pause and a full rollback from the
    error's *type*, and a cluster-local pause must not become a global rollback
    just because a phase running beside it also died.
    """

    if not notes:
        return
    suffix = "; ".join(notes)
    head = str(error.args[0]) if error.args else ""
    error.args = (f"{head}; {suffix}" if head else suffix,) + tuple(error.args[1:])


def run_upgrade_phases(
    self: Any,
    *,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
    previous: dict[str, Any],
    completed_phases: set[str],
    completed_clusters: set[str],
    registry_staged: bool,
) -> bool:
    # Phases that have completed but are not in the ConfigMap yet, oldest first.
    # Nothing is allowed to depend on a phase being *durable* until the next
    # state write returns, which is why every mutation is followed by one.
    pending_phases: dict[str, dict[str, Any]] = {}

    def phase_done(name: str) -> bool:
        """True when this release has nothing left to do for `name`."""

        if name in completed_phases:
            return True
        gate = PHASE_COMPONENT_GATES.get(name)
        return gate is not None and not plan.has(gate)

    def write_state(stamp: str, **updates: Any) -> None:
        merged: dict[str, Any] = {}
        carried = sorted(
            (
                phase
                for phase in pending_phases
                if phase != stamp and phase in UPGRADE_PHASE_ORDER
            ),
            key=UPGRADE_PHASE_ORDER.index,
        )
        for values in pending_phases.values():
            merged.update(values)
        pending_phases.clear()
        merged.update(updates)
        self._save_state(
            stamp,
            previous=previous,
            release_diff=diff.as_dict(),
            execution_plan=plan.as_dict(),
            completed_phases=sorted(completed_phases),
            completed_cluster_ids=sorted(completed_clusters),
            registry_staged=registry_staged,
            **merged,
        )
        # `save_state` narrates the stamp and nothing else, so the phases this
        # one write also made durable would have no line at all -- and those are
        # exactly the phases an operator reads the log to time. They follow the
        # stamp because the write is what makes them true, and they carry
        # `elapsed=0.0` for the same reason: the interval they were reached in is
        # charged to the checkpoint that persisted all of them.
        for phase in carried:
            narrate_phase(self, phase)

    def phase_complete(name: str, **updates: Any) -> None:
        """Mark `name` complete; the next state write is what persists it."""

        completed_phases.add(name)
        pending_phases[name] = updates

    def flush_phases(**updates: Any) -> None:
        with state_transaction(self):
            current = str(self.state.get("phase") or "")
            # The phase already on record counts towards the stamp, not just the
            # phases this write carries. Without it a write whose only pending
            # phase is an early one (the candidate preflight, joined late) would
            # move the recorded phase *backwards*, and a phase outside
            # `RESUMABLE_PHASES` -- or simply an earlier one -- makes a crash
            # here look like a release that must start over.
            ranked = [
                phase
                for phase in {*pending_phases, current}
                if phase in UPGRADE_PHASE_ORDER
            ]
            stamp = (
                max(ranked, key=UPGRADE_PHASE_ORDER.index)
                if ranked
                else (current or "preflight")
            )
            write_state(stamp, **updates)

    def checkpoint(phase: str, **updates: Any) -> None:
        with state_transaction(self):
            phase_complete(phase)
            flush_phases(**updates)

    def mark_component(component: ReleaseComponent, status: str) -> dict[str, Any]:
        progress = update_component_progress(self.state, component, status)
        self.state["component_progress"] = progress
        return progress

    def start_component(component: ReleaseComponent) -> None:
        # One write says "the phases before this are complete" and "this
        # component has started". Both describe the same instant, and a crash
        # between them could not be told apart from a crash after them, so
        # splitting the write bought nothing but another ConfigMap apply.
        with state_transaction(self):
            flush_phases(component_progress=mark_component(component, PROGRESS_STARTED))

    def fail_component(component: ReleaseComponent) -> None:
        with state_transaction(self):
            flush_phases(component_progress=mark_component(component, PROGRESS_FAILED))

    def finish_component(component: ReleaseComponent) -> None:
        with state_transaction(self):
            mark_component(component, PROGRESS_COMPLETED)

    # The phases running in the pool, by name, until the main thread joins one.
    # A future nobody joins is a failure nobody sees: its exception is never
    # retrieved, so it never reaches the log or the failure record.
    pending_futures: dict[str, Future[Any]] = {}

    def submit_phase(name: str, action: Callable[..., Any], *args: Any) -> None:
        pending_futures[name] = pool.submit(action, *args)

    def join_phase(name: str) -> None:
        pending_futures.pop(name).result()

    def drain_background_phases() -> list[str]:
        """Wait out the phases still in the pool and report what they raised.

        Called only on the failure path, for two reasons: a rollback must not run
        while a mutation from this transaction is still in flight, and the
        exception of an unjoined future would otherwise be discarded with the
        pool -- unlogged, and absent from the error the operator is handed.
        """

        notes: list[str] = []
        for name in list(pending_futures):
            future = pending_futures.pop(name)
            try:
                future.result()
            except Exception as exc:
                narrate_step(
                    "release-phase-failed",
                    phase=name,
                    error=f"{type(exc).__name__}: {exc}",
                )
                notes.append(f"{name} also failed: {type(exc).__name__}: {exc}")
        return notes

    def join_component(
        component: ReleaseComponent,
        name: str,
        completes_phase: str,
        **phase_updates: Any,
    ) -> None:
        """Wait for a component that ran in the pool and record its outcome.

        The STARTED marker was written before the future was submitted, so a
        crash while it ran is already compensated by the rollback plan; this is
        where the same component becomes COMPLETED or FAILED, on the main thread,
        inside the transaction that is about to depend on it.
        """

        try:
            join_phase(name)
        except Exception:
            fail_component(component)
            raise
        finish_component(component)
        phase_complete(completes_phase, **phase_updates)

    def run_component(component: ReleaseComponent, action: Callable[[], Any]) -> Any:
        start_component(component)
        try:
            result = action()
        except Exception:
            fail_component(component)
            raise
        finish_component(component)
        return result

    # A phase that runs in the pool never writes state from its own thread:
    # it is joined at the phase that depends on it, and the join is what
    # checkpoints it. Leaving the pool waits for whatever is still running, so
    # a failure on the main thread cannot be recorded while a mutation from
    # this transaction is still in flight.
    with ThreadPoolExecutor(
        max_workers=UPGRADE_PHASE_WORKERS,
        thread_name_prefix="gpu-fault-release-phase",
    ) as pool:
        try:
            if not phase_done("uploaded"):
                self._upload_release(diff)
                checkpoint("uploaded")
            if not phase_done("data-converged"):
                # Read-only Jobs on GPU nodes; nothing the control-plane phases
                # below touch, and nothing that writes release state.
                submit_phase(
                    PHASE_CANDIDATE_PREFLIGHT, preflight_upgrade_mutations, self, plan
                )
            if not phase_done("schema-ready"):
                # The one backward path of an accepted schema change is a
                # database restore, so the snapshot is taken here, before the
                # schema Jobs, under the SCHEMA component: a failure to take it
                # stops the release before the database has changed.
                acceptance = recorded_acceptance(self.state)
                if acceptance is not None:
                    acceptance = run_component(
                        ReleaseComponent.SCHEMA,
                        lambda: ensure_schema_change_snapshot(self, acceptance),
                    )
                    self.state[SCHEMA_CHANGE_ACCEPTANCE_KEY] = acceptance
                run_component(ReleaseComponent.SCHEMA, self._ensure_schema)
                phase_complete(
                    "schema-ready",
                    **(
                        {SCHEMA_CHANGE_ACCEPTANCE_KEY: acceptance}
                        if acceptance is not None
                        else {}
                    ),
                )
            if not phase_done("registry-staged"):

                def stage_registry_durably() -> bool:
                    # Staging the Secret alone is invisible to running Pods,
                    # which serve the durable head; publish there too and wait
                    # for the fleet to converge, as join/remove do (H3).
                    staged = bool(self._stage_registry())
                    if staged:
                        self._publish_staged_registry()
                    return staged

                registry_staged = bool(
                    run_component(ReleaseComponent.REGISTRY, stage_registry_durably)
                )
                phase_complete("registry-staged")
            if not phase_done("cpu-staged"):

                def apply_staged_cpu() -> None:
                    ingress_rollout = "ingress" in control_plane_role_targets(diff)
                    expected_agents = (
                        previous.get("agent_identities")
                        or self._capture_active_agent_node_sets()
                        if ingress_rollout
                        else {}
                    )
                    # One apply, every role, on the main thread. The role split
                    # cannot be invoked per role while the image changes: its
                    # post-apply verifier checks all three tiers against the
                    # candidate image and compares the ingress and spool
                    # ConfigMaps against each other, and the first invocation
                    # consumes the pin-metadata change that tells the *other*
                    # roles to restart onto the new compatibility window.
                    self._apply_cpu(
                        finalize=False,
                        force_restart=registry_staged,
                        diff=diff,
                    )
                    if ingress_rollout:
                        self._wait_candidate_cpu_agent_heartbeats(expected_agents)

                run_component(ReleaseComponent.CPU_STAGE, apply_staged_cpu)
                phase_complete("cpu-staged", release_lifecycle="CPU_STAGED")
            if not phase_done("profile-ready"):
                run_component(
                    ReleaseComponent.RUNTIME_PROFILE,
                    lambda: ensure_runtime_profile(self),
                )
                phase_complete("profile-ready")
            if not phase_done("endpoint-ready"):
                # Submitted only once the staged apply and its heartbeat barrier
                # have returned: the endpoint wait counts *healthy NLB targets*,
                # and the ingress Pods are cycling for the whole staged restart,
                # so overlapping the two would let the wait pass on a target set
                # that is about to be replaced. What it does overlap is whatever
                # is left of the candidate node preflight, and the data plane
                # still waits for it -- every cluster verifies the control-plane
                # endpoint before it is mutated.
                start_component(ReleaseComponent.ENDPOINT)
                submit_phase(PHASE_ENDPOINT, self._apply_nlb)
            if not phase_done("observability-ready"):
                # Nothing in the rollout reads what the monitoring install
                # writes, so it rolls beside the clusters; it is joined before
                # the finalize, which is the step no rollback can undo.
                start_component(ReleaseComponent.OBSERVABILITY)
                submit_phase(PHASE_OBSERVABILITY, self._apply_observability)
            if PHASE_ENDPOINT in pending_futures:
                join_component(
                    ReleaseComponent.ENDPOINT, PHASE_ENDPOINT, "endpoint-ready"
                )
            if PHASE_CANDIDATE_PREFLIGHT in pending_futures:
                # The join, not the submit, is the gate: no GPU cluster may be
                # mutated until the candidate has been proven usable on its nodes.
                join_phase(PHASE_CANDIDATE_PREFLIGHT)
                if not phase_done("candidate-preflight-ready"):
                    phase_complete("candidate-preflight-ready")
            if not phase_done("data-converged"):
                flush_phases(release_lifecycle="ROLLING_CLUSTERS")
            upgrade_gpu_clusters(
                self,
                diff=diff,
                plan=plan,
                previous=previous,
                completed_phases=completed_phases,
                completed_clusters=completed_clusters,
                registry_staged=registry_staged,
            )
            if PHASE_OBSERVABILITY in pending_futures:
                join_component(
                    ReleaseComponent.OBSERVABILITY,
                    PHASE_OBSERVABILITY,
                    "observability-ready",
                )
            if not phase_done("data-converged"):
                phase_complete("data-converged", release_lifecycle="FINALIZING")
            if not phase_done("cpu-finalized"):
                if plan.has(ReleaseComponent.RUNTIME_PROFILE):
                    self._ensure_profile_transition_safe(
                        previous.get("runtime_profile_version")
                    )

                def apply_final_cpu() -> None:
                    if "ingress" not in control_plane_role_targets(diff):
                        self._apply_cpu(finalize=True, diff=diff)
                        return
                    expected_agents = (
                        previous.get("agent_identities")
                        or self._capture_active_agent_node_sets()
                    )
                    # Closing the compatibility window is the transaction's one
                    # irreversible step: afterwards the control plane rejects every Agent
                    # still on the previous pin, and neither rollback nor a staged resume
                    # can reopen it. So prove the whole fleet already reports the
                    # candidate identity *before* promoting, the way
                    # deploy/hyperpod/deploy.sh does. A failure here leaves the window
                    # open and the transaction still compensable.
                    self._wait_candidate_cpu_agent_heartbeats(
                        expected_agents,
                        required_identity=self._candidate_agent_pin_identity(),
                    )
                    self._apply_cpu(finalize=True, diff=diff)
                    # And prove the cutover itself did not cost us the fleet.
                    self._wait_candidate_cpu_agent_heartbeats(expected_agents)

                run_component(
                    ReleaseComponent.CPU_FINALIZE,
                    apply_final_cpu,
                )
                phase_complete("cpu-finalized", release_lifecycle="FINALIZING")
            if not phase_done("verified"):
                run_component(
                    ReleaseComponent.VERIFY,
                    lambda: self._validate_release_quick(plan),
                )
                phase_complete("verified")
            if registry_staged:
                self._commit_registry_update()
            checkpoint("complete", release_lifecycle="COMMITTED")
        except Exception as error:
            _annotate_with_background_failures(error, drain_background_phases())
            raise
    return registry_staged


def upgrade_release(
    self: Any,
    *,
    resume: bool = False,
    diff: ReleaseDiff | None = None,
    supersede: dict[str, Any] | None = None,
) -> None:
    """Run one upgrade transaction: new, resumed, or superseding a failed one.

    ``supersede`` is the loaded state of a fail-forward transaction that
    stopped in ``failed``/``partial-convergence`` and whose candidate is not
    this release. The new transaction then starts from that transaction's
    ``previous`` (the last committed release, see
    ``inherit_superseded_previous``) instead of capturing the mixed live state,
    records the transaction it replaced under ``superseded_transaction``, and
    carries the earlier schema-change acceptance when it still applies. It is
    otherwise an ordinary new transaction: fresh checkpoints, ``autoRollback``
    from the site, the manifest plan digest originated here (M-23) rather than
    pinned to the failed transaction's.
    """

    if resume and supersede is not None:
        raise ReleaseError("a superseding transaction cannot also be a resume")
    self._ensure_contexts()
    self._require_cpu_secrets()
    # Ensure the RDS CA bundle ConfigMap exists (M-6) before re-applying the
    # control-plane roles (or a schema-ensure Job) that mount it verify-full;
    # on the first upgrade after this change an existing cluster has no bundle
    # yet, so an un-wired apply would wedge in CreateContainerConfigError.
    self._apply_rds_ca_bundle()
    # Copy the current Aurora password into the Secret before anything restarts
    # a Pod against it; fail closed. A rotation that the refresh CronJob has not
    # caught up with yet makes every new control-plane Pod die on password
    # authentication failed, and a rollout that hits that window fails for a
    # reason that looks like a broken release (2026-09-07, rollback-failed).
    # After the CA bundle because the refresh Job mounts it.
    self._refresh_aurora_credentials()
    if not self._remote_commands_are_idle():
        raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
    # One store read; refuses by name while a driver/firmware/EFA install is
    # PENDING or WAITING (``regional_release_store_preflight``).
    verdict = self._require_no_inflight_installs(action="upgrade")
    active_diff = diff or _default_release_diff()
    plan = build_execution_plan(active_diff)
    acceptance = _validate_upgrade_transaction(
        self, active_diff, plan, resume=resume, supersede=supersede
    )
    (
        previous,
        completed_phases,
        completed_clusters,
        registry_staged,
    ) = _upgrade_context(
        self,
        resume=resume,
        diff=active_diff,
        plan=plan,
        supersede=supersede,
    )
    if not resume:
        # A new transaction must not inherit rollback checkpoints or failure
        # fields from the currently deployed release.
        self.state = {}
    if isinstance(verdict, dict):
        # The gate's verdict, persisted by the first checkpoint like the rollback's.
        self.state["inflight_installs"] = verdict
    if not resume:
        self._save_state(
            "preflight",
            previous=previous,
            **(
                {SUPERSEDED_TRANSACTION_KEY: superseded_transaction_record(supersede)}
                if supersede is not None
                else {}
            ),
            # Folded into every fleet deployment id of this transaction, so a
            # candidate re-applied after its rollback gets a fresh rollout.
            fleet_rollout_transaction=uuid.uuid4().hex[:12],
            release_diff=active_diff.as_dict(),
            execution_plan=plan.as_dict(),
            completed_phases=[],
            completed_cluster_ids=[],
            registry_staged=False,
            previous_snapshot_sha256=canonical_sha256(previous),
            transaction_committed=False,
            release_lifecycle="PREPARING",
            **(
                {SCHEMA_CHANGE_ACCEPTANCE_KEY: acceptance}
                if acceptance is not None
                else {}
            ),
            cluster_attempts={
                target.cluster_id: {
                    "state": "PENDING",
                    "attempt_generation": 1,
                    "updated_at_epoch": time.time(),
                }
                for target in self.config.clusters
            },
        )
    try:
        run_upgrade_phases(
            self,
            diff=active_diff,
            plan=plan,
            previous=previous,
            completed_phases=completed_phases,
            completed_clusters=completed_clusters,
            registry_staged=registry_staged,
        )
    except Exception as upgrade_error:
        recover_failed_upgrade(
            self,
            error=upgrade_error,
            diff=active_diff,
            plan=plan,
            previous=previous,
            completed_phases=completed_phases,
            completed_clusters=completed_clusters,
            registry_staged=registry_staged,
        )


def _rollback_context(
    self: Any,
    state: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = dict(self.state)
    if state is None:
        loaded = self._load_state()
    self.state = dict(loaded)
    previous = state or loaded.get("previous")
    if not previous:
        if loaded.get("phase") in {
            "bootstrap-started",
            "bootstrap-failed",
        }:
            self._cleanup_bootstrap()
            return loaded, {}
        raise ReleaseError("previous release state is unavailable")
    changed = set(((loaded.get("release_diff") or {}).get("changed") or []))
    unsupported = changed.intersection(NON_TRANSACTIONAL_CHANGES)
    if unsupported:
        raise ReleaseError(
            "rollback is not transactional for: " + ", ".join(sorted(unsupported))
        )
    if "database_schema" in changed:
        # Never compatible: the old wheel requires the exact previous schema.
        acceptance = recorded_acceptance(loaded)
        snapshot = (acceptance or {}).get("snapshot_id")
        raise ReleaseError(
            "rollback across this PostgreSQL schema change is not "
            "declared backward-compatible"
            + (
                f"; restore the database from the pre-schema snapshot {snapshot} "
                "first, then rerun the rollback"
                if snapshot
                else "; the database has to be restored to the previous schema "
                "version by hand before a rollback can run"
            )
        )
    return loaded, previous


def _restore_rollback_cpu(
    self: Any,
    *,
    previous: dict[str, Any],
    metadata: dict[str, Any],
    cpu_wheel: str,
    artifact: str,
    config_digest: str,
    runtime_profile_version: str,
    runtime_image: str,
) -> dict[str, Any]:
    """Put the previous control plane back and say how its env was rebuilt.

    Returns the phase details recorded in ``rollback_timing``: whether the
    previous Deployments' container environment came from the transaction's
    own snapshot or, for a transaction opened before that capture existed,
    from the current template. The distinction is what makes a later
    ``unknown GPU_FAULT environment variable`` rollback failure explainable.
    """

    # Validated before the first mutation: a malformed snapshot stops the
    # rollback while the Secret, registry and role ConfigMaps are untouched.
    container_env = previous_container_env_snapshot(
        previous.get("cpu_role_container_env")
    )
    if container_env is None:
        details = {
            "cpu_container_env": "current-template",
            "cpu_container_env_note": (
                "previous snapshot predates the container environment capture; "
                "the previous image starts with the current template's env names"
            ),
        }
        narrate_step(
            "rollback-cpu-container-env",
            source="current-template",
            reason="snapshot-predates-capture",
        )
    else:
        details = {"cpu_container_env": "snapshot"}
    previous_admin_config = AdminConfig.from_mapping(previous.get("admin_config") or {})
    cpu_secret = (previous.get("secret_backups") or {}).get("cpu") or {}
    if cpu_secret.get("backup"):
        self._restore_secret(
            self._cpu(),
            source=str(cpu_secret["source"]),
            backup=str(cpu_secret["backup"]),
        )
    if self._restore_registry_backup():
        self._publish_restored_registry()
    preserve_role_config_maps = self._restore_cpu_role_config_maps(
        previous.get("cpu_role_config_maps")
    )
    cpu_sha = self._config_map_sha(
        self._cpu(),
        cpu_wheel,
        self.config.wheel.name,
    )
    with tempfile.TemporaryDirectory(prefix="gpu-fault-rollback-inputs-") as inputs:
        container_env_file = (
            write_rollback_container_env(container_env, Path(inputs))
            if container_env is not None
            else None
        )
        environment = build_rollback_environment(
            rollback_config=self.config.for_rollback(
                config_digest,
                admin_config=previous_admin_config,
            ),
            metadata=metadata,
            cpu_wheel=cpu_wheel,
            cpu_sha=cpu_sha,
            artifact=artifact,
            config_digest=config_digest,
            runtime_profile_version=runtime_profile_version,
            runtime_image=runtime_image,
            preserve_role_config_maps=preserve_role_config_maps,
            previous_container_env_file=container_env_file,
        )
        _apply_rollback_cpu_environment(self, environment)
    refresh_exists = self.runner.probe(
        self._cpu(
            "-n",
            self.config.namespace,
            "get",
            "cronjob",
            "gpu-fault-aurora-credential-refresh",
        ),
    )
    if refresh_exists:
        self.runner.run(
            self._cpu(
                "-n",
                self.config.namespace,
                "set",
                "image",
                "cronjob/gpu-fault-aurora-credential-refresh",
                f"refresh={runtime_image}",
            )
        )
    return details


def _stage_rollback_controller(
    self: Any,
    *,
    previous: dict[str, Any],
    metadata: dict[str, Any],
    cpu_wheel: str,
    artifact: str,
    config_digest: str,
    runtime_profile_version: str,
) -> None:
    previous_admin_config = AdminConfig.from_mapping(previous.get("admin_config") or {})
    cpu_secret = (previous.get("secret_backups") or {}).get("cpu") or {}
    if cpu_secret.get("backup"):
        self._restore_secret(
            self._cpu(),
            source=str(cpu_secret["source"]),
            backup=str(cpu_secret["backup"]),
        )
    if self._restore_registry_backup():
        self._publish_restored_registry()
    controller_config = rollback_controller_config(
        previous.get("agent_identities") or {}
    )
    patch = json.dumps({"data": controller_config}, sort_keys=True)
    for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS:
        self.runner.run(
            self._cpu(
                "-n",
                self.config.namespace,
                "patch",
                "configmap",
                f"{deployment}-config-core",
                "--type=merge",
                "-p",
                patch,
            )
        )
    cpu_sha = self._config_map_sha(
        self._cpu(),
        cpu_wheel,
        self.config.wheel.name,
    )
    environment = build_rollback_environment(
        rollback_config=self.config.for_rollback(
            config_digest,
            admin_config=previous_admin_config,
        ),
        metadata=metadata,
        cpu_wheel=cpu_wheel,
        cpu_sha=cpu_sha,
        artifact=artifact,
        config_digest=config_digest,
        runtime_profile_version=runtime_profile_version,
        runtime_image=self.runtime_image,
        preserve_role_config_maps=True,
    )
    _apply_rollback_cpu_environment(self, environment)


def rollback_target(
    self: Any,
    target: ClusterTarget,
    *,
    previous: dict[str, Any],
    artifact: str,
    config_digest: str,
    runtime_profile_version: str,
    executor_artifact: str,
    executor_compatibility: str,
    node_compatibility: str,
    runtime_image: str,
    node_installer_image: str,
    components: frozenset[ReleaseComponent],
) -> None:
    old = (previous.get("clusters") or {}).get(target.cluster_id, {})
    if ReleaseComponent.ENDPOINT in components:
        secret = ((previous.get("secret_backups") or {}).get("clusters") or {}).get(
            target.cluster_id
        ) or {}
        if not secret.get("backup") or not secret.get("source"):
            raise ReleaseError(
                f"{target.cluster_id} rollback connection Secret backup is missing"
            )
        self._restore_secret(
            self._gpu(target),
            source=str(secret["source"]),
            backup=str(secret["backup"]),
        )
    wheel = old.get("wheel")
    if ReleaseComponent.ENDPOINT in components:
        self._verify_gpu_control_plane_endpoint(target)
    if ReleaseComponent.DCGM in components:
        if not old.get("dcgm_image"):
            raise ReleaseError(
                f"{target.cluster_id} previous DCGM image is unavailable"
            )
        self._apply_gpu_dcgm_exporter(
            target,
            image=old["dcgm_image"],
        )
    deployment_names = {
        deployment
        for component, deployment in (
            (ReleaseComponent.EXECUTOR, inventory.GPU_EXECUTOR_DEPLOYMENT),
            (ReleaseComponent.WATCHER, inventory.GPU_WATCHER_DEPLOYMENT),
            (ReleaseComponent.COLLECTOR, inventory.GPU_COLLECTOR_DEPLOYMENT),
        )
        if component in components
    }
    if deployment_names:
        if not wheel:
            raise ReleaseError(
                f"{target.cluster_id} previous Executor wheel is unavailable"
            )
        rollback_artifact = executor_artifact or self._config_map_sha(
            self._gpu(target),
            wheel,
            old.get("wheel_key") or self.config.executor_wheel.name,
        )
        self._apply_gpu_deployments(
            target,
            wheel,
            deployment_names=frozenset(deployment_names),
            runtime_profile_version=runtime_profile_version,
            executor_wheel_filename=old.get("wheel_key"),
            executor_artifact_sha=rollback_artifact,
            executor_compatibility_digest=(executor_compatibility or rollback_artifact),
            runtime_image=runtime_image,
        )
    if ReleaseComponent.AGENT in components:
        missing = [name for name in ("reconciler_wheel", "bundle") if not old.get(name)]
        if missing:
            raise ReleaseError(
                f"{target.cluster_id} previous Agent rollback resources are "
                "missing: " + ", ".join(missing)
            )
        agent_identity = (previous.get("agent_identities") or {}).get(
            target.cluster_id
        ) or {}
        legacy_agent_identity = validate_rollback_agent_identity(
            target.cluster_id,
            agent_identity,
            metadata=previous.get("metadata") or {},
            artifact=artifact,
            compatibility=node_compatibility,
            config_digest=config_digest,
            runtime_profile_version=runtime_profile_version,
        )
        self._roll_node_runtime(
            target,
            phase="rollback",
            wheel_cm=old["reconciler_wheel"],
            bundle_cm=old["bundle"],
            artifact_sha=artifact,
            config_digest=config_digest,
            runtime_profile_version=runtime_profile_version,
            executor_wheel_filename=old.get("reconciler_wheel_key"),
            node_compatibility_digest=node_compatibility,
            bundle_sha256=old.get("bundle_sha256"),
            template_sha256=old.get("template_sha256"),
            template_config_map=None,
            runtime_image=self.runtime_image,
            steady_runtime_image=runtime_image,
            steady_template_config_map=old.get("template"),
            node_installer_image=node_installer_image,
            allow_legacy_identity=legacy_agent_identity,
            agent_identity=agent_identity,
        )
    elif ReleaseComponent.RECONCILER in components:
        missing = [name for name in ("reconciler_wheel", "bundle") if not old.get(name)]
        if missing:
            raise ReleaseError(
                f"{target.cluster_id} previous Reconciler resources are "
                "missing: " + ", ".join(missing)
            )
        self._deploy_reconciler(
            target,
            wheel_cm=old["reconciler_wheel"],
            bundle_cm=old["bundle"],
            artifact_sha=artifact,
            config_digest=config_digest,
            runtime_profile_version=runtime_profile_version,
            executor_wheel_filename=old.get("reconciler_wheel_key"),
            node_compatibility_digest=node_compatibility,
            bundle_sha256=old.get("bundle_sha256"),
            template_sha256=old.get("template_sha256"),
            template_config_map=old.get("template"),
            allowed_node_names=None,
            runtime_image=runtime_image,
            node_installer_image=node_installer_image,
        )


def _rollback_gpu_clusters(
    self: Any,
    *,
    previous: dict[str, Any],
    loaded: dict[str, Any],
    completed_phases: set[str],
    completed_clusters: set[str],
    target_arguments: dict[str, Any],
    compensation: RollbackCompensationPlan,
    timing: dict[str, Any],
) -> None:
    pending = [
        target
        for target in self.config.clusters
        if target.cluster_id not in completed_clusters
        and compensation.for_cluster(target.cluster_id)
    ]
    progress_clusters = (loaded.get("component_progress") or {}).get("clusters") or {}
    config_order = {
        target.cluster_id: index for index, target in enumerate(self.config.clusters)
    }

    def changed_at(target: ClusterTarget) -> tuple[float, int]:
        entries = progress_clusters.get(target.cluster_id) or {}
        observed = max(
            (
                float(entry.get("updated_at_epoch") or 0.0)
                for entry in entries.values()
                if isinstance(entry, dict)
            ),
            default=0.0,
        )
        return observed, config_order[target.cluster_id]

    pending.sort(key=changed_at, reverse=True)
    for target in pending:
        start_timed_entry(
            timing,
            "clusters",
            target.cluster_id,
            details={
                "components": sorted(
                    component.value
                    for component in compensation.for_cluster(target.cluster_id)
                )
            },
        )
    if pending:
        self._save_state(
            "rollback-data-progress",
            previous=previous,
            rollback_completed_phases=sorted(completed_phases),
            rollback_completed_cluster_ids=sorted(completed_clusters),
            rollback_plan=compensation.as_dict(),
            rollback_timing=timing,
            release_lifecycle="ROLLING_BACK",
            original_failure=loaded.get("original_failure"),
        )
    for target in pending:
        cluster_id = target.cluster_id
        try:
            rollback_target(
                self,
                target,
                previous=previous,
                components=compensation.for_cluster(cluster_id),
                **target_arguments,
            )
        except Exception as exc:
            complete_timed_entry(
                timing,
                "clusters",
                cluster_id,
                status="FAILED",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
            merge_rollback_wave_timings(self, timing)
            self._save_state(
                "rollback-data-progress",
                previous=previous,
                rollback_completed_phases=sorted(completed_phases),
                rollback_completed_cluster_ids=sorted(completed_clusters),
                rollback_plan=compensation.as_dict(),
                rollback_timing=timing,
                release_lifecycle="FAILED",
                original_failure=loaded.get("original_failure"),
            )
            raise ReleaseError(f"{cluster_id} rollback failed: {exc}") from exc
        complete_timed_entry(timing, "clusters", cluster_id)
        merge_rollback_wave_timings(self, timing)
        completed_clusters.add(cluster_id)
        attempts = cluster_attempts_with(
            self.state,
            cluster_id,
            "ROLLED_BACK",
        )
        self._save_state(
            "rollback-data-progress",
            previous=previous,
            rollback_completed_phases=sorted(completed_phases),
            rollback_completed_cluster_ids=sorted(completed_clusters),
            rollback_plan=compensation.as_dict(),
            rollback_timing=timing,
            cluster_attempts=attempts,
            release_lifecycle="ROLLING_BACK",
            original_failure=loaded.get("original_failure"),
        )


def _retain_valid_rollback_clusters(
    self: Any,
    *,
    loaded: dict[str, Any],
    previous: dict[str, Any],
    runtime_image: str,
    compensation: RollbackCompensationPlan,
    completed_phases: set[str],
    completed_clusters: set[str],
) -> None:
    if (
        loaded.get("phase") != "rollback-data-restored"
        or "rollback-verified" in completed_phases
    ):
        return
    retained = set()
    for target in self.config.clusters:
        if target.cluster_id not in completed_clusters:
            continue
        try:
            validate_gpu_rollback_target(
                self,
                previous,
                runtime_image,
                target,
                components=compensation.for_cluster(target.cluster_id),
                run_verifier=False,
            )
        except Exception:
            continue
        retained.add(target.cluster_id)
    completed_clusters.intersection_update(retained)
    planned = {
        target.cluster_id
        for target in self.config.clusters
        if compensation.for_cluster(target.cluster_id)
    }
    if not planned.issubset(completed_clusters):
        completed_phases.discard("rollback-data-restored")


def _verify_and_complete_rollback(
    self: Any,
    *,
    previous: dict[str, Any],
    loaded: dict[str, Any],
    compensation: RollbackCompensationPlan,
    completed_phases: set[str],
    completed_clusters: set[str],
    timing: dict[str, Any],
    run_phase: Callable[[str, str, str, Callable[[], None]], None],
) -> None:
    run_phase(
        "verify",
        "rollback-verifying",
        "rollback-verified",
        lambda: self._validate_rollback(
            previous,
            restore_cpu=compensation.restores_cpu,
            cluster_components=compensation.cluster_components,
        ),
    )
    complete_rollback(
        self,
        previous=previous,
        completed_phases=completed_phases,
        completed_clusters=completed_clusters,
        rollback_plan=compensation.as_dict(),
        rollback_timing=timing,
        original_failure=loaded.get("original_failure"),
    )


def _restore_regional_singletons(
    self: Any,
    *,
    previous: dict[str, Any],
    compensation: RollbackCompensationPlan,
    run_phase: Callable[[str, str, str, Callable[[], object]], None],
) -> None:
    """Put back the components there is exactly one of for the whole region.

    Both are snapshot restores of objects applied straight from the candidate
    checkout, and both run before the data plane rather than after it. For the
    endpoint that ordering is load-bearing: the per-cluster half of the endpoint
    component restores each cluster's connection Secret and then runs
    ``_verify_gpu_control_plane_endpoint`` against it, and that verification only
    proves something once the Service and the Route53 record it resolves are
    already back.

    Observability's per-GPU-cluster half (the data-plane collector) is restored
    here too, from the per-cluster snapshots inside the observability snapshot:
    it is a global component in the compensation plan. The phase returns the
    record (path taken, per-cluster outcome) the phase runner persists.
    """

    if compensation.restores_observability:
        run_phase(
            "observability_restore",
            "rollback-observability-restoring",
            "rollback-observability-restored",
            lambda: restore_observability_with_dataplane(self, previous),
        )
    if compensation.restores_endpoint:
        run_phase(
            "endpoint_restore",
            "rollback-endpoint-restoring",
            "rollback-endpoint-restored",
            lambda: self._restore_endpoint_snapshot(previous.get("endpoint")),
        )


def _rollback_phase_runner(
    self: Any,
    *,
    timing: dict[str, Any],
    previous: dict[str, Any],
    loaded: dict[str, Any],
    compensation: Any,
    completed_phases: set[str],
    completed_clusters: set[str],
) -> tuple[
    Callable[[str, str], None],
    Callable[[str, str, Exception], None],
    Callable[..., None],
    Callable[[str, str, str, Callable[[], object]], None],
]:
    """The rollback's phase bookkeeping: start, fail, checkpoint, run.

    Every phase persists the same durable record (progress sets, plan, timing,
    lifecycle) so a crash mid-rollback resumes from the last checkpoint; these
    closures share that record so no phase can persist a partial one.
    """

    def save_progress(
        phase: str,
        *,
        release_lifecycle: str = "ROLLING_BACK",
    ) -> None:
        merge_rollback_wave_timings(self, timing)
        self._save_state(
            phase,
            previous=previous,
            rollback_completed_phases=sorted(completed_phases),
            rollback_completed_cluster_ids=sorted(completed_clusters),
            rollback_plan=compensation.as_dict(),
            rollback_timing=timing,
            release_lifecycle=release_lifecycle,
            original_failure=loaded.get("original_failure"),
        )

    def start_phase(name: str, phase: str) -> None:
        start_timed_entry(timing, "phases", name)
        save_progress(phase, release_lifecycle="FAILED")

    def fail_phase(name: str, phase: str, error: Exception) -> None:
        complete_timed_entry(
            timing,
            "phases",
            name,
            status="FAILED",
            details={"error": f"{type(error).__name__}: {error}"},
        )
        save_progress(phase)

    def checkpoint(
        phase: str,
        *,
        timing_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if timing_name is not None:
            complete_timed_entry(timing, "phases", timing_name, details=details)
        completed_phases.add(phase)
        save_progress(phase)

    def run_phase(
        name: str,
        started_phase: str,
        completed_phase: str,
        action: Callable[[], object],
    ) -> None:
        if completed_phase in completed_phases:
            return
        start_phase(name, started_phase)
        try:
            outcome = action()
        except Exception as exc:
            fail_phase(name, started_phase, exc)
            raise
        # A phase that returns a mapping is describing how it did its work
        # (the CPU restore says where the container env came from); that goes
        # into the durable timing record beside the phase's duration.
        checkpoint(
            completed_phase,
            timing_name=name,
            details=outcome if isinstance(outcome, dict) else None,
        )

    return start_phase, fail_phase, checkpoint, run_phase


def rollback_release(
    self: Any,
    *,
    state: dict[str, Any] | None = None,
    automatic: bool = False,
) -> None:
    loaded, previous = _rollback_context(self, state)
    if not previous:
        return
    # The compensating restore restarts the control-plane roles; on 2026-09-07
    # it restarted them into a password RDS had rotated mid-transaction and the
    # release landed in rollback-failed. Fresh credentials first, fail closed,
    # before any restore is even planned.
    self._refresh_aurora_credentials()
    # Before any restore is planned: the previous release's control plane would
    # re-submit an install that is PENDING or WAITING right now. Skipped once
    # the control plane is already restored (a cleanup re-entry). An automatic
    # rollback proceeds only when no Running control-plane Pod could run the
    # probe (StoreUnreachable); a manual one refuses. The verdict rides in the
    # state so ``rollback-started`` persists whether and on what this was
    # checked (``render_persisted_state`` writes the state whole).
    if "rollback-cpu-restored" not in set(
        loaded.get("rollback_completed_phases") or []
    ):
        verdict = self._require_no_inflight_installs(
            action="rollback",
            unreadable="proceed" if automatic else "refuse",
        )
        if isinstance(verdict, dict):
            self.state["inflight_installs"] = verdict
    compensation = build_rollback_compensation_plan(
        loaded,
        (target.cluster_id for target in self.config.clusters),
    )
    (
        metadata,
        cpu_wheel,
        artifact,
        config_digest,
        profile,
        executor_artifact,
        executor_compatibility,
        runtime_image,
    ) = _rollback_identity_context(self, previous, compensation)
    completed_phases = set(loaded.get("rollback_completed_phases") or [])
    completed_clusters = set(loaded.get("rollback_completed_cluster_ids") or [])
    timing = initialize_rollback_timing(loaded)
    if not hasattr(self, "_rollback_wave_timings"):
        self._rollback_wave_timings = {}
    if "rollback-restored" not in completed_phases:
        start_timed_entry(
            timing,
            "phases",
            "restore",
            details={"plan": compensation.as_dict()},
        )
    target_arguments = _rollback_target_arguments(
        self,
        previous=previous,
        metadata=metadata,
        artifact=artifact,
        config_digest=config_digest,
        profile=profile,
        executor_artifact=executor_artifact,
        executor_compatibility=executor_compatibility,
        runtime_image=runtime_image,
    )
    _retain_valid_rollback_clusters(
        self,
        loaded=loaded,
        previous=previous,
        runtime_image=runtime_image,
        compensation=compensation,
        completed_phases=completed_phases,
        completed_clusters=completed_clusters,
    )

    start_phase, fail_phase, checkpoint, run_phase = _rollback_phase_runner(
        self,
        timing=timing,
        previous=previous,
        loaded=loaded,
        compensation=compensation,
        completed_phases=completed_phases,
        completed_clusters=completed_clusters,
    )

    if "rollback-started" not in completed_phases:
        checkpoint("rollback-started")

    if compensation.needs_controller:
        run_phase(
            "controller_stage",
            "rollback-controller-staging",
            "rollback-controller-staged",
            lambda: _stage_rollback_controller(
                self,
                previous=previous,
                metadata=metadata,
                cpu_wheel=cpu_wheel,
                artifact=artifact,
                config_digest=config_digest,
                runtime_profile_version=profile,
            ),
        )
    _restore_regional_singletons(
        self,
        previous=previous,
        compensation=compensation,
        run_phase=run_phase,
    )
    if (
        compensation.restores_data_plane
        and "rollback-data-restored" not in completed_phases
    ):
        start_phase("data_restore", "rollback-data-restoring")
        try:
            _rollback_gpu_clusters(
                self,
                previous=previous,
                loaded=loaded,
                completed_phases=completed_phases,
                completed_clusters=completed_clusters,
                compensation=compensation,
                target_arguments=target_arguments,
                timing=timing,
            )
        except Exception as exc:
            fail_phase("data_restore", "rollback-data-restoring", exc)
            raise
        checkpoint("rollback-data-restored", timing_name="data_restore")
    # Deliberately not gated on `compensation.restores_data_plane`. Everything
    # inside is already scoped per cluster by the compensation plan, and the
    # stranded-rollout sweep has to run even when the plan claims no data-plane
    # component: the case that stranded a PLANNED fleet deployment for 34 hours
    # was an upgrade that created the record and then died before writing the
    # AGENT progress the plan is derived from.
    if "rollback-rollout-cleaned" not in completed_phases:
        run_phase(
            "rollout_cleanup",
            "rollback-rollout-cleaning",
            "rollback-rollout-cleaned",
            lambda: cleanup_candidate_rollout_state(self, compensation),
        )
    if compensation.restores_cpu:
        run_phase(
            "cpu_restore",
            "rollback-cpu-restoring",
            "rollback-cpu-restored",
            lambda: _restore_rollback_cpu(
                self,
                previous=previous,
                metadata=metadata,
                cpu_wheel=cpu_wheel,
                artifact=artifact,
                config_digest=config_digest,
                runtime_profile_version=profile,
                runtime_image=runtime_image,
            ),
        )
    if "rollback-restored" not in completed_phases:
        complete_timed_entry(timing, "phases", "restore")
        if "safe_at_epoch" not in timing:
            mark_rollback_safe(timing)
        checkpoint("rollback-restored")
    _verify_and_complete_rollback(
        self,
        previous=previous,
        loaded=loaded,
        compensation=compensation,
        completed_phases=completed_phases,
        completed_clusters=completed_clusters,
        timing=timing,
        run_phase=run_phase,
    )
