"""Snapshot-protected Aurora-only repair before Store-dependent release gates."""

from __future__ import annotations

import copy
import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from gpu_fault.admin.process_supervisor import ensure_supervision_safe
from gpu_fault_release.regional_aurora_credentials import (
    CRONJOB_NAME,
    read_aurora_refresh_cronjob,
    require_current_refresh_status,
)
from gpu_fault_release.regional_release_aurora_refresh import (
    apply_aurora_refresh,
    aurora_refresh_drift,
    capture_aurora_refresh_snapshot,
    restore_aurora_refresh_snapshot,
    validate_aurora_refresh_snapshot,
)
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_diff import (
    AURORA_PREREQUISITE_REPAIR_KEY,
    ReleaseComponent,
    ReleaseDiff,
    ReleaseExecutionPlan,
    build_execution_plan,
    diff_from_changed,
)
from gpu_fault_release.regional_release_probe_job import (
    Checkpoint,
    RUN_ANNOTATION,
    cleanup_probe_job,
    run_probe_job,
)
from gpu_fault_release.regional_release_store_proof import (
    bootstrap_store_proof,
    database_identity,
)
from gpu_fault_release.regional_release_retained_database import (
    HANDOFF_KEY,
    RETAINED_ORIGIN_KEY,
    retained_database_proof,
)
from gpu_fault_release.regional_release_runtime_identity import CONTROL_PLANE_PYTHON
from gpu_fault_release.regional_release_images import require_digest_pinned_image
from gpu_fault_release.regional_release_progress import (
    PROGRESS_COMPLETED,
    update_component_progress,
)
from gpu_fault_release.regional_release_prerequisite_baseline import (
    capture_baseline_representation,
    validated_prerequisite_baseline,
)
from gpu_fault_release.regional_release_prerequisite_handoff import (
    HANDOFF_AUDIT_KEY,
    bootstrap_handoff_audit,
    candidate_bootstrap_baseline,
    upgrade_handoff_audit,
    validate_bootstrap_repair_handoff,
    validate_upgrade_repair_handoff,
)
from gpu_fault_release.regional_release_transaction import save_recorded_state

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

REPAIR_KEY = AURORA_PREREQUISITE_REPAIR_KEY
BOOTSTRAP_ORIGIN_KEY = "bootstrap_database_origin"


def has_prerequisite_repair(state: dict[str, Any]) -> bool:
    return REPAIR_KEY in state


def _baseline(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return validated_prerequisite_baseline(state, record)


def _validate_record(release: RegionalRelease, state: dict[str, Any]) -> dict[str, Any]:
    record = state.get(REPAIR_KEY)
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ReleaseError("Aurora prerequisite repair context is invalid")
    binding = record.get("binding")
    if (
        not isinstance(binding, dict)
        or canonical_sha256(binding) != record.get("binding_sha256")
        or ("bootstrap" in binding and type(binding["bootstrap"]) is not bool)
        or binding.get("release_id") != release.release_id
        or binding.get("manifest_sha256") != release.rendered_manifest_digest
        or binding.get("cpu_eks_arn") != release.config.cpu_eks_arn
        or binding.get("namespace") != release.config.namespace
        or binding.get("runtime_image") != release.runtime_image
        or binding.get("wheel_config_map") != release.wheel_cm
        or canonical_sha256(record.get("previous_refresher"))
        != record.get("snapshot_sha256")
    ):
        raise ReleaseError(
            "Aurora prerequisite repair candidate or snapshot binding differs"
        )
    try:
        baseline = _baseline(state, record)
    except KeyError as exc:
        raise ReleaseError("Aurora prerequisite repair baseline is incomplete") from exc
    if canonical_sha256(baseline) != record.get("baseline_sha256"):
        raise ReleaseError("Aurora prerequisite repair baseline drifted")
    validate_aurora_refresh_snapshot(release, record["previous_refresher"])
    if record.get("status") not in {"PREPARED", "APPLYING", "READY", "FAILED"}:
        raise ReleaseError("Aurora prerequisite repair status is unknown")
    if not isinstance(record.get("jobs"), dict):
        raise ReleaseError("Aurora prerequisite repair Job journal is invalid")
    return record


def matches_pre_repair_state(
    release: RegionalRelease, state: dict[str, Any], expected: str
) -> bool:
    if not has_prerequisite_repair(state):
        return False
    record = _validate_record(release, state)
    if record["baseline_sha256"] == expected:
        return True
    # The deploy diff was pinned while a predecessor journal was still live;
    # the successor this candidate wrote in preflight recorded that digest.
    supersedes = record.get("supersedes")
    return bool(
        isinstance(supersedes, dict)
        and isinstance(supersedes.get("state_sha256"), str)
        and supersedes["state_sha256"] == expected
    )


def _persist(release: RegionalRelease, record: dict[str, Any]) -> None:
    save_recorded_state(release, str(release.state["phase"]), **{REPAIR_KEY: record})


def repair_job_checkpoint(release: RegionalRelease, name: str) -> Checkpoint:
    def checkpoint(job: dict[str, Any]) -> None:
        record = release.state[REPAIR_KEY]
        previous = record["jobs"].get(name)
        if (
            previous
            and previous.get("name") != job.get("name")
            and previous.get("status") != "REMOVED"
        ):
            raise ReleaseError("previous proof Job cleanup is incomplete")
        record["jobs"][name] = dict(job)
        _persist(release, record)

    return checkpoint


def _cleanup_jobs(release: RegionalRelease, record: dict[str, Any]) -> None:
    for name, job in list(record["jobs"].items()):
        if job.get("status") != "REMOVED":
            cleanup_probe_job(release, job, repair_job_checkpoint(release, name))


def _new_record(
    release: RegionalRelease,
    *,
    baseline: dict[str, Any],
    snapshot: object,
    identity: dict[str, Any],
    bootstrap: bool,
    commit_live: bool,
) -> dict[str, Any]:
    binding = {
        "release_id": release.release_id,
        "manifest_sha256": release.rendered_manifest_digest,
        "cpu_eks_arn": release.config.cpu_eks_arn,
        "namespace": release.config.namespace,
        "runtime_image": release.runtime_image,
        "wheel_config_map": release.wheel_cm,
        "bootstrap": bootstrap,
        "commit_live": commit_live,
        "database": identity,
        "execution_plan": ReleaseExecutionPlan(
            (ReleaseComponent.AURORA_REFRESH, ReleaseComponent.VERIFY)
        ).as_dict(),
    }
    return {
        "schema_version": 1,
        "attempt_id": uuid.uuid4().hex,
        "binding": binding,
        "binding_sha256": canonical_sha256(binding),
        "previous_refresher": copy.deepcopy(snapshot),
        "snapshot_sha256": canonical_sha256(snapshot),
        "baseline_sha256": canonical_sha256(baseline),
        "baseline_had_timestamp": "updated_at_epoch" in baseline,
        "baseline_timestamp": baseline.get("updated_at_epoch"),
        "snapshot_representation": capture_baseline_representation(
            baseline, bootstrap=bootstrap
        ),
        "baseline_commit_fields": {
            key: {"present": key in baseline, "value": baseline.get(key)}
            for key in (
                "transaction_committed",
                "release_lifecycle",
                "commit_cleanup_completed",
            )
        },
        "status": "PREPARED",
        "jobs": {},
    }


def _advance_bootstrap_repair(
    release: RegionalRelease,
    record: dict[str, Any],
    *,
    identity: dict[str, Any],
) -> dict[str, Any]:
    # Even removed Jobs are rechecked before their CronJob owner can be restored.
    for name, job in list(record["jobs"].items()):
        cleanup_probe_job(release, job, repair_job_checkpoint(release, name))
    ensure_supervision_safe()
    validate_bootstrap_repair_handoff(
        release, release.state, identity=database_identity(release)
    )
    restore_aurora_refresh_snapshot(release, record["previous_refresher"])
    if database_identity(release) != identity:
        raise ReleaseError("bootstrap prerequisite identity changed during restoration")
    original_state = release.state
    baseline = candidate_bootstrap_baseline(release, original_state)
    successor = _new_record(
        release,
        baseline=baseline,
        snapshot=record["previous_refresher"],
        identity=identity,
        bootstrap=True,
        commit_live=False,
    )
    successor["supersedes"] = bootstrap_handoff_audit(record)
    # One checkpoint replaces the journal and all candidate identity fields.
    # Before it, a retry still owns the original snapshot and repeats restoration.
    release.state = {**baseline, REPAIR_KEY: successor}
    try:
        _persist(release, successor)
    except BaseException:
        release.state = original_state
        raise
    return successor


def _retire_orphaned_repair(
    release: RegionalRelease, record: dict[str, Any], *, identity: dict[str, Any]
) -> None:
    """Restore the original refresher an abandoned upgrade candidate repaired over.

    ``record`` must be the journal object held in ``release.state``: the Job
    checkpoints persist through it. Nothing here removes the journal; a failure
    leaves the original record for the next attempt to repeat the restoration.
    """
    # Even removed Jobs are rechecked before their CronJob owner can be restored.
    for name, job in list(record["jobs"].items()):
        cleanup_probe_job(release, job, repair_job_checkpoint(release, name))
    ensure_supervision_safe()
    validate_upgrade_repair_handoff(
        release, release.state, identity=database_identity(release)
    )
    restore_aurora_refresh_snapshot(release, record["previous_refresher"])
    if database_identity(release) != identity:
        raise ReleaseError("upgrade prerequisite identity changed during restoration")


def _advance_upgrade_repair(
    release: RegionalRelease,
    record: dict[str, Any],
    *,
    identity: dict[str, Any],
) -> dict[str, Any]:
    # The digest the deploy driver pinned: the committed state with the orphan.
    superseded_state_sha256 = canonical_sha256(release.state)
    _retire_orphaned_repair(release, record, identity=identity)
    original_state = release.state
    baseline = copy.deepcopy(
        {key: value for key, value in original_state.items() if key != REPAIR_KEY}
    )
    successor = _new_record(
        release,
        baseline=baseline,
        snapshot=record["previous_refresher"],
        identity=identity,
        bootstrap=False,
        commit_live=False,
    )
    successor["supersedes"] = upgrade_handoff_audit(
        record, state_sha256=superseded_state_sha256
    )
    # One checkpoint replaces the journal. Before it, a retry still finds the
    # predecessor, revalidates it and repeats the restoration.
    release.state = {**baseline, REPAIR_KEY: successor}
    try:
        _persist(release, successor)
    except BaseException:
        release.state = original_state
        raise
    return successor


def _refresh(release: RegionalRelease) -> None:
    cronjob = read_aurora_refresh_cronjob(release)
    if cronjob is None:
        raise ReleaseError("prerequisite refresher is absent")
    spec = copy.deepcopy(cronjob["spec"]["jobTemplate"]["spec"])
    pod = spec["template"]["spec"]
    containers = pod.get("containers") or []
    environment = {
        item.get("name"): item.get("value")
        for container in containers
        for item in container.get("env", [])
    }
    if (
        len(containers) != 1
        or containers[0].get("image") != release.runtime_image
        or containers[0].get("command") != ["gpu-fault-aurora-credential-refresh"]
        or containers[0].get("args")
        or environment.get("GPU_FAULT_AURORA_REFRESH_RESTART_DEPLOYMENTS") != "false"
    ):
        raise ReleaseError(
            "prerequisite refresher may restart consumers or has drifted"
        )
    containers[0]["command"] = [
        CONTROL_PLANE_PYTHON,
        "-I",
        "-c",
        "import inspect\n"
        "from gpu_fault.aurora_credential_refresh import refresh_once, main\n"
        "if 'restart_deployments' not in inspect.signature(refresh_once).parameters:\n"
        "    raise RuntimeError('refresher cannot guarantee no consumer restart')\n"
        "main()\n",
    ]
    run_id = uuid.uuid4().hex
    spec.update(backoffLimit=0, activeDeadlineSeconds=300, ttlSecondsAfterFinished=300)
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"{CRONJOB_NAME}-{run_id[:8]}",
            "namespace": release.config.namespace,
            "labels": {"app": CRONJOB_NAME},
            "annotations": {
                RUN_ANNOTATION: run_id,
                "gpu-fault.io/cleanup-phase": "auxiliary",
                "gpu-fault.io/cleanup-order": "30",
            },
        },
        "spec": spec,
    }
    started = datetime.now(UTC).replace(microsecond=0)

    def verify() -> dict[str, Any]:
        require_current_refresh_status(release, started)
        return {"status": "refreshed"}

    run_probe_job(
        release, job, repair_job_checkpoint(release, "credential"), verify=verify
    )


def prepare_prerequisite_repair(
    release: RegionalRelease,
    *,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
    bootstrap: bool = False,
    previous_override: dict[str, Any] | None = None,
    resume: bool = False,
    commit_live: bool = False,
) -> dict[str, Any] | None:
    if type(bootstrap) is not bool:
        raise ReleaseError("prerequisite bootstrap selection must be a boolean")
    if release.runner.dry_run or not plan.has(ReleaseComponent.AURORA_REFRESH):
        return None
    require_digest_pinned_image("candidate Aurora refresher", release.runtime_image)
    predecessor = release.state.get(REPAIR_KEY)
    binding = predecessor.get("binding") if isinstance(predecessor, dict) else None
    handoff: dict[str, Any] | None = None
    orphan: dict[str, Any] | None = None
    identity: dict[str, Any] | None = None
    if isinstance(binding, dict) and binding.get("release_id") != release.release_id:
        if resume or commit_live or previous_override is not None:
            raise ReleaseError("Aurora prerequisite repair candidate binding differs")
        identity = database_identity(release)
        if bootstrap:
            handoff = validate_bootstrap_repair_handoff(
                release, release.state, identity=identity
            )
        else:
            # An upgrade candidate that died between READY and its application
            # never returns (a release id is a digest of its tree), so the next
            # candidate restores what it repaired over and takes the journal.
            orphan = validate_upgrade_repair_handoff(
                release, release.state, identity=identity
            )
    release.pin_approved_manifest_plan(
        release.approved_manifest_digest
        or (
            release.state.get("approved_manifest_sha256")
            if (bootstrap and handoff is None)
            or (
                release.state.get("transaction_committed") is False
                and release.state.get("release_id") == release.release_id
            )
            else None
        )
        or release.rendered_manifest_digest
    )
    release.enforce_manifest_plan_pin()
    if identity is None:
        identity = database_identity(release)
    if handoff is not None:
        record = _advance_bootstrap_repair(release, handoff, identity=identity)
    elif orphan is not None:
        record = _advance_upgrade_repair(release, orphan, identity=identity)
    elif has_prerequisite_repair(release.state):
        record = _validate_record(release, release.state)
        if record["binding"]["database"] != identity:
            raise ReleaseError("Aurora prerequisite database identity drifted")
    else:
        snapshot: object = capture_aurora_refresh_snapshot(release, bootstrap=bootstrap)
        if previous_override is not None:
            snapshot = previous_override.get("aurora_refresh")
            validate_aurora_refresh_snapshot(release, snapshot)
        elif resume:
            previous = release.state.get("previous")
            if (
                not isinstance(previous, dict)
                or release.state.get("transaction_committed") is True
                or release.state.get("release_id") != release.release_id
                or release.state.get("release_diff") != diff.as_dict()
                or release.state.get("execution_plan") != plan.as_dict()
                or canonical_sha256(previous)
                != release.state.get("previous_snapshot_sha256")
            ):
                raise ReleaseError(
                    "resumed repair transaction identity or snapshot differs"
                )
            snapshot = previous.get("aurora_refresh")
            validate_aurora_refresh_snapshot(release, snapshot)
        elif (
            not bootstrap
            and not commit_live
            and release.state.get("transaction_committed") is False
            and isinstance(release.state.get("previous"), dict)
        ):
            raise ReleaseError("incomplete application transaction requires resume")
        if commit_live and not (
            release.state.get("phase") == "complete"
            and release.state.get("transaction_committed") is False
            and release.state.get("release_id") != release.release_id
        ):
            raise ReleaseError("prerequisite baseline commit identity differs")
        if not release.state:
            release._save_state(
                "bootstrap-started", previous=None, transaction_committed=False
            )
        record = _new_record(
            release,
            baseline=release.state,
            snapshot=snapshot,
            identity=identity,
            bootstrap=bootstrap,
            commit_live=commit_live,
        )
        _persist(release, record)
    _cleanup_jobs(release, record)
    try:
        record["status"] = "APPLYING"
        _persist(release, record)
        release._apply_rds_ca_bundle()
        release._upload_release(diff_from_changed({"control_plane_wheel"}))
        changed = aurora_refresh_drift(release)
        if changed and aurora_refresh_drift(release, suspended=True):
            apply_aurora_refresh(release, refresh=False, suspended=True)
        _refresh(release)
        if changed:
            apply_aurora_refresh(release, refresh=False)
        record["status"] = "READY"
        _persist(release, record)
    except Exception:
        record["status"] = "FAILED"
        _persist(release, record)
        raise
    return record


def adopted_refresher_snapshot(release: RegionalRelease) -> dict[str, Any] | None:
    if not has_prerequisite_repair(release.state):
        return None
    record = _validate_record(release, release.state)
    if record["status"] != "READY" or any(
        job.get("status") != "REMOVED" for job in record["jobs"].values()
    ):
        raise ReleaseError("Aurora prerequisite repair or cleanup is incomplete")
    snapshot: dict[str, Any] = copy.deepcopy(record["previous_refresher"])
    return snapshot


def restore_prerequisite_repair(release: RegionalRelease) -> None:
    record = release.state.get(REPAIR_KEY)
    binding = record.get("binding") if isinstance(record, dict) else None
    if isinstance(binding, dict) and binding.get("release_id") != release.release_id:
        # Another candidate's abandoned upgrade repair over a committed release:
        # the standalone rollback restores the original refresher it journaled.
        identity = database_identity(release)
        record = validate_upgrade_repair_handoff(
            release, release.state, identity=identity
        )
        _retire_orphaned_repair(release, record, identity=identity)
    else:
        record = _validate_record(release, release.state)
        _cleanup_jobs(release, record)
        restore_aurora_refresh_snapshot(release, record["previous_refresher"])
    release.state.pop(REPAIR_KEY)
    save_recorded_state(release, str(release.state["phase"]))


def prepare_bootstrap_prerequisite_entry(release: RegionalRelease) -> None:
    """Validate or retire a repair before bootstrap can stamp candidate identity."""
    if not has_prerequisite_repair(release.state) or release.runner.dry_run:
        return
    record = release.state[REPAIR_KEY]
    binding = record.get("binding") if isinstance(record, dict) else None
    predecessor = binding.get("release_id") if isinstance(binding, dict) else None
    if (
        isinstance(predecessor, str)
        and predecessor
        and predecessor != release.release_id
    ):
        diff = diff_from_changed({"aurora_refresh_manifests"})
        prepare_prerequisite_repair(
            release, diff=diff, plan=build_execution_plan(diff), bootstrap=True
        )
        # Keep the journal until ordinary bootstrap adoption. A failed handoff
        # must escape the caller before its generic checkpoint/cleanup path.
        bootstrap_workflow_proof(release)
        return
    record = _validate_record(release, release.state)
    release.pin_approved_manifest_plan(
        release.approved_manifest_digest or record["binding"]["manifest_sha256"]
    )
    release.enforce_manifest_plan_pin()
    if record["binding"]["database"] != database_identity(release):
        raise ReleaseError("Aurora prerequisite database identity drifted")


def prepare_bootstrap_workflows(release: RegionalRelease) -> None:
    if release.runner.dry_run:
        return
    diff = diff_from_changed({"aurora_refresh_manifests"})
    prepare_prerequisite_repair(
        release, diff=diff, plan=build_execution_plan(diff), bootstrap=True
    )
    proof = bootstrap_workflow_proof(release)
    origin_key = RETAINED_ORIGIN_KEY if HANDOFF_KEY in proof else BOOTSTRAP_ORIGIN_KEY
    origin = release.state.get(origin_key) or proof
    adopted_refresher_snapshot(release)
    completed_jobs = {
        name: {
            key: job.get(key)
            for key in (
                "name",
                "namespace",
                "uid",
                "owner_uid",
                "run_id",
                "spec_sha256",
                "status",
            )
        }
        for name, job in release.state[REPAIR_KEY]["jobs"].items()
    }
    superseded = release.state[REPAIR_KEY].get("supersedes")
    release.state.pop(REPAIR_KEY)
    release._save_state(
        "bootstrap-started",
        bootstrap_store_safety={**proof, "proof_jobs": completed_jobs},
        **{origin_key: origin},
        **({HANDOFF_AUDIT_KEY: copy.deepcopy(superseded)} if superseded else {}),
    )


def bootstrap_workflow_proof(release: RegionalRelease) -> dict[str, Any]:
    origin = release.state.get(BOOTSTRAP_ORIGIN_KEY)
    if origin is not None and release.config.retained_database_handoff is not None:
        raise ReleaseError("retained reinstall cannot reuse an empty-database origin")
    retained = retained_database_proof(release, repair_job_checkpoint(release, "store"))
    if retained is not None:
        return retained
    if origin is not None:
        identity = database_identity(release)
        try:
            finished = datetime.fromisoformat(origin["finished_at"])
            dated = finished.tzinfo is not None and finished <= datetime.now(UTC)
        except (KeyError, TypeError, ValueError):
            dated = False
        if (
            not isinstance(origin, dict)
            or origin.get("safe") is not True
            or origin.get("database_state") != "uninitialized_empty"
            or origin.get("schema_version") != 0
            or origin.get("identity_sha256") != canonical_sha256(identity)
            or origin.get("blockers")
            != {"workflow": 0, "remote_command": 0, "observation": 0}
            or re.fullmatch(r"[a-f0-9]{32}", str(origin.get("run_id") or "")) is None
            or not isinstance(origin.get("job_uid"), str)
            or not origin["job_uid"]
            or not dated
        ):
            raise ReleaseError("bootstrap database origin proof is missing or unbound")
    # An origin proof only identifies a previously authorized bootstrap; it
    # never substitutes for the current row/schema read.
    return bootstrap_store_proof(
        release, repair_job_checkpoint(release, "store"), require_empty=origin is None
    )


def prepare_upgrade_credentials(
    release: RegionalRelease,
    *,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
    previous_override: dict[str, Any] | None = None,
    resume: bool = False,
) -> dict[str, Any] | None:
    if plan.has(ReleaseComponent.AURORA_REFRESH):
        if not release.state:
            release._load_state()
        return prepare_prerequisite_repair(
            release,
            diff=diff,
            plan=plan,
            previous_override=previous_override,
            resume=resume,
        )
    release._apply_rds_ca_bundle()
    release._refresh_aurora_credentials()
    return None


def checkpoint_prerequisite_adoption(
    release: RegionalRelease, completed_phases: set[str]
) -> None:
    release._save_state(
        str(release.state["phase"]),
        completed_phases=sorted(completed_phases),
        component_progress=update_component_progress(
            release.state, ReleaseComponent.AURORA_REFRESH, PROGRESS_COMPLETED
        ),
    )


def restore_standalone_prerequisite(release: RegionalRelease) -> None:
    release._load_state()
    if has_prerequisite_repair(release.state) and not (
        release.state.get("transaction_committed") is False
        and isinstance(release.state.get("previous"), dict)
    ):
        restore_prerequisite_repair(release)
        raise ReleaseError(
            "Aurora prerequisite repair restored; rerun rollback to roll back the application"
        )
