from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import repository_root
from gpu_fault_release.regional_admin_checks import (
    # The same tolerant wrapper the checks use: a release object that predates
    # the read cache, or a test double standing in for one, has no
    # `_read_snapshot` and gets a `nullcontext` instead of an AttributeError.
    _read_snapshot as read_snapshot,
)
from gpu_fault_release.regional_admin_checks import (
    build_health_report,
    build_quick_health_report,
)
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import (
    ReleaseChangeKind,
    ReleaseDiff,
    build_execution_plan,
    classify_release,
    diff_from_changed,
)
from gpu_fault_release.regional_release_narration import narrate_step
from gpu_fault_release.regional_release_orchestration import SUPERSEDABLE_PHASES
from gpu_fault_release.regional_release_reporting import build_release_status
from gpu_fault_release.regional_release_transaction import commit_live_release

STATE_CONFIG_MAP = "gpu-fault-regional-release-state"
ROOT = repository_root()
EXPECTED_STATE_SHA256_ENV = "GPU_FAULT_EXPECTED_RELEASE_STATE_SHA256"
# The operator's consent to open a new transaction for a different candidate
# over a fail-forward transaction that stopped in `failed`/`partial-convergence`.
# Set by `gpu-fault-admin deploy --supersede-failed-transaction` and read here,
# travelling through the deploy's process chain as an environment variable like
# `regional_schema_change.ACCEPT_SCHEMA_CHANGE_ENV`; `gpu_fault.admin.cli`
# mirrors the spelling and a test pins the two equal.
SUPERSEDE_FAILED_TRANSACTION_ENV = "GPU_FAULT_RELEASE_SUPERSEDE_FAILED_TRANSACTION"
SUPERSEDE_FAILED_TRANSACTION_FLAG = "--supersede-failed-transaction"
RESUMABLE_PHASES = frozenset(
    {
        "preflight",
        "uploaded",
        # The candidate node preflight runs beside the control-plane phases and
        # is joined before the data plane, so it can be the recorded phase of a
        # release that has already staged the CPU roles. Omitting it made a crash
        # there look terminal, which discards `completed_phases` and
        # `cluster_attempts` and starts the whole transaction over.
        "candidate-preflight-ready",
        "schema-ready",
        "registry-staged",
        "cpu-staged",
        "profile-ready",
        "endpoint-ready",
        "observability-ready",
        "data-plane-progress",
        "data-converged",
        "cpu-finalized",
        "verified",
        "failed",
        "partial-convergence",
    }
)
ROLLBACK_PHASES = frozenset(
    {
        "rollback-started",
        "rollback-controller-staging",
        "rollback-controller-staged",
        "rollback-observability-restoring",
        "rollback-observability-restored",
        "rollback-endpoint-restoring",
        "rollback-endpoint-restored",
        "rollback-data-restoring",
        "rollback-data-progress",
        "rollback-data-restored",
        "rollback-rollout-cleaning",
        "rollback-rollout-cleaned",
        "rollback-cpu-restoring",
        "rollback-cpu-restored",
        "rollback-dataplane-observability-restoring",
        "rollback-dataplane-observability-restored",
        "rollback-restored",
        "rollback-verifying",
        "rollback-verified",
        "rollback-failed",
    }
)
BOOTSTRAP_PHASES = frozenset(
    {
        "bootstrap-started",
        "bootstrap-cpu-ready",
        "bootstrap-endpoint-ready",
        "bootstrap-data-plane-progress",
        "bootstrap-failed",
        "bootstrap-cleanup-started",
        "bootstrap-cleanup-progress",
        "bootstrap-cleanup-failed",
        "bootstrap-cleaned",
    }
)


def stored_release_diff(state: dict[str, Any]) -> ReleaseDiff | None:
    value = state.get("release_diff")
    if not isinstance(value, dict):
        return None
    try:
        kind = ReleaseChangeKind(str(value["kind"]))
        changed = frozenset(str(item) for item in value["changed"])
    except (KeyError, TypeError, ValueError):
        return None
    return ReleaseDiff(kind=kind, changed=changed)


def release_state_sha256(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rollback_pending(state: dict[str, Any]) -> bool:
    phase = str(state.get("phase") or "")
    return phase in ROLLBACK_PHASES or (
        phase == "rolled-back" and state.get("rollback_cleanup_completed") is False
    )


def _commit_cleanup_pending(state: dict[str, Any]) -> bool:
    return (
        state.get("phase") == "complete"
        and state.get("transaction_committed") is True
        and state.get("commit_cleanup_completed") is False
    )


def _commit_pending(state: dict[str, Any]) -> bool:
    return (
        str(state.get("phase") or "") == "complete"
        and state.get("transaction_committed") is False
    )


def _upgrade_resume_required(state: dict[str, Any]) -> bool:
    return str(state.get("phase") or "") in RESUMABLE_PHASES or _commit_pending(state)


def supersede_requested(environment: dict[str, str] | None = None) -> bool:
    """Whether this command carried ``--supersede-failed-transaction``."""

    return bool(
        (environment if environment is not None else os.environ)
        .get(SUPERSEDE_FAILED_TRANSACTION_ENV, "")
        .strip()
    )


def _terminal_failed_transaction(state: dict[str, Any]) -> bool:
    """A fail-forward transaction that stopped: not mid-flight, not rolling back."""

    return str(state.get("phase") or "") in SUPERSEDABLE_PHASES


def _foreign_candidate(release: Any, state: dict[str, Any]) -> bool:
    return str(state.get("release_id") or "") != str(release.release_id)


def _uncommitted_bootstrap(release: Any, state: dict[str, Any]) -> bool:
    """A first bootstrap's ``complete`` state whose commit never happened.

    Only bootstrap writes ``previous: null``; every upgrade records the release
    it replaced. The same candidate cannot resume it as an upgrade (no diff or
    plan was recorded), and a foreign candidate takes the commit-live-release
    path instead.
    """

    return (
        _commit_pending(state)
        and "previous" in state
        and state.get("previous") is None
        and not _foreign_candidate(release, state)
    )


def _refuse_foreign_candidate_resume(release: Any, state: dict[str, Any]) -> None:
    """A failed fail-forward transaction only resumes the release that failed.

    Before this check the mismatch surfaced deep in the engine as
    ``resume release_id does not match the candidate``, which reads like a
    tooling bug and names neither release nor the way out; live on 2026-09-07
    the operator drove the engine from a hand-written script instead. The way
    out is the supersede flag, and the message says so.
    """

    if not (_terminal_failed_transaction(state) and _foreign_candidate(release, state)):
        return
    raise ReleaseError(
        foreign_candidate_resume_message(
            live_release_id=str(state.get("release_id") or ""),
            phase=str(state.get("phase") or ""),
            candidate_release_id=str(release.release_id),
        )
    )


def foreign_candidate_resume_message(
    *, live_release_id: str, phase: str, candidate_release_id: str
) -> str:
    """The refusal wording, shared with the deploy entry's first-minute check.

    ``gpu-fault-admin deploy`` reads the live phase and the candidate id as soon
    as the release is built and refuses there, before bootstrap re-validation
    and the rollout; the engine repeats the check as the last line of defence.
    Both say exactly this.
    """

    return (
        f"release {live_release_id or 'unknown'} stopped in phase "
        f"{phase or 'unknown'} and the candidate {candidate_release_id} is a "
        "different release; a failed fail-forward transaction only resumes the "
        "release that failed. To open a new transaction for the candidate on "
        "the last committed baseline, rerun the same deploy with "
        f"{SUPERSEDE_FAILED_TRANSACTION_FLAG} (the components the failed release "
        "already moved are re-rolled to the candidate)"
    )


def _require_supersede_target(release: Any, state: dict[str, Any] | None) -> None:
    """Refuse the supersede flag anywhere it does not apply, naming why."""

    if state is None:
        raise ReleaseError(
            f"{SUPERSEDE_FAILED_TRANSACTION_FLAG} requires a recorded release "
            "transaction; this site has no release state yet"
        )
    phase = str(state.get("phase") or "") or "unknown"
    if not _terminal_failed_transaction(state):
        if phase in ROLLBACK_PHASES or _rollback_pending(state):
            reason = "a rollback is in progress and must finish first"
        elif _upgrade_resume_required(state):
            reason = "the transaction is still resumable and belongs to no failure"
        else:
            reason = "there is no failed transaction to supersede"
        raise ReleaseError(
            f"{SUPERSEDE_FAILED_TRANSACTION_FLAG} only applies to a fail-forward "
            f"transaction in {'/'.join(sorted(SUPERSEDABLE_PHASES))}; the recorded "
            f"phase is {phase}: {reason}"
        )
    if not _foreign_candidate(release, state):
        raise ReleaseError(
            f"{SUPERSEDE_FAILED_TRANSACTION_FLAG} is for a different candidate; "
            f"{release.release_id} is the release that failed, so rerun deploy "
            "without the flag to resume it"
        )


def supersede_release_diff(release: Any, state: dict[str, Any]) -> ReleaseDiff:
    """The diff a superseding transaction rolls: everything the candidate changes
    against the committed baseline, plus everything the failed release moved.

    ``retry_release_diff`` is exactly that union. The failed state's top-level
    fields are the failed candidate's, so ``classify_release`` against it finds
    every component where the new candidate differs from the failed one; the
    persisted ``release_diff`` names every component the failed candidate
    differed from the committed baseline in, whether or not it got to move it.
    A component that differs between the new candidate and the committed
    baseline is in one of those two sets, and a component the failed release
    moved is in the second even when the new candidate matches the baseline
    there -- it has to be rolled back to the baseline value, which only a
    rollout of the new candidate does. The artifact checks against ``previous``
    add the physical wheel/bundle names the digest fields do not carry.
    """

    return retry_release_diff(release, state)


def next_deploy(release: Any, state: dict[str, Any]) -> dict[str, Any]:
    if (
        supersede_requested()
        and _terminal_failed_transaction(state)
        and _foreign_candidate(release, state)
    ):
        return {
            **supersede_release_diff(release, state).as_dict(),
            "resume": False,
            "action": "upgrade",
            "supersedes_release_id": state.get("release_id"),
        }
    if _rollback_pending(state):
        return {
            **retry_release_diff(release, state).as_dict(),
            "resume": True,
            "action": "rollback",
        }
    if _commit_cleanup_pending(state):
        return {
            **retry_release_diff(release, state).as_dict(),
            "resume": True,
            "action": "commit",
        }
    if _uncommitted_bootstrap(release, state):
        return {
            **retry_release_diff(release, state).as_dict(),
            "resume": False,
            "action": "commit",
        }
    if _commit_pending(state) and _foreign_candidate(release, state):
        committed = {**state, "transaction_committed": True}
        return {
            **classify_release(release, committed).as_dict(),
            "resume": False,
            "action": "upgrade",
            "commits_live_release_id": state.get("release_id"),
        }
    phase = str(state.get("phase") or "")
    if _upgrade_resume_required(state) or phase == "rolled-back":
        return {
            **retry_release_diff(release, state).as_dict(),
            "resume": _upgrade_resume_required(state),
            "action": "upgrade",
            # `complete` with an open transaction is a resume, but a resume of
            # one step: the commit. Everything is applied and quick validation
            # already passed, so the release driver may reuse the read-only
            # verifier evidence that deploy produced instead of re-running the
            # same probes a minute later. Naming the release the evidence has to
            # match is half of what makes that reuse safe; the other half is the
            # live-state digest `quick_validation_evidence` compares.
            **(
                {"pending_commit": True, "release_id": state.get("release_id")}
                if _commit_pending(state)
                else {}
            ),
        }
    return classify_release(release, state).as_dict()


def _require_expected_state(state: dict[str, Any]) -> None:
    expected = os.getenv(EXPECTED_STATE_SHA256_ENV, "").strip()
    if expected and release_state_sha256(state) != expected:
        raise ReleaseError(
            "regional release state changed after the deployment diff was calculated"
        )


def retry_release_diff(release: Any, state: dict[str, Any]) -> ReleaseDiff:
    persisted = stored_release_diff(state)
    changed = set(persisted.changed if persisted is not None else ())
    changed.update(classify_release(release, state).changed)
    previous = state.get("previous")
    if not isinstance(previous, dict):
        return diff_from_changed(changed)

    metadata = previous.get("metadata") or {}
    if previous.get("cpu_wheel") != release.wheel_cm:
        changed.add("control_plane_wheel")
    if (
        metadata.get("required-agent-artifact-sha256")
        and metadata.get("required-agent-artifact-sha256") != release.node_wheel_sha
    ):
        changed.add("node_runtime_wheel")
    if (
        metadata.get("required-regional-executor-artifact-sha256")
        and metadata.get("required-regional-executor-artifact-sha256")
        != release.executor_wheel_sha
    ):
        changed.add("executor_wheel")

    clusters = previous.get("clusters") or {}
    for target in release.config.clusters:
        old = clusters.get(target.cluster_id) or {}
        if any(
            old.get(name) and old.get(name) != release.executor_wheel_cm
            for name in ("wheel", "reconciler_wheel")
        ):
            changed.add("executor_wheel")
        if old.get("bundle") and old.get("bundle") != release.bundle_cm:
            changed.add("node_bundle")
    return diff_from_changed(changed)


def ensure_schema(release: Any) -> None:
    release.runner.run(
        [str(ROOT / "deploy/control-plane/tools/ensure-postgres-schema.sh")],
        env={
            **os.environ,
            "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": release.config.cpu_kubeconfig,
            "GPU_FAULT_NAMESPACE": release.config.namespace,
            "GPU_FAULT_WHEEL_CONFIGMAP": release.wheel_cm,
            "GPU_FAULT_RUNTIME_IMAGE": release.runtime_image,
        },
    )


def apply_rds_ca_bundle(release: Any) -> None:
    """Ship the pinned AWS RDS CA bundle ConfigMap into the CPU namespace (M-6).

    The control-plane DSN uses ``sslmode=verify-full`` (see
    ``gpu_fault.admin.bootstrap._aurora_dsn``), so the schema-ensure Job, the
    control-plane Deployment, the Aurora migration Jobs and the credential
    refresh CronJob all mount ``gpu-fault-rds-ca-bundle`` non-optionally and set
    ``GPU_FAULT_RDS_CA_BUNDLE`` at the mount path. This must run before any of
    those Aurora consumers, or they stay in ``CreateContainerConfigError`` --
    fail closed rather than fall back to an unverified connection. The bundle is
    fetched from the official AWS truststore and checked against the pin baked
    into the script; a digest mismatch aborts the deploy.
    """

    release.runner.run(
        ["bash", str(ROOT / "deploy/control-plane/tools/apply-rds-ca-bundle.sh")],
        env={
            **os.environ,
            "KUBECONFIG": release.config.cpu_kubeconfig,
            "GPU_FAULT_NAMESPACE": release.config.namespace,
        },
    )


def bootstrap_cpu_is_current(release: Any) -> bool:
    try:
        metadata = release._config_map_data("gpu-fault-release-metadata")
        expected = {
            "required-agent-artifact-sha256": release.node_wheel_sha,
            "required-agent-compatibility-digest": (
                release.config.component_digests.get("node_runtime")
                or release.node_wheel_sha
            ),
            "required-regional-executor-artifact-sha256": (release.executor_wheel_sha),
            "required-regional-executor-compatibility-digest": (
                release.config.component_digests.get("executor")
                or release.executor_wheel_sha
            ),
            "required-agent-config-digest": release.config.agent_config_digest,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if (
            release._deployment_wheel(
                release._cpu(),
                inventory.CPU_INGRESS_DEPLOYMENT,
            )
            != release.wheel_cm
        ):
            return False
        for deployment in inventory.CPU_DEPLOYMENTS:
            value = release._get_json(
                release._cpu(
                    "-n",
                    release.config.namespace,
                    "get",
                    "deployment",
                    deployment,
                )
            )
            desired = int((value.get("spec") or {}).get("replicas") or 0)
            status = value.get("status") or {}
            metadata_value = value.get("metadata") or {}
            if int(status.get("observedGeneration") or 0) < int(
                metadata_value.get("generation") or 0
            ):
                return False
            if any(
                int(status.get(field) or 0) != desired
                for field in (
                    "readyReplicas",
                    "updatedReplicas",
                    "availableReplicas",
                )
            ):
                return False
        return True
    except (ReleaseError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_deploy(release: Any) -> None:
    state_exists = release.runner.probe(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            STATE_CONFIG_MAP,
        ),
    )
    expected_state = os.getenv(EXPECTED_STATE_SHA256_ENV, "").strip()
    if expected_state and not state_exists:
        raise ReleaseError(
            "regional release state disappeared after the deployment diff was calculated"
        )
    state = release._load_state() if state_exists else None
    supersede = supersede_requested()
    if supersede:
        # The flag is consent to one specific thing; anywhere else it is refused
        # with the reason rather than ignored, so it cannot become a habit.
        _require_supersede_target(release, state)
    bootstrap_required = not state_exists or (
        state is not None and state.get("phase") in BOOTSTRAP_PHASES
    )
    if bootstrap_required:
        if not release.config.clusters:
            raise ReleaseError(
                "initial regional bootstrap requires at least one GPU cluster; "
                "an empty cluster set is only valid after a completed deployment"
            )
        # Resuming a partial bootstrap (state exists and its phase is a bootstrap
        # phase): pin the digest that bootstrap was first planned against so the
        # resumed apply refuses a working tree that drifted since it started
        # (M-23). A fresh bootstrap (no prior state) leaves the pin unset, and so
        # does a different candidate over a failed or interrupted attempt: that
        # is the fixed tree the operator reruns with, there is no baseline to
        # protect, and bootstrap re-plans and re-applies everything (live
        # 2026-09-12: the pin refused the fix and the refusal tore the partial
        # site down).
        if state is not None and not _foreign_candidate(release, state):
            release.pin_approved_manifest_plan(state.get("approved_manifest_sha256"))
        release.bootstrap()
        return
    assert state is not None
    _require_expected_state(state)
    phase = str(state.get("phase") or "")
    if _rollback_pending(state):
        release.rollback()
        raise ReleaseError(
            "rollback recovery completed; rerun deploy to start a new transaction"
        )
    if _commit_cleanup_pending(state):
        release.commit_release()
        return
    if _uncommitted_bootstrap(release, state):
        # The first bootstrap rolled out completely but its deploy stopped
        # before the commit (verify or stability failed, no baseline to roll
        # back to). There is no previous release to resume an upgrade against
        # -- bootstrap records neither a release diff nor an execution plan, so
        # `upgrade(resume=True)` would refuse -- and nothing left to apply: the
        # driver runs verify and commit again over this state.
        narrate_step(
            "bootstrap-complete",
            release_id=str(state.get("release_id") or ""),
            reason="complete but uncommitted with no previous release; nothing "
            "to re-apply, awaiting verify and commit",
        )
        return
    if _commit_pending(state) and _foreign_candidate(release, state):
        # The live release rolled out completely but its deploy stopped before
        # the commit (verify or stability failed, no rollback). Only that same
        # candidate could resume into the commit, and when the verify was the
        # defect it never could -- so a different candidate commits it as the
        # baseline and moves on (`commit_live_release`).
        narrate_step(
            "commit-live-release",
            release_id=str(state.get("release_id") or ""),
            reason="complete but uncommitted; committed as the baseline for "
            f"candidate {release.release_id}",
        )
        commit_live_release(release, state)
        state = release._load_state()
        phase = str(state.get("phase") or "")
    if supersede:
        # A new transaction for a different candidate over a stopped fail-forward
        # one: not a resume, so no plan pin (M-23 originates a fresh digest), a
        # baseline inherited from the failed transaction, and a diff that covers
        # everything the failed release moved (`supersede_release_diff`).
        release.upgrade(
            resume=False,
            diff=supersede_release_diff(release, state),
            supersede=state,
        )
        return
    _refuse_foreign_candidate_resume(release, state)
    if _upgrade_resume_required(state) or phase == "rolled-back":
        resume_required = _upgrade_resume_required(state)
        # Only a genuine resume pins the approved plan (M-23): the persisted
        # digest belongs to the in-progress transaction, so enforcing it on the
        # resumed apply refuses a tree that drifted since the plan. A bare
        # "rolled-back" re-entry renders a different (previous) manifest set, so
        # it must not be pinned to the interrupted transaction's digest.
        if resume_required:
            release.pin_approved_manifest_plan(state.get("approved_manifest_sha256"))
        release.upgrade(
            resume=resume_required,
            diff=retry_release_diff(release, state),
        )
        return
    diff = classify_release(release, state)
    if diff.kind == ReleaseChangeKind.NOOP:
        release.noop(diff)
        return
    release.upgrade(diff=diff)


def build_release_diff(release: Any) -> dict[str, Any]:
    state = release._load_state()
    return {
        "mode": "release-diff",
        "state_sha256": release_state_sha256(state),
        "next_deploy": next_deploy(release, state),
    }


def stage_noop_release(release: Any) -> None:
    state = release._load_state()
    _require_expected_state(state)
    diff = classify_release(release, state)
    if diff.kind is not ReleaseChangeKind.NOOP:
        raise ReleaseError(
            f"stage-noop requires a current NOOP release diff; got {diff.kind.value}"
        )
    release._ensure_contexts()
    release._require_cpu_secrets()
    # The admin `deploy` fast path lands here for an unchanged release, so this
    # is where GpuFaultCompletionActiveStateUnavailable is recovered from: the
    # watcher's missing state ConfigMap and its ClusterRole come back without
    # anyone applying the unrendered manifest by hand.
    for target in release.config.clusters:
        release._reassert_completion_watcher_state(target)
    release.state = dict(state)
    release._save_state(
        "complete",
        transaction_committed=False,
        previous=None,
        release_diff=diff.as_dict(),
        execution_plan=build_execution_plan(diff).as_dict(),
        completed_phases=[],
        completed_cluster_ids=[],
    )


def run_resume(release: Any) -> None:
    state = release._load_state()
    phase = str(state.get("phase") or "")
    if _rollback_pending(state):
        release.rollback()
        return
    if _commit_cleanup_pending(state):
        release.commit_release()
        return
    if not _upgrade_resume_required(state):
        raise ReleaseError(
            "resume requires an incomplete upgrade or rollback transaction; "
            f"current phase is {phase or 'unknown'}"
        )
    # `resume` is the low-level same-release command; a different candidate is
    # refused here with the deploy flag named rather than by the engine's
    # release_id mismatch, which reads like a tooling bug.
    _refuse_foreign_candidate_resume(release, state)
    # Resume enforces the digest the transaction was planned against (M-23), so
    # a working tree edited between the interrupted apply and this resume is
    # refused rather than silently applied.
    release.pin_approved_manifest_plan(state.get("approved_manifest_sha256"))
    release.upgrade(
        resume=True,
        diff=retry_release_diff(release, state),
    )


def build_release_summary(
    release: Any, *, state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The configured release beside the live one and what a deploy would do.

    ``state`` is the live release state when the caller has already read it:
    ``status`` reads the ConfigMap once for the health baseline, the summary
    and the snapshot together, where it used to read it three times.
    """

    result: dict[str, Any]
    try:
        result = build_release_status(release)
    except Exception as exc:
        result = {
            "site_name": release.config.site_name,
            "release_status_error": str(exc),
        }
    result["mode"] = "release-summary"
    try:
        if state is None:
            state = release._load_state()
        result["live_release"] = {
            "release_id": state.get("release_id"),
            "phase": state.get("phase"),
            "transaction_committed": state.get("transaction_committed") is True,
            "release_lifecycle": state.get("release_lifecycle"),
            "state_sha256": release_state_sha256(state),
            # Present only for a transaction that crossed a schema version with
            # --accept-schema-change: the operator reads the snapshot id here.
            "schema_change_acceptance": state.get("schema_change_acceptance"),
        }
        result["next_deploy"] = next_deploy(release, state)
    except Exception as exc:
        result["next_deploy_error"] = str(exc)
    return result


def build_full_status(
    release: Any, *, state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The full ``status`` report: every health check plus the release summary."""

    # One snapshot for both halves. `build_health_report` opens its own, and
    # everything `build_release_summary` reads -- the release state ConfigMap,
    # the live Deployments behind `next_deploy` -- the health checks have already
    # read inside it, so without this the summary re-issued each of those
    # `kubectl get` calls after the health report's snapshot had been torn down.
    # It also means the summary describes the same observation as the checks
    # reported beside it, which is the whole point of a status output.
    with read_snapshot(release):
        health = build_health_report(release, mode="status")
        result = build_release_summary(release, state=state)
    return _status_result(result, health, scope="full")


def build_status(release: Any, *, full: bool) -> dict[str, Any]:
    """The ``status`` mode: one snapshot, one state read, quick or full.

    The snapshot opens first so the single release-state read serves the
    rolled-back health baseline, the summary and every check inside it. Before,
    the baseline read the ConfigMap outside the snapshot and the summary read it
    again inside.
    """

    with read_snapshot(release):
        state = release._load_state()
        release._apply_health_baseline(state)
        if full:
            return build_full_status(release, state=state)
        return build_quick_status(release, state=state)


def build_quick_status(
    release: Any, *, state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The default ``status`` report: the release summary plus the cheap checks.

    Same document shape as :func:`build_full_status` -- ``mode``, ``healthy``,
    ``live_release``, ``configured_release``, ``next_deploy``, ``health`` -- so
    every reader of the JSON works on either. Only ``health.checks`` is shorter,
    and ``health_scope`` says which report this is.
    """

    with read_snapshot(release):
        health = build_quick_health_report(release)
        result = build_release_summary(release, state=state)
    return _status_result(result, health, scope="quick")


def _status_result(
    result: dict[str, Any], health: dict[str, Any], *, scope: str
) -> dict[str, Any]:
    result["mode"] = "status"
    result["healthy"] = health["healthy"]
    result["health"] = health
    result["health_scope"] = scope
    return result


def failing_checks(report: dict[str, Any]) -> list[str]:
    """Names of the checks in a health or status report that did not pass."""

    health = report.get("health")
    checks = (health if isinstance(health, dict) else report).get("checks")
    return [
        str(item.get("name"))
        for item in (checks if isinstance(checks, list) else [])
        if isinstance(item, dict) and item.get("status") not in ("PASS", "SKIP")
    ]


def status_header_lines(report: dict[str, Any], *, json_destination: str) -> list[str]:
    """The five lines an administrator reads before the JSON.

    The status document runs past a thousand lines with ``healthy`` sorted to
    the bottom. These name the verdict, the live release, what the next deploy
    would do and which checks failed, then say where the whole document went.
    """

    live = report.get("live_release")
    live = live if isinstance(live, dict) else {}
    next_deploy = report.get("next_deploy")
    next_deploy = next_deploy if isinstance(next_deploy, dict) else {}
    health = report.get("health")
    summary = (health if isinstance(health, dict) else {}).get("summary")
    total = sum(summary.values()) if isinstance(summary, dict) else 0
    failing = failing_checks(report)
    next_kind = str(
        next_deploy.get("kind") or report.get("next_deploy_error") or "unknown"
    )
    if next_deploy.get("resume"):
        next_kind += f" (resume {next_deploy.get('action')})"
    return [
        f"healthy: {'yes' if report.get('healthy') else 'NO'}"
        f" ({report.get('health_scope') or 'full'} health, {total} checks)",
        f"live release: {live.get('release_id') or 'unknown'}"
        f" phase={live.get('phase') or 'unknown'}"
        f" committed={'yes' if live.get('transaction_committed') else 'no'}",
        f"next deploy: {next_kind}",
        "failing checks: " + (", ".join(failing) if failing else "none"),
        f"full JSON report: {json_destination}",
    ]


FULL_REPORT_ENV = "GPU_FAULT_FULL_REPORT"


def full_report_requested(environment: dict[str, str] | None = None) -> bool:
    value = (environment if environment is not None else os.environ).get(
        FULL_REPORT_ENV, ""
    )
    return value.strip().lower() in ("1", "true", "yes")


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    """A health report with its passing checks folded to their names.

    A passing preflight used to print about 250 lines of check details in the
    middle of a deploy that nothing reads. Failures keep their details; the
    full document is one ``--full`` (or ``GPU_FAULT_FULL_REPORT=1``) away.
    """

    checks = [item for item in (report.get("checks") or []) if isinstance(item, dict)]
    return {
        **report,
        "checks": [item for item in checks if item.get("status") != "PASS"],
        "passed_checks": [
            str(item.get("name")) for item in checks if item.get("status") == "PASS"
        ],
    }
