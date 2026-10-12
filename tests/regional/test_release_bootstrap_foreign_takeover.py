"""A new candidate bootstraps over another candidate's journal-less bootstrap.

A bootstrap that adopted its Aurora repair keeps no journal; the next
candidate finds only the old attempt's failed, interrupted or cleaned state,
stamped with the old candidate's identity and approved manifest digest. The
deploy preflight of the next candidate pinned that old digest, although M-23
pins only the same candidate's resumed plan, and died before any mutation
with ``release manifests changed between plan and apply``. With the pin alone
corrected, the repair journal hashed the old identity, the new candidate's
first bootstrap checkpoint re-stamped it, and adoption refused with ``Aurora
prerequisite repair baseline drifted`` over a state nothing could resume.

Both processes of the deploy run here, each starting unpinned: the preflight
(``build_deploy_preflight_report``) and the deploy (``run_deploy`` into
``RegionalRelease.bootstrap``), with the real manifest pin enforcement.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import regional_release_bootstrap as BOOTSTRAP
from gpu_fault_release import regional_release_prerequisite_handoff as HANDOFF
from gpu_fault_release import regional_release_prerequisite_repair as REPAIR
from gpu_fault_release import regional_release_rendering as RENDERING
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_prerequisite_baseline import (
    validated_prerequisite_baseline,
)
from gpu_fault_release.regional_release_progress import BOOTSTRAP_PHASES
from tests.regional import test_release_bootstrap_handoff as handoff_support
from tests.regional.test_release_bootstrap_handoff import (
    CLEANUP_STEPS,
    MUTATIONS,
    select_candidate,
)

handoff_release = handoff_support.handoff_release

PAYLOAD = {"cpu": {"gpu-fault-api": [{"kind": "Deployment", "candidate": "b"}]}}
OLD_DIGEST = "a" * 64
CLEANUP_PHASES = {
    "bootstrap-cleanup-started",
    "bootstrap-cleanup-progress",
    "bootstrap-cleanup-failed",
    "bootstrap-cleaned",
}
CLEANUP_PROGRESS = {
    "bootstrap-cleanup-started": [],
    "bootstrap-cleanup-progress": ["installer-jobs-cancelled"],
    "bootstrap-cleanup-failed": ["gpu-scaled-down", "installer-jobs-cancelled"],
    "bootstrap-cleaned": CLEANUP_STEPS,
}
# Every bootstrap phase an adopted attempt can stop in carries its digest.
STALE_PHASES = sorted(BOOTSTRAP_PHASES)


def real_manifest_pin(instance: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    """Route the pin through ``render_release_payload``; return the tree digest."""
    monkeypatch.setattr(
        RENDERING, "_render_release_payload", lambda _release: copy.deepcopy(PAYLOAD)
    )
    monkeypatch.setattr(
        instance,
        "enforce_manifest_plan_pin",
        lambda: rollout.RegionalRelease.enforce_manifest_plan_pin(instance),
    )
    monkeypatch.setattr(
        instance,
        "pin_approved_manifest_plan",
        lambda digest: rollout.RegionalRelease.pin_approved_manifest_plan(
            instance, digest
        ),
    )
    return RENDERING.rendered_release_manifest_sha256(
        SimpleNamespace(approved_manifest_digest=None)
    )


def stub_bootstrap(instance: Any, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Everything after the prerequisite gates records instead of acting."""
    events = instance.runner.events
    gpu_calls: list[Any] = []
    run = instance.runner.run

    def bootstrap_run(arguments: list[str], **kwargs: Any) -> str:
        text = kwargs.get("input_text")
        if "apply" in arguments and text and "kind: Namespace" in text:
            events.append("namespace-prerequisites")
            return ""
        return str(run(arguments, **kwargs))

    def cpu_is_current() -> bool:
        events.append("live-cpu-check")
        return True

    monkeypatch.setattr(instance.runner, "run", bootstrap_run)
    monkeypatch.setattr(rollout, "ensure_runtime_profile", lambda _release: None)
    monkeypatch.setattr(
        rollout,
        "bootstrap_gpu_clusters",
        lambda _release, completed: gpu_calls.append(sorted(completed)),
    )
    monkeypatch.setattr(
        ADMIN,
        "build_preflight_report",
        lambda *_args, **_kwargs: {"checks": [], "healthy": True},
    )
    monkeypatch.delenv(ADMIN.EXPECTED_STATE_SHA256_ENV, raising=False)
    monkeypatch.delenv(ADMIN.SUPERSEDE_FAILED_TRANSACTION_ENV, raising=False)
    for name in ("_initialize_registry", "_prepare_nlb", "_wait_nlb"):
        monkeypatch.setattr(instance, name, lambda: None, raising=False)
    monkeypatch.setattr(
        instance, "_validate_release", lambda: events.append("verify"), raising=False
    )
    monkeypatch.setattr(instance, "_require_cpu_secrets", lambda **_kwargs: None)
    monkeypatch.setattr(instance, "_upload_release", lambda _diff=None: None)
    monkeypatch.setattr(
        instance, "_bootstrap_cpu_is_current", cpu_is_current, raising=False
    )
    monkeypatch.setattr(
        instance,
        "_prepare_bootstrap_workflows",
        lambda: REPAIR.prepare_bootstrap_workflows(instance),
        raising=False,
    )
    monkeypatch.setattr(
        instance,
        "_cleanup_bootstrap",
        lambda: BOOTSTRAP.cleanup_bootstrap(instance),
        raising=False,
    )
    monkeypatch.setattr(instance, "_gpu", lambda _target: ["kubectl"], raising=False)
    monkeypatch.setattr(
        instance,
        "_cancel_active_installer_jobs",
        lambda _target: events.append("cancel-installers"),
        raising=False,
    )
    monkeypatch.setattr(
        instance,
        "bootstrap",
        lambda: rollout.RegionalRelease.bootstrap(instance),
        raising=False,
    )
    return gpu_calls


def adopted_predecessor(instance: Any, phase: str) -> dict[str, Any]:
    """The old candidate adopted its repair, rolled out and stopped in ``phase``."""
    REPAIR.prepare_bootstrap_workflows(instance)
    assert REPAIR.REPAIR_KEY not in instance.state, "adoption kept the old journal"
    instance.runner.database_state = "initialized"
    cleanup = phase in CLEANUP_PHASES
    STATE.save_state(
        instance,
        phase,
        previous=None,
        transaction_committed=False,
        completed_cluster_ids=(
            [] if cleanup else [instance.config.clusters[0].cluster_id]
        ),
        bootstrap_cleanup_completed_steps=CLEANUP_PROGRESS.get(phase, []),
        bootstrap_cleanup_failure=(
            "ReleaseError: scale-down timed out"
            if phase == "bootstrap-cleanup-failed"
            else None
        ),
        resume_phase="bootstrap-started" if cleanup else "bootstrap-endpoint-ready",
    )
    instance.runner.events.clear()
    old = copy.deepcopy(instance.runner.cloud_state)
    assert REPAIR.REPAIR_KEY not in old and old["release_id"] == "candidate", (
        "the predecessor state is not the old candidate's journal-less state"
    )
    assert old["approved_manifest_sha256"] == OLD_DIGEST, (
        "the predecessor state does not carry the old candidate's approved digest"
    )
    return old


def deploy_preflight(instance: Any) -> dict[str, Any]:
    """``rollout-regional-release.sh preflight --for-deploy`` as its own process."""
    instance.approved_manifest_digest = None
    report = ADMIN.build_deploy_preflight_report(instance)
    assert report["healthy"] is True, f"deploy preflight failed: {report['checks']}"
    return report


def deploy(instance: Any) -> None:
    """``rollout-regional-release.sh deploy`` as the next, unpinned process."""
    instance.approved_manifest_digest = None
    ADMIN.run_deploy(instance)


@pytest.mark.parametrize("auto_rollback", [False, True])
@pytest.mark.parametrize("phase", STALE_PHASES)
def test_new_candidate_bootstraps_over_adopted_predecessor_state(
    handoff_release: Any,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    auto_rollback: bool,
) -> None:
    instance = handoff_release
    gpu_calls = stub_bootstrap(instance, monkeypatch)
    old = adopted_predecessor(instance, phase)
    instance.config.auto_rollback = auto_rollback
    select_candidate(instance)
    digest = real_manifest_pin(instance, monkeypatch)
    instance.rendered_manifest_digest = digest
    assert digest != OLD_DIGEST, "the two candidates must render different trees"

    deploy_preflight(instance)
    assert instance.approved_manifest_digest == digest, (
        "the new candidate's preflight pinned another candidate's approved digest"
    )
    taken_over = copy.deepcopy(instance.runner.cloud_state)
    record = taken_over[REPAIR.REPAIR_KEY]
    assert (
        taken_over["release_id"] == "candidate-b"
        and taken_over["approved_manifest_sha256"] == digest
        and taken_over["rendered_manifest_sha256"] == digest
    ), "the takeover checkpoint kept the old candidate's identity or digests"
    assert record["predecessor"]["release_id"] == "candidate" and (
        record["predecessor"]["approved_manifest_sha256"] == OLD_DIGEST
    ), "the takeover journal lost the audit of the state it replaced"
    baseline = validated_prerequisite_baseline(taken_over, record)
    assert baseline["completed_cluster_ids"] == [] and (
        baseline["resume_phase"] == "bootstrap-started"
    ), "the takeover kept the old attempt's completed clusters or checkpoint"
    owed = phase in CLEANUP_PHASES or (phase == "bootstrap-failed")
    assert baseline["phase"] == (phase if owed else "bootstrap-started"), (
        "the takeover dropped the cleanup the old attempt still owes"
    )
    assert rollout.bootstrap_resume_context(taken_over, "candidate-b") == (
        set(),
        False,
        False,
    ), "the old attempt's progress counted as the new candidate's"

    mark = len(instance.runner.snapshots)
    deploy(instance)
    written = instance.runner.snapshots[mark:]
    assert written and all(
        value["release_id"] == "candidate-b"
        and value["approved_manifest_sha256"] == digest
        and value["rendered_manifest_sha256"] == digest
        for value in written
    ), "a bootstrap checkpoint persisted the old candidate's identity or digests"
    final = instance.runner.cloud_state
    assert final["phase"] == "complete" and REPAIR.REPAIR_KEY not in final, (
        f"the new candidate's bootstrap did not complete: {final.get('phase')}"
    )
    assert final[HANDOFF.PREDECESSOR_AUDIT_KEY] == record["predecessor"], (
        "adoption discarded the audit of the state the takeover replaced"
    )
    assert final["bootstrap_database_origin"] == old["bootstrap_database_origin"], (
        "the empty-database origin of the first bootstrap was not carried forward"
    )
    events = instance.runner.events
    assert "live-cpu-check" not in events, (
        "the old candidate's CPU was trusted as this candidate's checkpoint"
    )
    assert {"schema", "cpu-finalize"} <= set(events) and gpu_calls == [[]], (
        "the new candidate skipped its schema, CPU or GPU bootstrap"
    )
    cleaned = [value["phase"] for value in written].count("bootstrap-cleaned")
    expected_cleanup = phase in CLEANUP_PHASES - {"bootstrap-cleaned"} or (
        phase == "bootstrap-failed" and auto_rollback
    )
    assert cleaned == int(expected_cleanup), (
        "the cleanup the old attempt owed did not run exactly once"
    )


@pytest.mark.parametrize("phase", ["bootstrap-cleaned", "bootstrap-failed"])
def test_same_candidate_resume_still_pins_its_approved_plan(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    instance = handoff_release
    stub_bootstrap(instance, monkeypatch)
    adopted_predecessor(instance, phase)
    instance.rendered_manifest_digest = real_manifest_pin(instance, monkeypatch)
    before = copy.deepcopy(instance.runner.cloud_state)
    with pytest.raises(ReleaseError, match="changed between plan and apply"):
        deploy_preflight(instance)
    assert instance.approved_manifest_digest == OLD_DIGEST, (
        "the same candidate's resume did not pin the digest it was planned against"
    )
    assert instance.runner.cloud_state == before, (
        "a drifted resume rewrote the release state"
    )
    assert not (MUTATIONS & set(instance.runner.events)), (
        "a drifted resume reached a repair or bootstrap mutation"
    )


def test_takeover_checkpoint_failure_keeps_the_predecessor_state(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = handoff_release
    stub_bootstrap(instance, monkeypatch)
    old = adopted_predecessor(instance, "bootstrap-cleaned")
    select_candidate(instance)
    instance.rendered_manifest_digest = real_manifest_pin(instance, monkeypatch)

    def lost_checkpoint(*_args: Any, **_kwargs: Any) -> None:
        raise ReleaseError("state ConfigMap write failed")

    monkeypatch.setattr(REPAIR, "save_recorded_state", lost_checkpoint)
    with pytest.raises(ReleaseError, match="state ConfigMap write failed"):
        deploy_preflight(instance)
    assert instance.state == old and instance.runner.cloud_state == old, (
        "a failed takeover checkpoint left a half-projected state behind"
    )


@pytest.mark.parametrize(
    ("phase", "projected"),
    [
        ("bootstrap-started", "bootstrap-started"),
        ("bootstrap-cpu-ready", "bootstrap-started"),
        ("bootstrap-data-plane-progress", "bootstrap-started"),
        ("complete", "bootstrap-started"),
        ("bootstrap-failed", "bootstrap-failed"),
        ("bootstrap-cleanup-progress", "bootstrap-cleanup-progress"),
        ("bootstrap-cleanup-failed", "bootstrap-cleanup-failed"),
        ("bootstrap-cleaned", "bootstrap-cleaned"),
    ],
)
def test_takeover_projection_keeps_only_owed_cleanup(
    handoff_release: Any, phase: str, projected: str
) -> None:
    instance = handoff_release
    old = adopted_predecessor(
        instance, "bootstrap-started" if phase == "complete" else phase
    )
    if phase == "complete":
        old.update(phase="complete", transaction_committed=True)
    select_candidate(instance)
    events = list(instance.runner.events)
    baseline = HANDOFF.foreign_bootstrap_baseline(instance, old)
    owed = projected != "bootstrap-started"
    assert baseline["phase"] == projected and (
        baseline["bootstrap_cleanup_completed_steps"]
        == (old["bootstrap_cleanup_completed_steps"] if owed else [])
        and baseline.get("bootstrap_cleanup_failure")
        == (old.get("bootstrap_cleanup_failure") if owed else None)
    ), "the takeover projection mis-kept the old attempt's cleanup progress"
    assert (
        baseline["completed_cluster_ids"] == []
        and baseline["resume_phase"] == "bootstrap-started"
        and baseline["transaction_committed"] is False
    ), "the takeover projection kept the old attempt's rollout progress"
    assert baseline["release_id"] == "candidate-b" and (
        baseline["approved_manifest_sha256"] == instance.rendered_manifest_digest
    ), "the takeover projection kept the old candidate's identity"
    assert baseline["bootstrap_database_origin"] == old["bootstrap_database_origin"], (
        "the takeover projection dropped the first bootstrap's database origin"
    )
    assert instance.runner.events == events and instance.runner.dry_run is False, (
        "the takeover projection issued a command or disabled the live runner"
    )


def test_takeover_refuses_inconsistent_predecessor_progress(
    handoff_release: Any,
) -> None:
    instance = handoff_release
    old = adopted_predecessor(instance, "bootstrap-cleaned")
    old["bootstrap_cleanup_completed_steps"] = ["gpu-scaled-down"]
    select_candidate(instance)
    with pytest.raises(ReleaseError, match="predecessor progress is invalid"):
        HANDOFF.foreign_bootstrap_baseline(instance, old)


def test_takeover_refuses_a_predecessor_with_a_previous_application(
    handoff_release: Any,
) -> None:
    instance = handoff_release
    old = adopted_predecessor(instance, "bootstrap-cleaned")
    old["previous"] = {"release_id": "older"}
    select_candidate(instance)
    with pytest.raises(ReleaseError, match="previous application"):
        HANDOFF.foreign_bootstrap_baseline(instance, old)


@pytest.mark.parametrize("marker", ["supersedes", "predecessor"])
def test_taken_over_journal_lends_no_progress_until_adopted(marker: str) -> None:
    state = {
        "phase": "bootstrap-failed",
        "resume_phase": "bootstrap-cpu-ready",
        "release_id": "candidate-b",
        "completed_cluster_ids": ["gpu-a"],
    }
    own = rollout.bootstrap_resume_context(
        {**state, REPAIR.REPAIR_KEY: {"status": "READY"}}, "candidate-b"
    )
    assert own == ({"gpu-a"}, True, False), (
        "the same candidate's own unadopted journal stopped its checkpoint resume"
    )
    taken = rollout.bootstrap_resume_context(
        {**state, REPAIR.REPAIR_KEY: {"status": "READY", marker: {}}}, "candidate-b"
    )
    assert taken == (set(), False, False), (
        "a journal that took over another bootstrap lent that bootstrap's progress"
    )
    live = rollout.bootstrap_resume_context(
        {
            **state,
            "resume_phase": "bootstrap-started",
            REPAIR.REPAIR_KEY: {"status": "READY", marker: {}},
        },
        "candidate-b",
    )
    assert live == (set(), False, False), (
        "a taken-over bootstrap trusted the live CPU of the candidate it replaced"
    )
