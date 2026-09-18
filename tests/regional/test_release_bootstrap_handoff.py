"""Mocked bootstrap repair handoffs preserve both sides of the transaction."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_prerequisite_repair as REPAIR
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_aurora_refresh import resource_argument
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_diff import (
    build_execution_plan,
    diff_from_changed,
)
from gpu_fault_release.regional_release_prerequisite_baseline import (
    capture_baseline_representation,
    validated_prerequisite_baseline,
)
from gpu_fault_release.regional_release_prerequisite_handoff import (
    HANDOFF_AUDIT_KEY,
    candidate_bootstrap_baseline,
)
from tests.regional._prerequisite_repair_support import repair_release
from tests.regional._release_orchestrator_support import config_file
from tests.regional.test_release_aurora_refresh_transaction import objects

DIFF = diff_from_changed({"aurora_refresh_manifests"})
PLAN = build_execution_plan(DIFF)
MUTATIONS = {"persist", "apply", "delete", "job-create", "job-delete"}
CLEANUP_STEPS = ["cpu-scaled-down", "gpu-scaled-down", "installer-jobs-cancelled"]


def prepare(instance: Any, **options: Any) -> dict[str, Any]:
    result = REPAIR.prepare_prerequisite_repair(
        instance, diff=DIFF, plan=PLAN, bootstrap=True, **options
    )
    assert isinstance(result, dict), "an enabled repair did not return its journal"
    return result


def select_candidate(instance: Any) -> None:
    instance.release_id = "candidate-b"
    instance.runtime_image = "registry/cpu@sha256:" + "3" * 64
    instance.executor_image = "registry/executor@sha256:" + "4" * 64
    instance.wheel_sha = "5" * 64
    instance.wheel_cm = "gpu-fault-control-plane-wheel-0100-" + instance.wheel_sha[:12]
    instance.rendered_manifest_digest = "6" * 64
    instance.approved_manifest_digest = None


def cleaned(instance: Any) -> None:
    REPAIR.save_recorded_state(
        instance,
        "bootstrap-cleaned",
        previous=None,
        transaction_committed=False,
        completed_cluster_ids=[],
        bootstrap_cleanup_completed_steps=CLEANUP_STEPS,
        bootstrap_cleanup_failure=None,
        resume_phase="bootstrap-started",
    )


@pytest.fixture
def handoff_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Any:
    prototype = rollout.RegionalRelease(
        rollout.ReleaseConfig.load(config_file(tmp_path)), rollout.Runner(dry_run=True)
    )
    instance = repair_release(monkeypatch, bootstrap=True)
    instance.config = SimpleNamespace(
        **(vars(prototype.config) | vars(instance.config))
    )
    instance.config.clusters = prototype.config.clusters
    for name, value in vars(prototype).items():
        if not hasattr(instance, name):
            setattr(instance, name, value)
    instance.executor_image = prototype.executor_image
    instance.wheel_cm = "gpu-fault-control-plane-wheel-0100-" + instance.wheel_sha[:12]
    runner = instance.runner
    if getattr(request, "param", False):
        runner.live = {
            resource_argument(item["apiVersion"], item["kind"]): item
            for item in objects()
        }
    original_run = runner.run

    def run(args: list[str], **kwargs: Any) -> str:
        if "apply" in args and kwargs.get("input_text"):
            try:
                document = json.loads(kwargs["input_text"])
            except ValueError:
                document = {}
            if (
                document.get("kind") == "ConfigMap"
                and document["metadata"]["name"] == STATE.STATE_CONFIG_MAP
            ):
                runner.cloud_state = json.loads(document["data"]["state.json"])
                runner.snapshots.append(copy.deepcopy(runner.cloud_state))
                runner.events.append("persist")
                return ""
        return str(original_run(args, **kwargs))

    def enforce_pin() -> None:
        runner.events.append("approval")
        if instance.approved_manifest_digest != instance.rendered_manifest_digest:
            raise ReleaseError("approved manifest differs")

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(STATE, "record_release_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(STATE, "narrate_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(instance, "enforce_manifest_plan_pin", enforce_pin)
    monkeypatch.setattr(
        instance,
        "_save_state",
        lambda phase, **updates: STATE.save_state(instance, phase, **updates),
    )
    prepare(instance)
    REPAIR.bootstrap_workflow_proof(instance)
    cleaned(instance)
    runner.events.clear()
    return instance


def test_new_candidate_restores_original_before_install_and_reproves(
    handoff_release: Any,
) -> None:
    instance = handoff_release
    original = copy.deepcopy(instance.state[REPAIR.REPAIR_KEY])
    select_candidate(instance)
    replacement = prepare(instance)
    proof = REPAIR.bootstrap_workflow_proof(instance)
    events = instance.runner.events
    assert events.index("delete") < events.index("apply"), (
        "candidate resources were applied before original absence was restored"
    )
    assert replacement["previous_refresher"] == original["previous_refresher"], (
        "handoff recaptured the old candidate as its original snapshot"
    )
    assert replacement["attempt_id"] != original["attempt_id"], (
        "handoff rebound the old attempt instead of creating a new journal"
    )
    assert replacement["binding"]["bootstrap"] is True, (
        "the new journal did not identify its bootstrap scope"
    )
    assert replacement["binding"]["runtime_image"] == instance.runtime_image, (
        "the successor retained its predecessor's image binding"
    )
    assert replacement["supersedes"]["attempt_id"] == original["attempt_id"], (
        "the successor lost its predecessor audit link"
    )
    assert replacement["supersedes"]["record_sha256"] == canonical_sha256(original), (
        "the predecessor audit digest was not of the fully cleaned original journal"
    )
    baseline = validated_prerequisite_baseline(instance.state, replacement)
    assert baseline["release_id"] == instance.release_id, (
        "new identity and new journal were not committed as one baseline"
    )
    assert baseline["executor_image"] == instance.executor_image, (
        "handoff projected only CPU fields rather than the complete candidate identity"
    )
    assert instance.approved_manifest_digest == instance.rendered_manifest_digest, (
        "a proven new candidate inherited the failed bootstrap's old plan pin"
    )
    assert events.count("credential-job") == events.count("store-job") == 1, (
        "handoff reused the predecessor's credential or database proof"
    )
    assert proof["safe"] is True and instance.runner.jobs == {}, (
        "the fresh proof failed or left owned Jobs behind"
    )
    assert not ({"schema", "cpu-finalize", "registry", "endpoint"} & set(events)), (
        "the prerequisite handoff performed an application mutation"
    )


def test_handoff_projects_state_without_io_or_mutating_original(
    handoff_release: Any,
) -> None:
    instance = handoff_release
    select_candidate(instance)
    original = copy.deepcopy(instance.state)
    events = list(instance.runner.events)
    projected = candidate_bootstrap_baseline(instance, instance.state)
    assert instance.state == original and instance.runner.dry_run is False, (
        "candidate projection mutated the original state or disabled its live runner"
    )
    assert instance.runner.events == events, "candidate projection issued a command"
    assert (
        REPAIR.REPAIR_KEY not in projected and projected["phase"] == "bootstrap-started"
    ), "candidate projection kept an active predecessor or the old cleanup phase"


def test_bootstrap_adoption_keeps_handoff_audit(handoff_release: Any) -> None:
    instance = handoff_release
    select_candidate(instance)
    replacement = prepare(instance)
    audit = copy.deepcopy(replacement["supersedes"])
    REPAIR.prepare_bootstrap_workflows(instance)
    assert instance.state[HANDOFF_AUDIT_KEY] == audit, (
        "bootstrap adoption discarded the predecessor restoration audit"
    )
    assert REPAIR.REPAIR_KEY not in instance.state, (
        "bootstrap adoption left the prerequisite independently active"
    )
    assert instance.state["bootstrap_store_safety"]["safe"] is True, (
        "bootstrap was adopted without a fresh Store proof"
    )


@pytest.mark.parametrize("explicit_pin", [False, True])
def test_same_release_drift_or_new_explicit_pin_mismatch_still_refuses(
    handoff_release: Any, explicit_pin: bool
) -> None:
    instance = handoff_release
    old_pin = instance.state["approved_manifest_sha256"]
    if explicit_pin:
        select_candidate(instance)
        instance.approved_manifest_digest = old_pin
    else:
        instance.rendered_manifest_digest = "7" * 64
        instance.approved_manifest_digest = None
    with pytest.raises(ReleaseError, match="approved manifest"):
        prepare(instance)
    assert not (MUTATIONS & set(instance.runner.events)), (
        "an unapproved manifest reached the handoff mutation path"
    )


@pytest.mark.parametrize(
    "field", ["runtime_image", "wheel_config_map", "manifest_sha256"]
)
def test_self_consistent_binding_must_still_match_original_baseline(
    handoff_release: Any, field: str
) -> None:
    instance = handoff_release
    record = instance.state[REPAIR.REPAIR_KEY]
    record["binding"][field] = "different"
    record["binding_sha256"] = canonical_sha256(record["binding"])
    select_candidate(instance)
    with pytest.raises(ReleaseError, match="predecessor identity"):
        prepare(instance)
    assert not (MUTATIONS & set(instance.runner.events)), (
        "a self-consistent but foreign candidate binding authorized restoration"
    )


@pytest.mark.parametrize(
    "damage",
    [
        "namespace",
        "database",
        "cpu",
        "members",
        "schema",
        "phase",
        "cleanup",
        "previous",
        "committed",
        "adopted",
        "completed",
        "unknown",
        "snapshot",
    ],
)
def test_scope_progress_or_integrity_drift_blocks_handoff(
    handoff_release: Any, damage: str
) -> None:
    instance = handoff_release
    if damage == "namespace":
        instance.runner.namespace_uid = "replacement-namespace"
    elif damage == "database":
        instance.runner.database_resource_id = "replacement-database"
    elif damage == "cpu":
        instance.config.cpu_eks_arn += "-other"
    elif damage == "members":
        instance.config.clusters = ()
    elif damage == "schema":
        instance.config.database_schema_version += 1
    elif damage == "phase":
        instance.state["phase"] = "bootstrap-cpu-ready"
    elif damage == "cleanup":
        instance.state["bootstrap_cleanup_completed_steps"] = []
    elif damage == "previous":
        instance.state["previous"] = {"release_id": "foreign"}
    elif damage == "committed":
        instance.state["transaction_committed"] = True
    elif damage == "adopted":
        instance.state["bootstrap_database_origin"] = {"safe": True}
    elif damage == "completed":
        instance.state["completed_cluster_ids"] = ["gpu-a"]
    elif damage == "unknown":
        instance.state["unowned_field"] = "do-not-ignore"
    else:
        instance.state[REPAIR.REPAIR_KEY]["previous_refresher"]["absent"].pop()
    select_candidate(instance)
    with pytest.raises(ReleaseError):
        prepare(instance)
    assert not (MUTATIONS & set(instance.runner.events)), (
        "scope or baseline drift authorized the handoff"
    )


@pytest.mark.parametrize("option", ["resume", "commit_live", "previous_override"])
def test_handoff_cannot_be_used_as_upgrade_or_supersede(
    handoff_release: Any, option: str
) -> None:
    instance = handoff_release
    select_candidate(instance)
    with pytest.raises(ReleaseError, match="candidate binding"):
        prepare(instance, **{option: {} if option == "previous_override" else True})
    assert not (MUTATIONS & set(instance.runner.events)), (
        "a bootstrap-only handoff widened an application transaction permission"
    )


def test_legacy_initial_record_handoff_keeps_exact_original_hash(
    handoff_release: Any,
) -> None:
    instance = handoff_release
    record = instance.state[REPAIR.REPAIR_KEY]
    baseline = validated_prerequisite_baseline(instance.state, record)
    record["binding"].pop("bootstrap")
    record["binding_sha256"] = canonical_sha256(record["binding"])
    record["snapshot_representation"] = capture_baseline_representation(
        baseline, bootstrap=False
    )
    original_hash = record["baseline_sha256"]
    select_candidate(instance)
    replacement = prepare(instance)
    assert replacement["supersedes"]["baseline_sha256"] == original_hash, (
        "legacy migration changed the originally recorded baseline hash"
    )


@pytest.mark.parametrize("failure", ["cleanup", "foreign-job", "restore"])
def test_original_journal_survives_cleanup_or_restore_failure(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    instance = handoff_release
    predecessor = copy.deepcopy(instance.state[REPAIR.REPAIR_KEY])
    if failure in {"cleanup", "foreign-job"}:
        job = copy.deepcopy(instance.runner.created_jobs[0])
        instance.runner.jobs[job["metadata"]["name"]] = job
        instance.runner.fail_delete = failure == "cleanup"
        instance.runner.foreign_job = failure == "foreign-job"
    else:
        monkeypatch.setattr(
            REPAIR,
            "restore_aurora_refresh_snapshot",
            lambda *_args: (_ for _ in ()).throw(ReleaseError("restore failed")),
        )
    select_candidate(instance)
    with pytest.raises(ReleaseError):
        prepare(instance)
    assert (
        instance.state[REPAIR.REPAIR_KEY]["attempt_id"] == predecessor["attempt_id"]
    ), "a failed restoration replaced the original journal"
    assert (
        instance.state[REPAIR.REPAIR_KEY]["previous_refresher"]
        == predecessor["previous_refresher"]
    ), "a failed restoration lost the original snapshot"
    assert "apply" not in instance.runner.events, (
        "candidate resources changed after cleanup or restoration failed"
    )


@pytest.mark.parametrize("persisted", [False, True])
def test_lost_handoff_checkpoint_ack_has_no_identity_only_intermediate_state(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch, persisted: bool
) -> None:
    instance = handoff_release
    old_attempt = instance.state[REPAIR.REPAIR_KEY]["attempt_id"]
    old_snapshot = copy.deepcopy(
        instance.state[REPAIR.REPAIR_KEY]["previous_refresher"]
    )
    select_candidate(instance)
    save = REPAIR.save_recorded_state
    interrupted = False

    def fail_checkpoint(release: Any, phase: str, **updates: Any) -> None:
        nonlocal interrupted
        record = updates.get(REPAIR.REPAIR_KEY, {})
        if record.get("supersedes") and not interrupted:
            interrupted = True
            if persisted:
                save(release, phase, **updates)
            raise KeyboardInterrupt()
        save(release, phase, **updates)

    monkeypatch.setattr(REPAIR, "save_recorded_state", fail_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        prepare(instance)
    stored = instance.runner.cloud_state
    stored_record = stored[REPAIR.REPAIR_KEY]
    assert (stored_record["attempt_id"] != old_attempt) is persisted, (
        "checkpoint interruption created a partially replaced journal"
    )
    assert stored["release_id"] == stored_record["binding"]["release_id"], (
        "candidate identity reached storage without its matching journal"
    )
    instance.state = copy.deepcopy(stored)
    instance.approved_manifest_digest = None
    replacement = prepare(instance)
    assert replacement["previous_refresher"] == old_snapshot, (
        "checkpoint recovery recaptured a candidate instead of the original snapshot"
    )
    assert replacement["status"] == "READY" and not instance.runner.jobs, (
        "checkpoint recovery did not finish the new repair and its cleanup"
    )


def test_interruption_after_restore_before_baseline_keeps_original_journal(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = handoff_release
    predecessor = copy.deepcopy(instance.state[REPAIR.REPAIR_KEY])
    select_candidate(instance)
    projection = REPAIR.candidate_bootstrap_baseline

    def interrupt(*_args: Any) -> dict[str, Any]:
        assert instance.runner.live == {}, (
            "baseline projection started before original absence was restored"
        )
        raise KeyboardInterrupt()

    monkeypatch.setattr(REPAIR, "candidate_bootstrap_baseline", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prepare(instance)
    assert instance.runner.cloud_state[REPAIR.REPAIR_KEY] == predecessor, (
        "restoration removed its journal before a new baseline could be recorded"
    )
    monkeypatch.setattr(REPAIR, "candidate_bootstrap_baseline", projection)
    instance.state = copy.deepcopy(instance.runner.cloud_state)
    replacement = prepare(instance)
    assert replacement["supersedes"]["attempt_id"] == predecessor["attempt_id"], (
        "retry lost the original repair after a completed restoration"
    )


def test_database_identity_change_during_restore_preserves_predecessor(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = handoff_release
    predecessor = instance.state[REPAIR.REPAIR_KEY]["attempt_id"]
    restore = REPAIR.restore_aurora_refresh_snapshot

    def replace_database(release: Any, snapshot: object) -> None:
        restore(release, snapshot)
        instance.runner.database_resource_id = "replaced-during-restore"

    monkeypatch.setattr(REPAIR, "restore_aurora_refresh_snapshot", replace_database)
    select_candidate(instance)
    with pytest.raises(ReleaseError, match="identity changed during restoration"):
        prepare(instance)
    assert instance.state[REPAIR.REPAIR_KEY]["attempt_id"] == predecessor, (
        "database replacement authorized a new journal"
    )
    assert "apply" not in instance.runner.events, (
        "the candidate was applied after database identity changed"
    )


@pytest.mark.parametrize("handoff_release", [True], indirect=True)
def test_existing_full_snapshot_is_restored_and_carried_without_recapture(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = handoff_release
    original = copy.deepcopy(instance.state[REPAIR.REPAIR_KEY]["previous_refresher"])
    restore = REPAIR.restore_aurora_refresh_snapshot
    restored: list[dict[str, Any]] = []

    def observe_restore(release: Any, snapshot: object) -> None:
        restore(release, snapshot)
        restored.append(copy.deepcopy(instance.runner.live))

    def refuse_recapture(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("handoff must not capture a replacement rollback target")

    monkeypatch.setattr(REPAIR, "restore_aurora_refresh_snapshot", observe_restore)
    monkeypatch.setattr(REPAIR, "capture_aurora_refresh_snapshot", refuse_recapture)
    select_candidate(instance)
    replacement = prepare(instance)
    expected = {
        resource_argument(item["apiVersion"], item["kind"]): item
        for item in original["objects"]
    }
    assert restored == [expected], "the complete original refresher was not restored"
    assert replacement["previous_refresher"] == original, (
        "new repair did not inherit the original full rollback snapshot"
    )


def test_new_candidate_store_refusal_does_not_adopt_or_run_consumers(
    handoff_release: Any,
) -> None:
    instance = handoff_release
    select_candidate(instance)
    instance.runner.unsafe_store = True
    with pytest.raises(ReleaseError, match="safety proof failed"):
        REPAIR.prepare_bootstrap_workflows(instance)
    assert (
        instance.state[REPAIR.REPAIR_KEY]["binding"]["release_id"]
        == instance.release_id
    ), "Store refusal discarded the new repair's recovery journal"
    assert REPAIR.BOOTSTRAP_ORIGIN_KEY not in instance.state, (
        "Store refusal authorized bootstrap database adoption"
    )
    assert not ({"schema", "cpu-finalize"} & set(instance.runner.events)), (
        "Store refusal still permitted an application mutation"
    )


def test_failed_new_candidate_restores_original_not_failed_predecessor(
    handoff_release: Any,
) -> None:
    instance = handoff_release
    select_candidate(instance)
    instance.runner.fail_job = "credential"
    with pytest.raises(ReleaseError, match="credential Job failed"):
        prepare(instance)
    assert instance.state[REPAIR.REPAIR_KEY]["status"] == "FAILED", (
        "a failed successor was not durably recorded"
    )
    REPAIR.restore_prerequisite_repair(instance)
    assert instance.runner.live == {}, (
        "successor rollback restored the failed candidate instead of original absence"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bootstrap", 1),
        ("bootstrap", "true"),
        ("bootstrap", False),
        ("commit_live", True),
        ("extra_policy", "unknown"),
    ],
)
def test_unknown_or_nonbootstrap_binding_cannot_take_handoff_path(
    handoff_release: Any, field: str, value: object
) -> None:
    instance = handoff_release
    record = instance.state[REPAIR.REPAIR_KEY]
    record["binding"][field] = value
    record["binding_sha256"] = canonical_sha256(record["binding"])
    select_candidate(instance)
    with pytest.raises(ReleaseError, match="handoff binding"):
        prepare(instance)
    assert not (MUTATIONS & set(instance.runner.events)), (
        "an unknown handoff binding was accepted"
    )


@pytest.mark.parametrize("field", ["status", "jobs", "job-status", "job-owner"])
def test_malformed_journal_is_a_release_refusal_before_mutation(
    handoff_release: Any, field: str
) -> None:
    instance = handoff_release
    record = instance.state[REPAIR.REPAIR_KEY]
    if field == "job-status":
        record["jobs"]["credential"]["status"] = []
    elif field == "job-owner":
        record["jobs"]["credential"]["owner_uid"] = None
    else:
        record[field] = []
    select_candidate(instance)
    with pytest.raises(ReleaseError, match="handoff"):
        prepare(instance)
    assert not (MUTATIONS & set(instance.runner.events)), (
        "malformed journal fields authorized restoration"
    )
