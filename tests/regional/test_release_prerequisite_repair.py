from __future__ import annotations

import copy
import json
import signal
from typing import Any

import pytest

from gpu_fault.admin.process_supervisor import (
    ProcessSupervisionLost,
    ensure_supervision_safe,
    interruption_scope,
)
from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_admin_commands as COMMANDS
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_prerequisite_repair as REPAIR
from gpu_fault_release import regional_release_probe_job as JOBS
from gpu_fault_release.regional_release_bootstrap import cleanup_bootstrap
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_diff import (
    build_execution_plan,
    diff_from_changed,
)
from gpu_fault_release.regional_release_store_proof import bootstrap_store_proof
from tests.regional._prerequisite_repair_support import repair_release

DIFF = diff_from_changed({"aurora_refresh_drift", "schema_manifests"})
PLAN = build_execution_plan(DIFF)


def prepare(instance: Any, *, bootstrap: bool = False) -> None:
    REPAIR.prepare_prerequisite_repair(
        instance, diff=DIFF, plan=PLAN, bootstrap=bootstrap
    )


def test_repair_persists_original_snapshot_and_baseline_before_mutation(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch)
    baseline = copy.deepcopy(instance.state)
    original = copy.deepcopy(instance.runner.live)
    prepare(instance)
    record = instance.state[REPAIR.REPAIR_KEY]
    assert record["status"] == "READY", (
        "repair did not record successful AWSCURRENT proof"
    )
    assert instance.state["release_id"] == "old", (
        "prerequisite overwrote the live release identity"
    )
    assert record["baseline_sha256"] == canonical_sha256(baseline), (
        "baseline binding is not original"
    )
    assert record["previous_refresher"]["objects"] == list(original.values()), (
        "the rollback snapshot was taken after candidate mutation"
    )
    assert instance.runner.events.index("persist") < instance.runner.events.index(
        "apply"
    ), "versioned repair started without durable rollback context"
    assert REPAIR.matches_pre_repair_state(
        instance, instance.state, canonical_sha256(baseline)
    ), "the deploy handoff rejected its own recorded prerequisite transition"
    assert instance.runner.jobs == {}, "successful repair left a temporary Job"


def test_upgrade_repairs_before_store_gates_and_adopts_original_snapshot(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch)
    original = copy.deepcopy(instance.runner.live)
    monkeypatch.setattr(
        ORCHESTRATION,
        "preflight_upgrade_mutations",
        lambda *_args: instance.runner.events.append("candidate-preflight"),
    )
    ORCHESTRATION.upgrade_release(instance, diff=DIFF)
    events = instance.runner.events
    assert (
        events.index("credential-job")
        < events.index("store-gate")
        < events.index("full-snapshot")
    ), "rotated credentials still blocked the prerequisite repair"
    assert events.index("credential-job") < events.index("schema"), (
        "DDL ran without credential proof"
    )
    assert instance.state["previous"]["aurora_refresh"]["objects"] == list(
        original.values()
    ), "the full transaction adopted the repaired candidate as its rollback baseline"
    assert REPAIR.REPAIR_KEY not in instance.state, (
        "the adopted prerequisite was left independently active"
    )


def test_retry_does_not_recapture_candidate_or_reinstall_healthy_refresher(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch)
    instance.runner.fail_job = "credential"
    with pytest.raises(ReleaseError, match="credential Job failed"):
        prepare(instance)
    previous = copy.deepcopy(instance.state[REPAIR.REPAIR_KEY]["previous_refresher"])
    instance.runner.fail_job = ""
    prepare(instance)
    assert instance.state[REPAIR.REPAIR_KEY]["previous_refresher"] == previous, (
        "retry overwrote the original rollback snapshot"
    )
    assert instance.runner.events.count("apply") == 2, (
        "repair must only install the suspended candidate then activate it after proof"
    )
    assert instance.runner.jobs == {}, "a failed credential Job survived retry cleanup"


def test_lost_create_ack_is_cleaned_by_nonce_and_uid(monkeypatch) -> None:
    instance = repair_release(monkeypatch)
    instance.runner.create_ack_lost = True
    with pytest.raises(ReleaseError, match="ACK lost"):
        prepare(instance)
    assert instance.runner.jobs == {}, "lost create ACK left an untracked Job"
    record = instance.state[REPAIR.REPAIR_KEY]["jobs"]["credential"]
    assert record["uid"] and record["status"] == "REMOVED", (
        "cleanup did not persist the recovered UID"
    )


def test_cleanup_failure_blocks_success_and_is_retried_before_next_job(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch)
    instance.runner.fail_delete = True
    with pytest.raises(ReleaseError, match="cleanup unavailable"):
        prepare(instance)
    assert instance.state[REPAIR.REPAIR_KEY]["status"] == "FAILED", (
        "cleanup failure was marked ready"
    )
    instance.runner.fail_delete = False
    events = instance.runner.events
    before = len(events)
    prepare(instance)
    assert events[before:].index("job-delete") < events[before:].index("job-create"), (
        "retry created another writer before removing the interrupted Job"
    )
    assert instance.runner.jobs == {}, "retry did not complete owned cleanup"


def test_replaced_job_is_never_trusted_or_deleted(monkeypatch) -> None:
    instance = repair_release(monkeypatch)
    instance.runner.replace_on_completion = True
    with pytest.raises(ReleaseError, match="ownership or UID differs"):
        prepare(instance)
    assert instance.runner.jobs, "UID-scoped cleanup deleted the replacement Job"
    assert "job-delete" not in instance.runner.events, "a foreign UID reached deletion"
    assert instance.state[REPAIR.REPAIR_KEY]["status"] == "FAILED", (
        "replacement Job completion was accepted as credential proof"
    )


@pytest.mark.parametrize("mutation", ["post-start", "pod-security", "host-network"])
def test_job_admission_cannot_add_execution_or_security_fields(
    monkeypatch, mutation: str
) -> None:
    instance = repair_release(monkeypatch)
    run = instance.runner.run

    def inject(args, **kwargs):
        result = run(args, **kwargs)
        if "--dry-run=server" in args and "-o" in args:
            document = json.loads(result)
            pod = document["spec"]["template"]["spec"]
            if mutation == "post-start":
                pod["containers"][0]["lifecycle"] = {
                    "postStart": {"exec": {"command": ["unapproved-program"]}}
                }
            elif mutation == "pod-security":
                pod["securityContext"] = {"runAsUser": 0}
            else:
                pod["hostNetwork"] = True
            return json.dumps(document)
        return result

    monkeypatch.setattr(instance.runner, "run", inject)
    with pytest.raises(ReleaseError, match="Job admission identity differs"):
        prepare(instance)
    assert instance.runner.created_jobs == [], (
        "admission changed the program before create"
    )
    assert instance.state[REPAIR.REPAIR_KEY]["status"] == "FAILED", (
        "an unapproved admission mutation was treated as successful repair"
    )


def test_job_comparison_accepts_only_known_server_defaults(monkeypatch) -> None:
    instance = repair_release(monkeypatch)
    run = instance.runner.run

    def defaults(args, **kwargs):
        result = run(args, **kwargs)
        if "--dry-run=server" in args and "-o" in args:
            document = json.loads(result)
            document["metadata"]["uid"] = "preview-uid"
            spec = document["spec"]
            spec["selector"] = {
                "matchLabels": {"batch.kubernetes.io/controller-uid": "preview-uid"}
            }
            spec.update(parallelism=1, completions=1, completionMode="NonIndexed")
            pod = spec["template"]["spec"]
            pod.update(dnsPolicy="ClusterFirst", schedulerName="default-scheduler")
            pod.setdefault("securityContext", {})
            pod["containers"][0].update(
                imagePullPolicy="IfNotPresent",
                terminationMessagePath="/dev/termination-log",
                terminationMessagePolicy="File",
            )
            return json.dumps(document)
        return result

    monkeypatch.setattr(instance.runner, "run", defaults)
    prepare(instance)
    assert instance.state[REPAIR.REPAIR_KEY]["status"] == "READY", (
        "ordinary API defaults prevented a verified Job from running"
    )


def test_ctrl_c_cleans_owned_job_without_bypassing_ordinary_interruption(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch)
    run = instance.runner.run

    def interrupt(args, **kwargs):
        if any(value.endswith("/wait-for-kubernetes-job.sh") for value in args):
            signal.raise_signal(signal.SIGINT)
            ensure_supervision_safe()
        return run(args, **kwargs)

    monkeypatch.setattr(instance.runner, "run", interrupt)
    with pytest.raises(KeyboardInterrupt), interruption_scope():
        prepare(instance)
    assert instance.runner.jobs == {}, "Ctrl-C skipped bounded UID-scoped cleanup"
    assert "job-delete" in instance.runner.events, (
        "the interrupted proof Job was not removed"
    )
    assert instance.state[REPAIR.REPAIR_KEY]["status"] != "READY", (
        "interruption was mistaken for successful credential proof"
    )


def test_supervision_poison_still_refuses_all_cleanup_commands(monkeypatch) -> None:
    instance = repair_release(monkeypatch)
    run = instance.runner.run

    def lost(*, allow_interrupted=False):
        assert allow_interrupted is True, (
            "cleanup did not opt into bounded interruption handling"
        )
        raise ProcessSupervisionLost("ownership unproved")

    def lose_on_wait(args, **kwargs):
        if any(value.endswith("/wait-for-kubernetes-job.sh") for value in args):
            monkeypatch.setattr(JOBS, "ensure_supervision_safe", lost)
            raise ProcessSupervisionLost("ownership unproved")
        return run(args, **kwargs)

    monkeypatch.setattr(instance.runner, "run", lose_on_wait)
    with pytest.raises(ProcessSupervisionLost):
        prepare(instance)
    assert "job-delete" not in instance.runner.events, (
        "poisoned ownership allowed a cleanup command"
    )
    assert instance.state[REPAIR.REPAIR_KEY]["status"] == "APPLYING", (
        "fatal supervision loss entered ordinary recovery/state writes"
    )


def test_unadopted_repair_restores_only_original_refresher_not_passwords(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch)
    original = copy.deepcopy(instance.runner.live)
    prepare(instance)
    status = copy.deepcopy(instance.runner.refresh_status)
    REPAIR.restore_prerequisite_repair(instance)
    assert instance.runner.live == original, (
        "prerequisite rollback did not restore all original objects"
    )
    assert instance.runner.refresh_status == status, (
        "rollback restored old database credentials/status"
    )
    assert REPAIR.REPAIR_KEY not in instance.runner.cloud_state, (
        "completed prerequisite rollback stayed active"
    )
    assert instance.state["release_id"] == "old", (
        "prerequisite rollback changed application identity"
    )


def test_store_refusal_leaves_resumable_prerequisite_not_a_false_committed_candidate(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch)
    monkeypatch.setattr(instance, "_remote_commands_are_idle", lambda: False)
    with pytest.raises(ReleaseError, match="remote commands"):
        ORCHESTRATION.upgrade_release(instance, diff=DIFF)
    assert instance.state["release_id"] == "old", (
        "Store refusal committed candidate application identity"
    )
    assert instance.state[REPAIR.REPAIR_KEY]["status"] == "READY", (
        "repair rollback context was discarded"
    )
    assert (
        "schema" not in instance.runner.events
        and "cpu-finalize" not in instance.runner.events
    ), "business consumers moved despite Store refusal"


def test_accepted_baseline_commit_preserves_pending_repair_binding(monkeypatch) -> None:
    instance = repair_release(monkeypatch)
    instance.state["transaction_committed"] = False
    REPAIR.prepare_prerequisite_repair(instance, diff=DIFF, plan=PLAN, commit_live=True)
    expected = instance.state[REPAIR.REPAIR_KEY]["baseline_sha256"]
    instance.state.update(
        transaction_committed=True,
        release_lifecycle="COMMITTED",
        commit_cleanup_completed=True,
    )
    assert REPAIR.matches_pre_repair_state(instance, instance.state, expected), (
        "the known commit-live transition invalidated its pending repair snapshot"
    )
    assert REPAIR.adopted_refresher_snapshot(instance) is not None, (
        "the new upgrade cannot adopt its original refresher after baseline commit"
    )


def test_baseline_or_candidate_drift_cannot_reuse_prerequisite_approval(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch)
    prepare(instance)
    before = list(instance.runner.events)
    instance.state["release_id"] = "foreign"
    with pytest.raises(ReleaseError, match="baseline drifted"):
        prepare(instance)
    assert "apply" not in instance.runner.events[len(before) :], (
        "baseline drift authorized another mutation"
    )


def test_bootstrap_retries_inspect_actual_rows_even_after_a_previous_success(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch, bootstrap=True)
    prepare(instance, bootstrap=True)
    first = bootstrap_store_proof(
        instance, REPAIR.repair_job_checkpoint(instance, "store")
    )
    assert first["safe"] is True, "initial database proof failed"
    instance.runner.unsafe_store = True
    with pytest.raises(ReleaseError, match="safety proof failed"):
        bootstrap_store_proof(instance, REPAIR.repair_job_checkpoint(instance, "store"))
    assert instance.runner.events.count("store-job") == 2, (
        "retry reused an old empty/created proof"
    )
    assert instance.runner.jobs == {}, "rejected Store proof did not clean its Job"


def test_for_deploy_runs_real_prerequisite_and_independent_bootstrap_gate(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch, bootstrap=True)
    instance.config.clusters = ("gpu-a",)
    for name in (
        "_check_tools",
        "_check_local_inputs",
        "_check_aws_identity",
        "_check_contexts",
        "_check_cpu_capacity",
        "check_cpu_secrets",
        "_check_load_balancer_controller",
        "_check_nlb_inputs",
        "_check_aurora",
        "check_control_record_archive_bucket",
        "check_email_notifications",
    ):
        monkeypatch.setattr(
            CHECKS, name, lambda *_args, **_kwargs: CHECKS.CheckValue("proved")
        )
    monkeypatch.setattr(
        CHECKS, "monitoring_repair_preflight", lambda *_args, **_kwargs: {}
    )
    report = COMMANDS.build_deploy_preflight_report(instance)
    assert report["healthy"] is True, report["checks"]
    workflow = next(
        item for item in report["checks"] if item["name"] == "workflow_safety"
    )
    assert workflow["details"]["database_state"] == "uninitialized_empty", (
        "bootstrap did not use a fresh empty-database proof"
    )
    assert instance.runner.events.index(
        "credential-job"
    ) < instance.runner.events.index("store-job"), (
        "bootstrap Store proof used credentials before AWSCURRENT was repaired"
    )
    assert instance.runner.jobs == {}, "admin preflight leaked proof Jobs"


def test_fresh_install_refuses_nonempty_database_even_without_active_workflows(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch, bootstrap=True)
    instance.runner.database_state = "initialized"
    prepare(instance, bootstrap=True)
    with pytest.raises(ReleaseError, match="safety proof failed"):
        REPAIR.bootstrap_workflow_proof(instance)
    assert REPAIR.BOOTSTRAP_ORIGIN_KEY not in instance.state, (
        "absence of the old release state adopted a nonempty database"
    )


def test_initialized_bootstrap_retry_requires_origin_and_a_fresh_read(
    monkeypatch,
) -> None:
    instance = repair_release(monkeypatch, bootstrap=True)
    REPAIR.prepare_bootstrap_workflows(instance)
    origin = copy.deepcopy(instance.state[REPAIR.BOOTSTRAP_ORIGIN_KEY])
    instance.runner.database_state = "initialized"
    REPAIR.prepare_bootstrap_workflows(instance)
    assert instance.state[REPAIR.BOOTSTRAP_ORIGIN_KEY] == origin, (
        "retry replaced the original empty-database evidence"
    )
    assert (
        instance.state["bootstrap_store_safety"]["database_state"] == "initialized"
    ), "retry trusted an old created/empty checkpoint instead of querying current state"
    instance.runner.unsafe_store = True
    with pytest.raises(ReleaseError, match="safety proof failed"):
        REPAIR.prepare_bootstrap_workflows(instance)
    assert instance.runner.events.count("store-job") == 3, (
        "retry reused stale row evidence"
    )


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("cleanup", [False, True])
def test_preflight_repair_survives_owned_bootstrap_checkpoints(
    monkeypatch: pytest.MonkeyPatch, legacy: bool, cleanup: bool
) -> None:
    instance = repair_release(monkeypatch, bootstrap=True)
    prepare(instance, bootstrap=True)
    REPAIR.bootstrap_workflow_proof(instance)
    record = instance.state[REPAIR.REPAIR_KEY]
    original = copy.deepcopy(record["previous_refresher"])
    if legacy:
        record["binding"].pop("bootstrap")
        record["binding_sha256"] = canonical_sha256(record["binding"])
        record["snapshot_representation"] = {
            key: record["snapshot_representation"][key]
            for key in ("previous_snapshot", "previous_snapshot_sha256")
        }
    REPAIR.save_recorded_state(
        instance,
        "bootstrap-started",
        previous=None,
        transaction_committed=False,
        completed_cluster_ids=[],
        bootstrap_cleanup_completed_steps=[],
        bootstrap_cleanup_failure=None,
    )
    if cleanup:
        REPAIR.save_recorded_state(
            instance, "bootstrap-failed", resume_phase="bootstrap-started"
        )
        cleanup_bootstrap(instance)
        assert instance.state["phase"] == "bootstrap-cleaned"
    assert REPAIR.adopted_refresher_snapshot(instance) == original, (
        "owned progress must not invalidate or replace the original rollback snapshot"
    )
    REPAIR.prepare_bootstrap_workflows(instance)
    assert REPAIR.REPAIR_KEY not in instance.state, (
        "bootstrap did not adopt its successfully revalidated prerequisite repair"
    )
    assert instance.state["bootstrap_store_safety"]["safe"] is True
    assert instance.runner.events.count("credential-job") == 2
    assert instance.runner.events.count("store-job") == 2, (
        "bootstrap reused preflight's old database proof instead of rereading"
    )
    assert instance.runner.jobs == {}, "bootstrap handoff left an owned proof Job"
