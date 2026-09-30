from __future__ import annotations

import copy

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional import boot020_release_candidates as candidates
from scripts.e2e.regional import boot029_receipts as receipts
from scripts.e2e.regional.boot019_revocation import RevocationCredentials
from scripts.e2e.regional.boot_acceptance_lifecycle import cleanup_isolated_site
from scripts.e2e.regional.run_boot020_release_rolling import (
    AcceptanceCheckError,
    _assert_data_plane_changed,
)


def test_bound_stage_resume_rejects_skips_input_changes_and_edited_logs(tmp_path):
    path = tmp_path / "receipt.json"
    inputs = {"cpu": "cpu", "gpu": "gpu"}
    with pytest.raises(ValueError, match="preceding"):
        receipts.initialize(path, inputs=inputs, stage=3)
    receipts.initialize(path, inputs=inputs, stage=1)
    receipts.start_stage(path, state=tmp_path / "state", stage=1)
    log = tmp_path / "first.log"
    log.write_text("completed")
    receipts.finish(path, stage=1, status="PASS", seconds=1, log=log, name="first")
    receipts.initialize(path, inputs=inputs, stage=2)
    with pytest.raises(ValueError, match="first unpassed"):
        receipts.initialize(path, inputs=inputs, stage=3)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="identity"):
        receipts.initialize(path, inputs={"cpu": "changed", "gpu": "gpu"}, stage=2)
    assert path.read_bytes() == before
    log.write_text("edited")
    with pytest.raises(ValueError, match="log proof"):
        receipts.initialize(path, inputs=inputs, stage=2)


def test_completion_journal_cannot_be_stale_or_from_a_different_target(tmp_path):
    path, state = tmp_path / "receipt.json", tmp_path / "state"
    old = state / "remove-cluster/gpu/state.json"
    value = {
        "phase": "COMPLETED",
        "attempt_id": "a" * 32,
        "target": {"eks_cluster_arn": "gpu"},
    }
    write_json_atomic(old, value)
    receipts.initialize(path, inputs={}, stage=1)
    receipts.start_stage(path, state=state, stage=2)
    with pytest.raises(ValueError, match="exactly one"):
        receipts.verify_journal(
            path, state=state, stage=2, gpu_arn="gpu", cpu_arn="cpu"
        )
    value["updated_at"] = "new-mtime-is-not-new-operation"
    write_json_atomic(old, value)
    with pytest.raises(ValueError, match="exactly one"):
        receipts.verify_journal(
            path, state=state, stage=2, gpu_arn="gpu", cpu_arn="cpu"
        )
    value["attempt_id"] = "b" * 32
    write_json_atomic(old, value)
    assert (
        receipts.verify_journal(
            path, state=state, stage=2, gpu_arn="gpu", cpu_arn="cpu"
        )["phase"]
        == "COMPLETED"
    )
    with pytest.raises(ValueError, match="exactly one"):
        receipts.verify_journal(
            path, state=state, stage=2, gpu_arn="sibling", cpu_arn="cpu"
        )


def test_failed_first_deploy_preserves_its_checkpoint_for_retry(tmp_path):
    path, state = tmp_path / "receipt.json", tmp_path / "state"
    receipts.initialize(path, inputs={}, stage=1)
    receipts.start_stage(path, state=state, stage=1)
    write_json_atomic(state / "bootstrap-state.json", {"phase": "failed"})
    log = tmp_path / "attempt1.log"
    log.write_text("injected failure after resources were created")
    receipts.finish(path, stage=1, status="FAIL", seconds=1, log=log, name="first")
    receipts.initialize(path, inputs={}, stage=1)
    assert receipts.start_stage(path, state=state, stage=1)["attempts"] == 2
    assert receipts.read(state / "bootstrap-state.json") == {"phase": "failed"}


def test_early_resource_disposition_does_not_infer_absence_from_missing_site(tmp_path):
    state = tmp_path / "state"
    write_json_atomic(
        state / "bootstrap-state.json", {"phase": "failed", "resources": {"aurora": {}}}
    )
    outcome = cleanup_isolated_site(
        state,
        tmp_path,
        log_name="cleanup.log",
        retain=False,
        uninstall=lambda *_args: pytest.fail("no site authorizes uninstall"),
    )
    assert outcome["disposition"] == "RECONCILIATION_REQUIRED"
    assert outcome["resource_state"] == "UNKNOWN"
    assert outcome["cleanup_verified"] is False
    assert (state / "bootstrap-state.json").exists(), (
        "unknown resource disposition must preserve the bootstrap checkpoint"
    )


def test_revocation_credential_survives_new_process_but_not_another_run(tmp_path):
    site = tmp_path / "site.yaml"
    first = RevocationCredentials(
        tmp_path / "run", site_path=site, join_arn="arn:example:gpu"
    )
    capture = first.capture("joined", b"fixture-value-for-revocation")
    second = RevocationCredentials(
        tmp_path / "run", site_path=site, join_arn="arn:example:gpu"
    )
    assert second.load(capture) == b"fixture-value-for-revocation"
    assert all(
        path.stat().st_mode & 0o777 == 0o600 for path in first.directory.iterdir()
    ), "persisted revocation credentials must remain owner-readable only"
    foreign = RevocationCredentials(
        tmp_path / "other-run", site_path=site, join_arn="arn:example:gpu"
    )
    with pytest.raises(ValueError, match="another run"):
        foreign.load(capture)
    second.remove(capture)
    second.remove(capture)
    assert not list(first.directory.iterdir()), (
        "idempotent revocation cleanup must remove every retained credential file"
    )


def test_executor_candidate_delta_checks_full_wheel_identity():
    base = {
        "components": {
            name: {field: "a" * 64}
            for name, field in [
                ("control_plane", "wheel_sha256"),
                ("executor", "wheel_sha256"),
                ("node_runtime", "wheel_sha256"),
                ("node_bundle", "bundle_sha256"),
            ]
        }
    }
    candidate = copy.deepcopy(base)
    candidate["components"]["executor"]["wheel_sha256"] = "b" * 64
    assert candidates.validate_candidate_delta(base, candidate, "B") == ["executor"]
    candidate["components"]["control_plane"]["wheel_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="delta"):
        candidates.validate_candidate_delta(base, candidate, "B")


def test_executor_image_rollout_includes_reconciler_but_never_agent():
    names = [
        "gpu-fault-cluster-executor",
        "gpu-fault-completion-watcher",
        "gpu-fault-kubernetes-node-resource-collector",
        "gpu-fault-node-installer-reconciler",
    ]
    before = {
        "live": {"clusters": {"gpu": {"pin": "old"}}},
        "gpu_generations": {"gpu": dict.fromkeys(names, 1)},
    }
    after = {
        "live": {"clusters": {"gpu": {"pin": "new"}}},
        "gpu_generations": {"gpu": dict.fromkeys(names, 2)},
    }
    diff = {
        "kind": "DATA_PLANE_COMPATIBLE",
        "changed": ["executor_wheel", "executor_image"],
    }
    _assert_data_plane_changed(before, after, scenario="executor", diff=diff)
    after["gpu_generations"]["gpu"][names[-1]] = 1
    with pytest.raises(AcceptanceCheckError, match="component plan"):
        _assert_data_plane_changed(before, after, scenario="executor", diff=diff)


@pytest.mark.parametrize(
    "missing", [None, "control_plane", "executor", "node_dependencies"]
)
def test_cold_build_requires_each_current_image_descriptor(tmp_path, missing):
    manifest_path = tmp_path / "dist/current-release.json"
    mapping = {
        "control_plane": "runtime",
        "executor": "executor",
        "node_dependencies": "node_dependencies",
    }
    images = {
        name: {
            "registry_reused": False,
            "deployable": True,
            "image_input_sha256": "a" * 64,
            "source_identity_sha256": "b" * 64,
            "reference": f"repository@sha256:{index:064d}",
        }
        for index, name in enumerate(mapping)
    }
    write_json_atomic(
        manifest_path,
        {
            "delivery": {
                "images": {
                    field: {"reference": images[name]["reference"]}
                    for name, field in mapping.items()
                }
            }
        },
    )
    write_json_atomic(
        tmp_path / "bootstrap-state.json",
        {"resources": {"release": {"manifest": str(manifest_path)}}},
    )
    if missing:
        images[missing]["registry_reused"] = True
    write_json_atomic(
        tmp_path / "dist/release-runtime-image.json",
        {"schema_version": 3, "source_identity_sha256": "b" * 64, "images": images},
    )
    log = tmp_path / "build.log"
    log.write_text("#5 extracting sha256:abcdef\n#6 [2/6] COPY\n")
    if missing:
        with pytest.raises(ValueError, match=missing):
            receipts.cold_build_proof(tmp_path, [log])
    else:
        assert set(
            receipts.cold_build_proof(tmp_path, [log])["cold_build"]["images"]
        ) == set(mapping)


def test_completed_record_admits_a_stage_five_rerun_and_keeps_the_passed_receipt(
    tmp_path,
):
    """2026-09-22: the first stage-5 pass needed a resumed first deploy after
    runner fixes; the user asked for one unbroken attempt. A fully passed record
    may run stage 5 again (the second site was uninstalled in between); earlier
    stages stay closed and the passed receipt survives under ``reruns``."""

    path = tmp_path / "receipt.json"
    inputs = {"cpu": "cpu", "gpu": "gpu"}
    receipts.initialize(path, inputs=inputs, stage=1)
    for stage in range(1, 6):
        if stage > 1:
            receipts.initialize(path, inputs=inputs, stage=stage)
        receipts.start_stage(path, state=tmp_path / "state", stage=stage)
        log = tmp_path / f"stage-{stage}.log"
        log.write_text(f"stage {stage} done")
        receipts.finish(
            path, stage=stage, status="PASS", seconds=1, log=log, name=f"s{stage}"
        )
    with pytest.raises(ValueError, match="first unpassed"):
        receipts.initialize(path, inputs=inputs, stage=3)

    record = receipts.initialize(path, inputs=inputs, stage=5)

    assert record["stages"]["5"]["status"] == "RERUN", "stage 5 is open again"
    assert record["reruns"][0]["previous"]["status"] == "PASS", (
        "the passed receipt is kept, not overwritten"
    )
    assert record["stages"]["5"]["attempt_logs"] == [
        {"name": "stage-5.log", "sha256": record["reruns"][0]["previous"]["log_sha256"]}
    ], "earlier attempt logs stay listed on the stage"
    assert record["stages"]["5"]["rerun_from_attempt"] == 2, (
        "the rerun's attempts are marked where they begin"
    )
    receipt = receipts.start_stage(path, state=tmp_path / "state", stage=5)
    assert receipt["attempts"] == 2 and receipt["status"] == "RUNNING", (
        "attempts keep counting so attempt log names stay unique"
    )
    log = tmp_path / "stage-5-rerun.log"
    log.write_text("one attempt")
    receipts.finish(path, stage=5, status="PASS", seconds=2, log=log, name="s5")
    final = receipts.read(path)
    assert final["stages"]["5"]["status"] == "PASS", "the rerun passed"
    assert (
        final["stages"]["5"]["attempts"] == final["stages"]["5"]["rerun_from_attempt"]
    ), "one attempt, one pass: the rerun began and ended on the same attempt"
    assert [item["name"] for item in final["stages"]["5"]["attempt_logs"]] == [
        "stage-5.log",
        "stage-5-rerun.log",
    ], "the log chain covers both runs"
