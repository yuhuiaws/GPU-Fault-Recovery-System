from __future__ import annotations

import copy
from typing import Any

import pytest

from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_diff import AURORA_PREREQUISITE_REPAIR_KEY
from gpu_fault_release.regional_release_prerequisite_baseline import (
    BOOTSTRAP_PROGRESS_FIELDS,
    CLEANUP_STEPS,
    capture_baseline_representation,
    validated_prerequisite_baseline,
)


def context(
    *, bootstrap: bool = True, legacy: bool = False
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline: dict[str, Any] = {
        "phase": "bootstrap-started" if bootstrap else "complete",
        "release_id": "candidate",
        "previous": None,
        "transaction_committed": False,
        "runtime_image": "image@sha256:" + "a" * 64,
        "rendered_manifest_sha256": "b" * 64,
        "updated_at_epoch": 10,
    }
    record: dict[str, Any] = {
        "binding": {"release_id": "candidate", "commit_live": False},
        "baseline_sha256": canonical_sha256(baseline),
        "baseline_had_timestamp": True,
        "baseline_timestamp": 10,
        "snapshot_representation": capture_baseline_representation(
            baseline, bootstrap=bootstrap and not legacy
        ),
        "baseline_commit_fields": {
            key: {"present": key in baseline, "value": baseline.get(key)}
            for key in (
                "transaction_committed",
                "release_lifecycle",
                "commit_cleanup_completed",
            )
        },
    }
    if not legacy:
        record["binding"]["bootstrap"] = bootstrap
    state = copy.deepcopy(baseline)
    state[AURORA_PREREQUISITE_REPAIR_KEY] = record
    return state, record, baseline


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "phase",
    [
        "bootstrap-started",
        "bootstrap-failed",
        "bootstrap-cleanup-started",
        "bootstrap-cleanup-progress",
        "bootstrap-cleaned",
    ],
)
def test_owned_bootstrap_bookkeeping_preserves_the_full_baseline(
    phase: str, legacy: bool
) -> None:
    state, record, original = context(legacy=legacy)
    state.update(
        phase=phase,
        updated_at_epoch=20,
        completed_cluster_ids=[],
        bootstrap_cleanup_completed_steps=(
            sorted(CLEANUP_STEPS) if phase == "bootstrap-cleaned" else []
        ),
        bootstrap_cleanup_failure=None,
        resume_phase="bootstrap-started",
    )
    before = copy.deepcopy(state)
    assert validated_prerequisite_baseline(state, record) == original
    assert state == before, "baseline validation must not edit the live journal"


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "updates",
    [
        {"runtime_image": "foreign-image"},
        {"release_id": "foreign-release"},
        {"rendered_manifest_sha256": "c" * 64},
        {"unexpected_field": "foreign"},
        {"approved_manifest_sha256": "c" * 64},
    ],
)
def test_progress_normalization_never_hides_other_state_drift(
    legacy: bool, updates: dict[str, Any]
) -> None:
    state, record, _original = context(legacy=legacy)
    state.update(
        phase="bootstrap-cleaned",
        completed_cluster_ids=[],
        bootstrap_cleanup_completed_steps=sorted(CLEANUP_STEPS),
        bootstrap_cleanup_failure=None,
        resume_phase="bootstrap-started",
        **updates,
    )
    with pytest.raises(ReleaseError, match="baseline drifted"):
        validated_prerequisite_baseline(state, record)


@pytest.mark.parametrize(
    "updates",
    [
        {"phase": "complete"},
        {"phase": "bootstrap-unknown"},
        {"phase": []},
        {"previous": {"release_id": "application"}},
        {"transaction_committed": True},
        {"completed_cluster_ids": ["unexpected"]},
        {"completed_cluster_ids": ["duplicate", "duplicate"]},
        {"completed_cluster_ids": "cluster"},
        {"bootstrap_cleanup_completed_steps": ["unknown"]},
        {"bootstrap_cleanup_completed_steps": ["cpu-scaled-down", "cpu-scaled-down"]},
        {"bootstrap_cleanup_completed_steps": {}},
        {"phase": "bootstrap-cleaned", "bootstrap_cleanup_completed_steps": []},
        {"bootstrap_cleanup_failure": False},
        {"resume_phase": "unknown"},
        {"resume_phase": []},
    ],
)
def test_invalid_bootstrap_progress_is_not_an_authorized_transition(
    updates: dict[str, Any],
) -> None:
    state, record, _original = context()
    state.update(updates)
    with pytest.raises(ReleaseError, match="bootstrap"):
        validated_prerequisite_baseline(state, record)


@pytest.mark.parametrize(
    "updates",
    [
        {"completed_cluster_ids": ["already-applied"]},
        {"bootstrap_store_safety": {"safe": True}},
        {"bootstrap_database_origin": {"safe": True}},
        {"retained_database_origin": {"safe": True}},
        {"bootstrap_cleanup_failure": "cleanup remains unconfirmed"},
        {"phase": "bootstrap-cpu-ready"},
        {"resume_phase": "bootstrap-cpu-ready"},
    ],
)
def test_legacy_recovery_is_only_for_initial_unadopted_bootstrap(
    updates: dict[str, Any],
) -> None:
    state, record, _original = context(legacy=True)
    state.update(
        phase="bootstrap-cleaned",
        completed_cluster_ids=[],
        bootstrap_cleanup_completed_steps=sorted(CLEANUP_STEPS),
        bootstrap_cleanup_failure=None,
        resume_phase="bootstrap-started",
    )
    state.update(updates)
    with pytest.raises(ReleaseError, match="baseline drifted"):
        validated_prerequisite_baseline(state, record)


def test_existing_bootstrap_completion_cannot_grow_before_repair_adoption() -> None:
    state, record, original = context()
    original["completed_cluster_ids"] = ["existing"]
    state["completed_cluster_ids"] = ["existing"]
    record["baseline_sha256"] = canonical_sha256(original)
    record["snapshot_representation"] = capture_baseline_representation(
        original, bootstrap=True
    )
    assert validated_prerequisite_baseline(state, record) == original
    state["completed_cluster_ids"] = []
    assert validated_prerequisite_baseline(state, record) == original
    state["completed_cluster_ids"] = ["existing", "new"]
    with pytest.raises(ReleaseError, match="completion scope"):
        validated_prerequisite_baseline(state, record)


def test_baseline_capture_does_not_share_mutable_lists_with_live_state() -> None:
    state: dict[str, Any] = {
        "phase": "bootstrap-started",
        "completed_cluster_ids": ["first"],
    }
    captured = capture_baseline_representation(state, bootstrap=True)
    state["completed_cluster_ids"].append("second")
    assert captured["completed_cluster_ids"]["value"] == ["first"]


@pytest.mark.parametrize("defect", ["missing", "extra", "present-type", "absent-value"])
def test_incomplete_or_unknown_representation_cannot_reconstruct_baseline(
    defect: str,
) -> None:
    state, record, _original = context()
    representation = record["snapshot_representation"]
    if defect == "missing":
        representation.pop("phase")
    elif defect == "extra":
        representation["runtime_image"] = {"present": False, "value": None}
    elif defect == "present-type":
        representation["phase"]["present"] = 1
    else:
        representation["completed_cluster_ids"]["value"] = []
    with pytest.raises(ReleaseError, match="baseline representation"):
        validated_prerequisite_baseline(state, record)


@pytest.mark.parametrize("flag", [1, "true", None, []])
def test_bootstrap_binding_is_a_strict_boolean(flag: Any) -> None:
    state, record, _original = context()
    record["binding"]["bootstrap"] = flag
    with pytest.raises(ReleaseError, match="bootstrap binding"):
        validated_prerequisite_baseline(state, record)


def test_non_bootstrap_snapshot_and_commit_transitions_remain_supported() -> None:
    state, record, original = context(bootstrap=False)
    record["binding"]["commit_live"] = True
    state.update(
        previous_snapshot={"immutable": "reference"},
        previous_snapshot_sha256="d" * 64,
        transaction_committed=True,
        release_lifecycle="COMMITTED",
        commit_cleanup_completed=False,
        updated_at_epoch=30,
    )
    assert validated_prerequisite_baseline(state, record) == original
    state["phase"] = "failed"
    with pytest.raises(ReleaseError, match="commit is incomplete"):
        validated_prerequisite_baseline(state, record)


def test_bootstrap_progress_is_not_normalized_for_an_upgrade() -> None:
    state, record, _original = context(bootstrap=False)
    state.update(phase="bootstrap-started", completed_cluster_ids=[])
    with pytest.raises(ReleaseError, match="baseline drifted"):
        validated_prerequisite_baseline(state, record)
    assert set(record["snapshot_representation"]).isdisjoint(
        BOOTSTRAP_PROGRESS_FIELDS
    ), "bootstrap normalization must not weaken the upgrade baseline"


def test_absent_original_timestamp_remains_absent_in_baseline() -> None:
    state, record, original = context()
    original.pop("updated_at_epoch")
    record.update(
        baseline_had_timestamp=False,
        baseline_timestamp=None,
        baseline_sha256=canonical_sha256(original),
    )
    state["updated_at_epoch"] = 30
    assert validated_prerequisite_baseline(state, record) == original
