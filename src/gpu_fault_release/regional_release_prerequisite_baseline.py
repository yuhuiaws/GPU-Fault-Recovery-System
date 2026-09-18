"""Preserve repair identity while allowing explicitly recorded bootstrap progress."""

from __future__ import annotations

import copy
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_diff import AURORA_PREREQUISITE_REPAIR_KEY
from gpu_fault_release.regional_release_progress import BOOTSTRAP_PHASES

SNAPSHOT_FIELDS = ("previous_snapshot", "previous_snapshot_sha256")
BOOTSTRAP_PROGRESS_FIELDS = (
    "phase",
    "completed_cluster_ids",
    "bootstrap_cleanup_completed_steps",
    "bootstrap_cleanup_failure",
    "resume_phase",
)
CLEANUP_STEPS = frozenset(
    {"installer-jobs-cancelled", "gpu-scaled-down", "cpu-scaled-down"}
)
EARLY_BOOTSTRAP_PHASES = BOOTSTRAP_PHASES - {
    "bootstrap-cpu-ready",
    "bootstrap-endpoint-ready",
    "bootstrap-data-plane-progress",
}


def capture_baseline_representation(
    state: dict[str, Any], *, bootstrap: bool
) -> dict[str, dict[str, Any]]:
    fields = (*SNAPSHOT_FIELDS, *(BOOTSTRAP_PROGRESS_FIELDS if bootstrap else ()))
    return {
        key: {"present": key in state, "value": copy.deepcopy(state.get(key))}
        for key in fields
    }


def _restore_fields(
    baseline: dict[str, Any], representation: Any, fields: set[str]
) -> None:
    if not isinstance(representation, dict) or set(representation) != fields:
        raise ReleaseError("prerequisite baseline representation is incomplete")
    for key, value in representation.items():
        if (
            not isinstance(value, dict)
            or set(value) != {"present", "value"}
            or type(value["present"]) is not bool
            or not value["present"]
            and value["value"] is not None
        ):
            raise ReleaseError("prerequisite baseline representation is invalid")
        if value["present"]:
            baseline[key] = copy.deepcopy(value["value"])
        else:
            baseline.pop(key, None)


def _valid_bootstrap_progress(state: dict[str, Any]) -> bool:
    phase = state.get("phase")
    clusters = state.get("completed_cluster_ids", [])
    steps = state.get("bootstrap_cleanup_completed_steps", [])
    failure = state.get("bootstrap_cleanup_failure")
    resume = state.get("resume_phase")
    return (
        isinstance(phase, str)
        and phase in BOOTSTRAP_PHASES
        and "previous" in state
        and state["previous"] is None
        and state.get("transaction_committed") is False
        and isinstance(clusters, list)
        and all(isinstance(item, str) and item for item in clusters)
        and len(clusters) == len(set(clusters))
        and isinstance(steps, list)
        and all(isinstance(item, str) and item in CLEANUP_STEPS for item in steps)
        and len(steps) == len(set(steps))
        and (failure is None or isinstance(failure, str))
        and (resume is None or isinstance(resume, str) and resume in BOOTSTRAP_PHASES)
        and (phase != "bootstrap-cleaned" or set(steps) == CLEANUP_STEPS)
    )


def _legacy_initial_baseline(
    state: dict[str, Any], baseline: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any] | None:
    if (
        "bootstrap" in record["binding"]
        or not _valid_bootstrap_progress(state)
        or state["phase"] not in EARLY_BOOTSTRAP_PHASES
        or state.get("completed_cluster_ids", []) != []
        or state.get("bootstrap_cleanup_failure") is not None
        or state.get("resume_phase") not in (None, "bootstrap-started")
        or state.get("release_id") != record["binding"].get("release_id")
        or any(
            key in state
            for key in (
                "bootstrap_store_safety",
                "bootstrap_database_origin",
                "retained_database_origin",
            )
        )
    ):
        return None
    # Older first-install journals predate these four owned bookkeeping writes.
    # Recovery is allowed only when the complete original hash proves the result.
    original = copy.deepcopy(baseline)
    original["phase"] = "bootstrap-started"
    for key in BOOTSTRAP_PROGRESS_FIELDS[1:]:
        original.pop(key, None)
    return (
        original
        if canonical_sha256(original) == record.get("baseline_sha256")
        else None
    )


def validated_prerequisite_baseline(
    state: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    try:
        if not isinstance(record["binding"], dict):
            raise ReleaseError("prerequisite baseline binding is invalid")
        bootstrap = record["binding"].get("bootstrap", False)
        if type(bootstrap) is not bool:
            raise ReleaseError("prerequisite bootstrap binding is invalid")
        baseline = copy.deepcopy(
            {
                key: value
                for key, value in state.items()
                if key != AURORA_PREREQUISITE_REPAIR_KEY
            }
        )
        if type(record["baseline_had_timestamp"]) is not bool:
            raise ReleaseError("prerequisite baseline timestamp binding is invalid")
        if record["baseline_had_timestamp"]:
            baseline["updated_at_epoch"] = record["baseline_timestamp"]
        else:
            baseline.pop("updated_at_epoch", None)
        fields = {*SNAPSHOT_FIELDS, *(BOOTSTRAP_PROGRESS_FIELDS if bootstrap else ())}
        _restore_fields(baseline, record["snapshot_representation"], fields)
        if bootstrap:
            if not _valid_bootstrap_progress(state):
                raise ReleaseError("prerequisite bootstrap progress is invalid")
            original_clusters = baseline.get("completed_cluster_ids", [])
            if state.get("completed_cluster_ids", []) not in (original_clusters, []):
                raise ReleaseError("prerequisite bootstrap completion scope changed")
        if (
            record["binding"].get("commit_live")
            and state.get("transaction_committed") is True
        ):
            if (
                state.get("phase") != "complete"
                or state.get("release_lifecycle") != "COMMITTED"
                or type(state.get("commit_cleanup_completed")) is not bool
            ):
                raise ReleaseError(
                    "prerequisite baseline commit is incomplete or unknown"
                )
            _restore_fields(
                baseline,
                record["baseline_commit_fields"],
                {
                    "transaction_committed",
                    "release_lifecycle",
                    "commit_cleanup_completed",
                },
            )
    except (KeyError, TypeError) as exc:
        raise ReleaseError("Aurora prerequisite repair baseline is incomplete") from exc
    if canonical_sha256(baseline) == record.get("baseline_sha256"):
        return baseline
    legacy = _legacy_initial_baseline(state, baseline, record)
    if legacy is not None:
        return legacy
    raise ReleaseError("Aurora prerequisite repair baseline drifted")
