"""The direct bootstrap entry must not overwrite a still-owned repair."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_prerequisite_repair as REPAIR
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_prerequisite_baseline import (
    validated_prerequisite_baseline,
)
from gpu_fault_release.regional_release_prerequisite_handoff import (
    validate_bootstrap_repair_handoff,
)
from tests.regional import test_release_bootstrap_handoff as handoff_support
from tests.regional.test_release_bootstrap_handoff import MUTATIONS, select_candidate

handoff_release = handoff_support.handoff_release


def test_absent_repair_entry_has_no_read_or_write_dependencies() -> None:
    instance = SimpleNamespace(state={})
    REPAIR.prepare_bootstrap_prerequisite_entry(instance)
    assert instance.state == {}, "an absent repair caused bootstrap entry mutation"


def test_same_candidate_entry_validates_without_repeating_jobs(
    handoff_release: Any,
) -> None:
    instance = handoff_release
    original = copy.deepcopy(instance.state)
    jobs = copy.deepcopy(instance.runner.created_jobs)
    REPAIR.prepare_bootstrap_prerequisite_entry(instance)
    assert instance.state == original, "same-candidate entry changed its journal"
    assert instance.runner.created_jobs == jobs, (
        "same-candidate entry repeated credential or Store proof Jobs"
    )
    assert not (MUTATIONS & set(instance.runner.events)), (
        "same-candidate entry performed a bootstrap mutation"
    )
    assert instance.approved_manifest_digest == original["approved_manifest_sha256"], (
        "same-candidate entry lost the original manifest pin"
    )


@pytest.mark.parametrize(
    "damage", ["baseline", "candidate", "database", "explicit-pin"]
)
def test_same_candidate_entry_rejects_drift_before_bootstrap_checkpoint(
    handoff_release: Any, damage: str
) -> None:
    instance = handoff_release
    if damage == "baseline":
        instance.state["foreign_progress"] = True
    elif damage == "candidate":
        instance.rendered_manifest_digest = "b" * 64
    elif damage == "database":
        instance.runner.database_resource_id = "replacement-database"
    else:
        instance.approved_manifest_digest = "b" * 64
    with pytest.raises(ReleaseError):
        REPAIR.prepare_bootstrap_prerequisite_entry(instance)
    assert not (MUTATIONS & set(instance.runner.events)), (
        "same-candidate drift reached a checkpoint or cleanup mutation"
    )


def test_foreign_entry_finishes_handoff_and_proof_without_generic_checkpoint(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = handoff_release
    previous = copy.deepcopy(instance.state[REPAIR.REPAIR_KEY])
    select_candidate(instance)

    def refuse_checkpoint(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("entry called the generic bootstrap checkpoint")

    monkeypatch.setattr(instance, "_save_state", refuse_checkpoint)
    REPAIR.prepare_bootstrap_prerequisite_entry(instance)
    record = instance.state[REPAIR.REPAIR_KEY]
    assert record["status"] == "READY", "entry did not prove the successor refresher"
    assert (
        record["binding"]["release_id"]
        == instance.state["release_id"]
        == instance.release_id
    ), "entry returned before the journal and candidate identity matched"
    assert record["previous_refresher"] == previous["previous_refresher"], (
        "entry lost the original pre-repair snapshot"
    )
    assert REPAIR.BOOTSTRAP_ORIGIN_KEY not in instance.state, (
        "entry prematurely adopted the bootstrap repair"
    )
    assert (
        instance.runner.events.count("credential-job")
        == instance.runner.events.count("store-job")
        == 1
    ), "foreign entry did not execute both current proofs"
    before = len(instance.runner.created_jobs)
    REPAIR.prepare_bootstrap_prerequisite_entry(instance)
    assert len(instance.runner.created_jobs) == before, (
        "re-entering the now-current candidate repeated proof Jobs"
    )


@pytest.mark.parametrize("failure", ["scope", "cleanup", "restore", "proof"])
def test_foreign_entry_propagates_failure_without_generic_error_checkpoint(
    handoff_release: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    instance = handoff_release
    previous = copy.deepcopy(instance.state[REPAIR.REPAIR_KEY])
    if failure == "scope":
        instance.runner.namespace_uid = "foreign-namespace"
    elif failure == "cleanup":
        job = copy.deepcopy(instance.runner.created_jobs[0])
        instance.runner.jobs[job["metadata"]["name"]] = job
        instance.runner.fail_delete = True
    elif failure == "restore":

        def fail_restore(*_args: Any) -> None:
            raise ReleaseError("restore failed")

        monkeypatch.setattr(REPAIR, "restore_aurora_refresh_snapshot", fail_restore)
    else:
        instance.runner.unsafe_store = True

    def refuse_checkpoint(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a refused handoff called generic bootstrap checkpoint")

    monkeypatch.setattr(instance, "_save_state", refuse_checkpoint)
    select_candidate(instance)
    with pytest.raises(ReleaseError):
        REPAIR.prepare_bootstrap_prerequisite_entry(instance)
    record = instance.runner.cloud_state[REPAIR.REPAIR_KEY]
    assert record["previous_refresher"] == previous["previous_refresher"], (
        "entry failure lost the original snapshot"
    )
    if failure != "proof":
        assert record["attempt_id"] == previous["attempt_id"], (
            "failed predecessor recovery overwrote its old journal"
        )
    else:
        assert record["binding"]["release_id"] == instance.release_id, (
            "Store refusal lost the durably prepared successor journal"
        )
    assert REPAIR.BOOTSTRAP_ORIGIN_KEY not in instance.state, (
        "a failed entry authorized bootstrap adoption"
    )


def set_historical_wheel_name(instance: Any, name: str) -> None:
    record = instance.state[REPAIR.REPAIR_KEY]
    baseline = validated_prerequisite_baseline(instance.state, record)
    baseline["wheel_config_map"] = name
    instance.state["wheel_config_map"] = name
    record["binding"]["wheel_config_map"] = name
    record["binding_sha256"] = canonical_sha256(record["binding"])
    record["baseline_sha256"] = canonical_sha256(baseline)


@pytest.mark.parametrize("version", ["0007", "0099", "0200", "v2.1"])
def test_old_wheel_version_is_not_compared_to_the_current_release_version(
    handoff_release: Any, version: str
) -> None:
    instance = handoff_release
    name = f"gpu-fault-control-plane-wheel-{version}-{instance.wheel_sha[:12]}"
    set_historical_wheel_name(instance, name)
    record = instance.state[REPAIR.REPAIR_KEY]
    select_candidate(instance)
    validated = validate_bootstrap_repair_handoff(
        instance, instance.state, identity=record["binding"]["database"]
    )
    assert validated is record, "a valid prior-version wheel name blocked recovery"


@pytest.mark.parametrize(
    "name",
    [
        "foreign-wheel-0099-{sha}",
        "gpu-fault-control-plane-wheel-0099-000000000000",
        "gpu-fault-control-plane-wheel-?-{sha}",
        "gpu-fault-control-plane-wheel-..-{sha}",
        "gpu-fault-control-plane-wheel-{sha}",
        "gpu-fault-control-plane-wheel-" + "v" * 254 + "-{sha}",
    ],
)
def test_old_wheel_name_still_requires_valid_scope_and_original_hash_suffix(
    handoff_release: Any, name: str
) -> None:
    instance = handoff_release
    set_historical_wheel_name(instance, name.format(sha=instance.wheel_sha[:12]))
    record = instance.state[REPAIR.REPAIR_KEY]
    select_candidate(instance)
    with pytest.raises(ReleaseError, match="predecessor snapshot"):
        validate_bootstrap_repair_handoff(
            instance, instance.state, identity=record["binding"]["database"]
        )
    assert not (MUTATIONS & set(instance.runner.events)), (
        "an invalid old artifact name authorized mutation"
    )
