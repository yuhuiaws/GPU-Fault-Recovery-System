from __future__ import annotations

import copy
from typing import Any

import pytest

from gpu_fault_release import regional_release_prerequisite_repair as repair
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_diff import (
    build_execution_plan,
    diff_from_changed,
)
from tests.regional._prerequisite_repair_support import repair_release

DIFF = diff_from_changed({"aurora_refresh_drift"})
PLAN = build_execution_plan(DIFF)


def prepare(instance: Any, **kwargs: Any) -> Any:
    return repair.prepare_prerequisite_repair(instance, diff=DIFF, plan=PLAN, **kwargs)


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("schema", "context is invalid"),
        ("binding", "candidate or snapshot binding differs"),
        ("baseline", "baseline is incomplete"),
        ("status", "status is unknown"),
        ("jobs", "Job journal is invalid"),
        ("pending-job", "repair or cleanup is incomplete"),
    ],
)
def test_adoption_rejects_incomplete_repair_journal(
    monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    instance = repair_release(monkeypatch)
    prepare(instance)
    record = instance.state[repair.REPAIR_KEY]
    if fault == "schema":
        record["schema_version"] = 2
    elif fault == "binding":
        record["binding_sha256"] = "wrong"
    elif fault == "baseline":
        record.pop("baseline_had_timestamp")
    elif fault == "status":
        record["status"] = "UNKNOWN"
    elif fault == "jobs":
        record["jobs"] = []
    else:
        record["jobs"]["store"] = {"status": "RUNNING"}
    before = list(instance.runner.events)
    with pytest.raises(ReleaseError, match=problem):
        repair.adopted_refresher_snapshot(instance)
    assert instance.runner.events == before


def test_repair_keeps_absent_timestamp_and_present_empty_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    instance.state.pop("updated_at_epoch")
    instance.state["previous_snapshot"] = None
    baseline = copy.deepcopy(instance.state)
    prepare(instance)
    assert repair.matches_pre_repair_state(
        instance, instance.state, canonical_sha256(baseline)
    ), "repair bookkeeping changed the baseline timestamp or snapshot representation"


def test_baseline_commit_requires_known_committed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    instance.state["transaction_committed"] = False
    prepare(instance, commit_live=True)
    baseline = instance.state[repair.REPAIR_KEY]["baseline_sha256"]
    instance.state.update(transaction_committed=True, release_lifecycle="FAILED")
    with pytest.raises(ReleaseError, match="baseline commit is incomplete"):
        repair.matches_pre_repair_state(instance, instance.state, baseline)


def test_checkpoint_refuses_new_job_before_previous_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    prepare(instance)
    jobs = instance.state[repair.REPAIR_KEY]["jobs"]
    jobs["store"] = {"name": "previous-job", "status": "RUNNING"}
    before = copy.deepcopy(jobs)
    checkpoint = repair.repair_job_checkpoint(instance, "store")
    with pytest.raises(ReleaseError, match="cleanup is incomplete"):
        checkpoint({"name": "new-job", "status": "PLANNED"})
    assert jobs == before


def test_repair_rechecks_database_binding_before_another_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    prepare(instance)
    record = instance.state[repair.REPAIR_KEY]
    record["binding"]["database"]["endpoint"] = "different.example.invalid"
    record["binding_sha256"] = canonical_sha256(record["binding"])
    before = list(instance.runner.events)
    with pytest.raises(ReleaseError, match="database identity drifted"):
        prepare(instance)
    assert "apply" not in instance.runner.events[len(before) :]


@pytest.mark.parametrize("dry_run", [False, True])
def test_unselected_or_dry_run_repair_has_no_transport(
    monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    instance = repair_release(monkeypatch)
    instance.runner.dry_run = dry_run
    plan = PLAN if dry_run else build_execution_plan(diff_from_changed(set()))
    assert repair.prepare_prerequisite_repair(instance, diff=DIFF, plan=plan) is None
    assert instance.runner.events == []


def test_supplied_previous_refresher_remains_the_repair_rollback_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    snapshot = repair.capture_aurora_refresh_snapshot(instance)
    snapshot["objects"][-1]["spec"]["schedule"] = "13 * * * *"
    record = prepare(instance, previous_override={"aurora_refresh": snapshot})
    assert record["previous_refresher"] == snapshot


@pytest.mark.parametrize(
    "mode,problem",
    [
        ("resume", "resumed repair transaction identity"),
        ("incomplete", "requires resume"),
        ("commit", "baseline commit identity differs"),
    ],
)
def test_repair_does_not_bypass_application_transaction_binding(
    monkeypatch: pytest.MonkeyPatch, mode: str, problem: str
) -> None:
    instance = repair_release(monkeypatch)
    if mode == "incomplete":
        instance.state.update(
            transaction_committed=False, previous={"release_id": "old"}
        )
    with pytest.raises(ReleaseError, match=problem):
        prepare(instance, resume=mode == "resume", commit_live=mode == "commit")
    assert "apply" not in instance.runner.events
    assert instance.runner.created_jobs == []


def test_standalone_repair_rollback_stops_before_application_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    original = copy.deepcopy(instance.runner.live)
    prepare(instance)
    with pytest.raises(ReleaseError, match="rerun rollback"):
        repair.restore_standalone_prerequisite(instance)
    assert instance.runner.live == original
    assert repair.REPAIR_KEY not in instance.state


@pytest.mark.parametrize("absent", [False, True])
def test_refresher_program_must_exist_and_forbid_consumer_restarts(
    monkeypatch: pytest.MonkeyPatch, absent: bool
) -> None:
    instance = repair_release(monkeypatch)
    read = repair.read_aurora_refresh_cronjob

    def unsafe(release: Any) -> Any:
        if absent:
            return None
        value = read(release)
        value["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0][
            "args"
        ] = ["--restart-deployments"]
        return value

    monkeypatch.setattr(repair, "read_aurora_refresh_cronjob", unsafe)
    with pytest.raises(ReleaseError, match="is absent|may restart consumers"):
        prepare(instance)
    assert instance.runner.created_jobs == []
    assert instance.state[repair.REPAIR_KEY]["status"] == "FAILED"


@pytest.mark.parametrize("origin", [False, [], {}, {"finished_at": "invalid"}])
def test_bootstrap_origin_must_be_a_bound_dated_empty_database_proof(
    monkeypatch: pytest.MonkeyPatch, origin: Any
) -> None:
    instance = repair_release(monkeypatch, bootstrap=True)
    instance.state[repair.BOOTSTRAP_ORIGIN_KEY] = origin
    with pytest.raises(ReleaseError, match="origin proof is missing or unbound"):
        repair.bootstrap_workflow_proof(instance)
    assert instance.runner.created_jobs == []


def test_bootstrap_dry_run_does_not_prepare_or_query_the_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch, bootstrap=True)
    instance.runner.dry_run = True
    repair.prepare_bootstrap_workflows(instance)
    assert instance.runner.events == []


def test_upgrade_credentials_load_existing_state_before_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = repair_release(monkeypatch)
    instance.state = {}
    record = repair.prepare_upgrade_credentials(instance, diff=DIFF, plan=PLAN)
    assert record["status"] == "READY"
    assert instance.state["release_id"] == "old"
    assert instance.runner.jobs == {}
