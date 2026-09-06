"""Unit tests for the GF-REGIONAL-PREEMPT-036 seeding and verdict module.

The runner owns the throwaway database, the shipped-source subprocess and the
evidence file. Everything decidable without those lives in
``scripts.e2e.regional.preempt036_verdicts`` and is tested here: the three stuck
shapes are seeded against a real SqliteStore and driven through the real
``workflow-reconcile`` plan/apply entry points, and every verdict function is
also shown to fail on the drift it exists to catch.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import pytest

from gpu_fault.admin.compile_blocked import (
    apply_compile_blocked_plan,
    build_compile_blocked_plan,
)
from gpu_fault.admin.orphaned_commands import (
    apply_orphaned_commands_plan,
    build_orphaned_commands_plan,
)
from gpu_fault.retired_generation import (
    apply_retired_generation_plan,
    build_retired_generation_plan,
)
from gpu_fault.store import SqliteStore
from scripts.e2e.regional.preempt036_verdicts import (
    CANCELLED_STATUS_SOURCE,
    CASE_ID,
    COMPILE_BLOCKED_MODE,
    DRIFT_REFUSALS,
    INELIGIBLE_REFUSALS,
    MODES,
    ORPHANED_COMMANDS_MODE,
    REFERENCE,
    RERUN_NO_OP,
    RERUN_REFUSED,
    RETIRED_GENERATION_MODE,
    ModeSeed,
    all_errors,
    apply_errors,
    case_verdict,
    command_errors,
    command_snapshot,
    open_command_errors,
    plan_errors,
    record_errors,
    refusal_errors,
    rerun_errors,
    safety_errors,
    seed_mode,
    workflow_snapshot,
)

BUILDERS: Mapping[str, Callable[..., dict[str, Any]]] = {
    COMPILE_BLOCKED_MODE: build_compile_blocked_plan,
    ORPHANED_COMMANDS_MODE: build_orphaned_commands_plan,
    RETIRED_GENERATION_MODE: build_retired_generation_plan,
}
APPLIERS: Mapping[str, Callable[..., dict[str, Any]]] = {
    COMPILE_BLOCKED_MODE: apply_compile_blocked_plan,
    ORPHANED_COMMANDS_MODE: apply_orphaned_commands_plan,
    RETIRED_GENERATION_MODE: apply_retired_generation_plan,
}
TAMPERED_DIGEST = "0" * 64


def _store(tmp_path: Any, mode: str) -> SqliteStore:
    return SqliteStore(str(tmp_path / f"{mode}.db"))


def _apply(mode: str, store: Any, seed: ModeSeed, digest: str) -> dict[str, Any]:
    return APPLIERS[mode](
        store,
        workflow_ids=list(seed.actionable_ids),
        expected_plan_sha256=digest,
        reference=REFERENCE,
    )


@pytest.mark.parametrize("mode", MODES)
def test_seeded_plan_names_the_actionable_record_and_refuses_the_twin(
    tmp_path: Any, mode: str
) -> None:
    store = _store(tmp_path, mode)
    seed = seed_mode(mode, store)

    plan = BUILDERS[mode](store, list(seed.workflow_ids))

    assert plan_errors(plan, seed) == [], (
        f"{mode}: the seeded store does not produce the plan the case asserts"
    )
    assert set(seed.actionable_ids) & set(seed.refused_ids) == set(), (
        f"{mode}: a record cannot be both actionable and refused"
    )
    assert set(seed.actionable_ids) | set(seed.refused_ids) == set(seed.workflow_ids), (
        f"{mode}: the seed leaves a workflow neither actionable nor refused"
    )


@pytest.mark.parametrize("mode", MODES)
def test_apply_over_the_ineligible_twin_is_refused_by_name(
    tmp_path: Any, mode: str
) -> None:
    store = _store(tmp_path, mode)
    seed = seed_mode(mode, store)
    plan = BUILDERS[mode](store, list(seed.workflow_ids))

    with pytest.raises(ValueError) as raised:
        APPLIERS[mode](
            store,
            workflow_ids=list(seed.workflow_ids),
            expected_plan_sha256=str(plan["plan_sha256"]),
            reference=REFERENCE,
        )

    message = str(raised.value)
    assert refusal_errors("ineligible", message, INELIGIBLE_REFUSALS[mode]) == [], (
        f"{mode}: the refusal does not say the plan holds ineligible records"
    )
    for request_id in seed.refused_ids:
        assert (
            refusal_errors(request_id, message, seed.refusal_substrings[request_id])
            == []
        ), f"{mode}: the refusal does not give {request_id} its own reason"


@pytest.mark.parametrize("mode", MODES)
def test_apply_bound_to_a_tampered_plan_digest_is_refused(
    tmp_path: Any, mode: str
) -> None:
    store = _store(tmp_path, mode)
    seed = seed_mode(mode, store)
    BUILDERS[mode](store, list(seed.actionable_ids))

    with pytest.raises(ValueError) as raised:
        _apply(mode, store, seed, TAMPERED_DIGEST)

    assert (
        refusal_errors("tampered plan", str(raised.value), DRIFT_REFUSALS[mode]) == []
    ), f"{mode}: a tampered plan digest was not refused as plan drift"


@pytest.mark.parametrize("mode", MODES)
def test_apply_closes_the_record_and_leaves_live_work_alone(
    tmp_path: Any, mode: str
) -> None:
    store = _store(tmp_path, mode)
    seed = seed_mode(mode, store)
    plan = BUILDERS[mode](store, list(seed.actionable_ids))
    digest = str(plan["plan_sha256"])
    commands_before = command_snapshot(store, seed.workflow_ids)

    result = _apply(mode, store, seed, digest)

    assert apply_errors(result, seed, approved_plan_sha256=digest) == [], (
        f"{mode}: the apply result does not match the approved plan"
    )
    assert record_errors(workflow_snapshot(store, seed.statuses_after), seed) == [], (
        f"{mode}: the workflow records did not reach the promised state"
    )
    assert (
        command_errors(
            commands_before, command_snapshot(store, seed.workflow_ids), seed
        )
        == []
    ), f"{mode}: the remote commands did not reach the promised state"


@pytest.mark.parametrize("mode", MODES)
def test_rerunning_the_same_reconcile_changes_nothing(tmp_path: Any, mode: str) -> None:
    store = _store(tmp_path, mode)
    seed = seed_mode(mode, store)
    digest = str(BUILDERS[mode](store, list(seed.actionable_ids))["plan_sha256"])
    _apply(mode, store, seed, digest)

    rerun_plan = BUILDERS[mode](store, list(seed.actionable_ids))
    rerun_result: dict[str, Any] | None = None
    output = ""
    if seed.rerun_contract == RERUN_REFUSED:
        with pytest.raises(ValueError) as raised:
            _apply(mode, store, seed, str(rerun_plan["plan_sha256"]))
        output = str(raised.value)
    else:
        rerun_result = _apply(mode, store, seed, str(rerun_plan["plan_sha256"]))

    assert (
        rerun_errors(seed, plan=rerun_plan, result=rerun_result, output=output) == []
    ), f"{mode}: the second reconcile pass did not honour this mode's contract"
    assert record_errors(workflow_snapshot(store, seed.statuses_after), seed) == [], (
        f"{mode}: the second pass moved a record that was already settled"
    )


def test_only_orphaned_commands_refuses_its_own_rerun() -> None:
    contracts = {mode: seed_contract(mode) for mode in MODES}

    assert contracts == {
        COMPILE_BLOCKED_MODE: RERUN_NO_OP,
        ORPHANED_COMMANDS_MODE: RERUN_REFUSED,
        RETIRED_GENERATION_MODE: RERUN_NO_OP,
    }, "the per-mode rerun contracts drifted from what the reconcile code does"


def seed_contract(mode: str) -> str:
    """The rerun contract a mode declares, read without touching a store."""

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        store = SqliteStore(f"{directory}/{mode}.db")
        return seed_mode(mode, store).rerun_contract


def test_seed_mode_refuses_an_unknown_mode(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="unsupported workflow-reconcile mode"):
        seed_mode("restore", SqliteStore(str(tmp_path / "unknown.db")))


def test_case_id_matches_the_catalog_entry() -> None:
    assert CASE_ID == "GF-REGIONAL-PREEMPT-036", (
        "the evidence case id must stay the catalog case id"
    )


def _seed(tmp_path: Any, mode: str) -> ModeSeed:
    return seed_mode(mode, _store(tmp_path, mode))


def test_plan_errors_reports_a_missing_digest_and_an_unplanned_record(
    tmp_path: Any,
) -> None:
    seed = _seed(tmp_path, COMPILE_BLOCKED_MODE)

    errors = plan_errors({"mode": seed.plan_mode, "items": []}, seed)

    assert any("sha256" in error for error in errors), (
        "a plan without a digest must be reported"
    )
    assert any("plan covers" in error for error in errors), (
        "a plan that skips the seeded records must be reported"
    )


def test_plan_errors_reports_an_eligibility_flag_that_flipped(tmp_path: Any) -> None:
    seed = _seed(tmp_path, COMPILE_BLOCKED_MODE)
    refused = seed.refused_ids[0]
    plan = {
        "mode": seed.plan_mode,
        "plan_sha256": "a" * 64,
        "items": [
            {"request_id": request_id, "eligible": True, "reasons": []}
            for request_id in seed.workflow_ids
        ],
    }

    errors = plan_errors(plan, seed)

    assert any(error.startswith(f"{refused}: plan eligible") for error in errors), (
        "a plan that calls the ineligible twin eligible must be reported"
    )
    assert any(
        error.startswith(f"{refused}: reasons do not name") for error in errors
    ), "a plan that drops the twin's refusal reason must be reported"


def test_plan_errors_reports_reasons_on_a_record_that_must_have_none(
    tmp_path: Any,
) -> None:
    seed = _seed(tmp_path, COMPILE_BLOCKED_MODE)
    actionable = seed.actionable_ids[0]
    plan = {
        "mode": seed.plan_mode,
        "plan_sha256": "a" * 64,
        "items": [
            {
                "request_id": request_id,
                **dict(seed.plan_flags.get(request_id) or {}),
                "reasons": ["unexpected"],
            }
            for request_id in seed.workflow_ids
        ],
    }

    errors = plan_errors(plan, seed)

    assert any(
        error.startswith(f"{actionable}: item was expected to carry no reasons")
        for error in errors
    ), "reasons on a record the plan must apply cleanly have to be reported"


def test_refusal_errors_needs_a_configured_substring() -> None:
    assert refusal_errors("rerun", "anything", "") == [
        "rerun: no refusal substring was configured"
    ], "an unconfigured refusal must not silently pass"
    assert refusal_errors("rerun", "boom", "boom") == [], (
        "a refusal that names the expected reason must pass"
    )


def test_apply_errors_reports_an_unbound_digest_and_a_deletion(tmp_path: Any) -> None:
    seed = _seed(tmp_path, COMPILE_BLOCKED_MODE)
    result = {
        "mode": seed.apply_mode,
        "reference": REFERENCE,
        "applied_workflow_ids": list(seed.actionable_ids),
        "records_deleted": 1,
        seed.approved_digest_field: TAMPERED_DIGEST,
        "settled_plan_sha256": TAMPERED_DIGEST,
    }

    errors = apply_errors(result, seed, approved_plan_sha256="b" * 64)

    assert any("records" in error for error in errors), (
        "an apply that deleted a record must be reported; reconcile never deletes"
    )
    assert any(seed.approved_digest_field in error for error in errors), (
        "an apply bound to a digest the operator never approved must be reported"
    )


def test_apply_errors_reports_a_wider_change_than_the_plan(tmp_path: Any) -> None:
    seed = _seed(tmp_path, COMPILE_BLOCKED_MODE)
    digest = "c" * 64
    result = {
        "mode": seed.apply_mode,
        "reference": REFERENCE,
        "applied_workflow_ids": [*seed.actionable_ids, *seed.refused_ids],
        "records_deleted": 0,
        seed.approved_digest_field: digest,
        "settled_plan_sha256": digest,
    }

    errors = apply_errors(result, seed, approved_plan_sha256=digest)

    assert any("apply changed" in error for error in errors), (
        "an apply that touched the refused twin must be reported"
    )


def test_apply_errors_reports_a_second_pass_digest_that_did_not_move(
    tmp_path: Any,
) -> None:
    seed = _seed(tmp_path, RETIRED_GENERATION_MODE)
    digest = "d" * 64
    result = {
        "mode": seed.apply_mode,
        "reference": REFERENCE,
        "applied_workflow_ids": list(seed.actionable_ids),
        "records_deleted": 0,
        seed.approved_digest_field: digest,
        "settled_plan_sha256": digest,
    }

    errors = apply_errors(result, seed, approved_plan_sha256=digest)

    assert seed.settled_digest_differs is True, (
        "retired-generation settles a second-pass digest, so it must differ"
    )
    assert any("cancel pass" in error for error in errors), (
        "a retired-generation apply whose cancel pass changed nothing must be reported"
    )


def test_record_errors_reports_a_status_and_a_missing_audit_reference(
    tmp_path: Any,
) -> None:
    seed = _seed(tmp_path, COMPILE_BLOCKED_MODE)
    closed = seed.actionable_ids[0]
    snapshot = {
        request_id: {
            "status": "RUNNING",
            "preemption_reason": "",
            "preempted_by_workflow_id": None,
            "execution_owner_id": None,
        }
        for request_id in seed.statuses_after
    }

    errors = record_errors(snapshot, seed)

    assert any(error.startswith(f"{closed}: status is") for error in errors), (
        "a record that never left RUNNING must be reported"
    )
    assert any(error.startswith(f"{closed}: preemption_reason") for error in errors), (
        "a closed record with no audit prose must be reported"
    )


def test_record_errors_reports_a_revoked_record_that_kept_its_owner(
    tmp_path: Any,
) -> None:
    seed = _seed(tmp_path, RETIRED_GENERATION_MODE)
    revoked = next(iter(seed.preempted_by))
    snapshot = {
        request_id: {
            "status": seed.statuses_after[request_id],
            "preemption_reason": " ".join(seed.reason_substrings.get(request_id) or ()),
            "preempted_by_workflow_id": seed.preempted_by.get(request_id),
            "execution_owner_id": "p035-executor-a",
        }
        for request_id in seed.statuses_after
    }

    errors = record_errors(snapshot, seed)

    assert [f"{revoked}: revoked record still holds an owner"] == errors, (
        "a revoked generation that kept its execution owner can still dispatch"
    )


def test_command_errors_reports_a_wrong_status_source_and_a_live_command(
    tmp_path: Any,
) -> None:
    seed = _seed(tmp_path, ORPHANED_COMMANDS_MODE)
    orphan = seed.cancelled_command_ids[0]
    live = seed.untouched_command_ids[0]
    before = {live: {"status": "WAITING", "lease_owner": None}}
    after = {
        orphan: {
            "status": "FAILED",
            "status_source": "node-agent",
            "error": f"operator reconciliation {REFERENCE}",
            "lease_owner": None,
        },
        live: {"status": "FAILED", "lease_owner": None},
    }

    errors = command_errors(before, after, seed)

    assert any(error.startswith(f"{orphan}: status_source") for error in errors), (
        "an orphan failed by something other than "
        f"{CANCELLED_STATUS_SOURCE!r} must be reported"
    )
    assert any(error.startswith(f"{live}: live command changed") for error in errors), (
        "cancelling the live workflow's command must be reported"
    )


def test_command_errors_reports_a_cancelled_command_that_kept_its_lease(
    tmp_path: Any,
) -> None:
    seed = _seed(tmp_path, ORPHANED_COMMANDS_MODE)
    orphan = seed.cancelled_command_ids[0]
    after = {
        orphan: {
            "status": "FAILED",
            "status_source": CANCELLED_STATUS_SOURCE,
            "error": f"operator reconciliation {REFERENCE}: cancelled",
            "lease_owner": "p035-node-agent",
        }
    }

    errors = command_errors({}, after, seed)

    assert errors == [f"{orphan}: cancelled command still holds a lease"], (
        "a cancelled command that kept its lease can still be executed"
    )


def test_safety_errors_reports_a_blocker_that_did_not_clear(tmp_path: Any) -> None:
    seed = _seed(tmp_path, COMPILE_BLOCKED_MODE)
    before = {
        "blockers": list(seed.blockers_before),
        "blocker_count": len(seed.blockers_before),
        "resolved_blocked": list(seed.resolved_blocked),
    }

    errors = safety_errors(before, before, seed)

    assert any("blockers after the apply" in error for error in errors), (
        "a release preflight that still names the closed record must be reported"
    )


def test_safety_errors_accepts_blockers_that_must_survive(tmp_path: Any) -> None:
    seed = _seed(tmp_path, RETIRED_GENERATION_MODE)
    before = {
        "blockers": list(seed.blockers_before),
        "blocker_count": len(seed.blockers_before),
        "resolved_blocked": list(seed.resolved_blocked),
    }
    after = {
        "blockers": list(seed.blockers_after),
        "blocker_count": len(seed.blockers_after),
        "resolved_blocked": list(seed.resolved_blocked),
    }

    assert safety_errors(before, after, seed) == [], (
        "the live successor and the human-owned record are blockers before and after"
    )
    assert seed.blockers_after, (
        "retired-generation must not claim it cleared the running successor"
    )


def test_safety_errors_reports_a_count_that_disagrees_with_the_list(
    tmp_path: Any,
) -> None:
    seed = _seed(tmp_path, COMPILE_BLOCKED_MODE)
    before = {
        "blockers": list(seed.blockers_before),
        "blocker_count": 99,
        "resolved_blocked": list(seed.resolved_blocked),
    }
    after = {
        "blockers": list(seed.blockers_after),
        "blocker_count": len(seed.blockers_after),
        "resolved_blocked": list(seed.resolved_blocked),
    }

    errors = safety_errors(before, after, seed)

    assert any("blocker_count before" in error for error in errors), (
        "a probe whose count disagrees with its own list must be reported"
    )


def test_open_command_errors_reports_a_counter_that_did_not_drop(tmp_path: Any) -> None:
    seed = _seed(tmp_path, ORPHANED_COMMANDS_MODE)
    stats = {"by_status": {"WAITING": 2, "PENDING": 0, "LEASED": 0}}

    assert open_command_errors(stats, stats, seed) != [], (
        "a release gate counter that never dropped must be reported"
    )
    assert (
        open_command_errors(
            stats, {"by_status": {"WAITING": 1, "PENDING": 0, "LEASED": 0}}, seed
        )
        == []
    ), "one orphan cancelled must drop the open-command counter by exactly one"


def test_open_command_errors_refuses_stats_without_by_status(tmp_path: Any) -> None:
    seed = _seed(tmp_path, ORPHANED_COMMANDS_MODE)

    with pytest.raises(ValueError, match="by_status"):
        open_command_errors({"total": 2}, {"total": 1}, seed)


def test_rerun_errors_reports_a_second_apply_that_changed_a_record(
    tmp_path: Any,
) -> None:
    seed = _seed(tmp_path, COMPILE_BLOCKED_MODE)
    plan = {
        "items": [
            {"request_id": request_id, "already_closed": True}
            for request_id in seed.actionable_ids
        ]
    }
    result = {
        "applied_workflow_ids": list(seed.actionable_ids),
        "already_closed_workflow_ids": list(seed.actionable_ids),
    }

    errors = rerun_errors(seed, plan=plan, result=result, output="")

    assert any("a second time" in error for error in errors), (
        "a rerun that wrote the record again must be reported"
    )


def test_rerun_errors_reports_a_missing_refusal(tmp_path: Any) -> None:
    seed = _seed(tmp_path, ORPHANED_COMMANDS_MODE)

    errors = rerun_errors(seed, plan=None, result=None, output="something else")

    assert errors == [
        f"rerun: refusal does not name {seed.rerun_refusal!r}, got 'something else'"
    ], "orphaned-commands must refuse its own rerun by name"


def test_verdict_and_error_aggregation_are_stage_ordered() -> None:
    stages = {"plan": [], "apply": ["b", "a"], "records": ["c"]}

    assert case_verdict(stages) == "FAIL", "any stage error must fail the case"
    assert case_verdict({"plan": [], "apply": []}) == "PASS", (
        "an empty stage set must pass"
    )
    assert all_errors(stages) == ["apply: b", "apply: a", "records: c"], (
        "aggregated errors must name their stage in stage order"
    )
