"""Unit tests for the GF-REGIONAL-PREEMPT-036 seeding and verdict module.

The runner owns the throwaway database, the probe subprocesses and the evidence
file. Everything decidable without those lives in
``scripts.e2e.regional.preempt036_verdicts`` and is tested here: the three stuck
shapes are seeded against a real SqliteStore and swept by the product's own
dispatcher, and every verdict function is also shown to fail on the drift it
exists to catch.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.store import SqliteStore
from scripts.e2e.regional.preempt036_verdicts import (
    CASE_ID,
    COMPILE_BLOCKED_SHAPE,
    DISPATCHER_ACTOR,
    ORPHANED_COMMANDS_SHAPE,
    RETIRED_GENERATION_SHAPE,
    SHAPES,
    ShapeSeed,
    all_errors,
    case_verdict,
    command_errors,
    command_snapshot,
    event_errors,
    held_errors,
    incident_errors,
    incident_snapshot,
    open_command_errors,
    record_errors,
    rerun_errors,
    safety_errors,
    seed_shape,
    sweep,
    workflow_snapshot,
)


def _store(tmp_path: Any, shape: str) -> SqliteStore:
    return SqliteStore(str(tmp_path / f"{shape}.db"))


def _snapshots(store: Any, seed: ShapeSeed) -> dict[str, Any]:
    return {
        "workflows": workflow_snapshot(store, seed.workflow_ids),
        "incidents": incident_snapshot(store, seed.incident_ids),
        "commands": command_snapshot(store, seed.workflow_ids),
    }


def _swept(tmp_path: Any, shape: str) -> tuple[Any, ShapeSeed, dict, dict, list]:
    store = _store(tmp_path, shape)
    seed = seed_shape(shape, store)
    before = _snapshots(store, seed)
    held = sweep(store, passes=seed.passes)
    return store, seed, before, _snapshots(store, seed), held


@pytest.mark.parametrize("shape", SHAPES)
def test_the_sweep_closes_the_record_and_leaves_live_work_alone(
    tmp_path: Any, shape: str
) -> None:
    _store_, seed, before, after, held = _swept(tmp_path, shape)

    assert held_errors(held, seed) == [], f"{shape}: the held set is wrong"
    assert record_errors(before["workflows"], after["workflows"], seed) == [], (
        f"{shape}: the workflow records did not reach the promised state"
    )
    assert incident_errors(before["incidents"], after["incidents"], seed) == [], (
        f"{shape}: the incidents did not end as promised"
    )
    assert command_errors(before["commands"], after["commands"], seed) == [], (
        f"{shape}: the remote commands did not reach the promised state"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_every_written_record_is_attributed_to_the_dispatcher(
    tmp_path: Any, shape: str
) -> None:
    _store_, seed, _before, after, _held = _swept(tmp_path, shape)

    assert event_errors(after["workflows"], seed) == [], (
        f"{shape}: the audit trail is not what the shape promises"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_one_more_pass_changes_nothing(tmp_path: Any, shape: str) -> None:
    store, seed, _before, settled, _held = _swept(tmp_path, shape)

    sweep(store, passes=1)

    assert rerun_errors(settled, _snapshots(store, seed)) == [], (
        f"{shape}: the sweep is not idempotent on this shape"
    )


def test_only_the_retired_generation_needs_two_passes_and_holds_one_for_a_human(
    tmp_path: Any,
) -> None:
    seeds = {shape: seed_shape(shape, _store(tmp_path, shape)) for shape in SHAPES}

    assert {shape: seed.passes for shape, seed in seeds.items()} == {
        COMPILE_BLOCKED_SHAPE: 1,
        ORPHANED_COMMANDS_SHAPE: 1,
        RETIRED_GENERATION_SHAPE: 2,
    }
    assert seeds[RETIRED_GENERATION_SHAPE].held_after == ("p036rg-mutated",), (
        "the generation that already reset a GPU must stay held for an operator"
    )


def test_the_retired_generation_writes_no_operator_event(tmp_path: Any) -> None:
    seeds = {shape: seed_shape(shape, _store(tmp_path, shape)) for shape in SHAPES}

    assert {shape: seed.event_kind for shape, seed in seeds.items()} == {
        COMPILE_BLOCKED_SHAPE: "OPERATOR_RECONCILED",
        ORPHANED_COMMANDS_SHAPE: "OPERATOR_RECONCILED",
        RETIRED_GENERATION_SHAPE: None,
    }, "the per-shape audit contracts drifted from what the sweeps do"


def test_seed_shape_refuses_an_unknown_shape(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="unsupported stuck-workflow shape"):
        seed_shape("restore", SqliteStore(str(tmp_path / "unknown.db")))


def test_case_id_matches_the_catalog_entry() -> None:
    assert CASE_ID == "GF-REGIONAL-PREEMPT-036", (
        "the evidence case id must stay the catalog case id"
    )


def test_seed_identifiers_carry_the_case_number(tmp_path: Any) -> None:
    for shape in SHAPES:
        seed = seed_shape(shape, _store(tmp_path, shape))
        for request_id in (*seed.workflow_ids, *seed.incident_ids):
            assert request_id.startswith("p036"), request_id


# --------------------------------------------------------------------------- #
# Each verdict fails on the drift it exists to catch
# --------------------------------------------------------------------------- #
def test_record_errors_reports_a_status_a_missing_audit_and_a_moved_twin(
    tmp_path: Any,
) -> None:
    _store_, seed, before, after, _held = _swept(tmp_path, COMPILE_BLOCKED_SHAPE)
    drifted = {
        key: dict(value, status="BLOCKED", preemption_reason="")
        if key == "p036cb-blocked"
        else dict(value, fencing_token=99)
        for key, value in after["workflows"].items()
    }

    errors = record_errors(before["workflows"], drifted, seed)

    assert any("status is 'BLOCKED'" in error for error in errors), errors
    assert any("preemption_reason does not carry" in error for error in errors), errors
    assert any("untouched record changed" in error for error in errors), errors


def test_record_errors_reports_a_revoked_record_that_kept_its_owner(
    tmp_path: Any,
) -> None:
    _store_, seed, before, after, _held = _swept(tmp_path, RETIRED_GENERATION_SHAPE)
    drifted = dict(after["workflows"])
    drifted["p036rg-retired"] = dict(
        drifted["p036rg-retired"],
        execution_owner_id="p036-executor-a",
        preempted_by_workflow_id=None,
    )

    errors = record_errors(before["workflows"], drifted, seed)

    assert any("still holds an owner" in error for error in errors), errors
    assert any("preempted_by_workflow_id is None" in error for error in errors), errors


def test_incident_errors_reports_a_moved_incident_and_a_missing_audit_line(
    tmp_path: Any,
) -> None:
    _store_, seed, before, after, _held = _swept(tmp_path, RETIRED_GENERATION_SHAPE)
    drifted = {
        key: dict(value, reasons=[], workflow_request_id="somebody-else")
        for key, value in after["incidents"].items()
    }

    errors = incident_errors(before["incidents"], drifted, seed)

    assert any("do not carry" in error for error in errors), errors
    assert any("moved incident workflow_request_id" in error for error in errors), (
        errors
    )

    _store2, seed2, before2, after2, _held2 = _swept(tmp_path, COMPILE_BLOCKED_SHAPE)
    touched = {
        key: dict(value, reasons=[*value["reasons"], "x"])
        for key, value in after2["incidents"].items()
    }
    assert any(
        "incident changed" in error
        for error in incident_errors(before2["incidents"], touched, seed2)
    ), "a shape that does not write the incident must notice when one moved"


def test_event_errors_reports_a_wrong_actor_a_missing_detail_and_a_stray_event(
    tmp_path: Any,
) -> None:
    _store_, seed, _before, after, _held = _swept(tmp_path, ORPHANED_COMMANDS_SHAPE)
    drifted = dict(after["workflows"])
    (event,) = drifted["p036oc-failed"]["events"]
    drifted["p036oc-failed"] = dict(
        drifted["p036oc-failed"],
        events=[
            dict(
                event,
                actor="arn:aws:sts::1:assumed-role/x",
                details={**event["details"], "command_ids": []},
            )
        ],
    )
    drifted["p036oc-running"] = dict(
        drifted["p036oc-running"], events=[dict(event, actor=DISPATCHER_ACTOR)]
    )

    errors = event_errors(drifted, seed)

    assert any("event actor is" in error for error in errors), errors
    assert any("event command_ids is []" in error for error in errors), errors
    assert any("untouched record carries 1" in error for error in errors), errors


def test_event_errors_reports_a_closed_record_with_no_event(tmp_path: Any) -> None:
    _store_, seed, _before, after, _held = _swept(tmp_path, COMPILE_BLOCKED_SHAPE)
    silent = {key: dict(value, events=[]) for key, value in after["workflows"].items()}

    assert event_errors(silent, seed) == [
        "p036cb-blocked: 0 OPERATOR_RECONCILED event(s), expected exactly 1"
    ]


def test_command_errors_reports_a_wrong_status_source_and_a_live_command(
    tmp_path: Any,
) -> None:
    _store_, seed, before, after, _held = _swept(tmp_path, ORPHANED_COMMANDS_SHAPE)
    drifted = dict(after["commands"])
    drifted["p036oc-command-orphan"] = dict(
        drifted["p036oc-command-orphan"],
        status_source="agent",
        error="",
        lease_owner="still-here",
    )
    drifted["p036oc-command-live"] = dict(
        drifted["p036oc-command-live"], status="FAILED"
    )

    errors = command_errors(before["commands"], drifted, seed)

    assert any("status_source is 'agent'" in error for error in errors), errors
    assert any("error does not carry" in error for error in errors), errors
    assert any("still holds a lease" in error for error in errors), errors
    assert any("live command changed" in error for error in errors), errors


def test_held_errors_reports_a_wrong_pass_count_and_a_released_operator_record(
    tmp_path: Any,
) -> None:
    seed = seed_shape(RETIRED_GENERATION_SHAPE, _store(tmp_path, "rg"))

    assert held_errors([[]], seed) == ["sweep ran 1 pass(es), expected 2"]
    assert held_errors([["p036rg-retired", "p036rg-mutated"], []], seed) == [
        "the last pass still holds [], expected ['p036rg-mutated']"
    ]


def test_safety_errors_reports_a_blocker_that_did_not_clear(tmp_path: Any) -> None:
    seed = seed_shape(RETIRED_GENERATION_SHAPE, _store(tmp_path, "rg"))
    before = {
        "blockers": ["p036rg-current", "p036rg-mutated", "p036rg-retired"],
        "blocker_count": 3,
        "resolved_blocked": [],
        "resolved_blocked_count": 0,
        "compile_blocked": [],
        "compile_blocked_count": 0,
    }
    after = dict(before)

    errors = safety_errors(before, after, seed)

    assert errors == [
        "blockers after are ['p036rg-current', 'p036rg-mutated', 'p036rg-retired'], "
        "expected ['p036rg-current', 'p036rg-mutated']"
    ]


def test_safety_errors_accepts_blockers_that_must_survive_and_counts_that_agree(
    tmp_path: Any,
) -> None:
    seed = seed_shape(ORPHANED_COMMANDS_SHAPE, _store(tmp_path, "oc"))
    snapshot = {
        "blockers": ["p036oc-running"],
        "blocker_count": 1,
        "resolved_blocked": [],
        "resolved_blocked_count": 0,
        "compile_blocked": [],
        "compile_blocked_count": 0,
    }

    assert safety_errors(snapshot, snapshot, seed) == []


def test_safety_errors_reports_a_compile_blocked_record_the_probe_still_counts(
    tmp_path: Any,
) -> None:
    seed = seed_shape(COMPILE_BLOCKED_SHAPE, _store(tmp_path, "cb"))
    before = {
        "blockers": ["p036cb-blocked"],
        "blocker_count": 1,
        "resolved_blocked": ["p036cb-restore-twin"],
        "resolved_blocked_count": 1,
        "compile_blocked": [],
        "compile_blocked_count": 1,
    }
    after = {
        "blockers": [],
        "blocker_count": 0,
        "resolved_blocked": ["p036cb-restore-twin"],
        "resolved_blocked_count": 1,
        "compile_blocked": [],
        "compile_blocked_count": 0,
    }

    errors = safety_errors(before, after, seed)

    assert any("blockers before are ['p036cb-blocked']" in error for error in errors), (
        errors
    )
    assert any("compile_blocked before are []" in error for error in errors), errors
    assert any(
        "compile_blocked_count before is 1 for 0" in error for error in errors
    ), errors


def test_open_command_errors_reports_a_counter_that_did_not_drop(tmp_path: Any) -> None:
    seed = seed_shape(ORPHANED_COMMANDS_SHAPE, _store(tmp_path, "oc"))
    stats = {"by_status": {"PENDING": 0, "LEASED": 0, "WAITING": 2}}

    assert open_command_errors(stats, stats, seed) == [
        "open remote commands dropped by 0, expected 1"
    ]
    with pytest.raises(ValueError, match="by_status"):
        open_command_errors({}, stats, seed)


def test_rerun_errors_reports_a_second_pass_that_changed_a_record(
    tmp_path: Any,
) -> None:
    _store_, _seed, _before, settled, _held = _swept(tmp_path, COMPILE_BLOCKED_SHAPE)
    rerun = {
        **settled,
        "workflows": {
            key: dict(value, events=[*value["events"], {"kind": "OPERATOR_RECONCILED"}])
            for key, value in settled["workflows"].items()
        },
    }

    errors = rerun_errors(settled, rerun)

    assert len(errors) == len(settled["workflows"])
    assert all("workflows: the rerun changed" in error for error in errors), errors


def test_verdict_and_error_aggregation_are_stage_ordered() -> None:
    stages = {"records": ["b"], "audit_events": [], "commands": ["a"]}

    assert case_verdict(stages) == "FAIL"
    assert case_verdict({"records": [], "commands": []}) == "PASS"
    assert all_errors(stages) == ["commands: a", "records: b"]
