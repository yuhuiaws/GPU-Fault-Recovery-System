"""GF-GENERAL-LIVE-011 NCCL stall verdicts over scripted EFA windows and records."""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import general_live_nccl as nc
from tests.regional.test_general_live_nccl import (
    HUNG_REASON,
    NODE,
    PEER,
    SINCE,
    STARTED,
    hung_records,
)


def reading(at: float, total: int, *, complete: bool = True) -> dict[str, Any]:
    return {
        "observed_at": at,
        "pair_bytes_total": total,
        "port_count": 16,
        "complete_pairs": complete,
    }


def test_a_baseline_without_any_usable_interval_names_each_gap() -> None:
    window = {NODE: [reading(0, 0, complete=False), reading(30, 10, complete=False)]}

    assert nc.baseline_errors(window, minimum_bps=1000.0, minimum_intervals=2) == [
        f"{NODE}: 1 of 1 EFA counter interval(s) unusable "
        "(incomplete pair, probe error or counter reset)",
        f"{NODE}: 0 usable EFA interval(s), need 2",
    ]


def test_a_resume_window_without_an_interval_is_named() -> None:
    assert nc.resume_errors({NODE: [reading(0, 0)]}, floor_bps=10.0) == [
        f"{NODE}: no usable EFA interval after the release"
    ]


def test_executions_without_an_integer_step_index_are_not_counted() -> None:
    records = hung_records(SINCE, ops=("FREEZE_EVIDENCE",))
    workflow = records["workflows"]["wf"]
    workflow["step_executions"] = [dict(workflow["step_executions"][0], step_index="0")]

    assert nc.execution_errors(records) == ["step 0 FREEZE_EVIDENCE never executed"]


def test_reported_nodes_come_from_the_reasons_when_the_event_id_names_none() -> None:
    incident = {
        "event_id": "manual-hung-report",
        "reasons": [HUNG_REASON, f"{PEER}{nc.CONCURRENT}{HUNG_REASON}"],
    }

    assert nc.reported_nodes(incident) == {PEER}


def test_both_nodes_needs_a_merged_incident_naming_every_attempt_node() -> None:
    assert nc.both_nodes_errors({"incidents": []}, [NODE, PEER]) == [
        "no merged hung incident to cover the attempt nodes"
    ]
    records = hung_records(SINCE)
    records["incidents"][0]["node_ids"] = [NODE]

    assert nc.both_nodes_errors(records, [NODE, PEER]) == [
        f"the merged hung incident does not name {[PEER]}"
    ]


def test_a_hung_incident_without_a_creation_time_cannot_be_timed() -> None:
    records = hung_records(SINCE)
    records["incidents"][0]["created_at"] = None

    assert nc.timing_errors(records, STARTED, [NODE, PEER]) == [
        "the hung incident has no creation time"
    ]


def test_a_repeat_read_without_the_earlier_view_still_judges_the_second_wave() -> None:
    first = hung_records(SINCE)
    second = hung_records(SINCE, tag="b", workflow="wf-2", predecessor="wf")

    assert nc.repeat_errors(first, second, {"incidents": []}) == []


def test_triage_failures_join_a_list_of_causes() -> None:
    records = hung_records(
        SINCE, triage_failures={NODE: ["ObservationIncomplete", "RESULT_SIZE_LIMIT"]}
    )

    assert nc.triage_node_failures(records) == {
        NODE: "ObservationIncomplete; RESULT_SIZE_LIMIT"
    }


def test_fr_errors_need_a_hung_incident_and_a_triage_execution() -> None:
    assert nc.fr_errors({"incidents": []}, expect_configured=True) == []
    records = hung_records(SINCE, ops=("FREEZE_EVIDENCE", "ESCALATE_SUPPORT"))

    assert nc.fr_errors(records, expect_configured=True) == [
        "no COLLECT_HUNG_TRIAGE execution to read FR status from"
    ]
