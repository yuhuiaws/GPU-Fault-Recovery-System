"""GF-GENERAL-LIVE-024 ``xid_twin_*`` oracles on hand-built product records.

``test_general_live_xid_twins_product.py`` drives every variant end to end over
the in-memory product, where the records come out right; here each oracle of
``general_live_xid_twin_records`` is handed the record shapes a broken product
(or a broken premise) leaves behind -- a report that never arrived, a clock that
runs backwards, a decision that names the wrong twin, a repeat read as the
earlier event, a remote command that ran a step the workflow never executed, a
remediation that stopped half way, a job restart without its ledger -- and must
name each one in the stage the verdict reports.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.e2e.regional import general_live_xid_twin_records as twin_records

G1 = "GPU-11111111-1111-4111-8111-111111111111"
START = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
#: The catalog's IGNORE decision of the safe variants (XID 92).
MONITOR = twin_records.TwinExpectation(
    xid=92,
    disposition="MONITOR_ONLY",
    official_action="IGNORE",
    actions=frozenset({"NO_ACTION", None}),
)
#: A remediating variant: one RESET_GPU chain, never a quarantine or a reboot.
RESET = twin_records.TwinExpectation(
    xid=94,
    disposition="RECOVERABLE",
    official_action="RESET_GPU",
    actions=frozenset({"RESET_GPU"}),
    executed={"RESET_GPU": 1},
    absent=frozenset({"QUARANTINE"}),
    chain=("STOP_WORKLOADS", "RESET_GPU"),
    forbidden=frozenset({"REBOOT_NODE"}),
    hardware=frozenset({"RESET_GPU"}),
)


def _at(seconds: float) -> str:
    return (START + timedelta(seconds=seconds)).isoformat()


def _event(
    event_id: str,
    source: str,
    seconds: float | None,
    *,
    linked: str | None = None,
    **decision: Any,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_source": source,
        "gpu_uuid": G1,
        "observed_at": _at(seconds) if seconds is not None else None,
        "linked_incident_id": linked,
        "decision": {
            "disposition": "MONITOR_ONLY",
            "official_action": "IGNORE",
            "action": "NO_ACTION",
            **decision,
        },
    }


# --------------------------------------------------------------------------- #
# Lookups and the product's clock
# --------------------------------------------------------------------------- #
def test_an_unknown_incident_is_an_empty_record_and_a_bad_stamp_no_gap() -> None:
    records = {"incidents": [{"incident_id": "inc-a", "state": "RECOVERED"}]}

    assert twin_records.incident_of(records, "inc-b") == {}
    assert twin_records.incident_of(records, None) == {}, (
        "no incident id matches nothing, even an incident without one"
    )
    assert twin_records.incident_of(records, "inc-a")["state"] == "RECOVERED"
    assert twin_records.seconds_between("not-a-timestamp", _at(0)) is None
    assert twin_records.seconds_between(None, _at(0)) is None
    assert twin_records.seconds_between(_at(0), "") is None
    assert twin_records.seconds_between(_at(45), _at(0)) == 45.0


# --------------------------------------------------------------------------- #
# The pair of reports (premises, INCONCLUSIVE)
# --------------------------------------------------------------------------- #
def test_the_pair_names_each_report_that_never_reached_the_product() -> None:
    pair, problems = twin_records.twin_pair(
        {"events": []}, gpu=G1, first_source="KERNEL_LOG", xid=92
    )
    assert pair is None
    assert problems == [
        f"no field-230 change of XID 92 on {G1} reached the product",
        f"no kernel line of XID 92 on {G1} reached the product",
    ]

    gauge = _event("g", "DCGM_EXPORTER", 0)
    other_gpu = {**_event("k", "KERNEL_LOG", 10), "gpu_uuid": "GPU-other"}
    pair, problems = twin_records.twin_pair(
        {"events": [gauge, other_gpu]}, gpu=G1, first_source="KERNEL_LOG", xid=92
    )
    assert pair is None
    assert problems == [f"no kernel line of XID 92 on {G1} reached the product"], (
        "a kernel line of another GPU is no report of this occurrence"
    )


def test_a_second_served_change_is_a_premise_failure_not_a_pair() -> None:
    kernel = _event("k", "KERNEL_LOG", 0)
    gauges = [_event("g1", "DCGM_EXPORTER", 20), _event("g2", "DCGM_EXPORTER", 40)]
    pair, problems = twin_records.twin_pair(
        {"events": [kernel, *gauges]}, gpu=G1, first_source="KERNEL_LOG", xid=92
    )
    assert pair is None
    assert len(problems) == 1 and problems[0].startswith(
        f"2 field-230 changes of XID 92 on {G1}"
    ), problems
    assert "a real XID would have aborted the replay" in problems[0], problems


def test_a_change_the_product_dates_before_its_leading_line_is_no_pair() -> None:
    kernel = _event("k", "KERNEL_LOG", 30)
    gauge = _event("g", "DCGM_EXPORTER", 10)
    pair, problems = twin_records.twin_pair(
        {"events": [kernel, gauge]}, gpu=G1, first_source="KERNEL_LOG", xid=92
    )
    assert pair is None
    assert problems == [
        "the product's clock puts the DCGM_EXPORTER report before the KERNEL_LOG "
        "one (-20.0 s)"
    ]

    undated = _event("g", "DCGM_EXPORTER", None)
    pair, problems = twin_records.twin_pair(
        {"events": [kernel, undated]}, gpu=G1, first_source="KERNEL_LOG", xid=92
    )
    assert pair is None and "(None s)" in problems[0], (
        f"a change without observed_at cannot be placed: {problems}"
    )


# --------------------------------------------------------------------------- #
# The decision classes and the one occurrence
# --------------------------------------------------------------------------- #
def test_a_report_without_a_saved_decision_is_named_before_its_fields() -> None:
    event = {"event_id": "k", "event_source": "KERNEL_LOG", "decision": None}
    assert twin_records.decision_class_errors(event, MONITOR, label="kernel line") == [
        "the kernel line has no saved FaultPolicyDecision"
    ]

    wrong = _event(
        "k",
        "KERNEL_LOG",
        0,
        disposition="RECOVERABLE",
        official_action="RESET_GPU",
        action="RESET_GPU",
        safety_action="QUARANTINE",
    )
    errors = twin_records.decision_class_errors(wrong, MONITOR, label="kernel line")
    assert errors == [
        "the kernel line is RECOVERABLE, expected MONITOR_ONLY",
        "the kernel line's official action is RESET_GPU, expected IGNORE",
        "the kernel line earned recovery action RESET_GPU",
        "the kernel line earned safety action QUARANTINE",
    ]


def test_a_primary_that_names_another_report_and_no_incident_fails_its_stage() -> None:
    primary = _event("k", "KERNEL_LOG", 0, linked="inc-stale", correlated_event_id="x")
    joiner = _event(
        "g", "DCGM_EXPORTER", 20, linked=None, correlated_event_id="k", duplicate=True
    )
    pair = twin_records.TwinPair(
        first=primary, second=joiner, joiner=joiner, primary=primary
    )
    stages = twin_records.twin_errors({"incidents": []}, pair, MONITOR)

    assert stages["primary"] == [
        "the KERNEL_LOG report k names x: the report the occurrence is decided on "
        "names no other",
        "the KERNEL_LOG report k names no incident",
        "the store links the KERNEL_LOG report k to inc-stale, its decision names None",
    ]
    assert stages["twin"] == [], stages
    assert stages["one_incident"] == ["incident None is not in the store"], (
        "an occurrence whose decision names no incident has none in the store"
    )
    assert pair.gap_seconds == 20.0


def test_an_occurrence_incident_missing_from_the_store_is_named() -> None:
    primary = _event("k", "KERNEL_LOG", 0, linked="inc-k", incident_id="inc-k")
    joiner = _event(
        "g",
        "DCGM_EXPORTER",
        20,
        linked="inc-k",
        incident_id="inc-k",
        correlated_event_id="k",
        duplicate=True,
    )
    pair = twin_records.TwinPair(
        first=primary, second=joiner, joiner=joiner, primary=primary
    )
    stages = twin_records.twin_errors(
        {"incidents": [{"incident_id": "inc-other", "event_id": "zz"}]}, pair, MONITOR
    )
    assert stages == {
        "primary": [],
        "twin": [],
        "one_incident": ["incident inc-k is not in the store"],
    }


# --------------------------------------------------------------------------- #
# The repeat outside the twin window
# --------------------------------------------------------------------------- #
def test_a_repeat_that_never_reached_the_store_is_its_only_error() -> None:
    assert twin_records.repeat_errors(
        {"events": [_event("k", "KERNEL_LOG", 0)]},
        "r",
        occurrence_incident_id="inc-k",
        expectation=MONITOR,
    ) == ["the repeated kernel line r is not in the store"]


def test_a_duplicate_repeat_needs_a_marker_association_to_be_excused() -> None:
    repeat = _event(
        "r", "KERNEL_LOG", 300, linked="inc-r", incident_id="inc-r", duplicate=True
    )
    records = {
        "events": [repeat],
        "incidents": [{"incident_id": "inc-r", "state": "RECOVERED"}],
    }
    errors = twin_records.repeat_errors(
        records, "r", occurrence_incident_id="inc-k", expectation=MONITOR
    )
    assert errors == [
        "the repeated kernel line r is flagged duplicate with no marker association: "
        "the repeat was read as the earlier event"
    ]

    repeat["decision"]["reasons"] = [
        f"{twin_records.MARKER_ASSOCIATION}inc-marker; provider marker active"
    ]
    assert twin_records.marker_associations(repeat["decision"]) == ["inc-marker"]
    assert (
        twin_records.repeat_errors(
            records, "r", occurrence_incident_id="inc-k", expectation=MONITOR
        )
        == []
    ), "a marker association is the one reason a repeat may be flagged duplicate"


def test_a_monitor_only_repeat_must_mint_no_workflow_and_close_recovered() -> None:
    repeat = _event(
        "r",
        "KERNEL_LOG",
        300,
        linked="inc-r",
        incident_id="inc-r",
        workflow_request_id="wf-r",
    )
    records = {
        "events": [repeat],
        "incidents": [{"incident_id": "inc-r", "state": "OPEN"}],
    }
    errors = twin_records.repeat_errors(
        records, "r", occurrence_incident_id="inc-k", expectation=MONITOR
    )
    assert errors == [
        "the repeated kernel line r minted workflow wf-r",
        "the repeated kernel line r's incident is OPEN; a monitor-only decision "
        "closes RECOVERED on creation",
    ]
    reset_repeat = _event(
        "r",
        "KERNEL_LOG",
        300,
        linked="inc-r",
        incident_id="inc-r",
        workflow_request_id="wf-r",
        disposition="RECOVERABLE",
        official_action="RESET_GPU",
        action="RESET_GPU",
    )
    assert (
        twin_records.repeat_errors(
            {"events": [reset_repeat], "incidents": records["incidents"]},
            "r",
            occurrence_incident_id="inc-k",
            expectation=RESET,
        )
        == []
    ), "a remediating repeat earns its own workflow on its own open incident"


# --------------------------------------------------------------------------- #
# Step executions and remote-command receipts
# --------------------------------------------------------------------------- #
ACTIVITY = {
    "workflows": [
        {
            "request_id": "wf-1",
            "step_executions": [
                {"operation": "RESET_GPU", "status": "SUCCEEDED", "step_index": 1},
                {"operation": "QUARANTINE", "status": "FAILED", "step_index": 2},
            ],
        }
    ],
    "commands": [
        {
            "workflow_request_id": "wf-1",
            "status": "SUCCEEDED",
            "step_index": 1,
            "step": {"operation": "RESET_GPU"},
        },
        {
            "workflow_request_id": "wf-1",
            "status": "FAILED",
            "step_index": 3,
            "step": {"operation": "STOP_WORKLOADS"},
            "batched_steps": [
                {"step_index": 3, "step": {"operation": "STOP_WORKLOADS"}},
                {"step_index": 4, "step": {"operation": "QUARANTINE"}},
                {"step_index": 5, "step": {"operation": "RESTART_WORKLOAD"}},
            ],
            "result_details": {
                "batched_results": {
                    "3": {"status": "FAILED"},
                    "4": {"status": "SUCCEEDED"},
                }
            },
        },
    ],
}


def test_receipts_count_direct_and_batched_successes_only() -> None:
    assert dict(twin_records.executed_steps(ACTIVITY)) == {
        "RESET_GPU": {("wf-1", 1)}
    }, "a FAILED step execution is no run"
    assert dict(twin_records.command_receipts(ACTIVITY)) == {
        "RESET_GPU": {("wf-1", 1)},
        "QUARANTINE": {("wf-1", 4)},
    }, "only SUCCEEDED commands and SUCCEEDED batched outcomes are receipts"


def test_a_receipt_without_its_step_execution_is_an_extra_run() -> None:
    errors, counts = twin_records.execution_errors(ACTIVITY, RESET)
    assert errors == [
        "QUARANTINE has remote-command receipts [('wf-1', 4)] beyond its step "
        "executions"
    ], "the batched QUARANTINE ran with no step execution of the workflow"
    assert counts == {
        "QUARANTINE": {"steps": [], "receipts": [["wf-1", 4]]},
        "RESET_GPU": {"steps": [["wf-1", 1]], "receipts": [["wf-1", 1]]},
    }


# --------------------------------------------------------------------------- #
# The occurrence's workflow and its chain
# --------------------------------------------------------------------------- #
def _workflow(request_id: str, incident_id: str, *ops: str, **extra: Any) -> dict:
    return {
        "request_id": request_id,
        "incident_id": incident_id,
        "created_at": extra.pop("created_at", _at(0)),
        "official_steps": [{"operation": item} for item in ops],
        **extra,
    }


def test_a_remediation_needs_exactly_one_workflow_and_no_other_actor() -> None:
    activity = {
        "workflows": [
            _workflow("wf-a", "inc-other", "VALIDATE_GPU", created_at=_at(5)),
            _workflow("wf-b", "inc-third", "RESET_GPU", created_at=_at(9)),
        ]
    }
    errors, mine = twin_records.workflow_errors(activity, "inc-k", RESET)
    assert mine == {}
    assert errors == [
        "0 workflows on the occurrence's incident inc-k, expected one",
        "workflow wf-b of incident inc-third also plans ['RESET_GPU']",
    ], "a validation-only workflow of another incident does not act on the node"

    activity["workflows"].append(
        _workflow("wf-k", "inc-k", "STOP_WORKLOADS", "REBOOT_NODE", created_at=_at(1))
    )
    errors, mine = twin_records.workflow_errors(activity, "inc-k", RESET)
    assert mine["request_id"] == "wf-k"
    assert errors == [
        "workflow wf-b of incident inc-third also plans ['RESET_GPU']",
        "workflow wf-k plans ['REBOOT_NODE']",
    ]


def test_the_chain_names_a_missing_unfinished_or_failed_remediation() -> None:
    assert twin_records.chain_errors({}, RESET) == [
        "the occurrence's remediation workflow is not in the product's records"
    ]
    running = _workflow("wf-k", "inc-k", "RESET_GPU", status="RUNNING")
    assert twin_records.chain_errors(running, RESET) == [
        "workflow steps ['RESET_GPU'] lack the chain ['STOP_WORKLOADS', 'RESET_GPU']",
        "the remediation is still RUNNING at the end of its budget",
        "STOP_WORKLOADS has no step execution",
        "RESET_GPU has no step execution",
    ]
    failed = _workflow(
        "wf-k",
        "inc-k",
        "STOP_WORKLOADS",
        "RESET_GPU",
        status="FAILED",
        terminal_failure_reason="reset refused",
        step_executions=[
            {"operation": "STOP_WORKLOADS", "status": "SUCCEEDED", "updated_at": "1"},
            {
                "operation": "RESET_GPU",
                "status": "FAILED",
                "updated_at": "2",
                "error": "nvidia-smi -r exited 1",
            },
        ],
    )
    assert twin_records.chain_errors(failed, RESET) == [
        "the remediation ended FAILED: reset refused",
        "RESET_GPU ended FAILED: nvidia-smi -r exited 1",
    ]


def test_the_incident_state_is_recovered_or_at_least_not_quarantined() -> None:
    records = {
        "incidents": [
            {"incident_id": "inc-open", "state": "REMEDIATING"},
            {"incident_id": "inc-q", "state": "QUARANTINED"},
        ]
    }
    assert twin_records.incident_state_errors(records, "inc-open") == [
        "the occurrence's incident inc-open is REMEDIATING, not RECOVERED"
    ]
    assert twin_records.incident_state_errors(records, "inc-q", recovered=False) == [
        "the occurrence's incident inc-q is QUARANTINED"
    ]
    assert (
        twin_records.incident_state_errors(records, "inc-open", recovered=False) == []
    ), "an open incident is fine where recovery is not required yet"


# --------------------------------------------------------------------------- #
# The partial-GPU job's stop and restart
# --------------------------------------------------------------------------- #
def test_every_defect_of_the_jobs_stop_and_restart_is_named() -> None:
    records = {
        "attempts": {
            "attempt-1": {
                "event": {"terminal_status": "FAILED"},
                "decision": {"status": "RESTART"},
                "containment_incident": "inc-contain",
            }
        },
        "restart_budget": None,
    }
    assert twin_records.job_restart_errors(
        records, attempt_id="attempt-1", restart_attempt_id=None
    ) == [
        "the job's attempt terminal is FAILED, expected STOPPED by the occurrence's "
        "STOP_WORKLOADS",
        "the attempt's completion decision is RESTART; an incident-initiated stop is "
        "NO_ACTION",
        "a passive containment incident was opened for the stop",
        "RESTART_WORKLOAD recorded no restart_attempt_id",
        "the job has no restart ledger after its restart",
    ]
    clean = {
        "attempts": {
            "attempt-1": {
                "event": {"terminal_status": "STOPPED"},
                "decision": {"status": "NO_ACTION"},
            }
        },
        "restart_budget": {"restart_count": 1},
    }
    assert (
        twin_records.job_restart_errors(
            clean, attempt_id="attempt-1", restart_attempt_id="attempt-2"
        )
        == []
    ), "one stop and one ledgered restart is the expected record"
