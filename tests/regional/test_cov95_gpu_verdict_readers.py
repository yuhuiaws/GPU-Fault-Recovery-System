"""GF-GENERAL-LIVE DCGM / GPU chain readers: the reasons each one gives.

``general_live_gpu_verdicts`` reads the product's records and the replay's
audit receipts and answers with human-readable errors. The category tests
(``test_general_live_gpu_chain``, ``..._recovery``, ``..._must_fail``) drive
these readers through whole variants; the cases here feed each reader the
record that makes it name a problem -- an incident without a workflow, a chain
that must not be present, a replay that aborted or never served the key, a
counter history with foreign and non-numeric samples, a finding the history
does not hold, a composite missing a component or missing altogether.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import general_live_gpu_verdicts as gv

GPU = "GPU-1d2c3b4a-0000-1111-2222-333344445555"
OTHER_GPU = "GPU-99999999-0000-1111-2222-333344445555"
CANONICAL = "gpu_dram_ecc_dbe_volatile_total"


def incident(incident_id: str, **changes: Any) -> dict[str, Any]:
    value = {
        "incident_id": incident_id,
        "event_type": "NODE_HEALTH",
        "event_id": f"gpu-b1-{GPU}-{CANONICAL}",
        "created_at": "2026-10-26T00:00:10+00:00",
        "state": "ESCALATED",
        "effective_action": "RUN_DIAGNOSTICS",
    }
    value.update(changes)
    return value


def workflow(incident_id: str, *operations: str, **changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "request_id": f"wf-{incident_id}",
        "incident_id": incident_id,
        "status": "RUNNING",
        "created_at": "2026-10-26T00:00:11+00:00",
        "official_steps": [{"operation": op} for op in operations],
        "step_executions": [],
    }
    value.update(changes)
    return value


# --------------------------------------------------------------------------- #
# chain readers
# --------------------------------------------------------------------------- #
def test_a_chain_without_a_workflow_has_not_executed() -> None:
    chain = gv.GpuChain("dbe", operations=("MARK_UNSCHEDULABLE",))
    activity = {"workflows": [workflow("inc-other", "MARK_UNSCHEDULABLE")]}

    assert gv.chain_executed(activity, [incident("inc-1")], chain) is False


def test_an_unexpected_chain_names_every_incident_it_minted() -> None:
    chain = gv.GpuChain("fabric", present=False)
    incidents = [incident("inc-1"), incident("inc-2")]

    judgement = gv.judge_chain({"workflows": []}, incidents, chain)

    assert judgement == gv.ChainJudgement(
        ["unexpected fabric incident inc-1", "unexpected fabric incident inc-2"], []
    )
    assert gv.judge_chain({}, [], chain) == gv.ChainJudgement([], []), (
        "an absent chain with no incident is the expected outcome"
    )


def test_an_incident_without_a_workflow_keeps_the_errors_found_before_it() -> None:
    chain = gv.GpuChain("dbe", actions=frozenset({"DRAIN"}))
    incidents = [incident("inc-1"), incident("inc-2")]

    errors = gv.chain_errors({"workflows": []}, incidents, chain)

    assert errors == [
        "2 dbe incidents (expected one): inc-1, inc-2",
        "dbe incident action RUN_DIAGNOSTICS, expected one of ['DRAIN']",
        "dbe incident inc-1 has no workflow",
    ]


def test_operation_count_and_step_started_read_every_workflow_of_the_incidents() -> (
    None
):
    first = workflow(
        "inc-1",
        "VALIDATE_GPU",
        "RESET_GPU",
        "VALIDATE_GPU",
        step_executions=[{"operation": "VALIDATE_GPU", "status": "SUCCEEDED"}],
    )
    second = workflow(
        "inc-1",
        "VALIDATE_GPU",
        created_at="2026-10-26T00:05:00+00:00",
        safety_steps=[{"operation": "MARK_UNSCHEDULABLE"}],
    )
    activity = {"workflows": [first, second, workflow("inc-9", "VALIDATE_GPU")]}
    incidents = [incident("inc-1")]

    assert gv.operation_count(activity, "VALIDATE_GPU", incidents) == 3
    assert gv.operation_count(activity, "MARK_UNSCHEDULABLE", incidents) == 1
    assert gv.operation_count(activity, "VALIDATE_GPU", []) == 0
    assert gv.step_started(activity, "VALIDATE_GPU", incidents) is True
    assert gv.step_started(activity, "RESET_GPU", incidents) is False, (
        "a planned step that has no execution has not started"
    )


# --------------------------------------------------------------------------- #
# replay audit
# --------------------------------------------------------------------------- #
KEY = f"{GPU}/{CANONICAL}"


def test_replay_effective_accepts_an_armed_read_that_served_the_key() -> None:
    records = [
        {"read": 1, "armed": False, "served": {}},
        {"read": 2, "armed": True, "served": {KEY: 1.0}},
    ]

    assert gv.replay_effective(records, KEY) == []


def test_replay_effective_names_an_abort_and_a_key_never_served() -> None:
    records = [
        {"read": 1, "armed": True, "served": {KEY: None}},
        {"read": 2, "armed": True, "aborted": "upstream lost GPU-1"},
        {"read": None, "armed": True, "served": {KEY: 5.0}},
        {"read": 3, "armed": True, "aborted": "upstream lost GPU-1"},
        {"read": 4, "armed": True, "aborted": "no scrape 2 reads old"},
    ]

    assert gv.replay_effective(records, KEY) == [
        "the replay aborted: no scrape 2 reads old; upstream lost GPU-1",
        f"the replay never served {KEY}",
    ]


# --------------------------------------------------------------------------- #
# product readings
# --------------------------------------------------------------------------- #
def test_counter_reset_ticks_skip_foreign_and_non_numeric_samples() -> None:
    state = {
        "batches": [
            {
                "batch_id": "b1",
                "observed_at": "2026-10-26T00:00:00+00:00",
                "samples": [
                    {"name": CANONICAL, "gpu": GPU, "value": 9},
                    {"name": CANONICAL, "gpu": OTHER_GPU, "value": 0},
                    {"name": "gpu_temperature_celsius", "gpu": GPU, "value": 0},
                ],
            },
            {
                "batch_id": "b2",
                "observed_at": "2026-10-26T00:00:30+00:00",
                "samples": [
                    {"name": CANONICAL, "gpu": GPU, "value": True},
                    {"name": CANONICAL, "gpu": GPU, "value": "3"},
                    {"name": CANONICAL, "gpu": GPU, "value": None},
                ],
            },
            {
                "batch_id": "b3",
                "observed_at": "2026-10-26T00:01:00+00:00",
                "edge_filter_reasons": ["counter-reset"],
                "samples": [{"name": CANONICAL, "gpu": GPU, "value": 2}],
            },
        ]
    }

    assert gv.counter_reset_ticks(state, CANONICAL, GPU) == [
        {
            "value": 2,
            "previous": 9,
            "observed_at": "2026-10-26T00:01:00+00:00",
            "batch_id": "b3",
            "edge_filter_reasons": ["counter-reset"],
        }
    ]


def test_an_incident_whose_finding_is_not_in_the_history_reads_as_absent() -> None:
    state = {
        "findings": [
            {"finding_id": "b0-other", "value": 1},
            {"finding_id": f"b1-{GPU}-{CANONICAL}", "value": 4, "delta": 4},
        ]
    }

    found = gv.incident_finding(state, incident("inc-1"))
    missing = gv.incident_finding(state, incident("inc-2", event_id="gpu-b7-x"))

    assert found == {"finding_id": f"b1-{GPU}-{CANONICAL}", "value": 4, "delta": 4}
    assert missing is None
    assert gv.finding_text(missing) == (
        "finding not in the product's history since the replay was armed"
    )
    assert gv.finding_text(found) == "value=4 delta=4 rate=None/min at None"


def test_renewal_without_any_accepted_reading_names_the_missing_reading() -> None:
    state = {
        "active_findings": [
            {
                "canonical_name": CANONICAL,
                "gpu_uuid": GPU,
                "value": 7,
                "delta": 7,
                "rate_per_minute": 7.0,
                "observed_at": "2026-10-26T00:02:00+00:00",
            }
        ],
        "batches": [
            {
                "observed_at": "2026-10-26T00:02:00+00:00",
                "samples": [{"name": CANONICAL, "gpu": GPU, "value": "7"}],
            }
        ],
    }

    errors = gv.renewal_errors(state, CANONICAL, GPU, need="recovery needs it")

    assert errors == [
        f"the {CANONICAL} finding on {GPU} is still active after the replay ended "
        "(value=7 delta=7 rate=7.0/min at 2026-10-26T00:02:00+00:00): "
        "recovery needs it",
        f"the product accepted no {CANONICAL} reading of {GPU}",
    ]


# --------------------------------------------------------------------------- #
# composites
# --------------------------------------------------------------------------- #
def test_composite_grade_without_a_finding_is_one_error() -> None:
    assert gv.composite_grade_errors(
        [], rule_id="THERMAL_STRESS", severity="WARNING", action="RUN_DIAGNOSTICS"
    ) == ["no THERMAL_STRESS composite finding was recorded"]


def test_composite_grade_names_the_components_it_lacks() -> None:
    findings = [
        {
            "finding_id": "c1",
            "severity": "WARNING",
            "automatic_action": "RUN_DIAGNOSTICS",
            "component_metrics": ["gpu_temperature_celsius"],
        }
    ]

    errors = gv.composite_grade_errors(
        findings,
        rule_id="THERMAL_STRESS",
        severity="WARNING",
        action="RUN_DIAGNOSTICS",
        required_components=("gpu_temperature_celsius", "gpu_power_watts"),
    )

    assert errors == [
        "THERMAL_STRESS components {'gpu_temperature_celsius'} lack ['gpu_power_watts']"
    ]


def test_composite_grade_with_no_component_at_all_shows_an_empty_set() -> None:
    findings = [{"finding_id": "c1", "severity": "WARNING", "automatic_action": "NONE"}]

    errors = gv.composite_grade_errors(
        findings,
        rule_id="THERMAL_STRESS",
        severity="WARNING",
        action="NONE",
        required_components=("gpu_power_watts",),
    )

    assert errors == ["THERMAL_STRESS components {} lack ['gpu_power_watts']"]


def test_component_dedup_without_a_composite_incident_is_one_error() -> None:
    errors = gv.component_dedup_errors(
        {"incidents": [incident("inc-1")]}, [], components=(CANONICAL,), gpu=GPU
    )

    assert errors == [f"no composite incident to de-duplicate against on {GPU}"]
