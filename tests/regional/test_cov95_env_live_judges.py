"""The GF-ENV-LIVE judges' rarer decisions, on hand-built polls.

Supporting logic coverage only: no cluster, node or store is touched, and a
passing test is never a LIVE PASS. Each test feeds ``env_live_judges`` the
documents one or more polls would carry (the GPU state, the node activity, the
Node) in the shapes ``tests/regional/_env_live_world.py`` serves, and pins
what the judge makes of a reading it must skip, a review chain it must fail or
leave INCONCLUSIVE at the deadline, and a read-only signal graded on the wrong
scale or never cleared.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.adapters.common import QUARANTINE_TAINT
from gpu_fault.health_recovery_contract import COOLDOWN_SECONDS
from scripts.e2e.regional import env_live_judges as judges
from scripts.e2e.regional.env_live_verdicts import ENVIRONMENT_REVIEW_OPERATIONS
from tests.regional._cov95_env_live_judges import (
    CLEAR_BATCH,
    CONSUMED,
    FIELD,
    HEALTHY,
    HOLD_END,
    REVIEW,
    REVIEW_HOLD,
    SEVERE,
    SEVERE_BATCHES,
    SOURCE,
    STAGES,
    WARNING,
    at,
    batch,
    expectation,
    finding,
    gpu_doc,
    incident,
    iso,
    judge_read_only,
    judge_review,
    latest,
    node_doc,
    notification,
    replay,
    review_scene,
    snapshot,
    unknown_rule,
    warning_rule,
    workflow,
)
from tests.regional._env_live_world import (
    BOOT,
    DIAGNOSTIC_OPERATIONS,
    GPU,
    REVIEW_OPERATIONS,
)


# --------------------------------------------------------------------------- #
# Intake and the shared facts
# --------------------------------------------------------------------------- #
def test_a_poll_counts_only_the_reads_it_carried_and_gpu_metrics_errors() -> None:
    judge = judges.judge_for(warning_rule(), repoint_at=at(-60))
    progress = replay(consumed=None, reads=2)
    statuses = [
        {
            "collector": "GPU_INVENTORY",
            "errors": ["inventory refresh failed"],
            "observed_at": iso(5),
        },
        {
            "collector": "GPU_METRICS",
            "errors": [f"invalid or ambiguous GPU observation: {GPU}/{FIELD}"],
            "observed_at": iso(5),
        },
    ]
    judge.observe(snapshot(5, gpu=gpu_doc(statuses=statuses)), progress)
    verdict = judge.decide(progress, at(30), final=True)
    assert verdict is not None and verdict.outcome == "INCONCLUSIVE", verdict
    assert verdict.evidence["store_reads"] == {
        "gpu_state": 1,
        "activity": 0,
        "node": 0,
    }, "a poll whose activity and Node reads failed counts only the GPU state"
    assert verdict.evidence["node"] == [], "no Node read, no node timeline entry"
    assert verdict.evidence["collector_errors"] == {
        f"invalid or ambiguous GPU observation: {GPU}/{FIELD}": iso(5)
    }, "only the GPU_METRICS collector's errors are kept"


def test_the_clear_after_the_trigger_skips_an_unparseable_stamp() -> None:
    judge = judges.judge_for(expectation("review"), repoint_at=at(-60))
    progress = replay()
    garbled = iso(45) + "-garbled"
    first = [
        batch("b-1", 45, {FIELD: SEVERE}),
        batch("b-garbled", garbled, {FIELD: HEALTHY}),
    ]
    judge.observe(snapshot(50, gpu=gpu_doc(batches=first)), progress)
    assert judge.after_trigger() is None, (
        "a reading whose stamp does not parse is no accepted clear"
    )
    judge.observe(
        snapshot(65, gpu=gpu_doc(batches=[batch("b-2", 60, {FIELD: HEALTHY})])),
        progress,
    )
    assert judge.after_trigger() == at(60), (
        "the first parseable reading after the trigger is the clear"
    )


def test_task_evidence_measures_owned_tasks_and_their_stop_executions() -> None:
    judge = judges.judge_for(expectation("review"), repoint_at=at(-60))
    owned = finding(
        "f-crit", 75, workload_state="ACTIVE", affected_workload_ids=["job-7"]
    )
    stopping = workflow(
        "wf-review",
        "f-crit",
        76,
        ENVIRONMENT_REVIEW_OPERATIONS,
        status="RUNNING",
        outcomes={"STOP_WORKLOADS": "FAILED"},
    )
    task = judge.task_evidence([owned], [stopping])
    assert task["owned_task"] == ["job-7"] and task["workload_state_active"], task
    assert task["measured"] is True and task["stopped"] is False, task
    assert task["stop_executions"] == [
        {"status": "FAILED", "error": "STOP_WORKLOADS refused"}
    ], task
    assert task["progress_measured"] is False, "no training-progress reporter"
    planned = workflow("wf-review", "f-crit", 76, ["STOP_WORKLOADS"], status="PENDING")
    task = judge.task_evidence([finding("f-crit", 75)], [planned])
    assert task["measured"] is True and task["owned_task"] is None, (
        "a planned STOP_WORKLOADS is measured even before it ran"
    )
    assert task["stop_executions"] == [] and task["stopped"] is False, task


def test_running_until_gives_an_unstamped_deadline_a_far_moment() -> None:
    now = at(200)
    flows = [
        {"status": "SUCCEEDED", "execution_deadline": iso(900)},
        {"status": "RUNNING"},
        {"status": "RUNNING", "execution_deadline": iso(400)},
    ]
    assert judges.ReviewJudge.running_until(flows, now) == now + timedelta(days=1), (
        "a running workflow without its own deadline keeps the window open"
    )
    assert judges.ReviewJudge.running_until(flows[::2], now) == at(400), (
        "a terminal workflow is skipped, a running one waits for its deadline"
    )


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-001 pass-through baseline
# --------------------------------------------------------------------------- #
def baseline_judge() -> judges.BaselineJudge:
    rule = expectation(
        "baseline",
        case_id="GF-ENV-LIVE-001",
        plan_name="passthrough",
        field=None,
        trigger_values=(),
        planned_reads=3,
        names=("gpu_temperature_c", "power_usage_w", "power_limit_w"),
        hold_seconds=0.0,
        threshold_source=None,
        boot_id=BOOT,
    )
    judge = judges.judge_for(rule, repoint_at=at(0), before=None)
    assert isinstance(judge, judges.BaselineJudge), judge
    return judge


def test_baseline_reads_only_the_repointed_collectors_batches_and_readings() -> None:
    judge = baseline_judge()
    progress = replay(consumed=None, armed=False, reads=1)
    gpu = gpu_doc(
        batches=[
            batch("b-old", -30, {"memory_temperature_c": 33.0}, "health-summary"),
            batch(
                "b-fresh",
                20,
                {"gpu_temperature_c": 31.0, "power_usage_w": 112.0},
                "initial-baseline",
            ),
        ],
        rows=[
            latest("power_limit_w", 25, 700.0),
            latest("clock_throttle_reasons", -30, 0.0),
        ],
    )
    judge.observe(snapshot(30, gpu=gpu, node=node_doc(held=False)), progress)
    assert judge.decide(progress, at(30), final=False) is None, (
        "an unconsumed pass-through waits for the deadline"
    )
    verdict = judge.decide(progress, at(90), final=True)
    assert verdict is not None and verdict.outcome == "INCONCLUSIVE", verdict
    assert verdict.inconclusive == [
        "the collector consumed 1 of 3 pass-through reads before the deadline"
    ], verdict.inconclusive
    assert not any(verdict.stages.values()), verdict.stages
    assert verdict.evidence["fresh_batch"]["batch_id"] == "b-fresh", (
        "the first initial-baseline batch after the repoint is the fresh one"
    )
    assert verdict.evidence["fresh_fields"] == [
        "gpu_temperature_c",
        "power_limit_w",
        "power_usage_w",
    ], "a pre-repoint batch or latest reading is never a pass-through reading"


def test_baseline_readings_from_another_boot_fail_at_once() -> None:
    judge = baseline_judge()
    progress = replay(consumed=None, armed=False, reads=1)
    fresh = batch(
        "b-fresh", 20, {"gpu_temperature_c": 31.0}, "initial-baseline", boot="boot-2"
    )
    judge.observe(snapshot(30, gpu=gpu_doc(batches=[fresh])), progress)
    verdict = judge.decide(progress, at(30), final=False)
    assert verdict is not None and verdict.outcome == "FAIL", verdict
    assert verdict.stages["baseline"] == [
        "GF-ENV-LIVE-001: accepted readings carry source_boot_id ['boot-2'], "
        f"the node booted {BOOT}"
    ], verdict.stages


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-004 review chain
# --------------------------------------------------------------------------- #
def test_review_merged_into_the_earlier_workflow_passes() -> None:
    scene = review_scene(
        findings=[finding("f-warn", 45, severity="WARNING"), finding("f-crit", 75)],
        incidents=[
            incident("f-warn", 46, state="ACTION_PENDING", workflow="wf-early"),
            incident("f-crit", 76, workflow="wf-early"),
        ],
        workflows=[workflow("wf-early", "f-warn", 46, REVIEW_OPERATIONS)],
    )
    verdict = judge_review(scene)
    assert verdict is not None and verdict.outcome == "PASS", verdict
    assert verdict.evidence["review_workflow"] == "wf-early", (
        "the confirmation merged into the earlier workflow is the review"
    )
    assert verdict.evidence["hold"]["held"] is True, verdict.evidence["hold"]


def test_review_after_an_isolating_pre_confirmation_workflow_fails() -> None:
    scene = review_scene(
        findings=[finding("f-warn", 60, severity="WARNING"), finding("f-crit", 75)],
        incidents=[
            incident("f-warn", 61, state="ACTION_PENDING", workflow="wf-early"),
            incident("f-crit", 76, workflow="wf-review"),
        ],
        workflows=[
            workflow(
                "wf-early",
                "f-warn",
                61,
                ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE"],
                status="RUNNING",
            ),
            workflow("wf-review", "f-crit", 76, REVIEW_OPERATIONS),
        ],
    )
    verdict = judge_review(scene, polls=(CONSUMED,))
    assert verdict is not None and verdict.outcome == "FAIL", verdict
    assert verdict.stages["reaction"] == [
        f"{REVIEW}: a pre-confirmation severe sample already planned "
        "['MARK_UNSCHEDULABLE'] (workflow wf-early)"
    ], verdict.stages


def test_review_confirmed_on_another_threshold_scale_fails() -> None:
    scene = review_scene(
        findings=[finding("f-crit", 75, threshold_source="NVIDIA_SLOWDOWN_LIMIT")]
    )
    verdict = judge_review(scene, polls=(CONSUMED,))
    assert verdict is not None and verdict.outcome == "FAIL", verdict
    assert verdict.stages["grading"] == [
        f"{REVIEW}: confirmed on NVIDIA_SLOWDOWN_LIMIT, the device grades on {SOURCE}"
    ], verdict.stages


@pytest.mark.parametrize("planned_by", ["review", "recovery"])
def test_restoring_scheduling_inside_the_review_hold_fails(planned_by: str) -> None:
    if planned_by == "review":
        flows = [
            workflow(
                "wf-review",
                "f-crit",
                76,
                [*REVIEW_OPERATIONS, "RESTORE_SCHEDULING"],
                status="RUNNING",
            )
        ]
    else:
        flows = [
            workflow("wf-review", "f-crit", 76, REVIEW_OPERATIONS),
            workflow("wf-restore", "f-recover", 80, ["RESTORE_SCHEDULING"]),
        ]
    verdict = judge_review(review_scene(workflows=flows), polls=(CONSUMED,))
    assert verdict is not None and verdict.outcome == "FAIL", verdict
    request = "wf-review" if planned_by == "review" else "wf-restore"
    assert verdict.stages["isolation"] == [
        f"{REVIEW}: workflow {request} restores scheduling inside the review hold"
    ], verdict.stages


ACTIVE_TASK = finding(
    "f-crit", 75, workload_state="ACTIVE", affected_workload_ids=["job-7"]
)
#: The review chain compiled for an ACTIVE attempt (with STOP_WORKLOADS).
WITH_STOP = ENVIRONMENT_REVIEW_OPERATIONS


@pytest.mark.parametrize(
    ("changes", "stage", "message"),
    [
        (
            {
                "workflows": [
                    workflow(
                        "wf-review",
                        "f-crit",
                        76,
                        [
                            name
                            for name in REVIEW_OPERATIONS
                            if name != "COLLECT_DIAGNOSTIC_BUNDLE"
                        ],
                    )
                ]
            },
            "workflow",
            "ENVIRONMENT_REVIEW workflow wf-review lacks ['COLLECT_DIAGNOSTIC_BUNDLE']",
        ),
        (
            {"findings": [ACTIVE_TASK]},
            "task",
            "the owned ACTIVE task ['job-7'] is not stopped by the review",
        ),
        (
            {
                "workflows": [
                    workflow(
                        "wf-review",
                        "f-crit",
                        76,
                        REVIEW_OPERATIONS,
                        status="FAILED",
                        pending_failure_error="QUARANTINE: taint patch refused",
                    )
                ]
            },
            "workflow",
            "ENVIRONMENT_REVIEW workflow ended FAILED: QUARANTINE: taint patch refused",
        ),
        (
            {
                "workflows": [
                    workflow(
                        "wf-review",
                        "f-crit",
                        76,
                        REVIEW_OPERATIONS,
                        status="BLOCKED",
                        blocked_reasons=["remediation budget exhausted"],
                    )
                ]
            },
            "workflow",
            "ENVIRONMENT_REVIEW workflow ended BLOCKED: ['remediation budget exhausted']",
        ),
        (
            {
                "findings": [ACTIVE_TASK],
                "workflows": [
                    workflow(
                        "wf-review",
                        "f-crit",
                        76,
                        WITH_STOP,
                        status="RUNNING",
                        outcomes={"STOP_WORKLOADS": "FAILED"},
                    )
                ],
            },
            "task",
            "the owned task's STOP_WORKLOADS failed",
        ),
        (
            {
                "workflows": [
                    workflow(
                        "wf-review",
                        "f-crit",
                        76,
                        REVIEW_OPERATIONS,
                        outcomes={"QUARANTINE": "FAILED"},
                    )
                ]
            },
            "workflow",
            "the review SUCCEEDED without receipts for ['QUARANTINE']",
        ),
    ],
    ids=[
        "lacks-receipt",
        "task-not-stopped",
        "failed",
        "blocked",
        "stop-failed",
        "no-receipts",
    ],
)
def test_a_wrong_review_workflow_fails_before_the_deadline(
    changes: dict[str, Any], stage: str, message: str
) -> None:
    verdict = judge_review(review_scene(**changes), polls=(CONSUMED,))
    assert verdict is not None and verdict.outcome == "FAIL", verdict
    assert any(
        item.startswith(f"{REVIEW}: {message}") for item in verdict.stages[stage]
    ), verdict.stages


def test_review_at_the_deadline_without_escalation_notice_or_isolation_fails() -> None:
    scene = review_scene(
        incidents=[incident("f-crit", 76, state="QUARANTINED", workflow="wf-review")],
        notifications=[],
    )
    assert judge_review(scene, held=False) is None, (
        "nothing definite before the deadline"
    )
    verdict = judge_review(scene, held=False, final=True)
    assert verdict is not None and verdict.outcome == "FAIL", verdict
    assert verdict.stages["workflow"] == [
        f"{REVIEW}: incident state is QUARANTINED, not ESCALATED",
        f"{REVIEW}: no delivered notification receipt for the review incident "
        "(statuses [])",
    ], verdict.stages
    assert verdict.stages["isolation"] == [
        f"{REVIEW}: node was not cordoned",
        f"{REVIEW}: quarantine taint {QUARANTINE_TAINT} is absent",
    ], verdict.stages
    assert verdict.inconclusive == [], "a failure carries no pending reasons"


@pytest.mark.parametrize(
    ("status", "cleared", "pending"),
    [
        (
            "QUEUED",
            True,
            [
                f"{REVIEW}: the review notification has no delivery result yet "
                "(['QUEUED'])",
                f"{REVIEW}: the replay window closed before the {REVIEW_HOLD:.0f}s "
                f"hold past the old {COOLDOWN_SECONDS}s cooldown ended "
                f"({at(HOLD_END)})",
            ],
        ),
        (
            None,
            False,
            [
                f"{REVIEW}: the review notification has no delivery result yet "
                "(['None'])",
                f"{REVIEW}: the cooled reading never reached the control plane, so "
                "the hold after cooling is not shown",
            ],
        ),
        (
            "SENT",
            True,
            [
                f"{REVIEW}: the replay window closed before the {REVIEW_HOLD:.0f}s "
                f"hold past the old {COOLDOWN_SECONDS}s cooldown ended "
                f"({at(HOLD_END)})"
            ],
        ),
    ],
    ids=["queued", "no-result-no-clear", "sent"],
)
def test_review_closed_before_its_hold_or_delivery_is_inconclusive(
    status: str | None, cleared: bool, pending: list[str]
) -> None:
    batches = [*SEVERE_BATCHES, CLEAR_BATCH] if cleared else list(SEVERE_BATCHES)
    scene = review_scene(
        batches=batches, notifications=[notification("f-crit", status)]
    )
    verdict = judge_review(scene, polls=(CONSUMED, 200.0), final=True)
    assert verdict is not None and verdict.outcome == "INCONCLUSIVE", verdict
    assert verdict.inconclusive == pending, verdict.inconclusive
    assert not any(verdict.stages.values()), verdict.stages


def test_the_deadline_fails_an_active_task_the_review_left_running() -> None:
    judge = judges.judge_for(expectation("review"), repoint_at=at(-60))
    judge.observe(snapshot(HOLD_END + 15, node=node_doc(held=True)), replay())
    facts = judges.ReviewFacts(
        run=3,
        finding=ACTIVE_TASK,
        incident=incident("f-crit", 76, workflow="wf-review"),
        workflow=workflow("wf-review", "f-crit", 76, WITH_STOP),
        receipts_ok=True,
        escalated=True,
        notification_statuses=["SENT"],
        held=True,
        task={"owned_task": ["job-7"], "workload_state_active": True, "stopped": False},
    )
    stages: dict[str, list[str]] = {name: [] for name in STAGES}
    assert judge.unfinished(facts, stages, at(HOLD_END + 15)) == [], (
        "a definite task failure leaves nothing pending"
    )
    assert stages["task"] == [f"{REVIEW}: the owned ACTIVE task was not stopped"], (
        stages
    )
    assert not any(items for name, items in stages.items() if name != "task"), stages


# --------------------------------------------------------------------------- #
# Read-only signals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "policy", "brake"),
    [
        (128.0, "SITE_NVIDIA_CLOCK_REASONS", True),
        (4.0, "SITE_NVIDIA_POWER_DELIVERY", True),
        (4.0, "SITE_NVIDIA_CLOCK_REASONS", False),
    ],
    ids=["brake-bit", "power-delivery-check", "ordinary-reason"],
)
def test_an_unusable_mask_read_as_a_power_brake_fails(
    value: float, policy: str, brake: bool
) -> None:
    graded = finding(
        "f-mask",
        30,
        name="clock_throttle_reasons",
        value=value,
        severity="WARNING",
        policy_source=policy,
    )
    judge, verdict = judge_read_only(unknown_rule(), findings=[graded])
    errors, proof = judge.grading()
    assert proof == [], "a finding never proves an unusable mask was graded"
    if brake:
        assert errors == [
            "GF-ENV-LIVE-008: an unusable mask was read as a Power Brake"
        ], errors
        assert verdict is not None and verdict.outcome == "FAIL", verdict
    else:
        assert errors == [], "an ordinary reason finding is no brake"
        assert verdict is None, "nothing definite before the deadline"


def test_a_warning_finding_on_another_value_proves_nothing() -> None:
    other = finding("f-other", 30, value=5.0, severity="WARNING")
    judge, verdict = judge_read_only(
        warning_rule(), findings=[other], batches=[batch("b-1", 30, {FIELD: WARNING})]
    )
    assert judge.grading() == ([], []), "a finding on an unreplayed value is skipped"
    assert verdict is None, "nothing definite before the deadline"
    _judge, verdict = judge_read_only(
        warning_rule(),
        findings=[other],
        batches=[batch("b-1", 30, {FIELD: WARNING})],
        final=True,
    )
    assert verdict is not None and verdict.outcome == "FAIL", verdict
    assert verdict.stages["grading"] == [
        f"GF-ENV-LIVE-003: 1 accepted {FIELD} sample(s) [{WARNING}] were never "
        "graded into a finding"
    ], verdict.stages


def test_a_warning_graded_on_another_scale_fails() -> None:
    graded = finding(
        "f-warn",
        30,
        value=WARNING,
        severity="WARNING",
        threshold_source="NVIDIA_SLOWDOWN_LIMIT",
    )
    _judge, verdict = judge_read_only(warning_rule(), findings=[graded])
    assert verdict is not None and verdict.outcome == "FAIL", verdict
    assert verdict.stages["grading"] == [
        "GF-ENV-LIVE-003: the warning was graded on NVIDIA_SLOWDOWN_LIMIT, the "
        f"device grades on {SOURCE}"
    ], verdict.stages


def test_an_omitted_field_is_traced_through_the_newest_latest_temperature() -> None:
    rows = [
        latest("gpu_temperature_c", 30, 31.0),
        latest("gpu_temperature_c", 10, 30.0),
        latest("gpu_temperature_c", "garbled", 29.0),
        latest("clock_throttle_reasons", -120, 0.0),
    ]
    judge, verdict = judge_read_only(unknown_rule(), rows=rows, final=True)
    assert judge.accepted_without_field(replay()) == [
        f"latest gpu_temperature_c at {iso(30)}"
    ], "an older or unparseable latest row never replaces the newest one"
    assert verdict is not None and verdict.outcome == "FAIL", verdict
    assert verdict.stages["grading"] == [
        f"GF-ENV-LIVE-008: ['latest gpu_temperature_c at {iso(30)}'] accepted the "
        "target GPU without its clock_throttle_reasons and nothing reported it "
        "UNKNOWN (the field-missing GPU was judged by the batch's other GPUs)"
    ], verdict.stages


def test_a_kind_without_a_grading_rule_never_passes() -> None:
    rule = expectation("interruption", case_id="GF-ENV-LIVE-009", hold_seconds=60.0)
    judge, verdict = judge_read_only(
        rule, batches=[batch("b-1", 30, {FIELD: SEVERE})], final=True
    )
    assert judge.graded(at(130)) == [], "no grading rule, no graded evidence"
    assert verdict is not None and verdict.outcome == "INCONCLUSIVE", verdict
    assert verdict.inconclusive == [
        f"GF-ENV-LIVE-009: no accepted record proves the replayed {FIELD} reached "
        "the control plane and was graded"
    ], verdict.inconclusive


WARN = finding("f-warn", 30, value=WARNING, severity="WARNING")
WARN_INCIDENT = incident("f-warn", 31, state="ACTION_PENDING", workflow="wf-diag")


def test_a_failed_diagnostic_or_validation_fails_the_read_only_case() -> None:
    failed = workflow(
        "wf-diag",
        "f-warn",
        31,
        DIAGNOSTIC_OPERATIONS,
        status="FAILED",
        outcomes={"FREEZE_EVIDENCE": "SUCCEEDED", "VALIDATE_GPU": "FAILED"},
        pending_failure_error="VALIDATE_GPU: dcgmi diag level 1 failed",
    )
    _judge, verdict = judge_read_only(
        warning_rule(), findings=[WARN], incidents=[WARN_INCIDENT], workflows=[failed]
    )
    assert verdict is not None and verdict.outcome == "FAIL", verdict
    assert verdict.stages["diagnostic"] == [
        "GF-ENV-LIVE-003: diagnostic workflow wf-diag ended FAILED: "
        "VALIDATE_GPU: dcgmi diag level 1 failed",
        "GF-ENV-LIVE-003: VALIDATE_GPU failed: VALIDATE_GPU refused",
    ], verdict.stages


@pytest.mark.parametrize("overdue", [True, False], ids=["overdue", "inside-deadline"])
def test_a_diagnostic_still_running_at_the_deadline(overdue: bool) -> None:
    running = workflow(
        "wf-diag",
        "f-warn",
        31,
        DIAGNOSTIC_OPERATIONS,
        status="RUNNING",
        deadline=60.0 if overdue else 1800.0,
    )
    _judge, verdict = judge_read_only(
        warning_rule(),
        findings=[WARN],
        incidents=[WARN_INCIDENT],
        workflows=[running],
        final=True,
    )
    assert verdict is not None, "the deadline always decides"
    if overdue:
        assert verdict.outcome == "FAIL", verdict
        assert verdict.stages["diagnostic"] == [
            "GF-ENV-LIVE-003: diagnostic workflow wf-diag is still running past "
            "its own execution deadline"
        ], verdict.stages
    else:
        assert verdict.outcome == "INCONCLUSIVE", verdict
        assert verdict.inconclusive == [
            "GF-ENV-LIVE-003: the replay window closed with the diagnostic "
            "workflow still running inside its own deadline",
            "GF-ENV-LIVE-003: the replay window closed before the 60s hold ended "
            f"({at(180).isoformat()})",
        ], verdict.inconclusive


def test_a_transient_not_yet_cleared_inside_its_settle_time_is_inconclusive() -> None:
    rule = expectation(
        "transient",
        case_id="GF-ENV-LIVE-005",
        plan_name="transient/once_then_clear",
        hold_seconds=60.0,
        max_confirmation=1,
        require_clear=True,
    )
    transient = finding("f-t", 110, severity="WARNING")
    judge, verdict = judge_read_only(
        rule,
        findings=[transient],
        active=[transient],
        batches=[batch("b-1", 110, {FIELD: SEVERE})],
        poll_at=150.0,
        final=True,
    )
    assert judge.cleared_at() is None, "the finding on the field is still active"
    assert verdict is not None and verdict.outcome == "INCONCLUSIVE", verdict
    assert verdict.inconclusive == [
        "GF-ENV-LIVE-005: the clear was not accepted before the deadline",
        "GF-ENV-LIVE-005: the replay window closed before the 60s hold ended "
        f"({at(210).isoformat()})",
    ], verdict.inconclusive
    assert verdict.evidence["graded_evidence"] == ["f-t"], verdict.evidence
