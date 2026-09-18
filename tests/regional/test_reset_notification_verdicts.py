"""Every reset case requires the batched RESET_GPU step's completion record.

DESTR-001, HA-003 and HA-004 passed live resets that produced no
GPU_RESET_COMPLETED record at all: the reset rides inside a QUIESCE-headed
carrier since the compound node command, and the control plane keyed the
notification on the head. The runners now read the incident's notification
records after the workflow turns terminal and require exactly one GPU reset
completion keyed by the RESET_GPU step's own idempotency key and none keyed by
the head. The delivery result follows the incident: a drill incident (every
kmsg-injected case) owes a record SKIPPED by the drill policy; a real incident
owes SENT with a provider message id and gets a bounded wait for the dispatcher.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import reset_notification_evidence as evidence
from scripts.e2e.regional import run_destr001_gpu_reset as reset_case
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
from tests.regional._cov95_destr_idle import IdleHarness
from tests.regional._cov95_ha_reset_harness import ResetHarness
from tests.regional._reset_notification_support import (
    drill_policy_reason,
    reset_notification_entry,
    skipped_drill_result,
)

CARRIER = {
    "command_id": "remote-carrier",
    "status": "SUCCEEDED",
    "step": {"operation": "QUIESCE_GPU_SERVICES"},
    "idempotency_key": "wf/2/QUIESCE_GPU_SERVICES",
    "batched_steps": [
        {"step_index": 3, "step": {"operation": "VERIFY_NO_GPU_CLIENTS"}},
        {
            "step_index": 4,
            "step": {"operation": "RESET_GPU"},
            "idempotency_key": "wf/4/RESET_GPU",
        },
        {
            "step_index": 5,
            "step": {"operation": "RESTORE_GPU_SERVICES"},
            "idempotency_key": "wf/5/RESTORE_GPU_SERVICES",
        },
    ],
}
MARK = {
    "command_id": "remote-mark",
    "status": "SUCCEEDED",
    "step": {"operation": "MARK_UNSCHEDULABLE"},
    "idempotency_key": "wf/1/MARK_UNSCHEDULABLE",
}
STANDALONE = {
    "command_id": "remote-reset",
    "status": "SUCCEEDED",
    "step": {"operation": "RESET_GPU"},
    "idempotency_key": "wf/4/RESET_GPU",
}
RESET_KEY = "wf/4/RESET_GPU"
HEAD_KEY = "wf/2/QUIESCE_GPU_SERVICES"
BUDGET = evidence.DELIVERY_WAIT_SECONDS // evidence.DELIVERY_POLL_SECONDS


def _entry(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "cluster_id": "cluster-a",
        "incident_id": "inc-a",
        "operation_id": RESET_KEY,
    }
    values.update(overrides)
    return reset_notification_entry(**values)


def _errors(
    notifications: list[dict[str, Any]], *, drill_id: str | None = None
) -> list[str]:
    return evidence.reset_notification_errors(
        notifications, operation_id=RESET_KEY, drill_id=drill_id
    )


# --- the RESET_GPU step's own idempotency key ------------------------------


def test_the_reset_key_is_read_off_the_passenger_or_the_head() -> None:
    assert evidence.reset_step_idempotency_key([MARK, CARRIER]) == RESET_KEY
    assert evidence.reset_step_idempotency_key([MARK, STANDALONE]) == RESET_KEY


@pytest.mark.parametrize(
    "commands",
    [
        [MARK],
        [CARRIER, STANDALONE],
        [{**CARRIER, "batched_steps": CARRIER["batched_steps"][:1]}],
        [
            {
                **CARRIER,
                "batched_steps": [
                    {"step_index": 4, "step": {"operation": "RESET_GPU"}}
                ],
            }
        ],
        [{**STANDALONE, "idempotency_key": ""}],
    ],
    ids=["no-reset", "two-carriers", "no-passenger", "unkeyed", "empty-key"],
)
def test_an_unidentifiable_reset_key_is_none(commands: list[dict[str, Any]]) -> None:
    assert evidence.reset_step_idempotency_key(commands) is None


# --- judging the incident's records -----------------------------------------


def test_expected_status_follows_the_incident() -> None:
    assert evidence.expected_status(None) == "SENT"
    assert evidence.expected_status("") == "SENT"
    assert evidence.expected_status("destr001-a1") == "SKIPPED"


def test_a_real_incidents_sent_record_keyed_by_the_step_passes() -> None:
    assert _errors([_entry()]) == []


def test_a_drill_incidents_record_skipped_by_the_policy_passes() -> None:
    assert _errors([_entry(drill_id="d1")], drill_id="d1") == []


def test_a_missing_key_fails_before_the_records_are_read() -> None:
    errors = evidence.reset_notification_errors([_entry()], operation_id=None)
    assert errors == [
        "RESET_GPU step idempotency key is not identifiable from the commands"
    ]


@pytest.mark.parametrize(
    "notifications,fragment",
    [
        ([], "found 0"),
        ([_entry(), _entry()], "found 2"),
        (
            [
                {
                    "notification": {
                        "deduplication_key": "cluster-a/inc-a/fault-detected/x",
                        "category": "FAULT_DETECTED",
                    },
                    "result": {"status": "SENT", "provider_message_id": "m"},
                }
            ],
            "found 0",
        ),
        (
            [_entry(operation_id=HEAD_KEY)],
            f"keyed by 'cluster-a/inc-a/gpu-reset/{HEAD_KEY}' instead of",
        ),
        (
            [_entry(operation_id=HEAD_KEY), _entry()],
            f"keyed by 'cluster-a/inc-a/gpu-reset/{HEAD_KEY}' instead of",
        ),
        ([_entry(drill_id="d1")], f"is SKIPPED: {drill_policy_reason('d1')}, not SENT"),
        ([_entry(status="QUEUED", provider_message_id=None)], "is QUEUED, not SENT"),
        ([{"notification": _entry()["notification"], "result": None}], "unsent"),
        ([_entry(provider_message_id="")], "has no provider message id"),
    ],
    ids=[
        "missing",
        "duplicate",
        "other-kind-only",
        "head-key-only",
        "head-key-beside-the-step",
        "skipped-drill",
        "queued",
        "no-result",
        "no-provider-id",
    ],
)
def test_a_real_incident_needs_one_sent_step_keyed_record(
    notifications: list[dict[str, Any]], fragment: str
) -> None:
    errors = _errors(notifications)
    assert errors, "the defect must be named"
    assert any(fragment in item for item in errors), (fragment, errors)


@pytest.mark.parametrize(
    "notifications,fragment",
    [
        ([], "found 0"),
        ([_entry(drill_id="d1"), _entry(drill_id="d1")], "found 2"),
        (
            [_entry(drill_id="d1", operation_id=HEAD_KEY), _entry(drill_id="d1")],
            f"keyed by 'cluster-a/inc-a/gpu-reset/{HEAD_KEY}' instead of",
        ),
        (
            [_entry()],
            "is SENT, not SKIPPED by the drill policy "
            "(GPU_FAULT_NOTIFICATION_DELIVER_DRILLS)",
        ),
        (
            [_entry(drill_id="d1", reason="operator muted the incident")],
            "is SKIPPED: operator muted the incident, not SKIPPED by the drill policy",
        ),
        (
            [_entry(drill_id="d1", status="QUEUED", reason=None)],
            "is QUEUED, not SKIPPED by the drill policy",
        ),
        (
            [{"notification": _entry(drill_id="d1")["notification"], "result": None}],
            "is unsent, not SKIPPED by the drill policy",
        ),
        ([_entry(drill_id="d2")], "is not labelled with the incident's drill d1"),
    ],
    ids=[
        "missing",
        "duplicate",
        "head-key-beside-the-step",
        "sent",
        "other-skip-reason",
        "queued",
        "no-result",
        "foreign-drill",
    ],
)
def test_a_drill_incident_needs_one_record_skipped_by_the_policy(
    notifications: list[dict[str, Any]], fragment: str
) -> None:
    errors = _errors(notifications, drill_id="d1")
    assert errors, "the defect must be named"
    assert any(fragment in item for item in errors), (fragment, errors)


def test_a_foreign_drill_label_is_the_only_error_when_the_policy_applied() -> None:
    errors = _errors([_entry(drill_id="d2")], drill_id="d1")
    assert errors == [
        "GPU reset completion notification is not labelled with the incident's drill d1"
    ]


def test_the_category_is_checked_too() -> None:
    entry = _entry()
    entry["notification"]["category"] = "ADVISORY"
    assert _errors([entry]) == [
        "GPU reset completion notification is not ACTION_COMPLETED"
    ]


# --- the bounded delivery wait ---------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.parametrize(
    "notifications,pending",
    [
        ([], True),
        ([_entry(status="QUEUED", provider_message_id=None)], True),
        ([_entry(status="FAILED", provider_message_id=None)], True),
        ([{"notification": _entry()["notification"], "result": None}], True),
        ([_entry(drill_id="d1")], False),
        ([_entry(), _entry()], False),
        ([_entry(operation_id=HEAD_KEY)], False),
        ([_entry(operation_id=HEAD_KEY, status="QUEUED")], False),
        ([_entry()], False),
    ],
    ids=[
        "not-yet-created",
        "queued",
        "failed-attempt",
        "no-result",
        "skipped",
        "duplicate",
        "foreign-key",
        "foreign-key-queued",
        "sent",
    ],
)
def test_only_an_unsent_or_absent_record_keeps_the_wait_alive(
    notifications: list[dict[str, Any]], pending: bool
) -> None:
    assert (
        evidence.delivery_pending(notifications, operation_id=RESET_KEY) is pending
    ), notifications


def _wait(
    state: dict[str, Any],
    snapshot: Any,
    clock: Clock,
    *,
    drill_id: str | None = None,
    timeout_seconds: float = evidence.DELIVERY_WAIT_SECONDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    return evidence.wait_for_reset_notification(
        state,
        operation_id=RESET_KEY,
        drill_id=drill_id,
        snapshot=snapshot,
        timeout_seconds=timeout_seconds,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def test_a_record_that_already_landed_costs_no_extra_store_read() -> None:
    clock = Clock()
    reads: list[int] = []
    notifications, errors = _wait(
        {"notifications": [_entry()]}, lambda: reads.append(1) or {}, clock
    )
    assert errors == [] and len(notifications) == 1
    assert reads == [] and clock.sleeps == []


def test_a_record_still_in_the_outbox_is_re_read_until_it_is_sent() -> None:
    clock = Clock()
    queued = _entry(status="QUEUED", provider_message_id=None)
    snapshots = iter(
        [
            {"notifications": [queued]},
            {"notifications": [queued]},
            {"notifications": [_entry()]},
        ]
    )
    notifications, errors = _wait({"notifications": []}, lambda: next(snapshots), clock)
    assert errors == []
    assert notifications[0]["result"]["status"] == "SENT"
    assert clock.sleeps == [evidence.DELIVERY_POLL_SECONDS] * 3


def test_the_wait_is_bounded_and_ends_in_the_last_observed_defect() -> None:
    clock = Clock()
    reads: list[int] = []
    notifications, errors = _wait(
        {"notifications": []}, lambda: reads.append(1) or {"notifications": []}, clock
    )
    assert errors == [
        f"expected exactly one GPU reset completion notification for {RESET_KEY}, "
        "found 0"
    ]
    assert notifications == []
    assert len(reads) == BUDGET
    assert clock.now >= evidence.DELIVERY_WAIT_SECONDS


def test_a_final_defect_ends_the_wait_without_spending_the_budget() -> None:
    clock = Clock()
    reads: list[int] = []
    skipped = _entry(drill_id="d1")
    _, errors = _wait(
        {"notifications": [skipped]},
        lambda: reads.append(1) or {"notifications": [skipped]},
        clock,
    )
    assert any("not SENT" in item for item in errors), errors
    assert reads == [] and clock.sleeps == [], "a SKIPPED result is final"


def test_a_drill_incident_is_judged_once_even_with_a_budget() -> None:
    """The drill policy skips the record synchronously when the carrier
    completes; there is nothing to wait for, a missing record is final."""

    clock = Clock()
    reads: list[int] = []
    _, errors = _wait(
        {"notifications": []},
        lambda: reads.append(1) or {"notifications": [_entry(drill_id="d1")]},
        clock,
        drill_id="d1",
    )
    assert any("found 0" in item for item in errors), errors
    assert reads == [] and clock.sleeps == []


def _regional(reads: list[dict[str, Any]], notifications: list[dict[str, Any]]) -> Any:
    def store_snapshot(**kwargs: Any) -> dict[str, Any]:
        reads.append(kwargs)
        return {"notifications": notifications}

    return type("Regional", (), {"store_snapshot": staticmethod(store_snapshot)})()


def test_a_run_that_already_failed_is_judged_once_without_the_budget() -> None:
    clock = Clock()
    reads: list[dict[str, Any]] = []
    document, errors = evidence.reset_notification_evidence(
        _regional(reads, []),
        {"commands": [MARK, CARRIER], "notifications": [], "incident": {}},
        node="node-a",
        marker="marker-a",
        observed_after="t0",
        wait=False,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert errors and reads == [] and clock.sleeps == []
    assert document["expected_operation_id"] == RESET_KEY
    assert document["drill"] is False and document["drill_id"] is None
    assert document["expected_status"] == "SENT"
    assert document["delivery_wait_seconds"] == 0
    assert document["reset_notifications"] == []
    assert document["errors"] == errors


def test_a_real_incident_reads_the_store_by_marker_and_samples_the_queue_once() -> None:
    clock = Clock()
    reads: list[dict[str, Any]] = []
    document, errors = evidence.reset_notification_evidence(
        _regional(reads, [_entry()]),
        {"commands": [MARK, CARRIER], "notifications": [], "incident": {}},
        node="node-a",
        marker="marker-a",
        observed_after="t0",
        wait=True,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert errors == []
    assert reads == [
        {
            "node": "node-a",
            "marker": "marker-a",
            "observed_after": "t0",
            "queue_attempts": 1,
        }
    ]
    assert document["delivery_wait_seconds"] == evidence.DELIVERY_WAIT_SECONDS
    assert document["expected_status"] == "SENT"
    assert document["reset_notifications"][0]["result"]["status"] == "SENT"
    assert document["notification_count"] == 1


def test_a_drill_incident_takes_the_skipped_branch_and_never_waits() -> None:
    clock = Clock()
    reads: list[dict[str, Any]] = []
    state = {
        "commands": [MARK, CARRIER],
        "notifications": [_entry(drill_id="destr001-a1")],
        "incident": {"incident_id": "inc-a", "drill_id": "destr001-a1"},
    }
    document, errors = evidence.reset_notification_evidence(
        _regional(reads, []),
        state,
        node="node-a",
        marker="marker-a",
        observed_after="t0",
        wait=True,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert errors == []
    assert document["drill"] is True and document["drill_id"] == "destr001-a1"
    assert document["expected_status"] == "SKIPPED"
    assert document["delivery_wait_seconds"] == 0
    assert reads == [] and clock.sleeps == []
    assert document["reset_notifications"][0]["result"]["status"] == "SKIPPED"


# --- the runners -------------------------------------------------------------


@pytest.fixture(params=[ha003, ha004], ids=["aurora", "executor"])
def harness(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> ResetHarness:
    return ResetHarness(monkeypatch, tmp_path, request.param)


def _ha_states(harness: ResetHarness) -> list[dict[str, Any]]:
    return [harness.api.final, *harness.api.states]


def _make_drill(harness: ResetHarness, drill_id: str) -> None:
    for state in _ha_states(harness):
        state["incident"]["drill_id"] = drill_id
        skipped_drill_result(state["notifications"][0], drill_id)


def test_ha_reset_cases_record_a_real_incidents_sent_completion(
    harness: ResetHarness,
) -> None:
    code, report = harness.execute()
    assert code == 0, report.get("error", report.get("errors"))
    mail = report["reset_notification"]
    assert mail["errors"] == []
    assert mail["drill"] is False and mail["expected_status"] == "SENT"
    assert mail["expected_operation_id"] == "workflow-unit/4"
    [entry] = mail["reset_notifications"]
    assert entry["result"]["status"] == "SENT"
    assert entry["notification"]["deduplication_key"].endswith(
        "/gpu-reset/workflow-unit/4"
    ), "the recorded mail is the one keyed by the RESET_GPU command's own key"
    written = harness.directory / "reset-notification.json"
    assert written.is_file(), "the mail evidence is written next to the workflow state"


def test_ha_reset_cases_accept_a_drill_incidents_record_skipped_by_the_policy(
    harness: ResetHarness,
) -> None:
    _make_drill(harness, "ha-unit")
    harness.events.clear()
    code, report = harness.execute()
    assert code == 0, report.get("error", report.get("errors"))
    mail = report["reset_notification"]
    assert mail["errors"] == []
    assert mail["drill"] is True and mail["drill_id"] == "ha-unit"
    assert mail["expected_status"] == "SKIPPED"
    assert mail["delivery_wait_seconds"] == 0, "a drill record is never waited for"
    [entry] = mail["reset_notifications"]
    assert entry["result"]["status"] == "SKIPPED"
    assert "GPU_FAULT_NOTIFICATION_DELIVER_DRILLS" in entry["result"]["reason"]


@pytest.mark.parametrize(
    "defect", ["missing", "skipped", "head-key", "duplicate", "no-provider-id"]
)
def test_ha_reset_cases_fail_a_real_incident_without_one_sent_step_keyed_record(
    harness: ResetHarness, defect: str
) -> None:
    for state in _ha_states(harness):
        entry = state["notifications"][0]
        if defect == "missing":
            state["notifications"] = []
        elif defect == "skipped":
            skipped_drill_result(entry, "ha-unit")
        elif defect == "head-key":
            entry["notification"]["deduplication_key"] = (
                "unit-cluster/incident-unit/gpu-reset/workflow-unit/2"
            )
        elif defect == "duplicate":
            state["notifications"].append(copy.deepcopy(entry))
        else:
            entry["result"]["provider_message_id"] = None
    code, report = harness.execute()
    assert code == 1
    assert report["verdict"] == "FAIL"
    assert any(
        "GPU reset completion notification" in item for item in report["errors"]
    ), report["errors"]
    if defect == "skipped":
        assert any("not SENT" in item for item in report["errors"]), report["errors"]
    if defect == "head-key":
        assert any("instead of" in item for item in report["errors"]), report["errors"]
    assert report["cleanup"]["errors"] == [], "cleanup still runs after a failed mail"
    assert "restore-quiesce" in harness.events


@pytest.mark.parametrize("defect", ["missing", "sent", "other-reason", "head-key"])
def test_ha_reset_cases_fail_a_drill_incident_not_skipped_by_the_policy(
    harness: ResetHarness, defect: str
) -> None:
    _make_drill(harness, "ha-unit")
    for state in _ha_states(harness):
        entry = state["notifications"][0]
        if defect == "missing":
            state["notifications"] = []
        elif defect == "sent":
            entry["result"].update(status="SENT", provider_message_id="m", reason=None)
        elif defect == "other-reason":
            entry["result"]["reason"] = "operator muted the incident"
        else:
            state["notifications"].append(
                {
                    **copy.deepcopy(entry),
                    "notification": {
                        **copy.deepcopy(entry["notification"]),
                        "notification_id": "notification-head",
                        "deduplication_key": (
                            "unit-cluster/incident-unit/gpu-reset/workflow-unit/2"
                        ),
                    },
                }
            )
    harness.events.clear()
    code, report = harness.execute()
    assert code == 1
    assert report["verdict"] == "FAIL"
    assert any(
        "GPU reset completion notification" in item for item in report["errors"]
    ), report["errors"]
    if defect in {"sent", "other-reason"}:
        assert any(
            "not SKIPPED by the drill policy" in item for item in report["errors"]
        ), report["errors"]
    assert report["reset_notification"]["delivery_wait_seconds"] == 0
    assert report["cleanup"]["errors"] == [], "cleanup still runs after a failed mail"


def test_ha_reset_cases_spend_the_delivery_budget_only_on_a_real_passing_run(
    harness: ResetHarness,
) -> None:
    for state in _ha_states(harness):
        state["notifications"] = []
    harness.events.clear()
    code, report = harness.execute()
    assert code == 1
    observed = harness.events.count("store-observe")
    assert observed >= BUDGET, (observed, BUDGET, harness.events)
    assert report["reset_notification"]["delivery_wait_seconds"] == (
        evidence.DELIVERY_WAIT_SECONDS
    )

    # A run that already failed its host contract is judged once, no budget.
    (harness.path / "second").mkdir()
    second = ResetHarness(pytest.MonkeyPatch(), harness.path / "second", harness.module)
    for state in _ha_states(second):
        state["notifications"] = []
    second.host.after["ledger"] = []
    second.events.clear()
    _, report = second.execute()
    assert second.events.count("store-observe") < BUDGET
    assert report["reset_notification"]["delivery_wait_seconds"] == 0


def _idle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> IdleHarness:
    h = IdleHarness(reset_case, tmp_path, monkeypatch)
    h.plan(tmp_path)
    return h


def _polls(h: IdleHarness) -> int:
    # One store read is the execute-phase preflight; the rest are the wait.
    return [name for name, _ in h.calls].count("store.snapshot") - 1


def test_destr001_records_a_real_incidents_sent_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _idle(tmp_path, monkeypatch)
    code, report = h.execute(tmp_path)
    assert code == 0 and report["verdict"] == "PASS", report
    mail = report["reset_notification"]
    assert mail["errors"] == []
    assert mail["drill"] is False and mail["expected_status"] == "SENT"
    assert mail["expected_operation_id"] == "wf-destr023/4/RESET_GPU"
    assert mail["reset_notifications"][0]["result"]["status"] == "SENT"
    assert (
        tmp_path / "cases" / reset_case.CASE_ID / "reset-notification.json"
    ).is_file(), "the mail evidence is written next to the workflow state"


def test_destr001_accepts_a_drill_incidents_record_skipped_by_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _idle(tmp_path, monkeypatch)
    h.workflow["incident"]["drill_id"] = "destr001-unit"
    skipped_drill_result(h.workflow["notifications"][0], "destr001-unit")
    h.calls.clear()
    code, report = h.execute(tmp_path)
    assert code == 0 and report["verdict"] == "PASS", report
    mail = report["reset_notification"]
    assert mail["drill"] is True and mail["expected_status"] == "SKIPPED"
    assert mail["delivery_wait_seconds"] == 0
    assert _polls(h) == 0, "a drill record is judged once from the terminal state"


@pytest.mark.parametrize("defect", ["missing", "skipped"])
def test_destr001_fails_a_real_incident_without_a_sent_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    h = _idle(tmp_path, monkeypatch)
    if defect == "missing":
        h.workflow["notifications"] = []
    else:
        skipped_drill_result(h.workflow["notifications"][0], "destr001-unit")
    h.calls.clear()
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    assert any(
        "GPU reset completion notification" in item for item in report["errors"]
    ), report["errors"]
    if defect == "skipped":
        assert any("not SENT" in item for item in report["errors"]), report["errors"]
        assert _polls(h) == 0, "a SKIPPED result is final; no budget is spent on it"
    else:
        # The workflow passed every other check, so the delivery budget was
        # spent re-reading the store before the record was declared missing.
        assert _polls(h) >= BUDGET, (_polls(h), BUDGET)
    names = [name for name, _ in h.calls]
    assert names.index("workflow.wait") < names.index("host.cleanup"), names
    assert "host.restore-quiesce" not in names, names


@pytest.mark.parametrize("defect", ["missing", "sent", "other-reason"])
def test_destr001_fails_a_drill_incident_not_skipped_by_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    h = _idle(tmp_path, monkeypatch)
    h.workflow["incident"]["drill_id"] = "destr001-unit"
    entry = skipped_drill_result(h.workflow["notifications"][0], "destr001-unit")
    if defect == "missing":
        h.workflow["notifications"] = []
    elif defect == "sent":
        entry["result"].update(status="SENT", provider_message_id="m", reason=None)
    else:
        entry["result"]["reason"] = "operator muted the incident"
    h.calls.clear()
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    assert any(
        "GPU reset completion notification" in item for item in report["errors"]
    ), report["errors"]
    if defect != "missing":
        assert any(
            "not SKIPPED by the drill policy" in item for item in report["errors"]
        ), report["errors"]
    assert _polls(h) == 0, "a drill incident never spends the delivery budget"
    assert report["reset_notification"]["expected_status"] == "SKIPPED"
