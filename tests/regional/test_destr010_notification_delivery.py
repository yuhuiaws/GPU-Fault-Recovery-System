"""DESTR-010 judges each of its notifications by the delivery result it ends with.

Both XID 45 lines the case writes carry a drill id -- ``<run_id>`` for the chain,
``<run_id>-observe`` for the observe-only phase -- so a site that keeps drills off
the mail (``GPU_FAULT_NOTIFICATION_DELIVER_DRILLS`` unset, the shipped default)
records the case's notifications SKIPPED by the drill policy, and a site that mails
drills records them SENT. The runner accepts exactly those two outcomes: SENT with a
provider message id, or SKIPPED on a notification labelled with the run's own drill
whose reason names the drill policy. A SKIPPED for any other reason (an expired
advisory, a disabled channel), a SKIPPED on a notification without that drill, a
SENT without its provider message id and every other status fail and are named; a
result still in flight is re-read within the shared delivery budget and fails as
"not SENT" at the deadline.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts.e2e.regional import destr010_observe_only as observe
from scripts.e2e.regional import fabric_restart_shape as shapes
from scripts.e2e.regional import reset_notification_evidence as evidence
from scripts.e2e.regional import run_destr010_fabric_manager_restart as destr010
from tests.regional._cov95_destr_warm import Clock
from tests.regional._reset_notification_support import drill_policy_reason
from tests.regional.test_destr010_fabric_reset_shape import (
    BASELINE,
    GPUS,
    _fabric_state,
    _observe_state,
    _ObserveRegional,
    _preflight,
    _run_observe_case,
)
from tests.regional.test_destr010_review_fixes import (
    _Calls,
    _destr010_fakes,
    _store_state,
)

# ``execute_case`` derives the run id from the run directory ``run-DESTR010``.
RUN_ID = "destr010-destr010-a1"
OBSERVE_DRILL = f"{RUN_ID}-observe"
POLICY_SKIP = {"status": "SKIPPED", "reason": drill_policy_reason(RUN_ID)}
SENT = {"status": "SENT", "provider_message_id": "provider-message-1"}
BUDGET_READS = evidence.DELIVERY_WAIT_SECONDS // evidence.DELIVERY_POLL_SECONDS
OTHER_SKIP_REASONS = [
    "expired: 7300s old, past its 3600s ACTION_COMPLETED shelf life "
    "(deadline 2026-01-01T01:00:00+00:00)",
    "SNS notification delivery is disabled",
    "suppressed as pre-watermark backlog by release-a",
]


def _fm_restart_state() -> dict[str, Any]:
    """The terminal read of the Fabric Manager restart, with its kernel evidence."""

    return {**_store_state(), "evidence": [{"record_id": "evidence-a"}]}


# (shape, terminal store read of that shape's chain, the GPUs it resets)
CHAIN_SHAPES = [
    pytest.param(shapes.FABRIC_MANAGER_RESTART, _fm_restart_state, (), id="fm-restart"),
    pytest.param(shapes.FABRIC_RESET, _fabric_state, tuple(GPUS), id="fabric-reset"),
]


def _set_delivery(
    state: dict[str, Any], result: dict[str, Any] | None, *, drill_id: str | None
) -> dict[str, Any]:
    """Label the state's one notification with ``drill_id`` and give it ``result``."""

    entry = state["notifications"][0]
    entry["notification"]["drill_id"] = drill_id
    entry["result"] = (
        None
        if result is None
        else {
            "notification_id": entry["notification"]["notification_id"],
            "provider_message_id": None,
            "reason": None,
            **result,
        }
    )
    return state


def _chain_errors(
    build: Callable[[], dict[str, Any]],
    shape: str,
    gpus: tuple[str, ...],
    result: dict[str, Any] | None,
    *,
    labelled: str | None = RUN_ID,
) -> list[str]:
    state = _set_delivery(build(), result, drill_id=labelled)
    return destr010.workflow_errors(state, shape, gpus, drill_id=RUN_ID)


# --- the chain's verdict, for both node-action shapes ---------------------------


@pytest.mark.parametrize(("shape", "build", "gpus"), CHAIN_SHAPES)
def test_a_skip_by_the_drill_policy_on_the_run_s_own_drill_passes(
    shape: str, build: Callable[[], dict[str, Any]], gpus: tuple[str, ...]
) -> None:
    errors = _chain_errors(build, shape, gpus, POLICY_SKIP)

    assert errors == [], "the shipped policy keeps the case's drill mail back"


@pytest.mark.parametrize(("shape", "build", "gpus"), CHAIN_SHAPES)
def test_a_sent_notification_with_its_provider_message_id_passes(
    shape: str, build: Callable[[], dict[str, Any]], gpus: tuple[str, ...]
) -> None:
    errors = _chain_errors(build, shape, gpus, SENT)

    assert errors == [], "a site that mails drills delivers the notification"


@pytest.mark.parametrize(
    "reason", OTHER_SKIP_REASONS, ids=["expired", "off", "backlog"]
)
@pytest.mark.parametrize(("shape", "build", "gpus"), CHAIN_SHAPES)
def test_a_skip_for_any_other_reason_fails_and_names_it(
    shape: str, build: Callable[[], dict[str, Any]], gpus: tuple[str, ...], reason: str
) -> None:
    errors = _chain_errors(build, shape, gpus, {"status": "SKIPPED", "reason": reason})

    assert any(
        f"notification is SKIPPED: {reason}, not SKIPPED by the drill policy "
        f"({evidence.DRILL_POLICY_MARKER})" in item
        for item in errors
    ), errors


@pytest.mark.parametrize(
    "labelled",
    [None, "destr010-other-a1", OBSERVE_DRILL],
    ids=["no-drill", "another-run", "the-observe-phase"],
)
@pytest.mark.parametrize(("shape", "build", "gpus"), CHAIN_SHAPES)
def test_a_policy_skip_on_a_notification_without_the_run_s_drill_fails(
    shape: str,
    build: Callable[[], dict[str, Any]],
    gpus: tuple[str, ...],
    labelled: str | None,
) -> None:
    errors = _chain_errors(build, shape, gpus, POLICY_SKIP, labelled=labelled)

    carried = f"drill {labelled}" if labelled else "no drill id"
    assert any(
        f"notification is SKIPPED but carries {carried}, not this run's drill "
        f"{RUN_ID}" in item
        for item in errors
    ), errors


@pytest.mark.parametrize(("shape", "build", "gpus"), CHAIN_SHAPES)
def test_a_sent_notification_without_its_provider_message_id_fails(
    shape: str, build: Callable[[], dict[str, Any]], gpus: tuple[str, ...]
) -> None:
    errors = _chain_errors(build, shape, gpus, {"status": "SENT"})

    assert any(
        "notification is SENT without a provider message id" in item for item in errors
    ), errors


@pytest.mark.parametrize(
    ("result", "described"),
    [
        (None, "unsent"),
        (
            {"status": "QUEUED", "reason": "queued for asynchronous delivery"},
            "QUEUED: queued for asynchronous delivery",
        ),
        (
            {"status": "FAILED", "reason": "dead-lettered after 8 attempt(s)"},
            "FAILED: dead-lettered after 8 attempt(s)",
        ),
        ({"status": "DUPLICATE"}, "DUPLICATE"),
        ({"status": "DEAD"}, "DEAD"),
    ],
    ids=["no-result", "queued", "failed", "duplicate", "dead"],
)
@pytest.mark.parametrize(("shape", "build", "gpus"), CHAIN_SHAPES)
def test_every_other_result_is_named_and_is_no_pass(
    shape: str,
    build: Callable[[], dict[str, Any]],
    gpus: tuple[str, ...],
    result: dict[str, Any] | None,
    described: str,
) -> None:
    errors = _chain_errors(build, shape, gpus, result)

    assert any(f"notification is {described}, not SENT" in item for item in errors), (
        errors
    )


def test_the_recorded_notification_keeps_what_the_verdict_reads() -> None:
    state = _set_delivery(_fm_restart_state(), POLICY_SKIP, drill_id=RUN_ID)

    (record,) = destr010.normalize_notifications(state["notifications"])

    assert record["drill_id"] == RUN_ID, record
    assert record["status"] == "SKIPPED", record
    assert record["reason"] == drill_policy_reason(RUN_ID), record
    assert record["provider_message_id"] is None, record
    assert destr010.normalize_notifications([record]) == [record], "idempotent"


# --- the chain waits for a result still in flight (execute_case) ----------------


def _execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    terminal: dict[str, Any] | None,
    reads: list[dict[str, Any] | None],
) -> tuple[int, dict[str, Any], _Calls, Clock]:
    """Drive the Fabric Manager restart with scripted notification results.

    ``terminal`` is the result the workflow wait returns; every later store
    read takes the next of ``reads`` (the last one repeats), labelled with the
    run's drill as the product labels a drill incident's notification.
    """

    calls = _Calls()
    settings, run_dir = _destr010_fakes(tmp_path, monkeypatch, calls=calls)
    shared = destr010.RegionalLiveFixture
    remaining = list(reads)

    class Delivering(shared):
        def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
            state: dict[str, Any] = super().wait_for_workflow(**kwargs)
            return _set_delivery(state, terminal, drill_id=RUN_ID)

        def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
            state: dict[str, Any] = super().store_snapshot(**kwargs)
            result = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            return _set_delivery(state, result, drill_id=RUN_ID)

    clock = Clock()
    monkeypatch.setattr(destr010, "RegionalLiveFixture", Delivering)
    monkeypatch.setattr(destr010, "time", clock)
    code = destr010.execute_case(
        settings, run_dir, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    report = json.loads(
        (run_dir / "cases" / destr010.CASE_ID / f"{destr010.CASE_ID}.json").read_text(
            encoding="utf-8"
        )
    )
    return code, report, calls, clock


def _wait_reads(calls: _Calls) -> int:
    # The post-replay read happens on every run; every other read is the wait's.
    return len(calls.named("regional:store_snapshot")) - 1


def test_the_chain_re_reads_a_result_in_flight_until_it_is_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, report, calls, clock = _execute(
        tmp_path,
        monkeypatch,
        terminal=None,
        reads=[None, {"status": "QUEUED", "reason": "queued"}, SENT],
    )

    assert code == 0 and report["verdict"] == "PASS", report
    assert _wait_reads(calls) == 3, calls.names()
    assert clock.elapsed == 3 * evidence.DELIVERY_POLL_SECONDS, clock.elapsed
    assert report["drill_id"] == RUN_ID, report
    assert report["notifications"][0]["provider_message_id"] == "provider-message-1"


def test_the_chain_s_wait_is_bounded_and_ends_not_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, report, calls, clock = _execute(
        tmp_path, monkeypatch, terminal=None, reads=[None]
    )

    assert code == 1 and report["verdict"] == "FAIL", report
    assert (
        "Fabric Manager action notification is unsent, not SENT" in report["errors"]
    ), report["errors"]
    assert _wait_reads(calls) == BUDGET_READS, calls.names()
    assert clock.elapsed >= evidence.DELIVERY_WAIT_SECONDS, clock.elapsed
    assert "regional:executor_python" not in calls.names(), "a FAIL is not replayed"


def test_a_policy_skip_lands_with_the_terminal_read_and_costs_no_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, report, calls, clock = _execute(
        tmp_path, monkeypatch, terminal=POLICY_SKIP, reads=[POLICY_SKIP]
    )

    assert code == 0 and report["verdict"] == "PASS", report
    assert _wait_reads(calls) == 0 and clock.elapsed == 0, calls.names()
    assert report["notifications"][0]["reason"] == drill_policy_reason(RUN_ID)


def test_a_result_that_is_already_final_is_judged_without_a_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expired = {"status": "SKIPPED", "reason": OTHER_SKIP_REASONS[0]}

    code, report, calls, clock = _execute(
        tmp_path, monkeypatch, terminal=expired, reads=[POLICY_SKIP]
    )

    assert code == 1 and report["verdict"] == "FAIL", report
    assert any(OTHER_SKIP_REASONS[0] in item for item in report["errors"]), report
    assert _wait_reads(calls) == 0 and clock.elapsed == 0, calls.names()


# --- the observe-only phase's investigatory notification (run_phase) ------------


class _PhaseRegional:
    """The control plane of the observe-only XID: one store read per call."""

    def __init__(self, reads: list[dict[str, Any] | None], labelled: str | None):
        self.reads = list(reads)
        self.labelled = labelled
        self.store_reads = 0

    def store_snapshot(self, **_kwargs: Any) -> dict[str, Any]:
        self.store_reads += 1
        result = self.reads.pop(0) if len(self.reads) > 1 else self.reads[0]
        return _set_delivery(_observe_state(), result, drill_id=self.labelled)

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return dict(_preflight()["node"])


class _PhaseHost:
    def __init__(self) -> None:
        self.injections: list[tuple[str, ...]] = []

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        if arguments[0] == "write-xid45":
            self.injections.append(arguments)
            return {"command": arguments[0]}
        return copy.deepcopy(BASELINE)


def _phase(
    tmp_path: Path,
    reads: list[dict[str, Any] | None],
    *,
    labelled: str | None = OBSERVE_DRILL,
) -> tuple[dict[str, Any], _PhaseRegional, _PhaseHost, Clock]:
    regional = _PhaseRegional(reads, labelled)
    host = _PhaseHost()
    clock = Clock()
    report = observe.run_phase(
        regional,
        host,
        tmp_path,
        node="node-a",
        run_id=RUN_ID,
        attempt=1,
        baseline_host=copy.deepcopy(BASELINE),
        baseline_node=dict(_preflight()["node"]),
        graded_gpus=GPUS,
        companion_window_seconds=0,
        node_state_errors=destr010.node_state_errors,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    return report, regional, host, clock


def test_phase_one_injects_and_judges_the_same_drill(tmp_path: Path) -> None:
    observe_skip = {"status": "SKIPPED", "reason": drill_policy_reason(OBSERVE_DRILL)}

    report, regional, host, _clock = _phase(tmp_path, [observe_skip])

    assert report["errors"] == [], report
    assert report["drill_id"] == OBSERVE_DRILL, report
    (injection,) = host.injections
    assert injection[injection.index("--drill-id") + 1] == OBSERVE_DRILL, injection
    assert regional.store_reads == 2, "the finalize read and the verdict read"


@pytest.mark.parametrize(
    ("result", "labelled", "message"),
    [
        (
            {"status": "SKIPPED", "reason": OTHER_SKIP_REASONS[0]},
            OBSERVE_DRILL,
            f"is SKIPPED: {OTHER_SKIP_REASONS[0]}, not SKIPPED by the drill policy",
        ),
        (
            POLICY_SKIP,
            RUN_ID,
            f"is SKIPPED but carries drill {RUN_ID}, not this run's drill "
            f"{OBSERVE_DRILL}",
        ),
        (
            {"status": "SKIPPED", "reason": drill_policy_reason(OBSERVE_DRILL)},
            None,
            f"is SKIPPED but carries no drill id, not this run's drill {OBSERVE_DRILL}",
        ),
        ({"status": "SENT"}, OBSERVE_DRILL, "is SENT without a provider message id"),
    ],
    ids=["expired", "the-chain-s-drill", "no-drill", "sent-without-id"],
)
def test_phase_one_fails_an_investigatory_notice_neither_sent_nor_its_drill_s(
    tmp_path: Path, result: dict[str, Any], labelled: str | None, message: str
) -> None:
    report, _regional, _host, _clock = _phase(tmp_path, [result], labelled=labelled)

    assert any(
        item.startswith("observe-only: XID investigatory notification")
        and message in item
        for item in report["errors"]
    ), report["errors"]


def test_phase_one_re_reads_a_result_in_flight_until_it_is_sent(tmp_path: Path) -> None:
    report, regional, _host, clock = _phase(
        tmp_path, [None, None, {"status": "QUEUED"}, SENT]
    )

    assert report["errors"] == [], report
    assert regional.store_reads == 4, "finalize, verdict and two re-reads"
    assert clock.elapsed == (
        observe.SETTLE_SECONDS + 2 * evidence.DELIVERY_POLL_SECONDS
    ), clock.elapsed


def test_phase_one_s_wait_is_bounded_and_ends_not_sent(tmp_path: Path) -> None:
    report, regional, _host, clock = _phase(tmp_path, [None])

    assert (
        "observe-only: XID investigatory notification is unsent, not SENT"
        in report["errors"]
    ), report["errors"]
    assert regional.store_reads == 2 + BUDGET_READS, regional.store_reads
    assert clock.elapsed >= observe.SETTLE_SECONDS + evidence.DELIVERY_WAIT_SECONDS


# --- end to end: observe-only first, then the opt-in chain ---------------------


def _wait_returns(
    monkeypatch: pytest.MonkeyPatch, result: dict[str, Any], *, labelled: str
) -> None:
    """The chain's terminal read, its notification labelled ``labelled``."""

    chain = _set_delivery(_fabric_state(), result, drill_id=labelled)

    def wait_for_workflow(self: _ObserveRegional, **_kwargs: Any) -> dict[str, Any]:
        self.events.append("chain.wait")
        return copy.deepcopy(chain)

    monkeypatch.setattr(_ObserveRegional, "wait_for_workflow", wait_for_workflow)


def test_the_graded_case_passes_with_both_notifications_kept_back_as_its_drills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wait_returns(monkeypatch, POLICY_SKIP, labelled=RUN_ID)

    code, report, events = _run_observe_case(tmp_path, monkeypatch)

    assert code == 0 and report["verdict"] == "PASS", report
    assert report["observe_only"]["drill_id"] == OBSERVE_DRILL, report
    assert report["drill_id"] == RUN_ID, report
    assert events.index("window.open") < events.index("chain.wait"), events


def test_a_chain_notice_kept_back_for_the_observe_phase_s_drill_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wait_returns(
        monkeypatch,
        {"status": "SKIPPED", "reason": drill_policy_reason(OBSERVE_DRILL)},
        labelled=OBSERVE_DRILL,
    )

    code, report, _events = _run_observe_case(tmp_path, monkeypatch)

    assert code == 1 and report["verdict"] == "FAIL", report
    assert (
        f"GPU/NVSwitch reset notification is SKIPPED but carries drill "
        f"{OBSERVE_DRILL}, not this run's drill {RUN_ID}" in report["errors"]
    ), report["errors"]


def test_a_phase_one_notice_skipped_for_another_reason_opens_no_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expired = _set_delivery(
        _observe_state(),
        {"status": "SKIPPED", "reason": OTHER_SKIP_REASONS[0]},
        drill_id=OBSERVE_DRILL,
    )

    code, report, events = _run_observe_case(
        tmp_path, monkeypatch, observe_state=expired
    )

    assert code == 1 and report["verdict"] == "FAIL", report
    assert any(
        item.startswith("observe-only: XID investigatory notification is SKIPPED")
        for item in report["errors"]
    ), report["errors"]
    assert "window.open" not in events and "chain.wait" not in events, events
