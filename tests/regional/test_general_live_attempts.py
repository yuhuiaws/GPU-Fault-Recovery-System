"""GF-GENERAL-LIVE-028/029 injector/verdict contract over scripted records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.store import NotFoundError
from scripts.e2e.regional import general_live_attempts as att
from scripts.e2e.regional import general_live_terminal_replay as replay
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_restart_budget import BUDGET_PROBE
from tests.regional._general_live_fakes import (
    FakeRegional,
    budget_ledger_store,
    budget_probe_run,
    make_context,
    make_settings,
    store_script,
    workflow,
)

SPECS = {spec.variant: spec for spec in att.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


class FakeJob:
    created: list["FakeJob"] = []

    def __init__(
        self,
        ctx: Any,
        *,
        name: str,
        mode: Any,
        restart_budget: int = 1,
        gpu_count: int = 8,
        ranks: int = 1,
    ) -> None:
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.mode = mode
        self.restart_budget = restart_budget
        self.gpu_count = gpu_count
        self.ranks = ranks
        self.suspended = False

        class _Fixture:
            def __init__(self, outer: "FakeJob") -> None:
                self.regional = ctx.regional

            def delete(self) -> None:
                pass

        self.fixture = _Fixture(self)
        self.prewarm = type("P", (), {"cleanup": staticmethod(lambda: None)})()
        FakeJob.created.append(self)

    def start(self) -> dict[str, Any]:
        return {}

    def suspend(self) -> None:
        self.suspended = True

    def delete(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _patch_job(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeJob.created = []
    monkeypatch.setattr(att, "ManagedFailJob", FakeJob)


def settings(spec: Any, tmp_path: Path):
    s = make_settings(spec)
    (tmp_path / "site.yaml").write_text("site: test\n")
    return type(s)(**{**s.__dict__, "site_file": tmp_path / "site.yaml"})


def attempt_doc(
    attempt_id: str,
    *,
    terminal: str = "FAILED",
    decision: str = "PLAN_CREATED",
    containment_ops=("FREEZE_EVIDENCE", "STOP_WORKLOADS"),
    recovery_ops=("RESTART_WORKLOAD",),
    restart_count: int = 1,
    reservations=None,
    restart_status: str = "SUCCEEDED",
    restart_reason: str = "",
    incident_state: str = "ACTION_PENDING",
    duplicate: bool = False,
    restart_attempt_id: str | None = None,
    restarts: int = 1,
    extra_attempts: dict[str, dict[str, Any]] | None = None,
    notifications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One attempt's records. ``restart_attempt_id`` is what the product's
    RESTART_WORKLOAD execution recorded; ``restarts`` recovery workflows carry
    it; ``extra_attempts`` are further attempt entries (a restart copy's);
    ``notifications`` the attempt's incidents' notifications (one by default)."""

    reservations = (
        reservations
        if reservations is not None
        else [f"res-{i}" for i in range(restart_count)]
    )
    containment_wf = workflow(
        "wf-c", "inc-c", list(containment_ops), status="SUCCEEDED"
    )
    recovery = []
    if recovery_ops:
        details: dict[str, Any] = {"reason": restart_reason}
        if restart_attempt_id:
            details["restart_attempt_id"] = restart_attempt_id
        executions = (
            [
                {
                    "operation": "RESTART_WORKLOAD",
                    "status": restart_status,
                    "details": details,
                }
            ]
            if "RESTART_WORKLOAD" in recovery_ops
            else []
        )
        for index in range(restarts):
            recovery_wf = workflow(
                f"wf-r{index}",
                f"inc-r{index}",
                list(recovery_ops),
                status="SUCCEEDED",
                executions=executions,
            )
            recovery.append(
                {
                    "incident": {
                        "incident_id": f"inc-r{index}",
                        "job_id": attempt_id.rsplit("-", 1)[0],
                        "state": incident_state,
                    },
                    "workflow": recovery_wf,
                }
            )
    return {
        "attempts": {
            attempt_id: {
                "event": {"terminal_status": terminal},
                "decision": {
                    "status": decision,
                    "duplicate": duplicate,
                    "reason": "r",
                    "event_key": f"gpu-eks-cluster/{attempt_id}/TrainingAttemptFailureDetected",
                    "recovery_plan_id": "plan-1",
                },
                "containment_incident": {
                    "incident_id": "inc-c",
                    "job_id": attempt_id.rsplit("-", 1)[0],
                    "state": incident_state,
                },
                "containment_workflows": [containment_wf],
                "recovery": recovery,
                "notifications": [notification("n-1")]
                if notifications is None
                else list(notifications),
            },
            **(extra_attempts or {}),
        },
        "restart_budget": {
            "restart_count": restart_count,
            "reservation_ids": reservations,
            "budget": max(restart_count, 1),
        },
    }


def notification(notification_id: str, incident_id: str = "inc-c") -> dict[str, Any]:
    return {
        "notification_id": notification_id,
        "incident_id": incident_id,
        "category": "INCIDENT",
        "deduplication_key": f"{incident_id}/{notification_id}",
        "created_at": "2026-10-25T00:00:00+00:00",
        "subject": "s",
    }


def context(spec: Any, tmp_path: Path, doc: dict[str, Any]):
    regional = FakeRegional(cpu_python=store_script({"attempt": doc}))
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        settings=settings(spec, tmp_path),
        probes={},
    )


def test_nonzero_exit_passes(tmp_path: Path) -> None:
    spec = SPECS["nonzero_exit"]
    ctx = context(spec, tmp_path, {})
    attempt_id = None

    # The job name is derived inside execute; capture it to build the doc.
    def responder(script: str, *a: str) -> dict[str, Any]:
        nonlocal attempt_id
        attempt_id = a[2]
        return attempt_doc(attempt_id)

    ctx.regional.cpu_python_responder = responder
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def _run(spec_name: str, tmp_path: Path, build) -> dict[str, Any]:
    spec = SPECS[spec_name]
    captured: dict[str, Any] = {}

    def responder(script: str, *a: str) -> dict[str, Any]:
        captured["attempt_id"] = a[2]
        return build(a[2])

    regional = FakeRegional(cpu_python=responder)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        settings=settings(spec, tmp_path),
        probes={},
    )
    return spec.execute(ctx)


def test_one_restart_and_budget_consumed(tmp_path: Path) -> None:
    assert (
        _run("one_restart", tmp_path, lambda aid: attempt_doc(aid, restart_count=1))[
            "verdict"
        ]
        == "PASS"
    )
    assert (
        _run(
            "budget_consumed", tmp_path, lambda aid: attempt_doc(aid, restart_count=1)
        )["verdict"]
        == "PASS"
    )


def test_one_restart_fails_on_double_deduction(tmp_path: Path) -> None:
    result = _run(
        "one_restart",
        tmp_path,
        lambda aid: attempt_doc(aid, restart_count=2, reservations=["a", "b"]),
    )
    assert result["verdict"] == "FAIL"


def test_cleanup_timeout(tmp_path: Path) -> None:
    assert (
        _run(
            "cleanup_timeout",
            tmp_path,
            lambda aid: attempt_doc(aid, terminal="TIMED_OUT", recovery_ops=()),
        )["verdict"]
        == "PASS"
    )
    assert (
        _run(
            "cleanup_timeout",
            tmp_path,
            lambda aid: attempt_doc(aid, terminal="FAILED", recovery_ops=()),
        )["verdict"]
        == "FAIL"
    )


def test_cleanup_timeout_injects_a_failed_rank_and_a_rank_that_ignores_the_stop(
    tmp_path: Path,
) -> None:
    """Live 2026-10-25 (eks): the variant ran one Pod whose only container
    ignored TERM and slept; nothing failed, the product saw a healthy attempt
    for 900 s and the terminal stayed None. The product's TIMED_OUT needs a
    detected failure first (watcher.py: the cleanup deadline starts at the
    FailureDetectedEvent) and a critical rank still not terminated at that
    deadline: two ranks, one exiting non-zero, one ignoring the stop, on one
    node (4 GPUs each), with no restart budget -- the restart is not what this
    variant proves."""

    _run(
        "cleanup_timeout",
        tmp_path,
        lambda aid: attempt_doc(aid, terminal="TIMED_OUT", recovery_ops=()),
    )
    (job,) = FakeJob.created
    assert job.mode == {"Master": "exit:1", "Worker": "hang"}, job.mode
    assert (job.ranks, job.gpu_count, job.restart_budget) == (2, 4, 0)


def test_controller_stop_suspends_the_job_before_reading_the_stop(
    tmp_path: Path,
) -> None:
    """Live 2026-10-25 (eksb): the variant started a job that waited for a stop
    and never stopped it; 900 s later the decision was None. The injection is
    the job controller's suspend of the workload (``spec.runPolicy.suspend``,
    what Kueue does), after which the product must read a STOPPED terminal and
    answer NO_ACTION."""

    suspended_when_read: list[bool] = []

    def build(aid: str) -> dict[str, Any]:
        suspended_when_read.append(FakeJob.created[-1].suspended)
        return attempt_doc(
            aid, terminal="STOPPED", decision="NO_ACTION", recovery_ops=()
        )

    result = _run("controller_stop", tmp_path, build)
    assert result["verdict"] == "PASS", result["errors"]
    assert suspended_when_read and all(suspended_when_read), (
        "the stop is read only after the suspend was sent"
    )
    assert FakeJob.created[-1].mode == "stop"
    # A terminal that is not a stop, or a decision that restarts, is a FAIL.
    for terminal, decision in (("FAILED", "NO_ACTION"), ("STOPPED", "PLAN_CREATED")):
        result = _run(
            "controller_stop",
            tmp_path,
            lambda aid, t=terminal, d=decision: attempt_doc(
                aid, terminal=t, decision=d, recovery_ops=()
            ),
        )
        assert result["verdict"] == "FAIL", (terminal, decision, result)


def test_user_and_controller_stop(tmp_path: Path) -> None:
    for variant in ("user_stop", "controller_stop"):
        result = _run(
            variant,
            tmp_path,
            lambda aid: attempt_doc(
                aid, terminal="STOPPED", decision="NO_ACTION", recovery_ops=()
            ),
        )
        assert result["verdict"] == "PASS", (variant, result["errors"])
    # a user stop that nonetheless restarted is a FAIL
    result = _run(
        "user_stop",
        tmp_path,
        lambda aid: attempt_doc(
            aid,
            terminal="STOPPED",
            decision="NO_ACTION",
            recovery_ops=("RESTART_WORKLOAD",),
        ),
    )
    assert result["verdict"] == "FAIL"


# --------------------------------------------------------------------------- #
# 028 old_terminal: the product's recorded terminal, redelivered
# --------------------------------------------------------------------------- #
def untouched() -> dict[str, Any]:
    return {
        "event": None,
        "decision": None,
        "containment_incident": None,
        "containment_workflows": [],
        "recovery": [],
        "notifications": [],
    }


def replayed(
    original: dict[str, Any], *, duplicate: bool = True, plan: str = "plan-1"
) -> dict[str, Any]:
    """What TERMINAL_REPLAY_PROBE prints for the old attempt."""

    return {
        "attempt_id": original["attempt_id"],
        "event_key": original["event_key"],
        "terminal_status": "FAILED",
        "ended_at": "2026-10-25T00:00:00+00:00",
        "before": original,
        "decision": {**original, "duplicate": duplicate, "recovery_plan_id": plan},
    }


def _run_old_terminal(
    tmp_path: Path,
    *,
    new_attempt_records: dict[str, Any] | None = None,
    restarts: int = 1,
    restart_count: int = 1,
    restart_attempt: bool = True,
    duplicate: bool = True,
    plan: str = "plan-1",
    refused: bool = False,
) -> tuple[dict[str, Any], list[tuple[str, ...]]]:
    spec = SPECS["old_terminal"]
    calls: list[tuple[str, ...]] = []

    def responder(script: str, *a: str) -> dict[str, Any]:
        if script == replay.TERMINAL_REPLAY_PROBE:
            calls.append(("replay", *a))
            if refused:
                return {"attempt_id": a[1], "refused": "no recorded terminal event"}
            original = {
                "cluster_id": a[0],
                "attempt_id": a[1],
                "event_key": f"{a[0]}/{a[1]}/TrainingAttemptFailureDetected",
                "status": "PLAN_CREATED",
                "reason": "r",
                "duplicate": False,
                "matched_marker_ids": [],
                "recovery_plan_id": "plan-1",
            }
            return replayed(original, duplicate=duplicate, plan=plan)
        calls.append(("attempt", *a))
        attempt_id = a[2]
        new_id = f"{attempt_id}-r-deadbeef"
        return attempt_doc(
            attempt_id,
            restart_attempt_id=new_id if restart_attempt else None,
            restarts=restarts,
            restart_count=restart_count,
            reservations=[f"res-{i}" for i in range(restart_count)],
            extra_attempts={new_id: new_attempt_records or untouched()},
        )

    regional = FakeRegional(cpu_python=responder)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        settings=settings(spec, tmp_path),
        probes={},
    )
    return spec.execute(ctx), calls


def test_old_terminal_replays_the_recorded_terminal_against_a_running_copy(
    tmp_path: Path,
) -> None:
    result, calls = _run_old_terminal(tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    (job,) = FakeJob.created
    assert job.mode == "exit_once:1" and job.restart_budget == 1, job.mode
    replays = [call for call in calls if call[0] == "replay"]
    assert replays == [("replay", "gpu-eks-cluster", job.attempt_id)], replays
    assert result["new_attempt"] == f"{job.attempt_id}-r-deadbeef"
    assert result["replayed_decision"]["duplicate"] is True
    # The replay came after the restart copy was known (attempt reads first).
    assert calls.index(replays[0]) > 1, calls
    final = calls[-1]
    assert final[0] == "attempt" and final[-1] == result["new_attempt"], (
        "the last read asks for both attempts"
    )
    spec = SPECS["old_terminal"]
    assert "handle_terminal" in spec.injection and "duplicate=true" in spec.expected


@pytest.mark.parametrize(
    ("overrides", "stage", "needle"),
    [
        ({"duplicate": False}, "replay_duplicate", "decided anew"),
        ({"plan": "plan-2"}, "replay_duplicate", "recovery_plan_id"),
        ({"restarts": 2}, "single_restart", "2 restart workflows"),
        ({"restart_count": 2}, "budget", "exceeds"),
        (
            {
                "new_attempt_records": {
                    **untouched(),
                    "containment_incident": {"incident_id": "inc-new"},
                }
            },
            "new_attempt_untouched",
            "containment_incident",
        ),
        (
            {
                "new_attempt_records": {
                    **untouched(),
                    "event": {"terminal_status": "STOPPED"},
                }
            },
            "new_attempt_untouched",
            "has a event",
        ),
    ],
)
def test_old_terminal_fails_when_the_replay_controls_anything(
    tmp_path: Path, overrides: dict[str, Any], stage: str, needle: str
) -> None:
    result, _calls = _run_old_terminal(tmp_path, **overrides)
    assert result["verdict"] == "FAIL", result
    assert any(needle in error for error in result["stages"][stage]), result["stages"]


def test_old_terminal_is_inconclusive_without_a_new_attempt_or_a_recorded_terminal(
    tmp_path: Path,
) -> None:
    with pytest.raises(VariantInconclusive, match="no new attempt"):
        _run_old_terminal(tmp_path, restart_attempt=False)
    with pytest.raises(VariantInconclusive, match="could not be replayed"):
        _run_old_terminal(tmp_path, refused=True)


def test_missing_allocation_is_deferred_with_the_watcher_reason() -> None:
    spec = SPECS["missing_allocation"]
    assert not spec.implemented and spec.deferred.startswith("FIXTURE_NOT_READY: "), (
        spec.deferred
    )
    for needle in ("spec.nodeName", "/v1/attempts/terminal", "from_missing_allocation"):
        assert needle in spec.deferred, needle
    assert spec.lanes == ("eks", "hp")


REPLAY_CODE = compile(replay.TERMINAL_REPLAY_PROBE, "<terminal-replay-probe>", "exec")


def _replay_probe_run(context: Any, arguments: list[str]) -> dict[str, Any]:
    import contextlib
    import io
    import sys
    from unittest import mock

    from gpu_fault.app import ApplicationContext

    output = io.StringIO()
    with (
        mock.patch.object(ApplicationContext, "from_environment", lambda: context),
        mock.patch.object(sys, "argv", ["terminal-replay-probe", *arguments]),
        contextlib.redirect_stdout(output),
    ):
        exec(REPLAY_CODE, {"__name__": "__main__"})
    return json.loads(output.getvalue().splitlines()[-1])


def test_terminal_replay_probe_redelivers_the_stored_event_and_refuses_strangers() -> (
    None
):
    import types
    from datetime import datetime, timezone

    from gpu_fault.models import (
        CompletionDecision,
        DecisionStatus,
        Environment,
        TerminalEvent,
        TerminalStatus,
    )
    from gpu_fault.store import InMemoryStore

    store = InMemoryStore()
    event = TerminalEvent(
        cluster_id="gpu-eks-cluster",
        environment=Environment.EKS,
        job_id="gl28ot-x",
        attempt_id="gl28ot-x-a001",
        terminal_status=TerminalStatus.FAILED,
        ended_at=datetime(2026, 10, 25, tzinfo=timezone.utc),
        runtime_profile_version="p1",
    )
    assert store.save_event_if_absent(event), "the event row is new"
    original = CompletionDecision(
        cluster_id=event.cluster_id,
        attempt_id=event.attempt_id,
        event_key=event.event_key,
        status=DecisionStatus.PLAN_CREATED,
        reason="r",
        recovery_plan_id="plan-1",
    )
    store.save_decision(original)
    handled: list[TerminalEvent] = []

    class Completion:
        def handle_terminal(self, item: TerminalEvent) -> CompletionDecision:
            handled.append(item)
            return original.model_copy(update={"duplicate": True})

    context = types.SimpleNamespace(store=store, completion=Completion())
    document = _replay_probe_run(context, ["gpu-eks-cluster", "gl28ot-x-a001"])
    assert handled == [event], "the product's own recorded event is what is redelivered"
    assert document["decision"]["duplicate"] is True
    assert document["before"]["recovery_plan_id"] == "plan-1"
    assert (
        document["event_key"] == event.event_key
        and document["terminal_status"] == "FAILED"
    )
    assert _replay_probe_run(context, ["gpu-eks-cluster", "gl28ot-y-a001"]) == {
        "attempt_id": "gl28ot-y-a001",
        "refused": "no recorded terminal event",
    }
    # 029 duplicate_event's jobs are gl29 test attempts: accepted, nothing else is.
    assert _replay_probe_run(context, ["gpu-eks-cluster", "gl29de-y-a001"]) == {
        "attempt_id": "gl29de-y-a001",
        "refused": "no recorded terminal event",
    }
    assert replay.REPLAY_ATTEMPT_PREFIXES == ("gl28", "gl29")
    with pytest.raises(ValueError, match="outside the test prefix"):
        _replay_probe_run(context, ["gpu-eks-cluster", "train-123-a001"])
    with pytest.raises(ValueError, match="outside the test prefix"):
        _replay_probe_run(context, ["gpu-eks-cluster", "gl30x-a001"])
    assert len(handled) == 1, "a refused replay redelivers nothing"


def test_restart_withheld_passes(tmp_path: Path) -> None:
    result = _run(
        "restart_withheld",
        tmp_path,
        lambda aid: attempt_doc(
            aid,
            restart_count=0,
            reservations=[],
            recovery_ops=("RESTART_WORKLOAD",),
            restart_status="WAITING",
            restart_reason="RESTART_BUDGET_EXHAUSTED",
            incident_state="ESCALATED",
        ),
    )
    assert result["verdict"] == "PASS", result["errors"]


def test_restart_withheld_fails_if_restart_succeeded(tmp_path: Path) -> None:
    result = _run(
        "restart_withheld",
        tmp_path,
        lambda aid: attempt_doc(
            aid,
            restart_count=0,
            reservations=[],
            recovery_ops=("RESTART_WORKLOAD",),
            restart_status="SUCCEEDED",
        ),
    )
    assert result["verdict"] == "FAIL"


# --------------------------------------------------------------------------- #
# 029 duplicate_event: the same terminal delivered twice
# --------------------------------------------------------------------------- #
def _run_duplicate_event(
    tmp_path: Path,
    *,
    after: dict[str, Any] | None = None,
    duplicate: bool = True,
    plan: str = "plan-1",
    refused: bool = False,
) -> tuple[dict[str, Any], list[tuple[str, ...]]]:
    """``after`` overrides the attempt records read after the redelivery."""

    spec = SPECS["duplicate_event"]
    calls: list[tuple[str, ...]] = []
    replayed_once = False

    def responder(script: str, *a: str) -> dict[str, Any]:
        nonlocal replayed_once
        if script == replay.TERMINAL_REPLAY_PROBE:
            calls.append(("replay", *a))
            replayed_once = True
            if refused:
                return {"attempt_id": a[1], "refused": "no recorded terminal event"}
            original = {
                "cluster_id": a[0],
                "attempt_id": a[1],
                "event_key": f"{a[0]}/{a[1]}/TrainingAttemptFailureDetected",
                "status": "PLAN_CREATED",
                "reason": "r",
                "duplicate": False,
                "matched_marker_ids": [],
                "recovery_plan_id": "plan-1",
            }
            return replayed(original, duplicate=duplicate, plan=plan)
        calls.append(("attempt", *a))
        attempt_id = a[2]
        new_id = f"{attempt_id}-r-deadbeef"
        fields: dict[str, Any] = {
            "restart_attempt_id": new_id,
            "extra_attempts": {new_id: untouched()},
        }
        if replayed_once and after:
            fields.update(after)
        return attempt_doc(attempt_id, **fields)

    regional = FakeRegional(cpu_python=responder)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        settings=settings(spec, tmp_path),
        probes={},
    )
    return spec.execute(ctx), calls


def test_duplicate_event_redelivers_the_same_terminal_and_nothing_changes(
    tmp_path: Path,
) -> None:
    result, calls = _run_duplicate_event(tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    (job,) = FakeJob.created
    assert job.mode == "exit_once:1" and job.restart_budget == 1, job.mode
    assert job.name.startswith("gl29de-"), job.name
    replays = [call for call in calls if call[0] == "replay"]
    assert replays == [("replay", "gpu-eks-cluster", job.attempt_id)], replays
    # The redelivery came after the first delivery settled (decision read,
    # restart copy recorded) and the last read is after it, for both attempts.
    assert calls.index(replays[0]) > 1, calls
    assert calls[-1][0] == "attempt" and calls[-1][-1] == result["new_attempt"]
    assert result["replayed_decision"]["duplicate"] is True
    assert result["notifications_before"] == 1
    assert result["stages"]["unchanged"] == []
    # The baseline is the read right before the redelivery, after the watcher
    # pass: it asks for both attempts, like the read after it.
    baseline = calls[calls.index(replays[0]) - 1]
    assert baseline[0] == "attempt" and baseline[-1] == result["new_attempt"], baseline
    spec = SPECS["duplicate_event"]
    assert "handle_terminal" in spec.injection and "duplicate=true" in spec.expected
    assert spec.proof_mode == "native" and spec.approvals == ()


@pytest.mark.parametrize(
    ("overrides", "stage", "needle"),
    [
        ({"duplicate": False}, "replay_duplicate", "decided anew"),
        ({"plan": "plan-2"}, "replay_duplicate", "recovery_plan_id"),
        ({"after": {"restart_count": 2}}, "budget", "exceeds"),
        ({"after": {"restart_count": 2}}, "unchanged", "restart_count changed"),
        (
            {"after": {"reservations": ["res-9"]}},
            "unchanged",
            "reservation_ids changed",
        ),
        ({"after": {"restarts": 2}}, "single_restart", "2 restart workflows"),
        ({"after": {"restarts": 2}}, "unchanged", "RESTART_WORKLOAD workflows after"),
        (
            {"after": {"notifications": [notification("n-1"), notification("n-2")]}},
            "unchanged",
            "1 new notification(s)",
        ),
        ({"after": {"duplicate": True}}, "unchanged", "flagged duplicate"),
        ({"after": {"decision": "NO_ACTION"}}, "unchanged", "status changed"),
    ],
)
def test_duplicate_event_fails_when_the_redelivery_changed_anything(
    tmp_path: Path, overrides: dict[str, Any], stage: str, needle: str
) -> None:
    result, _calls = _run_duplicate_event(tmp_path, **overrides)
    assert result["verdict"] == "FAIL", result
    assert any(needle in error for error in result["stages"][stage]), result["stages"]


def test_duplicate_event_is_inconclusive_without_a_recorded_terminal(
    tmp_path: Path,
) -> None:
    with pytest.raises(VariantInconclusive, match="could not be replayed"):
        _run_duplicate_event(tmp_path, refused=True)


def test_unchanged_errors_reads_only_the_attempts_own_records() -> None:
    before = attempt_doc("gl29de-x-a001")
    same = attempt_doc("gl29de-x-a001")
    assert replay.unchanged_errors(before, same, "gl29de-x-a001") == []
    # A notification of another incident is not the attempt's.
    other = attempt_doc("gl29de-x-a001", notifications=[notification("n-1")])
    other["attempts"]["gl29de-x-a001"]["notifications"].append(
        notification("n-7", incident_id="inc-other")
    )
    errors = replay.unchanged_errors(before, other, "gl29de-x-a001")
    assert errors == [
        "the redelivery produced 1 new notification(s) for the attempt's "
        "incidents: ['n-7']"
    ], "the probe already filtered by the attempt's incidents; any new id counts"


def test_single_notification(tmp_path: Path) -> None:
    result = _run(
        "single_notification",
        tmp_path,
        lambda aid: attempt_doc(
            aid,
            restart_count=0,
            reservations=[],
            recovery_ops=("RESTART_WORKLOAD",),
            restart_status="WAITING",
            restart_reason="RESTART_BUDGET_EXHAUSTED",
            incident_state="ESCALATED",
        ),
    )
    assert result["verdict"] == "PASS", result["errors"]


def test_deferred_variants() -> None:
    # The two dangerous watcher variants are executable behind their tokens
    # (test_general_live_network_isolation / test_general_live_watcher_restart).
    assert SPECS["api_unreachable"].implemented, "api_unreachable is implemented"
    assert SPECS["api_unreachable"].approvals == ("WATCHER_NETWORK_ISOLATION",)
    assert SPECS["worker_restart"].implemented, "worker_restart is implemented"
    assert SPECS["worker_restart"].approvals == ("CONTROL_PLANE_RESTART",)
    # 029 hardware_chain_continues moved to general_live_attempt_hardware.py.
    assert "hardware_chain_continues" not in SPECS, (
        "the hardware chain is implemented in its own module, not deferred here"
    )


# --------------------------------------------------------------------------- #
# 029 store-level variants over the product's restart ledger
# --------------------------------------------------------------------------- #
def budget_regional(store: Any, mutate=None) -> FakeRegional:
    def responder(script: str, *arguments: str) -> dict[str, Any]:
        assert script == BUDGET_PROBE, "store variants ship only the budget probe"
        document = budget_probe_run(store, arguments)
        if mutate is not None:
            mutate(arguments, document)
        return document

    return FakeRegional(cpu_python=responder)


def test_store_variants_are_implemented_lane_locked_and_unapproved() -> None:
    for variant in ("concurrent_claim", "missing_context"):
        spec = SPECS[variant]
        assert spec.implemented, variant
        assert spec.flags == "P" and spec.lanes == ("eks", "hp"), variant
        assert spec.approvals == () and spec.risk == "L", variant
        assert "test" in spec.cleanup and "release" in spec.cleanup, variant


@pytest.mark.parametrize("variant", ["concurrent_claim", "missing_context"])
def test_store_variants_pass_and_clean_up_over_the_product_ledger(
    tmp_path: Path, variant: str
) -> None:
    store = budget_ledger_store()
    spec = SPECS[variant]
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=budget_regional(store), probes={}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert ctx.cleanup.pending() == [], "the variant cleaned up itself"
    assert result["cleanup_receipt"]["clean"] is True
    assert result["job_id"].startswith("gl29"), "result['job_id'].startswith('gl29')"
    with pytest.raises(NotFoundError):
        store.get_restart_budget("gpu-eks-cluster", result["job_id"])


def test_concurrent_claim_fails_on_a_granted_rival(tmp_path: Path) -> None:
    def grant_rival(arguments: tuple[str, ...], document: dict[str, Any]) -> None:
        if arguments[0] == "concurrent":
            document["phases"]["rival"]["results"][0]["reserved"] = True

    store = budget_ledger_store()
    spec = SPECS["concurrent_claim"]
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=budget_regional(store, grant_rival), probes={}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("past the budget" in error for error in result["errors"]), result[
        "errors"
    ]
    assert result["cleanup_receipt"]["clean"] is True, "cleanup still runs on a FAIL"


def test_missing_context_fails_when_the_preflight_admits_the_step(
    tmp_path: Path,
) -> None:
    def admit(arguments: tuple[str, ...], document: dict[str, Any]) -> None:
        if arguments[0] == "missing-context":
            document["preflight"] = None

    store = budget_ledger_store()
    spec = SPECS["missing_context"]
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=budget_regional(store, admit), probes={}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("admitted" in error for error in result["errors"]), result["errors"]


def test_store_variants_fail_when_the_ledger_row_remains(tmp_path: Path) -> None:
    def keep_row(arguments: tuple[str, ...], document: dict[str, Any]) -> None:
        if arguments[0] == "cleanup":
            document["clean"] = False
            document["remaining"] = {"restart_count": 0, "reservation_ids": []}

    store = budget_ledger_store()
    spec = SPECS["missing_context"]
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=budget_regional(store, keep_row), probes={}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("ledger remains" in error for error in result["errors"]), result[
        "errors"
    ]
