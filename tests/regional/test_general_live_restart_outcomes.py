"""GF-GENERAL-LIVE-028/029 restart outcomes: shown, not assumed.

``one_restart``, ``budget_consumed`` and ``nonzero_exit`` read the records at
the first sight of a decision -- before the recovery workflow's claim reserved
the budget -- and passed a ledger at ``restart_count`` 0 and an empty recovery
history; ``restart_withheld`` and ``single_notification`` passed a WAITING
restart or none at all and never read a notification. The records here are
the shape the product writes (``service._decide``,
``restart_budget_preflight.withhold_exhausted_restart``,
``notifications/restart_guard.build_budget_exhausted``): each old vacuous
shape must now FAIL or be INCONCLUSIVE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_fault.execution.restart_budget_preflight import BUDGET_EXHAUSTED_REASON
from scripts.e2e.regional import general_live_attempts as att
from scripts.e2e.regional import general_live_network_isolation as isolation
from scripts.e2e.regional import general_live_restart_outcomes as outcomes
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_store import ATTEMPT_PROBE
from tests.regional._general_live_fakes import (
    FakeRegional,
    make_context,
    make_settings,
    store_poll_clock,
    workflow,
)

SPECS = {spec.variant: spec for spec in att.VARIANTS}
CLUSTER = "gpu-eks-cluster"  # make_settings' cluster id
COPY_SUFFIX = "-r-0123abcd"


class FakeJob:
    """The managed job double: its identity, no cluster."""

    created: list["FakeJob"] = []

    def __init__(self, ctx: Any, *, name: str, mode: Any, **options: Any) -> None:
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.mode = mode
        self.options = options
        self.fixture = type("F", (), {"regional": ctx.regional})()
        FakeJob.created.append(self)

    def start(self) -> dict[str, Any]:
        return {}

    def delete(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _patch_job(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeJob.created = []
    monkeypatch.setattr(att, "ManagedFailJob", FakeJob)


# --------------------------------------------------------------------------- #
# the product's records
# --------------------------------------------------------------------------- #
def containment(stop: str = "SUCCEEDED") -> list[dict[str, Any]]:
    return [
        workflow(
            "wf-c",
            "inc-c",
            ["FREEZE_EVIDENCE", "STOP_WORKLOADS"],
            status="SUCCEEDED",
            executions=[{"operation": "STOP_WORKLOADS", "status": stop}],
        )
    ]


def recovery(
    status: str, execution: dict[str, Any] | None, *, state: str
) -> list[dict[str, Any]]:
    return [
        {
            "incident": {"incident_id": "inc-r", "job_id": None, "state": state},
            "workflow": workflow(
                "wf-r",
                "inc-r",
                ["RESTART_WORKLOAD"],
                status=status,
                executions=[execution] if execution else [],
            ),
        }
    ]


def entry(
    *,
    recovery_items: list[dict[str, Any]],
    notifications: list[dict[str, Any]] = (),  # type: ignore[assignment]
    decision: str | None = "PLAN_CREATED",
    stop: str = "SUCCEEDED",
) -> dict[str, Any]:
    return {
        "event": {"terminal_status": "FAILED"},
        "decision": {"status": decision, "reason": "r"} if decision else None,
        "containment_incident": {"incident_id": "inc-c", "state": "RECOVERED"},
        "containment_workflows": containment(stop),
        "recovery": recovery_items,
        "notifications": list(notifications),
    }


def ledger(count: int, reservations: list[str], budget: int) -> dict[str, Any]:
    return {"restart_count": count, "reservation_ids": reservations, "budget": budget}


def restarted(job: str, *, copy: bool = True) -> dict[str, Any]:
    """Budget 1, restarted once: RESTART_WORKLOAD SUCCEEDED with the copy's id,
    the ledger holding the recovery workflow's own reservation."""

    details = {"restart_attempt_id": job + COPY_SUFFIX} if copy else {}
    run = {"operation": "RESTART_WORKLOAD", "status": "SUCCEEDED", "details": details}
    return {
        "attempts": {
            f"{job}-a001": entry(
                recovery_items=recovery("SUCCEEDED", run, state="RECOVERED")
            )
        },
        "restart_budget": ledger(1, ["wf-r/0/RESTART_WORKLOAD"], 1),
    }


def budget_notice(job: str, notification_id: str = "n-budget") -> dict[str, Any]:
    return {
        "notification_id": notification_id,
        "incident_id": "inc-r",
        "category": "GENERAL",
        "deduplication_key": f"{CLUSTER}/{job}{outcomes.BUDGET_NOTICE}0",
        "created_at": "2026-10-24T00:05:00+00:00",
        "subject": "[ACTION REQUIRED][GPU training automatic restarts stopped]",
    }


def withheld(job: str, *, notices: Any = None, run: Any = "default") -> dict[str, Any]:
    """Budget 0, withheld: one FAILED RESTART_WORKLOAD for the exhausted budget
    naming the one notice, the recovery FAILED and its incident ESCALATED."""

    if run == "default":
        run = {
            "operation": "RESTART_WORKLOAD",
            "status": "FAILED",
            "details": {
                "reason": BUDGET_EXHAUSTED_REASON,
                "restart_count": 0,
                "restart_budget": 0,
                "notification_id": "n-budget",
            },
        }
    return {
        "attempts": {
            f"{job}-a001": entry(
                recovery_items=recovery("FAILED", run, state="ESCALATED"),
                notifications=[budget_notice(job)] if notices is None else notices,
            )
        },
        "restart_budget": ledger(0, [], 0),
    }


def run_variant(
    tmp_path: Path,
    variant: str,
    build: Any,
    *,
    running: bool = True,
    poll_clock: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """``execute`` over records ``build(job, attempt_ids, read)`` returns; the
    watcher's observations show the restart copy RUNNING when ``running``."""

    spec = SPECS[variant]
    reads: list[str] = []

    def responder(script: str, *arguments: str) -> dict[str, Any]:
        if script is ATTEMPT_PROBE:
            reads.append("attempt")
            job, attempts = arguments[1], list(arguments[2:])
            document = build(job, len(reads))
            for attempt in attempts:
                document["attempts"].setdefault(attempt, {})
            return document
        if script is isolation.ISOLATION_RECORDS_PROBE:
            reads.append("observations")
            job = arguments[1]
            phase = "RUNNING" if running else "PENDING"
            return {
                "observations": [
                    {
                        "attempt_id": job + COPY_SUFFIX,
                        "job_id": job,
                        "workload_phase": phase,
                    }
                ]
            }
        raise AssertionError(f"unexpected probe for {variant}")

    settings = make_settings(spec)
    (tmp_path / "site.yaml").write_text("site: test\n")
    settings = type(settings)(
        **{**settings.__dict__, "site_file": tmp_path / "site.yaml"}
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=responder),
        settings=settings,
        probes={},
    )
    if poll_clock:
        store_poll_clock(ctx)
    return spec.execute(ctx), reads


# --------------------------------------------------------------------------- #
# graders (pure)
# --------------------------------------------------------------------------- #
def test_budget_errors_want_exactly_the_consumed_count() -> None:
    reserved = {"restart_budget": ledger(1, ["wf-r/0/RESTART_WORKLOAD"], 1)}
    assert outcomes.budget_errors(reserved, consumed=1) == []
    # The old helper only refused a count ABOVE the expected one.
    released = {"restart_budget": ledger(0, [], 1)}
    assert outcomes.budget_errors(released, consumed=1) == [
        "restart_count is 0, expected exactly 1"
    ]
    assert outcomes.budget_errors({"restart_budget": None}, consumed=1) == [
        "no restart-budget ledger row for the job"
    ]
    assert outcomes.budget_errors({"restart_budget": None}, consumed=0) == []
    doubled = {"restart_budget": ledger(2, ["a"], 1)}
    assert outcomes.budget_errors(doubled, consumed=1) == [
        "restart_count does not match the reservation ids",
        "restart_count is 2, expected exactly 1",
    ]


def test_restart_once_needs_the_recorded_copy_and_its_own_reservation() -> None:
    document = restarted("job")
    assert (
        outcomes.restart_once_errors(document["attempts"]["job-a001"], document) == []
    )
    blind = restarted("job", copy=False)
    assert outcomes.restart_once_errors(blind["attempts"]["job-a001"], blind) == [
        "the product's RESTART_WORKLOAD recorded no restart_attempt_id"
    ]
    # The old vacuous shape: decision only, no recovery, a count of 0.
    early = {
        "attempts": {"job-a001": entry(recovery_items=[])},
        "restart_budget": ledger(0, [], 1),
    }
    errors = outcomes.restart_once_errors(early["attempts"]["job-a001"], early)
    assert "expected one recovery workflow, found 0" in errors, errors
    assert any("RESTART_WORKLOAD succeeded 0 times" in item for item in errors), errors
    assert any("reservation was lost" in item for item in errors), errors


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (
            {"operation": "RESTART_WORKLOAD", "status": "WAITING", "details": {}},
            "expected one RESTART_WORKLOAD withheld",
        ),
        (None, "found no execution"),
        (
            {"operation": "RESTART_WORKLOAD", "status": "SUCCEEDED", "details": {}},
            "succeeded despite the exhausted budget",
        ),
        (
            {
                "operation": "RESTART_WORKLOAD",
                "status": "FAILED",
                "details": {"reason": BUDGET_EXHAUSTED_REASON, "restart_budget": 1},
            },
            "read budget 1 with None used, expected 0 of 0",
        ),
    ],
    ids=["waiting", "never-ran", "succeeded", "other-budget"],
)
def test_withheld_errors_need_the_budget_failure_itself(run: Any, message: str) -> None:
    document = withheld("job", run=run)
    errors = outcomes.withheld_errors(document["attempts"]["job-a001"], budget=0)
    assert any(message in item for item in errors), errors


def test_withheld_without_a_planned_restart_is_inconclusive() -> None:
    planned_nothing = entry(recovery_items=[])
    with pytest.raises(VariantInconclusive, match="never reached"):
        outcomes.withheld_errors(planned_nothing, budget=0)


def test_withheld_still_needs_the_containment_stop() -> None:
    document = withheld("job")
    item = document["attempts"]["job-a001"]
    item["containment_workflows"] = containment(stop="FAILED")
    assert outcomes.withheld_errors(item, budget=0) == [
        "the containment's STOP_WORKLOADS never succeeded"
    ]


@pytest.mark.parametrize(
    ("notices", "message"),
    [
        ([], "0 budget-exhausted notifications"),
        ("two", "2 budget-exhausted notifications"),
        ("other-key", "is keyed"),
        ("unnamed", "names notification"),
    ],
)
def test_budget_notice_errors_count_and_bind_the_notice(
    notices: Any, message: str
) -> None:
    if notices == "two":
        notices = [budget_notice("job"), budget_notice("job", "n-2")]
    elif notices == "other-key":
        notices = [
            {
                **budget_notice("job"),
                "deduplication_key": "c/x/restart-budget-exhausted/0",
            }
        ]
    elif notices == "unnamed":
        notices = [budget_notice("job", "n-other")]
    document = withheld("job", notices=notices)
    errors = outcomes.budget_notice_errors(
        document["attempts"]["job-a001"], cluster_id=CLUSTER, job_id="job", budget=0
    )
    assert any(message in item for item in errors), errors
    assert (
        outcomes.budget_notice_errors(
            withheld("job")["attempts"]["job-a001"],
            cluster_id=CLUSTER,
            job_id="job",
            budget=0,
        )
        == []
    ), "the product's one notice, named by the withheld restart"


# --------------------------------------------------------------------------- #
# the variants over the settled outcome
# --------------------------------------------------------------------------- #
def test_one_restart_passes_on_one_shown_restart_and_a_running_copy(
    tmp_path: Path,
) -> None:
    result, reads = run_variant(
        tmp_path, "one_restart", lambda job, _read: restarted(job)
    )
    assert result["verdict"] == "PASS", result["errors"]
    job = FakeJob.created[-1]
    assert job.mode == "exit_once:1", "a single non-zero exit; the copy keeps running"
    assert result["restart_attempt"] == job.name + COPY_SUFFIX, result
    assert reads.count("observations") == 1, reads
    assert set(result["stages"]) == {
        "restart_once",
        "budget",
        "copy_running",
        "still_once",
    }, result["stages"]


@pytest.mark.parametrize("variant", ["one_restart", "budget_consumed", "nonzero_exit"])
def test_a_decision_without_the_restart_is_not_a_budgeted_restart(
    tmp_path: Path, variant: str
) -> None:
    # What the old code passed: a decision, a settled recovery whose ledger
    # row reads 0 (the reservation released without a restart).
    def build(job: str, _read: int) -> dict[str, Any]:
        document = restarted(job, copy=False)
        document["restart_budget"] = ledger(0, [], 1)
        return document

    result, _reads = run_variant(tmp_path, variant, build)
    assert result["verdict"] == "FAIL", result
    assert "restart_count is 0, expected exactly 1" in result["stages"]["budget"], (
        result["stages"]
    )


def test_one_restart_is_inconclusive_while_the_recovery_never_settles(
    tmp_path: Path,
) -> None:
    def build(job: str, _read: int) -> dict[str, Any]:
        document = restarted(job)
        item = document["attempts"][f"{job}-a001"]
        item["recovery"][0]["workflow"]["status"] = "PENDING"
        item["recovery"][0]["workflow"]["step_executions"] = []
        document["restart_budget"] = None
        return document

    with pytest.raises(VariantInconclusive, match="did not settle"):
        run_variant(tmp_path, "one_restart", build, poll_clock=True)


def test_one_restart_fails_when_the_copy_never_runs(tmp_path: Path) -> None:
    result, _reads = run_variant(
        tmp_path,
        "one_restart",
        lambda job, _read: restarted(job),
        running=False,
        poll_clock=True,
    )
    assert result["verdict"] == "FAIL", result
    (error,) = result["stages"]["copy_running"]
    assert "never observed the restart attempt" in error, error


def test_restart_withheld_passes_on_the_products_withholding(tmp_path: Path) -> None:
    result, _reads = run_variant(
        tmp_path, "restart_withheld", lambda job, _read: withheld(job)
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert FakeJob.created[-1].options["restart_budget"] == 0


@pytest.mark.parametrize("shape", ["waiting", "no-recovery"])
def test_restart_withheld_old_vacuous_shapes_no_longer_pass(
    tmp_path: Path, shape: str
) -> None:
    def build(job: str, _read: int) -> dict[str, Any]:
        document = withheld(job)
        item = document["attempts"][f"{job}-a001"]
        if shape == "waiting":
            item["recovery"][0]["workflow"]["status"] = "RUNNING"
            item["recovery"][0]["workflow"]["step_executions"] = [
                {
                    "operation": "RESTART_WORKLOAD",
                    "status": "WAITING",
                    "details": {"reason": BUDGET_EXHAUSTED_REASON},
                }
            ]
        else:
            item["recovery"] = []
        return document

    with pytest.raises(VariantInconclusive, match="did not settle"):
        run_variant(tmp_path, "restart_withheld", build, poll_clock=True)


def test_single_notification_passes_on_one_named_notice(tmp_path: Path) -> None:
    result, _reads = run_variant(
        tmp_path, "single_notification", lambda job, _read: withheld(job)
    )
    assert result["verdict"] == "PASS", result["errors"]
    assert [item["notification_id"] for item in result["budget_notices"]] == [
        "n-budget"
    ], result["budget_notices"]


def test_single_notification_fails_without_the_notice(tmp_path: Path) -> None:
    # The old variant never read a notification: this record passed it.
    result, _reads = run_variant(
        tmp_path, "single_notification", lambda job, _read: withheld(job, notices=[])
    )
    assert result["verdict"] == "FAIL", result
    (error,) = result["stages"]["notification"]
    assert error.startswith("0 budget-exhausted notifications"), error


def test_single_notification_fails_on_a_second_notice_later(tmp_path: Path) -> None:
    def build(job: str, read: int) -> dict[str, Any]:
        notices = [budget_notice(job)]
        if read > 2:  # the read a watcher pass after the settled outcome
            notices.append(budget_notice(job, "n-again"))
        return withheld(job, notices=notices)

    result, _reads = run_variant(tmp_path, "single_notification", build)
    assert result["verdict"] == "FAIL", result
    (error,) = result["stages"]["no_second_notice"]
    assert "n-again" in error, error


def test_a_decision_that_plans_no_restart_is_graded_without_waiting(
    tmp_path: Path,
) -> None:
    def build(job: str, _read: int) -> dict[str, Any]:
        document = withheld(job)
        document["attempts"][f"{job}-a001"]["decision"] = {"status": "NO_ACTION"}
        return document

    result, reads = run_variant(tmp_path, "restart_withheld", build)
    assert result["verdict"] == "FAIL", result
    assert result["stages"] == {
        "decision": ["terminal decision was NO_ACTION, expected a budgeted restart"]
    }, result["stages"]
    assert reads == ["attempt"], "no outcome wait for a restart never planned"
