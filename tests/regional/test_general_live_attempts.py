"""GF-GENERAL-LIVE-028/029 injector/verdict contract over scripted records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_fault.store import NotFoundError
from scripts.e2e.regional import general_live_attempts as att
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
) -> dict[str, Any]:
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
        executions = (
            [
                {
                    "operation": "RESTART_WORKLOAD",
                    "status": restart_status,
                    "details": {"reason": restart_reason},
                }
            ]
            if "RESTART_WORKLOAD" in recovery_ops
            else []
        )
        recovery_wf = workflow(
            "wf-r",
            "inc-r",
            list(recovery_ops),
            status="SUCCEEDED",
            executions=executions,
        )
        recovery = [
            {
                "incident": {
                    "incident_id": "inc-r",
                    "job_id": attempt_id.rsplit("-", 1)[0],
                    "state": incident_state,
                },
                "workflow": recovery_wf,
            }
        ]
    return {
        "attempts": {
            attempt_id: {
                "event": {"terminal_status": terminal},
                "decision": {"status": decision, "duplicate": duplicate, "reason": "r"},
                "containment_incident": {
                    "incident_id": "inc-c",
                    "job_id": attempt_id.rsplit("-", 1)[0],
                    "state": incident_state,
                },
                "containment_workflows": [containment_wf],
                "recovery": recovery,
            }
        },
        "restart_budget": {
            "restart_count": restart_count,
            "reservation_ids": reservations,
            "budget": max(restart_count, 1),
        },
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


def test_missing_allocation_and_old_terminal(tmp_path: Path) -> None:
    assert (
        _run(
            "missing_allocation",
            tmp_path,
            lambda aid: attempt_doc(aid, decision="PLAN_CREATED"),
        )["verdict"]
        == "PASS"
    )
    assert (
        _run(
            "old_terminal",
            tmp_path,
            lambda aid: attempt_doc(aid, decision="NO_ACTION", recovery_ops=()),
        )["verdict"]
        == "PASS"
    )


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


def test_duplicate_event_and_single_notification(tmp_path: Path) -> None:
    assert (
        _run(
            "duplicate_event",
            tmp_path,
            lambda aid: attempt_doc(aid, restart_count=1, duplicate=True),
        )["verdict"]
        == "PASS"
    )
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
    assert SPECS["api_unreachable"].deferred.startswith("NOT_AUTHORIZED"), (
        "SPECS['api_unreachable'].deferred.startswith('NOT_AUTHORIZED')"
    )
    assert SPECS["worker_restart"].deferred.startswith("NOT_AUTHORIZED"), (
        "SPECS['worker_restart'].deferred.startswith('NOT_AUTHORIZED')"
    )
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
