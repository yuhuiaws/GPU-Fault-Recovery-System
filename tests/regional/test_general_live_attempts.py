"""GF-GENERAL-LIVE-028/029 injector/verdict contract over scripted records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_attempts as att
from tests.regional._general_live_fakes import (
    FakeRegional,
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
        mode: str,
        restart_budget: int = 1,
        gpu_count: int = 8,
    ) -> None:
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.mode = mode
        self.restart_budget = restart_budget

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

    ctx.regional._cpu_python = responder
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
    assert SPECS["api_unreachable"].deferred.startswith("NOT_AUTHORIZED")
    assert SPECS["worker_restart"].deferred.startswith("NOT_AUTHORIZED")
    assert SPECS["hardware_chain_continues"].deferred.startswith("FIXTURE_NOT_READY")
    assert SPECS["concurrent_claim"].deferred.startswith("FIXTURE_NOT_READY")
    assert SPECS["missing_context"].deferred.startswith("FIXTURE_NOT_READY")
