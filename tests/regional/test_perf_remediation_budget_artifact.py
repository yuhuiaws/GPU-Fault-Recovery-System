"""``remediation-budget.json`` of an action capacity run (性能压测验收方案 §11).

§11 required the document -- five budget tiers, peak active claims per scope,
waits, longest wait, workflows that never ran, lease-expiry releases -- but no
runner wrote it. The action runner now writes it from what it already holds:
``run.json.remediation_budget``, the occupancy verdict over ``history.json`` and
the run's final workflow rows; a field the release does not record is
``unsupported``, and lease-expiry releases are ``not-applicable`` because a
claim is held until its workflow ends.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.perf import regional_action_capacity_suite as action
from scripts.perf import regional_remediation_budget as budget
from tests.regional._perf_caller_support import CapacityWire, action_args

T0 = datetime(2026, 10, 14, 9, 0, tzinfo=timezone.utc)
RAW_LIMITS = {
    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": 128,
    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER": 4,
    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS": 4,
    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE": 1,
    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN": 1,
}


def workflow(
    index: int, *, refused: int | None, updated: float, claims: bool = True
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "created_at": T0.isoformat(),
        "updated_at": (T0 + timedelta(seconds=updated)).isoformat(),
    }
    if refused is not None:
        payload["remediation_budget_wait_count"] = refused
    if claims:
        payload["remediation_budget_claims"] = []
    return f"workflow-actionperf-run-a-c000-w{index:03d}", payload


class Cursor:
    def __init__(self, rows: list[list[Any]]) -> None:
        self.rows = rows
        self.result: list[Any] = []

    def execute(self, _query: str, _params: Any = None) -> None:
        self.result = self.rows.pop(0)

    def fetchall(self) -> list[Any]:
        return self.result


class Connection:
    def __init__(self, rows: list[list[Any]]) -> None:
        self.rows = rows

    def __enter__(self) -> Connection:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def cursor(self) -> Cursor:
        return Cursor(self.rows)


def run_details_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workflows: list[tuple[str, dict[str, Any]]],
    first_commands: dict[str, datetime],
) -> dict[str, Any]:
    """``database_details``' in-Pod program, run here against canned rows."""

    scripts: list[str] = []

    def control(*args: str, **_kwargs: Any) -> str:
        if args[0] == "get":
            return "api"
        scripts.append(args[args.index("-c") + 1])
        return "{}"

    monkeypatch.setattr(action, "control", control)
    action.database_details("run-a")
    rows = [workflows, list(first_commands.items())]
    psycopg = SimpleNamespace(connect=lambda _dsn: Connection(rows))
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    dsn = tmp_path / "postgres-url"
    dsn.write_text("postgresql://unused", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(dsn))
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(scripts[-1], "database_details", "exec"), {"__name__": "x"})
    result: dict[str, Any] = json.loads(output.getvalue().splitlines()[-1])
    return result


def test_the_details_program_reads_the_waits_from_the_runs_workflow_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [
        workflow(0, refused=0, updated=300),
        workflow(1, refused=3, updated=900),
        workflow(2, refused=2, updated=50),
    ]
    first = {
        rows[0][0]: T0 + timedelta(seconds=5),
        rows[1][0]: T0 + timedelta(seconds=400),
    }

    details = run_details_script(monkeypatch, tmp_path, rows, first)

    assert details["remediation_budget_waits"] == {
        "recorded": True,
        "claims_recorded": True,
        "refused_claims": 5,
        "waited_workflows": 2,
        "longest_wait_seconds": 400.0,
        "never_executed_workflows": 1,
    }, "refusals are the product's counts; the wait runs to the first command"
    assert details["first_action_max_seconds"] == 400.0, (
        "the existing percentiles are unchanged"
    )


def test_a_release_without_budget_fields_is_unsupported_not_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [workflow(0, refused=None, updated=30, claims=False)]
    details = run_details_script(
        monkeypatch, tmp_path, rows, {rows[0][0]: T0 + timedelta(seconds=2)}
    )

    document = budget.remediation_budget_artifact(
        run_id="run-a",
        tier={"tier": "unbounded", "limits": dict.fromkeys(RAW_LIMITS)},
        occupancy={"passed": True, "samples": 3, "peak_holders": {}},
        waits=details["remediation_budget_waits"],
    )

    assert document["peak_active_claims"] == budget.UNSUPPORTED
    assert document["waits"] == dict.fromkeys(budget.WAIT_FIELDS, budget.UNSUPPORTED)
    assert document["never_executed_workflows"] == 0, (
        "the first-command count is known on every release"
    )
    assert document["limits"] == dict.fromkeys(RAW_LIMITS), (
        "unset tiers are recorded as unset"
    )


def test_an_artifact_without_its_inputs_says_so() -> None:
    document = budget.remediation_budget_artifact(
        run_id="run-a", tier={}, occupancy={}, waits=None
    )

    assert document["tier"] == document["limits"] == budget.UNSUPPORTED
    assert document["peak_active_claims"] == budget.UNSUPPORTED
    assert document["never_executed_workflows"] == budget.UNSUPPORTED
    assert document["lease_expiry_releases"] == budget.NOT_APPLICABLE, (
        "a claim is held until its workflow ends: a lapsed lease releases nothing"
    )


def test_the_action_run_writes_the_budget_evidence_beside_its_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    CapacityWire(tmp_path, monkeypatch)
    waits = {
        "recorded": True,
        "claims_recorded": True,
        "refused_claims": 4,
        "waited_workflows": 1,
        "longest_wait_seconds": 61.5,
        "never_executed_workflows": 0,
    }
    monkeypatch.setattr(
        action, "database_details", lambda _run_id: {"remediation_budget_waits": waits}
    )

    assert action.main(action_args(tmp_path)) == 0, "the fake run completes"

    document = json.loads((tmp_path / budget.ARTIFACT).read_text())
    run = json.loads((tmp_path / "run.json").read_text())
    assert document["limits"] == run["remediation_budget"]["limits"] == RAW_LIMITS, (
        "the five tiers are the ones the run recorded"
    )
    assert document["tier"] == "raw-capacity"
    assert document["peak_active_claims"] == {
        "class": 1,
        "cluster": 1,
        "node": 1,
        "region": 1,
    }, "the per-scope peaks are the occupancy verdict's"
    assert document["occupancy"]["passed"] is True
    assert document["waits"] == {
        "refused_claims": 4,
        "waited_workflows": 1,
        "longest_wait_seconds": 61.5,
    }
    assert document["never_executed_workflows"] == 0
    assert document["lease_expiry_releases"] == budget.NOT_APPLICABLE
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["remediation_budget_waits"] == waits, (
        "summary.json carries the same wait side"
    )
