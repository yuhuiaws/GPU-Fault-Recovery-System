"""The in-flight install probe: what it scans, what it reports, how it fails.

The probe (``deploy/control-plane/regional/probes/inflight_installs.py``) is
the one store read behind the release engine's in-flight install gate
(``tests/regional/test_release_inflight_install_gate.py`` pins the gate). It is
shipped to a control-plane Pod as source and runs against whatever
``gpu_fault`` is already deployed, so these tests exec its source the way the
Pod does and monkeypatch ``ApplicationContext.from_environment`` underneath.
Split out of the gate file in fix round 4 when that file crossed the
architecture cap.
"""

from __future__ import annotations

import builtins
import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.models import (
    BlockedKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault_release import regional_release_probes as PROBES

# --- the probe --------------------------------------------------------------------


def _workflow(
    request_id: str,
    status: WorkflowStatus,
    steps: list[tuple[WorkflowOperation, list[str]]],
    *,
    executions: list[tuple[int, WorkflowStepStatus]] = (),
    completed: list[int] = (),
    superseded: list[int] = (),
) -> WorkflowRequest:
    return WorkflowRequest(
        request_id=request_id,
        incident_id=f"incident-{request_id}",
        status=status,
        fencing_token=1,
        official_steps=[
            WorkflowStepSpec(
                operation=operation, execution_owner="regional", node_ids=node_ids
            )
            for operation, node_ids in steps
        ],
        step_executions=[
            WorkflowStepExecution(
                step_index=index, operation=steps[index][0], status=step_status
            )
            for index, step_status in executions
        ],
        completed_step_indexes=list(completed),
        superseded_step_indexes=list(superseded),
    )


def _run_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workflows: list[WorkflowRequest],
) -> dict[str, Any]:
    from gpu_fault.app import ApplicationContext

    asked: list[tuple[set[WorkflowStatus], int]] = []

    def list_workflows(statuses=None, *, limit=100, **_kwargs):
        asked.append((set(statuses or ()), limit))
        rows = [workflow for workflow in workflows if workflow.status in statuses]
        return rows[:limit]

    context = SimpleNamespace(store=SimpleNamespace(list_workflows=list_workflows))
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    exec(  # noqa: S102 - the probe is a program, this is how the Pod runs it
        compile(PROBES.probe_source("inflight_installs"), "inflight_installs", "exec"),
        {"__name__": "__probe__"},
    )
    result = json.loads(capsys.readouterr().out)
    assert len(asked) == 1, "one store read, not one per workflow"
    # Executable statuses only: BLOCKED is never archived and grows without
    # bound, and (pinned below) never holds an official install step mid-flight.
    assert asked[0][0] == {
        WorkflowStatus.PENDING,
        WorkflowStatus.SAFETY_PENDING,
        WorkflowStatus.RUNNING,
    }
    assert asked[0][1] == 1001, "one row past the 1000-row window proves overflow"
    return result


def test_the_probe_reports_pending_and_waiting_install_steps_with_their_nodes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pending = _workflow(
        "wf-pending",
        WorkflowStatus.PENDING,
        [(WorkflowOperation.REMEDIATE_DRIVER, ["node-a"])],
    )
    waiting = _workflow(
        "wf-waiting",
        WorkflowStatus.RUNNING,
        [
            (WorkflowOperation.QUIESCE_GPU_SERVICES, ["node-b"]),
            (WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE, ["node-b"]),
        ],
        executions=[(0, WorkflowStepStatus.SUCCEEDED), (1, WorkflowStepStatus.WAITING)],
        completed=[0],
    )

    result = _run_probe(monkeypatch, capsys, [pending, waiting])

    assert result["inflight_count"] == 2
    by_id = {item["workflow_id"]: item for item in result["inflight"]}
    assert by_id["wf-pending"]["operation"] == "REMEDIATE_DRIVER"
    assert by_id["wf-pending"]["node_ids"] == ["node-a"]
    assert by_id["wf-pending"]["step_status"] == "PENDING"
    assert by_id["wf-waiting"]["operation"] == "UPDATE_SOFTWARE_FIRMWARE"
    assert by_id["wf-waiting"]["node_ids"] == ["node-b"]
    assert by_id["wf-waiting"]["step_index"] == 1
    assert by_id["wf-waiting"]["step_status"] == "WAITING"


def test_the_probe_ignores_finished_failed_superseded_and_unrelated_steps(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    done = _workflow(
        "wf-done",
        WorkflowStatus.RUNNING,
        [
            (WorkflowOperation.REMEDIATE_EFA_DRIVER, ["node-c"]),
            (WorkflowOperation.RESTORE_GPU_SERVICES, ["node-c"]),
        ],
        executions=[(0, WorkflowStepStatus.SUCCEEDED)],
        completed=[0],
    )
    failed = _workflow(
        "wf-failed",
        WorkflowStatus.RUNNING,
        [(WorkflowOperation.REMEDIATE_DRIVER, ["node-d"])],
        executions=[(0, WorkflowStepStatus.FAILED)],
    )
    superseded = _workflow(
        "wf-superseded",
        WorkflowStatus.RUNNING,
        [(WorkflowOperation.REMEDIATE_DRIVER, ["node-e"])],
        superseded=[0],
    )
    unrelated = _workflow(
        "wf-reset", WorkflowStatus.RUNNING, [(WorkflowOperation.RESET_GPU, ["node-f"])]
    )
    terminal = _workflow(
        "wf-succeeded",
        WorkflowStatus.SUCCEEDED,
        [(WorkflowOperation.REMEDIATE_DRIVER, ["node-g"])],
    )

    result = _run_probe(
        monkeypatch, capsys, [done, failed, superseded, unrelated, terminal]
    )

    assert result == {
        "inflight": [],
        "inflight_count": 0,
        "bounded": True,
        "scanned": 4,
    }


def test_the_probe_reports_its_own_failure_as_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The HIGH-1 scenario of fix round 3: the Pod is Running and its warm
    connection pool keeps dispatching, but the probe's FRESH store connection
    fails (Aurora rotation window). That is not "the store did not answer"; it
    is the probe answering "I could not read". It must reach stdout as JSON so
    the gate can refuse on it instead of reading a bare exit 1 as transport."""

    from gpu_fault.app import ApplicationContext

    def from_environment(cls):
        raise ConnectionError("password authentication failed for user gpu_fault")

    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(from_environment)
    )

    with pytest.raises(SystemExit) as exit_status:
        exec(  # noqa: S102 - the probe is a program, this is how the Pod runs it
            compile(
                PROBES.probe_source("inflight_installs"), "inflight_installs", "exec"
            ),
            {"__name__": "__probe__"},
        )

    assert exit_status.value.code == 1, "the probe ran and failed: non-zero"
    reported = json.loads(capsys.readouterr().out)
    assert reported["probe_error"].startswith("ConnectionError: "), reported
    assert "password authentication failed" in reported["probe_error"]
    assert "inflight" not in reported, "a failed read must not look like evidence"


def test_the_probe_reports_an_import_failure_as_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rollback target whose deployed ``gpu_fault`` lacks a name the probe
    imports used to die at import time, before ``run()``'s catch-all existed
    to report it; the wrapper then showed a bare traceback and exit 1. The
    imports live inside ``main`` so an ImportError is ``probe_error`` evidence
    like any other failure (fix round 4, LOW-5)."""

    import sys

    monkeypatch.setitem(sys.modules, "gpu_fault.app", None)

    with pytest.raises(SystemExit) as exit_status:
        exec(  # noqa: S102 - the probe is a program, this is how the Pod runs it
            compile(
                PROBES.probe_source("inflight_installs"), "inflight_installs", "exec"
            ),
            {"__name__": "__probe__"},
        )

    assert exit_status.value.code == 1, "the probe ran and failed: non-zero"
    reported = json.loads(capsys.readouterr().out)
    kind, _, detail = reported["probe_error"].partition(": ")
    assert issubclass(getattr(builtins, kind), ImportError), reported
    assert "gpu_fault.app" in detail, reported


def test_blocked_workflows_are_not_scanned(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """BLOCKED is excluded because no DISPATCHED BLOCKED row is ever reopened,
    not because it cannot hold an install execution (it can, see the next
    test). ``executor.py`` returns the recorded result for SUCCEEDED / BLOCKED /
    SUPERSEDED without executing; the operator levers close a BLOCKED row to
    SUPERSEDED and refuse rows that were already dispatched. The one writer
    that does rewrite BLOCKED to PENDING -- ``node_lifecycle._state`` merging a
    second node fault into an aggregation window -- only reaches a row whose
    ``not_before`` is still in the future and that has no execution owner, i.e.
    a row the dispatcher has never claimed, so it can carry no install execution
    (pinned in ``tests/orchestration/test_replacement_merge_reopens_only_undis-
    patched.py``). So no control plane -- old or new -- will ever derive a
    command id for a dispatched step in a BLOCKED row again, and the
    double-submit this gate exists for cannot start there. The table is also
    never archived, which is what let the newest RUNNING row fall past the
    window when BLOCKED was scanned (F1)."""

    blocked = _workflow(
        "wf-blocked-waiting",
        WorkflowStatus.BLOCKED,
        [(WorkflowOperation.REMEDIATE_DRIVER, ["node-i"])],
        executions=[(0, WorkflowStepStatus.WAITING)],
    )

    result = _run_probe(monkeypatch, capsys, [blocked])

    assert result["inflight"] == []
    assert result["bounded"] is True


def test_a_blocked_internal_error_row_with_a_waiting_install_is_not_counted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A RUNNING workflow CAN end up BLOCKED with a WAITING install execution:
    ``dispatcher._block_after_internal_error`` blocks a mid-dispatch
    ValidationError as ``BlockedKind.INTERNAL_ERROR`` and ``terminalize_claimed``
    leaves ``step_executions`` intact (``workflow_resolution`` and the postgres
    open-predecessor SQL both know this shape). The node agent may still be
    installing -- but no control plane will ever re-derive that step's command
    id (a BLOCKED row is never re-executed), so the old/new command-id shape
    cannot produce a second submit for it. Not in flight for this gate."""

    blocked = _workflow(
        "wf-internal-error",
        WorkflowStatus.BLOCKED,
        [
            (WorkflowOperation.QUIESCE_GPU_SERVICES, ["node-m"]),
            (WorkflowOperation.REMEDIATE_DRIVER, ["node-m"]),
        ],
        executions=[(0, WorkflowStepStatus.SUCCEEDED), (1, WorkflowStepStatus.WAITING)],
        completed=[0],
    ).model_copy(
        update={
            "blocked_kind": BlockedKind.INTERNAL_ERROR,
            "blocked_reasons": ["dispatcher internal error: ValidationError"],
        }
    )
    running_sibling = _workflow(
        "wf-running",
        WorkflowStatus.RUNNING,
        [(WorkflowOperation.REMEDIATE_DRIVER, ["node-n"])],
        executions=[(0, WorkflowStepStatus.WAITING)],
    )

    result = _run_probe(monkeypatch, capsys, [blocked, running_sibling])

    assert [item["workflow_id"] for item in result["inflight"]] == ["wf-running"]


def test_a_transiently_succeeded_execution_is_not_in_flight(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Between the agent's success and ``completed_step_indexes`` there is a
    window where the latest execution says SUCCEEDED; the node is done, so the
    step is not in flight (F5)."""

    finished = _workflow(
        "wf-finished",
        WorkflowStatus.RUNNING,
        [(WorkflowOperation.REMEDIATE_DRIVER, ["node-j"])],
        executions=[(0, WorkflowStepStatus.SUCCEEDED)],
    )

    result = _run_probe(monkeypatch, capsys, [finished])

    assert result == {
        "inflight": [],
        "inflight_count": 0,
        "bounded": True,
        "scanned": 1,
    }


def test_more_rows_than_the_window_is_reported_as_unbounded_not_as_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``list_workflows`` is oldest-first; with more executable rows than the
    window the newest RUNNING install would fall off the end and the probe used
    to print 0 -- failing open. Now it says it could not bound the scan."""

    old = [
        _workflow(
            f"wf-old-{index}",
            WorkflowStatus.PENDING,
            [(WorkflowOperation.RESET_GPU, ["node-k"])],
        )
        for index in range(1001)
    ]
    newest = _workflow(
        "wf-newest-install",
        WorkflowStatus.RUNNING,
        [(WorkflowOperation.REMEDIATE_DRIVER, ["node-z"])],
        executions=[(0, WorkflowStepStatus.WAITING)],
    )

    result = _run_probe(monkeypatch, capsys, [*old, newest])

    assert result["bounded"] is False
    assert result["scanned"] == 1001
    assert result["inflight_count"] == 0, "the install fell past the window"
