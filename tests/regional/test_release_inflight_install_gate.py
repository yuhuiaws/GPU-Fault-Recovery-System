"""A release transaction refuses to start while a node install is in flight.

R4 (2026-09-09) changed the node-action command id of REMEDIATE_DRIVER /
UPDATE_SOFTWARE_FIRMWARE / REMEDIATE_EFA_DRIVER from ``<key>/<node>/agent-N``
to ``<key>/<node>``. The new control plane reads the old rows back through a
one-release shim, but a rollback to the previous release runs old code that
derives ``agent-N`` again, finds no ledger row under that id, and submits a
SECOND install on top of the one the node agent is still running (install
timeout 1800 s). The same holds for an upgrade that lands while the old control
plane has such a step WAITING.

The ruling is a release-tooling precondition, pinned here: before ``deploy``
opens a transaction (upgrade, resume, supersede or rollback) and before the
engine's automatic rollback, one control-plane store read lists the workflows
whose install steps are PENDING or WAITING; any hit refuses with the workflow
ids and nodes named, unless the operator passed ``--allow-inflight-installs``.
A store that cannot answer refuses too (fail closed), naming the error, unless
no Running control-plane Pod could run the probe at all and the caller is the
automatic rollback or carries the consent. A refused automatic rollback logs why
and leaves the transaction in ``failed``, the phase the resume / supersede
levers already handle.
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault.models import (
    BlockedKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.execution.config import NODE_INSTALL_OPERATIONS
from gpu_fault.operation_registry import GENERATION_STABLE_COMMAND_OPERATIONS
from gpu_fault_release import regional_deployment_inventory as INVENTORY
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_store_preflight as GATE
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_probes as PROBES
from gpu_fault_release import rollout as MODULE

ROOT = Path(__file__).resolve().parents[2]
ENV = "GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS"
FLAG = "--allow-inflight-installs"
NAMESPACE = "gpu-fault-system"
# The line the shell wrapper prints after the probe, whatever the probe did.
MARKER = "__GPU_FAULT_PROBE_EXIT"


# --- the operation set and the spellings ---------------------------------------


def test_the_gate_covers_exactly_the_generation_stable_operations() -> None:
    """The three operations R4 moved off the agent-generation suffix are the
    three whose in-flight rows an old control plane would re-submit -- and the
    engine reads them from the executor's install set, not a private copy."""

    assert set(GATE.INSTALL_OPERATIONS) == {
        operation.value for operation in GENERATION_STABLE_COMMAND_OPERATIONS
    }
    assert GATE.INSTALL_OPERATIONS == tuple(
        operation.value for operation in NODE_INSTALL_OPERATIONS
    )
    assert set(GATE.INSTALL_OPERATIONS) == {
        "REMEDIATE_DRIVER",
        "UPDATE_SOFTWARE_FIRMWARE",
        "REMEDIATE_EFA_DRIVER",
    }


def test_the_probe_inlines_the_operation_set_instead_of_importing_it() -> None:
    """The probe runs against whatever ``gpu_fault`` is already deployed. For the
    release that first carries R4 -- and for every rollback target -- that is a
    module without ``GENERATION_STABLE_COMMAND_OPERATIONS``."""

    source = PROBES.probe_source("inflight_installs")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    assert "gpu_fault.operation_registry" not in imported, imported
    for name in GATE.INSTALL_OPERATIONS:
        assert f"WorkflowOperation.{name}" in source


def test_the_cli_and_the_engine_spell_the_variable_and_the_flag_the_same() -> None:
    assert admin_cli.ALLOW_INFLIGHT_INSTALLS_ENV == ENV
    assert GATE.ALLOW_INFLIGHT_INSTALLS_ENV == ENV
    assert GATE.ALLOW_INFLIGHT_INSTALLS_FLAG == FLAG


def test_the_release_object_binds_the_gate_like_the_other_preflights() -> None:
    assert (
        MODULE.RegionalRelease._require_no_inflight_installs
        is GATE.require_no_inflight_installs
    )


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


# --- the gate ---------------------------------------------------------------------


class Runner:
    """Every control-plane role has one Running Pod (``cpu-pod``); every exec
    answers the same way: a string is what ``kubectl exec`` printed, an
    exception is what ``Runner.run`` raised (kubectl exited non-zero, or timed
    out)."""

    dry_run = False

    def __init__(self, result: str | Exception) -> None:
        self.result = result
        self.execs: list[list[str]] = []
        self.exec_kwargs: list[dict[str, Any]] = []

    def run(self, args, **kwargs):
        if "get" in args and "pod" in args:
            return "cpu-pod"
        self.execs.append(list(args))
        self.exec_kwargs.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def probe(self, _args, **_kwargs) -> bool:
        return True


def _release(result: str | Exception) -> SimpleNamespace:
    return SimpleNamespace(
        runner=Runner(result),
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *args: ["kubectl", *args],
    )


def _probe_output(*lines: str, code: int = 0) -> str:
    """What the shell wrapper hands back: whatever the probe printed, then the
    exit marker. The marker is the proof that the probe process ran at all."""

    return "\n".join([*lines, f"{MARKER}={code}"])


def _snapshot(*items: dict[str, Any], bounded: bool = True) -> str:
    return _probe_output(
        json.dumps(
            {
                "inflight": list(items),
                "inflight_count": len(items),
                "bounded": bounded,
                "scanned": len(items),
            }
        )
    )


def _timed_out() -> MODULE.ReleaseError:
    """What ``Runner.run`` raises when ``timeout_seconds`` expires."""

    error = MODULE.ReleaseError("command timed out after 120s: kubectl")
    error.__cause__ = subprocess.TimeoutExpired(cmd="kubectl", timeout=120)
    return error


class RoleRunner:
    """Resolves one Running Pod per control-plane Deployment and answers an exec
    per Pod: a string is stdout, an exception is what ``kubectl exec`` raised."""

    dry_run = False

    def __init__(self, pods: dict[str, str], answers: dict[str, str | Exception]):
        self.pods = pods
        self.answers = answers
        self.execs: list[str] = []

    def run(self, args, **_kwargs):
        if "get" in args and "pod" in args:
            label = args[args.index("-l") + 1].removeprefix("app=")
            return self.pods.get(label, "")
        pod = next(
            argument
            for argument in args[args.index("exec") + 1 :]
            if not argument.startswith("-")
        )
        self.execs.append(pod)
        answer = self.answers[pod]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def probe(self, _args, **_kwargs) -> bool:
        return True


def _role_release(runner: RoleRunner) -> SimpleNamespace:
    return SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *args: ["kubectl", *args],
    )


def _no_pod_release() -> SimpleNamespace:
    """No control-plane role has a Running Pod: the probe provably never ran."""

    return _role_release(RoleRunner(pods={}, answers={}))


DRIVER_STEP = {
    "workflow_id": "workflow-7f3a",
    "incident_id": "incident-1",
    "workflow_status": "RUNNING",
    "operation": "REMEDIATE_DRIVER",
    "step_index": 3,
    "node_ids": ["ip-10-0-1-17"],
    "step_status": "WAITING",
}


def test_a_pending_driver_install_refuses_the_transaction_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    release = _release(_snapshot(DRIVER_STEP))

    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(release, action="upgrade")

    message = str(failure.value)
    assert isinstance(failure.value, MODULE.ReleaseError), (
        "the engine's main() only reports ReleaseError cleanly"
    )
    assert "upgrade" in message
    assert "workflow-7f3a" in message
    assert "ip-10-0-1-17" in message
    assert "REMEDIATE_DRIVER" in message
    assert FLAG in message
    assert len(release.runner.execs) == 1, "one store read"


def test_the_override_lets_the_transaction_proceed_and_says_what_it_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV, "1")
    release = _release(_snapshot(DRIVER_STEP))

    result = GATE.require_no_inflight_installs(release, action="rollback")

    assert result["inflight_count"] == 1
    assert result["verdict"] == "overridden"
    assert result["checked"] is True
    assert result["reason"] == FLAG
    assert len(result["steps"]) == 1 and "workflow-7f3a" in result["steps"][0], (
        "the verdict carries the listed set the consent covered"
    )
    logged = capsys.readouterr().err
    assert "workflow-7f3a" in logged
    assert "ip-10-0-1-17" in logged
    assert FLAG in logged


def test_no_in_flight_install_lets_the_transaction_proceed_quietly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(ENV, raising=False)

    result = GATE.require_no_inflight_installs(_release(_snapshot()), action="upgrade")

    assert result == {
        "checked": True,
        "verdict": "clear",
        "reason": None,
        "steps": [],
        "inflight_count": 0,
        "scanned": 0,
    }
    assert capsys.readouterr().err == ""


def test_the_probe_runs_through_a_shell_wrapper_that_marks_its_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Runner.run`` raises the same ``command failed (1): kubectl`` whether
    kubectl never reached the Pod or the probe launched and died on line one.
    So the probe is not the exec'd command: a ``sh`` wrapper is, it feeds the
    probe to ``python -`` on stdin, merges stderr into the captured output so
    the cause survives, and prints the exit marker last whatever happened, so
    the wrapper itself always exits 0. The marker's presence is the proof that
    the process ran; kubectl failing before it = transport. The read is not
    ``sensitive`` (its output is workflow ids the refusal prints anyway) and it
    has a timeout: a hung store read must refuse, not hang the engine."""

    monkeypatch.delenv(ENV, raising=False)
    release = _release(_snapshot())

    GATE.require_no_inflight_installs(release, action="upgrade")

    (args,) = release.runner.execs
    (kwargs,) = release.runner.exec_kwargs
    assert "-i" in args, "the probe travels on stdin"
    assert args[-3:-1] == ["sh", "-c"], args
    wrapper = args[-1]
    assert "python -" in wrapper and f'echo "{MARKER}=$?"' in wrapper, wrapper
    assert "2>&1" in wrapper, "the probe's own traceback must reach the log"
    assert kwargs["input_text"] == PROBES.probe_source("inflight_installs")
    assert kwargs.get("sensitive", False) is False, "the cause must survive"
    assert kwargs["timeout_seconds"] == GATE.PROBE_TIMEOUT_SECONDS == 120
    assert kwargs["capture"] is True, "the answer is read, not streamed"


def test_an_unreachable_store_refuses_and_names_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    release = _release(MODULE.ReleaseError("command failed (1): kubectl exec"))

    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(release, action="upgrade")

    message = str(failure.value)
    assert "command failed (1): kubectl exec" in message
    assert FLAG in message


def test_unreadable_evidence_refuses_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV, raising=False)

    for garbage in (
        "not json",  # no marker: kubectl exited 0 without the wrapper's line
        _probe_output("not json"),
        _probe_output("[]"),
        _probe_output('{"inflight": "x"}'),
    ):
        with pytest.raises(GATE.InflightInstallsRefused, match="evidence"):
            GATE.require_no_inflight_installs(_release(garbage), action="upgrade")


def test_the_override_also_covers_a_store_that_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rollback of a control plane that is itself down must stay possible."""

    monkeypatch.setenv(ENV, "1")

    result = GATE.require_no_inflight_installs(_no_pod_release(), action="rollback")

    assert result["verdict"] == "unchecked"
    assert result["checked"] is False
    assert result["reason"] == FLAG
    assert "could not reach any Running control-plane Pod" in capsys.readouterr().err


def test_consent_does_not_cover_an_evidence_defect(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--allow-inflight-installs`` is consent to proceed over a LISTED
    in-flight set, or over a store no Pod could answer for. It is not consent
    to proceed over a check that ran and could not vouch for anything: an
    unbounded scan, no JSON, the wrong shape, a probe that reported its own
    error, or a probe that hung. Those refuse in every mode and the message
    points at the cause, not at the flag (fix round 3, MEDIUM-4)."""

    monkeypatch.setenv(ENV, "1")
    cases: list[tuple[str | Exception, str]] = [
        (_snapshot(bounded=False), "could not bound"),
        (_probe_output("Traceback (most recent call last):", code=1), "exited 1"),
        (_probe_output("", code=0), "evidence"),
        (_probe_output('{"inflight": 1}'), "evidence"),
        (
            _probe_output(
                json.dumps({"probe_error": "OperationalError: connection refused"}),
                code=1,
            ),
            "OperationalError: connection refused",
        ),
        (_timed_out(), "did not answer within"),
    ]
    for evidence, cause in cases:
        for unreadable in ("refuse", "proceed"):
            with pytest.raises(GATE.InflightInstallsRefused) as failure:
                GATE.require_no_inflight_installs(
                    _release(evidence), action="rollback", unreadable=unreadable
                )
            message = str(failure.value)
            assert cause in message, (evidence, message)
            assert f"{FLAG} does not cover" in message, message
            assert "rerun" not in message.split("does not cover")[0], (
                "the lever is not offered as the way out"
            )
    assert capsys.readouterr().err == "", "an evidence defect is refused, not logged"


def test_the_lever_named_matches_the_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``deploy`` has the flag; ``rollout-regional-release.sh rollback`` has no
    flag at all, only the variable (F3)."""

    monkeypatch.delenv(ENV, raising=False)

    with pytest.raises(GATE.InflightInstallsRefused) as upgrade:
        GATE.require_no_inflight_installs(
            _release(_snapshot(DRIVER_STEP)), action="upgrade"
        )
    assert "gpu-fault-admin deploy" in str(upgrade.value)
    assert FLAG in str(upgrade.value)
    assert f"{ENV}=1" not in str(upgrade.value)

    with pytest.raises(GATE.InflightInstallsRefused) as rollback:
        GATE.require_no_inflight_installs(
            _release(_snapshot(DRIVER_STEP)), action="rollback"
        )
    assert f"{ENV}=1" in str(rollback.value)
    assert f"rerun with {FLAG}" not in str(rollback.value)

    with pytest.raises(GATE.InflightInstallsRefused) as unreachable:
        GATE.require_no_inflight_installs(
            _release(MODULE.ReleaseError("exec failed")), action="rollback"
        )
    assert f"{ENV}=1" in str(unreachable.value)


def test_an_unbounded_scan_refuses_instead_of_reading_zero_as_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    release = _release(_snapshot(bounded=False))

    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(release, action="upgrade")

    message = str(failure.value)
    assert "could not bound" in message
    assert "1000" in message
    assert f"{FLAG} does not cover" in message, "consent is not the way out"
    assert "drain" in message, "the message points at the cause"


def test_the_automatic_rollback_proceeds_only_when_no_pod_could_run_the_probe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The primary automatic-rollback scenario is a control plane that is down
    (CrashLoop, Aurora window, broken wheel); with no Running Pod in any role
    nothing dispatches, and wedging production in ``failed`` is the worse
    failure (F2). But a Running Pod whose probe ran and exited without evidence
    is the opposite case: the old dispatcher's warm pool may still be
    dispatching while the probe's fresh connection failed, so the automatic
    rollback REFUSES (fix round 3, HIGH-1). An actual in-flight install still
    refuses."""

    monkeypatch.delenv(ENV, raising=False)

    result = GATE.require_no_inflight_installs(
        _no_pod_release(), action="rollback", unreadable="proceed"
    )

    assert result["verdict"] == "unchecked"
    assert result["checked"] is False
    assert "no Running control-plane Pod" in result["reason"]
    logged = capsys.readouterr().err
    assert "inflight-installs-unchecked" in logged
    assert "could not reach any Running control-plane Pod" in logged
    assert "automatic" in logged

    died = _release(_probe_output("ImportError: cannot import name 'x'", code=1))
    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(died, action="rollback", unreadable="proceed")
    assert "exited 1" in str(failure.value)
    assert "ImportError: cannot import name 'x'" in str(failure.value), (
        "the cause the wrapper captured reaches the refusal"
    )

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        GATE.require_no_inflight_installs(
            _release(_snapshot(DRIVER_STEP)), action="rollback", unreadable="proceed"
        )


def test_only_a_probe_that_could_not_run_counts_as_unreadable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Classified by WHERE the failure happened. Once the wrapper's marker is
    on stdout the probe process ran, and everything after that is evidence,
    binding in every mode: an overflowed scan, no JSON at all, the wrong shape,
    the probe's own ``probe_error``, or a hang past the timeout (the exec did
    start; the store did not answer in time). Only "no Running Pod in any role"
    is unreachable."""

    monkeypatch.delenv(ENV, raising=False)
    evidence_defects: list[str | Exception] = [
        _snapshot(bounded=False),
        "not json",
        _probe_output("not json"),
        _probe_output("[]"),
        _probe_output('{"inflight": 1}'),
        _probe_output("", code=137),
        _probe_output(json.dumps({"probe_error": "OperationalError: timeout"})),
        _timed_out(),
    ]
    for evidence in evidence_defects:
        with pytest.raises(GATE.InflightInstallsRefused) as failure:
            GATE.require_no_inflight_installs(
                _release(evidence), action="rollback", unreadable="proceed"
            )
        assert not isinstance(failure.value.__cause__, GATE.StoreUnreachable), (
            evidence,
            str(failure.value),
        )
    assert capsys.readouterr().err == "", "an evidence defect is refused, not logged"

    result = GATE.require_no_inflight_installs(
        _no_pod_release(), action="rollback", unreadable="proceed"
    )
    assert result["verdict"] == "unchecked"
    assert "inflight-installs-unchecked" in capsys.readouterr().err


def test_a_probe_that_reports_its_own_error_refuses_even_the_automatic_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact HIGH-1 shape: Running Pod, probe ran, fresh store connection
    failed. The refusal names the probe's error so the operator sees the
    rotation window, not a bare exit code."""

    monkeypatch.delenv(ENV, raising=False)
    release = _release(
        _probe_output(
            json.dumps(
                {"probe_error": "OperationalError: password authentication failed"}
            ),
            code=1,
        )
    )

    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(
            release, action="rollback", unreadable="proceed"
        )

    message = str(failure.value)
    assert "OperationalError: password authentication failed" in message
    assert "probe reported" in message
    assert len(release.runner.execs) == 1, "an answer from one Pod is final"


def test_a_hung_probe_refuses_instead_of_hanging_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store read that does not return inside the timeout is an answer of
    sorts -- the store is not healthy -- and not "no Pod could run it": the exec
    started. Refused in every mode; the other roles are not tried (they read
    the same store, and each would cost the whole window again)."""

    monkeypatch.delenv(ENV, raising=False)
    release = _release(_timed_out())

    for unreadable in ("refuse", "proceed"):
        with pytest.raises(GATE.InflightInstallsRefused) as failure:
            GATE.require_no_inflight_installs(
                release, action="rollback", unreadable=unreadable
            )
        assert "did not answer within 120" in str(failure.value)
        assert not isinstance(failure.value.__cause__, GATE.StoreUnreachable), (
            "a timeout is an evidence defect, not an unreachable store"
        )
    assert len(release.runner.execs) == 2, "one exec per call, no role fallback"


def test_a_kubectl_failure_before_the_wrapper_answered_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The wrapper always exits 0, so a non-zero ``kubectl exec`` with no marker
    on stdout means kubectl never got the wrapper to run: the API server was
    unreachable, or the Pod is phase Running with its container in
    CrashLoopBackOff (which is what a broken control-plane wheel looks like).
    With no marker there is no evidence the probe started, so this stays
    ``StoreUnreachable`` (the automatic rollback proceeds, logging), and it is
    tried on every Running Pod of every role first."""

    monkeypatch.delenv(ENV, raising=False)
    release = _release(MODULE.ReleaseError("command failed (1): kubectl"))

    result = GATE.require_no_inflight_installs(
        release, action="rollback", unreadable="proceed"
    )

    assert result["verdict"] == "unchecked"
    assert len(release.runner.execs) == len(INVENTORY.CPU_RUNTIME_DEPLOYMENTS), (
        "every role's Running Pod was tried before giving up"
    )
    assert "command failed (1): kubectl" in capsys.readouterr().err

    with pytest.raises(GATE.InflightInstallsRefused, match="could not read"):
        GATE.require_no_inflight_installs(release, action="rollback")


def test_every_running_pod_of_a_role_is_tried_not_only_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck rollout leaves the old ReplicaSet's Pod Running beside the new
    CrashLooping one; ``items[0]`` may be the crashing Pod. The old Pod runs the
    same store read and is the one still dispatching."""

    monkeypatch.delenv(ENV, raising=False)
    ingress = INVENTORY.CPU_INGRESS_DEPLOYMENT
    runner = RoleRunner(
        pods={ingress: "ingress-new\ningress-old"},
        answers={
            "ingress-new": MODULE.ReleaseError("command failed (1): kubectl"),
            "ingress-old": _snapshot(DRIVER_STEP),
        },
    )

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        GATE.require_no_inflight_installs(_role_release(runner), action="rollback")

    assert runner.execs == ["ingress-new", "ingress-old"]


def test_the_two_failure_kinds_are_distinct_exception_types() -> None:
    assert issubclass(GATE.StoreUnreachable, GATE.ReleaseError), (
        "main() reports every refusal as a ReleaseError"
    )
    assert not issubclass(GATE.StoreUnreachable, GATE.InflightInstallsRefused), (
        "unreachable is a cause the gate decides on, not a refusal by itself"
    )
    with pytest.raises(GATE.StoreUnreachable):
        GATE.inflight_install_snapshot(_no_pod_release())
    with pytest.raises(GATE.ReleaseError) as defect:
        GATE.inflight_install_snapshot(_release(_snapshot(bounded=False)))
    assert not isinstance(defect.value, GATE.StoreUnreachable), (
        "an overflowed scan is an answer, not an unreachable store"
    )


def test_the_read_falls_back_to_another_running_control_plane_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ingress Pod is the one the failed upgrade just rolled; any Running
    control-plane role can answer the same store read (F2)."""

    monkeypatch.delenv(ENV, raising=False)
    ingress = INVENTORY.CPU_INGRESS_DEPLOYMENT
    other = next(name for name in INVENTORY.CPU_RUNTIME_DEPLOYMENTS if name != ingress)
    runner = RoleRunner(
        pods={ingress: "ingress-pod", other: "worker-pod"},
        answers={
            "ingress-pod": MODULE.ReleaseError("command failed (137): kubectl"),
            "worker-pod": _snapshot(DRIVER_STEP),
        },
    )

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        GATE.require_no_inflight_installs(_role_release(runner), action="rollback")

    assert runner.execs[0] == "ingress-pod"
    assert "worker-pod" in runner.execs
    assert runner.execs.count("ingress-pod") <= 2, (
        "one retry at most, then the next role"
    )


def test_when_no_role_answers_the_error_names_every_role_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV, raising=False)

    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(_no_pod_release(), action="upgrade")

    message = str(failure.value)
    for name in INVENTORY.CPU_RUNTIME_DEPLOYMENTS:
        assert name in message


# --- where the orchestrator calls it -------------------------------------------


class _Reached(RuntimeError):
    pass


def _upgrade_double(calls: list[str], **stubs: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "config": SimpleNamespace(
            auto_rollback=True, schema_rollback_compatible=True, clusters=()
        ),
        "state": {},
        "_ensure_contexts": lambda: None,
        "_require_cpu_secrets": lambda: None,
        "_apply_rds_ca_bundle": lambda: None,
        "_refresh_aurora_credentials": lambda: None,
        "_remote_commands_are_idle": lambda: (calls.append("idle"), True)[1],
        "_require_no_inflight_installs": lambda **kwargs: calls.append(
            f"inflight-installs:{kwargs['action']}"
        ),
    }
    fields.update(stubs)
    return SimpleNamespace(**fields)


def test_upgrade_checks_for_in_flight_installs_before_capturing_previous() -> None:
    calls: list[str] = []
    release = _upgrade_double(
        calls, _capture_previous=lambda **_kwargs: (_ for _ in ()).throw(_Reached())
    )
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    with pytest.raises(_Reached):
        ORCHESTRATION.upgrade_release(release, diff=diff)

    assert calls == ["idle", "inflight-installs:upgrade"]


def test_a_refused_upgrade_writes_no_state(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    release = _upgrade_double(
        calls,
        _require_no_inflight_installs=lambda **_kwargs: (_ for _ in ()).throw(
            GATE.InflightInstallsRefused("upgrade refused: workflow-7f3a")
        ),
        _capture_previous=lambda **_kwargs: pytest.fail("previous was captured"),
    )
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        ORCHESTRATION.upgrade_release(release, diff=diff)
    assert release.state == {}


def _rollback_double(calls: list[str], **stubs: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "config": SimpleNamespace(auto_rollback=True, clusters=()),
        "state": {},
        "_refresh_aurora_credentials": lambda: calls.append("aurora-refresh"),
        "_require_no_inflight_installs": lambda **kwargs: calls.append(
            f"inflight-installs:{kwargs['action']}:{kwargs['unreadable']}"
        ),
        "_save_state": lambda phase, **_updates: calls.append(f"state:{phase}"),
    }
    fields.update(stubs)
    return SimpleNamespace(**fields)


def _plan_reached(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    def plan(*_args, **_kwargs):
        calls.append("compensation-plan")
        raise _Reached()

    monkeypatch.setattr(ORCHESTRATION, "build_rollback_compensation_plan", plan)


def test_rollback_checks_for_in_flight_installs_after_the_credential_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the Aurora refresh -- a rotation-window failure gets its chance to
    be named first -- and before any restore is planned. A manual rollback
    fails closed on a store that cannot answer."""

    calls: list[str] = []
    _plan_reached(monkeypatch, calls)

    with pytest.raises(_Reached):
        ORCHESTRATION.rollback_release(
            _rollback_double(calls), state={"metadata": {}, "cpu_wheel": "w"}
        )

    assert calls == [
        "aurora-refresh",
        "inflight-installs:rollback:refuse",
        "compensation-plan",
    ]


def test_the_automatic_rollback_tells_the_gate_to_proceed_on_an_unreadable_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _plan_reached(monkeypatch, calls)

    with pytest.raises(_Reached):
        ORCHESTRATION.rollback_release(
            _rollback_double(calls),
            state={"metadata": {}, "cpu_wheel": "w"},
            automatic=True,
        )

    assert calls == [
        "aurora-refresh",
        "inflight-installs:rollback:proceed",
        "compensation-plan",
    ]


def test_the_rollback_keeps_the_gates_verdict_for_its_first_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``inflight-installs-unchecked`` used to be one stderr line and the
    snapshot the gate returned was dropped at the call site. The verdict now
    lands in the transaction state before ``rollback-started`` is written, so
    the durable record says whether the rollback was checked, and on what
    (fix round 3, MEDIUM-3). ``render_persisted_state`` writes ``release.state``
    whole, so a top-level key is all that is needed."""

    calls: list[str] = []
    _plan_reached(monkeypatch, calls)
    verdict = {
        "checked": False,
        "verdict": "unchecked",
        "reason": "automatic rollback: no Running control-plane Pod could run the probe",
        "steps": [],
    }
    release = _rollback_double(
        calls, _require_no_inflight_installs=lambda **_kwargs: dict(verdict)
    )

    with pytest.raises(_Reached):
        ORCHESTRATION.rollback_release(
            release, state={"metadata": {}, "cpu_wheel": "w"}, automatic=True
        )

    assert release.state["inflight_installs"] == verdict


def test_a_rollback_re_entered_after_the_control_plane_restore_skips_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once ``rollback-cpu-restored`` is checkpointed the previous control plane
    is already the one dispatching; refusing the cleanup re-entry would only
    strand the transaction."""

    calls: list[str] = []
    _plan_reached(monkeypatch, calls)
    release = _rollback_double(
        calls,
        state={
            "rollback_completed_phases": ["rollback-started", "rollback-cpu-restored"]
        },
    )

    with pytest.raises(_Reached):
        ORCHESTRATION.rollback_release(
            release, state={"metadata": {}, "cpu_wheel": "w"}
        )

    assert calls == ["aurora-refresh", "compensation-plan"]


def test_a_refused_rollback_touches_nothing_after_the_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    release = _rollback_double(
        calls,
        _require_no_inflight_installs=lambda **_kwargs: (_ for _ in ()).throw(
            GATE.InflightInstallsRefused("rollback refused: workflow-7f3a")
        ),
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "build_rollback_compensation_plan",
        lambda *_a, **_k: pytest.fail("the rollback was planned"),
    )

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        ORCHESTRATION.rollback_release(release, state={"metadata": {}})
    assert calls == ["aurora-refresh"], "the credential refresh is not a restore"


def _failing_upgrade(
    monkeypatch: pytest.MonkeyPatch, rollback: Any
) -> tuple[SimpleNamespace, DIFF.ReleaseDiff]:
    calls: list[str] = []
    # Late-bound on purpose: ``upgrade_release`` rebinds ``release.state`` to a
    # fresh dict for a new transaction, so the stubs must read the attribute
    # at call time rather than capture the initial dict.
    release = _upgrade_double(
        calls,
        rollback=rollback,
        _save_state=lambda phase, **updates: release.state.update(
            {"phase": phase, **updates}
        ),
        _load_state=lambda: dict(release.state),
    )
    monkeypatch.setattr(
        ORCHESTRATION, "_validate_upgrade_transaction", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "_upgrade_context",
        lambda *_a, **_k: ({"metadata": {}}, set(), set(), False),
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "run_upgrade_phases",
        lambda *_a, **_k: (_ for _ in ()).throw(
            MODULE.ReleaseError("cpu finalize barrier failed")
        ),
    )
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )
    return release, diff


def test_a_refused_automatic_rollback_is_logged_and_leaves_the_failed_phase(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Not ``rollback-failed``: nothing was restored, so the transaction stays
    where ``record_upgrade_failure`` put it -- ``failed``, which resume and
    ``--supersede-failed-transaction`` already know how to handle."""

    monkeypatch.delenv(ENV, raising=False)
    attempts: list[dict[str, Any]] = []

    def rollback(**kwargs):
        attempts.append(kwargs)
        raise GATE.InflightInstallsRefused(
            "rollback refused: workflow-7f3a REMEDIATE_DRIVER on ip-10-0-1-17"
        )

    release, diff = _failing_upgrade(monkeypatch, rollback)

    with pytest.raises(MODULE.ReleaseError) as failure:
        ORCHESTRATION.upgrade_release(release, diff=diff)

    message = str(failure.value)
    assert "automatic rollback" in message and "refused" in message
    assert "workflow-7f3a" in message
    assert "cpu finalize barrier failed" in message
    assert FLAG in message
    assert failure.value.__cause__ is not None
    assert len(attempts) == 1
    assert attempts[0]["automatic"] is True, (
        "the engine's rollback must know it is automatic (unreadable store → proceed)"
    )
    assert release.state["phase"] == "failed"
    assert release.state["release_lifecycle"] == "FAILED"
    assert "rollback_failure" not in release.state
    logged = capsys.readouterr().err
    assert "automatic-rollback-refused" in logged
    assert "workflow-7f3a" in logged


def test_a_rollback_that_fails_for_another_reason_still_records_rollback_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rollback(**_kwargs):
        raise MODULE.ReleaseError("previous Agent identities are unavailable")

    release, diff = _failing_upgrade(monkeypatch, rollback)

    with pytest.raises(MODULE.ReleaseError, match="rollback also failed"):
        ORCHESTRATION.upgrade_release(release, diff=diff)

    assert release.state["phase"] == "rollback-failed"


# --- the CLI ---------------------------------------------------------------------


def test_the_flag_travels_to_the_source_preparer_as_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 0
    )
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(admin_cli.ACCEPT_SCHEMA_CHANGE_ENV, raising=False)
    monkeypatch.delenv(admin_cli.SUPERSEDE_FAILED_TRANSACTION_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    base = [
        "deploy",
        "--cpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
        "--gpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
        "--state-dir",
        str(tmp_path / "state"),
        "--admin-email",
        "operations@example.com",
    ]

    assert admin_cli.run(admin_cli.parser().parse_args(base)) == 0
    assert ENV not in calls[-1]["extra_environment"], "no flag, no variable"
    assert admin_cli.run(admin_cli.parser().parse_args([*base, FLAG])) == 0
    assert calls[-1]["extra_environment"][ENV] == "1"
    assert GATE.inflight_installs_allowed(calls[-1]["extra_environment"]) is True
    assert GATE.inflight_installs_allowed({}) is False


def test_an_explicit_variable_wins_over_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV, "1")
    arguments = admin_cli.parser().parse_args(["deploy", FLAG, "--state-dir", "/tmp/x"])

    assert admin_cli.inflight_installs_environment(arguments) == {}


def test_the_flag_is_a_deploy_option_not_a_verb() -> None:
    """A flag on ``deploy``, not an eleventh-plus verb and not a global option."""

    parser = admin_cli.parser()
    arguments = parser.parse_args(["deploy", FLAG, "--state-dir", "/tmp/x"])
    assert arguments.allow_inflight_installs is True, "deploy accepts the flag"
    for argv in ([FLAG], [FLAG[2:]], ["status", FLAG]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


# --- the engine entrypoint --------------------------------------------------------


def test_the_engine_exits_with_the_refusal_code_the_driver_classifies_on(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``scripts/release_deploy.py`` sees only the exit code of
    ``rollout-regional-release.sh rollback``; a refusal must not look like the
    generic 2 that means "rollback failed" (F4)."""

    monkeypatch.setattr(
        MODULE,
        "parser",
        lambda: SimpleNamespace(
            parse_args=lambda argv=None: SimpleNamespace(
                mode="rollback", dry_run=False, config="/tmp/site.json"
            )
        ),
    )
    # The engine's first act is loading the config; raising there stands in for
    # the gate refusing a few calls later.
    monkeypatch.setattr(
        MODULE.ReleaseConfig,
        "load",
        classmethod(
            lambda cls, _path: (_ for _ in ()).throw(
                GATE.InflightInstallsRefused("rollback refused: workflow-7f3a")
            )
        ),
    )

    assert MODULE.main() == GATE.INFLIGHT_INSTALLS_REFUSED_EXIT_CODE
    assert GATE.INFLIGHT_INSTALLS_REFUSED_EXIT_CODE not in (0, 1, 2)
    assert "workflow-7f3a" in capsys.readouterr().err

    monkeypatch.setattr(
        MODULE.ReleaseConfig,
        "load",
        classmethod(
            lambda cls, _path: (_ for _ in ()).throw(MODULE.ReleaseError("broke"))
        ),
    )
    assert MODULE.main() == 2, "every other engine error keeps the generic code"


def test_the_rollback_mode_takes_an_automatic_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = MODULE.parser().parse_args(
        ["rollback", "--config", "/tmp/site.json", "--automatic"]
    )
    assert parsed.automatic is True, "the driver marks its rollback automatic"
    assert (
        MODULE.parser().parse_args(["rollback", "--config", "/tmp/x"]).automatic
        is False
    )

    seen: list[dict[str, Any]] = []
    release = SimpleNamespace(rollback=lambda **kwargs: seen.append(kwargs))
    monkeypatch.setattr(
        MODULE, "parser", lambda: SimpleNamespace(parse_args=lambda argv=None: parsed)
    )
    monkeypatch.setattr(MODULE.ReleaseConfig, "load", classmethod(lambda cls, _p: None))
    monkeypatch.setattr(MODULE, "RegionalRelease", lambda _config, _runner: release)

    assert MODULE.main() == 0
    assert seen == [{"automatic": True}]


def test_the_automatic_marker_is_engine_internal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hidden from ``--help`` (it is set by ``recover_failed_upgrade`` and the
    release driver, never typed by an operator) and refused on every mode but
    ``rollback``, where it would otherwise be accepted silently."""

    assert "--automatic" not in MODULE.parser().format_help()

    parsed = MODULE.parse_arguments(["rollback", "--config", "/tmp/x", "--automatic"])
    assert parsed.automatic is True, "rollback keeps the marker"
    assert MODULE.parse_arguments(["upgrade", "--config", "/tmp/x"]).automatic is False

    for mode in ("upgrade", "deploy", "resume", "status"):
        with pytest.raises(SystemExit) as failure:
            MODULE.parse_arguments([mode, "--config", "/tmp/x", "--automatic"])
        assert failure.value.code == 2, "argparse usage error, like any bad argv"
        assert "--automatic" in capsys.readouterr().err
