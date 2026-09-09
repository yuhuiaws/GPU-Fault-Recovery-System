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
A store that cannot answer refuses too (fail closed), naming the error. A
refused automatic rollback logs why and leaves the transaction in ``failed``,
the phase the resume / supersede levers already handle.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault.models import (
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


def test_blocked_workflows_are_not_scanned(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """BLOCKED is reached only at compile time (policy/compile errors) or as the
    terminal status of a safety-only execution (``executor._complete_claimed_
    workflow`` with ``is_safety``; ``coordinator`` SAFETY_PENDING → BLOCKED). No
    transition takes a RUNNING workflow with official steps to BLOCKED, so a
    BLOCKED row can never hold a WAITING install step -- and the table is never
    archived, so scanning it is what let the newest RUNNING row fall past the
    window (F1)."""

    blocked = _workflow(
        "wf-blocked-waiting",
        WorkflowStatus.BLOCKED,
        [(WorkflowOperation.REMEDIATE_DRIVER, ["node-i"])],
        executions=[(0, WorkflowStepStatus.WAITING)],
    )

    result = _run_probe(monkeypatch, capsys, [blocked])

    assert result["inflight"] == []
    assert result["bounded"] is True


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
    dry_run = False

    def __init__(self, result: str | Exception) -> None:
        self.result = result
        self.execs: list[list[str]] = []

    def run(self, args, **_kwargs):
        if "get" in args and "pod" in args:
            return "cpu-pod"
        self.execs.append(list(args))
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


def _snapshot(*items: dict[str, Any], bounded: bool = True) -> str:
    return json.dumps(
        {
            "inflight": list(items),
            "inflight_count": len(items),
            "bounded": bounded,
            "scanned": len(items),
        }
    )


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
        pod = args[args.index("exec") + 1]
        self.execs.append(pod)
        answer = self.answers[pod]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def probe(self, _args, **_kwargs) -> bool:
        return True


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
        "inflight": [],
        "inflight_count": 0,
        "bounded": True,
        "scanned": 0,
    }
    assert capsys.readouterr().err == ""


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

    for garbage in ("not json", "[]", '{"inflight": "x"}'):
        with pytest.raises(GATE.InflightInstallsRefused, match="evidence"):
            GATE.require_no_inflight_installs(_release(garbage), action="upgrade")


def test_the_override_also_covers_a_store_that_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rollback of a control plane that is itself down must stay possible."""

    monkeypatch.setenv(ENV, "1")
    release = _release(MODULE.ReleaseError("no running CPU ingress Pod"))

    result = GATE.require_no_inflight_installs(release, action="rollback")

    assert result["inflight_count"] == 0
    assert "no running CPU ingress Pod" in capsys.readouterr().err


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
    assert FLAG in message


def test_the_automatic_rollback_proceeds_when_the_store_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The primary automatic-rollback scenario is a control plane that is down
    (CrashLoop, Aurora window, broken wheel); a dead control plane dispatches
    nothing, and wedging production in ``failed`` is the worse failure (F2). An
    actual in-flight install still refuses."""

    monkeypatch.delenv(ENV, raising=False)
    down = _release(MODULE.ReleaseError("no running CPU ingress Pod"))

    result = GATE.require_no_inflight_installs(
        down, action="rollback", unreadable="proceed"
    )

    assert result["inflight_count"] == 0
    logged = capsys.readouterr().err
    assert "inflight-installs-unchecked" in logged
    assert "no running CPU ingress Pod" in logged
    assert "automatic" in logged

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        GATE.require_no_inflight_installs(
            _release(_snapshot(DRIVER_STEP)), action="rollback", unreadable="proceed"
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
    release = SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *args: ["kubectl", *args],
    )

    with pytest.raises(GATE.InflightInstallsRefused, match="workflow-7f3a"):
        GATE.require_no_inflight_installs(release, action="rollback")

    assert runner.execs[0] == "ingress-pod"
    assert "worker-pod" in runner.execs
    assert runner.execs.count("ingress-pod") <= 2, (
        "one retry at most, then the next role"
    )


def test_when_no_role_answers_the_error_names_every_role_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    runner = RoleRunner(pods={}, answers={})
    release = SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *args: ["kubectl", *args],
    )

    with pytest.raises(GATE.InflightInstallsRefused) as failure:
        GATE.require_no_inflight_installs(release, action="upgrade")

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
            parse_args=lambda: SimpleNamespace(mode="rollback", dry_run=False)
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_run_mode",
        lambda _arguments: (_ for _ in ()).throw(
            GATE.InflightInstallsRefused("rollback refused: workflow-7f3a")
        ),
    )

    assert MODULE.main() == GATE.INFLIGHT_INSTALLS_REFUSED_EXIT_CODE
    assert GATE.INFLIGHT_INSTALLS_REFUSED_EXIT_CODE not in (0, 1, 2)
    assert "workflow-7f3a" in capsys.readouterr().err

    monkeypatch.setattr(
        MODULE,
        "_run_mode",
        lambda _arguments: (_ for _ in ()).throw(MODULE.ReleaseError("broke")),
    )
    assert MODULE.main() == 2, "every other engine error keeps the generic code"


def test_the_rollback_mode_takes_an_automatic_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = MODULE.parser().parse_args(
        ["rollback", "--config", "/tmp/site.json", "--automatic"]
    )
    assert arguments.automatic is True, "the driver marks its rollback automatic"
    assert (
        MODULE.parser().parse_args(["rollback", "--config", "/tmp/x"]).automatic
        is False
    )

    seen: list[dict[str, Any]] = []
    release = SimpleNamespace(rollback=lambda **kwargs: seen.append(kwargs))
    monkeypatch.setattr(MODULE.ReleaseConfig, "load", classmethod(lambda cls, _p: None))
    monkeypatch.setattr(MODULE, "RegionalRelease", lambda _config, _runner: release)

    MODULE._run_mode(arguments)

    assert seen == [{"automatic": True}]
