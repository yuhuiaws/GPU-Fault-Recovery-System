from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
    record_focused_tests,
    reusable_focused_tests,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    install_abort_signals,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)

CASE_ID = "GF-REGIONAL-PREEMPT-012"
CONFIRMATION = "PREEMPT012_REAL_QUIESCE_BOUNDARIES"
PROBE_SCRIPT = Path(__file__).with_name("probes") / "preempt012_node_probe.py"
# How long the runner waits for the host cycle to report QUIESCED before the
# control audit starts, and for it to finish afterwards. The cycle is armed
# with a 5s delay and a 45s hold; the fail-safe restore is at 180s.
QUIESCE_WAIT_SECONDS = 120
CYCLE_WAIT_SECONDS = 600
CYCLE_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "INTERRUPTED"})
TIMER_LIVE_STATES = frozenset({"active", "activating", "waiting", "reloading"})
LIMITATIONS = [
    "The node performs a real quiesce/restore. Reset, reboot and replace "
    "remain deliberately unsubmitted because PREEMPT-012 targets the "
    "boundaries before or immediately after quiesce.",
    "Clean group: the predecessor is superseded at a step boundary before "
    "QUIESCE and the successor then executes with the predecessor's "
    "containment inherited, so only RESTART_NODE reaches an adapter. The "
    "pair is seeded as two records (the cross-record successor shape); the "
    "planner's own PREEMPT-002 clean boundary now preempts inside one record "
    "as a successor branch, which this case does not exercise.",
    "Dirty group: the boundary exercised is 'QUIESCE done, reset not "
    "submitted' (PREEMPT-008 shape): the quiesce is handed to the successor "
    "and no RESET_GPU is issued. The LEASED-reset boundary -- a reset already "
    "leased by an executor when the preemption lands -- is covered by "
    "DESTR-016, not here.",
]


class PreemptAcceptanceError(RuntimeError):
    pass


CONTROL_AUDIT = r"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.execution import (
    ProductionExecutorConfig,
    ProductionWorkflowExecutor,
    WorkflowStepOutcome,
)
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)

cluster_id, node_id, stamp = sys.argv[1:]
store = ApplicationContext.from_environment().store
owner = f"preempt012-{stamp}"
now = datetime.now(timezone.utc)
later = now + timedelta(hours=1)
created = []
# Every synthetic id this audit will write. The script is sent with a single
# exec attempt, but a leftover from an earlier run with the same stamp would
# otherwise be silently overwritten and its state mixed into this verdict.
planned_ids = []
for label in ("clean", "dirty"):
    planned_ids.append(("incident", f"incident-{label}-{stamp}"))
    planned_ids.append(("workflow", f"workflow-{label}-pred-{stamp}"))
    planned_ids.append(("workflow", f"workflow-{label}-succ-{stamp}"))
existing = [
    {"kind": kind, "key": key}
    for kind, key in planned_ids
    if store._get_optional(kind, key) is not None
]
if existing:
    print(json.dumps({"error": "audit ids already exist", "existing": existing}))
    raise SystemExit(1)


class AuditAdapter:
    owner = "preempt012-audit"

    def __init__(self):
        self.calls = []

    def supports(self, step):
        return step.execution_owner == self.owner

    def execute(self, context):
        self.calls.append(context.step.operation.value)
        if context.step.operation is WorkflowOperation.RESTART_NODE:
            return WorkflowStepOutcome.waiting(
                operation_id=f"audit/{context.step.operation.value}"
            )
        if context.step.operation in {
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            WorkflowOperation.REPLACE_NODE,
        }:
            raise RuntimeError("physical operation reached the audit adapter")
        return WorkflowStepOutcome.succeeded()


adapter = AuditAdapter()
executor = ProductionWorkflowExecutor(
    store,
    [adapter],
    ProductionExecutorConfig(
        enabled=True,
        executor_id=owner,
        allowed_operations=frozenset(
            {
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.STOP_WORKLOADS,
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESTORE_GPU_SERVICES,
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.RESTART_NODE,
            }
        ),
        workflow_preemption_enabled=True,
    ),
)

OPERATIONS = [
    WorkflowOperation.MARK_UNSCHEDULABLE,
    WorkflowOperation.STOP_WORKLOADS,
    WorkflowOperation.QUIESCE_GPU_SERVICES,
    WorkflowOperation.RESET_GPU,
    WorkflowOperation.RESTORE_GPU_SERVICES,
    WorkflowOperation.VALIDATE_GPU,
]


def step_for(operation):
    return WorkflowStepSpec(
        operation=operation,
        execution_owner="preempt012-audit",
        node_ids=[node_id],
        workload_ids=(
            [f"training/pytorchjob/preempt012-{stamp}"]
            if operation is WorkflowOperation.STOP_WORKLOADS
            else []
        ),
        gpu_uuids=(
            ["GPU-PREEMPT012"] if operation is WorkflowOperation.RESET_GPU else []
        ),
    )


def execution_for(index, operation, *, extra=None):
    details = {}
    if operation is WorkflowOperation.QUIESCE_GPU_SERVICES:
        details = {
            "agent_generations": {node_id: 1},
            "maintenance_window_expires_at": (
                now + timedelta(minutes=10)
            ).isoformat(),
        }
    details.update(extra or {})
    return WorkflowStepExecution(
        step_index=index,
        operation=operation,
        status=WorkflowStepStatus.SUCCEEDED,
        adapter_operation_id=f"audit/{operation.value}",
        details=details,
    )


def save_pair(label, completed, *, inherit_containment):
    incident = FaultIncident(
        incident_id=f"incident-{label}-{stamp}",
        event_id=f"event-{label}-{stamp}",
        event_type="PREEMPT012_AUDIT",
        cluster_id=cluster_id,
        node_ids=[node_id],
        policy_version="preempt012/v1",
        policy_source="ACCEPTANCE",
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
        created_at=now,
        updated_at=now,
    )
    predecessor = WorkflowRequest(
        request_id=f"workflow-{label}-pred-{stamp}",
        incident_id=incident.incident_id,
        runtime_profile_version="preempt012",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        not_before=later,
        official_steps=[step_for(operation) for operation in OPERATIONS],
        completed_step_indexes=list(range(completed)),
        completed_operations=OPERATIONS[:completed],
        step_executions=[
            execution_for(index, OPERATIONS[index]) for index in range(completed)
        ],
        created_at=now,
        updated_at=now,
    )
    if inherit_containment:
        # The cross-record successor shape: the stronger successor carries the
        # predecessor's completed containment as its own completed steps.
        successor_operations = [
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.RESTART_NODE,
        ]
        inherited = list(range(completed))
        successor_extra = {
            "completed_step_indexes": inherited,
            "completed_operations": successor_operations[:completed],
            "inherited_step_indexes": inherited,
            "step_executions": [
                execution_for(
                    index,
                    successor_operations[index],
                    extra={"inherited_from_workflow_id": predecessor.request_id},
                )
                for index in inherited
            ],
        }
    else:
        successor_operations = [WorkflowOperation.RESTART_NODE]
        successor_extra = {}
    successor = WorkflowRequest(
        request_id=f"workflow-{label}-succ-{stamp}",
        incident_id=incident.incident_id,
        predecessor_workflow_id=predecessor.request_id,
        preempt_predecessor=True,
        preemption_reason="PREEMPT-012 stronger reboot successor",
        runtime_profile_version="preempt012",
        status=WorkflowStatus.PENDING,
        fencing_token=1,
        not_before=later,
        official_steps=[step_for(operation) for operation in successor_operations],
        created_at=now,
        updated_at=now,
        **successor_extra,
    )
    incident = incident.model_copy(
        update={"workflow_request_id": successor.request_id}
    )
    store.save_incident(incident)
    store.save_workflow(predecessor)
    store.save_workflow(successor)
    created.extend(
        [
            ("workflow", predecessor.request_id),
            ("workflow", successor.request_id),
            ("incident", incident.incident_id),
        ]
    )
    return incident, predecessor, successor


def request():
    return WorkflowExecutionRequest(expected_fencing_token=1)


result = {"stamp": stamp, "executed_at": datetime.now(timezone.utc).isoformat()}
try:
    _incident, clean, clean_successor = save_pair(
        "clean", 2, inherit_containment=True
    )
    before_calls = list(adapter.calls)
    clean_result = executor.execute(clean.request_id, request())
    clean_saved = store.get_workflow(clean.request_id)
    predecessor_calls = adapter.calls[len(before_calls):]
    before_calls = list(adapter.calls)
    clean_successor_result = executor.execute(clean_successor.request_id, request())
    clean_successor_saved = store.get_workflow(clean_successor.request_id)
    result["clean"] = {
        "status": clean_result.status.value,
        "predecessor_id": clean.request_id,
        "preempted_by": clean_saved.preempted_by_workflow_id,
        "successor_id": clean_successor.request_id,
        "new_adapter_calls": predecessor_calls,
        "completed_operations": [
            item.value for item in clean_saved.completed_operations
        ],
        "successor_status": clean_successor_result.status.value,
        "successor_adapter_calls": adapter.calls[len(before_calls):],
        "successor_inherited_step_indexes": list(
            clean_successor_saved.inherited_step_indexes
        ),
        "successor_completed_operations": [
            item.value for item in clean_successor_saved.completed_operations
        ],
        "successor_inherited_from": sorted(
            {
                str(item.details.get("inherited_from_workflow_id"))
                for item in clean_successor_saved.step_executions
                if item.step_index in clean_successor_saved.inherited_step_indexes
            }
        ),
    }

    _incident, dirty, dirty_successor = save_pair(
        "dirty", 3, inherit_containment=False
    )
    before_calls = list(adapter.calls)
    dirty_result = executor.execute(dirty.request_id, request())
    before_claim = store.get_workflow(dirty_successor.request_id)
    handoff_result = executor.execute(dirty_successor.request_id, request())
    after_claim = store.get_workflow(dirty_successor.request_id)
    result["dirty"] = {
        "status": dirty_result.status.value,
        "predecessor_id": dirty.request_id,
        "preempted_by": store.get_workflow(
            dirty.request_id
        ).preempted_by_workflow_id,
        "successor_id": dirty_successor.request_id,
        "new_adapter_calls": adapter.calls[len(before_calls):],
        "handoff_status": handoff_result.status.value,
        "handoff_before_claim": before_claim.quiesce_handoff_from_workflow_id,
        "handoff_after_claim": after_claim.quiesce_handoff_from_workflow_id,
        "successor_operations": [
            item.operation.value for item in after_claim.official_steps
        ],
        "successor_dependencies": [
            item.depends_on_step_indexes for item in after_claim.official_steps
        ],
        "boundary": "QUIESCE done, reset not submitted (PREEMPT-008 shape)",
    }
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    result["physical_operations_called"] = [
        item
        for item in adapter.calls
        if item
        in {
            "RESET_GPU",
            "RESET_ALL_GPUS_NVSWITCHES",
            "REPLACE_NODE",
        }
    ]
    audit_workflow_ids = [key for kind, key in created if kind == "workflow"]
    result["remote_commands_for_audit_workflows"] = len(
        store.list_remote_commands(workflow_request_ids=audit_workflow_ids)
    )
finally:
    for kind, key in reversed(created):
        store._delete(kind, key)
result["residual_objects"] = [
    {"kind": kind, "key": key}
    for kind, key in created
    if store._get_optional(kind, key) is not None
]
print(json.dumps(result, sort_keys=True, default=str))
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def focused_tests(case_dir: Path, *, reuse_from: Path | None = None) -> dict[str, Any]:
    """The focused regression tests, or the plan's result when it still holds.

    ``--plan`` runs them and records the result with a source digest;
    ``--execute`` passes the plan path so a passing result taken against the
    same tree is reused instead of paid for twice.
    """

    if reuse_from is not None:
        reused = reusable_focused_tests(reuse_from)
        if reused is not None:
            return {**reused, "reused_from_plan": True}
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/execution/test_executor.py::"
        "test_executor_restores_when_successor_cannot_take_quiesce_handoff",
        "tests/execution/test_misc.py::"
        "test_reset_to_reboot_hands_off_quiesce_and_skips_low_reset",
        "tests/execution/test_misc.py::"
        "test_submitted_reset_restores_before_reboot_preemption",
        "tests/execution/test_validation.py::"
        "test_local_validation_waiting_is_safe_to_preempt",
    ]
    completed = RegionalLiveFixture.run(
        command,
        cwd=ROOT,
        check=False,
        timeout=600,
    )
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
    }


def read_only_preflight(
    regional: RegionalLiveFixture,
    *,
    node: str,
    predecessor_id: str,
    predecessor_path_value: Path,
    case_dir: Path,
    plan_path: Path | None = None,
) -> dict[str, Any]:
    node_state = regional.node_snapshot(node)
    store = regional.store_snapshot(node=node)
    tests = focused_tests(case_dir, reuse_from=plan_path)
    # The predecessor is whatever the formal order names -- PREEMPT-009 today,
    # the last PREEMPT contract case that writes evidence -- not a fixed id.
    predecessor = predecessor_evidence(predecessor_path_value, predecessor_id)
    errors = []
    if not predecessor["valid"]:
        errors.append(f"{predecessor_id} predecessor evidence is not PASS")
    if node_state["ready"] != "True" or node_state["unschedulable"]:
        errors.append("target node is not Ready and schedulable")
    if node_state["taints"] or node_state["ownership_annotations"]:
        errors.append("target node has pre-existing mutation ownership")
    if regional.business_workloads(node):
        errors.append("target node has a business workload")
    if (store.get("agent") or {}).get("lifecycle_state") != "ACTIVE":
        errors.append("target Node Agent is not ACTIVE")
    if (store.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        "node": node_state,
        "store": store,
        "predecessor": predecessor,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run PREEMPT-012 real-quiesce clean/dirty boundary acceptance."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--node", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def _parse_time(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def wait_for_cycle(
    host: HostProbeFixture,
    run_id: str,
    *,
    until: frozenset[str],
    timeout: float,
    poll_seconds: float = 2.0,
) -> dict[str, Any]:
    """Poll the host cycle's evidence until its status is one of ``until``.

    A terminal status (COMPLETED, FAILED, INTERRUPTED) always ends the wait,
    whether or not it was asked for: the caller decides what an early end
    means. A fixed sleep used to stand here, and it raced the QUIESCED write
    the control audit needs to overlap.
    """

    deadline = time.monotonic() + timeout
    cycle: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            cycle = host.execute("read", "--run-id", run_id)
        except Exception:
            time.sleep(poll_seconds)
            continue
        status = str(cycle.get("status") or "")
        if status in until or status in CYCLE_TERMINAL_STATUSES:
            return cycle
        time.sleep(poll_seconds)
    return cycle


def evaluate_checks(
    *,
    control: dict[str, Any],
    cycle: dict[str, Any],
    baseline: dict[str, Any],
    final_host: dict[str, Any],
    final_nodes: list[dict[str, Any]],
    provider: list[dict[str, Any]],
    cpu_blast_unchanged: bool,
) -> dict[str, bool]:
    """The verdict's checks, from the control audit and the host cycle."""

    audit_started_at = _parse_time(control["executed_at"])
    audit_completed_at = _parse_time(control["completed_at"])
    quiesced_at = _parse_time(cycle["quiesced_at"])
    restored_at = _parse_time(cycle["restored_at"])
    clean = control["clean"]
    dirty = control["dirty"]
    return {
        "clean_boundary_superseded": (
            clean["status"] == "SUPERSEDED" and not clean["new_adapter_calls"]
        ),
        "clean_successor_reused_containment": (
            clean["preempted_by"] == clean["successor_id"]
            and clean["successor_adapter_calls"] == ["RESTART_NODE"]
            and clean["successor_inherited_step_indexes"] == [0, 1]
            and clean["successor_completed_operations"][:2]
            == ["MARK_UNSCHEDULABLE", "STOP_WORKLOADS"]
            and clean["successor_inherited_from"] == [clean["predecessor_id"]]
        ),
        "dirty_boundary_superseded": dirty["status"] == "SUPERSEDED",
        "dirty_handoff_recorded": (
            dirty["handoff_after_claim"] == dirty["predecessor_id"]
        ),
        "no_physical_operation_called": not control["physical_operations_called"],
        "no_remote_command_created": (
            control["remote_commands_for_audit_workflows"] == 0
        ),
        "control_audit_overlapped_real_quiesce": (
            quiesced_at <= audit_started_at and audit_completed_at <= restored_at
        ),
        "host_services_restored": all(
            value == "active"
            for service, value in final_host["services"].items()
            if baseline["services"].get(service) == "active"
        ),
        "quiesce_state_removed": not final_host["quiesce_state_files"],
        "gpu_count_unchanged": final_host["gpu_count"] == baseline["gpu_count"],
        "all_gpu_nodes_ready_and_schedulable": all(
            item["ready"] == "True"
            and not item["unschedulable"]
            and not any(
                str(taint.get("key", "")).startswith("gpu-fault.io/")
                for taint in item["taints"]
            )
            for item in final_nodes
        ),
        "audit_objects_removed": not control["residual_objects"],
        "no_provider_mutation": not provider,
        "control_plane_eks_identical": cpu_blast_unchanged,
    }


def cleanup_checks(
    *,
    host_cleanup: dict[str, Any] | None,
    node_state: dict[str, Any] | None,
) -> dict[str, bool]:
    """What must be true of the node once the probe has been torn down."""

    return {
        "ownership_annotations_removed": (
            node_state is not None and not node_state.get("ownership_annotations")
        ),
        "cycle_timer_inactive": (
            host_cleanup is not None
            and str(host_cleanup.get("timer_active_state") or "unknown")
            not in TIMER_LIVE_STATES
            and "restore_error" not in host_cleanup
        ),
    }


def execute_case(
    arguments: argparse.Namespace,
    regional: RegionalLiveFixture,
    *,
    node: str,
    image: str,
    predecessor_id: str,
    predecessor_path_value: Path,
    environment: dict[str, str],
) -> int:
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = authorize_execution(
        arguments,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=environment,
    )
    preflight = read_only_preflight(
        regional,
        predecessor_id=predecessor_id,
        node=node,
        predecessor_path_value=predecessor_path_value,
        case_dir=case_dir,
        plan_path=case_dir / "plan.json",
    )
    if preflight["errors"]:
        raise PreemptAcceptanceError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if datetime.now(timezone.utc).timestamp() + 600 >= deadline.timestamp():
        raise PreemptAcceptanceError(
            "maintenance window must have at least 10 minutes remaining"
        )
    # The attempt number is part of every synthetic id (host unit, evidence
    # file, control-audit workflows), so a re-run never collides with the
    # objects an earlier attempt may have left behind.
    run_id = f"preempt012-a{arguments.attempt}-{int(time.time())}"
    host = HostProbeFixture(
        HostProbeSettings(
            kubeconfig=regional.settings.gpu_kubeconfig,
            context=regional.settings.gpu_context,
            namespace=regional.settings.namespace,
            node=node,
            image=image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=PROBE_SCRIPT,
            active_deadline_seconds=1800,
        )
    )
    started_at = utc_now()
    result: dict[str, Any] = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": CASE_ID,
        "verdict": "FAIL",
        "started_at": started_at,
        "predecessor": preflight["predecessor"],
        **regional.evidence_identity(),
    }
    checks: dict[str, bool] = {}
    try:
        host.create()
        baseline = host.execute("snapshot")
        if baseline["quiesce_state_files"]:
            raise PreemptAcceptanceError("pre-existing quiesce state exists")
        armed = host.execute(
            "arm",
            "--run-id",
            run_id,
            "--probe-script",
            host.host_script,
            "--delay-seconds",
            "5",
            "--hold-seconds",
            "45",
            "--failsafe-seconds",
            "180",
        )
        if not armed["timer_active"]:
            raise PreemptAcceptanceError("host cycle timer is not active")
        # The control audit has to run *inside* the real quiesce window: wait
        # for the host to report QUIESCED rather than guessing with a sleep.
        cycle = wait_for_cycle(
            host,
            run_id,
            until=frozenset({"QUIESCED"}),
            timeout=QUIESCE_WAIT_SECONDS,
        )
        if cycle.get("status") != "QUIESCED":
            raise PreemptAcceptanceError(
                f"host cycle did not reach QUIESCED before the audit: {cycle}"
            )
        # One attempt only: the script writes and executes synthetic workflows,
        # and a retry after a partial run would execute them twice.
        control = regional.cpu_python(
            CONTROL_AUDIT,
            regional.settings.cluster_id,
            node,
            run_id,
            attempts=1,
        )
        if "error" in control:
            raise PreemptAcceptanceError(f"control audit refused: {control}")
        cycle = wait_for_cycle(
            host,
            run_id,
            until=CYCLE_TERMINAL_STATUSES,
            timeout=CYCLE_WAIT_SECONDS,
        )
        if cycle.get("status") != "COMPLETED":
            raise PreemptAcceptanceError(f"host quiesce cycle failed: {cycle}")
        final_host = host.execute("snapshot")
        final_nodes = regional.gpu_nodes()
        provider = regional.provider_events(
            datetime.fromisoformat(started_at),
            datetime.now(timezone.utc),
        )
        checks = evaluate_checks(
            control=control,
            cycle=cycle,
            baseline=baseline,
            final_host=final_host,
            final_nodes=final_nodes,
            provider=provider,
            cpu_blast_unchanged=(
                regional.cpu_blast_snapshot() == preflight["cpu_blast"]
            ),
        )
        result.update(
            {
                "checks": checks,
                "control_audit": control,
                "host_cycle": cycle,
                "provider_events": provider,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        host_cleanup: dict[str, Any] | None = None
        try:
            host_cleanup = host.execute(
                "cleanup",
                "--run-id",
                run_id,
                timeout=300,
            )
            result["host_cleanup"] = host_cleanup
        except Exception as exc:
            result["host_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        try:
            residuals = host.cleanup()
        except Exception as exc:
            residuals = {"cleanup_error": True}
            result["probe_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["probe_residuals"] = residuals
        node_state: dict[str, Any] | None = None
        try:
            node_state = regional.node_snapshot(node)
        except Exception as exc:
            result["node_snapshot_error"] = f"{type(exc).__name__}: {exc}"
        result["node_after_cleanup"] = node_state
        checks.update(cleanup_checks(host_cleanup=host_cleanup, node_state=node_state))
        result["checks"] = checks
        result["verdict"] = (
            "PASS"
            if checks
            and all(checks.values())
            and "error" not in result
            and "host_cleanup_error" not in result
            and not any(residuals.values())
            else "FAIL"
        )
    result["executed_at"] = utc_now()
    result["limitations"] = list(LIMITATIONS)
    write_json_atomic(case_evidence_path(arguments.run_dir, CASE_ID), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    regional_settings = settings_from_arguments(arguments)
    node = required(
        arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""),
        "target node",
    )
    image = required(
        arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
        "host probe image",
    )
    if "@sha256:" not in image:
        raise PreemptAcceptanceError("host probe image must use an immutable digest")
    predecessor_id, predecessor_path_value = predecessor_path(
        arguments.run_dir,
        CASE_ID,
        arguments.predecessor_evidence,
    )
    if predecessor_id is None or predecessor_path_value is None:
        raise PreemptAcceptanceError("PREEMPT-012 predecessor resolution is invalid")
    environment = {
        **regional_settings.environment(),
        "GPU_FAULT_TARGET_NODE": node,
        "GPU_FAULT_HOST_PROBE_IMAGE": image,
        "GPU_FAULT_PREDECESSOR_EVIDENCE": str(predecessor_path_value),
    }
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    regional = RegionalLiveFixture(regional_settings)
    if not arguments.execute:
        preflight = read_only_preflight(
            regional,
            predecessor_id=predecessor_id,
            node=node,
            predecessor_path_value=predecessor_path_value,
            case_dir=case_dir,
        )
        details: dict[str, Any] = {
            "risk": "destructive",
            "predecessor": preflight["predecessor"],
            "target_node": node,
            "mutation": (
                "run one real host GPU-service quiesce/restore cycle while "
                "the deployed executor state machine evaluates clean and dirty "
                "preemption boundaries against the live PostgreSQL store"
            ),
            "stop_conditions": [
                f"{predecessor_id} evidence is not PASS",
                "target node is not Ready/schedulable/idle",
                "quiesce fail-safe timer is not armed",
                "host cycle does not report QUIESCED before the control audit",
                "any reset, reboot or replacement reaches an adapter",
                "quiesce state, timer, workflow object or probe resource remains",
            ],
            "rollback": {
                "node_quiesce_has_independent_failsafe_seconds": 180,
                "cycle_restores_services_after_45_seconds": True,
                "cycle_restores_on_sigterm": True,
                "runner_finally_stops_timer_then_restores_and_deletes_probe": True,
            },
            "preflight": preflight,
        }
        record_focused_tests(details, preflight["focused_tests"])
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=CASE_ID,
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            environment=environment,
            details=details,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 1
    return execute_case(
        arguments,
        regional,
        node=node,
        image=image,
        predecessor_id=predecessor_id,
        predecessor_path_value=predecessor_path_value,
        environment=environment,
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
