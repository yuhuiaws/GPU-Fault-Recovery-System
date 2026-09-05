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


def save_pair(label, completed):
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
    operations = [
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.VALIDATE_GPU,
    ]
    predecessor = WorkflowRequest(
        request_id=f"workflow-{label}-pred-{stamp}",
        incident_id=incident.incident_id,
        runtime_profile_version="preempt012",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        not_before=later,
        official_steps=[
            WorkflowStepSpec(
                operation=operation,
                execution_owner="preempt012-audit",
                node_ids=[node_id],
                workload_ids=(
                    [f"training/pytorchjob/preempt012-{stamp}"]
                    if operation is WorkflowOperation.STOP_WORKLOADS
                    else []
                ),
                gpu_uuids=(
                    ["GPU-PREEMPT012"]
                    if operation is WorkflowOperation.RESET_GPU
                    else []
                ),
            )
            for operation in operations
        ],
        completed_step_indexes=list(range(completed)),
        completed_operations=operations[:completed],
        step_executions=[
            WorkflowStepExecution(
                step_index=index,
                operation=operations[index],
                status=WorkflowStepStatus.SUCCEEDED,
                adapter_operation_id=f"audit/{operations[index].value}",
                details=(
                    {
                        "agent_generations": {node_id: 1},
                        "maintenance_window_expires_at": (
                            now + timedelta(minutes=10)
                        ).isoformat(),
                    }
                    if operations[index] is WorkflowOperation.QUIESCE_GPU_SERVICES
                    else {}
                ),
            )
            for index in range(completed)
        ],
        created_at=now,
        updated_at=now,
    )
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
        official_steps=[
            WorkflowStepSpec(
                operation=WorkflowOperation.RESTART_NODE,
                execution_owner="preempt012-audit",
                node_ids=[node_id],
            )
        ],
        created_at=now,
        updated_at=now,
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


baseline_remote = len(store.list_remote_commands())
result = {"executed_at": datetime.now(timezone.utc).isoformat()}
try:
    _incident, clean, clean_successor = save_pair("clean", 2)
    before_calls = list(adapter.calls)
    clean_result = executor.execute(
        clean.request_id,
        WorkflowExecutionRequest(expected_fencing_token=1),
    )
    clean_saved = store.get_workflow(clean.request_id)
    result["clean"] = {
        "status": clean_result.status.value,
        "preempted_by": clean_saved.preempted_by_workflow_id,
        "successor_id": clean_successor.request_id,
        "new_adapter_calls": adapter.calls[len(before_calls):],
        "completed_operations": [
            item.value for item in clean_saved.completed_operations
        ],
    }

    _incident, dirty, dirty_successor = save_pair("dirty", 3)
    before_calls = list(adapter.calls)
    dirty_result = executor.execute(
        dirty.request_id,
        WorkflowExecutionRequest(expected_fencing_token=1),
    )
    before_claim = store.get_workflow(dirty_successor.request_id)
    handoff_result = executor.execute(
        dirty_successor.request_id,
        WorkflowExecutionRequest(expected_fencing_token=1),
    )
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
    }
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
    result["remote_command_delta"] = (
        len(store.list_remote_commands()) - baseline_remote
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


def focused_tests(case_dir: Path) -> dict[str, Any]:
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
    predecessor_path_value: Path,
    case_dir: Path,
) -> dict[str, Any]:
    node_state = regional.node_snapshot(node)
    store = regional.store_snapshot(node=node)
    tests = focused_tests(case_dir)
    predecessor = predecessor_evidence(
        predecessor_path_value,
        "GF-REGIONAL-PREEMPT-011",
    )
    errors = []
    if not predecessor["valid"]:
        errors.append("PREEMPT-011 predecessor evidence is not PASS")
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


def execute_case(
    arguments: argparse.Namespace,
    regional: RegionalLiveFixture,
    *,
    node: str,
    image: str,
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
        node=node,
        predecessor_path_value=predecessor_path_value,
        case_dir=case_dir,
    )
    if preflight["errors"]:
        raise PreemptAcceptanceError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if datetime.now(timezone.utc).timestamp() + 600 >= deadline.timestamp():
        raise PreemptAcceptanceError(
            "maintenance window must have at least 10 minutes remaining"
        )
    run_id = f"preempt012-{arguments.attempt}-{int(time.time())}"
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
    }
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
        time.sleep(25)
        control = regional.cpu_python(
            CONTROL_AUDIT,
            regional.settings.cluster_id,
            node,
            run_id,
        )
        cycle: dict[str, Any] = {}
        deadline_wait = time.monotonic() + 600
        while time.monotonic() < deadline_wait:
            try:
                cycle = host.execute("read", "--run-id", run_id)
            except Exception:
                time.sleep(5)
                continue
            if cycle.get("status") in {"COMPLETED", "FAILED"}:
                break
            time.sleep(5)
        if cycle.get("status") != "COMPLETED":
            raise PreemptAcceptanceError(f"host quiesce cycle failed: {cycle}")
        audit_at = datetime.fromisoformat(
            str(control["executed_at"]).replace("Z", "+00:00")
        )
        quiesced_at = datetime.fromisoformat(
            str(cycle["quiesced_at"]).replace("Z", "+00:00")
        )
        restored_at = datetime.fromisoformat(
            str(cycle["restored_at"]).replace("Z", "+00:00")
        )
        final_host = host.execute("snapshot")
        final_nodes = regional.gpu_nodes()
        provider = regional.provider_events(
            datetime.fromisoformat(started_at),
            datetime.now(timezone.utc),
        )
        checks = {
            "clean_boundary_superseded": (
                control["clean"]["status"] == "SUPERSEDED"
                and not control["clean"]["new_adapter_calls"]
            ),
            "dirty_boundary_superseded": (control["dirty"]["status"] == "SUPERSEDED"),
            "dirty_handoff_recorded": (
                control["dirty"]["handoff_after_claim"]
                == control["dirty"]["predecessor_id"]
            ),
            "no_physical_operation_called": not control["physical_operations_called"],
            "no_remote_command_created": control["remote_command_delta"] == 0,
            "control_audit_overlapped_real_quiesce": (
                quiesced_at <= audit_at <= restored_at
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
            "control_plane_eks_identical": (
                regional.cpu_blast_snapshot() == preflight["cpu_blast"]
            ),
        }
        result.update(
            {
                "verdict": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
                "control_audit": control,
                "host_cycle": cycle,
                "provider_events": provider,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            result["host_cleanup"] = host.execute(
                "cleanup",
                "--run-id",
                run_id,
                timeout=300,
            )
        except Exception as exc:
            result["host_cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
        try:
            residuals = host.cleanup()
        except Exception as exc:
            residuals = {"cleanup_error": True}
            result["probe_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["probe_residuals"] = residuals
        if any(residuals.values()):
            result["verdict"] = "FAIL"
    result["executed_at"] = utc_now()
    result["limitations"] = [
        "The node performs a real quiesce/restore. Reset, reboot and replace "
        "remain deliberately unsubmitted because PREEMPT-012 targets the "
        "boundaries before or immediately after quiesce."
    ]
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
    if predecessor_id != "GF-REGIONAL-PREEMPT-011" or predecessor_path_value is None:
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
            node=node,
            predecessor_path_value=predecessor_path_value,
            case_dir=case_dir,
        )
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=CASE_ID,
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            environment=environment,
            details={
                "risk": "destructive",
                "predecessor": preflight["predecessor"],
                "target_node": node,
                "mutation": (
                    "run one real host GPU-service quiesce/restore cycle while "
                    "the deployed executor state machine evaluates clean and dirty "
                    "preemption boundaries against the live PostgreSQL store"
                ),
                "stop_conditions": [
                    "PREEMPT-011 evidence is not PASS",
                    "target node is not Ready/schedulable/idle",
                    "quiesce fail-safe timer is not armed",
                    "any reset, reboot or replacement reaches an adapter",
                    "quiesce state, timer, workflow object or probe resource remains",
                ],
                "rollback": {
                    "node_quiesce_has_independent_failsafe_seconds": 180,
                    "cycle_restores_services_after_45_seconds": True,
                    "runner_finally_calls_restore_and_deletes_probe": True,
                },
                "preflight": preflight,
            },
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 1
    return execute_case(
        arguments,
        regional,
        node=node,
        image=image,
        predecessor_path_value=predecessor_path_value,
        environment=environment,
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
