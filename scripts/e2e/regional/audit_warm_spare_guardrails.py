#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[3]
CPU_KUBECONFIG = Path()
CPU_CONTEXT = ""
GPU_KUBECONFIG = Path()
GPU_CONTEXT = ""
NAMESPACE = "gpu-fault-system"
AWS_REGION = ""
MANAGED_GPU_CLUSTER = ""
AUTOMATIC_NEGATIVE_CLUSTER = ""

CASE_IDS = (
    "GF-REGIONAL-DESTR-005",
    "GF-REGIONAL-DESTR-006",
    "GF-REGIONAL-DESTR-007",
)
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
OWNERSHIP_ANNOTATIONS = (
    "gpu-fault.io/incident-id",
    "gpu-fault.io/fencing-token",
    "gpu-fault.io/previous-unschedulable",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def configure(arguments: argparse.Namespace, selected_cases: set[str]) -> None:
    global AUTOMATIC_NEGATIVE_CLUSTER
    global AWS_REGION
    global CPU_CONTEXT
    global CPU_KUBECONFIG
    global GPU_CONTEXT
    global GPU_KUBECONFIG
    global MANAGED_GPU_CLUSTER
    global NAMESPACE

    gpu_kubeconfig = _text(
        arguments.gpu_kubeconfig
        or os.getenv("GPU_KUBECONFIG")
        or os.getenv("KUBECONFIG", "")
    )
    GPU_CONTEXT = _text(
        arguments.gpu_context
        or os.getenv("GPU_EKS_CONTEXT")
        or os.getenv("GPU_FAULT_DATAPLANE_CONTEXT", "")
    )
    AWS_REGION = _text(
        arguments.region
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION", "")
    )
    MANAGED_GPU_CLUSTER = _text(
        arguments.managed_gpu_cluster_name
        or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", "")
    )
    AUTOMATIC_NEGATIVE_CLUSTER = _text(
        arguments.automatic_negative_cluster_name
        or os.getenv("GPU_FAULT_AUTOMATIC_NEGATIVE_CLUSTER_NAME", "")
    )
    NAMESPACE = _text(arguments.namespace) or "gpu-fault-system"
    missing = []
    for name, value in (
        ("GPU kubeconfig", gpu_kubeconfig),
        ("GPU context", GPU_CONTEXT),
        ("AWS Region", AWS_REGION),
        ("managed GPU cluster", MANAGED_GPU_CLUSTER),
    ):
        if not value:
            missing.append(name)
    if "GF-REGIONAL-DESTR-006" in selected_cases and not AUTOMATIC_NEGATIVE_CLUSTER:
        missing.append("Automatic negative cluster")

    cpu_kubeconfig = _text(
        arguments.cpu_kubeconfig
        or os.getenv("GPU_FAULT_CONTROL_KUBECONFIG")
        or os.getenv("CPU_KUBECONFIG", "")
    )
    CPU_CONTEXT = _text(
        arguments.cpu_context
        or os.getenv("GPU_FAULT_CONTROL_CONTEXT")
        or os.getenv("CPU_EKS_CONTEXT", "")
    )
    if "GF-REGIONAL-DESTR-005" in selected_cases:
        if not cpu_kubeconfig:
            missing.append("CPU kubeconfig")
        if not CPU_CONTEXT:
            missing.append("CPU context")
    if missing:
        raise RuntimeError(
            "required audit configuration is missing: " + ", ".join(missing)
        )

    GPU_KUBECONFIG = Path(gpu_kubeconfig).expanduser().resolve()
    if not GPU_KUBECONFIG.is_file():
        raise RuntimeError("GPU kubeconfig does not exist")
    if cpu_kubeconfig:
        CPU_KUBECONFIG = Path(cpu_kubeconfig).expanduser().resolve()
        if not CPU_KUBECONFIG.is_file():
            raise RuntimeError("CPU kubeconfig does not exist")


def command(
    argv: list[str],
    *,
    timeout: int = 180,
    stdin: str | None = None,
) -> str:
    result = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(argv)}; "
            f"stderr={result.stderr.strip()}"
        )
    return result.stdout


def kubectl(
    kubeconfig: Path,
    context: str,
    *arguments: str,
    timeout: int = 180,
    stdin: str | None = None,
) -> str:
    return command(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "--context",
            context,
            *arguments,
        ],
        timeout=timeout,
        stdin=stdin,
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def cluster_recovery(name: str) -> dict[str, Any]:
    value = json.loads(
        command(
            [
                "aws",
                "sagemaker",
                "describe-cluster",
                "--region",
                AWS_REGION,
                "--cluster-name",
                name,
                "--output",
                "json",
            ]
        )
    )
    return {
        "cluster_name": value.get("ClusterName"),
        "status": value.get("ClusterStatus"),
        "node_recovery": value.get("NodeRecovery"),
    }


def _running_pods(
    kubeconfig: Path,
    context: str,
    selector: str,
) -> list[str]:
    return kubectl(
        kubeconfig,
        context,
        "-n",
        NAMESPACE,
        "get",
        "pod",
        "-l",
        selector,
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    ).split()


def executor_env() -> list[dict[str, Any]]:
    result = []
    for pod in _running_pods(
        GPU_KUBECONFIG,
        GPU_CONTEXT,
        "app=gpu-fault-cluster-executor",
    ):
        result.append(
            json.loads(
                kubectl(
                    GPU_KUBECONFIG,
                    GPU_CONTEXT,
                    "-n",
                    NAMESPACE,
                    "exec",
                    pod,
                    "--",
                    "python",
                    "-c",
                    (
                        "import json,os; print(json.dumps({"
                        "'pod':os.environ.get('HOSTNAME'),"
                        "'spare_failover':os.environ.get("
                        "'GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER'),"
                        "'remote_state':os.environ.get("
                        "'GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE'),"
                        "'allow_replace':os.environ.get("
                        "'GPU_FAULT_ALLOW_HYPERPOD_REPLACE')}))"
                    ),
                ).splitlines()[-1]
            )
        )
    return result


def _gpu_capacity(node: dict[str, Any]) -> int:
    raw = node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", "0")
    try:
        return int(str(raw))
    except ValueError:
        return 0


def node_snapshot() -> list[dict[str, Any]]:
    value = json.loads(
        kubectl(
            GPU_KUBECONFIG,
            GPU_CONTEXT,
            "get",
            "nodes",
            "-o",
            "json",
        )
    )
    nodes = []
    for item in value.get("items", []):
        if _gpu_capacity(item) <= 0:
            continue
        metadata = item.get("metadata", {})
        annotations = metadata.get("annotations", {})
        taints = sorted(
            (
                {
                    key: taint[key]
                    for key in ("key", "value", "effect", "timeAdded")
                    if key in taint
                }
                for taint in item.get("spec", {}).get("taints", [])
            ),
            key=lambda taint: (
                str(taint.get("key", "")),
                str(taint.get("value", "")),
                str(taint.get("effect", "")),
            ),
        )
        ready = next(
            (
                condition.get("status")
                for condition in item.get("status", {}).get("conditions", [])
                if condition.get("type") == "Ready"
            ),
            None,
        )
        nodes.append(
            {
                "name": metadata.get("name"),
                "uid": metadata.get("uid"),
                "ready": ready,
                "gpu_allocatable": _gpu_capacity(item),
                "unschedulable": bool(item.get("spec", {}).get("unschedulable", False)),
                "taints": taints,
                "ownership_annotations": {
                    key: annotations.get(key) for key in OWNERSHIP_ANNOTATIONS
                },
            }
        )
    return sorted(nodes, key=lambda node: str(node["name"]))


def node_preflight_errors(nodes: list[dict[str, Any]]) -> list[str]:
    errors = []
    if not nodes:
        return ["GPU node snapshot is empty"]
    for node in nodes:
        quarantine = [
            taint for taint in node["taints"] if taint.get("key") == QUARANTINE_TAINT
        ]
        ownership = {
            key: value
            for key, value in node["ownership_annotations"].items()
            if value is not None
        }
        if quarantine or ownership:
            errors.append(
                f"{node['name']} has pre-existing gpu-fault quarantine ownership"
            )
    return errors


def node_state_drift(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[str]:
    errors = []
    before_by_name = {str(node["name"]): node for node in before}
    after_by_name = {str(node["name"]): node for node in after}
    if set(before_by_name) != set(after_by_name):
        errors.append(
            "GPU node membership changed: "
            f"before={sorted(before_by_name)} after={sorted(after_by_name)}"
        )
    for name in sorted(set(before_by_name) & set(after_by_name)):
        previous = before_by_name[name]
        current = after_by_name[name]
        for field in (
            "uid",
            "ready",
            "gpu_allocatable",
            "unschedulable",
            "taints",
            "ownership_annotations",
        ):
            if previous[field] != current[field]:
                errors.append(
                    f"{name} changed {field}: "
                    f"before={previous[field]!r} after={current[field]!r}"
                )
    return errors


def replace_events(started_at: datetime, ended_at: datetime) -> list[dict[str, Any]]:
    value = json.loads(
        command(
            [
                "aws",
                "cloudtrail",
                "lookup-events",
                "--region",
                AWS_REGION,
                "--start-time",
                started_at.isoformat(),
                "--end-time",
                ended_at.isoformat(),
                "--lookup-attributes",
                "AttributeKey=EventSource,AttributeValue=sagemaker.amazonaws.com",
                "--output",
                "json",
            ]
        )
    )
    return [
        {
            "event_name": item.get("EventName"),
            "event_time": str(item.get("EventTime")),
        }
        for item in value.get("Events", [])
        if item.get("EventName") in {"BatchReplaceClusterNodes", "ReplaceClusterNodes"}
    ]


def _pod_python(
    kubeconfig: Path,
    context: str,
    pod: str,
    script: str,
) -> dict[str, Any]:
    output = kubectl(
        kubeconfig,
        context,
        "-n",
        NAMESPACE,
        "exec",
        "-i",
        pod,
        "--",
        "python",
        "-",
        stdin=script,
    )
    return cast(dict[str, Any], json.loads(output.splitlines()[-1]))


def deployed_managed_owner_probe() -> dict[str, Any]:
    pods = _running_pods(
        CPU_KUBECONFIG,
        CPU_CONTEXT,
        "app=gpu-fault-control-worker",
    )
    if not pods:
        raise RuntimeError("no running control worker for DESTR-005 deployed probe")
    script = r"""
import json
from types import SimpleNamespace
from gpu_fault.adapters.managed_recovery import ManagedRecoveryObserverAdapter
from gpu_fault.models import WorkflowOperation, WorkflowStepSpec

owner = "hyperpod-managed-node-recovery"
step = WorkflowStepSpec(
    operation=WorkflowOperation.REPLACE_NODE,
    execution_owner=owner,
    node_ids=["audit-node"],
    parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
)
outcome = ManagedRecoveryObserverAdapter({owner}).execute(
    SimpleNamespace(step=step)
)
print(json.dumps({
    "status": outcome.status.value,
    "error": outcome.error,
}, sort_keys=True))
"""
    return _pod_python(CPU_KUBECONFIG, CPU_CONTEXT, pods[0], script)


def deployed_executor_guard_probes() -> dict[str, Any]:
    pods = _running_pods(
        GPU_KUBECONFIG,
        GPU_CONTEXT,
        "app=gpu-fault-cluster-executor",
    )
    if not pods:
        raise RuntimeError("no running cluster executor for DESTR-007 deployed probe")
    script = r"""
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import gpu_fault.cluster_executor as cluster_executor
from gpu_fault.adapters.hyperpod.lifecycle import HyperPodLifecycleStepAdapter
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)

step = WorkflowStepSpec(
    operation=WorkflowOperation.REPLACE_NODE,
    execution_owner="gpu-fault-hyperpod-adapter",
    node_ids=["audit-node"],
    parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
)
now = datetime.now(timezone.utc)
incident = FaultIncident(
    incident_id="incident-destr007-deployed-probe",
    event_id="event-destr007-deployed-probe",
    event_type="REGIONAL_ACCEPTANCE",
    cluster_id=os.environ["GPU_FAULT_CLUSTER_ID"],
    node_ids=["audit-node"],
    policy_version="destr007-deployed-probe/v1",
    policy_source="ACCEPTANCE",
    state=IncidentState.ACTION_PENDING,
    workflow_request_id="workflow-destr007-deployed-probe",
    fencing_token=1,
    created_at=now,
    updated_at=now,
)
workflow = WorkflowRequest(
    request_id="workflow-destr007-deployed-probe",
    incident_id=incident.incident_id,
    status=WorkflowStatus.RUNNING,
    official_action="REPLACE_NODE",
    fencing_token=1,
    official_steps=[step],
    completed_operations=[WorkflowOperation.MARK_UNSCHEDULABLE],
    created_at=now,
    updated_at=now,
)
adapter = HyperPodLifecycleStepAdapter(object())
adapter.dispatcher.preflight = lambda *_args, **_kwargs: SimpleNamespace(
    safe_to_submit=True,
    gate_failures=[],
    node_recovery="None",
)
outcome = adapter.execute(
    WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(
            expected_fencing_token=1,
            confirm_cluster_name=os.environ["GPU_FAULT_HYPERPOD_CONFIRM_CLUSTER"],
            isolation_verified_nodes=["audit-node"],
        ),
        idempotency_key="workflow-destr007-deployed-probe/0/REPLACE_NODE",
    )
)

class FakeKubernetesAdapter:
    owner = "gpu-fault-kubernetes-adapter"
    core = object()

cluster_executor._persistent_store_from_environment = lambda: None
cluster_executor.KubernetesWorkflowAdapter = lambda **_kwargs: FakeKubernetesAdapter()
cluster_executor.HyperPodLifecycleAdapter = lambda *_args, **_kwargs: object()
cluster_executor.missing_aws_credentials = lambda: None
os.environ["GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER"] = "false"
os.environ["GPU_FAULT_ENABLE_HYPERPOD_ADAPTER"] = "true"
os.environ["GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER"] = "true"
os.environ["GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE"] = "false"
startup_error = None
try:
    cluster_executor.executor_from_environment()
except cluster_executor.ClusterExecutorError as exc:
    startup_error = str(exc)

print(json.dumps({
    "coordinator_guard": {
        "status": outcome.status.value,
        "error": outcome.error,
    },
    "startup_guard_error": startup_error,
}, sort_keys=True))
"""
    return _pod_python(GPU_KUBECONFIG, GPU_CONTEXT, pods[0], script)


def run_pytest(case_dir: Path, nodeids: list[str]) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *nodeids],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    path = case_dir / "pytest.log"
    path.write_text(result.stdout)
    path.chmod(0o600)
    return result.returncode == 0


def case_definitions() -> dict[str, list[str]]:
    return {
        "GF-REGIONAL-DESTR-005": [
            "tests/execution/test_misc.py::"
            "test_managed_recovery_rejects_warm_spare_replacement"
        ],
        "GF-REGIONAL-DESTR-006": [
            "tests/execution/test_node_action.py::"
            "test_warm_spare_rejects_automatic_node_recovery_before_allocation"
        ],
        "GF-REGIONAL-DESTR-007": [
            "tests/hyperpod/test_cluster_executor.py::"
            "test_executor_spare_failover_requires_remote_state",
            "tests/execution/test_node_action.py::"
            "test_warm_spare_requires_enabled_coordinator_before_submission",
        ],
    }


def _probe_errors(
    case_id: str,
    managed_probe: dict[str, Any] | None,
    executor_probes: dict[str, Any] | None,
) -> list[str]:
    errors = []
    if case_id == "GF-REGIONAL-DESTR-005":
        if managed_probe is None:
            errors.append("deployed managed-owner guard probe did not run")
        elif managed_probe != {
            "status": "FAILED",
            "error": (
                "healthy warm-spare replacement cannot be delegated "
                "to managed/provider node recovery"
            ),
        }:
            errors.append("deployed managed-owner guard returned an unexpected result")
    if case_id == "GF-REGIONAL-DESTR-007":
        if executor_probes is None:
            errors.append("deployed executor guard probes did not run")
        else:
            coordinator = executor_probes.get("coordinator_guard")
            if coordinator != {
                "status": "FAILED",
                "error": (
                    "healthy warm-spare replacement is required but the "
                    "spare coordinator is disabled"
                ),
            }:
                errors.append(
                    "deployed coordinator guard returned an unexpected result"
                )
            startup_error = str(executor_probes.get("startup_guard_error") or "")
            if (
                "regional HyperPod spare failover requires "
                "GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE=true"
            ) not in startup_error:
                errors.append(
                    "deployed startup guard did not reject remote-state=false"
                )
    return errors


def run_audit(run_dir: Path, selected_cases: list[str]) -> int:
    definitions = case_definitions()
    started_at = datetime.now(timezone.utc)
    baseline = node_snapshot()
    preflight_errors = node_preflight_errors(baseline)
    write_json(run_dir / "gpu-node-baseline.json", baseline)

    gpu: dict[str, Any] = {}
    automatic: dict[str, Any] | None = None
    env: list[dict[str, Any]] = []
    managed_probe: dict[str, Any] | None = None
    executor_probes: dict[str, Any] | None = None
    test_results: dict[str, bool] = {}
    fatal_error = None
    postflight: list[dict[str, Any]] = []
    drift_errors: list[str] = []
    events: list[dict[str, Any]] = []
    try:
        gpu = cluster_recovery(MANAGED_GPU_CLUSTER)
        if "GF-REGIONAL-DESTR-006" in selected_cases:
            automatic = cluster_recovery(AUTOMATIC_NEGATIVE_CLUSTER)
        env = executor_env()
        if "GF-REGIONAL-DESTR-005" in selected_cases:
            managed_probe = deployed_managed_owner_probe()
        if "GF-REGIONAL-DESTR-007" in selected_cases:
            executor_probes = deployed_executor_guard_probes()
        for case_id in selected_cases:
            case_dir = run_dir / "cases" / case_id
            case_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            test_results[case_id] = run_pytest(case_dir, definitions[case_id])
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
    finally:
        ended_at = datetime.now(timezone.utc)
        try:
            events = replace_events(started_at, ended_at)
        except Exception as exc:
            fatal_error = fatal_error or f"{type(exc).__name__}: {exc}"
        try:
            postflight = node_snapshot()
            drift_errors = node_state_drift(baseline, postflight)
            write_json(run_dir / "gpu-node-postflight.json", postflight)
        except Exception as exc:
            fatal_error = fatal_error or f"{type(exc).__name__}: {exc}"

    failed = False
    for case_id in selected_cases:
        errors = list(preflight_errors)
        if not test_results.get(case_id, False):
            errors.append("focused pytest failed")
        if fatal_error:
            errors.append(fatal_error)
        if events:
            errors.append("CloudTrail contains provider replace")
        errors.extend(drift_errors)
        errors.extend(_probe_errors(case_id, managed_probe, executor_probes))
        if gpu.get("node_recovery") != "None":
            errors.append("managed GPU cluster NodeRecovery is not None")
        if case_id == "GF-REGIONAL-DESTR-006":
            if automatic is None or automatic["node_recovery"] != "Automatic":
                errors.append("negative cluster is not Automatic")
        if case_id == "GF-REGIONAL-DESTR-007":
            if len(env) != 2:
                errors.append("expected two executor replicas")
            if any(
                item.get("spare_failover") != "true"
                or item.get("remote_state") != "true"
                or item.get("allow_replace") != "false"
                for item in env
            ):
                errors.append("production executor safety environment is inconsistent")
        result = {
            "case_id": case_id,
            "verdict": "PASS" if not errors else "FAIL",
            "acceptance_incomplete": bool(preflight_errors or drift_errors),
            "errors": errors,
            "focused_pytest": definitions[case_id],
            "gpu_cluster": gpu,
            "automatic_negative_cluster": automatic,
            "executor_env": env,
            "deployed_managed_owner_probe": managed_probe,
            "deployed_executor_guard_probes": executor_probes,
            "replace_events": events,
            "node_baseline": baseline,
            "node_postflight": postflight,
            "node_state_identical": not drift_errors,
        }
        case_dir = run_dir / "cases" / case_id
        write_json(case_dir / f"{case_id}.json", result)
        print(json.dumps(result, sort_keys=True))
        failed = failed or bool(errors)
    return int(failed)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run the read-only/isolated warm-spare guard acceptance for "
            "DESTR-005, DESTR-006 and DESTR-007."
        )
    )
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--case", action="append", choices=CASE_IDS, default=[])
    result.add_argument("--cpu-kubeconfig")
    result.add_argument("--cpu-context")
    result.add_argument("--gpu-kubeconfig")
    result.add_argument("--gpu-context")
    result.add_argument("--namespace", default="gpu-fault-system")
    result.add_argument("--region")
    result.add_argument("--managed-gpu-cluster-name")
    result.add_argument("--automatic-negative-cluster-name")
    return result


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    selected_cases = arguments.case or list(CASE_IDS)
    configure(arguments, set(selected_cases))
    if arguments.run_dir is not None:
        return run_audit(arguments.run_dir, selected_cases)
    configured = os.getenv("GPU_FAULT_ACCEPTANCE_RUN_DIR", "").strip()
    if configured:
        return run_audit(Path(configured), selected_cases)
    with tempfile.TemporaryDirectory(prefix="gpu-fault-warm-spare-guards-") as value:
        return run_audit(Path(value), selected_cases)


if __name__ == "__main__":
    raise SystemExit(main())
