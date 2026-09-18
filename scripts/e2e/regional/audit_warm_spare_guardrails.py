#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gpu_fault.admin.atomic_json import (  # noqa: E402
    write_json_atomic as _write_document,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.acceptance_scope import current_acceptance_scope  # noqa: E402
from scripts.e2e.regional.acceptance_supervision import (  # noqa: E402
    bind_command_supervision,
)
from scripts.e2e.regional.destr_net_isolation_probe import (  # noqa: E402
    ISOLATED_NODE_READER,
)
from scripts.e2e.regional.guardrail_audit_evidence import (  # noqa: E402
    REGISTRY_PROBE,
    complete_pod_population,
    registration_identity,
    validate_provider_record,
)
from scripts.e2e.regional.regional_case_contract import predecessor_path  # noqa: E402
from scripts.e2e.regional.regional_commands import run_fixture_command  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    PROVIDER_MUTATIONS,
    component_python,
    predecessor_evidence,
)
from scripts.e2e.regional.warm_node_evidence import project_gpu_nodes  # noqa: E402
from tools.pytest_result_identity import PytestReceipt  # noqa: E402

# CloudTrail's delivery guarantee: a negative read whose window ended inside
# this is provisional, not proof.
PROVIDER_EVENT_VISIBILITY_SECONDS = 900
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
AUTOMATIC_GUARD_ERROR = (
    "healthy warm-spare replacement requires HyperPod "
    "NodeRecovery=None; provider replacement is disabled"
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
    return run_fixture_command(argv, cwd=ROOT, input_text=stdin, timeout=timeout).stdout


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
    """Atomic, private, scoped: a case document through the shared writer, a
    node snapshot (a list) through the same all-or-nothing rename."""

    if isinstance(value, dict):
        write_json_atomic(path, cast(dict[str, Any], value))
    else:
        _write_document(path, cast(dict[str, Any], value))


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
    inventory = json.loads(
        kubectl(
            kubeconfig,
            context,
            "-n",
            NAMESPACE,
            "get",
            "pod",
            "-l",
            selector,
            "-o",
            "json",
        )
    )
    if not selector.startswith("app=") or selector.count("=") != 1:
        raise RuntimeError("guard audit Pod selector is unsupported")
    deployment = json.loads(
        kubectl(
            kubeconfig,
            context,
            "-n",
            NAMESPACE,
            "get",
            "deployment",
            selector.removeprefix("app="),
            "-o",
            "json",
        )
    )
    return [pod["name"] for pod in complete_pod_population(deployment, inventory)]


def executor_env() -> list[dict[str, Any]]:
    result = []
    for pod in _running_pods(
        GPU_KUBECONFIG,
        GPU_CONTEXT,
        "app=gpu-fault-cluster-executor",
    ):
        metadata = json.loads(
            kubectl(
                GPU_KUBECONFIG,
                GPU_CONTEXT,
                "-n",
                NAMESPACE,
                "get",
                "pod",
                pod,
                "-o",
                "json",
            )
        )["metadata"]
        uid = metadata.get("uid")
        if not isinstance(uid, str) or not uid:
            raise RuntimeError("executor Pod UID is missing")
        observation = json.loads(
            kubectl(
                GPU_KUBECONFIG,
                GPU_CONTEXT,
                "-n",
                NAMESPACE,
                "exec",
                pod,
                "--",
                component_python("gpu"),
                "-c",
                (
                    "import json,os; print(json.dumps({"
                    "'pod':os.environ.get('HOSTNAME'),"
                    "'cluster_id':os.environ.get('GPU_FAULT_CLUSTER_ID'),"
                    "'spare_failover':os.environ.get("
                    "'GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER'),"
                    "'remote_state':os.environ.get("
                    "'GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE'),"
                    "'allow_replace':os.environ.get("
                    "'GPU_FAULT_ALLOW_HYPERPOD_REPLACE')}))"
                ),
            ).splitlines()[-1]
        )
        after = json.loads(
            kubectl(
                GPU_KUBECONFIG,
                GPU_CONTEXT,
                "-n",
                NAMESPACE,
                "get",
                "pod",
                pod,
                "-o",
                "json",
            )
        )["metadata"]
        if after.get("uid") != uid or observation.get("pod") != pod:
            raise RuntimeError("executor Pod changed during its environment probe")
        result.append({**observation, "pod_uid": uid})
    return result


def audit_identity() -> dict[str, str]:
    pods = _running_pods(CPU_KUBECONFIG, CPU_CONTEXT, "app=gpu-fault-api-ha")
    if not pods:
        raise RuntimeError("no Ready API Pod for registry identity")
    registry = _pod_python(CPU_KUBECONFIG, CPU_CONTEXT, pods[0], REGISTRY_PROBE)
    registration = registration_identity(
        registry, MANAGED_GPU_CLUSTER, AUTOMATIC_NEGATIVE_CLUSTER, AWS_REGION
    )
    document = json.loads(
        kubectl(
            CPU_KUBECONFIG,
            CPU_CONTEXT,
            "-n",
            NAMESPACE,
            "get",
            "configmap",
            "gpu-fault-regional-release-state",
            "-o",
            "json",
        )
    )
    state = json.loads(document["data"]["state.json"])
    release_id = state.get("release_id")
    if (
        not isinstance(release_id, str)
        or not release_id.strip()
        or state.get("phase") != "complete"
    ):
        raise RuntimeError("guard audit requires a complete bound release")
    return {
        "release_id": release_id,
        "cluster_id": registration["cluster_id"],
        "eks_cluster_arn": registration["eks_cluster_arn"],
        "registry_generation": str(registry["generation"]),
    }


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
    return project_gpu_nodes(value, ownership_annotations=OWNERSHIP_ANNOTATIONS)


def node_identity_errors(nodes: list[dict[str, Any]]) -> list[str]:
    errors = []
    names: set[str] = set()
    uids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            errors.append("GPU node snapshot contains a non-object identity")
            continue
        for field, seen in (("name", names), ("uid", uids)):
            value = node.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"GPU node snapshot has an invalid {field}")
            elif value in seen:
                errors.append(f"GPU node snapshot has a duplicate {field}")
            else:
                seen.add(value)
    return errors


def node_preflight_errors(nodes: list[dict[str, Any]]) -> list[str]:
    errors = []
    if not nodes:
        return ["GPU node snapshot is empty"]
    errors = node_identity_errors(nodes)
    if errors:
        return errors
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
    errors = [f"baseline: {error}" for error in node_identity_errors(before)]
    errors.extend(f"postflight: {error}" for error in node_identity_errors(after))
    if errors:
        return errors
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
    if not isinstance(value, dict) or not isinstance(value.get("Events"), list):
        raise RuntimeError("CloudTrail event inventory is unreadable")
    return [
        {
            "event_name": item.get("EventName"),
            "event_time": str(item.get("EventTime")),
        }
        for item in value.get("Events", [])
        if item.get("EventName") in PROVIDER_MUTATIONS
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
        component_python(
            "cpu" if (kubeconfig, context) == (CPU_KUBECONFIG, CPU_CONTEXT) else "gpu"
        ),
        "-",
        stdin=script,
    )
    value = json.loads(output.splitlines()[-1])
    if not isinstance(value, dict) or not value:
        raise RuntimeError("guard probe did not return a nonempty object")
    return value


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


CLUSTER_FIELDS = ("ClusterName", "ClusterStatus", "NodeRecovery")
NODE_SUMMARY_FIELDS = (
    "InstanceGroupName",
    "InstanceId",
    "NodeLogicalId",
    "InstanceType",
    "InstanceStatus",
)
NODE_DETAIL_FIELDS = (
    "NodeLogicalId",
    "InstanceId",
    "InstanceGroupName",
    "InstanceType",
    "InstanceStatus",
    "PrivateDnsHostname",
    "PrivatePrimaryIp",
    "Placement",
    "CapacityType",
    "KubernetesConfig",
)


def _project(value: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {key: value[key] for key in fields if key in value}


def record_provider_snapshot(cluster_name: str) -> dict[str, Any]:
    """Record the negative cluster's real provider payloads, read-only.

    The auditor host holds SageMaker read rights on every cluster in the
    region; the deployed executor deliberately does not, because its IRSA role
    is scoped to the managed cluster alone (the boundary DESTR-011 and
    DESTR-013 assert). So the payloads DESTR-006's guard must judge are
    fetched here and replayed into the pod, rather than widening that role to
    make the probe convenient.

    Only the fields the deployed adapter actually reads are kept: the point is
    real provider data, not a copy of the cluster's network and IAM identity.
    """

    recorded_at = datetime.now(timezone.utc)
    described = json.loads(
        command(
            [
                "aws",
                "sagemaker",
                "describe-cluster",
                "--region",
                AWS_REGION,
                "--cluster-name",
                cluster_name,
                "--output",
                "json",
            ]
        )
    )
    pages: list[dict[str, Any]] = []
    next_token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        argv = [
            "aws",
            "sagemaker",
            "list-cluster-nodes",
            "--region",
            AWS_REGION,
            "--cluster-name",
            cluster_name,
            "--include-node-logical-ids",
            "--max-results",
            "100",
            "--no-paginate",
            "--output",
            "json",
        ]
        if next_token:
            argv += ["--next-token", next_token]
        listed = json.loads(command(argv))
        if not isinstance(listed, dict) or not isinstance(
            listed.get("ClusterNodeSummaries"), list
        ):
            raise RuntimeError("provider node page is unreadable")
        pages.append(
            {
                "ClusterNodeSummaries": [
                    _project(item, NODE_SUMMARY_FIELDS)
                    for item in listed.get("ClusterNodeSummaries", [])
                ],
                "NextToken": listed.get("NextToken"),
            }
        )
        next_token = listed.get("NextToken")
        if not next_token:
            break
        if not isinstance(next_token, str) or next_token in seen_tokens:
            raise RuntimeError("provider node pagination repeated a token")
        seen_tokens.add(next_token)
    details: dict[str, Any] = {}
    for page in pages:
        for item in page["ClusterNodeSummaries"]:
            logical_id = item["NodeLogicalId"]
            node = json.loads(
                command(
                    [
                        "aws",
                        "sagemaker",
                        "describe-cluster-node",
                        "--region",
                        AWS_REGION,
                        "--cluster-name",
                        cluster_name,
                        "--node-logical-id",
                        logical_id,
                        "--output",
                        "json",
                    ]
                )
            )
            details[logical_id] = _project(node["NodeDetails"], NODE_DETAIL_FIELDS)
    payloads = {
        "cluster_name": cluster_name,
        "describe_cluster": _project(described, CLUSTER_FIELDS),
        "list_cluster_nodes": pages,
        "describe_cluster_node": details,
    }
    serialized = json.dumps(payloads, sort_keys=True, separators=(",", ":"))
    result = {
        "payloads": payloads,
        "recorded_at": recorded_at.isoformat().replace("+00:00", "Z"),
        "payload_digest": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "recorded_by": "acceptance auditor host (SageMaker read-only)",
    }
    validate_provider_record(result, cluster_name)
    return result


def deployed_automatic_recovery_probe(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Fire DESTR-006's guard in the deployed executor, on a real Automatic cluster.

    Reading `NodeRecovery` off two clusters and running a repo-side unit test
    says nothing about the code that is deployed. This case's whole claim is
    that the shipped adapter refuses a warm-spare replacement the moment the
    provider's own recovery controller is armed, so the `Automatic` it refuses
    on has to be observed, not stubbed: the deployed adapter parses the real
    recorded DescribeCluster / ListClusterNodes / DescribeClusterNode payloads
    of the unmanaged Automatic cluster and its own preflight decides.

    A second arm runs the identical step without
    `replacement_strategy=HEALTHY_WARM_SPARE_ONLY`, where the warm-spare guard
    must not be what refuses. Without it a probe that failed for any reason at
    all would read as proof, and the recorded refusal could not be attributed
    to the observed `NodeRecovery`.

    Nothing here can mutate: the replayed client implements the three read
    calls and nothing else, no spare coordinator is attached, and the negative
    cluster's `NodeRecovery` is never touched. A target-bound, read-only
    synthetic Kubernetes reader satisfies the earlier isolation fence; this
    proves the recovery guard, not the live negative cluster's isolation.
    """

    validate_provider_record(snapshot, AUTOMATIC_NEGATIVE_CLUSTER)
    pods = _running_pods(
        GPU_KUBECONFIG,
        GPU_CONTEXT,
        "app=gpu-fault-cluster-executor",
    )
    if not pods:
        raise RuntimeError("no running cluster executor for DESTR-006 deployed probe")
    script = r"""
import json
import os
from datetime import datetime, timezone

from gpu_fault.adapters.hyperpod.lifecycle import HyperPodLifecycleStepAdapter
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.hyperpod import HyperPodAdapterConfig, HyperPodLifecycleAdapter
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)


class RecordedProviderClient:
    # Replays one recorded read-only SageMaker snapshot, and nothing else.
    # The three read calls the adapter makes are served verbatim from what
    # the auditor recorded off the live cluster. Every other API -- every
    # mutation -- is simply absent, so this probe cannot submit a
    # replacement even by mistake.

    def __init__(self, recorded):
        self._recorded = recorded

    def _check(self, cluster_name):
        if cluster_name != self._recorded["cluster_name"]:
            raise AssertionError(
                "recorded snapshot is for another cluster: " + str(cluster_name)
            )

    def describe_cluster(self, *, ClusterName):
        self._check(ClusterName)
        return dict(self._recorded["describe_cluster"])

    def list_cluster_nodes(
        self,
        *,
        ClusterName,
        MaxResults=None,
        IncludeNodeLogicalIds=False,
        NextToken=None,
    ):
        self._check(ClusterName)
        if not IncludeNodeLogicalIds:
            raise AssertionError("recorded snapshot only covers logical-id listing")
        pages = self._recorded["list_cluster_nodes"]
        index = 0
        if NextToken is not None:
            tokens = [page.get("NextToken") for page in pages]
            if NextToken not in tokens:
                raise AssertionError("unrecorded ListClusterNodes page token")
            index = tokens.index(NextToken) + 1
        page = pages[index]
        result = {"ClusterNodeSummaries": list(page["ClusterNodeSummaries"])}
        if page.get("NextToken"):
            result["NextToken"] = page["NextToken"]
        return result

    def describe_cluster_node(self, *, ClusterName, NodeLogicalId):
        self._check(ClusterName)
        details = self._recorded["describe_cluster_node"].get(NodeLogicalId)
        if details is None:
            raise AssertionError("unrecorded node: " + str(NodeLogicalId))
        return {"NodeDetails": dict(details)}


cluster = RECORDED["cluster_name"]
config = HyperPodAdapterConfig.from_environment(cluster_name=cluster)
adapter = HyperPodLifecycleAdapter(config, client=RecordedProviderClient(RECORDED))
described = adapter.client.describe_cluster(ClusterName=cluster)
nodes = adapter.list_nodes(enrich=True)
target = next((item for item in nodes if item.status == "Running"), None)
if target is None:
    raise SystemExit("the Automatic negative cluster has no Running node to preflight")

now = datetime.now(timezone.utc)
ISOLATION_READS = {}


def guard_outcome(arm, parameters):
    step = WorkflowStepSpec(
        operation=WorkflowOperation.REPLACE_NODE,
        execution_owner="gpu-fault-hyperpod-adapter",
        node_ids=[target.node_logical_id],
        parameters=parameters,
    )
    incident = FaultIncident(
        incident_id="incident-destr006-deployed-probe",
        event_id="event-destr006-deployed-probe",
        event_type="REGIONAL_ACCEPTANCE",
        cluster_id=os.environ["GPU_FAULT_CLUSTER_ID"],
        node_ids=[target.node_logical_id],
        policy_version="destr006-deployed-probe/v1",
        policy_source="ACCEPTANCE",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-destr006-deployed-probe",
        fencing_token=1,
        created_at=now,
        updated_at=now,
    )
    workflow = WorkflowRequest(
        request_id="workflow-destr006-deployed-probe",
        incident_id=incident.incident_id,
        status=WorkflowStatus.RUNNING,
        official_action="REPLACE_NODE",
        fencing_token=1,
        official_steps=[step],
        completed_operations=[WorkflowOperation.MARK_UNSCHEDULABLE],
        created_at=now,
        updated_at=now,
    )
    kubernetes, reader = isolated_kubernetes_adapter(
        incident.incident_id, incident.fencing_token, target.aliases
    )
    outcome = HyperPodLifecycleStepAdapter(
        adapter, kubernetes_adapter=kubernetes
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=1,
                confirm_cluster_name=cluster,
                isolation_verified_nodes=sorted(target.aliases),
            ),
            idempotency_key="workflow-destr006-deployed-probe/0/REPLACE_NODE",
        )
    )
    ISOLATION_READS[arm] = reader.reads
    return {"status": outcome.status.value, "error": outcome.error}


print(json.dumps({
    "cluster_name": described.get("ClusterName"),
    "observed_node_recovery": described.get("NodeRecovery"),
    "probed_node_status": target.status,
    "probed_node_count": len(nodes),
    "warm_spare_guard": guard_outcome(
        "warm_spare_guard", {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"}
    ),
    "control_without_warm_spare_strategy": guard_outcome(
        "control_without_warm_spare_strategy", {}
    ),
    "isolation_reads": ISOLATION_READS,
    "isolation_source": "synthetic-read-only",
}, sort_keys=True))
"""
    header = (
        f"import json\nRECORDED = json.loads({json.dumps(snapshot['payloads'])!r})\n"
    )
    probe = _pod_python(
        GPU_KUBECONFIG, GPU_CONTEXT, pods[0], header + ISOLATED_NODE_READER + script
    )
    probe["payload_provenance"] = {
        key: snapshot[key] for key in ("recorded_at", "payload_digest", "recorded_by")
    }
    return probe


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

import gpu_fault.cluster_executor.bootstrap as executor_bootstrap
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
kubernetes, reader = isolated_kubernetes_adapter(
    incident.incident_id, incident.fencing_token, step.node_ids
)
adapter = HyperPodLifecycleStepAdapter(object(), kubernetes_adapter=kubernetes)
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

executor_bootstrap._persistent_store_from_environment = lambda: None
executor_bootstrap.KubernetesWorkflowAdapter = lambda **_kwargs: FakeKubernetesAdapter()
executor_bootstrap.HyperPodLifecycleAdapter = lambda *_args, **_kwargs: object()
executor_bootstrap.missing_aws_credentials = lambda: None
os.environ["GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER"] = "false"
os.environ["GPU_FAULT_ENABLE_HYPERPOD_ADAPTER"] = "true"
os.environ["GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER"] = "true"
os.environ["GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE"] = "false"
startup_error = None
try:
    executor_bootstrap.executor_from_environment()
except executor_bootstrap.ClusterExecutorError as exc:
    startup_error = str(exc)

print(json.dumps({
    "coordinator_guard": {
        "status": outcome.status.value,
        "error": outcome.error,
    },
    "startup_guard_error": startup_error,
    "isolation_reads": reader.reads,
    "isolation_source": "synthetic-read-only",
}, sort_keys=True))
"""
    return _pod_python(
        GPU_KUBECONFIG, GPU_CONTEXT, pods[0], ISOLATED_NODE_READER + script
    )


def run_pytest(case_dir: Path, nodeids: list[str]) -> PytestReceipt | None:
    from scripts.e2e.regional.warm_pytest import run_pytest as execute_pytest

    return execute_pytest(case_dir, nodeids, root=ROOT)


def run_focused_pytest(
    run_dir: Path,
    definitions: dict[str, list[str]],
) -> dict[str, bool]:
    """Attribute complete discovered variants and passing phases to each case."""

    from scripts.e2e.regional.focused_pytest import PASSED_PHASES
    from tools.pytest_result_identity import normalized_pytest_nodeid

    if not definitions:
        return {}
    log_dir = run_dir / "cases"
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    nodeids = list(
        dict.fromkeys(nodeid for items in definitions.values() for nodeid in items)
    )
    if not nodeids:
        return {case_id: False for case_id in definitions}
    receipt = run_pytest(log_dir, nodeids)
    if (
        receipt is None
        or receipt.discovered_nodeids is None
        or receipt.collection_skips
    ):
        return {case_id: False for case_id in definitions}
    results = {}
    for case_id, items in definitions.items():
        passed = bool(items)
        for nodeid in items:
            records, error = receipt.selection(
                normalized_pytest_nodeid(nodeid, root=ROOT)
            )
            if (
                error
                or not records
                or any(
                    record.get("status") != "PASS"
                    or record.get("phases") != PASSED_PHASES
                    for record in records
                )
            ):
                passed = False
        results[case_id] = passed
    return results


def cloudtrail_provisional(ended_at: datetime, *, now: datetime | None = None) -> bool:
    current = now or datetime.now(timezone.utc)
    return current - ended_at < timedelta(seconds=PROVIDER_EVENT_VISIBILITY_SECONDS)


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


def probe_errors(
    case_id: str,
    managed_probe: dict[str, Any] | None,
    executor_probes: dict[str, Any] | None,
    automatic_probe: dict[str, Any] | None = None,
) -> list[str]:
    errors = []
    if case_id == "GF-REGIONAL-DESTR-006":
        if automatic_probe is None:
            errors.append("deployed automatic-recovery guard probe did not run")
        else:
            if automatic_probe.get("observed_node_recovery") != "Automatic":
                # Without this the probe still "passes" its error check while
                # having proved the guard on a cluster that was never Automatic.
                errors.append(
                    "deployed probe did not observe NodeRecovery=Automatic: "
                    + json.dumps(automatic_probe, sort_keys=True)
                )
            if automatic_probe.get("warm_spare_guard") != {
                "status": "FAILED",
                "error": AUTOMATIC_GUARD_ERROR,
            }:
                errors.append(
                    "deployed automatic-recovery guard returned an unexpected result"
                )
            control = automatic_probe.get("control_without_warm_spare_strategy") or {}
            control_error = str(control.get("error") or "")
            if (
                control.get("status") != "FAILED"
                or control_error == AUTOMATIC_GUARD_ERROR
                or "HyperPod automatic node recovery is enabled" not in control_error
            ):
                # The refusal has to be attributable to the NodeRecovery the
                # deployed preflight read. A probe that fails for any reason at
                # all, or one whose warm-spare message appears without the
                # strategy that gates it, proves nothing.
                errors.append(
                    "deployed control probe did not attribute the refusal to "
                    "the observed NodeRecovery"
                )
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


def automatic_cluster_recovery(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The negative cluster's recovery mode, read off the recorded snapshot.

    The snapshot already carries the ``DescribeCluster`` payload the deployed
    probe replays, so a second ``describe-cluster`` for the same three fields
    was a duplicate read -- and one that could disagree with what the probe
    was actually judged on.
    """

    described = (snapshot.get("payloads") or {}).get("describe_cluster") or {}
    return {
        "cluster_name": described.get("ClusterName"),
        "status": described.get("ClusterStatus"),
        "node_recovery": described.get("NodeRecovery"),
    }


class _Probes:
    """Every live read of one audit, each guarded on its own.

    One ``try`` around all of them meant a single probe failure -- an executor
    Pod that would not exec, a cluster that would not describe -- failed all
    three cases with the same error, and left the probes after it unrun. Each
    read now records its own failure against the cases that depend on it.
    """

    def __init__(self, selected_cases: list[str]) -> None:
        self.selected = list(selected_cases)
        self.errors: dict[str, list[str]] = {case: [] for case in selected_cases}

    def guarded(
        self,
        read: Callable[[], Any],
        *,
        cases: list[str] | None = None,
    ) -> Any | None:
        try:
            return read()
        except Exception as exc:  # noqa: BLE001 - recorded against its cases
            message = f"{type(exc).__name__}: {exc}"
            for case in cases if cases is not None else self.selected:
                if case in self.errors:
                    self.errors[case].append(message)
            return None


def run_audit(
    run_dir: Path, selected_cases: list[str], *, identity: dict[str, str] | None = None
) -> int:
    definitions = case_definitions()
    started_at = datetime.now(timezone.utc)
    for case_id in selected_cases:
        write_json(
            run_dir / "cases" / case_id / f"{case_id}.json",
            {
                "case_id": case_id,
                "verdict": "FAIL",
                "status": "RUNNING",
                "formal_sequence_satisfied": False,
            },
        )
    baseline = node_snapshot()
    preflight_errors = node_preflight_errors(baseline)
    write_json(run_dir / "gpu-node-baseline.json", baseline)
    for case_id in selected_cases:
        (run_dir / "cases" / case_id).mkdir(parents=True, exist_ok=True, mode=0o700)

    probes = _Probes(selected_cases)
    if identity is not None and preflight_errors:
        for case_id in selected_cases:
            write_json(
                run_dir / "cases" / case_id / f"{case_id}.json",
                {
                    "case_id": case_id,
                    "verdict": "FAIL",
                    **identity,
                    "errors": preflight_errors,
                    "formal_sequence_satisfied": False,
                },
            )
        return 1
    # The focused pytest first and once: it is independent of every live probe,
    # and running it afterwards meant one probe exception left it unrun and
    # reported as "focused pytest failed" -- a deployment problem disguised as
    # a repo-side regression.
    test_results: dict[str, bool] = (
        probes.guarded(
            lambda: run_focused_pytest(
                run_dir, {case_id: definitions[case_id] for case_id in selected_cases}
            ),
            cases=[],
        )
        or {}
    )
    gpu: dict[str, Any] = (
        probes.guarded(lambda: cluster_recovery(MANAGED_GPU_CLUSTER)) or {}
    )
    env: list[dict[str, Any]] = probes.guarded(executor_env) or []
    if identity is not None and any(
        item.get("cluster_id") != identity["cluster_id"] for item in env
    ):
        for errors in probes.errors.values():
            errors.append("executor cluster identity differs from the durable registry")
    automatic: dict[str, Any] | None = None
    automatic_probe: dict[str, Any] | None = None
    managed_probe: dict[str, Any] | None = None
    executor_probes: dict[str, Any] | None = None
    if "GF-REGIONAL-DESTR-006" in selected_cases:
        only = ["GF-REGIONAL-DESTR-006"]
        snapshot = probes.guarded(
            lambda: record_provider_snapshot(AUTOMATIC_NEGATIVE_CLUSTER), cases=only
        )
        if snapshot is not None:
            write_json(run_dir / "automatic-negative-cluster-payloads.json", snapshot)
            automatic = automatic_cluster_recovery(snapshot)
            automatic_probe = probes.guarded(
                lambda: deployed_automatic_recovery_probe(snapshot), cases=only
            )
    if "GF-REGIONAL-DESTR-005" in selected_cases:
        managed_probe = probes.guarded(
            deployed_managed_owner_probe, cases=["GF-REGIONAL-DESTR-005"]
        )
    if "GF-REGIONAL-DESTR-007" in selected_cases:
        executor_probes = probes.guarded(
            deployed_executor_guard_probes, cases=["GF-REGIONAL-DESTR-007"]
        )

    ended_at = datetime.now(timezone.utc)
    events: list[dict[str, Any]] | None = probes.guarded(
        lambda: replace_events(started_at, ended_at)
    )
    provisional = cloudtrail_provisional(ended_at)
    postflight: list[dict[str, Any]] = []
    drift_errors: list[str] = []

    def postflight_read() -> None:
        nonlocal postflight, drift_errors
        postflight = node_snapshot()
        drift_errors = node_state_drift(baseline, postflight)
        write_json(run_dir / "gpu-node-postflight.json", postflight)

    probes.guarded(postflight_read)
    post_env = probes.guarded(executor_env)
    if post_env != env:
        for errors in probes.errors.values():
            errors.append(
                "executor population or safety environment changed during audit"
            )
    if identity is not None and probes.guarded(audit_identity) != identity:
        for errors in probes.errors.values():
            errors.append("release or durable registry identity changed during audit")

    failed = False
    for case_id in selected_cases:
        errors = list(preflight_errors)
        if case_id not in test_results:
            errors.append("focused pytest did not run")
        elif not test_results[case_id]:
            errors.append("focused pytest failed")
        errors.extend(probes.errors[case_id])
        if events:
            errors.append("CloudTrail contains an unexpected provider mutation")
        errors.extend(drift_errors)
        errors.extend(
            probe_errors(
                case_id,
                managed_probe if case_id == "GF-REGIONAL-DESTR-005" else None,
                executor_probes if case_id == "GF-REGIONAL-DESTR-007" else None,
                automatic_probe if case_id == "GF-REGIONAL-DESTR-006" else None,
            )
        )
        if gpu.get("node_recovery") != "None":
            errors.append("managed GPU cluster NodeRecovery is not None")
        if case_id == "GF-REGIONAL-DESTR-006":
            if automatic is None or automatic["node_recovery"] != "Automatic":
                errors.append("negative cluster is not Automatic")
        if case_id == "GF-REGIONAL-DESTR-007":
            if not env:
                errors.append("complete Ready executor population is missing")
            if any(
                item.get("spare_failover") != "true"
                or item.get("remote_state") != "true"
                or item.get("allow_replace") != "false"
                for item in env
            ):
                errors.append("production executor safety environment is inconsistent")
        result: dict[str, Any] = {
            "case_id": case_id,
            "status": "COMPLETED" if not errors else "FAILED",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **(identity or {}),
            **current_acceptance_scope().result_fields(),
            "formal_sequence_satisfied": identity is not None
            and not current_acceptance_scope().selective,
            "verdict": "PASS" if not errors else "FAIL",
            "acceptance_incomplete": bool(preflight_errors or drift_errors),
            "errors": errors,
            "focused_pytest": definitions[case_id],
            "focused_pytest_log": str(run_dir / "cases" / "pytest.log"),
            "gpu_cluster": gpu,
            "executor_env": env,
            # The negative claim is provisional while CloudTrail may still be
            # delivering; DESTR-013 re-reads the whole run's window later.
            "replace_events": events if events is not None else [],
            "replace_events_provisional": provisional,
            "node_baseline": baseline,
            "node_postflight": postflight,
            "node_state_identical": bool(baseline and postflight) and not drift_errors,
        }
        # Each case carries only its own probe; the shared baseline/postflight
        # and cluster reads are the same for all three and are kept.
        if case_id == "GF-REGIONAL-DESTR-005":
            result["deployed_managed_owner_probe"] = managed_probe
        if case_id == "GF-REGIONAL-DESTR-006":
            result["automatic_negative_cluster"] = automatic
            result["deployed_automatic_recovery_probe"] = automatic_probe
        if case_id == "GF-REGIONAL-DESTR-007":
            result["deployed_executor_guard_probes"] = executor_probes
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
    result.add_argument("--predecessor-evidence", default="")
    return result


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    if len(arguments.case) != 1:
        raise RuntimeError(
            "select exactly one explicit --case for this read-only live audit"
        )
    selected_cases = arguments.case
    configured = os.getenv("GPU_FAULT_ACCEPTANCE_RUN_DIR", "").strip()
    run_dir = arguments.run_dir or (Path(configured) if configured else None)
    if run_dir is None:
        raise RuntimeError(
            "an explicit audit --run-dir or GPU_FAULT_ACCEPTANCE_RUN_DIR is required"
        )
    case_id = selected_cases[0]
    result_path = run_dir / "cases" / case_id / f"{case_id}.json"
    write_json(
        result_path,
        {
            "case_id": case_id,
            "verdict": "FAIL",
            "status": "RUNNING",
            "formal_sequence_satisfied": False,
        },
    )
    try:
        bind_command_supervision(run_dir)
        configure(arguments, set(selected_cases))
        identity = audit_identity()
        previous_id, path = predecessor_path(
            run_dir, case_id, arguments.predecessor_evidence
        )
        if previous_id is not None and path is not None:
            previous = predecessor_evidence(
                path,
                previous_id,
                release_id=identity["release_id"],
                cluster_id=identity["cluster_id"],
            )
            if previous.get("valid") is not True:
                raise RuntimeError(
                    "formal predecessor did not pass for this release and cluster"
                )
        return run_audit(run_dir, selected_cases, identity=identity)
    except BaseException as exc:
        write_json(
            result_path,
            {
                "case_id": case_id,
                "verdict": "FAIL",
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "formal_sequence_satisfied": False,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
