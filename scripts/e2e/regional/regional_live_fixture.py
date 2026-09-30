from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.acceptance_scope import (  # noqa: E402
    FORMAL_SCOPE,
    SELECTIVE_SCOPE,
    current_acceptance_scope,
)
from scripts.e2e.regional.kmsg_clock import marker_observed_after  # noqa: E402
from scripts.e2e.regional.regional_commands import (  # noqa: E402
    RegionalCommandTimeout as RegionalCommandTimeout,
    RegionalFixtureError as RegionalFixtureError,
    run_fixture_command,
)
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records  # noqa: E402
from scripts.e2e.regional.regional_store_probe import STORE_PROBE as STORE_PROBE  # noqa: E402
from scripts.e2e.regional.regional_workload_inventory import (  # noqa: E402
    SYSTEM_NAMESPACES as SYSTEM_NAMESPACES,
    business_workload_items as business_workload_items,
    pod_gpu_count as pod_gpu_count,
)

TERMINAL_WORKFLOW_STATUSES = {"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}
PROVIDER_MUTATIONS = {
    "BatchDeleteClusterNodes",
    "BatchRebootClusterNodes",
    "BatchReplaceClusterNodes",
    "DeleteClusterNodes",
    "RebootClusterNodes",
    "ReplaceClusterNodes",
}
RUNTIME_IDENTITY_DEPLOYMENTS = {
    "cpu": (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
    ),
    "gpu": (
        "gpu-fault-cluster-executor",
        "gpu-fault-completion-watcher",
        "gpu-fault-kubernetes-node-resource-collector",
    ),
}
# CloudTrail delivers management events eventually, typically within 5 minutes
# and documented at up to 15. A lookup that returns nothing inside that window
# has not shown that nothing happened.
PROVIDER_EVENT_VISIBILITY_SECONDS = 900


def component_python(plane: str) -> str:
    if plane not in {"cpu", "gpu"}:
        raise ValueError(f"unknown Kubernetes plane: {plane}")
    component = "control-plane" if plane == "cpu" else "executor"
    return f"/opt/gpu-fault/{component}/bin/python"


RELEASE_STATE_IDENTITY_FIELDS = (
    "release_id",
    "phase",
    "transaction_committed",
    "updated_at_epoch",
    "wheel_sha256",
    "executor_wheel_sha256",
    "runtime_image",
    "runtime_profile_version",
    "runtime_profile_sha256",
    "cpu_manifest_sha256",
    "executor_manifest_sha256",
    "cluster_registry_digest",
)


def provider_event_actor_matches_role(
    event: dict[str, str],
    expected_role_arn: str,
) -> bool:
    expected_role_name = expected_role_arn.rsplit("/", 1)[-1]
    session_issuer_role_name = event.get("session_issuer_role_name", "")
    if session_issuer_role_name:
        return session_issuer_role_name == expected_role_name
    return expected_role_name in event.get("username", "")


def runtime_identity_errors(value: dict[str, Any]) -> list[str]:
    errors = []
    release_state = value.get("release_state")
    if not isinstance(release_state, dict):
        errors.append("regional release identity is unavailable")
    else:
        phase = str(release_state.get("phase") or "").strip().lower()
        if phase.startswith("rollback-") or phase == "rolling-back":
            errors.append(f"regional release rollback is active at phase {phase}")
    deployments = value.get("deployments")
    if not isinstance(deployments, dict):
        errors.append("runtime deployment identity is unavailable")
        return errors
    for plane, names in RUNTIME_IDENTITY_DEPLOYMENTS.items():
        plane_deployments = deployments.get(plane)
        if not isinstance(plane_deployments, dict):
            errors.append(f"{plane} runtime deployment identity is unavailable")
            continue
        for name in names:
            item = plane_deployments.get(name)
            if not isinstance(item, dict):
                errors.append(f"{plane}/{name} runtime deployment is unavailable")
                continue
            generation = int(item.get("generation") or 0)
            desired = int(item.get("desired_replicas") or 0)
            observed = int(item.get("observed_generation") or 0)
            updated = int(item.get("updated_replicas") or 0)
            ready = int(item.get("ready_replicas") or 0)
            available = int(item.get("available_replicas") or 0)
            if (
                desired <= 0
                or observed != generation
                or updated != desired
                or ready != desired
                or available != desired
            ):
                errors.append(f"{plane}/{name} is not fully rolled out")
    return errors


class RegionalFixtureAbort(BaseException):
    """An operator abort delivered by SIGINT/SIGTERM.

    Deliberately *not* an ``Exception``. Every probe helper in this fixture
    retries on a bare ``except Exception`` -- ``pod_python`` for three attempts,
    the wait loops for their whole timeout -- so a signal raised as an
    ``Exception`` is swallowed by the very loop the operator is trying to
    interrupt, and ^C appears to do nothing until the timeout expires. On a
    destructive case that gap is the difference between an abort and a
    completed mutation.
    """

    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


def abort_on_signal(signum: int, _frame: object) -> None:
    raise RegionalFixtureAbort(signum)


def install_abort_signals() -> None:
    signal.signal(signal.SIGTERM, abort_on_signal)
    signal.signal(signal.SIGINT, abort_on_signal)


def run_case_main(entry: Callable[[], int]) -> int:
    """Run a case entry point so a signal abort exits instead of tracebacking.

    ``RegionalFixtureAbort`` has to escape the retry loops, which means it also
    escapes ``main``. Reporting it as ``128 + signum`` keeps the shell contract
    an operator expects from ^C while still distinguishing an abort from a
    ``FAIL`` verdict, which exits 1.
    """

    try:
        return entry()
    except RegionalFixtureAbort as abort:
        print(f"aborted: {abort}", file=sys.stderr)
        return 128 + abort.signum


def required(value: str, label: str) -> str:
    result = value.strip()
    if not result:
        raise RegionalFixtureError(f"{label} is required")
    return result


def predecessor_identity_errors(
    value: dict[str, Any],
    *,
    release_id: str | None,
    cluster_id: str | None,
) -> list[str]:
    """Why ``value`` is not evidence from this release and cluster.

    A PASS is only a predecessor if it was earned against the deployment the
    successor is about to mutate. A run directory reused across a redeploy, or
    an evidence file copied from another site, otherwise satisfies the chain
    while proving nothing about the code that is live now.
    """

    errors = []
    for field, expected in (("release_id", release_id), ("cluster_id", cluster_id)):
        if expected is None:
            continue
        actual = value.get(field)
        if not isinstance(expected, str) or not expected.strip():
            errors.append(f"expected {field} is empty or invalid")
        elif actual in (None, ""):
            errors.append(f"{field} missing from predecessor evidence")
        elif not isinstance(actual, str) or actual != expected:
            errors.append(f"{field} mismatch")
    return errors


def _predecessor_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate predecessor evidence field")
        value[key] = item
    return value


def _invalid_predecessor_constant(_value: str) -> NoReturn:
    raise ValueError("non-finite predecessor evidence value")


def _read_predecessor_evidence(
    path: Path,
) -> tuple[dict[str, Any] | None, str, str | None, str | None]:
    try:
        if not path.is_file():
            return None, "MISSING", "predecessor evidence does not exist", None
        data = path.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_predecessor_object,
            parse_constant=_invalid_predecessor_constant,
        )
    except (OSError, ValueError, RecursionError) as exc:
        return (
            None,
            "INVALID",
            f"cannot read predecessor evidence: {type(exc).__name__}",
            None,
        )
    if not isinstance(value, dict):
        return None, "INVALID", "predecessor evidence is not a JSON object", None
    return value, "READ", None, hashlib.sha256(data).hexdigest()


def read_predecessor_evidence(
    path: Path,
    expected_case_id: str,
    *,
    release_id: str | None = None,
    cluster_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return one parsed document and its scope-independent validation facts.

    Consumers reusing recorded steps must check ``facts["evidence_valid"]``
    before using the returned document; rereading the path would lose that
    binding. Omitted identity arguments retain the legacy unbound contract.
    """

    value, code, read_error, digest = _read_predecessor_evidence(path)
    facts: dict[str, Any] = {
        "evidence_valid": False,
        "evidence_case_id": None,
        "evidence_verdict": None,
        "evidence_execution_scope": None,
        "evidence_release_id": None,
        "evidence_cluster_id": None,
        "expected_release_id": release_id,
        "expected_cluster_id": cluster_id,
        "evidence_error": read_error,
        "evidence_read_status": code,
        "evidence_sha256": digest,
    }
    if value is None:
        return None, facts
    actual_case_id = value.get("case_id")
    verdict = value.get("verdict")
    evidence_scope = value.get("execution_scope", FORMAL_SCOPE)
    completed = "status" not in value or value["status"] == "COMPLETED"
    identity_errors = predecessor_identity_errors(
        value,
        release_id=release_id,
        cluster_id=cluster_id,
    )
    if (
        not isinstance(expected_case_id, str)
        or not expected_case_id.strip()
        or actual_case_id != expected_case_id
        or verdict != "PASS"
    ):
        error: str | None = "predecessor case must have verdict PASS"
    elif not completed:
        error = "predecessor execution must be completed"
    elif identity_errors:
        error = "; ".join(identity_errors)
    elif evidence_scope not in (FORMAL_SCOPE, SELECTIVE_SCOPE):
        error = "predecessor execution scope is invalid"
    elif "formal_sequence_satisfied" in value and not isinstance(
        value["formal_sequence_satisfied"], bool
    ):
        error = "predecessor formal sequence flag must be a boolean"
    else:
        error = None
    facts.update(
        {
            "evidence_valid": error is None,
            "evidence_case_id": actual_case_id,
            "evidence_verdict": verdict,
            "evidence_execution_scope": evidence_scope,
            "evidence_release_id": value.get("release_id"),
            "evidence_cluster_id": value.get("cluster_id"),
            "evidence_error": error,
        }
    )
    return value, facts


def predecessor_evidence_facts(
    path: Path,
    expected_case_id: str,
    *,
    release_id: str | None = None,
    cluster_id: str | None = None,
) -> dict[str, Any]:
    """Describe a completed bound PASS independently of the sequence gate."""

    _value, facts = read_predecessor_evidence(
        path, expected_case_id, release_id=release_id, cluster_id=cluster_id
    )
    return facts


def predecessor_evidence(
    path: Path,
    expected_case_id: str,
    *,
    release_id: str | None = None,
    cluster_id: str | None = None,
) -> dict[str, Any]:
    """Apply the sequence gate to the same document that supplies its facts.

    Selective scope waives only the sequence gate. It does not make missing,
    incomplete or foreign predecessor evidence safe to reuse.
    """

    scope = current_acceptance_scope()
    value, facts = read_predecessor_evidence(
        path, expected_case_id, release_id=release_id, cluster_id=cluster_id
    )
    if scope.selective:
        return {
            "path": str(path),
            "case_id": expected_case_id,
            "expected_case_id": expected_case_id,
            "verdict": "SKIPPED_BY_OPERATOR",
            "status": "SKIPPED_BY_OPERATOR",
            "valid": True,
            "execution_allowed": True,
            **facts,
            **scope.result_fields(),
            "error": None,
        }
    evidence_scope = facts["evidence_execution_scope"]
    formal_sequence_satisfied = value is not None and (
        value.get("formal_sequence_satisfied", evidence_scope == FORMAL_SCOPE) is True
    )
    valid = bool(
        facts["evidence_valid"]
        and evidence_scope == FORMAL_SCOPE
        and formal_sequence_satisfied
    )
    error = facts["evidence_error"]
    if error is None and not valid:
        error = "selective evidence cannot satisfy a formal predecessor"
    return {
        "path": str(path),
        "case_id": facts["evidence_case_id"] if value is not None else expected_case_id,
        "expected_case_id": expected_case_id,
        "verdict": (
            facts["evidence_verdict"]
            if value is not None
            else facts["evidence_read_status"]
        ),
        "valid": valid,
        "execution_allowed": valid,
        **facts,
        **scope.plan_fields(),
        "formal_sequence_satisfied": valid,
        "error": error,
    }


def waiting_step_executions(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """The WAITING step execution records, projected to what evidence keeps.

    ``started_at``/``updated_at`` ride along when the record has them -- the
    per-step waiting cap counts from when the step parked, and DESTR-017 reads
    that off this projection -- and are omitted when it does not, so a record
    without timestamps is projected exactly as before.
    """

    result = []
    for item in workflow.get("step_executions") or []:
        if not isinstance(item, dict) or item.get("status") != "WAITING":
            continue
        projected = {
            key: item.get(key)
            for key in (
                "step_index",
                "operation",
                "status",
                "adapter_operation_id",
                "details",
                "error",
            )
        }
        for key in ("started_at", "updated_at"):
            if item.get(key) is not None:
                projected[key] = item[key]
        result.append(projected)
    return result


class WaitingEvidence:
    """WAITING step executions seen across every store read a case makes.

    The workflow keeps one execution record per step, so a step's WAITING
    record -- the one carrying `mutation_submitted_by_control_plane: False` --
    exists only until the step succeeds. `wait_for_workflow` accumulates those
    while it polls, but a case that does its own polling first (waiting for a
    lease to appear, timing a failover) reaches `wait_for_workflow` after the
    early steps have already been replaced by their SUCCEEDED record, and the
    evidence check then reports them missing although they happened. Feed
    every snapshot through `observe` and `merged_into` the final state before
    evaluating it.
    """

    def __init__(self) -> None:
        self._seen: dict[tuple[int, str], dict[str, Any]] = {}

    def observe(self, snapshot: dict[str, Any]) -> None:
        for execution in waiting_step_executions(snapshot.get("workflow") or {}):
            step_index = execution.get("step_index")
            self._seen[
                (
                    step_index if isinstance(step_index, int) else -1,
                    str(execution.get("operation") or ""),
                )
            ] = execution

    def evidence(self) -> list[dict[str, Any]]:
        return [self._seen[key] for key in sorted(self._seen)]

    def merged_into(self, state: dict[str, Any]) -> dict[str, Any]:
        """The state with its own WAITING evidence plus everything seen here.

        The later observation wins for a step seen twice; both carry the same
        control-plane details, so the choice only affects timestamps.
        """

        combined = WaitingEvidence()
        combined.observe({"workflow": {"step_executions": self.evidence()}})
        combined.observe(
            {
                "workflow": {
                    "step_executions": (
                        state.get("observed_waiting_step_executions") or []
                    )
                }
            }
        )
        return {**state, "observed_waiting_step_executions": combined.evidence()}


@dataclass(frozen=True)
class RegionalLiveSettings:
    cpu_kubeconfig: Path
    gpu_kubeconfig: Path
    gpu_context: str
    namespace: str
    cluster_id: str
    region: str

    def __post_init__(self) -> None:
        if not self.cpu_kubeconfig.is_file():
            raise ValueError("CPU kubeconfig does not exist")
        if not self.gpu_kubeconfig.is_file():
            raise ValueError("GPU kubeconfig does not exist")
        if not all(
            (
                self.gpu_context,
                self.namespace,
                self.cluster_id,
                self.region,
            )
        ):
            raise ValueError("regional live settings contain an empty identity")

    def environment(self) -> dict[str, str]:
        return {
            "CPU_KUBECONFIG": str(self.cpu_kubeconfig),
            "GPU_KUBECONFIG": str(self.gpu_kubeconfig),
            "GPU_EKS_CONTEXT": self.gpu_context,
            "GPU_FAULT_NAMESPACE": self.namespace,
            "GPU_FAULT_CLUSTER_ID": self.cluster_id,
            "AWS_REGION": self.region,
        }


def settings_from_arguments(arguments: Any) -> RegionalLiveSettings:
    cpu = (
        Path(
            required(
                arguments.cpu_kubeconfig
                or os.getenv("GPU_FAULT_CONTROL_KUBECONFIG", "")
                or os.getenv("CPU_KUBECONFIG", ""),
                "CPU kubeconfig",
            )
        )
        .expanduser()
        .resolve()
    )
    gpu = (
        Path(
            required(
                arguments.gpu_kubeconfig
                or os.getenv("GPU_KUBECONFIG", "")
                or os.getenv("KUBECONFIG", ""),
                "GPU kubeconfig",
            )
        )
        .expanduser()
        .resolve()
    )
    return RegionalLiveSettings(
        cpu_kubeconfig=cpu,
        gpu_kubeconfig=gpu,
        gpu_context=required(
            arguments.gpu_context
            or os.getenv("GPU_EKS_CONTEXT", "")
            or os.getenv("GPU_FAULT_DATAPLANE_CONTEXT", ""),
            "GPU context",
        ),
        namespace=required(arguments.namespace, "namespace"),
        cluster_id=required(
            arguments.cluster_id or os.getenv("GPU_FAULT_CLUSTER_ID", ""),
            "cluster ID",
        ),
        region=required(
            arguments.region
            or os.getenv("AWS_REGION", "")
            or os.getenv("AWS_DEFAULT_REGION", ""),
            "AWS Region",
        ),
    )


EXECUTOR_XID_POST = r"""
import json
import os
import sys
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from gpu_fault.collectors.sinks import HttpEventSink

payload = json.loads(sys.argv[1])
os.environ["SSL_CERT_FILE"] = os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
sink = HttpEventSink(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
    bearer_token=os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
    timeout_seconds=15,
    max_attempts=1,
)
accepted = sink.post("/v1/collector-events/nvidia-kernel", payload)
request_id = accepted.get("processor_request_id")
receipt = None
if request_id:
    base = os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
    headers = {
        "Authorization": "Bearer " + os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
        "X-GPU-Fault-Cluster-ID": os.environ["GPU_FAULT_CLUSTER_ID"],
    }
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        request = Request(
            base + "/v1/processor/requests/" + quote(request_id, safe=""),
            headers=headers,
        )
        try:
            with urlopen(request, timeout=15) as response:
                body = response.read()
                receipt = {
                    "status": response.status,
                    "body": json.loads(body) if body else {},
                }
                if response.status != 202:
                    break
        except HTTPError as exc:
            body = exc.read()
            receipt = {
                "status": exc.code,
                "body": json.loads(body) if body else {},
            }
            break
        time.sleep(2)
print(json.dumps({
    "accepted": accepted,
    "receipt": receipt,
}, sort_keys=True))
"""


NODE_WORKLOAD_VIEW_SCRIPT = r"""
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext

context = ApplicationContext.from_environment()
resolved = context.topology.resolve(sys.argv[1], sys.argv[2], datetime.now(timezone.utc))
state = getattr(resolved.workload_state, "value", resolved.workload_state)
print(
    json.dumps(
        {
            "workload_state": str(state),
            "attempt_ids": list(resolved.attempt_ids),
            "workload_ids": list(resolved.workload_ids),
        }
    )
)
"""


SPARE_POOL_ANNOTATION_PREFIX = "gpu-fault.io/spare-"


def project_node_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    """The runner's view of one Node document.

    ``ownership_annotations`` are the marks an incident leaves while it holds
    the node (incident-id, fencing-token, previous-unschedulable, ...); the
    installer's identity annotations and the warm-spare pool's bookkeeping
    (``gpu-fault.io/spare-*``: a released spare keeps ``spare-pool-state:
    AVAILABLE``) are not ownership and are reported apart, so a validated
    restore is not judged "still isolated" by a node nothing holds (live
    COLLECT-016, 2026-09-24).
    """

    annotations = value["metadata"].get("annotations", {})
    return {
        "name": value["metadata"]["name"],
        "uid": value["metadata"]["uid"],
        "boot_id": value.get("status", {}).get("nodeInfo", {}).get("bootID"),
        "ready": next(
            (
                condition["status"]
                for condition in value["status"].get("conditions", [])
                if condition["type"] == "Ready"
            ),
            None,
        ),
        # When kubelet last flipped Ready: a node that just booted flips it
        # again ~30 s later when HyperPod restarts kubelet with its final
        # config, and an exec into a probe Pod across that restart fails.
        "ready_since": next(
            (
                condition.get("lastTransitionTime")
                for condition in value["status"].get("conditions", [])
                if condition["type"] == "Ready"
            ),
            None,
        ),
        "installer_boot_id": annotations.get("gpu-fault.io/installer-boot-id"),
        "unschedulable": value["spec"].get("unschedulable", False),
        "taints": value["spec"].get("taints", []),
        "gpu_allocatable": value["status"].get("allocatable", {}).get("nvidia.com/gpu"),
        "ownership_annotations": {
            key: item
            for key, item in annotations.items()
            if key.startswith("gpu-fault.io/")
            and not key.startswith("gpu-fault.io/installer-")
            and not key.startswith(SPARE_POOL_ANNOTATION_PREFIX)
        },
        "spare_pool_annotations": {
            key: item
            for key, item in annotations.items()
            if key.startswith(SPARE_POOL_ANNOTATION_PREFIX)
        },
    }


class RegionalLiveFixture:
    def __init__(self, settings: RegionalLiveSettings) -> None:
        self.settings = settings

    @staticmethod
    def run(
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        timeout: int = 300,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return run_fixture_command(
            command,
            input_text=input_text,
            check=check,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )

    def kubectl(
        self,
        plane: str,
        *arguments: str,
        input_text: str | None = None,
        check: bool = True,
        timeout: int = 300,
        all_namespaces: bool = False,
        namespace: str | None = None,
    ) -> str:
        command = ["kubectl", "--kubeconfig"]
        if plane == "cpu":
            command.append(str(self.settings.cpu_kubeconfig))
        elif plane == "gpu":
            command.extend(
                [
                    str(self.settings.gpu_kubeconfig),
                    "--context",
                    self.settings.gpu_context,
                ]
            )
        else:
            raise ValueError(f"unknown Kubernetes plane: {plane}")
        if all_namespaces and namespace is not None:
            raise ValueError("all_namespaces and namespace are mutually exclusive")
        if all_namespaces:
            command.extend(arguments)
            command.append("--all-namespaces")
        else:
            command.extend(["-n", namespace or self.settings.namespace])
            command.extend(arguments)
        return self.run(
            command,
            input_text=input_text,
            check=check,
            timeout=timeout,
        ).stdout

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        value = json.loads(
            self.kubectl(
                plane,
                "get",
                "pod",
                "-l",
                f"app={app}",
                "-o",
                "json",
            )
        )
        return ready_pod_records(value)

    def ready_pod(self, plane: str, app: str) -> str:
        pods = self.ready_pods(plane, app)
        if not pods:
            raise RegionalFixtureError(f"no Ready {plane} Pod for app={app}")
        return str(pods[0]["name"])

    def pod_python(
        self,
        plane: str,
        app: str,
        script: str,
        *arguments: str,
        timeout: int = 180,
        attempts: int = 3,
    ) -> dict[str, Any]:
        """Run ``script`` under the installed component interpreter in a Ready Pod.

        ``attempts`` is the retry budget for transient exec failures. The
        default suits read-only probes; a script that mutates -- posts an event,
        releases a spare -- must pass ``attempts=1``, because a retry after a
        failure that happened *after* the mutation performs it again. A timeout
        is never retried whatever the budget: the first attempt may have run to
        completion and only the receipt was lost.
        """

        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        last_error: Exception | None = None
        for _attempt in range(attempts):
            try:
                output = self.kubectl(
                    plane,
                    "exec",
                    "-i",
                    self.ready_pod(plane, app),
                    "--",
                    component_python(plane),
                    "-",
                    *arguments,
                    input_text=script,
                    timeout=timeout,
                )
                value = json.loads(output.splitlines()[-1])
                if not isinstance(value, dict):
                    raise RegionalFixtureError("Pod probe did not return a JSON object")
                return cast(dict[str, Any], value)
            except RegionalCommandTimeout as exc:
                raise RegionalFixtureError(
                    f"{plane} Pod probe timed out; not retried because the script "
                    f"may already have run: {exc}"
                ) from exc
            except Exception as exc:
                last_error = exc
                if _attempt + 1 < attempts:
                    time.sleep(1)
        raise RegionalFixtureError(f"{plane} Pod probe failed: {last_error}")

    def cpu_python(
        self,
        script: str,
        *arguments: str,
        timeout: int = 180,
        attempts: int = 3,
    ) -> dict[str, Any]:
        return self.pod_python(
            "cpu",
            "gpu-fault-api-ha",
            script,
            *arguments,
            timeout=timeout,
            attempts=attempts,
        )

    def executor_python(
        self,
        script: str,
        *arguments: str,
        timeout: int = 180,
        attempts: int = 3,
    ) -> dict[str, Any]:
        return self.pod_python(
            "gpu",
            "gpu-fault-cluster-executor",
            script,
            *arguments,
            timeout=timeout,
            attempts=attempts,
        )

    def store_snapshot(
        self,
        *,
        node: str = "",
        marker: str = "",
        observed_after: datetime | None = None,
        job_id: str = "",
        attempt_id: str = "",
        hyperpod_cluster: str = "",
        queue_attempts: int = 20,
        workflow_request_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """One store read; `queue_attempts=1` for wait loops, the default for gates.

        The default drains the processor queue for up to ~10s so a preflight
        does not refuse a healthy cluster on in-flight rows. A loop that polls
        the workflow does not need that reading and cannot afford it: with it,
        samples land 15s apart and a ten-second step's WAITING record is never
        seen.

        ``workflow_request_ids`` narrows the ``commands`` read to those
        workflows in the store itself. Without it the probe returns the
        marker's own workflow's commands, as it always has.
        """

        bound = marker_observed_after(marker, observed_after)
        arguments = [
            self.settings.cluster_id,
            node,
            marker,
            bound.isoformat() if bound is not None else "",
            job_id,
            attempt_id,
            hyperpod_cluster,
            str(queue_attempts),
        ]
        if workflow_request_ids:
            for request_id in workflow_request_ids:
                if "," in request_id or not request_id:
                    raise ValueError("workflow request IDs must be non-empty, no comma")
            arguments.append(",".join(workflow_request_ids))
        result = self.cpu_python(STORE_PROBE, *arguments)
        if not result.get("release_id"):
            result["release_id"] = self.release_id()
        return result

    def evidence_identity(self) -> dict[str, str]:
        """The release and cluster a case's evidence must be bound to.

        Written into every case result (``result.update(...)``) so the next case
        in the chain can pass the same values to ``predecessor_evidence`` and
        refuse a PASS that was earned against another deployment.
        """

        return {
            "release_id": self.release_id(),
            "cluster_id": self.settings.cluster_id,
        }

    def release_id(self) -> str:
        value = json.loads(
            self.kubectl(
                "cpu",
                "get",
                "configmap",
                "gpu-fault-regional-release-state",
                "-o",
                "json",
            )
        )
        state = json.loads(value["data"]["state.json"])
        return str(state.get("release_id") or "")

    def runtime_identity(self) -> dict[str, Any]:
        release_document = json.loads(
            self.kubectl(
                "cpu",
                "get",
                "configmap",
                "gpu-fault-regional-release-state",
                "-o",
                "json",
            )
        )
        try:
            release_state = json.loads(release_document["data"]["state.json"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RegionalFixtureError(
                "regional release state is not valid JSON"
            ) from exc
        if not isinstance(release_state, dict):
            raise RegionalFixtureError("regional release state is not an object")

        deployments: dict[str, dict[str, Any]] = {}
        for plane, names in RUNTIME_IDENTITY_DEPLOYMENTS.items():
            value = json.loads(
                self.kubectl(
                    plane,
                    "get",
                    "deployment",
                    *names,
                    "-o",
                    "json",
                )
            )
            items = {
                str(item.get("metadata", {}).get("name") or ""): item
                for item in value.get("items", [])
                if isinstance(item, dict)
            }
            if set(items) != set(names):
                missing = sorted(set(names) - set(items))
                raise RegionalFixtureError(
                    f"{plane} runtime deployments are missing: {missing}"
                )
            plane_identity: dict[str, Any] = {}
            for name in names:
                item = items[name]
                metadata = item.get("metadata") or {}
                spec = item.get("spec") or {}
                status = item.get("status") or {}
                template = spec.get("template") or {}
                template_spec = template.get("spec") or {}
                containers = [
                    *(template_spec.get("initContainers") or []),
                    *(template_spec.get("containers") or []),
                ]
                canonical = json.dumps(
                    template,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                plane_identity[name] = {
                    "generation": int(metadata.get("generation") or 0),
                    "desired_replicas": int(spec.get("replicas") or 0),
                    "observed_generation": int(status.get("observedGeneration") or 0),
                    "updated_replicas": int(status.get("updatedReplicas") or 0),
                    "ready_replicas": int(status.get("readyReplicas") or 0),
                    "available_replicas": int(status.get("availableReplicas") or 0),
                    "template_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                    "images": sorted(
                        str(container.get("image") or "")
                        for container in containers
                        if isinstance(container, dict)
                    ),
                }
            deployments[plane] = plane_identity
        return {
            "release_state": {
                field: release_state.get(field)
                for field in RELEASE_STATE_IDENTITY_FIELDS
            },
            "deployments": deployments,
        }

    def verify_runtime_identity(
        self,
        expected: dict[str, Any],
        *,
        evidence_path: Path,
        stage: str,
    ) -> dict[str, Any]:
        current = self.runtime_identity()
        write_json_atomic(evidence_path, current)
        if current != expected:
            raise RegionalFixtureError(
                f"{stage} release/runtime deployment identity drifted"
            )
        errors = runtime_identity_errors(current)
        if errors:
            raise RegionalFixtureError(
                f"{stage} runtime identity is unsafe: {'; '.join(errors)}"
            )
        return current

    def post_xid_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("cluster_id") != self.settings.cluster_id:
            raise RegionalFixtureError("XID payload cluster ID does not match settings")
        # One attempt: the sink itself is max_attempts=1, and a kubectl retry
        # after a lost receipt would post the same XID twice -- a second
        # collector event, a second incident, and a workflow the case did not
        # plan for.
        return self.executor_python(
            EXECUTOR_XID_POST,
            json.dumps(payload, sort_keys=True),
            timeout=240,
            attempts=1,
        )

    def wait_for_workflow(
        self,
        *,
        node: str,
        marker: str,
        observed_after: datetime,
        case_dir: Path,
        timeout_seconds: int,
        job_id: str = "",
        attempt_id: str = "",
        hyperpod_cluster: str = "",
        terminal: bool = True,
        workflow_request_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        timeline: list[dict[str, Any]] = []
        observed_waiting = WaitingEvidence()
        last: dict[str, Any] = {}
        snapshot_arguments: dict[str, Any] = {}
        if workflow_request_ids:
            snapshot_arguments["workflow_request_ids"] = list(workflow_request_ids)
        while time.monotonic() < deadline:
            last = self.store_snapshot(
                node=node,
                marker=marker,
                observed_after=observed_after,
                job_id=job_id,
                attempt_id=attempt_id,
                hyperpod_cluster=hyperpod_cluster,
                queue_attempts=1,
                **snapshot_arguments,
            )
            workflow = last.get("workflow") or {}
            current_waiting = waiting_step_executions(workflow)
            observed_waiting.observe(last)
            waiting_evidence = observed_waiting.evidence()
            timeline.append(
                {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "event_id": (last.get("event") or {}).get("event_id"),
                    "decision": (last.get("decision") or {}).get("disposition"),
                    "workflow_status": workflow.get("status"),
                    "completed_operations": workflow.get("completed_operations") or [],
                    "command_statuses": [
                        item.get("status") for item in last.get("commands") or []
                    ],
                    "waiting_step_executions": current_waiting,
                    "submission_state": ((last.get("submission") or {}).get("state")),
                }
            )
            write_json_atomic(
                case_dir / "timeline.json",
                {
                    "entries": timeline,
                    "observed_waiting_step_executions": waiting_evidence,
                },
            )
            status = workflow.get("status")
            if status and (not terminal or status in TERMINAL_WORKFLOW_STATUSES):
                return {
                    **last,
                    "observed_waiting_step_executions": waiting_evidence,
                }
            time.sleep(5)
        raise RegionalFixtureError(
            f"workflow did not reach the requested state: {last}"
        )

    def node_snapshot(self, node: str) -> dict[str, Any]:
        value = json.loads(self.kubectl("gpu", "get", "node", node, "-o", "json"))
        return project_node_snapshot(value)

    def node_workload_view(self, node: str) -> dict[str, Any]:
        """The control plane's own resolution of what runs on ``node`` now.

        ``kubectl get pods`` is not the product's view: the completion watcher
        re-posts a vanished attempt's last RUNNING observation for its
        missing-Pod grace, so a node can be empty and still bound to a dead
        attempt. A host-fault case that starts inside that window plans a
        STOP_WORKLOADS against a workload that no longer exists (NET-007).
        """

        return self.cpu_python(
            NODE_WORKLOAD_VIEW_SCRIPT, self.settings.cluster_id, node
        )

    def gpu_nodes(self) -> list[dict[str, Any]]:
        value = json.loads(self.kubectl("gpu", "get", "node", "-o", "json"))
        result = []
        for item in value.get("items", []):
            gpu_count = int(
                item.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0)
            )
            if gpu_count <= 0:
                continue
            result.append(
                {
                    "name": item["metadata"]["name"],
                    "uid": item["metadata"]["uid"],
                    "gpu_allocatable": gpu_count,
                    "ready": next(
                        (
                            condition["status"]
                            for condition in item["status"].get("conditions", [])
                            if condition["type"] == "Ready"
                        ),
                        None,
                    ),
                    "unschedulable": item["spec"].get("unschedulable", False),
                    "taints": item["spec"].get("taints", []),
                    "labels": item["metadata"].get("labels", {}),
                }
            )
        return sorted(result, key=lambda item: str(item["name"]))

    def node_metadata(self, node: str) -> dict[str, Any]:
        value = json.loads(self.kubectl("gpu", "get", "node", node, "-o", "json"))
        labels = value["metadata"].get("labels", {})
        return {
            "name": value["metadata"]["name"],
            "labels": labels,
            "product": next(
                (
                    labels[key]
                    for key in (
                        "nvidia.com/gpu.product",
                        "nvidia.com/gpu.machine",
                        "beta.kubernetes.io/instance-type",
                        "node.kubernetes.io/instance-type",
                    )
                    if labels.get(key)
                ),
                None,
            ),
        }

    @staticmethod
    def pod_gpu_count(item: dict[str, Any]) -> int:
        """The largest ``nvidia.com/gpu`` request or limit across containers."""

        return pod_gpu_count(item)

    def gpu_workloads(self) -> list[dict[str, Any]]:
        value = json.loads(
            self.kubectl(
                "gpu",
                "get",
                "pod",
                "-o",
                "json",
                all_namespaces=True,
            )
        )
        result = []
        for item in value.get("items", []):
            phase = item.get("status", {}).get("phase")
            if phase in {"Succeeded", "Failed"}:
                continue
            gpu_count = self.pod_gpu_count(item)
            if gpu_count <= 0:
                continue
            result.append(
                {
                    "namespace": item["metadata"].get("namespace"),
                    "name": item["metadata"].get("name"),
                    "node": item.get("spec", {}).get("nodeName"),
                    "phase": phase,
                    "gpu_count": gpu_count,
                }
            )
        return sorted(
            result,
            key=lambda item: (
                str(item["namespace"]),
                str(item["name"]),
            ),
        )

    def business_workloads(self, node: str) -> list[dict[str, str]]:
        value = json.loads(
            self.kubectl(
                "gpu",
                "get",
                "pod",
                "--field-selector",
                f"spec.nodeName={node},status.phase=Running",
                "-o",
                "json",
                all_namespaces=True,
            )
        )
        return business_workload_items(
            value.get("items", []), namespace=self.settings.namespace
        )

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        nodes = json.loads(
            self.kubectl(
                "cpu",
                "get",
                "node",
                "-o",
                "json",
            )
        )
        jobs = json.loads(
            self.kubectl(
                "cpu",
                "get",
                "job",
                "-o",
                "json",
                all_namespaces=True,
            )
        )
        events = json.loads(
            self.kubectl(
                "cpu",
                "get",
                "event",
                "-o",
                "json",
                all_namespaces=True,
            )
        )
        return {
            "nodes": {
                item["metadata"]["name"]: {
                    "taints": sorted(
                        (
                            taint["key"],
                            taint.get("value"),
                            taint["effect"],
                        )
                        for taint in item["spec"].get("taints", [])
                    ),
                    "unschedulable": item["spec"].get("unschedulable", False),
                    "labels": {
                        key: value
                        for key, value in item["metadata"].get("labels", {}).items()
                        if key.startswith("gpu-fault.io/")
                    },
                    "annotations": {
                        key: value
                        for key, value in item["metadata"]
                        .get("annotations", {})
                        .items()
                        if key.startswith("gpu-fault.io/")
                    },
                }
                for item in nodes.get("items", [])
            },
            "gpu_fault_jobs": sorted(
                (
                    str(item["metadata"].get("namespace", "")),
                    str(item["metadata"].get("name", "")),
                )
                for item in jobs.get("items", [])
                if any(
                    key.startswith("gpu-fault.io/")
                    for key in item["metadata"].get("labels", {})
                )
            ),
            "eviction_events": sorted(
                (
                    str(item["metadata"].get("uid", "")),
                    str(item.get("reason", "")),
                    str(item.get("involvedObject", {}).get("name", "")),
                )
                for item in events.get("items", [])
                if item.get("reason") in {"Evicted", "TaintManagerEviction"}
            ),
        }

    def provider_events(
        self,
        started_at: datetime,
        ended_at: datetime,
    ) -> list[dict[str, Any]]:
        from scripts.e2e.regional.collector_reboot_evidence import (
            normalize_provider_event,
            provider_event_page,
        )

        rows, token = provider_event_page(
            self.run(
                [
                    "aws",
                    "cloudtrail",
                    "lookup-events",
                    "--region",
                    self.settings.region,
                    "--start-time",
                    started_at.isoformat(),
                    "--end-time",
                    ended_at.isoformat(),
                    "--lookup-attributes",
                    "AttributeKey=EventSource,AttributeValue=sagemaker.amazonaws.com",
                    "--output",
                    "json",
                ],
                timeout=180,
            ).stdout
        )
        if token is not None:
            raise RegionalFixtureError(
                "CloudTrail lookup returned an incomplete inventory"
            )
        result: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item.get("EventName"), str) or not item["EventName"]:
                raise RegionalFixtureError(
                    "CloudTrail event name is missing or malformed"
                )
            if item.get("EventName") not in PROVIDER_MUTATIONS:
                continue
            result.append(dict(normalize_provider_event(item)))
        return result

    def wait_provider_events(
        self,
        started_at: datetime,
        *,
        event_names: set[str],
        expected_count: int,
        ended_at: datetime | None = None,
        timeout_seconds: int = PROVIDER_EVENT_VISIBILITY_SECONDS,
        poll_seconds: int = 60,
    ) -> list[dict[str, Any]]:
        """Poll CloudTrail until ``expected_count`` of ``event_names`` are visible.

        CloudTrail is eventually consistent -- a mutation typically appears
        within 5 minutes and is only guaranteed within 15 -- so a single lookup
        taken right after a reboot finished reads "no RebootClusterNodes" and
        fails a case that did exactly what it should. The intended pattern:

        * a *positive* claim ("the executor issued exactly one reboot") polls
          here and then asserts on the returned list; the poll stops as soon as
          enough events are visible, or at ``timeout_seconds``;
        * a *negative* claim ("no provider mutation happened") cannot be proven
          inside the window -- see ``provider_events_provisional`` -- and the
          case labels it ``provisional``; DESTR-013 re-checks the whole run's
          window once CloudTrail has caught up.

        ``ended_at`` defaults to *now at each poll*, so late-delivered events
        are seen. Returns whatever was visible when it stopped; the caller
        decides what the count means. No event is accepted as a successful
        scoped reboot merely because it is visible; the caller must validate
        the normalized identity, targets, response and submission interval.
        """

        if type(expected_count) is not int or expected_count < 0:
            raise ValueError("expected_count must be a nonnegative integer")
        if type(timeout_seconds) is not int or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be a nonnegative integer")
        if type(poll_seconds) is not int or poll_seconds < 1:
            raise ValueError("poll_seconds must be a positive integer")
        deadline = time.monotonic() + timeout_seconds
        events: list[dict[str, Any]] = []
        while True:
            window_end = ended_at or datetime.now(timezone.utc)
            events = [
                item
                for item in self.provider_events(started_at, window_end)
                if item.get("event_name") in event_names
            ]
            if len(events) >= expected_count or time.monotonic() >= deadline:
                return events
            time.sleep(max(1, min(poll_seconds, int(deadline - time.monotonic()))))

    @staticmethod
    def provider_events_provisional(
        ended_at: datetime,
        *,
        now: datetime | None = None,
    ) -> bool:
        """True while CloudTrail may still be delivering events up to ``ended_at``.

        A runner that saw no provider mutation and whose window ended inside
        the last 15 minutes has a provisional negative, not a PASS. Record it
        as ``provider_events_provisional: True`` so the auditor knows the claim
        rests on DESTR-013's later full-window lookup.
        """

        current = now or datetime.now(timezone.utc)
        if ended_at.tzinfo is None:
            ended_at = ended_at.replace(tzinfo=timezone.utc)
        return current - ended_at < timedelta(seconds=PROVIDER_EVENT_VISIBILITY_SECONDS)

    def wait_node_ready(
        self,
        node: str,
        *,
        timeout_seconds: int,
        expected_boot_id: str | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = self.node_snapshot(node)
            except Exception:
                time.sleep(10)
                continue
            # A node whose status has not repopulated bootID yet reports it as
            # None, which is "different from the old boot ID" only in the
            # trivial sense; requiring a real value keeps "rebooted" meaning a
            # new boot was observed rather than an empty field.
            observed_boot_id = str(last.get("boot_id") or "")
            rebooted = expected_boot_id is None or (
                bool(observed_boot_id) and observed_boot_id != expected_boot_id
            )
            if last.get("ready") == "True" and rebooted:
                return last
            time.sleep(10)
        raise RegionalFixtureError(f"node did not return Ready: {last}")
