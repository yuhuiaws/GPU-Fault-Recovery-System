"""Run the live, read-only GF-REGIONAL-BLAST-001..004 acceptance checks."""

from __future__ import annotations

import base64
import binascii
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpu_fault.admin.site import load_site  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import source_digest  # noqa: E402
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
)
from scripts.e2e.regional.regional_commands import run_fixture_command  # noqa: E402
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records  # noqa: E402

EXECUTION_TOKEN_NAME = re.compile(r"execution[_.-]?token", re.IGNORECASE)
GPU_FAULT_PREFIX = "gpu-fault.io/"
CASE_IDS = tuple(f"GF-REGIONAL-BLAST-{number:03d}" for number in range(1, 5))
E2E001_CASE_ID = "GF-REGIONAL-E2E-001"
# The E2E-001 artifacts BLAST-001 reads (run_workload_acceptance writes them).
E2E001_EXECUTION_CARD = "execution-card.json"
E2E001_CONTROL_PLANE_STATE = "control-plane-current.json"
E2E001_CPU_NODES_BEFORE = "cpu-nodes-before.json"
CONTAINMENT_CASE_ID = "GF-REGIONAL-DESTR-001"
# BLAST-001 binds its producers by release, not by case id: any PASS case of
# the run whose SUCCEEDED records prove these operations on the live release.
REQUIRED_WORKLOAD_OPERATIONS = ("STOP_WORKLOADS", "RESTART_WORKLOAD")
REQUIRED_ISOLATION_OPERATIONS = ("MARK_UNSCHEDULABLE",)
PRODUCER_EVIDENCE_LIMIT_BYTES = 32 * 1024 * 1024
# The four BLAST cases share one preflight (site/account/EKS/HyperPod
# bindings); a result this young is reused rather than paid for four times.
PREFLIGHT_REUSE_SECONDS = 30 * 60
PREFLIGHT_CACHE_NAME = "blast-preflight.json"
RELEASE_STATE_CONFIGMAP = "gpu-fault-regional-release-state"


def default_e2e_dir(run_dir: Path) -> Path:
    return run_dir / "cases" / E2E001_CASE_ID


def default_trusted_cpu_baseline(e2e_dir: Path) -> Path:
    return e2e_dir / E2E001_CPU_NODES_BEFORE


def blast001_input_errors(e2e_dir: Path, trusted_cpu_baseline: Path) -> list[str]:
    """Why BLAST-001 cannot run on these inputs: every file it reads must exist."""

    errors = []
    if not e2e_dir.is_dir():
        errors.append(f"E2E-001 case directory is not a directory: {e2e_dir}")
    else:
        for name in (E2E001_EXECUTION_CARD, E2E001_CONTROL_PLANE_STATE):
            if not (e2e_dir / name).is_file():
                errors.append(f"E2E-001 evidence file is missing: {e2e_dir / name}")
    if not trusted_cpu_baseline.is_file():
        errors.append(f"trusted CPU baseline is not a file: {trusted_cpu_baseline}")
    return errors


FORBIDDEN_CONTROL_PLANE_ACTIONS = (
    "sagemaker:BatchReplaceClusterNodes",
    "sagemaker:RebootClusterNodes",
    "sagemaker:BatchRebootClusterNodes",
    "sagemaker:BatchDeleteClusterNodes",
)
FORBIDDEN_EXECUTOR_ACTIONS = (
    "sagemaker:BatchReplaceClusterNodes",
    "sagemaker:BatchDeleteClusterNodes",
)


class CheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClusterTarget:
    cluster_id: str
    context: str
    hyperpod_cluster_name: str
    eks_cluster_arn: str
    executor_role_arn: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def observed_in_windows(
    timestamp: str, windows: Sequence[tuple[datetime, datetime]]
) -> bool:
    observed = parse_time(timestamp)
    if observed.tzinfo is None:
        raise CheckError("CPU audit timestamp has no timezone")
    return any(start <= observed <= end for start, end in windows)


def command(
    args: Sequence[str],
    *,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_fixture_command(
        args,
        input_text=input_text,
        check=check,
        timeout=180,
    )


def json_command(args: Sequence[str]) -> Any:
    result = command(args)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CheckError(
            f"command returned invalid JSON: {args[0]} {' '.join(args[1:4])}"
        ) from exc


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one evidence document all-or-nothing.

    The previous ``write_text`` left a truncated file behind when the process
    died mid-write; ``write_json_atomic`` renames a complete temporary file
    onto ``path`` and applies the same scope check the other runners use.
    """

    write_json_atomic(path, value)


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)


def arn_parts(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 5)
    if len(parts) != 6 or parts[0] != "arn":
        raise CheckError(f"invalid ARN in site configuration: {value!r}")
    return parts[3], parts[4], parts[5]


def arn_resource_name(value: str) -> str:
    resource = arn_parts(value)[2]
    return resource.rsplit("/", 1)[-1]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def successful_containment_command(
    state: dict[str, Any], source: dict[str, Any]
) -> str:
    """Bind the successful MARK step to the producer's exact remote command."""

    workflow, incident, event = (
        state.get("workflow") or {},
        state.get("incident") or {},
        state.get("event") or {},
    )
    if (
        not source.get("workflow_request_id")
        or not source.get("incident_id")
        or workflow.get("request_id") != source["workflow_request_id"]
        or workflow.get("incident_id") != source["incident_id"]
        or workflow.get("status") != "SUCCEEDED"
        or type(workflow.get("fencing_token")) is not int
        or workflow["fencing_token"] < 1
        or incident.get("incident_id") != source["incident_id"]
        or incident.get("workflow_request_id") != source["workflow_request_id"]
        or incident.get("cluster_id") != source["cluster_id"]
        or not event.get("event_id")
        or incident.get("event_id") != event["event_id"]
        or event.get("cluster_id") != source["cluster_id"]
        or event.get("node_id") != source["node"]
        or source["node"] not in (incident.get("node_ids") or [])
    ):
        raise CheckError("DESTR-001 containment workflow identity is inconsistent")
    executions = [
        step
        for step in workflow.get("step_executions") or []
        if step.get("operation") == "MARK_UNSCHEDULABLE"
        and step.get("status") == "SUCCEEDED"
    ]
    if len(executions) != 1:
        raise CheckError(
            "DESTR-001 has no unique successful MARK_UNSCHEDULABLE execution"
        )
    execution = executions[0]
    if type(execution.get("step_index")) is not int or execution["step_index"] < 0:
        raise CheckError("DESTR-001 containment step index is missing")
    matches = []
    for command in state.get("commands") or []:
        if (
            command.get("status") != "SUCCEEDED"
            or command.get("cluster_id") != source["cluster_id"]
            or command.get("workflow_request_id") != source["workflow_request_id"]
            or command.get("incident_id") != source["incident_id"]
            or command.get("fencing_token") != workflow.get("fencing_token")
            or not command.get("last_lease_owner")
            or execution.get("adapter_operation_id")
            != "remote/" + str(command.get("command_id"))
        ):
            continue
        covered = [
            {"step_index": command.get("step_index"), "step": command.get("step")},
            *(command.get("batched_steps") or []),
        ]
        if any(
            item.get("step_index") == execution.get("step_index")
            and (item.get("step") or {}).get("operation") == "MARK_UNSCHEDULABLE"
            and (item.get("step") or {}).get("node_ids") == [source["node"]]
            for item in covered
        ):
            matches.append(str(command["command_id"]))
    if len(matches) != 1:
        raise CheckError(
            "DESTR-001 successful containment command is missing or ambiguous"
        )
    return matches[0]


def containment_source(
    root_run_dir: Path,
    *,
    predecessor: dict[str, Any],
    release_id: str,
    cluster_id: str,
    cpu_snapshot: dict[str, Any],
) -> dict[str, Any]:
    source_dir = root_run_dir / "cases" / CONTAINMENT_CASE_ID
    verdict_path = source_dir / f"{CONTAINMENT_CASE_ID}.json"
    if (
        predecessor.get("valid") is not True
        or predecessor.get("case_id") != CONTAINMENT_CASE_ID
        or predecessor.get("verdict") != "PASS"
        or Path(str(predecessor.get("path") or "")).resolve() != verdict_path.resolve()
    ):
        raise CheckError("BLAST-001 requires this run's DESTR-001 PASS predecessor")
    documents = {}
    digests = {}
    for name in (
        verdict_path.name,
        "preflight.json",
        "workflow-state.json",
        "host-baseline.json",
        "host-after.json",
        "cpu-blast-after.json",
    ):
        path = source_dir / name
        if path.resolve().parent != source_dir.resolve():
            raise CheckError("DESTR-001 source file escaped its case directory")
        raw = path.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise CheckError("DESTR-001 source file is not an object")
        documents[name] = document
        digests[name] = sha256_bytes(raw)
    source = documents[verdict_path.name]
    preflight = documents["preflight.json"]
    state = documents["workflow-state.json"]
    residuals = source.get("probe_residuals")
    if (
        source.get("case_id") != CONTAINMENT_CASE_ID
        or source.get("verdict") != "PASS"
        or source.get("release_id") != release_id
        or source.get("cluster_id") != cluster_id
        or source.get("errors") != []
        or source.get("error")
        or any(
            source.get(key)
            for key in (
                "sampler_cleanup_error",
                "probe_cleanup_error",
                "postflight_error",
            )
        )
        or not isinstance(residuals, dict)
        or not residuals
        or any(value is not False for value in residuals.values())
        or not isinstance(source.get("quiesce_recovery"), dict)
        or source["quiesce_recovery"].get("error")
        or preflight.get("errors") != []
        or (preflight.get("evidence_identity") or {}).get("release_id") != release_id
        or (preflight.get("evidence_identity") or {}).get("cluster_id") != cluster_id
        or state.get("release_id") != release_id
    ):
        raise CheckError(
            "DESTR-001 source is not successful, cleaned and deployment-bound"
        )
    before_node, after_node = (
        preflight.get("node") or {},
        source.get("final_node") or {},
    )
    if (
        not source.get("node")
        or not before_node.get("uid")
        or before_node.get("name") != source["node"]
        or after_node.get("name") != source["node"]
        or after_node.get("uid") != before_node["uid"]
        or after_node.get("ready") != "True"
        or after_node.get("unschedulable") is not False
        or after_node.get("ownership_annotations")
        != before_node.get("ownership_annotations")
        or after_node.get("taints") != before_node.get("taints")
    ):
        raise CheckError("DESTR-001 node identity or cleanup is unproven")
    start_text = str(documents["host-baseline.json"].get("captured_at") or "")
    end_text = str(documents["host-after.json"].get("captured_at") or "")
    start, end = parse_time(start_text), parse_time(end_text)
    approved_end = parse_time(str(source.get("maintenance_window_end") or ""))
    event_time = parse_time(str((state.get("event") or {}).get("observed_at") or ""))
    if (
        any(item.tzinfo is None for item in (start, end, approved_end, event_time))
        or not start <= event_time <= end <= datetime.now(timezone.utc)
        or start == end
        or end > approved_end
    ):
        raise CheckError("DESTR-001 containment observation window is invalid")
    expected_nodes = {
        name: {
            "taints": [
                [item["key"], item["value"], item["effect"]] for item in node["taints"]
            ],
            "unschedulable": node["unschedulable"],
            "labels": node["gpu_fault_labels"],
            "annotations": node["gpu_fault_annotations"],
        }
        for name, node in cpu_snapshot.items()
    }
    before_cpu, after_cpu = (
        preflight.get("cpu_blast") or {},
        documents["cpu-blast-after.json"],
    )
    if (
        before_cpu.get("nodes") != expected_nodes
        or before_cpu != after_cpu
        or not isinstance(before_cpu.get("gpu_fault_jobs"), list)
        or not isinstance(before_cpu.get("eviction_events"), list)
    ):
        raise CheckError("DESTR-001 CPU snapshots do not match the trusted baseline")
    command_id = successful_containment_command(state, source)
    return {
        "case_id": CONTAINMENT_CASE_ID,
        "evidence_dir": str(source_dir),
        "release_id": release_id,
        "cluster_id": cluster_id,
        "workflow_request_id": source["workflow_request_id"],
        "command_id": command_id,
        "node": source["node"],
        "node_uid": before_node["uid"],
        "window_start": start_text,
        "window_end": end_text,
        "operations": ["MARK_UNSCHEDULABLE"],
        "source_sha256": digests,
    }


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def action_pattern_matches(pattern: str, action: str) -> bool:
    return fnmatch.fnmatchcase(action.lower(), pattern.lower())


def statement_actions(statement: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in as_list(statement.get("Action"))]


def statement_not_actions(statement: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in as_list(statement.get("NotAction"))]


def allow_statement_matches(statement: Mapping[str, Any], action: str) -> bool:
    if str(statement.get("Effect", "")).lower() != "allow":
        return False
    not_actions = statement_not_actions(statement)
    if not_actions:
        return not any(action_pattern_matches(item, action) for item in not_actions)
    return any(
        action_pattern_matches(item, action) for item in statement_actions(statement)
    )


def resources_for(statement: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in as_list(statement.get("Resource"))]


NOTIFICATION_CONFIGMAP = "gpu-fault-api-ha-config-notification"


def notification_channel(config: Mapping[str, Any]) -> str:
    """Which delivery channel the deployed control plane notifies through.

    Mirrors ``notifications.channel.notification_channel_from_environment``
    against the rendered control-plane config: the declared channel wins,
    then a topic ARN implies ``sns``, then a sender implies ``ses``, else
    ``disabled``. The release engine always renders the channel onto the
    role, so ``sns`` (the admin-CLI default) is the usual answer.
    """

    declared = str(config.get("GPU_FAULT_NOTIFICATION_CHANNEL") or "").strip().lower()
    if declared:
        if declared not in ("sns", "ses", "disabled"):
            raise CheckError("deployed notification channel is unknown")
        return declared
    if str(config.get("GPU_FAULT_SNS_TOPIC_ARN") or "").strip():
        return "sns"
    if config.get("GPU_FAULT_EMAIL_SENDER") or config.get("GPU_FAULT_EMAIL_RECIPIENTS"):
        return "ses"
    return "disabled"


def policy_statement_summary(statement: Mapping[str, Any]) -> dict[str, Any]:
    resources = resources_for(statement)
    return {
        "effect": statement.get("Effect"),
        "actions": statement_actions(statement),
        "not_actions": statement_not_actions(statement),
        "resource_wildcard": any(item == "*" for item in resources),
        "resource_count": len(resources),
        "resource_hashes": [sha256_text(item) for item in resources],
        "has_condition": bool(statement.get("Condition")),
    }


class BlastRunnerBase:
    def __init__(
        self,
        *,
        site_path: Path,
        run_dir: Path,
        case_id: str,
        e2e_dir: Path,
        trusted_cpu_baseline: Path,
        predecessor: dict[str, Any],
        preflight_reuse_seconds: int = PREFLIGHT_REUSE_SECONDS,
    ) -> None:
        self.site_path = site_path.resolve()
        self.root_run_dir = run_dir.resolve()
        self.case_id = case_id
        self.run_dir = self.root_run_dir / "cases" / case_id
        self.e2e_dir = e2e_dir.resolve()
        self.trusted_cpu_baseline = trusted_cpu_baseline.resolve()
        self.predecessor = predecessor
        self.preflight_reuse_seconds = preflight_reuse_seconds
        self._identity: dict[str, str | None] | None = None
        self.site = load_site(self.site_path)
        self.config = self.site.release_config
        self.region = str(self.config["aws_region"])
        self.namespace = str(self.config["namespace"])
        self.cpu_kubeconfig = str(self.config["cpu_kubeconfig"])
        self.gpu_kubeconfig = str(
            self.config.get("gpu_kubeconfig")
            or self.site.environment.get("KUBECONFIG")
            or ""
        )
        if not self.gpu_kubeconfig:
            raise CheckError("site does not resolve a GPU kubeconfig")
        self.cpu_eks_arn = str(self.config["cpu_eks_arn"])
        self.cpu_cluster_name = arn_resource_name(self.cpu_eks_arn)
        self.targets = [
            ClusterTarget(
                cluster_id=str(item["cluster_id"]),
                context=str(item["context"]),
                hyperpod_cluster_name=str(item["hyperpod_cluster_name"]),
                eks_cluster_arn=str(item["eks_cluster_arn"]),
                executor_role_arn=str(item["executor_irsa_role_arn"]),
            )
            for item in self.config["clusters"]
        ]
        if not self.targets:
            raise CheckError("site contains no GPU clusters")
        self._input_binding = self.input_binding()
        if self._input_binding["site_sha256"] != self.site.source_sha256:
            raise CheckError("site changed while loading BLAST inputs")
        self.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.run_dir.chmod(0o700)
        self.case_statuses: list[dict[str, Any]] = []

    def input_binding(self) -> dict[str, str]:
        binding = {}
        for key, path in (
            ("site_sha256", self.site_path),
            ("cpu_kubeconfig_sha256", Path(self.cpu_kubeconfig)),
            ("gpu_kubeconfig_sha256", Path(self.gpu_kubeconfig)),
        ):
            try:
                binding[key] = sha256_bytes(path.read_bytes())
            except OSError:
                raise CheckError(f"cannot bind BLAST input {key}") from None
        try:
            binding["source_digest"] = source_digest()
        except Exception:
            # Source identity errors may carry subprocess output.
            raise CheckError("cannot bind BLAST source identity") from None
        return binding

    def verify_input_binding(self) -> None:
        if self.input_binding() != self._input_binding:
            raise CheckError("BLAST input identity changed during the audit")

    def aws(self, *args: str) -> Any:
        return json_command(("aws", *args, "--region", self.region, "--output", "json"))

    def cpu_json(self, *args: str) -> Any:
        return json_command(("kubectl", "--kubeconfig", self.cpu_kubeconfig, *args))

    def cpu_text(self, *args: str, check: bool = True) -> str:
        return command(
            ("kubectl", "--kubeconfig", self.cpu_kubeconfig, *args),
            check=check,
        ).stdout

    def gpu_json(self, target: ClusterTarget, *args: str) -> Any:
        return json_command(
            (
                "kubectl",
                "--kubeconfig",
                self.gpu_kubeconfig,
                "--context",
                target.context,
                *args,
            )
        )

    def gpu_text(self, target: ClusterTarget, *args: str, check: bool = True) -> str:
        return command(
            (
                "kubectl",
                "--kubeconfig",
                self.gpu_kubeconfig,
                "--context",
                target.context,
                *args,
            ),
            check=check,
        ).stdout

    def notification_config(self) -> dict[str, str]:
        """The control plane's rendered notification config (channel + topic)."""

        document = self.cpu_json(
            "-n",
            self.namespace,
            "get",
            "configmap",
            NOTIFICATION_CONFIGMAP,
            "-o",
            "json",
        )
        data = document.get("data") or {}
        return {str(key): str(value) for key, value in data.items()}

    def evidence_identity(self) -> dict[str, str | None]:
        """The release (and, on a single-cluster site, the cluster) this audit reads.

        Written into every case result so the next case can require its
        predecessor to have passed against the same deployment. A site with
        several GPU clusters is audited as a whole and has no single
        ``cluster_id``; the release binding still applies.
        """

        document = self.cpu_json(
            "-n",
            self.namespace,
            "get",
            "configmap",
            RELEASE_STATE_CONFIGMAP,
            "-o",
            "json",
        )
        try:
            state = json.loads(document["data"]["state.json"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise CheckError("regional release state is not valid JSON") from exc
        if (
            not isinstance(state, dict)
            or not isinstance(state.get("release_id"), str)
            or not state["release_id"]
        ):
            raise CheckError("regional release identity is missing")
        current = {
            "release_id": state["release_id"],
            "cluster_id": self.targets[0].cluster_id
            if len(self.targets) == 1
            else None,
        }
        if self._identity is not None and self._identity != current:
            raise CheckError("regional release identity changed during the audit")
        self._identity = current
        return dict(self._identity)

    def record_case(
        self,
        case_id: str,
        status: str,
        *,
        checks: Mapping[str, Any],
        limitations: Sequence[str] = (),
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": 2,
            "report_type": "fault-acceptance",
            "case_id": case_id,
            "verdict": status,
            "executed_at": utc_now(),
            "checks": dict(checks),
            "input_binding": dict(self._input_binding),
            "limitations": list(limitations)
            or [
                "The audit proves the current live configuration and identities; "
                "it does not authorize or execute a GPU mutation."
            ],
            "predecessor": self.predecessor,
        }
        if error is not None:
            payload["error"] = error
        identity_error = None
        try:
            payload.update(self.evidence_identity())
        except Exception as exc:  # the identity read is itself a live call
            identity_error = f"{type(exc).__name__}: {exc}"
            payload["evidence_identity_error"] = identity_error
            payload["verdict"] = "FAIL"
        binding_error = None
        try:
            self.verify_input_binding()
        except Exception as exc:
            binding_error = f"{type(exc).__name__}: {exc}"
            payload["evidence_binding_error"] = binding_error
            payload["verdict"] = "FAIL"
        write_json(case_evidence_path(self.root_run_dir, case_id), payload)
        self.case_statuses.append(payload)
        print(f"{case_id}: {payload['verdict']}", flush=True)
        if identity_error is not None and status == "PASS":
            raise CheckError("the audit lost its deployment identity")
        if binding_error is not None and status == "PASS":
            raise CheckError("the audit lost its input identity")

    def ready_cpu_pod(self) -> str:
        pods = self.cpu_json(
            "-n",
            self.namespace,
            "get",
            "pods",
            "-l",
            "app=gpu-fault-api-ha",
            "-o",
            "json",
        )
        ready = ready_pod_records(pods)
        if not ready:
            raise CheckError("no Ready control-plane API Pod")
        return str(ready[-1]["name"])

    def ready_executor_pod(self, target: ClusterTarget) -> str:
        pods = self.gpu_json(
            target,
            "-n",
            self.namespace,
            "get",
            "pods",
            "-l",
            "app=gpu-fault-cluster-executor",
            "-o",
            "json",
        )
        ready = ready_pod_records(pods)
        if not ready:
            raise CheckError(f"no Ready executor Pod for cluster {target.cluster_id}")
        return str(ready[-1]["name"])

    def kubeconfig_binding(
        self,
        *,
        kube_args: Sequence[str],
        expected_arn: str,
        eks_description: Mapping[str, Any],
    ) -> dict[str, Any]:
        view = json_command(
            (
                "kubectl",
                *kube_args,
                "config",
                "view",
                "--minify",
                "-o",
                "json",
            )
        )
        contexts = view.get("contexts") or []
        clusters = view.get("clusters") or []
        if len(contexts) != 1 or len(clusters) != 1:
            raise CheckError("kubeconfig minified view is not singular")
        cluster_entry = clusters[0]
        cluster_data = cluster_entry.get("cluster", {})
        if cluster_data.get("insecure-skip-tls-verify") not in (None, False):
            raise CheckError("kubeconfig disables TLS verification")
        if eks_description["cluster"].get("arn") != expected_arn:
            raise CheckError("EKS description does not match the expected ARN")
        endpoint = str(cluster_data.get("server", ""))
        expected_endpoint = str(eks_description["cluster"]["endpoint"])
        context_cluster_name = str(contexts[0].get("context", {}).get("cluster", ""))
        if context_cluster_name != cluster_entry.get("name"):
            raise CheckError("kubeconfig context references a different cluster")
        expected_ca = str(
            eks_description["cluster"].get("certificateAuthority", {}).get("data", "")
        )
        kube_ca = command(
            (
                "kubectl",
                *kube_args,
                "config",
                "view",
                "--minify",
                "--raw",
                "-o",
                "jsonpath={.clusters[0].cluster.certificate-authority-data}",
            )
        ).stdout.strip()
        try:
            expected_ca_bytes = base64.b64decode(expected_ca, validate=True)
            kube_ca_bytes = base64.b64decode(kube_ca, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise CheckError(
                f"kubeconfig or EKS CA is not valid base64 for {expected_arn}"
            ) from exc
        ca_matches = bool(
            expected_ca_bytes and kube_ca_bytes and expected_ca_bytes == kube_ca_bytes
        )
        if endpoint != expected_endpoint:
            raise CheckError(f"kubeconfig endpoint does not match EKS {expected_arn}")
        if not ca_matches:
            raise CheckError(f"kubeconfig CA does not match EKS {expected_arn}")
        return {
            "expected_arn": expected_arn,
            "context_name": contexts[0].get("name"),
            "context_cluster_name": context_cluster_name,
            "endpoint_matches": endpoint == expected_endpoint,
            "ca_matches": ca_matches,
        }

    def preflight_binding(self) -> dict[str, Any]:
        return {
            **self.input_binding(),
            "region": self.region,
            "namespace": self.namespace,
            "cpu_kubeconfig": self.cpu_kubeconfig,
            "gpu_kubeconfig": self.gpu_kubeconfig,
            "cpu_eks_arn": self.cpu_eks_arn,
            "targets": [asdict(target) for target in self.targets],
            "identity": self.evidence_identity(),
        }

    def reusable_preflight(self) -> dict[str, Any] | None:
        """The run's cached preflight scope, if it is young enough and for this site.

        The scope holds the site path and the cluster IDs it was taken for;
        anything else -- another site, another set of clusters, or a capture
        older than ``preflight_reuse_seconds`` -- is not this run's preflight.
        """

        path = self.root_run_dir / PREFLIGHT_CACHE_NAME
        if not path.is_file():
            return None
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            captured_at = parse_time(str(cached["captured_at"]))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        if not isinstance(cached, dict):
            return None
        if captured_at.tzinfo is None:
            return None
        age = (datetime.now(timezone.utc) - captured_at).total_seconds()
        if age < 0 or age > self.preflight_reuse_seconds:
            return None
        if cached.get("site") != str(self.site_path):
            return None
        cached_clusters = sorted(
            str(item.get("cluster_id"))
            for item in cached.get("gpu_clusters") or []
            if isinstance(item, dict)
        )
        if cached_clusters != sorted(target.cluster_id for target in self.targets):
            return None
        if cached.get("binding") != self.preflight_binding():
            return None
        return cached

    def preflight(self) -> None:
        self.verify_input_binding()
        cached = self.reusable_preflight()
        if cached is not None:
            self.verify_input_binding()
            write_json(
                self.run_dir / "execution-scope.json",
                {
                    **cached,
                    "reused_from": str(self.root_run_dir / PREFLIGHT_CACHE_NAME),
                    "reused_at": utc_now(),
                },
            )
            print("preflight: REUSED", flush=True)
            return
        identity = self.aws("sts", "get-caller-identity")
        _, expected_account, _ = arn_parts(self.cpu_eks_arn)
        if str(identity.get("Account")) != expected_account:
            raise CheckError("AWS caller account does not match site CPU EKS ARN")

        cpu_eks = self.aws("eks", "describe-cluster", "--name", self.cpu_cluster_name)
        cpu_binding = self.kubeconfig_binding(
            kube_args=("--kubeconfig", self.cpu_kubeconfig),
            expected_arn=self.cpu_eks_arn,
            eks_description=cpu_eks,
        )
        cpu_ready = self.cpu_text("get", "--raw=/readyz").strip().endswith("ok")
        if not cpu_ready:
            raise CheckError("CPU EKS /readyz is not ok")
        cpu_nodes = self.cpu_json("get", "nodes", "-o", "json")

        gpu_results = []
        for target in self.targets:
            gpu_eks_name = arn_resource_name(target.eks_cluster_arn)
            gpu_eks = self.aws("eks", "describe-cluster", "--name", gpu_eks_name)
            binding = self.kubeconfig_binding(
                kube_args=(
                    "--kubeconfig",
                    self.gpu_kubeconfig,
                    "--context",
                    target.context,
                ),
                expected_arn=target.eks_cluster_arn,
                eks_description=gpu_eks,
            )
            ready = self.gpu_text(target, "get", "--raw=/readyz").strip().endswith("ok")
            if not ready:
                raise CheckError(f"GPU EKS /readyz is not ok for {target.cluster_id}")
            hyperpod = self.aws(
                "sagemaker",
                "describe-cluster",
                "--cluster-name",
                target.hyperpod_cluster_name,
            )
            orchestrator_arn = str(
                hyperpod.get("Orchestrator", {}).get("Eks", {}).get("ClusterArn", "")
            )
            node_recovery = str(hyperpod.get("NodeRecovery", ""))
            if orchestrator_arn != target.eks_cluster_arn:
                raise CheckError(
                    f"HyperPod EKS binding mismatch for {target.cluster_id}"
                )
            if node_recovery != "None":
                raise CheckError(
                    f"GPU HyperPod NodeRecovery is not None for {target.cluster_id}"
                )
            nodes = self.gpu_json(target, "get", "nodes", "-o", "json")
            gpu_results.append(
                {
                    "cluster_id": target.cluster_id,
                    "context": target.context,
                    "kubeconfig": self.gpu_kubeconfig,
                    "eks_binding": binding,
                    "readyz": ready,
                    "hyperpod_orchestrator_matches": True,
                    "node_recovery": node_recovery,
                    "node_names": sorted(
                        str(item.get("metadata", {}).get("name", ""))
                        for item in nodes.get("items", [])
                    ),
                }
            )

        scope = {
            "captured_at": utc_now(),
            "binding": self.preflight_binding(),
            "site": str(self.site_path),
            "aws_region": self.region,
            "aws_account": expected_account,
            "namespace": self.namespace,
            "cpu": {
                "eks_arn": self.cpu_eks_arn,
                "kubeconfig": self.cpu_kubeconfig,
                "binding": cpu_binding,
                "readyz": cpu_ready,
                "node_names": sorted(
                    str(item.get("metadata", {}).get("name", ""))
                    for item in cpu_nodes.get("items", [])
                ),
            },
            "gpu_clusters": gpu_results,
            "maintenance_window": "not-required-read-only-phase",
            "allowed_actions": [
                "Kubernetes GET/LIST/AUTH CAN-I",
                "Kubernetes pod exec for names, hashes, and file listings only",
                "AWS Describe/List/Get IAM policy calls",
            ],
            "forbidden_actions": [
                "Kubernetes create/update/patch/delete/apply",
                "fault injection",
                "GPU reset",
                "node reboot",
                "node replacement",
                "workload restart",
                "service restart",
            ],
            "stop_conditions": [
                "site, context, EKS, HyperPod, Region, or account mismatch",
                "GPU HyperPod NodeRecovery is not None",
                "unexpected write permission or credential placement",
                "any preceding BLAST case fails",
            ],
            "rollback": "none; this run creates no cluster or AWS resources",
        }
        self.verify_input_binding()
        write_json(self.run_dir / "execution-scope.json", scope)
        write_json(self.root_run_dir / PREFLIGHT_CACHE_NAME, scope)
        print("preflight: PASS", flush=True)

    def fail(self, started_at: str, exc: BaseException) -> int:
        """Record a stopped run: a FAIL evidence file when none was written yet.

        Only ``CheckError`` used to be caught; a KeyError from an unexpected
        API shape or an OSError reading evidence left no verdict file at all,
        so the chain's next case saw a MISSING predecessor instead of a FAIL
        with a reason.
        """

        error = f"{type(exc).__name__}: {exc}"
        if not any(item.get("case_id") == self.case_id for item in self.case_statuses):
            self.record_case(
                self.case_id,
                "FAIL",
                checks={},
                limitations=[
                    "The audit stopped before its checks completed; the error "
                    "field names the failure and no verdict was inferred."
                ],
                error=error,
            )
        write_json(
            self.run_dir / "phase-summary.json",
            {
                "phase": "爆炸半径",
                "case_id": self.case_id,
                "started_at": started_at,
                "ended_at": utc_now(),
                "status": "FAIL",
                "error": error,
                "cases": self.case_statuses,
            },
        )
        print(f"STOP: {error}", file=sys.stderr, flush=True)
        return 1

    def run(self) -> int:
        started_at = utc_now()
        try:
            if self.predecessor.get("valid") is not True:
                raise CheckError("formal predecessor evidence is not PASS")
            self.preflight()
            {
                "GF-REGIONAL-BLAST-001": self.blast_001,
                "GF-REGIONAL-BLAST-002": self.blast_002,
                "GF-REGIONAL-BLAST-003": self.blast_003,
                "GF-REGIONAL-BLAST-004": self.blast_004,
            }[self.case_id]()
        except Exception as exc:
            return self.fail(started_at, exc)
        write_json(
            self.run_dir / "phase-summary.json",
            {
                "phase": "爆炸半径",
                "case_id": self.case_id,
                "started_at": started_at,
                "ended_at": utc_now(),
                "status": "PASS",
                "cases": self.case_statuses,
            },
        )
        return 0

    def blast_001(self) -> None:
        raise NotImplementedError

    def blast_002(self) -> None:
        raise NotImplementedError

    def blast_003(self) -> None:
        raise NotImplementedError

    def blast_004(self) -> None:
        raise NotImplementedError


def _operation_name(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1]


def _read_json_document(path: Path) -> Any | None:
    try:
        if path.stat().st_size > PRODUCER_EVIDENCE_LIMIT_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def parse_optional_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = parse_time(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def succeeded_operations(document: Any) -> dict[str, set[str]]:
    """Operations proven by SUCCEEDED records anywhere in ``document``.

    Two shapes count, both as the runners write them: a workflow record whose
    ``status`` is SUCCEEDED, read through its SUCCEEDED ``step_executions``; and
    a remote command whose ``status`` is SUCCEEDED, read through
    ``step.operation``. Declared steps, ``completed_operations`` lists, WAITING
    or FAILED executions and non-terminal workflows prove nothing here -- the
    spec forbids taking a step name or an arbitrary terminal status as success.
    Returns ``{operation: {workflow ids}}``.
    """

    found: dict[str, set[str]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if str(value.get("status") or "").endswith("SUCCEEDED"):
                identity = str(
                    value.get("request_id") or value.get("workflow_request_id") or ""
                )
                steps = value.get("step_executions")
                if isinstance(steps, list):
                    for step in steps:
                        if (
                            isinstance(step, dict)
                            and str(step.get("status") or "").endswith("SUCCEEDED")
                            and step.get("operation")
                        ):
                            found.setdefault(
                                _operation_name(step["operation"]), set()
                            ).add(identity)
                step = value.get("step")
                if (
                    isinstance(step, dict)
                    and step.get("operation")
                    and value.get("command_id")
                ):
                    found.setdefault(_operation_name(step["operation"]), set()).add(
                        identity
                    )
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    return found


def release_bound_producers(
    root_run_dir: Path, *, release_id: str, cluster_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """PASS cases of this run bound to ``release_id`` with the operations they prove.

    A case counts when its verdict file says PASS for ``release_id`` and one of
    ``cluster_ids`` with no recorded errors; every JSON file in its directory is
    then read for SUCCEEDED records (``succeeded_operations``). The window a
    producer contributes runs from its ``started_at`` (or the earliest JSON it
    wrote) to its latest recorded end (or the verdict file's write time).
    """

    producers: dict[str, dict[str, Any]] = {}
    for case_dir in sorted((root_run_dir / "cases").glob("GF-REGIONAL-*")):
        if not case_dir.is_dir():
            continue
        # The case's own verdict file, or -- for E2E-001, whose in-process
        # producer writes only its execution card -- the card, which carries
        # the same case_id/verdict/release_id/cluster_id binding.
        verdict_path = case_dir / f"{case_dir.name}.json"
        if not verdict_path.is_file():
            verdict_path = case_dir / E2E001_EXECUTION_CARD
        verdict = _read_json_document(verdict_path)
        if (
            not isinstance(verdict, dict)
            or verdict.get("case_id") not in (None, case_dir.name)
            or verdict.get("verdict") != "PASS"
            or verdict.get("release_id") != release_id
            or verdict.get("cluster_id") not in cluster_ids
            or verdict.get("errors")
            or verdict.get("error")
        ):
            continue
        operations: dict[str, set[str]] = {}
        files: list[str] = []
        documents = sorted(case_dir.glob("*.json"))
        for path in documents:
            document = _read_json_document(path)
            if document is None:
                continue
            for operation, identities in succeeded_operations(document).items():
                operations.setdefault(operation, set()).update(identities)
            files.append(path.name)
        if not operations:
            continue
        # A run's own start stamp, else the window its card opened (E2E-001's
        # card carries no started_at), else the earliest JSON it wrote.
        window = verdict.get("maintenance_window")
        started = (
            parse_optional_time(verdict.get("started_at"))
            or parse_optional_time(
                window.get("start") if isinstance(window, dict) else None
            )
            or datetime.fromtimestamp(
                min(path.stat().st_mtime for path in documents), tz=timezone.utc
            )
        )
        ends = [
            parse_optional_time(verdict.get(key))
            for key in ("ended_at", "executed_at", "completed_at")
        ]
        ended = max(
            [item for item in ends if item is not None]
            or [datetime.fromtimestamp(verdict_path.stat().st_mtime, tz=timezone.utc)]
        )
        producers[case_dir.name] = {
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "operations": {
                operation: sorted(identities)
                for operation, identities in sorted(operations.items())
            },
            "files": files,
        }
    return producers


def producer_operations(producers: dict[str, dict[str, Any]]) -> set[str]:
    return {
        operation
        for producer in producers.values()
        for operation in producer["operations"]
    }


def missing_required_operations(producers: dict[str, dict[str, Any]]) -> list[str]:
    present = producer_operations(producers)
    return sorted(
        set(REQUIRED_WORKLOAD_OPERATIONS + REQUIRED_ISOLATION_OPERATIONS) - present
    )
