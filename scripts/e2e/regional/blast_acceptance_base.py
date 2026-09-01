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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpu_fault.admin_site import load_site  # noqa: E402
from scripts.e2e.regional.acceptance_scope import (  # noqa: E402
    scoped_case_evidence,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
)

EXECUTION_TOKEN_NAME = re.compile(r"execution[_.-]?token", re.IGNORECASE)
GPU_FAULT_PREFIX = "gpu-fault.io/"
CASE_IDS = tuple(f"GF-REGIONAL-BLAST-{number:03d}" for number in range(1, 5))
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


def command(
    args: Sequence[str],
    *,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(args),
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        raise CheckError(
            f"command failed with exit {result.returncode}: {args[0]} "
            f"{' '.join(args[1:4])}; stderr={stderr[:600]!r}"
        )
    return result


def json_command(args: Sequence[str]) -> Any:
    result = command(args)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CheckError(
            f"command returned invalid JSON: {args[0]} {' '.join(args[1:4])}"
        ) from exc


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            scoped_case_evidence(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


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
    ) -> None:
        self.site_path = site_path.resolve()
        self.root_run_dir = run_dir.resolve()
        self.case_id = case_id
        self.run_dir = self.root_run_dir / "cases" / case_id
        self.e2e_dir = e2e_dir.resolve()
        self.trusted_cpu_baseline = trusted_cpu_baseline.resolve()
        self.predecessor = predecessor
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
        self.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.run_dir.chmod(0o700)
        self.case_statuses: list[dict[str, Any]] = []

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

    def record_case(
        self,
        case_id: str,
        status: str,
        *,
        checks: Mapping[str, Any],
        limitations: Sequence[str] = (),
    ) -> None:
        payload = {
            "schema_version": 2,
            "report_type": "fault-acceptance",
            "case_id": case_id,
            "verdict": status,
            "executed_at": utc_now(),
            "checks": dict(checks),
            "limitations": list(limitations)
            or [
                "The audit proves the current live configuration and identities; "
                "it does not authorize or execute a GPU mutation."
            ],
            "predecessor": self.predecessor,
        }
        write_json(case_evidence_path(self.root_run_dir, case_id), payload)
        self.case_statuses.append(payload)
        print(f"{case_id}: {status}", flush=True)

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
        ready: list[Mapping[str, Any]] = []
        for pod in pods.get("items", []):
            metadata = pod.get("metadata", {})
            status = pod.get("status", {})
            conditions = status.get("conditions") or []
            if (
                not metadata.get("deletionTimestamp")
                and status.get("phase") == "Running"
                and any(
                    item.get("type") == "Ready" and item.get("status") == "True"
                    for item in conditions
                )
            ):
                ready.append(pod)
        if not ready:
            raise CheckError("no Ready control-plane API Pod")
        ready.sort(key=lambda item: item["metadata"].get("creationTimestamp", ""))
        return str(ready[-1]["metadata"]["name"])

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
        ready: list[Mapping[str, Any]] = []
        for pod in pods.get("items", []):
            metadata = pod.get("metadata", {})
            status = pod.get("status", {})
            conditions = status.get("conditions") or []
            if (
                not metadata.get("deletionTimestamp")
                and status.get("phase") == "Running"
                and any(
                    item.get("type") == "Ready" and item.get("status") == "True"
                    for item in conditions
                )
            ):
                ready.append(pod)
        if not ready:
            raise CheckError(f"no Ready executor Pod for cluster {target.cluster_id}")
        ready.sort(key=lambda item: item["metadata"].get("creationTimestamp", ""))
        return str(ready[-1]["metadata"]["name"])

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
        endpoint = str(cluster_data.get("server", ""))
        expected_endpoint = str(eks_description["cluster"]["endpoint"])
        context_cluster_name = str(contexts[0].get("context", {}).get("cluster", ""))
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
        if expected_ca_bytes and kube_ca_bytes and not ca_matches:
            raise CheckError(f"kubeconfig CA does not match EKS {expected_arn}")
        return {
            "expected_arn": expected_arn,
            "context_name": contexts[0].get("name"),
            "context_cluster_name": context_cluster_name,
            "endpoint_matches": endpoint == expected_endpoint,
            "ca_matches": ca_matches,
        }

    def preflight(self) -> None:
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
        write_json(self.run_dir / "execution-scope.json", scope)
        print("preflight: PASS", flush=True)

    def run(self) -> int:
        started_at = utc_now()
        try:
            if not self.predecessor.get("valid", True):
                raise CheckError("formal predecessor evidence is not PASS")
            self.preflight()
            {
                "GF-REGIONAL-BLAST-001": self.blast_001,
                "GF-REGIONAL-BLAST-002": self.blast_002,
                "GF-REGIONAL-BLAST-003": self.blast_003,
                "GF-REGIONAL-BLAST-004": self.blast_004,
            }[self.case_id]()
        except CheckError as exc:
            write_json(
                self.run_dir / "phase-summary.json",
                {
                    "phase": "爆炸半径",
                    "case_id": self.case_id,
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "status": "FAIL",
                    "error": str(exc),
                    "cases": self.case_statuses,
                },
            )
            print(f"STOP: {exc}", file=sys.stderr, flush=True)
            return 1
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
