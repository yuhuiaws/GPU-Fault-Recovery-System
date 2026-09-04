#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, cast


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    predecessor_evidence,
    required,
)


CASE_ID = "GF-REGIONAL-DESTR-013"
PREDECESSOR_CASE_ID = "GF-REGIONAL-HA-004"
EVENT_NAMES = (
    "BatchReplaceClusterNodes",
    "ReplaceClusterNodes",
    "BatchDeleteClusterNodes",
    "DeleteCluster",
    "UpdateCluster",
)


class AuditError(RuntimeError):
    pass


def parse_time(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuditError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise AuditError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def run(
    command: list[str],
    *,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise AuditError(
            f"command failed ({completed.returncode}): {' '.join(command)}; "
            f"stderr={completed.stderr.strip()}"
        )
    return completed


def kubectl(
    kubeconfig: Path,
    context: str,
    namespace: str,
    *arguments: str,
) -> str:
    return run(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "--context",
            context,
            "-n",
            namespace,
            *arguments,
        ]
    ).stdout


def cluster_recovery(region: str, cluster_name: str) -> dict[str, Any]:
    value = json.loads(
        run(
            [
                "aws",
                "sagemaker",
                "describe-cluster",
                "--region",
                region,
                "--cluster-name",
                cluster_name,
                "--output",
                "json",
            ]
        ).stdout
    )
    return {
        "cluster_name": value.get("ClusterName"),
        "cluster_arn": value.get("ClusterArn"),
        "cluster_status": value.get("ClusterStatus"),
        "node_recovery": value.get("NodeRecovery"),
        "orchestrator": value.get("Orchestrator"),
    }


def node_inventory(region: str, cluster_name: str) -> dict[str, Any]:
    value = json.loads(
        run(
            [
                "aws",
                "sagemaker",
                "list-cluster-nodes",
                "--region",
                region,
                "--cluster-name",
                cluster_name,
                "--output",
                "json",
            ]
        ).stdout
    )
    rows = sorted(
        [
            {
                "node_logical_id": item.get("NodeLogicalId"),
                "instance_id": item.get("InstanceId"),
                "instance_group": item.get("InstanceGroupName"),
                "instance_type": item.get("InstanceType"),
            }
            for item in value.get("ClusterNodeSummaries", [])
        ],
        key=lambda item: (
            str(item["node_logical_id"]),
            str(item["instance_id"]),
        ),
    )
    return {"count": len(rows), "nodes": rows}


def executor_environment(
    kubeconfig: Path,
    context: str,
    namespace: str,
) -> list[dict[str, str | None]]:
    value = json.loads(
        kubectl(
            kubeconfig,
            context,
            namespace,
            "get",
            "pod",
            "-l",
            "app=gpu-fault-cluster-executor",
            "--field-selector=status.phase=Running",
            "-o",
            "json",
        )
    )
    result = []
    for item in value.get("items", []):
        pod = str(item["metadata"]["name"])
        output = kubectl(
            kubeconfig,
            context,
            namespace,
            "exec",
            pod,
            "--",
            "python3",
            "-c",
            (
                "import json,os; print(json.dumps({"
                "'allow_replace':os.getenv("
                "'GPU_FAULT_ALLOW_HYPERPOD_REPLACE'),"
                "'allow_automatic':os.getenv("
                "'GPU_FAULT_ALLOW_WITH_AUTOMATIC_NODE_RECOVERY'),"
                "'legacy_mutation':os.getenv("
                "'GPU_FAULT_ALLOW_HYPERPOD_MUTATION')}))"
            ),
        )
        environment = json.loads(output.splitlines()[-1])
        result.append(
            {
                "pod": pod,
                "allow_replace": environment.get("allow_replace"),
                "allow_automatic": environment.get("allow_automatic"),
                "legacy_mutation": environment.get("legacy_mutation"),
            }
        )
    return sorted(result, key=lambda item: str(item["pod"]))


def iam_decisions(region: str, role_arn: str, cluster_arn: str) -> dict[str, str]:
    # Simulate against this cluster, not the default `*` resource. The Executor
    # policy scopes its SageMaker grants to one cluster ARN, so a `*` simulation
    # answers "implicitDeny" for every action -- including the reboot this case
    # relies on staying usable, and including a replace the policy might actually
    # grant on this very cluster. An unscoped deny would therefore satisfy the
    # replace assertion below without testing anything, which is exactly the
    # defect DESTR-011 hit on 2026-09-04.
    value = json.loads(
        run(
            [
                "aws",
                "iam",
                "simulate-principal-policy",
                "--policy-source-arn",
                role_arn,
                "--action-names",
                "sagemaker:BatchReplaceClusterNodes",
                "sagemaker:BatchRebootClusterNodes",
                "--resource-arns",
                cluster_arn,
                "--region",
                region,
                "--output",
                "json",
            ]
        ).stdout
    )
    return {
        str(item["EvalActionName"]): str(item["EvalDecision"])
        for item in value.get("EvaluationResults", [])
    }


def cloudtrail_events(
    region: str,
    started_at: datetime,
    ended_at: datetime,
) -> dict[str, list[dict[str, str]]]:
    result = {}
    for event_name in EVENT_NAMES:
        value = json.loads(
            run(
                [
                    "aws",
                    "cloudtrail",
                    "lookup-events",
                    "--region",
                    region,
                    "--start-time",
                    started_at.isoformat(),
                    "--end-time",
                    ended_at.isoformat(),
                    "--lookup-attributes",
                    f"AttributeKey=EventName,AttributeValue={event_name}",
                    "--output",
                    "json",
                ]
            ).stdout
        )
        result[event_name] = [
            {
                "event_time": str(item.get("EventTime") or ""),
                "event_name": str(item.get("EventName") or ""),
                "username": str(item.get("Username") or ""),
            }
            for item in value.get("Events", [])
        ]
    return result


def _walk(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def manifest_invariants() -> dict[str, Any]:
    yaml = importlib.import_module("yaml")

    executor = ROOT / "deploy/dataplane/cluster-action-executor.yaml"
    violations = []
    observed = []
    for path in sorted((ROOT / "deploy").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if (
            "GPU_FAULT_ALLOW_HYPERPOD_REPLACE" not in text
            and "GPU_FAULT_ALLOW_HYPERPOD_MUTATION" not in text
        ):
            continue
        try:
            documents = list(yaml.safe_load_all(text))
        except (OSError, yaml.YAMLError) as exc:
            violations.append(f"{path.relative_to(ROOT)} cannot be parsed: {exc}")
            continue
        for document in documents:
            for item in _walk(document):
                name = item.get("name")
                if name == "GPU_FAULT_ALLOW_HYPERPOD_REPLACE":
                    value = item.get("value")
                    observed.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "value": value,
                        }
                    )
                    if str(value).lower() != "false":
                        violations.append(
                            f"{path.relative_to(ROOT)} enables provider replace"
                        )
                if name == "GPU_FAULT_ALLOW_HYPERPOD_MUTATION":
                    violations.append(
                        f"{path.relative_to(ROOT)} contains legacy mutation switch"
                    )
    source = executor.read_text(encoding="utf-8")
    if "GPU_FAULT_ALLOW_HYPERPOD_REPLACE" not in source:
        violations.append("cluster-action-executor.yaml has no replace invariant")
    return {
        "source_manifest": str(executor.relative_to(ROOT)),
        "observed_replace_settings": observed,
        "violations": violations,
    }


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/hyperpod/test_hyperpod.py::"
        "test_replace_env_is_rejected_by_the_design_invariant",
        "tests/hyperpod/test_hyperpod.py::"
        "test_provider_replace_can_be_disabled_without_disabling_reboot",
        "tests/regional/test_regional_release_commands.py::"
        "test_executor_iam_boundary_rejects_excess_privilege",
    ]
    completed = run(command, timeout=300)
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {"passed": True, "command": command}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Audit the final DESTR-013 provider-replacement invariants."
    )
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--attempt", type=int, default=1)
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--region", default="")
    value.add_argument("--hyperpod-cluster", default="")
    value.add_argument("--executor-role-arn", default="")
    value.add_argument("--window-start", required=True)
    value.add_argument("--window-end", required=True)
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    kubeconfig = (
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
    context = required(
        arguments.gpu_context
        or os.getenv("GPU_EKS_CONTEXT", "")
        or os.getenv("GPU_FAULT_DATAPLANE_CONTEXT", ""),
        "GPU context",
    )
    region = required(
        arguments.region
        or os.getenv("AWS_REGION", "")
        or os.getenv("AWS_DEFAULT_REGION", ""),
        "AWS Region",
    )
    cluster = required(
        arguments.hyperpod_cluster or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", ""),
        "HyperPod cluster",
    )
    role_arn = required(
        arguments.executor_role_arn or os.getenv("GPU_FAULT_EXECUTOR_ROLE_ARN", ""),
        "executor role ARN",
    )
    started_at = parse_time(arguments.window_start, "window start")
    ended_at = parse_time(arguments.window_end, "window end")
    if ended_at <= started_at:
        raise AuditError("window end must be after window start")
    predecessor_path = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir
            / "cases"
            / PREDECESSOR_CASE_ID
            / f"{PREDECESSOR_CASE_ID}.json"
        ).resolve()
    )
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": arguments.attempt,
        "verdict": "FAIL",
        "window_start": started_at.isoformat(),
        "window_end": ended_at.isoformat(),
    }
    try:
        predecessor = predecessor_evidence(
            predecessor_path,
            PREDECESSOR_CASE_ID,
        )
        inventory_before = node_inventory(region, cluster)
        recovery = cluster_recovery(region, cluster)
        environments = executor_environment(
            kubeconfig,
            context,
            arguments.namespace,
        )
        cluster_arn = str(recovery.get("cluster_arn") or "")
        if not cluster_arn:
            raise AuditError("describe-cluster did not return a cluster ARN")
        decisions = iam_decisions(region, role_arn, cluster_arn)
        events = cloudtrail_events(region, started_at, ended_at)
        manifests = manifest_invariants()
        tests = focused_tests(case_dir)
        inventory_after = node_inventory(region, cluster)
        errors = []
        if not predecessor["valid"]:
            errors.append("HA-004 predecessor evidence is not PASS")
        if recovery.get("cluster_status") != "InService":
            errors.append("HyperPod cluster is not InService")
        if recovery.get("node_recovery") != "None":
            errors.append("HyperPod NodeRecovery is not None")
        if len(environments) < 1:
            errors.append("no running executor replicas were inspected")
        if any(
            str(item.get("allow_replace") or "").lower() != "false"
            or str(item.get("allow_automatic") or "").lower()
            not in {"", "false", "none"}
            or str(item.get("legacy_mutation") or "").lower()
            not in {"", "false", "none"}
            for item in environments
        ):
            errors.append("an executor replica violates the mutation invariants")
        if decisions.get("sagemaker:BatchReplaceClusterNodes") not in {
            "implicitDeny",
            "explicitDeny",
        }:
            errors.append("executor IAM permits BatchReplaceClusterNodes")
        # Prove the deny above is specific to replace. A policy that denied every
        # SageMaker verb on this cluster would satisfy the assertion while making
        # the reboot path DESTR-002 depends on impossible, so the scoped
        # simulation has to show reboot is still allowed.
        if decisions.get("sagemaker:BatchRebootClusterNodes") != "allowed":
            errors.append("executor IAM does not allow BatchRebootClusterNodes")
        replace_events = [
            item
            for name in ("BatchReplaceClusterNodes", "ReplaceClusterNodes")
            for item in events[name]
        ]
        if replace_events:
            errors.append("CloudTrail contains provider replacement")
        if manifests["violations"]:
            errors.extend(cast(list[str], manifests["violations"]))
        if inventory_before != inventory_after:
            errors.append("read-only audit changed HyperPod node inventory")
        if not tests["passed"]:
            errors.append("focused regression tests failed")
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "predecessor": predecessor,
                "cluster": recovery,
                "executor_environment": environments,
                "iam_decisions": decisions,
                "cloudtrail_events": events,
                "manifest_invariants": manifests,
                "inventory_before": inventory_before,
                "inventory_after": inventory_after,
                "focused_tests": tests,
                "window_limitation": (
                    "CloudTrail absence only applies to the explicit query window; "
                    "configuration and IAM evidence support the forward invariant."
                ),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
