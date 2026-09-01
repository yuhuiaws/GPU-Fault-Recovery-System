#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__:
    from .acceptance_scope import scoped_case_evidence
else:
    from acceptance_scope import scoped_case_evidence


ROOT = Path(__file__).resolve().parents[3]
CASE_ID = "GF-REGIONAL-DESTR-011"
GPU_KUBECONFIG = Path()
GPU_CONTEXT = ""
NAMESPACE = "gpu-fault-system"
ROLE_ARN = ""
HYPERPOD_CLUSTER = ""
AWS_REGION = ""


def configure(arguments: argparse.Namespace) -> None:
    global AWS_REGION
    global GPU_CONTEXT
    global GPU_KUBECONFIG
    global HYPERPOD_CLUSTER
    global NAMESPACE
    global ROLE_ARN

    kubeconfig = (
        arguments.gpu_kubeconfig
        or os.getenv("GPU_KUBECONFIG")
        or os.getenv("KUBECONFIG", "")
    )
    GPU_CONTEXT = (
        arguments.gpu_context
        or os.getenv("GPU_EKS_CONTEXT")
        or os.getenv("GPU_FAULT_DATAPLANE_CONTEXT", "")
    ).strip()
    AWS_REGION = (
        arguments.region
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION", "")
    ).strip()
    ROLE_ARN = (
        arguments.executor_role_arn or os.getenv("GPU_FAULT_EXECUTOR_ROLE_ARN", "")
    ).strip()
    HYPERPOD_CLUSTER = (
        arguments.hyperpod_cluster_name
        or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", "")
    ).strip()
    NAMESPACE = arguments.namespace
    if not all((kubeconfig, GPU_CONTEXT, AWS_REGION, ROLE_ARN, HYPERPOD_CLUSTER)):
        raise RuntimeError(
            "GPU kubeconfig/context, AWS Region, executor role ARN and "
            "HyperPod cluster name are required"
        )
    GPU_KUBECONFIG = Path(kubeconfig).expanduser().resolve()
    if not GPU_KUBECONFIG.is_file():
        raise RuntimeError("GPU kubeconfig does not exist")


def command(argv: list[str], *, timeout: int = 180) -> str:
    result = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
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


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(scoped_case_evidence(value), indent=2, sort_keys=True) + "\n"
    )
    path.chmod(0o600)


def node_inventory() -> dict:
    value = json.loads(
        command(
            [
                "aws",
                "sagemaker",
                "list-cluster-nodes",
                "--region",
                AWS_REGION,
                "--cluster-name",
                HYPERPOD_CLUSTER,
                "--output",
                "json",
            ]
        )
    )
    rows = sorted(
        [
            {
                "instance_id": item.get("InstanceId"),
                "instance_group": item.get("InstanceGroupName"),
                "instance_type": item.get("InstanceType"),
            }
            for item in value.get("ClusterNodeSummaries", [])
        ],
        key=lambda item: (
            str(item["instance_group"]),
            str(item["instance_id"]),
        ),
    )
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"count": len(rows), "sha256": digest}


def run_case(run_dir: Path, attempt: int) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    errors = []
    try:
        inventory_before = node_inventory()
        tests = command(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/hyperpod/test_hyperpod.py::"
                "test_provider_replace_can_be_disabled_without_disabling_reboot",
                "tests/hyperpod/test_hyperpod.py::"
                "test_replace_env_is_rejected_by_the_design_invariant",
                "tests/regional/test_regional_release_orchestrator.py::"
                "test_executor_iam_boundary_rejects_excess_privilege",
            ]
        )
        (case_dir / "pytest.log").write_text(tests)
        (case_dir / "pytest.log").chmod(0o600)

        pods = command(
            [
                "kubectl",
                "--kubeconfig",
                str(GPU_KUBECONFIG),
                "--context",
                GPU_CONTEXT,
                "-n",
                NAMESPACE,
                "get",
                "pod",
                "-l",
                "app=gpu-fault-cluster-executor",
                "--field-selector=status.phase=Running",
                "-o",
                "jsonpath={.items[*].metadata.name}",
            ]
        ).split()
        executor_env = []
        for pod in pods:
            value = json.loads(
                command(
                    [
                        "kubectl",
                        "--kubeconfig",
                        str(GPU_KUBECONFIG),
                        "--context",
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
                            "'allow_replace':os.environ.get("
                            "'GPU_FAULT_ALLOW_HYPERPOD_REPLACE'),"
                            "'allow_reboot':os.environ.get("
                            "'GPU_FAULT_ALLOW_HYPERPOD_REBOOT')}))"
                        ),
                    ]
                ).splitlines()[-1]
            )
            executor_env.append(value)

        simulation = json.loads(
            command(
                [
                    "aws",
                    "iam",
                    "simulate-principal-policy",
                    "--policy-source-arn",
                    ROLE_ARN,
                    "--action-names",
                    "sagemaker:BatchReplaceClusterNodes",
                    "sagemaker:BatchRebootClusterNodes",
                    "--output",
                    "json",
                ]
            )
        )
        decisions = {
            item["EvalActionName"]: item["EvalDecision"]
            for item in simulation.get("EvaluationResults", [])
        }

        now = datetime.now(timezone.utc)
        cloudtrail = json.loads(
            command(
                [
                    "aws",
                    "cloudtrail",
                    "lookup-events",
                    "--region",
                    AWS_REGION,
                    "--start-time",
                    (now - timedelta(hours=2)).isoformat(),
                    "--end-time",
                    now.isoformat(),
                    "--lookup-attributes",
                    "AttributeKey=EventSource,AttributeValue=sagemaker.amazonaws.com",
                    "--output",
                    "json",
                ]
            )
        )
        replace_events = [
            {
                "event_time": str(item.get("EventTime")),
                "event_name": item.get("EventName"),
                "username": item.get("Username"),
            }
            for item in cloudtrail.get("Events", [])
            if item.get("EventName")
            in {"BatchReplaceClusterNodes", "ReplaceClusterNodes"}
        ]
        inventory_after = node_inventory()
        if len(executor_env) != 2:
            errors.append("expected two running executor replicas")
        if any(item.get("allow_replace") != "false" for item in executor_env):
            errors.append("an executor replica allows provider replacement")
        if any(item.get("allow_reboot") != "true" for item in executor_env):
            errors.append("reboot was not independently enabled")
        if decisions.get("sagemaker:BatchReplaceClusterNodes") != "implicitDeny":
            errors.append("IAM does not implicitDeny BatchReplaceClusterNodes")
        if replace_events:
            errors.append("CloudTrail contains a provider replace event")
        if inventory_before != inventory_after:
            errors.append("HyperPod node inventory changed")
        result = {
            "case_id": CASE_ID,
            "attempt": attempt,
            "verdict": "PASS" if not errors else "FAIL",
            "errors": errors,
            "executor_env": executor_env,
            "iam_decisions": decisions,
            "replace_events": replace_events,
            "inventory_before": inventory_before,
            "inventory_after": inventory_after,
        }
    except Exception as exc:
        result = {
            "case_id": CASE_ID,
            "attempt": attempt,
            "verdict": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the DESTR-011 provider-replace safety invariant."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--gpu-kubeconfig")
    parser.add_argument("--gpu-context")
    parser.add_argument("--namespace", default="gpu-fault-system")
    parser.add_argument("--region")
    parser.add_argument("--executor-role-arn")
    parser.add_argument("--hyperpod-cluster-name")
    args = parser.parse_args()
    os.umask(0o077)
    configure(args)
    return run_case(args.run_dir, args.attempt)


if __name__ == "__main__":
    raise SystemExit(main())
