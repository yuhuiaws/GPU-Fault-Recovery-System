#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-013: the final provider-replacement invariant of a run.

Runs after every destructive case and reads the whole run's CloudTrail window
plus the deployed executor and API environments. It absorbs the checks that
were unique to ``GF-REGIONAL-DESTR-011`` (reboot independently enabled on every
executor replica, the replica count on record), which is now superseded by
this case.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
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
    PROVIDER_EVENT_VISIBILITY_SECONDS,
    predecessor_evidence,
    required,
)


CASE_ID = "GF-REGIONAL-DESTR-013"
PREDECESSOR_CASE_ID = "GF-REGIONAL-HA-004"
SUPERSEDED_CASE_IDS = ("GF-REGIONAL-DESTR-011",)
# The forbidden mutations, plus the one *expected* mutation: a run that drove
# DESTR-002/014/016 rebooted nodes, so a window with no BatchRebootClusterNodes
# is more likely the wrong region or the wrong hours than a quiet cluster.
FORBIDDEN_EVENT_NAMES = (
    "BatchReplaceClusterNodes",
    "ReplaceClusterNodes",
    "BatchDeleteClusterNodes",
    "DeleteCluster",
    "UpdateCluster",
)
POSITIVE_CONTROL_EVENT = "BatchRebootClusterNodes"
EVENT_NAMES = (*FORBIDDEN_EVENT_NAMES, POSITIVE_CONTROL_EVENT)
SYNTHETIC_ROUTE_ENV = "GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS"
API_APP = "gpu-fault-api-ha"
# Top-level keys of a case's evidence documents that carry a run timestamp,
# and the two timeline shapes the runners write.
RUN_TIMESTAMP_KEYS = (
    "started_at",
    "completed_at",
    "updated_at",
    "observed_at",
    "recorded_at",
    "window_start",
    "window_end",
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
    # The control plane kubeconfig names its own current context, so an empty
    # ``context`` means "the kubeconfig's" rather than a literal empty name.
    selector = ["--context", context] if context else []
    return run(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            *selector,
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
                "'allow_reboot':os.getenv("
                "'GPU_FAULT_ALLOW_HYPERPOD_REBOOT'),"
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
                "allow_reboot": environment.get("allow_reboot"),
                "allow_automatic": environment.get("allow_automatic"),
                "legacy_mutation": environment.get("legacy_mutation"),
            }
        )
    return sorted(result, key=lambda item: str(item["pod"]))


def api_environment(
    kubeconfig: Path,
    context: str,
    namespace: str,
) -> list[dict[str, str | None]]:
    """Whether each running API replica still carries the synthetic route switch.

    DESTR-003/008 open the route for their window; this final invariant is
    where a window nobody closed is caught, because a route that fabricates a
    REPLACE_NODE finding for any node must not outlive the cases that needed it.
    """

    value = json.loads(
        kubectl(
            kubeconfig,
            context,
            namespace,
            "get",
            "pod",
            "-l",
            f"app={API_APP}",
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
                f"'synthetic_route':os.getenv({SYNTHETIC_ROUTE_ENV!r})}}))"
            ),
        )
        environment = json.loads(output.splitlines()[-1])
        result.append(
            {"pod": pod, "synthetic_route": environment.get("synthetic_route")}
        )
    return sorted(result, key=lambda item: str(item["pod"]))


def environment_errors(environments: list[dict[str, str | None]]) -> list[str]:
    """The executor-tier invariants, DESTR-011's included."""

    errors = []
    if len(environments) < 1:
        errors.append("no running executor replicas were inspected")
    if any(
        str(item.get("allow_replace") or "").lower() != "false"
        or str(item.get("allow_automatic") or "").lower() not in {"", "false", "none"}
        or str(item.get("legacy_mutation") or "").lower() not in {"", "false", "none"}
        for item in environments
    ):
        errors.append("an executor replica violates the mutation invariants")
    # Reboot must stay independently usable: a fleet that disabled every
    # HyperPod verb would satisfy the replace invariant while making the
    # reboot path DESTR-002 depends on impossible.
    if any(
        str(item.get("allow_reboot") or "").lower() != "true" for item in environments
    ):
        errors.append("an executor replica does not independently enable reboot")
    return errors


def synthetic_route_errors(api_env: list[dict[str, str | None]]) -> list[str]:
    errors = []
    if len(api_env) < 1:
        errors.append("no running API replicas were inspected for the synthetic route")
    open_replicas = sorted(
        str(item["pod"]) for item in api_env if item.get("synthetic_route") is not None
    )
    if open_replicas:
        errors.append(
            f"{SYNTHETIC_ROUTE_ENV} is still set on API replicas: "
            + ", ".join(open_replicas)
        )
    return errors


def cloudtrail_invariant_errors(
    events: dict[str, list[dict[str, str]]],
) -> list[str]:
    """Every forbidden mutation that appeared in the window, by name.

    ``DeleteCluster`` and ``UpdateCluster`` were collected and never judged;
    an ``UpdateCluster`` is how NodeRecovery flips to Automatic, which is the
    one provider setting the whole warm-spare design depends on staying None.
    """

    errors = []
    replace = [
        item
        for name in ("BatchReplaceClusterNodes", "ReplaceClusterNodes")
        for item in events.get(name, [])
    ]
    if replace:
        errors.append("CloudTrail contains provider replacement")
    for name in ("BatchDeleteClusterNodes", "DeleteCluster", "UpdateCluster"):
        if events.get(name):
            errors.append(f"CloudTrail contains {name}")
    return errors


def window_errors(
    *,
    started_at: datetime,
    ended_at: datetime,
    now: datetime,
    accept_recent_window: bool,
    reboot_events: int,
    expect_no_reboots: bool,
) -> tuple[list[str], bool]:
    """Errors about the query window itself, and whether the negative is provisional.

    CloudTrail delivers within 15 minutes, so a window that ends inside that
    lag has not been fully read yet; the caller either waits or accepts a
    provisional conclusion explicitly. And an empty window proves nothing on
    its own: without at least one reboot from the destructive sequence it is
    indistinguishable from a wrong region or wrong hours.
    """

    errors = []
    provisional = False
    lag = timedelta(seconds=PROVIDER_EVENT_VISIBILITY_SECONDS)
    if now - ended_at < lag:
        if accept_recent_window:
            provisional = True
        else:
            errors.append(
                "window end is inside CloudTrail's delivery lag; wait until "
                f"{(ended_at + lag).isoformat()} or pass --accept-recent-window"
            )
    if reboot_events == 0 and not expect_no_reboots:
        errors.append(
            f"no {POSITIVE_CONTROL_EVENT} in the window; an empty query cannot "
            "distinguish a quiet cluster from the wrong region or hours "
            "(pass --expect-no-reboots if the run drove no reboot case)"
        )
    if reboot_events and expect_no_reboots:
        errors.append(
            f"--expect-no-reboots was given but {reboot_events} "
            f"{POSITIVE_CONTROL_EVENT} event(s) are in the window"
        )
    return errors, provisional


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def run_timestamps(run_dir: Path) -> list[dict[str, Any]]:
    """Every run timestamp recorded under the destructive cases' evidence.

    Top-level keys of each JSON document, plus the ``observed_at`` of timeline
    and step-transition entries. Nothing deeper: a store snapshot embeds
    records (agents, profiles) whose own timestamps predate the run.
    """

    found: list[dict[str, Any]] = []
    for case_dir in sorted((run_dir / "cases").glob("GF-REGIONAL-DESTR-*")):
        for path in sorted(case_dir.rglob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(document, dict):
                continue
            candidates: list[tuple[str, Any]] = [
                (key, document.get(key)) for key in RUN_TIMESTAMP_KEYS
            ]
            for entries_key in ("entries", "transitions"):
                for entry in document.get(entries_key) or []:
                    if isinstance(entry, dict):
                        candidates.append(
                            (f"{entries_key}[].observed_at", entry.get("observed_at"))
                        )
            for key, raw in candidates:
                parsed = _parse_timestamp(raw)
                if parsed is not None:
                    found.append(
                        {
                            "path": str(path.relative_to(run_dir)),
                            "key": key,
                            "at": parsed.isoformat(),
                        }
                    )
    return found


def coverage_errors(
    *,
    started_at: datetime,
    ended_at: datetime,
    timestamps: list[dict[str, Any]],
) -> list[str]:
    """Refuse a window that does not cover the run's own recorded timestamps."""

    if not timestamps:
        return ["--run-dir carries no destructive-case timestamps to check against"]
    outside = [
        item
        for item in timestamps
        if not (started_at <= datetime.fromisoformat(item["at"]) <= ended_at)
    ]
    if not outside:
        return []
    earliest = min(item["at"] for item in timestamps)
    latest = max(item["at"] for item in timestamps)
    return [
        f"the window does not cover the run's evidence ({len(outside)} timestamp(s) "
        f"outside): evidence spans {earliest} .. {latest}"
    ]


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
    # Not ``run()``: a failing test must become ``passed: False`` in the
    # verdict, not an exception that leaves the case with no verdict at all.
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Audit the final DESTR-013 provider-replacement invariants."
    )
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--attempt", type=int, default=1)
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--cpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--region", default="")
    value.add_argument("--hyperpod-cluster", default="")
    value.add_argument("--executor-role-arn", default="")
    value.add_argument("--window-start", required=True)
    value.add_argument("--window-end", required=True)
    value.add_argument(
        "--accept-recent-window",
        action="store_true",
        help=(
            "allow a --window-end inside CloudTrail's 15-minute delivery lag; "
            "the CloudTrail conclusion is then recorded as provisional"
        ),
    )
    value.add_argument(
        "--expect-no-reboots",
        action="store_true",
        help=(
            "the run drove no reboot case, so an empty BatchRebootClusterNodes "
            "query is not a wrong-window signal"
        ),
    )
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
    cpu_kubeconfig = (
        Path(
            required(
                arguments.cpu_kubeconfig
                or os.getenv("GPU_FAULT_CONTROL_KUBECONFIG", "")
                or os.getenv("CPU_KUBECONFIG", ""),
                "CPU kubeconfig (the synthetic replacement route is read on the API tier)",
            )
        )
        .expanduser()
        .resolve()
    )
    cpu_context = (
        arguments.cpu_context
        or os.getenv("GPU_FAULT_CONTROL_CONTEXT", "")
        or os.getenv("CPU_EKS_CONTEXT", "")
    ).strip()
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
        "supersedes": list(SUPERSEDED_CASE_IDS),
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
        api_env = api_environment(cpu_kubeconfig, cpu_context, arguments.namespace)
        cluster_arn = str(recovery.get("cluster_arn") or "")
        if not cluster_arn:
            raise AuditError("describe-cluster did not return a cluster ARN")
        decisions = iam_decisions(region, role_arn, cluster_arn)
        events = cloudtrail_events(region, started_at, ended_at)
        manifests = manifest_invariants()
        tests = focused_tests(case_dir)
        inventory_after = node_inventory(region, cluster)
        timestamps = run_timestamps(arguments.run_dir)
        errors = []
        if not predecessor["valid"]:
            errors.append("HA-004 predecessor evidence is not PASS")
        if recovery.get("cluster_status") != "InService":
            errors.append("HyperPod cluster is not InService")
        if recovery.get("node_recovery") != "None":
            errors.append("HyperPod NodeRecovery is not None")
        errors.extend(environment_errors(environments))
        errors.extend(synthetic_route_errors(api_env))
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
        errors.extend(cloudtrail_invariant_errors(events))
        window_problems, provisional = window_errors(
            started_at=started_at,
            ended_at=ended_at,
            now=datetime.now(timezone.utc),
            accept_recent_window=bool(arguments.accept_recent_window),
            reboot_events=len(events.get(POSITIVE_CONTROL_EVENT, [])),
            expect_no_reboots=bool(arguments.expect_no_reboots),
        )
        errors.extend(window_problems)
        errors.extend(
            coverage_errors(
                started_at=started_at, ended_at=ended_at, timestamps=timestamps
            )
        )
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
                "executor_replica_count": len(environments),
                "api_environment": api_env,
                "iam_decisions": decisions,
                "cloudtrail_events": events,
                "cloudtrail_provisional": provisional,
                "run_evidence_timestamps": timestamps,
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
