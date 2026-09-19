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
import importlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_fault.admin.diagnostics import diagnostic_text  # noqa: E402
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.acceptance_scope import current_acceptance_scope  # noqa: E402
from scripts.e2e.regional.acceptance_supervision import (  # noqa: E402
    bind_command_supervision,
)
from scripts.e2e.regional.destr013_audit_evidence import (  # noqa: E402
    SYNTHETIC_ROUTE_ENV,
    AuditError,
    arn_parts,
    json_object,
    run,
    target_evidence,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    PROVIDER_EVENT_VISIBILITY_SECONDS,
    RegionalFixtureAbort,
    RegionalFixtureError,
    install_abort_signals,
    predecessor_evidence,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    required as require_text,
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


def required(value: str, label: str) -> str:
    try:
        return require_text(value, label)
    except RegionalFixtureError:
        raise AuditError(f"{label} is required") from None


def parse_time(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f"{label} must be a nonempty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuditError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise AuditError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


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
    if ended_at <= started_at:
        errors.append("window end must be after window start")
    if ended_at > now or started_at > now:
        errors.append("window must not extend into the future")
    if errors:
        return errors, provisional
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


def timeline_entries(value: Any, label: str) -> list[dict[str, Any]]:
    """The timeline under an ``entries``/``transitions`` key, or nothing.

    The runners' two timeline shapes (``timeline.json`` entries and
    ``step-timeline.json`` transitions) carry ``observed_at`` on every element.
    Collectors reuse the same keys for lists that are not timelines -- per-Pod
    log classifications, coverage probes keyed ``probed_at`` -- and an empty
    list records nothing; none of those is judged, and none contributes a run
    timestamp. A non-list, or a list where only some elements carry
    ``observed_at``, is a timeline whose evidence went missing and stays an
    integrity failure.
    """

    if not isinstance(value, list):
        raise AuditError(f"destructive evidence timeline is malformed: {label}")
    carries = [isinstance(entry, dict) and "observed_at" in entry for entry in value]
    if not any(carries):
        return []
    if not all(carries):
        raise AuditError(f"destructive evidence timeline is malformed: {label}")
    return cast(list[dict[str, Any]], value)


def run_timestamps(run_dir: Path) -> list[dict[str, Any]]:
    """Every run timestamp recorded under the destructive cases' evidence.

    Top-level keys of each JSON document, plus the ``observed_at`` of timeline
    and step-transition entries (``timeline_entries`` decides which lists those
    are). Nothing deeper: a store snapshot embeds records (agents, profiles)
    whose own timestamps predate the run.
    """

    def unreadable(_error: OSError) -> None:
        raise AuditError("destructive evidence directory cannot be read")

    found: list[dict[str, Any]] = []
    try:
        cases = sorted((run_dir / "cases").iterdir())
    except OSError:
        raise AuditError(
            "--run-dir destructive evidence inventory cannot be read"
        ) from None
    for case_dir in cases:
        if (
            not case_dir.name.startswith("GF-REGIONAL-DESTR-")
            or case_dir.name == CASE_ID
        ):
            continue
        if not case_dir.is_dir() or case_dir.is_symlink():
            raise AuditError("destructive evidence case directory is invalid")
        case_count = len(found)
        paths: list[Path] = []
        for parent, directories, files in os.walk(case_dir, onerror=unreadable):
            if any((Path(parent) / name).is_symlink() for name in directories):
                raise AuditError("destructive evidence directory must not be a symlink")
            paths.extend(
                Path(parent) / name for name in files if name.endswith(".json")
            )
        for path in sorted(paths):
            label = str(path.relative_to(run_dir))
            if path.is_symlink():
                raise AuditError(f"destructive evidence must not be a symlink: {label}")
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise AuditError(
                    f"destructive evidence cannot be read as JSON: {label}"
                ) from None
            if not isinstance(document, dict):
                continue
            candidates: list[tuple[str, Any]] = [
                (key, document[key]) for key in RUN_TIMESTAMP_KEYS if key in document
            ]
            for entries_key in ("entries", "transitions"):
                if entries_key not in document:
                    continue
                candidates.extend(
                    (f"{entries_key}[].observed_at", entry["observed_at"])
                    for entry in timeline_entries(
                        document[entries_key], f"{label} {entries_key}"
                    )
                )
            for key, raw in candidates:
                parsed = parse_time(raw, f"destructive evidence {label} {key}")
                found.append({"path": label, "key": key, "at": parsed.isoformat()})
        if len(found) == case_count:
            raise AuditError(
                f"destructive case has no run timestamp evidence: {case_dir.name}"
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
        if not (
            started_at <= parse_time(item["at"], "run evidence timestamp") <= ended_at
        )
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
    *,
    cluster_arn: str,
    cluster_name: str,
    role_arn: str,
) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    account = arn_parts(cluster_arn, "sagemaker")[4]
    for event_name in EVENT_NAMES:
        result[event_name] = []
        token = ""
        seen: set[str] = set()
        for _page in range(1000):
            value = json_object(
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
                        "--no-paginate",
                        *(["--next-token", token] if token else []),
                    ]
                ).stdout,
                "CloudTrail page",
            )
            if not isinstance(value.get("Events"), list):
                raise AuditError("CloudTrail page has no complete Events inventory")
            for item in value["Events"]:
                if not isinstance(item, dict):
                    raise AuditError("CloudTrail event is malformed")
                detail = json_object(
                    item.get("CloudTrailEvent"), "CloudTrail event detail"
                )
                if detail.get("eventSource") != "sagemaker.amazonaws.com":
                    if not detail.get("eventSource"):
                        raise AuditError("CloudTrail event source is missing")
                    continue
                at = parse_time(detail.get("eventTime"), "CloudTrail event time")
                if (
                    detail.get("eventName") != event_name
                    or item.get("EventName") != event_name
                    or detail.get("awsRegion") != region
                    or detail.get("recipientAccountId") != account
                    or not started_at <= at <= ended_at
                    or parse_time(item.get("EventTime"), "CloudTrail lookup time") != at
                ):
                    raise AuditError(
                        "CloudTrail event identity/time differs from the target window"
                    )
                if event_name == POSITIVE_CONTROL_EVENT:
                    parameters = detail.get("requestParameters")
                    if not isinstance(parameters, dict):
                        raise AuditError(
                            "CloudTrail reboot control has no cluster binding"
                        )
                    targets = [
                        parameters[key]
                        for key in ("clusterName", "clusterArn")
                        if key in parameters
                    ]
                    if not targets or any(
                        not isinstance(target, str) or not target for target in targets
                    ):
                        raise AuditError(
                            "CloudTrail reboot control has no cluster binding"
                        )
                    matches = [
                        target in {cluster_name, cluster_arn} for target in targets
                    ]
                    if any(matches) and not all(matches):
                        raise AuditError(
                            "CloudTrail reboot control has conflicting cluster identities"
                        )
                    if not any(matches):
                        continue
                    identity = detail.get("userIdentity") or {}
                    issuer = (identity.get("sessionContext") or {}).get(
                        "sessionIssuer"
                    ) or {}
                    if detail.get("errorCode") or issuer.get("arn") != role_arn:
                        raise AuditError(
                            "CloudTrail reboot control is not a successful call by the bound executor role"
                        )
                # Forbidden verbs remain region-wide and actor-independent.
                result[event_name].append(
                    {"event_time": at.isoformat(), "event_name": event_name}
                )
            next_token = value.get("NextToken")
            if next_token is None:
                break
            if not isinstance(next_token, str) or not next_token or next_token in seen:
                raise AuditError("CloudTrail pagination is malformed or repeated")
            seen.add(next_token)
            token = next_token
        else:
            raise AuditError("CloudTrail pagination exceeds the audit bound")
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
        except (OSError, yaml.YAMLError):
            violations.append(f"{path.relative_to(ROOT)} cannot be parsed")
            continue
        for document in documents:
            for item in _walk(document):
                name = item.get("name")
                if name == "GPU_FAULT_ALLOW_HYPERPOD_REPLACE":
                    value = item.get("value")
                    observed.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "value": (
                                str(value).lower()
                                if str(value).lower() in {"true", "false"}
                                else "invalid"
                            ),
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
    # Test failure is evidence; transport/supervision failure still aborts.
    completed = run(
        command,
        timeout=300,
        check=False,
        env={
            "HOME": "/tmp",
            "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "AWS_CONFIG_FILE": "/dev/null",
            "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
            "AWS_EC2_METADATA_DISABLED": "true",
            "KUBECONFIG": "/dev/null",
        },
    )
    path = case_dir / "focused-tests.log"
    path.write_text(
        f"returncode={completed.returncode}\n"
        + diagnostic_text(completed.stdout + completed.stderr, sensitive=True)
        + "\n",
        encoding="utf-8",
    )
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
            "record a provisional INCOMPLETE result, never formal PASS"
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


def audit(
    arguments: argparse.Namespace, case_dir: Path, result: dict[str, Any]
) -> None:
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
    if ended_at > datetime.now(timezone.utc):
        raise AuditError("window must not extend into the future")
    result.update(window_start=started_at.isoformat(), window_end=ended_at.isoformat())
    timestamps = run_timestamps(arguments.run_dir)
    problems = coverage_errors(
        started_at=started_at, ended_at=ended_at, timestamps=timestamps
    )
    if problems:
        raise AuditError("; ".join(problems))
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
    recovery = cluster_recovery(region, cluster)
    target_arguments: dict[str, Any] = {
        "gpu_kubeconfig": kubeconfig,
        "gpu_context": context,
        "cpu_kubeconfig": cpu_kubeconfig,
        "cpu_context": cpu_context,
        "namespace": required(arguments.namespace, "namespace"),
        "region": region,
        "cluster": cluster,
        "role_arn": role_arn,
    }
    target = target_evidence(**target_arguments, recovery=recovery)
    result.update(
        release_id=target["release_id"],
        cluster_id=target["cluster_id"],
        target_binding=target,
    )
    predecessor = predecessor_evidence(
        predecessor_path,
        PREDECESSOR_CASE_ID,
        release_id=target["release_id"],
        cluster_id=target["cluster_id"],
    )
    result["predecessor"] = predecessor
    if predecessor["valid"] is not True:
        raise AuditError(
            "HA-004 predecessor is not formal PASS for this release/cluster"
        )
    inventory_before = node_inventory(region, cluster)
    environments = target["executor_environment"]
    api_env = target["api_environment"]
    cluster_arn = recovery["cluster_arn"]
    decisions = iam_decisions(region, role_arn, cluster_arn)
    events = cloudtrail_events(
        region,
        started_at,
        ended_at,
        cluster_arn=cluster_arn,
        cluster_name=cluster,
        role_arn=role_arn,
    )
    manifests = manifest_invariants()
    tests = focused_tests(case_dir)
    inventory_after = node_inventory(region, cluster)
    recovery_after = cluster_recovery(region, cluster)
    target_after = target_evidence(**target_arguments, recovery=recovery_after)
    errors = []
    if recovery.get("cluster_status") != "InService":
        errors.append("HyperPod cluster is not InService")
    if recovery.get("node_recovery") != "None":
        errors.append("HyperPod NodeRecovery is not None")
    if recovery_after != recovery or target_after != target:
        errors.append(
            "release/registration/EKS/IRSA or replica identity drifted during audit"
        )
    errors.extend(environment_errors(environments))
    errors.extend(synthetic_route_errors(api_env))
    if decisions.get("sagemaker:BatchReplaceClusterNodes") not in {
        "implicitDeny",
        "explicitDeny",
    }:
        errors.append("executor IAM permits BatchReplaceClusterNodes")
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
    timestamps_after = run_timestamps(arguments.run_dir)
    if timestamps_after != timestamps:
        errors.append("destructive run timestamp evidence changed during audit")
    errors.extend(
        coverage_errors(
            started_at=started_at, ended_at=ended_at, timestamps=timestamps_after
        )
    )
    if manifests["violations"]:
        errors.extend(cast(list[str], manifests["violations"]))
    if inventory_before != inventory_after:
        errors.append("HyperPod node inventory changed during the read-only audit")
    if not inventory_before["count"]:
        errors.append("HyperPod node inventory is empty")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    incomplete = provisional and not errors
    if provisional:
        errors.append(
            "CloudTrail conclusion is provisional; formal PASS requires a settled window"
        )
    result.update(
        {
            "verdict": "PASS" if not errors else "FAIL",
            "status": "INCOMPLETE"
            if incomplete
            else "FAILED"
            if errors
            else "COMPLETED",
            "formal_sequence_satisfied": not errors
            and not current_acceptance_scope().selective,
            "errors": errors,
            "cluster": recovery,
            "executor_environment": environments,
            "executor_replica_count": len(environments),
            "api_environment": api_env,
            "api_replica_count": len(api_env),
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


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    arguments.run_dir = arguments.run_dir.expanduser().resolve()
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": arguments.attempt,
        "verdict": "FAIL",
        "status": "RUNNING",
        "formal_sequence_satisfied": False,
        "supersedes": list(SUPERSEDED_CASE_IDS),
    }
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    code = 1
    try:
        install_abort_signals()
        try:
            bind_command_supervision(arguments.run_dir)
        except RuntimeError:
            raise AuditError(
                "acceptance run lost command supervision; independent recovery is required"
            ) from None
        result.update(current_acceptance_scope().plan_fields())
        audit(arguments, case_dir, result)
        code = 0 if result["verdict"] == "PASS" else 1
    except (
        Exception,
        ProcessSupervisionLost,
        RegionalFixtureAbort,
        KeyboardInterrupt,
    ) as exc:
        result.update(
            verdict="FAIL",
            status="FAILED",
            formal_sequence_satisfied=False,
            error=(
                str(exc)
                if isinstance(exc, AuditError)
                else f"{type(exc).__name__}: {diagnostic_text(str(exc), sensitive=True)}"
            ),
        )
        if isinstance(exc, ProcessSupervisionLost):
            result["status"] = "RECOVERY_REQUIRED"
        if isinstance(exc, RegionalFixtureAbort):
            code = 128 + exc.signum
        elif isinstance(exc, KeyboardInterrupt):
            code = 130
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
