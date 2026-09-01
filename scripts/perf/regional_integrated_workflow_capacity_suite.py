#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import signal
import sys
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

if __package__:
    from .regional_action_capacity_suite import (
        aggregate_executor_documents,
        collect_executor_logs,
        executor_identity,
        executor_job,
    )
    from .regional_capacity_results import artifact_dir, move_to_aborted, write_status
    from .regional_capacity_registry import (
        AWS_REGION,
        CONNECTION_SECRET,
        control,
        dataplane,
        register,
        run as shell_run,
        validate_registry_target,
    )
    from .regional_capacity_suite import (
        DEFAULT_ARTIFACT_ROOT,
        START_GATE_CONFIGMAP,
        aggregate,
        aurora_window,
        build_job,
        collect_logs,
        control_pods,
        drain_targets,
        postgres_counters,
        prepare_start_gate,
        processor_priority_latency,
        publish_fixtures,
        queue_drain,
        release_identity,
        release_start_gate,
        scrape_cgroup,
        scrape_metrics,
        teardown,
        upsert_configmap,
        wait_for_job,
        wait_for_load_pods,
    )
else:
    from regional_action_capacity_suite import (
        aggregate_executor_documents,
        collect_executor_logs,
        executor_identity,
        executor_job,
    )
    from regional_capacity_results import artifact_dir, move_to_aborted, write_status
    from regional_capacity_registry import (
        AWS_REGION,
        CONNECTION_SECRET,
        control,
        dataplane,
        register,
        run as shell_run,
        validate_registry_target,
    )
    from regional_capacity_suite import (
        DEFAULT_ARTIFACT_ROOT,
        START_GATE_CONFIGMAP,
        aggregate,
        aurora_window,
        build_job,
        collect_logs,
        control_pods,
        drain_targets,
        postgres_counters,
        prepare_start_gate,
        processor_priority_latency,
        publish_fixtures,
        queue_drain,
        release_identity,
        release_start_gate,
        scrape_cgroup,
        scrape_metrics,
        teardown,
        upsert_configmap,
        wait_for_job,
        wait_for_load_pods,
    )


PERF_DIR = Path(__file__).resolve().parent
LOAD_JOB = "gpu-fault-perf-suite-burst"
EXECUTOR_JOB = "gpu-fault-integrated-action-executors"
EXECUTOR_CONFIGMAP = "gpu-fault-integrated-action-executor"
TERMINAL_WORKFLOWS = {"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}
TERMINAL_COMMANDS = {"SUCCEEDED", "FAILED", "CANCELLED"}
FORMAL_NODES_PER_CLUSTER = 256
FORMAL_WORKFLOWS_PER_CLUSTER = 4
FORMAL_COMMANDS_PER_CLUSTER = 30
FORMAL_AURORA_MIN_ACU = 124.0
FORMAL_AURORA_MAX_ACU = 128.0
FORMAL_MODELS = {
    32: {
        "xid": 500,
        "sxid": 500,
        "gpu_evidence": 500,
        "host_evidence": 500,
        "p100": 24576,
        "total": 26576,
    },
    50: {
        "xid": 781,
        "sxid": 781,
        "gpu_evidence": 781,
        "host_evidence": 781,
        "p100": 38400,
        "total": 41524,
    },
}


@contextmanager
def cleanup_signal_guard() -> Iterator[None]:
    guarded = tuple(
        value
        for name in ("SIGINT", "SIGTERM", "SIGHUP")
        if (value := getattr(signal, name, None)) is not None
    )
    previous: dict[signal.Signals, object] = {}
    try:
        for value in guarded:
            previous[value] = signal.getsignal(value)
            signal.signal(value, signal.SIG_IGN)
    except (OSError, ValueError):
        for value, handler in previous.items():
            signal.signal(value, handler)
        previous.clear()
    try:
        yield
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)


def integrated_workload_residuals() -> dict[str, list[str]]:
    jobs = []
    for job in (LOAD_JOB, EXECUTOR_JOB):
        jobs.extend(
            dataplane(
                "get",
                "job",
                job,
                "--ignore-not-found",
                "-o",
                "name",
                timeout=60,
            ).split()
        )
    pods = dataplane(
        "get",
        "pod",
        "-l",
        f"job-name in ({LOAD_JOB},{EXECUTOR_JOB})",
        "-o",
        "name",
        timeout=60,
    ).split()
    return {"jobs": sorted(jobs), "pods": sorted(pods)}


def wait_for_integrated_workload_cleanup(
    *,
    timeout_seconds: int = 180,
    sample_seconds: int = 2,
) -> dict[str, list[str]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        residuals = integrated_workload_residuals()
        if not residuals["jobs"] and not residuals["pods"]:
            return residuals
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "integrated workload cleanup left residual resources: "
                + json.dumps(residuals, sort_keys=True)
            )
        time.sleep(sample_seconds)


def aws_json(*arguments: str, timeout: int = 120) -> dict:
    completed = shell_run(
        ["aws", *arguments, "--output", "json"],
        timeout=timeout,
    )
    return json.loads(completed.stdout)


def aurora_capacity_preflight(
    cluster_id: str,
    *,
    timeout_seconds: int,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last: dict = {}
    while time.monotonic() < deadline:
        cluster = aws_json(
            "rds",
            "describe-db-clusters",
            "--region",
            AWS_REGION,
            "--db-cluster-identifier",
            cluster_id,
        )["DBClusters"][0]
        scaling = cluster.get("ServerlessV2ScalingConfiguration") or {}
        members = cluster.get("DBClusterMembers") or []
        instances = {}
        for member in members:
            instance_id = member["DBInstanceIdentifier"]
            instance = aws_json(
                "rds",
                "describe-db-instances",
                "--region",
                AWS_REGION,
                "--db-instance-identifier",
                instance_id,
            )["DBInstances"][0]
            metrics = aws_json(
                "cloudwatch",
                "get-metric-statistics",
                "--region",
                AWS_REGION,
                "--namespace",
                "AWS/RDS",
                "--metric-name",
                "ServerlessDatabaseCapacity",
                "--dimensions",
                f"Name=DBInstanceIdentifier,Value={instance_id}",
                "--start-time",
                (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "--end-time",
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "--period",
                "60",
                "--statistics",
                "Maximum",
            )
            points = sorted(
                metrics.get("Datapoints") or [],
                key=lambda item: item["Timestamp"],
            )
            instances[instance_id] = {
                "writer": bool(member.get("IsClusterWriter")),
                "status": instance.get("DBInstanceStatus"),
                "class": instance.get("DBInstanceClass"),
                "actual_acu": (float(points[-1]["Maximum"]) if points else None),
            }
        last = {
            "cluster_id": cluster_id,
            "cluster_status": cluster.get("Status"),
            "min_acu": scaling.get("MinCapacity"),
            "max_acu": scaling.get("MaxCapacity"),
            "instances": instances,
        }
        if (
            last["cluster_status"] == "available"
            and float(last["min_acu"] or 0) == FORMAL_AURORA_MIN_ACU
            and float(last["max_acu"] or 0) == FORMAL_AURORA_MAX_ACU
            and len(instances) == 2
            and all(
                item["status"] == "available"
                and item["class"] == "db.serverless"
                and float(item["actual_acu"] or 0) >= FORMAL_AURORA_MIN_ACU
                for item in instances.values()
            )
        ):
            return last
        time.sleep(10)
    raise RuntimeError(f"Aurora formal capacity preflight failed: {last}")


def ensure_aurora_capacity(
    cluster_id: str,
    *,
    timeout_seconds: int,
    configure: bool,
) -> dict:
    cluster = aws_json(
        "rds",
        "describe-db-clusters",
        "--region",
        AWS_REGION,
        "--db-cluster-identifier",
        cluster_id,
    )["DBClusters"][0]
    scaling = cluster.get("ServerlessV2ScalingConfiguration") or {}
    initial = {
        "min_acu": scaling.get("MinCapacity"),
        "max_acu": scaling.get("MaxCapacity"),
    }
    modified = (
        float(initial["min_acu"] or 0) != FORMAL_AURORA_MIN_ACU
        or float(initial["max_acu"] or 0) != FORMAL_AURORA_MAX_ACU
    )
    if modified:
        if not configure:
            raise RuntimeError(
                "Aurora is not at 124/128 and automatic capacity configuration "
                "was not authorized"
            )
        shell_run(
            [
                "aws",
                "rds",
                "modify-db-cluster",
                "--region",
                AWS_REGION,
                "--db-cluster-identifier",
                cluster_id,
                "--serverless-v2-scaling-configuration",
                "MinCapacity=124,MaxCapacity=128",
                "--apply-immediately",
            ],
            timeout=120,
        )
    result = aurora_capacity_preflight(
        cluster_id,
        timeout_seconds=timeout_seconds,
    )
    result["initial_scaling"] = initial
    result["scaling_modified"] = modified
    return result


def remediation_budget_preflight(clusters: int) -> dict:
    expected = {
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": str(128 if clusters == 32 else 200),
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER": "4",
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS": "4",
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE": "1",
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN": "1",
    }
    pods = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-control-worker",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    ).split()
    if not pods:
        raise RuntimeError("no Running control-worker Pods for budget check")
    actual = {}
    names = sorted(expected)
    for pod in pods:
        values = control(
            "exec",
            pod,
            "--",
            "python3",
            "-c",
            (
                "import json,os; print(json.dumps({"
                + ",".join(f"{name!r}:os.environ.get({name!r})" for name in names)
                + "},sort_keys=True))"
            ),
        )
        actual[pod] = json.loads(values.splitlines()[-1])
    mismatched = {pod: values for pod, values in actual.items() if values != expected}
    if mismatched:
        raise RuntimeError(
            "live remediation budget does not match formal matrix: "
            + json.dumps(mismatched, sort_keys=True)
        )
    return {"expected": expected, "pods": actual}


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def seed_integrated_agents(
    *,
    run_id: str,
    clusters: int,
    workflows_per_cluster: int,
    lease_seconds: int,
    agent_identity: dict[str, object] | None,
) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    environment = [
        f"ACTION_RUN_ID={run_id}",
        f"ACTION_CLUSTERS={clusters}",
        f"ACTION_WORKFLOWS_PER_CLUSTER={workflows_per_cluster}",
        f"ACTION_AGENT_LEASE_SECONDS={lease_seconds}",
        "ACTION_SEED_MODE=integrated-agents",
    ]
    if agent_identity is not None:
        environment.append(
            "ACTION_AGENT_IDENTITY_JSON="
            + json.dumps(agent_identity, separators=(",", ":"), sort_keys=True)
        )
    output = control(
        "exec",
        "-i",
        pod,
        "--",
        "env",
        *environment,
        "python3",
        "-",
        stdin=(PERF_DIR / "seed_regional_action_workflows.py").read_bytes(),
        timeout=180,
    )
    return json.loads(output.splitlines()[-1])


def refresh_integrated_agents(
    *,
    run_id: str,
    clusters: int,
    workflows_per_cluster: int,
    lease_seconds: int,
) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    output = control(
        "exec",
        "-i",
        pod,
        "--",
        "env",
        f"ACTION_RUN_ID={run_id}",
        f"ACTION_CLUSTERS={clusters}",
        f"ACTION_WORKFLOWS_PER_CLUSTER={workflows_per_cluster}",
        f"ACTION_AGENT_LEASE_SECONDS={lease_seconds}",
        "ACTION_SEED_MODE=integrated-heartbeats",
        "python3",
        "-",
        stdin=(PERF_DIR / "seed_regional_action_workflows.py").read_bytes(),
        timeout=180,
    )
    return json.loads(output.splitlines()[-1])


def integrated_agent_heartbeat_refresher(
    *,
    run_id: str,
    clusters: int,
    workflows_per_cluster: int,
    lease_seconds: int,
    interval_seconds: int = 30,
) -> Callable[[], None]:
    next_refresh = 0.0

    def refresh() -> None:
        nonlocal next_refresh
        now = time.monotonic()
        if now < next_refresh:
            return
        refresh_integrated_agents(
            run_id=run_id,
            clusters=clusters,
            workflows_per_cluster=workflows_per_cluster,
            lease_seconds=lease_seconds,
        )
        next_refresh = now + interval_seconds

    return refresh


def build_load_manifest(
    *,
    run_id: str,
    clusters: int,
    workflows_per_cluster: int,
    runtime_profile_version: str,
    workers: int,
    lead_seconds: int,
) -> dict:
    model = FORMAL_MODELS[clusters]
    manifest = build_job(
        "burst",
        clusters=clusters,
        nodes_per_cluster=FORMAL_NODES_PER_CLUSTER,
        xid_total=model["xid"],
        sxid_total=model["sxid"],
        gpu_evidence_total=model["gpu_evidence"],
        host_evidence_total=model["host_evidence"],
        training_heartbeat_total=0,
        workload_observation_total=0,
        correlate_attempt_faults=False,
        start_epoch=time.time() + lead_seconds,
        workers=workers,
        duration_seconds=60,
        cpu_request="2",
        cpu_limit="4",
        include_telemetry=True,
        prewarm_connections=False,
    )
    environment = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    environment.extend(
        [
            {"name": "ACTION_RUN_ID", "value": run_id},
            {
                "name": "ACTION_WORKFLOWS_PER_CLUSTER",
                "value": str(workflows_per_cluster),
            },
            {
                "name": "RUNTIME_PROFILE_VERSION",
                "value": runtime_profile_version,
            },
        ]
    )
    return manifest


def build_executor_manifest(
    *,
    clusters: int,
    executor_workers: int,
    max_seconds: int,
    idle_exit_seconds: int,
    identity: dict,
) -> dict:
    manifest = executor_job(
        clusters=clusters,
        expected_commands=FORMAL_COMMANDS_PER_CLUSTER,
        workers=executor_workers,
        delay_scale=1.0,
        lease_seconds=120,
        inject_renew_failure_once=False,
        connection_secret=CONNECTION_SECRET,
        **identity,
    )
    manifest["metadata"]["name"] = EXECUTOR_JOB
    environment = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    environment.extend(
        [
            {"name": "ACTION_MAX_SECONDS", "value": str(max_seconds)},
            {
                "name": "ACTION_IDLE_EXIT_SECONDS",
                "value": str(idle_exit_seconds),
            },
            {"name": "ACTION_MIN_CONCURRENT_COMMANDS", "value": "1"},
        ]
    )
    manifest["spec"]["activeDeadlineSeconds"] = max_seconds + 120
    return manifest


def wait_for_executor_pods(clusters: int, timeout_seconds: int = 600) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        pods = dataplane(
            "get",
            "pod",
            "-l",
            f"batch.kubernetes.io/job-name={EXECUTOR_JOB}",
            "--field-selector=status.phase=Running",
            "-o",
            "name",
        ).splitlines()
        if len(pods) == clusters:
            return
        time.sleep(2)
    raise RuntimeError("integrated executor pods did not become Running")


def control_spool_mode(expected: str) -> dict:
    pods = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    ).split()
    if not pods:
        raise RuntimeError("no Running ingress Pods for spool-mode check")
    values = {}
    for pod in pods:
        values[pod] = (
            control(
                "exec",
                pod,
                "--",
                "python3",
                "-c",
                (
                    "import os; print("
                    "os.environ.get('GPU_FAULT_TELEMETRY_SPOOL','false'))"
                ),
            )
            .strip()
            .lower()
        )
    replicas = int(
        control(
            "get",
            "deployment",
            "gpu-fault-telemetry-spool-worker",
            "-o",
            "jsonpath={.spec.replicas}",
        ).strip()
        or 0
    )
    spool_pods = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-telemetry-spool-worker",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    ).split()
    spool_values = {}
    for pod in spool_pods:
        spool_values[pod] = (
            control(
                "exec",
                pod,
                "--",
                "python3",
                "-c",
                (
                    "import os; print("
                    "os.environ.get('GPU_FAULT_TELEMETRY_SPOOL','false'))"
                ),
            )
            .strip()
            .lower()
        )
    enabled = all(value in {"1", "true", "yes", "on"} for value in values.values())
    disabled = all(
        value in {"", "0", "false", "no", "off"} for value in values.values()
    )
    spool_enabled = all(
        value in {"1", "true", "yes", "on"} for value in spool_values.values()
    )
    if expected == "enabled" and (
        not enabled or replicas <= 0 or len(spool_pods) != replicas or not spool_enabled
    ):
        raise RuntimeError("live control plane is not in spool-enabled mode")
    if expected == "disabled" and (not disabled or replicas != 0 or spool_pods):
        raise RuntimeError("live control plane is not in spool-disabled mode")
    return {
        "expected": expected,
        "ingress_values": values,
        "spool_worker_replicas": replicas,
        "spool_worker_values": spool_values,
    }


def workflow_audit(run_id: str) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    script = f"""
import json, os, psycopg, statistics
from datetime import datetime
pattern='%integrated-{run_id}-%'
terminal_workflows={{'SUCCEEDED','FAILED','BLOCKED','SUPERSEDED'}}
terminal_commands={{'SUCCEEDED','FAILED','CANCELLED'}}
def percentile(values, ratio):
    values=sorted(values)
    return values[int((len(values)-1)*ratio)] if values else None
with psycopg.connect(os.environ['GPU_FAULT_STORE_URL']) as conn:
    cur=conn.cursor()
    cur.execute(
        "SELECT key,payload FROM gpu_fault_objects "
        "WHERE kind='incident' AND payload->>'event_id' LIKE %s",
        (pattern,),
    )
    incidents=cur.fetchall()
    incident_ids=[key for key,_ in incidents]
    cur.execute(
        "SELECT key,payload FROM gpu_fault_objects "
        "WHERE kind='workflow' AND payload->>'incident_id'=ANY(%s::text[])",
        (incident_ids,),
    )
    workflows=cur.fetchall()
    workflow_ids=[key for key,_ in workflows]
    cur.execute(
        "SELECT key,payload FROM gpu_fault_objects "
        "WHERE kind='remote_command' "
        "AND payload->>'workflow_request_id'=ANY(%s::text[])",
        (workflow_ids,),
    )
    commands=cur.fetchall()
durations=[]
step_counts=[]
budget_wait=0
operation_counts={{}}
for _,item in workflows:
    durations.append(
        (
            datetime.fromisoformat(item['updated_at'])
            - datetime.fromisoformat(item['created_at'])
        ).total_seconds()
    )
    step_counts.append(len(item.get('official_steps') or []))
    for step in item.get('official_steps') or []:
        operation=step.get('operation')
        operation_counts[operation]=operation_counts.get(operation,0)+1
    budget_wait += int(item.get('remediation_budget_wait_count',0))
command_durations=[
    (
        datetime.fromisoformat(item['updated_at'])
        - datetime.fromisoformat(item['created_at'])
    ).total_seconds()
    for _,item in commands
]
out={{
    'incident_count':len(incidents),
    'context_incident_count':sum(
        bool(item.get('job_id')) and bool(item.get('attempt_id'))
        for _,item in incidents
    ),
    'workflow_count':len(workflows),
    'workflow_statuses':{{
        status:sum(item.get('status')==status for _,item in workflows)
        for status in sorted({{item.get('status') for _,item in workflows}})
    }},
    'terminal_workflow_count':sum(
        item.get('status') in terminal_workflows for _,item in workflows
    ),
    'succeeded_workflow_count':sum(
        item.get('status')=='SUCCEEDED' for _,item in workflows
    ),
    'step_count_min':min(step_counts) if step_counts else None,
    'step_count_max':max(step_counts) if step_counts else None,
    'workflow_p50_seconds':percentile(durations,0.50),
    'workflow_p95_seconds':percentile(durations,0.95),
    'workflow_p99_seconds':percentile(durations,0.99),
    'command_count':len(commands),
    'terminal_command_count':sum(
        item.get('status') in terminal_commands for _,item in commands
    ),
    'simulated_command_count':sum(
        bool((item.get('result_details') or {{}}).get('simulated'))
        for _,item in commands
    ),
    'duplicate_idempotency_keys':len(commands)-len({{
        item.get('idempotency_key') for _,item in commands
    }}),
    'duplicate_workflow_steps':len(commands)-len({{
        (item.get('workflow_request_id'),item.get('step_index'))
        for _,item in commands
    }}),
    'fencing_mismatches':sum(
        int(item.get('fencing_token',0))
        != int((item.get('workflow') or {{}}).get('fencing_token',-1))
        for _,item in commands
    ),
    'budget_wait_total':budget_wait,
    'operation_counts':operation_counts,
    'permanent_budget_waiters':sum(
        item.get('status') not in terminal_workflows
        and int(item.get('remediation_budget_wait_count',0)) > 0
        for _,item in workflows
    ),
    'command_p50_seconds':percentile(command_durations,0.50),
    'command_p95_seconds':percentile(command_durations,0.95),
    'command_p99_seconds':percentile(command_durations,0.99),
}}
print(json.dumps(out,sort_keys=True))
"""
    output = control(
        "exec",
        pod,
        "--",
        "python3",
        "-c",
        script,
        timeout=180,
    )
    return json.loads(output.splitlines()[-1])


def wait_for_workflows(
    run_id: str,
    *,
    expected_workflows: int,
    timeout_seconds: int,
    heartbeat_refresh: Callable[[], None] | None = None,
) -> tuple[dict, list[dict]]:
    history = []
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if heartbeat_refresh is not None:
            heartbeat_refresh()
        audit = workflow_audit(run_id)
        audit["elapsed_seconds"] = timeout_seconds - max(
            0.0,
            deadline - time.monotonic(),
        )
        history.append(audit)
        if (
            audit["workflow_count"] == expected_workflows
            and audit["terminal_workflow_count"] == expected_workflows
            and audit["terminal_command_count"] == audit["command_count"]
        ):
            return audit, history
        time.sleep(2)
    return workflow_audit(run_id), history


def verdict(
    summary: dict,
    *,
    expected_workflows: int,
) -> tuple[str, list[str]]:
    errors = []
    ingress = summary["ingress"]
    audit = summary["workflow"]
    executor = summary["executor"]
    model = summary["fixed_matrix"]
    if ingress.get("requests") != model["total"]:
        errors.append("fixed ingress request total changed")
    if ingress.get("job_status") != "Complete":
        errors.append("mixed ingress Job did not complete")
    paths = ingress.get("paths") or {}
    for kind, expected in (
        ("NVIDIA_KERNEL", model["xid"]),
        ("FABRIC_MANAGER_LOG", model["sxid"]),
    ):
        path = paths.get(kind) or {}
        if int(path.get("count", 0)) != expected:
            errors.append(f"{kind} count changed from {expected}")
    for kind, path in sorted(paths.items()):
        status_counts = path.get("status_counts") or {}
        for status, count in sorted(status_counts.items()):
            if int(count) and not 200 <= int(status) < 300:
                errors.append(f"{kind} returned HTTP {status}")
        if path.get("errors"):
            errors.append(f"{kind} produced client errors")
    if audit["incident_count"] != expected_workflows:
        errors.append("action-bearing incident count mismatch")
    if audit["context_incident_count"] != expected_workflows:
        errors.append("action incidents are missing attempt context")
    if audit["workflow_count"] != expected_workflows:
        errors.append("causally linked workflow count mismatch")
    if audit["succeeded_workflow_count"] != expected_workflows:
        errors.append("not all action workflows succeeded")
    if audit["terminal_command_count"] != audit["command_count"]:
        errors.append("nonterminal remote commands remain")
    if audit["simulated_command_count"] != audit["command_count"]:
        errors.append("a remote command was not completed by the simulator")
    if not (
        audit["step_count_min"] is not None
        and audit["step_count_min"] >= 8
        and audit["step_count_max"] <= 12
    ):
        errors.append("workflow step count is outside 8..12")
    for operation in (
        "MARK_UNSCHEDULABLE",
        "STOP_WORKLOADS",
        "COLLECT_DIAGNOSTIC_BUNDLE",
        "RESET_GPU",
        "RESET_ALL_GPUS_NVSWITCHES",
        "RESTART_NODE",
        "RESTORE_SCHEDULING",
    ):
        if int((audit.get("operation_counts") or {}).get(operation, 0)) <= 0:
            errors.append(f"workflow operation coverage is missing {operation}")
    for key in (
        "duplicate_idempotency_keys",
        "duplicate_workflow_steps",
        "fencing_mismatches",
        "permanent_budget_waiters",
    ):
        if int(audit.get(key, 0)):
            errors.append(f"{key} is nonzero")
    for key in ("duplicate_claims", "claim_errors", "result_errors"):
        if int(executor.get(key, 0)):
            errors.append(f"executor {key} is nonzero")
    expected_commands = summary["clusters"] * FORMAL_COMMANDS_PER_CLUSTER
    if executor.get("expected_commands") != expected_commands:
        errors.append("executor expected command total changed")
    if executor.get("completed_commands") != expected_commands:
        errors.append("executor did not complete every remote command")
    if int(executor.get("long_commands", 0)) and not int(executor.get("renewals", 0)):
        errors.append("long simulated actions produced no lease renewals")
    return ("PASS" if not errors else "FAIL", errors)


def purge_integrated_rows(run_id: str) -> None:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    script = f"""
import os, psycopg
pattern='%integrated-{run_id}-%'
node_pattern='integrated-node-{run_id}-%'
with psycopg.connect(os.environ['GPU_FAULT_STORE_URL'],autocommit=True) as conn:
    cur=conn.cursor()
    cur.execute(
        "SELECT key FROM gpu_fault_objects "
        "WHERE kind='incident' AND payload->>'event_id' LIKE %s",
        (pattern,),
    )
    incidents=[row[0] for row in cur.fetchall()]
    cur.execute(
        "SELECT key FROM gpu_fault_objects "
        "WHERE kind='workflow' AND payload->>'incident_id'=ANY(%s::text[])",
        (incidents,),
    )
    workflows=[row[0] for row in cur.fetchall()]
    cur.execute(
        "DELETE FROM gpu_fault_objects WHERE kind='remote_command' "
        "AND payload->>'workflow_request_id'=ANY(%s::text[])",
        (workflows,),
    )
    cur.execute(
        "DELETE FROM gpu_fault_objects "
        "WHERE kind='workflow' AND key=ANY(%s::text[])",
        (workflows,),
    )
    cur.execute(
        "DELETE FROM gpu_fault_objects "
        "WHERE kind='incident' AND key=ANY(%s::text[])",
        (incidents,),
    )
    cur.execute(
        "DELETE FROM gpu_fault_objects WHERE kind='agent' "
        "AND payload->>'node_id' LIKE %s",
        (node_pattern,),
    )
"""
    control("exec", pod, "--", "python3", "-c", script, check=False, timeout=180)


def execute_integrated_jobs(
    args: argparse.Namespace,
    *,
    run_id: str,
    artifacts: Path,
    registry_scope: str,
    spool: dict,
    seeded: dict,
    expected_workflows: int,
    aurora_preflight: dict,
    heartbeat_refresh: Callable[[], None],
) -> int:
    publish_fixtures("burst")
    upsert_configmap(
        EXECUTOR_CONFIGMAP,
        text={
            "benchmark_regional_action_executor.py": (
                PERF_DIR / "benchmark_regional_action_executor.py"
            ).read_text()
        },
    )
    prepare_start_gate()
    load_manifest = build_load_manifest(
        run_id=run_id,
        clusters=args.clusters,
        workflows_per_cluster=args.workflows_per_cluster,
        runtime_profile_version=str(seeded["runtime_profile_version"]),
        workers=args.ingress_workers,
        lead_seconds=args.lead_seconds,
    )
    executor_manifest = build_executor_manifest(
        clusters=args.clusters,
        executor_workers=args.executor_workers,
        max_seconds=args.timeout_seconds,
        idle_exit_seconds=args.executor_idle_seconds,
        identity=executor_identity(require_dataplane_deployment=True),
    )
    executor_manifest["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"] = (
        EXECUTOR_CONFIGMAP
    )
    for manifest, name in (
        (load_manifest, "load-job.json"),
        (executor_manifest, "executor-job.json"),
    ):
        (artifacts / name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    for job in (LOAD_JOB, EXECUTOR_JOB):
        dataplane(
            "delete",
            "job",
            job,
            "--ignore-not-found",
            "--wait=true",
            check=False,
        )
    pods = control_pods()
    metrics_before = scrape_metrics(pods)
    cgroup_before = scrape_cgroup(pods)
    postgres_before = postgres_counters()
    target_queue, target_spool = drain_targets(metrics_before)
    dataplane("apply", "-f", "-", stdin=json.dumps(executor_manifest).encode())
    dataplane("apply", "-f", "-", stdin=json.dumps(load_manifest).encode())
    wait_for_executor_pods(args.clusters)
    wait_for_load_pods(LOAD_JOB, args.clusters, timeout_seconds=600)
    heartbeat_refresh()
    start_epoch = time.time() + args.lead_seconds
    release_start_gate(start_epoch)
    ingress_status = wait_for_job(
        LOAD_JOB,
        args.timeout_seconds + args.lead_seconds,
    )
    audit, history = wait_for_workflows(
        run_id,
        expected_workflows=expected_workflows,
        timeout_seconds=args.timeout_seconds,
        heartbeat_refresh=heartbeat_refresh,
    )
    executor_status = wait_for_job(
        EXECUTOR_JOB,
        args.executor_idle_seconds + 180,
    )
    window_end = time.time()
    load_documents = collect_logs(LOAD_JOB, artifacts / "load-pods")
    executor_documents = collect_executor_logs(
        artifacts / "executor-pods",
        job_name=EXECUTOR_JOB,
    )
    ingress = aggregate(load_documents)
    ingress.update(
        {
            "job_status": ingress_status,
            "requests": sum(int(item.get("events", 0)) for item in load_documents),
        }
    )
    executor = aggregate_executor_documents(executor_documents)
    executor["job_status"] = executor_status
    drain = queue_drain(
        pods,
        target_queue_depth=target_queue,
        target_spool_depth=target_spool,
    )
    metrics_after = scrape_metrics(pods)
    cgroup_after = scrape_cgroup(pods)
    postgres_after = postgres_counters()
    summary = {
        "run_id": run_id,
        "registry_scope": registry_scope,
        "spool_mode": spool,
        "clusters": args.clusters,
        "fixed_matrix": {
            **FORMAL_MODELS[args.clusters],
            "p0": FORMAL_MODELS[args.clusters]["xid"]
            + FORMAL_MODELS[args.clusters]["sxid"],
            "p50": FORMAL_MODELS[args.clusters]["gpu_evidence"]
            + FORMAL_MODELS[args.clusters]["host_evidence"],
        },
        "expected_workflows": expected_workflows,
        "seeded": seeded,
        "ingress": ingress,
        "workflow": audit,
        "executor": executor,
        "queue_drain": drain,
        "aurora_preflight": aurora_preflight,
        "aurora": aurora_window(
            start_epoch,
            window_end,
            aurora_instance=next(
                instance_id
                for instance_id, item in aurora_preflight["instances"].items()
                if item["writer"]
            ),
        ),
        "cpu_throttling": {
            pod: {
                key: int(after[key]) - int(cgroup_before.get(pod, {}).get(key, 0))
                for key in ("nr_throttled", "throttled_usec")
                if key in after
            }
            for pod, after in cgroup_after.items()
        },
        "processor_priority_latency": processor_priority_latency(),
        "postgres_deltas": {
            key: (postgres_after.get(key) or 0) - (postgres_before.get(key) or 0)
            for key in ("deadlocks", "xact_commit", "xact_rollback")
        },
        "window_start_epoch": start_epoch,
        "window_end_epoch": window_end,
    }
    status, errors = verdict(summary, expected_workflows=expected_workflows)
    summary["status"] = status
    summary["errors"] = errors
    for name, value in (
        ("summary.json", summary),
        ("workflow-history.json", history),
        ("metrics-before.json", metrics_before),
        ("metrics-after.json", metrics_after),
        ("cgroup-before.json", cgroup_before),
        ("cgroup-after.json", cgroup_after),
        ("postgres-before.json", postgres_before),
        ("postgres-after.json", postgres_after),
    ):
        (artifacts / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    return int(status != "PASS")


def run(args: argparse.Namespace) -> int:
    if args.clusters not in FORMAL_MODELS:
        raise ValueError("formal integrated run requires 32 or 50 clusters")
    if args.workflows_per_cluster != FORMAL_WORKFLOWS_PER_CLUSTER:
        raise ValueError("formal integrated run requires 4 workflows per cluster")
    registry_scope = validate_registry_target(
        allow_live_registry=args.allow_live_registry,
        confirmation=args.confirm_live_registry,
    )
    if registry_scope != "live":
        raise ValueError("integrated workflow capacity must target the live registry")
    spool = control_spool_mode(args.spool_mode)
    run_id = args.suite_id or datetime.now(timezone.utc).strftime(
        "integrated%Y%m%dT%H%M%S"
    )
    identity = release_identity()
    artifacts = artifact_dir(
        args.artifact_root,
        args.label or f"integrated-workflow-{args.clusters}c-{args.spool_mode}",
        identity["release_id"] or "unknown-release",
    )
    (artifacts / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "registry_scope": registry_scope,
                "spool_mode": spool,
                "clusters": args.clusters,
                "nodes_per_cluster": FORMAL_NODES_PER_CLUSTER,
                "workflows_per_cluster": args.workflows_per_cluster,
                "expected_workflows": (args.clusters * args.workflows_per_cluster),
                "fixed_matrix": {
                    **FORMAL_MODELS[args.clusters],
                    "p0": FORMAL_MODELS[args.clusters]["xid"]
                    + FORMAL_MODELS[args.clusters]["sxid"],
                    "p50": FORMAL_MODELS[args.clusters]["gpu_evidence"]
                    + FORMAL_MODELS[args.clusters]["host_evidence"],
                },
                "action_mix_per_cluster": [
                    "XID48_RESET_GPU",
                    "XID48_RESET_GPU",
                    "XID79_RESTART_NODE",
                    "SXID11001_DIAGNOSTIC_FABRIC_RESET",
                ],
                "simulated_reset_seconds": 8,
                "simulated_reboot_seconds": 120,
                "timeout_seconds": args.timeout_seconds,
                "executor_idle_seconds": args.executor_idle_seconds,
                "aurora_cluster_id": args.aurora_cluster_id,
                "required_aurora_min_acu": FORMAL_AURORA_MIN_ACU,
                "required_aurora_max_acu": FORMAL_AURORA_MAX_ACU,
                "started_at": datetime.now(timezone.utc).isoformat(),
                **identity,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_status(artifacts, status="running")
    expected_workflows = args.clusters * args.workflows_per_cluster
    failure: BaseException | None = None
    result = 1
    try:
        aurora_preflight = ensure_aurora_capacity(
            args.aurora_cluster_id,
            timeout_seconds=args.aurora_preflight_timeout_seconds,
            configure=args.configure_aurora_capacity,
        )
        budget_preflight = remediation_budget_preflight(args.clusters)
        (artifacts / "aurora-preflight.json").write_text(
            json.dumps(aurora_preflight, indent=2, sort_keys=True) + "\n"
        )
        (artifacts / "remediation-budget-preflight.json").write_text(
            json.dumps(budget_preflight, indent=2, sort_keys=True) + "\n"
        )
        register(
            args.clusters,
            artifacts,
            run_id=run_id,
            expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=args.synthetic_ttl_seconds),
            allow_live_registry=args.allow_live_registry,
            live_registry_confirmation=args.confirm_live_registry,
        )
        seeded = seed_integrated_agents(
            run_id=run_id,
            clusters=args.clusters,
            workflows_per_cluster=args.workflows_per_cluster,
            lease_seconds=args.timeout_seconds + 900,
            agent_identity=None,
        )
        heartbeat_refresh = integrated_agent_heartbeat_refresher(
            run_id=run_id,
            clusters=args.clusters,
            workflows_per_cluster=args.workflows_per_cluster,
            lease_seconds=args.timeout_seconds + 900,
        )
        result = execute_integrated_jobs(
            args,
            run_id=run_id,
            artifacts=artifacts,
            registry_scope=registry_scope,
            spool=spool,
            seeded=seeded,
            expected_workflows=expected_workflows,
            aurora_preflight=aurora_preflight,
            heartbeat_refresh=heartbeat_refresh,
        )
    except BaseException as exc:
        failure = exc
    finally:
        with cleanup_signal_guard():
            for job in (LOAD_JOB, EXECUTOR_JOB):
                dataplane(
                    "delete",
                    "job",
                    job,
                    "--ignore-not-found",
                    "--wait=true",
                    check=False,
                    timeout=300,
                )
            for configmap in (EXECUTOR_CONFIGMAP, START_GATE_CONFIGMAP):
                dataplane(
                    "delete",
                    "configmap",
                    configmap,
                    "--ignore-not-found",
                    check=False,
                )
            purge_integrated_rows(run_id)
            try:
                teardown(
                    purge=True,
                    deregister_clusters=True,
                    allow_live_registry=args.allow_live_registry,
                    live_registry_confirmation=args.confirm_live_registry,
                    artifacts=artifacts,
                    run_id=run_id,
                )
                workload_residuals = wait_for_integrated_workload_cleanup()
                (artifacts / "cleanup-workloads.json").write_text(
                    json.dumps(workload_residuals, indent=2, sort_keys=True) + "\n"
                )
                residuals = workflow_audit(run_id)
                (artifacts / "cleanup-residuals.json").write_text(
                    json.dumps(residuals, indent=2, sort_keys=True) + "\n"
                )
                if any(
                    int(residuals.get(key, 0))
                    for key in ("incident_count", "workflow_count", "command_count")
                ):
                    raise RuntimeError("integrated workflow cleanup left residual rows")
            except Exception as cleanup_error:
                if failure is None:
                    failure = cleanup_error
                else:
                    failure.add_note(
                        f"integrated teardown also failed: {cleanup_error}"
                    )
    if failure is not None:
        write_status(
            artifacts,
            status="aborted",
            reason=f"{type(failure).__name__}: {failure}",
        )
        move_to_aborted(args.artifact_root, artifacts)
        raise failure.with_traceback(failure.__traceback__)
    write_status(
        artifacts,
        status="ok" if result == 0 else "aborted",
        reason=None if result == 0 else "integrated verdict failed",
    )
    print(json.dumps({"artifacts": str(artifacts), "result": result}))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Live mixed-ingress and causally linked workflow capacity suite"
    )
    value.add_argument(
        "--clusters",
        type=int,
        choices=sorted(FORMAL_MODELS),
        required=True,
    )
    value.add_argument(
        "--spool-mode",
        choices=("enabled", "disabled"),
        required=True,
    )
    value.add_argument(
        "--workflows-per-cluster",
        type=int,
        default=FORMAL_WORKFLOWS_PER_CLUSTER,
    )
    value.add_argument("--ingress-workers", type=int, default=256)
    value.add_argument("--executor-workers", type=int, default=8)
    value.add_argument("--lead-seconds", type=int, default=20)
    value.add_argument("--timeout-seconds", type=int, default=1800)
    value.add_argument("--executor-idle-seconds", type=int, default=300)
    value.add_argument("--synthetic-ttl-seconds", type=int, default=3600)
    value.add_argument("--aurora-cluster-id", required=True)
    value.add_argument("--aurora-preflight-timeout-seconds", type=int, default=1800)
    value.add_argument(
        "--configure-aurora-capacity", action="store_true", required=True
    )
    value.add_argument("--confirm-aurora-scaling", required=True)
    value.add_argument("--suite-id")
    value.add_argument("--label")
    value.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    value.add_argument("--allow-live-registry", action="store_true", required=True)
    value.add_argument("--confirm-live-registry", required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.confirm_live_registry != "ALLOW_PERF_CAPACITY_LIVE_REGISTRY":
        parser().error("exact live-registry confirmation is required")
    if args.synthetic_ttl_seconds < args.timeout_seconds + 600:
        parser().error("synthetic TTL must exceed timeout by at least 600 seconds")
    if args.confirm_aurora_scaling != "SET_AURORA_MIN_124_MAX_128":
        parser().error("exact Aurora scaling confirmation is required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
