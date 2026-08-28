from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .regional_capacity_suite import (
        DEFAULT_ARTIFACT_ROOT,
        NAMESPACE,
        TOKEN_SECRET,
        artifact_dir,
        control,
        dataplane,
        move_to_aborted,
        postgres_counters,
        register,
        release_identity,
        scrape_cgroup,
        scrape_metrics,
        teardown,
        upsert_configmap,
        write_status,
    )
else:
    from regional_capacity_suite import (
        DEFAULT_ARTIFACT_ROOT,
        NAMESPACE,
        TOKEN_SECRET,
        artifact_dir,
        control,
        dataplane,
        move_to_aborted,
        postgres_counters,
        register,
        release_identity,
        scrape_cgroup,
        scrape_metrics,
        teardown,
        upsert_configmap,
        write_status,
    )


SCRIPT_CONFIGMAP = "gpu-fault-action-capacity-script"
JOB_NAME = "gpu-fault-action-capacity-executors"
TERMINAL = {"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}
REPO_ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = REPO_ROOT / "scripts" / "perf"


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def executor_job(
    *,
    clusters: int,
    expected_commands: int,
    workers: int,
    delay_scale: float,
    lease_seconds: int,
    inject_renew_failure_once: bool,
    executor_protocol_version: int,
    executor_artifact_sha256: str,
    executor_compatibility_digest: str,
) -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": NAMESPACE,
        },
        "spec": {
            "completions": clusters,
            "parallelism": clusters,
            "completionMode": "Indexed",
            "backoffLimit": 0,
            "activeDeadlineSeconds": 1200,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": ("gpu-fault-completion-watcher"),
                    "containers": [
                        {
                            "name": "executor",
                            "image": ("public.ecr.aws/docker/library/python:3.12-slim"),
                            "command": [
                                "python",
                                "/scripts/benchmark_regional_action_executor.py",
                            ],
                            "env": [
                                {
                                    "name": ("GPU_FAULT_CONTROL_PLANE_URL"),
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": ("gpu-fault-regional-connection"),
                                            "key": "control-plane-url",
                                        }
                                    },
                                },
                                {
                                    "name": "SSL_CERT_FILE",
                                    "value": "/tls/ca.crt",
                                },
                                {
                                    "name": "CLUSTERS_FILE",
                                    "value": "/tokens/clusters.json",
                                },
                                {
                                    "name": "CLUSTER_OFFSET",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": (
                                                "metadata.annotations["
                                                "'batch.kubernetes.io/"
                                                "job-completion-index']"
                                            )
                                        }
                                    },
                                },
                                {
                                    "name": "EXPECTED_COMMANDS",
                                    "value": str(expected_commands),
                                },
                                {
                                    "name": "ACTION_EXECUTOR_WORKERS",
                                    "value": str(workers),
                                },
                                {
                                    "name": "ACTION_DELAY_SCALE",
                                    "value": str(delay_scale),
                                },
                                {
                                    "name": "ACTION_LEASE_SECONDS",
                                    "value": str(lease_seconds),
                                },
                                {
                                    "name": "ACTION_INJECT_RENEW_FAILURE_ONCE",
                                    "value": str(inject_renew_failure_once).lower(),
                                },
                                {
                                    "name": "EXECUTOR_PROTOCOL_VERSION",
                                    "value": str(executor_protocol_version),
                                },
                                {
                                    "name": "EXECUTOR_ARTIFACT_SHA256",
                                    "value": executor_artifact_sha256,
                                },
                                {
                                    "name": "EXECUTOR_COMPATIBILITY_DIGEST",
                                    "value": executor_compatibility_digest,
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "250m",
                                    "memory": "128Mi",
                                },
                                "limits": {
                                    "cpu": "2",
                                    "memory": "512Mi",
                                },
                            },
                            "volumeMounts": [
                                {
                                    "name": "script",
                                    "mountPath": "/scripts",
                                    "readOnly": True,
                                },
                                {
                                    "name": "tokens",
                                    "mountPath": "/tokens",
                                    "readOnly": True,
                                },
                                {
                                    "name": "tls",
                                    "mountPath": "/tls",
                                    "readOnly": True,
                                },
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "script",
                            "configMap": {"name": SCRIPT_CONFIGMAP},
                        },
                        {
                            "name": "tokens",
                            "secret": {"secretName": TOKEN_SECRET},
                        },
                        {
                            "name": "tls",
                            "secret": {
                                "secretName": ("gpu-fault-regional-connection"),
                                "items": [
                                    {
                                        "key": "ca.crt",
                                        "path": "ca.crt",
                                    }
                                ],
                            },
                        },
                    ],
                }
            },
        },
    }


def executor_identity() -> dict:
    state = json.loads(
        control(
            "get",
            "configmap",
            "gpu-fault-regional-release-state",
            "-o",
            "jsonpath={.data.state\\.json}",
        )
    )
    deployment = json.loads(
        dataplane(
            "get",
            "deployment",
            "gpu-fault-cluster-executor",
            "-o",
            "json",
        )
    )
    environment = {
        item["name"]: item.get("value")
        for item in deployment["spec"]["template"]["spec"]["containers"][0].get(
            "env", []
        )
    }
    expected_artifact = str(state["executor_wheel_sha256"])
    expected_compatibility = str(state["component_digests"]["executor"])
    if environment.get("GPU_FAULT_EXECUTOR_ARTIFACT_SHA256") != expected_artifact:
        raise RuntimeError("live executor artifact pin differs from release state")
    if (
        environment.get("GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST")
        != expected_compatibility
    ):
        raise RuntimeError("live executor compatibility pin differs from release state")
    return {
        "executor_protocol_version": int(state["executor_protocol_version"]),
        "executor_artifact_sha256": expected_artifact,
        "executor_compatibility_digest": expected_compatibility,
    }


def seed(
    *,
    run_id: str,
    clusters: int,
    workflows_per_cluster: int,
    nodes_per_workflow: int,
) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    script = (PERF_DIR / "seed_regional_action_workflows.py").read_bytes()
    output = control(
        "exec",
        "-i",
        pod,
        "--",
        "env",
        f"ACTION_RUN_ID={run_id}",
        f"ACTION_CLUSTERS={clusters}",
        (f"ACTION_WORKFLOWS_PER_CLUSTER={workflows_per_cluster}"),
        f"ACTION_NODES_PER_WORKFLOW={nodes_per_workflow}",
        "python3",
        "-",
        stdin=script,
        timeout=300,
    )
    return json.loads(output.splitlines()[-1])


def database_snapshot(run_id: str) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    script = f"""
import json, os, psycopg
pattern = 'workflow-actionperf-{run_id}-%'
out = {{}}
with psycopg.connect(os.environ['GPU_FAULT_STORE_URL']) as conn:
    cur = conn.cursor()
    cur.execute(
        "SELECT payload->>'status', count(*) "
        "FROM gpu_fault_objects "
        "WHERE kind='workflow' AND key LIKE %s GROUP BY 1",
        (pattern,),
    )
    out['workflows'] = dict(cur.fetchall())
    cur.execute(
        "SELECT payload->>'status', count(*) "
        "FROM gpu_fault_objects "
        "WHERE kind='remote_command' "
        "AND payload->>'workflow_request_id' LIKE %s GROUP BY 1",
        (pattern,),
    )
    out['commands'] = dict(cur.fetchall())
    cur.execute(
        "SELECT min((payload->>'created_at')::timestamptz), "
        "max((payload->>'updated_at')::timestamptz) "
        "FROM gpu_fault_objects "
        "WHERE kind='workflow' AND key LIKE %s",
        (pattern,),
    )
    first, last = cur.fetchone()
    out['first_created_at'] = first.isoformat() if first else None
    out['last_updated_at'] = last.isoformat() if last else None
print(json.dumps(out))
"""
    output = control(
        "exec",
        pod,
        "--",
        "python3",
        "-c",
        script,
        timeout=120,
    )
    return json.loads(output.splitlines()[-1])


def database_details(run_id: str) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    script = f"""
import json, os, re, statistics, psycopg
from datetime import datetime
pattern = 'workflow-actionperf-{run_id}-%'
def stamp(value):
    return datetime.fromisoformat(value)
def percentile(values, ratio):
    values = sorted(values)
    return values[int((len(values) - 1) * ratio)] if values else None
with psycopg.connect(os.environ['GPU_FAULT_STORE_URL']) as conn:
    cur = conn.cursor()
    cur.execute(
        "SELECT key, payload FROM gpu_fault_objects "
        "WHERE kind='workflow' AND key LIKE %s",
        (pattern,),
    )
    workflows = cur.fetchall()
    cur.execute(
        "SELECT payload->>'workflow_request_id', "
        "min((payload->>'created_at')::timestamptz) "
        "FROM gpu_fault_objects WHERE kind='remote_command' "
        "AND payload->>'workflow_request_id' LIKE %s GROUP BY 1",
        (pattern,),
    )
    first_commands = dict(cur.fetchall())
durations = []
first_actions = []
cluster_finished = {{}}
for key, payload in workflows:
    created = stamp(payload['created_at'])
    updated = stamp(payload['updated_at'])
    durations.append((updated - created).total_seconds())
    first = first_commands.get(key)
    if first is not None:
        first_actions.append((first - created).total_seconds())
    match = re.search(r'-c(\\d{{3}})-w\\d{{3}}$', key)
    if match:
        cluster = match.group(1)
        cluster_finished[cluster] = max(
            cluster_finished.get(cluster, updated), updated
        )
finish_values = list(cluster_finished.values())
out = {{
    'workflow_duration_p50_seconds': percentile(durations, 0.50),
    'workflow_duration_p95_seconds': percentile(durations, 0.95),
    'workflow_duration_p99_seconds': percentile(durations, 0.99),
    'workflow_duration_max_seconds': max(durations) if durations else None,
    'first_action_p50_seconds': percentile(first_actions, 0.50),
    'first_action_p95_seconds': percentile(first_actions, 0.95),
    'first_action_p99_seconds': percentile(first_actions, 0.99),
    'first_action_max_seconds': max(first_actions)
        if first_actions else None,
    'cluster_completion_skew_seconds': (
        max(finish_values) - min(finish_values)
    ).total_seconds() if finish_values else None,
}}
print(json.dumps(out))
"""
    output = control(
        "exec",
        pod,
        "--",
        "python3",
        "-c",
        script,
        timeout=120,
    )
    return json.loads(output.splitlines()[-1])


def collect_executor_logs(target: Path) -> list[dict]:
    target.mkdir(parents=True, exist_ok=True)
    raw = dataplane(
        "get",
        "pod",
        "-l",
        f"batch.kubernetes.io/job-name={JOB_NAME}",
        "-o",
        "jsonpath={range .items[*]}{.metadata.name} "
        "{.metadata.annotations.batch\\.kubernetes\\.io/"
        "job-completion-index}{'\\n'}{end}",
    )
    documents = []
    for line in raw.splitlines():
        pod, _, index = line.partition(" ")
        body = dataplane("logs", pod, check=False, timeout=120)
        (target / f"{index.strip()}-{pod}.log").write_text(body)
        try:
            documents.append(json.loads(body))
        except json.JSONDecodeError:
            pass
    return documents


def aggregate_executor_documents(documents: list[dict]) -> dict:
    walls = [item.get("wall_seconds", 0.0) for item in documents]
    return {
        "executor_pods": len(documents),
        "executor_wall_p50": statistics.median(walls) if walls else None,
        "executor_wall_max": max(walls) if walls else None,
        "duplicate_claims": sum(item.get("duplicate_claims", 0) for item in documents),
        "claim_errors": sum(item.get("claim_errors", 0) for item in documents),
        "result_errors": sum(item.get("result_errors", 0) for item in documents),
        "renewals": sum(item.get("renewals", 0) for item in documents),
        "renewal_errors": sum(item.get("renewal_errors", 0) for item in documents),
        "injected_renewal_failures": sum(
            item.get("injected_renewal_failures", 0) for item in documents
        ),
        "long_commands": sum(item.get("long_commands", 0) for item in documents),
        "max_concurrent_commands": max(
            (item.get("max_concurrent_commands", 0) for item in documents),
            default=0,
        ),
    }


def execute_capacity_run(
    args,
    *,
    run_id: str,
    artifacts: Path,
    expected_workflows: int,
    expected_commands_per_cluster: int,
    started: float,
) -> int:
    register(args.clusters, artifacts)
    upsert_configmap(
        SCRIPT_CONFIGMAP,
        text={
            "benchmark_regional_action_executor.py": (
                PERF_DIR / "benchmark_regional_action_executor.py"
            ).read_text()
        },
    )
    dataplane(
        "delete",
        "job",
        JOB_NAME,
        "--ignore-not-found",
        "--wait=true",
        check=False,
    )
    identity = executor_identity()
    manifest = executor_job(
        clusters=args.clusters,
        expected_commands=expected_commands_per_cluster,
        workers=args.executor_workers,
        delay_scale=args.delay_scale,
        lease_seconds=args.lease_seconds,
        inject_renew_failure_once=args.inject_renew_failure_once,
        **identity,
    )
    (artifacts / "job.json").write_text(json.dumps(manifest, indent=2) + "\n")
    dataplane("apply", "-f", "-", stdin=json.dumps(manifest).encode())
    deadline = time.time() + 300
    while time.time() < deadline:
        running = dataplane(
            "get",
            "pod",
            "-l",
            f"batch.kubernetes.io/job-name={JOB_NAME}",
            "--field-selector=status.phase=Running",
            "-o",
            "name",
        ).splitlines()
        if len(running) == args.clusters:
            break
        time.sleep(5)
    else:
        raise RuntimeError("action executor pods did not start")

    pods = [
        item
        for item in control(
            "get",
            "pod",
            "-l",
            "app in (gpu-fault-api-ha,gpu-fault-control-worker)",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        ).split()
        if item
    ]
    before_metrics = scrape_metrics(pods)
    before_cgroup = scrape_cgroup(pods)
    before_postgres = postgres_counters()
    seeded_at = time.time()
    seeded = seed(
        run_id=run_id,
        clusters=args.clusters,
        workflows_per_cluster=args.workflows_per_cluster,
        nodes_per_workflow=args.nodes_per_workflow,
    )
    log(f"seeded {seeded['workflows_created']} workflows")

    history = []
    end = time.time() + args.timeout_seconds
    while time.time() < end:
        snapshot = database_snapshot(run_id)
        snapshot["elapsed_seconds"] = time.time() - seeded_at
        history.append(snapshot)
        terminal = sum(
            count
            for status, count in snapshot["workflows"].items()
            if status in TERMINAL
        )
        if terminal == expected_workflows:
            break
        time.sleep(2)
    job_succeeded = ""
    job_failed = ""
    job_completions = ""
    job_deadline = time.time() + 120
    while time.time() < job_deadline:
        state = dataplane(
            "get",
            "job",
            JOB_NAME,
            "-o",
            "jsonpath={.status.succeeded}|{.status.failed}|{.spec.completions}",
            check=False,
        )
        job_succeeded, job_failed, job_completions = (state.split("|") + ["", "", ""])[
            :3
        ]
        if job_succeeded and job_succeeded == job_completions:
            break
        if job_failed and job_failed != "0":
            break
        time.sleep(2)
    after_metrics = scrape_metrics(pods)
    after_cgroup = scrape_cgroup(pods)
    after_postgres = postgres_counters()
    documents = collect_executor_logs(artifacts / "executors")
    (artifacts / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    final = history[-1] if history else {}
    details = database_details(run_id)
    executor_metrics = aggregate_executor_documents(documents)
    workflows_succeeded = (
        final.get("workflows", {}).get("SUCCEEDED", 0) == expected_workflows
    )
    executor_job_succeeded = (
        bool(job_succeeded)
        and job_succeeded == job_completions
        and not (job_failed and job_failed != "0")
    )
    summary = {
        **seeded,
        **details,
        **executor_metrics,
        "job_status": (
            "Complete"
            if workflows_succeeded and executor_job_succeeded
            else "Incomplete"
        ),
        "executor_job_succeeded": int(job_succeeded or 0),
        "executor_job_failed": int(job_failed or 0),
        "executor_job_completions": int(job_completions or 0),
        "workflow_elapsed_seconds": (
            history[-1]["elapsed_seconds"] if history else None
        ),
        "workflow_statuses": final.get("workflows", {}),
        "command_statuses": final.get("commands", {}),
        "postgres_deltas": {
            key: (after_postgres.get(key) or 0) - (before_postgres.get(key) or 0)
            for key in ("deadlocks", "xact_commit", "xact_rollback")
        },
        "wall_seconds": time.time() - started,
    }
    for name, value in (
        ("metrics-before.json", before_metrics),
        ("metrics-after.json", after_metrics),
        ("cgroup-before.json", before_cgroup),
        ("cgroup-after.json", after_cgroup),
        ("postgres-before.json", before_postgres),
        ("postgres-after.json", after_postgres),
        ("summary.json", summary),
    ):
        (artifacts / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["job_status"] == "Complete":
        write_status(artifacts, status="ok")
        return 0
    write_status(
        artifacts,
        status="aborted",
        reason=f"job status: {summary['job_status']}",
    )
    move_to_aborted(args.artifact_root, artifacts)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--workflows-per-cluster", type=int, default=2)
    parser.add_argument("--nodes-per-workflow", type=int, default=4)
    parser.add_argument("--executor-workers", type=int, default=8)
    parser.add_argument("--delay-scale", type=float, default=1.0)
    parser.add_argument("--lease-seconds", type=int, default=120)
    parser.add_argument("--inject-renew-failure-once", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--suite-id")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("act%Y%m%dT%H%M%S")
    label = args.label or f"actions-{args.clusters}c"
    identity = release_identity()
    artifacts = artifact_dir(
        args.artifact_root,
        label,
        identity["release_id"] or "unknown-release",
    )
    expected_workflows = args.clusters * args.workflows_per_cluster
    expected_commands_per_cluster = args.workflows_per_cluster * 10
    log(f"artifacts: {artifacts}")
    (artifacts / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "clusters": args.clusters,
                "workflows_per_cluster": (args.workflows_per_cluster),
                "nodes_per_workflow": args.nodes_per_workflow,
                "executor_workers": args.executor_workers,
                "delay_scale": args.delay_scale,
                "lease_seconds": args.lease_seconds,
                "inject_renew_failure_once": args.inject_renew_failure_once,
                "suite_id": args.suite_id or run_id,
                **identity,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    write_status(artifacts, status="running")
    started = time.time()
    try:
        return execute_capacity_run(
            args,
            run_id=run_id,
            artifacts=artifacts,
            expected_workflows=expected_workflows,
            expected_commands_per_cluster=expected_commands_per_cluster,
            started=started,
        )
    except BaseException as exc:
        write_status(
            artifacts,
            status="aborted",
            reason=f"{type(exc).__name__}: {exc}",
        )
        move_to_aborted(args.artifact_root, artifacts)
        raise
    finally:
        dataplane(
            "delete",
            "job",
            JOB_NAME,
            "--ignore-not-found",
            check=False,
        )
        dataplane(
            "delete",
            "configmap",
            SCRIPT_CONFIGMAP,
            "--ignore-not-found",
            check=False,
        )
        teardown(purge=True, deregister_clusters=True)


if __name__ == "__main__":
    raise SystemExit(main())
