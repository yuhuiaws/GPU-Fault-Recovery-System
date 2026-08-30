from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__:
    from .regional_action_capacity_suite import (
        executor_identity,
        release_agent_identity,
    )
    from .regional_capacity_results import (
        artifact_dir,
        move_to_aborted,
        write_status,
    )
    from .regional_capacity_registry import (
        CONNECTION_SECRET,
        NAMESPACE,
        TOKEN_SECRET,
        control,
        dataplane,
        register,
        validate_registry_target,
    )
    from .regional_capacity_suite import (
        DEFAULT_ARTIFACT_ROOT,
        release_identity,
        teardown,
        upsert_configmap,
    )
else:
    from regional_action_capacity_suite import (
        executor_identity,
        release_agent_identity,
    )
    from regional_capacity_results import artifact_dir, move_to_aborted, write_status
    from regional_capacity_registry import (
        CONNECTION_SECRET,
        NAMESPACE,
        TOKEN_SECRET,
        control,
        dataplane,
        register,
        validate_registry_target,
    )
    from regional_capacity_suite import (
        DEFAULT_ARTIFACT_ROOT,
        release_identity,
        teardown,
        upsert_configmap,
    )


SCRIPT_CONFIGMAP = "gpu-fault-correlated-action-script"
JOB_NAME = "gpu-fault-correlated-action"
REPO_ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = REPO_ROOT / "scripts/perf"
TERMINAL_WORKFLOWS = {"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}
TERMINAL_COMMANDS = {"SUCCEEDED", "FAILED", "CANCELLED"}


def scenario_job(
    *,
    clusters: int,
    run_id: str,
    executor_protocol_version: int,
    executor_artifact_sha256: str,
    executor_compatibility_digest: str,
    runtime_profile_version: str,
    connection_secret: str = CONNECTION_SECRET,
) -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": JOB_NAME, "namespace": NAMESPACE},
        "spec": {
            "completions": clusters,
            "parallelism": clusters,
            "completionMode": "Indexed",
            "backoffLimit": 0,
            "activeDeadlineSeconds": 1200,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": "gpu-fault-completion-watcher",
                    "tolerations": [
                        {
                            "key": "node.kubernetes.io/unschedulable",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        },
                        {
                            "key": "gpu-fault.io/quarantined",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        },
                    ],
                    "containers": [
                        {
                            "name": "scenario",
                            "image": "public.ecr.aws/docker/library/python:3.12-slim",
                            "command": [
                                "python",
                                "/scripts/benchmark_correlated_action_scenario.py",
                            ],
                            "env": [
                                {
                                    "name": "GPU_FAULT_CONTROL_PLANE_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": connection_secret,
                                            "key": "control-plane-url",
                                        }
                                    },
                                },
                                {"name": "SSL_CERT_FILE", "value": "/tls/ca.crt"},
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
                                {"name": "CORRELATED_ACTION_RUN_ID", "value": run_id},
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
                                {
                                    "name": "RUNTIME_PROFILE_VERSION",
                                    "value": runtime_profile_version,
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "250m", "memory": "128Mi"},
                                "limits": {"cpu": "2", "memory": "512Mi"},
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
                                "secretName": connection_secret,
                                "items": [{"key": "ca.crt", "path": "ca.crt"}],
                            },
                        },
                    ],
                }
            },
        },
    }


def seed_agents(
    run_id: str,
    clusters: int,
    *,
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
        "ACTION_SEED_MODE=correlated-agents",
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
        timeout=120,
    )
    return json.loads(output.splitlines()[-1])


def validate_preemption_enabled() -> None:
    pods = control(
        "get",
        "pod",
        "-l",
        "app in (gpu-fault-api-ha,gpu-fault-control-worker)",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    ).split()
    if not pods:
        raise RuntimeError("no control-plane Pods available for preemption check")
    for pod in pods:
        value = control(
            "exec",
            pod,
            "--",
            "python3",
            "-c",
            (
                "import os; print("
                "os.environ.get('GPU_FAULT_ENABLE_WORKFLOW_PREEMPTION','false')"
                ")"
            ),
        ).strip()
        if value.lower() not in {"1", "true", "yes", "on"}:
            raise RuntimeError(f"workflow preemption is not enabled on {pod}")


def collect_logs(target: Path) -> list[dict]:
    target.mkdir(parents=True, exist_ok=True)
    raw = dataplane(
        "get",
        "pod",
        "-l",
        f"batch.kubernetes.io/job-name={JOB_NAME}",
        "-o",
        (
            "jsonpath={range .items[*]}{.metadata.name} "
            "{.metadata.annotations.batch\\.kubernetes\\.io/"
            "job-completion-index}{'\\n'}{end}"
        ),
    )
    documents = []
    for line in raw.splitlines():
        pod, _, index = line.partition(" ")
        body = dataplane("logs", pod, check=False, timeout=120)
        (target / f"{index.strip()}-{pod}.log").write_text(body)
        try:
            documents.append(json.loads(body))
        except json.JSONDecodeError:
            continue
    return documents


def database_audit(run_id: str) -> dict:
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
prefix='corr-live-{run_id}-%'
with psycopg.connect(os.environ['GPU_FAULT_STORE_URL']) as conn:
    cur=conn.cursor()
    cur.execute(
        "SELECT payload FROM gpu_fault_objects "
        "WHERE kind='incident' AND payload->>'event_id' LIKE %s",
        (prefix,),
    )
    incidents=[row[0] for row in cur.fetchall()]
    incident_ids=[item['incident_id'] for item in incidents]
    cur.execute(
        "SELECT payload FROM gpu_fault_objects WHERE kind='workflow'"
    )
    all_workflows=[row[0] for row in cur.fetchall()]
    selected={{
        item['request_id']:item
        for item in all_workflows
        if item.get('incident_id') in incident_ids
    }}
    changed=True
    while changed:
        changed=False
        expected_reboots={{
            f"workflow-reboot-after-{{request_id}}"
            for request_id in selected
        }}
        for item in all_workflows:
            if item['request_id'] in selected:
                continue
            if (
                item.get('predecessor_workflow_id') in selected
                or item['request_id'] in expected_reboots
            ):
                selected[item['request_id']]=item
                changed=True
    workflows=list(selected.values())
    workflow_ids=[item['request_id'] for item in workflows]
    cur.execute(
        "SELECT payload FROM gpu_fault_objects "
        "WHERE kind='remote_command' "
        "AND payload->>'workflow_request_id'=ANY(%s)",
        (workflow_ids,),
    )
    commands=[row[0] for row in cur.fetchall()]

by_id={{item['request_id']:item for item in workflows}}
def operations(item):
    return {{step.get('operation') for step in item.get('official_steps',[])}}
weak=[
    item for item in workflows
    if item.get('status')=='SUPERSEDED'
    and item.get('preempted_by_workflow_id')
    and 'RESET_GPU' in operations(item)
]
strong=[
    item for item in workflows
    if item.get('preempt_predecessor') is True
    and 'RESET_ALL_GPUS_NVSWITCHES' in operations(item)
]
reset_gpu=[
    item for item in workflows
    if item.get('status')=='FAILED'
    and any(
        execution.get('operation')=='RESET_GPU'
        and execution.get('status')=='FAILED'
        for execution in item.get('step_executions',[])
    )
]
reboot=[
    item for item in workflows
    if 'RESTART_NODE' in operations(item)
]
weak_ok=sum(
    item.get('status')=='SUPERSEDED'
    and bool(item.get('preempted_by_workflow_id'))
    for item in weak
)
strong_ok=sum(
    item.get('status')=='FAILED'
    and item.get('predecessor_workflow_id') in by_id
    and {{
        item['official_steps'][index].get('operation')
        for index in item.get('inherited_step_indexes',[])
    }} == {{'MARK_UNSCHEDULABLE','STOP_WORKLOADS'}}
    and any(
        execution.get('operation')=='RESET_ALL_GPUS_NVSWITCHES'
        and execution.get('status')=='FAILED'
        for execution in item.get('step_executions',[])
    )
    for item in strong
)
reboot_ok=sum(
    item.get('status')=='SUCCEEDED'
    and any(
        step.get('operation')=='RESTART_NODE'
        for step in item.get('official_steps',[])
    )
    for item in reboot
)
terminal_workflows=sum(
    item.get('status') in {sorted(TERMINAL_WORKFLOWS)!r}
    for item in workflows
)
terminal_commands=sum(
    item.get('status') in {sorted(TERMINAL_COMMANDS)!r}
    for item in commands
)
duplicates=len(commands)-len({{
    item.get('idempotency_key') for item in commands
}})
fencing=sum(
    int(item.get('fencing_token',0))
    != int((by_id.get(item.get('workflow_request_id')) or {{}}).get('fencing_token',-1))
    for item in commands
)
out={{
    'incident_count':len(incidents),
    'workflow_count':len(workflows),
    'command_count':len(commands),
    'weak_workflow_count':len(weak),
    'strong_workflow_count':len(strong),
    'reset_gpu_workflow_count':len(reset_gpu),
    'reboot_workflow_count':len(reboot),
    'weak_superseded_count':weak_ok,
    'strong_preempt_failed_fabric_reset_count':strong_ok,
    'reset_gpu_failed_count':len(reset_gpu),
    'reboot_succeeded_count':reboot_ok,
    'terminal_workflow_count':terminal_workflows,
    'terminal_command_count':terminal_commands,
    'duplicate_idempotency_keys':duplicates,
    'fencing_mismatches':fencing,
    'budget_wait_total':sum(
        int(item.get('remediation_budget_wait_count',0))
        for item in workflows
    ),
    'permanent_budget_waiters':sum(
        item.get('status') in ('PENDING','RUNNING','WAITING')
        and int(item.get('remediation_budget_wait_count',0)) > 0
        for item in workflows
    ),
    'workflow_statuses':{{
        status:sum(item.get('status')==status for item in workflows)
        for status in sorted({{item.get('status') for item in workflows}})
    }},
    'command_statuses':{{
        status:sum(item.get('status')==status for item in commands)
        for status in sorted({{item.get('status') for item in commands}})
    }},
}}
print(json.dumps(out,sort_keys=True))
"""
    output = control(
        "exec",
        "-i",
        pod,
        "--",
        "python3",
        "-",
        stdin=script.encode(),
        timeout=120,
    )
    return json.loads(output.splitlines()[-1])


def verdict(summary: dict, clusters: int) -> tuple[str, list[str]]:
    errors = []
    audit = summary["audit"]
    if len(summary["pod_results"]) != clusters:
        errors.append("scenario pod result count mismatch")
    for item in summary["pod_results"]:
        for key in (
            "strong_sent",
            "primary_reset_failed",
            "reset_attempt_started",
            "reset_gpu_failed",
            "idle_terminal",
        ):
            if not item.get(key):
                errors.append(f"scenario pod missing {key}")
        if int(item.get("duplicate_claims", 0)):
            errors.append("scenario duplicate claims are nonzero")
        if int(item.get("result_errors", 0)):
            errors.append("scenario result errors are nonzero")
    for key in (
        "weak_superseded_count",
        "strong_preempt_failed_fabric_reset_count",
        "reset_gpu_failed_count",
    ):
        if int(audit.get(key, 0)) != clusters:
            errors.append(f"{key} does not equal cluster count")
    if int(audit.get("reboot_succeeded_count", 0)) != clusters * 2:
        errors.append("reboot_succeeded_count does not equal twice cluster count")
    if audit["terminal_workflow_count"] != audit["workflow_count"]:
        errors.append("nonterminal workflows remain")
    if audit["terminal_command_count"] != audit["command_count"]:
        errors.append("nonterminal commands remain")
    for key in (
        "duplicate_idempotency_keys",
        "fencing_mismatches",
        "permanent_budget_waiters",
    ):
        if int(audit.get(key, 0)):
            errors.append(f"{key} is nonzero")
    return ("PASS" if not errors else "FAIL", errors)


def purge_scenario_rows(run_id: str) -> None:
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
prefix='corr-live-{run_id}-%'
with psycopg.connect(os.environ['GPU_FAULT_STORE_URL'],autocommit=True) as conn:
    cur=conn.cursor()
    cur.execute(
        "SELECT key FROM gpu_fault_objects "
        "WHERE kind='incident' AND payload->>'event_id' LIKE %s",
        (prefix,),
    )
    incidents=[row[0] for row in cur.fetchall()]
    cur.execute(
        "SELECT key,payload FROM gpu_fault_objects WHERE kind='workflow'"
    )
    all_workflows=cur.fetchall()
    selected={{
        key:payload
        for key,payload in all_workflows
        if payload.get('incident_id') in incidents
    }}
    changed=True
    while changed:
        changed=False
        expected_reboots={{
            f"workflow-reboot-after-{{request_id}}"
            for request_id in selected
        }}
        for key,payload in all_workflows:
            if key in selected:
                continue
            if (
                payload.get('predecessor_workflow_id') in selected
                or key in expected_reboots
            ):
                selected[key]=payload
                changed=True
    workflows=list(selected)
    incidents=sorted({{
        *incidents,
        *(
            payload.get('incident_id')
            for payload in selected.values()
            if payload.get('incident_id')
        ),
    }})
    cur.execute(
        "DELETE FROM gpu_fault_objects "
        "WHERE kind='remote_command' "
        "AND payload->>'workflow_request_id'=ANY(%s)",
        (workflows,),
    )
    cur.execute(
        "DELETE FROM gpu_fault_objects "
        "WHERE kind='workflow' AND key=ANY(%s)",
        (workflows,),
    )
    cur.execute(
        "DELETE FROM gpu_fault_objects "
        "WHERE kind='incident' AND key=ANY(%s)",
        (incidents,),
    )
"""
    control(
        "exec",
        "-i",
        pod,
        "--",
        "python3",
        "-",
        stdin=script.encode(),
        timeout=120,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--label")
    parser.add_argument("--suite-id")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allow-live-registry", action="store_true")
    parser.add_argument("--confirm-live-registry")
    parser.add_argument("--synthetic-ttl-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.synthetic_ttl_seconds < 300:
        parser.error("--synthetic-ttl-seconds must be at least 300")
    try:
        registry_scope = validate_registry_target(
            allow_live_registry=args.allow_live_registry,
            confirmation=args.confirm_live_registry,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    run_id = args.suite_id or datetime.now(timezone.utc).strftime(
        "corract%Y%m%dT%H%M%S"
    )
    label = args.label or f"correlated-action-{args.clusters}c"
    identity = release_identity()
    artifacts = artifact_dir(
        args.artifact_root,
        label,
        identity["release_id"] or "unknown-release",
    )
    write_status(artifacts, status="running")
    failure: BaseException | None = None
    result = 1
    try:
        validate_preemption_enabled()
        register(
            args.clusters,
            artifacts,
            run_id=run_id,
            expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=args.synthetic_ttl_seconds),
            allow_live_registry=args.allow_live_registry,
            live_registry_confirmation=args.confirm_live_registry,
        )
        seeded = seed_agents(
            run_id,
            args.clusters,
            agent_identity=(
                release_agent_identity() if registry_scope == "isolated" else None
            ),
        )
        upsert_configmap(
            SCRIPT_CONFIGMAP,
            text={
                "benchmark_correlated_action_scenario.py": (
                    PERF_DIR / "benchmark_correlated_action_scenario.py"
                ).read_text()
            },
        )
        identity = executor_identity(
            require_dataplane_deployment=registry_scope == "live",
        )
        manifest = scenario_job(
            clusters=args.clusters,
            run_id=run_id,
            runtime_profile_version=str(seeded["runtime_profile_version"]),
            **identity,
        )
        (artifacts / "job.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        dataplane("apply", "-f", "-", stdin=json.dumps(manifest).encode())
        wait = dataplane(
            "wait",
            "--for=condition=complete",
            f"job/{JOB_NAME}",
            f"--timeout={args.timeout_seconds}s",
            check=False,
            timeout=args.timeout_seconds + 60,
        )
        pod_results = collect_logs(artifacts / "pods")
        audit = database_audit(run_id)
        summary = {
            "run_id": run_id,
            "clusters": args.clusters,
            "registry_scope": registry_scope,
            "seeded": seeded,
            "job_wait_output": wait,
            "pod_results": pod_results,
            "audit": audit,
        }
        status, errors = verdict(summary, args.clusters)
        summary["status"] = status
        summary["errors"] = errors
        (artifacts / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        result = int(status != "PASS")
    except BaseException as exc:
        failure = exc
    finally:
        dataplane("delete", "job", JOB_NAME, "--ignore-not-found", check=False)
        dataplane(
            "delete",
            "configmap",
            SCRIPT_CONFIGMAP,
            "--ignore-not-found",
            check=False,
        )
        purge_scenario_rows(run_id)
        try:
            teardown(
                purge=True,
                deregister_clusters=True,
                allow_live_registry=args.allow_live_registry,
                live_registry_confirmation=args.confirm_live_registry,
                artifacts=artifacts,
                run_id=run_id,
            )
        except Exception as cleanup_error:
            if failure is None:
                failure = cleanup_error
            else:
                failure.add_note(f"scenario teardown also failed: {cleanup_error}")
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
        reason=None if result == 0 else "scenario verdict failed",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
