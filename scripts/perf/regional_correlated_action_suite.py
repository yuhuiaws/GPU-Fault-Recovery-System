from __future__ import annotations

import argparse
import json
import time
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
        STORE_DSN_SNIPPET,
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
        STORE_DSN_SNIPPET,
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
    scenario_max_seconds: int,
    active_deadline_seconds: int,
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
            "activeDeadlineSeconds": active_deadline_seconds,
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
                                    "name": "SCENARIO_MAX_SECONDS",
                                    "value": str(scenario_max_seconds),
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


def _database_audit_script_prefix(run_id: str) -> str:
    return f"""
import json, os, psycopg
{STORE_DSN_SNIPPET}
from datetime import datetime, timezone
prefix='corr-live-{run_id}-%'
with psycopg.connect(store_dsn()) as conn:
    cur=conn.cursor()
    cur.execute(
        "SELECT key,payload FROM gpu_fault_objects "
        "WHERE kind='incident' AND payload->>'event_id' LIKE %s",
        (prefix,),
    )
    incident_rows=cur.fetchall()
    incidents=[row[1] for row in incident_rows]
    incident_ids=[row[0] for row in incident_rows]
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
    related_incident_ids=sorted({{
        item.get('incident_id')
        for item in [*workflows,*commands]
        if item.get('incident_id')
    }})
    cur.execute(
        "SELECT key,payload FROM gpu_fault_objects "
        "WHERE kind='incident' AND key=ANY(%s)",
        (related_incident_ids,),
    )
    incident_by_id={{row[0]:row[1] for row in cur.fetchall()}}
"""


def _database_audit_script_shapes() -> str:
    """Classify the run's workflows: preempted, preempting, reset-lane, reboot."""
    return """

by_id={item['request_id']:item for item in workflows}
def operations(item):
    return {step.get('operation') for step in item.get('official_steps',[])}
def step_operation(item,index):
    steps=item.get('official_steps',[])
    return steps[index].get('operation') if 0<=index<len(steps) else None
def in_record_preemption(item):
    # Clean boundary: the stronger fabric reset preempted the weak GPU reset
    # inside one record -- the weak plan's pending steps (its RESET_GPU among
    # them) are superseded, the fabric reset lives in a successor branch, and
    # the record carries a PREEMPTION event.
    superseded=set(item.get('superseded_step_indexes') or [])
    return bool(
        superseded
        and any(step_operation(item,i)=='RESET_GPU' for i in superseded)
        and any(
            step.get('operation')=='RESET_ALL_GPUS_NVSWITCHES'
            and index not in superseded
            for index,step in enumerate(item.get('official_steps',[]))
        )
        and any(
            event.get('kind')=='PREEMPTION' for event in item.get('events',[])
        )
    )
in_record=[item for item in workflows if in_record_preemption(item)]
# Cross-record shape: the preemption landed on an in-flight physical step, so
# the fabric reset is its own successor record marked preempt_predecessor.
weak_cross=[
    item for item in workflows
    if item.get('status')=='SUPERSEDED'
    and item.get('preempted_by_workflow_id')
    and 'RESET_GPU' in operations(item)
]
strong_cross=[
    item for item in workflows
    if item.get('preempt_predecessor') is True
    and 'RESET_ALL_GPUS_NVSWITCHES' in operations(item)
]
weak=[*weak_cross,*in_record]
strong=[*strong_cross,*in_record]
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
weak_by_id={item['request_id']:item for item in weak_cross}
required_containment={'MARK_UNSCHEDULABLE','STOP_WORKLOADS'}
allowed_inherited=required_containment|{'QUIESCE_GPU_SERVICES'}
preemption_pairs=[]
for item in strong_cross:
    predecessor=weak_by_id.get(item.get('predecessor_workflow_id'))
    inherited={
        item['official_steps'][index].get('operation')
        for index in item.get('inherited_step_indexes',[])
    }
    if (
        predecessor is not None
        and predecessor.get('preempted_by_workflow_id')==item['request_id']
        and predecessor.get('incident_id')==item.get('incident_id')
        and required_containment<=inherited<=allowed_inherited
    ):
        preemption_pairs.append((predecessor,item))
for item in in_record:
    # In-record pair: the completed containment stays completed and is not
    # retired; the fabric reset sits in the node's successor branch.
    superseded=set(item.get('superseded_step_indexes') or [])
    completed=set(item.get('completed_step_indexes') or [])
    kept={step_operation(item,i) for i in completed-superseded}
    successor_branch=any(
        str(step.get('branch_id') or '').endswith(':successor:1')
        and step.get('operation')=='RESET_ALL_GPUS_NVSWITCHES'
        for step in item.get('official_steps',[])
    )
    if required_containment<=kept and successor_branch:
        preemption_pairs.append((item,item))
weak_ok=sum(
    item.get('status')=='SUPERSEDED'
    and bool(item.get('preempted_by_workflow_id'))
    for item in weak_cross
)+len(in_record)
strong_ok=sum(
    item.get('status')=='FAILED'
    and any(
        execution.get('operation')=='RESET_ALL_GPUS_NVSWITCHES'
        and execution.get('status')=='FAILED'
        for execution in item.get('step_executions',[])
    )
    for item in strong
)
"""


def _database_audit_script_metrics() -> str:
    """Reboot pairing, terminal drain, idempotency and fencing counters."""
    return f"""
failed_sources={{
    item['request_id']:item
    for item in [*strong,*reset_gpu]
}}
deterministic_reboot_sources={{
    f"workflow-reboot-after-{{request_id}}":request_id
    for request_id in failed_sources
}}
reboot_by_predecessor={{}}
unexpected_reboot_predecessors=0
for item in reboot:
    predecessor=(
        item.get('predecessor_workflow_id')
        or deterministic_reboot_sources.get(item['request_id'])
    )
    if predecessor not in failed_sources:
        unexpected_reboot_predecessors+=1
        continue
    reboot_by_predecessor.setdefault(predecessor,[]).append(item)
reboot_successor_pairs=sum(
    len(items)==1
    and items[0].get('status')=='SUCCEEDED'
    and 'RESTART_NODE' in operations(items[0])
    for items in reboot_by_predecessor.values()
)
duplicate_reboot_successors=sum(
    max(0,len(items)-1)
    for items in reboot_by_predecessor.values()
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
duplicate_workflow_steps=len(commands)-len({{
    (item.get('workflow_request_id'),item.get('step_index'))
    for item in commands
}})
fencing=sum(
    (
        int(item.get('fencing_token',0))
        != int(
            (by_id.get(item.get('workflow_request_id')) or {{}})
            .get('fencing_token',-1)
        )
        or int(item.get('fencing_token',0))
        != int(
            (incident_by_id.get(item.get('incident_id')) or {{}})
            .get('fencing_token',-1)
        )
    )
    for item in commands
)
observed_at=datetime.now(timezone.utc)
nonterminal=[
    item for item in workflows
    if item.get('status') in ('PENDING','RUNNING','WAITING')
]
lease_states={{'active':0,'expired':0,'missing':0}}
for item in nonterminal:
    raw=item.get('execution_lease_expires_at')
    if not raw:
        lease_states['missing']+=1
        continue
    expires=datetime.fromisoformat(str(raw).replace('Z','+00:00'))
    lease_states['active' if expires>observed_at else 'expired']+=1
commands_by_workflow={{}}
for item in commands:
    commands_by_workflow.setdefault(item.get('workflow_request_id'),[]).append(item)
terminal_command_nonterminal_workflows=sum(
    bool(commands_by_workflow.get(item['request_id']))
    and all(
        command.get('status') in {sorted(TERMINAL_COMMANDS)!r}
        for command in commands_by_workflow[item['request_id']]
    )
    for item in nonterminal
)
out={{
    'incident_count':len(incidents),
    'workflow_count':len(workflows),
    'command_count':len(commands),
    'weak_workflow_count':len(weak),
    'strong_workflow_count':len(strong),
    'in_record_preemption_count':len(in_record),
    'cross_record_preemption_count':len(strong_cross),
    'reset_gpu_workflow_count':len(reset_gpu),
    'reboot_workflow_count':len(reboot),
    'preemption_pair_count':len(preemption_pairs),
    'weak_superseded_count':weak_ok,
    'strong_preempt_failed_fabric_reset_count':strong_ok,
    'reset_gpu_failed_count':len(reset_gpu),
    'reboot_succeeded_count':reboot_ok,
    'reboot_successor_pair_count':reboot_successor_pairs,
    'unique_reboot_predecessor_count':len(reboot_by_predecessor),
    'duplicate_reboot_successors':duplicate_reboot_successors,
    'unexpected_reboot_predecessors':unexpected_reboot_predecessors,
    'terminal_workflow_count':terminal_workflows,
    'terminal_command_count':terminal_commands,
    'duplicate_idempotency_keys':duplicates,
    'duplicate_workflow_steps':duplicate_workflow_steps,
    'fencing_mismatches':fencing,
    'nonterminal_lease_states':lease_states,
    'terminal_command_nonterminal_workflows':(
        terminal_command_nonterminal_workflows
    ),
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


def _database_audit_script_suffix() -> str:
    return _database_audit_script_shapes() + _database_audit_script_metrics()


def database_audit(run_id: str) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    script = _database_audit_script_prefix(run_id) + _database_audit_script_suffix()
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


def audit_is_terminal(audit: dict) -> bool:
    return bool(audit.get("workflow_count")) and (
        int(audit.get("terminal_workflow_count", 0))
        == int(audit.get("workflow_count", 0))
        and int(audit.get("terminal_command_count", 0))
        == int(audit.get("command_count", 0))
        and int(audit.get("permanent_budget_waiters", 0)) == 0
    )


def wait_for_terminal_drain(
    run_id: str,
    *,
    timeout_seconds: int,
) -> tuple[dict, list[dict]]:
    deadline = time.monotonic() + timeout_seconds
    history = []
    while True:
        audit = database_audit(run_id)
        history.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "workflow_count": audit.get("workflow_count"),
                "terminal_workflow_count": audit.get("terminal_workflow_count"),
                "command_count": audit.get("command_count"),
                "terminal_command_count": audit.get("terminal_command_count"),
                "permanent_budget_waiters": audit.get("permanent_budget_waiters"),
                "workflow_statuses": audit.get("workflow_statuses"),
                "command_statuses": audit.get("command_statuses"),
            }
        )
        if audit_is_terminal(audit) or time.monotonic() >= deadline:
            return audit, history
        time.sleep(5)


def verdict(summary: dict, clusters: int) -> tuple[str, list[str]]:
    errors = []
    audit = summary["audit"]
    if summary.get("job_status") != "Complete":
        errors.append("scenario job did not complete")
    if int(summary.get("job_succeeded", 0)) != clusters:
        errors.append("scenario job succeeded count mismatch")
    if int(summary.get("job_failed", 0)):
        errors.append("scenario job failed count is nonzero")
    if len(summary["pod_results"]) != clusters:
        errors.append("scenario pod result count mismatch")
    for item in summary["pod_results"]:
        for key in (
            "aggregate_shared_incident",
            "aggregate_shared_workflow",
            "strong_sent",
            "primary_reset_failed",
            "primary_reboot_succeeded",
            "reset_attempt_started",
            "reset_gpu_failed",
            "reset_reboot_succeeded",
            "idle_terminal",
        ):
            if not item.get(key):
                errors.append(f"scenario pod missing {key}")
        if int(item.get("duplicate_claims", 0)):
            errors.append("scenario duplicate claims are nonzero")
        if int(item.get("claim_errors", 0)):
            errors.append("scenario claim errors are nonzero")
        if int(item.get("result_errors", 0)):
            errors.append("scenario result errors are nonzero")
        if int(item.get("ownership_errors", 0)):
            errors.append("scenario ownership errors are nonzero")
    # A clean-boundary preemption stays inside the weak record (one record per
    # cluster: preempted record, its reboot, the reset-lane record, its
    # reboot); a preemption on an in-flight physical step adds a separate
    # successor record. The two shapes must account for every cluster.
    in_record = int(audit.get("in_record_preemption_count", 0))
    cross_record = int(audit.get("cross_record_preemption_count", 0))
    if in_record + cross_record != clusters:
        errors.append(f"preemption shape counts do not add up to {clusters}")
    expected_counts = {
        "incident_count": clusters * 2,
        "workflow_count": clusters * 4 + cross_record,
        "weak_workflow_count": clusters,
        "strong_workflow_count": clusters,
        "reset_gpu_workflow_count": clusters,
        "preemption_pair_count": clusters,
        "weak_superseded_count": clusters,
        "strong_preempt_failed_fabric_reset_count": clusters,
        "reset_gpu_failed_count": clusters,
        "reboot_workflow_count": clusters * 2,
        "reboot_succeeded_count": clusters * 2,
        "reboot_successor_pair_count": clusters * 2,
        "unique_reboot_predecessor_count": clusters * 2,
    }
    for key, expected in expected_counts.items():
        if int(audit.get(key, 0)) != expected:
            errors.append(f"{key} does not equal {expected}")
    if int(audit.get("command_count", 0)) <= 0:
        errors.append("scenario created no remote commands")
    if audit["terminal_workflow_count"] != audit["workflow_count"]:
        errors.append("nonterminal workflows remain")
    if audit["terminal_command_count"] != audit["command_count"]:
        errors.append("nonterminal commands remain")
    for key in (
        "duplicate_idempotency_keys",
        "duplicate_workflow_steps",
        "duplicate_reboot_successors",
        "unexpected_reboot_predecessors",
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
{STORE_DSN_SNIPPET}
prefix='corr-live-{run_id}-%'
with psycopg.connect(store_dsn(), autocommit=True) as conn:
    cur=conn.cursor()
    cur.execute(
        "DELETE FROM gpu_fault_objects "
        "WHERE kind='xid_policy_decision' "
        "AND payload->>'event_id' LIKE %s",
        (prefix,),
    )
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
    parser.add_argument("--scenario-max-seconds", type=int, default=1800)
    parser.add_argument("--terminal-drain-seconds", type=int, default=600)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--allow-live-registry", action="store_true")
    parser.add_argument("--confirm-live-registry")
    parser.add_argument("--synthetic-ttl-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.synthetic_ttl_seconds < 300:
        parser.error("--synthetic-ttl-seconds must be at least 300")
    if args.scenario_max_seconds < 60:
        parser.error("--scenario-max-seconds must be at least 60")
    if args.terminal_drain_seconds < 0:
        parser.error("--terminal-drain-seconds must not be negative")
    timeout_seconds = args.timeout_seconds or args.scenario_max_seconds + 300
    if timeout_seconds < args.scenario_max_seconds + 60:
        parser.error(
            "--timeout-seconds must leave at least 60 seconds beyond "
            "--scenario-max-seconds"
        )
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
            scenario_max_seconds=args.scenario_max_seconds,
            active_deadline_seconds=timeout_seconds,
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
            f"--timeout={timeout_seconds}s",
            check=False,
            timeout=timeout_seconds + 60,
        )
        job = json.loads(dataplane("get", "job", JOB_NAME, "-o", "json"))
        conditions = {
            str(item.get("type")): str(item.get("status"))
            for item in job.get("status", {}).get("conditions", [])
        }
        job_status = (
            "Complete"
            if conditions.get("Complete") == "True"
            else "Failed"
            if conditions.get("Failed") == "True"
            else "Incomplete"
        )
        pod_results = collect_logs(artifacts / "pods")
        if job_status == "Complete":
            audit, drain_history = wait_for_terminal_drain(
                run_id,
                timeout_seconds=args.terminal_drain_seconds,
            )
        else:
            audit = database_audit(run_id)
            drain_history = []
        (artifacts / "terminal-drain.json").write_text(
            json.dumps(drain_history, indent=2, sort_keys=True) + "\n"
        )
        summary = {
            "run_id": run_id,
            "clusters": args.clusters,
            "registry_scope": registry_scope,
            "seeded": seeded,
            "job_wait_output": wait,
            "job_status": job_status,
            "job_succeeded": int(job.get("status", {}).get("succeeded", 0)),
            "job_failed": int(job.get("status", {}).get("failed", 0)),
            "scenario_max_seconds": args.scenario_max_seconds,
            "terminal_drain_seconds": args.terminal_drain_seconds,
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
