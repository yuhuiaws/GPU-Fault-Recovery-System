from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, cast

import boto3
import yaml  # type: ignore[import-untyped,unused-ignore]
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from gpu_fault.admin.site import load_site
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.boot_acceptance_common import (
    ROOT,
    BootAcceptanceError,
    SiteFixture,
    arn_resource_name,
    run,
    utc_now,
)

EXECUTOR_MANIFEST = ROOT / "deploy/dataplane/cluster-action-executor.yaml"
OPERATIONS_MANUAL = ROOT / "docs/部署和运维手册.md"
READINESS_PROBE = (ROOT / "scripts/e2e/regional/audit_executor_readiness.py").read_text(
    encoding="utf-8"
)
REMOTE_OWNER = "gpu-fault-boot015-nonexistent-adapter"


def secret_key_contract() -> dict[str, Any]:
    documents = [
        item
        for item in yaml.safe_load_all(EXECUTOR_MANIFEST.read_text(encoding="utf-8"))
        if item
    ]
    deployment = next(item for item in documents if item.get("kind") == "Deployment")
    required: list[tuple[str, str, str]] = []
    optional: list[tuple[str, str, str]] = []
    for container in deployment["spec"]["template"]["spec"]["containers"]:
        for item in container.get("env", []):
            reference = (item.get("valueFrom") or {}).get("secretKeyRef")
            if not reference:
                continue
            row = (str(reference["name"]), str(reference["key"]), str(item["name"]))
            (optional if reference.get("optional") else required).append(row)
    manual = OPERATIONS_MANUAL.read_text(encoding="utf-8")
    missing = [
        row
        for row in sorted(set(required))
        if re.search(
            rf"--from-(?:file|literal)={re.escape(row[1])}=",
            manual,
        )
        is None
    ]
    python_c_violations = []
    for container in deployment["spec"]["template"]["spec"]["containers"]:
        command_items = [str(item) for item in container.get("command", [])]
        args = [str(item) for item in container.get("args", [])]
        if command_items[-2:] == ["python", "-c"] or command_items[-2:] == [
            "python3",
            "-c",
        ]:
            if args and args[0][:1].isspace():
                python_c_violations.append(str(container.get("name")))
    return {
        "required": required,
        "optional": optional,
        "missing": missing,
        "python_c_leading_whitespace": python_c_violations,
        "passed": not missing and not python_c_violations,
    }


def run_boot011(
    fixture: SiteFixture,
    *,
    production_site: Path,
) -> dict[str, Any]:
    contract = secret_key_contract()
    production = load_site(production_site.resolve(), repository_root=ROOT)
    isolated_ids = {fixture.cluster_id}
    production_ids = {
        str(item["cluster_id"]) for item in production.release_config["clusters"]
    }
    deployment = json.loads(
        fixture.regional.kubectl(
            "gpu",
            "get",
            "deployment",
            "gpu-fault-cluster-executor",
            "-o",
            "json",
        )
    )
    secret = json.loads(
        fixture.regional.kubectl(
            "gpu",
            "get",
            "secret",
            "gpu-fault-regional-connection",
            "-o",
            "json",
        )
    )
    actual_keys = sorted((secret.get("data") or {}).keys())
    containers = deployment["spec"]["template"]["spec"]["containers"]
    environment = {
        item["name"]: item
        for container in containers
        for item in container.get("env", [])
    }
    pods = fixture.pods("gpu", "gpu-fault-cluster-executor")
    pod_state = json.loads(
        fixture.regional.kubectl(
            "gpu",
            "get",
            "pod",
            "-l",
            "app=gpu-fault-cluster-executor",
            "-o",
            "json",
        )
    )
    encoded = json.dumps(pod_state, sort_keys=True)
    logs = "\n".join(
        fixture.regional.kubectl(
            "gpu",
            "logs",
            pod,
            "--since=10m",
            check=False,
        )
        for pod in pods
    )
    service_account = json.loads(
        fixture.regional.kubectl(
            "gpu",
            "get",
            "serviceaccount",
            "gpu-fault-cluster-executor",
            "-o",
            "json",
        )
    )
    actual_role = (
        service_account.get("metadata", {})
        .get("annotations", {})
        .get("eks.amazonaws.com/role-arn")
    )
    expected_role = str(fixture.target["executor_irsa_role_arn"])
    role = run(
        [
            "aws",
            "iam",
            "get-role",
            "--role-name",
            arn_resource_name(expected_role),
            "--output",
            "json",
        ],
        timeout=120,
    )
    trust = json.loads(role.stdout)["Role"]["AssumeRolePolicyDocument"]
    trust_text = json.dumps(trust, sort_keys=True)
    checks = {
        "secret_contract": contract["passed"],
        "required_secret_keys_present": all(
            key in actual_keys for _name, key, _env in contract["required"]
        ),
        "optional_endpoint_key": bool(
            environment["GPU_FAULT_NODE_AGENT_ENDPOINTS"]["valueFrom"][
                "secretKeyRef"
            ].get("optional")
        ),
        "no_database_credentials": all(
            name != "GPU_FAULT_STORE_URL" and not name.startswith("POSTGRES_POOL_")
            for name in environment
        ),
        "executor_ready_within_180s": len(pods)
        == int(deployment["spec"].get("replicas") or 0),
        "no_config_or_tls_errors": not any(
            marker in encoded + logs
            for marker in (
                "CreateContainerConfigError",
                "couldn't find key",
                "CERTIFICATE_VERIFY_FAILED",
            )
        ),
        "isolated_registry_only": isolated_ids.isdisjoint(production_ids),
        "production_identity_absent": not any(
            cluster_id in logs for cluster_id in production_ids
        ),
        "dedicated_irsa_effective": actual_role == expected_role
        and "*" not in trust_text,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "details": {
            "secret_key_names": actual_keys,
            "secret_contract": contract,
            "executor_pods": len(pods),
            "isolated_cluster_count": len(isolated_ids),
            "production_cluster_count": len(production_ids),
        },
        "limitations": [
            "The fixture verifies an explicitly supplied isolated greenfield "
            "site; it never copies production Secret values."
        ],
    }


BOOT012_ENV_PROBE = r"""
import json
import os
from pathlib import Path

ca = os.environ.get("GPU_FAULT_CONTROL_PLANE_CA_FILE", "")
print(json.dumps({
    "ca_file": ca,
    "ca_exists": bool(ca and Path(ca).is_file()),
    "ssl_cert_file_set": bool(os.getenv("SSL_CERT_FILE")),
    "requests_ca_bundle_set": bool(os.getenv("REQUESTS_CA_BUNDLE")),
}, sort_keys=True))
"""

BOOT012_STS_PROBE = r"""
import json
import boto3
identity = boto3.client("sts").get_caller_identity()
print(json.dumps({"caller_identity_available": bool(identity.get("Arn"))}))
"""


def run_boot012(fixture: SiteFixture) -> dict[str, Any]:
    contract = secret_key_contract()
    pods = fixture.pods("gpu", "gpu-fault-cluster-executor")
    results = []
    for pod in pods:
        environment = fixture.pod_json("gpu", pod, BOOT012_ENV_PROBE)
        readiness = fixture.exec(
            "gpu",
            pod,
            "gpu-fault-cluster-executor-readiness",
            check=False,
        )
        empty_ca = fixture.exec(
            "gpu",
            pod,
            "sh",
            "-c",
            (
                ": > /tmp/boot012-empty-ca.pem; "
                "GPU_FAULT_CONTROL_PLANE_CA_FILE=/tmp/boot012-empty-ca.pem "
                "gpu-fault-cluster-executor-readiness"
            ),
            check=False,
        )
        sts = fixture.pod_json("gpu", pod, BOOT012_STS_PROBE)
        stale = fixture.exec(
            "gpu",
            pod,
            "python3",
            "-",
            input_text=READINESS_PROBE,
            check=False,
        )
        stale_payload = (
            json.loads(stale.stdout.splitlines()[-1])
            if stale.returncode == 0 and stale.stdout.strip()
            else {}
        )
        logs = fixture.regional.kubectl(
            "gpu",
            "logs",
            pod,
            "--since=5m",
            check=False,
        )
        error_count = sum(
            logs.lower().count(marker.lower())
            for marker in (
                "CERTIFICATE_VERIFY_FAILED",
                "SSLError",
                "URLError",
                "Traceback",
            )
        )
        results.append(
            {
                "pod": pod,
                "environment": environment,
                "authenticated_readiness": readiness.returncode == 0,
                "empty_ca_rejected": empty_ca.returncode != 0,
                "sts_public_trust": sts["caller_identity_available"],
                "stale_claim_503": (
                    (stale_payload.get("stale_claim") or {}).get("status") == 503
                ),
                "recent_error_count": error_count,
            }
        )
    checks = {
        "scoped_ca": contract["passed"]
        and all(
            item["environment"]["ca_exists"]
            and not item["environment"]["ssl_cert_file_set"]
            and not item["environment"]["requests_ca_bundle_set"]
            for item in results
        ),
        "authenticated_readiness": bool(results)
        and all(item["authenticated_readiness"] for item in results),
        "empty_ca_rejected": bool(results)
        and all(item["empty_ca_rejected"] for item in results),
        "sts_public_trust": bool(results)
        and all(item["sts_public_trust"] for item in results),
        "stale_claim_503": bool(results)
        and all(item["stale_claim_503"] for item in results),
        "no_recent_tls_errors": all(
            item["recent_error_count"] == 0 for item in results
        ),
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "replicas": results,
        "limitations": [
            "The empty-CA branch is a TLS negative probe; it does not modify "
            "the running executor process environment."
        ],
    }


BOOT013_PROBE = r"""
import json
import os
import subprocess
import sys

from botocore.exceptions import ParamValidationError
from gpu_fault.notifications import SesEmailNotifier, SesNotificationConfig

config = SesNotificationConfig.from_environment()
client = SesEmailNotifier._create_client(config)
local_validation = False
try:
    client.send_email(
        FromEmailAddress=config.sender,
        Destination={"ToAddresses": [1]},
        Content={"Simple": {
            "Subject": {"Data": "boot013"},
            "Body": {"Text": {"Data": "boot013"}},
        }},
    )
except ParamValidationError:
    local_validation = True

child = os.environ.copy()
child.pop("AWS_REGION", None)
child.pop("AWS_DEFAULT_REGION", None)
child["AWS_CONFIG_FILE"] = "/dev/null"
negative = subprocess.run(
    [
        sys.executable,
        "-c",
        (
            "from gpu_fault.notifications import "
            "SesEmailNotifier,SesNotificationConfig;"
            "SesEmailNotifier._create_client("
            "SesNotificationConfig.from_environment())"
        ),
    ],
    env=child,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
print(json.dumps({
    "region_present": bool(os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")),
    "config_region": config.region_name,
    "client_region": client.meta.region_name,
    "execution_enabled": config.execution_enabled,
    "allow_email": os.getenv("GPU_FAULT_ALLOW_EMAIL", "").lower() == "true",
    "local_param_validation": local_validation,
    "no_region_error": "NoRegionError" in negative.stderr,
}, sort_keys=True))
"""


def run_boot013(fixture: SiteFixture) -> dict[str, Any]:
    pods = fixture.pods("cpu", "gpu-fault-api-ha")
    results = [fixture.pod_json("cpu", pod, BOOT013_PROBE) for pod in pods]
    checks = {
        "api_replicas_present": bool(results),
        "region_present_all": all(item["region_present"] for item in results),
        "client_region_matches_all": all(
            item["config_region"] == fixture.region
            and item["client_region"] == fixture.region
            for item in results
        ),
        "execution_enabled_matches_all": all(
            item["execution_enabled"] == item["allow_email"] for item in results
        ),
        "local_zero_network_param_validation_all": all(
            item["local_param_validation"] for item in results
        ),
        "no_region_error_all": all(item["no_region_error"] for item in results),
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "replicas": results,
        "limitations": [
            "The mandatory path performs only botocore local parameter "
            "validation and does not send an acceptance email."
        ],
    }


BOOT014_PROBE = r"""
import json
import os
from collections import Counter

from gpu_fault.app import ApplicationContext

context = ApplicationContext.from_environment()
service = context.advisory_notifications
notifications = context.store.list_notifications()
results = [
    context.store.get_notification_result(item.notification_id)
    for item in notifications
]
statuses = Counter(
    item.status.value for item in results if item is not None
)
print(json.dumps({
    "allow_email": os.getenv("GPU_FAULT_ALLOW_EMAIL", "").lower() == "true",
    "async_delivery": service.async_delivery,
    "dispatcher_enabled": service.dispatcher_enabled,
    "deliver_backlog": service.deliver_backlog,
    "backlog_grace_seconds": service.backlog_grace_seconds,
    "notification_count": len(notifications),
    "result_count": sum(item is not None for item in results),
    "status_counts": dict(statuses),
}, sort_keys=True))
"""


def run_boot014(fixture: SiteFixture) -> dict[str, Any]:
    pods = fixture.pods("cpu", "gpu-fault-api-ha")
    values = [fixture.pod_json("cpu", pod, BOOT014_PROBE) for pod in pods]
    normalized = [
        {
            key: item[key]
            for key in (
                "allow_email",
                "async_delivery",
                "dispatcher_enabled",
                "deliver_backlog",
                "backlog_grace_seconds",
            )
        }
        for item in values
    ]
    backlog = sum(
        item["notification_count"] - item["result_count"] for item in values[:1]
    )
    checks = {
        "api_replicas_present": bool(values),
        "replica_env_consistent": len(
            {json.dumps(item, sort_keys=True) for item in normalized}
        )
        == 1,
        "allow_email_equals_dispatcher": all(
            item["allow_email"] == item["dispatcher_enabled"] for item in values
        ),
        "dangerous_combination_absent": all(
            not (
                item["async_delivery"]
                and item["allow_email"]
                and not item["dispatcher_enabled"]
            )
            for item in values
        ),
        "backlog_without_result_zero": backlog == 0,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "replicas": values,
        "limitations": [
            "This case audits live dispatcher configuration and backlog state; "
            "it does not toggle email delivery."
        ],
    }


PROFILE_OWNER_PROBE = r"""
import json
import os
import sys

from gpu_fault.app import ApplicationContext

cluster_id, owners_csv = sys.argv[1:]
executor_owners = {item for item in owners_csv.split(",") if item}
remote_owners = {
    item.strip()
    for item in os.getenv(
        "GPU_FAULT_REMOTE_EXECUTION_OWNERS",
        "gpu-fault-kubernetes-adapter,gpu-fault-node-agent,"
        "gpu-fault-hyperpod-adapter",
    ).split(",")
    if item.strip()
}
profiles = []
for profile in ApplicationContext.from_environment().store._list("profile"):
    if profile.cluster_id != cluster_id:
        continue
    owners = {
        item.owner
        for item in profile.capabilities
        if getattr(item.mode, "value", item.mode) in {"OWN", "DELEGATE"}
    }
    profiles.append({
        "profile_version": profile.profile_version,
        "remote_owners": sorted(owners & remote_owners),
        "orphan_owners": sorted((owners & remote_owners) - executor_owners),
    })
print(json.dumps({"profiles": profiles}, sort_keys=True))
"""

REMOTE_BASELINE_PROBE = r"""
import json
from collections import Counter
from gpu_fault.app import ApplicationContext
commands = ApplicationContext.from_environment().store.list_remote_commands()
print(json.dumps({
    "count": len(commands),
    "by_status": dict(Counter(item.status.value for item in commands)),
}, sort_keys=True))
"""

REMOTE_INJECT_PROBE = r"""
import json
import sys
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import RemoteActionCommand

cluster_id, command_id, owner = sys.argv[1:]
created = datetime.now(timezone.utc) - timedelta(seconds=610)
incident = FaultIncident(
    incident_id=command_id,
    event_id=command_id,
    event_type="BOOT015_PROBE",
    cluster_id=cluster_id,
    node_ids=["boot015-nonexistent-node"],
    policy_version="boot015",
    policy_source="boot015-probe",
)
command = RemoteActionCommand(
    command_id=command_id,
    cluster_id=cluster_id,
    workflow_request_id=command_id,
    incident_id=incident.incident_id,
    step_index=0,
    fencing_token=1,
    idempotency_key=command_id,
    step=WorkflowStepSpec(
        operation=WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
        execution_owner=owner,
        node_ids=["boot015-nonexistent-node"],
    ),
    workflow=WorkflowRequest(
        request_id=command_id,
        incident_id=incident.incident_id,
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
    ),
    incident=incident,
    created_at=created,
    updated_at=created,
)
stored = ApplicationContext.from_environment().store.ensure_remote_command(command)
print(json.dumps({"command_id": stored.command_id, "status": stored.status.value}))
"""

REMOTE_DELETE_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
store = ApplicationContext.from_environment().store
store._delete("remote_command", sys.argv[1])
print(json.dumps({"remaining": len(store.list_remote_commands())}))
"""

METRIC_PROBE = r"""
import json
import sys
import time
import urllib.request

cluster_id = sys.argv[1]
started = time.perf_counter()
text = urllib.request.urlopen("http://127.0.0.1:8080/metrics", timeout=5).read().decode()
elapsed = time.perf_counter() - started
pending = 0.0
oldest = None
for line in text.splitlines():
    if line.startswith('gpu_fault_remote_command_total{status="PENDING"}'):
        pending = float(line.rsplit(" ", 1)[1])
    if (
        line.startswith("gpu_fault_remote_command_oldest_unclaimed_seconds")
        and f'cluster_id="{cluster_id}"' in line
    ):
        oldest = float(line.rsplit(" ", 1)[1])
print(json.dumps({
    "duration_seconds": round(elapsed, 6),
    "pending": pending,
    "oldest_unclaimed_seconds": oldest,
}, sort_keys=True))
"""


def amp_request(
    *,
    region: str,
    workspace_id: str,
    method: str,
    path: str,
    parameters: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = (
        f"https://aps-workspaces.{region}.amazonaws.com/workspaces/{workspace_id}{path}"
    )
    body = urllib.parse.urlencode(parameters or {})
    headers = (
        {"Content-Type": "application/x-www-form-urlencoded"}
        if method == "POST"
        else {}
    )
    request = AWSRequest(
        method=method,
        url=url,
        data=body if method == "POST" else None,
        headers=headers,
    )
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise BootAcceptanceError("AWS credentials are unavailable for AMP query")
    SigV4Auth(credentials, "aps", region).add_auth(request)
    response = urllib.request.Request(
        url,
        data=body.encode() if method == "POST" else None,
        headers=dict(request.headers),
        method=method,
    )
    with urllib.request.urlopen(response, timeout=30) as handle:
        return cast(dict[str, Any], json.load(handle))


def wait_amp_alert(
    fixture: SiteFixture,
    *,
    expect_firing: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    workspace_id = str(fixture.config["health"]["amp_workspace_id"])
    deadline = time.monotonic() + timeout_seconds
    timeline = []
    while time.monotonic() < deadline:
        query = amp_request(
            region=fixture.region,
            workspace_id=workspace_id,
            method="POST",
            path="/api/v1/query",
            parameters={
                "query": (
                    "max by (cluster_id) "
                    "(gpu_fault_remote_command_oldest_unclaimed_seconds) > 600"
                )
            },
        )["data"]["result"]
        alerts = amp_request(
            region=fixture.region,
            workspace_id=workspace_id,
            method="GET",
            path="/api/v1/alerts",
        )["data"]["alerts"]
        vector = any(
            item.get("metric", {}).get("cluster_id") == fixture.cluster_id
            for item in query
        )
        states = [
            item.get("state")
            for item in alerts
            if item.get("labels", {}).get("alertname")
            == "GpuFaultRemoteCommandUnclaimed"
            and item.get("labels", {}).get("cluster_id") == fixture.cluster_id
        ]
        timeline.append({"vector": vector, "states": states, "at": utc_now()})
        if expect_firing and "firing" in states:
            return {"matched": True, "timeline": timeline}
        if not expect_firing and not vector and not states:
            return {"matched": True, "timeline": timeline}
        time.sleep(15)
    return {"matched": False, "timeline": timeline}


def run_boot015(
    fixture: SiteFixture,
    *,
    case_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    command_id = f"remote-boot015-{attempt}-{int(time.time())}"
    pods = fixture.pods("gpu", "gpu-fault-cluster-executor")
    owner_sets = []
    for pod in pods:
        value = fixture.pod_json(
            "gpu",
            pod,
            (
                "import json;"
                "from gpu_fault.cluster_executor import executor_from_environment;"
                "print(json.dumps({'owners':sorted("
                "executor_from_environment().execution_owners)}))"
            ),
        )
        owner_sets.append(value["owners"])
    if not owner_sets:
        raise BootAcceptanceError("no Ready executor Pods")
    profile = fixture.regional.cpu_python(
        PROFILE_OWNER_PROBE,
        fixture.cluster_id,
        ",".join(owner_sets[0]),
    )
    static_alert = run(
        [sys.executable, "scripts/verify-regional-alerting.py"],
        check=False,
    )
    baseline = fixture.regional.cpu_python(REMOTE_BASELINE_PROBE)
    result: dict[str, Any] = {"verdict": "FAIL"}
    try:
        fixture.regional.cpu_python(
            REMOTE_INJECT_PROBE,
            fixture.cluster_id,
            command_id,
            REMOTE_OWNER,
        )
        readiness = []
        for pod in pods:
            completed = fixture.exec(
                "gpu",
                pod,
                "gpu-fault-cluster-executor-readiness",
                check=False,
            )
            readiness.append(
                {
                    "pod": pod,
                    "returncode": completed.returncode,
                    "owner_backlog_reason": (
                        "open backlog needs execution owners"
                        in completed.stdout + completed.stderr
                    ),
                }
            )
        first = fixture.regional.cpu_python(METRIC_PROBE, fixture.cluster_id)
        time.sleep(30)
        second = fixture.regional.cpu_python(METRIC_PROBE, fixture.cluster_id)
        logs = [
            fixture.regional.kubectl(
                "gpu",
                "logs",
                pod,
                "--since=5m",
                check=False,
            )
            for pod in pods
        ]
        firing = wait_amp_alert(
            fixture,
            expect_firing=True,
            timeout_seconds=300,
        )
        checks = {
            "profile_remote_owners_subset": all(
                not item["orphan_owners"] for item in profile["profiles"]
            ),
            "executor_owner_sets_identical": all(
                item == owner_sets[0] for item in owner_sets
            ),
            "readiness_failed_closed_replicas": all(
                item["returncode"] != 0 and item["owner_backlog_reason"]
                for item in readiness
            ),
            "metrics_under_five_seconds": (
                first["duration_seconds"] < 5 and second["duration_seconds"] < 5
            ),
            "oldest_metric_increases": (
                first["oldest_unclaimed_seconds"] is not None
                and second["oldest_unclaimed_seconds"]
                > first["oldest_unclaimed_seconds"]
            ),
            "executor_logs_exclude_command": all(
                command_id not in text for text in logs
            ),
            "alert_reachability": static_alert.returncode == 0,
            "amp_firing": firing["matched"],
        }
        result = {
            "verdict": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "owner_sets": owner_sets,
            "profile": profile,
            "readiness": readiness,
            "metric_samples": [first, second],
            "amp_firing": firing,
        }
    finally:
        cleanup = fixture.regional.cpu_python(REMOTE_DELETE_PROBE, command_id)
        resolved = wait_amp_alert(
            fixture,
            expect_firing=False,
            timeout_seconds=180,
        )
        recovered = []
        for pod in pods:
            completed = fixture.exec(
                "gpu",
                pod,
                "gpu-fault-cluster-executor-readiness",
                check=False,
            )
            recovered.append(completed.returncode == 0)
        final = fixture.regional.cpu_python(REMOTE_BASELINE_PROBE)
        result["cleanup"] = {
            "delete": cleanup,
            "amp_resolved": resolved,
            "readiness_recovered": recovered,
            "remote_count_restored": final["count"] == baseline["count"],
        }
        if (
            not resolved["matched"]
            or not all(recovered)
            or final["count"] != baseline["count"]
        ):
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / "boot015-details.json", result)
    result["limitations"] = [
        "The current implementation validates owner compatibility at runtime "
        "readiness; registration-time owner validation remains a known gap."
    ]
    return result
