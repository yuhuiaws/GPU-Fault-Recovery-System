from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

import boto3
import yaml  # type: ignore[import-untyped,unused-ignore]
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from gpu_fault.admin.site import load_site
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.boot_acceptance_common import (
    parse_probe_json,
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
EXECUTOR_DEPLOYMENT = "gpu-fault-cluster-executor"
CA_FILE_ENV = "GPU_FAULT_CONTROL_PLANE_CA_FILE"
# Either of these would replace boto3's public trust roots with the private CA
# bundle and break STS/HyperPod calls; the manifest deliberately leaves them out.
FORBIDDEN_TRUST_ENV = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
EXECUTOR_READY_LIMIT_SECONDS = 180


def manifest_deployment() -> dict[str, Any]:
    documents = [
        item
        for item in yaml.safe_load_all(EXECUTOR_MANIFEST.read_text(encoding="utf-8"))
        if item
    ]
    return cast(
        dict[str, Any],
        next(item for item in documents if item.get("kind") == "Deployment"),
    )


def _container_env(deployment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["name"]): item
        for container in deployment["spec"]["template"]["spec"]["containers"]
        for item in container.get("env", [])
    }


def _ca_mount(deployment: dict[str, Any], ca_path: str) -> dict[str, Any] | None:
    """The volumeMount + volume that serve ``ca_path``, or ``None``."""

    spec = deployment["spec"]["template"]["spec"]
    volumes = {str(item["name"]): item for item in spec.get("volumes", [])}
    for container in spec["containers"]:
        for mount in container.get("volumeMounts", []):
            mount_path = str(mount.get("mountPath") or "").rstrip("/")
            if mount_path and ca_path.startswith(mount_path + "/"):
                volume = volumes.get(str(mount.get("name"))) or {}
                secret = volume.get("secret") or {}
                keys = sorted(str(item.get("key")) for item in secret.get("items", []))
                return {
                    "mount_path": mount_path,
                    "read_only": bool(mount.get("readOnly")),
                    "secret_name": secret.get("secretName"),
                    "secret_keys": keys,
                }
    return None


def ca_file_contract(
    manifest: dict[str, Any],
    live: dict[str, Any],
) -> dict[str, Any]:
    """The CA contract: the manifest's scoped CA file is what the live Pod runs.

    The secretKeyRef contract says nothing about the CA; this one does. The
    manifest declares ``GPU_FAULT_CONTROL_PLANE_CA_FILE`` and the read-only
    Secret mount that serves it, and the live Deployment must carry the same
    value and mount, with neither ``SSL_CERT_FILE`` nor ``REQUESTS_CA_BUNDLE``
    anywhere in either environment.
    """

    manifest_env = _container_env(manifest)
    live_env = _container_env(live)
    ca_path = str((manifest_env.get(CA_FILE_ENV) or {}).get("value") or "")
    live_path = str((live_env.get(CA_FILE_ENV) or {}).get("value") or "")
    manifest_mount = _ca_mount(manifest, ca_path) if ca_path else None
    live_mount = _ca_mount(live, live_path) if live_path else None
    forbidden = sorted(
        name for name in FORBIDDEN_TRUST_ENV if name in manifest_env or name in live_env
    )
    return {
        "ca_file": ca_path,
        "live_ca_file": live_path,
        "manifest_mount": manifest_mount,
        "live_mount": live_mount,
        "forbidden_trust_env_present": forbidden,
        "passed": bool(ca_path)
        and ca_path == live_path
        and manifest_mount is not None
        and manifest_mount == live_mount
        and bool(manifest_mount["read_only"])
        and bool(manifest_mount["secret_name"])
        and not forbidden,
    }


def secret_key_contract() -> dict[str, Any]:
    deployment = manifest_deployment()
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


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def ready_within(
    pod_state: dict[str, Any],
    *,
    replicas: int,
    limit_seconds: int = EXECUTOR_READY_LIMIT_SECONDS,
) -> dict[str, Any]:
    """Whether every desired executor replica became Ready within the limit.

    ``len(ready_pods) == replicas`` alone is true for a Deployment scaled to
    zero, and says nothing about *when* the Pods became Ready. This requires at
    least one replica, exactly that many Ready Pods, and for each of them a
    Ready transition no later than ``limit_seconds`` after creation.
    """

    durations: dict[str, float | None] = {}
    for item in pod_state.get("items", []):
        name = str(item["metadata"]["name"])
        created = _timestamp(item["metadata"].get("creationTimestamp"))
        ready_at = None
        for condition in item.get("status", {}).get("conditions", []):
            if condition.get("type") == "Ready" and condition.get("status") == "True":
                ready_at = _timestamp(condition.get("lastTransitionTime"))
        durations[name] = (
            (ready_at - created).total_seconds()
            if created is not None and ready_at is not None
            else None
        )
    ready = {name: value for name, value in durations.items() if value is not None}
    return {
        "replicas": replicas,
        "ready_seconds": durations,
        "passed": replicas >= 1
        and len(ready) == replicas
        and all(value <= limit_seconds for value in ready.values()),
    }


def foreign_lease_owners(
    owners: list[str],
    *,
    isolated_cluster_id: str,
    isolated_pods: list[str],
) -> list[str]:
    """Lease owners in the production store that belong to the isolated executor.

    The executor identifies itself as ``<cluster_id>/<hostname>`` unless
    ``GPU_FAULT_CLUSTER_EXECUTOR_ID`` overrides it, and the hostname is the Pod
    name; either form appearing as a production lease owner means the isolated
    executor claimed production commands, which turns the case from a
    deployability check into a production mutation.
    """

    prefix = f"{isolated_cluster_id}/"
    pods = set(isolated_pods)
    return sorted(
        owner
        for owner in owners
        if owner.startswith(prefix)
        or owner in pods
        or any(owner.endswith(f"/{pod}") for pod in pods)
    )


LEASE_OWNER_PROBE = r"""
import json
from gpu_fault.app import ApplicationContext
owners = set()
for item in ApplicationContext.from_environment().store.list_remote_commands():
    for value in (item.lease_owner, item.last_lease_owner):
        if value:
            owners.add(value)
print(json.dumps({"owners": sorted(owners)}, sort_keys=True))
"""


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
    if not production_ids:
        raise BootAcceptanceError("production site names no GPU clusters")
    production_fixture = SiteFixture(
        production_site.resolve(),
        sorted(production_ids)[0],
    )
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
    replicas = int(deployment["spec"].get("replicas") or 0)
    readiness = ready_within(pod_state, replicas=replicas)
    # Read-only: the production store's lease owners, to prove the isolated
    # executor never claimed a production command.
    production_owners = production_fixture.regional.cpu_python(LEASE_OWNER_PROBE)
    foreign = foreign_lease_owners(
        [str(item) for item in production_owners.get("owners") or []],
        isolated_cluster_id=fixture.cluster_id,
        isolated_pods=pods,
    )
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
        "executor_ready_within_180s": readiness["passed"],
        "isolated_executor_never_owned_production_lease": not foreign,
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
            "executor_readiness": readiness,
            "foreign_production_lease_owners": foreign,
            "isolated_cluster_count": len(isolated_ids),
            "production_cluster_count": len(production_ids),
        },
        **fixture.regional.evidence_identity(),
        "limitations": [
            "The fixture verifies an explicitly supplied isolated greenfield "
            "site; it never copies production Secret values, and reads the "
            "production store only for lease owners."
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


def _tail(completed: subprocess.CompletedProcess[str], limit: int = 600) -> str:
    return (completed.stdout + completed.stderr)[-limit:]


MATRIX_STATUS = {
    "valid": 200,
    "wrong_token": 403,
    "wrong_pin": 503,
    "no_owner": 503,
    "stale_claim": 503,
}


def readiness_matrix_verdict(replicas: list[dict[str, Any]]) -> dict[str, Any]:
    """The BOOT-021 conclusion from the per-Pod readiness matrices BOOT-012 ran.

    BOOT-012 already executes ``audit_executor_readiness.py`` in every executor
    Pod, which is the whole BOOT-021 case; recording it once here spares a
    second live run whose only difference would be the case id on the file.
    """

    matrices: list[dict[str, Any]] = []
    complete = bool(replicas)
    for item in replicas:
        matrix = item.get("readiness_matrix")
        if not isinstance(matrix, dict) or any(
            key not in matrix for key in MATRIX_STATUS
        ):
            complete = False
            continue
        matrices.append(matrix)
    checks: dict[str, bool] = {"matrix_recorded_per_replica": complete}
    for key, status in MATRIX_STATUS.items():
        checks[f"{key}_{status}"] = complete and all(
            (matrix[key] or {}).get("status") == status for matrix in matrices
        )
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "replicas": [
            {"pod": item["pod"], "readiness_matrix": item.get("readiness_matrix")}
            for item in replicas
        ],
    }


# The markers with which the executor client refuses an empty CA file: the TLS
# handshake failing verification (OpenSSL < 3.5) or the trust store refusing to
# load an empty PEM at all (OpenSSL 3.5, live 2026-09-13 after a cold image build).
EMPTY_CA_REJECTION_MARKERS = (
    "CERTIFICATE_VERIFY_FAILED",
    "NO_CERTIFICATE_OR_CRL_FOUND",
)


def empty_ca_rejection(output: str) -> bool:
    return any(marker in output for marker in EMPTY_CA_REJECTION_MARKERS)


def run_boot012(
    fixture: SiteFixture,
    *,
    boot021_evidence_path: Path | None = None,
) -> dict[str, Any]:
    started_at = utc_now()
    contract = secret_key_contract()
    live_deployment = json.loads(
        fixture.regional.kubectl(
            "gpu",
            "get",
            "deployment",
            EXECUTOR_DEPLOYMENT,
            "-o",
            "json",
        )
    )
    ca_contract = ca_file_contract(manifest_deployment(), live_deployment)
    pods = fixture.pods("gpu", EXECUTOR_DEPLOYMENT)
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
                f"{CA_FILE_ENV}=/tmp/boot012-empty-ca.pem "
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
        # audit_executor_readiness.py prints an indented JSON document, so the
        # last line alone is "}" and never parses (BOOT-012 failed on it live).
        stale_payload = (
            parse_probe_json(stale.stdout)
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
        empty_ca_output = empty_ca.stdout + empty_ca.stderr
        results.append(
            {
                "pod": pod,
                "environment": environment,
                "ca_file_matches_contract": (
                    environment["ca_file"] == ca_contract["ca_file"]
                    and environment["ca_exists"]
                    and not environment["ssl_cert_file_set"]
                    and not environment["requests_ca_bundle_set"]
                ),
                "authenticated_readiness": readiness.returncode == 0,
                # A non-zero exit alone could be any failure; the negative probe
                # has to fail *because* the empty CA cannot verify the server.
                # OpenSSL < 3.5 loaded an empty PEM silently and the handshake
                # then failed CERTIFICATE_VERIFY_FAILED; OpenSSL 3.5 refuses the
                # empty file when the client builds its context
                # (X509: NO_CERTIFICATE_OR_CRL_FOUND). Both are the rejection
                # the case wants; what must not happen is a completed request.
                "empty_ca_rejected": empty_ca.returncode != 0
                and empty_ca_rejection(empty_ca_output),
                "sts_public_trust": sts["caller_identity_available"],
                "stale_claim_503": (
                    (stale_payload.get("stale_claim") or {}).get("status") == 503
                ),
                "readiness_matrix": stale_payload or None,
                "recent_error_count": error_count,
                # stderr tails are kept only for probes that did not do what the
                # case expects, so a live failure is diagnosable without a rerun.
                "readiness_stderr_tail": (
                    _tail(readiness) if readiness.returncode != 0 else None
                ),
                "empty_ca_stderr_tail": (
                    _tail(empty_ca) if not empty_ca_rejection(empty_ca_output) else None
                ),
                "stale_probe_stderr_tail": (
                    _tail(stale) if stale.returncode != 0 else None
                ),
            }
        )
    checks = {
        "scoped_ca": contract["passed"]
        and ca_contract["passed"]
        and bool(results)
        and all(item["ca_file_matches_contract"] for item in results),
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
    identity = fixture.regional.evidence_identity()
    result = {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "ca_contract": ca_contract,
        "replicas": results,
        **identity,
        "limitations": [
            "The empty-CA branch is a TLS negative probe; it does not modify "
            "the running executor process environment."
        ],
    }
    if boot021_evidence_path is not None:
        boot021 = readiness_matrix_verdict(results)
        write_json_atomic(
            boot021_evidence_path,
            {
                "schema_version": 2,
                "report_type": "fault-acceptance",
                "case_id": "GF-REGIONAL-BOOT-021",
                "verdict": boot021["verdict"],
                "started_at": started_at,
                "executed_at": utc_now(),
                "recorded_by": "GF-REGIONAL-BOOT-012",
                "checks": boot021["checks"],
                "replicas": boot021["replicas"],
                **identity,
                "limitations": [
                    "Recorded from the readiness matrix BOOT-012 ran in every "
                    "executor Pod; the matrix is the BOOT-021 procedure."
                ],
            },
        )
        result["boot021_evidence"] = str(boot021_evidence_path)
    return result


BOOT013_PROBE = r"""
import json
import os
import subprocess
import sys

from botocore.exceptions import ParamValidationError
from gpu_fault.env import env_bool

# The live channel decides which client the case inspects. Both clients take
# their region from the environment, never from the Pod's SDK defaults; the SNS
# client additionally pins it to the topic ARN's region when the variables are
# absent, so its negative branch proves that pin instead of NoRegionError.
channel = (os.getenv("GPU_FAULT_NOTIFICATION_CHANNEL") or "ses").strip().lower()
if channel == "sns":
    from gpu_fault.notifications.sns import SnsNotificationConfig, SnsNotifier

    config = SnsNotificationConfig.from_environment()
    client = SnsNotifier._create_client(config)
    allow_default = True

    def zero_network_call():
        client.publish(TopicArn=config.topic_arn, Message=1)

    negative_source = (
        "from gpu_fault.notifications.sns import SnsNotificationConfig, SnsNotifier;"
        "c=SnsNotificationConfig.from_environment();"
        "print('REGION=' + str(SnsNotifier._create_client(c).meta.region_name))"
    )
else:
    from gpu_fault.notifications import SesEmailNotifier, SesNotificationConfig

    config = SesNotificationConfig.from_environment()
    client = SesEmailNotifier._create_client(config)
    allow_default = False

    def zero_network_call():
        client.send_email(
            FromEmailAddress=config.sender,
            Destination={"ToAddresses": [1]},
            Content={"Simple": {
                "Subject": {"Data": "boot013"},
                "Body": {"Text": {"Data": "boot013"}},
            }},
        )

    negative_source = (
        "from gpu_fault.notifications import "
        "SesEmailNotifier,SesNotificationConfig;"
        "SesEmailNotifier._create_client("
        "SesNotificationConfig.from_environment())"
    )

local_validation = False
try:
    zero_network_call()
except ParamValidationError:
    local_validation = True

child = os.environ.copy()
child.pop("AWS_REGION", None)
child.pop("AWS_DEFAULT_REGION", None)
child["AWS_CONFIG_FILE"] = "/dev/null"
negative = subprocess.run(
    [sys.executable, "-c", negative_source],
    env=child,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
if channel == "sns":
    topic_region = config.topic_arn.split(":")[3]
    no_region_error = (
        negative.returncode == 0 and f"REGION={topic_region}" in negative.stdout
    )
else:
    no_region_error = "NoRegionError" in negative.stderr
print(json.dumps({
    "channel": channel,
    "region_present": bool(os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")),
    "config_region": config.region_name,
    "client_region": client.meta.region_name,
    "execution_enabled": config.execution_enabled,
    "allow_email": env_bool("GPU_FAULT_ALLOW_EMAIL", allow_default),
    "local_param_validation": local_validation,
    "no_region_error": no_region_error,
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
            "validation and does not send an acceptance email.",
            "On the sns channel the negative branch proves the client region is "
            "pinned to the topic ARN without AWS_REGION, not NoRegionError.",
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
    "ids": sorted(item.command_id for item in commands),
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
    deadline: float | None = None,
) -> dict[str, Any]:
    """Poll AMP until the alert reaches the expected state or the deadline.

    ``deadline`` is a ``time.monotonic()`` value; BOOT-015 sets it right after
    the injection so the readiness, metric and log reads it does in between
    count against the same window instead of extending it.
    """

    workspace_id = str(fixture.config["health"]["amp_workspace_id"])
    if deadline is None:
        deadline = time.monotonic() + timeout_seconds
    timeline = []
    while True:
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
        if time.monotonic() >= deadline:
            return {"matched": False, "timeline": timeline}
        time.sleep(min(15.0, max(0.0, deadline - time.monotonic())))


REMOTE_STATE_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError
store = ApplicationContext.from_environment().store
try:
    command = store.get_remote_command(sys.argv[1])
except NotFoundError:
    print(json.dumps({"exists": False, "status": None, "lease_owner": None}))
else:
    print(json.dumps({
        "exists": True,
        "status": command.status.value,
        "lease_owner": command.lease_owner,
    }, sort_keys=True))
"""

AMP_FIRING_WINDOW_SECONDS = 300
AMP_RESOLVE_WINDOW_SECONDS = 180
# Two samples of an age metric are monotonic whatever the gap; 5 s is enough
# to see it move and does not stretch the alert window.
METRIC_SAMPLE_GAP_SECONDS = 5


def _cleanup_step(
    cleanup: dict[str, Any],
    name: str,
    operation: Callable[[], Any],
) -> Any:
    """Run one cleanup step and record its result or its error under ``name``.

    The steps are independent, so one failing (the delete exec dropping, AMP
    unreachable) must not skip the ones after it; the first version's
    ``finally`` ran them in sequence and an exception in the first left the
    injected command in place, which keeps every executor readiness probe
    failing until someone deletes it by hand.
    """

    try:
        value = operation()
    except Exception as exc:  # noqa: BLE001 - recorded, later steps still run
        cleanup[name] = {"error": f"{type(exc).__name__}: {exc}"}
        return None
    cleanup[name] = value
    return value


def boot015_cleanup(
    fixture: SiteFixture,
    *,
    command_id: str,
    pods: list[str],
    baseline_ids: set[str],
) -> dict[str, Any]:
    """Delete the injected command and record every observation about it."""

    cleanup: dict[str, Any] = {}
    _cleanup_step(
        cleanup,
        "delete",
        lambda: fixture.regional.cpu_python(REMOTE_DELETE_PROBE, command_id),
    )
    final = _cleanup_step(
        cleanup,
        "final_baseline",
        lambda: fixture.regional.cpu_python(REMOTE_BASELINE_PROBE),
    )
    final_ids = set((final or {}).get("ids") or [])
    cleanup["synthetic_command_removed"] = final is not None and (
        command_id not in final_ids
    )
    # The retention sweep deletes old terminal commands on its own schedule,
    # so the total count drifts during the case; what the case owns is that
    # its command is gone and it introduced no other.
    cleanup["commands_introduced"] = sorted(final_ids - baseline_ids)
    cleanup["remote_count_after"] = (final or {}).get("count")
    if cleanup["synthetic_command_removed"]:
        # Staleness of the AMP series is not a cleanup defect: the gauge may
        # keep its last value for a scrape interval after the command is gone.
        # The wait is recorded, never a verdict.
        _cleanup_step(
            cleanup,
            "resolve_observed",
            lambda: wait_amp_alert(
                fixture,
                expect_firing=False,
                timeout_seconds=AMP_RESOLVE_WINDOW_SECONDS,
            ),
        )

    def readiness() -> dict[str, bool]:
        recovered = {}
        for pod in pods:
            completed = fixture.exec(
                "gpu",
                pod,
                "gpu-fault-cluster-executor-readiness",
                check=False,
            )
            recovered[pod] = completed.returncode == 0
        return recovered

    recovered = _cleanup_step(cleanup, "readiness_recovered", readiness)
    cleanup["passed"] = bool(
        cleanup["synthetic_command_removed"]
        and not cleanup["commands_introduced"]
        and isinstance(recovered, dict)
        and recovered
        and all(recovered.values())
    )
    return cleanup


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
    baseline_ids = set(baseline.get("ids") or [])
    result: dict[str, Any] = {
        "verdict": "FAIL",
        "remote_count_before": baseline["count"],
    }
    injected = False
    try:
        fixture.regional.cpu_python(
            REMOTE_INJECT_PROBE,
            fixture.cluster_id,
            command_id,
            REMOTE_OWNER,
        )
        injected = True
        # The alert window starts at injection; the reads below happen inside it.
        alert_deadline = time.monotonic() + AMP_FIRING_WINDOW_SECONDS
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
        time.sleep(METRIC_SAMPLE_GAP_SECONDS)
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
            timeout_seconds=AMP_FIRING_WINDOW_SECONDS,
            deadline=alert_deadline,
        )
        # Before deletion the command must still be exactly what was injected:
        # pending in the catalog and never leased, or an executor did claim it.
        state = fixture.regional.cpu_python(REMOTE_STATE_PROBE, command_id)
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
            "pending_counted": second["pending"] >= 1,
            "oldest_metric_increases": (
                first["oldest_unclaimed_seconds"] is not None
                and second["oldest_unclaimed_seconds"]
                > first["oldest_unclaimed_seconds"]
            ),
            "never_leased": state["exists"]
            and state["status"] == "PENDING"
            and state["lease_owner"] is None,
            "executor_logs_exclude_command": all(
                command_id not in text for text in logs
            ),
            "alert_reachability": static_alert.returncode == 0,
            "amp_firing": firing["matched"],
        }
        result.update(
            {
                "verdict": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
                "owner_sets": owner_sets,
                "profile": profile,
                "readiness": readiness,
                "metric_samples": [first, second],
                "pre_delete_state": state,
                "amp_firing": firing,
            }
        )
    finally:
        # Runs for a PASS, a failed check and any exception alike; the details
        # file is written here so an interrupted run still leaves evidence, and
        # the original exception (if any) propagates after it.
        if injected:
            cleanup = boot015_cleanup(
                fixture,
                command_id=command_id,
                pods=pods,
                baseline_ids=baseline_ids,
            )
            result["cleanup"] = cleanup
            if not cleanup["passed"]:
                result["verdict"] = "FAIL"
        else:
            result["cleanup"] = {"injected": False}
        result["limitations"] = [
            "The current implementation validates owner compatibility at runtime "
            "readiness; registration-time owner validation remains a known gap.",
            "AMP alert resolution after cleanup is observed, not required: the "
            "gauge may hold its last value for one scrape interval.",
        ]
        write_json_atomic(case_dir / "boot015-details.json", result)
    return result
