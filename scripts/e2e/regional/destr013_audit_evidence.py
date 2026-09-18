"""Read-only deployment binding for the DESTR013 final invariant audit."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from scripts.e2e.regional.guardrail_audit_evidence import (
    REGISTRY_PROBE,
    complete_pod_population,
    registration_identity,
)
from scripts.e2e.regional.regional_commands import run_fixture_command
from scripts.e2e.regional.regional_live_fixture import component_python

ROOT = Path(__file__).resolve().parents[3]
yaml = importlib.import_module("yaml")
API_APP = "gpu-fault-api-ha"
EXECUTOR_APP = "gpu-fault-cluster-executor"
SYNTHETIC_ROUTE_ENV = "GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS"
ENVIRONMENT_FIELDS = {
    "allow_replace": "GPU_FAULT_ALLOW_HYPERPOD_REPLACE",
    "allow_reboot": "GPU_FAULT_ALLOW_HYPERPOD_REBOOT",
    "allow_automatic": "GPU_FAULT_ALLOW_WITH_AUTOMATIC_NODE_RECOVERY",
    "legacy_mutation": "GPU_FAULT_ALLOW_HYPERPOD_MUTATION",
}


class AuditError(RuntimeError):
    """A controlled diagnostic that contains no raw probe output."""


def run(
    command: list[str],
    *,
    timeout: int = 180,
    input_text: str | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_fixture_command(
        command, cwd=ROOT, timeout=timeout, input_text=input_text, check=check, env=env
    )


def kubectl(
    kubeconfig: Path,
    context: str,
    namespace: str,
    *arguments: str,
    input_text: str | None = None,
) -> str:
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
        ],
        input_text=input_text,
    ).stdout


def json_object(text: object, label: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise AuditError(f"{label} is not valid JSON")
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        raise AuditError(f"{label} is not valid JSON") from None
    if not isinstance(value, dict):
        raise AuditError(f"{label} is not a JSON object")
    return value


def arn_parts(value: Any, service: str) -> list[str]:
    if not isinstance(value, str):
        raise AuditError(f"target {service} ARN is missing")
    parts = value.split(":", 5)
    if (
        len(parts) != 6
        or parts[0] != "arn"
        or not parts[1]
        or parts[2] != service
        or re.fullmatch(r"\d{12}", parts[4]) is None
        or not parts[5].startswith("role/" if service == "iam" else "cluster/")
        or not parts[5].split("/", 1)[1]
    ):
        raise AuditError(f"target {service} ARN is malformed")
    return parts


def kubeconfig_cluster(path: Path, context: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = yaml.safe_load(raw)
        selected = context or value["current-context"]
        contexts = [row for row in value["contexts"] if row["name"] == selected]
        if len(contexts) != 1:
            raise ValueError
        name = contexts[0]["context"]["cluster"]
        clusters = [row["cluster"] for row in value["clusters"] if row["name"] == name]
        if len(clusters) != 1 or not isinstance(clusters[0], dict):
            raise ValueError
        cluster = clusters[0]
        if cluster.get("insecure-skip-tls-verify") not in (None, False):
            raise ValueError
        if not isinstance(cluster.get("server"), str) or not cluster[
            "server"
        ].startswith("https://"):
            raise ValueError
        return cluster, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        raise AuditError(
            "target kubeconfig context or verified TLS identity is invalid"
        ) from None


def eks_binding(
    kubeconfig: Path, context: str, namespace: str, region: str, eks_arn: str
) -> dict[str, Any]:
    parts = arn_parts(eks_arn, "eks")
    if parts[3] != region:
        raise AuditError("target EKS Region differs from the registry")
    value = json_object(
        run(
            [
                "aws",
                "eks",
                "describe-cluster",
                "--region",
                region,
                "--name",
                parts[5].split("/", 1)[1],
                "--output",
                "json",
            ]
        ).stdout,
        "EKS description",
    ).get("cluster")
    cluster, digest = kubeconfig_cluster(kubeconfig, context)
    try:
        if (
            not isinstance(value, dict)
            or value["arn"] != eks_arn
            or value["status"] != "ACTIVE"
        ):
            raise ValueError
        if cluster["server"] != value["endpoint"] or cluster.get("tls-server-name"):
            raise ValueError
        expected_ca = base64.b64decode(
            value["certificateAuthority"]["data"], validate=True
        )
        actual_ca = (
            base64.b64decode(cluster["certificate-authority-data"], validate=True)
            if cluster.get("certificate-authority-data")
            else (kubeconfig.parent / cluster["certificate-authority"]).read_bytes()
        )
        if not expected_ca or actual_ca != expected_ca:
            raise ValueError
    except (OSError, KeyError, TypeError, ValueError):
        raise AuditError(
            "target GPU kubeconfig does not match the registered EKS endpoint/CA"
        ) from None
    anchor = (
        json_object(
            kubectl(
                kubeconfig,
                context,
                namespace,
                "get",
                "namespace",
                "kube-system",
                "-o",
                "json",
            ),
            "GPU EKS namespace anchor",
        )
        .get("metadata", {})
        .get("uid")
    )
    if not isinstance(anchor, str) or not anchor:
        raise AuditError("target GPU EKS namespace UID is missing")
    return {
        "eks_cluster_arn": eks_arn,
        "kubeconfig_sha256": digest,
        "namespace_uid": anchor,
    }


def pod_population(
    kubeconfig: Path, context: str, namespace: str, app: str, container: str
) -> dict[str, Any]:
    deployment = json_object(
        kubectl(kubeconfig, context, namespace, "get", "deployment", app, "-o", "json"),
        f"{container} Deployment",
    )
    inventory = json_object(
        kubectl(
            kubeconfig,
            context,
            namespace,
            "get",
            "pod",
            "-l",
            f"app={app}",
            "-o",
            "json",
        ),
        f"{container} Pod inventory",
    )
    try:
        ready = complete_pod_population(deployment, inventory)
        template = deployment["spec"]["template"]["spec"]
        template_containers = [
            row for row in template["containers"] if row["name"] == container
        ]
        if len(template_containers) != 1:
            raise ValueError
        pods = []
        for record, item in zip(
            ready,
            sorted(inventory["items"], key=lambda row: row["metadata"]["name"]),
            strict=True,
        ):
            spec = item["spec"]
            selected = [row for row in spec["containers"] if row["name"] == container]
            statuses = [
                row
                for row in item["status"]["containerStatuses"]
                if row["name"] == container
            ]
            account = spec.get("serviceAccountName")
            if (
                len(selected) != 1
                or len(statuses) != 1
                or not isinstance(account, str)
                or not account
                or account != template.get("serviceAccountName")
                or selected[0].get("image") != template_containers[0].get("image")
                or not statuses[0].get("containerID")
                or type(statuses[0].get("restartCount")) is not int
            ):
                raise ValueError
            pods.append(
                {
                    **record,
                    "service_account": account,
                    "image": selected[0]["image"],
                    "container_id": statuses[0]["containerID"],
                    "restart_count": statuses[0]["restartCount"],
                }
            )
        return {
            "deployment_uid": deployment["metadata"]["uid"],
            "generation": deployment["metadata"]["generation"],
            "template_sha256": hashlib.sha256(
                json.dumps(deployment["spec"]["template"], sort_keys=True).encode()
            ).hexdigest(),
            "pods": pods,
        }
    except (RuntimeError, KeyError, TypeError, ValueError):
        raise AuditError(
            f"{container} requires a complete stable Ready Pod population"
        ) from None


def pod_json(
    kubeconfig: Path, context: str, namespace: str, pod: str, plane: str, script: str
) -> dict[str, Any]:
    container = "api" if plane == "cpu" else "executor"
    return json_object(
        kubectl(
            kubeconfig,
            context,
            namespace,
            "exec",
            "-i",
            pod,
            "-c",
            container,
            "--",
            component_python(plane),
            "-B",
            "-",
            input_text=script,
        ),
        f"{container} read-only probe",
    )


def environment_probe(plane: str) -> str:
    fields = (
        {
            "synthetic_route": SYNTHETIC_ROUTE_ENV,
            "executor_artifact": "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256",
            "executor_compatibility": "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST",
        }
        if plane == "cpu"
        else {
            **ENVIRONMENT_FIELDS,
            "cluster_id": "GPU_FAULT_CLUSTER_ID",
            "role_arn": "AWS_ROLE_ARN",
            "region": "AWS_REGION",
            "default_region": "AWS_DEFAULT_REGION",
            "executor_artifact": "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256",
            "executor_compatibility": "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST",
        }
    )
    script = (
        "import json, os\nfrom gpu_fault import module_digest\n"
        f"fields = {fields!r}\n"
        "value = {key: os.environ.get(name) for key, name in fields.items()}\n"
        "value.update(pod=os.environ.get('HOSTNAME'), module_digest=module_digest())\n"
    )
    if plane == "gpu":
        script += (
            "import boto3\nfrom botocore.config import Config\n"
            "session = boto3.Session()\n"
            "credentials = session.get_credentials()\n"
            "value['credential_method'] = credentials.method if credentials else None\n"
            "caller = session.client('sts', config=Config(connect_timeout=10, "
            "read_timeout=10, retries={'max_attempts': 0})).get_caller_identity()\n"
            "value.update(caller_arn=caller.get('Arn'), caller_account=caller.get('Account'))\n"
        )
    return script + "print(json.dumps(value))\n"


def executor_identity(
    kubeconfig: Path,
    context: str,
    namespace: str,
    service_account: str,
    observed: dict[str, Any],
    *,
    cluster_id: str,
    region: str,
    role_arn: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    account = json_object(
        kubectl(
            kubeconfig,
            context,
            namespace,
            "get",
            "serviceaccount",
            service_account,
            "-o",
            "json",
        ),
        "executor ServiceAccount",
    )["metadata"]
    role = arn_parts(role_arn, "iam")
    if (
        not account.get("uid")
        or account.get("deletionTimestamp")
        or account.get("annotations", {}).get("eks.amazonaws.com/role-arn") != role_arn
        or observed.get("role_arn") != role_arn
        or observed.get("cluster_id") != cluster_id
        or observed.get("region") != region
        or observed.get("default_region") not in (None, region)
        or observed.get("credential_method") != "assume-role-with-web-identity"
        or observed.get("caller_account") != role[4]
        or not isinstance(observed.get("caller_arn"), str)
        or not observed["caller_arn"].startswith(
            f"arn:{role[1]}:sts::{role[4]}:assumed-role/{role_arn.rsplit('/', 1)[-1]}/"
        )
    ):
        raise AuditError(
            "target executor cluster/Region/IRSA differs from the registered target"
        )
    switches: dict[str, Any] = {}
    for key in ENVIRONMENT_FIELDS:
        raw = observed.get(key)
        if raw is not None and (
            not isinstance(raw, str) or raw.lower() not in {"", "true", "false", "none"}
        ):
            raise AuditError(f"executor environment has an invalid {key} switch")
        switches[key] = raw
    return {"uid": account["uid"], "role_arn": role_arn}, switches


def target_evidence(
    *,
    gpu_kubeconfig: Path,
    gpu_context: str,
    cpu_kubeconfig: Path,
    cpu_context: str,
    namespace: str,
    region: str,
    cluster: str,
    role_arn: str,
    recovery: dict[str, Any],
) -> dict[str, Any]:
    """Bind the supplied targets to the durable registration and deployed pins."""
    try:
        _, cpu_digest = kubeconfig_cluster(cpu_kubeconfig, cpu_context)
        api = pod_population(cpu_kubeconfig, cpu_context, namespace, API_APP, "api")
        registry = pod_json(
            cpu_kubeconfig,
            cpu_context,
            namespace,
            api["pods"][0]["name"],
            "cpu",
            REGISTRY_PROBE,
        )
        try:
            registration = registration_identity(registry, cluster, "", region)
        except RuntimeError:
            raise AuditError(
                "target HyperPod cluster/Region does not match the durable registration"
            ) from None
        if type(registry.get("generation")) is not int or registry["generation"] < 1:
            raise AuditError("target registry generation is missing")
        hp = arn_parts(recovery.get("cluster_arn"), "sagemaker")
        eks = arn_parts(registration["eks_cluster_arn"], "eks")
        role = arn_parts(role_arn, "iam")
        if (
            recovery.get("cluster_name") != cluster
            or hp[1] != eks[1]
            or hp[1] != role[1]
            or hp[3] != region
            or eks[3] != region
            or role[3]
            or hp[4] != eks[4]
            or hp[4] != role[4]
            or recovery.get("orchestrator")
            != {"Eks": {"ClusterArn": registration["eks_cluster_arn"]}}
        ):
            raise AuditError("target HyperPod/EKS/IRSA identity is inconsistent")
        binding = eks_binding(
            gpu_kubeconfig,
            gpu_context,
            namespace,
            region,
            registration["eks_cluster_arn"],
        )
        document = json_object(
            kubectl(
                cpu_kubeconfig,
                cpu_context,
                namespace,
                "get",
                "configmap",
                "gpu-fault-regional-release-state",
                "-o",
                "json",
            ),
            "release state ConfigMap",
        )
        state = json_object(document["data"]["state.json"], "release state")
        if (
            not isinstance(state.get("release_id"), str)
            or not state["release_id"].strip()
            or state.get("phase") != "complete"
            or state.get("transaction_committed") is not True
            or not isinstance(state.get("cluster_ids"), list)
            or registration["cluster_id"] not in state.get("cluster_ids", [])
        ):
            raise AuditError(
                "target release is not complete, committed and cluster-bound"
            )
        digests = state["component_digests"]
        for pin in (
            state["wheel_sha256"],
            state["executor_wheel_sha256"],
            digests["control_plane"],
            digests["executor"],
        ):
            if not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
                raise AuditError(
                    "target release component pins are missing or malformed"
                )
        executor = pod_population(
            gpu_kubeconfig, gpu_context, namespace, EXECUTOR_APP, "executor"
        )
        accounts: dict[str, Any] = {}
        environments: dict[str, list[dict[str, Any]]] = {"cpu": [], "gpu": []}
        for plane, population, config, context, image in (
            ("cpu", api, cpu_kubeconfig, cpu_context, state["runtime_image"]),
            ("gpu", executor, gpu_kubeconfig, gpu_context, state["executor_image"]),
        ):
            for pod in population["pods"]:
                observed = pod_json(
                    config,
                    context,
                    namespace,
                    pod["name"],
                    plane,
                    environment_probe(plane),
                )
                fields = (
                    {"synthetic_route"} if plane == "cpu" else set(ENVIRONMENT_FIELDS)
                )
                if not fields <= observed.keys():
                    raise AuditError(
                        f"target {plane} replica environment evidence is incomplete"
                    )
                if (
                    not isinstance(image, str)
                    or not image
                    or pod["image"] != image
                    or observed.get("pod") != pod["name"]
                    or observed.get("module_digest")
                    != digests["control_plane" if plane == "cpu" else "executor"]
                    or observed.get("executor_artifact")
                    != state["executor_wheel_sha256"]
                    or observed.get("executor_compatibility") != digests["executor"]
                ):
                    raise AuditError(
                        f"target {plane} replica does not match the deployed release pins"
                    )
                safe: dict[str, Any] = {"pod": pod["name"], "pod_uid": pod["uid"]}
                if plane == "cpu":
                    # Record presence only: even an invalid switch must not echo its value.
                    safe["synthetic_route"] = (
                        "set" if observed.get("synthetic_route") is not None else None
                    )
                else:
                    account, switches = executor_identity(
                        config,
                        context,
                        namespace,
                        pod["service_account"],
                        observed,
                        cluster_id=registration["cluster_id"],
                        region=region,
                        role_arn=role_arn,
                    )
                    accounts[pod["service_account"]] = account
                    safe.update(switches)
                environments[plane].append(safe)
            app = API_APP if plane == "cpu" else EXECUTOR_APP
            container = "api" if plane == "cpu" else "executor"
            if pod_population(config, context, namespace, app, container) != population:
                raise AuditError(
                    f"target {plane} replica identity drifted during probes"
                )
        return {
            "release_id": state["release_id"],
            "cluster_id": registration["cluster_id"],
            "registration": registration,
            "registry_generation": registry["generation"],
            "release_pins": {
                key: state[key]
                for key in (
                    "release_id",
                    "wheel_sha256",
                    "executor_wheel_sha256",
                    "runtime_image",
                    "executor_image",
                )
            }
            | {
                "component_digests": {
                    key: digests[key] for key in ("control_plane", "executor")
                }
            },
            "gpu_eks_binding": binding,
            "cpu_kubeconfig_sha256": cpu_digest,
            "api_population": api,
            "executor_population": executor,
            "executor_service_accounts": accounts,
            "executor_environment": environments["gpu"],
            "api_environment": environments["cpu"],
        }
    except (KeyError, TypeError, ValueError, AttributeError):
        raise AuditError(
            "target deployment identity evidence is incomplete or malformed"
        ) from None
