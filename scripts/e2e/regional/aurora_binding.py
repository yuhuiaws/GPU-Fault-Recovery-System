"""Read-only CPU/Store/RDS identity proof for the three Aurora HA cases."""

from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gpu_fault.admin.bootstrap_common import Arn, ClusterIdentity
from gpu_fault.admin.bootstrap_services import (
    irsa_trust_document,
    pod_identity_trust,
)
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault_release.regional_deployment_inventory import CPU_RUNTIME_DEPLOYMENTS
from gpu_fault_release.regional_release_config import validate_eks_arn
from gpu_fault_release.regional_release_runtime_identity import CONTROL_PLANE_PATH
from gpu_fault_release.regional_release_store_probe import CA_PATH, connection_arguments

from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import component_python

DSN_FILE = "/etc/gpu-fault/aurora/postgres-url"
MASTER_FILE = "/etc/gpu-fault/aurora/master-secret-arn"
REFRESH_POLICY = "ReadAuroraManagedMasterSecret"


class BindingError(RegionalFixtureError):
    """Only fixed, credential-free reasons may cross this boundary."""


def require(value: object, reason: str) -> None:
    if not value:
        raise BindingError("Aurora binding: " + reason)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def identity(document: dict[str, Any], kind: str, name: str, namespace: str) -> str:
    metadata = document.get("metadata") or {}
    require(
        document.get("kind") == kind
        and metadata.get("name") == name
        and metadata.get("namespace", "") == namespace
        and isinstance(metadata.get("uid"), str)
        and metadata["uid"]
        and not metadata.get("deletionTimestamp"),
        "resource identity is missing or differs",
    )
    return str(metadata["uid"])


def dsn_arguments(dsn: str, database: dict[str, Any]) -> dict[str, Any]:
    from psycopg.conninfo import conninfo_to_dict

    parsed = conninfo_to_dict(dsn)
    allowed = {
        "host",
        "port",
        "dbname",
        "user",
        "password",
        "sslmode",
        "sslrootcert",
        "connect_timeout",
        "application_name",
    }
    if set(parsed) - allowed:
        raise ValueError("unsupported Store connection override")
    return dict(connection_arguments(dsn, database))


POD_PROBE = r"""
import hashlib
import json
import logging
import os
import pathlib
import sys

logging.disable(logging.CRITICAL)
try:
    import psycopg
    from gpu_fault import module_digest
    from gpu_fault.store.postgres.pool import StoreCredentials

    expected = json.loads(sys.argv[1])
    path = os.environ.get("GPU_FAULT_STORE_URL_FILE")
    if path != "/etc/gpu-fault/aurora/postgres-url":
        raise ValueError("credential file source differs")
    if any(value for key, value in os.environ.items() if key.startswith("PG")):
        raise ValueError("ambient libpq configuration is unsupported")
    text = pathlib.Path(path).read_text().strip()
    if not text:
        raise ValueError("projected credential is unavailable")
    credentials = StoreCredentials(os.environ["GPU_FAULT_STORE_URL"], path=path)
    current = credentials.conninfo()
    if credentials.source != "file" or current != text:
        raise ValueError("projected credential changed during proof")
    dsn_arguments(os.environ["GPU_FAULT_STORE_URL"], expected["database"])
    arguments = dsn_arguments(current, expected["database"])
    if pathlib.Path("/etc/gpu-fault/aurora/master-secret-arn").read_text().strip() != expected["database"]["master_secret_arn"]:
        raise ValueError("projected master Secret reference differs")
    if module_digest() != expected["module_digest"]:
        raise ValueError("CPU release module digest differs")
    actual_digest = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    if expected["authenticate"]:
        if actual_digest != expected["credential_sha256"]:
            raise ValueError("projected credential is not current")
        with psycopg.connect(**arguments) as connection:
            row = connection.execute(
                "SELECT current_database(), current_user, pg_is_in_recovery()"
            ).fetchone()
            if row != (
                expected["database"]["database"],
                expected["database"]["username"], False,
            ):
                raise ValueError("SQL target is not the bound writer")
    print(json.dumps({"verified": True, "credential_sha256": actual_digest}))
except Exception:
    print(json.dumps({"verified": False}))
"""


def pod_probe_source() -> str:
    # These pure release checks also run in CPU images, which exclude deploy-host.
    return (
        "from __future__ import annotations\n"
        f"CA_PATH = {CA_PATH!r}\n"
        + inspect.getsource(connection_arguments)
        + "\n"
        + inspect.getsource(dsn_arguments)
        + "\n"
        + POD_PROBE
    )


def credential_mount(spec: dict[str, Any], secret_name: str) -> dict[str, Any]:
    containers = spec.get("containers") or []
    require(len(containers) == 1, "CPU container selection is ambiguous")
    container = containers[0]
    references = [
        entry
        for entry in container.get("env", [])
        if entry.get("name") == "GPU_FAULT_STORE_URL"
    ]
    require(
        len(references) == 1
        and references[0].get("valueFrom")
        == {"secretKeyRef": {"name": secret_name, "key": "postgres-url"}}
        and "value" not in references[0],
        "CPU Store credential must use the selected Secret key",
    )
    volumes = [
        volume
        for volume in spec.get("volumes", [])
        if volume.get("name") == "aurora-credentials"
    ]
    mounts = [
        mount
        for mount in container.get("volumeMounts", [])
        if mount.get("name") == "aurora-credentials"
    ]
    require(
        len(volumes) == len(mounts) == 1
        and volumes[0].get("secret", {}).get("secretName") == secret_name
        and not volumes[0]["secret"].get("items")
        and not volumes[0]["secret"].get("optional")
        and mounts[0].get("mountPath") == "/etc/gpu-fault/aurora"
        and mounts[0].get("readOnly") is True
        and not mounts[0].get("subPath")
        and not mounts[0].get("subPathExpr"),
        "CPU must project the whole selected Secret read-only without subPath",
    )
    require(
        not any(
            mount.get("name") != "aurora-credentials"
            and str(mount.get("mountPath", "")).startswith("/etc/gpu-fault/aurora")
            for mount in container.get("volumeMounts", [])
        ),
        "CPU credential mount is shadowed",
    )
    return dict(container)


def rds_identity(
    cluster: dict[str, Any], cpu_arn: Arn, cluster_id: str, *, refreshing: bool
) -> dict[str, Any]:
    arn = f"arn:{cpu_arn.partition}:rds:{cpu_arn.region}:{cpu_arn.account}:cluster:{cluster_id}"
    master = cluster.get("MasterUserSecret") or {}
    secret_arn = Arn.parse(str(master.get("SecretArn") or ""))
    required = ("DbClusterResourceId", "Endpoint", "DatabaseName", "MasterUsername")
    require(
        cluster.get("DBClusterIdentifier") == cluster_id
        and cluster.get("DBClusterArn") == arn
        and cluster.get("Engine") == "aurora-postgresql"
        and cluster.get("Port") == 5432
        and all(isinstance(cluster.get(key), str) and cluster[key] for key in required)
        and not cluster.get("PendingModifiedValues")
        and cluster.get("Status")
        in (
            {"available", "modifying", "resetting-master-credentials"}
            if refreshing
            else {"available"}
        )
        and master.get("SecretStatus")
        in ({"active", "rotating"} if refreshing else {"active"})
        and secret_arn.service == "secretsmanager"
        and secret_arn.partition == cpu_arn.partition
        and secret_arn.region == cpu_arn.region
        and secret_arn.account == cpu_arn.account
        and secret_arn.resource.startswith("secret:"),
        "RDS incarnation, managed Secret or readiness differs",
    )
    return {
        "cluster_arn": arn,
        "cluster_resource_id": cluster["DbClusterResourceId"],
        "master_secret_arn": master["SecretArn"],
        "kms_key_arn": master.get("KmsKeyId"),
        "endpoint": cluster["Endpoint"],
        "port": cluster["Port"],
        "database": cluster["DatabaseName"],
        "username": cluster["MasterUsername"],
    }


def release_identity(document: dict[str, Any], namespace: str) -> dict[str, Any]:
    uid = identity(document, "ConfigMap", "gpu-fault-regional-release-state", namespace)
    state = json.loads(document["data"]["state.json"])
    require(
        state.get("phase") == "complete"
        and state.get("transaction_committed") is True
        and isinstance(state.get("release_id"), str)
        and state["release_id"]
        and re.fullmatch(r".+@sha256:[0-9a-f]{64}", state.get("runtime_image", ""))
        and re.fullmatch(
            r"[0-9a-f]{64}", state.get("component_digests", {}).get("control_plane", "")
        )
        and state.get("wheel_config_map"),
        "deployed release is incomplete or unpinned",
    )
    return {
        "uid": uid,
        "state_sha256": digest(state),
        "release_id": state["release_id"],
        "runtime_image": state["runtime_image"],
        "module_digest": state["component_digests"]["control_plane"],
        "wheel_config_map": state["wheel_config_map"],
    }


@dataclass
class AuroraBinding:
    control: Callable[..., str]
    aws: Callable[..., dict[str, Any]]
    region: str
    cluster_id: str
    namespace: str
    secret_name: str = "gpu-fault-aurora"
    cronjob_name: str | None = None

    def get(self, kind: str, name: str) -> dict[str, Any]:
        value = json.loads(self.control("get", kind, name, "-o", "json", timeout=60))
        require(isinstance(value, dict), "resource read is not an object")
        return dict(value)

    def cpu_identity(self) -> tuple[dict[str, Any], Arn, dict[str, Any]]:
        config = json.loads(self.control("config", "view", "--minify", "-o", "json"))
        require(
            len(config.get("contexts", [])) == len(config.get("clusters", [])) == 1,
            "CPU context is ambiguous",
        )
        arn_text = validate_eks_arn(
            config["contexts"][0]["context"]["cluster"],
            field="CPU cluster",
            expected_region=self.region,
        )
        arn = Arn.parse(arn_text)
        configured = config["clusters"][0]
        cluster = self.aws("eks", "describe-cluster", "--name", arn.resource_name)[
            "cluster"
        ]
        require(
            configured.get("name") == arn_text
            and cluster.get("arn") == arn_text
            and cluster.get("status") == "ACTIVE"
            and cluster.get("createdAt")
            and configured["cluster"].get("server") == cluster.get("endpoint")
            and str(cluster.get("endpoint", "")).startswith("https://")
            and configured["cluster"].get("insecure-skip-tls-verify") is not True,
            "CPU context does not identify the described EKS cluster",
        )
        namespace_uid = identity(
            self.get("namespace", self.namespace), "Namespace", self.namespace, ""
        )
        return (
            {
                "cluster_arn": arn_text,
                "created_at": cluster["createdAt"],
                "endpoint_sha256": digest(cluster["endpoint"]),
                "namespace": self.namespace,
                "namespace_uid": namespace_uid,
            },
            arn,
            cluster,
        )

    def consumer_identity(
        self,
        release: dict[str, Any],
        database: dict[str, Any],
        credential_sha256: str,
        *,
        refreshing: bool,
    ) -> dict[str, Any]:
        result = {}
        for name in CPU_RUNTIME_DEPLOYMENTS:
            deployment = self.get("deployment", name)
            uid = identity(deployment, "Deployment", name, self.namespace)
            spec, status = deployment["spec"], deployment.get("status", {})
            replicas = spec.get("replicas")
            require(
                type(replicas) is int
                and replicas >= 0
                and (name == "gpu-fault-telemetry-spool-worker" or replicas > 0)
                and status.get("observedGeneration")
                == deployment["metadata"].get("generation")
                and type(status.get("updatedReplicas", 0)) is int
                and status.get("updatedReplicas", 0) == replicas
                and all(
                    type(status.get(key, 0)) is int
                    and (
                        0 <= status.get(key, 0) <= replicas
                        if refreshing
                        else status.get(key, 0) == replicas
                    )
                    for key in ("readyReplicas", "availableReplicas")
                ),
                "CPU Deployment has not converged",
            )
            template = spec["template"]
            container = credential_mount(template["spec"], self.secret_name)
            require(
                container.get("image") == release["runtime_image"],
                "CPU image differs from release",
            )
            pods = json.loads(
                self.control("get", "pods", "-l", f"app={name}", "-o", "json")
            )["items"]
            require(len(pods) == replicas, "CPU Pod inventory is incomplete")
            pod_ids = {}
            for pod in pods:
                pod_name = pod["metadata"]["name"]
                pod_uid = identity(pod, "Pod", pod_name, self.namespace)
                actual = credential_mount(pod["spec"], self.secret_name)
                require(
                    actual.get("image") == release["runtime_image"]
                    and pod.get("status", {}).get("phase") == "Running"
                    and any(
                        c.get("type") == "Ready"
                        and c.get("status")
                        in ({"True", "False"} if refreshing else {"True"})
                        for c in pod.get("status", {}).get("conditions", [])
                    ),
                    "CPU Pod release or readiness differs",
                )
                payload = {
                    "database": database,
                    "module_digest": release["module_digest"],
                    "credential_sha256": credential_sha256,
                    "authenticate": not refreshing,
                }
                report = json.loads(
                    self.control(
                        "exec",
                        "-i",
                        pod_name,
                        "-c",
                        actual["name"],
                        "--",
                        component_python("cpu"),
                        "-",
                        json.dumps(payload),
                        stdin=pod_probe_source().encode(),
                        timeout=60,
                    )
                )
                require(
                    set(report) == {"verified", "credential_sha256"}
                    and report.get("verified") is True
                    and re.fullmatch(
                        r"[0-9a-f]{64}", report.get("credential_sha256", "")
                    )
                    and (
                        refreshing or report["credential_sha256"] == credential_sha256
                    ),
                    "CPU projected Store proof failed",
                )
                require(
                    identity(self.get("pod", pod_name), "Pod", pod_name, self.namespace)
                    == pod_uid,
                    "CPU Pod changed during Store proof",
                )
                pod_ids[pod_name] = pod_uid
            result[name] = {
                "uid": uid,
                "generation": deployment["metadata"].get("generation"),
                "template_sha256": digest(template),
                "pods": pod_ids,
            }
        return result

    def read(
        self,
        expected: dict[str, Any] | None = None,
        *,
        refreshing: bool = False,
    ) -> dict[str, Any]:
        try:
            proof, _cronjob = self.capture(refreshing=refreshing)
            if expected is not None:
                require(
                    proof["identity"] == expected.get("identity"),
                    "identity changed since approval",
                )
                require(
                    proof["writer"] == expected.get("writer"),
                    "writer changed since approval",
                )
                if not refreshing:
                    require(
                        proof["credential_sha256"] == expected.get("credential_sha256"),
                        "credential changed since approval",
                    )
            return proof
        except ProcessSupervisionLost:
            raise ProcessSupervisionLost(
                "Aurora binding supervision was lost"
            ) from None
        except BindingError:
            raise
        except Exception:
            raise BindingError(
                "Aurora binding: read failed or evidence is malformed"
            ) from None

    def capture(
        self, *, refreshing: bool
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        cpu, arn, eks = self.cpu_identity()
        release = release_identity(
            self.get("configmap", "gpu-fault-regional-release-state"), self.namespace
        )
        clusters = self.aws(
            "rds", "describe-db-clusters", "--db-cluster-identifier", self.cluster_id
        )["DBClusters"]
        require(len(clusters) == 1, "RDS lookup is ambiguous")
        database = rds_identity(
            clusters[0], arn, self.cluster_id, refreshing=refreshing
        )
        members = clusters[0].get("DBClusterMembers") or []
        writers = [
            m["DBInstanceIdentifier"]
            for m in members
            if m.get("IsClusterWriter") is True
        ]
        require(len(writers) == 1, "RDS writer identity is ambiguous")
        master = self.aws(
            "secretsmanager",
            "describe-secret",
            "--secret-id",
            database["master_secret_arn"],
        )
        require(
            master.get("ARN") == database["master_secret_arn"]
            and master.get("OwningService") == "rds"
            and not master.get("DeletedDate"),
            "managed master Secret ownership differs",
        )
        secret = self.get("secret", self.secret_name)
        secret_uid = identity(secret, "Secret", self.secret_name, self.namespace)
        data = secret.get("data") or {}
        require(
            base64.b64decode(data["master-secret-arn"], validate=True).decode()
            == database["master_secret_arn"],
            "CPU Secret master reference differs from RDS",
        )
        dsn = base64.b64decode(data["postgres-url"], validate=True)
        dsn_arguments(dsn.decode().strip(), database)
        credential_sha256 = hashlib.sha256(dsn).hexdigest()
        consumers = self.consumer_identity(
            release, database, credential_sha256, refreshing=refreshing
        )
        refresher, cronjob = (
            self.refresher_identity(release, database, arn, eks)
            if self.cronjob_name
            else (None, None)
        )
        current_secret = self.get("secret", self.secret_name)
        require(
            identity(current_secret, "Secret", self.secret_name, self.namespace)
            == secret_uid
            and all(
                current_secret.get("data", {}).get(key) == data.get(key)
                for key in ("postgres-url", "master-secret-arn")
            ),
            "CPU Secret changed during proof",
        )
        require(
            release_identity(
                self.get("configmap", "gpu-fault-regional-release-state"),
                self.namespace,
            )
            == release,
            "release changed during proof",
        )
        current_clusters = self.aws(
            "rds", "describe-db-clusters", "--db-cluster-identifier", self.cluster_id
        )["DBClusters"]
        require(
            len(current_clusters) == 1
            and rds_identity(
                current_clusters[0], arn, self.cluster_id, refreshing=refreshing
            )
            == database
            and current_clusters[0].get("DBClusterMembers") == members,
            "RDS changed during proof",
        )
        return {
            "schema_version": 1,
            "identity": {
                "cpu": cpu,
                "release": release,
                "database": database,
                "secret": {"name": self.secret_name, "uid": secret_uid},
                "consumers": consumers,
                "refresher": refresher,
            },
            "credential_sha256": credential_sha256,
            "writer": writers[0],
        }, cronjob

    def refresher_identity(
        self,
        release: dict[str, Any],
        database: dict[str, Any],
        arn: Arn,
        eks: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        name = str(self.cronjob_name)
        cronjob = self.get("cronjob", name)
        uid = identity(cronjob, "CronJob", name, self.namespace)
        spec = cronjob["spec"]
        require(
            spec.get("suspend", False) is False
            and spec.get("concurrencyPolicy") == "Forbid"
            and not cronjob.get("status", {}).get("active"),
            "refresher is suspended, active or permits concurrent Jobs",
        )
        pod = spec["jobTemplate"]["spec"]["template"]["spec"]
        containers = pod.get("containers") or []
        require(
            len(containers) == 1 and not pod.get("initContainers"),
            "refresher program is ambiguous",
        )
        container = containers[0]
        require(
            container.get("image") == release["runtime_image"]
            and container.get("command") == ["gpu-fault-aurora-credential-refresh"]
            and not container.get("args")
            and not container.get("envFrom"),
            "refresher program differs from deployed release",
        )
        entries = container.get("env") or []
        env = {item["name"]: item.get("value") for item in entries}
        allowed_env = {
            "PATH",
            "GPU_FAULT_NAMESPACE",
            "GPU_FAULT_AURORA_SECRET",
            "GPU_FAULT_CONTROL_PLANE_DEPLOYMENT",
            "GPU_FAULT_AURORA_RESTART_DEPLOYMENTS",
            "GPU_FAULT_AURORA_REFRESH_MAX_ATTEMPTS",
            "GPU_FAULT_AURORA_MASTER_SECRET_ARN",
            "GPU_FAULT_RDS_CA_BUNDLE",
            "GPU_FAULT_AURORA_REFRESH_RESTART_DEPLOYMENTS",
            "GPU_FAULT_AWS_REGION",
        }
        require(
            len(env) == len(entries)
            and not set(env) - allowed_env
            and all(set(item) == {"name", "value"} for item in entries)
            and env.get("PATH") == CONTROL_PLANE_PATH
            and env.get("GPU_FAULT_NAMESPACE") == self.namespace
            and env.get("GPU_FAULT_AURORA_SECRET") == self.secret_name
            and env.get("GPU_FAULT_AURORA_MASTER_SECRET_ARN")
            == database["master_secret_arn"]
            and env.get("GPU_FAULT_AURORA_REFRESH_RESTART_DEPLOYMENTS") == "false"
            and env.get("GPU_FAULT_RDS_CA_BUNDLE") == CA_PATH
            and env.get("GPU_FAULT_AWS_REGION", self.region) == self.region
            and not any(key.startswith("AWS_") for key in env)
            and set(
                str(env.get("GPU_FAULT_AURORA_RESTART_DEPLOYMENTS", ""))
                .replace(" ", "")
                .split(",")
            )
            == set(CPU_RUNTIME_DEPLOYMENTS),
            "refresher target, credential source or no-rollout setting differs",
        )
        volumes = {v["name"]: v for v in pod.get("volumes", [])}
        require(
            volumes.get("artifact", {}).get("configMap", {}).get("name")
            == release["wheel_config_map"]
            and volumes.get("rds-ca-bundle", {}).get("configMap", {}).get("name")
            == "gpu-fault-rds-ca-bundle",
            "refresher release artifact or CA source differs",
        )
        require(
            pod.get("serviceAccountName") == name, "refresher ServiceAccount differs"
        )
        service_account = self.get("serviceaccount", name)
        sa_uid = identity(service_account, "ServiceAccount", name, self.namespace)
        role = self.get("role", name)
        role_uid = identity(role, "Role", name, self.namespace)
        binding = self.get("rolebinding", name)
        binding_uid = identity(binding, "RoleBinding", name, self.namespace)
        require(
            binding.get("roleRef")
            == {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": name}
            and binding.get("subjects")
            == [{"kind": "ServiceAccount", "name": name, "namespace": self.namespace}],
            "refresher RBAC subject differs",
        )
        expected_rules = [
            {
                "apiGroups": [""],
                "resources": ["secrets"],
                "resourceNames": [self.secret_name],
                "verbs": ["get", "patch"],
            },
            {
                "apiGroups": ["apps"],
                "resources": ["deployments"],
                "resourceNames": list(CPU_RUNTIME_DEPLOYMENTS),
                "verbs": ["get", "patch"],
            },
            {"apiGroups": [""], "resources": ["events"], "verbs": ["create", "patch"]},
        ]
        require(role.get("rules") == expected_rules, "refresher RBAC targets differ")
        iam = self.refresher_iam(service_account, arn, eks, database)
        return {
            "uid": uid,
            "spec_sha256": digest(spec),
            "service_account_uid": sa_uid,
            "role_uid": role_uid,
            "role_binding_uid": binding_uid,
            "iam": iam,
        }, cronjob

    def refresher_iam(
        self,
        service_account: dict[str, Any],
        arn: Arn,
        eks: dict[str, Any],
        database: dict[str, Any],
    ) -> dict[str, Any]:
        name = str(self.cronjob_name)
        irsa = (
            service_account["metadata"]
            .get("annotations", {})
            .get("eks.amazonaws.com/role-arn")
        )
        associations = self.aws(
            "eks",
            "list-pod-identity-associations",
            "--cluster-name",
            arn.resource_name,
            "--namespace",
            self.namespace,
            "--service-account",
            name,
        )["associations"]
        require(
            (bool(irsa) and not associations) or (not irsa and len(associations) == 1),
            "refresher AWS identity is missing or ambiguous",
        )
        association_id = None
        if irsa:
            role_arn = str(irsa)
            issuer = eks["identity"]["oidc"]["issuer"].removeprefix("https://")
            trust = irsa_trust_document(
                provider_arn=f"arn:{arn.partition}:iam::{arn.account}:oidc-provider/{issuer}",
                issuer=issuer,
                namespace=self.namespace,
                service_account=name,
            )
        else:
            association_id = associations[0]["associationId"]
            association = self.aws(
                "eks",
                "describe-pod-identity-association",
                "--cluster-name",
                arn.resource_name,
                "--association-id",
                association_id,
            )["association"]
            require(
                association.get("clusterName") == arn.resource_name
                and association.get("namespace") == self.namespace
                and association.get("serviceAccount") == name
                and association.get("associationId") == association_id,
                "refresher Pod Identity association differs",
            )
            role_arn = association["roleArn"]
            trust = pod_identity_trust(
                ClusterIdentity(
                    input_arn=eks["arn"],
                    role="cpu",
                    region=arn.region,
                    account_id=arn.account,
                    hyperpod_arn="",
                    hyperpod_name="",
                    eks_arn=eks["arn"],
                    eks_name=arn.resource_name,
                    vpc_id="",
                    subnet_ids=(),
                    node_recovery="None",
                    context="",
                )
            )
        parsed = Arn.parse(role_arn)
        require(
            parsed.partition == arn.partition
            and parsed.account == arn.account
            and parsed.service == "iam"
            and parsed.resource.startswith("role/"),
            "refresher IAM role belongs to another account",
        )
        role_name = role_arn.rsplit("/", 1)[-1]
        role = self.aws("iam", "get-role", "--role-name", role_name)["Role"]
        require(
            role.get("Arn") == role_arn
            and role.get("RoleId")
            and role.get("AssumeRolePolicyDocument") == trust,
            "refresher IAM trust differs",
        )
        policy = self.aws(
            "iam",
            "get-role-policy",
            "--role-name",
            role_name,
            "--policy-name",
            REFRESH_POLICY,
        )["PolicyDocument"]
        statements = [
            {
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue"],
                "Resource": database["master_secret_arn"],
            }
        ]
        if database["kms_key_arn"]:
            statements.append(
                {
                    "Effect": "Allow",
                    "Action": ["kms:Decrypt"],
                    "Resource": database["kms_key_arn"],
                }
            )
        require(
            policy == {"Version": "2012-10-17", "Statement": statements},
            "refresher IAM Secret target differs",
        )
        return {
            "role_arn": role_arn,
            "role_id": role["RoleId"],
            "association_id": association_id,
            "policy_sha256": digest(policy),
            "trust_sha256": digest(trust),
        }

    def refresh_job(self, name: str, expected: dict[str, Any]) -> dict[str, Any]:
        try:
            proof, cronjob = self.capture(refreshing=True)
            require(
                proof["identity"] == expected.get("identity")
                and proof["writer"] == expected.get("writer")
                and cronjob is not None,
                "refresh target changed since approval",
            )
            assert cronjob is not None, (
                "verified refresh proof must include its CronJob"
            )
            template = copy.deepcopy(cronjob["spec"]["jobTemplate"])
            metadata = template.setdefault("metadata", {})
            metadata.update(
                name=name,
                namespace=self.namespace,
                ownerReferences=[
                    {
                        "apiVersion": "batch/v1",
                        "kind": "CronJob",
                        "name": self.cronjob_name,
                        "uid": cronjob["metadata"]["uid"],
                        "controller": False,
                        "blockOwnerDeletion": False,
                    }
                ],
            )
            return {"apiVersion": "batch/v1", "kind": "Job", **template}
        except ProcessSupervisionLost:
            raise ProcessSupervisionLost(
                "Aurora binding supervision was lost"
            ) from None
        except BindingError:
            raise
        except Exception:
            raise BindingError("Aurora binding: refresh proof failed") from None


def regional_binding(regional: Any, cluster_id: str) -> AuroraBinding:
    def control(*args: str, stdin: bytes | None = None, **kwargs: Any) -> str:
        return str(
            regional.kubectl(
                "cpu",
                *args,
                input_text=stdin.decode() if stdin is not None else None,
                **kwargs,
            )
        )

    def aws(service: str, *args: str) -> dict[str, Any]:
        from scripts.e2e.regional.regional_commands import run_fixture_command

        return dict(
            json.loads(
                run_fixture_command(
                    [
                        "aws",
                        service,
                        *args,
                        "--region",
                        regional.settings.region,
                        "--output",
                        "json",
                    ],
                    timeout=180,
                ).stdout
            )
        )

    return AuroraBinding(
        control, aws, regional.settings.region, cluster_id, regional.settings.namespace
    )
