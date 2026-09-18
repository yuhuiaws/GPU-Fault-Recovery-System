"""Bounded, namespace-owned resources; never inherit business runtime credentials."""

from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
from typing import Any

from scripts.e2e.regional.ha011_contracts import (
    BOUNDARY,
    LABEL,
    POD_NAME,
    PYTHON,
    Settings,
)

SOURCE_FILES = (
    "ha011_contracts.py",
    "probes/ha011_workers.py",
    "probes/ha011_processes.py",
    "probes/ha011_probe.py",
)


def source_bundle(root: Path) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for package in (
            "scripts",
            "scripts/e2e",
            "scripts/e2e/regional",
            "scripts/e2e/regional/probes",
        ):
            bundle.writestr(zipfile.ZipInfo(package + "/__init__.py"), "")
        for filename in SOURCE_FILES:
            bundle.writestr(
                zipfile.ZipInfo("scripts/e2e/regional/" + filename),
                (root / filename).read_bytes(),
            )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def namespace_manifest(settings: Settings) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": settings.isolated_namespace,
            "labels": {
                LABEL: settings.isolation_id,
                "pod-security.kubernetes.io/enforce": "restricted",
            },
        },
    }


def priority_class_manifest(
    settings: Settings, namespace_uid: str = "<created-namespace-uid>"
) -> dict[str, Any]:
    return {
        "apiVersion": "scheduling.k8s.io/v1",
        "kind": "PriorityClass",
        "metadata": {
            "name": settings.priority_class_name,
            "labels": {LABEL: settings.isolation_id},
            "ownerReferences": [
                {
                    "apiVersion": "v1",
                    "kind": "Namespace",
                    "name": settings.isolated_namespace,
                    "uid": namespace_uid,
                    "controller": False,
                    "blockOwnerDeletion": False,
                }
            ],
        },
        "value": 0,
        "globalDefault": False,
        "preemptionPolicy": "Never",
    }


def postgres_container(settings: Settings, security: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "postgres",
        "image": settings.postgres_image,
        "imagePullPolicy": "Always",
        "terminationMessagePath": "/dev/termination-log",
        "terminationMessagePolicy": "File",
        "command": [
            "/bin/sh",
            "-ec",
            'i=0; while [ "$i" -lt 300 ]; do '
            "if [ -f /arm/armed.json ]; then "
            'exec docker-entrypoint.sh "$@"; fi; '
            "i=$((i+1)); sleep 0.2; done; exit 70",
            "--",
        ],
        "args": [
            "postgres",
            "-c",
            "listen_addresses=127.0.0.1",
            "-c",
            "unix_socket_directories=/var/run/postgresql,/tmp",
            "-c",
            "max_connections=30",
        ],
        "env": [
            {"name": "POSTGRES_PASSWORD_FILE", "value": "/private/password"},
            {"name": "PGDATA", "value": "/var/lib/postgresql/data/isolated"},
        ],
        "envFrom": [],
        "securityContext": security,
        "resources": {
            "requests": {"cpu": "250m", "memory": "256Mi"},
            "limits": {"cpu": "250m", "memory": "512Mi"},
        },
        "volumeMounts": [
            {"name": "database", "mountPath": "/var/lib/postgresql/data"},
            {"name": "password", "mountPath": "/private", "readOnly": True},
            {"name": "postgres-tmp", "mountPath": "/tmp"},
            {"name": "postgres-socket", "mountPath": "/var/run/postgresql"},
            {"name": "arm", "mountPath": "/arm", "readOnly": True},
        ],
    }


def manifests(
    settings: Settings,
    *,
    runtime_image: str,
    bundle: str,
    password: str,
    cpu_node: dict[str, Any],
) -> list[dict[str, Any]]:
    metadata = {
        "namespace": settings.isolated_namespace,
        "labels": {LABEL: settings.isolation_id},
    }
    security = {
        "privileged": False,
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }
    common_env = [
        {"name": "HA011_ISOLATION_ID", "value": settings.isolation_id},
        {"name": "HA011_BOUNDARY", "value": BOUNDARY},
        {
            "name": "POD_UID",
            "valueFrom": {
                "fieldRef": {"apiVersion": "v1", "fieldPath": "metadata.uid"}
            },
        },
        {
            "name": "POD_NAMESPACE",
            "valueFrom": {
                "fieldRef": {"apiVersion": "v1", "fieldPath": "metadata.namespace"}
            },
        },
        {"name": "PYTHONPATH", "value": "/probe/probe.zip"},
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "AWS_CONFIG_FILE", "value": "/dev/null"},
        {"name": "AWS_SHARED_CREDENTIALS_FILE", "value": "/dev/null"},
        {"name": "AWS_EC2_METADATA_DISABLED", "value": "true"},
        {"name": "KUBECONFIG", "value": "/dev/null"},
        {"name": "HOME", "value": "/tmp"},
    ]
    return [
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {**metadata, "name": "loopback-only"},
            "spec": {
                "podSelector": {},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [],
                "egress": [],
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {**metadata, "name": "private-postgres"},
            "immutable": True,
            "type": "Opaque",
            "stringData": {"password": password},
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {**metadata, "name": "probe-source"},
            "immutable": True,
            "binaryData": {"probe.zip": bundle},
        },
        priority_class_manifest(settings),
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {**metadata, "name": POD_NAME},
            "spec": {
                "automountServiceAccountToken": False,
                "serviceAccountName": "default",
                "enableServiceLinks": False,
                "dnsPolicy": "Default",
                "schedulerName": "default-scheduler",
                "preemptionPolicy": "Never",
                "priorityClassName": settings.priority_class_name,
                "priority": 0,
                "restartPolicy": "Never",
                "activeDeadlineSeconds": 360,
                "terminationGracePeriodSeconds": 5,
                "hostNetwork": False,
                "hostPID": False,
                "hostIPC": False,
                "shareProcessNamespace": False,
                "nodeSelector": {
                    "kubernetes.io/os": "linux",
                    "kubernetes.io/hostname": cpu_node["hostname"],
                },
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchFields": [
                                        {
                                            "key": "metadata.name",
                                            "operator": "In",
                                            "values": [cpu_node["name"]],
                                        }
                                    ]
                                }
                            ],
                        }
                    }
                },
                "tolerations": [
                    {
                        "key": key,
                        "operator": "Exists",
                        "effect": "NoExecute",
                        "tolerationSeconds": 300,
                    }
                    for key in (
                        "node.kubernetes.io/not-ready",
                        "node.kubernetes.io/unreachable",
                    )
                ],
                "securityContext": {
                    "runAsNonRoot": True,
                    "runAsUser": 999,
                    "runAsGroup": 999,
                    "fsGroup": 999,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "initContainers": [],
                "imagePullSecrets": [],
                "schedulingGates": [{"name": "gpu-fault.nvidia.com/ha011-verified"}],
                "containers": [
                    {
                        "name": "runtime",
                        "image": runtime_image,
                        "imagePullPolicy": "Always",
                        "terminationMessagePath": "/dev/termination-log",
                        "terminationMessagePolicy": "File",
                        "command": [
                            PYTHON,
                            "-m",
                            "scripts.e2e.regional.probes.ha011_probe",
                        ],
                        "env": common_env,
                        "envFrom": [],
                        "securityContext": security,
                        "resources": {
                            "requests": {"cpu": "250m", "memory": "512Mi"},
                            "limits": {"cpu": "250m", "memory": "1Gi"},
                        },
                        "volumeMounts": [
                            {"name": "source", "mountPath": "/probe", "readOnly": True},
                            {
                                "name": "password",
                                "mountPath": "/private",
                                "readOnly": True,
                            },
                            {"name": "temporary", "mountPath": "/tmp"},
                            {"name": "arm", "mountPath": "/arm"},
                        ],
                    },
                    postgres_container(settings, security),
                ],
                "volumes": [
                    {
                        "name": "source",
                        "configMap": {"name": "probe-source", "defaultMode": 292},
                    },
                    {
                        "name": "password",
                        "secret": {
                            "secretName": "private-postgres",
                            "defaultMode": 288,
                        },
                    },
                    {"name": "database", "emptyDir": {"sizeLimit": "1Gi"}},
                    {"name": "temporary", "emptyDir": {"sizeLimit": "128Mi"}},
                    {"name": "postgres-tmp", "emptyDir": {"sizeLimit": "32Mi"}},
                    {"name": "postgres-socket", "emptyDir": {"sizeLimit": "1Mi"}},
                    {"name": "arm", "emptyDir": {"sizeLimit": "1Mi"}},
                ],
            },
        },
    ]
