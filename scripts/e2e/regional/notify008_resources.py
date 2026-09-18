"""NOTIFY008-specific manifests and admitted-Pod safety checks."""

from __future__ import annotations

import re
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from typing import Any

from scripts.e2e.regional.probes.notify008_protocol import (
    CASE_ID,
    ProbeError,
    Target,
    digest,
)

RUNTIME_PYTHON = "/opt/gpu-fault/control-plane/bin/python"
GATE = "gpu-fault.io/notify008-admission"
OWNER_LABEL = "gpu-fault.io/acceptance-run"
CASE_LABEL = "gpu-fault.io/acceptance-case"
NAME = "notify008"

POSTGRES_START = """set -eu
control=$1
remaining=$NOTIFY008_SECONDS
arm="$control/arm-$NOTIFY008_POD_UID"
stop="$control/stop-$NOTIFY008_POD_UID"
trap 'exit 0' TERM INT
while [ ! -f "$arm" ]; do
  [ ! -f "$stop" ] || exit 0
  [ "$remaining" -gt 0 ] || exit 75
  sleep 1
  remaining=$((remaining - 1))
done
[ ! -f "$stop" ] || exit 0
[ "$remaining" -gt 0 ] || exit 75
[ "$(cat "$arm")" = "$NOTIFY008_RUN_ID" ] || exit 76
# The image initializer clears PGHOST and uses its private default socket.
docker-entrypoint.sh postgres -c listen_addresses= -c unix_socket_directories=/socket,/var/run/postgresql -c unix_socket_permissions=0700 &
database=$!
while kill -0 "$database" 2>/dev/null; do
  [ ! -f "$stop" ] || exit 0
  [ "$remaining" -gt 0 ] || exit 75
  sleep 1
  remaining=$((remaining - 1))
done
wait "$database"
"""


def metadata(target: Target, *, namespaced: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": NAME,
        "labels": {OWNER_LABEL: target.run_id, CASE_LABEL: CASE_ID},
    }
    if namespaced:
        result["namespace"] = target.run_id
    else:
        result["name"] = target.run_id
        result["labels"].update(
            {
                "pod-security.kubernetes.io/enforce": "restricted",
                "pod-security.kubernetes.io/enforce-version": "v1.30",
            }
        )
    return result


def environment(target: Target) -> list[dict[str, Any]]:
    values = {
        "HOME": "/work",
        "PYTHONPATH": "/case",
        "PYTHONDONTWRITEBYTECODE": "1",
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "AWS_EC2_METADATA_DISABLED": "true",
        "KUBECONFIG": "/dev/null",
        "GPU_FAULT_ALLOW_HYPERPOD_REPLACE": "false",
        "NOTIFY008_RUN_ID": target.run_id,
        "NOTIFY008_SECONDS": str(target.seconds),
    }
    return [
        *({"name": key, "value": value} for key, value in values.items()),
        {
            "name": "NOTIFY008_POD_UID",
            "valueFrom": {
                "fieldRef": {"apiVersion": "v1", "fieldPath": "metadata.uid"}
            },
        },
    ]


def pod_spec(target: Target) -> dict[str, Any]:
    security = {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "runAsNonRoot": True,
        "runAsUser": 999,
        "runAsGroup": 999,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    common = {
        "imagePullPolicy": "IfNotPresent",
        "securityContext": security,
        "resources": {
            "requests": {"cpu": "100m", "memory": "256Mi"},
            "limits": {"cpu": "1", "memory": "512Mi"},
        },
    }
    return {
        "serviceAccountName": NAME,
        "automountServiceAccountToken": False,
        "enableServiceLinks": False,
        "hostNetwork": False,
        "hostPID": False,
        "hostIPC": False,
        "shareProcessNamespace": False,
        "restartPolicy": "Never",
        "preemptionPolicy": "Never",
        "priorityClassName": target.run_id,
        "priority": 0,
        "activeDeadlineSeconds": target.seconds,
        "terminationGracePeriodSeconds": 15,
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 999,
            "runAsGroup": 999,
            "fsGroup": 999,
        },
        "schedulingGates": [{"name": GATE}],
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchFields": [
                                {
                                    "key": "metadata.name",
                                    "operator": "In",
                                    "values": [target.node],
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "containers": [
            {
                **common,
                "name": "runtime",
                "image": target.runtime_image,
                "command": [RUNTIME_PYTHON],
                "args": ["-m", "scripts.e2e.regional.probes.notify008_probe", "idle"],
                "env": environment(target),
                "volumeMounts": [
                    {"name": "scripts", "mountPath": "/case", "readOnly": True},
                    {"name": "control", "mountPath": "/control"},
                    {"name": "socket", "mountPath": "/socket"},
                    {"name": "work", "mountPath": "/work"},
                ],
            },
            {
                **common,
                "name": "database",
                "image": target.postgres_image,
                "command": ["/bin/sh"],
                "args": ["/case/postgres-start", "/control"],
                "env": [
                    *environment(target),
                    {
                        "name": "PATH",
                        "value": "/usr/lib/postgresql/16/bin:/usr/local/bin:/usr/bin:/bin",
                    },
                    {"name": "PGDATA", "value": "/database/data"},
                    {"name": "POSTGRES_USER", "value": "notify008"},
                    {"name": "POSTGRES_DB", "value": "notify008"},
                    {"name": "POSTGRES_HOST_AUTH_METHOD", "value": "trust"},
                    {
                        "name": "POSTGRES_INITDB_ARGS",
                        "value": "--auth-host=reject --auth-local=trust",
                    },
                ],
                "volumeMounts": [
                    {"name": "scripts", "mountPath": "/case", "readOnly": True},
                    {"name": "control", "mountPath": "/control", "readOnly": True},
                    {"name": "socket", "mountPath": "/socket"},
                    {"name": "database", "mountPath": "/database"},
                    {"name": "work", "mountPath": "/work"},
                    {"name": "pg-run", "mountPath": "/var/run/postgresql"},
                ],
            },
        ],
        "volumes": [
            {"name": "scripts", "configMap": {"name": NAME, "defaultMode": 0o444}},
            *(
                {"name": name, "emptyDir": {"medium": "Memory", "sizeLimit": size}}
                for name, size in (
                    ("control", "1Mi"),
                    ("socket", "1Mi"),
                    ("work", "32Mi"),
                    ("database", "512Mi"),
                    ("pg-run", "1Mi"),
                )
            ),
        ],
    }


def manifests(
    target: Target, namespace_uid: str, sources: dict[str, str]
) -> dict[str, dict[str, Any]]:
    if not isinstance(namespace_uid, str) or not namespace_uid or not sources:
        raise ProbeError("namespace identity and complete probe bundle are required")
    data = {**sources, "postgres-start": POSTGRES_START}
    config = {
        "target": asdict(target),
        "namespace_uid": namespace_uid,
        "source_sha256": digest(data),
    }
    import json

    data["config.json"] = json.dumps(config, sort_keys=True)
    spec = pod_spec(target)
    spec["volumes"][0]["configMap"]["items"] = [
        {
            "key": key,
            "path": f"scripts/e2e/regional/probes/{key}"
            if key.endswith(".py")
            else key,
        }
        for key in data
    ]
    labels = metadata(target)["labels"]
    return {
        "priorityclass": {
            "apiVersion": "scheduling.k8s.io/v1",
            "kind": "PriorityClass",
            "metadata": {
                "name": target.run_id,
                "labels": dict(labels),
                "ownerReferences": [
                    {
                        "apiVersion": "v1",
                        "kind": "Namespace",
                        "name": target.run_id,
                        "uid": namespace_uid,
                        "controller": False,
                        "blockOwnerDeletion": False,
                    }
                ],
            },
            "value": 0,
            "globalDefault": False,
            "preemptionPolicy": "Never",
        },
        "serviceaccount": {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": metadata(target),
            "automountServiceAccountToken": False,
        },
        "networkpolicy": {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": metadata(target),
            "spec": {
                "podSelector": {},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [],
                "egress": [],
            },
        },
        "configmap": {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": metadata(target),
            "immutable": True,
            "data": data,
        },
        "job": {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": metadata(target),
            "spec": {
                "suspend": True,
                "parallelism": 1,
                "completions": 1,
                "backoffLimit": 0,
                "activeDeadlineSeconds": target.seconds,
                "ttlSecondsAfterFinished": 120,
                "template": {"metadata": {"labels": labels}, "spec": spec},
            },
        },
    }


def positive_cpu(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        number = Decimal(value.removesuffix("m"))
        return number.is_finite() and number > 0
    except InvalidOperation:
        return False


def cpu_node_errors(node: dict[str, Any], target: Target) -> list[str]:
    metadata_value = node.get("metadata") or {}
    status = node.get("status") or {}
    labels = metadata_value.get("labels") or {}
    instance = labels.get("node.kubernetes.io/instance-type", "")
    cpu_family = isinstance(instance, str) and re.fullmatch(
        r"(?:c[5-8]|m[5-8]|r[5-8]|t[3-4])[a-z0-9]*\.[a-z0-9]+",
        instance.removeprefix("ml."),
    )
    quantities = status.get("capacity") or {}
    allocatable = status.get("allocatable") or {}
    accelerators = [
        value
        for inventory in (quantities, allocatable)
        for key, value in inventory.items()
        if "gpu" in key.lower()
        or key.startswith(("nvidia.com/", "amd.com/", "habana.ai/"))
    ]
    if (
        metadata_value.get("name") != target.node
        or metadata_value.get("uid") != target.node_uid
        or metadata_value.get("deletionTimestamp")
        or labels.get("kubernetes.io/os") != "linux"
        or not cpu_family
        or not positive_cpu(quantities.get("cpu"))
        or not positive_cpu(allocatable.get("cpu"))
        or any(value != "0" for value in accelerators)
        or not any(
            item.get("type") == "Ready" and item.get("status") == "True"
            for item in status.get("conditions", [])
        )
    ):
        return [
            "CPU node identity, instance class, readiness or accelerator inventory differs"
        ]
    return []


def admitted_pod_errors(
    pod: dict[str, Any],
    expected: dict[str, Any],
    target: Target,
    job_uid: str,
    *,
    gated: bool,
) -> list[str]:
    errors = []
    meta, spec = pod.get("metadata") or {}, pod.get("spec") or {}
    labels = meta.get("labels") or {}
    owners = meta.get("ownerReferences") or []
    if (
        meta.get("namespace") != target.run_id
        or not isinstance(meta.get("uid"), str)
        or not meta["uid"]
        or meta.get("deletionTimestamp")
        or labels.get(OWNER_LABEL) != target.run_id
        or labels.get(CASE_LABEL) != CASE_ID
        or len(owners) != 1
        or owners[0].get("kind") != "Job"
        or owners[0].get("uid") != job_uid
        or owners[0].get("controller") is not True
    ):
        errors.append("admitted Pod ownership differs")
    errors.extend(pod_spec_errors(spec, expected, gated=gated))
    errors.extend(running_pod_errors(pod, target, gated=gated))
    return errors


def pod_spec_errors(
    spec: dict[str, Any], expected: dict[str, Any], *, gated: bool
) -> list[str]:
    errors = []
    if gated and spec.get("nodeName"):
        errors.append("a prebound Pod bypasses the required scheduling gate")
    allowed_defaults = {
        "nodeName",
        "dnsPolicy",
        "schedulerName",
        "tolerations",
        "serviceAccount",
    }
    if set(spec) - set(expected) - allowed_defaults:
        errors.append("admitted Pod contains unknown fields")
    for key, wanted in {
        "dnsPolicy": "ClusterFirst",
        "schedulerName": "default-scheduler",
        "serviceAccount": NAME,
    }.items():
        if key in spec and spec[key] != wanted:
            errors.append(f"admitted Pod default differs at {key}")
    for toleration in spec.get("tolerations", []):
        if (
            not isinstance(toleration, dict)
            or set(toleration) != {"key", "operator", "effect", "tolerationSeconds"}
            or toleration.get("key")
            not in {"node.kubernetes.io/not-ready", "node.kubernetes.io/unreachable"}
            or toleration.get("operator") != "Exists"
            or toleration.get("effect") != "NoExecute"
            or type(toleration.get("tolerationSeconds")) is not int
            or not 0 <= toleration["tolerationSeconds"] <= 300
        ):
            errors.append("admitted Pod tolerations differ")
    omitted_defaults = {
        "hostNetwork": False,
        "hostPID": False,
        "hostIPC": False,
        "shareProcessNamespace": False,
        "priority": 0,
    }
    for name, value in expected.items():
        if name == "schedulingGates":
            if spec.get(name, []) != (value if gated else []):
                errors.append("admitted Pod scheduling gate differs")
        elif name != "containers" and digest(
            spec.get(name, omitted_defaults.get(name))
        ) != digest(value):
            errors.append(f"admitted Pod differs at {name}")
    containers = spec.get("containers") or []
    if not isinstance(containers, list) or len(containers) != 2:
        return [*errors, "admitted Pod container inventory differs"]
    for actual, wanted in zip(containers, expected["containers"], strict=True):
        defaults = {"terminationMessagePath", "terminationMessagePolicy"}
        if not isinstance(actual, dict) or set(actual) - set(wanted) - defaults:
            errors.append("admitted container contains unknown fields")
            continue
        for name, value in {
            "terminationMessagePath": "/dev/termination-log",
            "terminationMessagePolicy": "File",
        }.items():
            if name in actual and actual[name] != value:
                errors.append("admitted container termination fields differ")
        for name, value in wanted.items():
            if digest(actual.get(name)) != digest(value):
                errors.append(f"admitted {wanted['name']} differs at {name}")
    return errors


def running_pod_errors(
    pod: dict[str, Any], target: Target, *, gated: bool
) -> list[str]:
    errors = []
    spec = pod.get("spec") or {}
    if not gated:
        status = pod.get("status") or {}
        if not isinstance(status, dict):
            return ["admitted Pod status is malformed"]
        states = status.get("containerStatuses") or []
        if not isinstance(states, list) or any(
            not isinstance(item, dict) for item in states
        ):
            return ["admitted container readiness or restart history differs"]
        if (
            len(states) != 2
            or {item.get("name") for item in states} != {"runtime", "database"}
            or any(
                item.get("ready") is not True
                or type(item.get("restartCount")) is not int
                or item["restartCount"] != 0
                or not isinstance(item.get("state"), dict)
                or not isinstance(item["state"].get("running"), dict)
                for item in states
            )
        ):
            errors.append("admitted container readiness or restart history differs")
        actual_images = {
            item.get("name"): str(item.get("imageID", "")).removeprefix(
                "docker-pullable://"
            )
            for item in states
        }
        if spec.get("nodeName") != target.node or status.get("phase") != "Running":
            errors.append("admitted Pod is not running on the approved CPU node")
        for name, image in (
            ("runtime", target.runtime_image),
            ("database", target.postgres_image),
        ):
            if actual_images.get(name) != image:
                errors.append(f"admitted {name} resolved image identity differs")
    return errors
