from __future__ import annotations


def _job_environment(
    *,
    case: str,
    namespace: str,
    start_gate_configmap: str,
    clusters: int,
    nodes_per_cluster: int,
    xid_total: int,
    sxid_total: int,
    gpu_evidence_total: int,
    host_evidence_total: int,
    start_epoch: float,
    workers: int,
    duration_seconds: int,
    include_telemetry: bool,
    prewarm_connections: bool,
) -> list[dict]:
    return [
        {
            "name": "GPU_FAULT_CONTROL_PLANE_URL",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "gpu-fault-regional-connection",
                    "key": "control-plane-url",
                }
            },
        },
        {"name": "SSL_CERT_FILE", "value": "/tls/ca.crt"},
        {"name": "CLUSTERS_FILE", "value": "/tokens/clusters.json"},
        {
            "name": "TEMPLATES_FILE",
            "value": "/templates/templates.json.gz",
        },
        {"name": "CLUSTER_COUNT", "value": str(clusters)},
        {
            "name": "CLUSTER_OFFSET",
            "valueFrom": {
                "fieldRef": {
                    "fieldPath": (
                        "metadata.annotations['batch.kubernetes.io/"
                        "job-completion-index']"
                    )
                }
            },
        },
        {"name": "NODES_PER_CLUSTER", "value": str(nodes_per_cluster)},
        {"name": "XID_TOTAL", "value": str(xid_total)},
        {"name": "SXID_TOTAL", "value": str(sxid_total)},
        {"name": "GPU_EVIDENCE_TOTAL", "value": str(gpu_evidence_total)},
        {"name": "HOST_EVIDENCE_TOTAL", "value": str(host_evidence_total)},
        {"name": "START_EPOCH", "value": f"{start_epoch:.0f}"},
        {
            "name": "START_GATE_NAME",
            "value": start_gate_configmap if case == "burst" else "",
        },
        {"name": "START_GATE_NAMESPACE", "value": namespace},
        {"name": "WORKERS", "value": str(workers)},
        {
            "name": "INCLUDE_TELEMETRY",
            "value": str(include_telemetry).lower(),
        },
        {
            "name": "PREWARM_CONNECTIONS",
            "value": str(prewarm_connections).lower(),
        },
        {"name": "DURATION_SECONDS", "value": str(duration_seconds)},
        {"name": "PYTHONPATH", "value": "/scripts"},
    ]


def build_job(
    case: str,
    *,
    spec: dict,
    namespace: str,
    token_secret: str,
    script_configmap: str,
    template_configmap: str,
    start_gate_configmap: str,
    clusters: int,
    nodes_per_cluster: int,
    xid_total: int,
    sxid_total: int,
    gpu_evidence_total: int,
    host_evidence_total: int,
    start_epoch: float,
    workers: int,
    duration_seconds: int,
    cpu_request: str,
    cpu_limit: str,
    include_telemetry: bool,
    prewarm_connections: bool,
) -> dict:
    container = {
        "name": "load",
        "image": "public.ecr.aws/docker/library/python:3.12-slim",
        "command": ["python", f"/scripts/{spec['script']}"],
        "env": _job_environment(
            case=case,
            namespace=namespace,
            start_gate_configmap=start_gate_configmap,
            clusters=clusters,
            nodes_per_cluster=nodes_per_cluster,
            xid_total=xid_total,
            sxid_total=sxid_total,
            gpu_evidence_total=gpu_evidence_total,
            host_evidence_total=host_evidence_total,
            start_epoch=start_epoch,
            workers=workers,
            duration_seconds=duration_seconds,
            include_telemetry=include_telemetry,
            prewarm_connections=prewarm_connections,
        ),
        "resources": {
            "requests": {"cpu": cpu_request, "memory": "1Gi"},
            "limits": {"cpu": cpu_limit, "memory": "2Gi"},
        },
        "volumeMounts": [
            {"name": "script", "mountPath": "/scripts", "readOnly": True},
            {"name": "templates", "mountPath": "/templates", "readOnly": True},
            {"name": "tokens", "mountPath": "/tokens", "readOnly": True},
            {"name": "tls", "mountPath": "/tls", "readOnly": True},
        ],
    }
    if case == "burst":
        container["readinessProbe"] = {
            "exec": {
                "command": [
                    "/bin/sh",
                    "-c",
                    "test -f /tmp/gpu-fault-load-ready",
                ]
            },
            "periodSeconds": 1,
            "timeoutSeconds": 1,
            "failureThreshold": 600,
        }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": spec["job"], "namespace": namespace},
        "spec": {
            "completions": clusters,
            "parallelism": clusters,
            "completionMode": "Indexed",
            "backoffLimit": 0,
            "activeDeadlineSeconds": spec["deadline"],
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
                    "containers": [container],
                    "volumes": [
                        {
                            "name": "script",
                            "configMap": {"name": script_configmap},
                        },
                        {
                            "name": "templates",
                            "configMap": {"name": template_configmap},
                        },
                        {
                            "name": "tokens",
                            "secret": {"secretName": token_secret},
                        },
                        {
                            "name": "tls",
                            "secret": {
                                "secretName": "gpu-fault-regional-connection",
                                "items": [{"key": "ca.crt", "path": "ca.crt"}],
                            },
                        },
                    ],
                }
            },
        },
    }
