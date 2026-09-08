"""Run CAP-001..004 against disposable control planes in the CPU EKS."""

from __future__ import annotations

import json
import math
import re
import secrets
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpu_fault.admin.site import load_site  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)

TERMINAL_COMMAND_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}
CASE_IDS = tuple(f"GF-REGIONAL-CAP-{number:03d}" for number in range(1, 5))
PROBE_DIR = Path(__file__).with_name("probes")
CASE_LIMITATIONS = [
    "The load is generated against disposable control-plane "
    "Deployments and isolated databases in the selected CPU EKS; "
    "it does not establish a fleet-wide production SLO."
]
# CAP-001: storm-phase B latency may degrade to at most this multiple of the
# B-only baseline measured before the storm. Overridable per run.
DEFAULT_B_LATENCY_FACTOR = 2.0


class CapError(RuntimeError):
    pass


@dataclass
class Probe:
    case: str
    deployment: str
    service: str
    database: str
    pod: str
    local_port: int
    url: str
    port_forward: subprocess.Popen[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def percentile(values: Sequence[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * ratio) - 1)]


def write_json(path: Path, value: Any) -> None:
    """Write one evidence document all-or-nothing (0600, fsynced, renamed).

    The case verdict file is what the next case's predecessor gate reads; a
    half-written one is worse than none. Lists (metric samples, timelines)
    go through the same writer.
    """

    write_json_atomic(path, value)


def command(
    args: Sequence[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout: float | None = 300,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(args),
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        env=dict(env) if env is not None else None,
    )
    if check and result.returncode != 0:
        raise CapError(
            f"command failed ({result.returncode}): {' '.join(args[:5])}; "
            f"stderr={result.stderr[-800:]!r}"
        )
    return result


class CapCoreHarness:
    def __init__(
        self,
        *,
        site_path: Path,
        run_dir: Path,
        case_id: str,
        predecessor: dict[str, Any],
        b_latency_factor: float = DEFAULT_B_LATENCY_FACTOR,
    ) -> None:
        if not b_latency_factor >= 1.0:
            raise CapError("b_latency_factor must be at least 1.0")
        self.site_path = site_path.resolve()
        self.site = load_site(self.site_path)
        self.config = self.site.release_config
        self.root_run_dir = run_dir.resolve()
        self.case_id = case_id
        self.run_dir = self.root_run_dir / "cases" / case_id
        self.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.run_dir.chmod(0o700)
        self.predecessor = predecessor
        self.b_latency_factor = float(b_latency_factor)
        self.cpu_kubeconfig = str(self.config["cpu_kubeconfig"])
        self.namespace = str(self.config["namespace"])
        self.region = str(self.config["aws_region"])
        self.workspace_id = str(self.config["health"]["amp_workspace_id"])
        self.aurora_cluster_id = str(self.config["health"]["aurora_cluster_id"])
        processor_config = self.kubectl_json(
            "get",
            "configmap",
            "gpu-fault-control-worker-config-processor",
            "-o",
            "json",
        )
        self.processor_mode = str(
            processor_config.get("data", {}).get("GPU_FAULT_PROCESSOR_MODE", "")
        )
        if self.processor_mode not in {"direct", "active-active"}:
            raise CapError(
                "live processor mode is not a supported probe mode: "
                f"{self.processor_mode!r}"
            )
        self.run_id = datetime.now(timezone.utc).strftime("cap%H%M%S")
        self.resource_prefix = f"gpu-fault-{self.run_id}"
        self.secret_name = f"{self.resource_prefix}-registry"
        self.configmap_name = f"{self.resource_prefix}-scripts"
        self.tokens = [secrets.token_urlsafe(48) for _ in range(20)]
        self.execution_replay_secret = secrets.token_urlsafe(48)
        self.registrations: list[dict[str, Any]] = [
            {
                "cluster_id": f"cap-cluster-{index:03d}",
                "region": self.region,
                "hyperpod_cluster_name": f"cap-hyperpod-{index:03d}",
                "eks_cluster_arn": (
                    f"arn:aws:eks:{self.region}:000000000000:"
                    f"cluster/cap-cluster-{index:03d}"
                ),
                "token": token,
                "allowed_namespaces": ["gpu-fault-system", "training"],
                "agent_endpoint_allowed_cidrs": ["192.0.2.0/24"],
            }
            for index, token in enumerate(self.tokens)
        ]
        self.live_worker = self._live_worker_deployment()
        self.runtime_image = str(
            self.live_worker["spec"]["template"]["spec"]["containers"][0]["image"]
        )
        self.case_results: list[dict[str, Any]] = []
        self.active_probe: Probe | None = None

    def kubectl(
        self,
        *args: str,
        input_text: str | None = None,
        check: bool = True,
        timeout: float | None = 300,
    ) -> subprocess.CompletedProcess[str]:
        return command(
            (
                "kubectl",
                "--kubeconfig",
                self.cpu_kubeconfig,
                "-n",
                self.namespace,
                *args,
            ),
            input_text=input_text,
            check=check,
            timeout=timeout,
        )

    def kubectl_json(self, *args: str) -> Any:
        result = self.kubectl(*args)
        return json.loads(result.stdout)

    def _live_worker_deployment(self) -> Mapping[str, Any]:
        return cast(
            Mapping[str, Any],
            self.kubectl_json(
                "get", "deployment", "gpu-fault-control-worker", "-o", "json"
            ),
        )

    def apply(self, value: Mapping[str, Any]) -> None:
        self.kubectl("apply", "-f", "-", input_text=json.dumps(value))

    def create_common_resources(self) -> None:
        registry_digest = [
            {
                "cluster_id": item["cluster_id"],
                "token_sha256": __import__("hashlib")
                .sha256(str(item["token"]).encode())
                .hexdigest(),
            }
            for item in self.registrations
        ]
        write_json(
            self.run_dir / "synthetic-registry-summary.json",
            {"clusters": registry_digest},
        )
        self.apply(
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": self.secret_name, "namespace": self.namespace},
                "type": "Opaque",
                "stringData": {
                    "clusters.json": json.dumps(self.registrations),
                    "processor-replay-secret": self.execution_replay_secret,
                },
            }
        )
        scripts = {}
        for name in (
            "cap_probe_app.py",
            "cap_probe_supervisor.py",
            "cap_store_lock.py",
            "cap_seed_commands.py",
        ):
            scripts[name] = (PROBE_DIR / name).read_text(encoding="utf-8")
        self.apply(
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": self.configmap_name,
                    "namespace": self.namespace,
                },
                "data": scripts,
            }
        )

    def base_environment(self, case: str) -> list[dict[str, Any]]:
        direct = {
            "AWS_REGION": self.region,
            "AWS_DEFAULT_REGION": self.region,
            "CAP_DATABASE_NAME": (
                f"gpu_fault_{self.run_id.replace('-', '_')}_{case.lower()}"
            ),
            "GPU_FAULT_SITE_ID": f"{self.run_id}-{case.lower()}",
            "GPU_FAULT_DEPLOYMENT_MODE": "regional",
            "GPU_FAULT_EXECUTOR_MODE": "active",
            "GPU_FAULT_ALLOWED_OPERATIONS": "COLLECT_DIAGNOSTIC_BUNDLE",
            "GPU_FAULT_ENABLE_KUBERNETES_ADAPTER": "false",
            "GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER": "false",
            "GPU_FAULT_ENABLE_HYPERPOD_ADAPTER": "false",
            "GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER": "false",
            "GPU_FAULT_ALLOW_HYPERPOD_REPLACE": "false",
            "GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER": "false",
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "false",
            "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": "true",
            "GPU_FAULT_ALLOW_EMAIL": "false",
            "GPU_FAULT_NOTIFICATION_ASYNC_DELIVERY": "false",
            "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED": "false",
            "GPU_FAULT_PROCESSOR_MODE": self.processor_mode,
            "GPU_FAULT_PROCESSOR_LOCAL_URL": "http://127.0.0.1:18080",
            "GPU_FAULT_PROCESSOR_WORKERS": "1",
            "GPU_FAULT_PROCESSOR_COMPLETION_CLUSTER_CONCURRENCY": "1",
            "GPU_FAULT_PROCESSOR_MAX_QUEUE_DEPTH": "10000",
            "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH": "1000",
            "GPU_FAULT_PROCESSOR_FAULT_RESERVED_QUEUE_DEPTH": "0",
            "GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH": "0",
            "GPU_FAULT_PROCESSOR_GLOBAL_ADMISSION_GUARD": "256",
            "GPU_FAULT_PROCESSOR_RETRY_AFTER_SECONDS": "2",
            "GPU_FAULT_TELEMETRY_SPOOL": "false",
            "GPU_FAULT_STORE_IO_WORKERS": "8",
            "GPU_FAULT_STORE_IO_MAX_IN_FLIGHT": "64",
            "GPU_FAULT_STORE_IO_ADMISSION_TIMEOUT_SECONDS": "5",
            "GPU_FAULT_POSTGRES_POOL_MIN_SIZE": "1",
            "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "8",
            "GPU_FAULT_POSTGRES_POOL_TIMEOUT_SECONDS": "30",
            "GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT": "false",
            "GPU_FAULT_SERVICE_ROLE": "all",
        }
        result: list[dict[str, Any]] = [
            {"name": key, "value": value} for key, value in direct.items()
        ]
        result.extend(
            [
                {
                    "name": "GPU_FAULT_BASE_STORE_URL",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "gpu-fault-aurora",
                            "key": "postgres-url",
                        }
                    },
                },
                {
                    "name": "GPU_FAULT_EXECUTION_TOKEN",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "gpu-fault-control-plane-active",
                            "key": "execution-token",
                        }
                    },
                },
                {
                    "name": "GPU_FAULT_REGIONAL_CLUSTERS_JSON",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": self.secret_name,
                            "key": "clusters.json",
                        }
                    },
                },
                {
                    "name": "GPU_FAULT_PROCESSOR_REPLAY_SECRET",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": self.secret_name,
                            "key": "processor-replay-secret",
                        }
                    },
                },
                {
                    "name": "POD_UID",
                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
                },
            ]
        )
        return result

    @staticmethod
    def set_env(environment: list[dict[str, Any]], values: Mapping[str, str]) -> None:
        by_name = {item["name"]: item for item in environment}
        for name, value in values.items():
            by_name[name] = {"name": name, "value": value}
        environment[:] = list(by_name.values())


class CapProbeHarness(CapCoreHarness):
    def _apply_probe_resources(
        self,
        *,
        suffix: str,
        deployment: str,
        service: str,
        environment: list[dict[str, Any]],
    ) -> None:
        labels = {
            "app": "gpu-fault-control-worker",
            "gpu-fault.io/capacity-probe": f"{self.run_id}-{suffix}",
        }
        live_pod_spec = self.live_worker["spec"]["template"]["spec"]
        live_container = live_pod_spec["containers"][0]
        self.apply(
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": service, "namespace": self.namespace},
                "spec": {
                    "selector": {
                        "gpu-fault.io/capacity-probe": f"{self.run_id}-{suffix}"
                    },
                    "ports": [
                        {
                            "name": "http",
                            "port": 18080,
                            "targetPort": "http",
                        }
                    ],
                },
            }
        )
        self.apply(
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": deployment,
                    "namespace": self.namespace,
                    "labels": {"gpu-fault.io/capacity-probe": self.run_id},
                },
                "spec": {
                    "replicas": 1,
                    "strategy": {"type": "Recreate"},
                    "selector": {
                        "matchLabels": {
                            "gpu-fault.io/capacity-probe": f"{self.run_id}-{suffix}"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": labels,
                            "annotations": {
                                "prometheus.io/path": "/metrics",
                                "prometheus.io/port": "18080",
                                "prometheus.io/scrape": "true",
                            },
                        },
                        "spec": {
                            "enableServiceLinks": False,
                            "serviceAccountName": live_pod_spec.get(
                                "serviceAccountName", "gpu-fault-control-plane"
                            ),
                            "terminationGracePeriodSeconds": 120,
                            "securityContext": live_pod_spec.get("securityContext", {}),
                            "containers": [
                                {
                                    "name": "probe",
                                    "image": self.runtime_image,
                                    "imagePullPolicy": live_container.get(
                                        "imagePullPolicy", "IfNotPresent"
                                    ),
                                    "command": [
                                        "/opt/gpu-fault/control-plane/bin/python",
                                        "/opt/cap/cap_probe_supervisor.py",
                                    ],
                                    "env": environment,
                                    "ports": [
                                        {
                                            "name": "http",
                                            "containerPort": 18080,
                                        }
                                    ],
                                    "readinessProbe": {
                                        "httpGet": {
                                            "path": "/healthz",
                                            "port": "http",
                                        },
                                        "initialDelaySeconds": 5,
                                        "periodSeconds": 3,
                                        "timeoutSeconds": 2,
                                        "failureThreshold": 40,
                                    },
                                    "resources": {
                                        "requests": {"cpu": "1", "memory": "2Gi"},
                                        "limits": {"cpu": "4", "memory": "4Gi"},
                                    },
                                    "securityContext": live_container.get(
                                        "securityContext", {}
                                    ),
                                    "volumeMounts": [
                                        {
                                            "name": "scripts",
                                            "mountPath": "/opt/cap",
                                            "readOnly": True,
                                        },
                                        {"name": "work", "mountPath": "/work"},
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "scripts",
                                    "configMap": {
                                        "name": self.configmap_name,
                                        "defaultMode": 0o555,
                                    },
                                },
                                {"name": "work", "emptyDir": {}},
                            ],
                        },
                    },
                },
            }
        )

    def _wait_for_probe(
        self,
        *,
        case: str,
        suffix: str,
        deployment: str,
        service: str,
        database: str,
    ) -> Probe:
        port_forward: subprocess.Popen[str] | None = None
        try:
            self.kubectl(
                "rollout",
                "status",
                f"deployment/{deployment}",
                "--timeout=240s",
                timeout=260,
            )
            pod = self.kubectl(
                "get",
                "pods",
                "-l",
                f"gpu-fault.io/capacity-probe={self.run_id}-{suffix}",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ).stdout.strip()
            local_port = self.reserve_port()
            port_forward = subprocess.Popen(
                [
                    "kubectl",
                    "--kubeconfig",
                    self.cpu_kubeconfig,
                    "-n",
                    self.namespace,
                    "port-forward",
                    f"service/{service}",
                    f"{local_port}:18080",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            url = f"http://127.0.0.1:{local_port}"
            deadline = time.monotonic() + 45
            error: Exception | None = None
            while time.monotonic() < deadline:
                if port_forward.poll() is not None:
                    raise CapError("kubectl port-forward exited before readiness")
                try:
                    response = httpx.get(f"{url}/healthz", timeout=2)
                    if (
                        response.status_code == 200
                        and response.json().get("status") == "ok"
                    ):
                        probe = Probe(
                            case=case,
                            deployment=deployment,
                            service=service,
                            database=database,
                            pod=pod,
                            local_port=local_port,
                            url=url,
                            port_forward=port_forward,
                        )
                        self.active_probe = probe
                        return probe
                except Exception as exc:
                    error = exc
                time.sleep(0.5)
            raise CapError(f"probe did not become ready: {type(error).__name__}")
        except BaseException:
            if port_forward is not None and port_forward.poll() is None:
                port_forward.terminate()
                try:
                    port_forward.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    port_forward.kill()
                    port_forward.wait(timeout=3)
            self.kubectl(
                "delete",
                "deployment",
                deployment,
                "--ignore-not-found",
                "--wait=true",
                "--timeout=180s",
                check=False,
                timeout=200,
            )
            self.kubectl(
                "delete",
                "service",
                service,
                "--ignore-not-found",
                check=False,
            )
            self.drop_database_fallback(database)
            raise

    def deploy_probe(self, case: str, overrides: Mapping[str, str]) -> Probe:
        suffix = case.lower().replace("-", "")
        deployment = f"{self.resource_prefix}-{suffix}"
        service = deployment
        database = f"gpu_fault_{self.run_id}_{suffix}".lower()
        environment = self.base_environment(case)
        self.set_env(environment, overrides)
        self._apply_probe_resources(
            suffix=suffix,
            deployment=deployment,
            service=service,
            environment=environment,
        )
        return self._wait_for_probe(
            case=case,
            suffix=suffix,
            deployment=deployment,
            service=service,
            database=database,
        )

    @staticmethod
    def reserve_port() -> int:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    def cleanup_probe(self, probe: Probe) -> dict[str, Any]:
        if probe.port_forward.poll() is None:
            probe.port_forward.terminate()
            try:
                probe.port_forward.wait(timeout=5)
            except subprocess.TimeoutExpired:
                probe.port_forward.kill()
                probe.port_forward.wait(timeout=3)
        self.kubectl(
            "delete",
            "deployment",
            probe.deployment,
            "--ignore-not-found",
            "--wait=true",
            "--timeout=180s",
            check=False,
            timeout=200,
        )
        self.kubectl(
            "delete",
            "service",
            probe.service,
            "--ignore-not-found",
            check=False,
        )
        self.drop_database_fallback(probe.database)
        residual = self.kubectl(
            "get",
            "pods",
            "-l",
            f"gpu-fault.io/capacity-probe={self.run_id}",
            "-o",
            "name",
            check=False,
        ).stdout.splitlines()
        self.active_probe = None
        return {"database_dropped": True, "residual_probe_pods": residual}

    def drop_database_fallback(self, database: str) -> None:
        worker = self.kubectl_json(
            "get", "pods", "-l", "app=gpu-fault-control-worker", "-o", "json"
        )
        pods = [
            item
            for item in worker.get("items", [])
            if item.get("status", {}).get("phase") == "Running"
            and not item.get("metadata", {}).get("deletionTimestamp")
            and not item.get("metadata", {})
            .get("labels", {})
            .get("gpu-fault.io/capacity-probe")
        ]
        if not pods:
            raise CapError("no production worker Pod available for DB cleanup")
        pod = sorted(
            pods, key=lambda item: item["metadata"].get("creationTimestamp", "")
        )[-1]["metadata"]["name"]
        cleanup = r"""
import os,sys
import psycopg
from psycopg import sql
database=sys.argv[1]
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"],autocommit=True) as c:
  with c.cursor() as cur:
    cur.execute("select pg_terminate_backend(pid) from pg_stat_activity where datname=%s and pid<>pg_backend_pid()",(database,))
    cur.execute(sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(database)))
"""
        self.kubectl(
            "exec",
            "-i",
            pod,
            "--",
            "python",
            "-",
            database,
            input_text=cleanup,
            timeout=90,
        )

    def cleanup_common(self) -> None:
        self.kubectl(
            "delete",
            "secret",
            self.secret_name,
            "--ignore-not-found",
            check=False,
        )
        self.kubectl(
            "delete",
            "configmap",
            self.configmap_name,
            "--ignore-not-found",
            check=False,
        )


class CapHarnessBase(CapProbeHarness):
    @staticmethod
    def cluster_headers(cluster_id: str, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "X-GPU-Fault-Cluster-ID": cluster_id,
            "Content-Type": "application/json",
        }

    @staticmethod
    def parse_metrics(text: str) -> list[tuple[str, dict[str, str], float]]:
        values = []
        pattern = re.compile(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+"
            r"([-+]?[0-9.eE]+)$"
        )
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            match = pattern.match(line)
            if not match:
                continue
            labels = {}
            raw_labels = match.group(2)
            if raw_labels:
                for item in re.findall(r'(\w+)="((?:\\.|[^"])*)"', raw_labels):
                    labels[item[0]] = item[1].replace('\\"', '"').replace("\\\\", "\\")
            values.append((match.group(1), labels, float(match.group(3))))
        return values

    def metrics(self, url: str) -> list[tuple[str, dict[str, str], float]]:
        response = httpx.get(f"{url}/metrics", timeout=10)
        response.raise_for_status()
        return self.parse_metrics(response.text)

    @staticmethod
    def metric_value(
        metrics: Iterable[tuple[str, dict[str, str], float]],
        name: str,
        **labels: str,
    ) -> float:
        candidates = [
            value
            for metric, metric_labels, value in metrics
            if metric == name
            and all(
                metric_labels.get(key) == expected for key, expected in labels.items()
            )
        ]
        return max(candidates, default=0.0)

    def amp_request(
        self, method: str, path: str, params: Mapping[str, str] | None = None
    ) -> Any:
        base = (
            f"https://aps-workspaces.{self.region}.amazonaws.com"
            f"/workspaces/{self.workspace_id}"
        )
        url = base + path
        body = urllib.parse.urlencode(params or {})
        headers = (
            {"Content-Type": "application/x-www-form-urlencoded"}
            if method == "POST"
            else {}
        )
        credentials = boto3.Session().get_credentials()
        if credentials is None:
            raise CapError("AWS credentials unavailable for AMP query")
        signed = AWSRequest(
            method=method,
            url=url,
            data=body if method == "POST" else None,
            headers=headers,
        )
        SigV4Auth(credentials, "aps", self.region).add_auth(signed)
        request = urllib.request.Request(
            url,
            data=body.encode() if method == "POST" else None,
            headers=dict(signed.headers),
            method=method,
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def alert_states(self, alert_name: str) -> list[str]:
        alerts = self.amp_request("GET", "/api/v1/alerts")["data"]["alerts"]
        return sorted(
            {
                str(item.get("state"))
                for item in alerts
                if item.get("labels", {}).get("alertname") == alert_name
            }
        )

    def probe_control(
        self,
        probe: Probe,
        path: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        script = r"""
import json,sys,urllib.request
path=sys.argv[1]
payload=json.loads(sys.argv[2])
request=urllib.request.Request(
    "http://127.0.0.1:18080"+path,
    data=json.dumps(payload,separators=(",",":")).encode(),
    headers={"Content-Type":"application/json"},
    method="POST",
)
with urllib.request.urlopen(request,timeout=30) as response:
    print(response.read().decode())
"""
        output = self.kubectl(
            "exec",
            "-i",
            probe.pod,
            "--",
            "/opt/gpu-fault/control-plane/bin/python",
            "-",
            path,
            json.dumps(payload, separators=(",", ":")),
            input_text=script,
            timeout=45,
        ).stdout
        return cast(dict[str, Any], json.loads(output))

    def pod_cpu_usage_usec(self, pod: str) -> int:
        output = self.kubectl(
            "exec", pod, "--", "cat", "/sys/fs/cgroup/cpu.stat"
        ).stdout
        for line in output.splitlines():
            key, _, raw = line.partition(" ")
            if key == "usage_usec":
                return int(raw)
        raise CapError("cgroup cpu.stat lacks usage_usec")

    def isolated_db_connections(self, pod: str) -> int:
        script = r"""
from pathlib import Path
import psycopg
url=Path("/work/store-url").read_text()
with psycopg.connect(url) as c:
  with c.cursor() as cur:
    cur.execute("select count(*) from pg_stat_activity where datname=current_database()")
    print(cur.fetchone()[0])
"""
        return int(
            self.kubectl(
                "exec",
                "-i",
                pod,
                "--",
                "/opt/gpu-fault/control-plane/bin/python",
                "-",
                input_text=script,
            ).stdout.strip()
        )

    def connection_budget(self) -> dict[str, Any]:
        deployments = self.kubectl_json("get", "deployments", "-o", "json")
        configmaps = self.kubectl_json(
            "get",
            "configmap",
            "gpu-fault-api-ha-config-postgres",
            "gpu-fault-control-worker-config-postgres",
            "gpu-fault-telemetry-spool-worker-config-postgres",
            "-o",
            "json",
        )
        pools = {
            item["metadata"]["name"]: int(
                item.get("data", {}).get("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "0")
            )
            for item in configmaps["items"]
        }
        rows = []
        total = 0
        config_by_deployment = {
            "gpu-fault-api-ha": "gpu-fault-api-ha-config-postgres",
            "gpu-fault-control-worker": "gpu-fault-control-worker-config-postgres",
            "gpu-fault-telemetry-spool-worker": (
                "gpu-fault-telemetry-spool-worker-config-postgres"
            ),
        }
        for item in deployments["items"]:
            name = item["metadata"]["name"]
            if name not in config_by_deployment:
                continue
            replicas = int(item["spec"].get("replicas", 0))
            args = " ".join(item["spec"]["template"]["spec"]["containers"][0]["args"])
            match = re.search(r"--workers\s+(\d+)", args)
            processes = int(match.group(1)) if match else 1
            pool = pools[config_by_deployment[name]]
            maximum = replicas * processes * pool
            rows.append(
                {
                    "deployment": name,
                    "replicas": replicas,
                    "processes_per_pod": processes,
                    "pool_max_size": pool,
                    "theoretical_connections": maximum,
                }
            )
            total += maximum
        worker = self.kubectl_json(
            "get", "pods", "-l", "app=gpu-fault-control-worker", "-o", "json"
        )
        # The capacity probes carry the same app label but talk to their own
        # isolated database through their own environment; only a production
        # worker knows GPU_FAULT_STORE_URL (live 2026-09-07: KeyError inside a
        # probe Pod ended CAP-003 after all four scale points had run).
        pod = next(
            item["metadata"]["name"]
            for item in worker["items"]
            if item.get("status", {}).get("phase") == "Running"
            and not item.get("metadata", {}).get("deletionTimestamp")
            and not item.get("metadata", {})
            .get("labels", {})
            .get("gpu-fault.io/capacity-probe")
        )
        script = r"""
import os,json,psycopg
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"]) as c:
  with c.cursor() as cur:
    cur.execute("select current_setting('max_connections')::int")
    print(cur.fetchone()[0])
"""
        max_connections = int(
            self.kubectl(
                "exec",
                "-i",
                pod,
                "--",
                "python",
                "-",
                input_text=script,
            ).stdout.strip()
        )
        return {
            "roles": rows,
            "theoretical_total": total,
            "max_connections": max_connections,
            "budget_ratio": total / max_connections,
        }

    def cloudwatch_window(self, start: datetime, end: datetime) -> dict[str, Any]:
        client = boto3.client("cloudwatch", region_name=self.region)
        result = {}
        for metric in ("CPUUtilization", "DatabaseConnections"):
            response = client.get_metric_statistics(
                Namespace="AWS/RDS",
                MetricName=metric,
                Dimensions=[
                    {
                        "Name": "DBClusterIdentifier",
                        "Value": self.aurora_cluster_id,
                    }
                ],
                StartTime=start,
                EndTime=end,
                Period=60,
                Statistics=["Average", "Maximum"],
            )
            result[metric] = sorted(
                [
                    {
                        "timestamp": item["Timestamp"].isoformat(),
                        "average": item.get("Average"),
                        "maximum": item.get("Maximum"),
                    }
                    for item in response.get("Datapoints", [])
                ],
                key=lambda item: item["timestamp"],
            )
        return result

    def production_baseline(self) -> dict[str, Any]:
        deployments = self.kubectl_json("get", "deployments", "-o", "json")
        pods = self.kubectl_json("get", "pods", "-o", "json")
        return {
            "deployments": sorted(
                [
                    {
                        "name": item["metadata"]["name"],
                        "generation": item["metadata"].get("generation"),
                        "replicas": item["spec"].get("replicas", 0),
                        "ready": item.get("status", {}).get("readyReplicas", 0),
                    }
                    for item in deployments.get("items", [])
                    if item["metadata"]["name"]
                    in {
                        "gpu-fault-api-ha",
                        "gpu-fault-control-worker",
                        "gpu-fault-telemetry-spool-worker",
                        "gpu-fault-adot",
                    }
                ],
                key=lambda item: item["name"],
            ),
            "pods": sorted(
                [
                    {
                        "name": item["metadata"]["name"],
                        "uid": item["metadata"]["uid"],
                        "phase": item.get("status", {}).get("phase"),
                        "restarts": sum(
                            status.get("restartCount", 0)
                            for status in item.get("status", {}).get(
                                "containerStatuses", []
                            )
                        ),
                    }
                    for item in pods.get("items", [])
                    if item.get("metadata", {}).get("labels", {}).get("app")
                    in {
                        "gpu-fault-api-ha",
                        "gpu-fault-control-worker",
                        "gpu-fault-telemetry-spool-worker",
                        "gpu-fault-adot",
                    }
                ],
                key=lambda item: item["name"],
            ),
        }

    def run(self) -> int:
        before = self.production_baseline()
        write_json(self.run_dir / "production-before.json", before)
        write_json(
            self.run_dir / "execution-scope.json",
            {
                "started_at": utc_now(),
                "region": self.region,
                "cpu_kubeconfig": self.cpu_kubeconfig,
                "namespace": self.namespace,
                "runtime_image": self.runtime_image,
                "temporary_resource_prefix": self.resource_prefix,
                "database_isolation": "one disposable database per case",
                "gpu_mutations": False,
                "production_registry_mutations": False,
                "maintenance_window": "not required",
                "selected_case": self.case_id,
                "predecessor": self.predecessor,
            },
        )
        if not self.predecessor.get("valid", True):
            raise CapError("formal predecessor evidence is not PASS")
        case_path = self.run_dir / f"{self.case_id}.json"
        # The verdict file exists from the start but says PENDING: the next
        # case's predecessor gate reads this path, and until the production
        # baseline has been compared after cleanup there is no PASS to claim.
        write_json(case_path, self.case_document("PENDING", checks={}))
        result: dict[str, Any] | None = None
        error: str | None = None
        cleanup_errors: list[str] = []
        try:
            self.create_common_resources()
            runner = {
                "GF-REGIONAL-CAP-001": self.case_001,
                "GF-REGIONAL-CAP-002": self.case_002_v2,
                "GF-REGIONAL-CAP-003": self.case_003,
                "GF-REGIONAL-CAP-004": self.case_004,
            }[self.case_id]
            result = runner()
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            cleanup_errors = self.cleanup_all()
            after: dict[str, Any] | None = None
            try:
                after = self.production_baseline()
            except Exception as exc:  # noqa: BLE001 - recorded, must not mask
                cleanup_errors.append(
                    f"production baseline after run: {type(exc).__name__}: {exc}"
                )
            production_unchanged = after is not None and before == after
            write_json(self.run_dir / "production-after.json", after)
            write_json(
                self.run_dir / "production-unchanged.json",
                {"unchanged": production_unchanged},
            )
            verdict = self.verdict(
                result=result,
                error=error,
                cleanup_errors=cleanup_errors,
                production_unchanged=production_unchanged,
            )
            case = self.case_document(
                verdict,
                checks=result or {},
                error=error,
                cleanup_errors=cleanup_errors,
                production_unchanged=production_unchanged,
            )
            self.case_results.append(case)
            write_json(case_path, case)
            write_json(
                self.run_dir / "phase-partial-summary.json",
                {
                    "status": verdict,
                    "error": error,
                    "cleanup_errors": cleanup_errors,
                    "cases": self.case_results,
                },
            )
            print(f"{self.case_id}: {verdict}", flush=True)
        if cleanup_errors:
            raise CapError("capacity run cleanup failed: " + "; ".join(cleanup_errors))
        if not production_unchanged:
            raise CapError(
                "production control-plane baseline changed during capacity run"
            )
        return 0

    @staticmethod
    def verdict(
        *,
        result: dict[str, Any] | None,
        error: str | None,
        cleanup_errors: list[str],
        production_unchanged: bool,
    ) -> str:
        """PASS only when the case passed *and* it left nothing behind."""

        if result is None or error is not None:
            return "FAIL"
        if cleanup_errors or not production_unchanged:
            return "FAIL"
        return "PASS"

    def case_document(
        self,
        verdict: str,
        *,
        checks: dict[str, Any],
        error: str | None = None,
        cleanup_errors: list[str] | None = None,
        production_unchanged: bool | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "report_type": "fault-acceptance",
            "case_id": self.case_id,
            "verdict": verdict,
            "executed_at": utc_now(),
            "checks": checks,
            "error": error,
            "cleanup_errors": list(cleanup_errors or []),
            "production_unchanged": production_unchanged,
            "predecessor": self.predecessor,
            "limitations": CASE_LIMITATIONS,
        }

    def cleanup_all(self) -> list[str]:
        """Remove every probe resource; return the failures instead of hiding them.

        A probe Deployment or database that survives the run is a defect of
        the run, so each failure is recorded and the verdict becomes FAIL.
        """

        errors: list[str] = []
        if self.active_probe is not None:
            probe = self.active_probe
            try:
                cleanup = self.cleanup_probe(probe)
                write_json(
                    self.run_dir / f"{probe.case.lower()}-cleanup-on-exit.json",
                    cleanup,
                )
                if cleanup.get("residual_probe_pods"):
                    errors.append(
                        "probe Pods remain after cleanup: "
                        f"{cleanup['residual_probe_pods']}"
                    )
            except Exception as exc:  # noqa: BLE001 - recorded, verdict FAIL
                errors.append(
                    f"probe cleanup {probe.deployment}: {type(exc).__name__}: {exc}"
                )
        try:
            self.cleanup_common()
        except Exception as exc:  # noqa: BLE001 - recorded, verdict FAIL
            errors.append(f"common resource cleanup: {type(exc).__name__}: {exc}")
        return errors

    def case_001(self) -> dict[str, Any]:
        raise NotImplementedError

    def case_002_v2(self) -> dict[str, Any]:
        raise NotImplementedError

    def case_003(self) -> dict[str, Any]:
        raise NotImplementedError

    def case_004(self) -> dict[str, Any]:
        raise NotImplementedError
