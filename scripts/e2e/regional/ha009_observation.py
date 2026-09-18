"""Credential-rotation health observations, without exporting credential values."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from typing import Any

from prometheus_client.parser import text_string_to_metric_families

POOL_METRIC_NAMES = (
    "gpu_fault_postgres_pool_size",
    "gpu_fault_postgres_pool_available",
    "gpu_fault_postgres_pool_requests_waiting",
    "gpu_fault_postgres_pool_requests_errors_total",
    "gpu_fault_postgres_pool_connections_errors_total",
    "gpu_fault_postgres_pool_connections_lost_total",
    "gpu_fault_aurora_credential_refresh_last_success_age_seconds",
)


def parse_pool_metrics(text: str) -> dict[str, float]:
    series: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name not in POOL_METRIC_NAMES:
                continue
            value = float(sample.value)
            labels = tuple(sorted(sample.labels.items()))
            observed = series.setdefault(sample.name, {})
            if labels in observed or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid or duplicate pool metric: {sample.name}")
            observed[labels] = value
    # Per-process counters must be summed, not overwritten by the last slot.
    # Freshness is a worst-case age, never a sum of per-process ages.
    return {
        name: (
            max(observed.values())
            if name.endswith("_age_seconds")
            else sum(observed.values())
        )
        for name, observed in series.items()
    }


def pod_observation(
    control: Callable[..., str], pod: str, port: int, *, python: str
) -> dict[str, Any]:
    # This handshake is by an exec child, not a forced recycle of application pools.
    raw = control(
        "exec",
        pod,
        "--",
        python,
        "-c",
        "import json, os, sys, urllib.error, urllib.request\n"
        "import psycopg\n"
        "from gpu_fault.store.postgres.pool import StoreCredentials\n"
        "def get(path):\n"
        "    try:\n"
        "        with urllib.request.urlopen('http://127.0.0.1:' + sys.argv[1] + path, "
        "timeout=10) as response:\n"
        "            return response.status, response.read().decode()\n"
        "    except urllib.error.HTTPError as exc:\n"
        "        return exc.code, ''\n"
        "health, _ = get('/healthz')\n"
        "status, body = get('/metrics')\n"
        "connected = False\n"
        "connection_error_type = None\n"
        "try:\n"
        "    credentials = StoreCredentials(os.environ['GPU_FAULT_STORE_URL'], "
        "path=os.environ.get('GPU_FAULT_STORE_URL_FILE'))\n"
        "    with psycopg.connect(credentials.conninfo(), connect_timeout=10) as conn:\n"
        "        connected = conn.execute('SELECT 1').fetchone() == (1,)\n"
        "except Exception as exc:\n"
        "    connection_error_type = type(exc).__name__\n"
        "print(json.dumps({'healthz_status': health, 'metrics_status': status, "
        "'metrics_text': body, 'fresh_connection': connected, "
        "'connection_error_type': connection_error_type}))",
        str(port),
        timeout=60,
    )
    try:
        payload = json.loads(raw.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise ValueError("Pod health probe did not return JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("Pod health probe did not return an object")
    return {
        "healthz_status": payload.get("healthz_status"),
        "metrics_status": payload.get("metrics_status"),
        "metrics": parse_pool_metrics(str(payload.get("metrics_text") or "")),
        "fresh_connection": payload.get("fresh_connection") is True,
        "connection_error_type": payload.get("connection_error_type"),
    }


def observation_errors(
    expected_pods: set[str], propagation: dict[str, Any], observation: dict[str, Any]
) -> list[str]:
    errors = []
    for label, observed in (
        ("Secret propagation", propagation.get("pods") or {}),
        ("post-idle samples", observation.get("samples") or {}),
        ("authentication log reads", observation.get("auth_failures_in_logs") or {}),
    ):
        if (
            not expected_pods
            or not isinstance(observed, dict)
            or set(observed) != expected_pods
        ):
            errors.append(f"{label} does not cover every expected Pod")
    for pod, digest in (propagation.get("pods") or {}).items():
        if not digest or digest != propagation.get("digest"):
            errors.append(
                f"{pod} projected postgres-url did not catch up with the Secret"
            )
    for pod, samples in (observation.get("samples") or {}).items():
        if not samples:
            errors.append(f"{pod} was not observed after the idle window")
            continue
        for index, sample in enumerate(samples):
            if sample.get("healthz_status") != 200:
                errors.append(
                    f"{pod} /healthz returned {sample.get('healthz_status')} after the idle window (sample {index})"
                )
            if sample.get("fresh_connection") is not True:
                errors.append(f"{pod} did not prove a fresh exec-child SQL connection")
            if sample.get("metrics_status") != 200:
                errors.append(f"{pod} /metrics was not successfully observed")
            metrics = sample.get("metrics") or {}
            if "gpu_fault_postgres_pool_connections_errors_total" not in metrics:
                errors.append(
                    f"{pod} /metrics does not export gpu_fault_postgres_pool_connections_errors_total"
                )
            if any(
                type(value) not in (float, int) or not math.isfinite(value) or value < 0
                for value in metrics.values()
            ):
                errors.append(f"{pod} /metrics contains an invalid value")
    for pod, count in (observation.get("auth_failures_in_logs") or {}).items():
        if type(count) is not int or count != 0:
            errors.append(
                f"{pod} password authentication failure count is nonzero or unknown"
            )
    return errors


def steady_deployments(
    before: dict[str, Any], current: dict[str, Any], names: Sequence[str]
) -> list[str]:
    errors = []
    for name in names:
        if name not in before or name not in current:
            errors.append(f"missing Deployment observation: {name}")
            continue
        baseline, item = before[name], current[name]
        if not baseline.get("uid") or item.get("uid") != baseline["uid"]:
            errors.append(f"{name} Deployment UID changed or was not observed")
        if item["generation"] != baseline["generation"]:
            errors.append(f"{name} generation changed: a rotation must not roll it")
        replicas = baseline.get("replicas")
        if (
            type(replicas) is not int
            or replicas < 0
            or item.get("replicas") != replicas
        ):
            errors.append(f"{name} replica target changed or is invalid")
            continue
        if name in names[:2] and replicas == 0:
            errors.append(f"{name} has no enabled replicas")
        before_pods, after_pods = dict(baseline["pods"]), dict(item["pods"])
        if len(before_pods) != replicas:
            errors.append(f"{name} baseline Pod observation is incomplete")
        if len(after_pods) != replicas:
            errors.append(f"{name} final Pod observation is incomplete")
        if {pod: value["uid"] for pod, value in before_pods.items()} != {
            pod: value["uid"] for pod, value in after_pods.items()
        }:
            errors.append(f"{name} Pod set changed: a rotation must not replace Pods")
        for pod, value in after_pods.items():
            if value.get("restarts") != (before_pods.get(pod) or {}).get("restarts"):
                errors.append(f"{name} Pod {pod} restarted during the rotation")
            if value.get("ready") is not True:
                errors.append(f"{name} Pod {pod} is not Ready after the rotation")
        if any(
            item.get(field) != replicas for field in ("ready", "updated", "available")
        ):
            errors.append(f"{name} is not fully Ready after the rotation")
    return errors
