from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import statistics
from datetime import datetime, timezone
from typing import Callable


MAX_POD_LOG_WORKERS = 16


def metric_value(snapshot: str, name: str) -> float | None:
    prefix = f"{name} "
    for line in snapshot.splitlines():
        if line.startswith(prefix):
            return float(line.split()[-1])
    return None


def drain_targets(
    metrics: dict[str, str],
) -> tuple[float, float]:
    queue_values = [
        value
        for snapshot in metrics.values()
        if (value := metric_value(snapshot, "gpu_fault_processor_queue_depth"))
        is not None
    ]
    spool_values = [
        value
        for snapshot in metrics.values()
        if (value := metric_value(snapshot, "gpu_fault_telemetry_spool_depth"))
        is not None
    ]

    def target(values: list[float]) -> float:
        baseline = max(values, default=0.0)
        return 1.0 if baseline <= 1.0 else baseline + 4.0

    return target(queue_values), target(spool_values)


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[
        min(
            len(ordered) - 1,
            int((len(ordered) - 1) * ratio),
        )
    ]


def collect_pod_json_logs(
    entries: list[tuple[str, str]],
    target: Path,
    *,
    fetch: Callable[[str], str],
    on_decode_error: Callable[[str], None] | None = None,
) -> list[dict]:
    target.mkdir(parents=True, exist_ok=True)

    def order_key(entry: tuple[str, str]) -> tuple[int, int | str, str]:
        index, pod = entry
        return (0, int(index), pod) if index.isdigit() else (1, index, pod)

    ordered = sorted(entries, key=order_key)
    if not ordered:
        return []
    workers = min(MAX_POD_LOG_WORKERS, len(ordered))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        bodies = list(executor.map(lambda entry: fetch(entry[1]), ordered))
    documents = []
    for (index, pod), body in zip(ordered, bodies, strict=True):
        (target / f"{index}-{pod}.log").write_text(body, encoding="utf-8")
        try:
            value = json.loads(body)
        except json.JSONDecodeError:
            if on_decode_error is not None:
                on_decode_error(pod)
            continue
        if isinstance(value, dict):
            documents.append(value)
        elif on_decode_error is not None:
            on_decode_error(pod)
    return documents


def aggregate(documents: list[dict]) -> dict:
    if not documents:
        return {"pods": 0}
    paths: dict[str, dict] = {}
    for document in documents:
        for kind, value in document.get("paths", {}).items():
            entry = paths.setdefault(
                kind,
                {
                    "latencies": [],
                    "server": [],
                    "network": [],
                    "stages": {},
                    "status": {},
                    "errors": {},
                    "transport_retries": {},
                },
            )
            entry["latencies"].extend(value.get("raw_latencies_ms", []))
            entry["server"].extend(value.get("server_duration_ms", {}).get("raw", []))
            entry["network"].extend(
                value.get("client_minus_server_ms", {}).get("raw", [])
            )
            for stage, stage_values in value.get("server_stages_ms", {}).items():
                entry["stages"].setdefault(stage, []).extend(
                    stage_values.get("raw", [])
                )
            for status, count in value.get("status_counts", {}).items():
                entry["status"][str(status)] = (
                    entry["status"].get(str(status), 0) + count
                )
            for error, count in (value.get("errors") or {}).items():
                entry["errors"][error] = entry["errors"].get(error, 0) + count
            for error, count in (value.get("transport_retries") or {}).items():
                entry["transport_retries"][error] = (
                    entry["transport_retries"].get(error, 0) + count
                )
    summary = {
        "pods": len(documents),
        "requests": sum(document.get("events", 0) for document in documents),
        "wall_seconds_max": max(
            document.get("wall_seconds", 0.0) for document in documents
        ),
        "start_lag_seconds_max": max(
            abs(document.get("start_lag_seconds", 0.0)) for document in documents
        ),
        "client_cpu_cores_max": max(
            document.get("client_cpu_cores", 0.0) for document in documents
        ),
        "client_cpu_cores_mean": statistics.mean(
            document.get("client_cpu_cores", 0.0) for document in documents
        ),
        "connection_modes": sorted(
            {
                document.get("connection_mode")
                for document in documents
                if document.get("connection_mode")
            }
        ),
        "prewarm_seconds_max": max(
            document.get("prewarm_seconds", 0.0) for document in documents
        ),
        "prewarm_errors": sum(
            document.get("prewarm_errors", 0) for document in documents
        ),
        "dns_modes": sorted(
            {
                document.get("dns_mode")
                for document in documents
                if document.get("dns_mode")
            }
        ),
        "dns_resolution_seconds_max": max(
            document.get("dns_resolution_seconds", 0.0) for document in documents
        ),
        "dns_resolution_attempts_max": max(
            document.get("dns_resolution_attempts", 0) for document in documents
        ),
        "dns_address_count_min": min(
            document.get("dns_address_count", 0) for document in documents
        ),
        "dns_address_count_max": max(
            document.get("dns_address_count", 0) for document in documents
        ),
        "dns_cache_hits": sum(
            document.get("dns_cache_hits", 0) for document in documents
        ),
        "paths": {},
    }
    for kind, entry in sorted(paths.items()):
        latencies = entry["latencies"]
        summary["paths"][kind] = {
            "count": len(latencies),
            "status_counts": entry["status"],
            "errors": entry["errors"],
            "transport_retries": entry["transport_retries"],
            "p50_ms": percentile(latencies, 0.50),
            "p95_ms": percentile(latencies, 0.95),
            "p99_ms": percentile(latencies, 0.99),
            "max_ms": max(latencies) if latencies else 0.0,
        }
        for name in ("server", "network"):
            values = entry[name]
            if values:
                summary["paths"][kind][f"{name}_duration_ms"] = {
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                    "p99": percentile(values, 0.99),
                    "max": max(values),
                }
        summary["paths"][kind]["server_stages_ms"] = {
            stage: {
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
                "max": max(values),
            }
            for stage, values in sorted(entry["stages"].items())
            if values
        }
    if summary["wall_seconds_max"]:
        summary["throughput_req_s"] = summary["requests"] / summary["wall_seconds_max"]
    fault_latencies = [
        value
        for kind in ("NVIDIA_KERNEL", "FABRIC_MANAGER_LOG")
        for value in paths.get(kind, {}).get("latencies", [])
    ]
    if fault_latencies:
        summary["fault_p99_ms"] = percentile(fault_latencies, 0.99)
        summary["fault_p50_ms"] = percentile(fault_latencies, 0.50)
    return summary


def artifact_dir(
    root: Path,
    label: str,
    release: str,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
    safe_release = re.sub(r"[^A-Za-z0-9._-]+", "-", release).strip("-")
    path = root / safe_label / safe_release / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_status(
    artifacts: Path,
    *,
    status: str,
    reason: str | None = None,
) -> None:
    document = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if reason:
        document["reason"] = reason[:2000]
    (artifacts / "status.json").write_text(
        json.dumps(document, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def move_to_aborted(root: Path, artifacts: Path) -> Path:
    relative = artifacts.relative_to(root)
    target = root / "_aborted" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target = target.with_name(
            target.name + "-" + datetime.now(timezone.utc).strftime("%H%M%S")
        )
    artifacts.rename(target)
    return target
