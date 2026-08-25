from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import ssl
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4


RATES = {
    "GPU_INVENTORY": 136,
    "GPU_METRICS": 27,
    "HOST_TELEMETRY": 27,
    "WORKLOAD_OBSERVATION": 20,
    "NODE_LOGS": 2,
    "NVIDIA_KERNEL": 1,
    "FABRIC_MANAGER_LOG": 1,
}

DRILL_ID = os.getenv("GPU_FAULT_PERF_DRILL_ID", "perf-burst").strip() or "perf-burst"
"""Marks every synthetic fault this suite injects as a drill.

The control plane copies the label onto the advisory notification it raises,
and the notification service refuses to mail anything carrying one.  Without
it a burst run mails one message per synthetic fault to the real recipients.
"""

PATHS = {
    "GPU_INVENTORY": "/v1/collector-events/gpu-inventory",
    "GPU_METRICS": "/v1/collector-events/gpu-metrics",
    "HOST_TELEMETRY": "/v1/collector-events/host-telemetry",
    "WORKLOAD_OBSERVATION": "/v1/workload-observations",
    "NODE_LOGS": "/v1/collector-events/node-logs",
    "NVIDIA_KERNEL": "/v1/collector-events/nvidia-kernel",
    "FABRIC_MANAGER_LOG": "/v1/collector-events/fabric-manager",
}


def percentile(values, value):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * value))]


def stamp(payload, kind, cluster_id, node_id, sequence):
    value = copy.deepcopy(payload)
    now = datetime.now(timezone.utc).isoformat()
    value["cluster_id"] = cluster_id
    value["node_id"] = node_id
    for field in ("observed_at", "collected_at", "ingested_at"):
        if field in value:
            value[field] = now
    for field in ("batch_id", "snapshot_id", "record_id"):
        if field in value:
            value[field] = f"mixed-{kind.lower()}-{sequence}-{uuid4().hex[:8]}"
    if kind in {"GPU_METRICS", "HOST_TELEMETRY"}:
        value["affected_workload_ids"] = []
        value["workload_state"] = "IDLE"
        value["context_history"] = []
        value["edge_filter_reasons"] = ["health-summary"]
    if kind == "HOST_TELEMETRY":
        for sample in value.get("samples", []):
            name = str(
                sample.get("canonical_name")
                or sample.get("name")
                or sample.get("metric_name")
                or ""
            ).lower()
            if any(
                token in name
                for token in (
                    "delta",
                    "error",
                    "drop",
                    "mismatch",
                    "unavailable",
                    "critical",
                    "link_down",
                )
            ):
                sample["value"] = 0
            elif "link_up" in name:
                sample["value"] = 1
    if kind == "WORKLOAD_OBSERVATION":
        value["job_id"] = f"mixed-job-{sequence}"
        value["attempt_id"] = f"mixed-attempt-{sequence}"
        value["workload_phase"] = "RUNNING"
        value["observed_at"] = now
        value["started_at"] = now
        value["workload_ids"] = [f"kubernetes/job/mixed-{sequence}"]
        value["containers"] = [
            {
                "pod_uid": f"mixed-pod-{sequence}",
                "pod_name": f"mixed-pod-{sequence}",
                "container_name": "trainer",
                "role": "worker",
                "rank": 0,
                "node_id": node_id,
                "gpu_count": 8,
                "terminated": False,
            }
        ]
    elif kind == "NVIDIA_KERNEL":
        value["message"] = (
            "NVRM: Xid (PCI:0000:00:00): 14, "
            f"Channel exception (audit load) drill_id={DRILL_ID}"
        )
        value["product"] = "H200"
        value["driver_branch"] = 575
        value["cuda_version"] = "12.9"
    elif kind == "FABRIC_MANAGER_LOG":
        value = {
            "cluster_id": cluster_id,
            "node_id": node_id,
            "record_id": f"mixed-fm-{sequence}-{uuid4().hex[:8]}",
            "observed_at": now,
            "collected_at": now,
            "message": (
                "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): "
                f"22013, Non-fatal, Link 12 SAW_MVB error drill_id={DRILL_ID}"
            ),
            "source": "journal",
            "unit": "nvidia-fabricmanager.service",
            "fields": {},
            "product": "H200",
            "driver_branch": 575,
            "cuda_version": "12.9",
            "workload_state": "IDLE",
            "affected_workload_ids": [],
        }
    elif kind == "NODE_LOGS":
        for entry in value.get("entries", []):
            entry["message"] = "mixed load informational message"
    return value


def main():
    base_url = os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
    all_clusters = json.loads(Path(os.environ["CLUSTERS_FILE"]).read_text())
    cluster_count = int(os.getenv("CLUSTER_COUNT", "32"))
    cluster_offset = int(os.getenv("CLUSTER_OFFSET", "0"))
    clusters = all_clusters[cluster_offset : cluster_offset + cluster_count]
    template_path = Path(os.environ["TEMPLATES_FILE"])
    templates = (
        json.loads(gzip.open(template_path, "rt").read())
        if template_path.suffix == ".gz"
        else json.loads(template_path.read_text())
    )
    duration = int(os.getenv("DURATION_SECONDS", "30"))
    workers = int(os.getenv("WORKERS", "256"))
    events = []
    sequence = 0
    for kind, global_rate in RATES.items():
        total_for_kind = max(
            1,
            round(global_rate * cluster_count * duration / 32),
        )
        for index in range(total_for_kind):
            events.append((index * duration / total_for_kind, kind, sequence))
            sequence += 1
    events.sort()
    ssl_context = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
    started = time.perf_counter()
    phase_offset = 0.0
    if (
        os.getenv("GPU_FAULT_PERF_RANDOMIZE_PHASE", "true").strip().lower() == "true"
        and len(clusters) == 1
    ):
        digest = hashlib.sha256(clusters[0]["cluster_id"].encode()).digest()
        phase_offset = int.from_bytes(digest[:8], "big") / float(2**64) * duration
    results = {kind: [] for kind in RATES}
    statuses = {kind: {} for kind in RATES}
    errors = {kind: {} for kind in RATES}

    def send(event):
        offset, kind, seq = event
        scheduled_offset = (
            (offset + phase_offset) % duration if phase_offset else offset
        )
        delay = started + scheduled_offset - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        registration = clusters[seq % len(clusters)]
        node_id = f"mixed-node-{seq % 256:04d}"
        payload = stamp(
            templates.get(kind) or {},
            kind,
            registration["cluster_id"],
            node_id,
            seq,
        )
        body = json.dumps(payload, separators=(",", ":")).encode()
        compressed = len(body) >= 64 * 1024
        if compressed:
            body = gzip.compress(body, compresslevel=6)
        begin = time.perf_counter()
        for attempt in range(2):
            try:
                request = urllib_request.Request(
                    base_url + PATHS[kind],
                    data=body,
                    headers={
                        "Authorization": (f"Bearer {registration['token']}"),
                        "X-GPU-Fault-Cluster-ID": (registration["cluster_id"]),
                        "Connection": "close",
                        "Content-Type": "application/json",
                        **({"Content-Encoding": "gzip"} if compressed else {}),
                    },
                    method="POST",
                )
                with urllib_request.urlopen(
                    request,
                    timeout=30,
                    context=ssl_context,
                ) as response:
                    status = response.status
                return (
                    kind,
                    status,
                    (time.perf_counter() - begin) * 1000,
                    None,
                )
            except urllib_error.HTTPError as exc:
                return (
                    kind,
                    exc.code,
                    (time.perf_counter() - begin) * 1000,
                    None,
                )
            except (urllib_error.URLError, TimeoutError, OSError) as exc:
                if attempt == 0:
                    continue
                return (
                    kind,
                    0,
                    (time.perf_counter() - begin) * 1000,
                    type(exc).__name__,
                )
            except Exception as exc:
                return (
                    kind,
                    0,
                    (time.perf_counter() - begin) * 1000,
                    type(exc).__name__,
                )

    cpu_started = time.process_time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for future in as_completed([executor.submit(send, item) for item in events]):
            kind, status, latency, error = future.result()
            results[kind].append(latency)
            statuses[kind][status] = statuses[kind].get(status, 0) + 1
            if error is not None:
                errors[kind][error] = errors[kind].get(error, 0) + 1
    wall = time.perf_counter() - started
    output = {
        "wall_seconds": wall,
        "client_cpu_cores": (time.process_time() - cpu_started) / wall,
        "paths": {
            kind: {
                "count": len(values),
                "status_counts": statuses[kind],
                "errors": errors[kind],
                "p50_ms": statistics.median(values),
                "p95_ms": percentile(values, 0.95),
                "p99_ms": percentile(values, 0.99),
            }
            for kind, values in results.items()
        },
    }
    if os.getenv("OUTPUT_RAW_LATENCIES", "false").lower() == "true":
        output["raw_latencies_ms"] = results
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    if os.getenv("ENFORCE", "true").lower() != "true":
        return
    failures = []
    for kind, item in output["paths"].items():
        accepted = item["status_counts"].get(202, 0)
        if accepted != item["count"]:
            failures.append(f"{kind} did not return all 202")
        if item["p99_ms"] >= 1000:
            failures.append(f"{kind} p99 exceeded 1 second")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
