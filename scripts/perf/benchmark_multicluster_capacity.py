from __future__ import annotations

import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx


def percentile(values: list[float], value: float) -> float:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int((len(ordered) - 1) * value)),
    )
    return ordered[index]


def parse_metrics(text: str) -> dict[str, float]:
    result = {}
    wanted = (
        "gpu_fault_processor_queue_depth",
        "gpu_fault_processor_queue_oldest_age_seconds",
        "gpu_fault_store_io_admission_wait_seconds_max",
        "gpu_fault_postgres_pool_checkout_wait_seconds_max",
    )
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        name, _, raw = line.rpartition(" ")
        base = name.split("{", 1)[0]
        if base in wanted:
            result[base] = max(result.get(base, 0.0), float(raw))
        if (
            base == "gpu_fault_processor_cluster_queue_depth"
            and 'cluster_id="perf-p216-' in name
        ):
            result["gpu_fault_processor_audit_queue_depth"] = result.get(
                "gpu_fault_processor_audit_queue_depth", 0.0
            ) + float(raw)
        if base == "gpu_fault_processor_lane_wait_seconds_max":
            result[base] = max(result.get(base, 0.0), float(raw))
    return result


def main() -> None:
    base_url = os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
    ca_file = os.environ["SSL_CERT_FILE"]
    registrations = json.loads(
        Path(os.environ["GPU_FAULT_PERF_CLUSTERS_FILE"]).read_text()
    )
    cluster_count = int(os.getenv("GPU_FAULT_PERF_CLUSTER_COUNT", "32"))
    per_cluster = int(os.getenv("GPU_FAULT_PERF_REQUESTS_PER_CLUSTER", "200"))
    workers = int(os.getenv("GPU_FAULT_PERF_WORKERS", "256"))
    target_rps = float(os.getenv("GPU_FAULT_PERF_TARGET_RPS", "0"))
    clusters = registrations[:cluster_count]
    total = cluster_count * per_cluster
    suffix = uuid4().hex[:10]
    limits = httpx.Limits(
        max_connections=workers,
        max_keepalive_connections=workers,
    )
    client = httpx.Client(
        base_url=base_url,
        verify=ca_file,
        timeout=30,
        limits=limits,
    )
    latencies = []
    responses = []
    receipt_urls = []
    monitor_stop = threading.Event()
    metric_maxima: dict[str, float] = {}

    def monitor() -> None:
        while not monitor_stop.is_set():
            try:
                values = parse_metrics(client.get("/metrics").text)
                for key, value in values.items():
                    metric_maxima[key] = max(metric_maxima.get(key, 0.0), value)
            except Exception:
                pass
            monitor_stop.wait(1)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    def send(index: int):
        if target_rps > 0:
            scheduled = wall_started + index / target_rps
            delay = scheduled - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        cluster_index = index % cluster_count
        sequence = index // cluster_count
        registration = clusters[cluster_index]
        cluster_id = registration["cluster_id"]
        observed_at = datetime.now(timezone.utc) + timedelta(microseconds=index)
        started = time.perf_counter()
        try:
            response = client.post(
                "/v1/workload-observations",
                headers={
                    "Authorization": (f"Bearer {registration['token']}"),
                    "X-GPU-Fault-Cluster-ID": cluster_id,
                },
                json={
                    "cluster_id": cluster_id,
                    "environment": "hyperpod-eks",
                    "job_id": f"job-{suffix}-{sequence:04d}",
                    "attempt_id": (f"attempt-{suffix}-{sequence:04d}"),
                    "workload_phase": "RUNNING",
                    "observed_at": observed_at.isoformat(),
                    "started_at": observed_at.isoformat(),
                    "expected_critical_ranks": 1,
                    "containers": [
                        {
                            "pod_uid": f"pod-{suffix}-{sequence:04d}",
                            "pod_name": f"pod-{suffix}-{sequence:04d}",
                            "container_name": "trainer",
                            "role": "worker",
                            "rank": 0,
                            "node_id": (f"node-{sequence % 256:04d}"),
                            "gpu_count": 8,
                        }
                    ],
                    "workload_ids": [f"kubernetes/job/{suffix}-{sequence:04d}"],
                    "runtime_profile_version": "hyperpod-v1",
                    "restart_budget": 0,
                },
            )
            body = response.json()
            return (
                response.status_code,
                body.get("status_url"),
                (time.perf_counter() - started) * 1000,
            )
        except Exception:
            return 0, None, (time.perf_counter() - started) * 1000

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(send, index) for index in range(total)]
        for future in as_completed(futures):
            status, status_url, latency = future.result()
            responses.append(status)
            latencies.append(latency)
            if status == 202 and status_url:
                receipt_urls.append(status_url)
    submit_wall = time.perf_counter() - wall_started

    receipt_sample_size = int(os.getenv("GPU_FAULT_PERF_RECEIPT_SAMPLE", "64"))
    pending = set(receipt_urls[:receipt_sample_size])
    completion_deadline = time.monotonic() + 180
    queue_drained = False
    zero_samples = 0
    while time.monotonic() < completion_deadline:
        batch = list(pending)[:workers]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(client.get, url): url for url in batch}
            for future in as_completed(futures):
                url = futures[future]
                try:
                    if future.result().status_code != 202:
                        pending.discard(url)
                except Exception:
                    pass
        if pending:
            time.sleep(0.25)
        try:
            queue_depth = parse_metrics(client.get("/metrics").text).get(
                "gpu_fault_processor_audit_queue_depth", 0
            )
            zero_samples = zero_samples + 1 if queue_depth == 0 else 0
            queue_drained = zero_samples >= 2
        except Exception:
            queue_drained = False
        if not pending and queue_drained:
            break
    total_wall = time.perf_counter() - wall_started
    monitor_stop.set()
    monitor_thread.join(timeout=2)
    client.close()
    status_counts = {status: responses.count(status) for status in set(responses)}
    result = {
        "cluster_count": cluster_count,
        "requests_per_cluster": per_cluster,
        "total_requests": total,
        "target_rps": target_rps,
        "status_counts": status_counts,
        "submit_wall_seconds": submit_wall,
        "submit_requests_per_second": total / submit_wall,
        "submit_p50_ms": statistics.median(latencies),
        "submit_p95_ms": percentile(latencies, 0.95),
        "submit_p99_ms": percentile(latencies, 0.99),
        "receipt_sample_size": min(receipt_sample_size, len(receipt_urls)),
        "receipts_completed": min(receipt_sample_size, len(receipt_urls))
        - len(pending),
        "receipts_pending": len(pending),
        "queue_drained": queue_drained,
        "total_wall_seconds": total_wall,
        "client_average_cpu_cores": (time.process_time() - cpu_started) / total_wall,
        "metric_maxima": metric_maxima,
        "suffix": suffix,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    failures = []
    if status_counts.get(202) != total:
        failures.append("not all requests returned HTTP 202")
    if pending or not queue_drained:
        failures.append(
            "processor queue or sampled receipts did not drain in 180 seconds"
        )
    if result["submit_p99_ms"] >= 1000:
        failures.append("submit p99 exceeded 1 second")
    if result["client_average_cpu_cores"] >= 8:
        failures.append("load generator exceeded 8 CPU cores")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
