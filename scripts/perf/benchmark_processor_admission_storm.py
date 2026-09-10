from __future__ import annotations

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import psycopg
from psycopg import sql

from gpu_fault.processor import ProcessorRequest
from gpu_fault.store import PostgresStore


@dataclass(frozen=True)
class Result:
    name: str
    requests: int
    client_workers: int
    accepted: int
    rejected: int
    errors: dict[str, int]
    wall_seconds: float
    requests_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    client_cpu_cores: float


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * ratio)]


def store_url_for_schema(url: str, schema: str) -> str:
    parsed = urlsplit(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )


def request(cluster_id: str, sequence: int) -> ProcessorRequest:
    return ProcessorRequest.from_http(
        method="POST",
        path="/v1/collector-events/gpu-inventory",
        query="",
        body=json.dumps(
            {
                "snapshot_id": f"inventory-{cluster_id}-{sequence}",
                "cluster_id": cluster_id,
                "node_id": f"node-{sequence:04d}",
                "observed_at": "2026-08-15T00:00:00Z",
            },
            separators=(",", ":"),
        ).encode(),
        content_type="application/json",
        cluster_id=cluster_id,
    )


def run_scenario(
    url: str,
    *,
    name: str,
    cluster_count: int,
    requests_per_cluster: int,
    client_workers: int,
    use_batch: bool,
    batch_size: int,
) -> Result:
    schema = f"gpu_fault_perf_admission_{uuid4().hex[:12]}"
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    store = PostgresStore(
        store_url_for_schema(url, schema),
        pool_min_size=1,
        pool_max_size=client_workers + 8,
        pool_timeout_seconds=30,
    )
    requests = [
        request(f"perf-cluster-{cluster}", sequence)
        for cluster in range(cluster_count)
        for sequence in range(requests_per_cluster)
    ]
    latencies: list[float] = []
    errors: dict[str, int] = {}
    accepted = 0
    rejected = 0

    def submit(item: ProcessorRequest):
        begin = time.perf_counter()
        try:
            stored, reason = store.try_enqueue_processor_request(
                item,
                max_depth=65536,
                max_cluster_depth=1024,
                reserved_fault_depth=8192,
                reserved_cluster_fault_depth=128,
                global_admission_guard=256,
            )
            return stored is not None, reason, None, time.perf_counter() - begin
        except Exception as exc:
            return False, None, type(exc).__name__, time.perf_counter() - begin

    try:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        if use_batch:
            chunks = [
                requests[index : index + batch_size]
                for index in range(0, len(requests), batch_size)
            ]

            def submit_batch(items: list[ProcessorRequest]):
                begin = time.perf_counter()
                try:
                    batch_results = store.try_enqueue_processor_requests_batch(
                        items,
                        max_depth=65536,
                        max_cluster_depth=1024,
                        reserved_fault_depth=8192,
                        reserved_cluster_fault_depth=128,
                        global_admission_guard=256,
                    )
                    return (
                        batch_results,
                        None,
                        time.perf_counter() - begin,
                        len(items),
                    )
                except Exception as exc:
                    return (
                        [],
                        type(exc).__name__,
                        time.perf_counter() - begin,
                        len(items),
                    )

            with ThreadPoolExecutor(max_workers=client_workers) as executor:
                futures = [executor.submit(submit_batch, chunk) for chunk in chunks]
                for future in as_completed(futures):
                    (
                        batch_results,
                        error,
                        latency,
                        item_count,
                    ) = future.result()
                    if error:
                        errors[error] = errors.get(error, 0) + 1
                        rejected += item_count
                        continue
                    latencies.extend([latency] * len(batch_results))
                    for stored, reason in batch_results:
                        if stored is not None:
                            accepted += 1
                        else:
                            rejected += 1
                            key = f"rejected:{reason}"
                            errors[key] = errors.get(key, 0) + 1
        else:
            with ThreadPoolExecutor(max_workers=client_workers) as executor:
                futures = [executor.submit(submit, item) for item in requests]
                for future in as_completed(futures):
                    stored, reason, error, latency = future.result()
                    latencies.append(latency)
                    if error:
                        errors[error] = errors.get(error, 0) + 1
                    elif stored:
                        accepted += 1
                    else:
                        rejected += 1
                        key = f"rejected:{reason}"
                        errors[key] = errors.get(key, 0) + 1
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
        return Result(
            name=name,
            requests=len(requests),
            client_workers=client_workers,
            accepted=accepted,
            rejected=rejected,
            errors=errors,
            wall_seconds=wall,
            requests_per_second=len(requests) / wall,
            latency_p50_ms=statistics.median(latencies) * 1000,
            latency_p95_ms=percentile(latencies, 0.95) * 1000,
            latency_p99_ms=percentile(latencies, 0.99) * 1000,
            latency_max_ms=max(latencies) * 1000,
            client_cpu_cores=cpu / wall,
        )
    finally:
        store.close()
        with psycopg.connect(url, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


def store_dsn() -> str:
    path = (
        os.environ.get("GPU_FAULT_STORE_URL_FILE")
        or "/etc/gpu-fault/aurora/postgres-url"
    )
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return os.environ["GPU_FAULT_STORE_URL"]


def main() -> None:
    url = store_dsn()
    client_workers = int(os.getenv("CLIENT_WORKERS", "256"))
    use_batch = os.getenv("USE_BATCH", "false").lower() == "true"
    batch_size = int(os.getenv("BATCH_SIZE", "64"))
    results = [
        run_scenario(
            url,
            name="single_cluster_800",
            cluster_count=1,
            requests_per_cluster=800,
            client_workers=client_workers,
            use_batch=use_batch,
            batch_size=batch_size,
        ),
        run_scenario(
            url,
            name="thirty_two_clusters_128",
            cluster_count=32,
            requests_per_cluster=128,
            client_workers=client_workers,
            use_batch=use_batch,
            batch_size=batch_size,
        ),
    ]
    rendered = [asdict(item) for item in results]
    print(json.dumps(rendered, indent=2, sort_keys=True), flush=True)
    failures = []
    for item in results:
        if item.accepted != item.requests or item.rejected or item.errors:
            failures.append(f"{item.name}: admission was not lossless")
        if item.latency_p99_ms >= 1000:
            failures.append(f"{item.name}: p99 {item.latency_p99_ms:.1f}ms >= 1000ms")
        if item.client_cpu_cores >= client_workers * 0.5:
            failures.append(f"{item.name}: load generator saturated")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
