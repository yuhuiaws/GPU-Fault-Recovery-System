from __future__ import annotations

import gzip
import json
import os
import ssl
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote
from urllib.parse import urlsplit

if __package__:
    from .benchmark_mixed_control_plane import PATHS, stamp
else:
    from benchmark_mixed_control_plane import PATHS, stamp


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * ratio)]


def server_timing(headers) -> dict[str, float]:
    values = {}
    for entry in (headers.get("Server-Timing") or "").split(","):
        parts = [part.strip() for part in entry.split(";")]
        if not parts or not parts[0].startswith("gpu_fault_"):
            continue
        for parameter in parts[1:]:
            if not parameter.startswith("dur="):
                continue
            try:
                values[parts[0].removeprefix("gpu_fault_")] = float(
                    parameter.removeprefix("dur=")
                )
            except ValueError:
                pass
    return values


def synchronized_start_epoch() -> float:
    gate = os.getenv("START_GATE_NAME", "").strip()
    if not gate:
        return float(os.environ["START_EPOCH"])
    namespace = os.getenv("START_GATE_NAMESPACE", "gpu-fault-system")
    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    token = token_path.read_text().strip()
    context = ssl.create_default_context(cafile=ca_path)
    url = (
        "https://kubernetes.default.svc/api/v1/namespaces/"
        f"{quote(namespace, safe='')}/configmaps/"
        f"{quote(gate, safe='')}"
    )
    while True:
        request = urllib_request.Request(
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            with urllib_request.urlopen(
                request,
                timeout=5,
                context=context,
            ) as response:
                document = json.loads(response.read())
            value = document.get("data", {}).get("start_epoch")
            if value:
                return float(value)
        except (
            KeyError,
            ValueError,
            json.JSONDecodeError,
            urllib_error.HTTPError,
            urllib_error.URLError,
            TimeoutError,
            OSError,
        ):
            pass
        time.sleep(0.2)


def build_events(
    *,
    cluster_count: int,
    cluster_offset: int,
    nodes_per_cluster: int,
    xid_total: int,
    sxid_total: int,
    gpu_evidence_total: int,
    host_evidence_total: int,
    include_telemetry: bool,
) -> list[tuple[str, str, int]]:
    def local_fault_count(total: int) -> int:
        base, remainder = divmod(total, cluster_count)
        return base + (1 if cluster_offset < remainder else 0)

    events: list[tuple[str, str, int]] = []
    if include_telemetry:
        for node_index in range(nodes_per_cluster):
            events.extend(
                (
                    ("GPU_INVENTORY", "GPU_INVENTORY", node_index),
                    ("GPU_METRICS", "GPU_METRICS", node_index),
                    ("HOST_TELEMETRY", "HOST_TELEMETRY", node_index),
                )
            )
    for kind, template_kind, total in (
        ("NVIDIA_KERNEL", "NVIDIA_KERNEL", xid_total),
        ("FABRIC_MANAGER_LOG", "FABRIC_MANAGER_LOG", sxid_total),
        ("GPU_METRICS_EVIDENCE", "GPU_METRICS", gpu_evidence_total),
        ("HOST_TELEMETRY_EVIDENCE", "HOST_TELEMETRY", host_evidence_total),
    ):
        events.extend(
            (
                kind,
                template_kind,
                index % nodes_per_cluster,
            )
            for index in range(local_fault_count(total))
        )
    return events


def build_output(
    *,
    registration: dict,
    cluster_offset: int,
    events: list[tuple[str, str, int]],
    start_epoch: float,
    actual_start: float,
    wall: float,
    client_cpu_cores: float,
    prewarm_connections: bool,
    prewarm_seconds: float,
    prewarm_errors: int,
    latencies: dict[str, list[float]],
    statuses: dict[str, dict[int, int]],
    errors: dict[str, dict[str, int]],
    server_durations: dict[str, list[float]],
    network_overheads: dict[str, list[float]],
    stage_durations: dict[str, dict[str, list[float]]],
) -> dict:
    output = {
        "cluster_id": registration["cluster_id"],
        "cluster_offset": cluster_offset,
        "events": len(events),
        "scheduled_start_epoch": start_epoch,
        "actual_start_epoch": actual_start,
        "start_lag_seconds": actual_start - start_epoch,
        "wall_seconds": wall,
        "client_cpu_cores": client_cpu_cores,
        "connection_mode": (
            "warm-keepalive" if prewarm_connections else "cold-connect"
        ),
        "prewarm_seconds": prewarm_seconds,
        "prewarm_errors": prewarm_errors,
        "paths": {},
    }
    for kind, values in sorted(latencies.items()):
        path = {
            "count": len(values),
            "status_counts": statuses[kind],
            "errors": errors.get(kind, {}),
            "p50_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
            "p99_ms": percentile(values, 0.99),
            "max_ms": max(values),
            "raw_latencies_ms": values,
        }
        durations = server_durations.get(kind, [])
        if durations:
            path["server_duration_ms"] = {
                "p50": percentile(durations, 0.50),
                "p95": percentile(durations, 0.95),
                "p99": percentile(durations, 0.99),
                "max": max(durations),
                "raw": durations,
            }
            overheads = network_overheads[kind]
            path["client_minus_server_ms"] = {
                "p50": percentile(overheads, 0.50),
                "p95": percentile(overheads, 0.95),
                "p99": percentile(overheads, 0.99),
                "max": max(overheads),
                "raw": overheads,
            }
        path["server_stages_ms"] = {
            stage: {
                "p50": percentile(durations, 0.50),
                "p95": percentile(durations, 0.95),
                "p99": percentile(durations, 0.99),
                "max": max(durations),
                "raw": durations,
            }
            for stage, durations in sorted(stage_durations.get(kind, {}).items())
            if durations
        }
        output["paths"][kind] = path
    return output


def prewarm_connection_pool(
    events: list[tuple[str, str, int]],
    *,
    workers: int,
    enabled: bool,
    connection_factory,
) -> tuple[list[HTTPConnection | None], int, float]:
    connections: list[HTTPConnection | None] = [None for _ in events]
    started = time.perf_counter()
    errors = 0
    if enabled:
        with ThreadPoolExecutor(max_workers=min(workers, len(events))) as executor:
            futures = {
                executor.submit(connection_factory): index
                for index in range(len(events))
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    connections[index] = future.result()
                except OSError:
                    errors += 1
    return connections, errors, time.perf_counter() - started


def wait_for_synchronized_start() -> tuple[float, float]:
    Path(
        os.getenv(
            "GPU_FAULT_PERF_READY_FILE",
            "/tmp/gpu-fault-load-ready",
        )
    ).touch()
    start_epoch = synchronized_start_epoch()
    delay = start_epoch - time.time()
    if delay > 0:
        time.sleep(delay)
    return start_epoch, time.time()


def main() -> None:
    base_url = os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
    clusters = json.loads(Path(os.environ["CLUSTERS_FILE"]).read_text())
    cluster_count = int(os.environ["CLUSTER_COUNT"])
    cluster_offset = int(os.environ["CLUSTER_OFFSET"])
    registration = clusters[cluster_offset]
    nodes_per_cluster = int(os.getenv("NODES_PER_CLUSTER", "256"))
    xid_total = int(os.environ["XID_TOTAL"])
    sxid_total = int(os.environ["SXID_TOTAL"])
    gpu_evidence_total = int(os.getenv("GPU_EVIDENCE_TOTAL", "0"))
    host_evidence_total = int(os.getenv("HOST_EVIDENCE_TOTAL", "0"))
    workers = int(os.getenv("WORKERS", "128"))
    include_telemetry = os.getenv("INCLUDE_TELEMETRY", "true").lower() == "true"
    prewarm_connections = os.getenv("PREWARM_CONNECTIONS", "false").lower() == "true"
    templates_path = Path(os.environ["TEMPLATES_FILE"])
    templates = json.loads(gzip.open(templates_path, "rt").read())
    ssl_context = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
    parsed_base = urlsplit(base_url)
    base_path = parsed_base.path.rstrip("/")
    connection_port = parsed_base.port or (443 if parsed_base.scheme == "https" else 80)

    def new_connection() -> HTTPConnection:
        if parsed_base.scheme == "https":
            connection = HTTPSConnection(
                parsed_base.hostname,
                connection_port,
                timeout=60,
                context=ssl_context,
            )
        else:
            connection = HTTPConnection(
                parsed_base.hostname,
                connection_port,
                timeout=60,
            )
        connection.connect()
        return connection

    events = build_events(
        cluster_count=cluster_count,
        cluster_offset=cluster_offset,
        nodes_per_cluster=nodes_per_cluster,
        xid_total=xid_total,
        sxid_total=sxid_total,
        gpu_evidence_total=gpu_evidence_total,
        host_evidence_total=host_evidence_total,
        include_telemetry=include_telemetry,
    )

    connections, prewarm_errors, prewarm_seconds = prewarm_connection_pool(
        events,
        workers=workers,
        enabled=prewarm_connections,
        connection_factory=new_connection,
    )
    start_epoch, actual_start = wait_for_synchronized_start()
    latencies: dict[str, list[float]] = {}
    server_durations: dict[str, list[float]] = {}
    network_overheads: dict[str, list[float]] = {}
    stage_durations: dict[str, dict[str, list[float]]] = {}
    statuses: dict[str, dict[int, int]] = {}
    errors: dict[str, dict[str, int]] = {}

    def send(indexed_event: tuple[int, tuple[str, str, int]]):
        index, (kind, template_kind, node_index) = indexed_event
        sequence = cluster_offset * 1_000_000 + index
        node_id = f"burst-node-{node_index:04d}"
        try:
            payload = stamp(
                templates.get(template_kind) or {},
                template_kind,
                registration["cluster_id"],
                node_id,
                sequence,
            )
        except Exception as exc:
            return kind, 0, 0.0, type(exc).__name__, {}
        if kind.endswith("_EVIDENCE"):
            payload["edge_filter_reasons"] = ["threshold:synthetic-priority-50"]
            payload["collection_errors"] = []
            payload["context_history"] = []
        body = json.dumps(payload, separators=(",", ":")).encode()
        compressed = len(body) >= 64 * 1024
        if compressed:
            body = gzip.compress(body, compresslevel=6)
        begin = time.perf_counter()
        for attempt in range(2):
            try:
                headers = {
                    "Authorization": (f"Bearer {registration['token']}"),
                    "X-GPU-Fault-Cluster-ID": (registration["cluster_id"]),
                    "Content-Type": "application/json",
                    "Connection": ("keep-alive" if prewarm_connections else "close"),
                    **({"Content-Encoding": "gzip"} if compressed else {}),
                }
                connection = connections[index]
                if connection is not None:
                    connection.request(
                        "POST",
                        base_path + PATHS[template_kind],
                        body=body,
                        headers=headers,
                    )
                    response = connection.getresponse()
                    response.read()
                    status = response.status
                    timing = server_timing(response.headers)
                else:
                    request = urllib_request.Request(
                        base_url + PATHS[template_kind],
                        data=body,
                        headers=headers,
                        method="POST",
                    )
                    with urllib_request.urlopen(
                        request,
                        timeout=60,
                        context=ssl_context,
                    ) as response:
                        status = response.status
                        timing = server_timing(response.headers)
                return (
                    kind,
                    status,
                    (time.perf_counter() - begin) * 1000,
                    None,
                    timing,
                )
            except urllib_error.HTTPError as exc:
                return (
                    kind,
                    exc.code,
                    (time.perf_counter() - begin) * 1000,
                    None,
                    server_timing(exc.headers),
                )
            except (urllib_error.URLError, TimeoutError, OSError) as exc:
                connection = connections[index]
                if connection is not None:
                    connection.close()
                    connections[index] = None
                if attempt == 0:
                    continue
                return (
                    kind,
                    0,
                    (time.perf_counter() - begin) * 1000,
                    type(exc).__name__,
                    {},
                )

    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(send, item) for item in enumerate(events)]
        for future in as_completed(futures):
            kind, status, latency, error, timing = future.result()
            latencies.setdefault(kind, []).append(latency)
            statuses.setdefault(kind, {})[status] = (
                statuses.setdefault(kind, {}).get(status, 0) + 1
            )
            if error:
                errors.setdefault(kind, {})[error] = (
                    errors.setdefault(kind, {}).get(error, 0) + 1
                )
            if "total" in timing:
                total = timing["total"]
                server_durations.setdefault(kind, []).append(total)
                network_overheads.setdefault(kind, []).append(max(0.0, latency - total))
            for stage, duration in timing.items():
                stage_durations.setdefault(kind, {}).setdefault(stage, []).append(
                    duration
                )
    for connection in connections:
        if connection is not None:
            connection.close()
    wall = time.perf_counter() - wall_started
    output = build_output(
        registration=registration,
        cluster_offset=cluster_offset,
        events=events,
        start_epoch=start_epoch,
        actual_start=actual_start,
        wall=wall,
        client_cpu_cores=(time.process_time() - cpu_started) / wall,
        prewarm_connections=prewarm_connections,
        prewarm_seconds=prewarm_seconds,
        prewarm_errors=prewarm_errors,
        latencies=latencies,
        statuses=statuses,
        errors=errors,
        server_durations=server_durations,
        network_overheads=network_overheads,
        stage_durations=stage_durations,
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
