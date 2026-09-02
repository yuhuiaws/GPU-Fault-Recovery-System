from __future__ import annotations

import gzip
import json
import os
import socket
import ssl
import statistics
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterator, cast
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote
from urllib.parse import urlsplit

if __package__:
    from .benchmark_mixed_control_plane import PATHS, stamp
else:
    from benchmark_mixed_control_plane import PATHS, stamp

EVENT_PATHS = {
    **PATHS,
    "TRAINING_PROGRESS": "/v1/training-progress",
}
INTEGRATED_ACTION_KINDS = (
    "RESET_GPU",
    "RESET_GPU",
    "RESTART_NODE",
    "FABRIC_RESET",
)
AddressInfo = tuple[int, int, int, str, tuple[Any, ...]]
Resolver = Callable[..., list[AddressInfo]]


@dataclass(frozen=True)
class DnsResolution:
    addresses: tuple[AddressInfo, ...]
    attempts: int
    seconds: float


class ProcessDnsCache:
    def __init__(
        self,
        *,
        hostname: str,
        port: int,
        addresses: tuple[AddressInfo, ...],
        fallback: Resolver,
    ) -> None:
        if not addresses:
            raise ValueError("control-plane DNS cache requires at least one address")
        self._hostname = self._normalize_hostname(hostname)
        self._port = str(port)
        self._addresses = addresses
        self._fallback = fallback
        self._lock = Lock()
        self._next_address = 0
        self._hits = 0

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @staticmethod
    def _normalize_hostname(hostname: str | bytes | None) -> str:
        if isinstance(hostname, bytes):
            return hostname.decode("ascii").rstrip(".").lower()
        return str(hostname or "").rstrip(".").lower()

    def getaddrinfo(
        self,
        host: str | bytes | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[AddressInfo]:
        if (
            self._normalize_hostname(host) != self._hostname
            or str(port) != self._port
            or type not in {0, socket.SOCK_STREAM}
            or proto not in {0, socket.IPPROTO_TCP}
            or flags != 0
        ):
            return self._fallback(host, port, family, type, proto, flags)
        addresses = [
            item for item in self._addresses if family in {0, socket.AF_UNSPEC, item[0]}
        ]
        if not addresses:
            return self._fallback(host, port, family, type, proto, flags)
        with self._lock:
            offset = self._next_address % len(addresses)
            self._next_address += 1
            self._hits += 1
        return addresses[offset:] + addresses[:offset]


def resolve_control_plane_dns(
    hostname: str,
    port: int,
    *,
    attempts: int = 5,
    base_delay_seconds: float = 0.1,
    resolver: Resolver | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> DnsResolution:
    if attempts < 1:
        raise ValueError("DNS resolution attempts must be positive")
    if base_delay_seconds < 0:
        raise ValueError("DNS resolution retry delay must not be negative")
    active_resolver = resolver or cast(Resolver, socket.getaddrinfo)
    started = time.perf_counter()
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            values = active_resolver(
                hostname,
                port,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                0,
            )
        except OSError as exc:
            last_error = exc
        else:
            addresses = tuple(dict.fromkeys(values))
            if addresses:
                return DnsResolution(
                    addresses=addresses,
                    attempts=attempt,
                    seconds=time.perf_counter() - started,
                )
            last_error = socket.gaierror("control-plane DNS returned no addresses")
        if attempt < attempts:
            sleeper(base_delay_seconds * (2 ** (attempt - 1)))
    raise RuntimeError(
        f"control-plane DNS resolution failed after {attempts} attempts"
    ) from last_error


@contextmanager
def process_dns_cache(cache: ProcessDnsCache | None) -> Iterator[None]:
    if cache is None:
        yield
        return
    previous = socket.getaddrinfo
    socket.getaddrinfo = cache.getaddrinfo  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.getaddrinfo = previous


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


def transport_error_category(exc: BaseException) -> str:
    if isinstance(exc, urllib_error.URLError):
        return f"{type(exc).__name__}:{type(exc.reason).__name__}"
    return type(exc).__name__


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
    training_heartbeat_total: int = 0,
    workload_observation_total: int = 0,
    correlate_attempt_faults: bool = False,
) -> list[tuple[str, str, int]]:
    def local_fault_count(total: int) -> int:
        base, remainder = divmod(total, cluster_count)
        return base + (1 if cluster_offset < remainder else 0)

    events: list[tuple[str, str, int]] = []
    local_observations = local_fault_count(workload_observation_total)
    for index in range(local_observations):
        events.append(
            (
                "WORKLOAD_OBSERVATION",
                "WORKLOAD_OBSERVATION",
                (index * 2) % nodes_per_cluster,
            )
        )
    for index in range(local_fault_count(training_heartbeat_total)):
        events.append(
            (
                "TRAINING_PROGRESS",
                "TRAINING_PROGRESS",
                index % nodes_per_cluster,
            )
        )
    if include_telemetry:
        for node_index in range(nodes_per_cluster):
            events.extend(
                (
                    ("GPU_INVENTORY", "GPU_INVENTORY", node_index),
                    ("GPU_METRICS", "GPU_METRICS", node_index),
                    ("HOST_TELEMETRY", "HOST_TELEMETRY", node_index),
                )
            )
    for kind, template_kind, total, attempt_node_offset in (
        ("NVIDIA_KERNEL", "NVIDIA_KERNEL", xid_total, 0),
        ("FABRIC_MANAGER_LOG", "FABRIC_MANAGER_LOG", sxid_total, 1),
        ("GPU_METRICS_EVIDENCE", "GPU_METRICS", gpu_evidence_total, None),
        ("HOST_TELEMETRY_EVIDENCE", "HOST_TELEMETRY", host_evidence_total, None),
    ):
        local_count = local_fault_count(total)
        events.extend(
            (
                kind,
                template_kind,
                (
                    (2 * (index % local_observations) + int(attempt_node_offset))
                    % nodes_per_cluster
                    if correlate_attempt_faults
                    and local_observations
                    and attempt_node_offset is not None
                    else index % nodes_per_cluster
                ),
            )
            for index in range(local_count)
        )
    return events


def attempt_identity(node_index: int) -> dict[str, str | int]:
    attempt_index = node_index // 2
    suffix = f"{attempt_index:04d}"
    return {
        "attempt_index": attempt_index,
        "job_id": f"burst-job-{suffix}",
        "attempt_id": f"burst-attempt-{suffix}",
        "workload_id": f"kubernetes/job/burst-{suffix}",
    }


def integrated_action_identity(
    run_id: str,
    cluster_offset: int,
    workflow_index: int,
) -> dict[str, str]:
    suffix = f"{run_id}-c{cluster_offset:03d}-w{workflow_index:02d}"
    return {
        "job_id": f"integrated-job-{suffix}",
        "attempt_id": f"integrated-attempt-{suffix}",
        "workload_id": f"training/PyTorchJob/integrated-{suffix}",
        "node_id": f"integrated-node-{suffix}",
        "gpu_uuid": f"GPU-integrated-{cluster_offset:03d}-{workflow_index:02d}",
        "event_id": f"integrated-{suffix}",
        "action_kind": INTEGRATED_ACTION_KINDS[workflow_index],
    }


def build_event_payload(
    *,
    templates: dict,
    kind: str,
    template_kind: str,
    cluster_id: str,
    node_index: int,
    sequence: int,
    correlate_attempt_faults: bool,
    action_identity: dict[str, str] | None = None,
    runtime_profile_version: str = "hyperpod-v1",
) -> dict:
    node_id = (
        action_identity["node_id"]
        if action_identity is not None
        else f"burst-node-{node_index:04d}"
    )
    identity = attempt_identity(node_index)
    now = datetime.now(timezone.utc).isoformat()
    if action_identity is not None:
        if action_identity["action_kind"] == "FABRIC_RESET":
            return {
                "event_id": action_identity["event_id"],
                "cluster_id": cluster_id,
                "node_id": action_identity["node_id"],
                "observed_at": now,
                "source_event_time": now,
                "ingested_at": now,
                "event_source": "PERF_INTEGRATED",
                "sxid": 11001,
                "classification": "FATAL",
                "classification_source": "NVIDIA_FABRIC_MANAGER_CATALOG",
                "link_scope": "TRUNK",
                "link_scope_source": "TRUSTED_NVSWITCH_TOPOLOGY",
                "product": "H200",
                "fabric_partition": (
                    f"{cluster_id}/{action_identity['node_id']}/local-nvswitch"
                ),
                "participating_gpu_uuids": [action_identity["gpu_uuid"]],
                "runtime_profile_version": runtime_profile_version,
                "workload_state": "ACTIVE",
                "affected_workload_ids": [action_identity["workload_id"]],
                "drill_id": action_identity["event_id"],
                "synthetic": True,
            }
        return {
            "event_id": action_identity["event_id"],
            "cluster_id": cluster_id,
            "node_id": action_identity["node_id"],
            "observed_at": now,
            "source_event_time": now,
            "ingested_at": now,
            "event_source": "PERF_INTEGRATED",
            "xid": (79 if action_identity["action_kind"] == "RESTART_NODE" else 48),
            "gpu_uuid": action_identity["gpu_uuid"],
            "product": "H100",
            "driver_branch": 575,
            "cuda_version": "12.9",
            "runtime_profile_version": runtime_profile_version,
            "workload_state": "ACTIVE",
            "affected_workload_ids": [action_identity["workload_id"]],
            "drill_id": action_identity["event_id"],
            "synthetic": True,
        }
    if kind == "TRAINING_PROGRESS":
        return {
            "heartbeat_id": f"burst-progress-{sequence}",
            "cluster_id": cluster_id,
            "attempt_id": identity["attempt_id"],
            "rank": node_index % 2,
            "observed_at": now,
            "node_id": node_id,
            "pod_uid": f"burst-pod-{node_index:04d}",
            "container_name": "trainer",
            "step": 1,
            "samples_per_second": 1.0,
            "loss": 1.0,
            "labels": {"drill_id": "perf-burst"},
        }
    payload = stamp(
        templates.get(template_kind) or {},
        template_kind,
        cluster_id,
        node_id,
        sequence,
    )
    if kind == "WORKLOAD_OBSERVATION":
        payload.pop("node_id", None)
        first_node = int(identity["attempt_index"]) * 2
        payload.update(
            {
                "job_id": identity["job_id"],
                "attempt_id": identity["attempt_id"],
                "workload_phase": "RUNNING",
                "observed_at": now,
                "started_at": now,
                "expected_critical_ranks": 2,
                "workload_ids": [identity["workload_id"]],
                "restart_budget": 0,
                "containers": [
                    {
                        "pod_uid": f"burst-pod-{first_node + rank:04d}",
                        "pod_name": f"burst-pod-{first_node + rank:04d}",
                        "container_name": "trainer",
                        "role": "worker",
                        "rank": rank,
                        "node_id": f"burst-node-{first_node + rank:04d}",
                        "gpu_count": 8,
                        "terminated": False,
                    }
                    for rank in range(2)
                ],
            }
        )
    elif correlate_attempt_faults and kind in {
        "NVIDIA_KERNEL",
        "FABRIC_MANAGER_LOG",
    }:
        payload["affected_workload_ids"] = [identity["workload_id"]]
        payload["workload_state"] = "ACTIVE"
    return payload


def action_context_payloads(
    *,
    cluster_id: str,
    run_id: str,
    cluster_offset: int,
    workflows_per_cluster: int,
    runtime_profile_version: str,
) -> list[tuple[str, dict]]:
    observed_at = datetime.now(timezone.utc).isoformat()
    payloads = []
    for workflow_index in range(workflows_per_cluster):
        identity = integrated_action_identity(
            run_id,
            cluster_offset,
            workflow_index,
        )
        payloads.extend(
            [
                (
                    "/v1/workload-observations",
                    {
                        "cluster_id": cluster_id,
                        "environment": "hyperpod-eks",
                        "job_id": identity["job_id"],
                        "attempt_id": identity["attempt_id"],
                        "workload_phase": "RUNNING",
                        "observed_at": observed_at,
                        "started_at": observed_at,
                        "expected_critical_ranks": 1,
                        "workload_ids": [identity["workload_id"]],
                        "restart_budget": 1,
                        "runtime_profile_version": runtime_profile_version,
                        "containers": [
                            {
                                "pod_uid": f"pod-{identity['attempt_id']}",
                                "pod_name": f"pod-{identity['attempt_id']}",
                                "container_name": "trainer",
                                "role": "worker",
                                "rank": 0,
                                "node_id": identity["node_id"],
                                "gpu_count": 8,
                                "gpu_uuids": [identity["gpu_uuid"]],
                                "terminated": False,
                            }
                        ],
                    },
                ),
                (
                    "/v1/training-progress",
                    {
                        "heartbeat_id": f"heartbeat-{identity['attempt_id']}",
                        "cluster_id": cluster_id,
                        "attempt_id": identity["attempt_id"],
                        "rank": 0,
                        "observed_at": observed_at,
                        "node_id": identity["node_id"],
                        "pod_uid": f"pod-{identity['attempt_id']}",
                        "container_name": "trainer",
                        "gpu_uuids": [identity["gpu_uuid"]],
                        "step": 1,
                        "samples_per_second": 1.0,
                        "loss": 1.0,
                        "labels": {"drill_id": run_id},
                    },
                ),
            ]
        )
    return payloads


def submit_setup_context(
    *,
    base_url: str,
    registration: dict,
    ssl_context: ssl.SSLContext,
    path: str,
    payload: dict,
    timeout_seconds: float = 120,
) -> None:
    headers = {
        "Authorization": f"Bearer {registration['token']}",
        "X-GPU-Fault-Cluster-ID": registration["cluster_id"],
        "Content-Type": "application/json",
    }
    value = urllib_request.Request(
        base_url + path,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers=headers,
        method="POST",
    )
    with urllib_request.urlopen(
        value,
        timeout=30,
        context=ssl_context,
    ) as response:
        body = json.loads(response.read() or b"{}")
    request_id = body.get("processor_request_id")
    if not request_id:
        return
    deadline = time.monotonic() + timeout_seconds
    status_path = f"/v1/processor/requests/{quote(str(request_id), safe='')}"
    while time.monotonic() < deadline:
        status_request = urllib_request.Request(
            base_url + status_path,
            headers=headers,
            method="GET",
        )
        try:
            with urllib_request.urlopen(
                status_request,
                timeout=30,
                context=ssl_context,
            ) as response:
                response.read()
                if response.status != 202:
                    return
        except urllib_error.HTTPError as exc:
            exc.read()
            if exc.code != 202:
                raise
        time.sleep(0.2)
    raise TimeoutError(f"setup processor request did not complete: {request_id}")


def prepare_integrated_actions(
    *,
    base_url: str,
    registration: dict,
    ssl_context: ssl.SSLContext,
    events: list[tuple[str, str, int]],
    cluster_offset: int,
) -> tuple[str, str, dict[int, int], int]:
    workflows = int(os.getenv("ACTION_WORKFLOWS_PER_CLUSTER", "0"))
    run_id = os.getenv("ACTION_RUN_ID", "")
    profile = os.getenv("RUNTIME_PROFILE_VERSION", "hyperpod-v1")
    if workflows < 0:
        raise ValueError("ACTION_WORKFLOWS_PER_CLUSTER cannot be negative")
    if workflows and not run_id:
        raise ValueError("ACTION_RUN_ID is required for action-bearing P0")
    candidates = {
        kind: [
            event_index for event_index, event in enumerate(events) if event[0] == kind
        ]
        for kind in ("NVIDIA_KERNEL", "FABRIC_MANAGER_LOG")
    }
    indexes = {}
    for workflow_index in range(workflows):
        action_kind = INTEGRATED_ACTION_KINDS[workflow_index]
        source_kind = (
            "FABRIC_MANAGER_LOG" if action_kind == "FABRIC_RESET" else "NVIDIA_KERNEL"
        )
        if not candidates[source_kind]:
            break
        indexes[candidates[source_kind].pop(0)] = workflow_index
    if len(indexes) != workflows:
        raise ValueError("local XID allocation is below ACTION_WORKFLOWS_PER_CLUSTER")
    payloads = action_context_payloads(
        cluster_id=registration["cluster_id"],
        run_id=run_id,
        cluster_offset=cluster_offset,
        workflows_per_cluster=workflows,
        runtime_profile_version=profile,
    )
    for path, payload in payloads:
        submit_setup_context(
            base_url=base_url,
            registration=registration,
            ssl_context=ssl_context,
            path=path,
            payload=payload,
        )
    return run_id, profile, indexes, len(payloads)


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
    dns_mode: str,
    dns_resolution_seconds: float,
    dns_resolution_attempts: int,
    dns_address_count: int,
    dns_cache_hits: int,
    latencies: dict[str, list[float]],
    statuses: dict[str, dict[int, int]],
    errors: dict[str, dict[str, int]],
    transport_retries: dict[str, dict[str, int]],
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
        "dns_mode": dns_mode,
        "dns_resolution_seconds": dns_resolution_seconds,
        "dns_resolution_attempts": dns_resolution_attempts,
        "dns_address_count": dns_address_count,
        "dns_cache_hits": dns_cache_hits,
        "paths": {},
    }
    for kind, values in sorted(latencies.items()):
        path = {
            "count": len(values),
            "status_counts": statuses[kind],
            "errors": errors.get(kind, {}),
            "transport_retries": transport_retries.get(kind, {}),
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


def open_connection(parsed_base, connection_port: int, ssl_context) -> HTTPConnection:
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


def send_event(
    indexed_event: tuple[int, tuple[str, str, int]],
    *,
    templates: dict,
    registration: dict,
    cluster_offset: int,
    correlate_attempt_faults: bool,
    action_event_indexes: dict[int, int],
    action_run_id: str,
    runtime_profile_version: str,
    connections: list[HTTPConnection | None],
    prewarm_connections: bool,
    base_url: str,
    base_path: str,
    ssl_context: ssl.SSLContext,
) -> tuple[str, int, float, str | None, dict[str, float], str | None]:
    index, (kind, template_kind, node_index) = indexed_event
    sequence = cluster_offset * 1_000_000 + index
    workflow_index = action_event_indexes.get(index)
    action_identity = (
        integrated_action_identity(
            action_run_id,
            cluster_offset,
            workflow_index,
        )
        if workflow_index is not None
        else None
    )
    try:
        payload = build_event_payload(
            templates=templates,
            kind=kind,
            template_kind=template_kind,
            cluster_id=registration["cluster_id"],
            node_index=node_index,
            sequence=sequence,
            correlate_attempt_faults=correlate_attempt_faults,
            action_identity=action_identity,
            runtime_profile_version=runtime_profile_version,
        )
    except Exception as exc:
        return kind, 0, 0.0, type(exc).__name__, {}, None
    if kind.endswith("_EVIDENCE"):
        payload["edge_filter_reasons"] = ["threshold:synthetic-priority-50"]
        payload["collection_errors"] = []
        payload["context_history"] = []
    event_path = EVENT_PATHS[template_kind]
    if action_identity is not None:
        event_path = (
            "/v1/gpu-events/sxid"
            if action_identity["action_kind"] == "FABRIC_RESET"
            else "/v1/gpu-events/xid"
        )
    body = json.dumps(payload, separators=(",", ":")).encode()
    compressed = len(body) >= 64 * 1024
    if compressed:
        body = gzip.compress(body, compresslevel=6)
    begin = time.perf_counter()
    transport_retry: str | None = None
    for attempt in range(2):
        try:
            headers = {
                "Authorization": f"Bearer {registration['token']}",
                "X-GPU-Fault-Cluster-ID": registration["cluster_id"],
                "Content-Type": "application/json",
                "Connection": ("keep-alive" if prewarm_connections else "close"),
                **({"Content-Encoding": "gzip"} if compressed else {}),
            }
            connection = connections[index]
            if connection is not None:
                connection.request(
                    "POST",
                    base_path + event_path,
                    body=body,
                    headers=headers,
                )
                response = connection.getresponse()
                response.read()
                status = response.status
                timing = server_timing(response.headers)
            else:
                value = urllib_request.Request(
                    base_url + event_path,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with urllib_request.urlopen(
                    value,
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
                transport_retry,
            )
        except urllib_error.HTTPError as exc:
            return (
                kind,
                exc.code,
                (time.perf_counter() - begin) * 1000,
                None,
                server_timing(exc.headers),
                transport_retry,
            )
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            category = transport_error_category(exc)
            connection = connections[index]
            if connection is not None:
                connection.close()
                connections[index] = None
            if attempt == 0:
                transport_retry = category
                continue
            return (
                kind,
                0,
                (time.perf_counter() - begin) * 1000,
                type(exc).__name__,
                {},
                transport_retry,
            )
    raise AssertionError("unreachable send retry state")


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
    training_heartbeat_total = int(os.getenv("TRAINING_HEARTBEAT_TOTAL", "0"))
    workload_observation_total = int(os.getenv("WORKLOAD_OBSERVATION_TOTAL", "0"))
    correlate_attempt_faults = (
        os.getenv("CORRELATE_ATTEMPT_FAULTS", "false").lower() == "true"
    )
    workers = int(os.getenv("WORKERS", "128"))
    include_telemetry = os.getenv("INCLUDE_TELEMETRY", "true").lower() == "true"
    prewarm_connections = os.getenv("PREWARM_CONNECTIONS", "false").lower() == "true"
    cache_control_plane_dns = (
        os.getenv("CACHE_CONTROL_PLANE_DNS", "true").lower() == "true"
    )
    templates_path = Path(os.environ["TEMPLATES_FILE"])
    templates = json.loads(gzip.open(templates_path, "rt").read())
    ssl_context = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
    parsed_base = urlsplit(base_url)
    if parsed_base.hostname is None:
        raise ValueError("GPU_FAULT_CONTROL_PLANE_URL must include a hostname")
    base_path = parsed_base.path.rstrip("/")
    connection_port = parsed_base.port or (443 if parsed_base.scheme == "https" else 80)
    dns_resolution: DnsResolution | None = None
    dns_cache: ProcessDnsCache | None = None
    if cache_control_plane_dns:
        dns_resolution = resolve_control_plane_dns(
            parsed_base.hostname,
            connection_port,
        )
        dns_cache = ProcessDnsCache(
            hostname=parsed_base.hostname,
            port=connection_port,
            addresses=dns_resolution.addresses,
            fallback=cast(Resolver, socket.getaddrinfo),
        )

    events = build_events(
        cluster_count=cluster_count,
        cluster_offset=cluster_offset,
        nodes_per_cluster=nodes_per_cluster,
        xid_total=xid_total,
        sxid_total=sxid_total,
        gpu_evidence_total=gpu_evidence_total,
        host_evidence_total=host_evidence_total,
        include_telemetry=include_telemetry,
        training_heartbeat_total=training_heartbeat_total,
        workload_observation_total=workload_observation_total,
        correlate_attempt_faults=correlate_attempt_faults,
    )
    with process_dns_cache(dns_cache):
        (
            action_run_id,
            runtime_profile_version,
            action_event_indexes,
            setup_request_count,
        ) = prepare_integrated_actions(
            base_url=base_url,
            registration=registration,
            ssl_context=ssl_context,
            events=events,
            cluster_offset=cluster_offset,
        )
        connections, prewarm_errors, prewarm_seconds = prewarm_connection_pool(
            events,
            workers=workers,
            enabled=prewarm_connections,
            connection_factory=lambda: open_connection(
                parsed_base,
                connection_port,
                ssl_context,
            ),
        )
        start_epoch, actual_start = wait_for_synchronized_start()
        latencies: dict[str, list[float]] = {}
        server_durations: dict[str, list[float]] = {}
        network_overheads: dict[str, list[float]] = {}
        stage_durations: dict[str, dict[str, list[float]]] = {}
        statuses: dict[str, dict[int, int]] = {}
        errors: dict[str, dict[str, int]] = {}
        transport_retries: dict[str, dict[str, int]] = {}

        sender = partial(
            send_event,
            templates=templates,
            registration=registration,
            cluster_offset=cluster_offset,
            correlate_attempt_faults=correlate_attempt_faults,
            action_event_indexes=action_event_indexes,
            action_run_id=action_run_id,
            runtime_profile_version=runtime_profile_version,
            connections=connections,
            prewarm_connections=prewarm_connections,
            base_url=base_url,
            base_path=base_path,
            ssl_context=ssl_context,
        )
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(sender, item) for item in enumerate(events)]
            for future in as_completed(futures):
                kind, status, latency, error, timing, transport_retry = future.result()
                latencies.setdefault(kind, []).append(latency)
                statuses.setdefault(kind, {})[status] = (
                    statuses.setdefault(kind, {}).get(status, 0) + 1
                )
                if error:
                    errors.setdefault(kind, {})[error] = (
                        errors.setdefault(kind, {}).get(error, 0) + 1
                    )
                if transport_retry:
                    retries = transport_retries.setdefault(kind, {})
                    retries[transport_retry] = retries.get(transport_retry, 0) + 1
                if "total" in timing:
                    total = timing["total"]
                    server_durations.setdefault(kind, []).append(total)
                    network_overheads.setdefault(kind, []).append(
                        max(0.0, latency - total)
                    )
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
        dns_mode=("process-cache" if dns_cache is not None else "system"),
        dns_resolution_seconds=(
            dns_resolution.seconds if dns_resolution is not None else 0.0
        ),
        dns_resolution_attempts=(
            dns_resolution.attempts if dns_resolution is not None else 0
        ),
        dns_address_count=(
            len(dns_resolution.addresses) if dns_resolution is not None else 0
        ),
        dns_cache_hits=(dns_cache.hits if dns_cache is not None else 0),
        latencies=latencies,
        statuses=statuses,
        errors=errors,
        transport_retries=transport_retries,
        server_durations=server_durations,
        network_overheads=network_overheads,
        stage_durations=stage_durations,
    )
    action_workflows = len(action_event_indexes)
    output["action_workflows_per_cluster"] = action_workflows
    output["action_context_requests"] = setup_request_count
    output["action_event_ids"] = [
        integrated_action_identity(action_run_id, cluster_offset, index)["event_id"]
        for index in range(action_workflows)
    ]
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
