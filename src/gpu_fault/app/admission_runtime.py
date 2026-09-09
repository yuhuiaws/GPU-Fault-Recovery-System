from __future__ import annotations

import json
import logging
import os
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

from fastapi import HTTPException

from gpu_fault.app.admission import (
    _ProcessorAdmissionBatcher,
    _StripedAdmissionScope,
)
from gpu_fault.app.runtime import ProcessorDispatchState
from gpu_fault.async_store import (
    AsyncStoreExecutor,
    RequestDeadlineExceeded,
    StoreIoCapacityExceeded,
)
from gpu_fault.channel_registry import CHANNEL_REGISTRY
from gpu_fault.env import env_bool
from gpu_fault.processor import ProcessorCoordinator
from gpu_fault.processor.settings import ProcessorPoolSettings
from gpu_fault.processor_diagnostics import report_processor_replay_phase

LOGGER = logging.getLogger(__name__)

# Worst-case deflate expansion, used to bound the *wire* length of a compressed
# body against the same limit as a plain one. A deflate stream may store data
# uncompressed, in which case each block carries at most 65535 bytes behind a
# 5-byte header, so a payload of n bytes cannot need more than
# ``n + ceil(n / 65535) * 5`` bytes of deflate output. The gzip wrapper adds a
# 10-byte header, an 8-byte trailer, and optional FNAME/FCOMMENT fields that are
# arbitrary length in principle; ``GZIP_ENVELOPE_SLACK_BYTES`` allows a generous
# fixed amount for them rather than parsing the header before the size check.
DEFLATE_STORED_BLOCK_BYTES = 65535
DEFLATE_STORED_BLOCK_OVERHEAD_BYTES = 5
GZIP_ENVELOPE_SLACK_BYTES = 1024


def max_compressed_request_bytes(max_bytes: int) -> int:
    """Largest wire length a compressed body may declare and still be in bounds.

    ``_declared_oversize`` used to return early for any request carrying a
    ``Content-Encoding``, on the correct observation that a compressed body can
    be slightly larger than its output and so the wire length is not the limit
    itself. The consequence was that it was not *any* bound: a caller announcing
    a 200MB gzip body got 200MB buffered before anything looked at the size.

    This is that missing bound. It is deliberately loose — a body between
    ``max_bytes`` and this value is admitted here and then rejected by the
    decoder on its decompressed size — because the only job at this layer is to
    stop unbounded buffering, and nothing here can know the output size yet.
    """

    if max_bytes <= 0:
        raise ValueError("max request bytes must be positive")
    blocks = -(-max_bytes // DEFLATE_STORED_BLOCK_BYTES)
    return (
        max_bytes
        + blocks * DEFLATE_STORED_BLOCK_OVERHEAD_BYTES
        + GZIP_ENVELOPE_SLACK_BYTES
    )


def declared_body_oversize(headers: Mapping[str, str], max_bytes: int) -> bool:
    """Whether the announced ``Content-Length`` already exceeds the budget.

    Shared by the two hops that read a body -- the regional cluster
    authentication middleware and processor dispatch -- so that whichever runs
    first refuses to buffer an announced flood (A-3). A compressed body is held
    to :func:`max_compressed_request_bytes`; the decoder stays authoritative
    for the decompressed size. No usable declaration (chunked, malformed) means
    the post-decode check decides.
    """

    try:
        length = int(headers.get("Content-Length", ""))
    except ValueError:
        return False
    limit = (
        max_compressed_request_bytes(max_bytes)
        if headers.get("Content-Encoding", "").strip()
        else max_bytes
    )
    return length > limit


class NulInRequestBody(ValueError):
    """The JSON body carries a ``\\u0000`` escape (E-3).

    PostgreSQL ``jsonb`` refuses a NUL in a string (SQLSTATE 22P05), so a body
    that decodes fine here would fail the statement that stores it -- and in
    the telemetry spool that statement is one ``INSERT`` for a stripe of up to
    64 requests from unrelated clusters. Refusing it at the edge with a 422
    (terminal for the sink, which dead-letters it) keeps one node's kernel log
    line from failing 63 other requests every collection cycle.
    """


def _contains_nul(value: object) -> bool:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            if "\x00" in current:
                return True
        elif isinstance(current, dict):
            stack.extend(current.keys())
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return False


def inflate_bounded(body: bytes, max_bytes: int) -> bytes:
    """Inflate a gzip body, refusing to allocate more than ``max_bytes`` for it.

    ``gzip.decompress`` places no bound on its output, and the ``len(body) >
    max_bytes`` check downstream only runs once the whole result is already
    resident. A few hundred kilobytes of zeros inflate to gigabytes, so a caller
    inside the authenticated wire limit could exhaust the ingress replica's
    memory. ``_declared_oversize`` did not help, because it deliberately skipped
    compressed bodies entirely.

    ``max_length`` caps a single ``decompress`` call, so a body whose output fits
    is produced in one call and one that does not stops at the cap, leaving the
    surplus input in ``unconsumed_tail`` or the stream unfinished. Either way the
    allocation stops at ``max_bytes + 1`` and we raise instead of continuing.

    Exception choice is load-bearing. Both call sites (``middleware/auth.py`` and
    ``middleware/dispatch.py``) map ``OverflowError`` to 413 and
    ``OSError``/``EOFError``/``ValueError`` to 400, and let everything else become
    a 500. ``zlib.error`` derives from ``Exception`` and neither, so a truncated
    or corrupt gzip body would turn a client mistake into a server error — hence
    the translation to ``ValueError`` here. ``gzip.decompress`` raised
    ``BadGzipFile`` (an ``OSError``) and ``EOFError`` for these cases, so the
    responses are unchanged from before.
    """

    inflater = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        decoded = inflater.decompress(body, max_length=max_bytes + 1)
    except zlib.error as exc:
        raise ValueError("gzip request body is not decodable") from exc
    if len(decoded) > max_bytes or inflater.unconsumed_tail:
        raise OverflowError("processor request body is too large")
    if not inflater.eof:
        # All input consumed, output within budget, stream never finished: the
        # body was cut short. ``gzip.decompress`` reported this as EOFError.
        raise EOFError("gzip request body ended before the stream finished")
    if inflater.unused_data:
        # A member decoded cleanly and bytes remain. ``gzip.decompress`` would
        # decode further concatenated members; a single ``decompressobj`` stops
        # at the first one. Rejecting is the only option that neither reads an
        # unbounded number of members nor silently drops the caller's data.
        raise ValueError("gzip request body has trailing data after the stream")
    return decoded


@dataclass(frozen=True)
class PostgresPoolCapacity:
    """What this role can ask of its connection pool at full load (F-E5).

    ``demand_by_consumer`` is every thread family that checks a connection out
    of the pool; ``unpooled_connections`` are the LISTEN connections opened
    beside it, which count against Aurora's ``max_connections`` but never
    against the pool, so they are reported and kept out of the ratio.
    """

    pool_max: int
    demand_by_consumer: dict[str, int]
    unpooled_connections: int

    # Connections the steady consumers must leave free: one for the
    # ``/metrics`` scrape, one for a route that goes straight to the store
    # (heartbeat, receipt poll). An estimate that came out exactly equal to
    # ``pool_max`` used to pass, and the registry refresh -- the readiness
    # criterion -- then queued 10 s behind saturated store threads (A-5).
    REQUIRED_HEADROOM = 2

    @property
    def demand(self) -> int:
        return sum(self.demand_by_consumer.values())

    @property
    def oversubscription_ratio(self) -> float:
        return self.demand / self.pool_max if self.pool_max > 0 else float("inf")

    @property
    def headroom(self) -> int:
        """Connections left once every steady consumer holds one."""

        return self.pool_max - self.demand

    @property
    def has_headroom(self) -> bool:
        return self.headroom >= self.REQUIRED_HEADROOM


@dataclass
class AdmissionRuntime:
    max_queue_depth: int
    max_cluster_queue_depth: int
    fault_reserved_depth: int
    fault_reserved_cluster_depth: int
    retry_after_seconds: int
    response_timeout_seconds: float
    max_request_bytes: int
    global_admission_guard: int
    admission_rejections: dict[str, int]
    admission_rejections_by_path: dict[str, int]
    dispatch_state: ProcessorDispatchState
    queue_bypass_paths: frozenset[str]
    queue_bypass_enabled: bool
    queue_bypasses_by_path: dict[str, int]
    spool_enabled: bool
    spool_max_depth: int
    spool_max_cluster_depth: int
    spool_max_item_bytes: int
    spool_rejections: dict[str, int]
    spool_admitted_by_path: dict[str, int]
    decode_rejections: dict[str, int]
    store_io: AsyncStoreExecutor
    decode_io: AsyncStoreExecutor
    fault_store_io: AsyncStoreExecutor
    evidence_store_io: AsyncStoreExecutor
    fault_decode_io: AsyncStoreExecutor
    spool_store_io: AsyncStoreExecutor
    admission_batcher: _ProcessorAdmissionBatcher
    fault_batcher: _ProcessorAdmissionBatcher
    evidence_batcher: _ProcessorAdmissionBatcher
    spool_batcher: _ProcessorAdmissionBatcher
    decode_json_body: object
    bounded_io_endpoint: object


class AdmissionRuntimeFactory:
    def __init__(self, context, processor) -> None:
        self.context = context
        self.processor = processor

    def build(self) -> AdmissionRuntime:
        limits = self.limits()
        executors = self._executors(limits)
        # Published on the context the way the periodic runner is (F-L1), so
        # /metrics can export the ratio; a startup WARNING is invisible during a
        # rolling release.
        self.context.postgres_pool_capacity = self.pool_capacity(limits)
        batchers = self._batchers(limits, executors)
        return AdmissionRuntime(
            **limits,
            **executors,
            **batchers,
            decode_json_body=self._decode_json_body(limits["max_request_bytes"]),
            bounded_io_endpoint=self._bounded_endpoint(
                executors["store_io"],
                retry_after_seconds=limits["retry_after_seconds"],
            ),
        )

    def limits(self) -> dict:
        max_depth = int(os.getenv("GPU_FAULT_PROCESSOR_MAX_QUEUE_DEPTH", "10000"))
        max_cluster = int(
            os.getenv("GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH", "1000")
        )
        reserved = int(
            os.getenv(
                "GPU_FAULT_PROCESSOR_FAULT_RESERVED_QUEUE_DEPTH",
                str((max_depth * 2) // 5),
            )
        )
        reserved_cluster = int(
            os.getenv(
                "GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH",
                str((max_cluster * 2) // 5),
            )
        )
        max_bytes = int(
            os.getenv(
                "GPU_FAULT_PROCESSOR_MAX_REQUEST_BYTES",
                str(16 * 1024 * 1024),
            )
        )
        guard = int(
            os.getenv(
                "GPU_FAULT_PROCESSOR_GLOBAL_ADMISSION_GUARD",
                str(min(256, max_depth)),
            )
        )
        retry = int(os.getenv("GPU_FAULT_PROCESSOR_RETRY_AFTER_SECONDS", "2"))
        # Parsed here rather than inside the dispatch loop: a malformed value
        # used to raise ValueError per request, so the deployment looked healthy
        # and every synchronous call returned 500 instead of failing at startup.
        response_timeout = float(
            os.getenv("GPU_FAULT_PROCESSOR_RESPONSE_TIMEOUT_SECONDS", "115")
        )
        if (
            max_depth <= 0
            or max_cluster <= 0
            or not 0 <= reserved <= max_depth
            or not 0 <= reserved_cluster <= max_cluster
            or max_bytes <= 0
            or not 0 <= guard <= max_depth
            or retry <= 0
            or response_timeout <= 0
        ):
            raise RuntimeError(
                "processor queue limits, fault reserves, retry delay, and "
                "response timeout are invalid"
            )
        # The declared topology (0 = not declared). A correlated whole-cluster
        # fault is one fault-priority request per node on that many lanes at
        # once; a reserve that cannot hold it turns the fault lane into the
        # first thing that returns 429 (性能压测验收方案 §13.4).
        largest_cluster_nodes = int(
            os.getenv("GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT", "0")
        )
        managed_nodes = int(os.getenv("GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT", "0"))
        if largest_cluster_nodes < 0 or managed_nodes < 0:
            raise RuntimeError("declared capacity node counts cannot be negative")
        if 0 < largest_cluster_nodes and reserved_cluster < largest_cluster_nodes:
            raise RuntimeError(
                f"fault-reserved cluster depth {reserved_cluster} cannot hold one "
                "correlated whole-cluster fault wave of "
                f"{largest_cluster_nodes} nodes"
            )
        spool_enabled = env_bool("GPU_FAULT_TELEMETRY_SPOOL", False)
        spool_item_bytes = int(
            os.getenv(
                "GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES",
                str(4 * 1024 * 1024),
            )
        )
        partitions = int(
            os.getenv("GPU_FAULT_TELEMETRY_SPOOL_ADMISSION_PARTITIONS", "4")
        )
        if spool_enabled and not 0 < spool_item_bytes <= max_bytes:
            raise ValueError(
                "telemetry spool item byte limit must be positive "
                "and not exceed the request byte limit"
            )
        if spool_enabled and partitions <= 0:
            raise ValueError(
                "telemetry spool admission partition count must be positive"
            )
        if spool_enabled and not os.getenv("GPU_FAULT_STORE_URL", "").startswith(
            "postgres"
        ):
            LOGGER.warning(
                "telemetry spool is enabled on a non-Postgres store; "
                "spooled samples will not survive a restart"
            )
        if (
            spool_enabled
            and self.processor is not None
            and spool_item_bytes + ProcessorCoordinator._TELEMETRY_BATCH_ENVELOPE_BYTES
            > self.processor.telemetry_spool_replay_batch_max_bytes
        ):
            raise ValueError(
                "telemetry spool item plus replay envelope exceeds "
                "the replay batch byte limit"
            )
        return {
            "max_queue_depth": max_depth,
            "max_cluster_queue_depth": max_cluster,
            "fault_reserved_depth": reserved,
            "fault_reserved_cluster_depth": reserved_cluster,
            "retry_after_seconds": retry,
            "response_timeout_seconds": response_timeout,
            "max_request_bytes": max_bytes,
            "global_admission_guard": guard,
            "admission_rejections": {
                "global": 0,
                "cluster": 0,
                "global_reserved": 0,
                "cluster_reserved": 0,
            },
            "admission_rejections_by_path": {},
            "dispatch_state": ProcessorDispatchState(),
            "queue_bypass_paths": frozenset(
                path
                for path, channel in CHANNEL_REGISTRY.items()
                if channel.snapshot_bypass
            ),
            "queue_bypass_enabled": env_bool(
                "GPU_FAULT_PROCESSOR_SNAPSHOT_BYPASS", False
            ),
            "queue_bypasses_by_path": {},
            "spool_enabled": spool_enabled,
            "spool_max_depth": int(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_MAX_DEPTH",
                    str(max_depth),
                )
            ),
            "spool_max_cluster_depth": int(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_MAX_CLUSTER_DEPTH",
                    str(max_cluster),
                )
            ),
            "spool_max_item_bytes": spool_item_bytes,
            "spool_rejections": {"global": 0, "cluster": 0},
            "spool_admitted_by_path": {},
            # Bodies refused by the decoder for reasons that are neither size
            # nor syntax: today only a NUL escape (E-3). Rendered on /metrics
            # as ``gpu_fault_ingress_decode_rejections_total{reason=...}``.
            "decode_rejections": {"nul": 0},
            "_spool_partitions": partitions,
        }

    def _executors(self, limits: dict) -> dict:
        timeout = float(os.getenv("GPU_FAULT_STORE_IO_ADMISSION_TIMEOUT_SECONDS", "2"))
        executors = {
            "store_io": _executor(
                "GPU_FAULT_STORE_IO_WORKERS",
                "8",
                "GPU_FAULT_STORE_IO_MAX_IN_FLIGHT",
                "64",
                timeout,
            ),
            "decode_io": _executor(
                "GPU_FAULT_INGRESS_DECODE_WORKERS",
                "8",
                "GPU_FAULT_INGRESS_DECODE_MAX_IN_FLIGHT",
                "128",
                float(os.getenv("GPU_FAULT_INGRESS_DECODE_TIMEOUT_SECONDS", "5")),
                prefix="gpu-fault-request-decode",
            ),
            "fault_store_io": _executor(
                "GPU_FAULT_FAULT_STORE_IO_WORKERS",
                "8",
                "GPU_FAULT_FAULT_STORE_IO_MAX_IN_FLIGHT",
                "256",
                timeout,
                prefix="gpu-fault-store-io-fault",
            ),
            "evidence_store_io": _executor(
                "GPU_FAULT_EVIDENCE_STORE_IO_WORKERS",
                "4",
                "GPU_FAULT_EVIDENCE_STORE_IO_MAX_IN_FLIGHT",
                "256",
                timeout,
                prefix="gpu-fault-store-io-evidence",
            ),
            "fault_decode_io": _executor(
                "GPU_FAULT_FAULT_DECODE_WORKERS",
                "8",
                "GPU_FAULT_FAULT_DECODE_MAX_IN_FLIGHT",
                "256",
                float(os.getenv("GPU_FAULT_INGRESS_DECODE_TIMEOUT_SECONDS", "5")),
                prefix="gpu-fault-request-decode-fault",
            ),
            "spool_store_io": _executor(
                "GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS",
                "16",
                "GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_MAX_IN_FLIGHT",
                "512",
                timeout,
                prefix="gpu-fault-store-io-spool-admission",
            ),
        }
        return executors

    @staticmethod
    def pool_capacity(limits: dict) -> PostgresPoolCapacity | None:
        """Estimate the role's connection demand against its pool (F-E5).

        The guard used to count only the Store I/O threads: 8 in the worker
        role against a pool of 8, so it never warned in the one role where the
        processor threads, the workflow dispatcher and the periodic runner are
        the consumers that actually queue on checkout.

        Every entry is a thread family that exists in that role (see
        ``lifespan.py`` / ``lifespan_workers.py``), sized as the coordinator
        sizes it: the processor pools come from ``ProcessorPoolSettings``,
        not from the ``GPU_FAULT_PROCESSOR_WORKERS`` budget they are carved
        from (G-7: 24 declared, 14 started). Per-request lease renewal
        threads are deliberately not listed -- they hold a connection for one
        statement every few seconds and are bounded by the pool threads that
        spawn them -- which is part of why the baseline keeps two spare.
        """

        if not os.getenv("GPU_FAULT_STORE_URL", "").startswith("postgres"):
            return None
        env = os.getenv
        pool_max = int(env("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "8"))
        service_role = env("GPU_FAULT_SERVICE_ROLE", "all").strip().lower()
        spool_enabled = bool(limits["spool_enabled"])
        queued_processor = (
            env("GPU_FAULT_PROCESSOR_MODE", "direct").strip().lower() == "active-active"
        )
        regional = env("GPU_FAULT_DEPLOYMENT_MODE", "").strip().lower() == "regional"
        background_services = service_role in {"all", "worker"}
        try:
            pools = ProcessorPoolSettings.from_environment()
        except ValueError:
            # An invalid split fails the processor factory in the roles that
            # start one; an estimate must not be what fails a role that does
            # not. Fall back to the budget, the pre-G-7 reading.
            pools = ProcessorPoolSettings(
                worker_count=max(1, int(env("GPU_FAULT_PROCESSOR_WORKERS", "4")))
            )
        demand: dict[str, int] = {}
        if service_role in {"all", "ingress", "worker"}:
            demand["store_io_general"] = int(env("GPU_FAULT_STORE_IO_WORKERS", "8"))
        if service_role in {"all", "ingress"}:
            demand["store_io_fault"] = int(env("GPU_FAULT_FAULT_STORE_IO_WORKERS", "8"))
            demand["store_io_evidence"] = int(
                env("GPU_FAULT_EVIDENCE_STORE_IO_WORKERS", "4")
            )
            if spool_enabled:
                demand["store_io_spool"] = int(
                    env("GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS", "16")
                )
        if background_services and queued_processor:
            # ``gpu-fault-processor-inbox`` claims; the pools execute.
            demand["processor_claim_loop"] = 1
            demand["processor_workers"] = sum(pools.worker_counts().values())
        if background_services and env_bool(
            "GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER", True
        ):
            demand["workflow_dispatcher"] = int(
                env("GPU_FAULT_WORKFLOW_DISPATCHER_WORKERS", "8")
            )
        if background_services:
            demand["periodic_services"] = 1
            demand["xid_correlation"] = 1
            demand["notification_dispatcher"] = 1
            if queued_processor:
                demand["processor_diagnostics"] = 1
        # Started by ``lifespan.py`` in every role, 30 s lease + snapshot read.
        demand["collector_metrics_snapshot"] = 1
        if regional:
            # 1 s ``refresh_once``: two reads and a member write, in every
            # role; its failure is what fails ``/healthz``.
            demand["regional_registry"] = 1
        if (
            spool_enabled
            and queued_processor
            and service_role in {"all", "spool-worker"}
        ):
            demand["telemetry_spool_replay"] = int(
                env(
                    "GPU_FAULT_TELEMETRY_SPOOL_WORKERS",
                    str(pools.default_pool * 2),
                )
            )
        unpooled = 0
        if queued_processor and background_services:
            unpooled += 1  # LISTEN gpu_fault_processor_queue
        if (
            queued_processor
            and spool_enabled
            and service_role in {"all", "spool-worker"}
        ):
            unpooled += 1  # LISTEN gpu_fault_telemetry_spool
        estimate = PostgresPoolCapacity(
            pool_max=pool_max,
            demand_by_consumer=demand,
            unpooled_connections=unpooled,
        )
        if not estimate.has_headroom:
            LOGGER.warning(
                "postgres pool max %s leaves fewer than %s spare connections "
                "beyond the %s the %s role can hold at once (%s); the registry "
                "refresh and /metrics will queue on checkout",
                pool_max,
                PostgresPoolCapacity.REQUIRED_HEADROOM,
                estimate.demand,
                service_role,
                ", ".join(f"{name}={count}" for name, count in demand.items()),
            )
        return estimate

    def _batchers(self, limits: dict, executors: dict) -> dict:
        common = {
            "max_depth": limits["max_queue_depth"],
            "max_cluster_depth": limits["max_cluster_queue_depth"],
            "reserved_fault_depth": limits["fault_reserved_depth"],
            "reserved_cluster_fault_depth": (limits["fault_reserved_cluster_depth"]),
            "global_admission_guard": limits["global_admission_guard"],
        }
        admission = self._queue_batcher(
            executors["store_io"], common, "", "64", "16", "1.0"
        )
        fault = self._queue_batcher(
            executors["fault_store_io"],
            common,
            "FAULT_",
            "64",
            str(int(os.getenv("GPU_FAULT_FAULT_STORE_IO_WORKERS", "8"))),
            "0",
            label="fault admission",
        )
        evidence = self._queue_batcher(
            executors["evidence_store_io"],
            common,
            "EVIDENCE_",
            "32",
            "4",
            "0",
            label="evidence admission",
        )
        size = int(os.getenv("GPU_FAULT_TELEMETRY_SPOOL_BATCH_SIZE", "64"))

        def admit(items):
            return self.context.store.try_spool_telemetry_requests(
                items,
                max_depth=limits["spool_max_depth"],
                max_cluster_depth=limits["spool_max_cluster_depth"],
            )

        spool = _ProcessorAdmissionBatcher(
            self.context.store,
            executors["spool_store_io"],
            max_depth=limits["spool_max_depth"],
            max_cluster_depth=limits["spool_max_cluster_depth"],
            reserved_fault_depth=0,
            reserved_cluster_fault_depth=0,
            global_admission_guard=0,
            max_batch_size=size,
            max_flush_groups=int(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_BATCH_GROUPS",
                    str(limits["_spool_partitions"]),
                )
            ),
            flush_delay_seconds=float(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_BATCH_DELAY_SECONDS",
                    "0.002",
                )
            ),
            projection_margin=float(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_PROJECTION_MARGIN",
                    "1.0",
                )
            ),
            admit_batch=admit,
            label="telemetry spool admission",
            scope_key=_StripedAdmissionScope(
                batch_size=size,
                partitions=limits["_spool_partitions"],
            ),
        )
        limits.pop("_spool_partitions")
        return {
            "admission_batcher": admission,
            "fault_batcher": fault,
            "evidence_batcher": evidence,
            "spool_batcher": spool,
        }

    def _queue_batcher(
        self,
        executor,
        common,
        prefix,
        size,
        groups,
        margin,
        *,
        label="processor admission",
    ):
        base = "GPU_FAULT_PROCESSOR_"
        return _ProcessorAdmissionBatcher(
            self.context.store,
            executor,
            **common,
            max_batch_size=int(os.getenv(f"{base}{prefix}ADMISSION_BATCH_SIZE", size)),
            max_flush_groups=int(
                os.getenv(f"{base}{prefix}ADMISSION_BATCH_GROUPS", groups)
            ),
            flush_delay_seconds=float(
                os.getenv(
                    f"{base}{prefix}ADMISSION_BATCH_DELAY_SECONDS",
                    "0.002",
                )
            ),
            projection_margin=float(
                os.getenv(
                    f"{base}{prefix}ADMISSION_PROJECTION_MARGIN",
                    margin,
                )
            ),
            label=label,
        )

    @staticmethod
    def _decode_json_body(max_bytes: int):
        def decode(body: bytes, content_encoding: str):
            if content_encoding.lower() == "gzip":
                body = inflate_bounded(body, max_bytes)
            if len(body) > max_bytes:
                raise OverflowError("processor request body is too large")
            if not body:
                return body, {}
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("JSON request body must be an object")
            # A raw NUL byte is a control character inside a string and
            # ``json.loads`` already rejects it; only the escape gets
            # through. The bytes search is the cheap gate, the walk
            # confirms it, so ``"\\\\u0000"`` (an escaped backslash) does
            # not cost a legitimate log line its batch.
            if b"\\u0000" in body and _contains_nul(payload):
                raise NulInRequestBody(
                    "JSON request body contains a NUL character (\\u0000), "
                    "which PostgreSQL jsonb cannot store"
                )
            return body, payload

        return decode

    @staticmethod
    def _bounded_endpoint(
        store_io: AsyncStoreExecutor, *, retry_after_seconds: int = 2
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        retry_after = {"Retry-After": str(retry_after_seconds)}

        def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(function)
            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                report_processor_replay_phase(f"store_io_admission:{function.__name__}")

                def invoke() -> Any:
                    report_processor_replay_phase(f"handler:{function.__name__}")
                    return function(*args, **kwargs)

                try:
                    result = await store_io.run(invoke)
                    report_processor_replay_phase(
                        f"response_serialization:{function.__name__}"
                    )
                    return result
                except StoreIoCapacityExceeded as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "request deadline exceeded"
                            if isinstance(exc, RequestDeadlineExceeded)
                            else "store I/O capacity exceeded"
                        ),
                        headers=retry_after,
                    ) from exc

            return wrapped

        return decorate


def _executor(
    workers_name,
    workers_default,
    in_flight_name,
    in_flight_default,
    timeout,
    *,
    prefix=None,
):
    kwargs = {
        "workers": int(os.getenv(workers_name, workers_default)),
        "max_in_flight": int(os.getenv(in_flight_name, in_flight_default)),
        "admission_timeout_seconds": timeout,
    }
    if prefix is not None:
        kwargs["thread_name_prefix"] = prefix
    return AsyncStoreExecutor(**kwargs)
