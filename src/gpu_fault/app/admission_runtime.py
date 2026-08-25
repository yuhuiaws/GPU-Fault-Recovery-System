from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import gzip
import json
import logging
import os

from fastapi import HTTPException

from gpu_fault.app.admission import (
    _ProcessorAdmissionBatcher,
    _StripedAdmissionScope,
)
from gpu_fault.app.runtime import ProcessorDispatchState
from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.channel_registry import CHANNEL_REGISTRY
from gpu_fault.processor import ProcessorCoordinator
from gpu_fault.processor_diagnostics import report_processor_replay_phase


LOGGER = logging.getLogger(__name__)


@dataclass
class AdmissionRuntime:
    max_queue_depth: int
    max_cluster_queue_depth: int
    fault_reserved_depth: int
    fault_reserved_cluster_depth: int
    retry_after_seconds: int
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
        limits = self._limits()
        executors = self._executors(limits)
        batchers = self._batchers(limits, executors)
        return AdmissionRuntime(
            **limits,
            **executors,
            **batchers,
            decode_json_body=self._decode_json_body(limits["max_request_bytes"]),
            bounded_io_endpoint=self._bounded_endpoint(executors["store_io"]),
        )

    def _limits(self) -> dict:
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
        if (
            max_depth <= 0
            or max_cluster <= 0
            or not 0 <= reserved <= max_depth
            or not 0 <= reserved_cluster <= max_cluster
            or max_bytes <= 0
            or not 0 <= guard <= max_depth
            or retry <= 0
        ):
            raise RuntimeError(
                "processor queue limits, fault reserves, and retry delay are invalid"
            )
        spool_enabled = _enabled("GPU_FAULT_TELEMETRY_SPOOL")
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
            "queue_bypass_enabled": _enabled("GPU_FAULT_PROCESSOR_SNAPSHOT_BYPASS"),
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
        self._warn_pool_capacity(limits)
        return executors

    @staticmethod
    def _warn_pool_capacity(limits: dict) -> None:
        if not os.getenv("GPU_FAULT_STORE_URL", "").startswith("postgres"):
            return
        pool_max = int(os.getenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "8"))
        service_role = os.getenv("GPU_FAULT_SERVICE_ROLE", "all").strip().lower()
        general = int(os.getenv("GPU_FAULT_STORE_IO_WORKERS", "8"))
        fault = int(os.getenv("GPU_FAULT_FAULT_STORE_IO_WORKERS", "8"))
        evidence = int(os.getenv("GPU_FAULT_EVIDENCE_STORE_IO_WORKERS", "4"))
        spool = int(os.getenv("GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS", "16"))
        active = general if service_role in {"all", "ingress", "worker"} else 0
        if service_role in {"all", "ingress"}:
            active += fault + evidence
            if limits["spool_enabled"]:
                active += spool
        if pool_max < active:
            LOGGER.warning(
                "postgres pool max %s is smaller than %s active "
                "Store I/O workers; callers will queue on checkout",
                pool_max,
                active,
            )

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
                body = gzip.decompress(body)
            if len(body) > max_bytes:
                raise OverflowError("processor request body is too large")
            if not body:
                return body, {}
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("JSON request body must be an object")
            return body, payload

        return decode

    @staticmethod
    def _bounded_endpoint(store_io):
        def decorate(function):
            @wraps(function)
            async def wrapped(*args, **kwargs):
                report_processor_replay_phase(f"store_io_admission:{function.__name__}")

                def invoke():
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
                        detail="store I/O capacity exceeded",
                        headers={"Retry-After": "2"},
                    ) from exc

            return wrapped

        return decorate


def _enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


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
