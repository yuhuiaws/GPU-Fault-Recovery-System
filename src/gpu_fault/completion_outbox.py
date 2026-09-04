from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from gpu_fault.collectors.sinks import EventSink, HttpEventSink

CRITICAL_COMPLETION_PATHS = frozenset(
    {
        "/v1/attempts/failure-detected",
        "/v1/attempts/terminal",
    }
)
WORKLOAD_OBSERVATION_PATH = "/v1/workload-observations"


class CompletionOutboxFull(RuntimeError):
    pass


class KubernetesCompletionOutbox:
    """ConfigMap-backed write-ahead delivery for critical watcher events."""

    DATA_KEY = "events.json"
    ATTEMPTS_KEY = "active-attempts.json"

    def __init__(
        self,
        core_api: Any,
        sink: EventSink,
        *,
        namespace: str = "gpu-fault-system",
        name: str = "gpu-fault-completion-watcher-outbox",
        max_records: int = 256,
        max_bytes: int = 900_000,
        replay_batch_size: int = 32,
    ) -> None:
        if max_records < 1 or max_bytes < 1024 or replay_batch_size < 1:
            raise ValueError("completion outbox bounds must be positive")
        self.core_api = core_api
        self.sink = sink
        self.namespace = namespace
        self.name = name
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.replay_batch_size = replay_batch_size
        self._attempt_digests: dict[str, str] = {}

    @staticmethod
    def _record_key(path: str, payload: dict[str, Any]) -> str:
        cluster_id = str(payload.get("cluster_id") or "")
        attempt_id = str(payload.get("attempt_id") or "")
        if not cluster_id or not attempt_id:
            raise ValueError(
                "critical completion event requires cluster_id and attempt_id"
            )
        return f"{cluster_id}/{attempt_id}/{path.rsplit('/', 1)[-1]}"

    @staticmethod
    def _data(value: Any) -> dict[str, str]:
        raw = value.get("data", {}) if isinstance(value, dict) else value.data
        return dict(raw or {})

    @staticmethod
    def _resource_version(value: Any) -> str | None:
        metadata = (
            value.get("metadata", {}) if isinstance(value, dict) else value.metadata
        )
        if isinstance(metadata, dict):
            return metadata.get("resourceVersion") or metadata.get("resource_version")
        return getattr(metadata, "resource_version", None)

    def _read_data(self) -> tuple[dict[str, str], str | None]:
        value = self.core_api.read_namespaced_config_map(self.name, self.namespace)
        return self._data(value), self._resource_version(value)

    def _events(self, data: dict[str, str]) -> list[dict[str, Any]]:
        document = json.loads(data.get(self.DATA_KEY, "[]"))
        if not isinstance(document, list) or any(
            not isinstance(item, dict) for item in document
        ):
            raise RuntimeError("completion outbox ConfigMap contains invalid JSON")
        return document

    def _attempts(self, data: dict[str, str]) -> dict[str, dict[str, Any]]:
        document = json.loads(data.get(self.ATTEMPTS_KEY, "{}"))
        if not isinstance(document, dict) or any(
            not isinstance(key, str) or not isinstance(item, dict)
            for key, item in document.items()
        ):
            raise RuntimeError("completion attempt state contains invalid JSON")
        return document

    def _read(self) -> tuple[list[dict[str, Any]], str | None]:
        data, resource_version = self._read_data()
        return self._events(data), resource_version

    def _write_data(
        self,
        data: dict[str, str],
        resource_version: str | None,
    ) -> None:
        records = self._events(data)
        attempts = self._attempts(data)
        payload_bytes = sum(
            len(key.encode()) + len(value.encode()) for key, value in data.items()
        )
        if (
            len(records) > self.max_records
            or len(attempts) > self.max_records
            or payload_bytes > self.max_bytes
        ):
            raise CompletionOutboxFull(
                "completion outbox or active attempt state exceeds its bounds"
            )
        self.core_api.replace_namespaced_config_map(
            self.name,
            self.namespace,
            {
                "metadata": {
                    "name": self.name,
                    **(
                        {"resourceVersion": resource_version}
                        if resource_version is not None
                        else {}
                    ),
                },
                "data": data,
            },
        )

    def _mutate_data(
        self,
        function: Callable[[dict[str, str]], dict[str, str]],
    ) -> None:
        for attempt in range(3):
            data, resource_version = self._read_data()
            try:
                self._write_data(function(data), resource_version)
                return
            except Exception as exc:
                if getattr(exc, "status", None) != 409 or attempt == 2:
                    raise

    def _mutate(
        self,
        function: Callable[
            [list[dict[str, Any]]],
            list[dict[str, Any]],
        ],
    ) -> None:
        def update(data: dict[str, str]) -> dict[str, str]:
            result = dict(data)
            result[self.DATA_KEY] = json.dumps(
                function(self._events(data)),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            return result

        self._mutate_data(update)

    def _append(self, path: str, payload: dict[str, Any]) -> str:
        key = self._record_key(path, payload)

        def append(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if any(item.get("key") == key for item in records):
                return records
            return [*records, {"key": key, "path": path, "payload": payload}]

        self._mutate(append)
        return key

    def _upsert_latest(self, path: str, payload: dict[str, Any]) -> str:
        key = self._record_key(path, payload)

        def upsert(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            retained = [item for item in records if item.get("key") != key]
            return [*retained, {"key": key, "path": path, "payload": payload}]

        self._mutate(upsert)
        return key

    def _remove(self, key: str) -> None:
        self._mutate(
            lambda records: [item for item in records if item.get("key") != key]
        )

    @staticmethod
    def _attempt_key(payload: dict[str, Any]) -> str:
        cluster_id = str(payload.get("cluster_id") or "")
        attempt_id = str(payload.get("attempt_id") or "")
        if not cluster_id or not attempt_id:
            raise ValueError("attempt observation requires cluster_id and attempt_id")
        return f"{cluster_id}/{attempt_id}"

    @staticmethod
    def _attempt_digest(payload: dict[str, Any]) -> str:
        structural = dict(payload)
        structural.pop("observed_at", None)
        return hashlib.sha256(
            json.dumps(
                structural,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()

    def load_attempt_observations(self) -> list[dict[str, Any]]:
        data, _resource_version = self._read_data()
        attempts = self._attempts(data)
        self._attempt_digests = {
            key: self._attempt_digest(payload) for key, payload in attempts.items()
        }
        return [dict(attempts[key]) for key in sorted(attempts)]

    def save_attempt_observation(self, payload: dict[str, Any]) -> bool:
        key = self._attempt_key(payload)
        digest = self._attempt_digest(payload)
        if self._attempt_digests.get(key) == digest:
            return False

        def update(data: dict[str, str]) -> dict[str, str]:
            attempts = self._attempts(data)
            attempts[key] = dict(payload)
            result = dict(data)
            result[self.ATTEMPTS_KEY] = json.dumps(
                attempts,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            return result

        self._mutate_data(update)
        self._attempt_digests[key] = digest
        return True

    def remove_attempt_observation(self, payload: dict[str, Any]) -> None:
        key = self._attempt_key(payload)

        def update(data: dict[str, str]) -> dict[str, str]:
            attempts = self._attempts(data)
            if key not in attempts:
                return data
            attempts.pop(key)
            result = dict(data)
            result[self.ATTEMPTS_KEY] = json.dumps(
                attempts,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            return result

        self._mutate_data(update)
        self._attempt_digests.pop(key, None)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == WORKLOAD_OBSERVATION_PATH:
            try:
                return self.sink.post(path, payload)
            except Exception:
                self._upsert_latest(path, payload)
                raise
        if path not in CRITICAL_COMPLETION_PATHS:
            return self.sink.post(path, payload)
        key = self._append(path, payload)
        result = self.sink.post(path, payload)
        self._remove(key)
        return result

    def replay(self) -> int:
        records, _resource_version = self._read()
        replayed = 0
        for record in records[: self.replay_batch_size]:
            path = str(record["path"])
            payload = dict(record["payload"])
            self.sink.post(path, payload)
            if path == WORKLOAD_OBSERVATION_PATH and str(
                payload.get("workload_phase") or ""
            ) not in {"PENDING", "RUNNING"}:
                self.remove_attempt_observation(payload)
            self._remove(str(record["key"]))
            replayed += 1
        return replayed

    def depth(self) -> int:
        records, _resource_version = self._read()
        return len(records)


def completion_sink_from_environment(core_api: Any) -> KubernetesCompletionOutbox:
    return KubernetesCompletionOutbox(
        core_api,
        HttpEventSink(
            os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
            bearer_token=os.getenv("GPU_FAULT_CONTROL_PLANE_TOKEN"),
            timeout_seconds=10,
            processor_receipt_timeout_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_RECEIPT_TIMEOUT_SECONDS", "120")
            ),
            processor_receipt_poll_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_RECEIPT_POLL_SECONDS", "0.25")
            ),
        ),
        namespace=os.getenv("GPU_FAULT_NAMESPACE", "gpu-fault-system"),
        max_records=int(os.getenv("GPU_FAULT_COMPLETION_OUTBOX_MAX_RECORDS", "256")),
        max_bytes=int(os.getenv("GPU_FAULT_COMPLETION_OUTBOX_MAX_BYTES", "900000")),
        replay_batch_size=int(
            os.getenv("GPU_FAULT_COMPLETION_OUTBOX_REPLAY_BATCH_SIZE", "32")
        ),
    )


def replay_completion_outbox(sink: Any, logger: logging.Logger) -> None:
    replay = getattr(sink, "replay", None)
    if replay is None:
        return
    try:
        replay()
    except Exception:
        logger.exception("cannot replay Completion Watcher outbox")
