"""Emergency workload stop with training-log capture (Completion Watcher).

``KubernetesWorkloadStopper`` is the last line of defence when the control
plane has not accepted a failure event: it captures and archives the training
logs, annotates the Pods and suspends the workload objects. It holds no
controller state. Split out of ``completion_controller`` as a pure move (F6b).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlparse

from gpu_fault.completion_observation import TERMINATION_INCIDENT_ANNOTATION
from gpu_fault.completion_pod_parsing import (
    ATTEMPT_LABEL,
    MANAGED_LABEL,
    TRAINING_CONTAINER_ANNOTATION,
    WORKLOAD_LOG_SNAPSHOT_ANNOTATION,
    CompletionControllerError,
)

# Borrowed on purpose: every line below used to be logged by
# ``gpu_fault.completion_controller`` and the log format prints the logger
# name, so the split must not rename what operators grep for.
LOGGER = logging.getLogger("gpu_fault.completion_controller")


class KubernetesWorkloadStopper:
    """Emergency fallback when control-plane containment is unavailable."""

    def __init__(
        self,
        batch_api,
        custom_api,
        core_api=None,
        *,
        workload_log_tail_lines: int = 2000,
        workload_log_max_bytes: int = 262144,
        workload_log_s3_uri: str | None = None,
        workload_log_s3_max_bytes: int = 104857600,
        workload_log_annotation_tail_bytes: int = 8192,
        workload_log_uploader=None,
        workload_log_timeout_seconds: float = 30.0,
    ) -> None:
        if workload_log_timeout_seconds <= 0:
            raise CompletionControllerError(
                "workload log capture timeout must be positive"
            )
        self.batch = batch_api
        self.custom = custom_api
        self.core = core_api
        self.workload_log_timeout_seconds = workload_log_timeout_seconds
        self.workload_log_tail_lines = workload_log_tail_lines
        self.workload_log_max_bytes = workload_log_max_bytes
        uri = workload_log_s3_uri
        self.workload_log_s3_uri = uri.rstrip("/") if uri else None
        self.workload_log_s3_max_bytes = workload_log_s3_max_bytes
        self.workload_log_annotation_tail_bytes = workload_log_annotation_tail_bytes
        self.workload_log_uploader = workload_log_uploader
        # Liveness (M1): the controller points this at its progress stamp, so
        # each finished capture is a step and not the whole batch.
        self.note_progress: Callable[[], None] = lambda: None

    def stop(
        self,
        workload_ids: tuple[str, ...],
        attempt_id: str,
        incident_id: str | None = None,
        *,
        capture_logs: bool = True,
    ) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        if incident_id is not None and self.core is not None:
            items = self._attempt_pods(attempt_id)
            if capture_logs:
                snapshots = self.capture_logs(
                    attempt_id,
                    incident_id,
                    pods=items,
                    annotate=False,
                )
            snapshots_by_uid = {
                str(snapshot.get("pod_uid")): snapshot
                for snapshot in snapshots
                if snapshot.get("pod_uid")
            }
            for pod in items:
                name, namespace, pod_uid = self._pod_identity(pod)
                if not name:
                    raise CompletionControllerError(
                        "managed Pod selected for emergency stop has no metadata.name"
                    )
                pod_annotations = {
                    TERMINATION_INCIDENT_ANNOTATION: incident_id,
                    "gpu-fault.io/passive-stop-attempt": attempt_id,
                    "gpu-fault.io/passive-stop-mode": ("emergency-fallback"),
                }
                snapshot = snapshots_by_uid.get(pod_uid)
                if snapshot is not None and snapshot.get("record_id"):
                    pod_annotations[WORKLOAD_LOG_SNAPSHOT_ANNOTATION] = (
                        self._snapshot_annotation(snapshot)
                    )
                self.core.patch_namespaced_pod(
                    name,
                    namespace,
                    {"metadata": {"annotations": pod_annotations}},
                )
        for workload_id in dict.fromkeys(workload_ids):
            namespace, kind, name = self._parse(workload_id)
            body: dict[str, Any] = {
                "metadata": {
                    "annotations": {
                        "gpu-fault.io/passive-stop-attempt": attempt_id,
                        "gpu-fault.io/passive-stop-mode": ("emergency-fallback"),
                    }
                },
                "spec": (
                    {"runPolicy": {"suspend": True}}
                    if kind == "pytorchjob"
                    else {"suspend": True}
                ),
            }
            if kind == "job":
                self.batch.patch_namespaced_job(name, namespace, body)
                continue
            group, version, plural = {
                "pytorchjob": (
                    "kubeflow.org",
                    "v1",
                    "pytorchjobs",
                ),
                "jobset": (
                    "jobset.x-k8s.io",
                    "v1alpha2",
                    "jobsets",
                ),
            }[kind]
            self.custom.patch_namespaced_custom_object(
                group, version, namespace, plural, name, body
            )
        return snapshots

    def _attempt_pods(self, attempt_id: str) -> list[Any]:
        listing = self.core.list_pod_for_all_namespaces(
            label_selector=(f"{MANAGED_LABEL}=true,{ATTEMPT_LABEL}={attempt_id}")
        )
        return (
            listing.get("items", [])
            if isinstance(listing, dict)
            else list(getattr(listing, "items", []))
        )

    @staticmethod
    def _pod_identity(pod: Any) -> tuple[str, str, str]:
        metadata = (
            pod.get("metadata", {})
            if isinstance(pod, dict)
            else getattr(pod, "metadata", None)
        )
        name = (
            metadata.get("name")
            if isinstance(metadata, dict)
            else getattr(metadata, "name", None)
        )
        namespace = (
            metadata.get("namespace", "default")
            if isinstance(metadata, dict)
            else getattr(metadata, "namespace", "default")
        )
        uid = (
            metadata.get("uid")
            if isinstance(metadata, dict)
            else getattr(metadata, "uid", None)
        )
        return str(name or ""), str(namespace or "default"), str(uid or name or "")

    def capture_logs(
        self,
        attempt_id: str,
        incident_id: str,
        *,
        pods: list[Any] | None = None,
        annotate: bool = True,
    ) -> list[dict[str, Any]]:
        if self.core is None:
            return []
        items = list(pods) if pods is not None else self._attempt_pods(attempt_id)
        if not items:
            return []

        def capture(pod: Any) -> tuple[Any, dict[str, Any]]:
            name, namespace, _pod_uid = self._pod_identity(pod)
            try:
                snapshot = self._capture_workload_log(
                    pod,
                    namespace=namespace,
                    attempt_id=attempt_id,
                    incident_id=incident_id,
                )
            except Exception as exc:
                LOGGER.exception(
                    "cannot capture failure workload log for %s/%s",
                    namespace,
                    name,
                )
                snapshot = {
                    "pod_name": name,
                    "namespace": namespace,
                    "capture_error": str(exc),
                }
            return pod, snapshot

        # Bounded: each read carries ``_request_timeout`` and the whole batch
        # gets one budget per wave of workers. A capture that has not finished
        # by then is recorded as an error and left to finish on its own thread;
        # the reconcile loop that called us must not wait for a hung kubelet.
        completed: list[tuple[Any, dict[str, Any]]] = []
        max_workers = min(8, len(items))
        budget = self.workload_log_timeout_seconds * math.ceil(len(items) / max_workers)
        executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="gpu-fault-workload-log",
        )
        futures = {executor.submit(capture, pod): pod for pod in items}
        try:
            for future in as_completed(futures, timeout=budget):
                completed.append(future.result())
                self.note_progress()
        except TimeoutError:
            for future, pod in futures.items():
                if future.done():
                    continue
                name, namespace, _pod_uid = self._pod_identity(pod)
                LOGGER.error(
                    "workload log capture for %s/%s timed out after %ss",
                    namespace,
                    name,
                    budget,
                )
                completed.append(
                    (
                        pod,
                        {
                            "pod_name": name,
                            "namespace": namespace,
                            "capture_error": (
                                f"workload log capture timed out after {budget}s"
                            ),
                        },
                    )
                )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        snapshots = []
        for pod, snapshot in completed:
            snapshots.append(snapshot)
            if not annotate or not snapshot.get("record_id"):
                continue
            name, namespace, _pod_uid = self._pod_identity(pod)
            self.core.patch_namespaced_pod(
                name,
                namespace,
                {
                    "metadata": {
                        "annotations": {
                            WORKLOAD_LOG_SNAPSHOT_ANNOTATION: (
                                self._snapshot_annotation(snapshot)
                            )
                        }
                    }
                },
            )
        return sorted(
            snapshots,
            key=lambda item: (
                str(item.get("node_id") or ""),
                str(item.get("pod_uid") or item.get("pod_name") or ""),
            ),
        )

    def _snapshot_annotation(self, snapshot: dict[str, Any]) -> str:
        annotation_snapshot = dict(snapshot)
        tail = str(annotation_snapshot.get("tail") or "")
        annotation_snapshot["tail"] = tail.encode("utf-8")[
            -self.workload_log_annotation_tail_bytes :
        ].decode("utf-8", errors="replace")
        annotation_snapshot["annotation_tail_truncated"] = len(
            annotation_snapshot["tail"]
        ) < len(tail)
        return json.dumps(
            annotation_snapshot,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _capture_workload_log(
        self,
        pod: Any,
        *,
        namespace: str,
        attempt_id: str,
        incident_id: str,
    ) -> dict[str, Any]:
        metadata = (
            pod.get("metadata", {}) if isinstance(pod, dict) else pod.metadata.to_dict()
        )
        spec = pod.get("spec", {}) if isinstance(pod, dict) else pod.spec.to_dict()
        annotations = metadata.get("annotations") or {}
        pod_name = str(metadata.get("name") or "")
        pod_uid = str(metadata.get("uid") or pod_name)
        container_name = annotations.get(TRAINING_CONTAINER_ANNOTATION)
        containers = spec.get("containers") or []
        if not container_name and len(containers) == 1:
            container_name = containers[0].get("name")
        if not pod_name or not container_name:
            raise CompletionControllerError(
                "emergency log capture requires Pod and training container identity"
            )
        captured_at = datetime.now(timezone.utc)
        if self.workload_log_s3_uri:
            value = self.core.read_namespaced_pod_log(
                pod_name,
                namespace,
                container=container_name,
                timestamps=True,
                limit_bytes=self.workload_log_s3_max_bytes,
                _request_timeout=self.workload_log_timeout_seconds,
            )
            raw = str(value).encode("utf-8", errors="replace")
            archive_truncated = len(raw) >= self.workload_log_s3_max_bytes
        else:
            value = self.core.read_namespaced_pod_log(
                pod_name,
                namespace,
                container=container_name,
                timestamps=True,
                tail_lines=self.workload_log_tail_lines,
                _request_timeout=self.workload_log_timeout_seconds,
            )
            raw = str(value).encode("utf-8", errors="replace")
            archive_truncated = len(raw.splitlines()) >= self.workload_log_tail_lines
        tail = raw[-self.workload_log_max_bytes :]
        digest = hashlib.sha256(raw).hexdigest()
        record_hash = hashlib.sha256(
            (f"{incident_id}\x1f{pod_uid}\x1f{container_name}").encode()
        ).hexdigest()[:24]
        snapshot = {
            "record_id": f"workload-log/{record_hash}",
            "node_id": str(spec.get("nodeName") or "UNKNOWN"),
            "attempt_id": attempt_id,
            "namespace": namespace,
            "pod_name": pod_name,
            "pod_uid": pod_uid,
            "container_name": container_name,
            "captured_at": captured_at.isoformat(),
            "sha256": digest,
            "tail": tail.decode("utf-8", errors="replace"),
            "tail_bytes": len(tail),
            "tail_lines": self.workload_log_tail_lines,
            "truncated": (len(raw) > len(tail) or archive_truncated),
            "archive_truncated": archive_truncated,
            "s3_uri": None,
        }
        if self.workload_log_s3_uri:
            try:
                snapshot["s3_uri"] = self._upload_workload_log(
                    data=raw,
                    attempt_id=attempt_id,
                    incident_id=incident_id,
                    pod_uid=pod_uid,
                    container_name=str(container_name),
                    captured_at=captured_at,
                )
            except Exception as exc:
                LOGGER.exception(
                    "cannot archive workload log to S3 for %s/%s",
                    namespace,
                    pod_name,
                )
                snapshot["archive_error"] = f"{type(exc).__name__}: {exc}"
        return snapshot

    def _upload_workload_log(
        self,
        *,
        data: bytes,
        attempt_id: str,
        incident_id: str,
        pod_uid: str,
        container_name: str,
        captured_at: datetime,
    ) -> str:
        parsed = urlparse(self.workload_log_s3_uri or "")
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("GPU_FAULT_WORKLOAD_LOG_S3_URI must be an s3:// URI")
        safe_container = re.sub(r"[^A-Za-z0-9_.-]+", "-", container_name)
        suffix = (
            f"{attempt_id}/{incident_id}/{pod_uid}/"
            f"{safe_container}-{captured_at.strftime('%Y%m%dT%H%M%SZ')}"
            ".log.gz"
        )
        prefix = parsed.path.strip("/")
        key = f"{prefix}/{suffix}" if prefix else suffix
        destination = f"s3://{parsed.netloc}/{key}"
        compressed = gzip.compress(data)
        if self.workload_log_uploader is not None:
            return str(self.workload_log_uploader(destination, compressed))
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for workload log S3 archival"
            ) from exc
        boto3.client("s3").put_object(
            Bucket=parsed.netloc,
            Key=key,
            Body=compressed,
            ContentType="text/plain",
            ContentEncoding="gzip",
            Metadata={"sha256": hashlib.sha256(data).hexdigest()},
        )
        return destination

    @staticmethod
    def _parse(value: str) -> tuple[str, str, str]:
        parts = value.split("/")
        if len(parts) == 2:
            namespace, name = parts
            kind = "job"
        elif len(parts) == 3:
            namespace, kind, name = parts
            kind = kind.lower()
        else:
            raise CompletionControllerError(
                "workload ID must be namespace/name or "
                "namespace/{job|pytorchjob|jobset}/name"
            )
        if kind not in {"job", "pytorchjob", "jobset"} or not namespace or not name:
            raise CompletionControllerError(f"invalid workload ID: {value}")
        return namespace, kind, name
