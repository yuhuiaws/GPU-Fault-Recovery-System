from __future__ import annotations

import os
import time
from typing import Any, Callable

from gpu_fault.adapters.common import (
    ANNOTATION_EXECUTION_EPOCH,
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    ANNOTATION_OPERATION,
    ANNOTATION_STEP_INDEX,
    ANNOTATION_WORKFLOW,
)
from gpu_fault.execution import (
    WorkflowStepContext,
)

NODE_PATCH_ATTEMPTS = 3
NODE_PATCH_BACKOFF_SECONDS = 0.1


class NodePatchConflict(RuntimeError):
    """``patch_node`` still returned 409 after every bounded attempt.

    The node is unchanged; the caller decides whether that is WAITING
    (the operation is idempotent and a later pass may win) or FAILED.
    """


def kubernetes_request_timeout_seconds() -> float:
    """Bounded per-request timeout for every Kubernetes API call.

    The kubernetes client has no default: an apiserver that accepts the TCP
    connection and never answers used to hold the executor thread forever,
    and with it the workflow lease.
    """
    try:
        value = float(os.getenv("GPU_FAULT_KUBERNETES_REQUEST_TIMEOUT_SECONDS", "30"))
    except ValueError as exc:
        raise ValueError(
            "GPU_FAULT_KUBERNETES_REQUEST_TIMEOUT_SECONDS must be a positive number"
        ) from exc
    if value <= 0:
        raise ValueError(
            "GPU_FAULT_KUBERNETES_REQUEST_TIMEOUT_SECONDS must be a positive number"
        )
    return value


def build_kubernetes_clients(
    configuration: Any, *, request_timeout_seconds: float
) -> tuple[Any, Any, Any]:
    """CoreV1/BatchV1/CustomObjects APIs sharing one timeout-bounded client.

    Every generated API method forwards ``_request_timeout`` from its own
    kwargs, so the one place that can give all of them a default is the
    ``ApiClient`` they share.
    """
    from kubernetes import client

    class TimeoutApiClient(client.ApiClient):  # type: ignore[misc]
        def call_api(self, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("_request_timeout") is None:
                kwargs["_request_timeout"] = request_timeout_seconds
            return super().call_api(*args, **kwargs)

    api_client = TimeoutApiClient(configuration)
    return (
        client.CoreV1Api(api_client),
        client.BatchV1Api(api_client),
        client.CustomObjectsApi(api_client),
    )


def node_scheduling_snapshot(node: Any) -> dict[str, Any]:
    """The scheduling baseline a node carries: what isolation/restore change."""
    return {
        "unschedulable": KubernetesPrimitivesMixin._unschedulable(node),
        "taint_keys": sorted(
            str(item.get("key")) for item in KubernetesPrimitivesMixin._taints(node)
        ),
        "resource_version": KubernetesPrimitivesMixin._resource_version(node),
    }


def patch_node_with_retry(
    core: Any,
    node_id: str,
    build_body: Callable[[Any], dict[str, Any] | None],
    *,
    attempts: int = NODE_PATCH_ATTEMPTS,
    backoff_seconds: float = NODE_PATCH_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Read the node, patch what ``build_body`` derives from it, retry 409s.

    Each retry re-reads the node so the mutation is recomputed against the
    current object and its ``resourceVersion``; a stale body is never
    resent. ``build_body`` returning ``None`` means nothing to patch. A 404
    on the read propagates: only the caller knows whether an absent node is
    a no-op or a safety rejection. Returns the node the successful (or
    skipped) patch was derived from.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        node = core.read_node(node_id)
        body = build_body(node)
        if body is None:
            return node
        try:
            core.patch_node(node_id, body)
            return node
        except Exception as exc:
            if getattr(exc, "status", None) != 409:
                raise
            last_error = exc
            if attempt + 1 < attempts:
                sleep(backoff_seconds * (attempt + 1))
    raise NodePatchConflict(
        f"node {node_id} patch conflicted {attempts} times"
    ) from last_error


class KubernetesPrimitivesMixin:
    # Attributes supplied by the composed concrete implementation.
    _incident_workflow_is_terminal: Callable[..., Any]
    _serialize_workload: Any
    batch: Any
    core: Any
    custom: Any

    @staticmethod
    def _clients(request_timeout_seconds: float) -> tuple[Any, Any, Any]:
        try:
            from kubernetes import client, config
        except ImportError as exc:
            raise RuntimeError("install gpu-fault-control-plane[collectors]") from exc
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        return build_kubernetes_clients(
            client.Configuration.get_default_copy(),
            request_timeout_seconds=request_timeout_seconds,
        )

    @staticmethod
    def _node_allocatable(node: Any, resource_name: str) -> int:
        status = node.get("status", {}) if isinstance(node, dict) else node.status
        allocatable = (
            status.get("allocatable", {})
            if isinstance(status, dict)
            else status.allocatable
        )
        try:
            return int((allocatable or {}).get(resource_name, 0))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _pod_uid(cls, pod: Any) -> str:
        metadata = cls._metadata(pod)
        value = metadata.get("uid") if isinstance(metadata, dict) else metadata.uid
        return str(value or "")

    @classmethod
    def _pod_name(cls, pod: Any) -> str:
        metadata = cls._metadata(pod)
        value = metadata.get("name") if isinstance(metadata, dict) else metadata.name
        return str(value or "")

    @staticmethod
    def _pod_ready(pod: Any) -> bool:
        status = pod.get("status", {}) if isinstance(pod, dict) else pod.status
        conditions = (
            status.get("conditions", [])
            if isinstance(status, dict)
            else status.conditions
        )
        return any(
            (
                condition.get("type") == "Ready"
                and str(condition.get("status")).lower() == "true"
            )
            if isinstance(condition, dict)
            else (condition.type == "Ready" and str(condition.status).lower() == "true")
            for condition in conditions or []
        )

    def _efa_plugin_pods(
        self,
        node_id: str,
        *,
        namespace: str,
        label_selector: str,
    ) -> list[Any]:
        response = self.core.list_namespaced_pod(
            namespace,
            label_selector=label_selector,
            field_selector=f"spec.nodeName={node_id}",
        )
        return list(response.items or [])

    def _read_workload(self, namespace: str, kind: str, name: str) -> Any:
        if kind == "job":
            return self.batch.read_namespaced_job(name, namespace)
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
        return self.custom.get_namespaced_custom_object(
            group, version, namespace, plural, name
        )

    def _patch_workload(
        self,
        namespace: str,
        kind: str,
        name: str,
        body: dict[str, Any],
    ) -> None:
        for attempt in range(3):
            try:
                if kind == "job":
                    self.batch.patch_namespaced_job(name, namespace, body)
                else:
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
                        group,
                        version,
                        namespace,
                        plural,
                        name,
                        body,
                    )
                return
            except Exception as exc:
                if getattr(exc, "status", None) != 409 or attempt == 2:
                    raise
                latest = self._read_workload(namespace, kind, name)
                resource_version = self._resource_version(latest)
                if resource_version is None:
                    body["metadata"].pop("resourceVersion", None)
                else:
                    body["metadata"]["resourceVersion"] = resource_version

    def _operation_already_applied(
        self,
        annotations: dict[str, str],
        context: WorkflowStepContext,
    ) -> bool:
        existing_incident = annotations.get(ANNOTATION_INCIDENT)
        existing_operation = annotations.get(ANNOTATION_OPERATION)
        if existing_incident and (existing_incident != context.incident.incident_id):
            if not self._incident_workflow_is_terminal(existing_incident):
                raise ValueError("workload is controlled by another incident")
        try:
            existing_fencing = int(annotations.get(ANNOTATION_FENCING, "0"))
            existing_epoch = int(annotations.get(ANNOTATION_EXECUTION_EPOCH, "0"))
            existing_step = int(annotations.get(ANNOTATION_STEP_INDEX, "-1"))
        except ValueError as exc:
            raise ValueError(
                "workload has invalid gpu-fault fencing annotations"
            ) from exc
        newer_same_incident_generation = (
            existing_incident == context.incident.incident_id
            and context.workflow.fencing_token > existing_fencing
        )
        if (
            existing_incident == context.incident.incident_id
            and existing_fencing > context.workflow.fencing_token
        ):
            raise ValueError("workload is controlled by a newer workflow generation")
        if (
            not newer_same_incident_generation
            and existing_epoch > context.workflow.execution_epoch
        ):
            raise ValueError("workload has a newer execution epoch")
        if (
            annotations.get(ANNOTATION_WORKFLOW) == context.workflow.request_id
            and existing_step > context.step_index
        ):
            raise ValueError("workload has already advanced past this step")
        return existing_operation == context.idempotency_key

    @staticmethod
    def _parse_workload(value: str) -> tuple[str, str, str]:
        parts = value.split("/")
        if len(parts) == 2:
            namespace, name = parts
            kind = "job"
        elif len(parts) == 3:
            namespace, kind, name = parts
            kind = kind.lower()
        else:
            raise ValueError(
                "workload ID must be namespace/name or "
                "namespace/{job|pytorchjob|jobset}/name"
            )
        if kind not in {"job", "pytorchjob", "jobset"}:
            raise ValueError(f"unsupported workload kind: {kind}")
        if not namespace or not name:
            raise ValueError("workload namespace/name cannot be empty")
        return namespace, kind, name

    @classmethod
    def _workload_terminal(cls, kind: str, workload: Any) -> bool:
        if kind not in {"pytorchjob", "jobset"}:
            return False
        serialized = cls._serialize_workload(workload)
        for condition in serialized.get("status", {}).get("conditions", []):
            if str(condition.get("status", "")).lower() == "true" and condition.get(
                "type"
            ) in {"Failed", "Succeeded", "Completed"}:
                return True
        return False

    @staticmethod
    def _suspend_spec(kind: str, suspend: bool) -> dict[str, Any]:
        if kind == "pytorchjob":
            return {"runPolicy": {"suspend": suspend}}
        return {"suspend": suspend}

    @staticmethod
    def _metadata(node: Any) -> Any:
        return node.get("metadata", {}) if isinstance(node, dict) else node.metadata

    @classmethod
    def _annotations(cls, node: Any) -> dict[str, str]:
        metadata = cls._metadata(node)
        value = (
            metadata.get("annotations", {})
            if isinstance(metadata, dict)
            else metadata.annotations
        )
        return dict(value or {})

    @classmethod
    def _labels(cls, node: Any) -> dict[str, str]:
        metadata = cls._metadata(node)
        value = (
            metadata.get("labels", {})
            if isinstance(metadata, dict)
            else metadata.labels
        )
        return dict(value or {})

    @staticmethod
    def _spec(node: Any) -> Any:
        return node.get("spec", {}) if isinstance(node, dict) else node.spec

    @classmethod
    def _taints(cls, node: Any) -> list[dict[str, Any]]:
        spec = cls._spec(node)
        value = spec.get("taints", []) if isinstance(spec, dict) else spec.taints
        result = []
        for item in value or []:
            if isinstance(item, dict):
                result.append(dict(item))
            else:
                result.append(
                    {
                        "key": item.key,
                        "value": item.value,
                        "effect": item.effect,
                    }
                )
        return result

    @classmethod
    def _unschedulable(cls, node: Any) -> bool:
        spec = cls._spec(node)
        value = (
            spec.get("unschedulable", False)
            if isinstance(spec, dict)
            else spec.unschedulable
        )
        return bool(value)

    @classmethod
    def _resource_version(cls, node: Any) -> str | None:
        metadata = cls._metadata(node)
        return (
            metadata.get("resourceVersion")
            if isinstance(metadata, dict)
            else metadata.resource_version
        )
