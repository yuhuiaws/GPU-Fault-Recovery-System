"""Allowlisted failure summaries for owned proof Jobs; raw output stays private."""

from __future__ import annotations

import builtins
import json
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.execution import DeploymentDeadlineExceeded, remaining_timeout
from gpu_fault.admin.process_supervisor import ensure_supervision_safe
from gpu_fault_release.regional_resource_probe import ResourceRef

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

_MAX_PODS = 2
_MAX_CONTAINERS = 2
_LOG_BYTES = 8192
_JSON_BYTES = 262144
_PYTHON_ERRORS = frozenset(
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, Exception)
)
_REASONS = frozenset(
    {
        "Completed",
        "Error",
        "OOMKilled",
        "ContainerCannotRun",
        "StartError",
        "DeadlineExceeded",
        "CrashLoopBackOff",
        "CreateContainerConfigError",
        "CreateContainerError",
        "RunContainerError",
        "ErrImagePull",
        "ImagePullBackOff",
        "InvalidImageName",
        "ContainerCreating",
        "PodInitializing",
        "Evicted",
        "NodeLost",
    }
)


def _read(release: RegionalRelease, *arguments: str) -> str:
    ensure_supervision_safe()
    try:
        return release.runner.run(
            release._cpu(*arguments, "--request-timeout=5s"),
            capture=True,
            sensitive=True,
            timeout_seconds=remaining_timeout(5),
        )
    finally:
        ensure_supervision_safe()
        remaining_timeout(1)


def _json(release: RegionalRelease, *arguments: str) -> dict[str, Any]:
    raw = _read(release, *arguments)
    if len(raw.encode("utf-8")) > _JSON_BYTES:
        raise ValueError("proof Job diagnostic response exceeds its size limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("proof Job diagnostic response is not an object")
    return value


def _pod_identity(
    pod: dict[str, Any], job: dict[str, Any], *, listed: bool = False
) -> tuple[str, str]:
    metadata = pod["metadata"]
    expected = job["metadata"]
    owners = metadata.get("ownerReferences")
    if (
        pod.get("kind", "Pod" if listed else None) != "Pod"
        or metadata.get("namespace") != expected["namespace"]
        or not isinstance(metadata.get("uid"), str)
        or not metadata["uid"]
        or not isinstance(owners, list)
        or len(owners) != 1
        or owners[0].get("controller") is not True
        or any(
            owners[0].get(key) != value
            for key, value in (
                ("apiVersion", "batch/v1"),
                ("kind", "Job"),
                ("name", expected["name"]),
                ("uid", expected["uid"]),
            )
        )
    ):
        raise ValueError("proof Job diagnostic Pod ownership differs")
    reference = ResourceRef("pod", "Pod", metadata["name"], metadata["namespace"])
    return reference.name, metadata["uid"]


def _container_state(container: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("state", "lastState"):
        for state, value in (container.get(field) or {}).items():
            if state not in {"running", "waiting", "terminated"}:
                continue
            details: dict[str, Any] = {"state": state}
            reason = value.get("reason")
            if reason is not None:
                details["reason"] = reason if reason in _REASONS else "unknown"
            for key, maximum in (("exitCode", 255), ("signal", 64)):
                number = value.get(key)
                if type(number) is int and 0 <= number <= maximum:
                    details[key] = number
            result[field] = details
    return result


def _log_summary(raw: str) -> dict[str, Any]:
    # A bounded tail may begin inside a credential or JSON document. Only emit
    # known categories, never arbitrary text or a partially redacted traceback.
    if len(raw.encode("utf-8")) > _LOG_BYTES:
        raise ValueError("proof Job diagnostic logs exceed their size limit")
    return {
        "python_errors": sorted(
            set(re.findall(r"(?m)^([A-Za-z][A-Za-z0-9_]{0,63})(?::|$)", raw))
            & _PYTHON_ERRORS
        ),
        "output": diagnostic_text(raw, sensitive=True),
        "limit_reached": len(raw.encode("utf-8")) >= _LOG_BYTES,
    }


def collect_job_diagnostics(
    release: RegionalRelease, job: dict[str, Any]
) -> dict[str, Any]:
    """Read a small owned Pod sample and recheck each UID after its log reads."""
    metadata = job["metadata"]
    namespace = metadata["namespace"]
    query = urlencode(
        {
            "labelSelector": f"batch.kubernetes.io/controller-uid={metadata['uid']}",
            "limit": _MAX_PODS,
        }
    )
    listing = _json(
        release, "get", "--raw", f"/api/v1/namespaces/{namespace}/pods?{query}"
    )
    pods = listing.get("items")
    if (
        listing.get("kind") != "PodList"
        or not isinstance(pods, list)
        or len(pods) > _MAX_PODS
    ):
        raise ValueError("proof Job diagnostic Pod list is invalid")
    identities = [_pod_identity(pod, job, listed=True) for pod in pods]
    if any(
        len({item[index] for item in identities}) != len(identities) for index in (0, 1)
    ):
        raise ValueError("proof Job diagnostic Pod identities repeat")
    spec = job["spec"]["template"]["spec"]
    containers = [
        (field, item["name"])
        for field, key in (
            ("initContainerStatuses", "initContainers"),
            ("containerStatuses", "containers"),
        )
        for item in spec.get(key, [])
    ]
    summaries: list[dict[str, Any]] = []
    for pod, (name, uid) in zip(pods, identities, strict=True):
        status = pod.get("status") or {}
        phase = status.get("phase")
        summary: dict[str, Any] = {
            "pod_index": len(summaries),
            "phase": phase
            if phase in {"Pending", "Running", "Succeeded", "Failed", "Unknown"}
            else "Unknown",
            "containers": [],
            "containers_omitted": max(0, len(containers) - _MAX_CONTAINERS),
        }
        for field, container_name in containers[:_MAX_CONTAINERS]:
            container: dict[str, Any] = next(
                (
                    item
                    for item in status.get(field, [])
                    if item.get("name") == container_name
                ),
                {},
            )
            details = _container_state(container)
            details["container_index"] = len(summary["containers"])
            try:
                raw = _read(
                    release,
                    "-n",
                    namespace,
                    "logs",
                    f"pod/{name}",
                    "-c",
                    container_name,
                    "--tail=80",
                    f"--limit-bytes={_LOG_BYTES}",
                )
                details["logs"] = _log_summary(raw)
            except DeploymentDeadlineExceeded:
                raise
            except Exception as exc:
                details["logs"] = {"unavailable": type(exc).__name__}
            summary["containers"].append(details)
        current = _json(release, "-n", namespace, "get", "pod", name, "-o", "json")
        if _pod_identity(current, job) != (name, uid):
            raise ValueError("proof Job diagnostic Pod changed during log reads")
        summaries.append(summary)
    return {
        "pods": summaries,
        "more_pods": bool((listing.get("metadata") or {}).get("continue")),
    }
