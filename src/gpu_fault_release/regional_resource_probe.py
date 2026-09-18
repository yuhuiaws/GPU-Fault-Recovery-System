"""Three-state Kubernetes reads; transport failure is never resource absence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast

from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault_release.regional_release_config import ReleaseError


class ProbeState(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    ERROR = "error"


@dataclass(frozen=True)
class ResourceRef:
    resource: str
    kind: str
    name: str
    namespace: str | None

    def __post_init__(self) -> None:
        if (
            not re.fullmatch(r"[a-z][a-z0-9.-]{0,127}", self.resource)
            or not re.fullmatch(r"[A-Z][A-Za-z0-9]{0,127}", self.kind)
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", self.name)
            or self.namespace is not None
            and not re.fullmatch(
                r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", self.namespace
            )
        ):
            raise ValueError("resource probe scope is invalid")


@dataclass(frozen=True)
class ResourceObservation:
    state: ProbeState
    reference: ResourceRef
    document: dict[str, object] | None = field(default=None, repr=False)
    error: str = ""

    def __bool__(self) -> bool:
        raise TypeError("resource observations require an explicit state check")

    def require_readable(self) -> dict[str, object] | None:
        if self.state is ProbeState.ERROR:
            raise ReleaseError(
                f"cannot read {self.reference.resource}/{self.reference.name}: {self.error}"
            )
        return self.document

    def exists(self) -> bool:
        return self.require_readable() is not None


class ProbeRunner(Protocol):
    def probe_output(
        self, args: list[str], *, timeout_seconds: float | None = None
    ) -> tuple[int, str, str]: ...


def probe_resource(
    runner: ProbeRunner,
    kubectl: list[str],
    reference: ResourceRef,
    *,
    timeout_seconds: float = 20,
) -> ResourceObservation:
    namespace = ["-n", reference.namespace] if reference.namespace else []
    arguments = [
        *kubectl,
        *namespace,
        "get",
        reference.resource,
        reference.name,
        "--ignore-not-found",
        "-o",
        "json",
        "--request-timeout=15s",
    ]
    try:
        code, output, error = runner.probe_output(
            arguments, timeout_seconds=timeout_seconds
        )
    except (ReleaseError, TimeoutError, OSError) as exc:
        return ResourceObservation(
            ProbeState.ERROR, reference, error=diagnostic_text(str(exc))
        )
    if code:
        return ResourceObservation(
            ProbeState.ERROR,
            reference,
            error=f"read failed ({code}): {diagnostic_text(error)}",
        )
    if not output.strip():
        return ResourceObservation(ProbeState.ABSENT, reference)
    try:
        value = json.loads(output)
        if not isinstance(value, dict) or value.get("kind") != reference.kind:
            raise ValueError("resource kind differs")
        metadata = value.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("name") != reference.name
            or metadata.get("namespace") != reference.namespace
            or not isinstance(metadata.get("uid"), str)
            or not metadata["uid"]
        ):
            raise ValueError("resource identity differs")
    except (ValueError, TypeError, KeyError) as exc:
        return ResourceObservation(
            ProbeState.ERROR, reference, error=f"invalid response: {exc}"
        )
    return ResourceObservation(
        ProbeState.PRESENT, reference, document=cast(dict[str, object], value)
    )
