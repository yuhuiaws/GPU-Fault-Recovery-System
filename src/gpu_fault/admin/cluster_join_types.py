"""Shared join data without a dependency on the lifecycle driver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import ClusterIdentity
from gpu_fault.admin.site import RenderedSite

DEFAULT_ALLOWED_NAMESPACES = ("gpu-fault-system", "training")


@dataclass(frozen=True)
class JoinClusterRequest:
    site: RenderedSite
    gpu_cluster_arn: str
    cluster_id: str | None = None
    allowed_namespaces: tuple[str, ...] = DEFAULT_ALLOWED_NAMESPACES
    state_dir: Path | None = None


@dataclass(frozen=True)
class JoinInputs:
    target: ClusterIdentity
    cluster_id: str
    discovery: dict[str, Any]
    local: dict[str, Any]


@dataclass(frozen=True)
class JoinExecution:
    target: ClusterIdentity
    cluster_id: str
    discovery: dict[str, Any]
    local: dict[str, Any]
    prerequisites: dict[str, Any]
    candidate: RenderedSite


@dataclass
class JoinAttempt:
    request: JoinClusterRequest
    state_dir: Path
    state_path: Path
    state: dict[str, Any]
    execution: JoinExecution | None = None
    inputs: JoinInputs | None = None
