"""Short-lived, control-plane-owned scopes for passive communication observation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from urllib.parse import urlparse
import time
import threading
import weakref
import json

from pydantic import Field, field_validator

from gpu_fault.hung_lifecycle import cleanup_reason
from gpu_fault.models import StrictModel, WorkflowOperation

ScopeIndex = dict[str, list[tuple[dict[str, Any], float]]]
_SCOPES: weakref.WeakKeyDictionary[Any, dict[str, tuple[float, ScopeIndex]]] = (
    weakref.WeakKeyDictionary()
)
_SCOPE_LOCK = threading.RLock()


def target_index(
    store: Any, cluster: str, now: datetime
) -> dict[str, list[tuple[dict[str, Any], float]]]:
    with _SCOPE_LOCK:
        cache = _SCOPES.setdefault(store, {})
        cached = cache.get(cluster)
        if cached and time.monotonic() - cached[0] < 10:
            return cached[1]
        index: dict[str, list[tuple[dict[str, Any], float]]] = {}
        for observation in store.list_attempt_observations(cluster):
            if not 0 <= (
                now - observation.observed_at
            ).total_seconds() <= 120 or cleanup_reason(observation):
                continue
            by_node: dict[str, list[Any]] = {}
            for row in observation.containers:
                if row.node_id and not row.terminated and not row.deletion_requested:
                    by_node.setdefault(row.node_id, []).append(row)
            coordinator = min(by_node, default=None)
            for node, local in by_node.items():
                paths = sorted(
                    {
                        row.cgroup_path
                        for row in local
                        if row.cgroup_path and row.cgroup_path != "/"
                    }
                )
                if any(
                    not row.cgroup_path
                    or not row.cgroup_path.startswith("/")
                    or row.cgroup_path == "/"
                    or ".." in row.cgroup_path.split("/")
                    for row in local
                ):
                    paths = []
                index.setdefault(node, []).append(
                    (
                        {
                            "job_id": observation.job_id,
                            "attempt_id": observation.attempt_id,
                            "cgroup_paths": paths,
                            "gpu_uuids": sorted(
                                {gpu for row in local for gpu in row.gpu_uuids}
                            ),
                            "ras_coordinator": node == coordinator,
                        },
                        observation.observed_at.timestamp() + 120,
                    )
                )
        if len(json.dumps(index).encode()) <= 4 * 1024 * 1024:
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[cluster] = (time.monotonic(), index)
        return index


class CommunicationTarget(StrictModel):
    job_id: str = Field(min_length=1, max_length=256)
    attempt_id: str = Field(min_length=1, max_length=256)
    cgroup_paths: list[str] = Field(min_length=1, max_length=256)
    gpu_uuids: list[str] = Field(default_factory=list, max_length=256)
    ras_coordinator: bool = False

    @field_validator("cgroup_paths")
    @classmethod
    def scoped_paths(cls, values: list[str]) -> list[str]:
        if any(
            not path.startswith("/") or path == "/" or ".." in path.split("/")
            for path in values
        ):
            raise ValueError(
                "communication observation needs explicit workload cgroups"
            )
        return values


class CommunicationPlan(StrictModel):
    version: Literal["communication-monitor/v1"] = "communication-monitor/v1"
    cluster_id: str
    node_id: str
    boot_id: str
    generation: int
    issued_at: datetime
    expires_at: datetime
    targets: list[CommunicationTarget] = Field(default_factory=list, max_length=8)
    expected_target_count: int = Field(ge=0)
    node_addresses: list[str] = Field(default_factory=list, max_length=8)


def observation_plan(
    store: Any, record: Any, now: datetime | None = None
) -> CommunicationPlan:
    now = now or datetime.now(timezone.utc)
    targets = []
    expected = 0
    if (
        record.lifecycle_state.value == "ACTIVE"
        and WorkflowOperation.COLLECT_HUNG_SUMMARY in record.allowed_operations
        and record.boot_id
    ):
        for target, expiry in target_index(store, record.cluster_id, now).get(
            record.node_id, []
        ):
            if now.timestamp() > expiry:
                continue
            expected += 1
            if target["cgroup_paths"]:
                targets.append(CommunicationTarget.model_validate(target))
    return CommunicationPlan(
        cluster_id=record.cluster_id,
        node_id=record.node_id,
        boot_id=record.boot_id or "",
        generation=record.generation,
        issued_at=now,
        expires_at=now + timedelta(seconds=90),
        targets=sorted(targets, key=lambda item: (item.job_id, item.attempt_id))[:8],
        expected_target_count=expected,
        node_addresses=list(
            dict.fromkeys(
                [record.node_id, urlparse(record.endpoint).hostname or record.node_id]
            )
        ),
    )
