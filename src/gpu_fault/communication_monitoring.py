"""Short-lived, control-plane-owned scopes for passive communication observation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Final, Literal
from urllib.parse import urlparse
import time
import threading
import weakref
import json

from pydantic import Field, field_validator, model_validator

from gpu_fault.hung_lifecycle import cleanup_reason
from gpu_fault.models import StrictModel, WorkflowOperation
from gpu_fault.workload_container_scope import (
    MAX_NODE_CONTAINER_IDS,
    runtime_container_id,
)

#: The lease protocol an Agent's heartbeat advertises
#: (``AgentHeartbeat.communication_monitor_version``) and every plan names.
COMMUNICATION_MONITOR_V1: Final = "communication-monitor/v1"
#: v2 adds ``CommunicationTarget.container_ids``. A v1 Agent validates its plan
#: with ``extra="forbid"`` and refuses the whole lease over an unknown field, so
#: only an Agent that advertises v2 is sent a target with ids.
COMMUNICATION_MONITOR_V2: Final = "communication-monitor/v2"
MonitorVersion = Literal["communication-monitor/v1", "communication-monitor/v2"]
COMMUNICATION_MONITOR_VERSIONS = frozenset(
    {COMMUNICATION_MONITOR_V1, COMMUNICATION_MONITOR_V2}
)

ScopeIndex = dict[str, list[tuple[dict[str, Any], float]]]
_SCOPES: weakref.WeakKeyDictionary[Any, dict[str, tuple[float, ScopeIndex]]] = (
    weakref.WeakKeyDictionary()
)
_SCOPE_LOCK = threading.RLock()


def _node_scope(local: list[Any]) -> tuple[list[str], list[str], bool]:
    """``(cgroup paths, runtime container ids, scoped)`` of one node's containers.

    A container names its cgroup (a ``gpu-fault.io/cgroup-path`` annotation) or,
    without one, is named by the runtime id the kubelet reports. One container
    with neither -- not running yet, an id no node could match, a root or
    escaping path -- leaves the node unscoped: sampling the others would read as
    the whole job. So do more containers than one target may name.
    """

    paths: set[str] = set()
    container_ids: set[str] = set()
    scoped = True
    for row in local:
        if row.cgroup_path:
            if (
                not row.cgroup_path.startswith("/")
                or row.cgroup_path == "/"
                or ".." in row.cgroup_path.split("/")
            ):
                scoped = False
            else:
                paths.add(row.cgroup_path)
        elif (identifier := runtime_container_id(row.container_id)) is not None:
            container_ids.add(identifier)
        else:
            scoped = False
    return (
        sorted(paths),
        sorted(container_ids),
        scoped and len(container_ids) <= MAX_NODE_CONTAINER_IDS,
    )


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
                paths, container_ids, scoped = _node_scope(local)
                index.setdefault(node, []).append(
                    (
                        {
                            "job_id": observation.job_id,
                            "attempt_id": observation.attempt_id,
                            "cgroup_paths": paths,
                            "container_ids": container_ids,
                            "scoped": scoped,
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
    cgroup_paths: list[str] = Field(default_factory=list, max_length=256)
    #: v2 only: the runtime ids of the attempt's containers on the node that
    #: name no cgroup; the Agent finds their cgroups itself. Left out when empty,
    #: so a target without ids is the v1 target byte for byte.
    container_ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_NODE_CONTAINER_IDS,
        exclude_if=lambda value: not value,
    )
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

    @field_validator("container_ids")
    @classmethod
    def runtime_ids(cls, values: list[str]) -> list[str]:
        if any(runtime_container_id(value) != value for value in values):
            raise ValueError(
                "communication observation names containers by 64-hex runtime ids"
            )
        return values

    @model_validator(mode="after")
    def scoped(self) -> CommunicationTarget:
        if not self.cgroup_paths and not self.container_ids:
            raise ValueError(
                "communication observation needs explicit workload cgroups or "
                "runtime container ids"
            )
        return self


class CommunicationPlan(StrictModel):
    version: MonitorVersion = COMMUNICATION_MONITOR_V1
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
    store: Any,
    record: Any,
    now: datetime | None = None,
    *,
    version: MonitorVersion = COMMUNICATION_MONITOR_V1,
) -> CommunicationPlan:
    """The lease for one Agent, in the monitor ``version`` its heartbeat spoke.

    Every target of the node's attempts counts toward ``expected_target_count``;
    one the Agent cannot be given stays out of ``targets``: an unscoped node,
    and for a v1 Agent any node that needs a container id.
    """

    if version not in COMMUNICATION_MONITOR_VERSIONS:
        raise ValueError(f"unknown communication monitor version: {version}")
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
            if not target["scoped"] or (
                target["container_ids"] and version == COMMUNICATION_MONITOR_V1
            ):
                continue
            targets.append(
                CommunicationTarget(
                    job_id=target["job_id"],
                    attempt_id=target["attempt_id"],
                    cgroup_paths=target["cgroup_paths"],
                    container_ids=target["container_ids"],
                    gpu_uuids=target["gpu_uuids"],
                    ras_coordinator=target["ras_coordinator"],
                )
            )
    return CommunicationPlan(
        version=version,
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
