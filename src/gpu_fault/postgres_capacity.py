"""Shared PostgreSQL pool and listener budgets for runtime, admin and acceptance."""

from __future__ import annotations

from dataclasses import dataclass

from gpu_fault.store.contracts import WakeupChannel


@dataclass(frozen=True)
class PostgresPoolCapacity:
    """Pooled demand and separate LISTEN connections of one application process."""

    pool_max: int
    demand_by_consumer: dict[str, int]
    unpooled_connections: int
    REQUIRED_HEADROOM = 2

    @property
    def demand(self) -> int:
        return sum(self.demand_by_consumer.values())

    @property
    def oversubscription_ratio(self) -> float:
        return self.demand / self.pool_max if self.pool_max > 0 else float("inf")

    @property
    def headroom(self) -> int:
        return self.pool_max - self.demand

    @property
    def has_headroom(self) -> bool:
        return self.headroom >= self.REQUIRED_HEADROOM

    @staticmethod
    def listener_connections(
        service_role: str,
        *,
        queued_processor: bool,
        spool_enabled: bool,
        workflow_dispatcher_enabled: bool,
        regional: bool,
    ) -> dict[str, int]:
        if service_role not in {"all", "ingress", "worker", "spool-worker"}:
            raise ValueError("unknown service role in PostgreSQL listener budget")
        background = service_role in {"all", "worker"}
        result: dict[str, int] = {}
        if queued_processor and background:
            result["processor_queue"] = 1
        if (
            queued_processor
            and spool_enabled
            and service_role in {"all", "spool-worker"}
        ):
            result["telemetry_spool"] = 1
        if background and workflow_dispatcher_enabled:
            result["workflow_dispatch"] = len(WakeupChannel)
        if regional:
            # The claim hub starts lazily. Its connection must be reserved before
            # the first long poll, not only after it appears in pg_stat_activity.
            result["remote_command_claim"] = 1
        return result

    @classmethod
    def role_connection_ceiling(
        cls,
        *,
        pool_max: int,
        processes_per_pod: int,
        replicas: int,
        service_role: str,
        queued_processor: bool,
        spool_enabled: bool,
        workflow_dispatcher_enabled: bool,
        regional: bool,
    ) -> int:
        if (
            type(pool_max) is not int
            or pool_max < 1
            or type(processes_per_pod) is not int
            or processes_per_pod < 1
            or type(replicas) is not int
            or replicas < 0
        ):
            raise ValueError("PostgreSQL role capacity requires valid integer counts")
        listeners = cls.listener_connections(
            service_role,
            queued_processor=queued_processor,
            spool_enabled=spool_enabled,
            workflow_dispatcher_enabled=workflow_dispatcher_enabled,
            regional=regional,
        )
        return replicas * processes_per_pod * (pool_max + sum(listeners.values()))
