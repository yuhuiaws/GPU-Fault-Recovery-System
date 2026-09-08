from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.store.shared.telemetry_models import (
    TELEMETRY_SPOOL_MAX_ATTEMPTS,
    SpooledTelemetry,
)


class MemoryTelemetrySpoolMixin:
    """The in-process spool.

    ``InMemoryStore`` uses it because it has nothing else; ``SqliteStore``
    composes it too, so a single-process deployment spools telemetry in
    memory and loses it with the process -- acceptable for a stream whose
    next sample restates the same node state, and why the ingress warns when
    the spool is enabled against a store that is not Postgres.
    """

    # Attributes supplied by the composed concrete implementation.
    _telemetry_spool: Any

    _lock: Any

    TELEMETRY_SPOOL_MAX_ATTEMPTS = TELEMETRY_SPOOL_MAX_ATTEMPTS

    def try_spool_telemetry_requests(
        self,
        requests,
        *,
        max_depth: int,
        max_cluster_depth: int,
        now: datetime | None = None,
    ) -> list[tuple[object, str | None]]:
        """Admit telemetry to its own spool instead of the queue.

        Result shape matches ``try_enqueue_processor_requests_batch`` --
        ``(item, None)``, ``(item, "coalesced")``, ``(None, "global")``,
        ``(None, "cluster")`` -- so the ingress path handles a spooled and
        a queued admission the same way.

        There is no fault reserve here and there does not need to be: the
        reserve exists because telemetry and faults share the queue, and
        the point of the spool is that they no longer do.

        Only ``PostgresStore`` makes this durable. The in-memory spool is
        what tests and single-process deployments use, and it is lost with
        the process - acceptable for a stream whose next sample restates
        the same node state, but it is why the ingress warns when the
        spool is enabled against a store that is not Postgres.
        """

        observed_at = now or datetime.now(timezone.utc)
        results: list[tuple[object, str | None]] = []
        with self._lock:
            for request in requests:
                key = request.spool_key()
                existing = self._telemetry_spool.get(key)
                if existing is not None:
                    existing["payload"] = json.loads(request.body())
                    existing["request_id"] = request.request_id
                    existing["cluster_id"] = request.cluster_id
                    existing["path"] = request.path
                    existing["revision"] += 1
                    # A row whose lease has not expired still becomes
                    # claimable now: the consumer holding it is carrying a
                    # payload this sample just superseded, and its
                    # completion is fenced on the old revision.
                    existing["available_at"] = min(
                        existing["available_at"],
                        observed_at,
                    )
                    existing["updated_at"] = observed_at
                    results.append((request, "coalesced"))
                    continue
                if len(self._telemetry_spool) >= max_depth:
                    results.append((None, "global"))
                    continue
                cluster_depth = sum(
                    row["cluster_id"] == request.cluster_id
                    for row in self._telemetry_spool.values()
                )
                if cluster_depth >= max_cluster_depth:
                    results.append((None, "cluster"))
                    continue
                self._telemetry_spool[key] = {
                    "spool_key": key,
                    "cluster_id": request.cluster_id,
                    "path": request.path,
                    "request_id": request.request_id,
                    "revision": 0,
                    "attempts": 0,
                    "lease_owner": None,
                    "available_at": observed_at,
                    "created_at": observed_at,
                    "updated_at": observed_at,
                    "payload": json.loads(request.body()),
                }
                results.append((request, None))
        return results

    @staticmethod
    def _spooled_telemetry(row: dict) -> SpooledTelemetry:
        payload_bytes = len(
            json.dumps(
                row["payload"],
                separators=(",", ":"),
            ).encode()
        )
        return SpooledTelemetry(
            spool_key=row["spool_key"],
            revision=row["revision"],
            cluster_id=row["cluster_id"],
            path=row["path"],
            request_id=row["request_id"],
            attempts=row["attempts"],
            created_at=row["created_at"],
            payload=row["payload"],
            payload_bytes=payload_bytes,
        )

    def claim_telemetry_spool(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
        max_bytes: int | None = None,
        path: str | None = None,
    ) -> list[SpooledTelemetry]:
        """Take the oldest available rows and lease them.

        A lease is nothing but ``available_at`` moved into the future, so
        a consumer that dies holding rows needs no reaper: the rows fall
        back inside the claim window when the lease would have expired.
        """

        if limit <= 0:
            return []
        byte_limit = 2**63 - 1 if max_bytes is None else max_bytes
        if byte_limit <= 0:
            return []
        with self._lock:
            eligible = sorted(
                (
                    row
                    for row in self._telemetry_spool.values()
                    if row["available_at"] <= now
                    and (path is None or row["path"] == path)
                ),
                key=lambda row: (
                    row["available_at"],
                    row["spool_key"],
                ),
            )[:limit]
            claimed = []
            claimed_bytes = 0
            for row in eligible:
                item = self._spooled_telemetry(row)
                if claimed and claimed_bytes + item.payload_bytes > byte_limit:
                    break
                row["lease_owner"] = owner_id
                row["available_at"] = now + lease_duration
                row["attempts"] += 1
                row["updated_at"] = now
                item = self._spooled_telemetry(row)
                claimed.append(item)
                claimed_bytes += item.payload_bytes
            return claimed

    def complete_telemetry_spool(self, items: list[SpooledTelemetry]) -> int:
        with self._lock:
            removed = 0
            for item in items:
                row = self._telemetry_spool.get(item.spool_key)
                if row is None or row["revision"] != item.revision:
                    continue
                del self._telemetry_spool[item.spool_key]
                removed += 1
            return removed

    def abandon_telemetry_spool_claims(
        self,
        items: list[SpooledTelemetry],
        *,
        now: datetime,
    ) -> int:
        """Undo claims that never reached the replay executor."""

        abandoned = 0
        with self._lock:
            for item in items:
                row = self._telemetry_spool.get(item.spool_key)
                if row is None or row["revision"] != item.revision:
                    continue
                row["lease_owner"] = None
                row["available_at"] = now
                row["attempts"] = max(0, row["attempts"] - 1)
                row["updated_at"] = now
                abandoned += 1
        return abandoned

    def release_telemetry_spool(
        self,
        items: list[SpooledTelemetry],
        *,
        now: datetime,
        backoff: timedelta = timedelta(0),
        max_attempts: int | None = None,
    ) -> tuple[int, int]:
        """Return failed rows to the queue, or drop the hopeless ones.

        Returns ``(released, dropped)``. A row is only touched if its
        revision still matches: a sample that arrived while the replay was
        failing has already reset the row's availability, and pushing the
        backoff onto it would delay the newer payload.
        """

        limit = (
            self.TELEMETRY_SPOOL_MAX_ATTEMPTS if max_attempts is None else max_attempts
        )
        released = 0
        dropped = 0
        with self._lock:
            for item in items:
                row = self._telemetry_spool.get(item.spool_key)
                if row is None or row["revision"] != item.revision:
                    continue
                if row["attempts"] >= limit:
                    del self._telemetry_spool[item.spool_key]
                    dropped += 1
                    continue
                row["lease_owner"] = None
                row["available_at"] = now + backoff
                row["updated_at"] = now
                released += 1
        return released, dropped

    def telemetry_spool_stats(self, *, now: datetime | None = None) -> dict:
        observed_at = now or datetime.now(timezone.utc)
        with self._lock:
            rows = list(self._telemetry_spool.values())
        by_cluster: dict[str, int] = {}
        payload_bytes = 0
        leased_bytes = 0
        for row in rows:
            key = row["cluster_id"] or "__unscoped__"
            by_cluster[key] = by_cluster.get(key, 0) + 1
            size = len(
                json.dumps(
                    row["payload"],
                    separators=(",", ":"),
                ).encode()
            )
            payload_bytes += size
            if row["available_at"] > observed_at:
                leased_bytes += size
        return {
            "depth": len(rows),
            "leased": sum(row["available_at"] > observed_at for row in rows),
            "oldest_age_seconds": max(
                (
                    max(
                        0.0,
                        (observed_at - row["created_at"]).total_seconds(),
                    )
                    for row in rows
                ),
                default=0.0,
            ),
            "by_cluster": by_cluster,
            "payload_bytes": payload_bytes,
            "leased_bytes": leased_bytes,
        }
