from __future__ import annotations

from typing import Any, Callable

from datetime import datetime

from gpu_fault.store.shared.time import (
    utc_text as _utc_text,
)


class PostgresFleetMixin:
    # Attributes supplied by the composed concrete implementation.
    _agent_key: Callable[..., Any]
    _db: Any
    _decode: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def _regional_cluster_region(self, cluster_id: str) -> str | None:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload->>'region'
                FROM gpu_fault_objects
                WHERE kind='regional_cluster' AND key=%s
                """,
                (cluster_id,),
            )
            row = cursor.fetchone()
        return str(row[0]) if row and row[0] is not None else None

    def list_regional_cluster_ids(self) -> list[str]:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT key
                FROM gpu_fault_objects
                WHERE kind='regional_cluster'
                ORDER BY key
                """
            )
            rows = cursor.fetchall()
        return [str(row[0]) for row in rows]

    def list_regional_clusters(self):
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='regional_cluster'
                ORDER BY payload->>'cluster_id', key
                """
            )
            rows = cursor.fetchall()
        return [self._decode("regional_cluster", row[0]) for row in rows]

    def list_barriers(self):
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='barrier'
                ORDER BY payload->>'created_at', key
                """
            )
            rows = cursor.fetchall()
        return [self._decode("barrier", row[0]) for row in rows]

    def list_agents(self, cluster_id: str | None = None):
        clauses = ["kind='agent'"]
        parameters = []
        if cluster_id is not None:
            clauses.append("payload->>'cluster_id'=%s")
            parameters.append(cluster_id)
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE """
                + " AND ".join(clauses)
                + """
                ORDER BY payload->>'cluster_id',
                         payload->>'node_id'
                """,
                parameters,
            )
            rows = cursor.fetchall()
        return [self._decode("agent", row[0]) for row in rows]

    def list_fleet_deployments(self):
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='fleet_deployment'
                ORDER BY payload->>'created_at', key
                """
            )
            rows = cursor.fetchall()
        return [self._decode("fleet_deployment", row[0]) for row in rows]

    def list_active_fleet_deployments(self, cluster_id: str):
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='fleet_deployment'
                  AND payload->>'cluster_id'=%s
                  AND payload->>'status'
                      NOT IN ('SUCCEEDED', 'FAILED')
                ORDER BY payload->>'created_at', key
                """,
                (cluster_id,),
            )
            rows = cursor.fetchall()
        return [self._decode("fleet_deployment", row[0]) for row in rows]

    def cleanup_terminal_fleet_deployments(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        """Set-based retention delete, ordering and limit kept in SQL.

        The parent implementation decodes every deployment to sort them
        in Python, which is exactly the cost the retention drain exists
        to remove.
        """

        with self._state_transaction("fleet_deployment/cleanup"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    WITH victims AS (
                        SELECT key
                        FROM gpu_fault_objects
                        WHERE kind='fleet_deployment'
                          AND payload->>'status' IN (
                              'SUCCEEDED', 'FAILED'
                          )
                          AND payload->>'updated_at' <= %s
                        ORDER BY payload->>'updated_at', key
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    ),
                    deleted AS (
                        DELETE FROM gpu_fault_objects AS objects
                        USING victims
                        WHERE objects.kind='fleet_deployment'
                          AND objects.key=victims.key
                        RETURNING objects.key
                    )
                    SELECT count(*) FROM deleted
                    """,
                    (_utc_text(older_than), limit),
                )
                row = cursor.fetchone()
            return int(row[0]) if row else 0

    def replace_agent_if_matches(self, replacement, expected) -> bool:
        key = self._agent_key(
            replacement.cluster_id,
            replacement.node_id,
        )
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                        hashtextextended(%s, 0)
                    )
                    """,
                    (f"agent/{key}",),
                )
            current = self._get_optional("agent", key)
            if current != expected:
                return False
            self._put("agent", key, replacement)
            return True
