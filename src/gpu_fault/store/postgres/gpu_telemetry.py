from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.gpu_metric_models import GpuFindingState
from gpu_fault.store.shared.time import (
    utc_text as _utc_text,
)


class PostgresGpuTelemetryMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _decode: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _put: Callable[..., Any]
    _state_key: Callable[..., Any]
    _state_transaction: Callable[..., Any]
    hot_state_mode: Any

    def observe_gpu_inventory_snapshots(self, snapshots):
        if not snapshots:
            return []
        storage_keys = [
            self._state_key((snapshot.cluster_id, snapshot.node_id))
            for snapshot in snapshots
        ]
        unique_keys = list(dict.fromkeys(storage_keys))
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT key, payload
                FROM gpu_fault_objects
                WHERE kind='gpu_inventory_snapshot'
                  AND key=ANY(%s)
                """,
                (unique_keys,),
            )
            current_by_key = {
                key: self._decode("gpu_inventory_snapshot", payload)
                for key, payload in cursor.fetchall()
            }
        results = []
        final_by_key = {}
        for storage_key, snapshot in zip(storage_keys, snapshots, strict=True):
            previous = current_by_key.get(storage_key)
            if previous is not None and snapshot.observed_at <= previous.observed_at:
                results.append(False)
                continue
            results.append(previous)
            current_by_key[storage_key] = snapshot
            final_by_key[storage_key] = snapshot
        if final_by_key:
            keys = list(final_by_key)
            payloads = [final_by_key[key].model_dump_json() for key in keys]
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_objects(kind, key, payload)
                    SELECT
                        'gpu_inventory_snapshot',
                        batch.key,
                        batch.payload::jsonb
                    FROM unnest(
                        %s::text[], %s::text[]
                    ) AS batch(key, payload)
                    ON CONFLICT(kind, key) DO UPDATE SET
                        payload=excluded.payload
                    WHERE
                        (
                            excluded.payload->>'observed_at'
                        )::timestamptz >
                        (
                            gpu_fault_objects.payload->>'observed_at'
                        )::timestamptz
                    """,
                    (keys, payloads),
                )
        return results

    def get_gpu_metrics_batch(self, key):
        if self.hot_state_mode == "legacy":
            return super().get_gpu_metrics_batch(key)
        storage_key = self._state_key(key)
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_gpu_metrics_batches
                WHERE key=%s
                """,
                (storage_key,),
            )
            row = cursor.fetchone()
        if row is not None:
            return self._decode("gpu_metrics_batch", row[0])
        if self.hot_state_mode == "dual":
            return self._get_optional("gpu_metrics_batch", storage_key)
        return None

    def save_gpu_metrics_batch(self, key, result):
        if self.hot_state_mode == "legacy":
            return super().save_gpu_metrics_batch(key, result)
        storage_key = self._state_key(key)
        cluster_id, node_id, _batch_id = key
        with self._state_transaction(f"gpu_metrics_batch/{storage_key}"):
            existing = self.get_gpu_metrics_batch(key)
            if existing is not None:
                return existing.model_copy(update={"duplicate": True})
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_gpu_metrics_batches(
                        key, cluster_id, node_id, created_at, payload
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT(key) DO NOTHING
                    """,
                    (
                        storage_key,
                        cluster_id,
                        node_id,
                        datetime.now(timezone.utc),
                        result.model_dump_json(),
                    ),
                )
            if self.hot_state_mode == "dual":
                self._put("gpu_metrics_batch", storage_key, result)
            return result

    def observe_gpu_metrics(self, items):
        if not items:
            return []
        if self.hot_state_mode == "legacy":
            return super().observe_gpu_metrics(items)
        scopes = {(key[0], key[1]) for key, _ in items}
        if len(scopes) != 1:
            raise ValueError("one GPU metric batch must target one cluster/node")
        cluster_id, node_id = next(iter(scopes))
        storage_keys = [self._state_key(key) for key, _ in items]
        with self._state_transaction(f"gpu_metric_latest/batch/{cluster_id}/{node_id}"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT key, payload
                    FROM gpu_fault_gpu_metric_latest
                    WHERE key=ANY(%s)
                    """,
                    (storage_keys,),
                )
                previous_by_key = {
                    key: self._decode("gpu_metric_latest", payload)
                    for key, payload in cursor.fetchall()
                }
                if self.hot_state_mode == "dual":
                    missing = [
                        key for key in storage_keys if key not in previous_by_key
                    ]
                    if missing:
                        cursor.execute(
                            """
                            SELECT key, payload
                            FROM gpu_fault_objects
                            WHERE kind='gpu_metric_latest'
                              AND key=ANY(%s)
                            """,
                            (missing,),
                        )
                        previous_by_key.update(
                            {
                                key: self._decode(
                                    "gpu_metric_latest",
                                    payload,
                                )
                                for key, payload in cursor.fetchall()
                            }
                        )
            results = []
            final_by_key = {}
            current_by_key = dict(previous_by_key)
            for storage_key, (_, latest) in zip(storage_keys, items, strict=True):
                previous = current_by_key.get(storage_key)
                if previous is not None and latest.observed_at <= previous.observed_at:
                    results.append(False)
                    continue
                results.append(previous)
                current_by_key[storage_key] = latest
                final_by_key[storage_key] = latest
            if final_by_key:
                keys = list(final_by_key)
                payloads = [final_by_key[key].model_dump_json() for key in keys]
                with self._db.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO gpu_fault_gpu_metric_latest(
                            key, cluster_id, node_id,
                            observed_at, payload
                        )
                        SELECT
                            batch.key,
                            %s,
                            %s,
                            batch.observed_at::timestamptz,
                            batch.payload::jsonb
                        FROM unnest(
                            %s::text[],
                            %s::text[],
                            %s::text[]
                        ) AS batch(key, observed_at, payload)
                        ON CONFLICT(key) DO UPDATE SET
                            cluster_id=excluded.cluster_id,
                            node_id=excluded.node_id,
                            observed_at=excluded.observed_at,
                            payload=excluded.payload
                        """,
                        (
                            cluster_id,
                            node_id,
                            keys,
                            [_utc_text(final_by_key[key].observed_at) for key in keys],
                            payloads,
                        ),
                    )
                    if self.hot_state_mode == "dual":
                        cursor.execute(
                            """
                            INSERT INTO gpu_fault_objects(
                                kind, key, payload
                            )
                            SELECT
                                'gpu_metric_latest',
                                batch.key,
                                batch.payload::jsonb
                            FROM unnest(
                                %s::text[], %s::text[]
                            ) AS batch(key, payload)
                            ON CONFLICT(kind, key)
                            DO UPDATE SET payload=excluded.payload
                            """,
                            (keys, payloads),
                        )
            return results

    def observe_gpu_metric(self, key, latest):
        return self.observe_gpu_metrics([(key, latest)])[0]

    def get_gpu_finding_states(self, keys):
        if not keys:
            return []
        storage_keys = [self._state_key(key) for key in keys]
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT key, payload
                FROM gpu_fault_objects
                WHERE kind='gpu_finding_state'
                  AND key=ANY(%s)
                """,
                (storage_keys,),
            )
            by_key = {
                key: self._decode("gpu_finding_state", payload)
                for key, payload in cursor.fetchall()
            }
        return [by_key.get(key) for key in storage_keys]

    def update_gpu_findings(self, items) -> list[bool]:
        if not items:
            return []
        scopes = {(key[0], key[1]) for key, _, _ in items}
        if len(scopes) != 1:
            raise ValueError("one GPU finding batch must target one cluster/node")
        cluster_id, node_id = next(iter(scopes))
        storage_keys = [self._state_key(key) for key, _, _ in items]
        with self._state_transaction(f"gpu_finding_state/batch/{cluster_id}/{node_id}"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT key, payload
                    FROM gpu_fault_objects
                    WHERE kind='gpu_finding_state'
                      AND key=ANY(%s)
                    """,
                    (storage_keys,),
                )
                current_by_key = {
                    key: self._decode("gpu_finding_state", payload)
                    for key, payload in cursor.fetchall()
                }
            activated_values = []
            final_states = {}
            history = {}
            for (
                storage_key,
                (_, finding, observed_at),
            ) in zip(storage_keys, items, strict=True):
                previous = current_by_key.get(storage_key)
                if previous is not None and observed_at <= previous.observed_at:
                    activated_values.append(False)
                    continue
                activated = finding is not None and (
                    previous is None
                    or previous.finding is None
                    or finding.severity != previous.finding.severity
                    or finding.automatic_action != previous.finding.automatic_action
                )
                consecutive_breaches = (
                    previous.consecutive_breaches + 1
                    if finding is not None
                    and previous is not None
                    and previous.finding is not None
                    else 1
                    if finding is not None
                    else 0
                )
                state = GpuFindingState(
                    observed_at=observed_at,
                    finding=finding,
                    consecutive_breaches=consecutive_breaches,
                )
                current_by_key[storage_key] = state
                final_states[storage_key] = state
                activated_values.append(activated)
                if activated:
                    history[finding.finding_id] = finding
            with self._db.cursor() as cursor:
                if final_states:
                    keys = list(final_states)
                    payloads = [final_states[key].model_dump_json() for key in keys]
                    cursor.execute(
                        """
                        INSERT INTO gpu_fault_objects(
                            kind, key, payload
                        )
                        SELECT
                            'gpu_finding_state',
                            batch.key,
                            batch.payload::jsonb
                        FROM unnest(
                            %s::text[], %s::text[]
                        ) AS batch(key, payload)
                        ON CONFLICT(kind, key)
                        DO UPDATE SET payload=excluded.payload
                        """,
                        (keys, payloads),
                    )
                if history:
                    keys = list(history)
                    payloads = [history[key].model_dump_json() for key in keys]
                    cursor.execute(
                        """
                        INSERT INTO gpu_fault_objects(
                            kind, key, payload
                        )
                        SELECT
                            'gpu_finding_history',
                            batch.key,
                            batch.payload::jsonb
                        FROM unnest(
                            %s::text[], %s::text[]
                        ) AS batch(key, payload)
                        ON CONFLICT(kind, key)
                        DO UPDATE SET payload=excluded.payload
                        """,
                        (keys, payloads),
                    )
            return activated_values

    def update_gpu_finding(self, key, finding, observed_at) -> bool:
        return self.update_gpu_findings([(key, finding, observed_at)])[0]

    def get_gpu_finding_state(self, key):
        return self.get_gpu_finding_states([key])[0]

    def list_gpu_metrics_latest(self, cluster_id: str, node_id: str):
        if self.hot_state_mode == "legacy":
            return super().list_gpu_metrics_latest(cluster_id, node_id)
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT key, payload
                FROM gpu_fault_gpu_metric_latest
                WHERE cluster_id=%s
                  AND node_id=%s
                ORDER BY key
                """,
                (cluster_id, node_id),
            )
            rows = cursor.fetchall()
        by_key = {
            key: self._decode("gpu_metric_latest", payload) for key, payload in rows
        }
        if self.hot_state_mode == "dual":
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT key, payload
                    FROM gpu_fault_objects
                    WHERE kind='gpu_metric_latest'
                      AND payload->>'cluster_id'=%s
                      AND payload->>'node_id'=%s
                    """,
                    (cluster_id, node_id),
                )
                legacy_rows = cursor.fetchall()
            legacy = {
                key: self._decode("gpu_metric_latest", payload)
                for key, payload in legacy_rows
            }
            legacy.update(by_key)
            by_key = legacy
        return [by_key[key] for key in sorted(by_key)]

    def list_gpu_findings(
        self,
        cluster_id: str,
        node_id: str,
        *,
        active_only: bool,
    ):
        if active_only:
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT payload
                    FROM gpu_fault_objects
                    WHERE kind='gpu_finding_state'
                      AND payload->'finding' IS NOT NULL
                      AND payload->'finding'->>'cluster_id'=%s
                      AND payload->'finding'->>'node_id'=%s
                    ORDER BY key
                    """,
                    (cluster_id, node_id),
                )
                rows = cursor.fetchall()
            return [
                state.finding
                for state in (self._decode("gpu_finding_state", row[0]) for row in rows)
                if state.finding is not None
            ]
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='gpu_finding_history'
                  AND payload->>'cluster_id'=%s
                  AND payload->>'node_id'=%s
                ORDER BY key
                """,
                (cluster_id, node_id),
            )
            rows = cursor.fetchall()
        return [self._decode("gpu_finding_history", row[0]) for row in rows]
