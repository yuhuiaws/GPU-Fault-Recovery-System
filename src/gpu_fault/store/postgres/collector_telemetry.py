from __future__ import annotations

from typing import Any, Callable

from contextlib import contextmanager
from threading import Event as ThreadEvent

from gpu_fault.store.shared.time import (
    utc_text as _utc_text,
)
from gpu_fault.telemetry_models import (
    WorkloadObservationState,
)
from gpu_fault.training_models import TrainingProgressState


class PostgresCollectorTelemetryMixin:
    # Attributes supplied by the composed concrete implementation.
    _attempt_observation_queue: Any

    _attempt_observation_condition: Any
    _db: Any
    _decode: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _put: Callable[..., Any]
    _state_key: Callable[..., Any]
    _state_transaction: Callable[..., Any]
    hot_state_mode: Any

    @contextmanager
    def collector_ingestion_transaction(
        self, cluster_id: str, node_id: str, batch_id: str
    ):
        with self._state_transaction(
            f"collector_ingestion/{cluster_id}/{node_id}/{batch_id}"
        ):
            yield

    def save_collector_statuses_batch(self, statuses) -> list[bool]:
        if not statuses:
            return []
        storage_keys = [
            self._state_key(
                (
                    status.cluster_id,
                    status.node_id,
                    status.collector.value,
                )
            )
            for status in statuses
        ]
        unique_keys = list(dict.fromkeys(storage_keys))
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT key, payload
                FROM gpu_fault_objects
                WHERE kind='collector_status'
                  AND key=ANY(%s)
                """,
                (unique_keys,),
            )
            current_by_key = {
                key: self._decode("collector_status", payload)
                for key, payload in cursor.fetchall()
            }
        results = []
        final_by_key = {}
        for storage_key, status in zip(storage_keys, statuses, strict=True):
            previous = current_by_key.get(storage_key)
            if previous is not None and status.observed_at < previous.observed_at:
                results.append(False)
                continue
            if previous is not None:
                status = status.model_copy(
                    update={
                        "last_success_at": (
                            status.last_success_at or previous.last_success_at
                        ),
                        "last_error_at": (
                            status.last_error_at or previous.last_error_at
                        ),
                    }
                )
            current_by_key[storage_key] = status
            final_by_key[storage_key] = status
            results.append(True)
        if final_by_key:
            keys = list(final_by_key)
            payloads = [final_by_key[key].model_dump_json() for key in keys]
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_objects(kind, key, payload)
                    SELECT
                        'collector_status',
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
                        )::timestamptz >=
                        (
                            gpu_fault_objects.payload->>'observed_at'
                        )::timestamptz
                    """,
                    (keys, payloads),
                )
        return results

    def list_collector_statuses(self, cluster_id: str, node_id: str | None = None):
        clauses = [
            "kind='collector_status'",
            "payload->>'cluster_id'=%s",
        ]
        parameters: list[object] = [cluster_id]
        if node_id is not None:
            clauses.append("payload->>'node_id'=%s")
            parameters.append(node_id)
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE """
                + " AND ".join(clauses)
                + """
                ORDER BY payload->>'node_id',
                         payload->>'collector',
                         key
                """,
                parameters,
            )
            rows = cursor.fetchall()
        return [self._decode("collector_status", row[0]) for row in rows]

    def list_telemetry_metrics_latest(self, cluster_id: str, node_id: str):
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='telemetry_metric_latest'
                  AND payload->>'cluster_id'=%s
                  AND payload->>'node_id'=%s
                ORDER BY payload->>'device',
                         payload->>'name',
                         key
                """,
                (cluster_id, node_id),
            )
            rows = cursor.fetchall()
        return [self._decode("telemetry_metric_latest", row[0]) for row in rows]

    def observe_training_progress(self, progress):
        if self.hot_state_mode == "legacy":
            return super().observe_training_progress(progress)
        storage_key = self._state_key(
            (
                progress.cluster_id,
                progress.attempt_id,
                progress.rank,
            )
        )
        with self._state_transaction(f"training_progress/{storage_key}"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT payload
                    FROM gpu_fault_training_progress
                    WHERE key=%s
                    """,
                    (storage_key,),
                )
                row = cursor.fetchone()
            previous = (
                self._decode("training_progress", row[0]) if row is not None else None
            )
            if previous is None and self.hot_state_mode == "dual":
                previous = self._get_optional("training_progress", storage_key)
            if (
                previous is not None
                and progress.observed_at <= previous.heartbeat.observed_at
            ):
                return False
            advanced = (
                previous is None
                or progress.step is None
                or previous.heartbeat.step is None
                or progress.step > previous.heartbeat.step
            )
            state = TrainingProgressState(
                heartbeat=progress,
                last_progress_at=(
                    progress.observed_at if advanced else previous.last_progress_at
                ),
            )
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_training_progress(
                        key, cluster_id, attempt_id,
                        rank, observed_at, payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT(key) DO UPDATE SET
                        cluster_id=excluded.cluster_id,
                        attempt_id=excluded.attempt_id,
                        rank=excluded.rank,
                        observed_at=excluded.observed_at,
                        payload=excluded.payload
                    """,
                    (
                        storage_key,
                        progress.cluster_id,
                        progress.attempt_id,
                        progress.rank,
                        progress.observed_at,
                        state.model_dump_json(),
                    ),
                )
            if self.hot_state_mode == "dual":
                self._put("training_progress", storage_key, state)
            return previous.heartbeat if previous is not None else None

    def list_training_progress(self, cluster_id: str, attempt_id: str | None = None):
        return [
            item.heartbeat
            for item in self.list_training_progress_states(cluster_id, attempt_id)
        ]

    def list_training_progress_states(
        self, cluster_id: str, attempt_id: str | None = None
    ):
        if self.hot_state_mode == "legacy":
            return super().list_training_progress_states(cluster_id, attempt_id)
        clauses = ["cluster_id=%s"]
        parameters: list[object] = [cluster_id]
        if attempt_id is not None:
            clauses.append("attempt_id=%s")
            parameters.append(attempt_id)
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT key, payload
                FROM gpu_fault_training_progress
                WHERE """
                + " AND ".join(clauses)
                + """
                ORDER BY attempt_id, rank, key
                """,
                parameters,
            )
            rows = cursor.fetchall()
        by_key = {
            key: self._decode("training_progress", payload) for key, payload in rows
        }
        if self.hot_state_mode == "dual":
            legacy_clauses = [
                "kind='training_progress'",
                "payload->'heartbeat'->>'cluster_id'=%s",
            ]
            legacy_parameters = [cluster_id]
            if attempt_id is not None:
                legacy_clauses.append("payload->'heartbeat'->>'attempt_id'=%s")
                legacy_parameters.append(attempt_id)
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT key, payload
                    FROM gpu_fault_objects
                    WHERE """
                    + " AND ".join(legacy_clauses),
                    legacy_parameters,
                )
                legacy_rows = cursor.fetchall()
            legacy = {
                key: self._decode("training_progress", payload)
                for key, payload in legacy_rows
            }
            legacy.update(by_key)
            by_key = legacy
        return sorted(
            by_key.values(),
            key=lambda item: (
                item.heartbeat.attempt_id,
                item.heartbeat.rank,
            ),
        )

    def list_attempt_observations(self, cluster_id: str):
        return [
            state.observation
            for state in self.list_attempt_observation_states(cluster_id)
        ]

    def save_attempt_observation(self, observation) -> bool:
        if self.hot_state_mode == "legacy":
            return super().save_attempt_observation(observation)
        if self.hot_state_mode == "dedicated":
            entry = {
                "observation": observation,
                "event": ThreadEvent(),
                "result": False,
                "error": None,
            }
            with self._attempt_observation_condition:
                leader = not self._attempt_observation_queue
                self._attempt_observation_queue.append(entry)
                if not leader:
                    self._attempt_observation_condition.notify()
            if leader:
                with self._attempt_observation_condition:
                    self._attempt_observation_condition.wait(timeout=0.01)
                    batch = self._attempt_observation_queue[:64]
                    del self._attempt_observation_queue[: len(batch)]
                try:
                    accepted = self._save_attempt_observations_batch(
                        [item["observation"] for item in batch]
                    )
                    for item, value in zip(batch, accepted, strict=True):
                        item["result"] = value
                except Exception as exc:
                    for item in batch:
                        item["error"] = exc
                for item in batch:
                    item["event"].set()
            if not entry["event"].wait(timeout=30):
                raise TimeoutError("attempt observation batch did not flush")
            if entry["error"] is not None:
                raise entry["error"]
            return entry["result"]
        return self._save_attempt_observation_now(observation)

    def _save_attempt_observation_now(self, observation) -> bool:
        storage_key = self._state_key((observation.cluster_id, observation.attempt_id))
        with self._state_transaction(f"attempt_observation/{storage_key}"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT payload
                    FROM gpu_fault_attempt_observations
                    WHERE key=%s
                    """,
                    (storage_key,),
                )
                row = cursor.fetchone()
            previous = (
                self._decode("attempt_observation", row[0]) if row is not None else None
            )
            if previous is None and self.hot_state_mode == "dual":
                previous = self._get_optional("attempt_observation", storage_key)
            if (
                previous is not None
                and observation.observed_at < previous.observation.observed_at
            ):
                return False
            state = WorkloadObservationState(
                first_observed_at=(
                    previous.first_observed_at
                    if previous is not None
                    else observation.observed_at
                ),
                observation=observation,
            )
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_attempt_observations(
                        key, cluster_id, attempt_id,
                        observed_at, payload
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT(key) DO UPDATE SET
                        cluster_id=excluded.cluster_id,
                        attempt_id=excluded.attempt_id,
                        observed_at=excluded.observed_at,
                        payload=excluded.payload
                    """,
                    (
                        storage_key,
                        observation.cluster_id,
                        observation.attempt_id,
                        observation.observed_at,
                        state.model_dump_json(),
                    ),
                )
            if self.hot_state_mode == "dual":
                self._put(
                    "attempt_observation",
                    storage_key,
                    state,
                )
            return True

    def _save_attempt_observations_batch(self, observations) -> list[bool]:
        if not observations:
            return []
        selected_by_key = {}
        selected_indexes = {}
        for index, observation in enumerate(observations):
            storage_key = self._state_key(
                (
                    observation.cluster_id,
                    observation.attempt_id,
                )
            )
            previous = selected_by_key.get(storage_key)
            if previous is None or observation.observed_at >= previous.observed_at:
                selected_by_key[storage_key] = observation
                selected_indexes[storage_key] = index
        keys = list(selected_by_key)
        values = [selected_by_key[key] for key in keys]
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    WITH input AS (
                        SELECT *
                        FROM unnest(
                            %s::text[], %s::text[], %s::text[],
                            %s::text[], %s::text[]
                        ) AS value(
                            key, cluster_id, attempt_id,
                            observed_at, observation
                        )
                    ),
                    upserted AS (
                        INSERT INTO gpu_fault_attempt_observations(
                            key, cluster_id, attempt_id,
                            observed_at, payload
                        )
                        SELECT
                            key,
                            cluster_id,
                            attempt_id,
                            observed_at::timestamptz,
                            jsonb_build_object(
                                'first_observed_at',
                                observed_at::text,
                                'observation',
                                observation::jsonb
                            )
                        FROM input
                        ON CONFLICT(key) DO UPDATE SET
                            cluster_id=excluded.cluster_id,
                            attempt_id=excluded.attempt_id,
                            observed_at=excluded.observed_at,
                            payload=jsonb_set(
                                excluded.payload,
                                '{first_observed_at}',
                                gpu_fault_attempt_observations.payload
                                    ->'first_observed_at'
                            )
                        WHERE excluded.observed_at >=
                              gpu_fault_attempt_observations.observed_at
                        RETURNING key
                    )
                    SELECT key FROM upserted
                    """,
                    (
                        keys,
                        [item.cluster_id for item in values],
                        [item.attempt_id for item in values],
                        [_utc_text(item.observed_at) for item in values],
                        [item.model_dump_json() for item in values],
                    ),
                )
                accepted_keys = {row[0] for row in cursor.fetchall()}
        results = [False] * len(observations)
        for storage_key in accepted_keys:
            results[selected_indexes[storage_key]] = True
        return results

    def save_attempt_observations_batch(self, observations):
        if self.hot_state_mode == "dedicated":
            return self._save_attempt_observations_batch(observations)
        return super().save_attempt_observations_batch(observations)

    def list_attempt_observation_states(self, cluster_id: str | None = None):
        if self.hot_state_mode == "legacy":
            return super().list_attempt_observation_states(cluster_id)
        clauses = []
        parameters = []
        if cluster_id is not None:
            clauses.append("cluster_id=%s")
            parameters.append(cluster_id)
        query = "SELECT key, payload FROM gpu_fault_attempt_observations"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY observed_at, key"
        with self._db.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
        by_key = {
            key: self._decode("attempt_observation", payload) for key, payload in rows
        }
        if self.hot_state_mode == "dual":
            legacy_clauses = ["kind='attempt_observation'"]
            legacy_parameters = []
            if cluster_id is not None:
                legacy_clauses.append("payload->'observation'->>'cluster_id'=%s")
                legacy_parameters.append(cluster_id)
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT key, payload
                    FROM gpu_fault_objects
                    WHERE """
                    + " AND ".join(legacy_clauses),
                    legacy_parameters,
                )
                legacy_rows = cursor.fetchall()
            legacy = {
                key: self._decode("attempt_observation", payload)
                for key, payload in legacy_rows
            }
            legacy.update(by_key)
            by_key = legacy
        return sorted(
            by_key.values(),
            key=lambda item: (
                item.observation.observed_at,
                item.observation.attempt_id,
            ),
        )

    def _legacy_list_attempt_observation_states(self, cluster_id: str | None = None):
        clauses = ["kind='attempt_observation'"]
        parameters = []
        if cluster_id is not None:
            clauses.append("payload->'observation'->>'cluster_id'=%s")
            parameters.append(cluster_id)
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE """
                + " AND ".join(clauses)
                + """
                ORDER BY
                    payload->'observation'->>'observed_at',
                    key
                """,
                parameters,
            )
            rows = cursor.fetchall()
        return [self._decode("attempt_observation", row[0]) for row in rows]

    def observe_telemetry_metrics(self, items) -> list[bool]:
        if not items:
            return []
        storage_keys = [
            self._state_key(
                (
                    latest.cluster_id,
                    latest.node_id,
                    latest.device or "node",
                    latest.name,
                )
            )
            for latest in items
        ]
        unique_keys = list(dict.fromkeys(storage_keys))
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT key, payload
                FROM gpu_fault_objects
                WHERE kind='telemetry_metric_latest'
                  AND key=ANY(%s)
                """,
                (unique_keys,),
            )
            current_by_key = {
                key: self._decode("telemetry_metric_latest", payload)
                for key, payload in cursor.fetchall()
            }
        results = []
        final_by_key = {}
        for storage_key, latest in zip(storage_keys, items, strict=True):
            previous = current_by_key.get(storage_key)
            if previous is not None and latest.observed_at <= previous.observed_at:
                results.append(False)
                continue
            current_by_key[storage_key] = latest
            final_by_key[storage_key] = latest
            results.append(True)
        if final_by_key:
            keys = list(final_by_key)
            payloads = [final_by_key[key].model_dump_json() for key in keys]
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_objects(kind, key, payload)
                    SELECT
                        'telemetry_metric_latest',
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
