from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event as ThreadEvent
from typing import Any, Callable

from gpu_fault.attempt_observation_state import (
    TERMINAL_WORKLOAD_PHASES,
    terminal_attempt_observation_state,
)
from gpu_fault.models import TerminalEvent
from gpu_fault.store.shared.group_commit import submit_group_commit
from gpu_fault.store.shared.time import (
    utc_text as _utc_text,
)
from gpu_fault.telemetry_models import (
    WorkloadObservationState,
)
from gpu_fault.training_models import TrainingProgressState

LOGGER = logging.getLogger(__name__)

_TERMINAL_PHASE_LIST = ", ".join(
    f"'{phase.value}'" for phase in sorted(TERMINAL_WORKLOAD_PHASES)
)
_TERMINAL_EVENT_KEY = (
    "{cluster_id} || '/' || {attempt_id} || '/TrainingAttemptTerminal'"
)

# One statement per hot-state mode. Each is driven by the observation rows that
# are still non-terminal (or, in dual mode, present on one side only) and joins
# the terminal event to them -- never the other way round. Driving from the
# event kind made the sweep scan every terminal event the cluster ever
# recorded, and read "event without observation row" as an inconsistency to
# repair, re-creating rows that retention had just deleted (F-G7 / P1-50D,
# P2-50G). ``%(cursor)s`` is the progress cursor: the observation key the
# previous sweep stopped at, or NULL.
_SWEEP_CANDIDATES = {
    "dedicated": f"""
        SELECT dedicated.key, event.payload
        FROM gpu_fault_attempt_observations AS dedicated
        JOIN gpu_fault_objects AS event
          ON event.kind='event'
         AND event.key={
        _TERMINAL_EVENT_KEY.format(
            cluster_id="dedicated.cluster_id", attempt_id="dedicated.attempt_id"
        )
    }
        WHERE dedicated.payload->'observation'->>'workload_phase'
              NOT IN ({_TERMINAL_PHASE_LIST})
          AND (%(cursor)s::text IS NULL OR dedicated.key > %(cursor)s)
        ORDER BY dedicated.key
        LIMIT %(limit)s
    """,
    "legacy": f"""
        SELECT legacy.key, event.payload
        FROM gpu_fault_objects AS legacy
        JOIN gpu_fault_objects AS event
          ON event.kind='event'
         AND event.key={
        _TERMINAL_EVENT_KEY.format(
            cluster_id="(legacy.payload->'observation'->>'cluster_id')",
            attempt_id="(legacy.payload->'observation'->>'attempt_id')",
        )
    }
        WHERE legacy.kind='attempt_observation'
          AND legacy.payload->'observation'->>'workload_phase'
              NOT IN ({_TERMINAL_PHASE_LIST})
          AND (%(cursor)s::text IS NULL OR legacy.key > %(cursor)s)
        ORDER BY legacy.key
        LIMIT %(limit)s
    """,
    "dual": f"""
        WITH legacy AS (
            SELECT key, payload
            FROM gpu_fault_objects
            WHERE kind='attempt_observation'
        ),
        rows AS (
            SELECT
                COALESCE(dedicated.key, legacy.key) AS key,
                COALESCE(
                    dedicated.cluster_id,
                    legacy.payload->'observation'->>'cluster_id'
                ) AS cluster_id,
                COALESCE(
                    dedicated.attempt_id,
                    legacy.payload->'observation'->>'attempt_id'
                ) AS attempt_id,
                dedicated.payload->'observation'->>'workload_phase'
                    AS dedicated_phase,
                legacy.payload->'observation'->>'workload_phase'
                    AS legacy_phase
            FROM gpu_fault_attempt_observations AS dedicated
            FULL OUTER JOIN legacy ON legacy.key=dedicated.key
        )
        SELECT rows.key, event.payload
        FROM rows
        JOIN gpu_fault_objects AS event
          ON event.kind='event'
         AND event.key={
        _TERMINAL_EVENT_KEY.format(
            cluster_id="rows.cluster_id", attempt_id="rows.attempt_id"
        )
    }
        WHERE (
              rows.dedicated_phase IS NULL
              OR rows.dedicated_phase NOT IN ({_TERMINAL_PHASE_LIST})
              OR rows.legacy_phase IS NULL
              OR rows.legacy_phase NOT IN ({_TERMINAL_PHASE_LIST})
          )
          AND (%(cursor)s::text IS NULL OR rows.key > %(cursor)s)
        ORDER BY rows.key
        LIMIT %(limit)s
    """,
}


@dataclass(frozen=True)
class _ObservationScanSource:
    """One of the two tables an attempt-observation read can come from.

    ``hot_state_mode`` decides which: ``dedicated`` reads the purpose-built table,
    ``legacy`` the generic object table, and ``dual`` both, merged. The two differ
    only in the table name, the ``ORDER BY`` expression and a mandatory kind
    predicate, so they share one query builder rather than two near-copies.
    """

    table: str
    order_expression: str
    cluster_expression: str
    kind_clause: str | None = None

    def query(
        self,
        cluster_id: str | None,
        *,
        limit: int | None,
        newest_first: bool,
    ) -> tuple[str, list[Any]]:
        clauses = [] if self.kind_clause is None else [self.kind_clause]
        parameters: list[Any] = []
        if cluster_id is not None:
            clauses.append(f"{self.cluster_expression}=%s")
            parameters.append(cluster_id)
        query = f"SELECT key, payload FROM {self.table}"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        direction = "DESC" if newest_first else "ASC"
        query += f" ORDER BY {self.order_expression} {direction}, key"
        if limit is not None:
            # Pushed into SQL rather than sliced after the fetch: the whole point
            # of the bound is that a week of rows is never decoded, and slicing in
            # Python would decode them all first.
            query += " LIMIT %s"
            parameters.append(limit)
        return query, parameters


_DEDICATED_OBSERVATION_SCAN = _ObservationScanSource(
    table="gpu_fault_attempt_observations",
    order_expression="observed_at",
    cluster_expression="cluster_id",
)
_LEGACY_OBSERVATION_SCAN = _ObservationScanSource(
    table="gpu_fault_objects",
    order_expression="payload->'observation'->>'observed_at'",
    cluster_expression="payload->'observation'->>'cluster_id'",
    kind_clause="kind='attempt_observation'",
)


class PostgresCollectorTelemetryMixin:
    # Attributes supplied by the composed concrete implementation.
    _attempt_observation_queue: Any

    _attempt_observation_condition: Any
    _attempt_observation_leader_active: bool = False
    # Where the terminalization sweep stopped last time (observation key), or
    # None once it has wrapped. Per store instance; a lost cursor only costs
    # one pass from the start.
    _attempt_observation_sweep_cursor: str | None = None
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

            def flush(batch: list[Any]) -> None:
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

            try:
                submit_group_commit(
                    entry,
                    condition=self._attempt_observation_condition,
                    queue=self._attempt_observation_queue,
                    host=self,
                    active_attr="_attempt_observation_leader_active",
                    flush=flush,
                )
            except TimeoutError as exc:
                raise TimeoutError("attempt observation batch did not flush") from exc
            if entry["error"] is not None:
                raise entry["error"]
            return bool(entry["result"])
        return self._save_attempt_observation_now(observation)

    def _save_attempt_observation_now(self, observation) -> bool:
        storage_key = self._state_key((observation.cluster_id, observation.attempt_id))
        with self._state_transaction(f"attempt_observation/{storage_key}"):
            event = self._get_optional(
                "event",
                (
                    f"{observation.cluster_id}/{observation.attempt_id}/"
                    "TrainingAttemptTerminal"
                ),
            )
            if event is not None:
                self._terminalize_attempt_observation(event)
                return False
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

    def _read_dedicated_observation(self, storage_key: str) -> Any:
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
        return self._decode("attempt_observation", row[0]) if row is not None else None

    def _terminalize_attempt_observation(self, event: TerminalEvent) -> bool:
        """Write the terminal observation state for ``event`` where it differs.

        In ``dual`` mode the two tables are compared and written separately:
        comparing only against the dedicated row meant a stale legacy row (or a
        legacy row with no dedicated twin) was never rewritten, so the sweep
        selected it again on every tick and never converged (F-G7 / P1-50E).
        """

        storage_key = self._state_key((event.cluster_id, event.attempt_id))
        writes_dedicated = self.hot_state_mode != "legacy"
        writes_legacy = self.hot_state_mode != "dedicated"
        dedicated_previous = (
            self._read_dedicated_observation(storage_key) if writes_dedicated else None
        )
        legacy_previous = (
            self._get_optional("attempt_observation", storage_key)
            if writes_legacy
            else None
        )
        previous = (
            dedicated_previous if dedicated_previous is not None else legacy_previous
        )
        terminal = terminal_attempt_observation_state(event, previous)
        written = False
        if writes_dedicated and terminal != dedicated_previous:
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
                        event.cluster_id,
                        event.attempt_id,
                        terminal.observation.observed_at,
                        terminal.model_dump_json(),
                    ),
                )
            written = True
        if writes_legacy and terminal != legacy_previous:
            self._put("attempt_observation", storage_key, terminal)
            written = True
        return written

    def _reconcile_terminal_attempt_observations(self, limit: int) -> int:
        """Terminalize observation rows contradicted by a terminal event.

        Bounded by the non-terminal observation rows, not by the event history
        (see ``_SWEEP_CANDIDATES``). Each row is rewritten under its own
        ``attempt_observation/<key>`` advisory lock -- the lock every other
        writer of that row takes (P1-44L) -- and a row that cannot be processed
        is logged and skipped rather than aborting the sweep: the progress
        cursor moves past it, so one poisoned row no longer pins the head of
        every subsequent sweep.
        """

        if limit < 1:
            return 0
        cursor_key = self._attempt_observation_sweep_cursor
        with self._db.cursor() as cursor:
            cursor.execute(
                _SWEEP_CANDIDATES[self.hot_state_mode],
                {"cursor": cursor_key, "limit": limit},
            )
            rows = cursor.fetchall()
        self._attempt_observation_sweep_cursor = (
            rows[-1][0] if len(rows) >= limit else None
        )
        terminalized = 0
        for storage_key, payload in rows:
            try:
                event = self._decode("event", payload)
                with self._state_transaction(f"attempt_observation/{storage_key}"):
                    terminalized += int(self._terminalize_attempt_observation(event))
            except Exception:
                LOGGER.exception(
                    "attempt observation sweep skipped %s; it will be retried "
                    "after the cursor wraps",
                    storage_key,
                )
        return terminalized

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
                    SELECT pg_advisory_xact_lock(
                        hashtextextended(
                            'attempt_observation/' || value,
                            0
                        )
                    )
                    FROM (
                        SELECT value
                        FROM unnest(%s::text[]) AS value
                        ORDER BY value
                    ) AS ordered
                    """,
                    (sorted(keys),),
                )
                cursor.execute(
                    """
                    SELECT payload
                    FROM gpu_fault_objects
                    WHERE kind='event'
                      AND key=ANY(%s::text[])
                    ORDER BY key
                    """,
                    (
                        [
                            (
                                f"{item.cluster_id}/{item.attempt_id}/"
                                "TrainingAttemptTerminal"
                            )
                            for item in values
                        ],
                    ),
                )
                terminal_events = [
                    self._decode("event", row[0]) for row in cursor.fetchall()
                ]
            for event in terminal_events:
                self._terminalize_attempt_observation(event)
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
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM gpu_fault_objects AS terminal
                            WHERE terminal.kind='event'
                              AND terminal.key=(
                                  input.cluster_id || '/' ||
                                  input.attempt_id ||
                                  '/TrainingAttemptTerminal'
                              )
                        )
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

    def list_attempt_observation_states(
        self,
        cluster_id: str | None = None,
        *,
        limit: int | None = None,
        newest_first: bool = False,
    ):
        if self.hot_state_mode == "legacy":
            return super().list_attempt_observation_states(
                cluster_id,
                limit=limit,
                newest_first=newest_first,
            )
        by_key = self._scan_attempt_observations(
            _DEDICATED_OBSERVATION_SCAN,
            cluster_id,
            limit=limit,
            newest_first=newest_first,
        )
        if self.hot_state_mode == "dual":
            # Taking the newest ``limit`` from each source and then the newest
            # ``limit`` of the merge is exactly the newest ``limit`` of the union,
            # because any row in the union's top slice is also in its own source's
            # top slice. The dedicated table wins on key collisions.
            legacy = self._scan_attempt_observations(
                _LEGACY_OBSERVATION_SCAN,
                cluster_id,
                limit=limit,
                newest_first=newest_first,
            )
            legacy.update(by_key)
            by_key = legacy
        ordered = sorted(
            by_key.values(),
            key=lambda item: (
                item.observation.observed_at,
                item.observation.attempt_id,
            ),
            reverse=newest_first,
        )
        return ordered if limit is None else ordered[:limit]

    def _scan_attempt_observations(
        self,
        source: _ObservationScanSource,
        cluster_id: str | None,
        *,
        limit: int | None,
        newest_first: bool,
    ) -> dict[str, Any]:
        query, parameters = source.query(
            cluster_id,
            limit=limit,
            newest_first=newest_first,
        )
        with self._db.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
        return {
            key: self._decode("attempt_observation", payload) for key, payload in rows
        }

    def _legacy_list_attempt_observation_states(
        self,
        cluster_id: str | None = None,
        *,
        limit: int | None = None,
        newest_first: bool = False,
    ):
        return list(
            self._scan_attempt_observations(
                _LEGACY_OBSERVATION_SCAN,
                cluster_id,
                limit=limit,
                newest_first=newest_first,
            ).values()
        )

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
