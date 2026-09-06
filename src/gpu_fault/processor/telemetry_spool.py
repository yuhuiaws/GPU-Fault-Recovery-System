from __future__ import annotations

from typing import Any, Callable

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import logging
import time
from urllib import request as urllib_request

from gpu_fault.transport.http_client import urlopen

LOGGER = logging.getLogger(__name__)


class TelemetrySpoolCoordinatorMixin:
    # Attributes supplied by the composed concrete implementation.
    _telemetry_spool_batch_bytes: Callable[..., Any]
    spool_replay_seconds_sum: Any
    telemetry_spool_enabled: Any
    telemetry_spool_max_in_flight_bytes: Any
    telemetry_spool_workers: Any

    _TELEMETRY_BATCH_ENVELOPE_BYTES: Any
    _TELEMETRY_SPOOL_PATH_SCHEDULE: Any
    _spool_fallback_polls_total: Any
    _spool_work_available: Any
    _state_lock: Any
    _stop: Any
    claim_lane_blocked_total: Any
    claim_probes_total: Any
    internal_token: Any
    is_healthy: Callable[..., Any]
    local_url: Any
    owner_id: Any
    poll_seconds: Any
    request_max_execution_seconds: Any
    spool_abandoned_total: Any
    spool_claim_empty_total: Any
    spool_claim_rounds_total: Any
    spool_claim_rows_by_path: Any
    spool_claim_rows_total: Any
    spool_completed_total: Any
    spool_direct_replay_total: Any
    spool_dropped_total: Any
    spool_errors_total: Any
    spool_http_replay_total: Any
    spool_released_total: Any
    spool_superseded_total: Any
    store: Any
    telemetry_spool_fault_pressure_poll_seconds: Any
    telemetry_spool_fault_pressure_workers: Any
    telemetry_spool_lease_seconds: Any
    telemetry_spool_notification_fallback_seconds: Any
    telemetry_spool_replay_batch_max_bytes: Any
    telemetry_spool_replay_batch_max_items: Any
    telemetry_spool_replay_handler: Callable[[dict[str, Any]], dict[str, Any]] | None
    telemetry_spool_retry_backoff_seconds: Any

    def run_telemetry_spool(self) -> None:
        """Drain telemetry through the dedicated spool path."""

        if not self.telemetry_spool_enabled:
            return
        pool = ThreadPoolExecutor(
            max_workers=self.telemetry_spool_workers,
            thread_name_prefix="gpu-fault-telemetry-spool",
        )
        with self._state_lock:
            self._spool_consumer_running = True
        in_flight: dict[Future, int] = {}
        path_cursor = 0
        next_fault_pressure_check = 0.0
        try:
            while not self._stop.is_set():
                completed = {future for future in in_flight if future.done()}
                for future in completed:
                    try:
                        future.result()
                    except Exception:
                        LOGGER.exception("telemetry spool replay failed")
                for future in completed:
                    in_flight.pop(future, None)
                with self._state_lock:
                    self._spool_in_flight_bytes = sum(in_flight.values())
                if not self.is_healthy():
                    self._stop.wait(self.poll_seconds)
                    continue
                monotonic = time.monotonic()
                if monotonic >= next_fault_pressure_check:
                    try:
                        fault_backlog = self.store.processor_fault_backlog_depth()
                    except Exception:
                        LOGGER.exception("telemetry spool fault backlog probe failed")
                    else:
                        with self._state_lock:
                            self._spool_fault_backlog_depth = fault_backlog
                            self._spool_fault_pressure_active = fault_backlog > 0
                    next_fault_pressure_check = (
                        monotonic + self.telemetry_spool_fault_pressure_poll_seconds
                    )
                effective_workers = (
                    self.telemetry_spool_fault_pressure_workers
                    if self._spool_fault_pressure_active
                    else self.telemetry_spool_workers
                )
                available = effective_workers - len(in_flight)
                if available <= 0:
                    self._stop.wait(self.poll_seconds)
                    continue
                # Clearing before the claim closes the notification race:
                # an earlier signal already has a row visible to the claim,
                # while a signal arriving during the claim remains set and
                # prevents the fallback sleep.
                self._spool_work_available.clear()
                claimed_any = False
                claim_failed = False
                empty_paths: set[str] = set()
                # Derived from the registry, not written down: the schedule
                # repeats weighted paths, and "every path came back empty"
                # means every distinct one (F-E6 - a literal 3 against four
                # spoolable paths left the fourth to the fallback sleep).
                spool_paths = frozenset(self._TELEMETRY_SPOOL_PATH_SCHEDULE)
                for _slot in range(available):
                    in_flight_bytes = sum(in_flight.values())
                    available_bytes = (
                        self.telemetry_spool_max_in_flight_bytes - in_flight_bytes
                    )
                    payload_budget = (
                        min(
                            self.telemetry_spool_replay_batch_max_bytes,
                            available_bytes,
                        )
                        - self._TELEMETRY_BATCH_ENVELOPE_BYTES
                    )
                    if payload_budget <= 0:
                        break
                    rows = []
                    selected_path = None
                    for _path_attempt in range(
                        len(self._TELEMETRY_SPOOL_PATH_SCHEDULE)
                    ):
                        selected_path = self._TELEMETRY_SPOOL_PATH_SCHEDULE[path_cursor]
                        path_cursor = (path_cursor + 1) % len(
                            self._TELEMETRY_SPOOL_PATH_SCHEDULE
                        )
                        if selected_path in empty_paths:
                            continue
                        try:
                            rows = self.store.claim_telemetry_spool(
                                self.owner_id,
                                now=datetime.now(timezone.utc),
                                lease_duration=timedelta(
                                    seconds=(self.telemetry_spool_lease_seconds)
                                ),
                                limit=(self.telemetry_spool_replay_batch_max_items),
                                max_bytes=payload_budget,
                                path=selected_path,
                            )
                        except Exception:
                            LOGGER.exception(
                                "telemetry spool claim failed path=%s",
                                selected_path,
                            )
                            claim_failed = True
                            break
                        with self._state_lock:
                            self.spool_claim_rounds_total += 1
                            self.spool_claim_rows_total += len(rows)
                            if not rows:
                                self.spool_claim_empty_total += 1
                            else:
                                self.spool_claim_rows_by_path[selected_path] = (
                                    self.spool_claim_rows_by_path.get(selected_path, 0)
                                    + len(rows)
                                )
                        if rows:
                            break
                        empty_paths.add(selected_path)
                        if empty_paths >= spool_paths:
                            break
                    if claim_failed:
                        break
                    if not rows:
                        break
                    claimed_any = True
                    batch_bytes = self._telemetry_spool_batch_bytes(rows)
                    if (
                        self._stop.is_set()
                        or batch_bytes > self.telemetry_spool_replay_batch_max_bytes
                        or in_flight_bytes + batch_bytes
                        > self.telemetry_spool_max_in_flight_bytes
                    ):
                        self._abandon_telemetry_spool(rows)
                        break
                    try:
                        future = pool.submit(
                            self._replay_telemetry_spool,
                            rows,
                        )
                    except Exception:
                        self._abandon_telemetry_spool(rows)
                        raise
                    in_flight[future] = batch_bytes
                    with self._state_lock:
                        self._spool_in_flight_bytes = sum(in_flight.values())
                        self._spool_in_flight_max_bytes = max(
                            self._spool_in_flight_max_bytes,
                            self._spool_in_flight_bytes,
                        )
                if claim_failed:
                    self._stop.wait(self.poll_seconds)
                    continue
                if claimed_any:
                    continue
                # LISTEN/NOTIFY is the normal wake path. Poll only as a
                # bounded safety net for reconnect gaps, lost notifications
                # and retry rows whose future available_at has arrived.
                with self._state_lock:
                    self._spool_fallback_polls_total += 1
                self._spool_work_available.wait(
                    self.telemetry_spool_notification_fallback_seconds
                )
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            with self._state_lock:
                self._spool_consumer_running = False

    @staticmethod
    def _telemetry_spool_item_bytes(item) -> int:
        return len(
            json.dumps(
                {
                    "request_id": item.request_id,
                    "path": item.path,
                    "payload": item.payload,
                },
                separators=(",", ":"),
            ).encode()
        )

    def _telemetry_spool_batches(
        self,
        items: list,
        *,
        max_items: int,
    ):
        batch = []
        batch_bytes = 0
        for item in items:
            item_bytes = self._telemetry_spool_item_bytes(item) + 1
            if batch and (
                len(batch) >= max_items
                or batch_bytes + item_bytes
                > self.telemetry_spool_replay_batch_max_bytes
            ):
                yield batch
                batch = []
                batch_bytes = 0
            batch.append(item)
            batch_bytes += item_bytes
        if batch:
            yield batch

    def _abandon_telemetry_spool(self, items: list) -> None:
        try:
            abandoned = self.store.abandon_telemetry_spool_claims(
                items,
                now=datetime.now(timezone.utc),
            )
        except Exception:
            LOGGER.exception(
                "telemetry spool claim abandon failed keys=%s",
                [item.spool_key for item in items],
            )
            return
        with self._state_lock:
            self.spool_abandoned_total += abandoned

    def _release_telemetry_spool(
        self,
        items: list,
        *,
        backoff: timedelta | None = None,
    ) -> None:
        interval = (
            timedelta(seconds=self.telemetry_spool_retry_backoff_seconds)
            if backoff is None
            else backoff
        )
        try:
            released, dropped = self.store.release_telemetry_spool(
                items,
                now=datetime.now(timezone.utc),
                backoff=interval,
            )
        except Exception:
            LOGGER.exception(
                "telemetry spool release failed keys=%s",
                [item.spool_key for item in items],
            )
            return
        with self._state_lock:
            self.spool_released_total += released
            self.spool_dropped_total += dropped
        if dropped:
            LOGGER.error(
                "telemetry spool dropped %s samples after %s failed replays paths=%s",
                dropped,
                self.store.TELEMETRY_SPOOL_MAX_ATTEMPTS,
                sorted({item.path for item in items}),
            )

    def _replay_telemetry_spool(self, items: list) -> None:
        started = time.monotonic()
        batch_request = {
            "items": [
                {
                    "request_id": item.request_id,
                    "path": item.path,
                    "payload": item.payload,
                }
                for item in items
            ]
        }
        try:
            if self.telemetry_spool_replay_handler is not None:
                with self._state_lock:
                    self.spool_direct_replay_total += 1
                body = self.telemetry_spool_replay_handler(batch_request)
            else:
                with self._state_lock:
                    self.spool_http_replay_total += 1
                request = urllib_request.Request(
                    self.local_url + "/v1/internal/processor/telemetry-batch",
                    data=json.dumps(
                        batch_request,
                        separators=(",", ":"),
                    ).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "X-GPU-Fault-Processor-Replay": (self.internal_token),
                        "Idempotency-Key": ",".join(item.request_id for item in items),
                    },
                    method="POST",
                )
                with urlopen(
                    request,
                    timeout=self.request_max_execution_seconds,
                ) as response:
                    body = json.loads(response.read())
        except Exception:
            with self._state_lock:
                self.spool_errors_total += 1
            self._release_telemetry_spool(items)
            raise
        finally:
            elapsed = time.monotonic() - started
            with self._state_lock:
                self.spool_replay_seconds_sum += elapsed
                self.spool_replay_seconds_max = max(
                    self.spool_replay_seconds_max, elapsed
                )
        by_id = {result["request_id"]: result for result in body.get("results", [])}
        done = []
        retry = []
        for item in items:
            result = by_id.get(item.request_id)
            status = 500 if result is None else int(result.get("status", 500))
            # A 4xx is the endpoint's verdict on the payload, not a
            # transient failure, so retrying it would only occupy spool
            # depth until the attempt cap dropped it anyway. Deleting it
            # loses one sample of a stream whose next sample restates the
            # same node state - and it is logged, which a queued request
            # answering 422 into a response body nobody reads is not.
            if status >= 500:
                retry.append(item)
                continue
            if status >= 400:
                LOGGER.warning(
                    "telemetry spool replay rejected request_id=%s "
                    "path=%s status=%s detail=%s",
                    item.request_id,
                    item.path,
                    status,
                    (result or {}).get("body"),
                )
            done.append(item)
        if done:
            completed = self.store.complete_telemetry_spool(done)
            with self._state_lock:
                self.spool_completed_total += completed
                # The difference is rows a newer sample took over while
                # this replay was in flight. The newer payload is still
                # spooled and claimable, which is the whole reason
                # completion is fenced on the revision.
                self.spool_superseded_total += len(done) - completed
        if retry:
            self._release_telemetry_spool(retry)

    def _backlog_is_lane_blocked(
        self,
        *,
        now: datetime,
        include_paths: set[str] | None,
        exclude_paths: set[str] | None,
    ) -> bool:
        """Was the last claim empty because lanes are leased elsewhere?

        Only asked when this stream is about to back off past the busy
        ceiling, so the probe rate is bounded by one query per stream per
        ``busy_backoff_max_seconds`` per replica. A store that predates
        the probe just keeps the old behaviour.
        """

        probe = getattr(
            self.store,
            "active_backlog_is_lane_blocked",
            None,
        )
        if probe is None:
            return False
        self.claim_probes_total += 1
        try:
            blocked = bool(
                probe(
                    now=now,
                    include_paths=include_paths,
                    exclude_paths=exclude_paths,
                )
            )
        except Exception:
            LOGGER.exception("processor lane-block probe failed")
            return False
        if blocked:
            self.claim_lane_blocked_total += 1
        return blocked
