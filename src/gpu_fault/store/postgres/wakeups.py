from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

from gpu_fault.store.contracts import WakeupChannel
from gpu_fault.store.postgres.pool import _reject_reader

LOGGER = logging.getLogger(__name__)


class PostgresWakeupMixin:
    """``run_wakeup_listener`` over ``LISTEN`` on a dedicated connection."""

    # Supplied by the composed concrete implementation. Declared as a read-only
    # property, not an attribute, because ``PostgresStore.url`` is a property
    # that re-reads the mounted Secret file, so a LISTEN reconnect after a
    # password rotation uses the new DSN (CP-3).
    @property
    def url(self) -> str:
        raise NotImplementedError("PostgresStore supplies the DSN")

    # How often an idle LISTEN connection re-checks that it is still on the
    # writer (G-11). A class attribute rather than a keyword so every backend
    # keeps the one ``run_wakeup_listener`` signature; tests lower it.
    _WAKEUP_WRITER_CHECK_SECONDS = 5.0

    def run_wakeup_listener(
        self,
        channel: WakeupChannel,
        stop_event: threading.Event,
        on_notification: Callable[[dict[str, Any]], None],
        *,
        timeout_seconds: float = 1.0,
        on_state: Callable[[bool], None] | None = None,
    ) -> None:
        """Forward every NOTIFY on ``channel`` until ``stop_event`` is set.

        Same shape as ``listen_processor_queue_notifications`` and
        ``listen_telemetry_spool_notifications``: the LISTEN connection is
        opened outside the pool, autocommit, with a connect timeout, so the
        pool's writer probes never see it (G-11); after an Aurora failover it
        would stay on the demoted instance -- connected, deaf, reported
        healthy -- so once per ``_WAKEUP_WRITER_CHECK_SECONDS`` the loop asks
        the server whether it is a reader and treats "yes" like a dropped
        connection: close, report disconnected, reconnect through the writer
        endpoint.

        No shard lock, unlike the processor listener: every dispatcher replica
        and every executor wants every wakeup, and the claim that follows is
        what serialises them. Delivery is best-effort -- NOTIFY sent while the
        connection is down is lost, and PostgreSQL coalesces identical
        payloads in one transaction -- so the consumer keeps its poll and
        treats a payload as "scan now", never as the row.
        """

        import psycopg

        if timeout_seconds <= 0:
            raise ValueError("wakeup listener timeout must be positive")
        channel_name = WakeupChannel(channel).value
        while not stop_event.is_set():
            try:
                with psycopg.connect(
                    self.url,
                    autocommit=True,
                    connect_timeout=5,
                ) as connection:
                    connection.execute(f"LISTEN {channel_name}")
                    if on_state is not None:
                        on_state(True)
                    last_writer_check = time.monotonic()
                    while not stop_event.is_set():
                        monotonic = time.monotonic()
                        if (
                            monotonic - last_writer_check
                            >= self._WAKEUP_WRITER_CHECK_SECONDS
                        ):
                            last_writer_check = monotonic
                            _reject_reader(
                                connection,
                                f"{channel_name} LISTEN connection is on a "
                                "read-only replica; NOTIFY is not forwarded there",
                            )
                        payloads: set[str] = set()
                        for notification in connection.notifies(
                            timeout=timeout_seconds,
                            stop_after=256,
                        ):
                            if notification.payload not in payloads:
                                payloads.add(notification.payload)
                                decoded = _decode_wakeup(
                                    channel_name, notification.payload
                                )
                                if decoded is not None:
                                    on_notification(decoded)
                            if stop_event.is_set():
                                break
            except Exception:
                if on_state is not None:
                    on_state(False)
                if stop_event.wait(1.0):
                    return
        if on_state is not None:
            on_state(False)


def _decode_wakeup(channel_name: str, payload: str) -> dict[str, Any] | None:
    """The trigger's JSON object, or None for anything else on the channel.

    Only the trigger publishes here, but a hand-typed ``NOTIFY`` during an
    incident must not take the listener down into its reconnect loop.
    """

    try:
        decoded = json.loads(payload)
    except ValueError:
        decoded = None
    if not isinstance(decoded, dict):
        LOGGER.warning(
            "ignoring a non-object payload on %s: %.120r", channel_name, payload
        )
        return None
    return decoded
