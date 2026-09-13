from __future__ import annotations

import argparse
import errno
import hashlib
import io
import logging
import os
import re
import select
import signal
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    NVIDIA_KERNEL_PATH,
)
from gpu_fault.collectors.logs.fabric_manager import (
    SXID_SUMMARY_PATTERN,
)
from gpu_fault.collectors.models import CollectorContext, CollectorStats
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import CollectorError, EventSink, deliver_event
from gpu_fault.telemetry import CollectorKind

LOGGER = logging.getLogger(__name__)

#: Counters the kernel collector carries in its health summary (ARCH-G4). The
#: summary used to say only "alive"; a stream that was quietly dropping records
#: to its own delivery failures or to kernel ring overflows looked identical to
#: a healthy one. The token is ``<name-with-dashes>:<count>`` and is appended to
#: ``edge_filter_reasons`` only when the count is non-zero.
HEALTH_COUNTER_NAMES = (
    "delivery_failures",
    "kmsg_overflow",
    "boot_time_reestimates",
    "delivery_queue_drops",
    "health_summary_queue_drops",
    "delivery_buffered_at_shutdown",
    "delivery_dropped_at_shutdown",
    "delivery_thread_deaths",
)

#: How many records may wait for the sink before the oldest is dropped. The
#: reader must never wait on delivery -- ``HttpEventSink`` spends up to ~47 s per
#: record on retries before its outbox takes over, and the kernel ring keeps
#: overwriting records the whole time -- so the queue is the only place where
#: back pressure can land. Dropping the oldest keeps the newest, which is what
#: an operator needs during a storm, and every drop is counted and reported in
#: the health summary: delivered, buffered by the sink's outbox, or counted.
DEFAULT_DELIVERY_QUEUE_SIZE = 2048

#: One log line per drop would itself be a load source during a storm, so the
#: first drop is logged and then every ``n``-th, always with the running total.
DELIVERY_DROP_LOG_INTERVAL = 256

#: How long ``stop_delivery`` waits for the in-flight record before it drains
#: what is left in the caller's own thread.
DELIVERY_JOIN_TIMEOUT_SECONDS = 5.0

#: How long the shutdown drain may spend in total when the sink has no durable
#: outbox to hand records to. A full queue of 2048 records at ~47 s of retries
#: each fits inside no stop timeout, so the drain is single-shot and bounded;
#: what does not fit is counted as lost, never dropped in silence.
#:
#: The budget is checked between records, so the drain's upper bound is this
#: budget plus ONE single-attempt post (``HttpEventSink.deliver_once``, bounded
#: by ``GPU_FAULT_COLLECTOR_HTTP_TIMEOUT_SECONDS``, 10 s by default), plus the
#: ``DELIVERY_JOIN_TIMEOUT_SECONDS`` join before it, plus the ≤ 1 s ``select``
#: and ≤ 1 s reopen sleep the main loop may be inside when SIGTERM lands:
#: 2 + 5 + 5 + 10 = 22 s at the defaults. ``deploy/systemd/gpu-fault-kernel-collector.service`` states its
#: ``TimeoutStopSec`` from this sum; change one and the other.
DELIVERY_DRAIN_BUDGET_SECONDS = 5.0

#: How many times a delivery thread that died on its own may be replaced before
#: the collector gives up on it and delivers inline. The loop swallows every
#: ``Exception``, so a death means something worse; restarting for ever would
#: hide it.
DELIVERY_THREAD_MAX_RESTARTS = 3

#: How far a monotonic-derived ``observed_at`` may sit from the collection time
#: before the boot-time estimate is suspected of being stale (ARCH-G9). Records
#: read late are legitimately old, so the re-estimate is rate limited and only
#: changes anything when ``/proc/uptime`` and the wall clock disagree with it.
BOOT_TIME_DRIFT_SECONDS = 5.0
BOOT_TIME_REESTIMATE_MIN_INTERVAL_SECONDS = 60.0

NVIDIA_EVENT_PATTERN = re.compile(r"(?:\bNVRM\b.*\bXid\b|\bSXid\b)", re.IGNORECASE)

SXID_PATTERN = re.compile(r"\bSXid\b", re.IGNORECASE)

KMSG_RECORD_PATTERN = re.compile(
    r"^(?P<priority>\d+),(?P<sequence>\d+),"
    r"(?P<monotonic_us>\d+),(?P<flags>[^;]*);"
    r"(?P<message>.*)$"
)


def _report_lost_at_shutdown(
    counters: dict[str, int], lost: list[str], *, saved: int, reason: str
) -> None:
    """Count and name the records a shutdown drain could not account for."""

    if not lost:
        return
    counters["delivery_dropped_at_shutdown"] += len(lost)
    LOGGER.error(
        "kernel collector stopped with %d record(s) it could neither deliver nor "
        "buffer because %s; buffered %d in this drain, and these are lost: %s",
        len(lost),
        reason,
        saved,
        ", ".join(lost[:10]),
    )


def _deliver_once_at_shutdown(
    sink: EventSink,
    counters: dict[str, int],
    deliver_one: Callable[[str, str, dict[str, Any]], bool],
    record_id: str,
    path: str,
    payload: dict[str, Any],
) -> bool:
    """One bounded attempt for a record the outbox refused; never raises.

    ``deliver_event`` walks the sink's full retry ladder (~47 s at the defaults)
    and then tries the same failed outbox write again, so a drain of a few
    unbuffered records outlived the unit's ``TimeoutStopSec`` and was SIGKILLed
    before the lost-records report could run. ``HttpEventSink.deliver_once`` is
    bounded by one client timeout; a sink without it gets the caller's regular
    single delivery, which is all the fakes and the SQS sink have.
    """

    deliver_once = getattr(sink, "deliver_once", None)
    if not callable(deliver_once):
        return deliver_one(record_id, path, payload)
    try:
        deliver_once(path, payload)
    except Exception as exc:
        counters["delivery_failures"] += 1
        LOGGER.warning(
            "kernel event delivery failed at shutdown and was not buffered: "
            "record=%s error=%s",
            record_id,
            exc,
        )
        return False
    return True


def _count_queue_drop(
    counters: dict[str, int], *, dropped_id: str, dropped_path: str, queue_size: int
) -> None:
    """Count one record a full delivery queue could not hold, and say so.

    A dropped health summary is a lost liveness report, not lost fault
    evidence; counting it as an XID drop would make the counter operators page
    on untrue. One log line per drop would itself be a load source during a
    storm, so the first drop is logged and then every ``n``-th.
    """

    summary = dropped_path == COLLECTOR_HEALTH_PATH
    counter = "health_summary_queue_drops" if summary else "delivery_queue_drops"
    counters[counter] += 1
    drops = counters[counter]
    if drops == 1 or drops % DELIVERY_DROP_LOG_INTERVAL == 0:
        LOGGER.warning(
            "kernel delivery queue is full at %d records; dropped entry=%s "
            "(%s, dropped=%d). The control plane is not keeping up and these "
            "%s are lost.",
            queue_size,
            dropped_id,
            "health summary" if summary else "kmsg record",
            drops,
            "health summaries" if summary else "records",
        )


class KernelLogCollector:
    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        kmsg_path: str = "/dev/kmsg",
        boot_id: str | None = None,
        now: Callable[[], datetime] | None = None,
        deduplication_window: int = 4096,
        start_at_end: bool = True,
        reopen_delay_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        uptime_path: str = "/proc/uptime",
        delivery_queue_size: int = DEFAULT_DELIVERY_QUEUE_SIZE,
    ) -> None:
        if reopen_delay_seconds <= 0:
            raise ValueError("kernel log reopen delay must be positive")
        if delivery_queue_size < 1:
            raise ValueError("kernel delivery queue size must be at least 1")
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.kmsg_path = kmsg_path
        self.uptime_path = uptime_path
        self.boot_id = boot_id or self._read_boot_id()
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.start_at_end = start_at_end
        self.reopen_delay_seconds = reopen_delay_seconds
        self.sleep = sleep
        self._seen_order: deque[str] = deque()
        self._seen: set[str] = set()
        self._deduplication_window = deduplication_window
        self._boot_time: datetime | None = None
        self._boot_time_reestimated_at: datetime | None = None
        self._fallback_sequence = 0
        self.health_counters: dict[str, int] = dict.fromkeys(HEALTH_COUNTER_NAMES, 0)
        self.health_summary_seconds = int(
            os.getenv("GPU_FAULT_KERNEL_HEALTH_SUMMARY_SECONDS", "300")
        )
        # The first summary goes out on the first collection: a freshly
        # installed (or restarted) collector proves itself alive within
        # seconds instead of one period later (live 2026-09-12: a bootstrap's
        # first verify ran before the 300 s summary and read the node as
        # silent). Later summaries keep their stable phase.
        self._next_health_summary = self.now()
        self._last_read_at: datetime | None = None
        self.delivery_queue_size = delivery_queue_size
        # (record_id, path, payload): the health summary goes to its own
        # channel, and it must not block the reader either.
        self._delivery_queue: deque[tuple[str, str, dict[str, Any]]] = deque()
        self._delivery_wakeup = threading.Condition()
        self._delivery_thread: threading.Thread | None = None
        # One stop Event per thread, never a shared flag: a thread that outlived
        # its join must stay stopped for ever, or a later ``start_delivery``
        # leaves two consumers on one queue and only joins the newer.
        self._delivery_stop: threading.Event | None = None
        self._retired_delivery_threads: list[threading.Thread] = []
        # The record the delivery thread is posting right now. It has left the
        # queue, so without this the shutdown drain cannot see it and the thread
        # is killed with the process in the middle of a ~47 s retry ladder.
        self._delivery_inflight: tuple[str, str, dict[str, Any]] | None = None
        self._shutdown = threading.Event()

    def collect_lines(
        self, lines: Iterable[str], *, limit: int | None = None
    ) -> CollectorStats:
        """Read kmsg records and hand the NVIDIA ones on for delivery.

        The counters are plain integers and the model is built once at the end:
        over 99 % of kmsg lines are not NVIDIA events, and each of them used to
        allocate two ``CollectorStats`` copies on the hot read path.

        ``delivered`` counts only what this call saw accepted. With a delivery
        thread running (``start_delivery``) the outcome is not known yet, so
        records are counted as observed and the delivery counters in the health
        summary are what say whether they arrived.
        """

        observed = 0
        skipped = 0
        duplicates = 0
        delivered = 0
        for line in lines:
            if limit is not None and observed >= limit:
                break
            observed += 1
            parsed = self._parse_record(line.rstrip("\n"))
            message = parsed["message"] or ""
            if not NVIDIA_EVENT_PATTERN.search(message):
                skipped += 1
                continue
            if SXID_PATTERN.search(message) and not SXID_SUMMARY_PATTERN.search(
                message
            ):
                skipped += 1
                continue
            record_id = self._record_id(parsed)
            if record_id in self._seen:
                duplicates += 1
                continue
            collected_at = self.now()
            monotonic = parsed.get("monotonic_us")
            observed_at = self._observed_at(parsed, collected_at=collected_at)
            payload: dict[str, Any] = {
                **self.context.model_dump(mode="json"),
                "node_id": self.node_id,
                "record_id": record_id,
                "observed_at": observed_at.isoformat(),
                "source_monotonic_us": (int(monotonic) if monotonic else None),
                "source_boot_id": self.boot_id,
                "collected_at": collected_at.isoformat(),
                "message": message,
                "evidence_ref": (
                    f"kmsg://{self.node_id}/{self.boot_id}/"
                    f"{parsed.get('sequence') or record_id}"
                ),
            }
            # Remembered before delivery, not after: the stream will not show
            # the record again, re-posting a duplicate would help nobody, and a
            # record still waiting in the delivery queue must not be queued a
            # second time.
            self._remember(record_id)
            if self._submit(record_id, NVIDIA_KERNEL_PATH, payload):
                continue
            if self._deliver_one(record_id, NVIDIA_KERNEL_PATH, payload):
                delivered += 1
        return CollectorStats(
            observed=observed,
            skipped=skipped,
            duplicates=duplicates,
            delivered=delivered,
        )

    def start_delivery(self) -> None:
        """Move delivery off the read loop into one daemon thread.

        ``run`` starts it for the live stream. Without it ``collect_lines``
        delivers inline and returns the outcome, which is what a caller that
        feeds a fixed list of lines wants.
        """

        with self._delivery_wakeup:
            running = self._delivery_thread
            if running is not None and running.is_alive():
                return
            self._retired_delivery_threads = [
                item for item in self._retired_delivery_threads if item.is_alive()
            ]
            if self._retired_delivery_threads:
                LOGGER.warning(
                    "%d earlier kernel delivery thread(s) are still finishing a "
                    "post; their stop is set, so they will not take records from "
                    "the new one",
                    len(self._retired_delivery_threads),
                )
            stop = threading.Event()
            thread = threading.Thread(
                target=self._delivery_loop,
                args=(stop,),
                name="kernel-collector-delivery",
                daemon=True,
            )
            self._delivery_stop = stop
            self._delivery_thread = thread
            thread.start()

    def stop_delivery(
        self,
        timeout_seconds: float = DELIVERY_JOIN_TIMEOUT_SECONDS,
        drain_budget_seconds: float = DELIVERY_DRAIN_BUDGET_SECONDS,
    ) -> None:
        """Stop the delivery thread and give every queued record a verdict.

        A daemon thread is killed with the process, so this is what keeps the
        records read just before shutdown from disappearing without one. The
        drain is single-shot and bounded: what the sink can buffer for replay is
        buffered, what is left gets one delivery attempt inside
        ``drain_budget_seconds``, and the remainder is counted and logged as
        lost. Delivered, buffered or counted -- there is no fourth state. That
        includes the record the thread was posting when the join timed out: it
        is drained here too, because a duplicate the control plane dedupes on
        ``event_id`` is better than a record nobody accounted for.
        """

        with self._delivery_wakeup:
            thread = self._delivery_thread
            stop = self._delivery_stop
            if stop is not None:
                stop.set()
            self._delivery_wakeup.notify_all()
        if thread is not None:
            thread.join(timeout=timeout_seconds)
        with self._delivery_wakeup:
            if thread is not None and thread.is_alive():
                # Its own stop Event stays set, so it delivers the record it is
                # holding and then exits without touching the queue again.
                self._retired_delivery_threads.append(thread)
                LOGGER.warning(
                    "kernel delivery thread is still posting after %.1fs; it is "
                    "stopped and %d queued record(s) are drained here",
                    timeout_seconds,
                    len(self._delivery_queue),
                )
            self._delivery_thread = None
            self._delivery_stop = None
            pending = list(self._delivery_queue)
            self._delivery_queue.clear()
            inflight = self._delivery_inflight
            self._delivery_inflight = None
        if inflight is not None:
            # It is first in the queue again. A thread that is still posting it
            # may yet succeed, and then the control plane sees the same
            # ``event_id`` twice and dedupes it; the alternative is a record
            # with no verdict at all.
            pending.insert(0, inflight)
        self._drain_pending(pending, drain_budget_seconds)

    def _delivery_loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            with self._delivery_wakeup:
                while not self._delivery_queue and not stop.is_set():
                    self._delivery_wakeup.wait(1.0)
                if stop.is_set():
                    # Whatever is left belongs to ``stop_delivery``'s drain, in
                    # the caller's thread. Taking one more here would race it.
                    return
                item = self._delivery_queue.popleft()
                self._delivery_inflight = item
            record_id, path, payload = item
            # Posted outside the lock: the reader keeps queueing while this
            # record is in flight, which is the whole point of the thread.
            try:
                self._deliver_one(record_id, path, payload)
            finally:
                with self._delivery_wakeup:
                    # Identity, not equality: a retired thread finishing its
                    # last post must not clear the new thread's record.
                    if self._delivery_inflight is item:
                        self._delivery_inflight = None

    def _drain_pending(
        self,
        pending: list[tuple[str, str, dict[str, Any]]],
        budget_seconds: float,
    ) -> None:
        """Give every record left in the queue a verdict, once, inside a budget.

        Two things end it early -- the budget, and a *second* stop signal --
        and the ``finally`` reports whichever did: an interrupt out of a post or
        an outbox lock used to unwind straight out of the process, leaving the
        rest with no verdict, no counter and no log line.
        """

        if not pending:
            return
        buffer_for_replay = getattr(self.sink, "buffer_for_replay", None)
        deadline = time.monotonic() + max(0.0, budget_seconds)
        lost: list[str] = []
        saved = index = 0
        reason = f"the {budget_seconds:.1f}s shutdown budget ran out"
        try:
            for index, (record_id, path, payload) in enumerate(pending):
                if time.monotonic() >= deadline:
                    # The budget is checked here, between records, so the drain
                    # can overrun it by at most one single-attempt post (see
                    # ``DELIVERY_DRAIN_BUDGET_SECONDS`` for the sum the unit's
                    # ``TimeoutStopSec`` is derived from). The rest is reported
                    # lost rather than retried past SIGKILL.
                    lost.extend(item[0] for item in pending[index:])
                    break
                if buffer_for_replay is not None:
                    try:
                        buffered = bool(buffer_for_replay(path, payload))
                    except Exception:
                        LOGGER.exception(
                            "kernel record could not be buffered for replay at "
                            "shutdown: record=%s",
                            record_id,
                        )
                        buffered = False
                    if buffered:
                        saved += 1
                        self.health_counters["delivery_buffered_at_shutdown"] += 1
                        continue
                    # ``False`` means no outbox is configured, or the write
                    # failed on a read-only or full volume: the record is still
                    # ours, so it gets its one delivery attempt anyway -- one
                    # bounded attempt, never the sink's retry ladder.
                _deliver_once_at_shutdown(
                    self.sink,
                    self.health_counters,
                    self._deliver_one,
                    record_id,
                    path,
                    payload,
                )
        except BaseException:
            # Including the record this raised on: it has no verdict either.
            lost = [item[0] for item in pending[index:]]
            reason = "the shutdown drain was interrupted"
            raise
        finally:
            _report_lost_at_shutdown(
                self.health_counters, lost, saved=saved, reason=reason
            )

    def _submit(self, record_id: str, path: str, payload: dict[str, Any]) -> bool:
        """Queue one record for the delivery thread; False when there is none.

        Never blocks: a full queue drops a record, counts it and says so. Which
        record depends on what arrived: an XID evicts the oldest entry (the
        newest evidence is what an operator needs during a storm), a health
        summary drops *itself* -- a liveness report must never evict fault
        evidence -- and is counted as ``health_summary_queue_drops``; ``True``
        is still answered because the summary has had its verdict. A thread
        that died on its own is counted, reported and replaced -- it used to
        fall back to inline delivery in silence, leaving everything already
        queued waiting on nobody until the process exited.
        """

        for _attempt in (0, 1):
            with self._delivery_wakeup:
                thread = self._delivery_thread
                if thread is None:
                    # Delivery was never started (or has been stopped): the
                    # caller delivers inline and reports the outcome.
                    return False
                if thread.is_alive():
                    if len(self._delivery_queue) >= self.delivery_queue_size:
                        if path == COLLECTOR_HEALTH_PATH:
                            _count_queue_drop(
                                self.health_counters,
                                dropped_id=record_id,
                                dropped_path=path,
                                queue_size=self.delivery_queue_size,
                            )
                            return True
                        dropped_id, dropped_path, _payload = (
                            self._delivery_queue.popleft()
                        )
                        _count_queue_drop(
                            self.health_counters,
                            dropped_id=dropped_id,
                            dropped_path=dropped_path,
                            queue_size=self.delivery_queue_size,
                        )
                    self._delivery_queue.append((record_id, path, payload))
                    self._delivery_wakeup.notify()
                    return True
                self._delivery_thread = None
                self._delivery_stop = None
                inflight = self._delivery_inflight
                self._delivery_inflight = None
                if inflight is not None:
                    # It died holding this record: nobody else has it, so it
                    # goes back to the front for whoever drains the queue next.
                    self._delivery_queue.appendleft(inflight)
                self.health_counters["delivery_thread_deaths"] += 1
                deaths = self.health_counters["delivery_thread_deaths"]
                queued = len(self._delivery_queue)
                recover = deaths <= DELIVERY_THREAD_MAX_RESTARTS
                # A replacement drains what is already queued; without one the
                # queue has no consumer at all, so it is drained here.
                stranded: list[tuple[str, str, dict[str, Any]]] = []
                if not recover:
                    stranded = list(self._delivery_queue)
                    self._delivery_queue.clear()
            LOGGER.error(
                "kernel delivery thread died with %d record(s) queued (deaths=%d); %s",
                queued,
                deaths,
                "restarting it" if recover else "delivering inline from now on",
            )
            if not recover:
                self._drain_pending(stranded, DELIVERY_DRAIN_BUDGET_SECONDS)
                return False
            self.start_delivery()
        return False

    def _deliver_one(self, record_id: str, path: str, payload: dict[str, Any]) -> bool:
        """Post one record and say whether the control plane took it.

        Nothing raises out of here. ``deliver_event`` converts
        ``CollectorError`` only, so any other exception -- a proxy answering 200
        with an HTML body, say -- used to leave ``collect_lines`` and reopen
        ``/dev/kmsg`` at the live tail, losing every record written in between
        (ARCH-G4). It is counted and the stream keeps reading.
        """

        try:
            result = deliver_event(self.sink, path, payload)
        except Exception:
            self.health_counters["delivery_failures"] += 1
            LOGGER.exception(
                "kernel event delivery raised an unexpected error; "
                "continuing live kmsg collection: record=%s",
                record_id,
            )
            return False
        if result.buffered:
            LOGGER.warning(
                "kernel event persisted to the collector outbox; "
                "continuing live kmsg collection: record=%s error=%s",
                record_id,
                result.error,
            )
            return False
        if result.failed:
            # A rejected or unbuffered event is lost, and used to take the
            # stream with it: the exception reopened /dev/kmsg at the live tail
            # and dropped everything written in between (ARCH-G4). It is
            # counted and the stream keeps reading.
            self.health_counters["delivery_failures"] += 1
            LOGGER.warning(
                "kernel event delivery failed and was not buffered; "
                "continuing live kmsg collection: record=%s error=%s",
                record_id,
                result.error,
            )
            return False
        return True

    def run(self) -> None:
        self._shutdown.clear()
        self.start_delivery()
        restore_signals = self._install_stop_signals()
        try:
            self._run_forever()
        finally:
            try:
                # The reader is leaving: whatever is still queued is delivered,
                # buffered for replay or counted before the process goes away.
                # The handlers stay installed for this: the drain may take
                # seconds, and systemd repeats SIGTERM on a slow stop -- with
                # the default disposition back in place that second signal
                # killed the process mid-drain.
                self.stop_delivery()
            finally:
                restore_signals()

    def _install_stop_signals(self) -> Callable[[], None]:
        """Turn SIGTERM/SIGINT into a stop request; return how to undo it.

        systemd stops the unit with SIGTERM (no ``KillSignal=``), and CPython's
        default action for it terminates the process without unwinding, so the
        ``finally`` that drains the delivery queue never ran: every deploy, and
        every OOM-kill-adjacent restart, dropped up to ``delivery_queue_size``
        queued XID/SXID payloads with no post, no outbox record and no counter.
        The handler only sets a flag -- the drain happens on the way out of
        ``run`` -- and whatever handler was installed before still runs, so a
        Ctrl-C keeps raising ``KeyboardInterrupt``.
        """

        if threading.current_thread() is not threading.main_thread():
            # Only the main thread may install handlers, and a collector driven
            # from a worker thread is not the process's owner anyway.
            return lambda: None

        def make_handler(previous: Any) -> Callable[[int, Any], None]:
            def handler(signum: int, frame: Any) -> None:
                self._shutdown.set()
                if callable(previous):
                    previous(signum, frame)

            return handler

        installed: list[tuple[int, Any]] = []
        for number in (signal.SIGTERM, signal.SIGINT):
            try:
                previous = signal.getsignal(number)
                if previous is signal.SIG_IGN:
                    # Whoever started this process asked for the signal to be
                    # swallowed (a supervisor that stops its children itself).
                    # Arming it would turn that into a shutdown.
                    continue
                signal.signal(number, make_handler(previous))
            except (OSError, ValueError):
                continue
            installed.append((number, previous))

        def restore() -> None:
            for number, previous in installed:
                try:
                    signal.signal(
                        number,
                        previous if previous is not None else signal.SIG_DFL,
                    )
                except (OSError, ValueError):
                    continue

        return restore

    def _run_forever(self) -> None:
        while not self._shutdown.is_set():
            try:
                with open(
                    self.kmsg_path,
                    "r",
                    encoding="utf-8",
                    errors="replace",
                    buffering=1,
                ) as stream:
                    if self.start_at_end:
                        try:
                            self._seek_to_live_tail(stream)
                        except OSError as exc:
                            raise CollectorError(
                                "cannot seek kernel message device to "
                                "the live tail; refusing to replay the "
                                "ring buffer"
                            ) from exc
                    self.refresh_boot_time()
                    self._collect_live_stream(stream)
                    if not self._shutdown.is_set():
                        LOGGER.warning(
                            "kernel message stream ended; reopening %s",
                            self.kmsg_path,
                        )
            except PermissionError as exc:
                raise CollectorError(
                    f"cannot read {self.kmsg_path}; CAP_SYSLOG or an "
                    "equivalent privileged device mount is required"
                ) from exc
            except FileNotFoundError as exc:
                raise CollectorError(
                    f"kernel message device not found: {self.kmsg_path}"
                ) from exc
            except Exception:
                LOGGER.exception(
                    "kernel log collection failed; reopening %s",
                    self.kmsg_path,
                )
            if self._shutdown.is_set():
                return
            self.sleep(self.reopen_delay_seconds)

    def _collect_live_stream(self, stream: Any) -> None:
        try:
            descriptor = stream.fileno()
        except (AttributeError, io.UnsupportedOperation):
            self.collect_lines(stream)
            self._last_read_at = self.now()
            self._maybe_health_summary_without_interrupting_stream(self._last_read_at)
            return
        while not self._shutdown.is_set():
            now = self.now()
            timeout = min(
                1.0,
                max(
                    0.0,
                    (self._next_health_summary - now).total_seconds(),
                ),
            )
            ready, _, _ = select.select([descriptor], [], [], timeout)
            self._last_read_at = self.now()
            if ready:
                try:
                    line = stream.readline()
                except OSError as exc:
                    if exc.errno != errno.EPIPE:
                        raise
                    # The kernel ring buffer overwrote records this reader
                    # had not consumed yet. The next read returns the oldest
                    # surviving record; reopening would seek to the live
                    # tail and lose those too (ARCH-G4).
                    self.health_counters["kmsg_overflow"] += 1
                    LOGGER.warning(
                        "kernel message ring overflow; records were lost "
                        "before this reader consumed them (overflows=%d)",
                        self.health_counters["kmsg_overflow"],
                    )
                    continue
                if not line:
                    return
                self.collect_lines([line])
            self._maybe_health_summary_without_interrupting_stream(self._last_read_at)

    def _maybe_health_summary_without_interrupting_stream(
        self,
        observed_at: datetime,
    ) -> None:
        try:
            self._maybe_health_summary(observed_at)
        except CollectorError as exc:
            LOGGER.warning(
                "kernel health summary delivery failed; continuing live stream: %s",
                exc,
            )

    def _maybe_health_summary(self, observed_at: datetime) -> None:
        if observed_at < self._next_health_summary:
            return
        self._next_health_summary = next_stable_phase(
            observed_at,
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            channel=CollectorKind.NVIDIA_KERNEL.value,
            interval_seconds=self.health_summary_seconds,
        )
        self._send_health_summary(observed_at)

    def health_summary_reasons(self) -> list[str]:
        """``["health-summary"]`` plus one ``name:count`` token per non-zero counter.

        A healthy stream keeps exactly the routine reason so the summary stays
        on its dedicated ``collector-health-nvidia_kernel`` lane and coalesces
        as before. A stream with losses to report falls to the plain node lane
        (``processor/models.py`` ``_compute_ordering_key``), which only orders
        it behind that node's other work; coalescing still requires the same
        path, so it can only be superseded by a later summary of its own.
        """

        reasons = ["health-summary"]
        for name in HEALTH_COUNTER_NAMES:
            count = self.health_counters.get(name, 0)
            if count:
                reasons.append(f"{name.replace('_', '-')}:{count}")
        return reasons

    def _send_health_summary(self, observed_at: datetime) -> None:
        """Queue the summary behind the delivery thread, or post it inline.

        This used to post from the read loop, so an unreachable control plane
        stalled the kmsg reader for ~47 s once every 300 s while the kernel ring
        kept overwriting records.
        """

        summary_id = f"kernel-health-{self.node_id}-{int(observed_at.timestamp())}"
        payload: dict[str, Any] = {
            "summary_id": summary_id,
            "cluster_id": self.context.cluster_id,
            "node_id": self.node_id,
            "collector": CollectorKind.NVIDIA_KERNEL.value,
            "observed_at": observed_at.isoformat(),
            "edge_filter_reasons": self.health_summary_reasons(),
        }
        if self._submit(summary_id, COLLECTOR_HEALTH_PATH, payload):
            return
        result = deliver_event(self.sink, COLLECTOR_HEALTH_PATH, payload)
        # A summary the outbox took is on its way; only one that went nowhere
        # is reported by the caller's guard (ARCH-G3).
        result.raise_for_failure()

    @staticmethod
    def _seek_to_live_tail(stream: Any) -> None:
        try:
            descriptor = stream.fileno()
        except (AttributeError, io.UnsupportedOperation):
            # StringIO and other test streams have no file descriptor.
            stream.seek(0, os.SEEK_END)
            return
        os.lseek(descriptor, 0, os.SEEK_END)

    @staticmethod
    def _parse_record(line: str) -> dict[str, str | None]:
        match = KMSG_RECORD_PATTERN.match(line)
        if not match:
            return {
                "priority": None,
                "sequence": None,
                "monotonic_us": None,
                "flags": None,
                "message": line,
            }
        return match.groupdict()

    def _record_id(self, parsed: dict[str, str | None]) -> str:
        sequence = parsed.get("sequence")
        if sequence:
            suffix = sequence
        elif parsed.get("monotonic_us"):
            digest = hashlib.sha256((parsed.get("message") or "").encode()).hexdigest()[
                :12
            ]
            suffix = f"{parsed['monotonic_us']}-{digest}"
        else:
            self._fallback_sequence += 1
            digest = hashlib.sha256((parsed.get("message") or "").encode()).hexdigest()[
                :12
            ]
            suffix = f"fallback-{self._fallback_sequence}-{digest}"
        return f"kmsg-{self.boot_id}-{suffix}"

    def _observed_at(
        self,
        parsed: dict[str, str | None],
        *,
        collected_at: datetime,
    ) -> datetime:
        monotonic = parsed.get("monotonic_us")
        if monotonic is None or self._boot_time is None:
            return collected_at
        try:
            offset = timedelta(microseconds=int(monotonic))
            observed_at = self._boot_time + offset
        except (TypeError, ValueError, OverflowError):
            return collected_at
        drift = abs((observed_at - collected_at).total_seconds())
        if drift > BOOT_TIME_DRIFT_SECONDS and self._maybe_reestimate_boot_time(
            collected_at
        ):
            # The wall clock stepped (NTP) after the boot time was estimated,
            # so every offset since was skewed by the step (ARCH-G9).
            if self._boot_time is None:
                return collected_at
            observed_at = self._boot_time + offset
        if observed_at > collected_at + timedelta(seconds=BOOT_TIME_DRIFT_SECONDS):
            LOGGER.warning(
                "kernel event monotonic timestamp is in the future; "
                "using collection time record=%s",
                parsed.get("sequence"),
            )
            return collected_at
        return observed_at

    def refresh_boot_time(self) -> None:
        """Estimate the boot time from ``uptime_path`` and the wall clock now."""

        self._boot_time = self._estimate_boot_time()
        self._boot_time_reestimated_at = self.now()

    def _maybe_reestimate_boot_time(self, collected_at: datetime) -> bool:
        """Re-estimate once per rate-limit window; True when the estimate moved.

        Only an estimate this collector made (``refresh_boot_time``) is ever
        revised, and at most once a minute: a record that is merely old -- the
        reader fell behind -- re-reads ``uptime_path`` and finds the same boot
        time, so the check costs one small read and changes nothing.
        """

        previous_at = self._boot_time_reestimated_at
        if (
            previous_at is None
            or (collected_at - previous_at).total_seconds()
            < BOOT_TIME_REESTIMATE_MIN_INTERVAL_SECONDS
        ):
            return False
        self._boot_time_reestimated_at = collected_at
        previous = self._boot_time
        estimate = self._estimate_boot_time()
        if estimate is None or previous is None:
            return False
        if abs((estimate - previous).total_seconds()) <= 1.0:
            return False
        self._boot_time = estimate
        self.health_counters["boot_time_reestimates"] += 1
        LOGGER.warning(
            "kernel boot time estimate moved by %.1fs; the wall clock stepped "
            "since the stream was opened (reestimates=%d)",
            (estimate - previous).total_seconds(),
            self.health_counters["boot_time_reestimates"],
        )
        return True

    def _estimate_boot_time(self) -> datetime | None:
        try:
            uptime = float(
                Path(self.uptime_path).read_text(encoding="ascii").split(maxsplit=1)[0]
            )
        except (OSError, ValueError, IndexError):
            LOGGER.warning(
                "cannot read host uptime; kernel event timestamps "
                "will use collection time"
            )
            return None
        return self.now() - timedelta(seconds=uptime)

    def _remember(self, record_id: str) -> None:
        self._seen.add(record_id)
        self._seen_order.append(record_id)
        while len(self._seen_order) > self._deduplication_window:
            self._seen.discard(self._seen_order.popleft())

    @staticmethod
    def _read_boot_id() -> str:
        """The kernel's boot id, or a marker unique to this collector.

        The fallback used to be the constant ``unknown-boot``. kmsg sequence
        numbers restart at boot and the control plane's ``event_id`` is scoped
        to the cluster, so ``kmsg-unknown-boot-42`` recurred after every reboot
        and the first XID of the new boot was dropped as a duplicate of the last
        one before it. One random marker per collector -- so one per process,
        since the unit runs a single collector -- keeps the ids apart across
        reboots and across nodes; it is read once, in ``__init__``, so the ids
        of one process stay stable.
        """

        try:
            return (
                Path("/proc/sys/kernel/random/boot_id")
                .read_text(encoding="ascii")
                .strip()
            ) or f"unknown-{uuid4().hex[:12]}"
        except OSError:
            return f"unknown-{uuid4().hex[:12]}"


def build_from_environment(
    sink: EventSink, context: CollectorContext, arguments: argparse.Namespace
) -> KernelLogCollector:
    """The ``gpu-fault-collector kernel`` factory named by the registry."""

    if not arguments.node_id:
        raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
    return KernelLogCollector(
        sink,
        context,
        node_id=arguments.node_id,
        kmsg_path=arguments.kmsg_path,
    )
