"""Process-local liveness: the report behind ``/livez`` and a listener of its own.

``/livez`` is deliberately trivial - the event loop answers and the threads
this role cannot run without are alive - yet the 50-cluster cold-connect
burst had the kubelet kill all three api-ha Pods on it while every request
was being answered 202. The local reproduction showed why: the probe's
connection joins the main listening socket's accept queue, a FIFO it shares
with the storm. 4,250 cold connections against four uvicorn processes kept
3,600-3,900 connections waiting in that queue for the whole burst, so a probe
waited queue-depth over accept-rate (3.6 s at 1,090 req/s; 0.1 s once inside
the app). At the live scale - 4,266 connections per Pod at 390 req/s - that
is 11 s, two 5 s probes in a row, a restart. Event-loop lag, GIL contention
and GC pauses stayed under 0.6 s throughout; the queue was the mechanism.

So liveness gets a listener the storm cannot queue behind. Every worker
process binds :data:`LIVENESS_PORT_ENV` with ``SO_REUSEPORT`` - the kernel
spreads the kubelet's connections over the processes, the way the shared main
socket already did - and answers ``GET /livez`` from its own event loop, with
the same report the route gives. A stalled loop still fails the probe, because
the accept and the answer both run on it; a loop that is merely busy answers
within one iteration. The main-port ``/livez`` stays for single-process runs
and operators.

Readiness moved here as well. It stayed on the service port's ``/healthz``
(5 s x 3) on the theory that a Pod that cannot accept a request promptly
should be NotReady - but under a fleet-wide burst every replica is that Pod
at the same moment. In two 50-cluster rounds the probe timed out three times
in a row on all three api-ha Pods within 2.3 s while every load request was
answered 2xx; the AWS Load Balancer Controller deregistered all three NLB
targets (and revoked the NLB's ingress rule on the Pods' security group), and
for the ~40 s the re-registered targets needed to pass the NLB health check
again the load balancer had no healthy target, so executor claims opened in
that window were closed without a response. Readiness asks whether *this*
process should receive traffic - processor, spool consumer and registry
freshness, the ``/healthz`` verdict - and a backlog all replicas share is not
such a reason: taking every replica out turns a slow burst into an outage.
So the listener answers ``GET`` :data:`READINESS_PATH` too, with the route's
own document (:meth:`LivenessListener.serve`), and the renderer points the
readiness probe here. The service port keeps the startup probe and, for the
ingress, the NLB's health check, which gates a roll through the Pod readiness
gate and fails open rather than deregistering when every target misses it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

LIVENESS_PORT_ENV = "GPU_FAULT_LIVENESS_PORT"
LIVENESS_PATH = "/livez"
#: The readiness document (``/healthz``) the listener serves once the app wires
#: it (:meth:`LivenessListener.serve`); the kubelet's readiness probe reads it.
READINESS_PATH = "/healthz"
# A probe request is one line and a few headers; anything larger is not one.
MAX_REQUEST_BYTES = 8192
REQUEST_TIMEOUT_SECONDS = 2.0

LivenessReport = Callable[[], tuple[int, dict[str, Any]]]


def processor_consumer_dead(processor: Any) -> bool:
    """Whether the ``gpu-fault-processor-inbox`` loop has died or wedged (B-6).

    The signal is the coordinator's ``processor_consumer_running`` flag and
    the age of its last cycle. Before the first cycle the age is ``None``:
    the thread has not started yet, and a probe that restarted the Pod for
    that would never let it start, so that reads as alive. A coordinator
    that predates the signal has neither attribute and is left alone.
    """

    running = getattr(processor, "processor_consumer_running", None)
    age = getattr(processor, "consumer_last_cycle_age_seconds", None)
    if running is None or age is None:
        return False
    if not running:
        # It cycled once and is no longer running: the thread exited.
        return True
    # The loop's longest voluntary wait is the notification fallback (or the
    # idle backoff); several multiples of it separate "idle" from "wedged".
    fallback = float(getattr(processor, "processor_notification_fallback_seconds", 5.0))
    return not processor.consumer_is_live(max_cycle_age_seconds=max(30.0, 6 * fallback))


def liveness_report(
    *,
    service_role: str,
    processor: Any | None,
    environment: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    """The ``/livez`` verdict: HTTP status and payload.

    Deliberately blind to Aurora: it asserts that the event loop answers and
    that the background threads this process cannot run without are alive.
    Registry freshness, store round trips and leadership belong to readiness
    (``/healthz``); a writer failover must make Pods NotReady, never restart
    them, because the restarted process would need Aurora again to start.
    """

    dead_threads: list[str] = []
    if (
        service_role == "spool-worker"
        and processor is not None
        and not processor.spool_consumer_running
    ):
        dead_threads.append("telemetry-spool-consumer")
    if (
        service_role in {"worker", "all"}
        and processor is not None
        and getattr(processor, "active_consumers", False)
        and processor_consumer_dead(processor)
    ):
        dead_threads.append("processor-consumer")
    payload = {
        "status": "alive" if not dead_threads else "dead",
        "service_role": environment.get("GPU_FAULT_SERVICE_ROLE") or "combined",
        "dead_threads": dead_threads,
    }
    return (503 if dead_threads else 200), payload


def liveness_port_from_environment(environ: Mapping[str, str] | None = None) -> int:
    """``GPU_FAULT_LIVENESS_PORT``; 0 (the default) keeps the listener off."""

    env = os.environ if environ is None else environ
    port = int(env.get("GPU_FAULT_LIVENESS_PORT", "0"))
    if not 0 <= port <= 65535:
        raise ValueError(f"{LIVENESS_PORT_ENV} must be a TCP port or 0")
    return port


_REASONS = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    503: "Service Unavailable",
}


class LivenessListener:
    """A per-process HTTP/1.1 responder for the kubelet's probes on its own port.

    ``GET /livez`` from construction, ``GET /healthz`` once the application
    hands it the readiness verdict (:meth:`serve`). Bound with ``SO_REUSEPORT``
    so every uvicorn worker in the Pod serves the same port; the kubelet
    connects to the Pod IP and the kernel picks a process. Runs entirely on
    the process's event loop: accepting the probe and answering it are the
    liveness assertion.
    """

    def __init__(
        self,
        *,
        port: int,
        report: LivenessReport,
        host: str = "0.0.0.0",
        path: str = LIVENESS_PATH,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("liveness listener port must be a TCP port")
        self.port = port
        self.host = host
        self.path = path
        self.report = report
        self.routes: dict[str, LivenessReport] = {path: report}
        self.bound_port: int | None = None
        self.requests_total = 0
        self.failures_total = 0
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> int:
        self._server = await asyncio.start_server(
            self._serve,
            host=self.host,
            port=self.port,
            reuse_port=True,
            backlog=128,
            limit=MAX_REQUEST_BYTES,
        )
        self.bound_port = self._server.sockets[0].getsockname()[1]
        LOGGER.info(
            "liveness listener pid=%s port=%s paths=%s",
            os.getpid(),
            self.bound_port,
            ",".join(sorted(self.routes)),
        )
        return self.bound_port

    def serve(self, path: str, report: LivenessReport) -> None:
        """Answer ``GET path`` with ``report`` as well (the readiness document)."""

        self.routes[path] = report

    async def close(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        server.close()
        await server.wait_closed()

    def _answer(self, head: bytes) -> tuple[int, dict[str, Any]]:
        request_line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = request_line.split()
        if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
            return 400, {"detail": "malformed liveness request"}
        method, target = parts[0], parts[1].split("?", 1)[0]
        report = self.routes.get(target)
        if report is None:
            return 404, {"detail": "Not Found"}
        if method not in {"GET", "HEAD"}:
            return 405, {"detail": "Method Not Allowed"}
        return report()

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.requests_total += 1
        head_only = False
        try:
            try:
                head = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), REQUEST_TIMEOUT_SECONDS
                )
            except (
                asyncio.IncompleteReadError,
                asyncio.LimitOverrunError,
                TimeoutError,
            ):
                status, payload = 400, {"detail": "malformed liveness request"}
            else:
                head_only = head.startswith(b"HEAD ")
                status, payload = self._answer(head)
            body = json.dumps(payload, separators=(",", ":")).encode()
            response = (
                f"HTTP/1.1 {status} {_REASONS.get(status, 'Error')}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode()
            if not head_only:
                response += body
            writer.write(response)
            await asyncio.wait_for(writer.drain(), REQUEST_TIMEOUT_SECONDS)
            if status >= 500:
                self.failures_total += 1
        except (OSError, TimeoutError):
            # The prober gave up or went away; nothing to answer any more.
            self.failures_total += 1
        finally:
            writer.close()
