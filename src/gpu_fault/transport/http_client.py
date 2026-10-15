"""Keep-alive HTTP client for every node-side poster, with a connection policy.

``urllib.request.urlopen`` opens a new TCP (and for HTTPS a new TLS)
connection per call. For a fleet whose every collector posts on a fixed
cadence that is one handshake per post, and after a control-plane roll or a
fleet-wide restart those handshakes arrive together: the storm the ingress
measures as its accept-queue depth. The pool here keeps one connection per
(scheme, host, port) per thread and reuses it, so a steady-state post is a
write and a read on a socket that already exists.

What a long-lived connection has to respect (:class:`ConnectionPolicy`):

* the NLB in front of the ingress drops a flow idle for 350 s and cannot be
  told otherwise, so a connection idle longer than ``max_idle_seconds``
  (below that, and below the server's ``--timeout-keep-alive``) is closed
  here before it is reused, instead of failing a request on a flow that is
  already gone;
* TCP keepalive probes (``tcp_keepalive_seconds``) keep the NLB's idle clock
  from expiring between two posts and detect a peer that vanished;
* a connection older than ``max_age_seconds`` (jittered so a fleet does not
  reconnect in step) is replaced, which is what lets clients re-spread over
  Pods and workers after a roll - with ``SO_REUSEPORT`` service sockets a
  connection is pinned to the worker that accepted it;
* a reused connection the peer already closed (its FIN or RST is queued: a
  keep-alive expiry, a Pod leaving a roll) is replaced *before* anything is
  written to it, and a request whose write failed is sent again on a fresh
  connection; neither can have been acted on (the first wrote nothing, the
  second at most a head without its whole body), so both hold for every
  request, a non-idempotent claim included;
* a reused connection that fails after the request was written fails
  closed: it is retried once on a fresh connection only when the request is
  safe to repeat (``GET``, or a ``POST`` carrying one of
  :data:`RETRY_IDENTITY_HEADERS`: the caller vouches that the server
  deduplicates a repeat).

Every outcome is counted (:meth:`_KeepAliveConnectionPool.stats`) and
summarised to the log every :data:`STATS_LOG_INTERVAL_SECONDS`; the node
Agent publishes the same counters on its ``/healthz``.
"""

from __future__ import annotations

import io
import logging
import os
import random
import select
import socket
import ssl
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.client import (
    HTTPConnection,
    HTTPException,
    HTTPResponse,
    HTTPSConnection,
)
from threading import Lock, local
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    Request,
    getproxies,
    proxy_bypass,
)
from urllib.request import (
    urlopen as _urllib_urlopen,
)
from urllib.response import addinfourl

LOGGER = logging.getLogger(__name__)

#: The Network Load Balancer's idle timeout for a TCP flow: fixed by AWS,
#: not configurable. A client connection idle longer than this is dead.
NLB_IDLE_TIMEOUT_SECONDS = 350.0
CONNECTION_IDLE_ENV = "GPU_FAULT_HTTP_CONNECTION_IDLE_SECONDS"
CONNECTION_MAX_AGE_ENV = "GPU_FAULT_HTTP_CONNECTION_MAX_AGE_SECONDS"
TCP_KEEPALIVE_ENV = "GPU_FAULT_HTTP_TCP_KEEPALIVE_SECONDS"
#: Spread of the per-connection lifetime, so a fleet that connected together
#: after a roll does not reconnect together ``max_age_seconds`` later.
AGE_JITTER_RATIO = 0.1
#: TCP keepalive probe spacing and count once the idle time has elapsed.
TCP_KEEPALIVE_INTERVAL_SECONDS = 15
TCP_KEEPALIVE_PROBES = 4
STATS_LOG_INTERVAL_SECONDS = 300.0

#: The headers that license the one repeat of a ``POST`` whose reused
#: connection failed after the request was written. Each names a server-side
#: dedupe the caller relies on: ``Idempotency-Key`` derives the processor queue
#: row (``gpu_fault.app.middleware.dispatch``); the request id marks a post the
#: server answers alike when it arrives twice (the Agent heartbeat's CAS, the
#: executor's reads and keyed writes); the command id names a lease-fenced
#: command post (result, renewal, progress) the control plane answers
#: idempotently and checks against the command in the path.
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
REQUEST_ID_HEADER = "X-GPU-Fault-Request-ID"
COMMAND_ID_HEADER = "X-GPU-Fault-Command-ID"
RETRY_IDENTITY_HEADERS = (IDEMPOTENCY_KEY_HEADER, REQUEST_ID_HEADER, COMMAND_ID_HEADER)

ConnectionKey = tuple[str, str, int] | tuple[str, str, int, int]


@dataclass(frozen=True)
class ConnectionPolicy:
    """How long a pooled connection may idle, how long it may live, and the
    TCP keepalive that holds the NLB flow open in between."""

    max_idle_seconds: float = 90.0
    max_age_seconds: float = 1800.0
    tcp_keepalive_seconds: float = 60.0

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> ConnectionPolicy:
        env = os.environ if environ is None else environ
        idle = float(env.get("GPU_FAULT_HTTP_CONNECTION_IDLE_SECONDS", "90"))
        if idle <= 0:
            raise ValueError(f"{CONNECTION_IDLE_ENV} must be positive")
        if idle >= 350:
            raise ValueError(
                f"{CONNECTION_IDLE_ENV} must stay below the NLB idle timeout"
            )
        max_age = float(env.get("GPU_FAULT_HTTP_CONNECTION_MAX_AGE_SECONDS", "1800"))
        if max_age < 0:
            raise ValueError(
                f"{CONNECTION_MAX_AGE_ENV} cannot be negative (0 disables)"
            )
        keepalive = float(env.get("GPU_FAULT_HTTP_TCP_KEEPALIVE_SECONDS", "60"))
        if keepalive < 0:
            raise ValueError(f"{TCP_KEEPALIVE_ENV} cannot be negative (0 disables)")
        if keepalive >= 350:
            raise ValueError(
                f"{TCP_KEEPALIVE_ENV} must stay below the NLB idle timeout"
            )
        return cls(
            max_idle_seconds=idle,
            max_age_seconds=max_age,
            tcp_keepalive_seconds=keepalive,
        )


@dataclass
class _PooledConnection:
    connection: HTTPConnection
    opened_at: float
    last_used_at: float
    #: ``None`` when lifetimes are not bounded.
    expires_at: float | None
    keepalive_applied: bool = False


#: Counter names, in the order the log line prints them.
STAT_NAMES = (
    "connections_opened_total",
    "requests_reused_total",
    "stale_connection_retries_total",
    "stale_connection_failures_total",
    "idle_expired_total",
    "age_expired_total",
    "evicted_total",
    "peer_closed_total",
)


class _RequestNotWritten(Exception):
    """Writing the request failed, so the peer's application never had all of
    it: nothing can have been acted on. Internal to the pool -- the caller
    sees ``error``, the exception the write raised."""

    def __init__(self, error: HTTPException | OSError) -> None:
        super().__init__(str(error))
        self.error = error


def peer_closed(connection: object) -> bool:
    """Whether an idle pooled connection can no longer carry a request.

    Between two requests the peer owes nothing on a keep-alive connection, so
    a socket that polls readable holds the peer's FIN or RST (it closed the
    connection) or bytes nobody asked for; neither may carry the next request.
    Asked before anything is written, which is what makes replacing the
    connection safe for every request. A socket without a descriptor (a test
    double) is taken as open.
    """

    fileno = getattr(getattr(connection, "sock", None), "fileno", None)
    if fileno is None:
        return False
    try:
        descriptor = fileno()
    except (OSError, ValueError):
        return True
    if not isinstance(descriptor, int):
        return False
    if descriptor < 0:
        return True
    try:
        if hasattr(select, "poll"):
            poller = select.poll()
            poller.register(descriptor, select.POLLIN | select.POLLPRI)
            return bool(poller.poll(0))
        readable, _, _ = select.select([descriptor], [], [], 0)
    except (OSError, ValueError):
        return True
    return bool(readable)


def enable_tcp_keepalive(sock: object, idle_seconds: float) -> bool:
    """Arm TCP keepalive on ``sock``; ``False`` when the socket cannot take it."""

    if idle_seconds <= 0:
        return False
    setsockopt = getattr(sock, "setsockopt", None)
    if setsockopt is None:
        return False
    try:
        setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, max(1, int(idle_seconds)))
        setsockopt(
            socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, TCP_KEEPALIVE_INTERVAL_SECONDS
        )
        setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, TCP_KEEPALIVE_PROBES)
    except (OSError, AttributeError, TypeError):
        return False
    return True


class _KeepAliveConnectionPool:
    """Per-thread HTTP/1.1 connection cache under one :class:`ConnectionPolicy`.

    Connections are cached per thread because ``http.client`` connections
    are not safe to share; the counters and the policy are process-wide.
    An idle keep-alive connection can be closed by the peer between requests:
    one found closed before the write is replaced (``peer_closed_total``), a
    request whose write failed is sent again, and one that failed after the
    write is repeated only when it is safe to repeat.
    """

    max_connections = 8

    def __init__(self, policy: ConnectionPolicy | None = None) -> None:
        self._local = local()
        self._ssl_context: ssl.SSLContext | None = None
        self._proxies = getproxies()
        self.policy = (
            policy if policy is not None else ConnectionPolicy.from_environment()
        )
        self._stats_lock = Lock()
        self._counters: dict[str, int] = dict.fromkeys(STAT_NAMES, 0)
        self._last_log = time.monotonic()
        #: Draw for the per-connection lifetime spread; a test seam.
        self.jitter: Callable[[float, float], float] = random.uniform

    def context(self) -> ssl.SSLContext:
        if self._ssl_context is None:
            # Matches urllib's default verification, including the
            # SSL_CERT_FILE override the node installer relies on.
            self._ssl_context = ssl.create_default_context()
        return self._ssl_context

    def _cache(self) -> OrderedDict[ConnectionKey, _PooledConnection]:
        cache = getattr(self._local, "connections", None)
        if cache is None:
            cache = OrderedDict()
            self._local.connections = cache
        return cache

    def close(self) -> None:
        cache = self._cache()
        for record in cache.values():
            record.connection.close()
        cache.clear()

    # -- counters -----------------------------------------------------------

    def _count(self, name: str) -> None:
        with self._stats_lock:
            self._counters[name] += 1

    def stats(self) -> dict[str, int | float]:
        """The process-wide counters plus the policy they were counted under."""

        with self._stats_lock:
            snapshot: dict[str, int | float] = dict(self._counters)
        snapshot["max_idle_seconds"] = self.policy.max_idle_seconds
        snapshot["max_age_seconds"] = self.policy.max_age_seconds
        snapshot["tcp_keepalive_seconds"] = self.policy.tcp_keepalive_seconds
        return snapshot

    def reset_stats(self) -> None:
        with self._stats_lock:
            self._counters = dict.fromkeys(STAT_NAMES, 0)

    def _maybe_log(self, now: float) -> None:
        with self._stats_lock:
            if now - self._last_log < STATS_LOG_INTERVAL_SECONDS:
                return
            self._last_log = now
            counters = dict(self._counters)
        LOGGER.info(
            "transport pool: opened=%d reused=%d stale_retried=%d stale_failed=%d "
            "idle_expired=%d age_expired=%d evicted=%d peer_closed=%d (idle max "
            "%.0f s, age max %.0f s, tcp keepalive %.0f s)",
            *(counters[name] for name in STAT_NAMES),
            self.policy.max_idle_seconds,
            self.policy.max_age_seconds,
            self.policy.tcp_keepalive_seconds,
        )

    # -- the cache ----------------------------------------------------------

    def _expiry(self, now: float) -> float | None:
        if self.policy.max_age_seconds <= 0:
            return None
        spread = self.policy.max_age_seconds * AGE_JITTER_RATIO
        return now + self.policy.max_age_seconds + self.jitter(-spread, spread)

    def _expired(self, record: _PooledConnection, now: float) -> str | None:
        """Why a cached connection must not be reused, or ``None``."""

        if now - record.last_used_at >= self.policy.max_idle_seconds:
            return "idle_expired_total"
        if record.expires_at is not None and now >= record.expires_at:
            return "age_expired_total"
        return None

    def _store(self, key: ConnectionKey, record: _PooledConnection) -> None:
        cache = self._cache()
        previous = cache.pop(key, None)
        if previous is not None and previous is not record:
            previous.connection.close()
        cache[key] = record
        while len(cache) > self.max_connections:
            _, evicted = cache.popitem(last=False)
            evicted.connection.close()
            self._count("evicted_total")

    def _connect(
        self,
        scheme: str,
        host: str,
        port: int,
        timeout: float | None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> HTTPConnection:
        if scheme == "https":
            return HTTPSConnection(
                host,
                port,
                timeout=timeout,
                context=ssl_context or self.context(),
            )
        return HTTPConnection(host, port, timeout=timeout)

    # -- requests -----------------------------------------------------------

    def send(
        self,
        target: Request | str,
        timeout: float | None = None,
        *,
        ssl_context: ssl.SSLContext | None = None,
    ) -> addinfourl:
        request = target if isinstance(target, Request) else Request(target)
        parsed = urlsplit(request.full_url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        if (
            scheme not in {"http", "https"}
            or not host
            or (scheme in self._proxies and not proxy_bypass(host))
        ):
            return _urllib_urlopen(
                request,
                timeout=timeout,
                context=ssl_context,
            )
        port = parsed.port or (443 if scheme == "https" else 80)
        key: ConnectionKey = (
            (scheme, host, port, id(ssl_context))
            if ssl_context is not None
            else (scheme, host, port)
        )
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        headers = dict(request.header_items())
        headers.setdefault("Connection", "keep-alive")
        now = time.monotonic()
        self._maybe_log(now)

        reused = self._cache().pop(key, None)
        if reused is not None:
            expired = self._expired(reused, now)
            if expired is None and peer_closed(reused.connection):
                # The peer closed it while it idled (its keep-alive timeout, a
                # Pod leaving a roll). Nothing has been written yet, so a fresh
                # connection carries any request, a claim included.
                expired = "peer_closed_total"
            if expired is not None:
                # Closed here, cleanly, rather than discovered dead by the
                # request: the NLB has dropped (or is about to drop) an idle
                # flow, and an aged connection is replaced so the fleet
                # re-spreads over Pods and workers.
                reused.connection.close()
                self._count(expired)
                reused = None
        if reused is not None:
            try:
                response = self._exchange(reused, request, key, path, headers, timeout)
            except HTTPError:
                # A received response is not a broken keep-alive connection.
                self._count("requests_reused_total")
                raise
            except _RequestNotWritten:
                # The write itself failed: the peer never had the whole
                # request, so sending it again is safe for any request.
                reused.connection.close()
                self._count("stale_connection_retries_total")
            except (HTTPException, OSError) as exc:
                reused.connection.close()
                if not self._retryable_request(request, headers):
                    self._count("stale_connection_failures_total")
                    if isinstance(exc, HTTPException):
                        raise URLError(exc) from exc
                    raise
                self._count("stale_connection_retries_total")
            else:
                self._count("requests_reused_total")
                return response
        connection = self._connect(
            scheme,
            host,
            port,
            timeout,
            ssl_context,
        )
        self._count("connections_opened_total")
        record = _PooledConnection(
            connection, opened_at=now, last_used_at=now, expires_at=self._expiry(now)
        )
        try:
            return self._exchange(record, request, key, path, headers, timeout)
        except HTTPError:
            raise
        except _RequestNotWritten as failure:
            connection.close()
            unsent = failure.error
        except HTTPException as exc:
            connection.close()
            raise URLError(exc) from exc
        except OSError:
            connection.close()
            raise
        # A fresh connection is never retried here: the caller sees what the
        # write (or the connect inside it) raised, as before -- raised outside
        # the handler so it carries no link back to the internal wrapper.
        if isinstance(unsent, HTTPException):
            raise URLError(unsent) from unsent
        raise unsent

    @staticmethod
    def _retryable_request(
        request: Request,
        headers: dict[str, str],
    ) -> bool:
        if request.get_method() in {
            "GET",
            "HEAD",
            "OPTIONS",
            "PUT",
            "DELETE",
        }:
            return True
        normalized = {key.lower(): value for key, value in headers.items()}
        return any(
            str(normalized.get(name.lower(), "")).strip()
            for name in RETRY_IDENTITY_HEADERS
        )

    def _exchange(
        self,
        record: _PooledConnection,
        request: Request,
        key: ConnectionKey,
        path: str,
        headers: dict[str, str],
        timeout: float | None,
    ) -> addinfourl:
        connection = record.connection
        if timeout is not None:
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
        try:
            connection.request(
                request.get_method(),
                path,
                body=request.data,
                headers=headers,
            )
        except (HTTPException, OSError) as exc:
            # ``http.client`` writes the head, then the body: a failure here
            # leaves the peer at most a head without its whole body, which no
            # server acts on (ours read the body before any route runs).
            raise _RequestNotWritten(exc) from exc
        response = connection.getresponse()
        body = response.read()
        if response.will_close or response.version != 11:
            connection.close()
        else:
            record.last_used_at = time.monotonic()
            if not record.keepalive_applied:
                # The socket exists only after the first request; keepalive
                # matters only for the idle time between requests, so now is
                # exactly early enough.
                record.keepalive_applied = True
                enable_tcp_keepalive(connection.sock, self.policy.tcp_keepalive_seconds)
            self._store(key, record)
        return self._result(request.full_url, response, body)

    @staticmethod
    def _result(url: str, response: HTTPResponse, body: bytes) -> addinfourl:
        stream = io.BytesIO(body)
        if response.status >= 400:
            raise HTTPError(
                url,
                response.status,
                response.reason,
                response.msg,
                stream,
            )
        return addinfourl(stream, response.msg, url, response.status)


CONNECTION_POOL = _KeepAliveConnectionPool()


def urlopen(
    target: Request | str,
    timeout: float | None = None,
    *,
    ssl_context: ssl.SSLContext | None = None,
) -> addinfourl:
    """Keep-alive capable stand-in for ``urllib.request.urlopen``."""

    return CONNECTION_POOL.send(
        target,
        timeout=timeout,
        ssl_context=ssl_context,
    )


def transport_stats() -> dict[str, int | float]:
    """The process's connection reuse and reconnect counters."""

    return CONNECTION_POOL.stats()
