from __future__ import annotations

import io
import ssl
from collections import OrderedDict
from http.client import (
    HTTPConnection,
    HTTPException,
    HTTPResponse,
    HTTPSConnection,
)
from threading import local
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    Request,
    getproxies,
    proxy_bypass,
    urlopen as _urllib_urlopen,
)
from urllib.response import addinfourl


class _KeepAliveConnectionPool:
    """Per-thread HTTP/1.1 connection cache.

    ``urllib.request.urlopen`` opens a new TCP (and for HTTPS a new TLS)
    connection per call, which at collector volume costs one handshake
    per event. Connections are cached per thread because
    ``http.client`` connections are not safe to share, and a reused
    connection is retried once because an idle keep-alive connection can
    be closed by the peer between requests.
    """

    max_connections = 8

    def __init__(self) -> None:
        self._local = local()
        self._ssl_context: ssl.SSLContext | None = None
        self._proxies = getproxies()

    def context(self) -> ssl.SSLContext:
        if self._ssl_context is None:
            # Matches urllib's default verification, including the
            # SSL_CERT_FILE override the node installer relies on.
            self._ssl_context = ssl.create_default_context()
        return self._ssl_context

    def _cache(
        self,
    ) -> OrderedDict[
        tuple[str, str, int] | tuple[str, str, int, int],
        HTTPConnection,
    ]:
        cache = getattr(self._local, "connections", None)
        if cache is None:
            cache = OrderedDict()
            self._local.connections = cache
        return cache

    def close(self) -> None:
        cache = self._cache()
        for connection in cache.values():
            connection.close()
        cache.clear()

    def _store(
        self,
        key: tuple[str, str, int] | tuple[str, str, int, int],
        connection: HTTPConnection,
    ) -> None:
        cache = self._cache()
        previous = cache.pop(key, None)
        if previous is not None and previous is not connection:
            previous.close()
        cache[key] = connection
        while len(cache) > self.max_connections:
            _, evicted = cache.popitem(last=False)
            evicted.close()

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
        key = (
            (scheme, host, port, id(ssl_context))
            if ssl_context is not None
            else (scheme, host, port)
        )
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        headers = dict(request.header_items())
        headers.setdefault("Connection", "keep-alive")

        reused = self._cache().pop(key, None)
        if reused is not None:
            try:
                return self._exchange(reused, request, key, path, headers, timeout)
            except (HTTPException, OSError) as exc:
                reused.close()
                if not self._retryable_request(request, headers):
                    if isinstance(exc, HTTPException):
                        raise URLError(exc) from exc
                    raise
        connection = self._connect(
            scheme,
            host,
            port,
            timeout,
            ssl_context,
        )
        try:
            return self._exchange(connection, request, key, path, headers, timeout)
        except HTTPException as exc:
            connection.close()
            raise URLError(exc) from exc
        except OSError:
            connection.close()
            raise

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
        return bool(
            normalized.get("idempotency-key")
            or normalized.get("x-gpu-fault-request-id")
            or normalized.get("x-gpu-fault-command-id")
        )

    def _exchange(
        self,
        connection: HTTPConnection,
        request: Request,
        key: tuple[str, str, int] | tuple[str, str, int, int],
        path: str,
        headers: dict[str, str],
        timeout: float | None,
    ) -> addinfourl:
        if timeout is not None:
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
        connection.request(
            request.get_method(),
            path,
            body=request.data,
            headers=headers,
        )
        response = connection.getresponse()
        body = response.read()
        if response.will_close or response.version != 11:
            connection.close()
        else:
            self._store(key, connection)
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
