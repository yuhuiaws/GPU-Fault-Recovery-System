"""Production Executor HTTP protocol restricted to an isolated capacity API."""

from __future__ import annotations

from urllib.parse import urlsplit
from urllib.request import Request

from gpu_fault.cluster_executor.regional_client import RegionalExecutorClient
from gpu_fault.transport.http_client import CONNECTION_POOL

PAYLOAD_BUDGET_BYTES = 1024 * 1024


class CapacityWireError(RuntimeError):
    pass


class CapacityThreadsRunning(CapacityWireError):
    """Cleanup cannot remove an API still used by unconfirmed Executor threads."""


class CapacityWireClient(RegionalExecutorClient):
    def __init__(self, url: str, token: str, *, cluster_id: str) -> None:
        target = urlsplit(url)
        if (
            target.scheme != "http"
            or target.hostname != "127.0.0.1"
            or not target.port
            or target.path not in {"", "/"}
            or target.username is not None
            or target.password is not None
            or target.query
            or target.fragment
        ):
            raise CapacityWireError("capacity probe requires the isolated loopback URL")
        super().__init__(url, cluster_id, token, timeout_seconds=3)
        self.claim_response_bytes = 0

    def _send(self, request: Request, *, timeout_seconds: float | None = None) -> bytes:
        try:
            body = super()._send(request, timeout_seconds=timeout_seconds)
        finally:
            CONNECTION_POOL.close()
        if request.full_url.endswith("/claim"):
            self.claim_response_bytes = len(body)
            if len(body) > PAYLOAD_BUDGET_BYTES:
                raise CapacityWireError("capacity claim exceeded the payload budget")
        return body
