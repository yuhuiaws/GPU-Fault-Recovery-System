"""Production Executor HTTP protocol restricted to an isolated capacity API."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit
from urllib.request import Request

from gpu_fault.cluster_executor.regional_client import RegionalExecutorClient
from gpu_fault.transport.http_client import CONNECTION_POOL

PAYLOAD_BUDGET_BYTES = 1024 * 1024


class CapacityWireError(RuntimeError):
    pass


class CapacityThreadsRunning(CapacityWireError):
    """Cleanup cannot remove an API still used by unconfirmed Executor threads."""


ExecutorPins = Mapping[str, str | None]


def executor_pin_arguments(pins: ExecutorPins | None) -> dict[str, str]:
    """``RegionalExecutorClient`` pin keywords from the fleet's declared pins.

    The probe control plane enforces the namespace's executor pins, so every
    runner executor presents them the way the live executor Deployment does
    (``GPU_FAULT_EXECUTOR_ARTIFACT_SHA256`` and its compatibility digest).
    """

    if not pins:
        return {}
    allowed = {"executor_artifact_sha256", "executor_compatibility_digest"}
    unknown = set(pins) - allowed
    if unknown:
        raise CapacityWireError(f"unknown executor pin fields: {sorted(unknown)}")
    return {name: value for name, value in pins.items() if value}


class CapacityWireClient(RegionalExecutorClient):
    def __init__(
        self,
        url: str,
        token: str,
        *,
        cluster_id: str,
        executor_pins: ExecutorPins | None = None,
    ) -> None:
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
        super().__init__(
            url,
            cluster_id,
            token,
            timeout_seconds=3,
            **executor_pin_arguments(executor_pins),
        )
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
