from __future__ import annotations

import errno
import socket
from urllib.error import URLError


_RETRYABLE_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }
)
_RETRYABLE_BOTOCORE_ERRORS = frozenset(
    {
        "ConnectTimeoutError",
        "ConnectionClosedError",
        "EndpointConnectionError",
        "HTTPClientError",
        "ReadTimeoutError",
    }
)


def retryable_transport_error(
    exc: BaseException,
) -> str | None:
    """Return a stable reason for a transient network failure."""

    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(5):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        if isinstance(current, socket.gaierror):
            # Every resolver failure is transient for the fixed provider and
            # control-plane endpoints this code calls: EAI_AGAIN by definition,
            # and EAI_NONAME/EAI_FAIL when the cluster resolver itself is
            # disrupted -- live 2026-09-08 (DESTR-014) a sibling node's reboot
            # took a CoreDNS replica with it and BatchRebootClusterNodes on the
            # other branch failed with "[Errno -2] Name or service not known",
            # turning a DNS blip into a FAILED step and a support escalation.
            return f"temporary DNS failure: {current}"
        if isinstance(current, (TimeoutError, ConnectionError)):
            return f"temporary transport failure: {current}"
        if isinstance(current, OSError) and current.errno in _RETRYABLE_ERRNOS:
            return f"temporary transport failure: {current}"
        if (
            type(current).__module__.startswith("botocore")
            and type(current).__name__ in _RETRYABLE_BOTOCORE_ERRORS
        ):
            return (
                f"temporary AWS transport failure: {type(current).__name__}: {current}"
            )
        reason = (
            current.reason
            if isinstance(current, URLError)
            and isinstance(current.reason, BaseException)
            else None
        )
        current = reason or current.__cause__ or current.__context__
    return None


def retryable_transport_result(
    exc: BaseException,
    *,
    lease_token: str,
    executor_id: str,
):
    reason = retryable_transport_error(exc)
    if reason is None:
        return None
    from gpu_fault.regional import (
        RemoteCommandResult,
        RemoteCommandStatus,
    )

    return RemoteCommandResult(
        lease_token=lease_token,
        status=RemoteCommandStatus.WAITING,
        status_source="executor-retryable-transport",
        details={
            "retryable_transport_error": True,
            "reason": reason,
            "executor_id": executor_id,
            "exception_type": type(exc).__name__,
        },
    )
