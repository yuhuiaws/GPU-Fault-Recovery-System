from __future__ import annotations

import socket
from urllib.error import URLError

from gpu_fault.transport_errors import retryable_transport_error


def test_wrapped_temporary_dns_failure_is_retryable() -> None:
    error = URLError(
        socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
    )

    assert "temporary DNS failure" in (retryable_transport_error(error) or "")


def test_a_name_error_for_a_fixed_endpoint_is_retryable() -> None:
    """Live 2026-09-08 (DESTR-014): a sibling node's reboot took a CoreDNS
    replica down and BatchRebootClusterNodes failed with EAI_NONAME for the
    SageMaker endpoint -- a resolver disruption, not a permanent name error.
    The endpoints this code calls never change, so every resolver failure is
    transient."""
    error = socket.gaierror(socket.EAI_NONAME, "Name or service not known")
    reason = retryable_transport_error(RuntimeError("wrapped").with_traceback(None))
    assert reason is None, "an unrelated error stays non-retryable"
    assert retryable_transport_error(error) is not None, "EAI_NONAME must retry"
    assert "DNS" in str(retryable_transport_error(error))
