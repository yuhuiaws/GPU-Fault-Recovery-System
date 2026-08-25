from __future__ import annotations

import socket
from urllib.error import URLError

from gpu_fault.transport_errors import retryable_transport_error


def test_wrapped_temporary_dns_failure_is_retryable() -> None:
    error = URLError(
        socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
    )

    assert "temporary DNS failure" in (retryable_transport_error(error) or "")


def test_permanent_dns_name_error_is_not_retryable() -> None:
    error = socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    assert retryable_transport_error(error) is None
