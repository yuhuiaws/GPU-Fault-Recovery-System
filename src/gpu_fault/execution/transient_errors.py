from __future__ import annotations

from gpu_fault.store.shared.errors import operation_should_retry
from gpu_fault.transport_errors import retryable_transport_error

# ``kubernetes.client.ApiException`` statuses that describe the API server's
# moment rather than the request: a conflict on a stale resourceVersion,
# throttling, and the 5xx family. Anything else (400/401/403/404/422) is the
# API server's definitive answer and stays a step failure.
RETRYABLE_KUBERNETES_STATUSES = frozenset({409, 429, 500, 502, 503, 504})
# urllib3 raises these under the Kubernetes client for a socket that timed
# out, a pool that gave up, a torn connection or a refused handshake.
_RETRYABLE_URLLIB3_ERRORS = frozenset(
    {
        "ConnectTimeoutError",
        "MaxRetryError",
        "NewConnectionError",
        "ProtocolError",
        "ReadTimeoutError",
    }
)
_CAUSE_CHAIN_LIMIT = 5


def transient_store_error(exc: BaseException) -> bool:
    """Whether a failed store operation should simply be retried.

    The one shared classifier (``gpu_fault.store.shared.errors``): a
    serialization failure, a deadlock or a statement timeout is a retry, not a
    reason to write the workflow BLOCKED (F-J1).
    """

    return operation_should_retry(exc)


def retryable_adapter_error(exc: BaseException) -> bool:
    """Whether an adapter exception says nothing about the step it ran (ARCH-B1).

    Symmetric with ``transient_store_error``: a Kubernetes 409/429/5xx, a
    urllib3 timeout or connection error, or a transport-level failure (a reset
    connection, a timeout, a retryable errno, a transient AWS transport error)
    is safe to try again, so the step waits instead of failing into isolation
    and hardware escalation. The Kubernetes and urllib3 classes are matched by
    module and class name so the classifier works when those extras are not
    installed. The ``__cause__``/``__context__`` chain is walked because
    adapters wrap what they catch.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_CAUSE_CHAIN_LIMIT):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        module = type(current).__module__ or ""
        name = type(current).__name__
        if module.startswith("kubernetes") and name == "ApiException":
            return getattr(current, "status", None) in RETRYABLE_KUBERNETES_STATUSES
        if module.startswith("urllib3") and name in _RETRYABLE_URLLIB3_ERRORS:
            return True
        current = current.__cause__ or current.__context__
    return retryable_transport_error(exc) is not None
