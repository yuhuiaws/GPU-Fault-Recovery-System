from __future__ import annotations

from gpu_fault.store.shared.errors import operation_should_retry


def transient_store_error(exc: BaseException) -> bool:
    """Whether a failed store operation should simply be retried.

    The one shared classifier (``gpu_fault.store.shared.errors``): a
    serialization failure, a deadlock or a statement timeout is a retry, not a
    reason to write the workflow BLOCKED (F-J1).
    """

    return operation_should_retry(exc)
