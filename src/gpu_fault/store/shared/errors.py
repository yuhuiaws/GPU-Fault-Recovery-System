from collections.abc import Iterator


class NotFoundError(KeyError):
    pass


class WorkflowLeaseError(ValueError):
    pass


class StaleFencingTokenError(WorkflowLeaseError):
    """The caller's fencing token no longer matches the record's generation.

    A ``WorkflowLeaseError`` so that every caller that already treats "someone
    else owns this record" as a reason to stand aside does the same here. It
    used to be a bare ``ValueError``, which the dispatcher read as an internal
    error and answered by writing the workflow BLOCKED (F-J2).
    """


class WorkflowMergedError(WorkflowLeaseError):
    """The workflow was merged into since the caller read it.

    Raised by the leased writers when ``merge_revision`` moved: the executor's
    copy predates a merge that widened the record, and writing it would erase
    the merged targets (F-B1). A ``WorkflowLeaseError`` so the executor stands
    aside exactly as it does for a lost lease and resumes from the stored
    record on its next renewal.
    """


class RemediationBudgetError(WorkflowLeaseError):
    pass


class TransactionRequiredError(RuntimeError):
    """A row-locking read was attempted outside a transaction (F-J4).

    The pool is autocommit: a ``SELECT ... FOR UPDATE`` on its own connection
    releases the lock before the caller sees the row, so the read protects
    nothing. Raised instead of silently returning an unlocked row.
    """


class StaleWriteError(ValueError):
    """A conditional write found the row changed (or gone) since it was read."""


class EfaTrafficAdminConflict(ValueError):
    pass


# One SQLSTATE table, three questions (F-J1). The sets nest:
# connection_is_lost ⊆ writer_unavailable ⊆ operation_should_retry.
#
# Classification is by SQLSTATE and by psycopg exception *class*, walking the
# cause chain; matching class names let every psycopg subclass through as
# "not transient", which is how a serialization failure became a BLOCKED
# workflow (P0-71A).

# The connection itself is gone: the pool must discard it.
_CONNECTION_LOST_SQLSTATES = frozenset(
    {
        "25006",  # read_only_sql_transaction: pinned to a demoted writer
        "57P01",  # admin_shutdown
        "57P02",  # crash_shutdown
        "57P03",  # cannot_connect_now
    }
)
# The writer cannot take work right now: ingress answers 503 and the caller
# retries later. Includes every lost connection.
_WRITER_UNAVAILABLE_SQLSTATES = _CONNECTION_LOST_SQLSTATES | frozenset(
    {
        "53000",  # insufficient_resources
        "53100",  # disk_full
        "53200",  # out_of_memory
        "53300",  # too_many_connections
        "53400",  # configuration_limit_exceeded
    }
)
# The statement lost a race or a budget and the same operation can simply be
# run again. Includes everything above.
_OPERATION_RETRY_SQLSTATES = _WRITER_UNAVAILABLE_SQLSTATES | frozenset(
    {
        "40001",  # serialization_failure
        "40P01",  # deadlock_detected
        "55P03",  # lock_not_available
        "57014",  # query_canceled (statement_timeout, lock_timeout)
    }
)
_MAX_CHAIN_DEPTH = 6


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_MAX_CHAIN_DEPTH):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _psycopg_class(error: BaseException) -> str | None:
    error_type = type(error)
    module = error_type.__module__ or ""
    if module.startswith("psycopg_pool"):
        return f"pool.{error_type.__name__}"
    if module.startswith("psycopg"):
        return error_type.__name__
    return None


def _sqlstate(error: BaseException) -> str | None:
    value = getattr(error, "sqlstate", None)
    return value if isinstance(value, str) else None


def _is_operational(error: BaseException) -> bool:
    """psycopg raises OperationalError for socket-level failures without SQLSTATE.

    Matched by module and class name rather than ``isinstance`` so this module
    never imports the optional driver; a pool timeout is excluded because it
    never held a connection to lose.
    """

    return _psycopg_class(error) == "OperationalError" and _sqlstate(error) is None


def connection_is_lost(exc: BaseException) -> bool:
    """Whether the connection that raised ``exc`` must be discarded."""

    for error in _chain(exc):
        sqlstate = _sqlstate(error)
        if _psycopg_class(error) is None:
            continue
        if sqlstate is not None and (
            sqlstate.startswith("08") or sqlstate in _CONNECTION_LOST_SQLSTATES
        ):
            return True
        if _is_operational(error):
            return True
    return False


def writer_unavailable(exc: BaseException) -> bool:
    """Whether the PostgreSQL writer cannot take work right now (answer 503)."""

    if connection_is_lost(exc):
        return True
    for error in _chain(exc):
        kind = _psycopg_class(error)
        if kind is None:
            continue
        if kind == "pool.PoolTimeout":
            return True
        sqlstate = _sqlstate(error)
        if sqlstate is not None and sqlstate in _WRITER_UNAVAILABLE_SQLSTATES:
            return True
    return False


def operation_should_retry(exc: BaseException) -> bool:
    """Whether the failed operation can simply be run again (never BLOCK on it)."""

    if writer_unavailable(exc):
        return True
    for error in _chain(exc):
        kind = _psycopg_class(error)
        if kind is None:
            continue
        if kind in {"pool.PoolClosed", "pool.TooManyRequests", "ConnectionTimeout"}:
            return True
        sqlstate = _sqlstate(error)
        if sqlstate is not None and sqlstate in _OPERATION_RETRY_SQLSTATES:
            return True
    return False


def is_retryable_store_unavailable(exc: BaseException) -> bool:
    """Whether a PostgreSQL writer/connection failure should return 503.

    Kept as the ingress-facing name; it is the ``writer_unavailable`` question.
    """

    return writer_unavailable(exc)
