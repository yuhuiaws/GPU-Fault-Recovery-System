class NotFoundError(KeyError):
    pass


class WorkflowLeaseError(ValueError):
    pass


class RemediationBudgetError(WorkflowLeaseError):
    pass


class EfaTrafficAdminConflict(ValueError):
    pass


_RETRYABLE_POSTGRES_SQLSTATES = frozenset(
    {
        "25006",  # read_only_sql_transaction
        "57P01",  # admin_shutdown
        "57P02",  # crash_shutdown
        "57P03",  # cannot_connect_now
    }
)


def is_retryable_store_unavailable(exc: BaseException) -> bool:
    """Whether a PostgreSQL writer/connection failure should return 503."""

    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(6):
        if current is None or id(current) in seen:
            return False
        seen.add(id(current))
        error_type = type(current)
        module = error_type.__module__
        name = error_type.__name__
        sqlstate = getattr(current, "sqlstate", None)
        if module.startswith("psycopg_pool") and name == "PoolTimeout":
            return True
        if module.startswith("psycopg") and (
            name == "OperationalError"
            or sqlstate == "25006"
            or (isinstance(sqlstate, str) and sqlstate.startswith("08"))
            or sqlstate in _RETRYABLE_POSTGRES_SQLSTATES
        ):
            return True
        current = current.__cause__ or current.__context__
    return False
