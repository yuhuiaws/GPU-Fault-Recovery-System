from __future__ import annotations


_TRANSIENT_PSYCOPG_ERRORS = frozenset(
    {
        "ConnectionTimeout",
        "InterfaceError",
        "OperationalError",
        "PoolClosed",
        "PoolTimeout",
        "TooManyRequests",
    }
)
_MAX_CHAIN_DEPTH = 5


def transient_store_error(exc: BaseException) -> bool:
    """Return true for database connectivity errors safe to retry."""

    current: BaseException | None = exc
    seen: set[int] = set()
    for _ in range(_MAX_CHAIN_DEPTH):
        if current is None or id(current) in seen:
            return False
        seen.add(id(current))
        module = type(current).__module__ or ""
        name = type(current).__name__
        sqlstate = getattr(current, "sqlstate", None)
        if module.startswith(("psycopg", "psycopg_pool")) and (
            name in _TRANSIENT_PSYCOPG_ERRORS
            or (isinstance(sqlstate, str) and sqlstate.startswith("08"))
        ):
            return True
        current = current.__cause__ or current.__context__
    return False
