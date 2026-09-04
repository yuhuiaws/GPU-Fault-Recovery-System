from importlib import import_module
from typing import Any

_EXPORTS = {
    "EfaTrafficAdminConflict": (
        "gpu_fault.store.shared.errors",
        "EfaTrafficAdminConflict",
    ),
    "InMemoryStore": ("gpu_fault.store.memory.store", "InMemoryStore"),
    "NotFoundError": ("gpu_fault.store.shared.errors", "NotFoundError"),
    "RemediationBudgetError": (
        "gpu_fault.store.shared.errors",
        "RemediationBudgetError",
    ),
    "POSTGRES_SCHEMA_VERSION": (
        "gpu_fault.store.postgres.store",
        "POSTGRES_SCHEMA_VERSION",
    ),
    "PostgresStore": ("gpu_fault.store.postgres.store", "PostgresStore"),
    "SimulatedDiagnosticAdapter": (
        "gpu_fault.store.memory.store",
        "SimulatedDiagnosticAdapter",
    ),
    "SqliteStore": ("gpu_fault.store.sqlite.store", "SqliteStore"),
    "WorkflowLeaseError": ("gpu_fault.store.shared.errors", "WorkflowLeaseError"),
    "_PooledPostgresDatabase": (
        "gpu_fault.store.postgres.store",
        "_PooledPostgresDatabase",
    ),
}

__all__ = [name for name in _EXPORTS if not name.startswith("_")]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
