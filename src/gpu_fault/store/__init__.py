from gpu_fault.lazy_exports import lazy_module

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

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
