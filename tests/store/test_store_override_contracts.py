"""Guard the store subclasses against silently dropping a dict key.

``PostgresStore`` re-implements dozens of ``InMemoryStore`` methods as SQL
so the work happens in the database instead of in Python. Several of them
return a statistics mapping whose keys are read positionally by name
somewhere else -- ``/metrics`` being the worst case, because it reads
``remote_command_stats()['unclaimed_expired_total']`` unconditionally, so
an override that forgets the key does not degrade one gauge: the whole
endpoint answers 500 and every other metric disappears with it. That is
exactly what shipped once, and it was invisible to the suite because the
Postgres tests only run when ``GPU_FAULT_TEST_POSTGRES_URL`` is set.

This test needs no database. It parses ``store.py`` and compares the
literal dict keys each override returns against the ones the base
implementation (or the module-level helper the base delegates to)
returns.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "src" / "gpu_fault"
TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = tuple(
    sorted(
        path for path in (PACKAGE / "store").rglob("*.py") if path.name != "__init__.py"
    )
)
LAYERS = (
    (
        (
            "SqliteStore",
            "SqliteCoreMixin",
            "SqliteControlRecordMixin",
            "SqliteEfaTrafficMixin",
            "SqliteFleetMixin",
            "SqliteNotificationMixin",
            "SqliteRemoteCommandMixin",
            "TransactionalWorkflowMixin",
            "SqliteWorkflowMixin",
            "SqliteXidMixin",
            "SqliteTelemetryMixin",
            "SqliteProcessorQueueMixin",
            "SqliteProcessorLeaseMixin",
        ),
        (
            "InMemoryStore",
            "MemoryControlRecordMixin",
            "MemoryEfaTrafficMixin",
            "MemoryNotificationMixin",
            "MemoryRemoteCommandMixin",
            "MemoryWorkflowMixin",
            "MemoryXidMixin",
            "MemoryTelemetryMixin",
            "MemoryTelemetrySpoolMixin",
            "MemoryProcessorQueueMixin",
            "MemoryProcessorLeaseMixin",
            "MemoryFleetMixin",
        ),
    ),
    (
        (
            "PostgresStore",
            "PostgresCoreMixin",
            "PostgresSchemaMixin",
            "PostgresControlRecordMixin",
            "PostgresFleetMixin",
            "PostgresNotificationMixin",
            "PostgresRemoteCommandMixin",
            "PostgresWorkflowMixin",
            "PostgresXidMixin",
            "PostgresGpuTelemetryMixin",
            "PostgresCollectorTelemetryMixin",
            "PostgresTelemetrySpoolMixin",
            "PostgresProcessorAdminMixin",
            "PostgresProcessorAdmissionMixin",
            "PostgresProcessorClaimsMixin",
            "PostgresProcessorCompletionMixin",
            "PostgresProcessorLeaseMixin",
            "PostgresProcessorStorageMixin",
        ),
        (
            "SqliteStore",
            "SqliteCoreMixin",
            "SqliteControlRecordMixin",
            "SqliteEfaTrafficMixin",
            "SqliteFleetMixin",
            "SqliteNotificationMixin",
            "SqliteRemoteCommandMixin",
            "TransactionalWorkflowMixin",
            "SqliteWorkflowMixin",
            "SqliteXidMixin",
            "SqliteTelemetryMixin",
            "SqliteProcessorQueueMixin",
            "SqliteProcessorLeaseMixin",
            "InMemoryStore",
            "MemoryControlRecordMixin",
            "MemoryEfaTrafficMixin",
            "MemoryNotificationMixin",
            "MemoryRemoteCommandMixin",
            "MemoryWorkflowMixin",
            "MemoryXidMixin",
            "MemoryTelemetryMixin",
            "MemoryTelemetrySpoolMixin",
            "MemoryProcessorQueueMixin",
            "MemoryProcessorLeaseMixin",
            "MemoryFleetMixin",
        ),
    ),
)


def _literal_dict_key_sets(function: ast.FunctionDef) -> list[set[str]]:
    """Keys of every ``return {...}`` whose keys are all string literals."""

    key_sets: list[set[str]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Return):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        keys: set[str] = set()
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
            else:
                # A computed or splatted key means the shape is not
                # statically knowable; skip rather than guess.
                keys = set()
                break
        if keys:
            key_sets.append(keys)
    return key_sets


def _methods(class_node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)
    }


def _source_definitions():
    classes: dict[str, ast.ClassDef] = {}
    helpers: dict[str, ast.FunctionDef] = {}
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes.update(
            {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        )
        helpers.update(
            {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        )
    return classes, helpers


def _layer_methods(
    classes: dict[str, ast.ClassDef], names: tuple[str, ...]
) -> dict[str, ast.FunctionDef]:
    methods: dict[str, ast.FunctionDef] = {}
    for name in reversed(names):
        methods.update(_methods(classes[name]))
    return methods


def test_store_overrides_return_the_same_dict_keys() -> None:
    classes, helpers = _source_definitions()

    drift: list[str] = []
    compared = 0
    for override_names, base_names in LAYERS:
        overrides = _layer_methods(classes, override_names)
        base = _layer_methods(classes, base_names)
        for name, override in overrides.items():
            if name not in base:
                continue
            base_keys = _literal_dict_key_sets(base[name])
            if not base_keys:
                # The base often delegates to a module-level helper so
                # the in-memory and SQLite stores share one shape; that
                # helper is the contract the SQL override must match.
                for node in ast.walk(base[name]):
                    if (
                        isinstance(node, ast.Return)
                        and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id in helpers
                    ):
                        base_keys = _literal_dict_key_sets(helpers[node.value.func.id])
            override_keys = _literal_dict_key_sets(override)
            if not base_keys or not override_keys:
                continue
            expected = set().union(*base_keys)
            actual = set().union(*override_keys)
            compared += 1
            if expected != actual:
                drift.append(
                    f"{override_names[0]}.{name}: missing "
                    f"{sorted(expected - actual)}, extra "
                    f"{sorted(actual - expected)}"
                )

    assert compared, "no override returned a literal dict to compare"
    assert not drift, "store override dict keys drifted: " + "; ".join(drift)


# Drivers that only the optional Postgres tests need. Importing one at
# module scope defeats the skip marker underneath it: pytest imports
# every test module before it looks at markers, so on a machine without
# the driver collection raises and *the entire suite* is interrupted --
# not one module skipped, no results at all. Import them inside the
# tests that use them instead.
OPTIONAL_DRIVERS = ("psycopg", "psycopg2")


def test_optional_drivers_are_not_imported_at_module_scope() -> None:
    offenders: list[str] = []
    for path in sorted(TEST_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in OPTIONAL_DRIVERS:
                    offenders.append(f"{path.name}:{node.lineno} {name}")

    assert not offenders, (
        "optional driver imported at module scope, which breaks "
        "collection for the whole suite when it is absent: " + "; ".join(offenders)
    )
