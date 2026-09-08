"""Every writer ``PostgresStore`` inherits is one statement or one transaction.

Architecture review 2026-09-07, item D2. ``PostgresStore.__init__`` sets
``self._lock = nullcontext()``: the process-wide RLock the in-memory and SQLite
stores serialize on means nothing across replicas, so an inherited writer that
does two store round-trips under ``with self._lock`` runs them as two
autocommit statements with nothing between them. A single statement is atomic
on its own; anything more must run inside ``_state_transaction`` or
``_db.transaction()`` -- either in the shared mixin itself or in a Postgres
override.

This is a structural test: it walks the AST of the module that defines each
inherited public method, counts the store round-trips in its body and checks
that more than one of them sits under a transaction ``with`` block. It reads
module files through ``__file__`` rather than ``inspect.getsource`` because it
audits the shape of the code, not its behaviour; the behavioural half of item
D2 is ``test_save_plan_guard.py``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from gpu_fault.store import PostgresStore

POSTGRES_PACKAGE = "gpu_fault.store.postgres"

# Method calls on ``self`` that are one store round-trip each.
ROUND_TRIP_METHODS = frozenset(
    {
        "_put",
        "_link",
        "_delete",
        "_get",
        "_get_optional",
        "_get_for_update",
        "_get_link",
        "_list",
        "get_incident",
        "get_workflow",
        "get_plan",
    }
)
# Cursor / connection statements: ``self._db.execute(...)``,
# ``cursor.execute(...)`` and the like.
EXECUTE_METHODS = frozenset({"execute", "executemany"})
WRITE_METHODS = frozenset({"_put", "_link", "_delete"})
TRANSACTION_CONTEXTS = frozenset({"_state_transaction", "transaction"})

# The inherited writers as reviewed for item D2. A new inherited writer has to
# be added here on purpose, after checking its statement count on Postgres.
REVIEWED_INHERITED_WRITERS = frozenset(
    {
        "acquire_periodic_task_lease",
        "acquire_processor_leadership",
        "add_marker",
        "amend_workflow",
        "apply_efa_traffic_admin_action",
        "complete_remote_command",
        "complete_xid_correlation",
        "create_incident_workflow_if_absent",
        "delete_regional_cluster",
        "enqueue_notification_delivery",
        "ensure_remote_command",
        "establish_notification_watermark",
        "merge_attempt_fault_workflow",
        "merge_replacement_workflow",
        "observe_efa_traffic",
        "publish_regional_registry_revision",
        "reconcile_restored_workflow",
        "reconcile_retired_generation_workflow",
        "record_xid74_occurrences",
        "release_job_restart",
        "renew_remote_command_lease",
        "reserve_hyperpod_submission",
        "reserve_job_restart",
        "save_agent",
        "save_barrier",
        "save_collector_metrics_snapshot",
        "save_decision",
        "save_fleet_deployment",
        "save_hyperpod_node_identity",
        "save_hyperpod_submission",
        "save_incident_and_workflow",
        "save_notification_result",
        "save_profile",
        "save_regional_cluster",
        "save_regional_registry_member",
        "save_workload_coverage",
        "save_xid_policy_decision",
    }
)


def _defining_class(name: str) -> type:
    for cls in PostgresStore.__mro__:
        if name in vars(cls):
            return cls
    raise AttributeError(name)


def _method_node(cls: type, name: str) -> ast.FunctionDef:
    module = sys.modules[cls.__module__]
    tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return child
    raise LookupError(f"{cls.__module__}.{cls.__name__}.{name}")


def _method_name(call: ast.Call) -> str | None:
    return call.func.attr if isinstance(call.func, ast.Attribute) else None


def _round_trips(function: ast.FunctionDef) -> list[ast.Call]:
    trips: list[ast.Call] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        method = _method_name(node)
        if method is None:
            continue
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        on_self = isinstance(receiver, ast.Name) and receiver.id == "self"
        if (on_self and method in ROUND_TRIP_METHODS) or method in EXECUTE_METHODS:
            trips.append(node)
    return trips


def _writes(function: ast.FunctionDef) -> bool:
    for call in _round_trips(function):
        method = _method_name(call)
        if method in WRITE_METHODS:
            return True
        if method in EXECUTE_METHODS and any(
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and any(
                verb in arg.value.upper() for verb in ("INSERT", "UPDATE", "DELETE")
            )
            for arg in call.args
        ):
            return True
    return False


def _under_transaction(function: ast.FunctionDef, call: ast.Call) -> bool:
    """Whether ``call`` is nested inside a ``with`` that opens a transaction."""

    def enclosing(node: ast.AST, target: ast.Call, stack: list[ast.With]) -> bool:
        if node is target:
            return any(
                isinstance(item.context_expr, ast.Call)
                and _method_name(item.context_expr) in TRANSACTION_CONTEXTS
                for block in stack
                for item in block.items
            )
        if isinstance(node, ast.With):
            stack.append(node)
            found = any(
                enclosing(child, target, stack) for child in ast.iter_child_nodes(node)
            )
            stack.pop()
            return found
        return any(
            enclosing(child, target, stack) for child in ast.iter_child_nodes(node)
        )

    return enclosing(function, call, [])


def _inherited_public_methods() -> dict[str, type]:
    inherited: dict[str, type] = {}
    for name in dir(PostgresStore):
        if name.startswith("_") or not callable(getattr(PostgresStore, name)):
            continue
        cls = _defining_class(name)
        if not cls.__module__.startswith(POSTGRES_PACKAGE):
            inherited[name] = cls
    return inherited


def _inherited_writers() -> dict[str, tuple[type, ast.FunctionDef]]:
    writers: dict[str, tuple[type, ast.FunctionDef]] = {}
    for name, cls in _inherited_public_methods().items():
        try:
            function = _method_node(cls, name)
        except LookupError:
            continue  # a descriptor or an alias, not a def
        if _writes(function):
            writers[name] = (cls, function)
    return writers


def test_the_inherited_writer_set_is_the_reviewed_one() -> None:
    assert set(_inherited_writers()) == REVIEWED_INHERITED_WRITERS, (
        "an inherited writer was added or removed on PostgresStore; review its "
        "statement count against the docstring of this test and update the set"
    )


@pytest.mark.parametrize("name", sorted(REVIEWED_INHERITED_WRITERS))
def test_inherited_writer_is_single_statement_or_transactional(name: str) -> None:
    cls, function = _inherited_writers()[name]
    trips = _round_trips(function)
    if len(trips) <= 1:
        return
    outside = [
        ast.get_source_segment(
            Path(sys.modules[cls.__module__].__file__ or "").read_text(
                encoding="utf-8"
            ),
            call,
        )
        for call in trips
        if not _under_transaction(function, call)
    ]
    assert outside == [], (
        f"{cls.__module__}.{cls.__name__}.{name} is inherited by PostgresStore and "
        f"makes {len(trips)} store round-trips, of which these run outside a "
        f"transaction (the process lock is a nullcontext there): {outside}; "
        "override it on Postgres inside _state_transaction"
    )
