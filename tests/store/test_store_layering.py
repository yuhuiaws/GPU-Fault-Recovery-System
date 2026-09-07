"""The three stores are peers composed from shared mixins, not a chain.

``PostgresStore`` used to inherit from ``SqliteStore``, which inherited from
``InMemoryStore``. Dozens of the methods production ran on PostgreSQL were
therefore defined in classes named ``Sqlite*Mixin`` -- template methods over
the ``_put``/``_get``/``_state_transaction`` primitives, correct on every
backend, but filed under the wrong name -- and the Postgres store had to undo
an inherited process-wide ``RLock`` with ``self._lock = nullcontext()``.

These tests pin the replacement: every public ``ControlPlaneStore`` method the
Postgres store runs is defined either by a Postgres mixin or by a
dialect-neutral mixin under ``gpu_fault.store.shared``; the concrete stores do
not inherit from one another; and the shared code reaches the dialect's lock
through an explicit ``_statement_guard`` hook that SQLite implements with its
RLock and Postgres implements, deliberately, as a no-op.
"""

from __future__ import annotations

import ast
import inspect
import pkgutil
import threading
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path

import gpu_fault.store.postgres
import gpu_fault.store.sqlite
from gpu_fault.fleet import AgentRecord
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.store.memory.store import InMemoryStore
from gpu_fault.store.postgres.core import PostgresCoreMixin
from gpu_fault.store.postgres.store import PostgresStore
from gpu_fault.store.sqlite.store import SqliteStore

# How a shared mixin must spell each primitive it requires: the alias exported
# by ``gpu_fault.store.shared.primitives``, so mypy checks the dialect's
# definition against one signature instead of an ad-hoc ``Callable[..., Any]``.
PRIMITIVE_CONTRACTS = {
    "_put": "PutRecord",
    "_delete": "DeleteRecord",
    "_get": "GetRecord",
    "_get_optional": "GetOptionalRecord",
    "_list": "ListRecords",
    "_link": "LinkRecord",
    "_get_link": "GetLink",
    "_state_key": "StateKey",
    "_state_transaction": "StateTransaction",
    "_statement_guard": "StatementGuard",
}


def _public_protocol_methods() -> list[str]:
    return sorted(
        name
        for name in dir(ControlPlaneStore)
        if not name.startswith("_") and callable(getattr(ControlPlaneStore, name))
    )


def _defining_class(cls: type, name: str) -> type:
    for base in cls.__mro__:
        if name in vars(base):
            return base
    raise AssertionError(f"{cls.__name__} does not define {name}")


def test_postgres_runs_only_postgres_or_shared_code() -> None:
    misfiled = {}
    for name in _public_protocol_methods():
        module = _defining_class(PostgresStore, name).__module__
        if not module.startswith(
            ("gpu_fault.store.postgres", "gpu_fault.store.shared")
        ):
            misfiled[name] = module
    assert misfiled == {}, (
        "PostgresStore resolves these ControlPlaneStore methods from another "
        f"dialect's package: {misfiled}"
    )


def test_concrete_stores_do_not_inherit_from_each_other() -> None:
    assert SqliteStore not in PostgresStore.__mro__
    assert InMemoryStore not in PostgresStore.__mro__
    assert InMemoryStore not in SqliteStore.__mro__


def test_shared_mixins_are_written_over_the_primitives_protocol() -> None:
    """A shared mixin may only require attributes the primitives Protocol names,
    besides the public contract it composes and the dialect helpers it
    declares under the same convention."""

    shared_classes = {
        base
        for store in (PostgresStore, SqliteStore, InMemoryStore)
        for base in store.__mro__
        if base.__module__.startswith("gpu_fault.store.shared")
        and base.__name__.endswith("Mixin")
    }
    assert shared_classes, "no shared mixins composed into the stores"
    misspelled = {}
    for cls in shared_classes:
        for name, annotation in inspect.get_annotations(cls).items():
            expected = PRIMITIVE_CONTRACTS.get(name)
            if expected is not None and annotation != expected:
                misspelled[f"{cls.__name__}.{name}"] = annotation
    assert misspelled == {}, (
        "shared mixins must declare primitives with the aliases from "
        f"gpu_fault.store.shared.primitives: {misspelled}"
    )


def _super_calls(package) -> dict[str, set[str]]:
    """``{class name: {method names called through super()}}`` for every class
    defined in the package's modules."""

    calls: dict[str, set[str]] = {}
    for info in pkgutil.iter_modules(package.__path__):
        module = import_module(f"{package.__name__}.{info.name}")
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in ast.walk(node):
                if (
                    isinstance(item, ast.Call)
                    and isinstance(item.func, ast.Attribute)
                    and isinstance(item.func.value, ast.Call)
                    and isinstance(item.func.value.func, ast.Name)
                    and item.func.value.func.id == "super"
                ):
                    calls.setdefault(node.name, set()).add(item.func.attr)
    return calls


def test_every_super_call_has_a_target_later_in_the_store_mro() -> None:
    """A dialect override that falls back with ``super().name(...)`` -- the
    Postgres ``hot_state_mode == "legacy"`` paths -- used to land on the SQLite
    template it inherited. With the chain gone, the target must be a class that
    follows the caller in the concrete store's MRO, or the fallback raises
    ``AttributeError`` on the one deployment mode nothing else exercises."""

    dangling = []
    for store, package in (
        (PostgresStore, gpu_fault.store.postgres),
        (SqliteStore, gpu_fault.store.sqlite),
    ):
        mro = list(store.__mro__)
        by_name = {cls.__name__: cls for cls in mro}
        for class_name, names in _super_calls(package).items():
            caller = by_name.get(class_name)
            if caller is None:
                continue
            later = mro[mro.index(caller) + 1 :]
            for name in sorted(names):
                target = next((cls for cls in later if name in vars(cls)), None)
                if target is None:
                    dangling.append(f"{store.__name__}: {class_name}.{name}")
    assert dangling == [], (
        "super() fallbacks with no target later in the MRO: " + ", ".join(dangling)
    )


def test_postgres_store_declares_no_process_lock() -> None:
    """No class the Postgres store composes declares or assigns ``_lock``; the
    only lock hook shared code may use is ``_statement_guard``."""

    for base in PostgresStore.__mro__:
        assert "_lock" not in vars(base), f"{base.__name__} defines _lock"
        assert "_lock" not in inspect.get_annotations(base), (
            f"{base.__name__} declares a _lock contract"
        )
    guard = _defining_class(PostgresStore, "_statement_guard")
    assert guard is PostgresCoreMixin
    assert isinstance(PostgresCoreMixin()._statement_guard(), nullcontext), (
        "Postgres must hold no statement lock"
    )


def test_sqlite_statement_guard_is_the_store_lock(tmp_path) -> None:
    """The SQLite store still serialises single-statement writes: a shared
    template method blocks while another thread holds the statement guard."""

    store = SqliteStore(str(tmp_path / "layering.db"))
    try:
        lock = store._statement_guard()
        assert isinstance(lock, type(threading.RLock())), (
            "sqlite must guard statements with one RLock"
        )
        assert store._statement_guard() is lock
        now = datetime.now(timezone.utc)
        agent = AgentRecord(
            cluster_id="cluster-a",
            node_id="node-a",
            endpoint="http://node-a:9099",
            agent_protocol_version=3,
            node_action_key_version=2,
            agent_version="0.10.0",
            artifact_sha256="b" * 64,
            policy_version="catalog-a",
            runtime_profile_version="hyperpod-v1",
            config_digest="c" * 64,
            allowed_operations=[],
            first_seen_at=now,
            last_seen_at=now,
            lease_expires_at=now + timedelta(minutes=5),
        )
        done = threading.Event()

        def write() -> None:
            store.save_agent(agent)
            done.set()

        lock.acquire()
        try:
            worker = threading.Thread(target=write, daemon=True)
            worker.start()
            assert not done.wait(0.2), "save_agent ran while the guard was held"
        finally:
            lock.release()
        assert done.wait(5), "save_agent never ran once the guard was released"
        worker.join(timeout=5)
        assert store.get_agent("cluster-a", "node-a") == agent
    finally:
        store.close()
