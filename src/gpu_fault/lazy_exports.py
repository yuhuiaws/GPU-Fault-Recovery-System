"""The one module-level ``__getattr__`` behind every lazy ``_EXPORTS`` table.

A package ``__init__.py`` that wants ``from gpu_fault.execution import
WorkflowDispatcher`` to work without importing the dispatcher (and everything
it drags in) at package import time declares the table and binds the hooks::

    _EXPORTS = {"WorkflowDispatcher": ("gpu_fault.execution.dispatcher",
                                       "WorkflowDispatcher")}

    __getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)

The table stays a literal dict display in the ``__init__.py`` because
``scripts/check-lazy-exports.py`` (resolves every tuple before deploy) and
``scripts/component_wheels.py`` (walks the tuples to pick wheel contents) read
it with ``ast`` rather than importing the package. Only the hooks live here.

The hooks are bound by name at module level, not injected into the namespace
from inside this function, so mypy sees the ``__getattr__`` and types a lazy
name as ``Any`` exactly as it did for the pasted ``def``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any

#: ``(__getattr__, __dir__, __all__)`` in the order a module binds them.
LazyModuleHooks = tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]


def lazy_module(
    namespace: dict[str, Any], exports: Mapping[str, tuple[str, str]]
) -> LazyModuleHooks:
    """Hooks that resolve ``exports`` into ``namespace`` on first access.

    ``namespace`` is the calling module's ``globals()``. A resolved value is
    stored there, so the interpreter finds it as an ordinary global on every
    later access and ``__getattr__`` runs once per name. An unknown name raises
    ``AttributeError(name)``, which is what ``hasattr``, ``getattr`` defaults
    and ``from package import missing`` all expect from a module.

    ``__dir__`` lists the table next to whatever the module already defines so
    ``dir()`` and tab completion show the lazy names without importing them.
    ``__all__`` is the public part of the table: a leading underscore keeps a
    name out of ``import *`` while ``__getattr__`` still serves it.
    """

    def __getattr__(name: str) -> Any:
        target = exports.get(name)
        if target is None:
            raise AttributeError(name)
        module_name, attribute = target
        value = getattr(import_module(module_name), attribute)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted(set(namespace) | set(exports))

    return (
        __getattr__,
        __dir__,
        [name for name in exports if not name.startswith("_")],
    )
