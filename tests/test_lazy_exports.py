"""One lazy-export hook for every package ``__init__`` (S21).

Fifteen package ``__init__.py`` files publish their names through an
``_EXPORTS = {"Name": ("module", "attribute")}`` table resolved on first
attribute access. The resolving ``__getattr__`` used to be pasted into each of
them, three of them in a one-name variant and three raising ``KeyError`` for
an unknown name, and none of them answered ``dir()``. ``gpu_fault.lazy_exports``
is the single copy: the same table, the same caching, ``AttributeError`` for
every unknown name, a ``__dir__`` that lists the lazy names, and an ``__all__``
derived from the table.

The tables themselves stay literal dict displays in each ``__init__.py``:
``scripts/check-lazy-exports.py`` and ``scripts/component_wheels.py`` read
them with ``ast`` and never import the package.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest

from gpu_fault.lazy_exports import lazy_module
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
GATE = lazy_script_module(ROOT / "scripts/check-lazy-exports.py")

LAZY_PACKAGES = frozenset(
    {
        "gpu_fault.adapters",
        "gpu_fault.adapters.node_action",
        "gpu_fault.adapters.hyperpod",
        "gpu_fault.adapters.kubernetes",
        "gpu_fault.processor",
        "gpu_fault.node_agent",
        "gpu_fault.collectors",
        "gpu_fault.collectors.cloud",
        "gpu_fault.collectors.gpu",
        "gpu_fault.collectors.logs",
        "gpu_fault.policy",
        "gpu_fault.execution",
        "gpu_fault.store",
        "gpu_fault.notifications",
        "gpu_fault.orchestration",
    }
)

TARGET_MODULE = "s21_lazy_target"


def _hooked_module(name: str, exports: dict[str, tuple[str, str]]) -> ModuleType:
    """A module wired exactly the way the package ``__init__`` files are."""
    module = ModuleType(name)
    namespace = vars(module)
    namespace["_EXPORTS"] = exports
    namespace["__getattr__"], namespace["__dir__"], namespace["__all__"] = lazy_module(
        namespace, exports
    )
    return module


def _target(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    target = ModuleType(TARGET_MODULE)
    target.value = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, TARGET_MODULE, target)
    return target


def test_unknown_name_raises_attribute_error_carrying_the_name() -> None:
    module = _hooked_module("s21_unknown", {"value": (TARGET_MODULE, "value")})

    with pytest.raises(AttributeError) as excinfo:
        module.absent

    assert excinfo.value.args == ("absent",)


def test_first_access_imports_the_target_and_caches_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target(monkeypatch)
    module = _hooked_module("s21_cache", {"value": (TARGET_MODULE, "value")})

    first = module.value

    assert first is target.value
    assert vars(module)["value"] is first
    # Once cached the name is an ordinary global: a second read must not go
    # back through import_module, which would now fail.
    monkeypatch.delitem(sys.modules, TARGET_MODULE)
    assert module.value is first


def test_dir_lists_lazy_names_next_to_the_module_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _target(monkeypatch)
    module = _hooked_module(
        "s21_dir",
        {"value": (TARGET_MODULE, "value"), "_hidden": (TARGET_MODULE, "value")},
    )

    listed = dir(module)

    assert "value" in listed
    assert "_hidden" in listed
    assert "_EXPORTS" in listed
    assert listed == sorted(listed)
    # dir() must not import anything: the names are known from the table.
    assert "value" not in vars(module)


def test_all_is_the_public_half_of_the_table() -> None:
    module = _hooked_module(
        "s21_all",
        {"value": (TARGET_MODULE, "value"), "_hidden": (TARGET_MODULE, "value")},
    )

    assert module.__all__ == ["value"]


def test_gate_discovers_every_lazy_package_table() -> None:
    tables = GATE.discover_export_tables(SOURCE)

    assert set(tables) == LAZY_PACKAGES
    assert all(tables[package] for package in LAZY_PACKAGES), (
        "every lazy package must have a non-empty table"
    )


@pytest.mark.parametrize("package", sorted(LAZY_PACKAGES))
def test_package_serves_and_lists_every_table_name(package: str) -> None:
    table = GATE.discover_export_tables(SOURCE)[package]
    module = import_module(package)

    listed = set(dir(module))
    for name, (target_module, attribute) in table.items():
        assert name in listed, f"{package}.{name} missing from dir()"
        value = getattr(module, name)
        assert value is getattr(import_module(target_module), attribute)
    assert set(module.__all__) == {name for name in table if not name.startswith("_")}
    with pytest.raises(AttributeError):
        getattr(module, "s21_no_such_export")
