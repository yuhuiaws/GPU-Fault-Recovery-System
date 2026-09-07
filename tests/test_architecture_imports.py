"""No compatibility shims inside the package (S20).

``gpu_fault.orchestrator`` and ``gpu_fault.runtime_adapters`` used to re-export
names that live in ``gpu_fault.orchestration`` and ``gpu_fault.adapters``. A
module under ``src/gpu_fault`` that imported the shim instead of the real
package put two import paths on one symbol and drew a false edge (for example
``execution -> orchestrator -> orchestration``) in the architecture import
graph that ``scripts/check-python-architecture.py`` reasons about. The shims
are gone; these tests keep them gone and keep every import spelled once. The
checker's own graph excludes ``TYPE_CHECKING`` imports; the AST walk below
does not, because a type-only import through a shim is still a second
spelling of the same dependency.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "gpu_fault"
architecture = lazy_script_module(ROOT / "scripts/check-python-architecture.py")

RETIRED_SHIMS = frozenset({"gpu_fault.orchestrator", "gpu_fault.runtime_adapters"})
SCANNED_ROOTS = ("src", "deploy", "scripts", "tools", "tests")


def _package_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        modules[architecture.module_name(path)] = path
    return modules


def _imported_modules(path: Path) -> set[str]:
    """Every module an ``import``/``from ... import`` names, guarded or not."""
    targets: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            targets.add(node.module)
    return targets


def test_retired_shim_modules_do_not_exist() -> None:
    modules = _package_modules()
    present = sorted(RETIRED_SHIMS & set(modules))
    assert present == [], f"compatibility shims must stay deleted: {present}"


def test_nothing_in_the_repository_imports_a_retired_shim() -> None:
    offenders: list[str] = []
    for root in SCANNED_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts or path == Path(__file__):
                continue
            try:
                imported = _imported_modules(path)
            except SyntaxError:
                continue
            offenders.extend(
                f"{path.relative_to(ROOT).as_posix()} imports {shim}"
                for shim in sorted(imported & RETIRED_SHIMS)
            )
    assert offenders == [], "\n".join(offenders)


def test_architecture_import_graph_has_no_shim_node() -> None:
    graph = architecture.import_graph(sorted(_package_modules().values()))
    assert not (RETIRED_SHIMS & set(graph)), "a shim is back in the import graph"
    edges = sorted(
        f"{name} -> {target}"
        for name, targets in graph.items()
        for target in targets & RETIRED_SHIMS
    )
    assert edges == [], "\n".join(edges)
