"""The release orchestrator is reached as a package, never as a directory.

The orchestrator modules used to import each other by bare name
(``from regional_release_config import ReleaseError``), which only works when
their directory is on ``sys.path``. That forced every consumer to put the
directory there: the test loader did it temporarily, and one e2e driver did it
for real and then ``importlib.import_module``-ed five modules by bare name. Each
such consumer is a second copy of the "how to find the orchestrator" rule, and
each one breaks silently the day the modules move.

These cases pin the import graph from the outside: no file in the repository
names an orchestrator module by its bare stem, and no file other than the test
loader puts the orchestrator directory on ``sys.path``. The package itself
reaches its siblings through absolute ``gpu_fault_release.*`` imports, so the
allowed set for bare names is empty.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_DIR = ROOT / "src/gpu_fault_release"
#: Where the orchestrator lives, as a repository-relative string. A consumer
#: that spells this out next to a ``sys.path`` call is re-deriving the import
#: rule by hand.
ORCHESTRATOR_RELATIVE = ORCHESTRATOR_DIR.relative_to(ROOT).as_posix()
#: Directories whose files may still import orchestrator modules by bare stem.
#: Empty: the package imports its own modules absolutely.
BARE_IMPORT_ALLOWED_DIRS: tuple[Path, ...] = ()
#: The only file allowed to put a script directory on ``sys.path``: it does so
#: for the duration of one ``exec_module`` and removes the entry afterwards.
SYS_PATH_ALLOWED_FILES = frozenset({"tests/_script_loader.py"})
SCANNED_ROOTS = ("deploy", "scripts", "src", "tests", "tools")


def orchestrator_stems() -> frozenset[str]:
    stems = frozenset(
        path.stem
        for path in ORCHESTRATOR_DIR.glob("*.py")
        if path.name != "__init__.py"
    )
    assert len(stems) >= 40, f"orchestrator directory looks wrong: {ORCHESTRATOR_DIR}"
    return stems


def scanned_files() -> list[Path]:
    result: list[Path] = []
    for root in SCANNED_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts or path == Path(__file__):
                continue
            result.append(path)
    return result


def _string_constant(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_import_module_call(node: ast.Call) -> bool:
    function = node.func
    if isinstance(function, ast.Attribute):
        return function.attr == "import_module"
    return isinstance(function, ast.Name) and function.id == "import_module"


def bare_orchestrator_imports(path: Path, stems: frozenset[str]) -> list[str]:
    """Every place ``path`` names an orchestrator module by bare stem.

    Covers ``import X``, ``from X import ...`` (absolute only; a relative
    import cannot leave its own package) and ``import_module("X")`` with a
    literal argument.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.partition(".")[0] in stems:
                    found.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                if node.module.partition(".")[0] in stems:
                    found.append(f"line {node.lineno}: from {node.module} import")
        elif isinstance(node, ast.Call) and _is_import_module_call(node):
            if node.args:
                target = _string_constant(node.args[0])
                if target is not None and target.partition(".")[0] in stems:
                    found.append(f"line {node.lineno}: import_module({target!r})")
    return found


def _mentions_sys_path_mutation(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute):
            continue
        if function.attr not in {"insert", "append"}:
            continue
        target = function.value
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "path"
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
        ):
            return True
    return False


def puts_orchestrator_directory_on_sys_path(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if not _mentions_sys_path_mutation(tree):
        return False
    return any(
        (constant := _string_constant(node)) is not None
        and constant.rstrip("/").endswith(ORCHESTRATOR_RELATIVE)
        for node in ast.walk(tree)
        if isinstance(node, ast.expr)
    )


def test_orchestrator_directory_is_a_package() -> None:
    assert (ORCHESTRATOR_DIR / "__init__.py").is_file(), (
        f"{ORCHESTRATOR_RELATIVE} must be an importable package, not a directory of "
        "scripts"
    )


def test_no_file_imports_an_orchestrator_module_by_bare_name() -> None:
    stems = orchestrator_stems()
    offenders: list[str] = []
    for path in scanned_files():
        if any(path.is_relative_to(allowed) for allowed in BARE_IMPORT_ALLOWED_DIRS):
            continue
        try:
            found = bare_orchestrator_imports(path, stems)
        except SyntaxError:
            continue
        relative = path.relative_to(ROOT).as_posix()
        offenders.extend(f"{relative} {item}" for item in found)
    assert offenders == [], (
        "orchestrator modules are imported as gpu_fault_release.<module>; these "
        "still use the bare stem:\n" + "\n".join(offenders)
    )


def test_only_the_script_loader_puts_the_orchestrator_directory_on_sys_path() -> None:
    offenders: list[str] = []
    for path in scanned_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative in SYS_PATH_ALLOWED_FILES:
            continue
        try:
            if puts_orchestrator_directory_on_sys_path(path):
                offenders.append(relative)
        except SyntaxError:
            continue
    assert offenders == [], (
        f"only {sorted(SYS_PATH_ALLOWED_FILES)} may put {ORCHESTRATOR_RELATIVE} on "
        "sys.path; import the package instead:\n" + "\n".join(offenders)
    )
