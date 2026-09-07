"""Resolve every lazy ``_EXPORTS`` tuple and every manifest-declared entry point.

Fifteen package ``__init__.py`` files under ``src/gpu_fault/`` publish their
public names through ``_EXPORTS = {"Name": ("module", "attribute")}`` and a
module-level ``__getattr__``. The tuple is only executed when a caller asks
for the name, so a misspelled attribute or a function that moved survives
``compileall``, mypy and every import-time test, and fails as ``AttributeError``
in the first process that needs it.

The Lambda template is the same defect one level up: ``Handler:`` names a
dotted Python path that the runtime resolves at cold start. ``yamllint``
proves the file is YAML; nothing proved the handler existed.

The tables are found by parsing the ``__init__.py`` with ``ast``, never by
importing the package, so a package that fails at import is reported as a
resolution failure rather than aborting the scan. Manifest kinds are an
explicit list of ``(glob, extractor)`` pairs so a second manifest format can
be added without touching the resolution code.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import yaml  # type: ignore[import-untyped,unused-ignore]

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src"
ExportTable = dict[str, tuple[str, str]]
LAMBDA_FUNCTION_TYPES = frozenset(
    {"AWS::Lambda::Function", "AWS::Serverless::Function"}
)


@dataclass(frozen=True)
class ManifestScan:
    """One manifest kind: the files it lives in and how to read entry points."""

    glob: str
    extract: Callable[[Path], list[str]]


def _walk(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def cloudformation_python_handlers(path: Path) -> list[str]:
    """``Handler`` of every Python-runtime Lambda function in a CFN template.

    ``BaseLoader`` rather than ``safe_load``: the templates carry ``!GetAtt``
    and ``!Sub`` tags, which the safe loader rejects outright. Non-Python
    runtimes are skipped because their handler strings are not module paths.
    """
    handlers: list[str] = []
    for document in yaml.load_all(
        path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    ):
        for mapping in _walk(document):
            if mapping.get("Type") not in LAMBDA_FUNCTION_TYPES:
                continue
            properties = mapping.get("Properties")
            if not isinstance(properties, dict):
                continue
            runtime = str(properties.get("Runtime", ""))
            handler = properties.get("Handler")
            if runtime.startswith("python") and isinstance(handler, str):
                handlers.append(handler)
    return handlers


MANIFEST_SCANS: tuple[ManifestScan, ...] = (
    ManifestScan("deploy/aws/lambda/*.yaml", cloudformation_python_handlers),
)


def _export_table(tree: ast.Module, path: Path) -> ExportTable | None:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "_EXPORTS"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError as exc:
            raise ValueError(f"{path}: _EXPORTS must be a literal dict") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}: _EXPORTS must be a dict")
        table: ExportTable = {}
        for name, target in value.items():
            if not (
                isinstance(name, str)
                and isinstance(target, tuple)
                and len(target) == 2
                and all(isinstance(part, str) for part in target)
            ):
                raise ValueError(
                    f"{path}: _EXPORTS[{name!r}] must be a (module, attribute) "
                    "tuple of strings"
                )
            table[name] = (target[0], target[1])
        return table
    return None


def discover_export_tables(source_root: Path) -> dict[str, ExportTable]:
    """Every ``_EXPORTS`` table under ``source_root``, keyed by package name.

    Found by parsing, not importing: the point of the gate is to run before
    anything has proven the package imports at all.
    """
    tables: dict[str, ExportTable] = {}
    for init in sorted(source_root.rglob("__init__.py")):
        tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
        table = _export_table(tree, init)
        if table is None:
            continue
        package = ".".join(init.parent.relative_to(source_root).parts)
        tables[package] = table
    return tables


def _resolve(module_name: str, attribute: str) -> str | None:
    """Why ``module_name.attribute`` does not resolve, or ``None`` when it does."""
    try:
        module = import_module(module_name)
    except Exception as exc:  # an import can raise anything; the gate reports it
        return f"cannot import {module_name}: {exc!r}"
    try:
        getattr(module, attribute)
    except AttributeError:
        return f"{module_name} has no attribute {attribute!r}"
    except Exception as exc:  # a lazy __getattr__ that raises KeyError, say
        return f"{module_name}.{attribute} raised {exc!r} on lookup"
    return None


def lazy_export_failures(source_root: Path) -> list[str]:
    failures: list[str] = []
    for package, table in discover_export_tables(source_root).items():
        for name, (module_name, attribute) in table.items():
            problem = _resolve(module_name, attribute)
            if problem is not None:
                failures.append(f"{package}.{name}: {problem}")
    return failures


def manifest_entry_point_failures(
    root: Path, scans: Sequence[ManifestScan] = MANIFEST_SCANS
) -> list[str]:
    failures: list[str] = []
    for scan in scans:
        for path in sorted(root.glob(scan.glob)):
            relative = path.relative_to(root).as_posix()
            for entry in scan.extract(path):
                module_name, _, attribute = entry.rpartition(".")
                if not module_name or not attribute:
                    failures.append(
                        f"{relative}: entry point {entry!r} is not module.attribute"
                    )
                    continue
                problem = _resolve(module_name, attribute)
                if problem is not None:
                    failures.append(f"{relative}: entry point {entry!r}: {problem}")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root holding src/ and the manifests (default: this repo)",
    )
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    # The checkout under test must win over an installed copy of the package.
    sys.path.insert(0, str(root / SOURCE))
    failures = lazy_export_failures(root / SOURCE)
    failures.extend(manifest_entry_point_failures(root))
    for failure in failures:
        print(failure)
    if failures:
        print(f"{len(failures)} unresolvable lazy export(s) or entry point(s)")
        return 1
    print("lazy exports and manifest entry points resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
