from __future__ import annotations

import ast
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
DEPLOY = ROOT / "deploy"
SCRIPTS = ROOT / "scripts"
TOOLS = ROOT / "tools"
TESTS = ROOT / "tests"
SOURCE_ROOTS = (SOURCE, DEPLOY, SCRIPTS, TOOLS, TESTS)
BASELINE = ROOT / "architecture-baseline.json"
DEFAULT_FILE_LIMIT = 1500
DEFAULT_FUNCTION_LIMIT = 200
DEFAULT_CLASS_LIMIT = 800
LIMITS = {
    "files": DEFAULT_FILE_LIMIT,
    "functions": DEFAULT_FUNCTION_LIMIT,
    "classes": DEFAULT_CLASS_LIMIT,
}


class DefinitionCollector(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.stack: list[str] = []
        self.functions: dict[str, int] = {}
        self.classes: dict[str, int] = {}

    def _key(self, name: str) -> str:
        return f"{self.path}:{'.'.join([*self.stack, name])}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes[self._key(node.name)] = node.end_lineno - node.lineno + 1
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions[self._key(node.name)] = node.end_lineno - node.lineno + 1
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef


def module_name(path: Path) -> str:
    if path.is_relative_to(SOURCE):
        relative = path.relative_to(SOURCE).with_suffix("")
    else:
        relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(part.replace("-", "_") for part in parts)


class RuntimeImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.targets: list[str] = []

    def visit_If(self, node: ast.If) -> None:
        is_type_checking = (
            isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        ) or (
            isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING"
        )
        if is_type_checking:
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.targets.extend(item.name for item in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.targets.append(node.module)


def import_graph(paths: list[Path]) -> dict[str, set[str]]:
    modules = {module_name(path): path for path in paths}
    graph = {name: set() for name in modules}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        collector = RuntimeImportCollector()
        collector.visit(tree)
        for target in collector.targets:
            candidate = target
            while candidate:
                if candidate in modules and candidate != name:
                    graph[name].add(candidate)
                    break
                candidate = candidate.rpartition(".")[0]
    return graph


def strongly_connected(
    graph: dict[str, set[str]],
) -> list[list[str]]:
    index = 0
    indexes: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    result: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = index
        low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph[node]:
            if target not in indexes:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], indexes[target])
        if low[node] != indexes[node]:
            return
        component = []
        while True:
            target = stack.pop()
            on_stack.remove(target)
            component.append(target)
            if target == node:
                break
        if len(component) > 1:
            result.append(sorted(component))

    for node in sorted(graph):
        if node not in indexes:
            visit(node)
    return sorted(result)


def collect_architecture() -> tuple[
    dict[str, dict[str, int]],
    list[list[str]],
]:
    paths = sorted(
        path for source_root in SOURCE_ROOTS for path in source_root.rglob("*.py")
    )
    current_files: dict[str, int] = {}
    current_functions: dict[str, int] = {}
    current_classes: dict[str, int] = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        current_files[relative] = len(text.splitlines())
        collector = DefinitionCollector(relative)
        collector.visit(ast.parse(text))
        current_functions.update(collector.functions)
        current_classes.update(collector.classes)
    return (
        {
            "files": current_files,
            "functions": current_functions,
            "classes": current_classes,
        },
        strongly_connected(import_graph(paths)),
    )


def generated_baseline(
    current: dict[str, dict[str, int]],
    cycles: list[list[str]],
) -> dict:
    return {
        "classes": {
            key: size
            for key, size in sorted(current["classes"].items())
            if size > DEFAULT_CLASS_LIMIT
        },
        "cycles": cycles,
        "files": {
            key: size
            for key, size in sorted(current["files"].items())
            if size > DEFAULT_FILE_LIMIT
        },
        "functions": {
            key: size
            for key, size in sorted(current["functions"].items())
            if size > DEFAULT_FUNCTION_LIMIT
        },
    }


def baseline_failures(
    baseline: dict,
    current: dict[str, dict[str, int]],
    cycles: list[list[str]],
) -> list[str]:
    failures: list[str] = []
    for kind, default in LIMITS.items():
        sizes = current[kind]
        allowed = {key: int(value) for key, value in baseline.get(kind, {}).items()}
        current_keys = set(sizes)
        allowed_keys = set(allowed)
        for key in sorted(allowed_keys - current_keys):
            failures.append(f"stale {kind[:-1]} baseline entry: {key}")
        for key in sorted(current_keys - allowed_keys):
            size = sizes[key]
            if size > default:
                failures.append(
                    f"{kind[:-1]} {key} is {size} lines; "
                    f"default limit {default}; add it with "
                    "--write-baseline after review"
                )
        for key in sorted(current_keys & allowed_keys):
            size = sizes[key]
            limit = allowed[key]
            if size > limit:
                failures.append(f"{kind[:-1]} {key} grew from {limit} to {size} lines")
            elif size < limit:
                failures.append(
                    f"{kind[:-1]} {key} baseline has slack: "
                    f"current {size}, recorded {limit}; run "
                    "--write-baseline to tighten it"
                )
            elif size <= default:
                failures.append(
                    f"{kind[:-1]} {key} no longer needs a baseline "
                    f"entry at {size} lines; run --write-baseline"
                )

    current_cycles = {tuple(item) for item in cycles}
    allowed_cycles = {tuple(item) for item in baseline.get("cycles", [])}
    for cycle in sorted(current_cycles - allowed_cycles):
        failures.append("new import cycle: " + " -> ".join(cycle))
    for cycle in sorted(allowed_cycles - current_cycles):
        failures.append("stale grandfathered import cycle: " + " -> ".join(cycle))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="replace the baseline with current oversized definitions",
    )
    args = parser.parse_args(argv)
    current, cycles = collect_architecture()
    if args.write_baseline:
        BASELINE.write_text(
            json.dumps(
                generated_baseline(current, cycles),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {BASELINE.relative_to(ROOT)}")
        return 0

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    failures = baseline_failures(baseline, current, cycles)

    if failures:
        print("architecture check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(
        "architecture check passed: "
        f"{len(current['files'])} modules, "
        f"{len(cycles)} grandfathered cycles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
