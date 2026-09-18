"""Structural credential-flow checks for Python and embedded CPU probes."""

from __future__ import annotations

import ast
from textwrap import dedent

ENV_URL = "GPU_FAULT_STORE_URL"
ENV_FILE = "GPU_FAULT_STORE_URL_FILE"


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}"
    return ""


def _environment_key(node: ast.AST) -> str | None:
    key: ast.AST | None = None
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        if _name(node.value) == "os.environ":
            key = node.slice
    elif (
        isinstance(node, ast.Call)
        and _name(node.func) in {"os.getenv", "os.environ.get"}
        and node.args
    ):
        key = node.args[0]
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def _scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            break
    return node


def _file_source(
    node: ast.AST, parents: dict[ast.AST, ast.AST], seen: frozenset[str] = frozenset()
) -> bool:
    if _environment_key(node) == ENV_FILE:
        return True
    if isinstance(node, ast.Name):
        if node.id in seen:
            return False
        scope = _scope(node, parents)
        assignments = [
            entry.value
            for entry in ast.walk(scope)
            if isinstance(entry, ast.Assign)
            and _scope(entry, parents) is scope
            and entry.lineno < node.lineno
            and any(
                isinstance(target, ast.Name) and target.id == node.id
                for target in entry.targets
            )
        ]
        return len(assignments) == 1 and _file_source(
            assignments[0], parents, seen | {node.id}
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "strip" and not node.args and not node.keywords:
            return _file_source(node.func.value, parents, seen)
    return False


def _file_read(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    return any(
        isinstance(entry, ast.Call)
        and isinstance(entry.func, ast.Attribute)
        and entry.func.attr == "read_text"
        and isinstance(entry.func.value, ast.Call)
        and _name(entry.func.value.func) in {"Path", "pathlib.Path"}
        and bool(entry.func.value.args)
        and _file_source(entry.func.value.args[0], parents)
        for entry in ast.walk(node)
    )


def _credential_flow(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.Call):
            if _name(current.func) == "StoreCredentials" and current.args:
                path = next(
                    (item.value for item in current.keywords if item.arg == "path"),
                    None,
                )
                if node in ast.walk(current.args[0]) and path is not None:
                    return _file_source(path, parents)
            # This discarded validator checks the startup URL's target identity.
            # Its output cannot supply a SQL connection's credentials.
            if _name(current.func) == "dsn_arguments" and isinstance(
                parents.get(current), ast.Expr
            ):
                return True
        if isinstance(current, ast.IfExp) and node in ast.walk(current.orelse):
            return _file_source(current.test, parents) and _file_read(
                current.body, parents
            )
        if isinstance(current, ast.stmt):
            break
    return False


def _function_body(node: ast.FunctionDef) -> str:
    return ast.dump(
        ast.Module(
            body=[
                entry
                for entry in node.body
                if not isinstance(entry, (ast.Import, ast.ImportFrom))
                and not (
                    isinstance(entry, ast.Expr)
                    and isinstance(entry.value, ast.Constant)
                    and isinstance(entry.value.value, str)
                )
            ],
            type_ignores=[],
        )
    )


def scan_source(
    source: str, label: str, *, canonical_reader: str
) -> tuple[list[str], list[str]]:
    tree = ast.parse(source)
    canonical = ast.parse(canonical_reader).body[0]
    if not isinstance(canonical, ast.FunctionDef):
        raise ValueError("the canonical DSN reader is not a function")
    units = [(label, tree)]
    outside: list[str] = []
    incomplete: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and ENV_URL in node.value
        ):
            try:
                embedded = ast.parse(dedent(node.value))
            except SyntaxError:
                if "os.environ[" in node.value or "os.getenv(" in node.value:
                    incomplete.append(f"{label}:{node.lineno}:unparsed SQL probe")
            else:
                units.append((f"{label}:{node.lineno}:embedded", embedded))
    for unit_label, unit in units:
        parents = {
            child: parent
            for parent in ast.walk(unit)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(unit):
            if _environment_key(node) != ENV_URL:
                continue
            scope = _scope(node, parents)
            location = f"{unit_label}:{node.lineno}"
            if isinstance(scope, ast.FunctionDef) and scope.name == "store_dsn":
                if _function_body(scope) != _function_body(canonical):
                    incomplete.append(location)
            elif not _credential_flow(node, parents):
                outside.append(location)
    return outside, incomplete
