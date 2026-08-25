from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path


@dataclass(frozen=True)
class FileSize:
    path: str
    lines: int


@dataclass(frozen=True)
class FunctionSize:
    path: str
    line: int
    name: str
    lines: int


class FunctionCollector(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.scope: list[str] = []
        self.functions: list[FunctionSize] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        self._record(node)

    def _record(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        name = ".".join([*self.scope, node.name])
        self.functions.append(
            FunctionSize(
                path=self.path.as_posix(),
                line=node.lineno,
                name=name,
                lines=node.end_lineno - node.lineno + 1,
            )
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def collect(root: Path) -> tuple[list[FileSize], list[FunctionSize]]:
    files = []
    functions = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        files.append(
            FileSize(
                path=path.as_posix(),
                lines=len(text.splitlines()),
            )
        )
        collector = FunctionCollector(path)
        collector.visit(ast.parse(text, filename=str(path)))
        functions.extend(collector.functions)
    return files, functions


def area_summary(files: list[FileSize], root: Path) -> list[dict]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item in files:
        relative = Path(item.path).relative_to(root)
        area = relative.parts[0] if len(relative.parts) > 1 else relative.name
        totals[area][0] += item.lines
        totals[area][1] += 1
    return [
        {"area": area, "lines": values[0], "files": values[1]}
        for area, values in sorted(
            totals.items(),
            key=lambda item: (-item[1][0], item[0]),
        )
    ]


def markdown(
    payload: dict,
    file_limit: int,
    function_limit: int,
) -> str:
    summary = payload["summary"]
    lines = [
        f"# Python Implementation Size Audit - {date.today().isoformat()}",
        "",
        "- Generator: `scripts/audit-code-size.py`",
        f"- Root: `{payload['root']}`",
        f"- Files: {summary['files']:,}",
        f"- Physical lines: {summary['physical_lines']:,}",
        f"- Functions/methods: {summary['functions']:,}",
        "",
        f"## Files over {file_limit:,} lines",
        "",
        "| Lines | File |",
        "|---:|---|",
    ]
    lines.extend(
        f"| {item['lines']:,} | `{item['path']}` |"
        for item in payload["files_over_limit"]
    )
    lines.extend(
        [
            "",
            f"## Functions over {function_limit:,} lines",
            "",
            "| Lines | Function |",
            "|---:|---|",
        ]
    )
    lines.extend(
        "| {lines:,} | `{path}:{line} {name}` |".format(**item)
        for item in payload["functions_over_limit"]
    )
    lines.extend(
        [
            "",
            "## Area distribution",
            "",
            "| Lines | Files | Area |",
            "|---:|---:|---|",
        ]
    )
    lines.extend(
        "| {lines:,} | {files:,} | `{area}` |".format(**item)
        for item in payload["areas"]
    )
    lines.extend(
        [
            "",
            "## All implementation files",
            "",
            "| Lines | File |",
            "|---:|---|",
        ]
    )
    lines.extend(
        f"| {item['lines']:,} | `{item['path']}` |" for item in payload["files"]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="src/gpu_fault")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--file-limit", type=int, default=2000)
    parser.add_argument("--function-limit", type=int, default=300)
    args = parser.parse_args()

    root = Path(args.root)
    files, functions = collect(root)
    files_desc = sorted(files, key=lambda item: (-item.lines, item.path))
    functions_desc = sorted(
        functions,
        key=lambda item: (-item.lines, item.path, item.line),
    )
    payload = {
        "schema_version": 1,
        "generator": "scripts/audit-code-size.py",
        "generated_on": date.today().isoformat(),
        "root": root.as_posix(),
        "limits": {
            "file_lines": args.file_limit,
            "function_lines": args.function_limit,
        },
        "summary": {
            "files": len(files),
            "physical_lines": sum(item.lines for item in files),
            "functions": len(functions),
        },
        "files_over_limit": [
            asdict(item) for item in files_desc if item.lines > args.file_limit
        ],
        "functions_over_limit": [
            asdict(item) for item in functions_desc if item.lines > args.function_limit
        ],
        "areas": area_summary(files, root),
        "files": [asdict(item) for item in files_desc],
    }
    Path(args.json_output).write_text(json.dumps(payload, indent=2) + "\n")
    Path(args.markdown_output).write_text(
        markdown(payload, args.file_limit, args.function_limit)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
