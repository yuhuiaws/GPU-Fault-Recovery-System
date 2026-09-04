"""Ratchet tests that assert on the source text of the code they test.

A test that greps ``inspect.getsource(...)`` for a call, or reads a module's
``.py`` file and looks for a substring, passes as long as the implementation
keeps spelling itself the same way. It does not run the behaviour, so it stays
green when the behaviour breaks and goes red on a pure rename. The remedy is a
call-sequence spy or an observable effect.

A handful of files legitimately audit static text -- shebangs, banned flags,
suite structure -- rather than behaviour. Those are listed in the baseline with
a reason and a site count, so they cannot grow unnoticed either.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
BASELINE = ROOT / "test-source-assertion-baseline.json"
SOURCE_READERS = frozenset(
    {"getsource", "getsourcelines", "getsourcefile", "findsource"}
)


class SourceTextCollector(ast.NodeVisitor):
    """Count the places a test turns implementation code into a string."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.sites = 0

    def visit_Call(self, node: ast.Call) -> None:
        if self._reads_source(node.func) or self._reads_a_python_file(node):
            self.sites += 1
        self.generic_visit(node)

    def _reads_source(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Attribute):
            return func.attr in SOURCE_READERS
        return isinstance(func, ast.Name) and func.id in SOURCE_READERS

    def _reads_a_python_file(self, node: ast.Call) -> bool:
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in {
            "read_text",
            "read_bytes",
        }:
            return False
        # The receiver is the path expression. Only a path that names a ``.py``
        # file is implementation source; reading a fixture or a manifest is not.
        receiver = ast.get_source_segment(self.text, func.value) or ""
        return any(literal.endswith(".py") for literal in _string_literals(receiver))


def _string_literals(expression: str) -> list[str]:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return []
    return [
        node.value
        for node in ast.walk(parsed)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def collect() -> Counter[str]:
    result: Counter[str] = Counter()
    for path in sorted(TESTS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        collector = SourceTextCollector(text)
        collector.visit(ast.parse(text))
        if collector.sites:
            result[path.relative_to(ROOT).as_posix()] = collector.sites
    return result


def failures(baseline: dict[str, dict], current: Counter[str]) -> list[str]:
    allowed = baseline.get("allowed") or {}
    result = []
    for relative in sorted(set(allowed) | set(current)):
        entry = allowed.get(relative)
        sites = current.get(relative, 0)
        if entry is None:
            result.append(
                f"test asserts on implementation source text: {relative} "
                f"({sites} sites); replace it with a call-sequence spy or an "
                "observable effect"
            )
        elif not entry.get("reason"):
            result.append(f"static-text audit needs a reason: {relative}")
        elif sites == 0:
            result.append(
                f"stale static-text audit baseline entry: {relative}; "
                "run --write-baseline"
            )
        elif sites > entry["sites"]:
            result.append(
                f"static-text audit grew: {relative} {entry['sites']} -> {sites}"
            )
        elif sites < entry["sites"]:
            result.append(
                f"static-text audit baseline has slack: {relative} "
                f"{entry['sites']} -> {sites}; run --write-baseline"
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="refresh the site counts of the already-allowed files",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="list every extraction site instead of gating",
    )
    args = parser.parse_args()
    current = collect()
    if args.report:
        for relative, sites in sorted(current.items(), key=lambda item: -item[1]):
            print(f"{sites:3d}  {relative}")
        print(f"total {sum(current.values())} sites in {len(current)} files")
        return 0
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if args.write_baseline:
        allowed = baseline.get("allowed") or {}
        unknown = sorted(set(current) - set(allowed))
        if unknown:
            print(
                "refusing to allow new source-text assertions: " + ", ".join(unknown),
                file=sys.stderr,
            )
            return 1
        for relative, entry in allowed.items():
            entry["sites"] = current.get(relative, 0)
        BASELINE.write_text(
            json.dumps({"allowed": allowed}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {BASELINE.relative_to(ROOT)}")
        return 0
    problems = failures(baseline, current)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(
        "test source-assertion check passed: "
        f"{sum(current.values())} audited static-text sites in {len(current)} files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
