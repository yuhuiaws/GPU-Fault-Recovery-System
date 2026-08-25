"""Ratchet tests that reach through public boundaries into private members."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
BASELINE = ROOT / "test-private-coupling-baseline.json"


class PrivateAccessCollector(ast.NodeVisitor):
    def __init__(self, relative: str) -> None:
        self.relative = relative
        self.accesses: Counter[str] = Counter()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr.startswith("_")
            and not node.attr.startswith("__")
            and isinstance(node.value, ast.Name)
            and node.value.id not in {"self", "cls"}
        ):
            key = f"{self.relative}:{node.value.id}.{node.attr}"
            self.accesses[key] += 1
        self.generic_visit(node)


def collect() -> Counter[str]:
    result: Counter[str] = Counter()
    for path in sorted(TESTS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        collector = PrivateAccessCollector(relative)
        collector.visit(ast.parse(path.read_text(encoding="utf-8")))
        result.update(collector.accesses)
    return result


def failures(
    baseline: dict[str, int],
    current: Counter[str],
) -> list[str]:
    result = []
    for key in sorted(set(baseline) | set(current)):
        expected = baseline.get(key)
        actual = current.get(key, 0)
        if expected is None:
            result.append(f"new private test coupling: {key} ({actual})")
        elif actual == 0:
            result.append(f"stale private coupling baseline entry: {key}")
        elif actual > expected:
            result.append(f"private test coupling grew: {key} {expected} -> {actual}")
        elif actual < expected:
            result.append(
                f"private coupling baseline has slack: {key} "
                f"{expected} -> {actual}; run --write-baseline"
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    current = collect()
    if args.write_baseline:
        BASELINE.write_text(
            json.dumps(dict(sorted(current.items())), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {BASELINE.relative_to(ROOT)}")
        return 0
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = failures(baseline, current)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(
        "private test coupling check passed: "
        f"{len(current)} symbols, {sum(current.values())} accesses"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
