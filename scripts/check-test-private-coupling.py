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

# Dunders are otherwise skipped, because ``self.__class__``, ``obj.__dict__``
# and ``func.__name__`` are everywhere and say nothing about coupling. These
# three do say something, and what they say is worse than ``module._helper``:
# they reach into a callee's own namespace instead of merely reading a private
# name, so the scan below would rate them as no coupling at all.
#
#   * ``__globals__`` swaps a module-level name out from under a function. The
#     test keeps passing after the function stops looking that name up, because
#     nothing about the patch is tied to the call the function actually makes.
#     ``tests/_script_loader.py`` exists partly to get tests off this.
#   * ``__wrapped__`` runs the undecorated function, i.e. not the one that ships.
#   * ``__code__`` replaces what runs outright.
#
# Each is a legitimate tool in a couple of places, so they are ratcheted like
# every other private access rather than banned.
NAMESPACE_BYPASS_DUNDERS = frozenset({"__globals__", "__wrapped__", "__code__"})


def receiver_name(node: ast.expr) -> str:
    """Last dotted segment of the expression an attribute is read from.

    Keys stay in the same ``owner.attribute`` shape as the private-name scan.
    ``rollout.run_fleet_waves.__globals__`` and
    ``module.run_fleet_waves.__globals__`` are the same coupling seen through two
    handles, so both count against one entry.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return "<expression>"


class PrivateAccessCollector(ast.NodeVisitor):
    def __init__(self, relative: str) -> None:
        self.relative = relative
        self.accesses: Counter[str] = Counter()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in NAMESPACE_BYPASS_DUNDERS:
            owner = receiver_name(node.value)
            self.accesses[f"{self.relative}:{owner}.{node.attr}"] += 1
        elif (
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
