"""Ratchet diagnostic messages on bare boolean test assertions."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
BASELINE = ROOT / "assert-message-baseline.json"


def is_bare_boolean(test: ast.expr) -> bool:
    if isinstance(test, (ast.Call, ast.Attribute, ast.Name)):
        return True
    return (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and isinstance(test.operand, (ast.Call, ast.Attribute, ast.Name))
    )


def missing_messages() -> tuple[int, dict[str, int]]:
    total = 0
    by_file: dict[str, int] = {}
    for path in sorted(TESTS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        count = sum(
            isinstance(node, ast.Assert)
            and node.msg is None
            and is_bare_boolean(node.test)
            for node in ast.walk(tree)
        )
        if count:
            relative = path.relative_to(ROOT).as_posix()
            by_file[relative] = count
            total += count
    return total, by_file


def failures(
    baseline: dict[str, int],
    current: int,
) -> list[str]:
    recorded = baseline["missing_messages"]
    initial = baseline["initial_missing_messages"]
    target = int(initial * 0.60)
    result = []
    if current > recorded:
        result.append(
            f"bare boolean asserts without messages grew from {recorded} to {current}"
        )
    elif current < recorded:
        result.append(
            "assert message baseline has slack: "
            f"current {current}, recorded {recorded}; run --write-baseline"
        )
    if current > target:
        result.append(
            f"bare boolean assert target exceeded: {current} > {target} "
            f"(60% of initial {initial})"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    arguments = parser.parse_args()
    current, by_file = missing_messages()
    if arguments.write_baseline:
        initial = current
        if BASELINE.is_file():
            initial = json.loads(BASELINE.read_text(encoding="utf-8")).get(
                "initial_missing_messages", current
            )
        BASELINE.write_text(
            json.dumps(
                {
                    "initial_missing_messages": initial,
                    "missing_messages": current,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"wrote {BASELINE.name}: {current} missing messages "
            f"across {len(by_file)} files"
        )
        return 0
    if not BASELINE.is_file():
        print(
            f"{BASELINE.name} is missing; run --write-baseline",
        )
        return 1
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = failures(baseline, current)
    if problems:
        print("assert message check failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(
        "assert message check passed: "
        f"{current} bare boolean asserts without messages "
        f"across {len(by_file)} files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
