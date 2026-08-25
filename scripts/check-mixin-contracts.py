"""Require explicit annotations for composed Mixin state."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "gpu_fault"
BASELINE = ROOT / "mixin-contract-baseline.json"


def mixin_contracts() -> tuple[list[str], int, int]:
    missing: list[str] = []
    contract_attributes = 0
    any_annotations = 0
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not (
                node.name.endswith("Mixin")
                or any(
                    isinstance(child, ast.Expr)
                    and isinstance(child.value, ast.Constant)
                    and child.value.value
                    == ("Attributes supplied by the composed concrete implementation.")
                    for child in node.body
                )
            ):
                continue
            declared = {
                child.name
                for child in node.body
                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                )
            }
            annotations = {
                child.target.id: child.annotation
                for child in node.body
                if isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
            }
            declared.update(annotations)
            declared.update(
                target.id
                for child in node.body
                if isinstance(child, ast.Assign)
                for target in child.targets
                if isinstance(target, ast.Name)
            )
            assigned: set[str] = set()
            loaded: set[str] = set()
            for child in node.body:
                for item in ast.walk(child):
                    if not (
                        isinstance(item, ast.Attribute)
                        and isinstance(item.value, ast.Name)
                        and item.value.id == "self"
                    ):
                        continue
                    if isinstance(item.ctx, ast.Store):
                        assigned.add(item.attr)
                    elif isinstance(item.ctx, ast.Load):
                        loaded.add(item.attr)
            required = loaded - declared - assigned
            relative = path.relative_to(ROOT).as_posix()
            missing.extend(
                f"{relative}:{node.name}.{name}" for name in sorted(required)
            )
            contract_attributes += len(annotations)
            any_annotations += sum(
                isinstance(annotation, ast.Name) and annotation.id == "Any"
                for annotation in annotations.values()
            )
    return missing, contract_attributes, any_annotations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    missing, attributes, any_annotations = mixin_contracts()
    if missing:
        print(
            "mixin contract check failed; annotate these attributes:",
            file=sys.stderr,
        )
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        return 1
    current = {
        "contract_attributes": attributes,
        "any_annotations": any_annotations,
    }
    if args.write_baseline:
        BASELINE.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {BASELINE.name}")
        return 0
    if not BASELINE.is_file():
        print(
            f"{BASELINE.name} is missing; run --write-baseline",
            file=sys.stderr,
        )
        return 1
    expected = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = []
    if attributes < expected["contract_attributes"]:
        problems.append("contract attribute baseline has slack; run --write-baseline")
    elif attributes > expected["contract_attributes"]:
        problems.append(
            "new Mixin contract attributes were added without tightening their types"
        )
    if any_annotations < expected["any_annotations"]:
        problems.append("Any annotation baseline has slack; run --write-baseline")
    elif any_annotations > expected["any_annotations"]:
        problems.append("Mixin Any annotations increased")
    if problems:
        for problem in problems:
            print(
                f"mixin contract check failed: {problem}",
                file=sys.stderr,
            )
        return 1
    print(
        f"mixin contract check passed: {attributes} attributes, {any_annotations} Any"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
