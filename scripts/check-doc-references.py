from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.doc_inventory import (  # noqa: E402
    DOCS,
    PRIVATE_DIRECTORIES,
    PRIVATE_FILES,
    ROOT,
    URL,
    documents,
    is_private,
    mask_urls,
)

__all__ = [
    "DOCS",
    "PRIVATE_DIRECTORIES",
    "PRIVATE_FILES",
    "ROOT",
    "URL",
    "check_references",
    "documents",
    "is_private",
    "mask_urls",
]

DISALLOWED_CURRENT_TARGETS = {
    "src/gpu_fault/api.py": "compatibility shim; reference gpu_fault.app",
}
LEGACY_MONOLITH = re.compile(
    r"(?<![/A-Za-z0-9_])"
    r"(?:api|store|node_agent|collectors|orchestrator|policy|execution|"
    r"processor|notifications)\.py\b"
)
LEGACY_API_ATTRIBUTE = re.compile(r"\bapi\._[A-Za-z_][A-Za-z0-9_]*")

REFERENCE = re.compile(
    r"(?P<path>(?:src|tests|deploy|scripts|tools)/"
    r"[A-Za-z0-9_./-]+\.(?:py|sh|ya?ml))"
    r"(?:(?:::(?P<symbol>[A-Za-z_][A-Za-z0-9_.]*))|"
    r"(?::(?P<start>[0-9]+)(?:-(?P<end>[0-9]+))?))?"
)


@dataclass(frozen=True)
class Violation:
    document: Path
    line: int
    reference: str
    reason: str


def _python_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.add(prefix + node.name)
            elif isinstance(node, ast.ClassDef):
                name = prefix + node.name
                result.add(name)
                visit(node.body, name + ".")
            elif (
                not prefix
                and path.is_relative_to(ROOT / "tests")
                and isinstance(node, ast.ImportFrom)
            ):
                result.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if (alias.asname or alias.name).startswith("test_")
                )

    visit(tree.body)
    return result


def check_references() -> list[Violation]:
    violations: list[Violation] = []
    line_counts: dict[Path, int] = {}
    symbols: dict[Path, set[str]] = {}

    for document in documents():
        for line_number, raw_line in enumerate(
            document.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            line = mask_urls(raw_line)
            if "按符号名定位" in line:
                violations.append(
                    Violation(
                        document,
                        line_number,
                        "按符号名定位",
                        "placeholder text is not a resolvable reference",
                    )
                )
            for match in LEGACY_MONOLITH.finditer(line):
                violations.append(
                    Violation(
                        document,
                        line_number,
                        match.group(0),
                        "legacy monolith name; reference the current package or symbol",
                    )
                )
            for match in LEGACY_API_ATTRIBUTE.finditer(line):
                violations.append(
                    Violation(
                        document,
                        line_number,
                        match.group(0),
                        "legacy api attribute; reference gpu_fault.app symbol",
                    )
                )
            for match in REFERENCE.finditer(line):
                reference = match.group(0)
                path_text = match.group("path")
                if "/.../" in path_text:
                    continue
                if path_text in DISALLOWED_CURRENT_TARGETS:
                    violations.append(
                        Violation(
                            document,
                            line_number,
                            reference,
                            DISALLOWED_CURRENT_TARGETS[path_text],
                        )
                    )
                    continue
                target = ROOT / path_text
                if not target.is_file():
                    violations.append(
                        Violation(
                            document,
                            line_number,
                            reference,
                            "target file does not exist",
                        )
                    )
                    continue

                start_text = match.group("start")
                if start_text is not None:
                    if target.suffix == ".py":
                        violations.append(
                            Violation(
                                document,
                                line_number,
                                reference,
                                "Python references must use ::symbol, not line numbers",
                            )
                        )
                        continue
                    count = line_counts.setdefault(
                        target,
                        len(target.read_text(encoding="utf-8").splitlines()),
                    )
                    start = int(start_text)
                    end = int(match.group("end") or start)
                    if start < 1 or end < start or end > count:
                        violations.append(
                            Violation(
                                document,
                                line_number,
                                reference,
                                f"line range is outside 1..{count}",
                            )
                        )

                symbol = match.group("symbol")
                if symbol is not None:
                    if target.suffix != ".py":
                        violations.append(
                            Violation(
                                document,
                                line_number,
                                reference,
                                "symbol locators are supported only for Python",
                            )
                        )
                        continue
                    available = symbols.setdefault(
                        target,
                        _python_symbols(target),
                    )
                    if symbol not in available:
                        violations.append(
                            Violation(
                                document,
                                line_number,
                                reference,
                                "Python symbol does not exist",
                            )
                        )
    return violations


def main() -> None:
    violations = check_references()
    if violations:
        for item in violations:
            document = item.document.relative_to(ROOT)
            print(f"{document}:{item.line}: {item.reference}: {item.reason}")
        raise SystemExit(1)
    print("Documentation code/test references are valid.")


if __name__ == "__main__":
    main()
