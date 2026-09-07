"""Hold a declared set of documentation statements to the code they describe.

``check-doc-references.py`` proves that a path or symbol a document names
exists; ``check-doc-anchors.py`` proves a link lands. Neither asks whether a
sentence is still *true*. Two design documents carried a hand-counted total of
``tests/`` files that had drifted by a factor of two, and ``COLLECTORS.md``
said the application performs no bearer-token validation long after the
regional middleware started doing exactly that.

The table below is deliberately small: a fact earns a row when it is cheap to
recompute from the tree and expensive to be wrong about. Two kinds of rows:

* ``Forbidden`` -- a phrase that must not appear, because it is the shape of
  a claim that always goes stale (a literal file count).
* ``Statement`` -- a phrase the document must carry, paired with a predicate
  over the checkout that must hold. Removing the sentence is reported too,
  otherwise the guard silently stops guarding anything.

Predicates read source through ``ast`` rather than substring search so that a
comment or docstring mentioning a name cannot satisfy them.
"""

from __future__ import annotations

import argparse
import ast
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGN = "docs/详细设计.md"
DESIGN_V2 = "docs/详细设计-v2.md"
COLLECTORS = "COLLECTORS.md"
AUTH_MIDDLEWARE = "src/gpu_fault/app/middleware/auth.py"
APP_FACTORY = "src/gpu_fault/app/factory.py"
REGIONAL = "src/gpu_fault/regional.py"

Predicate = Callable[[Path], str | None]


@dataclass(frozen=True)
class Forbidden:
    """``pattern`` must not match anywhere in ``documents``."""

    id: str
    documents: tuple[str, ...]
    pattern: str
    reason: str


@dataclass(frozen=True)
class Statement:
    """``document`` must carry ``phrase`` and ``truth`` must hold for the tree.

    ``truth`` returns ``None`` when the code backs the statement, otherwise a
    short description of what contradicts it.
    """

    id: str
    document: str
    phrase: str
    truth: Predicate


Fact = Forbidden | Statement


def _parse(root: Path, relative: str) -> ast.Module | None:
    path = root / relative
    if not path.is_file():
        return None
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _calls(tree: ast.Module, owner: str, attribute: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
        for node in ast.walk(tree)
    )


def _has_string(tree: ast.Module, text: str) -> bool:
    return any(
        isinstance(node, ast.Constant) and node.value == text for node in ast.walk(tree)
    )


def _python_file_counts(tests: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in tests.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(tests)
        group = relative.parts[0] if len(relative.parts) > 1 else "."
        counts[group] = counts.get(group, 0) + 1
    return counts


def largest_test_directory_is_regional(root: Path) -> str | None:
    counts = _python_file_counts(root / "tests")
    if not counts:
        return "tests/ holds no Python files"
    largest = max(counts, key=lambda group: (counts[group], group))
    if largest != "regional":
        return (
            f"the largest test directory is tests/{largest} "
            f"({counts[largest]} files), not tests/regional"
        )
    return None


def cluster_bearer_auth_is_implemented(root: Path) -> str | None:
    """Missing bearer is refused (401) and tokens compare in constant time."""
    problems: list[str] = []
    factory = _parse(root, APP_FACTORY)
    if factory is None or not _has_string(
        factory, "regional cluster bearer token is required"
    ):
        problems.append(
            f"{APP_FACTORY} no longer refuses a missing bearer with "
            "'regional cluster bearer token is required'"
        )
    regional = _parse(root, REGIONAL)
    if regional is None or not _calls(regional, "secrets", "compare_digest"):
        problems.append(
            f"{REGIONAL} no longer compares cluster token digests with "
            "secrets.compare_digest"
        )
    middleware = _parse(root, AUTH_MIDDLEWARE)
    if middleware is None or not _calls(middleware, "secrets", "compare_digest"):
        problems.append(
            f"{AUTH_MIDDLEWARE} no longer compares the execution token with "
            "secrets.compare_digest"
        )
    return "; ".join(problems) or None


def default_deny_bucket_is_implemented(root: Path) -> str | None:
    middleware = _parse(root, AUTH_MIDDLEWARE)
    if middleware is None or not _has_string(
        middleware, "regional route has no declared authorization bucket"
    ):
        return (
            f"{AUTH_MIDDLEWARE} no longer refuses routes without a declared "
            "authorization bucket"
        )
    return None


FACTS: tuple[Fact, ...] = (
    Forbidden(
        id="test-file-count-not-literal",
        documents=(DESIGN, DESIGN_V2),
        pattern=r"\d+\s*个\s*Python\s*文件",
        reason=(
            "a literal tests/ file count goes stale on every added test; "
            "describe the structure instead"
        ),
    ),
    Forbidden(
        id="test-directory-count-not-literal",
        documents=(DESIGN,),
        pattern=r"`tests/[^`]*`\s*（\s*\d+\s*）",
        reason="per-directory test counts go stale; name the directories only",
    ),
    Forbidden(
        id="check-script-count-not-literal",
        documents=(DESIGN,),
        pattern=r"\d+\s*个\s*`scripts/check-\*\.py`",
        reason="the number of scripts/check-*.py gates changes; do not count them",
    ),
    Statement(
        id="largest-test-directory",
        document=DESIGN,
        phrase="`tests/regional/` 是其中最大的一组",
        truth=largest_test_directory_is_regional,
    ),
    Forbidden(
        id="collectors-no-stale-auth-claim",
        documents=(COLLECTORS,),
        pattern=r"尚未实现 bearer token 校验",
        reason=(
            "the regional middleware validates cluster bearer tokens; "
            "see src/gpu_fault/app/middleware/auth.py"
        ),
    ),
    Statement(
        id="collectors-bearer-auth",
        document=COLLECTORS,
        phrase="区域模式下控制面自身校验集群 bearer token",
        truth=cluster_bearer_auth_is_implemented,
    ),
    Statement(
        id="collectors-default-deny",
        document=COLLECTORS,
        phrase="未声明授权桶的路由默认拒绝",
        truth=default_deny_bucket_is_implemented,
    ),
)


def _read(root: Path, relative: str) -> str | None:
    path = root / relative
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _check_forbidden(root: Path, fact: Forbidden) -> list[str]:
    failures: list[str] = []
    pattern = re.compile(fact.pattern)
    for relative in fact.documents:
        text = _read(root, relative)
        if text is None:
            failures.append(f"{fact.id}: {relative}: document is missing")
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            match = pattern.search(line)
            if match is not None:
                failures.append(
                    f"{fact.id}: {relative}:{number}: stale phrasing "
                    f"{match.group(0)!r}; {fact.reason}"
                )
    return failures


def _check_statement(root: Path, fact: Statement) -> list[str]:
    text = _read(root, fact.document)
    if text is None:
        return [f"{fact.id}: {fact.document}: document is missing"]
    if fact.phrase not in text:
        return [
            f"{fact.id}: {fact.document}: statement not found: {fact.phrase!r}; "
            "the guarded sentence was removed or reworded -- update the fact "
            "table together with the document"
        ]
    problem = fact.truth(root)
    if problem is not None:
        return [
            f"{fact.id}: {fact.document} states {fact.phrase!r} but the code "
            f"no longer backs it: {problem}"
        ]
    return []


def check_facts(root: Path, facts: Iterable[Fact] = FACTS) -> list[str]:
    failures: list[str] = []
    for fact in facts:
        if isinstance(fact, Forbidden):
            failures.extend(_check_forbidden(root, fact))
        else:
            failures.extend(_check_statement(root, fact))
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--fact",
        action="append",
        dest="facts",
        metavar="ID",
        help="check only this fact id (repeatable; default: every fact)",
    )
    arguments = parser.parse_args(argv)
    known = {fact.id: fact for fact in FACTS}
    if arguments.facts:
        unknown = sorted(set(arguments.facts) - set(known))
        if unknown:
            parser.error(f"unknown fact id(s): {', '.join(unknown)}")
        selected: list[Fact] = [known[fact_id] for fact_id in arguments.facts]
    else:
        selected = list(FACTS)
    failures = check_facts(arguments.root.resolve(), selected)
    for failure in failures:
        print(failure)
    if failures:
        print(f"{len(failures)} documentation fact(s) out of step with the code")
        return 1
    print(f"{len(selected)} documentation fact(s) hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
