from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GITIGNORE = ROOT / ".gitignore"
DISALLOWED_CURRENT_TARGETS = {
    "src/gpu_fault/api.py": "compatibility shim; reference gpu_fault.app",
    "src/gpu_fault/orchestrator.py": (
        "compatibility shim; reference gpu_fault.orchestration or planner"
    ),
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
# 外链里的路径不是本仓引用。上游仓库正好也有 ``src/...`` 布局：
# CLOUD_PROVIDER_SOURCE_REVIEW.md 的 cluster-health-scanner 链接里有
# ``src/gpu_healthcheck/gpu_healthcheck.py``，HYPERPOD_SOURCE_REVIEW.md 的
# aws-do-hyperpod 链接里有 ``src/manifests/health-monitoring-agent.yaml``，
# 两者在本仓都不存在也不该存在。同理 ``policy.py`` 这类名字出现在上游
# URL 里也不是本仓的遗留命名。
URL = re.compile(r"<?https?://[^\s)>\]]+")


def private_markdown_rules() -> tuple[frozenset[str], tuple[str, ...]]:
    """Markdown that ``.gitignore`` keeps out of the public repository.

    豁免面必须等于「不进公开仓库的那批文件」，所以直接从 ``.gitignore``
    派生，而不是手抄一份名单。手抄的那份已经在漂移：``IMPLEMENTATION.md``
    等 6 个名字被移进 ``internal-docs/`` 之后，根目录的豁免条目就成了死条目
    ——哪天有人在根目录新建同名公开文档，它会被静默豁免，而这条豁免正是
    ``LEGACY_MONOLITH`` 这类硬错误的开关。

    只解析带 ``/`` 前缀的锚定行：以 ``/`` 结尾的按目录前缀匹配，以 ``.md``
    结尾的按路径精确匹配。其余条目（``__pycache__/``、``artifacts/*`` 等）
    和 Markdown 扫描面无关。
    """
    files: set[str] = set()
    directories: list[str] = []
    for raw in GITIGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("/"):
            continue
        entry = line[1:]
        if entry.endswith("/"):
            directories.append(entry)
        elif entry.endswith(".md"):
            files.add(entry)
    return frozenset(files), tuple(sorted(directories))


PRIVATE_FILES, PRIVATE_DIRECTORIES = private_markdown_rules()


def is_private(relative: str) -> bool:
    return relative in PRIVATE_FILES or any(
        relative.startswith(prefix) for prefix in PRIVATE_DIRECTORIES
    )


def documents() -> list[Path]:
    """Every reviewed Markdown file, not just the ones under docs/.

    The blocklists above also apply to root-level public Markdown such as
    ``README.md``; a ``docs/``-only scan would miss those references.
    """
    return [
        path
        for path in sorted(ROOT.glob("*.md")) + sorted(DOCS.rglob("*.md"))
        if not is_private(path.relative_to(ROOT).as_posix())
    ]


def mask_urls(line: str) -> str:
    """Blank out URL spans, keeping offsets so reported columns stay usable."""
    return URL.sub(lambda match: " " * len(match.group(0)), line)


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
