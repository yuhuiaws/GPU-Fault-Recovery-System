"""Apply the local artifacts retention policy."""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_DOCS = ROOT / "docs"
# 文档引用一轮压测有两种写法，只认第一种会让「被文档引用的轮次不删」这条
# 保护在真实仓库上恒为空：
#   1. 直接写原始路径 ``artifacts/perf/<case>/<release>/<utc>/...``；
#   2. 引用人工提升到证据库后的扁平副本
#      ``docs/evidence/perf/<case>-<utc>/...``，这是 docs/ 里实际存在的形态
#      （提升目前没有脚本，靠 artifacts/README.md「Retention」一节的约定）。
# 第二种写法里 case 与 release 已经被拼进一个目录名，无法反解，但结尾的 UTC
# 戳在一轮压测里唯一，可以直接用它保护对应的原始目录。
RAW_REFERENCE = re.compile(r"artifacts/perf/([^\s`)'\"<>]+)")
PROMOTED_REFERENCE = re.compile(
    r"docs/evidence/perf/[^\s`)'\"<>]*?(\d{8}T\d{6}Z)",
)
UTC_NAME = re.compile(r"^(\d{8}T\d{6}Z)(?:-\d+)?$")


@dataclass(frozen=True)
class References:
    """Run directories that documentation asks us to keep."""

    patterns: set[str] = field(default_factory=set)
    stamps: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.patterns) + len(self.stamps)


def referenced_patterns(docs: Path) -> References:
    patterns: set[str] = set()
    stamps: set[str] = set()
    if not docs.is_dir():
        return References(patterns, stamps)
    for path in docs.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for match in RAW_REFERENCE.finditer(text):
            value = match.group(1).rstrip(".,;:")
            if "<" not in value:
                patterns.add(value)
        stamps.update(match.group(1) for match in PROMOTED_REFERENCE.finditer(text))
    return References(patterns, stamps)


def is_referenced(relative: str, references: References) -> bool:
    leaf = UTC_NAME.fullmatch(relative.rsplit("/", 1)[-1])
    if leaf is not None and leaf.group(1) in references.stamps:
        return True
    return any(
        fnmatch.fnmatch(relative, pattern)
        or relative.startswith(pattern.rstrip("/") + "/")
        or pattern.startswith(relative.rstrip("/") + "/")
        for pattern in references.patterns
    )


def run_time(path: Path) -> datetime:
    match = UTC_NAME.fullmatch(path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


@dataclass(frozen=True)
class Report:
    """What the policy would delete, and what documentation saved."""

    candidates: list[Path]
    protected: list[Path]


def retention_report(
    artifacts: Path,
    docs: Path,
    *,
    keep: int,
    aborted_days: int,
    now: datetime,
) -> Report:
    perf = artifacts / "perf"
    references = referenced_patterns(docs)
    candidates: list[Path] = []
    protected: list[Path] = []
    # ``artifacts/`` 整个目录都在 .gitignore 里，新克隆和 CI 上根本不存在。
    # 原来这里直接 ``perf.iterdir()``，加入 Makefile 入口后会变成
    # FileNotFoundError 而不是「没有可删除的轮次」。
    if not perf.is_dir():
        return Report(candidates, protected)

    def classify(run: Path) -> None:
        if is_referenced(run.relative_to(perf).as_posix(), references):
            protected.append(run)
        else:
            candidates.append(run)

    for case in sorted(
        path for path in perf.iterdir() if path.is_dir() and path.name != "_aborted"
    ):
        for release in sorted(path for path in case.iterdir() if path.is_dir()):
            runs = sorted(
                (path for path in release.iterdir() if path.is_dir()),
                key=run_time,
                reverse=True,
            )
            for run in runs[keep:]:
                classify(run)

    aborted = perf / "_aborted"
    if aborted.is_dir():
        cutoff = now.timestamp() - aborted_days * 86400
        for run in sorted(aborted.glob("*/*/*")):
            if run.is_dir() and run_time(run).timestamp() < cutoff:
                classify(run)
    return Report(sorted(set(candidates)), sorted(set(protected)))


def retention_candidates(
    artifacts: Path,
    docs: Path,
    *,
    keep: int,
    aborted_days: int,
    now: datetime,
) -> list[Path]:
    return retention_report(
        artifacts,
        docs,
        keep=keep,
        aborted_days=aborted_days,
        now=now,
    ).candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS,
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=DEFAULT_DOCS,
    )
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--aborted-days", type=int, default=7)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.keep < 1 or args.aborted_days < 0:
        parser.error("keep must be >= 1 and aborted-days must be >= 0")

    report = retention_report(
        args.artifacts_root,
        args.docs_root,
        keep=args.keep,
        aborted_days=args.aborted_days,
        now=datetime.now(timezone.utc),
    )
    candidates = report.candidates
    total = sum(
        path.stat().st_size
        for directory in candidates
        for path in directory.rglob("*")
        if path.is_file()
    )
    for path in candidates:
        print(
            ("DELETE " if args.apply else "WOULD_DELETE ")
            + path.relative_to(args.artifacts_root).as_posix()
        )
        if args.apply:
            shutil.rmtree(path)
    # 报的是「真的被保护下来的轮次数」，不是白名单条数。文档里出现的
    # ``artifacts/perf/**/registry-baseline.json`` 这类散文 glob 会进白名单
    # 但一个轮次都保护不了，按条数报会打印 referenced=1 而实际保护为 0。
    for path in report.protected:
        print("KEEP " + path.relative_to(args.artifacts_root).as_posix())
    print(
        f"candidates={len(candidates)} bytes={total} "
        f"protected={len(report.protected)} applied={str(args.apply).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
