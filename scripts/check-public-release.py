"""Reject live-environment identities from the public repository tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Capped well below the core count because this gate runs as one member of a
# parallel static-gate group: taking every core would slow the gates it shares
# the machine with by more than it saves here.
WORKER_LIMIT = 16
# Below this a pool costs more than it saves, and it keeps the gate's own tests
# -- which scan a handful of fixture files -- on the straightforward path.
MIN_PARALLEL_FILES = 32
# A file larger than this is cut into shards of roughly this size, so the wall
# clock is not set by whichever single file is biggest.
SHARD_BYTES = 256 * 1024
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".lua",
    ".md",
    ".py",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SCANNER_SOURCES = {
    "scripts/check-public-release.py",
    "tests/test_public_release.py",
    "tests/test_documentation_contracts.py",
    "tests/test_fault_scenario_catalog.py",
    "tests/test_script_assets.py",
}
PUBLIC_AWS_ACCOUNTS = {
    "000000000000",
    "111122223333",
    "123456789012",
    # AWS-owned public ECR registry used by the ADOT image.
    "602401143452",
}
SYNTHETIC_INSTANCE_IDS = {
    "i-0000000000000000",
    "i-00000000000000001",
    "i-00000000000000002",
}
PLACEHOLDER_RESOURCE_IDS = {
    "sg-0123456789abcdef0",
}

ARN_ACCOUNT = re.compile(
    r"arn:(?:aws|aws-cn|aws-us-gov):[^:\s]+:[^:\s]*:"
    r"(?P<account>\d{12}):"
)
ECR_ACCOUNT = re.compile(r"\b(?P<account>\d{12})\.dkr\.ecr\.")
AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
INSTANCE_ID = re.compile(r"\bi-[0-9a-f]{16,17}\b", re.IGNORECASE)
RESOURCE_ID = re.compile(
    r"\b(?:vpc|subnet|sg|eni|vol|snap|ami|igw|nat|rtb|vpce|fs)"
    r"-[0-9a-f]{8,17}\b",
    re.IGNORECASE,
)
UUID = re.compile(
    r"(?<![0-9a-f])"
    r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
    r"(?![0-9a-f])",
    re.IGNORECASE,
)
BARE_NODE_ID = re.compile(
    r"(?<![0-9a-f-])[0-9a-f]{17}(?![0-9a-f])",
    re.IGNORECASE,
)
CONCRETE_POD = re.compile(r"\bgpu-fault-[a-z0-9-]+-[0-9a-f]{8,10}-[a-z0-9]{5}\b")
PERSONAL_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@(?:amazon\.com|amazon\.aws)\b",
    re.IGNORECASE,
)
SITE_PRIVATE_ADDRESS = re.compile(r"\b10\.(?:91|92)(?:\.\d{1,3}){2}\b")
LOCAL_REPOSITORY = re.compile(r"/(?:home|Users)/[^/\s]+/GPU_failure_handling")
SITE_MARKERS = (
    re.compile(r"\b(?:hp|eks)-cluster-hypd-[A-Za-z0-9-]+\b", re.IGNORECASE),
    re.compile(r"\bcontrol-plane-GPU-fault-solution\b", re.IGNORECASE),
    re.compile(r"\bhypd-\d+\b", re.IGNORECASE),
    re.compile(r"\bspot-p5en-usw2az3\b", re.IGNORECASE),
    re.compile(r"\baccelerated-liangtest-\d+\b", re.IGNORECASE),
    re.compile(r"\bliang(?:aws|200)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    label: str
    value: str


@dataclass(frozen=True)
class Check:
    """One pattern, and what a match of it is reported as.

    ``group`` names the capture holding the reported value when the pattern
    matches more context than it is about -- an ARN is matched whole so the
    account is not confused with any other twelve digits. ``lower`` folds the
    value before both the ``allowed`` lookup and the report, so a placeholder
    written in either case is recognised as one.
    """

    label: str
    pattern: re.Pattern[str]
    group: str | None = None
    allowed: frozenset[str] = frozenset()
    lower: bool = False


CHECKS = (
    Check(
        "customer AWS account",
        ARN_ACCOUNT,
        group="account",
        allowed=frozenset(PUBLIC_AWS_ACCOUNTS),
    ),
    Check(
        "customer ECR account",
        ECR_ACCOUNT,
        group="account",
        allowed=frozenset(PUBLIC_AWS_ACCOUNTS),
    ),
    Check(
        "concrete EC2 instance ID",
        INSTANCE_ID,
        allowed=frozenset(SYNTHETIC_INSTANCE_IDS),
        lower=True,
    ),
    Check(
        "concrete AWS resource ID",
        RESOURCE_ID,
        allowed=frozenset(PLACEHOLDER_RESOURCE_IDS),
        lower=True,
    ),
    Check("AWS access key", AWS_ACCESS_KEY),
    Check("private key", PRIVATE_KEY),
    Check("personal email", PERSONAL_EMAIL),
    Check("site private address", SITE_PRIVATE_ADDRESS),
    Check("local repository path", LOCAL_REPOSITORY),
    *(Check("site-specific resource name", pattern) for pattern in SITE_MARKERS),
)
MARKDOWN_CHECKS = (
    Check("raw UUID in public documentation", UUID),
    Check("raw node identity in public documentation", BARE_NODE_ID),
    Check("concrete Kubernetes Pod in public documentation", CONCRETE_POD),
)
NEWLINE = re.compile("\n")


def _git_public_files(root: Path) -> list[Path] | None:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        return None
    return [
        root / Path(item.decode(errors="surrogateescape"))
        for item in completed.stdout.split(b"\0")
        if item
    ]


def public_text_files(root: Path) -> tuple[list[Path], str]:
    """The files this gate scans, and a digest of exactly what it scanned.

    The digest covers every scanned path and its bytes, plus this script's own
    source, and it is what a cached verdict may be keyed on. Nothing cheaper is
    sound: the deploy's ``prepared_tree_sha256`` covers ``git ls-files`` only, so
    keying on that would let an untracked-but-unignored file -- which this gate
    does scan, and which is exactly where a live identity arrives from -- change
    without invalidating the verdict.

    It costs nothing to produce because the walk already reads every candidate
    whole to decide whether it is text at all.
    """

    digests: dict[str, str] = {}
    candidates = _git_public_files(root)
    if candidates is None:
        import pathspec

        ignore_path = root / ".gitignore"
        ignore = pathspec.GitIgnoreSpec.from_lines(
            ignore_path.read_text(encoding="utf-8").splitlines()
            if ignore_path.is_file()
            else ()
        )
        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and not ignore.match_file(path.relative_to(root).as_posix())
        ]
    result = []
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".git/"):
            continue
        if relative in SCANNER_SOURCES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data:
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        digests[relative] = hashlib.sha256(data).hexdigest()
        result.append(path)
    combined = hashlib.sha256(Path(__file__).resolve().read_bytes())
    for relative, digest in sorted(digests.items()):
        combined.update(b"\0path\0")
        combined.update(relative.encode())
        combined.update(b"\0content\0")
        combined.update(digest.encode())
    return sorted(result), combined.hexdigest()


def _file_violations(
    relative: str,
    text: str,
    checks: tuple[Check, ...],
    *,
    line_offset: int = 0,
) -> list[Violation]:
    """Every violation in one file, matched against the whole text at once.

    Per line is the obvious way to write this and it is what made the gate the
    slowest thing in a deploy: fourteen patterns against 376k lines is 5.3
    million interpreter-level ``finditer`` calls for a repository that has no
    violations at all. Whole-text matching is fourteen calls per file, and the
    line number is recovered from the match offset only for the matches that
    exist.

    The two are equivalent because no pattern here can match across a line
    break: every one is anchored on word boundaries or on character classes
    that exclude whitespace, so a match found in the joined text lies wholly
    within one line.

    ``text`` may be a whole-line slice of a larger file, in which case
    ``line_offset`` is the number of lines preceding it, so the reported number
    is the one an editor will open.
    """

    found: list[tuple[Check, list[re.Match[str]]]] = []
    for check in checks:
        matches = list(check.pattern.finditer(text))
        if matches:
            found.append((check, matches))
    if not found:
        return []
    # Built once per offending file, never for a clean one, which is every file
    # on the path this gate is normally on.
    starts = [0, *(match.end() for match in NEWLINE.finditer(text))]
    violations = []
    for check, matches in found:
        for match in matches:
            value = match.group(check.group) if check.group else match.group()
            if check.lower:
                value = value.lower()
            if value in check.allowed:
                continue
            violations.append(
                Violation(
                    relative,
                    line_offset + bisect_right(starts, match.start()),
                    check.label,
                    value,
                )
            )
    return violations


def _shard_bounds(text: str, count: int) -> list[int]:
    """Cut ``text`` into at most ``count`` pieces, each holding whole lines.

    Every cut lands just after a newline, which is what keeps a shard equivalent
    to the file it came from: the patterns cannot match across a line break, so
    no match can straddle a boundary drawn on one. Cuts are aimed at even sizes
    and then moved forward to the next newline, so a file with one very long
    line simply yields fewer, larger shards rather than an unsafe split.
    """

    bounds = [0]
    for index in range(1, count):
        cut = text.find("\n", len(text) * index // count)
        if cut < 0:
            break
        if cut + 1 > bounds[-1]:
            bounds.append(cut + 1)
    if len(text) > bounds[-1]:
        bounds.append(len(text))
    return bounds


def _work_items(root: Path, relatives: list[str]) -> list[tuple[str, str, int, int]]:
    """One item per shard, sized so no single unit dominates the wall clock.

    Whole files are the natural unit and they do not balance: four generated HTML
    documents are half of the repository's public bytes, so a per-file fan-out
    finishes in the time the largest file takes on one core. Splitting those
    across shards is what turns the core count into speed.

    Ordered biggest first because the pool hands work out in order, and the
    longest unit is the one that must not be picked up last.
    """

    sized = sorted(
        ((root / relative).stat().st_size, relative) for relative in relatives
    )
    items = []
    for size, relative in reversed(sized):
        count = max(1, -(-size // SHARD_BYTES))
        items.extend((str(root), relative, index, count) for index in range(count))
    return items


def _scan_one(item: tuple[str, str, int, int]) -> list[Violation]:
    """Scan one shard, addressed by name so the item can cross a process.

    A worker re-reads and re-cuts the file rather than being handed its text:
    30MB of source pickled to the pool would cost more than the matching it is
    meant to spread, and the cut is derived from the text so every worker agrees
    on it without being told.
    """

    root, relative, index, count = item
    path = Path(root) / relative
    checks = CHECKS
    if path.suffix.lower() == ".md":
        checks = CHECKS + MARKDOWN_CHECKS
    text = path.read_text(encoding="utf-8")
    bounds = _shard_bounds(text, count)
    if index + 1 >= len(bounds):
        # Fewer shards than asked for, because the file ran out of line breaks
        # to cut on. The shards that do exist still cover it whole.
        return []
    start, end = bounds[index], bounds[index + 1]
    return _file_violations(
        relative,
        text[start:end],
        checks,
        line_offset=text.count("\n", 0, start),
    )


def scan_files(
    root: Path,
    paths: list[Path],
    *,
    workers: int | None = None,
) -> list[Violation]:
    """Every violation in ``paths``.

    Spread over processes because this gate is pure regex over every public byte
    -- 30M characters against seventeen patterns, some of which backtrack -- and
    that is one core's worth of work no matter how it is written. Measured at
    19s serial, which was the largest single item in a deploy's static gates and
    is paid several times per deploy.

    The alternative was a literal prefilter per pattern, which would have been
    faster still and was rejected: it states each pattern's meaning a second
    time, in a list that can silently stop matching the pattern it guards, and a
    safety gate that quietly scans less is worse than a slow one.
    """

    relatives = [path.relative_to(root).as_posix() for path in paths]
    items = _work_items(root, relatives)
    if workers is None:
        workers = min(WORKER_LIMIT, len(os.sched_getaffinity(0)))
    violations: list[Violation] = []
    if workers > 1 and len(relatives) > MIN_PARALLEL_FILES:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for found in executor.map(_scan_one, items):
                violations.extend(found)
    else:
        for item in items:
            violations.extend(_scan_one(item))
    # Sorted rather than in pattern order: the report is read as a to-do list,
    # and grouping a file's findings by position is what an editor needs.
    return sorted(
        violations,
        key=lambda item: (item.path, item.line, item.label, item.value),
    )


def scan(root: Path, *, workers: int | None = None) -> tuple[list[Violation], int]:
    """Every violation in the public tree, with the number of files scanned.

    The count is returned rather than recomputed by the caller: establishing it
    means walking the tree and decoding every candidate again, which the passing
    report used to do purely to print one number.
    """

    paths, _ = public_text_files(root)
    return scan_files(root, paths, workers=workers), len(paths)


def _cached_pass(cache: Path, digest: str) -> bool:
    """Whether ``cache`` records a pass for exactly this content.

    A hit means the same bytes under the same paths were scanned by the same
    scanner and found clean, so re-running is arithmetic. A miss for any reason
    -- absent, unreadable, malformed, different digest -- scans.

    This does not widen who can suppress the gate. The cache lives in the deploy
    state directory, whose writers already hold the signing material and the
    ``gpu-fault-admin`` virtualenv the deploy executes.
    """

    try:
        record = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and record.get("digest") == digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--verdict-cache",
        type=Path,
        default=None,
        help=(
            "reuse a recorded pass when the scanned content is byte-identical, "
            "and record this run's pass there"
        ),
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    cache: Path | None = arguments.verdict_cache
    paths, digest = public_text_files(root)
    if cache is not None and _cached_pass(cache, digest):
        print(
            f"public release safety check reused: {len(paths)} text file(s) "
            f"unchanged since {digest[:12]}"
        )
        return 0
    violations = scan_files(root, paths)
    scanned = len(paths)
    if violations:
        print("public release safety check failed:", file=sys.stderr)
        for item in violations:
            print(
                f"- {item.path}:{item.line}: {item.label}: {item.value}",
                file=sys.stderr,
            )
        return 1
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({"digest": digest, "scanned": scanned}) + "\n",
            encoding="utf-8",
        )
    print(f"public release safety check passed: scanned {scanned} text file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
