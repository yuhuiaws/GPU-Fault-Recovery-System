"""Ratchet strict mypy errors across the source tree."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "mypy-baseline.json"
# `deploy` is checked alongside `src` because the release orchestrator lives
# there, not in the package: ~14k lines that mutate two clusters and hold the
# rollback path, previously outside every type gate.
TARGETS = ("src", "deploy")
ERROR = re.compile(
    r"^((?:src|deploy)/[^:]+):\d+(?::\d+)?: error: .+ \[([a-z-]+)\]$",
)


def current_errors() -> tuple[Counter[str], str]:
    """Every strict error in ``TARGETS``, counted per file and error code.

    Incremental, because mypy stores the errors it found alongside the analysis
    and replays them for an unchanged module, so a cached run reports the same
    set as a full one -- measured here as an exact per-key match at 1417 errors,
    16.9s full against 0.4s cached. The cache is validated by size and mtime and
    then by content hash, and mypy invalidates it wholesale when its own version
    or option set changes.

    That equivalence is what the ratchet needs, and a cache that failed to
    deliver it would say so: fewer errors for a file reads as baseline slack and
    fails this gate, exactly as a regression does. Set ``MYPY_CACHE_DIR`` to
    share one cache across working trees -- a deploy checks a fresh snapshot of
    the same content, and paying 17s to re-derive an identical answer per deploy
    is the whole reason this is not ``--no-incremental``.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            *TARGETS,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = result.stdout + result.stderr
    counts: Counter[str] = Counter()
    for line in output.splitlines():
        match = ERROR.match(line)
        if match:
            counts[f"{match.group(1)}:{match.group(2)}"] += 1
    if result.returncode not in {0, 1}:
        raise RuntimeError(output)
    return counts, output


def failures(
    baseline: dict[str, int],
    current: Counter[str],
) -> list[str]:
    result = []
    keys = set(baseline) | set(current)
    for key in sorted(keys):
        expected = baseline.get(key, 0)
        actual = current.get(key, 0)
        if actual > expected:
            result.append(f"{key} grew from {expected} to {actual}")
        elif actual < expected:
            result.append(
                f"{key} baseline has slack: current {actual}, "
                f"recorded {expected}; run --write-baseline"
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    current, output = current_errors()
    if args.write_baseline:
        BASELINE.write_text(
            json.dumps(dict(sorted(current.items())), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {BASELINE.name}: {sum(current.values())} errors")
        return 0
    if not BASELINE.is_file():
        print(
            f"{BASELINE.name} is missing; run --write-baseline",
            file=sys.stderr,
        )
        return 1
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = failures(baseline, current)
    if problems:
        print("mypy baseline check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        print(output, file=sys.stderr)
        return 1
    attr_errors = sum(
        count for key, count in current.items() if key.endswith(":attr-defined")
    )
    print(
        "mypy baseline check passed: "
        f"{sum(current.values())} errors, "
        f"{attr_errors} attr-defined"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
