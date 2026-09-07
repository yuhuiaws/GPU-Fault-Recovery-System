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
# The per-file baseline stops growth but has no opinion about where the 1,400
# recorded errors should shrink first. The convergence file does: a ceiling per
# directory, ordered by priority, that only ever ratchets down (S19). 74% of the
# baseline is `no-untyped-def`/`type-arg`/`no-any-return`, concentrated in the
# store mixins, the orchestration families and the route handlers -- the three
# layers where "dict shape written wrong" is the main failure mode.
CONVERGENCE = ROOT / "mypy-convergence.json"
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


Targets = dict[str, dict[str, object]]


def load_baseline() -> dict[str, int]:
    return dict(json.loads(BASELINE.read_text(encoding="utf-8")))


def load_convergence_targets() -> Targets:
    if not CONVERGENCE.is_file():
        return {}
    return dict(json.loads(CONVERGENCE.read_text(encoding="utf-8")))


def _key_directory_matches(key: str, directory: str) -> bool:
    path = key.rsplit(":", 1)[0]
    return path == directory or path.startswith(directory.rstrip("/") + "/")


def directory_totals(targets: Targets, current: Counter[str]) -> dict[str, int]:
    """Strict errors under each convergence directory (a file counts once)."""

    return {
        directory: sum(
            count
            for key, count in current.items()
            if _key_directory_matches(key, directory)
        )
        for directory in targets
    }


def convergence_failures(targets: Targets, current: Counter[str]) -> list[str]:
    """Only growth past a ceiling fails; shrinking is the point, not slack."""

    problems = []
    for directory, total in directory_totals(targets, current).items():
        ceiling = int(str(targets[directory]["ceiling"]))
        if total > ceiling:
            problems.append(
                f"{directory} has {total} strict mypy errors, above its "
                f"convergence ceiling of {ceiling}; fix errors in that "
                "directory, do not raise the ceiling"
            )
    return problems


def tightened_ceilings(targets: Targets, current: Counter[str]) -> Targets:
    """Ceilings follow the current totals downwards and never move up."""

    totals = directory_totals(targets, current)
    return {
        directory: {
            **target,
            "ceiling": min(int(str(target["ceiling"])), totals[directory]),
        }
        for directory, target in targets.items()
    }


def report_lines(targets: Targets, current: Counter[str]) -> list[str]:
    totals = directory_totals(targets, current)
    ordered = sorted(targets.items(), key=lambda item: int(str(item[1]["priority"])))
    return [
        f"P{target['priority']} {directory}: {totals[directory]}/"
        f"{target['ceiling']} ({target['why']})"
        for directory, target in ordered
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the per-directory convergence table and exit",
    )
    args = parser.parse_args()

    current, output = current_errors()
    targets = load_convergence_targets()
    if args.report:
        for line in report_lines(targets, current):
            print(line)
        return 0
    if args.write_baseline:
        BASELINE.write_text(
            json.dumps(dict(sorted(current.items())), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {BASELINE.name}: {sum(current.values())} errors")
        if targets:
            CONVERGENCE.write_text(
                json.dumps(tightened_ceilings(targets, current), indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {CONVERGENCE.name}")
        return 0
    if not BASELINE.is_file():
        print(
            f"{BASELINE.name} is missing; run --write-baseline",
            file=sys.stderr,
        )
        return 1
    baseline = load_baseline()
    problems = failures(baseline, current) + convergence_failures(targets, current)
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
