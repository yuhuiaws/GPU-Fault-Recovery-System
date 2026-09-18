"""Per-module coverage floors for the sharded unit coverage gate.

Separate from ``ci_coverage_gate.py`` because it is the one part of the gate that
reads a finished coverage report and judges it: everything else there is about
partitioning tests, hashing inputs and moving signed artifacts around. It takes
the configuration as an argument and depends on nothing else in the gate, so the
dependency runs one way and the floors can be exercised on a report alone.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING or __package__:
    from scripts.ci_gate_artifacts import CoverageGateError, repository_files
else:
    from ci_gate_artifacts import CoverageGateError, repository_files

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_TARGET = 95
COVERAGE_SCOPES = {
    "production": (
        "src/gpu_fault",
        "src/gpu_fault_release",
        "deploy/control-plane/tools",
    ),
    "runner": ("scripts/e2e/regional", "tools"),
}


@dataclass(frozen=True)
class CoverageCounts:
    statements: int
    covered_lines: int
    branches: int
    covered_branches: int

    @classmethod
    def from_summary(cls, summary: object, *, source: str) -> CoverageCounts:
        if not isinstance(summary, dict):
            raise CoverageGateError(f"coverage summary is missing for {source}")
        fields = (
            "num_statements",
            "covered_lines",
            "missing_lines",
            "num_branches",
            "covered_branches",
            "missing_branches",
        )
        if any(type(summary.get(key)) is not int or summary[key] < 0 for key in fields):
            raise CoverageGateError(f"coverage counters are invalid for {source}")
        if (
            summary["covered_lines"] + summary["missing_lines"]
            != summary["num_statements"]
            or summary["covered_branches"] + summary["missing_branches"]
            != summary["num_branches"]
        ):
            raise CoverageGateError(f"coverage counters disagree for {source}")
        return cls(
            summary["num_statements"],
            summary["covered_lines"],
            summary["num_branches"],
            summary["covered_branches"],
        )

    @property
    def measured(self) -> int:
        return self.statements + self.branches

    @property
    def covered(self) -> int:
        return self.covered_lines + self.covered_branches


def validate_module_floors(raw: Sequence[Any]) -> None:
    """Reject a module floor that cannot fail.

    A group with no globs, or a per-file floor of zero, is decoration: it would
    pass whatever the code does while still reading like a guarantee in review.
    A group floor below ``coverage.floor`` is expected rather than suspicious --
    these groups exist precisely because they sit under the repository average,
    and a floor above it could only ever fail.
    """

    if not raw:
        raise CoverageGateError("coverage module floors are empty")
    identifiers = set()
    for entry in raw:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or not entry["id"]
            or not isinstance(entry.get("description"), str)
            or not isinstance(entry.get("globs"), list)
            or not entry["globs"]
            or type(entry.get("group_floor")) is not int
            or type(entry.get("file_floor")) is not int
            or not 0 < entry["file_floor"] <= entry["group_floor"] <= 100
        ):
            raise CoverageGateError("coverage module floor entry is invalid")
        if entry["id"] in identifiers:
            raise CoverageGateError(
                f"duplicate coverage module floor: {entry['id']}",
            )
        identifiers.add(entry["id"])


def module_floor_violations(
    coverage_json: Path,
    *,
    config: Mapping[str, Any],
    root: Path = ROOT,
) -> list[str]:
    """Per-group and per-file coverage failures a whole-repository floor hides.

    One floor over ``src/gpu_fault`` is satisfiable by a well-covered majority
    while a whole family of modules sits near zero -- and the administrator CLI
    is exactly that shape, because it is measured only in the deployment shard.
    The group floor stops the family from being carried by the rest of the tree;
    the file floor stops one well-covered module inside the family from carrying
    its siblings.
    """

    try:
        report = json.loads(coverage_json.read_text(encoding="utf-8"))
        files = report["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CoverageGateError("coverage JSON report is unreadable") from exc
    if not isinstance(files, dict):
        raise CoverageGateError("coverage JSON file inventory is invalid")
    sources = {
        path.relative_to(root).as_posix()
        for path in repository_files(root)
        if path.suffix == ".py"
    }
    violations = []
    for entry in config["coverage"]["module_floors"]:
        expected = {
            relative
            for relative in sources
            if any(
                fnmatch.fnmatchcase(relative, str(pattern))
                for pattern in entry["globs"]
            )
        }
        matched = {
            relative: value
            for relative, value in files.items()
            if any(
                fnmatch.fnmatchcase(relative, str(pattern))
                for pattern in entry["globs"]
            )
        }
        missing = sorted(expected - matched.keys())
        unexpected = sorted(matched.keys() - expected)
        if missing or unexpected:
            if missing:
                violations.append(
                    f"coverage module floor {entry['id']} has unmeasured sources: "
                    + ", ".join(missing)
                )
            if unexpected:
                violations.append(
                    f"coverage module floor {entry['id']} reports unknown sources: "
                    + ", ".join(unexpected)
                )
            continue
        if not matched:
            violations.append(
                f"coverage module floor {entry['id']} matched no measured file"
            )
            continue
        covered = 0
        total = 0
        for relative, value in sorted(matched.items()):
            counts = CoverageCounts.from_summary(
                value.get("summary") if isinstance(value, dict) else None,
                source=relative,
            )
            measurable = counts.measured
            reached = counts.covered
            covered += reached
            total += measurable
            if measurable == 0:
                # Empty modules cannot supply a meaningful per-file percentage.
                continue
            percent = 100.0 * reached / measurable
            if percent < entry["file_floor"]:
                violations.append(
                    f"{relative} covers {percent:.1f}% of "
                    f"{measurable} measurable points, below the "
                    f"{entry['id']} file floor of {entry['file_floor']}%"
                )
        if total == 0:
            violations.append(
                f"coverage module floor {entry['id']} has nothing to measure"
            )
            continue
        percent = 100.0 * covered / total
        if percent < entry["group_floor"]:
            violations.append(
                f"module group {entry['id']} covers {percent:.1f}%, below its "
                f"group floor of {entry['group_floor']}%"
            )
    return violations
