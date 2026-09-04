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
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from scripts.ci_gate_artifacts import CoverageGateError
else:
    from ci_gate_artifacts import CoverageGateError


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
            or not isinstance(entry.get("group_floor"), int)
            or not isinstance(entry.get("file_floor"), int)
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
    violations = []
    for entry in config["coverage"]["module_floors"]:
        matched = {
            relative: value
            for relative, value in files.items()
            if any(
                fnmatch.fnmatchcase(relative, str(pattern))
                for pattern in entry["globs"]
            )
        }
        if not matched:
            violations.append(
                f"coverage module floor {entry['id']} matched no measured file"
            )
            continue
        covered = 0
        total = 0
        for relative, value in sorted(matched.items()):
            summary = value.get("summary") or {}
            statements = int(summary.get("num_statements") or 0)
            branches = int(summary.get("num_branches") or 0)
            missing = int(summary.get("missing_lines") or 0)
            partial = int(summary.get("num_partial_branches") or 0)
            measurable = statements + branches
            reached = measurable - missing - partial
            covered += reached
            total += measurable
            if measurable == 0:
                # A module with nothing to measure (only imports and constants)
                # cannot fall below a floor, and failing it would push authors
                # towards deleting the file rather than testing it.
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
