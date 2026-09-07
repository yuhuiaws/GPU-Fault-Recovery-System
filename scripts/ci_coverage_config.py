"""The coverage gate's configuration layer: ``config/ci-unit-gate.json`` and the
test/source partition it declares.

Split out of ``ci_coverage_gate.py`` when the deploy roots joined the measured
sources (S16) and pushed that file past the 1500-line size ratchet -- the
``scripts/`` tree deliberately allows no size exceptions, so the answer is a
second module, not a baseline entry. Everything here is pure configuration
reading and partition logic; the gate module re-exports the names it always had.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from scripts.ci_coverage_floors import validate_module_floors
    from scripts.ci_gate_artifacts import CoverageGateError, repository_files
else:
    from ci_coverage_floors import validate_module_floors
    from ci_gate_artifacts import CoverageGateError, repository_files

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/ci-unit-gate.json"
CONFIG_SCHEMA_VERSION = 3
RUNTIME_SHARDS = ("runtime_0", "runtime_1", "runtime_2")
SHARDS = (*RUNTIME_SHARDS, "deployment", "fault_runner", "postgres")
TEST_DOMAINS = ("runtime", "deployment", "fault_runner", "postgres")
IDENTITY_GROUPS = {
    "dependencies",
    "deployment_source",
    "deployment_tests",
    "fault_runner_source",
    "fault_runner_tests",
    "postgres_tests",
    "protocol",
    "runtime_source",
    "runtime_tests",
    "shared_tests",
}


def load_config(root: Path = ROOT) -> dict[str, Any]:
    path = root / CONFIG_PATH.relative_to(ROOT)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageGateError("coverage shard config is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != CONFIG_SCHEMA_VERSION
        or value.get("domain") != "unit"
        or not isinstance(value.get("coverage"), dict)
        or not isinstance(value.get("tests"), dict)
        or not isinstance(value.get("identity"), dict)
        or not isinstance(value.get("protocol"), dict)
        or set(value.get("shards", {})) != set(SHARDS)
        or value["protocol"].get("runtime_partitions") != len(RUNTIME_SHARDS)
    ):
        raise CoverageGateError("coverage shard config is incomplete")
    identity = value["identity"]
    tests = value["tests"]
    coverage = value["coverage"]
    required_lists = (
        (identity, "exclude_prefixes"),
        (identity, "exclude_globs"),
        (identity, "exclude_files"),
        (identity, "protocol_files"),
        (tests, "coverage_excluded_files"),
        (tests, "postgres_files"),
        (tests, "fault_runner_files"),
        (tests, "deployment_prefixes"),
        (tests, "deployment_globs"),
        (tests, "shared_files"),
        (coverage, "sources"),
        (coverage, "deployment_only_sources"),
        (coverage, "deployment_only_globs"),
        (coverage, "application_shared_files"),
        (coverage, "module_floors"),
    )
    if any(not isinstance(mapping.get(name), list) for mapping, name in required_lists):
        raise CoverageGateError("coverage shard config lists are incomplete")
    _validate_coverage_sources(coverage)
    validate_module_floors(coverage["module_floors"])
    for shard in SHARDS:
        raw = value["shards"][shard]
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("identity_groups"), list)
            or not isinstance(raw.get("omit_deployment_source"), bool)
            or not set(raw["identity_groups"]) <= IDENTITY_GROUPS
        ):
            raise CoverageGateError(f"coverage shard config is invalid: {shard}")
    return value


def _validate_coverage_sources(coverage: Mapping[str, Any]) -> None:
    """Reject a source declaration the shards could not agree on.

    Schema 2 had one ``source`` string, which is why the release orchestrator
    under ``deploy/`` sat outside every floor. Schema 3 lists ``sources`` and
    the ``deployment_only_sources`` subset that runtime shards must not measure;
    the old key is refused rather than aliased so a stale config fails loudly
    instead of silently measuring one root.
    """

    sources = coverage["sources"]
    deployment_only = coverage["deployment_only_sources"]
    if (
        "source" in coverage
        or not sources
        or any(not isinstance(item, str) or not item for item in sources)
        or len(set(sources)) != len(sources)
        or any(item not in sources for item in deployment_only)
        or not set(sources) - set(deployment_only)
    ):
        raise CoverageGateError("coverage shard config sources are invalid")


def shard_coverage_sources(config: Mapping[str, Any], shard: str) -> tuple[str, ...]:
    """Source roots ``shard`` measures.

    A shard that omits deployment source drops the deployment-only roots
    entirely, not merely their files: its identity excludes ``deploy/``, so
    measuring anything there would let a stale shard be reused against a changed
    release orchestrator.
    """

    coverage = config["coverage"]
    sources = tuple(str(item) for item in coverage["sources"])
    if config["shards"][shard]["omit_deployment_source"]:
        excluded = {str(item) for item in coverage["deployment_only_sources"]}
        sources = tuple(item for item in sources if item not in excluded)
    return sources


def _is_excluded(relative: str, config: Mapping[str, Any]) -> bool:
    identity = config["identity"]
    return (
        relative in set(identity["exclude_files"])
        or any(
            relative.startswith(str(value)) for value in identity["exclude_prefixes"]
        )
        or any(
            fnmatch.fnmatchcase(relative, str(pattern))
            for pattern in identity["exclude_globs"]
        )
    )


def deployment_only_source_files(
    root: Path = ROOT,
    *,
    config: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    current = load_config(root) if config is None else config
    coverage = current["coverage"]
    shared = set(str(value) for value in coverage["application_shared_files"])
    result = {
        path.relative_to(root).as_posix()
        for path in repository_files(root)
        if any(
            fnmatch.fnmatchcase(
                path.relative_to(root).as_posix(),
                str(pattern),
            )
            for pattern in coverage["deployment_only_globs"]
        )
        and path.relative_to(root).as_posix() not in shared
    }
    if not result:
        raise CoverageGateError("deployment-only coverage source is empty")
    return tuple(sorted(result))


def _test_owner(relative: str, config: Mapping[str, Any]) -> str | None:
    tests = config["tests"]
    if relative in set(tests["shared_files"]):
        return "shared_tests"
    if relative in set(tests["coverage_excluded_files"]):
        return None
    if relative in set(tests["postgres_files"]) or relative.startswith(
        "tests/store/_postgres"
    ):
        return "postgres_tests"
    if relative in set(tests["fault_runner_files"]):
        return "fault_runner_tests"
    if any(relative.startswith(str(value)) for value in tests["deployment_prefixes"]):
        return "deployment_tests"
    if any(
        fnmatch.fnmatchcase(relative, str(pattern))
        for pattern in tests["deployment_globs"]
    ):
        return "deployment_tests"
    return "runtime_tests"


def logical_test_domain(shard: str) -> str:
    return "runtime" if shard in RUNTIME_SHARDS else shard


def pytest_targets(root: Path, shard: str) -> tuple[str, ...]:
    if shard not in SHARDS:
        raise CoverageGateError(f"unknown coverage shard: {shard}")
    config = load_config(root)
    expected_group = f"{logical_test_domain(shard)}_tests"
    targets = []
    for path in repository_files(root):
        relative = path.relative_to(root).as_posix()
        if (
            relative.startswith("tests/")
            and path.name.startswith("test_")
            and path.suffix == ".py"
            and _test_owner(relative, config) == expected_group
        ):
            targets.append(relative)
    if not targets:
        raise CoverageGateError(f"coverage shard has no pytest targets: {shard}")
    return tuple(sorted(targets))


def validate_test_partition(root: Path = ROOT) -> dict[str, int]:
    config = load_config(root)
    excluded = set(config["tests"]["coverage_excluded_files"])
    assigned: dict[str, str] = {}
    counts = {domain: 0 for domain in TEST_DOMAINS}
    for path in repository_files(root):
        relative = path.relative_to(root).as_posix()
        if (
            not relative.startswith("tests/")
            or not path.name.startswith("test_")
            or path.suffix != ".py"
        ):
            continue
        owner = _test_owner(relative, config)
        if owner is None:
            if relative not in excluded:
                raise CoverageGateError(f"unclassified coverage test: {relative}")
            continue
        shard = owner.removesuffix("_tests")
        if shard not in TEST_DOMAINS or relative in assigned:
            raise CoverageGateError(f"coverage test has invalid owner: {relative}")
        assigned[relative] = shard
        counts[shard] += 1
    configured = {
        *config["tests"]["postgres_files"],
        *config["tests"]["fault_runner_files"],
        *excluded,
    }
    missing = sorted(
        relative for relative in configured if not (root / relative).is_file()
    )
    if missing:
        raise CoverageGateError(
            "configured coverage tests are missing: " + ", ".join(missing)
        )
    if any(not count for count in counts.values()):
        raise CoverageGateError("one or more coverage shards are empty")
    return counts
