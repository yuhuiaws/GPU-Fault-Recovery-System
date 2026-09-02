from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError

if __package__:
    from scripts.ci_gate_artifacts import (
        GateArtifactError,
        canonical_sha256,
        download,
        evidence_entry,
        extract_archive,
        find_reusable_artifact,
        git,
        repository_files,
        sha256,
        verify_evidence,
        write_outputs,
    )
else:
    from ci_gate_artifacts import (
        GateArtifactError,
        canonical_sha256,
        download,
        evidence_entry,
        extract_archive,
        find_reusable_artifact,
        git,
        repository_files,
        sha256,
        verify_evidence,
        write_outputs,
    )


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/ci-unit-gate.json"
SCHEMA_VERSION = 1
RUNTIME_SHARDS = ("runtime_0", "runtime_1", "runtime_2")
SHARDS = (*RUNTIME_SHARDS, "deployment", "fault_runner", "postgres")
TEST_DOMAINS = ("runtime", "deployment", "fault_runner", "postgres")
GATE_NAME = "coverage-shard-gate.json"
BUNDLE_NAME = "coverage-shard-gate.bundle.json"
BASE_GATE_NAME = "base-coverage-shard-gate.json"
BASE_BUNDLE_NAME = "base-coverage-shard-gate.bundle.json"
COVERAGE_DATA_NAME = "coverage-data"
PYTEST_RESULTS_NAME = "pytest-results.json"
STRESS_RESULTS_NAME = "postgres-stress-results.json"
DURATIONS_NAME = "durations.json"
ARTIFACT_PREFIX = "gpu-fault-coverage-"
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
FAULT_RUNNER_FILES = {
    "scripts/build-regional-case-index.py",
    "tools/case_scheduler.py",
    "tools/pytest_case_reporter.py",
    "tools/pytest_result_identity.py",
    "tools/run_fault_test_cases.py",
    "tools/run_regional_acceptance.py",
}


class CoverageGateError(RuntimeError):
    pass


def _clean_directory(root: Path, path: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved == resolved_root or resolved_root.is_relative_to(resolved):
        raise CoverageGateError(f"{label} must not contain the repository root")
    shutil.rmtree(resolved, ignore_errors=True)
    resolved.mkdir(parents=True)
    return resolved


def load_config(root: Path = ROOT) -> dict[str, Any]:
    path = root / CONFIG_PATH.relative_to(ROOT)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageGateError("coverage shard config is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 2
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
        (coverage, "deployment_only_globs"),
        (coverage, "application_shared_files"),
    )
    if any(not isinstance(mapping.get(name), list) for mapping, name in required_lists):
        raise CoverageGateError("coverage shard config lists are incomplete")
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


def _is_dependency(relative: str) -> bool:
    return (
        relative == "pyproject.toml"
        or relative == "uv.lock"
        or relative.startswith("requirements/")
    )


def _is_deployment_input(relative: str, deployment_only: set[str]) -> bool:
    name = Path(relative).name
    return (
        relative in deployment_only
        or relative.startswith("deploy/")
        or "release" in name
        or "deploy_host" in name
        or "staging_deploy" in name
        or "component_artifact" in name
    )


def _identity_group(
    relative: str,
    *,
    config: Mapping[str, Any],
    deployment_only: set[str],
) -> str | None:
    if _is_excluded(relative, config):
        return None
    if _is_dependency(relative):
        return "dependencies"
    if relative in set(config["identity"]["protocol_files"]):
        return "protocol"
    if relative.startswith("tests/"):
        return _test_owner(relative, config)
    if relative.startswith("testcases/") or relative in FAULT_RUNNER_FILES:
        return "fault_runner_source"
    if _is_deployment_input(relative, deployment_only):
        return "deployment_source"
    return "runtime_source"


def _file_entry(root: Path, path: Path) -> tuple[str, dict[str, object]]:
    relative = path.relative_to(root).as_posix()
    return relative, {
        "mode": f"{path.stat().st_mode & 0o777:04o}",
        "sha256": sha256(path),
    }


def _group_identity(entries: Mapping[str, dict[str, object]]) -> dict[str, object]:
    ordered = dict(sorted(entries.items()))
    return {
        "sha256": canonical_sha256(ordered),
        "file_count": len(ordered),
        "files": ordered,
    }


def input_groups(root: Path = ROOT) -> dict[str, dict[str, object]]:
    root = root.resolve()
    config = load_config(root)
    deployment_only = set(deployment_only_source_files(root, config=config))
    groups: dict[str, dict[str, dict[str, object]]] = {
        name: {} for name in sorted(IDENTITY_GROUPS)
    }
    for path in repository_files(root):
        relative = path.relative_to(root).as_posix()
        group = _identity_group(
            relative,
            config=config,
            deployment_only=deployment_only,
        )
        if group is None:
            continue
        name, entry = _file_entry(root, path)
        groups[group][name] = entry
    required = {
        group
        for shard in SHARDS
        for group in config["shards"][shard]["identity_groups"]
    }
    empty = sorted(group for group in required if not groups[group])
    if empty:
        raise CoverageGateError(
            "coverage identity groups are empty: " + ", ".join(empty)
        )
    return {name: _group_identity(entries) for name, entries in sorted(groups.items())}


def installed_distributions() -> dict[str, str]:
    values: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = str(distribution.metadata.get("Name") or "").strip()
        if not raw_name:
            continue
        name = raw_name.lower().replace("_", "-")
        version = str(distribution.version)
        existing = values.get(name)
        if existing is not None and existing != version:
            raise CoverageGateError(f"installed distribution versions differ: {name}")
        values[name] = version
    return dict(sorted(values.items()))


def postgres_image_identity(
    *,
    container: str | None = None,
    image: str | None = None,
) -> str:
    if image:
        return image.strip()
    if not container:
        raise CoverageGateError("PostgreSQL container or image identity is required")
    completed = subprocess.run(
        ["docker", "inspect", "--format={{.Image}}", container],
        text=True,
        capture_output=True,
        check=False,
    )
    value = completed.stdout.strip()
    if completed.returncode or not value:
        raise CoverageGateError("cannot resolve PostgreSQL image identity")
    return value


def _environment_identity(
    *,
    postgres_image: str,
    environment: Mapping[str, str],
) -> dict[str, object]:
    return {
        "machine": platform.machine(),
        "postgres_image": postgres_image,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "runner_environment": environment.get("RUNNER_ENVIRONMENT", ""),
        "runner_image_os": environment.get("ImageOS", ""),
        "runner_image_version": environment.get("ImageVersion", ""),
        "runner_label": environment.get("CI_TEST_RUNNER_LABEL", ""),
        "sysconfig_platform": sysconfig.get_platform(),
    }


def shard_identity(
    root: Path,
    shard: str,
    *,
    postgres_image: str = "",
    distributions: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
    environment_identity: Mapping[str, object] | None = None,
    pytest_workers: str | None = None,
) -> dict[str, Any]:
    if shard not in SHARDS:
        raise CoverageGateError(f"unknown coverage shard: {shard}")
    config = load_config(root)
    groups = input_groups(root)
    selected = {
        name: groups[name] for name in config["shards"][shard]["identity_groups"]
    }
    protocol = dict(config["protocol"])
    protocol["coverage_branch"] = bool(config["coverage"]["branch"])
    protocol["coverage_source"] = str(config["coverage"]["source"])
    protocol["pytest_workers"] = (
        pytest_workers
        or os.getenv("PYTEST_XDIST_WORKERS")
        or str(protocol["pytest_workers"])
    )
    if shard != "postgres":
        protocol.pop("postgres_stress_rounds")
        protocol.pop("postgres_stress_workers")
    if shard in RUNTIME_SHARDS:
        protocol["runtime_partition_index"] = RUNTIME_SHARDS.index(shard)
    else:
        protocol.pop("runtime_partitions")
    current_environment = os.environ if environment is None else environment
    resolved_environment = (
        dict(environment_identity)
        if environment_identity is not None
        else _environment_identity(
            postgres_image=postgres_image if shard == "postgres" else "",
            environment=current_environment,
        )
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": "coverage-shard",
        "shard": shard,
        "groups": selected,
        "environment": resolved_environment,
        "installed_distributions": dict(
            sorted((distributions or installed_distributions()).items())
        ),
        "protocol": protocol,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def artifact_name(shard: str, identity_sha256: str) -> str:
    if shard not in SHARDS or len(identity_sha256) != 64:
        raise CoverageGateError("coverage shard artifact identity is invalid")
    return f"{ARTIFACT_PREFIX}{shard}-{identity_sha256}"


def _quality_gates(shard: str) -> dict[str, str]:
    result = {
        "branch_coverage_data": "PASSED",
        "pytest": "PASSED",
    }
    if shard == "postgres":
        result["postgres_contract"] = "PASSED"
        result["postgres_stress"] = "PASSED"
    return result


def _evidence_names(shard: str) -> set[str]:
    result = {"coverage_data", "durations", "pytest_results"}
    if shard == "postgres":
        result.add("postgres_stress_results")
    return result


def _load_json(path: Path, message: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageGateError(message) from exc
    if not isinstance(value, dict):
        raise CoverageGateError(message)
    return value


def _verify_pytest_results(path: Path) -> dict[str, Any]:
    value = _load_json(path, "coverage shard pytest evidence is invalid")
    if value.get("schema_version") != 1 or not isinstance(value.get("records"), dict):
        raise CoverageGateError("coverage shard pytest evidence is invalid")
    return value


def _relative_measured_file(root: Path, value: str) -> str | None:
    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return None
    return path.as_posix()


def verify_coverage_data(root: Path, shard: str, path: Path) -> tuple[str, ...]:
    try:
        from coverage import CoverageData

        data = CoverageData(basename=str(path))
        data.read()
        measured = tuple(sorted(data.measured_files()))
    except Exception as exc:
        raise CoverageGateError("coverage shard data is invalid") from exc
    if not measured and shard != "fault_runner":
        raise CoverageGateError("coverage shard data contains no measured files")
    if shard != "deployment":
        deployment_only = set(deployment_only_source_files(root))
        leaked = sorted(
            relative
            for value in measured
            if (relative := _relative_measured_file(root, value)) in deployment_only
        )
        if leaked:
            raise CoverageGateError(
                "non-deployment coverage shard measured deploy-host-only source: "
                + ", ".join(leaked)
            )
    return measured


def verify_shard_gate(
    gate_path: Path,
    artifact_root: Path,
    *,
    expected_identity: str | None = None,
    expected_shard: str | None = None,
    require_trusted: bool = False,
    source_root: Path = ROOT,
    verify_coverage_payload: bool = False,
) -> dict[str, Any]:
    gate = _load_json(gate_path, "coverage shard gate is invalid")
    identity = gate.get("identity")
    shard = str(gate.get("shard") or "")
    identity_sha = str((identity or {}).get("sha256") or "")
    if (
        gate.get("schema_version") != SCHEMA_VERSION
        or gate.get("domain") != "coverage-shard"
        or shard not in SHARDS
        or gate.get("quality_gates") != _quality_gates(shard)
        or not isinstance(identity, dict)
        or identity.get("shard") != shard
        or canonical_sha256(
            {key: value for key, value in identity.items() if key != "sha256"}
        )
        != identity_sha
        or gate.get("artifact_name") != artifact_name(shard, identity_sha)
    ):
        raise CoverageGateError("coverage shard identity or quality gates are invalid")
    if expected_shard is not None and shard != expected_shard:
        raise CoverageGateError("coverage shard name does not match")
    if expected_identity is not None and identity_sha != expected_identity:
        raise CoverageGateError(
            "coverage shard does not match current content identity"
        )
    producer = gate.get("producer")
    if not isinstance(producer, dict):
        raise CoverageGateError("coverage shard producer identity is invalid")
    repository = str(producer.get("repository") or "")
    workflow_ref = str(producer.get("workflow_ref") or "")
    expected_workflow = f"{repository}/.github/workflows/ci.yml@refs/heads/main"
    trusted = bool(producer.get("trusted"))
    if (
        not workflow_ref.startswith(f"{repository}/.github/workflows/ci.yml@")
        or trusted != (workflow_ref == expected_workflow)
        or (require_trusted and not trusted)
        or not str(producer.get("run_id") or "").isdecimal()
        or not str(producer.get("git_commit") or "")
    ):
        raise CoverageGateError("coverage shard producer identity is invalid")
    verified = verify_evidence(
        artifact_root,
        gate.get("evidence"),
        expected_names=_evidence_names(shard),
    )
    coverage_path = artifact_root / str(verified["coverage_data"]["path"])
    if verify_coverage_payload:
        verify_coverage_data(source_root, shard, coverage_path)
    pytest_path = artifact_root / str(verified["pytest_results"]["path"])
    _verify_pytest_results(pytest_path)
    durations_path = artifact_root / str(verified["durations"]["path"])
    durations = _load_json(durations_path, "coverage duration evidence is invalid")
    if durations.get("schema_version") != 1 or durations.get("shard") != shard:
        raise CoverageGateError("coverage duration evidence is invalid")
    if shard == "postgres":
        stress_path = artifact_root / str(verified["postgres_stress_results"]["path"])
        _verify_pytest_results(stress_path)
    reused_from = gate.get("reused_from")
    if reused_from is not None and (
        not isinstance(reused_from, dict)
        or len(str(reused_from.get("gate_sha256") or "")) != 64
        or len(str(reused_from.get("identity_sha256") or "")) != 64
        or not str(reused_from.get("producer_git_commit") or "")
        or not str(reused_from.get("producer_run_id") or "").isdecimal()
    ):
        raise CoverageGateError("reused coverage shard provenance is invalid")
    return gate


def _is_ancestor(root: Path, commit: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
    )
    return completed.returncode == 0


def restore_reusable_shard(
    *,
    root: Path = ROOT,
    repository: str,
    token: str,
    identity: dict[str, Any],
    destination: Path,
    current_run_id: int | None,
) -> dict[str, object]:
    shard = str(identity["shard"])
    name = artifact_name(shard, str(identity["sha256"]))
    destination = _clean_directory(root, destination, "coverage restore destination")
    base = {
        "artifact_name": name,
        "identity": identity["sha256"],
        "reused": "false",
        "source_run_id": "",
    }
    try:
        found = find_reusable_artifact(
            repository=repository,
            token=token,
            name=name,
            current_run_id=current_run_id,
        )
    except (HTTPError, URLError) as exc:
        print(
            f"ci-coverage-gate: signed shard lookup unavailable; running fresh: {exc}",
            file=sys.stderr,
        )
        return base
    if found is None:
        return base
    artifact, run = found
    try:
        data = download(str(artifact["archive_download_url"]), token)
    except (HTTPError, URLError) as exc:
        print(
            f"ci-coverage-gate: signed shard download unavailable; running fresh: {exc}",
            file=sys.stderr,
        )
        return base
    expected_digest = str(artifact.get("digest") or "")
    if (
        expected_digest
        and expected_digest != "sha256:" + hashlib.sha256(data).hexdigest()
    ):
        raise CoverageGateError(
            "downloaded coverage shard artifact digest does not match"
        )
    with tempfile.TemporaryDirectory(prefix="gpu-fault-coverage-shard-") as directory:
        extracted = Path(directory)
        extract_archive(data, extracted)
        gates = list(extracted.rglob(GATE_NAME))
        if len(gates) != 1:
            raise CoverageGateError(
                "coverage shard artifact must contain exactly one gate"
            )
        source_root = gates[0].parent
        gate = verify_shard_gate(
            gates[0],
            source_root,
            expected_identity=str(identity["sha256"]),
            expected_shard=shard,
            require_trusted=True,
            source_root=root,
            verify_coverage_payload=True,
        )
        producer = gate["producer"]
        if int(producer["run_id"]) != int(run["id"]):
            raise CoverageGateError(
                "coverage shard producer run does not match artifact"
            )
        if not _is_ancestor(root, str(producer["git_commit"])):
            raise CoverageGateError("coverage shard producer commit is not an ancestor")
        shutil.copytree(source_root, destination, dirs_exist_ok=True)
        (destination / GATE_NAME).replace(destination / BASE_GATE_NAME)
        bundle = destination / BUNDLE_NAME
        if bundle.is_file():
            bundle.replace(destination / BASE_BUNDLE_NAME)
    return {
        **base,
        "reused": "true",
        "source_run_id": run["id"],
    }


def _copy_evidence(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def build_shard_gate(
    root: Path,
    artifact_root: Path,
    *,
    identity: dict[str, Any],
    coverage_data: Path,
    pytest_results: Path,
    durations: Path,
    stress_results: Path | None = None,
) -> dict[str, Any]:
    shard = str(identity["shard"])
    verify_coverage_data(root, shard, coverage_data)
    _verify_pytest_results(pytest_results)
    duration_value = _load_json(durations, "coverage duration evidence is invalid")
    if (
        duration_value.get("schema_version") != 1
        or duration_value.get("shard") != shard
    ):
        raise CoverageGateError("coverage duration evidence is invalid")
    if shard == "postgres":
        if stress_results is None:
            raise CoverageGateError("PostgreSQL stress evidence is required")
        _verify_pytest_results(stress_results)
    elif stress_results is not None:
        raise CoverageGateError("only PostgreSQL shard accepts stress evidence")
    artifact_root.mkdir(parents=True, exist_ok=True)
    copied = {
        "coverage_data": _copy_evidence(
            coverage_data, artifact_root / COVERAGE_DATA_NAME
        ),
        "durations": _copy_evidence(durations, artifact_root / DURATIONS_NAME),
        "pytest_results": _copy_evidence(
            pytest_results, artifact_root / PYTEST_RESULTS_NAME
        ),
    }
    if stress_results is not None:
        copied["postgres_stress_results"] = _copy_evidence(
            stress_results,
            artifact_root / STRESS_RESULTS_NAME,
        )
    repository = os.getenv("GITHUB_REPOSITORY", "")
    workflow_ref = os.getenv("GITHUB_WORKFLOW_REF", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    expected_workflow = f"{repository}/.github/workflows/ci.yml@refs/heads/main"
    if (
        not repository
        or not workflow_ref.startswith(f"{repository}/.github/workflows/ci.yml@")
        or not run_id.isdecimal()
    ):
        raise CoverageGateError("coverage shard gate must be built by the CI workflow")
    trusted = workflow_ref == expected_workflow
    reused_from = None
    base_gate_path = artifact_root / BASE_GATE_NAME
    if base_gate_path.is_file():
        base_gate = verify_shard_gate(
            base_gate_path,
            artifact_root,
            expected_identity=str(identity["sha256"]),
            expected_shard=shard,
            source_root=root,
            verify_coverage_payload=True,
        )
        reused_from = {
            "gate_sha256": sha256(base_gate_path),
            "identity_sha256": base_gate["identity"]["sha256"],
            "producer_git_commit": base_gate["producer"]["git_commit"],
            "producer_run_id": base_gate["producer"]["run_id"],
        }
    gate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": "coverage-shard",
        "shard": shard,
        "artifact_name": artifact_name(shard, str(identity["sha256"])),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "producer": {
            "git_commit": git(root, "rev-parse", "HEAD"),
            "git_tree": git(root, "rev-parse", "HEAD^{tree}"),
            "repository": repository,
            "run_id": run_id,
            "trusted": trusted,
            "workflow_ref": workflow_ref,
        },
        "quality_gates": _quality_gates(shard),
        "evidence": {
            name: evidence_entry(artifact_root, path)
            for name, path in sorted(copied.items())
        },
    }
    if reused_from is not None:
        gate["reused_from"] = reused_from
    target = artifact_root / GATE_NAME
    target.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    base_gate_path.unlink(missing_ok=True)
    (artifact_root / BASE_BUNDLE_NAME).unlink(missing_ok=True)
    return gate


def write_coverage_config(root: Path, shard: str, output: Path) -> None:
    config = load_config(root)
    lines = [
        "[run]",
        f"branch = {str(bool(config['coverage']['branch'])).lower()}",
        "relative_files = true",
        "source =",
        f"    {config['coverage']['source']}",
    ]
    if config["shards"][shard]["omit_deployment_source"]:
        lines.extend(["omit ="])
        lines.extend(
            f"    {relative}"
            for relative in deployment_only_source_files(root, config=config)
        )
    lines.extend(["", "[report]", "show_missing = true", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _run(command: list[str], *, root: Path, environment: Mapping[str, str]) -> float:
    print("+ " + shlex.join(command), flush=True)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=root, env=environment, check=False)
    elapsed = time.monotonic() - started
    if completed.returncode:
        raise CoverageGateError(
            f"coverage shard command failed with exit {completed.returncode}"
        )
    return elapsed


def _duration_records(path: Path) -> list[dict[str, object]]:
    value = _verify_pytest_results(path)
    records = []
    for nodeid, raw in value["records"].items():
        if not isinstance(raw, dict):
            raise CoverageGateError("pytest duration record is invalid")
        records.append(
            {
                "nodeid": str(nodeid),
                "duration_seconds": float(raw.get("duration_seconds") or 0.0),
                "status": str(raw.get("status") or ""),
            }
        )
    return sorted(
        records,
        key=lambda item: (-float(item["duration_seconds"]), str(item["nodeid"])),
    )


def write_durations(
    *,
    shard: str,
    pytest_results: Path,
    output: Path,
    wall_seconds: float,
    limit: int,
    stress_results: Path | None = None,
    stress_wall_seconds: float | None = None,
) -> None:
    records = _duration_records(pytest_results)
    stress = _duration_records(stress_results) if stress_results is not None else []
    value = {
        "schema_version": 1,
        "shard": shard,
        "pytest": {
            "record_count": len(records),
            "slowest": records[:limit],
            "wall_seconds": round(wall_seconds, 6),
        },
        "postgres_stress": (
            {
                "record_count": len(stress),
                "slowest": stress[:limit],
                "wall_seconds": round(float(stress_wall_seconds or 0.0), 6),
            }
            if stress_results is not None
            else None
        ),
    }
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_shard(
    *,
    root: Path,
    shard: str,
    python: str,
    artifact_root: Path,
    workers: str,
    distribution: str,
    durations: int,
    include_stress: bool,
) -> None:
    validate_test_partition(root)
    if shard == "postgres" and not os.getenv("GPU_FAULT_TEST_POSTGRES_URL", ""):
        raise CoverageGateError("GPU_FAULT_TEST_POSTGRES_URL is required")
    if include_stress and shard != "postgres":
        raise CoverageGateError("only PostgreSQL shard supports stress")
    artifact_root = _clean_directory(root, artifact_root, "coverage artifact root")
    coverage_config = artifact_root / "coverage.ini"
    coverage_data = artifact_root / COVERAGE_DATA_NAME
    pytest_results = artifact_root / PYTEST_RESULTS_NAME
    durations_path = artifact_root / DURATIONS_NAME
    write_coverage_config(root, shard, coverage_config)
    config = load_config(root)
    environment = dict(os.environ)
    environment["COVERAGE_FILE"] = str(coverage_data)
    environment["PYTEST_GPU_FAULT_CASE_REPORT"] = str(pytest_results)
    if shard != "postgres":
        environment["GPU_FAULT_TEST_POSTGRES_URL"] = ""
    if shard in RUNTIME_SHARDS:
        environment["PYTEST_GPU_FAULT_PARTITION_COUNT"] = str(
            config["protocol"]["runtime_partitions"]
        )
        environment["PYTEST_GPU_FAULT_PARTITION_INDEX"] = str(
            RUNTIME_SHARDS.index(shard)
        )
    else:
        environment.pop("PYTEST_GPU_FAULT_PARTITION_COUNT", None)
        environment.pop("PYTEST_GPU_FAULT_PARTITION_INDEX", None)
    command = [
        python,
        "-m",
        "pytest",
        *pytest_targets(root, shard),
    ]
    if shard != "postgres":
        command.extend(["-n", workers, f"--dist={distribution}"])
    command.extend(
        [
            "-p",
            "tools.pytest_case_reporter",
            f"--cov={config['coverage']['source']}",
            "--cov-branch",
            f"--cov-config={coverage_config}",
            "--cov-report=",
            f"--durations={durations}",
        ]
    )
    wall_seconds = _run(command, root=root, environment=environment)
    stress_results = None
    stress_wall = None
    if include_stress:
        stress_results = artifact_root / STRESS_RESULTS_NAME
        stress_environment = dict(os.environ)
        stress_environment.pop("PYTEST_GPU_FAULT_PARTITION_COUNT", None)
        stress_environment.pop("PYTEST_GPU_FAULT_PARTITION_INDEX", None)
        stress_environment["PYTEST_GPU_FAULT_CASE_REPORT"] = str(stress_results)
        stress_environment["GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS"] = str(
            config["protocol"]["postgres_stress_workers"]
        )
        stress_environment["GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS"] = str(
            config["protocol"]["postgres_stress_rounds"]
        )
        stress_command = [
            python,
            "-m",
            "pytest",
            *pytest_targets(root, "postgres"),
            "-p",
            "tools.pytest_case_reporter",
            f"--durations={durations}",
        ]
        stress_wall = _run(
            stress_command,
            root=root,
            environment=stress_environment,
        )
    verify_coverage_data(root, shard, coverage_data)
    write_durations(
        shard=shard,
        pytest_results=pytest_results,
        output=durations_path,
        wall_seconds=wall_seconds,
        limit=durations,
        stress_results=stress_results,
        stress_wall_seconds=stress_wall,
    )
    coverage_config.unlink()


def discover_shard_gates(
    shards_root: Path,
    *,
    source_root: Path = ROOT,
    verify_coverage_payload: bool = False,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for gate_path in sorted(shards_root.rglob(GATE_NAME)):
        gate = verify_shard_gate(
            gate_path,
            gate_path.parent,
            source_root=source_root,
            verify_coverage_payload=verify_coverage_payload,
        )
        shard = str(gate["shard"])
        if shard in result:
            raise CoverageGateError(f"duplicate coverage shard gate: {shard}")
        result[shard] = (gate_path, gate)
    if set(result) != set(SHARDS):
        raise CoverageGateError("coverage shard gate set is incomplete")
    return result


def verify_current_shards(
    root: Path,
    shards_root: Path,
    *,
    require_run_id: str | None = None,
    require_trusted: bool = False,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    gates = discover_shard_gates(
        shards_root,
        source_root=root,
        verify_coverage_payload=True,
    )
    for shard, (path, gate) in gates.items():
        identity = gate["identity"]
        current = shard_identity(
            root,
            shard,
            postgres_image=str(identity["environment"].get("postgres_image") or ""),
            distributions=identity["installed_distributions"],
            environment_identity=identity["environment"],
            pytest_workers=str(identity["protocol"]["pytest_workers"]),
        )
        verify_shard_gate(
            path,
            path.parent,
            expected_identity=str(current["sha256"]),
            expected_shard=shard,
            require_trusted=require_trusted,
            source_root=root,
        )
        if require_run_id is not None and str(gate["producer"]["run_id"]) != str(
            require_run_id
        ):
            raise CoverageGateError(
                f"coverage shard was not produced by current run: {shard}"
            )
    return gates


def _merge_pytest_results(
    root: Path,
    gates: Mapping[str, tuple[Path, dict[str, Any]]],
    output: Path,
) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for shard, (gate_path, gate) in sorted(gates.items()):
        evidence = gate["evidence"]["pytest_results"]
        path = gate_path.parent / str(evidence["path"])
        value = _verify_pytest_results(path)
        for nodeid, raw in value["records"].items():
            if nodeid in merged and merged[nodeid] != raw:
                raise CoverageGateError(
                    f"pytest result differs across shards: {nodeid}"
                )
            merged[str(nodeid)] = raw
    if __package__:
        from tools.pytest_result_identity import source_identity
    else:
        sys.path.insert(0, str(root))
        from tools.pytest_result_identity import source_identity
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_identity": source_identity(root),
                "records": dict(sorted(merged.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _aggregate_durations(
    gates: Mapping[str, tuple[Path, dict[str, Any]]],
    output: Path,
) -> dict[str, Any]:
    shards: dict[str, Any] = {}
    slowest: list[dict[str, object]] = []
    for shard, (gate_path, gate) in sorted(gates.items()):
        evidence = gate["evidence"]["durations"]
        value = _load_json(
            gate_path.parent / str(evidence["path"]),
            "coverage duration evidence is invalid",
        )
        shards[shard] = {
            "producer_run_id": gate["producer"]["run_id"],
            "reused": gate.get("reused_from") is not None,
            "pytest": value["pytest"],
            "postgres_stress": value.get("postgres_stress"),
        }
        for record in value["pytest"]["slowest"]:
            slowest.append({**record, "shard": shard, "suite": "pytest"})
        stress = value.get("postgres_stress")
        if isinstance(stress, dict):
            for record in stress["slowest"]:
                slowest.append({**record, "shard": shard, "suite": "postgres_stress"})
    slowest.sort(
        key=lambda item: (-float(item["duration_seconds"]), str(item["nodeid"]))
    )
    result = {
        "schema_version": 1,
        "shards": shards,
        "slowest": slowest[:50],
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _write_job_summary(value: Mapping[str, Any]) -> None:
    raw_path = os.getenv("GITHUB_STEP_SUMMARY", "")
    if not raw_path:
        return
    lines = [
        "### Coverage shard timings",
        "",
        "| Shard | Reused | Pytest wall | Stress wall |",
        "|---|---:|---:|---:|",
    ]
    for shard, raw in value["shards"].items():
        stress = raw.get("postgres_stress")
        lines.append(
            f"| `{shard}` | {str(bool(raw['reused'])).lower()} | "
            f"{float(raw['pytest']['wall_seconds']):.1f}s | "
            f"{float((stress or {}).get('wall_seconds') or 0.0):.1f}s |"
        )
    lines.extend(["", "Slowest tests are retained in `test-durations.json`.", ""])
    with Path(raw_path).open("a", encoding="utf-8") as destination:
        destination.write("\n".join(lines))


def combine_shards(
    *,
    root: Path,
    shards_root: Path,
    output_root: Path,
    python: str,
    require_run_id: str | None,
) -> None:
    gates = verify_current_shards(
        root,
        shards_root,
        require_run_id=require_run_id,
    )
    resolved_output = output_root.resolve()
    resolved_shards = shards_root.resolve()
    if (
        resolved_output == resolved_shards
        or resolved_output.is_relative_to(resolved_shards)
        or resolved_shards.is_relative_to(resolved_output)
    ):
        raise CoverageGateError("coverage output root must not contain shard inputs")
    output_root = _clean_directory(root, output_root, "coverage output root")
    combine_root = output_root / "coverage-parts"
    combine_root.mkdir()
    for shard, (gate_path, gate) in gates.items():
        evidence = gate["evidence"]["coverage_data"]
        shutil.copy2(
            gate_path.parent / str(evidence["path"]),
            combine_root / f"coverage-data.{shard}",
        )
    final_config = output_root / "coverage.ini"
    config = load_config(root)
    final_config.write_text(
        "\n".join(
            [
                "[run]",
                f"branch = {str(bool(config['coverage']['branch'])).lower()}",
                "relative_files = true",
                "source =",
                f"    {config['coverage']['source']}",
                "",
                "[report]",
                "show_missing = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["COVERAGE_FILE"] = str(output_root / "coverage-data")
    _run(
        [python, "-m", "coverage", "combine", "--keep", str(combine_root)],
        root=root,
        environment=environment,
    )
    _run(
        [
            python,
            "-m",
            "coverage",
            "report",
            f"--rcfile={final_config}",
            f"--fail-under={config['coverage']['floor']}",
        ],
        root=root,
        environment=environment,
    )
    _run(
        [
            python,
            "-m",
            "coverage",
            "json",
            f"--rcfile={final_config}",
            "-o",
            str(output_root / "coverage.json"),
        ],
        root=root,
        environment=environment,
    )
    _merge_pytest_results(
        root,
        gates,
        output_root / "pytest-case-results.json",
    )
    durations = _aggregate_durations(
        gates,
        output_root / "test-durations.json",
    )
    _write_job_summary(durations)
    shutil.rmtree(combine_root)
    final_config.unlink()


def _identity_from_options(options: argparse.Namespace) -> dict[str, Any]:
    image = ""
    if options.shard == "postgres":
        image = postgres_image_identity(
            container=options.postgres_container,
            image=options.postgres_image,
        )
    return shard_identity(
        ROOT,
        options.shard,
        postgres_image=image,
        pytest_workers=options.pytest_workers,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--github-output", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--shard", choices=SHARDS, required=True)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--workers", default=os.getenv("PYTEST_XDIST_WORKERS", "4"))
    run.add_argument("--dist", default=os.getenv("PYTEST_XDIST_DIST", "worksteal"))
    run.add_argument("--durations", type=int, default=50)
    run.add_argument("--include-stress", action="store_true")
    combine = commands.add_parser("combine")
    combine.add_argument("--python", default=sys.executable)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--output-root", type=Path, required=True)
    combine.add_argument("--require-run-id")
    for name in ("identity", "restore", "build", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--shard", choices=SHARDS, required=True)
        command.add_argument("--postgres-container")
        command.add_argument("--postgres-image")
        command.add_argument("--pytest-workers")
    identity_parser = commands.choices["identity"]
    identity_parser.add_argument("--github-output", type=Path)
    restore = commands.choices["restore"]
    restore.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY"))
    restore.add_argument("--token-env", default="GITHUB_TOKEN")
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument("--current-run-id", type=int)
    restore.add_argument("--github-output", type=Path)
    build = commands.choices["build"]
    build.add_argument("--artifact-root", type=Path, required=True)
    build.add_argument("--coverage-data", type=Path, required=True)
    build.add_argument("--pytest-results", type=Path, required=True)
    build.add_argument("--durations", type=Path, required=True)
    build.add_argument("--stress-results", type=Path)
    build.add_argument("--github-output", type=Path)
    verify = commands.choices["verify"]
    verify.add_argument("--artifact-root", type=Path, required=True)
    verify.add_argument("--gate", type=Path, required=True)
    verify.add_argument("--expected-identity")
    options = parser.parse_args(arguments)
    try:
        if options.command == "validate":
            counts = validate_test_partition(ROOT)
            write_outputs(
                options.github_output,
                {f"{name}_test_files": value for name, value in counts.items()},
            )
            print(json.dumps(counts, sort_keys=True))
        elif options.command == "run":
            run_shard(
                root=ROOT,
                shard=options.shard,
                python=options.python,
                artifact_root=options.artifact_root.resolve(),
                workers=options.workers,
                distribution=options.dist,
                durations=options.durations,
                include_stress=options.include_stress,
            )
        elif options.command == "combine":
            combine_shards(
                root=ROOT,
                shards_root=options.shards_root.resolve(),
                output_root=options.output_root.resolve(),
                python=options.python,
                require_run_id=options.require_run_id,
            )
        elif options.command == "identity":
            identity = _identity_from_options(options)
            outputs = {
                "artifact_name": artifact_name(options.shard, str(identity["sha256"])),
                "identity": identity["sha256"],
            }
            write_outputs(options.github_output, outputs)
            print(json.dumps(outputs, sort_keys=True))
        elif options.command == "restore":
            token = os.getenv(options.token_env, "")
            if not options.repository or not token:
                raise CoverageGateError("repository and GitHub token are required")
            identity = _identity_from_options(options)
            outputs = restore_reusable_shard(
                repository=options.repository,
                token=token,
                identity=identity,
                destination=options.destination.resolve(),
                current_run_id=options.current_run_id,
            )
            write_outputs(options.github_output, outputs)
            print(json.dumps(outputs, sort_keys=True))
        elif options.command == "build":
            identity = _identity_from_options(options)
            gate = build_shard_gate(
                ROOT,
                options.artifact_root.resolve(),
                identity=identity,
                coverage_data=options.coverage_data.resolve(),
                pytest_results=options.pytest_results.resolve(),
                durations=options.durations.resolve(),
                stress_results=(
                    options.stress_results.resolve()
                    if options.stress_results is not None
                    else None
                ),
            )
            outputs = {
                "artifact_name": gate["artifact_name"],
                "identity": identity["sha256"],
                "reused": str(gate.get("reused_from") is not None).lower(),
            }
            write_outputs(options.github_output, outputs)
            print(json.dumps(outputs, sort_keys=True))
        else:
            expected = options.expected_identity or str(
                _identity_from_options(options)["sha256"]
            )
            verify_shard_gate(
                options.gate.resolve(),
                options.artifact_root.resolve(),
                expected_identity=expected,
                expected_shard=options.shard,
                source_root=ROOT,
                verify_coverage_payload=True,
            )
    except (
        CoverageGateError,
        GateArtifactError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ci-coverage-gate: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
