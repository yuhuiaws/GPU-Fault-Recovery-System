from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/ci-unit-gate.json"
SCHEMA_VERSION = 1
GATE_NAME = "unit-gate.json"
BUNDLE_NAME = "unit-gate.bundle.json"
BASE_GATE_NAME = "base-unit-gate.json"
BASE_BUNDLE_NAME = "base-unit-gate.bundle.json"
DELTA_PLAN_NAME = "delta-plan.json"
PYTEST_RESULTS_NAME = "pytest-case-results.json"
FAULT_REPORT_NAME = "fault-report.json"
ARTIFACT_PREFIX = "gpu-fault-unit-gate-"
QUALITY_GATES = {
    "coverage": "PASSED",
    "fault_catalog": "PASSED",
    "postgres_contract": "PASSED",
    "postgres_stress": "PASSED",
}


class UnitGateError(RuntimeError):
    pass


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise UnitGateError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def load_config(root: Path = ROOT) -> dict[str, Any]:
    try:
        value = json.loads((root / CONFIG_PATH.relative_to(ROOT)).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UnitGateError("unit gate identity config is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("domain") != "unit"
        or not isinstance(value.get("exclude_prefixes"), list)
        or not isinstance(value.get("exclude_globs"), list)
        or not isinstance(value.get("exclude_files"), list)
        or not isinstance(value.get("protocol"), dict)
        or not isinstance(value.get("reuse"), dict)
        or not isinstance(value["reuse"].get("coverage_groups"), list)
        or not isinstance(value["reuse"].get("static_only_domains"), list)
    ):
        raise UnitGateError("unit gate identity config is incomplete")
    return value


def _repository_files(root: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise UnitGateError("cannot enumerate unit gate inputs")
    result = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if relative.is_absolute() or ".." in relative.parts:
            raise UnitGateError(f"unit gate input leaves repository: {relative}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise UnitGateError(f"unsupported unit gate input: {relative}")
        result.append(path)
    return tuple(sorted(result))


def _is_excluded(relative: str, config: Mapping[str, Any]) -> bool:
    return (
        relative in set(config["exclude_files"])
        or any(
            relative.startswith(str(prefix)) for prefix in config["exclude_prefixes"]
        )
        or any(
            fnmatch.fnmatchcase(relative, str(pattern))
            for pattern in config["exclude_globs"]
        )
    )


def _entry(root: Path, path: Path) -> tuple[str, dict[str, object]]:
    relative = path.relative_to(root).as_posix()
    return relative, {
        "mode": f"{path.stat().st_mode & 0o777:04o}",
        "sha256": _sha256(path),
    }


def _group_identity(entries: Mapping[str, dict[str, object]]) -> dict[str, object]:
    ordered = dict(sorted(entries.items()))
    return {
        "sha256": _canonical_sha256(ordered),
        "file_count": len(ordered),
        "files": ordered,
    }


def _identity_group(relative: str) -> str:
    if (
        relative == "pyproject.toml"
        or relative == "uv.lock"
        or relative.startswith("requirements/")
    ):
        return "dependencies"
    if relative in {
        "testcases/fault-scenarios.yaml",
        "testcases/regional-execution-order.yaml",
        "tests/test_fault_scenario_catalog.py",
        "tools/case_scheduler.py",
        "tools/pytest_case_reporter.py",
        "tools/pytest_result_identity.py",
        "tools/run_fault_test_cases.py",
    }:
        return "fault_runner"
    if (
        relative.startswith(("deploy/", "tests/admin/"))
        or relative.startswith("src/gpu_fault/admin_")
        or relative.startswith("src/gpu_fault/installation_")
        or relative.startswith("tests/regional/test_regional_admin_")
        or "release" in Path(relative).name
        or "deploy_host" in Path(relative).name
        or "staging_deploy" in Path(relative).name
    ):
        return "deployment"
    if relative.startswith(("tests/", "testcases/")):
        return "tests"
    return "runtime"


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
            raise UnitGateError(f"installed distribution versions differ: {name}")
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
        raise UnitGateError("PostgreSQL container or image identity is required")
    completed = subprocess.run(
        ["docker", "inspect", "--format={{.Image}}", container],
        text=True,
        capture_output=True,
        check=False,
    )
    value = completed.stdout.strip()
    if completed.returncode or not value:
        raise UnitGateError("cannot resolve PostgreSQL image identity")
    return value


def unit_identity(
    root: Path,
    *,
    postgres_image: str,
    distributions: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    groups: dict[str, dict[str, dict[str, object]]] = {
        "deployment": {},
        "dependencies": {},
        "fault_runner": {},
        "runtime": {},
        "tests": {},
    }
    for path in _repository_files(root):
        relative = path.relative_to(root).as_posix()
        if _is_excluded(relative, config):
            continue
        name, value = _entry(root, path)
        groups[_identity_group(relative)][name] = value
    if any(not entries for entries in groups.values()):
        raise UnitGateError("unit gate identity group is empty")
    current_environment = os.environ if environment is None else environment
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": "unit",
        "groups": {
            name: _group_identity(entries) for name, entries in sorted(groups.items())
        },
        "environment": {
            "machine": platform.machine(),
            "postgres_image": postgres_image,
            "python_cache_tag": sys.implementation.cache_tag,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "runner_image_os": current_environment.get("ImageOS", ""),
            "runner_image_version": current_environment.get("ImageVersion", ""),
            "sysconfig_platform": sysconfig.get_platform(),
        },
        "installed_distributions": dict(
            sorted((distributions or installed_distributions()).items())
        ),
        "protocol": config["protocol"],
    }
    coverage_payload = {
        "environment": payload["environment"],
        "groups": {
            name: payload["groups"][name] for name in config["reuse"]["coverage_groups"]
        },
        "installed_distributions": payload["installed_distributions"],
        "protocol": payload["protocol"],
        "schema_version": SCHEMA_VERSION,
    }
    payload["coverage_sha256"] = _canonical_sha256(coverage_payload)
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def artifact_name(coverage_sha256: str) -> str:
    if len(coverage_sha256) != 64:
        raise UnitGateError("unit gate identity SHA-256 is invalid")
    return ARTIFACT_PREFIX + coverage_sha256


def _write_outputs(path: Path | None, values: Mapping[str, object]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as destination:
        for name, value in values.items():
            destination.write(f"{name}={value}\n")


def _api_json(url: str, token: str) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _download(url: str, token: str) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=60) as response:
        return response.read()


def _successful_main_run(
    *,
    repository: str,
    token: str,
    run_id: int,
) -> dict[str, Any] | None:
    run = _api_json(
        f"https://api.github.com/repos/{quote(repository, safe='/')}/"
        f"actions/runs/{run_id}",
        token,
    )
    if not isinstance(run, dict):
        raise UnitGateError("GitHub workflow run response is invalid")
    return (
        run
        if run.get("status") == "completed"
        and run.get("conclusion") == "success"
        and run.get("event") == "push"
        and run.get("head_branch") == "main"
        and run.get("path") == ".github/workflows/ci.yml"
        else None
    )


def find_reusable_artifact(
    *,
    repository: str,
    token: str,
    name: str,
    current_run_id: int | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    value = _api_json(
        f"https://api.github.com/repos/{quote(repository, safe='/')}/"
        f"actions/artifacts?name={quote(name)}&per_page=100",
        token,
    )
    artifacts = value.get("artifacts") if isinstance(value, dict) else None
    if not isinstance(artifacts, list):
        raise UnitGateError("GitHub artifact response is invalid")
    for artifact in sorted(
        (item for item in artifacts if isinstance(item, dict)),
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    ):
        workflow_run = artifact.get("workflow_run")
        run_id = int((workflow_run or {}).get("id") or 0)
        if (
            artifact.get("expired") is True
            or not run_id
            or run_id == current_run_id
            or (workflow_run or {}).get("head_branch") != "main"
        ):
            continue
        run = _successful_main_run(
            repository=repository,
            token=token,
            run_id=run_id,
        )
        if run is not None:
            return artifact, run
    return None


def extract_unit_gate_archive(data: bytes, destination: Path) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                path = Path(member.filename)
                mode = member.external_attr >> 16
                if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
                    raise UnitGateError(
                        f"unsafe unit gate artifact member: {member.filename}"
                    )
            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise UnitGateError("unit gate artifact archive is invalid") from exc


def _evidence_entry(root: Path, path: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise UnitGateError(f"unit gate evidence leaves artifact root: {path}") from exc
    if not resolved.is_file():
        raise UnitGateError(f"unit gate evidence is missing: {path}")
    return {
        "path": relative,
        "sha256": _sha256(resolved),
        "size": resolved.stat().st_size,
    }


def verify_unit_gate(
    gate_path: Path,
    artifact_root: Path,
    *,
    expected_identity: str | None = None,
    expected_coverage_identity: str | None = None,
) -> dict[str, Any]:
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnitGateError("unit gate is invalid") from exc
    identity = gate.get("identity") if isinstance(gate, dict) else None
    identity_sha = str((identity or {}).get("sha256") or "")
    coverage_sha = str((identity or {}).get("coverage_sha256") or "")
    if (
        not isinstance(gate, dict)
        or gate.get("schema_version") != SCHEMA_VERSION
        or gate.get("domain") != "unit"
        or not isinstance(identity, dict)
        or _canonical_sha256(
            {key: value for key, value in identity.items() if key != "sha256"}
        )
        != identity_sha
        or len(coverage_sha) != 64
        or gate.get("artifact_name") != artifact_name(coverage_sha)
        or gate.get("quality_gates") != QUALITY_GATES
    ):
        raise UnitGateError("unit gate identity or quality gates are invalid")
    if expected_identity is not None and identity_sha != expected_identity:
        raise UnitGateError("unit gate does not match current unit content identity")
    if (
        expected_coverage_identity is not None
        and coverage_sha != expected_coverage_identity
    ):
        raise UnitGateError(
            "unit gate does not match current coverage content identity"
        )
    producer = gate.get("producer")
    expected_workflow = (
        f"{(producer or {}).get('repository')}/.github/workflows/ci.yml@refs/heads/main"
    )
    if (
        not isinstance(producer, dict)
        or producer.get("workflow_ref") != expected_workflow
        or not str(producer.get("run_id") or "").isdecimal()
        or not str(producer.get("git_commit") or "")
    ):
        raise UnitGateError("unit gate producer identity is invalid")
    evidence = gate.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "fault_report",
        "pytest_results",
    }:
        raise UnitGateError("unit gate evidence inventory is invalid")
    for name, raw in evidence.items():
        if not isinstance(raw, dict):
            raise UnitGateError(f"unit gate evidence entry is invalid: {name}")
        path = (artifact_root / str(raw.get("path") or "")).resolve()
        try:
            path.relative_to(artifact_root.resolve())
        except ValueError as exc:
            raise UnitGateError(
                f"unit gate evidence leaves artifact root: {name}"
            ) from exc
        if (
            not path.is_file()
            or _sha256(path) != raw.get("sha256")
            or path.stat().st_size != raw.get("size")
        ):
            raise UnitGateError(f"unit gate evidence does not match: {name}")
    reused_from = gate.get("reused_from")
    if reused_from is not None and (
        not isinstance(reused_from, dict)
        or not isinstance(gate.get("delta"), dict)
        or len(str(reused_from.get("gate_sha256") or "")) != 64
        or len(str(reused_from.get("identity_sha256") or "")) != 64
        or not str(reused_from.get("producer_git_commit") or "")
        or not str(reused_from.get("producer_run_id") or "").isdecimal()
    ):
        raise UnitGateError("reused unit gate provenance is invalid")
    return gate


def _is_ancestor(root: Path, commit: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
    )
    return completed.returncode == 0


def change_impact_plan(root: Path, base: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/select-affected-tests.py"),
            "--base",
            base,
            "--format",
            "json",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise UnitGateError("change impact plan is invalid") from exc
    if (
        completed.returncode
        or not isinstance(value, dict)
        or not isinstance(value.get("domains"), list)
        or not isinstance(value.get("full"), bool)
        or not isinstance(value.get("postgres"), bool)
    ):
        raise UnitGateError("change impact plan is incomplete")
    return value


def delta_required_for_plan(
    plan: Mapping[str, Any],
    *,
    static_only_domains: set[str],
) -> bool | None:
    if plan.get("full") is True or plan.get("postgres") is True:
        return None
    domains = plan.get("domains")
    if not isinstance(domains, list) or any(
        not isinstance(item, str) for item in domains
    ):
        raise UnitGateError("change impact plan domains are invalid")
    return bool(set(domains) - static_only_domains)


def restore_reusable_gate(
    *,
    root: Path = ROOT,
    repository: str,
    token: str,
    identity: dict[str, Any],
    destination: Path,
    current_run_id: int | None,
) -> dict[str, object]:
    name = artifact_name(str(identity["coverage_sha256"]))
    found = find_reusable_artifact(
        repository=repository,
        token=token,
        name=name,
        current_run_id=current_run_id,
    )
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    if found is None:
        return {
            "artifact_name": name,
            "identity": identity["sha256"],
            "coverage_identity": identity["coverage_sha256"],
            "base_commit": "",
            "delta_required": "false",
            "reused": "false",
            "source_run_id": "",
        }
    artifact, run = found
    data = _download(str(artifact["archive_download_url"]), token)
    expected_digest = str(artifact.get("digest") or "")
    if (
        expected_digest
        and expected_digest != "sha256:" + hashlib.sha256(data).hexdigest()
    ):
        raise UnitGateError("downloaded unit gate artifact digest does not match")
    with tempfile.TemporaryDirectory(prefix="gpu-fault-unit-gate-") as directory:
        extracted = Path(directory)
        extract_unit_gate_archive(data, extracted)
        gates = list(extracted.rglob(GATE_NAME))
        if len(gates) != 1:
            raise UnitGateError("unit gate artifact must contain exactly one gate")
        source_root = gates[0].parent
        verify_unit_gate(
            gates[0],
            source_root,
            expected_coverage_identity=str(identity["coverage_sha256"]),
        )
        producer = json.loads(gates[0].read_text(encoding="utf-8"))["producer"]
        if int(producer["run_id"]) != int(run["id"]):
            raise UnitGateError("unit gate producer run does not match artifact")
        base_commit = str(producer["git_commit"])
        if not _is_ancestor(root, base_commit):
            raise UnitGateError("unit gate producer commit is not an ancestor")
        plan = change_impact_plan(root, base_commit)
        static_only = set(load_config(root)["reuse"]["static_only_domains"])
        delta_required = delta_required_for_plan(
            plan,
            static_only_domains=static_only,
        )
        if delta_required is None:
            return {
                "artifact_name": name,
                "identity": identity["sha256"],
                "coverage_identity": identity["coverage_sha256"],
                "base_commit": "",
                "delta_required": "false",
                "reused": "false",
                "source_run_id": "",
            }
        shutil.copytree(source_root, destination, dirs_exist_ok=True)
        (destination / GATE_NAME).replace(destination / BASE_GATE_NAME)
        bundle = destination / BUNDLE_NAME
        if bundle.is_file():
            bundle.replace(destination / BASE_BUNDLE_NAME)
        (destination / DELTA_PLAN_NAME).write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "artifact_name": name,
        "identity": identity["sha256"],
        "coverage_identity": identity["coverage_sha256"],
        "base_commit": base_commit,
        "delta_required": str(delta_required).lower(),
        "reused": "true",
        "source_run_id": run["id"],
    }


def _copy_evidence(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def build_unit_gate(
    root: Path,
    artifact_root: Path,
    *,
    identity: dict[str, Any],
    pytest_results: Path,
    fault_report: Path,
) -> dict[str, Any]:
    try:
        pytest_value = json.loads(pytest_results.read_text(encoding="utf-8"))
        fault_value = json.loads(fault_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnitGateError("unit gate evidence is invalid") from exc
    if (
        not isinstance(pytest_value, dict)
        or pytest_value.get("schema_version") != 1
        or not isinstance(pytest_value.get("records"), dict)
        or not isinstance(fault_value, dict)
        or fault_value.get("schema_version") != 2
        or fault_value.get("verdict") != "PASS"
    ):
        raise UnitGateError("unit gate evidence did not pass")
    artifact_root.mkdir(parents=True, exist_ok=True)
    pytest_copy = _copy_evidence(
        pytest_results,
        artifact_root / PYTEST_RESULTS_NAME,
    )
    fault_copy = _copy_evidence(
        fault_report,
        artifact_root / FAULT_REPORT_NAME,
    )
    repository = os.getenv("GITHUB_REPOSITORY", "")
    workflow_ref = os.getenv("GITHUB_WORKFLOW_REF", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    expected_workflow = f"{repository}/.github/workflows/ci.yml@refs/heads/main"
    if not repository or workflow_ref != expected_workflow or not run_id.isdecimal():
        raise UnitGateError("unit gate must be built by the main CI workflow")
    reused_from = None
    base_gate_path = artifact_root / BASE_GATE_NAME
    delta_path = artifact_root / DELTA_PLAN_NAME
    if base_gate_path.is_file():
        base_gate = verify_unit_gate(
            base_gate_path,
            artifact_root,
            expected_coverage_identity=str(identity["coverage_sha256"]),
        )
        if not delta_path.is_file():
            raise UnitGateError("reused unit gate has no delta plan")
        try:
            delta_plan = json.loads(delta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UnitGateError("reused unit gate delta plan is invalid") from exc
        if not isinstance(delta_plan, dict):
            raise UnitGateError("reused unit gate delta plan is invalid")
        reused_from = {
            "gate_sha256": _sha256(base_gate_path),
            "identity_sha256": base_gate["identity"]["sha256"],
            "producer_git_commit": base_gate["producer"]["git_commit"],
            "producer_run_id": base_gate["producer"]["run_id"],
        }
    gate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": "unit",
        "artifact_name": artifact_name(str(identity["coverage_sha256"])),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "producer": {
            "git_commit": _git(root, "rev-parse", "HEAD"),
            "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
            "repository": repository,
            "run_id": run_id,
            "workflow_ref": workflow_ref,
        },
        "quality_gates": dict(QUALITY_GATES),
        "evidence": {
            "fault_report": _evidence_entry(artifact_root, fault_copy),
            "pytest_results": _evidence_entry(artifact_root, pytest_copy),
        },
    }
    if reused_from is not None:
        gate["reused_from"] = reused_from
        gate["delta"] = delta_plan
    target = artifact_root / GATE_NAME
    target.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    base_gate_path.unlink(missing_ok=True)
    (artifact_root / BASE_BUNDLE_NAME).unlink(missing_ok=True)
    return gate


def _identity_from_options(options: argparse.Namespace) -> dict[str, Any]:
    image = postgres_image_identity(
        container=options.postgres_container,
        image=options.postgres_image,
    )
    return unit_identity(ROOT, postgres_image=image)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("identity", "restore", "build", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--postgres-container")
        command.add_argument("--postgres-image")
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
    build.add_argument("--pytest-results", type=Path, required=True)
    build.add_argument("--fault-report", type=Path, required=True)
    build.add_argument("--github-output", type=Path)
    verify = commands.choices["verify"]
    verify.add_argument("--artifact-root", type=Path, required=True)
    verify.add_argument("--gate", type=Path, required=True)
    verify.add_argument("--expected-identity")
    options = parser.parse_args(arguments)
    try:
        if options.command == "identity":
            identity = _identity_from_options(options)
            outputs = {
                "artifact_name": artifact_name(str(identity["coverage_sha256"])),
                "coverage_identity": identity["coverage_sha256"],
                "identity": identity["sha256"],
            }
            _write_outputs(options.github_output, outputs)
            print(json.dumps(outputs, sort_keys=True))
        elif options.command == "restore":
            token = os.getenv(options.token_env, "")
            if not options.repository or not token:
                raise UnitGateError("repository and GitHub token are required")
            identity = _identity_from_options(options)
            outputs = restore_reusable_gate(
                repository=options.repository,
                token=token,
                identity=identity,
                destination=options.destination.resolve(),
                current_run_id=options.current_run_id,
            )
            _write_outputs(options.github_output, outputs)
            print(json.dumps(outputs, sort_keys=True))
        elif options.command == "build":
            identity = _identity_from_options(options)
            gate = build_unit_gate(
                ROOT,
                options.artifact_root.resolve(),
                identity=identity,
                pytest_results=options.pytest_results.resolve(),
                fault_report=options.fault_report.resolve(),
            )
            outputs = {
                "artifact_name": gate["artifact_name"],
                "coverage_identity": identity["coverage_sha256"],
                "identity": identity["sha256"],
                "reused": "false",
                "source_run_id": os.getenv("GITHUB_RUN_ID", ""),
            }
            _write_outputs(options.github_output, outputs)
            print(json.dumps(outputs, sort_keys=True))
        else:
            expected = options.expected_identity
            if options.postgres_container or options.postgres_image:
                expected = str(_identity_from_options(options)["sha256"])
            verify_unit_gate(
                options.gate.resolve(),
                options.artifact_root.resolve(),
                expected_identity=expected,
            )
    except (
        OSError,
        UnitGateError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"ci-unit-gate: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
