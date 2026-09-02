from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

if __package__:
    from scripts.ci_unit_gate import verify_unit_gate
else:
    from ci_unit_gate import verify_unit_gate


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
QUALITY_GATES = {
    "artifact": "PASSED",
    "coverage": "PASSED",
    "fault_catalog": "PASSED",
    "postgres_contract": "PASSED",
    "postgres_stress": "PASSED",
    "static": "PASSED",
}


class CiGateError(RuntimeError):
    pass


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise CiGateError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(dist: Path) -> dict[str, dict[str, Any]]:
    if not dist.is_dir():
        raise CiGateError(f"candidate dist directory is missing: {dist}")
    files = {}
    for path in sorted(dist.rglob("*")):
        if path.is_symlink():
            raise CiGateError(f"candidate artifact must not be a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(dist).as_posix()
        if relative in {"ci-gate.json", "ci-gate.bundle.json"}:
            continue
        files[relative] = {
            "mode": f"{path.stat().st_mode & 0o777:04o}",
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
    if not files:
        raise CiGateError("candidate dist directory is empty")
    return files


def _candidate_release(dist: Path) -> tuple[dict[str, Any], Path]:
    current = dist / "current-release.json"
    try:
        raw = current.read_bytes()
        release = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise CiGateError("CI candidate release manifest is invalid") from exc
    if (
        not isinstance(release, dict)
        or release.get("schema_version") != 3
        or release.get("deployable") is not False
    ):
        raise CiGateError("CI candidate must contain a source-only schema v3 release")
    release_id = str(release.get("release_id") or "")
    immutable = dist / release_id / "release.json"
    if not release_id or not immutable.is_file() or immutable.read_bytes() != raw:
        raise CiGateError(
            "CI candidate release differs from its content-addressed copy"
        )
    return release, current


def _unit_domain(dist: Path, path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        relative = path.relative_to(dist.resolve()).as_posix()
    except ValueError as exc:
        raise CiGateError("unit domain gate leaves candidate dist") from exc
    unit = verify_unit_gate(path, path.parent)
    producer = unit["producer"]
    return {
        "path": relative,
        "sha256": _sha256(path),
        "identity_sha256": unit["identity"]["sha256"],
        "coverage_identity_sha256": unit["identity"]["coverage_sha256"],
        "producer_run_id": producer["run_id"],
        "producer_git_commit": producer["git_commit"],
        "reused": bool(unit["reuse"]["reused_shards"]),
        "reused_shards": unit["reuse"]["reused_shards"],
    }


def build_gate(dist: Path, unit_gate_path: Path) -> dict[str, Any]:
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise CiGateError("CI gate requires a clean source tree")
    repository = os.getenv("GITHUB_REPOSITORY", "")
    workflow_ref = os.getenv("GITHUB_WORKFLOW_REF", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    expected_workflow_ref = f"{repository}/.github/workflows/ci.yml@refs/heads/main"
    if (
        not repository
        or workflow_ref != expected_workflow_ref
        or not run_id.isdecimal()
    ):
        raise CiGateError("CI gate must be built by the main CI workflow")
    release, current = _candidate_release(dist)
    unit = _unit_domain(dist, unit_gate_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "workflow_ref": workflow_ref,
        "run_id": run_id,
        "source": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_tree": _git("rev-parse", "HEAD^{tree}"),
        },
        "candidate": {
            "release_id": release.get("release_id"),
            "manifest_sha256": _sha256(current),
            "files": artifact_inventory(dist),
        },
        "domains": {"unit": unit},
        "quality_gates": dict(QUALITY_GATES),
    }


def verify_gate(path: Path, dist: Path) -> dict[str, Any]:
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CiGateError("CI gate is invalid") from exc
    if not isinstance(gate, dict) or gate.get("schema_version") != SCHEMA_VERSION:
        raise CiGateError("CI gate schema_version is invalid")
    if gate.get("quality_gates") != QUALITY_GATES:
        raise CiGateError("CI gate quality gates are incomplete")
    source = gate.get("source")
    if not isinstance(source, dict):
        raise CiGateError("CI gate source identity is missing")
    if source.get("git_commit") != _git("rev-parse", "HEAD") or source.get(
        "git_tree"
    ) != _git("rev-parse", "HEAD^{tree}"):
        raise CiGateError("CI gate source identity does not match checkout")
    repository = os.getenv("GITHUB_REPOSITORY", "")
    if repository and gate.get("repository") != repository:
        raise CiGateError("CI gate repository does not match")
    expected_workflow_ref = (
        f"{gate.get('repository')}/.github/workflows/ci.yml@refs/heads/main"
    )
    if (
        gate.get("workflow_ref") != expected_workflow_ref
        or not str(gate.get("run_id") or "").isdecimal()
    ):
        raise CiGateError("CI gate workflow identity is invalid")
    candidate = gate.get("candidate")
    if not isinstance(candidate, dict):
        raise CiGateError("CI gate candidate identity is missing")
    release, current = _candidate_release(dist)
    if (
        candidate.get("manifest_sha256") != _sha256(current)
        or candidate.get("files") != artifact_inventory(dist)
        or candidate.get("release_id") != release.get("release_id")
    ):
        raise CiGateError("CI gate candidate artifacts do not match")
    domains = gate.get("domains")
    unit = domains.get("unit") if isinstance(domains, dict) else None
    if not isinstance(unit, dict):
        raise CiGateError("CI gate unit domain is missing")
    unit_path = (dist / str(unit.get("path") or "")).resolve()
    try:
        unit_path.relative_to(dist.resolve())
    except ValueError as exc:
        raise CiGateError("CI gate unit domain leaves candidate dist") from exc
    if not unit_path.is_file() or _sha256(unit_path) != unit.get("sha256"):
        raise CiGateError("CI gate unit domain artifact does not match")
    unit_gate = verify_unit_gate(unit_path, unit_path.parent)
    if (
        unit.get("identity_sha256") != unit_gate["identity"]["sha256"]
        or unit.get("coverage_identity_sha256")
        != unit_gate["identity"]["coverage_sha256"]
        or str(unit.get("producer_run_id") or "")
        != str(unit_gate["producer"]["run_id"])
        or unit.get("producer_git_commit") != unit_gate["producer"]["git_commit"]
        or unit.get("reused") != bool(unit_gate["reuse"]["reused_shards"])
        or unit.get("reused_shards") != unit_gate["reuse"]["reused_shards"]
    ):
        raise CiGateError("CI gate unit domain identity does not match")
    return gate


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--dist", type=Path, default=ROOT / "dist")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--unit-gate", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--dist", type=Path, default=ROOT / "dist")
    verify.add_argument("--gate", type=Path, required=True)
    options = parser.parse_args(arguments)
    try:
        if options.command == "build":
            value = build_gate(
                options.dist.resolve(),
                options.unit_gate.resolve(),
            )
            options.output.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            verify_gate(options.gate.resolve(), options.dist.resolve())
    except (CiGateError, OSError, subprocess.SubprocessError) as exc:
        print(f"ci-gate: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
