from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote

if __package__:
    from scripts.ci_gate_artifacts import (
        GateArtifactError,
        api_json,
        completed_main_run,
        download,
        extract_archive,
    )
    from scripts.resolve_ci_run import ResolveCiRunError, resolve_ci_run
else:
    from ci_gate_artifacts import (
        GateArtifactError,
        api_json,
        completed_main_run,
        download,
        extract_archive,
    )
    from resolve_ci_run import ResolveCiRunError, resolve_ci_run


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAME = "gpu-fault-ci-candidate"
EXPECTED_SHARDS = frozenset(
    {
        "deployment",
        "fault_runner",
        "postgres",
        "runtime_0",
        "runtime_1",
        "runtime_2",
    }
)
REPOSITORY_PATTERN = re.compile(
    r"(?:github\.com[:/])(?P<repository>[^/\s]+/[^/\s]+?)(?:\.git)?$"
)


class CandidateUnavailable(RuntimeError):
    pass


class CandidateTrustError(RuntimeError):
    pass


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise CandidateUnavailable(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def repository_slug(root: Path) -> str:
    configured = (
        os.getenv("GPU_FAULT_GITHUB_REPOSITORY") or os.getenv("GITHUB_REPOSITORY") or ""
    ).strip()
    if configured:
        if configured.count("/") != 1:
            raise CandidateUnavailable("configured GitHub repository is invalid")
        return configured
    remote = _git(root, "config", "--get", "remote.origin.url")
    match = REPOSITORY_PATTERN.search(remote)
    if match is None:
        raise CandidateUnavailable("origin is not a GitHub repository")
    return match.group("repository")


def _private_token_file(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CandidateUnavailable("GitHub token file is missing")
    if resolved.stat().st_mode & 0o077:
        raise CandidateTrustError(
            "GitHub token file must not be group/other accessible"
        )
    value = resolved.read_text(encoding="utf-8").strip()
    if not value:
        raise CandidateUnavailable("GitHub token file is empty")
    return value


def github_token() -> str:
    token_file = os.getenv("GPU_FAULT_GITHUB_TOKEN_FILE", "").strip()
    if token_file:
        return _private_token_file(Path(token_file))
    token = (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()
    if not token:
        raise CandidateUnavailable(
            "GitHub token is unavailable; configure GPU_FAULT_GITHUB_TOKEN_FILE"
        )
    return token


def matches_local_main_ref(root: Path) -> bool:
    if _git(root, "status", "--porcelain", "--untracked-files=normal"):
        return False
    head = _git(root, "rev-parse", "HEAD")
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "refs/remotes/origin/main^{commit}"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == head


def _run(
    arguments: Sequence[str], *, cwd: Path, env: dict[str, str] | None = None
) -> None:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise CandidateTrustError(
            f"candidate verification failed: {arguments[0]}: "
            + (completed.stderr or "").strip()
        )


def _verify_blob(path: Path, *, repository: str) -> None:
    bundle = path.with_name(path.stem + ".bundle.json")
    if not bundle.is_file():
        raise CandidateTrustError(f"candidate signature bundle is missing: {bundle}")
    _run(
        [
            "cosign",
            "verify-blob",
            "--bundle",
            str(bundle),
            "--certificate-identity",
            (
                f"https://github.com/{repository}/.github/workflows/"
                "ci.yml@refs/heads/main"
            ),
            "--certificate-oidc-issuer",
            "https://token.actions.githubusercontent.com",
            str(path),
        ],
        cwd=path.parent,
    )


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateTrustError(f"{description} is invalid") from exc
    if not isinstance(value, dict):
        raise CandidateTrustError(f"{description} must be an object")
    return value


def verify_candidate(
    root: Path,
    candidate: Path,
    *,
    repository: str,
) -> dict[str, Any]:
    gate_path = candidate / "ci-gate.json"
    if not gate_path.is_file():
        raise CandidateTrustError("candidate CI gate is missing")
    gate = _load_object(gate_path, "candidate CI gate")
    if gate.get("repository") != repository:
        raise CandidateTrustError("candidate repository does not match")
    current_commit = _git(root, "rev-parse", "HEAD")
    current_tree = _git(root, "rev-parse", "HEAD^{tree}")
    source = gate.get("source")
    if (
        not isinstance(source, dict)
        or source.get("git_commit") != current_commit
        or source.get("git_tree") != current_tree
    ):
        raise CandidateTrustError("candidate source identity does not match checkout")

    _verify_blob(gate_path, repository=repository)
    domains = gate.get("domains")
    unit = domains.get("unit") if isinstance(domains, dict) else None
    if not isinstance(unit, dict):
        raise CandidateTrustError("candidate unit domain is missing")
    unit_path = (candidate / str(unit.get("path") or "")).resolve()
    try:
        unit_path.relative_to(candidate.resolve())
    except ValueError as exc:
        raise CandidateTrustError("candidate unit gate leaves artifact root") from exc
    if not unit_path.is_file():
        raise CandidateTrustError("candidate unit gate is missing")
    _verify_blob(unit_path, repository=repository)

    shard_root = unit_path.parent / "shards"
    shard_gates = sorted(shard_root.glob("*/coverage-shard-gate.json"))
    shard_names = {path.parent.name for path in shard_gates}
    if shard_names != EXPECTED_SHARDS:
        raise CandidateTrustError(
            "candidate coverage shard set is incomplete: "
            + ", ".join(sorted(shard_names))
        )
    for path in shard_gates:
        _verify_blob(path, repository=repository)

    _run(
        [
            sys.executable,
            str(root / "scripts/ci_gate.py"),
            "verify",
            "--gate",
            str(gate_path),
            "--dist",
            str(candidate),
        ],
        cwd=root,
        env={**os.environ, "GITHUB_REPOSITORY": repository},
    )
    return gate


def _existing_candidate(
    root: Path,
    destination: Path,
    *,
    repository: str,
) -> dict[str, Any] | None:
    gate = destination / "ci-gate.json"
    if not gate.is_file():
        return None
    value = verify_candidate(root, destination, repository=repository)
    return {
        "available": True,
        "reused": True,
        "repository": repository,
        "run_id": int(value["run_id"]),
        "ci_gate": str(gate),
    }


def _candidate_artifact(
    *,
    repository: str,
    token: str,
    run_id: int,
) -> dict[str, Any]:
    value = api_json(
        f"https://api.github.com/repos/{quote(repository, safe='/')}/"
        f"actions/runs/{run_id}/artifacts?name={quote(ARTIFACT_NAME)}&per_page=100",
        token,
    )
    artifacts = value.get("artifacts") if isinstance(value, dict) else None
    matches = [
        item
        for item in artifacts or []
        if isinstance(item, dict)
        and item.get("name") == ARTIFACT_NAME
        and item.get("expired") is not True
    ]
    if len(matches) != 1:
        raise CandidateUnavailable(
            f"trusted CI run has {len(matches)} usable {ARTIFACT_NAME} artifacts"
        )
    return matches[0]


def _replace_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(source, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def restore_candidate(root: Path, destination: Path) -> dict[str, Any]:
    repository = repository_slug(root)
    existing = _existing_candidate(root, destination, repository=repository)
    if existing is not None:
        return existing
    if not matches_local_main_ref(root):
        raise CandidateUnavailable(
            "HEAD does not equal local refs/remotes/origin/main; using local gates"
        )
    token = github_token()
    commit = _git(root, "rev-parse", "HEAD")
    try:
        run_id = resolve_ci_run(
            repository=repository,
            commit=commit,
            token=token,
        )
        run = completed_main_run(
            repository=repository,
            token=token,
            run_id=run_id,
        )
        if (
            run is None
            or run.get("conclusion") != "success"
            or run.get("head_sha") != commit
        ):
            raise CandidateUnavailable("resolved CI run is not a successful main run")
        artifact = _candidate_artifact(
            repository=repository,
            token=token,
            run_id=run_id,
        )
        archive = download(str(artifact["archive_download_url"]), token)
    except (HTTPError, URLError, OSError, ResolveCiRunError, GateArtifactError) as exc:
        raise CandidateUnavailable(
            f"cannot restore trusted CI candidate: {exc}"
        ) from exc

    with tempfile.TemporaryDirectory(
        prefix=".ci-candidate-",
        dir=destination.parent,
    ) as directory:
        extracted = Path(directory) / "extracted"
        try:
            extract_archive(archive, extracted)
        except GateArtifactError as exc:
            raise CandidateTrustError(str(exc)) from exc
        candidate = (
            extracted / "dist"
            if (extracted / "dist/ci-gate.json").is_file()
            else extracted
        )
        verify_candidate(root, candidate, repository=repository)
        staged = destination.parent / f".{destination.name}.candidate-{os.getpid()}"
        shutil.rmtree(staged, ignore_errors=True)
        shutil.copytree(candidate, staged)
        _replace_directory(staged, destination)
    return {
        "available": True,
        "reused": False,
        "repository": repository,
        "run_id": run_id,
        "ci_gate": str(destination / "ci-gate.json"),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path)
    options = parser.parse_args(arguments)
    root = options.root.expanduser().resolve()
    destination = (
        options.destination.expanduser().resolve()
        if options.destination is not None
        else root / "dist"
    )
    try:
        result = restore_candidate(root, destination)
    except CandidateUnavailable as exc:
        result = {
            "available": False,
            "reason": str(exc),
        }
    except (
        CandidateTrustError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"restore-ci-candidate: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
