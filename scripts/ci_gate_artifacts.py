from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any
from urllib.parse import quote, urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
    urlopen,
)
import zipfile


class GateArtifactError(RuntimeError):
    pass


# The coverage gate's own failure type lives here rather than in
# ci_coverage_gate.py so that the gate and the modules it delegates to can both
# raise it without importing each other.
class CoverageGateError(RuntimeError):
    pass


class ArtifactRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> Request | None:
        redirected = super().redirect_request(
            request,
            file_pointer,
            code,
            message,
            headers,
            new_url,
        )
        if (
            redirected is not None
            and urlsplit(request.full_url).netloc != urlsplit(new_url).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise GateArtifactError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def repository_files(root: Path) -> tuple[Path, ...]:
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
        raise GateArtifactError("cannot enumerate gate inputs")
    result = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if relative.is_absolute() or ".." in relative.parts:
            raise GateArtifactError(f"gate input leaves repository: {relative}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise GateArtifactError(f"unsupported gate input: {relative}")
        result.append(path)
    return tuple(sorted(result))


def api_json(url: str, token: str) -> Any:
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


def download(url: str, token: str) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    opener = build_opener(ArtifactRedirectHandler())
    with opener.open(request, timeout=60) as response:
        return response.read()


def completed_main_run(
    *,
    repository: str,
    token: str,
    run_id: int,
) -> dict[str, Any] | None:
    run = api_json(
        f"https://api.github.com/repos/{quote(repository, safe='/')}/"
        f"actions/runs/{run_id}",
        token,
    )
    if not isinstance(run, dict):
        raise GateArtifactError("GitHub workflow run response is invalid")
    return (
        run
        if run.get("status") == "completed"
        and run.get("event") == "push"
        and run.get("head_branch") == "main"
        and run.get("path") == ".github/workflows/ci.yml"
        else None
    )


def successful_run_job(
    *,
    repository: str,
    token: str,
    run_id: int,
    name: str,
) -> bool:
    value = api_json(
        f"https://api.github.com/repos/{quote(repository, safe='/')}/"
        f"actions/runs/{run_id}/jobs?per_page=100",
        token,
    )
    jobs = value.get("jobs") if isinstance(value, dict) else None
    if not isinstance(jobs, list):
        raise GateArtifactError("GitHub workflow jobs response is invalid")
    return any(
        isinstance(job, dict)
        and job.get("name") == name
        and job.get("status") == "completed"
        and job.get("conclusion") == "success"
        for job in jobs
    )


def find_reusable_artifact(
    *,
    repository: str,
    token: str,
    name: str,
    current_run_id: int | None,
    required_job_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    value = api_json(
        f"https://api.github.com/repos/{quote(repository, safe='/')}/"
        f"actions/artifacts?name={quote(name)}&per_page=100",
        token,
    )
    artifacts = value.get("artifacts") if isinstance(value, dict) else None
    if not isinstance(artifacts, list):
        raise GateArtifactError("GitHub artifact response is invalid")
    candidates = sorted(
        (item for item in artifacts if isinstance(item, dict)),
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )
    for artifact in candidates:
        workflow_run = artifact.get("workflow_run")
        run_id = int((workflow_run or {}).get("id") or 0)
        if (
            artifact.get("expired") is True
            or not run_id
            or run_id == current_run_id
            or (workflow_run or {}).get("head_branch") != "main"
        ):
            continue
        run = completed_main_run(
            repository=repository,
            token=token,
            run_id=run_id,
        )
        if run is None:
            continue
        if required_job_name is None:
            eligible = run.get("conclusion") == "success"
        else:
            eligible = successful_run_job(
                repository=repository,
                token=token,
                run_id=run_id,
                name=required_job_name,
            )
        if eligible:
            return artifact, run
    return None


def extract_archive(data: bytes, destination: Path) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            for member in members:
                path = Path(member.filename)
                mode = member.external_attr >> 16
                if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
                    raise GateArtifactError(
                        f"unsafe gate artifact member: {member.filename}"
                    )
            archive.extractall(destination)
            for member in members:
                mode = member.external_attr >> 16
                path = destination / member.filename
                if path.is_file() and mode:
                    path.chmod(mode & 0o777)
    except zipfile.BadZipFile as exc:
        raise GateArtifactError("gate artifact archive is invalid") from exc


def evidence_entry(root: Path, path: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise GateArtifactError(f"gate evidence leaves artifact root: {path}") from exc
    if not resolved.is_file():
        raise GateArtifactError(f"gate evidence is missing: {path}")
    return {
        "path": relative,
        "sha256": sha256(resolved),
        "size": resolved.stat().st_size,
    }


def verify_evidence(
    artifact_root: Path,
    evidence: object,
    *,
    expected_names: set[str],
) -> dict[str, dict[str, object]]:
    if not isinstance(evidence, dict) or set(evidence) != expected_names:
        raise GateArtifactError("gate evidence inventory is invalid")
    verified: dict[str, dict[str, object]] = {}
    for name, raw in evidence.items():
        if not isinstance(raw, dict):
            raise GateArtifactError(f"gate evidence entry is invalid: {name}")
        path = (artifact_root / str(raw.get("path") or "")).resolve()
        try:
            path.relative_to(artifact_root.resolve())
        except ValueError as exc:
            raise GateArtifactError(
                f"gate evidence leaves artifact root: {name}"
            ) from exc
        if (
            not path.is_file()
            or sha256(path) != raw.get("sha256")
            or path.stat().st_size != raw.get("size")
        ):
            raise GateArtifactError(f"gate evidence does not match: {name}")
        verified[str(name)] = raw
    return verified


def write_outputs(path: Path | None, values: dict[str, object]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as destination:
        for name, value in values.items():
            destination.write(f"{name}={value}\n")
