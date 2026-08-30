from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from gpu_fault.admin_bootstrap_common import (
    DEFAULT_DCGM_IMAGE,
    DEFAULT_NODE_INSTALLER_IMAGE,
    DEFAULT_RUNTIME_IMAGE,
    BootstrapError,
    CommandRunner,
    compute_agent_config_digest,
)


DEFAULT_ADOT_IMAGE_AMD64 = (
    "public.ecr.aws/aws-observability/aws-otel-collector@"
    "sha256:bb72328152c72fb9662056759b275f7cc85e115db12bbb114fbea9f68dc4816c"
)


def _private_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BootstrapError(f"{description} is missing: {resolved}")
    if resolved.stat().st_mode & 0o077:
        raise BootstrapError(f"{description} must not be group/other accessible")
    return resolved


@contextmanager
def isolated_postgres_url(runner: CommandRunner) -> Iterator[str]:
    configured = os.getenv("GPU_FAULT_TEST_POSTGRES_URL", "").strip()
    if configured:
        yield configured
        return
    if runner.dry_run:
        yield "postgresql://postgres@127.0.0.1:5432/postgres"
        return
    name = f"gpu-fault-release-postgres-{os.getpid()}"
    runner.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            name,
            "-p",
            "127.0.0.1::5432",
            "-e",
            "POSTGRES_HOST_AUTH_METHOD=trust",
            "postgres:16",
        ],
        mutate=True,
    )
    try:
        for _attempt in range(60):
            ready = subprocess.run(
                ["docker", "exec", name, "pg_isready", "-U", "postgres"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            raise BootstrapError("temporary PostgreSQL 16 did not become ready")
        port = runner.run(
            [
                "docker",
                "inspect",
                "-f",
                '{{(index (index .NetworkSettings.Ports "5432/tcp") 0).HostPort}}',
                name,
            ]
        )
        yield f"postgresql://postgres@127.0.0.1:{port}/postgres"
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def verify_prebuilt_release(
    runner: CommandRunner,
    *,
    repository_root: Path,
    state_dir: Path,
    cosign_public_key: Path | None = None,
) -> None:
    if runner.dry_run:
        return
    attestation = repository_root / "dist/current-attestation.json"
    bundle = repository_root / "dist/current-attestation.bundle.json"
    public_key = (
        (cosign_public_key or state_dir / "release-signing/cosign.pub")
        .expanduser()
        .resolve()
    )
    if not attestation.is_file() or not bundle.is_file():
        raise BootstrapError("signed release artifacts are missing after release-build")
    if not public_key.is_file():
        raise BootstrapError(f"cosign public key is missing: {public_key}")
    script = repository_root / "scripts/verify-release-attestation.py"
    runner.run(
        [
            sys.executable,
            str(script),
            "--attestation",
            str(attestation),
            "--bundle",
            str(bundle),
            "--cosign-key",
            str(public_key),
        ],
        cwd=repository_root,
    )


def build_signed_release(
    runner: CommandRunner,
    *,
    repository_root: Path,
    state_dir: Path,
    region: str,
    runtime_repository: str,
    cache_repository: str | None,
    runtime_profile: str,
    cosign_signing_key: Path | None = None,
    cosign_public_key: Path | None = None,
    cosign_password_file: Path | None = None,
) -> dict[str, Any]:
    if runner.dry_run:
        return load_prebuilt_release(
            runner,
            repository_root=repository_root,
            runtime_profile=runtime_profile,
        )
    signing_dir = state_dir / "release-signing"
    signing_key = _private_file(
        cosign_signing_key or signing_dir / "cosign.key",
        "cosign signing key",
    )
    password_path = cosign_password_file or signing_dir / "cosign.password"
    password = None
    if password_path.expanduser().is_file():
        password = (
            _private_file(
                password_path,
                "cosign password file",
            )
            .read_text(encoding="utf-8")
            .strip()
        )
    registry = runtime_repository.split("/", 1)[0]
    login_password = runner.run(
        [
            "aws",
            "ecr",
            "get-login-password",
            "--region",
            region,
        ],
        sensitive=True,
    )
    runner.run(
        [
            "docker",
            "login",
            "--username",
            "AWS",
            "--password-stdin",
            registry,
        ],
        input_text=login_password + "\n",
        sensitive=True,
        mutate=True,
    )
    command = [
        "make",
        f"PYTHON={sys.executable}",
        "release-build",
        f"RUNTIME_IMAGE_REPOSITORY={runtime_repository}",
        f"COSIGN_SIGNING_KEY={signing_key}",
    ]
    if cache_repository:
        cache_ref = f"{cache_repository}:buildcache-linux-amd64"
        command.extend(
            [
                f"RUNTIME_IMAGE_CACHE_FROM=type=registry,ref={cache_ref}",
                (
                    "RUNTIME_IMAGE_CACHE_TO=type=registry,"
                    f"ref={cache_ref},mode=max,image-manifest=true,"
                    "oci-mediatypes=true"
                ),
            ]
        )
    with isolated_postgres_url(runner) as postgres_url:
        environment = {
            **os.environ,
            "GPU_FAULT_TEST_POSTGRES_URL": postgres_url,
        }
        if password is not None:
            environment["COSIGN_PASSWORD"] = password
        runner.run(
            command,
            cwd=repository_root,
            env=environment,
            mutate=True,
            capture=False,
        )
    verify_prebuilt_release(
        runner,
        repository_root=repository_root,
        state_dir=state_dir,
        cosign_public_key=cosign_public_key,
    )
    return load_prebuilt_release(
        runner,
        repository_root=repository_root,
        runtime_profile=runtime_profile,
    )


def load_prebuilt_release(
    runner: CommandRunner,
    *,
    repository_root: Path,
    runtime_profile: str,
) -> dict[str, Any]:
    manifest = repository_root / "dist/current-release.json"
    if runner.dry_run:
        return {
            "manifest": str(manifest),
            "release_id": "dry-run",
            "images": {
                "runtime": DEFAULT_RUNTIME_IMAGE,
                "node_installer": DEFAULT_NODE_INSTALLER_IMAGE,
                "dcgm_exporter": DEFAULT_DCGM_IMAGE,
                "adot": DEFAULT_ADOT_IMAGE_AMD64,
            },
            "agent_config_digest": "0" * 64,
        }
    if not manifest.is_file():
        raise BootstrapError(
            "dist/current-release.json is missing; build and sign the release in CI"
        )
    try:
        release = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"release manifest is invalid: {exc}") from exc
    if (
        int(release.get("schema_version", 0)) < 3
        or release.get("deployable") is not True
    ):
        raise BootstrapError("ARN bootstrap requires a deployable schema v3 release")
    images = dict((release.get("delivery") or {}).get("images") or {})
    required_images = {
        name: str((images.get(name) or {}).get("reference") or "")
        for name in ("runtime", "node_installer", "dcgm_exporter", "adot")
    }
    if any("@sha256:" not in value for value in required_images.values()):
        raise BootstrapError("release manifest image identity is incomplete")
    return {
        "manifest": str(manifest),
        "release_id": str(release["release_id"]),
        "images": required_images,
        "agent_config_digest": compute_agent_config_digest(
            runner,
            repository_root=repository_root,
            runtime_profile_version=runtime_profile,
        ),
    }
