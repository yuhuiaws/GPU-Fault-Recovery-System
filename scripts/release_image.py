from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable

if __package__:
    from scripts.component_wheels import COMPONENTS, build_component
    from scripts.release_identity import build_release_identity
else:
    from component_wheels import COMPONENTS, build_component
    from release_identity import build_release_identity


DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
RUNTIME_COMPONENTS = ("control_plane", "executor")


class ReleaseImageError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_runtime_image(
    root: Path,
    *,
    repository: str,
    platform: str = "linux/amd64",
    push: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    repository = repository.strip().rstrip(":")
    if not repository or any(character.isspace() for character in repository):
        raise ReleaseImageError("runtime image repository is invalid")
    identity = build_release_identity(root)
    source_sha = str(identity["sha256"])
    tag = f"{repository}:build-{source_sha[:16]}"
    with tempfile.TemporaryDirectory(prefix="gpu-fault-image-") as directory:
        temporary = Path(directory)
        context = temporary / "context"
        wheels = context / "wheels"
        requirements = context / "requirements"
        wheels.mkdir(parents=True)
        requirements.mkdir()
        shutil.copy2(root / "deploy/image/Dockerfile", context / "Dockerfile")
        shutil.copy2(
            root / "requirements/runtime.lock",
            requirements / "runtime.lock",
        )
        components: dict[str, dict[str, str]] = {}
        for name in RUNTIME_COMPONENTS:
            wheel, module_digest, _modules = build_component(
                python=sys.executable,
                name=name,
                build_root=temporary / "components",
                output=wheels,
            )
            components[name] = {
                "distribution": COMPONENTS[name].distribution,
                "wheel_sha256": _sha256(wheel),
                "module_digest": module_digest,
            }
        metadata_path = temporary / "metadata.json"
        command = [
            "docker",
            "buildx",
            "build",
            "--file",
            str(context / "Dockerfile"),
            "--platform",
            platform,
            "--tag",
            tag,
            "--metadata-file",
            str(metadata_path),
        ]
        for name, component in sorted(components.items()):
            command.extend(
                (
                    "--label",
                    "gpu-fault.component."
                    f"{name}.wheel-sha256={component['wheel_sha256']}",
                    "--label",
                    "gpu-fault.component."
                    f"{name}.module-digest={component['module_digest']}",
                )
            )
        command.extend(("--push" if push else "--load", str(context)))
        completed = runner(
            command,
            cwd=root,
            check=False,
            text=True,
        )
        if completed.returncode:
            raise ReleaseImageError(
                f"runtime image build failed with status {completed.returncode}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = str(metadata.get("containerimage.digest") or "")
    if push and not DIGEST_PATTERN.fullmatch(digest):
        raise ReleaseImageError("pushed runtime image has no OCI manifest digest")
    return {
        "schema_version": 2,
        "deployable": push,
        "repository": repository,
        "tag": tag,
        "reference": f"{repository}@{digest}" if push else None,
        "digest": digest or None,
        "platform": platform,
        "source_identity_sha256": source_sha,
        "dockerfile_sha256": _sha256(root / "deploy/image/Dockerfile"),
        "dependency_lock_sha256": _sha256(root / "requirements/runtime.lock"),
        "components": components,
    }
