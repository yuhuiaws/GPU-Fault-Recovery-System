from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts import release_image


def test_pushed_runtime_image_descriptor_uses_registry_digest(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    commands = []
    contexts = []

    def build_component(*, name, output, **_kwargs):
        wheel = output / (
            "gpu_fault_control_plane-0.10.0-py3-none-any.whl"
            if name == "control_plane"
            else "gpu_fault_cluster_executor-0.10.0-py3-none-any.whl"
        )
        wheel.write_bytes(name.encode())
        digest = hashlib.sha256(f"{name}-module".encode()).hexdigest()
        return wheel, digest, set()

    monkeypatch.setattr(release_image, "build_component", build_component)

    def runner(command, **_kwargs):
        commands.append(command)
        context = Path(command[-1])
        contexts.append(
            {
                "dockerfile": (context / "Dockerfile").read_text(encoding="utf-8"),
                "wheels": sorted(path.name for path in (context / "wheels").iterdir()),
            }
        )
        metadata = Path(command[command.index("--metadata-file") + 1])
        metadata.write_text(
            json.dumps({"containerimage.digest": "sha256:" + "a" * 64}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    descriptor = release_image.build_runtime_image(
        root, repository="registry.example/gpu-fault-runtime", push=True, runner=runner
    )

    assert "--push" in commands[0]
    assert descriptor["schema_version"] == 2
    assert descriptor["deployable"] is True
    assert descriptor["reference"] == (
        "registry.example/gpu-fault-runtime@sha256:" + "a" * 64
    )
    assert set(descriptor["components"]) == {"control_plane", "executor"}
    assert "pip wheel" not in contexts[0]["dockerfile"]
    assert contexts[0]["wheels"] == [
        "gpu_fault_cluster_executor-0.10.0-py3-none-any.whl",
        "gpu_fault_control_plane-0.10.0-py3-none-any.whl",
    ]
