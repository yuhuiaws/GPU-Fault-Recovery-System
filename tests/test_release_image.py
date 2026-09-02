from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import release_image

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIGEST = "sha256:" + "a" * 64


def fake_component_builder(*, name, output, **_kwargs):
    wheel = output / (
        "gpu_fault_control_plane-0.10.0-py3-none-any.whl"
        if name == "control_plane"
        else "gpu_fault_cluster_executor-0.10.0-py3-none-any.whl"
    )
    wheel.write_bytes(name.encode())
    digest = hashlib.sha256(f"{name}-module".encode()).hexdigest()
    return wheel, digest, set()


def inspected_image(
    labels: dict[str, str],
    *,
    digest: str = REGISTRY_DIGEST,
    platform: str = "linux/amd64",
) -> str:
    operating_system, architecture, *variant = platform.split("/")
    image = {
        "os": operating_system,
        "architecture": architecture,
        "config": {"Labels": labels},
    }
    if variant:
        image["variant"] = variant[0]
    return json.dumps({"manifest": {"digest": digest}, "image": {platform: image}})


def test_pushed_runtime_image_uses_complete_identity_and_remote_cache(
    monkeypatch,
) -> None:
    commands = []
    contexts = []
    labels: dict[str, str] = {}
    inspect_count = 0
    monkeypatch.setattr(release_image, "build_component", fake_component_builder)

    def runner(command, **_kwargs):
        nonlocal inspect_count
        commands.append(command)
        if command[:3] == ["docker", "buildx", "imagetools"]:
            inspect_count += 1
            if inspect_count == 1:
                return subprocess.CompletedProcess(command, 1, "", "manifest unknown")
            return subprocess.CompletedProcess(command, 0, inspected_image(labels), "")
        context = Path(command[-1])
        contexts.append(
            {
                "dockerfile": (context / "Dockerfile").read_text(encoding="utf-8"),
                "wheels": sorted(path.name for path in (context / "wheels").iterdir()),
            }
        )
        labels.update(
            value.split("=", 1)
            for index, value in enumerate(command)
            if index > 0 and command[index - 1] == "--label"
        )
        metadata = Path(command[command.index("--metadata-file") + 1])
        metadata.write_text(
            json.dumps({"containerimage.digest": REGISTRY_DIGEST}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    descriptor = release_image.build_runtime_image(
        ROOT,
        repository="registry.example/gpu-fault-runtime",
        push=True,
        cache_from=("type=registry,ref=registry.example/cache:runtime",),
        cache_to=("type=registry,ref=registry.example/cache:runtime,mode=max",),
        runner=runner,
    )

    build_command = next(
        command for command in commands if command[:3] == ["docker", "buildx", "build"]
    )
    assert "--push" in build_command
    assert "--cache-from" in build_command
    assert "--cache-to" in build_command
    assert descriptor["schema_version"] == 2
    assert descriptor["deployable"] is True
    assert descriptor["registry_reused"] is False
    assert descriptor["reference"] == (
        "registry.example/gpu-fault-runtime@" + REGISTRY_DIGEST
    )
    assert descriptor["tag"].endswith(descriptor["image_input_sha256"]), (
        "runtime image tag must be content-addressed by the full input digest"
    )
    assert descriptor["image_inputs"]["base_images"] == [
        "registry.access.redhat.com/ubi9/python-312-minimal@sha256:"
        "ddf83888de3388bce2fc9c3d66f917bf683790d4db9f956644a3e1b9b6fd15e7"
    ]
    assert set(descriptor["components"]) == {"control_plane", "executor"}
    assert "pip wheel" not in contexts[0]["dockerfile"]
    assert contexts[0]["wheels"] == [
        "gpu_fault_cluster_executor-0.10.0-py3-none-any.whl",
        "gpu_fault_control_plane-0.10.0-py3-none-any.whl",
    ]
    assert inspect_count == 2


def test_matching_registry_image_skips_docker_build(monkeypatch) -> None:
    inspected = []
    monkeypatch.setattr(release_image, "build_component", fake_component_builder)

    def inspect(root, **kwargs):
        inspected.append((root, kwargs))
        return REGISTRY_DIGEST

    monkeypatch.setattr(release_image, "inspect_registry_image", inspect)

    def runner(command, **_kwargs):
        pytest.fail(f"registry reuse must skip docker build: {command}")

    descriptor = release_image.build_runtime_image(
        ROOT, repository="registry.example/gpu-fault-runtime", push=True, runner=runner
    )

    assert descriptor["registry_reused"] is True
    assert descriptor["digest"] == REGISTRY_DIGEST
    assert len(inspected) == 1
    assert (
        inspected[0][1]["expected_labels"]["gpu-fault.image-input.sha256"]
        == descriptor["image_input_sha256"]
    )


def test_runtime_image_reuses_validated_component_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    wheels = {}
    module_digests = {}
    for name in ("control_plane", "executor"):
        path = tmp_path / f"{name}.whl"
        path.write_bytes(name.encode())
        wheels[name] = path
        module_digests[name] = hashlib.sha256(f"{name}-module".encode()).hexdigest()
    monkeypatch.setattr(
        release_image,
        "load_component_artifacts",
        lambda *_args, **_kwargs: SimpleNamespace(
            wheels=wheels, module_digests=module_digests
        ),
    )
    monkeypatch.setattr(
        release_image,
        "build_component",
        lambda **_kwargs: pytest.fail("validated component wheel was rebuilt"),
    )
    monkeypatch.setattr(
        release_image,
        "inspect_registry_image",
        lambda *_args, **_kwargs: REGISTRY_DIGEST,
    )

    descriptor = release_image.build_runtime_image(
        ROOT,
        repository="registry.example/gpu-fault-runtime",
        push=True,
        component_artifacts=tmp_path / "current-release.json",
        runner=lambda command, **_kwargs: pytest.fail(
            f"registry reuse must skip docker build: {command}"
        ),
    )

    assert descriptor["registry_reused"] is True
    assert (
        descriptor["components"]["control_plane"]["module_digest"]
        == (module_digests["control_plane"])
    )


def test_registry_image_label_mismatch_fails_closed() -> None:
    expected = {"gpu-fault.image-input.sha256": "b" * 64}

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, inspected_image({"gpu-fault.image-input.sha256": "c" * 64}), ""
        )

    with pytest.raises(release_image.ReleaseImageError, match="labels do not match"):
        release_image.inspect_registry_image(
            ROOT,
            tag="registry.example/runtime:build-test",
            platform="linux/amd64",
            expected_labels=expected,
            runner=runner,
        )


def test_image_input_digest_covers_platform_base_args_and_wheels() -> None:
    components = {
        "control_plane": {
            "distribution": "gpu-fault-control-plane",
            "wheel_sha256": "a" * 64,
            "module_digest": "b" * 64,
        },
        "executor": {
            "distribution": "gpu-fault-cluster-executor",
            "wheel_sha256": "c" * 64,
            "module_digest": "d" * 64,
        },
    }
    first = release_image.runtime_image_inputs(
        ROOT, platform="linux/amd64", build_args={}, components=components
    )
    different_platform = release_image.runtime_image_inputs(
        ROOT, platform="linux/arm64", build_args={}, components=components
    )
    different_base = release_image.runtime_image_inputs(
        ROOT,
        platform="linux/amd64",
        build_args={
            "RUNTIME_BASE_IMAGE": ("registry.example/python@sha256:" + "e" * 64)
        },
        components=components,
    )
    changed_components = json.loads(json.dumps(components))
    changed_components["executor"]["wheel_sha256"] = "f" * 64
    different_wheel = release_image.runtime_image_inputs(
        ROOT, platform="linux/amd64", build_args={}, components=changed_components
    )

    digests = {
        release_image.canonical_sha256(value)
        for value in (first, different_platform, different_base, different_wheel)
    }

    assert len(digests) == 4
