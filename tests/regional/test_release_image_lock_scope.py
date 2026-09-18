"""Registry-only drain cannot weaken image-consuming entrypoints."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest

from gpu_fault_release import rollout
from gpu_fault_release.regional_release_arguments import validate_cluster_arguments
from gpu_fault_release.regional_release_config import (
    ReleaseConfig,
    ReleaseError,
    resolve_release_image,
)
from gpu_fault_release.regional_release_rendering import build_cpu_apply_environment
from tests.regional._release_orchestrator_support import config_file

LOCKED = {
    "runtime": "registry.example/runtime@sha256:" + "a" * 64,
    "node_installer": "registry.example/installer@sha256:" + "b" * 64,
    "dcgm_exporter": "registry.example/dcgm@sha256:" + "c" * 64,
    "adot": "registry.example/adot@sha256:" + "d" * 64,
    "executor": "registry.example/executor@sha256:" + "e" * 64,
    "node_dependencies": "registry.example/dependencies@sha256:" + "f" * 64,
}
VARIABLES = {
    "runtime": "GPU_FAULT_RUNTIME_IMAGE",
    "node_installer": "GPU_FAULT_NODE_INSTALLER_IMAGE",
    "dcgm_exporter": "GPU_FAULT_DCGM_EXPORTER_IMAGE",
    "adot": "GPU_FAULT_ADOT_IMAGE",
    "executor": "GPU_FAULT_EXECUTOR_IMAGE",
}
OTHER = "registry.example/other@sha256:" + "1" * 64
CLUSTER_MODES = (
    "join-cluster",
    "activate-cluster",
    "fail-cluster",
    "rollback-cluster",
    "drain-cluster",
    "remove-cluster",
)


def locked_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, schema: int = 3
) -> ReleaseConfig:
    for variable in VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    return replace(
        ReleaseConfig.load(config_file(tmp_path)),
        release_manifest_schema_version=schema,
        locked_images=dict(LOCKED),
        release_delivery_identity={
            "images": {
                name: {"reference": reference, "source": f"source.example/{name}:base"}
                for name, reference in LOCKED.items()
            }
        },
    )


@pytest.mark.parametrize("schema", [3, 4])
@pytest.mark.parametrize(
    "image", ["runtime", "node_installer", "dcgm_exporter", "adot"]
)
def test_programmatic_construction_refuses_conflicts_before_any_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema: int, image: str
) -> None:
    config = locked_config(tmp_path, monkeypatch, schema=schema)
    monkeypatch.setenv(VARIABLES[image], OTHER)
    monkeypatch.setenv("GPU_FAULT_ADMIN_ALLOW_UNBOUND", "1")
    monkeypatch.setattr(
        rollout,
        "rendered_release_manifest_sha256",
        lambda _release: pytest.fail("a conflicting release reached the renderer"),
    )

    with pytest.raises(ReleaseError, match=f"{VARIABLES[image]}.*v{schema} image lock"):
        rollout.RegionalRelease(config, rollout.Runner(dry_run=True))


def test_split_executor_conflicts_do_not_fall_back_to_the_cpu_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = locked_config(tmp_path, monkeypatch, schema=4)
    monkeypatch.setenv("GPU_FAULT_EXECUTOR_IMAGE", LOCKED["runtime"])

    with pytest.raises(ReleaseError, match="GPU_FAULT_EXECUTOR_IMAGE.*v4 image lock"):
        rollout.RegionalRelease(config, rollout.Runner(dry_run=True))


@pytest.mark.parametrize("schema", [3, 4])
@pytest.mark.parametrize("mode", ["plan", "upgrade", "remove-cluster"])
def test_cli_image_consumers_refuse_conflicts_before_the_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    schema: int,
    mode: str,
) -> None:
    config = locked_config(tmp_path, monkeypatch, schema=schema)
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", OTHER)
    monkeypatch.setattr(rollout.ReleaseConfig, "load", lambda _path: config)
    monkeypatch.setattr(
        rollout.RegionalRelease,
        mode.replace("-", "_"),
        lambda *_args, **_kwargs: pytest.fail("conflicting image reached the handler"),
    )
    monkeypatch.setattr(
        rollout.sys,
        "argv",
        ["rollout", mode, "--config", "release.json", "--cluster-id", "gpu-a"],
    )

    assert rollout.main() == 2
    assert f"schema v{schema} image lock" in capsys.readouterr().err


@pytest.mark.parametrize("schema", [3, 4])
@pytest.mark.parametrize("override", [None, "", "locked", "source", "mirror"])
def test_image_resolution_keeps_source_alias_and_same_digest_mirror_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema: int, override: str | None
) -> None:
    config = locked_config(tmp_path, monkeypatch, schema=schema)
    mirror = "mirror.example/runtime@sha256:" + "a" * 64
    values = {
        "locked": LOCKED["runtime"],
        "source": "source.example/runtime:base",
        "mirror": mirror,
        "": "  ",
    }
    if override is not None:
        monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", values[override])

    assert resolve_release_image(config, "runtime", "GPU_FAULT_RUNTIME_IMAGE", "") == (
        mirror if override == "mirror" else LOCKED["runtime"]
    )


@pytest.mark.parametrize("reference", ["", "not-pinned"])
def test_missing_or_mutable_lock_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: str
) -> None:
    config = replace(
        locked_config(tmp_path, monkeypatch), locked_images={"runtime": reference}
    )

    with pytest.raises(ReleaseError, match="lock is missing or invalid"):
        resolve_release_image(config, "runtime", "GPU_FAULT_RUNTIME_IMAGE", "")


@pytest.mark.parametrize("reference", ["with space", "with#fragment", ""])
def test_legacy_images_still_validate_oci_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: str
) -> None:
    config = locked_config(tmp_path, monkeypatch, schema=1)
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", reference)

    with pytest.raises(ReleaseError, match="non-empty OCI image reference"):
        resolve_release_image(config, "runtime", "GPU_FAULT_RUNTIME_IMAGE", "")


@pytest.mark.parametrize("configured", [None, "legacy.example/runtime:stable"])
def test_legacy_images_keep_their_unlocked_default_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: str | None
) -> None:
    config = locked_config(tmp_path, monkeypatch, schema=2)
    if configured is not None:
        monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", configured)

    assert resolve_release_image(
        config, "runtime", "GPU_FAULT_RUNTIME_IMAGE", "legacy.example/default:base"
    ) == (configured or "legacy.example/default:base")


def test_later_environment_changes_cannot_replace_selected_split_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = locked_config(tmp_path, monkeypatch, schema=4)
    release = rollout.RegionalRelease(config, rollout.Runner(dry_run=True))
    for variable in VARIABLES.values():
        monkeypatch.setenv(variable, OTHER)

    assert release.runtime_image == LOCKED["runtime"]
    assert release.executor_image == LOCKED["executor"]
    assert (
        build_cpu_apply_environment(release, finalize=False)["GPU_FAULT_RUNTIME_IMAGE"]
        == LOCKED["runtime"]
    )


@pytest.mark.parametrize("schema", [3, 4])
def test_cli_batch_drain_uses_only_registry_access_despite_image_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema: int
) -> None:
    config = locked_config(tmp_path, monkeypatch, schema=schema)
    for variable in VARIABLES.values():
        monkeypatch.setenv(variable, OTHER)
    seen: list[list[str]] = []

    def drain(context: rollout.RegistryDrainContext, cluster_ids: list[str]) -> None:
        assert type(context) is rollout.RegistryDrainContext
        assert context.config is config
        assert not hasattr(context, "runtime_image"), (
            "registry-only context must not expose a runtime image"
        )
        assert not hasattr(context, "rendered_manifest_digest"), (
            "registry-only context must not contain rendered manifest state"
        )
        seen.append(cluster_ids)

    monkeypatch.setattr(rollout.ReleaseConfig, "load", lambda _path: config)
    monkeypatch.setattr(
        rollout,
        "RegionalRelease",
        lambda *_args: pytest.fail("registry drain constructed an image consumer"),
    )
    monkeypatch.setattr(rollout, "drain_registry_clusters", drain)
    monkeypatch.setattr(rollout, "deployment_api_budget", nullcontext)
    monkeypatch.setattr(rollout, "deployment_deadline", lambda *_args: nullcontext())
    monkeypatch.setattr(
        rollout.sys,
        "argv",
        [
            "rollout",
            "drain-cluster",
            "--config",
            "release.json",
            "--cluster-id",
            "gpu-a",
            "--cluster-id",
            "gpu-b",
        ],
    )

    assert rollout.main() == 0
    assert seen == [["gpu-a", "gpu-b"]]


def test_registry_only_drain_still_refuses_unknown_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = locked_config(tmp_path, monkeypatch)
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", OTHER)
    monkeypatch.setattr(rollout.ReleaseConfig, "load", lambda _path: config)
    monkeypatch.setattr(
        rollout.Runner,
        "run",
        lambda *_args, **_kwargs: pytest.fail("an unknown target reached I/O"),
    )
    monkeypatch.setattr(
        rollout.sys,
        "argv",
        [
            "rollout",
            "drain-cluster",
            "--config",
            "release.json",
            "--cluster-id",
            "other",
        ],
    )

    assert rollout.main() == 2


@pytest.mark.parametrize("mode", CLUSTER_MODES)
def test_cluster_modes_refuse_a_missing_selector_before_config_load(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(
        rollout.ReleaseConfig, "load", lambda _path: pytest.fail("loaded config")
    )
    monkeypatch.setattr(
        rollout.sys, "argv", ["rollout", mode, "--config", "release.json"]
    )

    assert rollout.main() == 2


@pytest.mark.parametrize(
    "mode", [*CLUSTER_MODES[:4], "remove-cluster", "sync-state", "upgrade", "plan"]
)
def test_single_cluster_modes_refuse_repeated_selectors(
    mode: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        rollout.ReleaseConfig, "load", lambda _path: pytest.fail("loaded config")
    )
    monkeypatch.setattr(
        rollout.sys,
        "argv",
        [
            "rollout",
            mode,
            "--config",
            "release.json",
            "--cluster-id",
            "gpu-a",
            "--cluster-id",
            "gpu-b",
        ],
    )

    assert rollout.main() == 2
    assert "takes exactly one --cluster-id" in capsys.readouterr().err


@pytest.mark.parametrize("cluster_ids", ["gpu-a", [""], [" "], [3]])
def test_programmatic_selector_validation_rejects_malformed_values(
    cluster_ids: object,
) -> None:
    with pytest.raises(ReleaseError, match="non-empty strings"):
        validate_cluster_arguments(
            argparse.Namespace(mode="drain-cluster", cluster_ids=cluster_ids)
        )


@pytest.mark.parametrize("cluster_id", [None, "gpu-a"])
def test_sync_state_keeps_its_optional_single_selector(cluster_id: str | None) -> None:
    arguments = argparse.Namespace(mode="sync-state", cluster_id=cluster_id)
    validate_cluster_arguments(arguments)

    assert arguments.cluster_id == cluster_id
    assert arguments.cluster_ids == ([] if cluster_id is None else [cluster_id])
