"""The schema v3 image lock is enforced where an image is read, not at construction.

Live 2026-09-15: ``gpu-fault-admin uninstall`` run from a checkout ahead of the
deployed release (``GPU_FAULT_ADMIN_ALLOW_UNBOUND=1``, the sanctioned path for a
fix that has to run before it can be deployed) died at CLUSTERS_DRAINING with
"GPU_FAULT_RUNTIME_IMAGE does not match the schema v3 image lock": the site's
deployed pin (``spec.images.runtime``) and the checkout's
``dist/current-release.json`` named different digests, and ``RegionalRelease``
compared them while it was built, for every mode -- ``drain-cluster`` included,
which publishes one registry revision and reads no image. The bound CLI never
sees the conflict (its snapshot's lock is the deployed image); only the
override path did, which defeated the override's purpose.

The tests drive ``main()``: the parser, the config load and the mode dispatch
are the path the uninstall script takes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from gpu_fault_release import rollout as ROLLOUT
from tests.regional._release_orchestrator_support import config_file

LOCKED = {
    "runtime": "registry.example/gpu-fault/runtime@sha256:" + "a" * 64,
    "node_installer": "registry.example/amazonlinux@sha256:" + "b" * 64,
    "dcgm_exporter": "registry.example/dcgm-exporter@sha256:" + "c" * 64,
    "adot": "registry.example/adot@sha256:" + "d" * 64,
}
OTHER_DIGEST = "registry.example/gpu-fault/runtime@sha256:" + "e" * 64
SAME_DIGEST_OTHER_REGISTRY = "mirror.example/runtime@sha256:" + "a" * 64
IMAGE_VARIABLES = (
    "GPU_FAULT_RUNTIME_IMAGE",
    "GPU_FAULT_NODE_INSTALLER_IMAGE",
    "GPU_FAULT_DCGM_EXPORTER_IMAGE",
    "GPU_FAULT_ADOT_IMAGE",
)
REFUSAL = "GPU_FAULT_RUNTIME_IMAGE does not match the schema v3 image lock"


class _NoKubeconfigCache:
    """Stands in for the exec-plugin token cache; the tests never run kubectl."""

    def __init__(self, *_args: object, **_keywords: object) -> None:
        pass

    def __enter__(self) -> _NoKubeconfigCache:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    @staticmethod
    def rewrite_config(config: ROLLOUT.ReleaseConfig) -> ROLLOUT.ReleaseConfig:
        return config

    @staticmethod
    def refresh_if_needed() -> None:
        return None


def _schema_v3_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **environment: str
) -> ROLLOUT.ReleaseConfig:
    for variable in IMAGE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)
    config = dataclasses.replace(
        ROLLOUT.ReleaseConfig.load(config_file(tmp_path)),
        release_manifest_schema_version=3,
        release_delivery_identity={},
        locked_images=dict(LOCKED),
    )
    monkeypatch.setattr(ROLLOUT.ReleaseConfig, "load", staticmethod(lambda _p: config))
    monkeypatch.setattr(ROLLOUT, "ReleaseKubeconfigCache", _NoKubeconfigCache)
    return config


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(
        ROLLOUT.sys, "argv", ["rollout", *argv, "--config", "release.json"]
    )
    return ROLLOUT.main()


def test_construction_records_a_conflicting_pin_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _schema_v3_site(
        tmp_path, monkeypatch, GPU_FAULT_RUNTIME_IMAGE=OTHER_DIGEST
    )

    release = ROLLOUT.RegionalRelease(config, ROLLOUT.Runner(dry_run=True))

    assert release.image_lock_conflicts == {
        "GPU_FAULT_RUNTIME_IMAGE": (OTHER_DIGEST, LOCKED["runtime"])
    }
    # Until a mode is refused the value is the lock, never the conflicting pin.
    assert release.runtime_image == LOCKED["runtime"]


def test_the_same_digest_or_no_pin_is_not_a_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _schema_v3_site(
        tmp_path, monkeypatch, GPU_FAULT_RUNTIME_IMAGE=SAME_DIGEST_OTHER_REGISTRY
    )

    release = ROLLOUT.RegionalRelease(config, ROLLOUT.Runner(dry_run=True))

    assert release.image_lock_conflicts == {}
    assert release.runtime_image == SAME_DIGEST_OTHER_REGISTRY
    assert release.adot_image == LOCKED["adot"]


def test_drain_cluster_publishes_under_a_conflicting_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _schema_v3_site(tmp_path, monkeypatch, GPU_FAULT_RUNTIME_IMAGE=OTHER_DIGEST)
    drained: list[list[str]] = []
    monkeypatch.setattr(
        ROLLOUT,
        "drain_registry_clusters",
        lambda release, cluster_ids: drained.append(list(cluster_ids)),
    )

    assert _run(monkeypatch, "drain-cluster", "--cluster-id", "gpu-a") == 0

    assert drained == [["gpu-a"]]
    assert REFUSAL not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (("plan",), "plan"),
        (("upgrade",), "upgrade"),
        (("remove-cluster", "--cluster-id", "gpu-a"), "remove_cluster"),
    ],
)
def test_a_mode_that_reads_an_image_refuses_before_it_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: tuple[str, ...],
    handler: str,
) -> None:
    _schema_v3_site(tmp_path, monkeypatch, GPU_FAULT_RUNTIME_IMAGE=OTHER_DIGEST)
    monkeypatch.setattr(
        ROLLOUT.RegionalRelease,
        handler,
        lambda *_args, **_keywords: pytest.fail(
            f"{argv[0]} ran under a conflicting image pin"
        ),
    )

    assert _run(monkeypatch, *argv) == 2

    err = capsys.readouterr().err
    assert (
        f"{REFUSAL}: {argv[0]} reads this image "
        f"(configured {OTHER_DIGEST}, locked {LOCKED['runtime']})"
    ) in err


def test_only_the_registry_only_mode_is_exempt() -> None:
    # A new mode enforces the lock unless it is added here on purpose.
    assert ROLLOUT.IMAGE_LOCK_EXEMPT_MODES == frozenset({"drain-cluster"})
