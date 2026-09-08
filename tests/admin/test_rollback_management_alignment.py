from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin.config import (
    AuroraCapacityConfig,
    default_admin_config,
    load_desired_admin_config,
    persist_desired_admin_config,
)
from gpu_fault.admin.rollback_alignment import (
    find_previous_release_manifest,
    materialized_rollback_status_site,
    reconcile_rollback_management,
    rollback_management_document,
)
from gpu_fault.admin.site import SiteConfigError


def _fixture(tmp_path: Path):
    repository = tmp_path / "current-repository"
    repository.mkdir()
    old_repository = tmp_path / "source-snapshots" / "snapshot-a" / "repository-a"
    manifest = old_repository / "dist" / "old-release" / "release.json"
    manifest.parent.mkdir(parents=True)
    wheel = manifest.parent / "runtime.whl"
    bundle = manifest.parent / "bundle.tar.gz"
    wheel.write_bytes(b"wheel")
    bundle.write_bytes(b"bundle")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "release_id": "old-release",
                "wheel": "dist/old-release/runtime.whl",
                "bundle": "dist/old-release/bundle.tar.gz",
                "components": {
                    "control_plane": {"wheel": "dist/old-release/runtime.whl"},
                    "executor": {"wheel": "dist/old-release/runtime.whl"},
                    "node_runtime": {"wheel": "dist/old-release/runtime.whl"},
                },
                "delivery": {
                    "sha256": "d" * 64,
                    "images": {
                        "runtime": {"reference": "old-runtime"},
                        "node_installer": {"reference": "old-installer"},
                        "dcgm_exporter": {"reference": "old-dcgm"},
                        "adot": {"reference": "old-adot"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "profiles" / "profile-v1.yaml"
    profile.parent.mkdir()
    profile.write_text("profile_version: profile-v1\n", encoding="utf-8")
    site = tmp_path / "site.yaml"
    site.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "RegionalSite",
                "metadata": {"name": "test-site"},
                "spec": {
                    "repositoryRoot": str(repository),
                    "release": {
                        "manifest": "dist/current-release.json",
                        "agentConfigDigest": "c" * 64,
                    },
                    "runtimeProfile": {
                        "source": str(profile),
                        "templateSource": str(profile),
                        "version": "profile-v2",
                    },
                    "images": {
                        "runtime": "candidate-runtime",
                        "nodeInstaller": "candidate-installer",
                        "dcgmExporter": "candidate-dcgm",
                        "adot": "candidate-adot",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    admin_config = default_admin_config()
    live_state = {
        "phase": "rolled-back",
        "rollback_result": {"status": "PASSED"},
        "previous": {
            "release_id": "old-release",
            "release_delivery_sha256": "d" * 64,
            "runtime_profile_version": "profile-v1",
            "runtime_image": "old-runtime",
            "node_installer_image": "old-installer",
            "adot_image": "old-adot",
            "admin_config": admin_config.as_dict(),
            "metadata": {"required-agent-config-digest": "a" * 64},
            "clusters": {"gpu-a": {"dcgm_image": "old-dcgm"}},
        },
    }
    return site, manifest, live_state, admin_config


def test_rollback_management_document_uses_immutable_previous_release(
    tmp_path: Path,
) -> None:
    site, manifest, live_state, admin_config = _fixture(tmp_path)

    document, restored_config, repository_root = rollback_management_document(
        site, live_state
    )

    assert document["spec"]["repositoryRoot"] == str(repository_root)
    assert document["spec"]["release"]["manifest"] == "dist/old-release/release.json"
    assert document["spec"]["release"]["agentConfigDigest"] == "a" * 64
    assert document["spec"]["runtimeProfile"]["version"] == "profile-v1"
    assert document["spec"]["images"] == {
        "runtime": "old-runtime",
        "nodeInstaller": "old-installer",
        "dcgmExporter": "old-dcgm",
        "adot": "old-adot",
    }
    assert restored_config == admin_config


def test_reconcile_rollback_management_restores_site_and_admin_config(
    tmp_path: Path,
) -> None:
    site, manifest, live_state, admin_config = _fixture(tmp_path)

    reconcile_rollback_management(site, live_state, source="rollback:test")

    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    management_manifest = Path(document["spec"]["release"]["manifest"])
    assert management_manifest.name == "release.status.json"
    assert management_manifest.is_file(), "rollback management manifest was not written"
    assert document["spec"]["repositoryRoot"].endswith("current-repository"), (
        "rollback management site did not restore the previous repository"
    )
    assert load_desired_admin_config(tmp_path) == admin_config


def test_rolled_back_status_uses_temporary_management_baseline(tmp_path: Path) -> None:
    site, manifest, live_state, admin_config = _fixture(tmp_path)

    with materialized_rollback_status_site(site, live_state) as temporary:
        document = yaml.safe_load(temporary.read_text(encoding="utf-8"))
        management_manifest = Path(document["spec"]["release"]["manifest"])
        assert management_manifest.name == "release.status.json"
        assert management_manifest.is_file(), (
            "rollback management manifest was not reused"
        )
        assert document["spec"]["repositoryRoot"].endswith("current-repository"), (
            "idempotent rollback alignment changed the previous repository"
        )
        assert load_desired_admin_config(temporary.parent) == admin_config

    assert not temporary.exists(), "temporary rollback status site was not removed"


def test_previous_manifest_identity_ignores_build_timestamp(tmp_path: Path) -> None:
    site, manifest, live_state, _admin_config = _fixture(tmp_path)
    first = json.loads(manifest.read_text(encoding="utf-8"))
    first["created_at"] = "2026-09-03T00:00:00Z"
    manifest.write_text(json.dumps(first), encoding="utf-8")
    duplicate = (
        tmp_path
        / "source-snapshots"
        / "snapshot-b"
        / "repository-b"
        / "dist"
        / "old-release"
        / "release.json"
    )
    duplicate.parent.mkdir(parents=True)
    second = dict(first)
    second["created_at"] = "2026-09-03T01:00:00Z"
    duplicate.write_text(json.dumps(second), encoding="utf-8")

    selected = find_previous_release_manifest(tmp_path, live_state["previous"])

    assert selected in {manifest, duplicate}


def test_previous_manifest_prefers_physical_identity_over_stale_release_id(
    tmp_path: Path,
) -> None:
    snapshots = tmp_path / "source-snapshots"

    def write_manifest(
        snapshot: str,
        release_id: str,
        delivery_sha256: str,
        *,
        control_plane: str,
        executor: str,
        executor_compatibility: str,
        agent: str,
        agent_compatibility: str,
        bundle: str,
        template: str,
    ) -> Path:
        path = (
            snapshots / snapshot / "repository-a" / "dist" / release_id / "release.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "release_id": release_id,
                    "bundle_sha256": bundle,
                    "protocol_versions": {"agent": 3, "executor": 2},
                    "components": {
                        "control_plane": {"wheel_sha256": control_plane},
                        "executor": {
                            "wheel_sha256": executor,
                            "module_digest": executor_compatibility,
                        },
                        "node_runtime": {
                            "wheel_sha256": agent,
                            "module_digest": agent_compatibility,
                        },
                    },
                    "delivery": {
                        "sha256": delivery_sha256,
                        "components": {"node_bundle": {"template_sha256": template}},
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    actual = {
        "control_plane": "1" * 64,
        "executor": "2" * 64,
        "executor_compatibility": "3" * 64,
        "agent": "4" * 64,
        "agent_compatibility": "5" * 64,
        "bundle": "6" * 64,
        "template": "7" * 64,
    }
    stale = write_manifest(
        "stale",
        "stale-release",
        "a" * 64,
        control_plane="8" * 64,
        executor="8" * 64,
        executor_compatibility="8" * 64,
        agent="8" * 64,
        agent_compatibility="8" * 64,
        bundle="8" * 64,
        template="8" * 64,
    )
    physical = write_manifest("physical", "actual-release", "b" * 64, **actual)
    previous = {
        "release_id": stale.parent.name,
        "release_delivery_sha256": "a" * 64,
        "cpu_wheel_sha256": actual["control_plane"],
        "metadata": {
            "required-regional-executor-artifact-sha256": actual["executor"],
            "required-regional-executor-compatibility-digest": (
                actual["executor_compatibility"]
            ),
            "required-regional-executor-protocol-version": "2",
            "required-agent-artifact-sha256": actual["agent"],
            "required-agent-compatibility-digest": actual["agent_compatibility"],
            "required-agent-protocol-version": "3",
        },
        "clusters": {
            "gpu-a": {
                "bundle_sha256": actual["bundle"],
                "template_sha256": actual["template"],
            }
        },
    }

    selected = find_previous_release_manifest(tmp_path, previous)

    assert selected == physical


def test_ambiguous_physical_identity_is_narrowed_by_the_named_release(
    tmp_path: Path,
) -> None:
    """Config-only releases build byte-identical artifacts, so physical ties.

    Every accumulated source snapshot that pins the same wheels, images and
    node template satisfies the physical check at once, which used to refuse the
    rollback outright. Exactly one of them also carries the previous
    ``release_id`` and delivery digest, and that is the answer -- already on
    disk, so refusing would strand the rollback for no reason.
    """

    snapshots = tmp_path / "source-snapshots"
    artifacts = {
        "control_plane": "1" * 64,
        "executor": "2" * 64,
        "executor_compatibility": "3" * 64,
        "agent": "4" * 64,
        "agent_compatibility": "5" * 64,
        "bundle": "6" * 64,
        "template": "7" * 64,
    }

    def write_manifest(snapshot: str, release_id: str, delivery_sha256: str) -> Path:
        path = (
            snapshots / snapshot / "repository-a" / "dist" / release_id / "release.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "release_id": release_id,
                    "bundle_sha256": artifacts["bundle"],
                    "protocol_versions": {"agent": 3, "executor": 2},
                    "components": {
                        "control_plane": {"wheel_sha256": artifacts["control_plane"]},
                        "executor": {
                            "wheel_sha256": artifacts["executor"],
                            "module_digest": artifacts["executor_compatibility"],
                        },
                        "node_runtime": {
                            "wheel_sha256": artifacts["agent"],
                            "module_digest": artifacts["agent_compatibility"],
                        },
                    },
                    "delivery": {
                        "sha256": delivery_sha256,
                        "components": {
                            "node_bundle": {"template_sha256": artifacts["template"]}
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    write_manifest("earlier", "2cec792611e4", "a" * 64)
    write_manifest("later", "7b024e0390dd", "b" * 64)
    named = write_manifest("named", "6fe471cb20d4", "c" * 64)
    previous = {
        "release_id": "6fe471cb20d4",
        "release_delivery_sha256": "c" * 64,
        "cpu_wheel_sha256": artifacts["control_plane"],
        "metadata": {
            "required-regional-executor-artifact-sha256": artifacts["executor"],
            "required-regional-executor-compatibility-digest": (
                artifacts["executor_compatibility"]
            ),
            "required-regional-executor-protocol-version": "2",
            "required-agent-artifact-sha256": artifacts["agent"],
            "required-agent-compatibility-digest": artifacts["agent_compatibility"],
            "required-agent-protocol-version": "3",
        },
        "clusters": {
            "gpu-a": {
                "bundle_sha256": artifacts["bundle"],
                "template_sha256": artifacts["template"],
            }
        },
    }

    selected = find_previous_release_manifest(tmp_path, previous)

    assert selected == named


def test_rollback_keeps_capacity_the_snapshot_never_recorded(tmp_path: Path) -> None:
    """A snapshot's silence is not an instruction to reset the field.

    ``previous.admin_config`` only carries the fields the release that wrote it
    knew about, and the parser reads a missing ``aurora`` block as the 0.5/8 ACU
    window such records used to imply. On a site running 8/32 ACU that guess is
    wrong twice over: the rollback persists it as though an administrator chose
    it, and the next deployment reconciles the live cluster down to it.
    """

    site, _manifest, live_state, admin_config = _fixture(tmp_path)
    desired = replace(
        admin_config, aurora=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0)
    )
    persist_desired_admin_config(tmp_path, config=desired, source="approved:test")
    snapshot = dict(admin_config.as_dict())
    del snapshot["aurora"]
    live_state["previous"]["admin_config"] = snapshot

    _document, restored, _root = rollback_management_document(site, live_state)

    assert restored.aurora == desired.aurora, (
        "the rollback invented an ACU window the snapshot never recorded"
    )
    assert restored.capacity == admin_config.capacity, (
        "the rollback stopped restoring the fields the snapshot does record"
    )


def test_rollback_still_restores_capacity_the_snapshot_does_record(
    tmp_path: Path,
) -> None:
    """Keeping unrecorded fields must not turn the rollback into a no-op."""

    site, _manifest, live_state, admin_config = _fixture(tmp_path)
    persist_desired_admin_config(
        tmp_path,
        config=replace(
            admin_config, aurora=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0)
        ),
        source="approved:test",
    )
    recorded = replace(
        admin_config, aurora=AuroraCapacityConfig(min_acu=16.0, max_acu=64.0)
    )
    live_state["previous"]["admin_config"] = recorded.as_dict()

    _document, restored, _root = rollback_management_document(site, live_state)

    assert restored.aurora == recorded.aurora


@pytest.mark.parametrize(
    "release_id", ["../../etc/evil", "with/slash", "..", "x" * 200]
)
def test_a_malformed_previous_release_id_is_rejected(
    tmp_path: Path, release_id: str
) -> None:
    """M-13: the recorded release_id becomes a directory below the state dir.

    It arrives from live release state the CLI did not author, so a value with a
    path separator or ``..`` segment would place the management manifest outside
    the state directory. It is bounded to the anchored release_id shape first.
    """

    site, _manifest, live_state, _admin_config = _fixture(tmp_path)
    live_state["previous"]["release_id"] = release_id

    with pytest.raises(SiteConfigError, match="release_id is malformed"):
        reconcile_rollback_management(site, live_state, source="rollback:test")
