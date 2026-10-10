"""The CPU apply environment names the schedulability mode and, when the
administrator CLI materialized the config, where the apply keeps its
evidence; the rollback context asks the preflight to warn only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_rendering as rendering
from gpu_fault_release import regional_release_rollback_context as rollback_context
from gpu_fault_release import rollout as MODULE
from tests.regional._release_orchestrator_support import config_file


def test_evidence_directory_follows_the_managed_state_directory(tmp_path: Path) -> None:
    assert rendering.control_plane_apply_evidence_environment(
        SimpleNamespace(admin_state_dir=tmp_path)
    ) == {"GPU_FAULT_CONTROL_PLANE_EVIDENCE_DIR": str(tmp_path / "control-plane-apply")}
    assert (
        rendering.control_plane_apply_evidence_environment(
            SimpleNamespace(admin_state_dir=None)
        )
        == {}
    )
    assert rendering.control_plane_apply_evidence_environment(object()) == {}


def test_forward_apply_refuses_and_records_under_the_state_directory(
    tmp_path: Path,
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))

    environment = rendering.build_cpu_apply_environment(release, finalize=True)
    assert environment["GPU_FAULT_CONTROL_PLANE_SCHEDULABILITY"] == "refuse"
    assert "GPU_FAULT_CONTROL_PLANE_EVIDENCE_DIR" not in environment, (
        "a hand-written config has no managed state directory"
    )

    release.config = config.__class__(
        **{**config.__dict__, "admin_state_dir": tmp_path / "state"}
    )
    environment = rendering.build_cpu_apply_environment(release, finalize=True)
    assert environment["GPU_FAULT_CONTROL_PLANE_EVIDENCE_DIR"] == str(
        tmp_path / "state" / "control-plane-apply"
    )


def test_rollback_environment_asks_the_preflight_to_warn_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        rollback_context, "admin_config_renderer_environment", lambda _config: {}
    )
    monkeypatch.setattr(rollback_context, "notification_digest", lambda _n: "f" * 64)
    environment = rollback_context.build_rollback_environment(
        rollback_config=SimpleNamespace(
            admin_config=SimpleNamespace(
                sha256=lambda: "a" * 64,
                role_sha256=lambda: {
                    "ingress": "i" * 64,
                    "worker": "w" * 64,
                    "spool": "s" * 64,
                },
            ),
            cpu_kubeconfig="/nonexistent/cpu.kubeconfig",
            aws_region="us-west-2",
            namespace="gpu-fault-system",
            notifications=SimpleNamespace(
                allow_email=True,
                acknowledge_external_alert_channel=False,
                ses_configuration_set=None,
            ),
            notification_environment=lambda: {},
            admin_state_dir=tmp_path,
        ),
        metadata={},
        cpu_wheel="previous-cpu-wheel",
        cpu_sha="c" * 64,
        artifact="artifact",
        config_digest="config",
        runtime_profile_version="profile-v1",
        runtime_image="registry.example/runtime@sha256:" + "d" * 64,
    )
    assert environment["GPU_FAULT_CONTROL_PLANE_SCHEDULABILITY"] == "warn"
    assert environment["GPU_FAULT_CONTROL_PLANE_EVIDENCE_DIR"] == str(
        tmp_path / "control-plane-apply"
    )
    assert environment["GPU_FAULT_FORCE_ROLE_RESTART"] == "true"
