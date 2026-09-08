from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault.admin.config_patch import preset_admin_config
from gpu_fault_release import regional_release_diff as DIFF_MODULE
from gpu_fault_release import rollout as MODULE
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]


def test_admin_config_renders_and_targets_only_changed_cpu_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = config_file(tmp_path)
    document = json.loads(path.read_text())
    admin_config = preset_admin_config("32-disabled")
    document["admin_config"] = {
        "config": admin_config.as_dict(),
        "config_sha256": admin_config.sha256(),
        "role_sha256": admin_config.role_sha256(),
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    config = MODULE.ReleaseConfig.load(path)

    class RecordingRunner:
        dry_run = True

        def __init__(self) -> None:
            self.calls = []

        def run(self, args, **kwargs):
            self.calls.append((args, kwargs))
            return ""

        def probe(self, _args, **_kwargs):
            return False

    runner = RecordingRunner()
    release = MODULE.RegionalRelease(config, runner)
    monkeypatch.setattr(release, "_ensure_contexts", lambda: None)
    monkeypatch.setattr(release, "_require_cpu_secrets", lambda: None)
    monkeypatch.setattr(release, "_apply_rds_ca_bundle", lambda: None)
    monkeypatch.setattr(release, "_refresh_aurora_credentials", lambda: None)
    monkeypatch.setattr(release, "_remote_commands_are_idle", lambda: True)
    monkeypatch.setattr(
        release, "_capture_previous", lambda **_kwargs: {"metadata": {}}
    )
    monkeypatch.setattr(release, "_backup_release_secrets", lambda: {})
    monkeypatch.setattr(release, "_save_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(release, "_upload_release", lambda _diff: None)
    monkeypatch.setattr(release, "_stage_registry", lambda: False)
    monkeypatch.setattr(release, "_validate_release_quick", lambda _plan: None)
    monkeypatch.setattr(MODULE, "ensure_notification_secret", lambda _release: None)
    diff = DIFF_MODULE.ReleaseDiff(
        kind=DIFF_MODULE.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"admin_config_worker"}),
    )

    release.upgrade(diff=diff)

    render_call, apply_call = runner.calls
    assert "render-control-plane-role-split.sh" in render_call[0][1]
    assert "apply-control-plane-role-split.sh" in apply_call[0][1]
    assert apply_call[1]["env"]["GPU_FAULT_CONTROL_PLANE_ROLE_TARGETS"] == "worker"
    assert render_call[1]["env"]["GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION"] == "128"
    assert render_call[1]["env"]["GPU_FAULT_TELEMETRY_SPOOL"] == "false"
    assert apply_call[1]["env"]["GPU_FAULT_ROLE_SPLIT_GENERATED_DIR"]
