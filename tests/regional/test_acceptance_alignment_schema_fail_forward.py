from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_diff as diff
from gpu_fault_release import regional_release_orchestration as orchestration
from gpu_fault_release import regional_schema_change as schema
from tests.regional._release_orchestrator_support import phase_release
from tests.regional.test_release_schema_change_acceptance import _AwsRunner


def test_actual_phase_driver_snapshots_before_schema_and_never_rolls_back_after_failure(
    monkeypatch,
):
    events = []
    monkeypatch.setenv(schema.ACCEPT_SCHEMA_CHANGE_ENV, "snapshot")

    def snapshot(instance, acceptance):
        value = schema.ensure_schema_change_snapshot(
            instance, acceptance, sleep=lambda _seconds: None
        )
        events.append("snapshot-available")
        return value

    def ensure():
        assert events[-1] == "snapshot-available"
        events.append("ensure")

    def verify(_plan):
        assert "ensure" in events
        raise RuntimeError("candidate verification failed after schema changed")

    release = phase_release(
        events,
        _load_state=lambda: copy.deepcopy(release.state),
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _remote_commands_are_idle=lambda: True,
        _backup_release_secrets=lambda: {},
        _ensure_schema=ensure,
        _validate_release_quick=verify,
        rollback=lambda **_kwargs: pytest.fail("schema transition entered rollback"),
    )
    release.config = SimpleNamespace(
        **vars(release.config),
        auto_rollback=True,
        database_schema_version=18,
        aws_region="us-east-1",
        namespace="gpu-fault-system",
        health=SimpleNamespace(aurora_cluster_id="isolated-database"),
    )
    release.release_id = "schema-candidate"
    release.runner = _AwsRunner()
    monkeypatch.setattr(orchestration, "ensure_schema_change_snapshot", snapshot)
    with pytest.raises(RuntimeError, match="after schema"):
        orchestration.upgrade_release(
            release,
            diff=diff.ReleaseDiff(
                diff.ReleaseChangeKind.FULL, frozenset({"database_schema"})
            ),
        )
    assert release.state["phase"] == "failed"
    assert "schema-ready" in release.state["completed_phases"]
    assert release.state["schema_change_acceptance"]["snapshot_status"] == "available"
    assert events.index("snapshot-available") < events.index("ensure")
