"""``gpu-fault-admin deploy --accept-schema-change`` in the release engine.

A release that changes the PostgreSQL schema cannot be rolled back, so the
transaction gate refused it under ``autoRollback: true`` and the operator had
to edit ``site.yaml`` to fail-forward and remember to edit it back. The
acceptance now travels as one environment variable, is recorded in the
transaction state, takes an Aurora snapshot before the schema Jobs, and turns
this one transaction fail-forward without touching the site.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli as admin_cli
from scripts import release_failure_recovery
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/rollout_regional_release.py"
)
DIFF = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_diff.py"
)
ORCHESTRATION = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_orchestration.py"
)
SCHEMA_CHANGE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_schema_change.py"
)

ENV = "GPU_FAULT_RELEASE_ACCEPT_SCHEMA_CHANGE"


def _schema_diff() -> object:
    return DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.FULL, changed=frozenset({"database_schema"})
    )


def _release(*, state: dict | None = None, load_state: dict | None = None):
    class ValidationPassed(RuntimeError):
        pass

    release = SimpleNamespace(
        config=SimpleNamespace(
            auto_rollback=True,
            schema_rollback_compatible=False,
            database_schema_version=12,
            aws_region="us-west-2",
            namespace="gpu-fault-system",
            health=SimpleNamespace(aurora_cluster_id=None),
            clusters=(),
        ),
        release_id="rel-abcdef1234567890",
        state=state if state is not None else {},
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _remote_commands_are_idle=lambda: True,
        _load_state=lambda: dict(load_state or {}),
        _capture_previous=lambda **_kwargs: (_ for _ in ()).throw(ValidationPassed()),
    )
    release.ValidationPassed = ValidationPassed
    return release


def test_the_three_places_that_spell_the_variable_agree() -> None:
    assert SCHEMA_CHANGE.ACCEPT_SCHEMA_CHANGE_ENV == ENV
    assert admin_cli.ACCEPT_SCHEMA_CHANGE_ENV == ENV
    assert release_failure_recovery.ACCEPT_SCHEMA_CHANGE_ENV == ENV
    assert admin_cli.SCHEMA_CHANGE_SNAPSHOT_MODE == SCHEMA_CHANGE.SNAPSHOT_MODE
    assert admin_cli.SCHEMA_CHANGE_NO_SNAPSHOT_MODE == SCHEMA_CHANGE.NO_SNAPSHOT_MODE


def test_schema_change_is_refused_with_the_flag_named(monkeypatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    release = _release()

    with pytest.raises(MODULE.ReleaseError, match="--accept-schema-change") as excinfo:
        ORCHESTRATION.upgrade_release(release, diff=_schema_diff())
    assert "PostgreSQL schema change" in str(excinfo.value)


def test_an_unknown_mode_value_is_refused_not_read_as_consent(monkeypatch) -> None:
    monkeypatch.setenv(ENV, "yes")
    with pytest.raises(MODULE.ReleaseError, match="must be snapshot or no-snapshot"):
        ORCHESTRATION.upgrade_release(_release(), diff=_schema_diff())


def test_accepted_schema_change_passes_the_gate_and_is_recorded(monkeypatch) -> None:
    monkeypatch.setenv(ENV, "snapshot")
    release = _release()

    acceptance = SCHEMA_CHANGE.resolve_acceptance(
        release, changed=_schema_diff().changed, resume=False
    )

    assert acceptance["mode"] == "snapshot"
    assert acceptance["database_schema_version"] == 12
    assert acceptance["snapshot_id"] is None, "the snapshot is taken at the schema step"
    with pytest.raises(release.ValidationPassed):
        ORCHESTRATION.upgrade_release(release, diff=_schema_diff())


def test_a_resumed_transaction_keeps_its_recorded_acceptance(monkeypatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    recorded = {
        "mode": "snapshot",
        "accepted_at": "2026-09-07T10:00:00+00:00",
        "database_schema_version": 12,
        "snapshot_id": "gpu-fault-pre-v12-rel-abcdef1",
        "snapshot_status": "available",
        "cluster_id": "gpu-fault-aurora",
    }
    release = _release(load_state={"schema_change_acceptance": recorded})

    acceptance = SCHEMA_CHANGE.resolve_acceptance(
        release, changed=_schema_diff().changed, resume=True
    )

    assert acceptance == recorded


def test_no_acceptance_is_needed_when_the_schema_does_not_change(monkeypatch) -> None:
    monkeypatch.setenv(ENV, "snapshot")
    release = _release()
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    assert (
        SCHEMA_CHANGE.resolve_acceptance(release, changed=diff.changed, resume=False)
        is None
    )


def test_an_accepted_transaction_does_not_roll_back_on_failure(monkeypatch) -> None:
    """The rollback would be refused at the schema anyway; the state keeps the
    acceptance so the recovery can name the snapshot."""

    monkeypatch.setenv(ENV, "no-snapshot")
    rolled_back: list[object] = []

    class FakeRelease:
        # ``upgrade_release`` rebinds ``self.state`` for a new transaction, so
        # the state writer has to follow the attribute, not a captured dict.
        config = _release().config
        release_id = "rel-abcdef1234567890"
        state: dict = {}

        def _ensure_contexts(self) -> None:
            pass

        def _require_cpu_secrets(self) -> None:
            pass

        def _remote_commands_are_idle(self) -> bool:
            return True

        def _load_state(self) -> dict:
            return {}

        def _capture_previous(self, **_kwargs) -> dict:
            return {"cpu": {"wheel": "previous"}}

        def _backup_release_secrets(self) -> dict:
            return {}

        def _save_state(self, phase: str, **updates) -> None:
            self.state.update({"phase": phase, **updates})

        def rollback(self, **kwargs) -> None:
            rolled_back.append(kwargs)

    release = FakeRelease()
    monkeypatch.setattr(
        ORCHESTRATION,
        "run_upgrade_phases",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("verify failed")),
    )

    with pytest.raises(RuntimeError, match="verify failed"):
        ORCHESTRATION.upgrade_release(release, diff=_schema_diff())

    assert rolled_back == [], "an accepted schema change must run fail-forward"
    assert release.state["phase"] == "failed"
    assert release.state["schema_change_acceptance"]["mode"] == "no-snapshot"


def test_rollback_across_the_schema_names_the_snapshot_to_restore() -> None:
    release = SimpleNamespace(
        config=SimpleNamespace(auto_rollback=True, schema_rollback_compatible=False),
        state={
            "release_diff": {"changed": ["database_schema"]},
            "schema_change_acceptance": {
                "mode": "snapshot",
                "snapshot_id": "gpu-fault-pre-v12-rel-abcdef1",
            },
        },
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _remote_commands_are_idle=lambda: True,
    )

    with pytest.raises(MODULE.ReleaseError, match="gpu-fault-pre-v12-rel-abcdef1"):
        ORCHESTRATION.rollback_release(release, state={"metadata": {}})


class _AwsRunner:
    """A runner that plays Aurora: snapshots appear on create and become
    available after a configurable number of describes."""

    def __init__(self, *, available_after: int = 1, fail_create: bool = False):
        self.commands: list[list[str]] = []
        self.snapshots: dict[str, str] = {}
        self.describes = 0
        self.available_after = available_after
        self.fail_create = fail_create

    def run(self, args, *, capture=False, sensitive=False, **_kwargs):
        self.commands.append(list(args))
        if args[:3] == ["aws", "rds", "describe-db-cluster-snapshots"]:
            self.describes += 1
            items = []
            for name, status in list(self.snapshots.items()):
                if status == "creating" and self.describes >= self.available_after:
                    self.snapshots[name] = status = "available"
                items.append({"DBClusterSnapshotIdentifier": name, "Status": status})
            return json.dumps({"DBClusterSnapshots": items})
        if args[:3] == ["aws", "rds", "create-db-cluster-snapshot"]:
            if self.fail_create:
                raise MODULE.ReleaseError("command failed (254): aws")
            name = args[args.index("--db-cluster-snapshot-identifier") + 1]
            self.snapshots[name] = "creating"
            return json.dumps({"DBClusterSnapshot": {"Status": "creating"}})
        raise AssertionError(f"unexpected command {args}")


def _snapshot_release(
    runner: _AwsRunner, *, cluster_id: str | None = "gpu-fault-aurora"
):
    url = "postgresql://user:pw@gpu-fault-us-west-2-cp.cluster-abc.us-west-2.rds.amazonaws.com:5432/gpu_fault"
    secret = {"data": {"postgres-url": base64.b64encode(url.encode()).decode()}}
    return SimpleNamespace(
        config=SimpleNamespace(
            aws_region="us-west-2",
            namespace="gpu-fault-system",
            database_schema_version=12,
            health=SimpleNamespace(aurora_cluster_id=cluster_id),
        ),
        release_id="Rel_ABCDEF1234567890",
        runner=runner,
        _cpu=lambda *args: ["kubectl", *args],
        _get_json=lambda _args: secret,
    )


def _acceptance(mode: str = "snapshot") -> dict:
    return {
        "mode": mode,
        "accepted_at": "2026-09-07T10:00:00+00:00",
        "database_schema_version": 12,
        "snapshot_id": None,
        "snapshot_status": None,
        "cluster_id": None,
    }


def test_snapshot_is_created_named_per_release_and_waited_for() -> None:
    runner = _AwsRunner(available_after=3)
    sleeps: list[float] = []

    result = SCHEMA_CHANGE.ensure_schema_change_snapshot(
        _snapshot_release(runner), _acceptance(), sleep=sleeps.append, clock=lambda: 0.0
    )

    assert result["snapshot_id"] == "gpu-fault-pre-v12-rel-abcdef12"
    assert result["snapshot_status"] == "available"
    assert result["cluster_id"] == "gpu-fault-aurora"
    creates = [c for c in runner.commands if c[2] == "create-db-cluster-snapshot"]
    assert len(creates) == 1
    assert "--db-cluster-identifier" in creates[0]
    assert (
        creates[0][creates[0].index("--db-cluster-identifier") + 1]
        == "gpu-fault-aurora"
    )
    assert sleeps, "the engine waited for the snapshot to become available"


def test_an_existing_snapshot_with_the_release_name_is_reused() -> None:
    runner = _AwsRunner()
    runner.snapshots["gpu-fault-pre-v12-rel-abcdef12"] = "available"

    result = SCHEMA_CHANGE.ensure_schema_change_snapshot(
        _snapshot_release(runner), _acceptance(), sleep=lambda _s: None
    )

    assert result["snapshot_status"] == "available"
    assert not any(c[2] == "create-db-cluster-snapshot" for c in runner.commands), (
        "a resume must not take a second snapshot"
    )


def test_no_snapshot_mode_touches_nothing() -> None:
    runner = _AwsRunner()

    result = SCHEMA_CHANGE.ensure_schema_change_snapshot(
        _snapshot_release(runner), _acceptance("no-snapshot")
    )

    assert result["snapshot_id"] is None
    assert runner.commands == []


def test_cluster_identifier_falls_back_to_the_aurora_secret() -> None:
    runner = _AwsRunner()

    result = SCHEMA_CHANGE.ensure_schema_change_snapshot(
        _snapshot_release(runner, cluster_id=None), _acceptance(), sleep=lambda _s: None
    )

    assert result["cluster_id"] == "gpu-fault-us-west-2-cp"


def test_a_snapshot_that_cannot_be_taken_stops_before_the_schema_with_a_remedy() -> (
    None
):
    runner = _AwsRunner(fail_create=True)

    with pytest.raises(MODULE.ReleaseError) as excinfo:
        SCHEMA_CHANGE.ensure_schema_change_snapshot(
            _snapshot_release(runner), _acceptance(), sleep=lambda _s: None
        )

    message = str(excinfo.value)
    assert "rds:CreateDBClusterSnapshot" in message
    assert "--accept-schema-change-without-snapshot" in message


def test_a_snapshot_that_never_becomes_available_times_out_before_the_schema(
    monkeypatch,
) -> None:
    monkeypatch.setenv(SCHEMA_CHANGE.SNAPSHOT_WAIT_SECONDS_ENV, "30")
    runner = _AwsRunner(available_after=10_000)
    ticks = iter([0.0, 10.0, 20.0, 31.0, 40.0])

    with pytest.raises(MODULE.ReleaseError, match="did not become available"):
        SCHEMA_CHANGE.ensure_schema_change_snapshot(
            _snapshot_release(runner),
            _acceptance(),
            sleep=lambda _s: None,
            clock=lambda: next(ticks),
        )


def test_release_driver_mirrors_the_acceptance_only_for_schema_releases() -> None:
    accepted = {ENV: "snapshot"}
    schema = {"next_deploy": {"changed": ["database_schema", "control_plane_wheel"]}}
    plain = {"next_deploy": {"changed": ["control_plane_wheel"]}}
    unknown = {"status": "APPLIED", "fast_path": False}

    assert release_failure_recovery.schema_change_fail_forward(schema, accepted), (
        "an accepted schema release must run fail-forward"
    )
    assert release_failure_recovery.schema_change_fail_forward(plain, accepted) is None
    assert release_failure_recovery.schema_change_fail_forward(unknown, accepted), (
        "with no readable diff the acceptance alone decides"
    )
    assert release_failure_recovery.schema_change_fail_forward(schema, {}) is None
