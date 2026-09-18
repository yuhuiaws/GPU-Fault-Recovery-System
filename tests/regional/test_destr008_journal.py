"""Real private journal transitions; no host, cluster or database operations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional import destr008_journal as journal
from scripts.e2e.regional.regional_commands import RegionalFixtureError

SCENARIOS = ("active-gpu-pod", "agent-unavailable", "no-spare")
STAGES: tuple[journal.Stage, ...] = (
    "workload_started",
    "safety_started",
    "fixture_started",
    "post_started",
)


def execution_record(
    binding: dict[str, Any], **changes: Any
) -> journal.ExecutionRecord:
    return journal.ExecutionRecord.model_validate(
        {
            "schema_version": 1,
            "binding": copy.deepcopy(binding),
            "attempt": 7,
            "maintenance_expires_at": 1700000000,
            "release_id": "release-original",
            "profile_version": "profile-original",
            "fault_uid": "fault-original-uid",
            "spare_uid": "spare-original-uid",
            "scenarios": {name: {} for name in SCENARIOS},
            **changes,
        }
    )


def make_journal(directory: Path, **changes: Any) -> journal.ExecutionJournal:
    scenarios = changes.get("scenarios", {name: {} for name in SCENARIOS})
    binding = {
        "host_sha256": "a" * 64,
        "connections": {"gpu": {"path": "/private/config", "sha256": "b" * 64}},
        "inputs": {
            "run_dir": str(directory.resolve()),
            "scenarios": list(scenarios),
            "manifest_sha256": "c" * 64,
        },
    }
    return journal.ExecutionJournal(
        directory / "execution-owner.json",
        lambda: copy.deepcopy(binding),
        initial=execution_record(binding, **changes),
    )


def reload_journal(value: journal.ExecutionJournal) -> journal.ExecutionJournal:
    return journal.ExecutionJournal(value.path, value.binding)


def start(value: journal.ExecutionJournal, scenario: str = "active-gpu-pod") -> None:
    value.start_prewarm()
    value.start_scenario(scenario)


def test_new_journal_is_private_and_round_trips_original_authority(
    tmp_path: Path,
) -> None:
    value = make_journal(tmp_path / "private")
    assert value.resuming is False
    assert value.path.stat().st_mode & 0o777 == 0o600
    assert value.path.parent.stat().st_mode & 0o777 == 0o700
    loaded = reload_journal(value)
    assert loaded.resuming is True
    assert loaded.record == value.record
    assert loaded.record.attempt == 7
    assert loaded.record.maintenance_expires_at == 1700000000
    assert loaded.record.completed is False


def test_missing_original_is_not_recreated_by_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "private" / "execution-owner.json"
    with pytest.raises(RegionalFixtureError, match="original execution journal"):
        journal.ExecutionJournal(path, lambda: {})
    assert not path.exists(), "cleanup must not recreate the original execution journal"


def test_initial_binding_mismatch_does_not_publish_an_owner(tmp_path: Path) -> None:
    path = tmp_path / "execution-owner.json"
    binding = {"cluster": "original", "inputs": {"scenarios": list(SCENARIOS)}}
    initial = execution_record(binding)
    changed = {**binding, "cluster": "other"}
    with pytest.raises(RegionalFixtureError, match="initial binding differs"):
        journal.ExecutionJournal(path, lambda: copy.deepcopy(changed), initial=initial)
    assert not path.exists(), "a mismatched binding must not publish an owner"


def test_existing_attempt_ignores_new_execution_parameters(tmp_path: Path) -> None:
    value = make_journal(tmp_path)
    start(value)
    value.stage("active-gpu-pod", "workload_started")
    original = value.path.read_bytes()
    later = execution_record(
        value.binding(),
        attempt=99,
        maintenance_expires_at=1900000000,
        release_id="new-release",
        fault_uid="recreated-node",
    )
    resumed = journal.ExecutionJournal(value.path, value.binding, initial=later)
    assert resumed.resuming is True
    assert resumed.record == value.record
    assert resumed.path.read_bytes() == original
    with pytest.raises(RegionalFixtureError, match="cleanup-only"):
        resumed.start_prewarm()
    with pytest.raises(RegionalFixtureError, match="cannot be started"):
        resumed.start_scenario("agent-unavailable")
    with pytest.raises(RegionalFixtureError, match="no execution authority"):
        resumed.stage("active-gpu-pod", "post_started")
    assert resumed.path.read_bytes() == original


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_real_stages_survive_reopen_and_cleanup_never_grants_reexecution(
    tmp_path: Path, scenario: str
) -> None:
    value = make_journal(tmp_path)
    start(value, scenario)
    stages: tuple[journal.Stage, ...] = (
        ("fixture_started", "workload_started", "post_started")
        if scenario == "no-spare"
        else STAGES
    )
    for stage in stages:
        value.stage(scenario, stage)
        loaded = reload_journal(value)
        assert loaded.record.scenarios[scenario].state == "STARTED"
        assert getattr(loaded.record.scenarios[scenario], stage) is True
    resumed = reload_journal(value)
    resumed.cleaned_scenario(scenario)
    resumed.cleaned_scenario(scenario)
    resumed.cleaned_prewarm()
    resumed.complete()
    final = reload_journal(value)
    assert final.record.completed is True
    assert final.record.prewarm_cleaned is True
    assert final.record.scenarios[scenario].state == "CLEANED"
    assert all(getattr(final.record.scenarios[scenario], stage) for stage in stages), (
        "cleanup must preserve every recorded stage"
    )
    assert final.record.attempt == 7
    assert final.record.maintenance_expires_at == 1700000000
    with pytest.raises(RegionalFixtureError, match="cannot be started"):
        final.start_scenario(scenario)


@pytest.mark.parametrize("phase", ["started", "resumed", "completed"])
def test_prewarm_cannot_be_reissued(tmp_path: Path, phase: str) -> None:
    value = make_journal(tmp_path)
    if phase == "started":
        value.start_prewarm()
    elif phase == "resumed":
        value = reload_journal(value)
    else:
        value.cleaned_prewarm()
        value.complete()
    original = value.path.read_bytes()
    with pytest.raises(RegionalFixtureError, match="cleanup-only"):
        value.start_prewarm()
    assert value.path.read_bytes() == original


@pytest.mark.parametrize(
    "phase", ["no-prewarm", "unknown", "already-started", "other-started", "completed"]
)
def test_scenario_start_requires_the_original_available_slot(
    tmp_path: Path, phase: str
) -> None:
    value = make_journal(tmp_path)
    scenario = "active-gpu-pod"
    if phase != "no-prewarm":
        value.start_prewarm()
    if phase == "unknown":
        scenario = "not-approved"
    elif phase in {"already-started", "other-started"}:
        value.start_scenario("active-gpu-pod")
        if phase == "other-started":
            scenario = "agent-unavailable"
    elif phase == "completed":
        value.cleaned_prewarm()
        value.complete()
    original = value.path.read_bytes()
    with pytest.raises(RegionalFixtureError, match="cannot be started"):
        value.start_scenario(scenario)
    assert value.path.read_bytes() == original


def test_cleanup_releases_only_the_original_scenario_slot(tmp_path: Path) -> None:
    value = make_journal(tmp_path)
    start(value)
    resumed = reload_journal(value)
    resumed.cleaned_scenario("active-gpu-pod")
    value.start_scenario("agent-unavailable")
    persisted = reload_journal(value).record
    assert persisted.scenarios["active-gpu-pod"].state == "CLEANED"
    assert persisted.scenarios["agent-unavailable"].state == "STARTED"
    assert persisted.scenarios["no-spare"].state == "PENDING"


@pytest.mark.parametrize("stage", STAGES)
def test_each_stage_intent_is_one_shot(tmp_path: Path, stage: journal.Stage) -> None:
    value = make_journal(tmp_path)
    start(value)
    for preceding in STAGES:
        value.stage("active-gpu-pod", preceding)
        if preceding == stage:
            break
    original = value.path.read_bytes()
    with pytest.raises(RegionalFixtureError, match="stage cannot be repeated"):
        value.stage("active-gpu-pod", stage)
    assert value.path.read_bytes() == original


@pytest.mark.parametrize("phase", ["unknown", "pending", "cleaned", "resuming"])
def test_stage_requires_a_live_original_scenario(tmp_path: Path, phase: str) -> None:
    value = make_journal(tmp_path)
    start(value)
    scenario = "active-gpu-pod"
    if phase == "unknown":
        scenario = "unknown"
    elif phase == "pending":
        scenario = "agent-unavailable"
    elif phase == "cleaned":
        value.cleaned_scenario(scenario)
    else:
        value = reload_journal(value)
    original = value.path.read_bytes()
    with pytest.raises(RegionalFixtureError, match="no execution authority"):
        value.stage(scenario, "workload_started")
    assert value.path.read_bytes() == original


@pytest.mark.parametrize("scenario", ["active-gpu-pod", "unknown"])
def test_never_started_scenario_cannot_be_recorded_as_cleaned(
    tmp_path: Path, scenario: str
) -> None:
    value = make_journal(tmp_path)
    original = value.path.read_bytes()
    with pytest.raises(RegionalFixtureError, match="no cleanup intent"):
        value.cleaned_scenario(scenario)
    assert value.path.read_bytes() == original


@pytest.mark.parametrize("active", [False, True])
def test_completion_requires_prewarm_cleanup_and_no_unfinished_scenario(
    tmp_path: Path, active: bool
) -> None:
    value = make_journal(tmp_path)
    if active:
        start(value)
        value.cleaned_prewarm()
    original = value.path.read_bytes()
    with pytest.raises(ValidationError, match="lifecycle"):
        value.complete()
    assert value.path.read_bytes() == original
    assert reload_journal(value).record.completed is False


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 0},
        {"schema_version": 2},
        {"schema_version": True},
        {"attempt": 0},
        {"attempt": "7"},
        {"maintenance_expires_at": 0},
        {"release_id": ""},
        {"profile_version": ""},
        {"fault_uid": ""},
        {"spare_uid": ""},
        {"prewarm_started": 1},
        {"completed": "false"},
        {"unexpected_authority": True},
        {"scenarios": {}},
        {"scenarios": {"active-gpu-pod": {"state": "UNKNOWN"}}},
        {"scenarios": {"active-gpu-pod": {"post_started": 1}}},
        {"scenarios": {"active-gpu-pod": {"unknown": True}}},
        {
            "prewarm_started": True,
            "scenarios": {
                "active-gpu-pod": {"state": "STARTED"},
                "agent-unavailable": {"state": "STARTED"},
            },
        },
        {"completed": True},
    ],
)
def test_untrusted_record_shapes_do_not_grant_cleanup_or_execution(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    value = make_journal(tmp_path)
    document = value.record.model_dump(mode="json")
    document.update(change)
    write_json_atomic(value.path, document)
    original = value.path.read_bytes()
    with pytest.raises(RegionalFixtureError, match="invalid"):
        reload_journal(value)
    assert value.path.read_bytes() == original


@pytest.mark.parametrize("stage", STAGES)
def test_pending_scenario_cannot_hide_a_started_resource(
    tmp_path: Path, stage: journal.Stage
) -> None:
    value = make_journal(tmp_path)
    document = value.record.model_dump(mode="json")
    document["prewarm_started"] = True
    document["scenarios"]["active-gpu-pod"][stage] = True
    write_json_atomic(value.path, document)
    with pytest.raises(RegionalFixtureError, match="invalid"):
        reload_journal(value)


def test_started_scenario_requires_original_prewarm_intent(tmp_path: Path) -> None:
    value = make_journal(tmp_path)
    document = value.record.model_dump(mode="json")
    document["scenarios"]["active-gpu-pod"]["state"] = "STARTED"
    write_json_atomic(value.path, document)
    with pytest.raises(RegionalFixtureError, match="invalid"):
        reload_journal(value)


@pytest.mark.parametrize("change", ["missing", "unknown", "path-traversal"])
def test_scenario_inventory_cannot_diverge_from_the_bound_approval(
    tmp_path: Path, change: str
) -> None:
    value = make_journal(tmp_path)
    document = value.record.model_dump(mode="json")
    if change == "missing":
        document["scenarios"].pop("active-gpu-pod")
    else:
        key = "unapproved" if change == "unknown" else "../../other-run"
        document["scenarios"][key] = {"state": "PENDING"}
    write_json_atomic(value.path, document)
    with pytest.raises(RegionalFixtureError):
        reload_journal(value)


@pytest.mark.parametrize(
    "binding",
    [
        {},
        {"inputs": None},
        {"inputs": []},
        {"inputs": {}},
        {"inputs": {"scenarios": None}},
        {"inputs": {"scenarios": "active-gpu-pod"}},
        {"inputs": {"scenarios": []}},
        {"inputs": {"scenarios": [True]}},
        {"inputs": {"scenarios": [1]}},
        {"inputs": {"scenarios": [None]}},
        {"inputs": {"scenarios": [{}]}},
        {"inputs": {"scenarios": ["unknown"]}},
        {"inputs": {"scenarios": ["../../other-run"]}},
        {"inputs": {"scenarios": [*SCENARIOS, "active-gpu-pod"]}},
    ],
)
def test_approval_requires_an_explicit_unique_list_of_known_scenarios(
    binding: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="lifecycle"):
        execution_record(binding)


@pytest.mark.parametrize(
    "content", ["{", "[]", '{"schema_version":1,"schema_version":1}']
)
def test_corrupt_or_ambiguous_json_is_not_a_new_execution(
    tmp_path: Path, content: str
) -> None:
    value = make_journal(tmp_path)
    value.path.write_text(content)
    with pytest.raises(RegionalFixtureError):
        reload_journal(value)
    assert value.path.read_text() == content


def test_deleted_journal_during_an_action_is_not_recreated(tmp_path: Path) -> None:
    value = make_journal(tmp_path)
    value.path.unlink()
    with pytest.raises(RegionalFixtureError, match="invalid"):
        value.start_prewarm()
    assert not value.path.exists(), (
        "an action must not recreate a deleted ownership journal"
    )


@pytest.mark.parametrize(
    "action",
    [
        "start_prewarm",
        "start_scenario",
        "stage",
        "cleaned_scenario",
        "cleaned_prewarm",
        "complete",
    ],
)
def test_every_transition_reloads_current_connection_and_source_binding(
    tmp_path: Path, action: str
) -> None:
    value = make_journal(tmp_path)
    start(value)
    binding = value.binding()
    value.binding = lambda: copy.deepcopy(binding)
    binding["inputs"]["manifest_sha256"] = "d" * 64
    original = value.path.read_bytes()
    arguments: tuple[Any, ...] = {
        "start_scenario": ("agent-unavailable",),
        "stage": ("active-gpu-pod", "workload_started"),
        "cleaned_scenario": ("active-gpu-pod",),
    }.get(action, ())
    with pytest.raises(RegionalFixtureError, match="input or connection drifted"):
        getattr(value, action)(*arguments)
    assert value.path.read_bytes() == original


def test_source_loss_never_overwrites_the_last_durable_intent(tmp_path: Path) -> None:
    value = make_journal(tmp_path)
    start(value)
    original = value.path.read_bytes()

    def lost_source() -> dict[str, Any]:
        raise FileNotFoundError("original bound source is unavailable")

    value.binding = lost_source
    with pytest.raises(FileNotFoundError, match="bound source"):
        value.cleaned_scenario("active-gpu-pod")
    assert value.path.read_bytes() == original


def test_failed_atomic_write_does_not_become_persisted_execution_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = make_journal(tmp_path)
    original = value.path.read_bytes()

    def fail_write(path: Path, document: dict[str, Any]) -> None:
        assert path == value.path
        assert document["prewarm_started"] is True
        raise OSError("temporary storage unavailable")

    monkeypatch.setattr(journal, "write_json_atomic", fail_write)
    with pytest.raises(OSError, match="storage unavailable"):
        value.start_prewarm()
    assert value.path.read_bytes() == original
    assert json.loads(original)["prewarm_started"] is False
    assert reload_journal(value).record.prewarm_started is False
