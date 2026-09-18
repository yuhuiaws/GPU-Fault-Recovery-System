"""COLLECT004 rejects unknown ownership and retains recovery evidence on failure."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.e2e.regional import collector_inventory_reboot as runner
from scripts.e2e.regional.collector_action_guard import action_window
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeMissingResponseError,
    HostProbeTransportError,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture
from tests.regional._collector_inventory_reboot_support import (
    APPLIED,
    BASELINE,
    INTENT,
    ORIGIN,
    InventoryHarness,
)


def test_inventory_reboot_samples_then_publishes_without_configuration_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    result = harness.run()
    assert result["verdict"] == "PASS", result
    assert harness.restores == 0
    assert (
        harness.calls.index("sample-gpu-inventory")
        < harness.calls.index("publish-gpu-inventory")
        < harness.calls.index("node-ready")
    )
    assert "override-expected-gpu-count" not in harness.calls
    assert "restore-collector-env" not in harness.calls
    assert result["production_configuration_modified"] is False
    assert result["production_service_restarted_by_runner"] is False
    assert result["isolated_sampling"]["live_delivery_proven"] is True
    receipt = json.loads((tmp_path / "inventory-operation-a1.json").read_text())
    assert receipt["run_id"] == harness.run_id
    assert len(receipt["owner_nonce"]) == 32
    assert (tmp_path / "inventory-operation-a1.json").stat().st_mode & 0o077 == 0
    assert "owner_nonce" not in result
    assert harness.cleanup.finish(profile_version="v1", reason="test")["errors"] == []
    assert harness.calls[-1] == "incident-cleanup"


@pytest.mark.parametrize(
    "problem",
    [
        "sample",
        "extra-workflow",
        "missing-restart",
        "failed-workflow",
        "digest",
        "same-boot",
        "node-identity",
        "provider",
        "provider-identity",
        "recovery-read",
        "publication",
        "poll",
    ],
)
def test_inventory_reboot_evidence_failures_never_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, problem: str
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    harness.problem = problem
    result = harness.run()
    assert result["verdict"] == "FAIL", result
    assert harness.restores == 0
    assert "override-expected-gpu-count" not in harness.calls
    assert "restore-collector-env" not in harness.calls
    stored = json.loads((tmp_path / "inventory-reboot-progress-a1.json").read_text())
    assert stored["verdict"] == "FAIL"
    assert stored["errors"] or stored["cleanup_errors"]


@pytest.mark.parametrize("problem", ["unhealthy", "debounce-config"])
def test_inventory_admission_rejection_never_attempts_override_or_restore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, problem: str
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    harness.problem = problem
    with pytest.raises(RegionalFixtureError):
        harness.run()
    assert harness.restores == 0 and not harness.run_id
    assert not list(tmp_path.glob("inventory-operation-*.json")), (
        'test_inventory_admission_rejection_never_attempts_override_or_restore: expected no list(tmp_path.glob("inventory-operation-*.json"))'
    )
    assert not harness.cleanup.state_readers, (
        "test_inventory_admission_rejection_never_attempts_override_or_restore: expected no harness.cleanup.state_readers"
    )


def test_expired_admission_never_authorizes_a_previous_runs_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    with action_window(datetime.now(timezone.utc) + timedelta(seconds=10)):
        with pytest.raises(RegionalFixtureError, match="maintenance"):
            harness.run()
    assert not harness.run_id and harness.restores == 0
    assert not harness.cleanup.state_readers, (
        "test_expired_admission_never_authorizes_a_previous_runs_cleanup: expected no harness.cleanup.state_readers"
    )


def test_distinct_campaigns_have_distinct_ids_and_retry_is_not_a_new_owner(
    tmp_path: Path,
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    first = runner.create_operation(tmp_path, 1)
    second = runner.create_operation(other, 1)
    assert first[0] != second[0] and first[1] != second[1]
    receipt = (tmp_path / "inventory-operation-a1.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.create_operation(tmp_path, 1)
    assert (tmp_path / "inventory-operation-a1.json").read_bytes() == receipt


@pytest.mark.parametrize("attempt", [0, -1, True])
def test_invalid_attempt_does_not_create_custody(tmp_path: Path, attempt: Any) -> None:
    with pytest.raises(RegionalFixtureError, match="attempt"):
        runner.create_operation(tmp_path, attempt)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", [RuntimeError("partial"), KeyboardInterrupt()])
def test_sampling_failure_has_no_production_undo_or_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: BaseException
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    harness.failure = failure
    if isinstance(failure, Exception):
        result = harness.run()
        assert "RuntimeError" in result["errors"][0]
    else:
        with pytest.raises(type(failure)):
            harness.run()
    result = json.loads((tmp_path / "inventory-reboot-progress-a1.json").read_text())
    assert result["verdict"] == "FAIL"
    assert result["cleanup_errors"] == []
    assert result["publication_started"] is False
    assert harness.restores == 0
    assert "publish-gpu-inventory" not in harness.calls
    assert len(harness.cleanup.state_readers) == 0


def test_publication_poll_failure_preserves_hold_when_workflow_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    harness.problem = "recovery-read"
    monkeypatch.setattr(
        runner,
        "wait_mismatch_finding",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RegionalFixtureError("evidence poll timed out")
        ),
    )
    result = harness.run()
    assert any("poll timed out" in item for item in result["errors"]), (
        'test_publication_poll_failure_preserves_hold_when_workflow_cannot_be_read: expected any("poll timed out" in item for item in result["erro...'
    )
    assert result["cleanup_errors"] == []
    assert result["publication_started"] is True
    released = harness.cleanup.finish(profile_version="v1", reason="test")
    assert "recovery lookup unresolved" in released["errors"][0]
    assert "incident-cleanup" not in harness.calls


@pytest.mark.parametrize(
    "error_type", [HostProbeMissingResponseError, HostProbeTransportError]
)
def test_lost_publication_response_is_read_back_without_republication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: type[Exception]
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    harness.publication_failure = error_type("response lost")
    result = harness.run()
    assert result["verdict"] == "PASS", result
    assert result["publication"] == {
        "response_unknown": True,
        "error_type": error_type.__name__,
        "replayed": False,
    }
    assert harness.calls.count("publish-gpu-inventory") == 1
    assert harness.restores == 0


def test_sample_and_publication_intent_exist_before_the_live_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    original = harness.collector.execute

    def execute(verb: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        if verb == "publish-gpu-inventory":
            sample = json.loads((tmp_path / "isolated-inventory-a1.json").read_text())
            intent = json.loads(
                (tmp_path / "inventory-publication-a1.json").read_text()
            )
            assert intent["sample_sha256"] == sample["sha256"]
            assert intent["batch_ids"] == sorted(
                batch["batch_id"] for batch in sample["batches"]
            )
            assert sample["publication_performed"] is False
        return cast(dict[str, Any], original(verb, *args, **kwargs))

    harness.collector.execute = execute
    assert harness.run()["verdict"] == "PASS"


def recovery(harness: InventoryHarness) -> runner.InventoryRecovery:
    return runner.InventoryRecovery(
        cast(RegionalLiveFixture, harness.regional),
        harness.settings,
        harness.scope,
        "run-a",
        "d" * 32,
        ORIGIN,
        ORIGIN,
        9,
        BASELINE,
        harness.case_dir,
        1,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "other"),
        ("mutation_started", False),
        ("timer_armed", False),
        ("boot_restore_armed", False),
        ("baseline_sha256", APPLIED),
        ("applied_sha256", "not-a-digest"),
        ("boot_id", "other"),
        ("intent_sha256", "f" * 63),
    ],
)
def test_arming_requires_complete_matching_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: Any
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    receipt = {
        "run_id": "run-a",
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "boot_id": "boot-a",
        "mutation_started": True,
        "timer_armed": True,
        "boot_restore_armed": True,
        "baseline_sha256": BASELINE,
        "applied_sha256": APPLIED,
        "intent_sha256": INTENT,
    }
    subject.accept_arming(receipt)
    with pytest.raises(RegionalFixtureError, match="ARMED"):
        subject.accept_arming({**receipt, field: value})


@pytest.mark.parametrize("problem", ["device", "batch", "record", "count", "empty"])
def test_finding_custody_is_not_derived_from_arbitrary_node_activity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, problem: str
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    rows = harness.records()
    if problem == "device":
        rows[-1]["samples"][0]["device"] = "gpu-a"
    elif problem == "batch":
        rows[-1]["batch_id"] = "host-other-batch-2"
    elif problem == "record":
        rows[-1]["record_id"] = "another-record"
    elif problem == "count":
        rows[-1]["samples"][0]["labels"]["expected_count"] = "8"
    else:
        rows.clear()
    with pytest.raises(RegionalFixtureError, match="finding"):
        subject.bind_finding(rows)


def test_cleanup_refresh_rejects_workflow_disappearance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = InventoryHarness(monkeypatch, tmp_path)
    subject = recovery(harness)
    subject.bind_finding(harness.records())
    subject.read()
    changed = deepcopy(harness.state())
    changed["workflows"][0]["request_id"] = "other"
    changed["node_workflow_ids"] = ["other"]
    harness.regional.cpu_python = lambda *args: changed
    with pytest.raises(RegionalFixtureError, match="lost a tracked"):
        subject.read()


def test_scope_comparison_does_not_treat_a_fresh_sts_session_as_a_new_executor() -> (
    None
):
    first = {
        "node_uid": "node-a",
        "boot_id": "boot-a",
        "observed_at": ORIGIN.isoformat(),
        "executor_role_arn": "bound-role",
        "executor_template_sha256": BASELINE,
        "executor_pods": [{"uid": "pod-a", "caller_arn": "first-verified-session"}],
    }
    second = deepcopy(first)
    second["observed_at"] = (ORIGIN + timedelta(seconds=1)).isoformat()
    second["executor_pods"] = [{"uid": "pod-a", "caller_arn": "new-verified-session"}]
    assert runner.scope_identity(first) == runner.scope_identity(second)
    assert first["executor_pods"] == [
        {"uid": "pod-a", "caller_arn": "first-verified-session"}
    ]
    second["executor_pods"] = [{"uid": "other-pod", "caller_arn": "new-session"}]
    assert runner.scope_identity(first) != runner.scope_identity(second)
    assert runner.scope_identity(first, after_reboot=True) == runner.scope_identity(
        second, after_reboot=True
    )
    second["executor_template_sha256"] = APPLIED
    assert runner.scope_identity(first, after_reboot=True) != runner.scope_identity(
        second, after_reboot=True
    )
