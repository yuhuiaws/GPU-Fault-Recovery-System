from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from scripts.e2e.regional import collect018_verdicts as rejected
from scripts.e2e.regional import collect019_verdicts as hang
from scripts.e2e.regional import collect020_verdicts as identity
from scripts.e2e.regional import run_collect016_training_recovery as training
from scripts.e2e.regional import run_collect021_late_xid_after_pod_death as late
from scripts.e2e.regional import run_collector_destructive as destructive


def unparsed_activity(operations: list[str]) -> dict[str, Any]:
    return {
        "incidents": [{"incident_id": "inc-unparsed_xid_line"}],
        "workflows": [
            {
                "incident_id": "inc-unparsed_xid_line",
                "status": "SUCCEEDED",
                "official_steps": [{"operation": item} for item in operations],
            }
        ],
        "notifications": [
            {"incident_id": "inc-unparsed_xid_line", "category": "OPERATOR_REVIEW"}
        ],
        "evidence": [{"payload": {"message": "marker=review-case"}}],
    }


@pytest.mark.parametrize(
    "operation",
    [
        item.value
        for item in WorkflowOperation
        if item is not WorkflowOperation.FREEZE_EVIDENCE
    ]
    + ["UNKNOWN_OPERATION", "FREEZE_EVIDENCE"],
)
def test_unparsed_xid_refuses_every_extra_operation(operation: str) -> None:
    errors = rejected.unparsed_finding_errors(
        unparsed_activity(["FREEZE_EVIDENCE", operation]),
        marker="review-case",
        metrics_before=[
            'gpu_fault_ingest_unresolved_fault_signals_total{kind="unparsed_xid_line"} 0'
        ],
        metrics_after=[
            'gpu_fault_ingest_unresolved_fault_signals_total{kind="unparsed_xid_line"} 1'
        ],
    )
    assert any("exactly FREEZE_EVIDENCE" in item for item in errors), (
        f"extra operation {operation!r} was not named by the freeze-only guard: {errors!r}"
    )


@pytest.mark.parametrize("status", ["FAILED", "BLOCKED", "RUNNING", None])
def test_unparsed_freeze_must_succeed(status: str | None) -> None:
    activity = unparsed_activity(["FREEZE_EVIDENCE"])
    activity["workflows"][0]["status"] = status
    errors = rejected.unparsed_finding_errors(
        activity, marker="review-case", metrics_before=[], metrics_after=[]
    )
    assert "the unparsed-line freeze workflow did not succeed" in errors


@pytest.mark.parametrize(
    "values",
    [
        {"MainPID": None},
        {"MainPID": "0"},
        {"MainPID": "unknown"},
        {"NRestarts": None},
        {"NRestarts": "-1"},
    ],
)
def test_hang_requires_present_valid_process_identity(values: dict[str, Any]) -> None:
    snapshot = {"ActiveState": "active", "MainPID": "42", "NRestarts": "0", **values}
    assert hang.service_errors(snapshot, dict(snapshot)), (
        f"service identity accepted missing or invalid process fields: {values!r}"
    )


def identity_activity() -> dict[str, Any]:
    return {
        "incidents": [
            {
                "incident_id": "identity-incident",
                "state": "RECOVERED",
                "reasons": ["GPU inventory identity changed: removed GPU-target"],
            }
        ],
        "workflows": [
            {
                "incident_id": "identity-incident",
                "request_id": "identity-workflow",
                "status": "SUCCEEDED",
                "official_steps": [
                    {"operation": "RUN_DCGM_DIAGNOSTIC"},
                    {"operation": "VALIDATE_GPU"},
                ],
            }
        ],
        "notifications": [
            {"incident_id": "identity-incident", "category": "OPERATOR_REVIEW"},
            {"incident_id": "identity-incident", "category": "DCGM_DIAGNOSTIC"},
        ],
        "markers": [
            {
                "incident_id": "identity-incident",
                "marker_id": "identity-marker",
                "retired_at": "2030-01-01T00:00:00Z",
                "retired_reason": "validated",
            }
        ],
    }


@pytest.mark.parametrize("missing", ["RUN_DCGM_DIAGNOSTIC", "VALIDATE_GPU"])
def test_identity_requires_both_diagnostic_operations(missing: str) -> None:
    activity = identity_activity()
    activity["workflows"][0]["official_steps"] = [
        item
        for item in activity["workflows"][0]["official_steps"]
        if item["operation"] != missing
    ]
    errors = identity.identity_finding_errors(activity, dropped_uuid="GPU-target")
    assert any("lacks DCGM diagnostic or GPU validation" in item for item in errors), (
        f"missing {missing} did not fail the diagnostic contract: {errors!r}"
    )


def test_identity_does_not_borrow_another_incidents_dcgm_notice() -> None:
    activity = identity_activity()
    activity["notifications"][1]["incident_id"] = "unrelated-incident"
    assert identity.dcgm_notification_errors(activity), (
        "an unrelated incident's DCGM notice satisfied this incident's notification proof"
    )


def test_identity_requires_markers_for_each_recovered_incident() -> None:
    activity = identity_activity()
    activity["markers"] = []
    assert identity.marker_retirement_errors(activity) == [
        "recovered incident identity-incident has no retirement markers"
    ]


@pytest.mark.parametrize(
    "devices",
    [
        [],
        [{"gpu_uuid": "GPU-other"}],
        [{"gpu_uuid": "GPU-a"}, {"gpu_uuid": "GPU-a"}],
        [{"gpu_uuid": "GPU-a"}, {"gpu_uuid": ""}],
        [{"gpu_uuid": "GPU-a"}, {"gpu_uuid": "GPU-unrelated"}],
    ],
)
def test_inventory_requires_exactly_the_baseline_minus_one_uuid(
    devices: list[dict[str, str]],
) -> None:
    assert identity.inventory_evidence_errors(
        [
            {
                "record_id": "inventory",
                "payload": {"devices": devices, "expected_gpu_count": 3},
            }
        ],
        dropped_uuid="GPU-target",
        expected_count=3,
        expected_uuids={"GPU-a", "GPU-b", "GPU-target"},
    ), (
        f"inventory accepted devices other than the exact baseline minus GPU-target: {devices!r}"
    )


def test_exact_shortened_inventory_is_accepted() -> None:
    assert (
        identity.inventory_evidence_errors(
            [
                {
                    "record_id": "inventory",
                    "payload": {
                        "devices": [{"gpu_uuid": "GPU-a"}, {"gpu_uuid": "GPU-b"}],
                        "expected_gpu_count": 3,
                    },
                }
            ],
            dropped_uuid="GPU-target",
            expected_count=3,
            expected_uuids={"GPU-a", "GPU-b", "GPU-target"},
        )
        == []
    )


@pytest.mark.parametrize(
    "source", ["NO_ACTIVE_MANAGED_ATTEMPT", "AMBIGUOUS_ACTIVE_ATTEMPTS", "unknown"]
)
def test_training_requires_the_exact_sole_active_identity(source: str) -> None:
    errors = training.restart_budget_section_errors(
        {"incident": {"workload_identity_source": source}}, {}, job_id="job-a"
    )
    assert any("workload_identity_source" in item for item in errors), (
        f"source {source!r} bypassed sole-active-attempt identity: {errors!r}"
    )


def death_observation() -> dict[str, Any]:
    return {
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "workload_phase": "FAILED",
        "containers": [
            {
                "pod_uid": "pod-a",
                "node_id": "node-a",
                "terminated": True,
                "exit_code": 137,
            }
        ],
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"job_id": "another-job"},
        {"attempt_id": "another-attempt"},
        {"workload_phase": "UNKNOWN"},
        {"workload_phase": None},
        {"workload_phase": "STOPPED"},
        {"containers": []},
        {
            "containers": [
                {
                    "pod_uid": "other",
                    "node_id": "node-a",
                    "terminated": True,
                    "exit_code": 137,
                }
            ]
        },
        {
            "containers": [
                {
                    "pod_uid": "pod-a",
                    "node_id": "other",
                    "terminated": True,
                    "exit_code": 137,
                }
            ]
        },
        {
            "containers": [
                {
                    "pod_uid": "pod-a",
                    "node_id": "node-a",
                    "terminated": False,
                    "exit_code": 137,
                }
            ]
        },
        {
            "containers": [
                {
                    "pod_uid": "pod-a",
                    "node_id": "node-a",
                    "terminated": True,
                    "exit_code": 0,
                }
            ]
        },
    ],
)
def test_death_wait_ignores_wrong_or_incomplete_identity(
    overrides: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = {**death_observation(), **overrides}
    valid = death_observation()
    reads = iter([{"observations": [invalid]}, {"observations": [valid]}])
    calls: list[dict[str, Any]] = []

    def snapshot(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return deepcopy(next(reads))

    monkeypatch.setattr(late.time, "sleep", lambda _: None)
    result = late.wait_death_observation(
        SimpleNamespace(store_snapshot=snapshot),
        node="node-a",
        job_id="job-a",
        attempt_id="attempt-a",
        pod_uid="pod-a",
        poll_seconds=0,
    )
    assert result == valid
    assert len(calls) == 2


def test_unknown_death_state_is_not_idle() -> None:
    assert not late.node_reads_idle({}, "node-a"), (
        "missing workload observation was treated as proof that node-a is idle"
    )
    assert not late.node_reads_idle({"workload_phase": "UNKNOWN"}, "node-a"), (
        "UNKNOWN workload phase was treated as proof that node-a is idle"
    )


@pytest.mark.parametrize("present", [True, False])
def test_full_fabric_negative_checks_legacy_inventory_even_without_a_snapshot(
    present: bool,
) -> None:
    event = datetime.now(timezone.utc) - timedelta(minutes=8)
    inventory = {
        "present": present,
        "observed_at": (event + timedelta(minutes=7)).isoformat(),
        "legacy_observed_at": [(event - timedelta(seconds=10)).isoformat()],
    }
    assert destructive.fail_closed_timing_errors(inventory, event_time=event), (
        f"legacy inventory could authorize the SXID despite snapshot present={present}"
    )
    inventory["legacy_observed_at"] = [(event + timedelta(minutes=7)).isoformat()]
    assert destructive.fail_closed_timing_errors(inventory, event_time=event) == []


@pytest.mark.parametrize(
    "inventory",
    [
        {},
        {"present": False},
        {"present": "false", "legacy_observed_at": []},
        {"present": False, "legacy_observed_at": None},
        {"present": False, "legacy_observed_at": ["unknown"]},
        {
            "present": True,
            "observed_at": "2030-01-01T00:00:00",
            "legacy_observed_at": [],
        },
    ],
)
def test_full_fabric_negative_refuses_unknown_inventory_proof(
    inventory: dict[str, Any],
) -> None:
    assert destructive.fail_closed_timing_errors(
        inventory, event_time=datetime.now(timezone.utc)
    ), f"unknown inventory proof allowed the fail-closed injection: {inventory!r}"
