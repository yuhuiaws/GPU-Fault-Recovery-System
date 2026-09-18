from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.collector_reset_evidence import (
    full_fabric_reset_errors,
    physical_reset_errors,
)
from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional._collector_reset_support import reset_documents


@pytest.mark.parametrize("operation", ["RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES"])
def test_physical_audit_binds_one_successful_attempt_to_exact_targets(
    operation: str,
) -> None:
    assert (
        physical_reset_errors(
            *reset_documents(operation), node="node-a", operation=operation
        )
        == []
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("workflow_request_id", "other-workflow"),
        ("incident_id", "other-incident"),
        ("fencing_token", 1),
        ("fencing_token", "2"),
        ("gpu_uuids", ["GPU-b"]),
        ("gpu_uuids", ["GPU-a", "GPU-a"]),
        ("attempt", 2),
        ("attempt", True),
        ("state", "FAILED"),
        ("signature_digest_present", False),
        ("parameters_digest", "unrelated"),
        ("started_at", "not-a-timestamp"),
        ("completed_at", "2020-01-01T00:00:00+00:00"),
    ],
)
def test_same_count_or_operation_name_cannot_hide_wrong_reset_identity(
    key: str, value: Any
) -> None:
    before, after, state = reset_documents()
    after["ledger"][0][key] = value
    assert physical_reset_errors(before, after, state, node="node-a"), (
        f"reset audit accepted an invalid identity field: {key}={value!r}"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("reset_gpu_uuids", ["GPU-b"]),
        ("reset_attempts", 0),
        ("reset_attempts", 2),
        ("reset_attempts", True),
        ("verified_no_gpu_clients", False),
    ],
)
def test_result_must_prove_a_single_completed_reset(key: str, value: Any) -> None:
    before, after, state = reset_documents()
    after["ledger"][0]["result"]["details"][key] = value
    assert physical_reset_errors(before, after, state, node="node-a"), (
        f"reset result accepted an invalid completion proof: {key}={value!r}"
    )


@pytest.mark.parametrize("operation", ["RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES"])
@pytest.mark.parametrize("busy_refusals", [0, 1, 2])
def test_known_busy_refusals_do_not_count_as_successful_physical_resets(
    operation: str, busy_refusals: int
) -> None:
    before, after, state = reset_documents(operation)
    after["ledger"][0]["result"]["details"].update(
        reset_attempts=busy_refusals + 1,
        reset_successes=1,
        reset_busy_refusals=busy_refusals,
    )
    errors = physical_reset_errors(
        before, after, state, node="node-a", operation=operation
    )
    assert errors == [], (
        "one signed successful reset plus bounded proven busy refusals is valid"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"reset_successes": 2},
        {"reset_successes": True},
        {"reset_busy_refusals": True},
        {"reset_busy_refusals": -1},
        {"reset_busy_refusals": 2},
        {"reset_attempts": 4, "reset_busy_refusals": 3},
        {"reset_successes": None},
        {"reset_busy_refusals": None},
        {"outcome_unknown": True},
        {"manual_confirmation_required": True},
        {"reset_outcome_unknown": ["GPU-a"]},
        {"reset_failed": ["GPU-a"]},
        {"reset_not_attempted": ["GPU-a"]},
    ],
)
def test_busy_retry_evidence_never_accepts_duplicate_or_unknown_execution(change):
    before, after, state = reset_documents()
    after["ledger"][0]["result"]["details"].update(
        {"reset_attempts": 2, "reset_successes": 1, "reset_busy_refusals": 1, **change}
    )
    assert physical_reset_errors(before, after, state, node="node-a"), (
        "retry counts cannot override missing, contradictory or unresolved evidence"
    )


def test_equal_inventory_counts_do_not_prove_uuid_restoration() -> None:
    before, after, state = reset_documents()
    after["gpu_inventory"][0]["uuid"] = "GPU-other"
    assert physical_reset_errors(before, after, state, node="node-a"), (
        "equal inventory counts hid replacement of a baseline GPU UUID"
    )


def test_previous_reset_is_not_a_new_physical_action() -> None:
    before, after, state = reset_documents()
    before["ledger"] = deepcopy(after["ledger"])
    assert physical_reset_errors(before, after, state, node="node-a"), (
        "a reset already present in the baseline was counted as this case's action"
    )


@pytest.mark.parametrize(
    "flag", ["inventory_verified_before", "inventory_verified_after", "reset_scope"]
)
def test_full_fabric_reset_requires_both_inventory_proofs(flag: str) -> None:
    before, after, state = reset_documents("RESET_ALL_GPUS_NVSWITCHES")
    del after["ledger"][0]["result"]["details"][flag]
    assert physical_reset_errors(
        before, after, state, node="node-a", operation="RESET_ALL_GPUS_NVSWITCHES"
    ), f"full fabric reset accepted a result missing {flag}"


def test_full_fabric_verdict_selects_the_reset_and_ignores_prior_ledger_rows() -> None:
    before, after, state = reset_documents("RESET_ALL_GPUS_NVSWITCHES")
    prior = {**deepcopy(after["ledger"][0]), "command_id": "prior-reset"}
    before["ledger"].append(prior)
    after["ledger"].append(prior)
    activity = {
        "workflows": [
            {"official_steps": [{"operation": "FREEZE_EVIDENCE"}]},
            state["workflow"],
        ],
        "incidents": [state["incident"]],
    }
    assert (
        full_fabric_reset_errors(
            before,
            after,
            activity,
            audit_before=before,
            audit_after=after,
            node="node-a",
        )
        == []
    )


@pytest.mark.parametrize(
    "change,fragment",
    [
        ("workflow", "no workflow planned"),
        ("status", "workflow is not SUCCEEDED"),
        ("execution", "did not succeed"),
        ("ledger", "ledger count is not one"),
        ("inventory", "GPU inventory changed"),
        ("boot", "reset became a reboot"),
        ("incident", "binding is incomplete"),
    ],
)
def test_full_fabric_verdict_retains_each_positive_phase_guard(
    change: str, fragment: str
) -> None:
    before, after, state = reset_documents("RESET_ALL_GPUS_NVSWITCHES")
    activity = {"workflows": [state["workflow"]], "incidents": [state["incident"]]}
    if change == "workflow":
        activity["workflows"] = []
    elif change == "status":
        state["workflow"]["status"] = "FAILED"
    elif change == "execution":
        state["workflow"]["step_executions"] = []
    elif change == "ledger":
        before["ledger"] = deepcopy(after["ledger"])
    elif change == "inventory":
        after["gpu_inventory"].pop()
    elif change == "boot":
        after["boot_id"] = "other-boot"
    else:
        activity["incidents"] = []
    errors = full_fabric_reset_errors(
        before, after, activity, audit_before=before, audit_after=after, node="node-a"
    )
    assert any(fragment in error for error in errors), errors


def test_reset_audit_reads_only_projected_metadata_without_opening_a_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    before, after, _ = reset_documents()
    row = after["ledger"][0]
    ledger = tmp_path / "node-actions.db"
    ledger.touch()
    boot = tmp_path / "boot-id"
    boot.write_text(before["boot_id"])
    payload = {**row["result"], "not_evidence": "do-not-publish"}
    values = [
        row[key]
        for key in (
            "command_id",
            "attempt",
            "state",
            "operation",
            "started_at",
            "completed_at",
            "incident_id",
            "workflow_request_id",
            "fencing_token",
        )
    ] + [
        json.dumps(row["gpu_uuids"]),
        row["parameters_digest"],
        "unit-signature-digest",
        json.dumps(payload),
    ]
    connection = SimpleNamespace(
        execute=lambda _: SimpleNamespace(fetchall=lambda: [tuple(values)]),
        close=lambda: None,
    )
    opened = []

    def connect(path: str, *, uri: bool) -> Any:
        opened.append((path, uri))
        return connection

    monkeypatch.setattr(probe, "NODE_LEDGER", ledger)
    monkeypatch.setattr(probe, "BOOT_ID_FILE", boot)
    monkeypatch.setattr(probe.sqlite3, "connect", connect)
    monkeypatch.setattr(probe, "gpu_inventory", lambda: before["gpu_inventory"])
    probe.reset_audit(SimpleNamespace())
    result = json.loads(capsys.readouterr().out)
    assert opened == [(f"file:{ledger}?mode=ro", True)]
    assert result["ledger"][0]["signature_digest_present"] is True
    assert set(result["ledger"][0]["result"]) == {
        "command_id",
        "operation",
        "status",
        "attempt",
        "details",
    }
