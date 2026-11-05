"""GF-GENERAL-LIVE hardware chain under an exhausted budget: verdict edges."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_attempt_hardware as hw
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from scripts.e2e.regional.general_live_kernel_xid import catalog_entry
from tests.regional._general_live_fakes import FakeRegional, make_context
from tests.regional.test_general_live_attempt_hardware import (
    G1,
    INC,
    NOTIFICATION,
    SPEC,
    WF,
    chain_workflow,
    decision,
)


def test_a_refused_check_that_moved_the_ledger_is_named() -> None:
    document = {
        "reservations": [{"reserved": True, "reservation_id": "r-1"}],
        "ledger": {
            "after": {"restart_count": 1, "budget": 1, "reservation_ids": ["r-1"]}
        },
        "exhausted_check": {"reserved": False, "restart_count": 2},
    }

    assert hw.seed_errors(document, budget=1, reservation_ids=["r-1"]) == [
        "the refused check changed the ledger"
    ]


def test_two_fresh_decisions_for_one_line_are_named() -> None:
    entry = catalog_entry(hw.HARDWARE_XID)
    errors, graded = hw.decision_errors(
        [decision(), decision(event_id="copy")], entry=entry, gpu_uuid=G1
    )

    assert errors == [
        f"2 non-duplicate decisions for XID {hw.HARDWARE_XID}; the reviewed "
        "record's copies did not dedup to one event"
    ]
    assert graded["event_id"] == decision()["event_id"]


def test_a_decision_without_an_incident_is_named() -> None:
    errors, _ = hw.decision_errors(
        [decision(incident_id=None)], entry=catalog_entry(hw.HARDWARE_XID), gpu_uuid=G1
    )

    assert errors == [f"XID {hw.HARDWARE_XID} decision names no incident"]


def test_a_hardware_step_that_did_not_succeed_is_named_with_its_error() -> None:
    workflow = chain_workflow()
    reset = next(
        item for item in workflow["step_executions"] if item["operation"] == "RESET_GPU"
    )
    reset.update(status="FAILED", error="nvidia-smi -r timed out")

    assert "RESET_GPU ended FAILED: nvidia-smi -r timed out" in (
        hw.chain_receipt_errors(workflow)
    )


def test_a_restart_step_without_any_record_is_named() -> None:
    workflow = chain_workflow(skip=("RESTART_WORKLOAD",))

    assert hw.withheld_errors(workflow, [], budget=1) == [
        "RESTART_WORKLOAD has no record: it was neither withheld nor run"
    ]


def test_two_restart_records_are_named() -> None:
    workflow = chain_workflow()
    record = next(
        item
        for item in workflow["step_executions"]
        if item["operation"] == "RESTART_WORKLOAD"
    )
    workflow["step_executions"].append(copy.deepcopy(record))

    errors = hw.withheld_errors(workflow, [], budget=1)

    assert errors[0] == (
        "2 RESTART_WORKLOAD records (expected the one withheld record)"
    )


def test_a_restart_record_that_is_neither_withheld_nor_run_is_named() -> None:
    workflow = chain_workflow(restart_status="RUNNING")

    assert hw.withheld_errors(workflow, [], budget=1) == [
        "RESTART_WORKLOAD is RUNNING; a withheld restart is recorded FAILED before "
        "any adapter runs"
    ]


def test_remote_commands_of_other_steps_are_not_restart_commands() -> None:
    commands = [
        {"command_id": "cmd-reset", "step": {"operation": "RESET_GPU"}},
        {"command_id": "cmd-batch", "batched_steps": [{"step": {}}]},
    ]

    assert hw.withheld_errors(chain_workflow(), commands, budget=1) == []


def test_a_budget_mail_whose_delivery_failed_is_named() -> None:
    mails = [
        {
            "notification_id": NOTIFICATION,
            "deduplication_key": f"restart-budget-exhausted/{INC}",
            "status": "FAILED",
        }
    ]

    assert hw.notification_errors(
        mails, notification_id=NOTIFICATION, incident_id=INC
    ) == [f"budget notification {NOTIFICATION} delivery FAILED"]


def test_the_workflow_is_found_by_its_incident_when_the_id_is_unknown() -> None:
    document = {
        "workflows": [
            {"request_id": "wf-other", "incident_id": "inc-other"},
            {"request_id": WF, "incident_id": INC, "created_at": "t"},
        ]
    }

    assert hw.workflow_of(document, workflow_id="wf-gone", incident_id=INC) == {
        "request_id": WF,
        "incident_id": INC,
        "created_at": "t",
    }
    assert hw.workflow_of(document, workflow_id=None, incident_id="inc-none") == {}


def test_a_failed_validated_restore_is_returned_as_a_cleanup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = make_context(SPEC, tmp_path=tmp_path, regional=FakeRegional(), probes={})
    calls: list[tuple[Any, ...]] = []

    def restore(context: Any, incidents: Any, *, reason: str) -> None:
        calls.append((incidents, reason))
        raise RuntimeError("restore refused: node not quarantined")

    monkeypatch.setattr(hw, "restore_incidents", restore)

    assert hw.validated_restore(ctx, INC) == [
        "validated restore failed: RuntimeError: restore refused: node not quarantined"
    ]
    assert calls == [
        ([{"incident_id": INC}], f"{SPEC.key} hardware chain acceptance restore")
    ]


def test_a_managed_job_needs_the_site_file(tmp_path: Path) -> None:
    ctx = make_context(SPEC, tmp_path=tmp_path, regional=FakeRegional(), probes={})
    assert ctx.settings.site_file is None

    with pytest.raises(VariantNotRun, match="a managed training job needs --site-file"):
        hw.ManagedTrainingJob(ctx, name="gl-hw-job")
