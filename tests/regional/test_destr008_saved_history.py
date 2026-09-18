"""Saved history must retain its original plan and acknowledged control-map identity."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional import destr008_cancellation as lifecycle
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._destr008_cancellation_controller import CpuApi, build_api


def saved_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, closed: bool
) -> tuple[CpuApi, lifecycle.CancellationWatchdog, dict[str, Any]]:
    api, plan, runtime = build_api(tmp_path, monkeypatch)
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    if closed:
        control.request_close()
        watchdog.wait_quiescence()
        watchdog.cleanup()
    return api, watchdog, api.journal()


@pytest.mark.parametrize("closed", [False, True], ids=["armed", "closed"])
@pytest.mark.parametrize("field", ["runtime_profile_version", "workload_ids"])
def test_changed_saved_plan_cannot_reuse_its_original_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, closed: bool, field: str
) -> None:
    api, watchdog, saved = saved_history(tmp_path, monkeypatch, closed=closed)
    saved["plan"][field] = (
        ["training/changed"] if field == "workload_ids" else "changed-profile"
    )
    changed = wire.Plan.model_validate(saved["plan"])
    write_json_atomic(watchdog.path, saved)
    before = list(api.calls)
    operations = [
        lambda: lifecycle.has_saved_plan(api.directory, changed.run_id),
        lambda: lifecycle.load_saved_plan(api.directory, changed.run_id),
        lambda: api.watchdog(changed, watchdog.runtime),
        watchdog.cleanup,
    ]
    for operation in operations:
        with pytest.raises(RegionalFixtureError, match="private journal is invalid"):
            operation()
    assert api.calls == before, (
        "contradictory saved history must fail before any CPU API work"
    )


@pytest.mark.parametrize(
    "change",
    [
        "control-plan",
        "receipt-plan",
        "receipt-uid",
        "owner-uid",
        "owner-missing",
        "control-missing",
        "last-receipt-missing",
        "quiescence-missing",
        "quiescence-changed",
        "revocation-changed",
        "receipt-before-plan",
    ],
)
def test_closed_history_is_bound_to_plan_control_uid_and_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    api, watchdog, saved = saved_history(tmp_path, monkeypatch, closed=True)
    owner = "configmap/" + watchdog.name
    if change == "control-plan":
        saved["last_control"]["plan_sha256"] = "a" * 64
    elif change in {"receipt-plan", "receipt-uid"}:
        key = "plan_sha256" if change == "receipt-plan" else "configmap_uid"
        value = "a" * 64 if change == "receipt-plan" else "replaced-control"
        saved["last_receipt"][key] = value
        saved["quiescence"][key] = value
    elif change == "owner-uid":
        saved["support"][owner]["uid"] = "replaced-control"
    elif change == "owner-missing":
        del saved["support"][owner]
    elif change == "control-missing":
        saved["last_control"] = None
    elif change == "last-receipt-missing":
        saved["last_receipt"] = None
    elif change == "quiescence-missing":
        saved["quiescence"] = None
    elif change == "quiescence-changed":
        saved["quiescence"]["sequence"] += 1
    elif change == "revocation-changed":
        saved["last_control"]["revocation"]["reason"] = "FAILURE"
    else:
        for key in ("last_receipt", "quiescence"):
            saved[key]["observed_at"] = saved["plan"]["created_at"] - 1
            saved[key]["quiet_since"] = saved["plan"]["created_at"] - 6
    write_json_atomic(watchdog.path, saved)
    before = list(api.calls)
    with pytest.raises(RegionalFixtureError, match="private journal is invalid"):
        lifecycle.load_saved_plan(api.directory, watchdog.plan.run_id)
    with pytest.raises(RegionalFixtureError, match="private journal is invalid"):
        watchdog.cleanup()
    assert api.calls == before, "a closed marker cannot bypass saved-history binding"


def test_nonterminal_receipt_cannot_be_installed_as_saved_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api, watchdog, saved = saved_history(tmp_path, monkeypatch, closed=False)
    saved["quiescence"] = copy.deepcopy(saved["last_receipt"])
    write_json_atomic(watchdog.path, saved)
    before = list(api.calls)
    with pytest.raises(RegionalFixtureError, match="private journal is invalid"):
        watchdog.cleanup()
    assert api.calls == before, (
        "an ARMED receipt is never terminal retirement authority"
    )


def test_armed_journal_cannot_erase_all_history_to_rebind_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api, watchdog, saved = saved_history(tmp_path, monkeypatch, closed=False)
    saved.update(last_control=None, last_receipt=None, quiescence=None)
    saved["plan"]["runtime_profile_version"] = "changed-profile"
    write_json_atomic(watchdog.path, saved)
    before = list(api.calls)
    with pytest.raises(RegionalFixtureError, match="private journal is invalid"):
        watchdog.cleanup()
    assert api.calls == before, (
        "erased authority history cannot become an unstarted run"
    )


def test_valid_closed_history_survives_expiry_without_rearming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api, watchdog, saved = saved_history(tmp_path, monkeypatch, closed=True)
    api.clock.sleep(86400)
    plan, runtime = lifecycle.load_saved_plan(api.directory, watchdog.plan.run_id)
    fresh = api.watchdog(plan, runtime)
    before = [call for call in api.calls if call[0] in {"create", "patch", "delete"}]
    fresh.cleanup()
    assert fresh.record.quiescence is not None
    assert fresh.record.quiescence.model_dump(mode="json") == saved["quiescence"], (
        "expiry does not rewrite a correctly bound historical retirement receipt"
    )
    assert [
        call for call in api.calls if call[0] in {"create", "patch", "delete"}
    ] == before


def test_stopped_submitting_receipt_can_precede_a_bound_late_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api, watchdog, saved = saved_history(tmp_path, monkeypatch, closed=False)
    plan = watchdog.plan
    control = wire.Control.model_validate(saved["last_control"])
    now = int(api.clock.now())
    submitted = wire.claim_submission(plan, control, claim_id="late-claim", now=now)
    revoked = wire.revoke(submitted, now=now, reason="FAILURE")
    failed = wire.receipt(
        plan,
        revoked,
        wire.Receipt.model_validate(saved["last_receipt"]),
        uid=saved["last_receipt"]["configmap_uid"],
        now=now,
        state="FAILED",
        error_code="SOURCE_UNRESOLVED",
        monitoring=False,
    )
    acknowledged = wire.acknowledge_submission(
        plan,
        revoked,
        acknowledgement=wire.Acknowledgement(
            incident_id="incident-a",
            workflow_request_id="workflow-a",
            claim_id="late-claim",
            event_id=plan.event_id,
            completed_at=now + 1,
        ),
        now=now + 1,
    )
    saved.update(
        last_control=acknowledged.model_dump(mode="json"),
        last_receipt=failed.model_dump(mode="json"),
    )
    write_json_atomic(watchdog.path, saved)
    before = list(api.calls)
    loaded, runtime = lifecycle.load_saved_plan(api.directory, plan.run_id)
    fresh = api.watchdog(loaded, runtime)
    assert fresh.record.last_control == acknowledged
    assert fresh.record.last_receipt == failed, (
        "valid late ACK must preserve the original failure"
    )
    assert api.calls == before, (
        "historical consistency checks do not query or mutate Kubernetes"
    )
