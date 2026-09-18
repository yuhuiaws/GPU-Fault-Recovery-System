from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gpu_fault import installation_inventory as inventory
from gpu_fault import models
from gpu_fault.async_store import AsyncStoreExecutor
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceSnapshot,
)
from tests._builders import workflow_request


def resource(**changes):
    return InstallationResource.model_validate(
        {
            "site_id": "unit-site",
            "resource_key": "unit-resource",
            "resource_type": "s3_bucket",
            "resource_id": "unit-bucket",
            "ownership": "CREATED",
            "delete_policy": "DELETE",
            **changes,
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"resource_key": "invalid key"},
        {"site_id": ""},
        {"resource_id": "invalid id"},
        {"dependencies": ["invalid dependency"]},
        {"attributes": {"execution_token": "example-only"}},
        {"delete_policy": "PRESERVE"},
    ],
)
def test_installation_resource_rejects_unsafe_identity_or_policy(changes) -> None:
    with pytest.raises(ValidationError):
        resource(**changes)


def test_installation_resource_normalizes_dependencies_and_preserves_external_assets() -> (
    None
):
    value = resource(
        ownership="REUSED",
        delete_policy="PRESERVE",
        dependencies=["second", "first", "first", " "],
    )
    assert value.dependencies == ["first", "second"], (
        "equivalent dependency sets have one identity"
    )
    before = value.immutable_identity()
    value.status = "PRESERVED"
    assert value.immutable_identity() == before, (
        "lifecycle status is not immutable ownership authority"
    )
    assert resource(resource_type="rds_snapshot", delete_policy="PRESERVE"), (
        "approved backup retention remains supported for created snapshots"
    )


@pytest.mark.parametrize("problem", ["duplicate", "foreign-site", "bad-digest"])
def test_resource_snapshot_requires_one_site_and_exact_content(problem) -> None:
    value = resource()
    payload = {"site_id": "unit-site", "resources": [value]}
    if problem == "duplicate":
        payload["resources"].append(value)
    elif problem == "foreign-site":
        payload["site_id"] = "other-site"
    else:
        payload["source_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        InstallationResourceSnapshot.model_validate(payload)


def test_resource_snapshot_requires_a_seal_after_loading() -> None:
    unsealed = InstallationResourceSnapshot(site_id="unit-site", resources=[resource()])
    with pytest.raises(ValueError, match="not sealed"):
        unsealed.require_source_binding()
    sealed = unsealed.model_copy(update={"source_sha256": unsealed.digest()})
    assert sealed.require_source_binding() is sealed, (
        "exact content identity authorizes use"
    )
    changed = sealed.model_copy(update={"resources": []})
    with pytest.raises(ValueError, match="digest mismatch"):
        changed.require_source_binding()


@pytest.mark.parametrize(
    "units",
    [
        "gpu-fault-a.service",
        b"gpu-fault-a.service",
        ["sshd.service"],
        [f"gpu-fault-unit-{index}.service" for index in range(129)],
    ],
)
def test_installed_unit_inventory_rejects_wrong_type_scope_and_size(units) -> None:
    with pytest.raises(ValueError):
        inventory.normalize_installed_units(units)


@pytest.mark.parametrize(
    "model", [inventory.InstalledUnitReport, inventory.InstalledUnitInventory]
)
def test_installed_unit_digest_must_describe_the_actual_units(model) -> None:
    with pytest.raises(ValidationError, match="digest mismatch"):
        model(digest="0" * 64, units=["gpu-fault-a.service"])
    with pytest.raises(ValueError, match="SHA-256"):
        inventory.validate_inventory_digest("invalid")


def test_inventory_delta_requires_full_list_when_digest_changes(tmp_path) -> None:
    units = ["gpu-fault-a.service", "gpu-fault-b.timer"]
    report = inventory.installed_unit_report(units, include_units=True)
    current, changed = inventory.resolve_agent_inventory(report, None)
    assert current is not None and not changed, (
        "the initial complete inventory is retained"
    )
    digest_only = inventory.installed_unit_report(units, include_units=False)
    resolved, changed = inventory.resolve_agent_inventory(
        digest_only, SimpleNamespace(installed_unit_inventory=current)
    )
    assert resolved is current and not changed, (
        "an unchanged digest reuses verified data"
    )
    with pytest.raises(ValueError, match="full installed"):
        inventory.resolve_installed_unit_inventory(digest_only, None)
    (tmp_path / "gpu-fault-a.service").write_text("unit", encoding="utf-8")
    (tmp_path / "gpu-fault-b.timer").write_text("timer", encoding="utf-8")
    assert (
        inventory.read_installed_systemd_units(tmp_path / "absent", tmp_path) == units
    ), "fallback discovery is bounded to the product's unit names"
    path = tmp_path / "inventory.txt"
    path.write_text("gpu-fault-a.service\n", encoding="utf-8")
    assert inventory.read_installed_systemd_units(path, tmp_path) == units[:1], (
        "an explicit inventory is authoritative"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"restart_count": 1, "reservation_ids": []},
        {"budget": 0, "restart_count": 1, "reservation_ids": ["one"]},
        {"budget": 2, "restart_count": 2, "reservation_ids": ["one", "one"]},
    ],
)
def test_restart_budget_cannot_be_reconstructed_from_inconsistent_reservations(
    changes,
) -> None:
    with pytest.raises(ValidationError):
        models.RestartBudgetState(
            **{"cluster_id": "unit", "job_id": "job", "budget": 2, **changes}
        )


def test_marker_window_must_have_positive_lifetime() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError, match="after observed_at"):
        models.NodeMarker(
            source="unit",
            incident_id="unit",
            observed_at=now,
            expires_at=now,
            scope=models.MarkerScope(node_ids=["node"]),
            severity=models.Severity.CRITICAL,
            mapping_version="unit",
        )


def test_nonobject_completion_decision_is_not_coerced() -> None:
    with pytest.raises(ValidationError):
        models.CompletionDecision.model_validate([])


@pytest.mark.parametrize(
    ("status", "blocked_kind", "expected"),
    [
        (models.WorkflowStatus.PENDING, None, True),
        (models.WorkflowStatus.RUNNING, None, True),
        (models.WorkflowStatus.SAFETY_PENDING, None, True),
        (models.WorkflowStatus.BLOCKED, models.BlockedKind.NEEDS_OPERATOR, True),
        (models.WorkflowStatus.BLOCKED, models.BlockedKind.INTERNAL_ERROR, True),
        (models.WorkflowStatus.BLOCKED, models.BlockedKind.SAFETY_SETTLED, False),
        (models.WorkflowStatus.BLOCKED, None, False),
        (models.WorkflowStatus.SUCCEEDED, None, False),
        (models.WorkflowStatus.FAILED, None, False),
        (models.WorkflowStatus.SUPERSEDED, None, False),
    ],
)
def test_workflow_occupancy_keeps_operator_holds_but_releases_terminal_work(
    status, blocked_kind, expected
) -> None:
    assert models.workflow_is_open(status, blocked_kind) is expected, (
        "occupancy is determined by executable state or an explicit operator hold"
    )


@pytest.mark.parametrize(
    ("status", "exit_codes", "expected"),
    [
        ("SUCCEEDED", [0, 0], False),
        ("SUCCEEDED", [0, 1], True),
        ("FAILED", [], True),
        ("TIMED_OUT", [], True),
        ("STOPPED", [0], False),
    ],
)
def test_terminal_result_preserves_failed_ranks_even_with_success_summary(
    status, exit_codes, expected
) -> None:
    event = models.TerminalEvent(
        cluster_id="unit",
        environment=models.Environment.EKS,
        job_id="job",
        attempt_id="attempt",
        terminal_status=status,
        ended_at=datetime(2026, 9, 12, tzinfo=timezone.utc),
        rank_exit_status=[
            models.RankExitStatus(rank=index, node_id="node", exit_code=code)
            for index, code in enumerate(exit_codes)
        ],
        runtime_profile_version="unit",
    )
    assert event.is_failure is expected, (
        "a nonzero rank cannot be hidden by a successful aggregate status"
    )
    assert event.event_key == "unit/attempt/TrainingAttemptTerminal", (
        "terminal deduplication identity is stable across delivery attempts"
    )
    by_uuids = event.model_copy(
        update={
            "allocation": [
                models.AllocationEntry(node_id="node", gpu_uuids=["gpu-a", "gpu-b"]),
                models.AllocationEntry(node_id="node", gpu_uuids=["gpu-b"]),
            ]
        }
    )
    assert by_uuids.gpu_count == 2, "legacy GPU identities are deduplicated"
    declared = by_uuids.model_copy(
        update={"allocation": [models.AllocationEntry(node_id="node", gpu_count=4)]}
    )
    assert declared.gpu_count == 4, "declared allocation counts remain authoritative"


@pytest.mark.parametrize("legacy_status", [False, True])
def test_legacy_completion_rows_decode_without_mutating_the_stored_payload(
    legacy_status,
) -> None:
    payload = {
        "cluster_id": "unit",
        "attempt_id": "attempt",
        "event_key": "unit-event",
        "status": "PENDING_TRIAGE" if legacy_status else "NO_ACTION",
        "reason": "legacy reason",
        "diagnostic_request_id": "retired",
    }
    decoded = models.CompletionDecision.model_validate(payload)
    assert decoded.status is models.DecisionStatus.NO_ACTION, (
        "retired triage cannot initiate a new recovery action"
    )
    assert decoded.reason.endswith("legacy reason"), (
        "legacy decisions retain the operator's explanatory context"
    )
    assert ("pre-cutover" in decoded.reason) is legacy_status, (
        "only a retired pending-triage state receives the migration explanation"
    )
    assert payload["diagnostic_request_id"] == "retired", (
        "decoding must not modify the caller's original payload"
    )


def test_valid_lifetimes_and_restart_reservations_remain_usable() -> None:
    now = datetime(2026, 9, 12, tzinfo=timezone.utc)
    marker = models.NodeMarker(
        source="unit",
        incident_id="unit",
        observed_at=now,
        expires_at=now + timedelta(seconds=1),
        scope=models.MarkerScope(node_ids=["node"]),
        severity=models.Severity.CRITICAL,
        mapping_version="unit",
    )
    assert marker.expires_at > marker.observed_at, "a positive marker window is valid"
    state = models.RestartBudgetState(
        cluster_id="unit",
        job_id="job",
        budget=2,
        restart_count=1,
        reservation_ids=["first"],
    )
    assert state.restart_count == len(state.reservation_ids) == 1, (
        "one durable reservation consumes exactly one restart"
    )
    workflow = workflow_request("unit-workflow", "unit-incident")
    assert not models.lifetime_exceeded(workflow, now), (
        "a never-claimed workflow has no implicit expired deadline"
    )
    claimed = workflow.model_copy(update={"lifetime_deadline_at": now})
    assert models.lifetime_exceeded(claimed, now), (
        "the exact deadline is already outside the execution window"
    )
    assert not models.lifetime_exceeded(claimed, now - timedelta(microseconds=1)), (
        "the lifetime remains open immediately before its boundary"
    )
    assert models.bounded_reasons(["only"], limit=2) == ["only"], (
        "a small history within its budget needs no truncation"
    )


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_small_reason_budgets_are_actual_upper_bounds(limit) -> None:
    values = ["first", "middle", "last"]
    result = models.bounded_reasons(values, limit=limit)
    assert len(result) <= limit, (
        "zero and small limits must not retain the entire history"
    )
    if limit == 1:
        assert result == ["last"], "one-slot history keeps the newest reason"
    elif limit == 2:
        assert result == ["first", "last"], "two slots retain the endpoints"


def test_negative_reason_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        models.bounded_reasons(["first"], limit=-1)


def test_repeated_history_markers_do_not_crash_or_duplicate_recent_events() -> None:
    now = datetime.now(timezone.utc)
    marker = models.WorkflowEvent(
        kind=models.WorkflowEventKind.HISTORY_TRUNCATED,
        at=now,
        details={"dropped": 1, "dropped_from": now.isoformat()},
    )
    old = workflow_request("unit-workflow", "unit-incident").model_copy(
        update={"events": [marker] * models.WORKFLOW_EVENTS_LIMIT}
    )
    event = models.WorkflowEvent(
        kind=models.WorkflowEventKind.CLAIM, at=now + timedelta(seconds=1)
    )
    result = models.append_workflow_event(old, event)
    assert (
        sum(item.kind is models.WorkflowEventKind.CLAIM for item in result.events) == 1
    ), "rebuilt history must not repeat its only real event"
    markers = [
        item
        for item in result.events
        if item.kind is models.WorkflowEventKind.HISTORY_TRUNCATED
    ]
    assert (
        len(markers) == 1
        and markers[0].details["dropped"] == models.WORKFLOW_EVENTS_LIMIT
    ), "all prior truncation counts are carried into one marker"


def test_operator_details_are_bounded_and_deterministic() -> None:
    result = models.bounded_event_details(
        {
            "long": "x" * 600,
            "set": {"b", "a"},
            "many": list(range(55)),
            "nested": {"deeper": {"value": {"too": "deep"}}},
        }
    )
    assert len(result["long"]) == models.OPERATOR_EVENT_TEXT_LIMIT, (
        "large text is bounded"
    )
    assert result["set"] == ["a", "b"], "set details have deterministic order"
    assert len(result["many"]) == models.OPERATOR_EVENT_ITEMS_LIMIT + 1, (
        "truncation retains one explicit count marker"
    )
    assert isinstance(result["nested"]["deeper"]["value"], str), (
        "unbounded nesting becomes text"
    )


@pytest.mark.parametrize(
    "options",
    [
        {"workers": 0, "max_in_flight": 1, "admission_timeout_seconds": 1},
        {"workers": 2, "max_in_flight": 1, "admission_timeout_seconds": 1},
        {"workers": 1, "max_in_flight": 1, "admission_timeout_seconds": 0},
    ],
)
def test_invalid_async_store_capacity_is_rejected_before_threads(options) -> None:
    with pytest.raises(ValueError):
        AsyncStoreExecutor(**options)


def test_async_store_rejects_unknown_rejection_labels() -> None:
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )
    try:
        with pytest.raises(ValueError, match="unknown store I/O rejection"):
            executor.record_rejection("unregistered")
        assert executor.rejected_total == 0, (
            "invalid labels cannot distort rejection metrics"
        )
    finally:
        executor.close()


def test_async_store_preserves_nonretryable_errors_and_releases_capacity() -> None:
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )

    def fail():
        raise ValueError("invalid domain request")

    async def execute():
        with pytest.raises(ValueError, match="invalid domain request"):
            await executor.run(fail)
        assert executor.in_flight == 0 and executor.rejected_total == 0, (
            "a domain failure is not a retryable backend outage or retained slot"
        )

    try:
        asyncio.run(execute())
    finally:
        executor.close()
