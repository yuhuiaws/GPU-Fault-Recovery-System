"""Regression cases for owned CMD/NET/PREEMPT safety checks, without live I/O."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import cmd017_verdicts as cmd017
from scripts.e2e.regional import net007_verdicts as net007
from scripts.e2e.regional import net008_verdicts as net008
from scripts.e2e.regional import preempt036_verdicts as preempt036
from scripts.e2e.regional import preempt037_verdicts as preempt037
from scripts.e2e.regional import preempt038_verdicts as preempt038
from scripts.e2e.regional import run_net007_transient_api_outage as outage
from scripts.e2e.regional import run_net008_outbox_dead_letter as outbox
from scripts.e2e.regional import run_preempt012_acceptance as preempt012
from scripts.e2e.regional import seeded_command_fixture as seeded


def test_failed_resource_read_is_not_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_read(*args: str, **kwargs: Any) -> str:
        if kwargs.get("check", True):
            raise RuntimeError("API unavailable")
        return ""

    monkeypatch.setattr(seeded, "dataplane", failed_read)
    with pytest.raises(RuntimeError, match="API unavailable"):
        seeded.kubernetes_residuals(SimpleNamespace(pod="p", configmap="c"))


@pytest.mark.parametrize("state", [{}, {"status": None}, {"status": "UNKNOWN"}])
def test_barrier_final_snapshot_requires_a_known_open_state(state: dict) -> None:
    assert cmd017.final_command_errors(state), (
        f"unknown final command state was accepted: {state}"
    )


@pytest.mark.parametrize(
    "expression",
    [
        "true || request.userInfo.username == '{username}'",
        "request.userInfo.username != '{username}'",
        "request.userInfo.username.startsWith('{username}')",
    ],
)
def test_webhook_username_mention_does_not_prove_exact_scope(expression: str) -> None:
    username = "system:serviceaccount:ns:executor"
    manifest = net007.webhook_manifest(
        name="test-webhook",
        node="node-a",
        namespace="ns",
        username=username,
        run_id="r",
    )
    manifest["webhooks"][0]["matchConditions"][0]["expression"] = expression.format(
        username=username
    )
    assert net007.webhook_errors(manifest, node="node-a", username=username), (
        f"widened webhook expression was accepted: {expression}"
    )


def test_failed_webhook_delete_keeps_the_deadman() -> None:
    calls: list[tuple[str, ...]] = []

    def kubectl(plane: str, *args: str, **kwargs: Any) -> str:
        calls.append(args)
        if args[:2] == ("delete", "--raw"):
            raise RuntimeError("webhook delete failed")
        if args[:2] == ("get", "validatingwebhookconfiguration"):
            return json.dumps(
                {
                    "uid": "webhook-uid",
                    "resourceVersion": "1",
                    "labels": {seeded.RUN_LABEL: "review"},
                }
            )
        return ""

    fixture = outage.OutageFixture(
        SimpleNamespace(kubectl=kubectl, settings=SimpleNamespace(namespace="ns")),
        names=outage.resource_names("review"),
        node="node-a",
        username="system:serviceaccount:ns:executor",
        image="image",
        run_id="review",
        deadman_seconds=840,
    )
    fixture.webhook_created = True
    with pytest.raises(RuntimeError, match="webhook delete failed"):
        fixture.cleanup()
    assert not any(
        args[0] == "delete" and "validatingwebhookconfigurations" not in args[2]
        for args in calls
    ), "the independent rollback must remain while webhook deletion is unproven"
    assert fixture.webhook_created is True


@pytest.mark.parametrize(
    "stats",
    [
        {},
        {"replayable": None},
        {"replayable": False},
        {"replayable": -1},
        {"replayable": 1},
    ],
)
def test_outbox_delivery_needs_a_measured_zero(stats: dict[str, Any]) -> None:
    assert net008.delivered_errors(
        [{"payload": "marker"}], [], stats, markers=["marker"]
    ), f"unproven replayable count was accepted: {stats}"


def test_failed_outbox_replay_cannot_mask_backlog_or_requeue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel = {
        "stats": {"replayable": 1, "dead": 1},
        "records": [
            {
                "path": net008.RETIRED_CHANNEL_PATH,
                "marker_present": True,
                "replayable": False,
                "error": "HTTP 404",
            }
        ],
    }
    run = SimpleNamespace(
        run_id="review-008",
        marker_base="marker",
        case_dir=tmp_path,
        transient_markers=["marker-t1"],
        stages={},
        evidence={},
        blocked=False,
        seeded=False,
        window_open=False,
        unit_stopped=False,
        kernel_activity=lambda: {"evidence": [{"payload": "marker-t1"}]},
        fixture=SimpleNamespace(
            snapshot=lambda marker: {"outboxes": {net008.COLLECTOR: kernel}},
            wait_until=lambda callback, **k: callback(),
            execute=lambda *a, **k: pytest.fail(
                "failed replay must not list or requeue"
            ),
        ),
    )
    monkeypatch.setattr(outbox, "_prepare_run", lambda *a: run)
    monkeypatch.setattr(outbox, "_phase_transient_403", lambda r: None)
    monkeypatch.setattr(outbox, "_phase_seed_dead_letter", lambda r: None)
    monkeypatch.setattr(
        outbox,
        "_phase_blackout_stream",
        lambda r: pytest.fail("failed replay must not start another outage"),
    )
    result = outbox.execute(
        SimpleNamespace(),
        run.fixture,
        tmp_path,
        1,
        datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert result["verdict"] == "FAIL"
    assert result["stages"]["delivered"] == ["1 replayable record(s) remain"]
    assert result["dead_letter"]["stats"]["replayable"] == 1


def test_invalid_outbox_window_stops_before_writing_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    run = SimpleNamespace(
        run_id="review-008",
        case_dir=tmp_path,
        stages={},
        evidence={},
        blocked=False,
        seeded=False,
        window_open=False,
        unit_stopped=False,
        transient_markers=["marker-t1"],
        write_kmsg=lambda marker: pytest.fail("invalid window cannot inject a marker"),
        fixture=SimpleNamespace(execute=lambda verb, *a, **k: calls.append(verb) or {}),
    )
    monkeypatch.setattr(outbox, "_prepare_run", lambda *a: run)
    monkeypatch.setattr(outbox, "open_window_or_rollback", lambda *a, **k: {})
    result = outbox.execute(
        SimpleNamespace(),
        run.fixture,
        tmp_path,
        1,
        datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert result["verdict"] == "FAIL"
    assert result["stages"]["window"]
    assert calls == ["close-window"]


def run_cleanup_path(
    run: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    def abort_phase(current: Any) -> None:
        raise RuntimeError("stop before the next phase")

    monkeypatch.setattr(outbox, "_prepare_run", lambda *a: run)
    monkeypatch.setattr(outbox, "_phase_transient_403", abort_phase)
    return outbox.execute(
        SimpleNamespace(),
        run.fixture,
        tmp_path,
        1,
        datetime.now(timezone.utc) + timedelta(hours=1),
    )


def test_outbox_cleanup_attempts_independent_restores_after_unblock_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def execute(verb: str, *args: str, **kwargs: Any) -> dict:
        calls.append(verb)
        if verb == "unblock":
            raise RuntimeError("network cleanup failed")
        return {"removed": 1}

    run = SimpleNamespace(
        blocked=True,
        window_open=True,
        seeded=True,
        fixture=SimpleNamespace(execute=execute),
        tag="test-tag",
        run_id="r",
        marker_base="m",
        stages={},
        evidence={},
        ip_arguments=lambda: ["--ip", "192.0.2.1"],
        stop_unit=lambda: execute("stop-unit"),
        start_unit=lambda: execute("start-unit"),
    )
    result = run_cleanup_path(run, tmp_path, monkeypatch)
    assert calls == [
        "unblock",
        "close-window",
        "stop-unit",
        "purge-outbox-record",
        "start-unit",
    ]
    assert result["verdict"] == "FAIL"
    assert any("unblock" in error for error in result["stages"]["cleanup"]), (
        f"network restore failure was lost: {result['stages']['cleanup']}"
    )


def test_outbox_cleanup_restarts_the_unit_even_when_purge_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def execute(verb: str, *args: str, **kwargs: Any) -> dict:
        calls.append(verb)
        raise RuntimeError("purge failed")

    run = SimpleNamespace(
        blocked=False,
        window_open=False,
        seeded=True,
        fixture=SimpleNamespace(execute=execute),
        run_id="r",
        marker_base="m",
        stages={},
        evidence={},
        stop_unit=lambda: calls.append("stop-unit"),
        start_unit=lambda: calls.append("start-unit"),
    )
    result = run_cleanup_path(run, tmp_path, monkeypatch)
    assert calls == ["stop-unit", "purge-outbox-record", "start-unit"]
    assert result["verdict"] == "FAIL"
    assert any("purge" in error for error in result["stages"]["cleanup"]), (
        f"outbox purge failure was lost: {result['stages']['cleanup']}"
    )


def test_blackout_needs_observed_destinations_and_verified_timer_cleanup() -> None:
    assert net008.blackout_errors(
        {"connectivity": {}, "timer": {"ActiveState": "active"}},
        {"rules": [], "connectivity": {}},
    ), "empty connectivity maps cannot prove a blackout or restoration"
    assert net008.blackout_errors(
        {"connectivity": {"192.0.2.1": False}, "timer": {"ActiveState": "active"}},
        {
            "rules": [],
            "connectivity": {"192.0.2.1": True},
            "timer": {"ActiveState": "active"},
        },
    ), "an active rollback timer must prevent successful blackout cleanup"


@pytest.mark.parametrize("missing", ["invocation_id", "pos"])
def test_kernel_stream_identity_cannot_be_proved_from_missing_fields(
    missing: str,
) -> None:
    state: dict[str, Any] = {
        "pid": 10,
        "invocation_id": "invocation",
        "kmsg_streams": [{"fd": 5, "pos": 100}],
    }
    if missing == "pos":
        state["kmsg_streams"][0].pop("pos")
    else:
        state.pop(missing)
    assert net008.stream_identity_errors(state, state), (
        f"kernel stream identity was accepted without {missing}: {state}"
    )


@pytest.mark.parametrize("timer", [None, "", "unknown", "failed", "misspelled"])
def test_quiesce_cleanup_does_not_accept_an_unknown_timer(timer: str | None) -> None:
    checked = preempt012.cleanup_checks(
        host_cleanup={"timer_active_state": timer},
        node_state={"ownership_annotations": {}},
    )
    assert checked["cycle_timer_inactive"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"execution_epoch": 77},
        {"remediation_budget_claims": ["foreign-claim"]},
        {"official_steps": []},
        {"completed_step_indexes": [0]},
    ],
)
def test_sweeper_snapshot_detects_changes_outside_its_old_projection(
    changes: dict[str, Any],
) -> None:
    store = InMemoryStore()
    seed = preempt036.seed_shape(preempt036.COMPILE_BLOCKED_SHAPE, store)
    request_id = seed.untouched_ids[0]
    before = preempt036.workflow_snapshot(store, seed.workflow_ids)
    workflow = store.get_workflow(request_id)
    changed = workflow.model_copy(update=changes)
    reader = SimpleNamespace(
        get_workflow=lambda key: changed
        if key == request_id
        else store.get_workflow(key)
    )
    after = preempt036.workflow_snapshot(reader, seed.workflow_ids)
    assert before[request_id] != after[request_id]
    assert any(
        "untouched record changed" in error
        for error in preempt036.record_errors(before, after, seed)
    ), f"untouched-workflow mutation escaped the canonical snapshot: {changes}"


def test_missing_metric_is_not_proof_that_the_alert_expression_fired() -> None:
    with pytest.raises(ValueError, match="metric|series"):
        preempt037.stalled(
            [f"{preempt037.PERIODIC_METRIC} 1000"], now=1000, threshold_seconds=300
        )


@pytest.mark.parametrize("stamp", ["NaN", "+Inf", "2000"])
def test_invalid_or_future_dispatch_metric_cannot_prove_recovery(stamp: str) -> None:
    with pytest.raises(ValueError, match="metric|timestamp|finite"):
        preempt037.stalled(
            [f"{preempt037.DISPATCH_METRIC} {stamp}"], now=1000, threshold_seconds=300
        )


def test_empty_evidence_keys_cannot_match_arbitrary_cleanup_logs() -> None:
    assert preempt038.log_errors(
        "cleanup raw_evidence deleted 1 rows; keys[:20]=unrelated-row",
        unrelated_key="",
        pinned_key="",
    ), "empty evidence keys must not match an unrelated cleanup log"
