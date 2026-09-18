"""HA-005: shared continuity evaluation, outbox vacuity, rollout predicate, sample polling."""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import pytest

from scripts.e2e.regional import run_ha005_rollout_continuity as ha005
from tests.regional.test_acceptance_alignment_ha_telemetry import telemetry_proof


def _probe(
    attempts: int, accepted: int, failures: int, buffered: int | None = None, **extra
) -> dict:
    base, _ = telemetry_proof(spooled=False)
    final = base["attempted_events"][0]
    events = [
        {**final, "batch_id": f"earlier-{index}"} for index in range(attempts - 1)
    ] + [final]
    admissions = [
        {
            **base["admissions"][0],
            "batch_id": event["batch_id"],
            "request_id": f"r{index + 1}",
        }
        for index, event in enumerate(events[:accepted])
    ]
    if accepted < attempts:
        admissions.append(
            {
                **base["admissions"][0],
                "batch_id": final["batch_id"],
                "request_id": None,
                "replay": True,
                "spooled": True,
                "coalesced": True,
            }
        )
    return {
        **base,
        "attempted_events": events,
        "attempted_batch_ids": [event["batch_id"] for event in events],
        "admissions": admissions,
        "wire_responses": [
            {"path": "/v1/collector-events/host-telemetry", "status": 202}
            for _ in admissions
        ]
        + [
            {
                "path": "/v1/collector-events/host-telemetry",
                "status": 503,
                "retry_after": "2",
            }
            for _ in range(failures)
        ],
        "counters": {
            "event_attempts": attempts,
            "event_accepted": accepted,
            "event_failures": failures,
            "event_buffered": failures if buffered is None else buffered,
            "claim_success": 1,
        },
        "outbox": {"records": 0, "replayable": 0},
        "error_types": {},
        **extra,
    }


def _receipts(ids: list[str]) -> dict:
    _, base = telemetry_proof()
    return {
        "telemetry": base["telemetry"],
        "requests": [
            {"request_id": value, "status": "COMPLETED", "response_status": 200}
            for value in ids
        ],
        "missing": [],
    }


def test_continuity_errors_pass_a_clean_probe_and_catch_each_defect() -> None:
    clean = ha005.continuity_errors(
        _probe(3, 1, 2), _receipts(["r1"]), accepted_ids=["r1"]
    )
    assert clean == []
    assert ha005.continuity_errors(
        _probe(3, 1, 2), {"requests": [], "missing": []}, accepted_ids=["r1"]
    ), "an accepted request without a persisted receipt must invalidate continuity"
    assert ha005.continuity_errors(
        _probe(10, 8, 2), _receipts(["r1"]), accepted_ids=["r1"]
    ), "eight accepted events cannot be proved by one accepted request ID"
    no_claim = _probe(3, 1, 2)
    no_claim["counters"]["claim_success"] = 0
    assert "probe observed no successful claim" in ha005.continuity_errors(
        no_claim, _receipts(["r1"]), accepted_ids=["r1"]
    )

    assert (
        "event attempts minus accepted does not equal failures"
        in ha005.continuity_errors(
            _probe(10, 8, 1), _receipts(["r1"]), accepted_ids=["r1"]
        )
    )
    assert "not every event failure was durably buffered" in ha005.continuity_errors(
        _probe(10, 8, 2, buffered=1), _receipts(["r1"]), accepted_ids=["r1"]
    )
    assert "probe outbox is not empty after recovery" in ha005.continuity_errors(
        {**_probe(10, 8, 2), "outbox": {"records": 1, "replayable": 1}},
        _receipts(["r1"]),
        accepted_ids=["r1"],
    )
    assert "probe observed a nonretryable HTTP error" in ha005.continuity_errors(
        {**_probe(10, 8, 2), "error_types": {"http-500": 1}},
        _receipts(["r1"]),
        accepted_ids=["r1"],
    )
    assert (
        "queue receipt IDs do not cover every observed queue admission"
        in ha005.continuity_errors(_probe(10, 8, 2), _receipts([]), accepted_ids=[])
    )
    incomplete = _receipts(["r1"])
    incomplete["requests"][0]["status"] = "LEASED"
    assert (
        "an accepted processor request did not complete with 200"
        in ha005.continuity_errors(_probe(10, 8, 2), incomplete, accepted_ids=["r1"])
    )


def test_outbox_exercised_is_false_when_no_event_failed() -> None:
    assert ha005.outbox_exercised(_probe(10, 10, 0)) is False
    assert ha005.outbox_exercised(_probe(10, 8, 2)) is True


def test_rollout_complete_requires_updated_available_and_old_uids_gone() -> None:
    def snapshot(
        ready: int, updated: int, available: int, uids: list[str], observed: int = 5
    ) -> dict:
        return {
            "replicas": 3,
            "ready": ready,
            "updated": updated,
            "available": available,
            "generation": 5,
            "observed_generation": observed,
            "pods": [(f"p-{uid}", {"uid": uid, "ready": True}) for uid in uids],
        }

    old = {"old-1", "old-2", "old-3"}
    assert ha005.rollout_complete(snapshot(3, 3, 3, ["n1", "n2", "n3"]), old) is True
    assert (
        ha005.rollout_complete(snapshot(3, 2, 3, ["n1", "n2", "old-3"]), old) is False
    ), "ready == replicas mid-rollout is not completion"
    assert (
        ha005.rollout_complete(snapshot(3, 3, 3, ["n1", "n2", "old-3"]), old) is False
    )
    assert ha005.rollout_complete(snapshot(3, 3, 2, ["n1", "n2", "n3"]), old) is False
    assert (
        ha005.rollout_complete(snapshot(3, 3, 3, ["n1", "n2", "n3"], observed=4), old)
        is False
    )


def test_wait_probe_samples_polls_counters_instead_of_sleeping_thirty_seconds() -> None:
    reads = iter(
        [
            {"counters": {"event_attempts": 0, "claim_attempts": 0}},
            {"counters": {"event_attempts": 2, "claim_attempts": 5}},
            {"counters": {"event_attempts": 5, "claim_attempts": 10}},
        ]
    )
    slept: list[float] = []
    clock = {"now": 0.0}

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["now"] += seconds

    probe = ha005.wait_probe_samples(
        read=lambda: next(reads),
        sleep=sleep,
        clock=lambda: clock["now"],
        timeout_seconds=60,
    )

    assert probe["counters"]["event_attempts"] == 5
    assert len(slept) == 2
    assert sum(slept) < 30, "the baseline waits for samples, not for a fixed 30s"


def test_wait_probe_samples_times_out_with_the_counters_in_the_message() -> None:
    clock = {"now": 0.0}

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    with pytest.raises(ha005.CaseError, match="claim samples"):
        ha005.wait_probe_samples(
            read=lambda: {"counters": {"event_attempts": 1, "claim_attempts": 1}},
            sleep=sleep,
            clock=lambda: clock["now"],
            timeout_seconds=4,
        )


def test_rollout_targets_skip_disabled_roles(monkeypatch) -> None:
    replicas = {
        "gpu-fault-api-ha": 3,
        "gpu-fault-control-worker": 6,
        "gpu-fault-telemetry-spool-worker": 0,
    }
    monkeypatch.setattr(
        ha005,
        "deployment_snapshot",
        lambda name=ha005.INGRESS_DEPLOYMENT: {"replicas": replicas[name]},
    )

    assert ha005.rollout_targets(False) == ["gpu-fault-api-ha"]
    assert ha005.rollout_targets(True) == [
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
    ]


def test_parser_accepts_all_deployments_flag() -> None:
    import argparse

    from scripts.e2e.regional.live_driver_guard import add_live_arguments

    parser = argparse.ArgumentParser()
    add_live_arguments(parser, confirmation=ha005.CONFIRMATION)
    parser.add_argument("--all-deployments", action="store_true")
    assert (
        parser.parse_args(["--run-dir", "/tmp/x", "--all-deployments"]).all_deployments
        is True
    )


def test_formal_case_refuses_an_ingress_subset_before_any_read(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("GPU_FAULT_ACCEPTANCE_EXECUTION_SCOPE", "formal")
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_SELECTION_REFERENCE", raising=False)
    monkeypatch.setattr(ha005, "install_site_profile", lambda: None)
    calls = []
    monkeypatch.setattr(ha005, "database_residuals", lambda: calls.append("read"))
    monkeypatch.setattr(sys, "argv", [ha005.__file__, "--run-dir", str(tmp_path)])
    with pytest.raises(ha005.CaseError, match="requires --all-deployments"):
        ha005.main()
    with pytest.raises(ha005.CaseError, match="requires --all-deployments"):
        ha005.run_case(tmp_path, 1, datetime(2099, 1, 1, tzinfo=timezone.utc))
    assert calls == []


def test_probe_manifest_uses_the_executor_component_python() -> None:
    manifest = ha005.pod_manifest(
        "unit-image",
        {"executor_artifact_sha256": "unit", "executor_compatibility_digest": "unit"},
        "unit-run",
    )
    assert manifest["spec"]["containers"][0]["command"][0] == ha005.component_python(
        "gpu"
    )
