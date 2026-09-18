from __future__ import annotations

import json
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request
from urllib.response import addinfourl

import pytest

from scripts.e2e.regional import ha_telemetry_evidence as proof
from scripts.e2e.regional.probes import ha005_probe
from scripts.e2e.regional.run_ha003_aurora_failover_reset import (
    ALLOWED_RESULT_SUBMISSION_CODES,
    result_submission_codes,
)
from scripts.e2e.regional.run_ha005_rollout_continuity import continuity_errors


def telemetry_proof(*, spooled=True):
    batch = {
        "batch_id": "ha005-captest-00000",
        "observed_at": "2026-09-15T10:00:00+00:00",
    }
    request_id = None if spooled else "processor-unit"
    probe = {
        "run_id": "captest",
        "cluster_id": "cap-cluster-000",
        "attempted_events": [batch],
        "attempted_batch_ids": [batch["batch_id"]],
        "payload_contract": {
            "path": proof.HOST_PATH,
            "node_id": "ha005-node-captest",
            "edge_filter_reasons": ["health-summary"],
            "collection_errors": [],
            "sample_count": 9,
        },
        "admissions": [
            {
                "batch_id": batch["batch_id"],
                "replay": False,
                "accepted": True,
                "spooled": spooled,
                "coalesced": False,
                "request_id": request_id,
            }
        ],
        "wire_responses": [{"path": proof.HOST_PATH, "status": 202}],
        "counters": {
            "event_attempts": 1,
            "event_accepted": 1,
            "event_failures": 0,
            "event_buffered": 0,
            "claim_success": 1,
        },
        "error_types": {},
        "outbox": {"records": 0, "replayable": 0},
        "replay_stopped": True,
    }
    receipt = {
        "cluster_id": probe["cluster_id"],
        "node_id": "ha005-node-captest",
        "batch_id": batch["batch_id"],
        "collector": "HOST_TELEMETRY",
        "observed_at": batch["observed_at"],
        "last_success_at": batch["observed_at"],
        "errors": [],
        "sample_count": 9,
        "spool_depth": 0,
    }
    return probe, {
        "requests": []
        if spooled
        else [
            {"request_id": request_id, "status": "COMPLETED", "response_status": 200}
        ],
        "missing": [],
        "telemetry": receipt,
    }


@pytest.mark.parametrize("spooled", [True, False])
def test_queue_and_spool_have_different_success_receipts(spooled) -> None:
    probe, receipts = telemetry_proof(spooled=spooled)
    ids = [] if spooled else ["processor-unit"]
    assert continuity_errors(probe, receipts, accepted_ids=ids) == []


def test_routine_coalescing_proves_latest_state_not_one_row_per_sample() -> None:
    probe, receipts = telemetry_proof()
    first = probe["attempted_events"][0]
    newer = {
        **first,
        "batch_id": first["batch_id"] + "-new",
        "observed_at": "2026-09-15T10:00:01+00:00",
    }
    probe["attempted_events"].append(newer)
    probe["attempted_batch_ids"].append(newer["batch_id"])
    probe["admissions"].append(
        {**probe["admissions"][0], "batch_id": newer["batch_id"], "coalesced": True}
    )
    probe["wire_responses"].append({"path": proof.HOST_PATH, "status": 202})
    probe["counters"].update(event_attempts=2, event_accepted=2)
    receipts["telemetry"].update(
        batch_id=newer["batch_id"],
        observed_at=newer["observed_at"],
        last_success_at=newer["observed_at"],
    )
    assert continuity_errors(probe, receipts, accepted_ids=[]) == []
    receipts["telemetry"]["batch_id"] = first["batch_id"]
    assert continuity_errors(probe, receipts, accepted_ids=[]), (
        "stale final telemetry cannot establish replay completion"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "events",
        "admissions",
        "malformed",
        "duplicate-id",
        "contract",
        "count",
        "batch-binding",
        "accepted",
        "replay-type",
        "spool-shape",
        "queue-id",
        "mode",
        "foreground",
        "id-census",
        "last-admission",
        "replay-running",
    ],
)
def test_admission_gaps_cannot_be_replaced_with_an_empty_outbox(defect) -> None:
    probe, _ = telemetry_proof()
    ids = []
    if defect in {"events", "admissions"}:
        probe["attempted_events" if defect == "events" else "admissions"] = None
    elif defect == "malformed":
        probe["admissions"] = [None]
    elif defect == "duplicate-id":
        probe["attempted_events"] *= 2
    elif defect == "contract":
        probe["payload_contract"]["edge_filter_reasons"] = ["fault"]
    elif defect == "count":
        probe["counters"]["event_attempts"] += 1
    elif defect == "id-census":
        ids = ["missing"]
    elif defect == "last-admission":
        probe["admissions"] = []
    elif defect == "replay-running":
        probe["replay_stopped"] = False
    else:
        item = probe["admissions"][0]
        item.update(
            {
                "batch-binding": {"batch_id": "foreign"},
                "accepted": {"accepted": False},
                "replay-type": {"replay": 0},
                "spool-shape": {"coalesced": None},
                "queue-id": {"spooled": False},
                "mode": {"spooled": None},
                "foreground": {"replay": True},
            }[defect]
        )
    assert proof.admission_errors(probe, ids), defect


@pytest.mark.parametrize(
    "key",
    [
        "cluster_id",
        "node_id",
        "batch_id",
        "collector",
        "sample_count",
        "errors",
        "spool_depth",
        "observed_at",
        "last_success_at",
    ],
)
def test_a_replay_receipt_must_match_the_latest_batch_and_be_healthy(key) -> None:
    probe, receipts = telemetry_proof()
    receipts["telemetry"][key] = None
    assert proof.telemetry_replay_errors(probe, receipts["telemetry"]), key


@pytest.mark.parametrize("status", [401, 403, 404, 409, 422, 500, 505, 0, True])
def test_nonretryable_http_cannot_pass_even_when_other_requests_succeed(status) -> None:
    probe, receipts = telemetry_proof()
    probe["wire_responses"].append(
        {"path": "/v1/regional/executors/claim", "status": status}
    )
    assert continuity_errors(probe, receipts, accepted_ids=[]), (
        "a nonretryable response must fail continuity"
    )


@pytest.mark.parametrize("retry_after", [None, "", "bad", "-1", "0", "NaN", "1000"])
def test_retryable_capacity_status_needs_retry_after(retry_after) -> None:
    assert proof.wire_errors(
        {
            "wire_responses": [
                {"status": 503, "retry_after": retry_after, "path": proof.HOST_PATH}
            ]
        }
    ), "503 requires a finite positive bounded Retry-After"


def test_retryable_http_and_recognized_transport_have_explicit_evidence() -> None:
    assert (
        proof.wire_errors(
            {
                "wire_responses": [
                    {"status": 503, "retry_after": "2", "path": proof.HOST_PATH},
                    {"status": 429, "retry_after": "2", "path": proof.HOST_PATH},
                    {"status": None, "transport_error": "TimeoutError"},
                    {"status": 502},
                    {"status": 504},
                    {"status": 408},
                    {"status": 425},
                ]
            }
        )
        == []
    )
    for bad in ({}, {"wire_responses": [None]}, {"wire_responses": [{"status": None}]}):
        assert proof.wire_errors(bad), "missing or malformed wire evidence must fail"


def test_probe_observes_real_sink_admissions_and_transport_headers(
    monkeypatch, tmp_path
) -> None:
    responses = [
        (503, {"Retry-After": "2"}, {"detail": "capacity"}),
        (202, {}, {"accepted": True, "spooled": True, "coalesced": False}),
        (202, {}, {"accepted": True, "spooled": True, "coalesced": True}),
    ]

    def open_request(request, **kwargs):
        status, headers, body = responses.pop(0)
        stream = BytesIO(json.dumps(body).encode())
        if status >= 400:
            raise HTTPError(request.full_url, status, "isolated", headers, stream)
        return addinfourl(stream, headers, request.full_url, status)

    monkeypatch.setattr(ha005_probe.sinks, "urlopen", open_request)
    evidence = ha005_probe.ProbeEvidence()
    sink = ha005_probe.ObservedSink(
        "http://unit.invalid",
        max_attempts=1,
        evidence=evidence,
        outbox_path=str(tmp_path / "outbox"),
    )
    payload = {"batch_id": "ha005-unit", "cluster_id": "cluster", "node_id": "node"}
    with evidence.observe_http():
        with pytest.raises(ha005_probe.CollectorError):
            sink.post(ha005_probe.HOST_PATH, payload)
        sink.post(ha005_probe.HOST_PATH, {**payload, "batch_id": "ha005-unit-new"})
        assert sink.wait_for_outbox_replay(2), (
            "the bounded real sink replay must finish"
        )
    result = evidence.snapshot()
    assert result["wire_responses"][0]["status"] == 503
    assert result["wire_responses"][0]["retry_after"] == "2"
    assert result["admissions"][-1] == {
        "batch_id": payload["batch_id"],
        "accepted": True,
        "replay": True,
        "spooled": True,
        "coalesced": True,
        "request_id": None,
    }
    assert ha005_probe.sinks.urlopen is open_request


def test_probe_restores_http_observer_after_transport_failure(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise TimeoutError("isolated timeout")

    monkeypatch.setattr(ha005_probe.regional_client, "urlopen", unavailable)
    evidence = ha005_probe.ProbeEvidence()
    with evidence.observe_http():
        with pytest.raises(TimeoutError):
            ha005_probe.regional_client.urlopen(Request("http://unit.invalid/claim"))
    assert evidence.snapshot()["wire_responses"][0]["transport_error"] == "TimeoutError"
    assert ha005_probe.regional_client.urlopen is unavailable


def test_ha003_reads_status_on_the_same_retry_log_line() -> None:
    logs = (
        "could not report result; retrying: rejected request (503)\n"
        "could not report result\nTraceback\nrejected request (409)\n"
        "unrelated rejected request (422)\n"
    )
    assert result_submission_codes(logs) == [503, 409]
    assert 503 in ALLOWED_RESULT_SUBMISSION_CODES
    assert 500 not in ALLOWED_RESULT_SUBMISSION_CODES
