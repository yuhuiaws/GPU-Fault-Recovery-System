"""Unit contracts for the NOTIFY-001..005 runner and its drill probe."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.models import (
    AdvisoryNotification,
    NotificationResult,
    NotificationStatus,
)
from scripts.e2e.regional import run_notification_acceptance as notification
from scripts.e2e.regional.probes import notification_drill as drill_probe

NODES = ("node-a", "node-b", "node-c")


class RecordingNotifier:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, item: AdvisoryNotification) -> NotificationResult:
        self.sent.append(item.notification_id)
        return NotificationResult(
            notification_id=item.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id="ses-1",
        )


# --- NOTIFY-005: one template object per role ------------------------------


def test_low_utilization_manifest_has_no_yaml_alias_and_distinct_templates() -> None:
    document = notification.low_utilization_manifest(
        name="notify005-3n", nodes=NODES, image="img@sha256:abc"
    )

    assert notification.manifest_alias_errors(document) == []
    rendered = yaml.safe_dump(document, sort_keys=False)
    assert "&id" not in rendered and "*id" not in rendered
    replicas = document["spec"]["pytorchReplicaSpecs"]
    assert replicas["Master"]["template"] is not replicas["Worker"]["template"]

    # What gpu-training-submit does per role must land on one role only.
    master_metadata = replicas["Master"]["template"]["metadata"]
    master_metadata.setdefault("annotations", {})["gpu-fault.io/rank-offset"] = "0"
    assert "annotations" not in replicas["Worker"]["template"]["metadata"]


def test_manifest_alias_errors_flags_a_shared_template() -> None:
    template = {"metadata": {"labels": {"app": "x"}}, "spec": {"containers": []}}
    shared = {
        "kind": "PyTorchJob",
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {"template": template},
                "Worker": {"template": template},
            }
        },
    }

    errors = notification.manifest_alias_errors(shared)

    assert "manifest serializes with a YAML alias" in errors
    assert "replica specs share one template object" in errors


def _rendered_workload(master_offset: str, worker_offset: str) -> dict[str, Any]:
    def template(role: str, offset: str) -> dict[str, Any]:
        return {
            "metadata": {
                "labels": {"gpu-fault.io/role": role},
                "annotations": {
                    "gpu-fault.io/rank-offset": offset,
                    "gpu-fault.io/expected-critical-ranks": "3",
                },
            }
        }

    return {
        "kind": "PyTorchJob",
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "template": template(
                        "master" if master_offset == "0" else "worker", master_offset
                    )
                },
                "Worker": {"template": template("worker", worker_offset)},
            }
        },
    }


def test_low_utilization_metadata_errors_catch_master_injected_twice() -> None:
    assert (
        notification.low_utilization_metadata_errors(
            _rendered_workload("0", "1"), expected_pods=3
        )
        == []
    )
    # The alias defect: Master ended up with Worker's role and rank offset.
    errors = notification.low_utilization_metadata_errors(
        _rendered_workload("1", "1"), expected_pods=3
    )
    assert "Master role label is 'worker'" in errors
    assert "Master rank offset is '1'" in errors
    assert "replica specs share one rank offset" in errors
    # A plain Job (single-node phase) has no replica specs to judge.
    assert (
        notification.low_utilization_metadata_errors({"kind": "Job"}, expected_pods=1)
        == []
    )


# --- external evidence -----------------------------------------------------


def _evidence(tmp_path: Path, name: str, value: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_external_evidence_requires_method_reference_and_a_covering_window(
    tmp_path: Path,
) -> None:
    drill_time = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    window = {
        "window_start": "2026-09-07T11:55:00+00:00",
        "window_end": "2026-09-07T12:10:00+00:00",
    }
    receipt = _evidence(
        tmp_path,
        "receipt.json",
        {"received": True, "reference": "ses-cw-1", "method": "inbox", **window},
    )
    dedup = _evidence(
        tmp_path,
        "dedup.json",
        {
            "send_count_delta": 0,
            "duplicate_inbox_count": 0,
            "reference": "SES/Send 11:55-12:10",
            "method": "cloudwatch-send-metric",
            **window,
        },
    )

    assert notification.validate_external_evidence(
        receipt, "receipt", record_times=[drill_time]
    )["valid"]
    assert notification.validate_external_evidence(
        dedup, "dedup", record_times=[drill_time]
    )["valid"]

    # The window must cover the record it vouches for.
    late = notification.validate_external_evidence(
        receipt, "receipt", record_times=[drill_time + timedelta(hours=1)]
    )
    assert not late["valid"]
    assert any("does not cover" in item for item in late["errors"]), (
        f"a record outside the receipt window must be named: {late['errors']}"
    )

    # A dedup file without method/reference/window is a guess, not evidence.
    bare = _evidence(
        tmp_path, "bare.json", {"send_count_delta": 0, "duplicate_inbox_count": 0}
    )
    verdict = notification.validate_external_evidence(bare, "dedup")
    assert not verdict["valid"]
    assert "method is required" in verdict["errors"]
    assert "reference is required" in verdict["errors"]
    assert any("window_start" in item for item in verdict["errors"]), (
        f"a dedup file without a window must fail on window_start: {verdict['errors']}"
    )

    missing = notification.validate_external_evidence(None, "receipt")
    assert not missing["valid"] and missing["errors"] == [
        "receipt evidence is required"
    ]


# --- NOTIFY-001: live records first, drill as fallback ---------------------


def _record_entry(
    *, notification_id: str, incident_id: str, key: str, cluster: str = "cluster-a"
) -> dict[str, Any]:
    return {
        "notification": {
            "notification_id": notification_id,
            "incident_id": incident_id,
            "category": "ACTION_COMPLETED",
            "cluster_name": cluster,
            "deduplication_key": key,
            "created_at": "2026-09-07T09:00:00+00:00",
        },
        "result": {"status": "SENT", "provider_message_id": "ses-x"},
    }


def _write_case_evidence(run_dir: Path, case_id: str, body: dict[str, Any]) -> None:
    path = run_dir / "cases" / case_id / f"{case_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"case_id": case_id, **body}), encoding="utf-8")


def test_action_completed_records_are_read_from_this_runs_evidence(
    tmp_path: Path,
) -> None:
    _write_case_evidence(
        tmp_path,
        "GF-REGIONAL-DESTR-009",
        {
            "verdict": "PASS",
            "state": {
                "notifications": [
                    _record_entry(
                        notification_id="n-restart",
                        incident_id="inc-1",
                        key="cluster-a/job/workload-restarted/op-1",
                    ),
                    {
                        "notification": {
                            "notification_id": "n-detected",
                            "category": "FAULT_DETECTED",
                            "cluster_name": "cluster-a",
                            "deduplication_key": "cluster-a/inc-1/fault",
                        },
                        "result": None,
                    },
                ]
            },
        },
    )
    _write_case_evidence(
        tmp_path,
        "GF-REGIONAL-DESTR-001",
        {
            "verdict": "PASS",
            "primary_state": {
                "notifications": [
                    _record_entry(
                        notification_id="n-reset-b",
                        incident_id="inc-2",
                        key="cluster-b/inc-2/gpu-reset/op-2",
                        cluster="cluster-b",
                    )
                ]
            },
        },
    )
    # Our own earlier NOTIFY evidence is never a source.
    _write_case_evidence(
        tmp_path,
        "GF-REGIONAL-NOTIFY-001",
        {
            "verdict": "PASS",
            "x": {
                "notifications": [
                    _record_entry(
                        notification_id="n-old",
                        incident_id="inc-9",
                        key="cluster-a/inc-9/gpu-reset/op-9",
                    )
                ]
            },
        },
    )

    records = notification.action_completed_records_from_evidence(
        tmp_path, cluster_id="cluster-a"
    )

    assert [item["notification_id"] for item in records["workload-restart"]] == [
        "n-restart"
    ]
    assert records["workload-restart"][0]["case_id"] == "GF-REGIONAL-DESTR-009"
    assert records["gpu-reset"] == []


def test_select_live_record_wants_a_sent_non_drill_record_for_the_cluster() -> None:
    records = [
        {
            "kind": "gpu-reset",
            "cluster_name": "cluster-a",
            "status": "SENT",
            "provider_message_id_present": True,
            "created_at": "2026-09-07T08:00:00+00:00",
            "drill_id": None,
        },
        {
            "kind": "gpu-reset",
            "cluster_name": "cluster-a",
            "status": "SENT",
            "provider_message_id_present": True,
            "created_at": "2026-09-07T09:00:00+00:00",
            "drill_id": "notify001-x",
        },
        {
            "kind": "gpu-reset",
            "cluster_name": "cluster-a",
            "status": "FAILED",
            "provider_message_id_present": False,
            "created_at": "2026-09-07T10:00:00+00:00",
            "drill_id": None,
        },
        {"notification_id": "gone", "missing": True},
    ]

    chosen = notification.select_live_record(
        records, kind="gpu-reset", cluster_id="cluster-a"
    )

    assert chosen is not None and chosen["created_at"].startswith("2026-09-07T08")
    assert (
        notification.select_live_record(
            records, kind="gpu-reset", cluster_id="cluster-b"
        )
        is None
    )


def _fake_drill(kind: str, drill_id: str, **_: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "drill_id": drill_id,
        "executed_at": "2026-09-07T12:00:00+00:00",
        "notification_id": f"n-{drill_id}",
        "statuses": ["SENT"] * 4,
        "provider_message_id_present": True,
        "provider_message_id_stable": True,
    }


def _install_notify001_fakes(
    monkeypatch: pytest.MonkeyPatch, *, live_records: list[dict[str, Any]]
) -> list[str]:
    drills: list[str] = []

    def fake_drill(
        site: Any, target: Any, *, kind: str, drill_id: str, pod: Any = None
    ) -> dict[str, Any]:
        drills.append(kind)
        return _fake_drill(kind, drill_id)

    monkeypatch.setattr(
        notification, "control_worker_pod", lambda site, target: "worker-0"
    )
    monkeypatch.setattr(notification, "drill", fake_drill)
    monkeypatch.setattr(
        notification,
        "live_action_completed_records",
        lambda site, target, *, run_dir, pod: {
            "candidates": {},
            "records": live_records,
        },
    )
    return drills


def _live(kind: str, count: int = 1) -> dict[str, Any]:
    return {
        "notification_id": f"n-{kind}",
        "missing": False,
        "kind": kind,
        "cluster_name": "cluster-a",
        "status": "SENT",
        "provider_message_id_present": True,
        "created_at": "2026-09-07T11:00:00+00:00",
        "drill_id": None,
        "same_incident_kind_count": count,
    }


def test_notify001_reuses_live_records_and_still_proves_dedup_with_one_drill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drills = _install_notify001_fakes(
        monkeypatch, live_records=[_live("gpu-reset"), _live("workload-restart")]
    )
    window = {
        "method": "cloudwatch",
        "reference": "SES/Send",
        "window_start": "2026-09-07T10:00:00+00:00",
        "window_end": "2026-09-07T13:00:00+00:00",
    }
    receipt = _evidence(tmp_path, "receipt.json", {"received": True, **window})
    dedup = _evidence(
        tmp_path,
        "dedup.json",
        {"send_count_delta": 0, "duplicate_inbox_count": 0, **window},
    )

    result = notification.run_notify001(
        SimpleNamespace(),
        SimpleNamespace(cluster_id="cluster-a"),
        attempt=1,
        run_dir=tmp_path,
        receipt_evidence=receipt,
        ses_window_evidence=dedup,
    )

    assert result["verdict"] == "PASS", result["checks"]
    assert result["sources"] == {"gpu-reset": "live", "workload-restart": "live"}
    # Exactly one drill: the four-submission deduplication proof that NOTIFY-002
    # used to carry; no completion is re-executed or re-mailed for a live kind.
    assert drills == ["gpu-reset"]
    assert result["checks"]["four_submissions_return_one_provider_id"] is True
    assert result["dedup_drill"]["drill_id"].startswith("notify001-dedup-"), result[
        "dedup_drill"
    ]
    assert result["drills"][0]["executed_at"] == "2026-09-07T12:00:00+00:00"
    assert "2026-09-07T11:00:00+00:00" in result["record_times"]


def test_notify001_falls_back_to_drills_and_fails_on_a_duplicate_live_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drills = _install_notify001_fakes(
        monkeypatch, live_records=[_live("workload-restart", count=2)]
    )

    result = notification.run_notify001(
        SimpleNamespace(),
        SimpleNamespace(cluster_id="cluster-a"),
        attempt=1,
        run_dir=tmp_path,
        receipt_evidence=None,
        ses_window_evidence=None,
    )

    assert result["sources"] == {"gpu-reset": "drill", "workload-restart": "live"}
    # The gpu-reset fallback drill doubles as the deduplication drill.
    assert drills == ["gpu-reset"]
    assert result["checks"]["gpu_reset_sent"] is True
    assert result["checks"]["workload_restart_deduplicated"] is False
    assert result["checks"]["receipt_confirmed_outside_the_solution"] is False
    assert result["verdict"] == "FAIL"


def test_notify002_declares_that_notify001_supersedes_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        notification,
        "drill",
        lambda site, target, *, kind, drill_id, pod=None: _fake_drill(kind, drill_id),
    )

    result = notification.run_notify002(
        SimpleNamespace(),
        SimpleNamespace(cluster_id="cluster-a"),
        attempt=1,
        ses_window_evidence=None,
    )

    assert result["superseded_by"] == "GF-REGIONAL-NOTIFY-001"
    assert "superseded by GF-REGIONAL-NOTIFY-001" in capsys.readouterr().err
    assert result["checks"]["four_submissions_return_one_provider_id"] is True


# --- NOTIFY-003: focused tests recorded in --plan, reused in --execute ------


def test_notify003_focused_tests_reuse_the_plan_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = {"passed": True, "returncode": 0, "command": ["pytest"]}
    monkeypatch.setattr(
        notification, "reusable_focused_tests", lambda path: dict(recorded)
    )

    def unexpected_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pytest must not run when the plan result is reusable")

    monkeypatch.setattr(notification, "run", unexpected_run)

    result = notification.notify003_focused_tests(tmp_path, reuse=True)

    assert result["focused_tests_reused"] is True and result["passed"] is True
    assert notification.NOTIFY003_FOCUSED_TESTS[0].startswith(
        "tests/notifications/test_notifications.py::"
    ), notification.NOTIFY003_FOCUSED_TESTS


def test_drill_reuses_the_resolved_control_worker_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def unexpected_lookup(site: Any, target: Any) -> str:
        raise AssertionError("ready_pods must not be listed again per drill")

    monkeypatch.setattr(notification, "control_worker_pod", unexpected_lookup)
    site = SimpleNamespace(
        pod_json=lambda plane, target, pod, script, *args, timeout: (
            calls.append(pod) or {"statuses": ["SENT"]}
        )
    )

    result = notification.drill(
        site,
        SimpleNamespace(cluster_id="cluster-a"),
        kind="gpu-reset",
        drill_id="d1",
        pod="worker-7",
    )

    assert calls == ["worker-7"]
    assert result["drill_id"] == "d1" and "executed_at" in result


# --- the drill probe ---------------------------------------------------------


def test_drill_probe_reports_when_it_sent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        drill_probe, "notification_notifier_from_environment", RecordingNotifier
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["drill", "--kind", "gpu-reset", "--drill-id", "d1", "--cluster-id", "c-a"],
    )

    assert drill_probe.main() == 0

    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert payload["drill_id"] == "d1"
    # One real send, three deduplicated repeats sharing its provider ID.
    assert payload["statuses"][0] == "SENT" and len(payload["statuses"]) == 4
    assert payload["provider_message_id_stable"] is True
    executed_at = datetime.fromisoformat(payload["executed_at"])
    assert executed_at.tzinfo is not None
    assert datetime.now(timezone.utc) - executed_at < timedelta(minutes=5)


def test_main_records_evidence_identity_and_plan_text_names_supersession() -> None:
    source = Path(notification.__file__).read_text(encoding="utf-8")

    # The predecessor is judged against, and the result carries, this
    # deployment's release/cluster identity.
    assert "predecessor_evidence(path, predecessor_id, **identity)" in source
    assert "**identity," in source

    plan = notification.case_plan(
        "GF-REGIONAL-NOTIFY-002",
        target=SimpleNamespace(cluster_id="c", context="x"),
        nodes=(),
        predecessor={},
    )
    assert plan["mutation"].startswith("superseded by GF-REGIONAL-NOTIFY-001"), plan[
        "mutation"
    ]
