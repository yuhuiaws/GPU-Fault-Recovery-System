from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import notify007_verdicts as verdicts
from scripts.e2e.regional import run_notification_acceptance as runner
from scripts.e2e.regional import run_notify007_delivery_states as notify007
from scripts.e2e.regional.probes.notification_drill import (
    DrillNotifier,
    build_notification,
)


@pytest.mark.parametrize("received", [False, "false", 1, [], {"received": True}])
def test_receipt_verdict_cannot_be_supplied_by_the_evidence(
    tmp_path: Path, received: object
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "received": received,
                "valid": True,
                "errors": [],
                "unreviewed": "not evidence",
                "method": "inbox",
                "reference": "receipt-ref",
                "window_start": "2026-09-01T00:00:00Z",
                "window_end": "2026-09-01T00:01:00Z",
            }
        )
    )
    result = runner.validate_external_evidence(path, "receipt")
    assert result["valid"] is False
    assert result["errors"] == ["received is not true"]
    assert "unreviewed" not in result


@pytest.mark.parametrize("count", [False, 0.1, "0", None])
def test_dedup_count_must_be_an_actual_integer(tmp_path: Path, count: object) -> None:
    path = tmp_path / "dedup.json"
    path.write_text(
        json.dumps(
            {
                "send_count_delta": count,
                "duplicate_inbox_count": 0,
                "method": "counter",
                "reference": "ref",
                "window_start": "2026-09-01T00:00:00Z",
                "window_end": "2026-09-01T00:01:00Z",
            }
        )
    )
    result = runner.validate_external_evidence(path, "dedup")
    assert result["valid"] is False
    assert any("integer 0" in item for item in result["errors"]), result


def test_window_without_a_timezone_does_not_prove_a_send(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "received": True,
                "method": "inbox",
                "reference": "ref",
                "window_start": "2026-09-01T00:00:00",
                "window_end": "2026-09-01T00:01:00",
            }
        )
    )
    assert not runner.validate_external_evidence(
        path, "receipt", record_times=[datetime(2026, 9, 1, tzinfo=timezone.utc)]
    )["valid"]


@pytest.mark.parametrize("status", ["FAILED", "PENDING", None])
def test_failed_live_completion_is_not_hidden_by_a_successful_drill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str | None
) -> None:
    monkeypatch.setattr(runner, "control_worker_pod", lambda *_: "worker")
    monkeypatch.setattr(
        runner,
        "live_action_completed_records",
        lambda *a, **kw: {
            "candidates": {"gpu-reset": [{"notification_id": "n"}]},
            "records": [
                {"kind": "gpu-reset", "notification_id": "n", "status": status}
            ],
        },
    )
    sends = []
    monkeypatch.setattr(runner, "drill", lambda *a, **kw: sends.append(kw))
    with pytest.raises(runner.NotificationAcceptanceError, match="cannot replace"):
        runner.run_notify001(
            SimpleNamespace(),
            SimpleNamespace(cluster_id="c"),
            attempt=1,
            run_dir=tmp_path,
            receipt_evidence=None,
            ses_window_evidence=None,
        )
    assert sends == []


def test_unlabelled_drill_never_reaches_the_provider() -> None:
    sends = []
    notifier = DrillNotifier(SimpleNamespace(send=sends.append), "drill-a")
    notification = build_notification("gpu-reset", "drill-a", "c")
    with pytest.raises(RuntimeError, match="unlabelled"):
        notifier.send(notification.model_copy(update={"subject": "not a drill"}))
    assert sends == []


@pytest.mark.parametrize("enabled,ready", [(False, 1), (True, 0)])
def test_notify003_rejects_disabled_or_missing_workers(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, ready: int
) -> None:
    config = {
        app: {
            "service_role": role,
            "dispatcher_enabled": str(enabled).lower(),
            "async_delivery": str(enabled).lower(),
            "desired_replicas": 1,
            "ready_replicas": ready,
            "replicas_agree": True,
        }
        for app, role in zip(
            runner.NOTIFICATION_DEPLOYMENTS, ("ingress", "worker", "spool-worker")
        )
    }
    monkeypatch.setattr(
        runner,
        "notify003_focused_tests",
        lambda *a, **kw: {"passed": True, "returncode": 0},
    )
    monkeypatch.setattr(runner, "deployment_notification_config", lambda *a: config)
    assert (
        runner.run_notify003(SimpleNamespace(), SimpleNamespace())["verdict"] == "FAIL"
    )


def metric_text(failed: int, dead: int) -> str:
    return "\n".join(
        [
            f'{verdicts.RESULT_METRIC}{{status="FAILED"}} {failed}',
            *(
                f'{verdicts.DELIVERY_METRIC}{{status="{status}"}} {dead if status == "DEAD" else 0}'
                for status in ("PENDING", "LEASED", "RETRY", "SENT", "DEAD")
            ),
            f"{verdicts.OLDEST_PENDING_METRIC} 0",
            f"{verdicts.DEAD_LETTERED_METRIC} 0",
            f"{verdicts.DISPATCH_CYCLE_METRIC} 1",
            f"{verdicts.TERMINAL_FAILURE_METRIC} 0",
        ]
    )


def test_production_metric_absence_and_hidden_replica_growth_are_not_zero() -> None:
    before = {"a/uid-a": metric_text(10, 0), "b/uid-b": metric_text(0, 0)}
    after = {**before, "b/uid-b": metric_text(1, 0)}
    assert verdicts.production_untouched_errors(before, after), {
        "before": before,
        "after": after,
    }
    assert verdicts.production_untouched_errors(before, {}), (
        "missing post-run replica metrics must not count as unchanged production"
    )
    assert verdicts.production_untouched_errors({}, {}), (
        "absent metrics before and after cannot prove production was untouched"
    )
    assert verdicts.production_untouched_errors(before, {**before, "b/uid-b": ""}), (
        "an empty metric response from one replica must invalidate the comparison"
    )
    assert verdicts.exported_family_errors([metric_text(0, 0), ""]), (
        "one complete replica must not hide missing metric families on another"
    )


def test_notification_metrics_refuse_empty_ready_pods() -> None:
    regional = SimpleNamespace(
        kubectl=lambda *a, **kw: json.dumps({"spec": {"replicas": 1}}),
        ready_pods=lambda *a: [],
    )
    with pytest.raises(Exception, match="Ready replicas"):
        notify007.control_plane_metrics(regional)
