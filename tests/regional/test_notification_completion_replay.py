from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.models import NotificationResult, NotificationStatus
from scripts.e2e.regional import run_notification_acceptance as runner
from scripts.e2e.regional.probes import notification_drill


class Notifier:
    def __init__(self):
        self.notifications = []

    def send(self, notification):
        self.notifications.append(notification)
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id="unit-message",
        )


@pytest.mark.parametrize("kind", ["gpu-reset", "workload-restart"])
def test_four_actual_completion_handler_calls_create_one_notification_and_send(
    kind,
) -> None:
    notifier = Notifier()
    result = notification_drill.replay_completion(
        kind, "unit-drill", "unit-cluster", notifier
    )
    assert result["injection_path"] == notification_drill.INJECTION_PATH
    assert result["completion_calls"] == 4
    assert result["command_statuses"] == ["SUCCEEDED"] * 4
    assert result["notification_count"] == 1
    assert result["notifier_calls"] == len(notifier.notifications) == 1
    assert result["statuses"] == ["SENT", "DUPLICATE", "DUPLICATE", "DUPLICATE"]
    assert result["provider_message_id_stable"] is True
    assert result["http_authorization_exercised"] is False
    sent = notifier.notifications[0]
    assert sent.drill_id == "unit-drill"
    assert sent.subject.startswith("[DRILL:unit-drill]"), sent.subject
    assert sent.body_text.startswith("DRILL /"), sent.body_text
    assert "lease_token" not in json.dumps(result)


def test_expired_window_never_calls_the_real_notifier() -> None:
    notifier = Notifier()
    with pytest.raises(RuntimeError, match="window ended"):
        notification_drill.replay_completion(
            "gpu-reset",
            "unit-drill",
            "unit-cluster",
            notifier,
            deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
    assert notifier.notifications == []


@pytest.mark.parametrize(
    "defect",
    [
        {"verdict": "FAIL"},
        {"status": "FAILED"},
        {"release_id": "other"},
        {"formal_sequence_satisfied": "false"},
    ],
)
def test_live_candidate_cannot_borrow_a_failed_or_foreign_case(
    tmp_path, defect
) -> None:
    case_id = "GF-REGIONAL-DESTR-001"
    path = tmp_path / "cases" / case_id / f"{case_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "verdict": "PASS",
                "release_id": "unit-release",
                "cluster_id": "unit-cluster",
                "notifications": [
                    {
                        "notification": {
                            "notification_id": "unit-id",
                            "category": "ACTION_COMPLETED",
                            "cluster_name": "unit-cluster",
                            "deduplication_key": "unit/gpu-reset/op",
                        },
                        "result": {"status": "SENT"},
                    }
                ],
                **defect,
            }
        )
    )
    with pytest.raises(runner.NotificationAcceptanceError, match="failed or unbound"):
        runner.action_completed_records_from_evidence(
            tmp_path, cluster_id="unit-cluster", release_id="unit-release"
        )


@pytest.mark.parametrize(
    "result",
    [
        {"result": "DENIED", "code": "MessageRejected"},
        {"result": "DENIED", "code": "NotFound"},
        {"result": "ALLOWED"},
        {},
    ],
)
def test_executor_denial_is_checked_on_every_observed_replica(result) -> None:
    calls = []
    site = SimpleNamespace(
        ready_pods=lambda *a: ["executor-a", "executor-b"],
        pod_json=lambda plane, target, pod, script, **kw: (
            calls.append((pod, kw["timeout"]))
            or (
                {"result": "DENIED", "code": "AccessDenied"}
                if pod == "executor-a"
                else result
            )
        ),
    )
    report = runner.run_notify004(site, SimpleNamespace())
    assert report["verdict"] == "FAIL"
    assert calls == [("executor-a", 30), ("executor-b", 30)]
