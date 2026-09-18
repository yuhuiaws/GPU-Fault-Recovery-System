from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.models import NotificationResult, NotificationStatus
from scripts.e2e.regional import notification_evidence
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


@pytest.mark.parametrize(
    "kind,command_shape",
    [
        ("gpu-reset", "compound"),
        ("gpu-reset", "standalone"),
        ("workload-restart", "standalone"),
    ],
)
def test_four_actual_completion_handler_calls_create_one_notification_and_send(
    kind, command_shape
) -> None:
    notifier = Notifier()
    result = notification_drill.replay_completion(
        kind, "unit-drill", "unit-cluster", notifier, command_shape=command_shape
    )
    assert result["injection_path"] == notification_drill.INJECTION_PATH
    assert result["completion_calls"] == 4
    assert result["command_statuses"] == ["SUCCEEDED"] * 4
    assert result["notification_count"] == 1
    assert result["notifier_calls"] == len(notifier.notifications) == 1
    assert result["statuses"] == ["SENT", "DUPLICATE", "DUPLICATE", "DUPLICATE"]
    assert result["provider_message_id_stable"] is True
    assert result["http_authorization_exercised"] is False
    assert result["command_shape"] == command_shape
    sent = notifier.notifications[0]
    assert sent.drill_id == "unit-drill"
    assert sent.subject.startswith("[DRILL:unit-drill]"), sent.subject
    assert sent.body_text.startswith("DRILL /"), sent.body_text
    assert "lease_token" not in json.dumps(result)


def test_the_compound_drill_mails_the_batched_reset_step_and_never_the_head() -> None:
    """The production reset carrier: QUIESCE heads it, RESET_GPU rides along.

    The one mail is keyed by the passenger's own idempotency key, exactly the
    key a standalone RESET_GPU command would have used; the head produced none.
    """

    notifier = Notifier()
    result = notification_drill.replay_completion(
        "gpu-reset", "unit-drill", "unit-cluster", notifier
    )
    assert result["command_shape"] == "compound", "compound is the default shape"
    assert result["head_operation"] == "QUIESCE_GPU_SERVICES"
    assert result["batched_operations"] == [
        "VERIFY_NO_GPU_CLIENTS",
        "RESET_GPU",
        "RESTORE_GPU_SERVICES",
    ]
    assert result["expected_operation_id"] == "workflow-unit-drill/2/RESET_GPU"
    assert result["notification_operation_ids"] == ["workflow-unit-drill/2/RESET_GPU"]
    [sent] = notifier.notifications
    assert sent.deduplication_key.endswith(
        "/gpu-reset/workflow-unit-drill/2/RESET_GPU"
    ), sent.deduplication_key
    assert "QUIESCE_GPU_SERVICES" not in sent.deduplication_key
    assert notification_evidence.drill_mails_the_batched_reset(result) is True
    assert (
        runner.completion_checks(
            {
                "gpu-reset": {"sent": True, "deduplicated": True},
                "workload-restart": {"sent": True, "deduplicated": True},
            },
            result,
            {"valid": True},
            {"valid": True},
        )["gpu_reset_mail_keyed_by_the_batched_step"]
        is True
    )


def test_the_standalone_drill_still_proves_the_head_path() -> None:
    """A DAG workflow or a site with batching off still issues RESET_GPU alone;
    the shape stays selectable but does not satisfy the compound check."""

    notifier = Notifier()
    result = notification_drill.replay_completion(
        "gpu-reset", "unit-drill", "unit-cluster", notifier, command_shape="standalone"
    )
    assert result["head_operation"] == "RESET_GPU"
    assert result["batched_operations"] == []
    assert result["expected_operation_id"] == "workflow-unit-drill/0/RESET_GPU"
    assert result["notification_operation_ids"] == ["workflow-unit-drill/0/RESET_GPU"]
    assert notification_evidence.drill_mails_the_batched_reset(result) is False


@pytest.mark.parametrize(
    "override",
    [
        {"command_shape": "standalone"},
        {"head_operation": "RESET_GPU"},
        {"batched_operations": ["VERIFY_NO_GPU_CLIENTS", "RESTORE_GPU_SERVICES"]},
        {"notification_operation_ids": []},
        {
            "notification_operation_ids": [
                "workflow-d/0/QUIESCE_GPU_SERVICES",
                "workflow-d/2/RESET_GPU",
            ]
        },
        {"notification_operation_ids": ["workflow-d/0/QUIESCE_GPU_SERVICES"]},
        {"expected_operation_id": None},
    ],
)
def test_a_drill_that_mails_the_head_or_nothing_fails_the_batched_check(
    override,
) -> None:
    drill = {
        "command_shape": "compound",
        "head_operation": "QUIESCE_GPU_SERVICES",
        "batched_operations": [
            "VERIFY_NO_GPU_CLIENTS",
            "RESET_GPU",
            "RESTORE_GPU_SERVICES",
        ],
        "expected_operation_id": "workflow-d/2/RESET_GPU",
        "notification_operation_ids": ["workflow-d/2/RESET_GPU"],
    }
    assert notification_evidence.drill_mails_the_batched_reset(drill) is True
    assert notification_evidence.drill_mails_the_batched_reset(
        {**drill, **override}
    ) is (False), override


def test_an_unknown_command_shape_is_refused_before_any_send() -> None:
    notifier = Notifier()
    with pytest.raises(ValueError, match="command shape"):
        notification_drill.replay_completion(
            "gpu-reset", "unit-drill", "unit-cluster", notifier, command_shape="chain"
        )
    assert notifier.notifications == []


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
