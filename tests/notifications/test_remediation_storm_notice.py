"""The operator notice of a tripped remediation storm breaker."""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.notifications.registry import NOTIFICATION_REGISTRY, NotificationKind
from gpu_fault.notifications.remediation_storm import (
    REMEDIATION_STORM_TEMPLATE_VERSION,
    RemediationStormEmailBuilder,
)
from gpu_fault.remediation_storm import RemediationStormBreaker, StormBreakerState

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)


def _tripped() -> RemediationStormBreaker:
    return RemediationStormBreaker(
        cluster_id="gpu-eks-cluster",
        state=StormBreakerState.TRIPPED,
        trip_id=f"gpu-eks-cluster@{NOW.isoformat()}",
        tripped_at=NOW,
        trip_node_ids=["node-a", "node-b", "node-c"],
        trip_domains=3,
        trip_threshold=3,
        trip_cluster_nodes=4,
        trip_workflow_id="workflow-c",
        updated_at=NOW,
    )


def test_the_notice_names_the_trip_and_both_exits() -> None:
    notice = RemediationStormEmailBuilder().build(record=_tripped())

    assert notice.deduplication_key == (
        f"gpu-eks-cluster/remediation-storm-tripped/gpu-eks-cluster@{NOW.isoformat()}"
    )
    assert notice.category == "OPERATOR_REVIEW"
    assert notice.priority == 0, "ahead of a storm's per-node mail"
    for text in (
        "node-a, node-b, node-c",
        "max(3, ceil(20% of 4 GPU nodes))",
        "workflow-c",
        "remediation-storm release",
        "remediation-storm abort",
        "Work already in flight goes on",
        REMEDIATION_STORM_TEMPLATE_VERSION,
    ):
        assert text in notice.body_text, text


def test_the_notice_kind_is_registered() -> None:
    definition = NOTIFICATION_REGISTRY[NotificationKind.REMEDIATION_STORM_TRIPPED]

    assert definition.builder_type is RemediationStormEmailBuilder
