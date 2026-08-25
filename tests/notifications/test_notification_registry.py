from __future__ import annotations

from gpu_fault.notifications.registry import (
    NOTIFICATION_REGISTRY,
    NotificationBuilderRegistry,
    NotificationKind,
    validate_notification_registry,
)


def test_notification_registry_is_exhaustive() -> None:
    validate_notification_registry()
    assert set(NOTIFICATION_REGISTRY) == set(NotificationKind)


def test_restart_notification_kinds_share_one_builder() -> None:
    registry = NotificationBuilderRegistry()
    first = registry.builder(NotificationKind.GPU_COUNT_CHANGE)
    second = registry.builder(NotificationKind.NODE_RESTARTED)
    assert first is second
