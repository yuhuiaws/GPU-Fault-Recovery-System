from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from gpu_fault import notification_service
from gpu_fault.models import AdvisoryNotification
from gpu_fault.store import InMemoryStore, SqliteStore

NOW = datetime(2030, 1, 1, tzinfo=UTC)


@pytest.fixture(name="runtime_store", params=["memory", "sqlite"])
def runtime_store_fixture(request, tmp_path, monkeypatch):
    store = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "notifications.db"))
    )
    store.establish_notification_watermark(
        established_at=NOW - timedelta(days=1), established_by="earlier-owner"
    )
    clock = SimpleNamespace(value=NOW)

    class Clock:
        @classmethod
        def now(cls, timezone):
            return clock.value

    monkeypatch.setattr(notification_service, "datetime", Clock)
    try:
        yield store, clock
    finally:
        if isinstance(store, SqliteStore):
            store.close()


def saved(store, index=0, **values):
    arguments = {
        "notification_id": f"notification-{index}",
        "deduplication_key": f"dedup-{index}",
        "incident_id": f"incident-{index}",
        "cluster_name": "cluster-local",
        "subject": "Local notification",
        "body_text": "Local notification body",
        "support_case_draft": "",
        "created_at": NOW,
        "not_before": NOW,
        "priority": 50 + index,
    }
    arguments.update(values)
    return store.save_notification_if_absent(AdvisoryNotification(**arguments))
