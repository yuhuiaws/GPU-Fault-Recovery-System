"""``spec.notificationDelivery.batchSize``: the dispatcher's per-cycle claim.

One bound (``gpu_fault.notification_limits``) is shared by the admin config,
the outbox dispatcher and the dispatch API. The admin config used to accept
1..1000 while ``dispatch_outbox`` refuses anything above 100, so a value in
between was accepted, rendered as ``GPU_FAULT_NOTIFICATION_BATCH_SIZE`` and
left every control worker failing its dispatch cycle every poll without
delivering a single notification.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gpu_fault.admin.config import (
    AdminConfig,
    admin_config_desired_path,
    canonical_sha256,
    default_admin_config,
    load_desired_admin_config,
)
from gpu_fault.admin.config_parser import AdminConfigError
from gpu_fault.admin.config_patch import apply_patch
from gpu_fault.app import lifespan_workers as workers
from gpu_fault.models import NotificationDispatchRequest
from gpu_fault.notification_limits import (
    NOTIFICATION_DISPATCH_BATCH_MAX,
    NOTIFICATION_DISPATCH_BATCH_MIN,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import HyperPodAdvisoryEmailBuilder
from tests._builders import build_store, copy_model
from tests.app_services._cov95_runtime_workers import Stop, ThreadRecorder
from tests.notifications._support import RecordingNotifier, advisory

MAX = NOTIFICATION_DISPATCH_BATCH_MAX


def _batch(size: int) -> dict[str, object]:
    return {"notificationDelivery": {"batchSize": size}}


def _record(size: int) -> dict[str, object]:
    record = default_admin_config().as_dict()
    section = record["notification_delivery"]
    assert isinstance(section, dict), "notification_delivery is a section"
    section["batch_size"] = size
    return record


def test_the_admin_config_accepts_exactly_the_dispatchers_range() -> None:
    for size in (NOTIFICATION_DISPATCH_BATCH_MIN, 25, MAX):
        config = apply_patch(default_admin_config(), _batch(size))
        assert config.notification_delivery.batch_size == size, (
            f"{size} is a batch the dispatcher claims, so the config takes it"
        )
    for size in (0, MAX + 1, 1000):
        with pytest.raises(
            AdminConfigError,
            match=rf"batchSize must be within 1\.\.{MAX}: the notification dispatcher",
        ):
            apply_patch(default_admin_config(), _batch(size))


def test_the_dispatcher_and_its_api_take_the_same_bound() -> None:
    service = AdvisoryNotificationService(
        build_store(), RecordingNotifier(), async_delivery=True
    )

    assert service.dispatch_outbox("owner", limit=MAX).attempted == 0, (
        "the largest batch the admin config accepts is one the dispatcher claims"
    )
    with pytest.raises(ValueError, match=f"between 1 and {MAX}"):
        service.dispatch_outbox("owner", limit=MAX + 1)
    with pytest.raises(ValueError, match=f"between 1 and {MAX}"):
        service.dispatch_pending(limit=MAX + 1)
    assert NotificationDispatchRequest(limit=MAX).limit == MAX, (
        "the dispatch API takes the largest batch"
    )
    with pytest.raises(ValidationError):
        NotificationDispatchRequest(limit=MAX + 1)


def test_a_record_from_before_the_bound_reads_and_has_to_be_planned_down(
    tmp_path: Path,
) -> None:
    """A persisted value above the bound is what that site runs: reading it
    must not fail, or the administrator could not load the state to fix it."""

    record = _record(200)
    legacy = AdminConfig.from_mapping(record)
    path = admin_config_desired_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": record,
                "config_sha256": canonical_sha256(record),
                "role_sha256": legacy.role_sha256(),
                "source": "approved-plan:large-notification-batch",
                "updated_at": "2026-10-30T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    assert load_desired_admin_config(tmp_path) == legacy, "the record still loads"
    assert legacy.notification_delivery.batch_size == 200, "read as recorded"
    with pytest.raises(AdminConfigError, match=rf"within 1\.\.{MAX}"):
        apply_patch(legacy, {"workflow": {"dispatcherWorkers": 12}})
    planned = apply_patch(legacy, _batch(MAX))
    assert planned.notification_delivery.batch_size == MAX, (
        "the next config apply lowers it into the dispatcher's range"
    )
    with pytest.raises(AdminConfigError, match=r"within 1\.\.1000"):
        AdminConfig.from_mapping(_record(1001))


def test_the_worker_claims_at_most_the_bound_when_a_larger_value_is_rendered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A role rendered from such a record still carries the larger value; the
    dispatcher refused it on every cycle and delivered nothing."""

    monkeypatch.setattr(workers, "Thread", ThreadRecorder())
    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_BATCH_SIZE", "200")
    store = build_store()
    notifier = RecordingNotifier()
    notification = store.save_notification_if_absent(
        copy_model(
            HyperPodAdvisoryEmailBuilder().build(
                advisory(),
                cluster_name="hp-cluster",
                incident_id="incident-large-batch",
                node_ids=["worker-1"],
                issue_summary="dispatcher batch above the bound",
            ),
            created_at=datetime.now(timezone.utc) - timedelta(seconds=60),
        )
    )
    worker = workers.start_notification_worker(
        context=SimpleNamespace(
            advisory_notifications=AdvisoryNotificationService(
                store, notifier, async_delivery=True
            )
        ),
        stop=Stop(1),
        owner="worker-0",
        throttle_delay=lambda previous, **_kwargs: previous,
    )

    with caplog.at_level(logging.WARNING):
        worker.run()

    assert [item.notification_id for item in notifier.notifications] == [
        notification.notification_id
    ], "the pending notification is delivered in the first cycle"
    assert "notification dispatch cycle failed" not in caplog.text, (
        "no cycle fails on the rendered value"
    )
    assert caplog.text.count("GPU_FAULT_NOTIFICATION_BATCH_SIZE=200") == 1, (
        "the clamp is reported once per process, not every cycle"
    )
