"""The SES client has bounded timeouts and a bounded dedup cache.

Control-plane review 2026-09-08, F-4. ``boto3.client("sesv2")`` was built
without a ``botocore.config.Config``: 60 s connect + 60 s read and up to five
legacy-mode attempts, so one hung ``send_email`` could outlive the 120 s
delivery lease while ``SesEmailNotifier.send`` held the process RLock -- and the
next replica to claim the row mailed it again in parallel. ``_results`` grew by
one entry per notification for the life of the process.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from gpu_fault.models import AdvisoryNotification, NotificationStatus
from gpu_fault.notifications import SesEmailNotifier, SesNotificationConfig
from tests.notifications._support import FakeSesV2Client


def _config() -> SesNotificationConfig:
    return SesNotificationConfig(
        sender="alerts@example.com",
        recipients=["oncall@example.com"],
        region_name="us-west-2",
        execution_enabled=True,
    )


def test_the_ses_client_is_created_with_bounded_timeouts_and_retries(monkeypatch):
    created: list[tuple[tuple, dict]] = []

    def client(*args, **kwargs):
        created.append((args, kwargs))
        return FakeSesV2Client()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=client))

    SesEmailNotifier(_config())

    assert len(created) == 1
    args, kwargs = created[0]
    assert args == ("sesv2",)
    assert kwargs["region_name"] == "us-west-2"
    config = kwargs["config"]
    assert config.connect_timeout == 5
    assert config.read_timeout == 20
    assert config.retries == {"max_attempts": 2, "mode": "standard"}


def _notification(index: int) -> AdvisoryNotification:
    return AdvisoryNotification(
        deduplication_key=f"dedup-{index}",
        cluster_name="cluster-a",
        incident_id=f"incident-{index}",
        subject="subject",
        body_text="body",
        support_case_draft="",
    )


def test_the_dedup_cache_is_bounded_and_keeps_the_newest_keys():
    notifier = SesEmailNotifier(_config(), client=FakeSesV2Client())
    limit = SesEmailNotifier.RESULT_CACHE_LIMIT
    assert 0 < limit <= 4096

    for index in range(limit + 100):
        assert notifier.send(_notification(index)).status is NotificationStatus.SENT

    assert len(notifier._results) == limit
    # The newest key is still deduplicated in-process ...
    newest = notifier.send(_notification(limit + 99))
    assert newest.status is NotificationStatus.DUPLICATE
    # ... and the oldest has been evicted; the shared notification_result row
    # is the cross-process truth, so this is only a hint.
    assert notifier.send(_notification(0)).status is NotificationStatus.SENT
