from __future__ import annotations

import io
import json
import subprocess
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest
from botocore.exceptions import ParamValidationError

import gpu_fault.notifications as notifications
from gpu_fault.notifications import sns
from scripts.e2e.regional import boot_acceptance_runtime as runtime


@pytest.mark.parametrize("channel", ["sns", "ses"])
def test_region_probe_uses_the_active_notification_channel_without_delivery(
    channel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    region = "us-west-2"
    topic = f"arn:aws:sns:{region}:000000000000:unit-alerts"
    calls = []
    config = SimpleNamespace(
        region_name=region,
        topic_arn=topic,
        sender="sender@example.com",
        execution_enabled=False,
    )

    class Client:
        meta = SimpleNamespace(region_name=region)

        def publish(self, **kwargs):
            calls.append(("sns", kwargs))
            raise ParamValidationError(report="unit invalid Message type")

        def send_email(self, **kwargs):
            calls.append(("ses", kwargs))
            raise ParamValidationError(report="unit invalid Destination type")

    class Notifier:
        @staticmethod
        def _create_client(value):
            assert value is config
            return Client()

    fake_config = SimpleNamespace(from_environment=lambda: config)
    monkeypatch.setattr(sns, "SnsNotificationConfig", fake_config)
    monkeypatch.setattr(sns, "SnsNotifier", Notifier)
    monkeypatch.setattr(notifications, "SesNotificationConfig", fake_config)
    monkeypatch.setattr(notifications, "SesEmailNotifier", Notifier)
    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_CHANNEL", channel)
    monkeypatch.setenv("AWS_REGION", region)
    monkeypatch.setenv("GPU_FAULT_ALLOW_EMAIL", "false")

    def negative(command, *, env, **_kwargs):
        assert "AWS_REGION" not in env and "AWS_DEFAULT_REGION" not in env
        assert env["AWS_CONFIG_FILE"] == "/dev/null"
        if channel == "sns":
            assert "SnsNotificationConfig" in command[-1]
            return subprocess.CompletedProcess(command, 0, f"REGION={region}\n", "")
        assert "SesNotificationConfig" in command[-1]
        return subprocess.CompletedProcess(command, 1, "", "NoRegionError")

    monkeypatch.setattr(subprocess, "run", negative)
    output = io.StringIO()
    with redirect_stdout(output):
        exec(runtime.BOOT013_PROBE, {})
    result = json.loads(output.getvalue())
    assert result == {
        "channel": channel,
        "region_present": True,
        "config_region": region,
        "client_region": region,
        "execution_enabled": False,
        "allow_email": False,
        "local_param_validation": True,
        "no_region_error": True,
    }
    assert len(calls) == 1 and calls[0][0] == channel
    if channel == "sns":
        assert calls[0][1] == {"TopicArn": topic, "Message": 1}
    else:
        assert calls[0][1]["Destination"] == {"ToAddresses": [1]}


@pytest.mark.parametrize("channel", ["unknown", "disabled"])
def test_unavailable_channel_cannot_fall_back_to_an_unrelated_client(
    channel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_CHANNEL", channel)
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: pytest.fail("must not launch a client")
    )
    with pytest.raises(ValueError):
        exec(runtime.BOOT013_PROBE, {})
