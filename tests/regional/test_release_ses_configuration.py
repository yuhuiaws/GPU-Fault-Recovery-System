"""SES site configuration, release identity and CPU environment boundaries."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin.site import NotificationSiteConfig, SiteConfigError, load_site
from gpu_fault_release import regional_notifications as notifications
from gpu_fault_release import regional_release_rendering as rendering
from gpu_fault_release import regional_release_rollback_context as rollback
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import (
    RegionalNotificationConfig,
    ReleaseConfig,
    ReleaseError,
)
from tests.admin.test_admin_site import site_file
from tests.regional._release_orchestrator_support import config_file

SES_ENV = "GPU_FAULT_SES_CONFIGURATION_SET"
SES_BLOCK = {
    "allow_email": True,
    "acknowledge_external_alert_channel": False,
    "admin_email": "ops@example.com",
    "email_sender": "sender@example.com",
    "email_recipients": ["ops@example.com"],
    "email_subject_prefix": "[PROD]",
    "channel": "ses",
}
SITE_BLOCK = {
    "allowEmail": True,
    "adminEmail": "ops@example.com",
    "emailSender": "sender@example.com",
}


@pytest.mark.parametrize("channel", ["ses", "sns"])
@pytest.mark.parametrize("value", [None, " alerts-set_1 ", "a" * 64])
def test_site_and_low_level_config_preserve_valid_configuration_sets(
    channel: str, value: str | None
) -> None:
    site = NotificationSiteConfig.from_value(
        {**SITE_BLOCK, "channel": channel, "sesConfigurationSet": value}
    )
    release = RegionalNotificationConfig.from_mapping(
        {**SES_BLOCK, "channel": channel, "ses_configuration_set": value}
    )

    assert site.ses_configuration_set == release.ses_configuration_set
    assert site.ses_configuration_set == (value.strip() if value is not None else None)
    assert site.channel == release.channel == channel


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "my set",
        "a/b",
        "a#b",
        "a'b",
        "a\nb",
        "a\rb",
        "a" * 65,
        "\u00e9",
        7,
        [],
        {},
    ],
)
def test_both_config_boundaries_reject_invalid_configuration_sets(
    value: object,
) -> None:
    with pytest.raises(SiteConfigError, match="sesConfigurationSet"):
        NotificationSiteConfig.from_value({**SITE_BLOCK, "sesConfigurationSet": value})
    with pytest.raises(ReleaseError, match="ses_configuration_set"):
        RegionalNotificationConfig.from_mapping(
            {**SES_BLOCK, "ses_configuration_set": value}
        )


def test_site_materializes_the_configuration_set_into_release_input(
    tmp_path: Path,
) -> None:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["spec"]["notifications"] = {
        **SITE_BLOCK,
        "sesConfigurationSet": " alerts-set ",
    }
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    assert load_site(path).release_config["notifications"]["ses_configuration_set"] == (
        "alerts-set"
    )


def test_an_absent_configuration_set_preserves_the_legacy_notification_digest() -> None:
    legacy_payload = {"schema_version": 4, **SES_BLOCK}
    expected = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    legacy_shape = SimpleNamespace(**SES_BLOCK)
    parsed = RegionalNotificationConfig.from_mapping(dict(SES_BLOCK))

    assert notifications.notification_digest(legacy_shape) == expected
    assert notifications.notification_digest(parsed) == expected
    changed = replace(parsed, ses_configuration_set="alerts-set")
    assert notifications.notification_digest(changed) != expected
    assert (
        notifications.notification_digest(replace(changed, ses_configuration_set=None))
        == expected
    )


@pytest.mark.parametrize("value", [None, "new-set"])
def test_structured_rendering_handles_noncanonical_yaml_without_touching_other_data(
    value: str | None,
) -> None:
    documents = [
        {"kind": "Deployment", "metadata": {"name": "unchanged"}},
        {"kind": "ConfigMap", "data": {"unrelated": "old-set"}},
        {"kind": "ConfigMap", "data": {SES_ENV: "old-set", "other": "keep"}},
        None,
        "not-an-object",
        {"kind": "ConfigMap", "data": None},
    ]
    text = yaml.safe_dump_all(documents, default_style='"')
    rendered = rendering.render_notification_config_maps(text, configuration_set=value)

    documents[2]["data"][SES_ENV] = value or ""
    assert list(yaml.safe_load_all(rendered)) == documents


def test_unchanged_notification_manifest_bytes_are_not_reformatted() -> None:
    text = 'kind: ConfigMap\ndata: {"GPU_FAULT_SES_CONFIGURATION_SET": ""}\n'

    assert (
        rendering.render_notification_config_maps(text, configuration_set=None) == text
    )


def test_structured_renderer_refuses_invalid_names_and_invalid_output(
    monkeypatch,
) -> None:
    with pytest.raises(ReleaseError, match="ses_configuration_set"):
        rendering.render_notification_config_maps("", configuration_set="bad set")
    monkeypatch.setattr(rendering.yaml, "safe_dump_all", lambda *_args, **_kwargs: None)
    with pytest.raises(ReleaseError, match="did not produce YAML text"):
        rendering.render_notification_config_maps(
            f"kind: ConfigMap\ndata:\n  {SES_ENV}: ''\n", configuration_set="alerts-set"
        )


def release_config(tmp_path: Path, configuration_set: str | None) -> ReleaseConfig:
    return replace(
        ReleaseConfig.load(config_file(tmp_path)),
        notifications=RegionalNotificationConfig.from_mapping(
            {**SES_BLOCK, "ses_configuration_set": configuration_set}
        ),
    )


@pytest.mark.parametrize("value", [None, "alerts-set"])
def test_apply_environment_is_site_owned_and_cannot_inherit_rollback_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    for key, injected in {
        SES_ENV: "inherited-set",
        "GPU_FAULT_NOTIFICATION_CHANNEL": "sns",
        "GPU_FAULT_SNS_TOPIC_ARN": "inherited-topic",
        "GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE": "/untrusted/snapshot.json",
        "GPU_FAULT_LEGACY_COMPONENT_PINS": "true",
        "GPU_FAULT_PRESERVE_ROLE_CONFIG_MAPS": "true",
        "GPU_FAULT_FORCE_ROLE_RESTART": "true",
    }.items():
        monkeypatch.setenv(key, injected)
    release = rollout.RegionalRelease(
        release_config(tmp_path, value), rollout.Runner(dry_run=True)
    )
    environment = rendering.build_cpu_apply_environment(release, finalize=False)

    assert environment[SES_ENV] == (value or "")
    assert environment["GPU_FAULT_NOTIFICATION_CHANNEL"] == "ses"
    assert "GPU_FAULT_SNS_TOPIC_ARN" not in environment
    assert "GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE" not in environment
    assert environment["GPU_FAULT_LEGACY_COMPONENT_PINS"] == "false"
    assert environment["GPU_FAULT_PRESERVE_ROLE_CONFIG_MAPS"] == "false"
    assert environment["GPU_FAULT_FORCE_ROLE_RESTART"] == "false"


def rollback_environment(config: ReleaseConfig, **options: object) -> dict[str, str]:
    return rollback.build_rollback_environment(
        rollback_config=config,
        metadata={},
        cpu_wheel="previous-wheel",
        cpu_sha="a" * 64,
        artifact="b" * 64,
        config_digest="c" * 64,
        runtime_profile_version="previous-profile",
        runtime_image="registry.example/previous@sha256:" + "d" * 64,
        **options,
    )


@pytest.mark.parametrize("value", [None, "previous-set"])
def test_rollback_clears_inherited_ses_and_sns_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    monkeypatch.setenv(SES_ENV, "candidate-set")
    monkeypatch.setenv("GPU_FAULT_SNS_TOPIC_ARN", "untrusted-topic")

    environment = rollback_environment(release_config(tmp_path, value))

    assert environment[SES_ENV] == (value or "")
    assert "GPU_FAULT_SNS_TOPIC_ARN" not in environment


def checked_channel(
    monkeypatch: pytest.MonkeyPatch,
    *,
    configuration_set: str | None = "alerts-set",
    channel: str = "ses",
    response: dict[str, object] | None = None,
) -> tuple[list[list[str]], dict[str, object]]:
    config = RegionalNotificationConfig.from_mapping(
        {**SES_BLOCK, "channel": channel, "ses_configuration_set": configuration_set}
    )
    calls: list[list[str]] = []
    payload = (
        {
            "ConfigurationSetName": configuration_set,
            "SendingOptions": {"SendingEnabled": True},
        }
        if response is None
        else response
    )

    def aws_json(arguments: list[str]) -> dict[str, object]:
        calls.append(arguments)
        if arguments[1] == "get-email-identity":
            return {"VerifiedForSendingStatus": True}
        if arguments[1] == "get-account":
            return {"SendingEnabled": True}
        assert arguments == [
            "sesv2",
            "get-configuration-set",
            "--configuration-set-name",
            configuration_set,
        ]
        return payload

    values = {
        "email-sender": "sender@example.com",
        "email-recipients": "ops@example.com",
        "email-subject-prefix": "[PROD]",
        "site-id": "site-a",
        "aws-account-id": "123456789012",
    }
    secret = {
        "data": {
            key: base64.b64encode(value.encode()).decode()
            for key, value in values.items()
        }
    }
    release = SimpleNamespace(
        config=SimpleNamespace(
            notifications=config,
            site_name="site-a",
            cpu_eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/cpu",
            health=SimpleNamespace(
                sns_topic_arn="arn:aws:sns:us-east-1:123456789012:alerts"
            ),
        )
    )
    _summary, details = notifications.check_notification_channel(
        release,
        aws_json=aws_json,
        read_secret=lambda _name: secret,
        decode_secret=base64.b64decode,
    )
    return calls, details


def test_ses_preflight_checks_the_declared_set_and_sending_state(monkeypatch) -> None:
    calls, details = checked_channel(monkeypatch)
    assert [call[1] for call in calls] == [
        "get-email-identity",
        "get-account",
        "get-configuration-set",
    ]
    assert details["ses_configuration_set"] == "alerts-set"


@pytest.mark.parametrize(
    "response",
    [
        {},
        {
            "ConfigurationSetName": "other-set",
            "SendingOptions": {"SendingEnabled": True},
        },
        {"ConfigurationSetName": "alerts-set"},
        {
            "ConfigurationSetName": "alerts-set",
            "SendingOptions": {"SendingEnabled": False},
        },
        {
            "ConfigurationSetName": "alerts-set",
            "SendingOptions": {"SendingEnabled": "true"},
        },
        {"ConfigurationSetName": "alerts-set", "SendingOptions": "unknown"},
    ],
)
def test_ses_preflight_refuses_missing_drifted_or_disabled_sets(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, object]
) -> None:
    with pytest.raises(ReleaseError, match="configuration set"):
        checked_channel(monkeypatch, response=response)


def test_absent_set_and_sns_do_not_query_unrelated_ses_configuration(
    monkeypatch,
) -> None:
    calls, details = checked_channel(monkeypatch, configuration_set=None)
    assert [call[1] for call in calls] == ["get-email-identity", "get-account"]
    assert "ses_configuration_set" not in details
    calls, details = checked_channel(monkeypatch, channel="sns")
    assert calls == []
    assert details["channel"] == "sns"
