"""The control plane's own alerts go to the site's SNS topic by default.

``spec.notifications.channel`` is ``sns`` (default) or ``ses`` (opt-in). A
site written before the key existed carries an ``emailSender`` and therefore
stays on ``ses`` until the operator says otherwise, so an upgrade never moves
a live site's notifications. The channel travels site -> release config ->
control-plane apply and rollback environments -> every role's container:
each role builds the notifier and runs the fail-closed alert-channel guard at
startup, so unlike retention the variables reach ingress, worker and spool.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.site import (
    NotificationSiteConfig,
    RegionalSite,
    SiteConfigError,
    load_site,
    site_notification_channel,
)
from gpu_fault_release import regional_notifications
from gpu_fault_release import regional_release_config as release_config_module
from gpu_fault_release import regional_release_rendering as rendering
from tests.admin.test_admin_site import site_file as write_site_file
from tests.regional._release_orchestrator_support import SNS_TOPIC_ARN, config_file

ROOT = Path(__file__).resolve().parents[2]
RENDER_SCRIPT = ROOT / "deploy/control-plane/tools/render_control_plane_role_split.py"
CHANNEL_ENV = ("GPU_FAULT_NOTIFICATION_CHANNEL", "GPU_FAULT_SNS_TOPIC_ARN")
ROLES = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)
LEGACY_BLOCK = {
    "allowEmail": True,
    "acknowledgeExternalAlertChannel": False,
    "adminEmail": "ops@example.com",
    "emailSender": "sender@example.com",
    "emailRecipients": ["ops@example.com", "oncall@example.com"],
    "emailSubjectPrefix": "[PROD]",
}


@pytest.fixture
def site_file(tmp_path: Path) -> Path:
    return write_site_file(tmp_path)


def _site_document(site_file: Path, notifications: dict[str, Any] | None) -> Path:
    document = yaml.safe_load(site_file.read_text(encoding="utf-8"))
    if notifications is None:
        document["spec"].pop("notifications", None)
    else:
        document["spec"]["notifications"] = notifications
    site_file.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return site_file


def _parse(site_file: Path, notifications: dict[str, Any] | None) -> RegionalSite:
    return RegionalSite.from_value(
        yaml.safe_load(_site_document(site_file, notifications).read_text())
    )


# --------------------------------------------------------------------------
# site.yaml
# --------------------------------------------------------------------------


def test_an_admin_email_alone_means_sns_notifications(site_file: Path) -> None:
    """The bootstrap block of a new site: one address, no sender, no channel."""

    site = _parse(site_file, {"adminEmail": "ops@example.com"})

    assert site.spec.notifications == NotificationSiteConfig(
        allow_email=True,
        acknowledge_external_alert_channel=True,
        admin_email="ops@example.com",
        email_recipients=("ops@example.com",),
        channel="sns",
    )


def test_a_site_without_a_notifications_block_stays_acknowledged_off(
    site_file: Path,
) -> None:
    site = _parse(site_file, None)

    assert site.spec.notifications.allow_email is False
    assert site.spec.notifications.channel == "sns"


def test_a_legacy_block_with_a_sender_keeps_ses(site_file: Path) -> None:
    """An upgrade must not move a site whose operator verified an SES sender."""

    site = _parse(site_file, dict(LEGACY_BLOCK))

    assert site.spec.notifications.channel == "ses"
    assert site.spec.notifications.email_sender == "sender@example.com"


def test_declaring_sns_on_a_legacy_block_flips_the_channel_only(
    site_file: Path,
) -> None:
    """Moving a live site edits one key; the stale addresses are carried, not
    rejected, so the operator does not have to strip them first."""

    site = _parse(site_file, {**LEGACY_BLOCK, "channel": "sns"})

    assert site.spec.notifications.channel == "sns"
    assert site.spec.notifications.allow_email is True
    assert site.spec.notifications.email_sender == "sender@example.com"


@pytest.mark.parametrize(
    ("notifications", "message"),
    [
        ({"adminEmail": "ops@example.com", "channel": "smtp"}, "must be one of"),
        ({"adminEmail": "ops@example.com", "channel": 1}, "must be one of"),
        ({"allowEmail": True, "channel": "sns"}, "require notifications.adminEmail"),
        (
            {"allowEmail": True, "adminEmail": "ops@example.com", "channel": "ses"},
            "emailSender",
        ),
    ],
)
def test_channel_and_its_required_addresses_are_validated(
    site_file: Path, notifications: dict[str, Any], message: str
) -> None:
    with pytest.raises(SiteConfigError, match=message):
        load_site(_site_document(site_file, notifications))


def test_the_channel_is_rendered_into_the_release_config(site_file: Path) -> None:
    rendered = load_site(_site_document(site_file, {"adminEmail": "ops@example.com"}))

    assert rendered.release_config["notifications"] == {
        "allow_email": True,
        "acknowledge_external_alert_channel": True,
        "admin_email": "ops@example.com",
        "email_sender": None,
        "email_recipients": ["ops@example.com"],
        "email_subject_prefix": "",
        "channel": "sns",
    }


@pytest.mark.parametrize(
    ("document", "channel"),
    [
        ({}, "sns"),
        ({"spec": {}}, "sns"),
        ({"spec": {"notifications": {"adminEmail": "a@example.com"}}}, "sns"),
        ({"spec": {"notifications": {"emailSender": "s@example.com"}}}, "ses"),
        (
            {
                "spec": {
                    "notifications": {"emailSender": "s@example.com", "channel": "sns"}
                }
            },
            "sns",
        ),
        ({"spec": {"notifications": {"channel": " ses "}}}, "ses"),
        ({"spec": {"notifications": {"channel": "smtp"}}}, "sns"),
    ],
)
def test_raw_document_channel_answers_before_validation(
    document: dict[str, Any], channel: str
) -> None:
    assert site_notification_channel(document) == channel


# --------------------------------------------------------------------------
# release config -> apply environment
# --------------------------------------------------------------------------


def _release_config(
    tmp_path: Path, notifications: dict[str, Any], *, topic: str | None
) -> release_config_module.ReleaseConfig:
    path = config_file(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["notifications"] = notifications
    if topic is not None:
        value["health"] = {"sns_topic_arn": topic}
    path.write_text(json.dumps(value), encoding="utf-8")
    return release_config_module.ReleaseConfig.load(path)


def test_release_config_threads_the_channel_and_defaults_it_like_the_site(
    tmp_path: Path,
) -> None:
    sns = _release_config(
        tmp_path, {"allow_email": True, "admin_email": "ops@example.com"}, topic=None
    )
    assert sns.notifications.channel == "sns"

    legacy = _release_config(
        tmp_path,
        {
            "allow_email": True,
            "admin_email": "ops@example.com",
            "email_sender": "sender@example.com",
        },
        topic=None,
    )
    assert legacy.notifications.channel == "ses", (
        "a release config written before the key existed rolls back onto ses"
    )

    with pytest.raises(release_config_module.ReleaseError, match="must be one of"):
        _release_config(
            tmp_path,
            {"allow_email": True, "admin_email": "ops@example.com", "channel": "x"},
            topic=None,
        )
    with pytest.raises(release_config_module.ReleaseError, match="admin_email"):
        _release_config(tmp_path, {"allow_email": True, "channel": "sns"}, topic=None)


def test_notification_environment_names_the_channel_and_the_topic_for_sns(
    tmp_path: Path,
) -> None:
    sns = _release_config(
        tmp_path,
        {"allow_email": True, "admin_email": "ops@example.com"},
        topic=SNS_TOPIC_ARN,
    )
    assert sns.notification_environment() == {
        "GPU_FAULT_NOTIFICATION_CHANNEL": "sns",
        "GPU_FAULT_SNS_TOPIC_ARN": SNS_TOPIC_ARN,
    }

    ses = _release_config(
        tmp_path,
        {
            "allow_email": True,
            "admin_email": "ops@example.com",
            "email_sender": "sender@example.com",
            "channel": "ses",
        },
        topic=SNS_TOPIC_ARN,
    )
    assert ses.notification_environment() == {"GPU_FAULT_NOTIFICATION_CHANNEL": "ses"}

    no_topic = _release_config(
        tmp_path, {"allow_email": True, "admin_email": "ops@example.com"}, topic=None
    )
    assert no_topic.notification_environment() == {
        "GPU_FAULT_NOTIFICATION_CHANNEL": "sns"
    }, "the runtime guard, not the renderer, reports a missing topic"


def test_cpu_apply_environment_carries_the_channel_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in CHANNEL_ENV:
        monkeypatch.delenv(name, raising=False)
    release = SimpleNamespace(
        config=_release_config(
            tmp_path,
            {"allow_email": True, "admin_email": "ops@example.com"},
            topic=SNS_TOPIC_ARN,
        ),
        wheel_cm="wheel-cm",
        wheel_sha="a" * 64,
        runtime_image="registry.example/runtime@sha256:" + "b" * 64,
        node_wheel_sha="c" * 64,
        executor_wheel_sha="d" * 64,
        admin_config_digest="e" * 64,
        admin_config_role_digests={
            "ingress": "f" * 64,
            "worker": "1" * 64,
            "spool": "2" * 64,
        },
        release_id="release-a",
    )

    environment = rendering.build_cpu_apply_environment(release, finalize=False)

    assert environment["GPU_FAULT_NOTIFICATION_CHANNEL"] == "sns"
    assert environment["GPU_FAULT_SNS_TOPIC_ARN"] == SNS_TOPIC_ARN
    assert environment["GPU_FAULT_ALLOW_EMAIL"] == "true"


# --------------------------------------------------------------------------
# the rendered containers
# --------------------------------------------------------------------------


def _source_deployment() -> dict[str, Any]:
    """The smallest gpu-fault-api-ha Deployment the renderer accepts."""

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "gpu-fault-api-ha"},
        "spec": {
            "replicas": 3,
            "selector": {"matchLabels": {"app": "gpu-fault-api-ha"}},
            "template": {
                "metadata": {
                    "labels": {"app": "gpu-fault-api-ha"},
                    "annotations": {"prometheus.io/port": "8080"},
                },
                "spec": {
                    "terminationGracePeriodSeconds": 120,
                    "containers": [
                        {
                            "name": "api",
                            "args": [
                                'exec "$@" gpu_fault.app:create_app --factory '
                                "--host 0.0.0.0 --port 8080"
                            ],
                            "env": [
                                {"name": "GPU_FAULT_PROCESSOR_WORKERS", "value": "24"}
                            ],
                            "readinessProbe": {"httpGet": {"port": 8080}},
                            "livenessProbe": {
                                "httpGet": {"path": "/healthz", "port": 8080}
                            },
                            "resources": {
                                "requests": {"cpu": "4", "memory": "4Gi"},
                                "limits": {"cpu": "8", "memory": "8Gi"},
                            },
                        }
                    ],
                },
            },
        },
    }


def _render_role_env(extra_env: dict[str, str]) -> dict[str, dict[str, str]]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(ROOT / "src"),
        "GPU_FAULT_CONTROL_WORKER_REPLICAS": "6",
        **extra_env,
    }
    completed = subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), "--json"],
        input=json.dumps(_source_deployment()),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    items = json.loads(completed.stdout)["items"]
    config_maps = {
        item["metadata"]["name"]: item.get("data", {})
        for item in items
        if item["kind"] == "ConfigMap"
    }
    result: dict[str, dict[str, str]] = {}
    for item in items:
        if item["kind"] != "Deployment":
            continue
        container = item["spec"]["template"]["spec"]["containers"][0]
        values: dict[str, str] = {}
        # Literal values are externalised into per-role ConfigMaps (envFrom).
        for ref in container.get("envFrom", []):
            values.update(config_maps.get(ref["configMapRef"]["name"], {}))
        for entry in container.get("env", []):
            if "value" in entry:
                values[entry["name"]] = entry["value"]
            else:
                ref = entry["valueFrom"].get("configMapKeyRef")
                if ref and ref["name"] in config_maps:
                    values[entry["name"]] = config_maps[ref["name"]].get(ref["key"], "")
        result[item["metadata"]["name"]] = values
    return result


def test_render_script_puts_the_channel_on_every_role_only_when_set() -> None:
    off = _render_role_env({})
    assert set(off) == set(ROLES)
    for deployment in off.values():
        assert not set(CHANNEL_ENV) & set(deployment)

    on = _render_role_env(
        {
            "GPU_FAULT_NOTIFICATION_CHANNEL": "sns",
            "GPU_FAULT_SNS_TOPIC_ARN": SNS_TOPIC_ARN,
        }
    )
    for name in ROLES:
        assert on[name]["GPU_FAULT_NOTIFICATION_CHANNEL"] == "sns", (
            f"{name} builds the notifier and runs the alert-channel guard at "
            "startup, so it must be told the channel"
        )
        assert on[name]["GPU_FAULT_SNS_TOPIC_ARN"] == SNS_TOPIC_ARN


# --------------------------------------------------------------------------
# release identity and the Secret
# --------------------------------------------------------------------------


def _notifications(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "allow_email": True,
        "acknowledge_external_alert_channel": False,
        "admin_email": "ops@example.com",
        "email_sender": "sender@example.com",
        "email_recipients": ("ops@example.com",),
        "email_subject_prefix": "[PROD]",
        "channel": "ses",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_notification_digest_changes_with_the_channel() -> None:
    ses = regional_notifications.notification_digest(_notifications())
    sns = regional_notifications.notification_digest(_notifications(channel="sns"))

    assert ses != sns, "flipping the channel must render as a release change"
    assert ses == regional_notifications.notification_digest(_notifications())


class _SecretRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, args: list[str], **_kwargs: Any) -> str:
        self.commands.append(list(args))
        return "kind: Secret\n"


def _secret_release(notifications: SimpleNamespace) -> SimpleNamespace:
    runner = _SecretRunner()
    return SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(
            notifications=notifications,
            namespace="gpu-fault-system",
            site_name="site-a",
            cpu_eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/cpu",
        ),
        _cpu=lambda *args: ["kubectl", *args],
    )


def test_sns_secret_carries_the_site_context_without_addresses() -> None:
    """Every channel reads site-id, account and subject prefix from the Secret;
    only ses puts the sender and recipients in it."""

    sns = _secret_release(
        _notifications(channel="sns", email_sender=None, email_recipients=())
    )
    regional_notifications.ensure_notification_secret(sns)
    create = sns.runner.commands[0]
    literals = [item for item in create if item.startswith("--from-literal=")]
    assert literals == [
        "--from-literal=email-subject-prefix=[PROD]",
        "--from-literal=site-id=site-a",
        "--from-literal=aws-account-id=123456789012",
    ]
    assert sns.runner.commands[1] == ["kubectl", "apply", "-f", "-"]

    ses = _secret_release(_notifications())
    regional_notifications.ensure_notification_secret(ses)
    assert "--from-literal=email-sender=sender@example.com" in ses.runner.commands[0]

    broken = _secret_release(_notifications(email_sender=None))
    with pytest.raises(release_config_module.ReleaseError, match="addresses"):
        regional_notifications.ensure_notification_secret(broken)
