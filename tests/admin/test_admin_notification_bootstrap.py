"""Which address the bootstrap sends mail from, and which one it alerts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import notification_bootstrap as admin_notification_bootstrap
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
)
from gpu_fault.admin.bootstrap_site import finalize_bootstrap_site
from gpu_fault.admin.notification_bootstrap import (
    notification_bootstrap_tasks,
    notification_routing,
)
from gpu_fault.admin.notifications import NotificationRouting
from gpu_fault.admin.site import NotificationSiteConfig


class Runner:
    dry_run = False

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.commands: list[tuple[str, ...]] = []

    def aws_json(self, _region: str, *arguments: str, **_kwargs: Any) -> Any:
        self.commands.append(arguments)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _cpu() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn="arn:aws:eks:us-west-2:123456789012:cluster/cpu",
        role="cpu",
        region="us-west-2",
        account_id="123456789012",
        hyperpod_arn="arn:aws:sagemaker:us-west-2:123456789012:cluster/cpu",
        hyperpod_name="cpu",
        eks_arn="arn:aws:eks:us-west-2:123456789012:cluster/cpu",
        eks_name="cpu",
        vpc_id="vpc-cpu",
        subnet_ids=("subnet-a", "subnet-b"),
        node_recovery="None",
        context="cpu",
    )


def _request(tmp_path: Path, **overrides: Any) -> BootstrapRequest:
    values: dict[str, Any] = {
        "cpu_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu",
        "gpu_cluster_arns": (),
        "repository_root": tmp_path,
        "state_dir": tmp_path / "state",
        "alert_email": "ops@example.com",
    }
    values.update(overrides)
    return BootstrapRequest(**values)


def _state(tmp_path: Path) -> BootstrapState:
    return BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")


def test_configured_alert_email_is_recorded_as_the_routing_source(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    runner = Runner()

    admin_email, routing = notification_routing(
        runner, _cpu(), _request(tmp_path), state
    )

    assert admin_email == "ops@example.com"
    assert routing == NotificationRouting(
        sender="ops@example.com", recipients=("ops@example.com",), subject_prefix=""
    )
    assert routing.channel == "sns", "a new site notifies over the site topic"
    assert state.value["resources"]["admin_email_source"] == "configured"
    assert state.value["resources"]["notification_channel"] == "sns"
    assert runner.commands == [], "a configured address must not be looked up in AWS"


@pytest.mark.parametrize(
    ("notifications", "expected"),
    [
        (None, "sns"),
        ({"adminEmail": "ops@example.com"}, "sns"),
        # Written before the channel existed: bootstrap always filled a sender,
        # and that sender is a verified SES identity the operator confirmed.
        ({"adminEmail": "ops@example.com", "emailSender": "ops@example.com"}, "ses"),
        ({"channel": "ses", "adminEmail": "ops@example.com"}, "ses"),
        (
            {
                "channel": "sns",
                "adminEmail": "ops@example.com",
                "emailSender": "ops@example.com",
            },
            "sns",
        ),
    ],
)
def test_the_channel_comes_from_the_existing_site(
    tmp_path: Path, notifications: dict[str, Any] | None, expected: str
) -> None:
    """A routine upgrade never moves a live site's notifications; the operator
    flips ``spec.notifications.channel`` and reruns deploy."""

    existing_site = (
        {"spec": {"notifications": notifications}} if notifications else None
    )

    _admin_email, routing = notification_routing(
        Runner(), _cpu(), _request(tmp_path), _state(tmp_path), existing_site
    )

    assert routing.channel == expected
    assert routing.uses_ses is (expected == "ses")


def test_discovered_alert_email_is_both_sender_and_recipient(tmp_path: Path) -> None:
    """There is no separate sender/recipient/prefix input any more: the
    discovered administrator address is the whole routing."""

    state = _state(tmp_path)
    runner = Runner([{"Account": {"Email": "root@example.com"}}])

    admin_email, routing = notification_routing(
        runner, _cpu(), _request(tmp_path, alert_email=None), state
    )

    assert admin_email == "root@example.com"
    assert routing == NotificationRouting(
        sender="root@example.com", recipients=("root@example.com",), subject_prefix=""
    ), "the discovered address is sender and recipient, with no prefix"
    assert state.value["resources"]["admin_email_source"] == "organizations"
    assert not hasattr(_request(tmp_path), "email_sender"), (
        "the sender/recipient/prefix overrides left BootstrapRequest"
    )


def test_undiscoverable_admin_email_stops_the_bootstrap(tmp_path: Path) -> None:
    state = _state(tmp_path)
    runner = Runner(
        [BootstrapError("organizations denied"), BootstrapError("account denied")]
    )

    with pytest.raises(BootstrapError, match="--admin-email"):
        notification_routing(
            runner, _cpu(), _request(tmp_path, alert_email=None), state
        )

    assert "admin_email_source" not in state.value["resources"], (
        "a failed lookup must not leave a routing source behind"
    )


def test_notification_tasks_send_as_the_routing_sender_and_alert_the_admin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, dict[str, Any]] = {}

    def spy(name: str):
        def record(runner: Any, **kwargs: Any) -> dict[str, str]:
            calls[name] = kwargs
            return {"name": name}

        return record

    for target in (
        "ensure_control_plane_role",
        "ensure_email_notifications",
        "ensure_monitoring_resources",
    ):
        monkeypatch.setattr(admin_notification_bootstrap, target, spy(target))

    state = _state(tmp_path)
    routing = NotificationRouting(
        sender="sender@example.com",
        recipients=("oncall@example.com",),
        subject_prefix="[PROD]",
        channel="ses",
    )
    tasks = notification_bootstrap_tasks(
        Runner(),
        state=state,
        cpu=_cpu(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        admin_email="ops@example.com",
        routing=routing,
    )

    assert sorted(tasks) == [
        "control_plane_role",
        "email_notifications",
        "monitoring_resources",
    ]
    assert calls == {}, "building the task table must not run any of the tasks"

    for task in tasks.values():
        task()

    assert calls["ensure_control_plane_role"]["email_sender"] == "sender@example.com"
    assert calls["ensure_control_plane_role"]["sns_topic_arn"] is None, (
        "an SES site's role gets no Publish grant it would never use"
    )
    assert calls["ensure_email_notifications"]["routing"] == routing, (
        "the SES task receives the resolved routing whole"
    )
    assert calls["ensure_email_notifications"]["admin_email"] == "ops@example.com"
    # The alert address is the administrator's, not the sender identity: SES mail
    # comes from the sender, but a monitoring alarm has to reach a human.
    assert calls["ensure_monitoring_resources"]["alert_email"] == "ops@example.com"
    assert calls["ensure_monitoring_resources"]["state"] is state


@pytest.mark.parametrize(
    ("channel", "granted"), [("ses", "gpu-fault-alerts"), ("sns", None)]
)
def test_the_role_task_grants_the_declared_configuration_set_on_ses_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, channel: str, granted: str | None
) -> None:
    """The set the site declares reaches the role's policy on the SES channel;
    on SNS it is carried unused and the role holds no SES grant at all."""

    calls: dict[str, dict[str, Any]] = {}

    def spy(name: str):
        def record(runner: Any, **kwargs: Any) -> dict[str, str]:
            calls[name] = kwargs
            return {"name": name}

        return record

    for target in (
        "ensure_control_plane_role",
        "ensure_email_notifications",
        "ensure_monitoring_resources",
    ):
        monkeypatch.setattr(admin_notification_bootstrap, target, spy(target))
    routing = NotificationRouting(
        sender="alerts@example.com",
        recipients=("oncall@example.com",),
        subject_prefix="",
        channel=channel,
        ses_configuration_set="gpu-fault-alerts",
    )

    tasks = notification_bootstrap_tasks(
        Runner(),
        state=_state(tmp_path),
        cpu=_cpu(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        admin_email="ops@example.com",
        routing=routing,
    )
    tasks["control_plane_role"]()

    assert calls["ensure_control_plane_role"]["ses_configuration_set"] == granted


def test_sns_tasks_grant_publish_on_the_site_topic_and_no_ses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default channel: the role may publish to the one site topic, whose
    ARN is known before the monitoring task creates it, and no SES sender is
    granted or verified."""

    calls: dict[str, dict[str, Any]] = {}

    def spy(name: str):
        def record(runner: Any, **kwargs: Any) -> dict[str, str]:
            calls[name] = kwargs
            return {"name": name}

        return record

    for target in (
        "ensure_control_plane_role",
        "ensure_email_notifications",
        "ensure_monitoring_resources",
    ):
        monkeypatch.setattr(admin_notification_bootstrap, target, spy(target))
    routing = NotificationRouting(
        sender="ops@example.com", recipients=("ops@example.com",), subject_prefix=""
    )

    tasks = notification_bootstrap_tasks(
        Runner(),
        state=_state(tmp_path),
        cpu=_cpu(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        admin_email="ops@example.com",
        routing=routing,
    )
    for task in tasks.values():
        task()

    assert calls["ensure_control_plane_role"]["email_sender"] is None
    assert calls["ensure_control_plane_role"]["sns_topic_arn"] == (
        "arn:aws:sns:us-west-2:123456789012:gpu-fault-site-a-alerts"
    )
    assert calls["ensure_email_notifications"]["routing"].channel == "sns"
    assert calls["ensure_monitoring_resources"]["alert_email"] == "ops@example.com"


# --- a deploy that regenerates site.yaml keeps the notification contract -------

# What bootstrap writes for an SES site that declares nothing beyond the channel.
SES_BLOCK: dict[str, Any] = {
    "channel": "ses",
    "allowEmail": True,
    "acknowledgeExternalAlertChannel": False,
    "adminEmail": "ops@example.com",
    "emailSender": "ops@example.com",
    "emailRecipients": ["ops@example.com"],
    "emailSubjectPrefix": "",
}


def _regenerate(
    tmp_path: Path, declared: dict[str, Any], *, alert_email: str = "ops@example.com"
) -> tuple[NotificationRouting, dict[str, Any]]:
    """One deploy's notification path over a site that declares ``declared``.

    The routing is what the role, SES identity and Secret tasks run with; the
    block is what the regenerated site.yaml says. ``_site_document`` writes
    ``routing.site_notifications(admin_email)`` and ``finalize_bootstrap_site``
    merges the existing site over it, exactly as here.
    """

    existing = {"spec": {"notifications": deepcopy(declared), "clusters": []}}
    state = _state(tmp_path)
    admin_email, routing = notification_routing(
        Runner(), _cpu(), _request(tmp_path, alert_email=alert_email), state, existing
    )
    generated = {
        "spec": {
            "notifications": routing.site_notifications(admin_email),
            "clusters": [],
        }
    }
    written: list[dict[str, Any]] = []
    finalize_bootstrap_site(
        tmp_path / "site.yaml",
        generated,
        existing,
        [],
        state,
        lambda _path, document: written.append(document),
    )
    return routing, written[0]["spec"]["notifications"]


@pytest.mark.parametrize(
    "custom",
    [
        {"emailSender": "alerts@example.com"},
        {"emailRecipients": ["oncall@example.com", "ops@example.com"]},
        {"emailSubjectPrefix": "[PROD]"},
        {"sesConfigurationSet": "gpu-fault-alerts"},
        {"acknowledgeExternalAlertChannel": True},
        {"allowEmail": False, "acknowledgeExternalAlertChannel": True},
    ],
    ids=[
        "sender",
        "recipients",
        "subject-prefix",
        "configuration-set",
        "external-channel",
        "email-off",
    ],
)
def test_a_regenerated_site_keeps_each_declared_notification_field(
    tmp_path: Path, custom: dict[str, Any]
) -> None:
    """Security reference 5.1: sender, recipients and administrator address are
    independent, and the subject prefix and configuration set are site
    settings. A deploy used to rebuild the block from the administrator address
    and keep only ``channel``, so none of these could take effect."""

    declared = {**SES_BLOCK, **custom}

    routing, written = _regenerate(tmp_path, declared)

    assert written == declared, "the regenerated site.yaml lost a declared field"
    config = NotificationSiteConfig.from_value(written)
    assert routing == NotificationRouting(
        sender=str(config.email_sender),
        recipients=config.email_recipients,
        subject_prefix=config.email_subject_prefix,
        channel="ses",
        ses_configuration_set=config.ses_configuration_set,
    ), "the notification tasks must act on the same contract the site declares"


def test_a_generated_sns_site_regenerates_unchanged(tmp_path: Path) -> None:
    """The default site's block -- what every first deploy writes -- comes back
    field for field, so a deploy of such a site moves no notification setting."""

    generated = NotificationRouting(
        sender="ops@example.com", recipients=("ops@example.com",), subject_prefix=""
    ).site_notifications("ops@example.com")

    routing, written = _regenerate(tmp_path, generated)

    assert list(written.items()) == list(generated.items()), (
        "same fields, same values, same order"
    )
    assert routing == NotificationRouting(
        sender="ops@example.com", recipients=("ops@example.com",), subject_prefix=""
    )


def test_an_sns_site_keeps_its_prefix_and_the_ses_fields_it_carries(
    tmp_path: Path,
) -> None:
    """A site flipped back to ``channel: sns`` carries its SES sender and
    recipients unused, so flipping it to ``ses`` again is one edit."""

    declared = {
        "channel": "sns",
        "allowEmail": True,
        "acknowledgeExternalAlertChannel": False,
        "adminEmail": "ops@example.com",
        "emailSender": "alerts@example.com",
        "emailRecipients": ["oncall@example.com"],
        "emailSubjectPrefix": "[PROD]",
        "sesConfigurationSet": "gpu-fault-alerts",
    }

    routing, written = _regenerate(tmp_path, declared)

    assert written == declared
    assert routing.channel == "sns" and routing.uses_ses is False
    assert routing.subject_prefix == "[PROD]", "the Secret carries the declared prefix"


def test_the_command_admin_email_wins_and_moves_only_the_addresses_derived_from_it(
    tmp_path: Path,
) -> None:
    """``--admin-email`` replaces ``adminEmail``. A sender or recipient list that
    is still the address it replaces is what bootstrap derived from it and
    follows it; an address the operator chose stays."""

    derived = {**SES_BLOCK, "emailSubjectPrefix": "[PROD]"}
    routing, written = _regenerate(tmp_path, derived, alert_email="new@example.com")

    assert written == {
        **derived,
        "adminEmail": "new@example.com",
        "emailSender": "new@example.com",
        "emailRecipients": ["new@example.com"],
    }
    assert routing.sender == "new@example.com"
    assert routing.recipients == ("new@example.com",)

    chosen = {
        **SES_BLOCK,
        "emailSender": "alerts@example.com",
        "emailRecipients": ["ops@example.com", "oncall@example.com"],
    }
    routing, written = _regenerate(tmp_path, chosen, alert_email="new@example.com")

    assert written == {**chosen, "adminEmail": "new@example.com"}, (
        "an operator-chosen sender or recipient list must survive a new admin email"
    )
    assert routing.sender == "alerts@example.com"
    assert routing.recipients == ("ops@example.com", "oncall@example.com")


def test_the_regenerated_ses_block_fills_only_what_the_site_leaves_out(
    tmp_path: Path,
) -> None:
    """The documented switch is one edit -- ``channel: ses`` -- on the block an
    SNS site was generated with; the sender and recipient it lacks are the
    administrator address."""

    sns_generated = {
        "channel": "ses",
        "allowEmail": True,
        "acknowledgeExternalAlertChannel": False,
        "adminEmail": "ops@example.com",
        "emailSubjectPrefix": "[PROD]",
    }

    routing, written = _regenerate(tmp_path, sns_generated)

    assert written == {
        **sns_generated,
        "emailSender": "ops@example.com",
        "emailRecipients": ["ops@example.com"],
    }
    assert routing.sender == "ops@example.com" and routing.uses_ses


@pytest.mark.parametrize(
    ("custom", "message"),
    [
        ({"emailSender": "not-an-address"}, "emailSender must be a valid email"),
        ({"emailSenders": ["a@example.com"]}, "unknown fields: emailSenders"),
        ({"channel": "email"}, "channel must be one of"),
        ({"allowEmail": "yes"}, "allowEmail must be a boolean"),
        ({"emailSubjectPrefix": "x" * 65}, "emailSubjectPrefix must be a single line"),
    ],
)
def test_an_invalid_declared_notification_field_stops_before_any_task(
    tmp_path: Path, custom: dict[str, Any], message: str
) -> None:
    """Kept verbatim, a bad value would only be refused by the release after
    the whole bootstrap ran; it stops the run before the first task instead,
    where the generator used to drop it silently."""

    state = _state(tmp_path)
    existing = {"spec": {"notifications": {**SES_BLOCK, **custom}}}

    with pytest.raises(BootstrapError, match=message):
        notification_routing(Runner(), _cpu(), _request(tmp_path), state, existing)

    assert "notification_channel" not in state.value["resources"], (
        "a refused contract must not leave a routing behind"
    )


# --- what the command's administrator address is for ---------------------------


def test_admin_email_help_says_what_the_address_is_used_for(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The address is the SNS subscription (AMP alerts, and the control plane's
    own notifications on the default channel), ``spec.notifications.adminEmail``
    and the Grafana ADMIN lookup. SES sends from ``emailSender``: the address is
    the SES sender and recipient only where the site declares none."""

    with pytest.raises(SystemExit, match="0"):
        admin_cli.parser().parse_args(["deploy", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    start = help_text.index("--admin-email EMAIL", help_text.index("options:"))
    entry = help_text[start : help_text.index("--rollback", start)]

    assert "SES fault notifications" not in entry, entry
    for phrase in (
        "SNS",
        "spec.notifications.adminEmail",
        "Grafana ADMIN",
        "emailSender",
        "required on the first deploy",
    ):
        assert phrase in entry, f"--admin-email help does not mention {phrase}: {entry}"
