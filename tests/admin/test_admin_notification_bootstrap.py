"""Which address the bootstrap sends mail from, and which one it alerts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import notification_bootstrap as admin_notification_bootstrap
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
)
from gpu_fault.admin.notification_bootstrap import (
    notification_bootstrap_tasks,
    notification_routing,
)
from gpu_fault.admin.notifications import NotificationRouting


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
    assert state.value["resources"]["admin_email_source"] == "configured"
    assert runner.commands == [], "a configured address must not be looked up in AWS"


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
    assert calls["ensure_email_notifications"]["routing"] == routing, (
        "the SES task receives the resolved routing whole"
    )
    assert calls["ensure_email_notifications"]["admin_email"] == "ops@example.com"
    # The alert address is the administrator's, not the sender identity: SES mail
    # comes from the sender, but a monitoring alarm has to reach a human.
    assert calls["ensure_monitoring_resources"]["alert_email"] == "ops@example.com"
    assert calls["ensure_monitoring_resources"]["state"] is state
