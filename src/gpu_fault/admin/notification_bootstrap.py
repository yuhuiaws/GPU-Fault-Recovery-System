from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from gpu_fault.admin.bootstrap_common import (
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_services import (
    ensure_control_plane_role,
    ensure_monitoring_resources,
    site_sns_topic_arn,
)
from gpu_fault.admin.notifications import (
    NotificationRouting,
    ensure_email_notifications,
    resolve_admin_email,
    resolve_notification_routing,
)
from gpu_fault.admin.site import site_notification_channel


def notification_routing(
    runner: CommandRunner,
    cpu: ClusterIdentity,
    request: BootstrapRequest,
    state: BootstrapState,
    existing_site: Mapping[str, Any] | None = None,
) -> tuple[str, NotificationRouting]:
    """The administrator address and the channel it is reached over.

    The channel is the existing site's (``spec.notifications.channel``, or
    ``ses`` for a site written before the field existed and still carrying an
    ``emailSender``); a new site gets ``sns``. Flipping a live site is an edit
    of that one field followed by a ``deploy`` rerun, never a side effect of
    the rerun itself.
    """

    admin_email, source = resolve_admin_email(
        runner,
        account_id=cpu.account_id,
        configured=request.alert_email,
    )
    routing = resolve_notification_routing(
        admin_email=admin_email,
        channel=site_notification_channel(existing_site),
    )
    state.record("admin_email_source", source)
    state.record("notification_channel", routing.channel)
    return admin_email, routing


def notification_bootstrap_tasks(
    runner: CommandRunner,
    *,
    state: BootstrapState | None,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
    namespace: str,
    site_id: str,
    admin_email: str,
    routing: NotificationRouting,
    archive_s3_uri: str | None = None,
) -> dict[str, Callable[[], Any]]:
    # The role is granted exactly the channel the site uses: Publish on the
    # site topic for sns, SendEmail from the verified sender for ses.
    uses_ses = routing.uses_ses
    return {
        "control_plane_role": lambda: ensure_control_plane_role(
            runner,
            cpu=cpu,
            cpu_kubeconfig=cpu_kubeconfig,
            namespace=namespace,
            site_id=site_id,
            email_sender=routing.sender if uses_ses else None,
            sns_topic_arn=None if uses_ses else site_sns_topic_arn(cpu, site_id),
            archive_s3_uri=archive_s3_uri,
        ),
        "email_notifications": lambda: ensure_email_notifications(
            runner,
            cpu=cpu,
            cpu_kubeconfig=cpu_kubeconfig,
            namespace=namespace,
            site_id=site_id,
            admin_email=admin_email,
            routing=routing,
        ),
        "monitoring_resources": lambda: ensure_monitoring_resources(
            runner,
            state=state,
            cpu=cpu,
            site_id=site_id,
            alert_email=admin_email,
        ),
    }
