from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from gpu_fault.admin_bootstrap_common import (
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin_bootstrap_services import (
    ensure_control_plane_role,
    ensure_monitoring_resources,
)
from gpu_fault.admin_notifications import (
    NotificationRouting,
    ensure_email_notifications,
    resolve_admin_email,
    resolve_notification_routing,
)


def notification_routing(
    runner: CommandRunner,
    cpu: ClusterIdentity,
    request: BootstrapRequest,
    state: BootstrapState,
) -> tuple[str, NotificationRouting]:
    admin_email, source = resolve_admin_email(
        runner,
        account_id=cpu.account_id,
        configured=request.alert_email,
    )
    routing = resolve_notification_routing(
        admin_email=admin_email,
        sender_email=request.email_sender,
        recipients=request.email_recipients,
        subject_prefix=request.email_subject_prefix,
    )
    state.record("admin_email_source", source)
    return admin_email, routing


def notification_bootstrap_tasks(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
    namespace: str,
    site_id: str,
    admin_email: str,
    routing: NotificationRouting,
) -> dict[str, Callable[[], Any]]:
    return {
        "control_plane_role": lambda: ensure_control_plane_role(
            runner,
            cpu=cpu,
            cpu_kubeconfig=cpu_kubeconfig,
            namespace=namespace,
            site_id=site_id,
            email_sender=routing.sender,
        ),
        "email_notifications": lambda: ensure_email_notifications(
            runner,
            cpu=cpu,
            cpu_kubeconfig=cpu_kubeconfig,
            namespace=namespace,
            site_id=site_id,
            admin_email=admin_email,
            sender_email=routing.sender,
            recipients=routing.recipients,
            subject_prefix=routing.subject_prefix,
        ),
        "monitoring_resources": lambda: ensure_monitoring_resources(
            runner,
            cpu=cpu,
            site_id=site_id,
            alert_email=admin_email,
        ),
    }
