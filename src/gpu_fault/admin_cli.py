from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from gpu_fault.admin_bootstrap import bootstrap_from_arns, discover_cluster
from gpu_fault.admin_bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    BootstrapState,
    CommandRunner,
)
from gpu_fault.admin_bootstrap_services import ensure_control_plane_role
from gpu_fault.admin_cluster_join import JoinClusterRequest, join_cluster
from gpu_fault.admin_cluster_removal import (
    RemoveClusterRequest,
    remove_cluster,
)
from gpu_fault.admin_legacy_site import LegacySiteRequest, discover_legacy_site
from gpu_fault.admin_notifications import (
    ensure_email_notifications,
    resolve_admin_email,
    validate_admin_email,
)
from gpu_fault.admin_resource_registry import sync_installation_resource_registry
from gpu_fault.admin_site import (
    RenderedSite,
    SiteConfigError,
    effective_environment,
    load_site,
    materialized_release_config,
)
from gpu_fault.admin_uninstall import UninstallRequest, uninstall

COMMANDS = {
    "preflight": "preflight",
    "deploy": "deploy",
    "verify": "verify",
    "status": "status",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="gpu-fault-admin",
        description="Regional GPU fault deployment and health administration",
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("preflight", "verify", "status"):
        command = commands.add_parser(name)
        command.add_argument(
            "-f",
            "--file",
            required=True,
            type=Path,
            help="RegionalSite YAML",
        )
        command.add_argument(
            "--repo-root",
            type=Path,
            help="override spec.repositoryRoot",
        )
        command.add_argument(
            "--show-effective-config",
            action="store_true",
            help="print the redacted generated release configuration",
        )
    deploy = commands.add_parser("deploy")
    deploy.add_argument("-f", "--file", type=Path, help="existing RegionalSite YAML")
    deploy.add_argument(
        "--cpu-cluster-arn",
        help="existing CPU EKS or HyperPod cluster ARN",
    )
    deploy.add_argument(
        "--gpu-cluster-arn",
        action="append",
        default=[],
        help="existing GPU EKS or HyperPod cluster ARN; repeat for more clusters",
    )
    deploy.add_argument("--repo-root", type=Path)
    deploy.add_argument("--state-dir", type=Path)
    deploy.add_argument(
        "--allow-legacy-python-foundation",
        action="store_true",
        help=(
            "allow the deprecated ARN-only Python AWS foundation bootstrap; "
            "new sites must provision AWS resources through IaC"
        ),
    )
    deploy.add_argument(
        "--admin-email",
        "--alert-email",
        dest="alert_email",
        help=(
            "administrator email for SES fault notifications and SNS alerts; "
            "defaults to the AWS account email when discoverable"
        ),
    )
    deploy.add_argument(
        "--email-sender",
        help="verified SES sender; defaults to the administrator email",
    )
    deploy.add_argument(
        "--email-recipient",
        action="append",
        default=[],
        help="notification recipient; repeat for multiple recipients",
    )
    deploy.add_argument(
        "--email-subject-prefix",
        help="optional site-specific prefix prepended to every email subject",
    )
    deploy.add_argument("--show-effective-config", action="store_true")
    join = commands.add_parser(
        "join-cluster",
        help="discover and attach one existing GPU EKS/HyperPod cluster",
    )
    join.add_argument("-f", "--file", required=True, type=Path)
    join.add_argument("--gpu-cluster-arn", required=True)
    join.add_argument("--cluster-id")
    join.add_argument(
        "--allowed-namespace",
        action="append",
        default=[],
        help="additional workload namespace; may be repeated",
    )
    join.add_argument("--state-dir", type=Path)
    join.add_argument("--repo-root", type=Path)
    detach = commands.add_parser(
        "remove-cluster",
        help="uninstall one managed GPU data plane and keep the CPU control plane",
    )
    detach.add_argument("-f", "--file", required=True, type=Path)
    detach.add_argument("--cluster-id", required=True)
    detach.add_argument(
        "--confirm",
        required=True,
        help="must be REMOVE_GPU_CLUSTER",
    )
    detach.add_argument("--repo-root", type=Path)
    remove = commands.add_parser("uninstall")
    remove.add_argument("-f", "--file", type=Path)
    remove.add_argument(
        "--cpu-cluster-arn",
        help="existing online CPU EKS or HyperPod cluster ARN",
    )
    remove.add_argument(
        "--gpu-cluster-arn",
        action="append",
        default=[],
        help="existing online GPU EKS or HyperPod cluster ARN; repeat as needed",
    )
    remove.add_argument("--state-dir", type=Path)
    remove.add_argument(
        "--cpu-cluster",
        choices=("keep", "delete"),
        default="keep",
    )
    remove.add_argument("--confirm", required=True)
    remove.add_argument(
        "--aurora-final-snapshot",
        choices=("retain", "skip"),
        default="retain",
        help="retain a final Aurora cluster snapshot, or skip it",
    )
    remove.add_argument("--repo-root", type=Path)
    remove.add_argument("--show-effective-config", action="store_true")
    return result


def _redacted_config(value: dict[str, Any]) -> dict[str, Any]:
    redacted = copy.deepcopy(value)
    for cluster in redacted.get("clusters", []):
        for key in ("token_file", "ca_file", "fleet_master_file"):
            if key in cluster:
                cluster[key] = f"<{key}-path>"
    return redacted


def _configure_site_notifications(
    site: RenderedSite,
    *,
    configured_email: str | None,
    configured_sender: str | None,
    configured_recipients: tuple[str, ...],
    configured_subject_prefix: str | None,
) -> RenderedSite:
    notifications = dict(site.release_config.get("notifications") or {})
    requested = configured_email or notifications.get("admin_email")
    runner = CommandRunner()
    cpu = discover_cluster(
        runner,
        cluster_arn=str(site.release_config["cpu_eks_arn"]),
        role="cpu",
        context="gpu-fault-admin-cpu",
    )
    admin_email, _source = resolve_admin_email(
        runner,
        account_id=cpu.account_id,
        configured=str(requested) if requested else None,
    )
    sender = validate_admin_email(
        configured_sender or str(notifications.get("email_sender") or admin_email)
    )
    recipients = tuple(
        dict.fromkeys(
            validate_admin_email(item)
            for item in (
                configured_recipients
                or tuple(notifications.get("email_recipients") or ())
                or (admin_email,)
            )
        )
    )
    subject_prefix = (
        configured_subject_prefix
        if configured_subject_prefix is not None
        else str(notifications.get("email_subject_prefix") or "")
    ).strip()
    email = ensure_email_notifications(
        runner,
        cpu=cpu,
        cpu_kubeconfig=Path(site.release_config["cpu_kubeconfig"]),
        namespace=str(site.release_config["namespace"]),
        site_id=str(site.release_config["site_name"]),
        admin_email=admin_email,
        sender_email=sender,
        recipients=recipients,
        subject_prefix=subject_prefix,
    )
    role = ensure_control_plane_role(
        runner,
        cpu=cpu,
        cpu_kubeconfig=Path(site.release_config["cpu_kubeconfig"]),
        namespace=str(site.release_config["namespace"]),
        site_id=str(site.release_config["site_name"]),
        email_sender=str(email["sender_email"]),
    )
    state_path = site.source.parent / "bootstrap-state.json"
    if state_path.is_file():
        state = BootstrapState(
            state_path,
            site_id=str(site.release_config["site_name"]),
        )
        for name, value in (
            ("email_notifications", email),
            ("control_plane_role", role),
        ):
            state.record(name, value)
            state.complete(name)
    document = yaml.safe_load(site.source.read_text(encoding="utf-8"))
    desired = {
        "allowEmail": True,
        "acknowledgeExternalAlertChannel": False,
        "adminEmail": admin_email,
        "emailSender": str(email["sender_email"]),
        "emailRecipients": list(email["email_recipients"]),
        "emailSubjectPrefix": str(email["email_subject_prefix"]),
    }
    if document["spec"].get("notifications") != desired:
        document["spec"]["notifications"] = desired
        temporary = site.source.with_suffix(site.source.suffix + ".tmp")
        temporary.write_text(
            yaml.safe_dump(document, sort_keys=False),
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(site.source)
    return load_site(site.source, repository_root=site.repository_root)


def run(arguments: argparse.Namespace) -> int:
    if arguments.command == "join-cluster":
        site = load_site(
            arguments.file,
            repository_root=arguments.repo_root,
        )
        result = join_cluster(
            JoinClusterRequest(
                site=site,
                gpu_cluster_arn=arguments.gpu_cluster_arn,
                cluster_id=arguments.cluster_id,
                allowed_namespaces=tuple(arguments.allowed_namespace),
                state_dir=arguments.state_dir,
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "remove-cluster":
        site = load_site(
            arguments.file,
            repository_root=arguments.repo_root,
        )
        result = remove_cluster(
            RemoveClusterRequest(
                site=site,
                cluster_id=arguments.cluster_id,
                confirmation=arguments.confirm,
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "uninstall":
        if arguments.file is not None:
            site = load_site(
                arguments.file,
                repository_root=arguments.repo_root,
            )
        else:
            if not arguments.cpu_cluster_arn or not arguments.gpu_cluster_arn:
                raise SiteConfigError(
                    "uninstall requires -f site.yaml, or --cpu-cluster-arn "
                    "with at least one --gpu-cluster-arn"
                )
            repository_root = (arguments.repo_root or Path.cwd()).resolve()
            identity = "|".join([arguments.cpu_cluster_arn, *arguments.gpu_cluster_arn])
            default_state = (
                Path.home()
                / ".gpu-fault/legacy-uninstall"
                / hashlib.sha256(identity.encode()).hexdigest()[:12]
            )
            site = discover_legacy_site(
                LegacySiteRequest(
                    cpu_cluster_arn=arguments.cpu_cluster_arn,
                    gpu_cluster_arns=tuple(arguments.gpu_cluster_arn),
                    repository_root=repository_root,
                    state_dir=(arguments.state_dir or default_state).expanduser(),
                )
            )
        uninstall_result = uninstall(
            UninstallRequest(
                site=site,
                cpu_disposition=arguments.cpu_cluster,
                confirmation=arguments.confirm,
                final_snapshot_policy=arguments.aurora_final_snapshot,
            )
        )
        print(json.dumps(uninstall_result, indent=2, sort_keys=True))
        return 0
    site_file = arguments.file
    automatic = False
    if arguments.command == "deploy" and site_file is None:
        if not arguments.cpu_cluster_arn or not arguments.gpu_cluster_arn:
            raise SiteConfigError(
                "deploy requires -f site.yaml, or --cpu-cluster-arn "
                "with at least one --gpu-cluster-arn"
            )
        if not getattr(arguments, "allow_legacy_python_foundation", False):
            raise SiteConfigError(
                "ARN-only Python foundation bootstrap is disabled by default; "
                "provision deploy/aws/regional-foundation with Terraform and "
                "deploy its site.yaml, or explicitly set "
                "--allow-legacy-python-foundation for migration"
            )
        repository_root = (arguments.repo_root or Path.cwd()).resolve()
        identity = arguments.cpu_cluster_arn
        default_state = (
            Path.home()
            / ".gpu-fault/bootstrap"
            / hashlib.sha256(identity.encode()).hexdigest()[:12]
        )
        bootstrap_result = bootstrap_from_arns(
            BootstrapRequest(
                cpu_cluster_arn=arguments.cpu_cluster_arn,
                gpu_cluster_arns=tuple(arguments.gpu_cluster_arn),
                repository_root=repository_root,
                state_dir=(arguments.state_dir or default_state).expanduser(),
                alert_email=arguments.alert_email,
                email_sender=getattr(arguments, "email_sender", None),
                email_recipients=tuple(getattr(arguments, "email_recipient", ()) or ()),
                email_subject_prefix=(
                    getattr(arguments, "email_subject_prefix", None) or ""
                ),
            )
        )
        site_file = bootstrap_result.site_file
        automatic = True
    if site_file is None:
        raise SiteConfigError(f"{arguments.command} requires -f site.yaml")
    site = load_site(site_file, repository_root=arguments.repo_root)
    if arguments.command == "deploy":
        notification_override = any(
            (
                getattr(arguments, "alert_email", None),
                getattr(arguments, "email_sender", None),
                tuple(getattr(arguments, "email_recipient", ()) or ()),
                getattr(arguments, "email_subject_prefix", None),
            )
        )
        if notification_override and not automatic:
            raise SiteConfigError(
                "notification AWS changes must be applied through IaC and "
                "site.yaml before application deployment"
            )
    print(
        json.dumps(
            {"gpu_fault_admin": site.audit_summary},
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    if arguments.show_effective_config:
        print(
            json.dumps(
                _redacted_config(site.release_config),
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    rollout = (
        site.repository_root
        / "deploy/control-plane/regional/rollout-regional-release.sh"
    )
    with materialized_release_config(site) as config:
        environment = effective_environment(site)
        if arguments.command == "deploy":
            preflight = subprocess.run(
                [str(rollout), "preflight", "--config", str(config)],
                cwd=site.repository_root,
                env=environment,
                check=False,
            )
            if preflight.returncode:
                return preflight.returncode
        completed = subprocess.run(
            [
                str(rollout),
                COMMANDS[arguments.command],
                "--config",
                str(config),
            ],
            cwd=site.repository_root,
            env=environment,
            check=False,
        )
        if completed.returncode:
            return completed.returncode
        if arguments.command != "deploy":
            return 0
        sync_installation_resource_registry(site)
        if automatic:
            verification = subprocess.run(
                [str(rollout), "verify", "--config", str(config)],
                cwd=site.repository_root,
                env=environment,
                check=False,
            )
            return verification.returncode
        return 0


def main() -> int:
    arguments = parser().parse_args()
    try:
        return run(arguments)
    except (
        BootstrapError,
        OSError,
        SiteConfigError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"gpu-fault-admin: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
