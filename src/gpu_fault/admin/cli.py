from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.aurora_capacity import (
    aurora_capacity_changed,
    reconcile_aurora_capacity,
)
from gpu_fault.admin.bootstrap import bootstrap_from_arns, discover_cluster
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    BootstrapState,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_services import ensure_control_plane_role
from gpu_fault.admin.cluster_batch_join import join_clusters
from gpu_fault.admin.cluster_join import (
    DEFAULT_ALLOWED_NAMESPACES,
    JoinClusterRequest,
    join_cluster,
)
from gpu_fault.admin.cluster_removal import (
    RemoveClusterRequest,
    remove_cluster,
)
from gpu_fault.admin.command_log import (
    ADMIN_LOG_KIND_MUTATING,
    ADMIN_LOG_KIND_READONLY,
    announce,
    command_log,
)
from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigError,
    AuroraCapacityConfig,
    admin_config_approval_path,
    admin_config_plan_path,
    complete_admin_config_apply,
    create_admin_config_plan,
    load_admin_config_plan,
    load_desired_admin_config,
    prepare_admin_config_apply,
    preview_admin_config_plan,
)
from gpu_fault.admin.config_patch import apply_capacity_patch
from gpu_fault.admin.config_file import (
    admin_config_file_path,
    initialize_desired_admin_config,
    load_admin_config_file,
    write_admin_config_file,
)
from gpu_fault.admin.legacy_site import LegacySiteRequest, discover_legacy_site
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.notifications import (
    ensure_email_notifications,
    resolve_admin_email,
    validate_admin_email,
)
from gpu_fault.admin.operation_lock import inherited_lock_pass_fds
from gpu_fault.admin.profile_approval import approve_profile
from gpu_fault.admin.release_artifacts import verify_prebuilt_release
from gpu_fault.admin.release_state import live_release_state as _live_release_state
from gpu_fault.admin.resource_registry import sync_installation_resource_registry
from gpu_fault.admin.rollback_alignment import materialized_rollback_status_site
from gpu_fault.admin.site import (
    RenderedSite,
    SiteConfigError,
    effective_environment,
    load_site,
    materialized_release_config,
)
from gpu_fault.admin.failure_domain_map import (
    add_failure_domain_map_command,
    run_failure_domain_map_command,
)
from gpu_fault.admin.grafana import (
    add_grafana_arguments,
    grafana_environment,
    grafana_request_fields,
)
from gpu_fault.admin.source_deploy import run_source_deploy
from gpu_fault.admin.uninstall import UninstallRequest, uninstall
from gpu_fault.models import BlockedKind
from gpu_fault.admin.workflow_reconcile import (
    RECONCILE_MODES,
    run_workflow_reconcile_mode,
)

COMMANDS = {
    "preflight": "preflight",
    "deploy": "deploy",
    "verify": "verify",
    "status": "status",
}
# The managed commands that write nothing; their console logs rotate under
# ``logs/readonly/`` so they cannot age out a failed deploy's log (I3).
READONLY_COMMANDS = frozenset({"preflight", "verify", "status"})
DEPLOY_HOST_STATE_BINDING = "gpu-fault-managed-state-dir.json"
QUICK_VALIDATION_EVIDENCE_ENV = "GPU_FAULT_QUICK_VALIDATION_EVIDENCE"
QUICK_VALIDATION_EVIDENCE_FILE = "quick-validation.json"


# Mirrors ``regional_schema_change.ACCEPT_SCHEMA_CHANGE_ENV`` and the two mode
# values; ``tests/regional/test_release_schema_change_acceptance.py`` pins them
# equal. The release engine reads the variable; the CLI only sets it.
ACCEPT_SCHEMA_CHANGE_ENV = "GPU_FAULT_RELEASE_ACCEPT_SCHEMA_CHANGE"
SCHEMA_CHANGE_SNAPSHOT_MODE = "snapshot"
SCHEMA_CHANGE_NO_SNAPSHOT_MODE = "no-snapshot"
# Mirrors ``regional_admin_commands.SUPERSEDE_FAILED_TRANSACTION_ENV``; the same
# test pins them equal. Set by ``--supersede-failed-transaction``, read by the
# release engine's deploy entrypoint.
SUPERSEDE_FAILED_TRANSACTION_ENV = "GPU_FAULT_RELEASE_SUPERSEDE_FAILED_TRANSACTION"


RELEASE_HISTORY_DIR_ENV = "GPU_FAULT_RELEASE_HISTORY_DIR"


def release_history_environment(arguments: argparse.Namespace) -> dict[str, str]:
    """Where the release engine mirrors its append-only audit history.

    The history ConfigMap lives in the namespace a `remove`/uninstall deletes,
    so the engine also appends each entry under the admin state directory --
    the one place that outlives the cluster and already holds the command logs.
    """

    state_dir = getattr(arguments, "state_dir", None)
    if not isinstance(state_dir, Path):
        return {}
    return {RELEASE_HISTORY_DIR_ENV: str(state_dir.expanduser().resolve() / "history")}


def schema_change_environment(arguments: argparse.Namespace) -> dict[str, str]:
    """The operator's consent to a schema-version release, for the release engine.

    A release that changes the PostgreSQL schema cannot be rolled back (the new
    wheel requires the exact version), so the engine refuses it under
    ``autoRollback: true``. ``--accept-schema-change`` says once, on the command,
    that the operator knows; the engine then takes an Aurora snapshot before the
    schema Jobs and runs this one transaction fail-forward without touching
    ``site.yaml``. ``--accept-schema-change-without-snapshot`` is the same
    consent for a database nobody would restore. An explicit environment value
    wins, as with the other release-engine variables. The engine ignores the
    variable when the release does not change the schema, so passing the flag on
    an ordinary release is harmless.
    """

    if os.environ.get(ACCEPT_SCHEMA_CHANGE_ENV, "").strip():
        return {}
    if getattr(arguments, "accept_schema_change_without_snapshot", False):
        return {ACCEPT_SCHEMA_CHANGE_ENV: SCHEMA_CHANGE_NO_SNAPSHOT_MODE}
    if getattr(arguments, "accept_schema_change", False):
        return {ACCEPT_SCHEMA_CHANGE_ENV: SCHEMA_CHANGE_SNAPSHOT_MODE}
    return {}


def supersede_environment(arguments: argparse.Namespace) -> dict[str, str]:
    """The operator's consent to replace a failed fail-forward transaction.

    A transaction that stopped in ``failed``/``partial-convergence`` only
    resumes the release that failed; a deploy of a *different* candidate (the
    fix) is refused with this flag named. ``--supersede-failed-transaction``
    tells the engine to open a new transaction for the candidate whose rollback
    baseline is the failed transaction's last committed release and whose diff
    re-rolls everything the failed release moved. It travels as one variable for
    the same reason the schema-change acceptance does; the engine refuses it
    when the recorded transaction is not such a failure, so it cannot be left on
    by habit.
    """

    if os.environ.get(SUPERSEDE_FAILED_TRANSACTION_ENV, "").strip():
        return {}
    if getattr(arguments, "supersede_failed_transaction", False):
        return {SUPERSEDE_FAILED_TRANSACTION_ENV: "1"}
    return {}


def quick_validation_evidence_environment(
    arguments: argparse.Namespace,
) -> dict[str, str]:
    """Name the evidence file a deploy writes and a `status` may reuse.

    The read-only verifiers -- the control-plane role split, and one data-plane
    executor probe per GPU cluster -- are the slowest part of a health report, and
    a deploy has just run them. Until now only the wrapper scripts knew where the
    evidence went, so an administrator driving the admin command directly re-ran
    every one of those probes on the next `status`, seconds after the deploy that
    proved them.

    This decides the path; it grants nothing. Reuse still has to get past
    `quick_validation_evidence`, which requires the release id, the delivery
    digest, the site identity and the live release state digest to match and the
    evidence to be under ten minutes old, and reports why it declined otherwise.
    An explicit environment setting always wins, and `verify` is untouched: an
    administrator who asks for the gate gets the probes.
    """

    if os.environ.get(QUICK_VALIDATION_EVIDENCE_ENV, "").strip():
        return {}
    command = getattr(arguments, "command", None)
    if command not in ("deploy", "status"):
        return {}
    state_dir = cast(Path | None, getattr(arguments, "state_dir", None))
    if state_dir is None:
        return {}
    path = state_dir.expanduser().resolve() / QUICK_VALIDATION_EVIDENCE_FILE
    if command == "deploy":
        # The writer replaces this at the end of quick validation. Removing it
        # first means a deploy that fails before then leaves no evidence behind
        # rather than evidence describing the release it replaced.
        path.unlink(missing_ok=True)
        return {QUICK_VALIDATION_EVIDENCE_ENV: str(path)}
    if not path.is_file():
        return {}
    return {QUICK_VALIDATION_EVIDENCE_ENV: str(path)}


def _bound_deploy_host_state_dir(prefix: Path | None = None) -> Path | None:
    binding = (prefix or Path(sys.prefix)).resolve() / DEPLOY_HOST_STATE_BINDING
    if not binding.is_file():
        return None
    try:
        value = json.loads(binding.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SiteConfigError("deploy-host state-dir binding is invalid") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise SiteConfigError("deploy-host state-dir binding schema is invalid")
    state_dir = value.get("state_dir")
    if not isinstance(state_dir, str) or not state_dir.strip():
        raise SiteConfigError("deploy-host state-dir binding has no state directory")
    return Path(state_dir).expanduser().resolve()


def enforce_deploy_host_state_dir(arguments: argparse.Namespace) -> None:
    bound = _bound_deploy_host_state_dir()
    if bound is None:
        return
    provided = cast(Path | None, getattr(arguments, "state_dir", None))
    if provided is not None:
        if provided.expanduser().resolve() == bound:
            return
        raise SiteConfigError(
            f"installed deploy-host is bound to --state-dir {bound}; "
            f"refusing {provided.expanduser().resolve()}"
        )
    explicit = cast(Path | None, getattr(arguments, "file", None))
    if explicit is not None and explicit.expanduser().resolve() == bound / "site.yaml":
        return
    raise SiteConfigError(
        f"installed deploy-host is bound to --state-dir {bound}; "
        f"{arguments.command} requires that managed state"
    )


def _add_managed_site_arguments(
    command: argparse.ArgumentParser,
    *,
    show_effective_config: bool = False,
) -> None:
    command.add_argument(
        "--state-dir",
        type=Path,
        metavar="STATE_DIR",
        help="private state directory containing the managed site",
    )
    command.add_argument(
        "-f",
        "--file",
        type=Path,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--repo-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    if show_effective_config:
        command.add_argument(
            "--show-effective-config",
            action="store_true",
            help="print the redacted generated release configuration",
        )


def _add_schema_change_arguments(deploy: argparse.ArgumentParser) -> None:
    deploy.add_argument(
        "--accept-schema-change",
        action="store_true",
        help=(
            "this release changes the PostgreSQL schema and cannot be rolled "
            "back: take an Aurora snapshot first and run this one transaction "
            "fail-forward without editing spec.autoRollback"
        ),
    )
    deploy.add_argument(
        "--accept-schema-change-without-snapshot",
        action="store_true",
        help=(
            "like --accept-schema-change but without the Aurora snapshot; only "
            "for a database nobody would restore"
        ),
    )
    deploy.add_argument(
        "--supersede-failed-transaction",
        action="store_true",
        help=(
            "the recorded transaction is a fail-forward release that stopped in "
            "failed/partial-convergence and this candidate is a different "
            "release: open a new transaction for it on the last committed "
            "baseline instead of resuming the failed one; refused in any other "
            "state"
        ),
    )


def _add_capacity_options(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--preset",
        choices=(
            "default",
            "32-disabled",
            "32-enabled",
            "50-disabled",
            "50-enabled",
        ),
    )
    command.add_argument("--control-worker-replicas", type=int)
    command.add_argument(
        "--spool",
        choices=("enabled", "disabled"),
        help="enable or disable telemetry spool admission",
    )
    command.add_argument("--spool-replicas", type=int)
    command.add_argument("--max-active-region", type=int)
    command.add_argument("--max-active-per-cluster", type=int)
    command.add_argument("--max-active-per-resource-class", type=int)
    command.add_argument("--largest-cluster-node-count", type=int)
    command.add_argument("--managed-node-count", type=int)


def _add_profile_approval_command(commands: Any) -> None:
    approve = commands.add_parser(
        "approve-profile",
        usage=(
            "gpu-fault-admin approve-profile --state-dir STATE_DIR "
            "--plan-sha256 SHA256 --reference REFERENCE"
        ),
        help="approve the pending Runtime Profile plan recorded in private state",
    )
    approve.add_argument(
        "--state-dir",
        required=True,
        type=Path,
        metavar="STATE_DIR",
        help="private state directory containing profile-plan.json",
    )
    approve.add_argument(
        "--plan-sha256",
        required=True,
        metavar="SHA256",
        help="exact plan_sha256 recorded from the reviewed profile-plan.json",
    )
    approve.add_argument(
        "--reference",
        required=True,
        metavar="REFERENCE",
        help="approved change or maintenance-window reference",
    )


def _add_admin_config_command(commands: Any) -> None:
    config = commands.add_parser(
        "config",
        usage=(
            "gpu-fault-admin config --state-dir STATE_DIR "
            "[--file ADMIN_CONFIG | --preset PRESET] --reference REFERENCE"
        ),
        help="validate and apply an audited administrator configuration",
    )
    config.add_argument(
        "--state-dir",
        required=True,
        type=Path,
        metavar="STATE_DIR",
    )
    config.add_argument(
        "--file",
        dest="admin_config_file",
        type=Path,
        metavar="ADMIN_CONFIG",
        help=("private AdminConfig YAML; defaults to <state-dir>/admin-config.yaml"),
    )
    _add_capacity_options(config)
    config.add_argument(
        "--reference",
        required=True,
        metavar="REFERENCE",
        help="approved change or maintenance-window reference",
    )
    config.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the internal change plan without applying it",
    )


def _add_workflow_reconcile_command(commands: Any) -> None:
    reconcile = commands.add_parser(
        "workflow-reconcile",
        usage=(
            "gpu-fault-admin workflow-reconcile --state-dir STATE_DIR "
            f"[--mode {{{','.join(RECONCILE_MODES)}}}] "
            "(--plan | --apply --plan-sha256 SHA256 --reference REFERENCE)"
        ),
        help="plan or apply an audited reconciliation of stuck workflow records",
    )
    _add_managed_site_arguments(reconcile)
    reconcile_mode = reconcile.add_mutually_exclusive_group(required=True)
    reconcile_mode.add_argument("--plan", action="store_true")
    reconcile_mode.add_argument("--apply", action="store_true")
    reconcile.add_argument(
        "--mode",
        choices=(
            "restore",
            "retired-generation",
            "compile-blocked",
            "orphaned-commands",
        ),
        default="restore",
    )
    reconcile.add_argument("--workflow-id", action="append", default=[])
    # Batch selectors for ``--mode restore --plan`` discovery: review one
    # incident's BLOCKED records, or one kind of BLOCKED, and cap the batch.
    reconcile.add_argument("--incident-id", action="append", default=[])
    reconcile.add_argument(
        "--blocked-kind",
        action="append",
        default=[],
        choices=[kind.value for kind in BlockedKind],
    )
    reconcile.add_argument("--max-items", type=int, default=None)
    reconcile.add_argument("--plan-sha256")
    reconcile.add_argument("--reference")


def _managed_site_file(
    arguments: argparse.Namespace,
    *,
    command: str,
    allow_missing: bool = False,
) -> Path | None:
    explicit = cast(Path | None, getattr(arguments, "file", None))
    if explicit is not None:
        return explicit
    state_dir = cast(Path | None, getattr(arguments, "state_dir", None))
    if state_dir is None:
        if allow_missing:
            return None
        raise SiteConfigError(f"{command} requires --state-dir")
    site_file = state_dir.expanduser().resolve() / "site.yaml"
    if site_file.is_file():
        return site_file
    if allow_missing:
        return None
    raise SiteConfigError(f"{command} found no managed site under --state-dir")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="gpu-fault-admin",
        description="Regional GPU fault deployment and health administration",
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("preflight", "verify", "status"):
        command = commands.add_parser(
            name,
            usage=f"gpu-fault-admin {name} --state-dir STATE_DIR",
        )
        _add_managed_site_arguments(
            command,
            show_effective_config=True,
        )
    deploy = commands.add_parser(
        "deploy",
        usage=(
            "gpu-fault-admin deploy --cpu-cluster-arn CPU_ARN "
            "--gpu-cluster-arn GPU_ARN --state-dir STATE_DIR "
            "--admin-email EMAIL"
        ),
        description=(
            "Bootstrap a new site or upgrade the existing site recorded under "
            "STATE_DIR using the same command"
        ),
    )
    deploy.add_argument("-f", "--file", type=Path, help=argparse.SUPPRESS)
    deploy.add_argument(
        "--cpu-cluster-arn",
        metavar="CPU_ARN",
        help="existing CPU EKS or HyperPod cluster ARN",
    )
    deploy.add_argument(
        "--gpu-cluster-arn",
        action="append",
        default=[],
        metavar="GPU_ARN",
        help="existing GPU EKS or HyperPod cluster ARN; repeat for more clusters",
    )
    deploy.add_argument("--repo-root", type=Path, help=argparse.SUPPRESS)
    deploy.add_argument(
        "--state-dir",
        type=Path,
        metavar="STATE_DIR",
        help=("private state directory used for both first deployment and upgrades"),
    )
    deploy.add_argument(
        "--config",
        dest="admin_config_file",
        type=Path,
        metavar="ADMIN_CONFIG",
        help=(
            "private AdminConfig YAML for first deployment; existing sites "
            "must use gpu-fault-admin config"
        ),
    )
    deploy.add_argument(
        "--allow-legacy-python-foundation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    deploy.add_argument(
        "--admin-email",
        dest="alert_email",
        metavar="EMAIL",
        help="administrator email for SES fault notifications and SNS alerts",
    )
    deploy.add_argument("--alert-email", dest="alert_email", help=argparse.SUPPRESS)
    add_grafana_arguments(deploy)
    deploy.add_argument(
        "--staging-only-release",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    _add_schema_change_arguments(deploy)
    deploy.add_argument("--impact-base", default="origin/main", help=argparse.SUPPRESS)
    deploy.add_argument(
        "--email-sender",
        help=argparse.SUPPRESS,
    )
    deploy.add_argument(
        "--email-recipient",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    deploy.add_argument(
        "--email-subject-prefix",
        help=argparse.SUPPRESS,
    )
    deploy.add_argument(
        "--prepared-source-release",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    deploy.add_argument(
        "--show-effective-config",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    _add_profile_approval_command(commands)
    _add_admin_config_command(commands)
    _add_workflow_reconcile_command(commands)
    add_failure_domain_map_command(commands, _add_managed_site_arguments)
    join = commands.add_parser(
        "join-cluster",
        usage=(
            "gpu-fault-admin join-cluster --state-dir STATE_DIR "
            "--gpu-cluster-arn GPU_ARN "
            "[--gpu-cluster-arn GPU_ARN ...]"
        ),
        help="discover and attach one or more existing GPU EKS/HyperPod clusters",
    )
    _add_managed_site_arguments(join)
    join.add_argument(
        "--gpu-cluster-arn",
        action="append",
        required=True,
        metavar="GPU_ARN",
    )
    join.add_argument("--cluster-id")
    join.add_argument(
        "--allowed-namespace",
        action="append",
        default=[],
        help="additional workload namespace; may be repeated",
    )
    detach = commands.add_parser(
        "remove-cluster",
        usage=(
            "gpu-fault-admin remove-cluster --state-dir STATE_DIR "
            "--cluster-id CLUSTER_ID --confirm REMOVE_GPU_CLUSTER"
        ),
        help="uninstall one managed GPU data plane and keep the CPU control plane",
    )
    _add_managed_site_arguments(detach)
    detach.add_argument("--cluster-id", required=True)
    detach.add_argument(
        "--confirm",
        required=True,
        help="must be REMOVE_GPU_CLUSTER",
    )
    remove = commands.add_parser(
        "uninstall",
        usage=(
            "gpu-fault-admin uninstall --state-dir STATE_DIR "
            "--cpu-cluster {keep,delete} --confirm CONFIRM"
        ),
    )
    _add_managed_site_arguments(remove)
    remove.add_argument(
        "--show-effective-config",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    remove.add_argument(
        "--cpu-cluster-arn",
        help=argparse.SUPPRESS,
    )
    remove.add_argument(
        "--gpu-cluster-arn",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
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


def _run_automatic_release(
    *,
    repository_root: Path,
    site_file: Path,
    state_dir: Path,
    staging_only_release: bool,
) -> int:
    command = [
        sys.executable,
        str(repository_root / "scripts/release_deploy.py"),
        "--site",
        str(site_file),
        "--prebuilt-attestation",
        str(repository_root / "dist/current-attestation.json"),
        "--prebuilt-bundle",
        str(repository_root / "dist/current-attestation.bundle.json"),
        "--cosign-key",
        str(state_dir / "release-signing/cosign.pub"),
    ]
    if staging_only_release:
        command.append("--allow-staging-release")
    completed = subprocess.run(
        command,
        cwd=repository_root,
        env={
            **os.environ,
            "PYTHONPATH": str(repository_root / "src"),
        },
        check=False,
        pass_fds=inherited_lock_pass_fds(),
    )
    if completed.returncode:
        return completed.returncode
    return 0


def _run_join_cluster(arguments: argparse.Namespace) -> int:
    site_file = _managed_site_file(arguments, command="join-cluster")
    assert site_file is not None
    site = load_site(
        site_file,
        repository_root=arguments.repo_root,
    )
    cluster_arns = tuple(str(value) for value in arguments.gpu_cluster_arn)
    if len(cluster_arns) > 1 and arguments.cluster_id:
        raise SiteConfigError(
            "join-cluster --cluster-id cannot be shared by multiple GPU clusters"
        )
    allowed_namespaces = (
        tuple(arguments.allowed_namespace) or DEFAULT_ALLOWED_NAMESPACES
    )
    requests = tuple(
        JoinClusterRequest(
            site=site,
            gpu_cluster_arn=cluster_arn,
            cluster_id=arguments.cluster_id,
            allowed_namespaces=allowed_namespaces,
        )
        for cluster_arn in cluster_arns
    )
    result = (
        join_cluster(requests[0]) if len(requests) == 1 else join_clusters(requests)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_remove_cluster(arguments: argparse.Namespace) -> int:
    site_file = _managed_site_file(arguments, command="remove-cluster")
    assert site_file is not None
    site = load_site(
        site_file,
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


def _run_uninstall(arguments: argparse.Namespace) -> int:
    site_file = _managed_site_file(
        arguments,
        command="uninstall",
        allow_missing=True,
    )
    if site_file is not None:
        site = load_site(
            site_file,
            repository_root=arguments.repo_root,
        )
    else:
        if not arguments.cpu_cluster_arn or not arguments.gpu_cluster_arn:
            raise SiteConfigError("uninstall found no managed site under --state-dir")
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


def _run_workflow_reconcile(arguments: argparse.Namespace) -> int:
    state_dir = arguments.state_dir.expanduser().resolve()
    site_file = _managed_site_file(arguments, command="workflow-reconcile")
    assert site_file is not None
    with administrator_operation_lock(state_dir):
        site = load_site(site_file, repository_root=arguments.repo_root)
        try:
            result = run_workflow_reconcile_mode(
                site,
                state_dir,
                mode=getattr(arguments, "mode", "restore"),
                plan=bool(arguments.plan),
                workflow_ids=tuple(arguments.workflow_id),
                incident_ids=tuple(getattr(arguments, "incident_id", ()) or ()),
                blocked_kinds=tuple(getattr(arguments, "blocked_kind", ()) or ()),
                max_items=getattr(arguments, "max_items", None),
                plan_sha256=arguments.plan_sha256,
                reference=arguments.reference,
            )
        except BootstrapError as exc:
            raise SiteConfigError(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    # A partial apply is reported in full and exits non-zero (see ``failures``).
    return 1 if result.get("failed_workflow_ids") else 0


def _run_failure_domain_map(arguments: argparse.Namespace) -> int:
    site_file = _managed_site_file(arguments, command="failure-domain-map")
    assert site_file is not None
    try:
        return run_failure_domain_map_command(
            arguments, site=load_site(site_file, repository_root=None)
        )
    except BootstrapError as exc:
        raise SiteConfigError(str(exc)) from exc


def _run_readonly_managed_command(
    arguments: argparse.Namespace,
    site_file: Path,
) -> int:
    current_site = load_site(
        site_file,
        repository_root=(
            None if arguments.command == "status" else arguments.repo_root
        ),
    )
    context: AbstractContextManager[Path] = nullcontext(site_file)
    if arguments.command == "status":
        try:
            live_state = _live_release_state(current_site)
        except SiteConfigError:
            live_state = {}
        else:
            rollback_result = live_state.get("rollback_result")
            if (
                str(live_state.get("phase") or "") == "rolled-back"
                and isinstance(rollback_result, dict)
                and rollback_result.get("status") == "PASSED"
            ):
                context = materialized_rollback_status_site(
                    site_file,
                    live_state,
                    management_repository_root=arguments.repo_root,
                )
    with context as effective_site_file:
        site = load_site(effective_site_file)
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
            completed = subprocess.run(
                [
                    str(rollout),
                    COMMANDS[arguments.command],
                    "--config",
                    str(config),
                ],
                cwd=site.repository_root,
                env={
                    **effective_environment(site),
                    **quick_validation_evidence_environment(arguments),
                },
                check=False,
            )
        return completed.returncode


def _run_profile_approval(arguments: argparse.Namespace) -> int:
    record = approve_profile(
        arguments.state_dir,
        reference=arguments.reference,
        expected_plan_sha256=arguments.plan_sha256,
    )
    print(
        json.dumps(
            {
                "status": "APPROVED",
                "site_identity": record["site_identity"],
                "site_identity_sha256": record["site_identity_sha256"],
                "desired_version": record["desired_version"],
                "change_kind": record["change_kind"],
                "plan_sha256": record["plan_sha256"],
                "reference": record["reference"],
                "approved_at": record["approved_at"],
                "approver_identity": record["approver_identity"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _admin_config_site_identity(site: RenderedSite) -> dict[str, str]:
    return {
        "site_name": str(site.release_config["site_name"]),
        "aws_region": str(site.release_config["aws_region"]),
        "cpu_eks_arn": str(site.release_config["cpu_eks_arn"]),
    }


def _capacity_candidate(
    arguments: argparse.Namespace,
    current: AdminConfig,
) -> AdminConfig:
    capacity: dict[str, Any] = {}
    if arguments.preset is not None:
        capacity["preset"] = arguments.preset
    if arguments.control_worker_replicas is not None:
        capacity["controlWorkerReplicas"] = arguments.control_worker_replicas
    spool: dict[str, Any] = {}
    if arguments.spool is not None:
        enabled = arguments.spool == "enabled"
        spool["enabled"] = enabled
        if arguments.spool_replicas is None:
            spool["replicas"] = 3 if enabled else 0
    if arguments.spool_replicas is not None:
        spool["replicas"] = arguments.spool_replicas
    if spool:
        capacity["telemetrySpool"] = spool
    remediation: dict[str, int] = {}
    for attribute, field in (
        ("max_active_region", "maxActiveRegion"),
        ("max_active_per_cluster", "maxActivePerCluster"),
        ("max_active_per_resource_class", "maxActivePerResourceClass"),
    ):
        value = cast(int | None, getattr(arguments, attribute))
        if value is not None:
            remediation[field] = value
    if remediation:
        capacity["remediation"] = remediation
    for attribute, field in (
        ("largest_cluster_node_count", "largestClusterNodeCount"),
        ("managed_node_count", "managedNodeCount"),
    ):
        node_count = cast(int | None, getattr(arguments, attribute))
        if node_count is not None:
            capacity[field] = node_count
    if not capacity:
        raise AdminConfigError(
            "capacity plan requires --preset or at least one explicit setting"
        )
    return apply_capacity_patch(current, capacity)


def _capacity_options_requested(arguments: argparse.Namespace) -> bool:
    return any(
        getattr(arguments, name, None) is not None
        for name in (
            "preset",
            "control_worker_replicas",
            "spool",
            "spool_replicas",
            "max_active_region",
            "max_active_per_cluster",
            "max_active_per_resource_class",
            "largest_cluster_node_count",
            "managed_node_count",
        )
    )


def _admin_config_candidate(
    arguments: argparse.Namespace,
    current: AdminConfig,
) -> tuple[AdminConfig, str]:
    configured_file = cast(
        Path | None,
        getattr(arguments, "admin_config_file", None),
    )
    capacity_options = _capacity_options_requested(arguments)
    if configured_file is not None and capacity_options:
        raise AdminConfigError(
            "--file cannot be combined with --preset or explicit capacity options"
        )
    if configured_file is None and not capacity_options:
        configured_file = admin_config_file_path(arguments.state_dir)
        if not configured_file.is_file():
            write_admin_config_file(
                configured_file,
                current,
                overwrite=False,
            )
            raise AdminConfigError(
                f"created {configured_file}; edit it and rerun gpu-fault-admin config"
            )
    if configured_file is not None:
        desired = load_admin_config_file(configured_file, base=current)
        return desired, f"file:{configured_file.expanduser().resolve()}"
    desired = _capacity_candidate(arguments, current)
    source = (
        f"preset:{arguments.preset}"
        if arguments.preset is not None
        else "explicit-capacity-options"
    )
    return desired, source


def _current_release_metadata(
    repository_root: Path,
) -> dict[str, object]:
    path = repository_root / "dist/current-release.json"
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SiteConfigError("current release manifest is invalid") from exc
    if not isinstance(value, dict):
        raise SiteConfigError("current release manifest must be a JSON object")
    release_id = str(value.get("release_id") or "").strip()
    if not release_id or Path(release_id).name != release_id:
        raise SiteConfigError("current release manifest has no release_id")
    immutable = repository_root / "dist" / release_id / "release.json"
    try:
        immutable_raw = immutable.read_bytes()
    except OSError as exc:
        raise SiteConfigError("content-addressed release manifest is missing") from exc
    if immutable_raw != raw:
        raise SiteConfigError(
            "current release manifest differs from its content-addressed copy"
        )
    staging_only = value.get("staging_only", False)
    if not isinstance(staging_only, bool):
        raise SiteConfigError("current release staging_only must be a boolean")
    return {
        "release_id": release_id,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "staging_only": staging_only,
    }


def _validate_live_release_identity(
    site: RenderedSite,
    release_identity: dict[str, object],
    *,
    uncommitted_target_sha256: str | None = None,
) -> None:
    state = _live_release_state(site)
    phase = str(state.get("phase") or "").strip().lower()
    live_release_id = str(state.get("release_id") or "").strip()
    if live_release_id != release_identity["release_id"]:
        raise SiteConfigError(
            "current signed release differs from the live regional release"
        )
    if state.get("transaction_committed") is True:
        if phase not in {
            "complete",
            "completed",
        }:
            raise SiteConfigError("live regional release is not complete")
        return
    if uncommitted_target_sha256 is None:
        raise SiteConfigError("live regional release is not committed")
    if "rollback" in phase or state.get("rollback_result") is not None:
        raise SiteConfigError("live regional release is rolling back")
    release_diff = state.get("release_diff")
    if (
        not isinstance(release_diff, dict)
        or release_diff.get("kind") != "CONTROL_PLANE_ONLY"
        or state.get("admin_config_sha256") != uncommitted_target_sha256
    ):
        raise SiteConfigError(
            "uncommitted live release does not match the active admin config plan"
        )
    if phase not in {
        "complete",
        "completed",
        # Mirrors `regional_admin_commands.RESUMABLE_PHASES`: every phase an
        # upgrade can be resumed from is a phase an approved admin-config plan
        # may find live.
        "candidate-preflight-ready",
        "cpu-staged",
        "cpu-finalized",
        "data-plane-progress",
        "data-converged",
        "endpoint-ready",
        "observability-ready",
        "profile-ready",
        "registry-staged",
        "schema-ready",
        "uploaded",
        "verified",
    }:
        raise SiteConfigError("uncommitted live release phase cannot be resumed")


def _pending_admin_config_plan(
    state_dir: Path,
    *,
    site_identity: dict[str, str],
    release_identity: dict[str, object],
    desired: AdminConfig,
) -> dict[str, Any] | None:
    if not admin_config_plan_path(state_dir).is_file():
        return None
    plan = load_admin_config_plan(state_dir)
    planned_desired = AdminConfig.from_mapping(plan.get("desired_config"))
    if (
        plan.get("site_identity") == site_identity
        and plan.get("release_identity") == release_identity
        and planned_desired == desired
    ):
        return plan
    return None


def _reconcile_site_aurora(
    site: RenderedSite,
    *,
    expected: AuroraCapacityConfig,
    desired: AuroraCapacityConfig,
) -> dict[str, Any]:
    return reconcile_aurora_capacity(
        aws_region=str(site.release_config["aws_region"]),
        cluster_id=str(site.release_config["health"]["aurora_cluster_id"]),
        expected=expected,
        desired=desired,
    )


def _rollback_site_aurora(
    site: RenderedSite,
    changed: bool,
    current: AdminConfig,
    desired: AdminConfig,
) -> dict[str, Any] | None:
    if not changed:
        return None
    return _reconcile_site_aurora(
        site,
        expected=desired.aurora,
        desired=current.aurora,
    )


def _run_admin_config(arguments: argparse.Namespace) -> int:
    site_file = _managed_site_file(arguments, command="config")
    assert site_file is not None
    site = load_site(site_file)
    release_identity = _current_release_metadata(site.repository_root)
    site_identity = _admin_config_site_identity(site)
    current = load_desired_admin_config(
        arguments.state_dir,
        migrate_legacy=True,
    )
    desired, source = _admin_config_candidate(arguments, current)
    verify_prebuilt_release(
        CommandRunner(),
        repository_root=site.repository_root,
        state_dir=arguments.state_dir,
        staging_only=bool(release_identity["staging_only"]),
    )
    active_plan = _pending_admin_config_plan(
        arguments.state_dir,
        site_identity=site_identity,
        release_identity=release_identity,
        desired=desired,
    )
    resumable = (
        not arguments.dry_run
        and active_plan is not None
        and admin_config_approval_path(arguments.state_dir).is_file()
    )
    _validate_live_release_identity(
        site,
        release_identity,
        uncommitted_target_sha256=(
            str(active_plan["desired_config_sha256"])
            if resumable and active_plan is not None
            else None
        ),
    )
    if arguments.dry_run:
        plan = preview_admin_config_plan(
            arguments.state_dir,
            site_identity=site_identity,
            release_identity=release_identity,
            desired=desired,
            source=source,
        )
        print(
            json.dumps(
                {"status": "DRY_RUN", **plan},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if active_plan is None:
        active_plan = create_admin_config_plan(
            arguments.state_dir,
            site_identity=site_identity,
            release_identity=release_identity,
            desired=desired,
            source=source,
        )
    plan_sha256 = str(active_plan["plan_sha256"])
    prepared = prepare_admin_config_apply(
        arguments.state_dir,
        expected_plan_sha256=plan_sha256,
        reference=arguments.reference,
        current_release_identity=release_identity,
    )
    write_admin_config_file(
        admin_config_file_path(arguments.state_dir),
        desired,
        overwrite=True,
    )
    release_id = str(release_identity["release_id"])
    staging_only = bool(release_identity["staging_only"])
    if prepared.no_op:
        result = complete_admin_config_apply(
            arguments.state_dir,
            expected_plan_sha256=plan_sha256,
            release_id=release_id,
            success=True,
        )
        print(
            json.dumps(
                {
                    "status": "NOOP",
                    "plan_sha256": plan_sha256,
                    "reference": arguments.reference,
                    "audit": str(result),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    reviewed_current = AdminConfig.from_mapping(prepared.plan["current_config"])
    aurora_changed = aurora_capacity_changed(
        reviewed_current.aurora,
        prepared.config.aurora,
    )
    aurora_result: dict[str, Any] | None = None
    rollback_result: dict[str, Any] | None = None

    try:
        if aurora_changed:
            aurora_result = _reconcile_site_aurora(
                site,
                expected=reviewed_current.aurora,
                desired=prepared.config.aurora,
            )
        returncode = _run_automatic_release(
            repository_root=site.repository_root,
            site_file=site_file,
            state_dir=arguments.state_dir,
            staging_only_release=staging_only,
        )
    except Exception as exc:
        rollback_error: Exception | None = None
        try:
            rollback_result = _rollback_site_aurora(
                site, aurora_changed, reviewed_current, prepared.config
            )
        except Exception as rollback_exc:  # noqa: BLE001
            rollback_error = rollback_exc
        error = f"{type(exc).__name__}: {exc}"
        if rollback_error is not None:
            error += (
                "; Aurora rollback failed: "
                f"{type(rollback_error).__name__}: {rollback_error}"
            )
        complete_admin_config_apply(
            arguments.state_dir,
            expected_plan_sha256=plan_sha256,
            release_id=release_id,
            success=False,
            error=error,
            details=(
                {"aurora_rollback": rollback_result}
                if rollback_result is not None
                else None
            ),
        )
        if rollback_error is not None:
            raise AdminConfigError(error) from exc
        raise
    if returncode:
        rollback_error = None
        try:
            rollback_result = _rollback_site_aurora(
                site, aurora_changed, reviewed_current, prepared.config
            )
        except Exception as exc:  # noqa: BLE001
            rollback_error = exc
        error = f"release-deploy exited with status {returncode}"
        if rollback_error is not None:
            error += (
                "; Aurora rollback failed: "
                f"{type(rollback_error).__name__}: {rollback_error}"
            )
        complete_admin_config_apply(
            arguments.state_dir,
            expected_plan_sha256=plan_sha256,
            release_id=release_id,
            success=False,
            error=error,
            details=(
                {"aurora_rollback": rollback_result}
                if rollback_result is not None
                else None
            ),
        )
        if rollback_error is not None:
            raise AdminConfigError(error)
        return returncode
    result = complete_admin_config_apply(
        arguments.state_dir,
        expected_plan_sha256=plan_sha256,
        release_id=release_id,
        success=True,
        details={"aurora": aurora_result} if aurora_result is not None else None,
    )
    print(
        json.dumps(
            {
                "status": "APPLIED",
                "release_id": release_id,
                "config_sha256": prepared.config.sha256(),
                "affected_roles": prepared.plan["affected_roles"],
                "affected_resources": ["aurora"] if aurora_changed else [],
                "plan_sha256": plan_sha256,
                "reference": arguments.reference,
                "audit": str(result),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run(arguments: argparse.Namespace) -> int:
    if arguments.command == "config":
        return _run_admin_config(arguments)
    if arguments.command == "approve-profile":
        return _run_profile_approval(arguments)
    if arguments.command == "join-cluster":
        return _run_join_cluster(arguments)
    if arguments.command == "remove-cluster":
        return _run_remove_cluster(arguments)
    if arguments.command == "uninstall":
        return _run_uninstall(arguments)
    if arguments.command == "workflow-reconcile":
        return _run_workflow_reconcile(arguments)
    if arguments.command == "failure-domain-map":
        return _run_failure_domain_map(arguments)
    if arguments.command in READONLY_COMMANDS:
        site_file = _managed_site_file(
            arguments,
            command=arguments.command,
        )
        assert site_file is not None
        return _run_readonly_managed_command(arguments, site_file)
    site_file = getattr(arguments, "file", None)
    automatic = False
    if arguments.command == "deploy" and site_file is None:
        if not arguments.cpu_cluster_arn or not arguments.gpu_cluster_arn:
            raise SiteConfigError(
                "deploy requires --cpu-cluster-arn and at least one --gpu-cluster-arn"
            )
        if arguments.state_dir is None:
            raise SiteConfigError("deploy requires --state-dir")
        if not arguments.alert_email:
            raise SiteConfigError("deploy requires --admin-email")
        existing_site = (
            arguments.state_dir.expanduser().resolve() / "site.yaml"
        ).is_file()
        initialize_desired_admin_config(
            arguments.state_dir,
            config_file=getattr(arguments, "admin_config_file", None),
            permit_change=not existing_site,
        )
        if not getattr(arguments, "prepared_source_release", False):
            return run_source_deploy(
                cpu_cluster_arn=arguments.cpu_cluster_arn,
                gpu_cluster_arns=tuple(arguments.gpu_cluster_arn),
                state_dir=arguments.state_dir,
                admin_email=arguments.alert_email,
                impact_base=getattr(arguments, "impact_base", "origin/main"),
                current_directory=Path.cwd(),
                extra_environment={
                    **schema_change_environment(arguments),
                    **supersede_environment(arguments),
                    **grafana_environment(arguments),
                },
            )
        repository_root = (arguments.repo_root or Path.cwd()).resolve()
        with administrator_operation_lock(arguments.state_dir):
            bootstrap_result = bootstrap_from_arns(
                BootstrapRequest(
                    cpu_cluster_arn=arguments.cpu_cluster_arn,
                    gpu_cluster_arns=tuple(arguments.gpu_cluster_arn),
                    repository_root=repository_root,
                    state_dir=arguments.state_dir.expanduser(),
                    alert_email=arguments.alert_email,
                    email_sender=getattr(arguments, "email_sender", None),
                    email_recipients=tuple(
                        getattr(arguments, "email_recipient", ()) or ()
                    ),
                    email_subject_prefix=(
                        getattr(arguments, "email_subject_prefix", None) or ""
                    ),
                    staging_only_release=getattr(
                        arguments,
                        "staging_only_release",
                        False,
                    ),
                    impact_base=getattr(arguments, "impact_base", "origin/main"),
                    **grafana_request_fields(arguments),
                )
            )
        site_file = bootstrap_result.site_file
        automatic = True
    if site_file is None:
        raise SiteConfigError(f"{arguments.command} requires -f site.yaml")
    if automatic:
        status = _run_automatic_release(
            repository_root=repository_root,
            site_file=site_file,
            state_dir=arguments.state_dir,
            staging_only_release=getattr(
                arguments,
                "staging_only_release",
                False,
            ),
        )
        if status:
            return status
        if bootstrap_result.pending_gpu_cluster_arns:
            site = load_site(site_file, repository_root=repository_root)
            join_clusters(
                tuple(
                    JoinClusterRequest(
                        site=site,
                        gpu_cluster_arn=gpu_cluster_arn,
                    )
                    for gpu_cluster_arn in bootstrap_result.pending_gpu_cluster_arns
                )
            )
        return 0
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
        environment = {
            **effective_environment(site),
            **quick_validation_evidence_environment(arguments),
            **schema_change_environment(arguments),
            **release_history_environment(arguments),
            **supersede_environment(arguments),
        }
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
        return 0


def _run_reporting_failures(arguments: argparse.Namespace) -> int:
    try:
        enforce_deploy_host_state_dir(arguments)
        return run(arguments)
    except (
        AdminConfigError,
        BootstrapError,
        OSError,
        SiteConfigError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"gpu-fault-admin: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    arguments = parser().parse_args()
    command = str(getattr(arguments, "command", "admin"))
    with command_log(
        getattr(arguments, "state_dir", None),
        command=command,
        # Read-only commands rotate separately (I3), so a habit of ``status``
        # cannot age out the log of the deploy that failed.
        kind=(
            ADMIN_LOG_KIND_READONLY
            if command in READONLY_COMMANDS
            else ADMIN_LOG_KIND_MUTATING
        ),
    ) as log_path:
        status = _run_reporting_failures(arguments)
        if status and log_path is not None:
            # The reason a deploy failed is usually hundreds of lines above the
            # exit code, so name the file at the point the operator is looking.
            announce(f"gpu-fault-admin: full output in {log_path}")
        return status


if __name__ == "__main__":
    raise SystemExit(main())
