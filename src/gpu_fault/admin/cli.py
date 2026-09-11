from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, cast

from gpu_fault.admin import operator_identity
from gpu_fault.admin.aurora_capacity import (
    await_aurora_capacity,
    reconcile_aurora_capacity,
    request_aurora_capacity,
)
from gpu_fault.admin.bootstrap import bootstrap_from_arns, discover_cluster
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    CommandRunner,
)
from gpu_fault.admin.cluster_batch_join import join_clusters
from gpu_fault.admin.cluster_join import (
    DEFAULT_ALLOWED_NAMESPACES,
    JoinClusterRequest,
    join_cluster,
)
from gpu_fault.admin.cluster_removal import (
    RemoveClusterRequest,
    remove_cluster,
    resolve_cluster_id,
)
from gpu_fault.admin.command_log import (
    ADMIN_LOG_ENVIRONMENT,
    ADMIN_LOG_KIND_MUTATING,
    ADMIN_LOG_KIND_READONLY,
    announce,
    command_log,
    report_failure,
)
from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigApply,
    AuroraCapacityConfig,
    admin_config_change_plan,
    begin_admin_config_apply,
    complete_admin_config_apply,
    config_command,
    load_desired_admin_config,
    matching_pending_admin_config_apply,
)
from gpu_fault.admin.config_file import (
    admin_config_file_path,
    load_admin_config_file,
    write_admin_config_file,
)
from gpu_fault.admin.config_parser import AdminConfigError
from gpu_fault.admin.deploy_command import DeployHooks, run_deploy
from gpu_fault.admin.failure_domain_map import (
    add_failure_domain_map_command,
    run_failure_domain_map_command,
)
from gpu_fault.admin.grafana import (
    add_grafana_arguments,
    grafana_environment,
    grafana_request_fields,
)
from gpu_fault.admin.incident_close import escalated_selector
from gpu_fault.admin.incident_close import exit_code as incident_close_exit_code
from gpu_fault.admin.incident_close import quarantined_selector
from gpu_fault.admin.incident_close import result_lines as incident_close_lines
from gpu_fault.admin.collector_outbox import (
    add_collector_outbox_command,
    run_collector_outbox_command,
)
from gpu_fault.admin.incident_close import run_incident_close
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.notification_precheck import WAIT_FLAG
from gpu_fault.admin.operation_lock import inherited_lock_pass_fds
from gpu_fault.admin.profile_approval import (
    ProfileApprovalError,
    approve_profile_plan_inline,
)
from gpu_fault.admin.release_artifacts import verify_prebuilt_release
from gpu_fault.admin.release_consent import (  # noqa: F401 - re-exported
    ACCEPT_SCHEMA_CHANGE_ENV,
    ALLOW_INFLIGHT_INSTALLS_ENV,
    SCHEMA_CHANGE_NO_SNAPSHOT_MODE,
    SCHEMA_CHANGE_SNAPSHOT_MODE,
    SUPERSEDE_FAILED_TRANSACTION_ENV,
    add_release_consent_arguments,
    inflight_installs_environment,
    release_consent_environment,
    schema_change_environment,
    supersede_environment,
)
from gpu_fault.admin.release_state import live_release_state as _live_release_state
from gpu_fault.admin.rollback_alignment import materialized_rollback_status_site
from gpu_fault.admin.rollback_command import run_rollback
from gpu_fault.admin.rotate_token import (
    add_rotate_token_command,
    run_rotate_token_command,
)
from gpu_fault.admin.site import (
    RenderedSite,
    SiteConfigError,
    effective_environment,
    load_site,
    materialized_release_config,
)
from gpu_fault.admin.source_deploy import run_source_deploy
from gpu_fault.admin.submit_remediation import (
    add_submit_remediation_command,
    run_submit_remediation_command,
)
from gpu_fault.admin.uninstall import UninstallRequest, uninstall
from gpu_fault.admin import warm_spare
from gpu_fault.admin.workflow_reconcile import run_workflow_reconcile
from gpu_fault_release.regional_admin_commands import status_header_lines
from gpu_fault_release.regional_validation_evidence import (
    QUICK_VALIDATION_EVIDENCE_ENV,
    QUICK_VALIDATION_EVIDENCE_FILE,  # noqa: F401  # re-exported; tests pin it
    quick_validation_evidence_path,
)

# Admin command -> release engine mode.
COMMANDS = {
    "preflight": "preflight",
    "deploy": "deploy",
    "verify": "verify",
    "status": "status",
}
# The one read-only verb an administrator is given. ``status`` and ``verify``
# had become the same 44-second report with `healthy` on line 1150, so the
# verb is `status`: five readable lines on stderr, the cheap checks by default,
# `--full` for the whole report acceptance evidence records.
PUBLIC_READONLY_COMMANDS = ("status",)
# Driver-only engine passthroughs: ``scripts/release_deploy.py`` runs `verify`
# beside the stability window and ``scripts/staging_deploy.py`` runs
# `preflight` for a deploy-host-only change. They are not advertised, not in
# `--help`, and not in the administrator task table; the drivers still need a
# CLI that materializes the site's release config and environment for them.
INTERNAL_READONLY_COMMANDS = ("preflight", "verify")
# The managed commands that write nothing; their console logs rotate under
# ``logs/readonly/`` so they cannot age out a failed deploy's log (I3).
READONLY_COMMANDS = frozenset(PUBLIC_READONLY_COMMANDS + INTERNAL_READONLY_COMMANDS)
DEPLOY_HOST_STATE_BINDING = "gpu-fault-managed-state-dir.json"


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

    The path is `quick_validation_evidence_path` of the managed state directory
    -- the directory holding `site.yaml`, whether it arrived as `--state-dir` or
    as the parent of `-f`. The release driver finalizes its evidence at the same
    path, so a `status` seconds after a driver-run deploy finds it.
    """

    if os.environ.get(QUICK_VALIDATION_EVIDENCE_ENV, "").strip():
        return {}
    command = getattr(arguments, "command", None)
    if command not in ("deploy", "status"):
        return {}
    state_dir = cast(Path | None, getattr(arguments, "state_dir", None))
    if state_dir is None:
        site_file = cast(Path | None, getattr(arguments, "file", None))
        if site_file is None:
            return {}
        state_dir = site_file.expanduser().resolve().parent
    path = quick_validation_evidence_path(state_dir)
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


def _add_managed_site_arguments(command: argparse.ArgumentParser) -> None:
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


def _add_deploy_action_arguments(deploy: argparse.ArgumentParser) -> None:
    """The flags that turn the one deploy command into its other two actions."""

    deploy.add_argument(
        "--rollback",
        action="store_true",
        help=(
            "roll the site back one step, to the release recorded as previous; "
            "refused while a transaction is in flight or when there is no "
            "previous release; takes no other deploy option"
        ),
    )
    deploy.add_argument(
        "--approve-profile-plan",
        metavar="PLAN_SHA256",
        help=(
            "approve the pending Runtime Profile plan with this plan_sha256 "
            "(printed when the deploy stopped for review) and continue the "
            "same deploy; requires --reference"
        ),
    )
    deploy.add_argument(
        "--reference",
        metavar="REFERENCE",
        help="approved change or maintenance-window reference for the approval",
    )
    deploy.add_argument(
        WAIT_FLAG,
        dest="wait_for_email_confirmation",
        type=int,
        default=0,
        metavar="MINUTES",
        help=(
            "wait up to MINUTES for the SNS subscription (and the SES sender on a "
            "spec.notifications.channel: ses site) to confirm instead of stopping"
        ),
    )


def _add_admin_config_command(commands: Any) -> None:
    config = commands.add_parser(
        "config",
        usage=(
            "gpu-fault-admin config --state-dir STATE_DIR "
            "[--file ADMIN_CONFIG] [--dry-run] [--reference REFERENCE]"
        ),
        help=(
            "apply the edited <state-dir>/admin-config.yaml to the live site "
            "(Aurora window and the affected control-plane roles)"
        ),
    )
    # Not ``required``: the ``config spare`` sub-action carries its own
    # ``--state-dir``; ``_managed_site_file`` refuses a bare ``config`` without it.
    config.add_argument("--state-dir", type=Path, metavar="STATE_DIR")
    config.add_argument(
        "--file",
        dest="admin_config_file",
        type=Path,
        metavar="ADMIN_CONFIG",
        help=("private AdminConfig YAML; defaults to <state-dir>/admin-config.yaml"),
    )
    config.add_argument(
        "--reference",
        metavar="REFERENCE",
        help=(
            "change or maintenance-window reference for the audit record; "
            "defaults to the approver identity and the start time"
        ),
    )
    config.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "print the change plan (affected roles, changed fields, Aurora "
            "change) from local state only; nothing is verified or written"
        ),
    )
    warm_spare.add_config_spare_command(config)


def _add_workflow_reconcile_command(commands: Any) -> None:
    reconcile = commands.add_parser(
        "workflow-reconcile",
        usage=(
            "gpu-fault-admin workflow-reconcile --state-dir STATE_DIR "
            "[--workflow-id ID ...] [--incident-id ID ...] [--max-items N] "
            "[--reference REFERENCE] [--dry-run]\n"
            "       gpu-fault-admin workflow-reconcile --state-dir STATE_DIR "
            "(--close-incident INCIDENT_ID ... | --close-escalated [--max-items N] "
            "| --close-quarantined [--max-items N]) "
            "[--reason TEXT] [--reference REFERENCE] [--dry-run]"
        ),
        help=(
            "close BLOCKED workflow records a later workflow already restored "
            "(plans and applies in one run; --dry-run only prints the plan); "
            "with --close-incident / --close-escalated close ESCALATED incidents, "
            "with --close-quarantined close QUARANTINED incidents whose node "
            "isolation is gone (a hand-released taint's leftover annotations are stripped)"
        ),
    )
    _add_managed_site_arguments(reconcile)
    reconcile.add_argument(
        "--workflow-id",
        action="append",
        default=[],
        metavar="ID",
        help="reconcile only this BLOCKED workflow (repeatable)",
    )
    # Discovery selectors: without --workflow-id the plan scans the BLOCKED
    # backlog; these narrow it to one incident's records and cap the batch.
    reconcile.add_argument(
        "--incident-id",
        action="append",
        default=[],
        metavar="ID",
        help="discover only this incident's BLOCKED records (repeatable)",
    )
    reconcile.add_argument(
        "--max-items",
        type=int,
        default=None,
        metavar="N",
        help="cap the discovered batch at N records",
    )
    reconcile.add_argument(
        "--reference",
        metavar="REFERENCE",
        help="approved change or maintenance-window reference; required unless --dry-run",
    )
    reconcile.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan with its node evidence and write nothing",
    )
    _add_incident_close_arguments(reconcile)


def _add_incident_close_arguments(reconcile: argparse.ArgumentParser) -> None:
    """``--close-incident``: the operator exit for an ESCALATED incident.

    Rides the ``workflow-reconcile`` verb because it is the same kind of
    disposition -- an operator ending a record the runtime handed over --
    and reaches the control plane the same way (a script in the CPU ingress
    Pod calling the service the API route calls). ``--reference`` is the
    verb's existing approved-change reference; ``--dry-run`` reports the
    verdicts and writes nothing.

    ``--close-escalated`` is the same disposition without the ids: it discovers
    every ESCALATED incident of the site (all registered clusters, oldest
    first, ``--max-items`` caps the batch) and judges or closes each one exactly
    as ``--close-incident`` would. ``--close-quarantined`` discovers the
    QUARANTINED queue the same way; those close only on node isolation
    evidence read through the site's GPU kubeconfig (no cordon, no quarantine
    taint of the incident, no isolation annotation of it; annotations left by a
    hand-released taint are stripped first). The three are mutually exclusive.
    """

    selection = reconcile.add_mutually_exclusive_group()
    selection.add_argument(
        "--close-incident",
        action="append",
        default=[],
        metavar="INCIDENT_ID",
        help=(
            "close this ESCALATED incident RECOVERED (repeatable); needs --reason "
            "and --reference; refused while a workflow of it is still open; a "
            "QUARANTINED id is judged on node isolation evidence read through "
            "the GPU kubeconfig"
        ),
    )
    selection.add_argument(
        "--close-escalated",
        action="store_true",
        help=(
            "discover every ESCALATED incident of the site (all registered "
            "clusters, oldest first; --max-items caps the batch) and close each "
            "one as --close-incident would; --dry-run lists the ids and verdicts "
            "without writing, otherwise --reason and --reference are required"
        ),
    )
    selection.add_argument(
        "--close-quarantined",
        action="store_true",
        help=(
            "discover every QUARANTINED incident of the site and close those "
            "whose nodes carry no cordon, no gpu-fault quarantine taint of the "
            "incident and no isolation annotation of it (read through the GPU "
            "kubeconfig); --dry-run lists the verdicts without writing, "
            "otherwise --reason and --reference are required"
        ),
    )
    reconcile.add_argument(
        "--reason",
        metavar="TEXT",
        help="why the incident is closed; recorded on the incident and its audit event",
    )


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


def _internal_readonly_parser(name: str) -> argparse.ArgumentParser:
    """The argument parser for a driver-only engine passthrough verb.

    Built outside the public sub-command table so `--help` and the usage line
    never mention it; the drivers pass the same site arguments `status` takes.
    """

    result = argparse.ArgumentParser(
        prog=f"gpu-fault-admin {name}",
        usage=f"gpu-fault-admin {name} --state-dir STATE_DIR",
    )
    result.set_defaults(command=name, full=False)
    _add_managed_site_arguments(result)
    return result


class _AdminParser(argparse.ArgumentParser):
    """`gpu-fault-admin` with the driver-only verbs routed off the public table."""

    def parse_known_args(  # type: ignore[override]
        self,
        args: Any = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        argv = list(sys.argv[1:] if args is None else args)
        if argv and argv[0] in INTERNAL_READONLY_COMMANDS:
            return _internal_readonly_parser(argv[0]).parse_known_args(
                argv[1:], namespace
            )
        return super().parse_known_args(argv, namespace)


def parser() -> argparse.ArgumentParser:
    result: argparse.ArgumentParser = _AdminParser(
        prog="gpu-fault-admin",
        description="Regional GPU fault deployment and health administration",
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name in PUBLIC_READONLY_COMMANDS:
        command = commands.add_parser(
            name,
            usage=f"gpu-fault-admin {name} --state-dir STATE_DIR [--full]",
            description=(
                "Report the live release and the control plane's health: five "
                "lines on stderr, the JSON document on stdout"
            ),
        )
        _add_managed_site_arguments(command)
        command.add_argument(
            "--full",
            action="store_true",
            help=(
                "run every health check (GPU clusters, role split, NLB, Aurora, "
                "monitoring) instead of the control-plane workload and API "
                "checks; this is the report acceptance evidence records"
            ),
        )
    deploy = commands.add_parser(
        "deploy",
        usage=(
            "gpu-fault-admin deploy --state-dir STATE_DIR "
            "[--cpu-cluster-arn CPU_ARN --gpu-cluster-arn GPU_ARN ... "
            "--admin-email EMAIL] [--rollback] "
            "[--approve-profile-plan PLAN_SHA256 --reference REFERENCE]"
        ),
        description=(
            "Bootstrap a new site (all four inputs), upgrade the site recorded "
            "under STATE_DIR (--state-dir alone), add a GPU cluster (the managed "
            "ARNs plus the new one), roll back one step (--rollback) or approve "
            "the pending Runtime Profile plan and continue "
            "(--approve-profile-plan): one command"
        ),
    )
    deploy.add_argument("-f", "--file", type=Path, help=argparse.SUPPRESS)
    deploy.add_argument(
        "--cpu-cluster-arn",
        metavar="CPU_ARN",
        help=(
            "existing CPU EKS or HyperPod cluster ARN; required on the first "
            "deploy, read from the site afterwards"
        ),
    )
    deploy.add_argument(
        "--gpu-cluster-arn",
        action="append",
        default=[],
        metavar="GPU_ARN",
        help=(
            "existing GPU EKS or HyperPod cluster ARN; repeat for more clusters. "
            "On a managed site the set must be the managed clusters or a superset "
            "of them: the extra clusters are joined after the release"
        ),
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
        "--admin-email",
        dest="alert_email",
        metavar="EMAIL",
        help=(
            "administrator email for SES fault notifications and SNS alerts; "
            "required on the first deploy, read from the site afterwards"
        ),
    )
    _add_deploy_action_arguments(deploy)
    add_grafana_arguments(deploy)
    deploy.add_argument(
        "--staging-only-release",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    add_release_consent_arguments(deploy)
    deploy.add_argument("--impact-base", default="origin/main", help=argparse.SUPPRESS)
    deploy.add_argument(
        "--prepared-source-release",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    _add_admin_config_command(commands)
    _add_workflow_reconcile_command(commands)
    add_failure_domain_map_command(commands, _add_managed_site_arguments)
    add_rotate_token_command(commands, _add_managed_site_arguments)
    add_submit_remediation_command(commands, _add_managed_site_arguments)
    add_collector_outbox_command(commands, _add_managed_site_arguments)
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
            "--gpu-cluster-arn GPU_ARN --confirm REMOVE_GPU_CLUSTER"
        ),
        help="uninstall one managed GPU data plane and keep the CPU control plane",
    )
    _add_managed_site_arguments(detach)
    detach.add_argument(
        "--gpu-cluster-arn",
        required=True,
        metavar="GPU_ARN",
        help=(
            "the managed GPU EKS or HyperPod cluster ARN to remove, as given to "
            "deploy or join-cluster"
        ),
    )
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
        "--cpu-cluster",
        choices=("keep", "delete"),
        default="keep",
        help=(
            "keep: reinstall later, keeping the CPU cluster and the Aurora "
            "records; delete: retire the CPU control plane and its Aurora cluster"
        ),
    )
    remove.add_argument(
        "--confirm",
        required=True,
        help=(
            "UNINSTALL_GPU_FAULT with --cpu-cluster keep, "
            "DELETE_CPU_CONTROL_PLANE with --cpu-cluster delete"
        ),
    )
    remove.add_argument(
        "--reset-database",
        action="store_true",
        help=(
            "with --cpu-cluster keep only: also delete the Aurora cluster so the "
            "next deploy starts from an empty database"
        ),
    )
    remove.add_argument(
        "--aurora-final-snapshot",
        choices=("retain", "skip"),
        default="retain",
        help=(
            "retain a final Aurora cluster snapshot for audit, or skip it; "
            "skip is only valid with --cpu-cluster delete"
        ),
    )
    return result


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

    def discover(cluster_arn: str) -> tuple[str, str]:
        identity = discover_cluster(
            CommandRunner(),
            cluster_arn=cluster_arn,
            role="gpu",
            context="gpu-fault-admin-remove",
        )
        return identity.eks_arn, identity.hyperpod_name

    result = remove_cluster(
        RemoveClusterRequest(
            site=site,
            cluster_id=resolve_cluster_id(
                site, arguments.gpu_cluster_arn, discover=discover
            ),
            confirmation=arguments.confirm,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_uninstall(arguments: argparse.Namespace) -> int:
    site_file = _managed_site_file(arguments, command="uninstall")
    assert site_file is not None
    site = load_site(site_file, repository_root=arguments.repo_root)
    try:
        request = UninstallRequest(
            site=site,
            cpu_disposition=arguments.cpu_cluster,
            confirmation=arguments.confirm,
            final_snapshot_policy=arguments.aurora_final_snapshot,
            reset_database=bool(arguments.reset_database),
        )
    except BootstrapError as exc:
        raise SiteConfigError(str(exc)) from exc
    uninstall_result = uninstall(request)
    print(json.dumps(uninstall_result, indent=2, sort_keys=True))
    return 0


def _run_incident_close(arguments: argparse.Namespace) -> int:
    site_file = _managed_site_file(arguments, command="workflow-reconcile")
    assert site_file is not None
    close_escalated = bool(getattr(arguments, "close_escalated", False))
    close_quarantined = bool(getattr(arguments, "close_quarantined", False))
    if close_escalated:
        flag = "--close-escalated"
    elif close_quarantined:
        flag = "--close-quarantined"
    else:
        flag = "--close-incident"
    if arguments.state_dir is None:
        raise SiteConfigError(f"workflow-reconcile {flag} requires --state-dir")
    selectors = sum(
        1
        for chosen in (
            close_escalated,
            close_quarantined,
            bool(getattr(arguments, "close_incident", None)),
        )
        if chosen
    )
    if selectors > 1:
        raise SiteConfigError(
            "workflow-reconcile takes one of --close-escalated, --close-quarantined "
            "or --close-incident, not several"
        )
    state_dir = arguments.state_dir.expanduser().resolve()
    options: dict[str, Any] = {}
    if close_escalated:
        # The discovery: ESCALATED only, every registered cluster, oldest
        # first, capped by the verb's --max-items.
        options["selector"] = escalated_selector(max_items=arguments.max_items)
    elif close_quarantined:
        # QUARANTINED only; each one is then judged on node isolation
        # evidence read through the site's GPU kubeconfig.
        options["selector"] = quarantined_selector(max_items=arguments.max_items)
    else:
        options["incident_ids"] = tuple(arguments.close_incident)
    try:
        with administrator_operation_lock(state_dir):
            site = load_site(site_file, repository_root=arguments.repo_root)
            result = run_incident_close(
                site,
                state_dir,
                reason=arguments.reason or "",
                reference=arguments.reference,
                dry_run=bool(getattr(arguments, "dry_run", False)),
                **options,
            )
    except BootstrapError as exc:
        raise SiteConfigError(str(exc)) from exc
    for line in incident_close_lines(result):
        print(line)
    return incident_close_exit_code(result)


def _run_workflow_reconcile(arguments: argparse.Namespace) -> int:
    if (
        getattr(arguments, "close_incident", None)
        or getattr(arguments, "close_escalated", False)
        or getattr(arguments, "close_quarantined", False)
    ):
        return _run_incident_close(arguments)
    site_file = _managed_site_file(arguments, command="workflow-reconcile")
    assert site_file is not None
    # ``-f`` names the site without a state directory, but the archive and the
    # administrator lock live under --state-dir, so it is required either way.
    if arguments.state_dir is None:
        raise SiteConfigError("workflow-reconcile requires --state-dir")
    state_dir = arguments.state_dir.expanduser().resolve()
    try:
        with administrator_operation_lock(state_dir):
            site = load_site(site_file, repository_root=arguments.repo_root)
            result = run_workflow_reconcile(
                site,
                state_dir,
                workflow_ids=tuple(arguments.workflow_id),
                incident_ids=tuple(arguments.incident_id),
                max_items=arguments.max_items,
                reference=arguments.reference,
                dry_run=bool(arguments.dry_run),
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
        rollout = (
            site.repository_root
            / "deploy/control-plane/regional/rollout-regional-release.sh"
        )
        is_status = arguments.command == "status"
        with materialized_release_config(site) as config:
            completed = subprocess.run(
                [
                    str(rollout),
                    COMMANDS[arguments.command],
                    "--config",
                    str(config),
                    *(
                        ["--full"]
                        if is_status and getattr(arguments, "full", False)
                        else []
                    ),
                ],
                cwd=site.repository_root,
                env={
                    **effective_environment(site),
                    **quick_validation_evidence_environment(arguments),
                },
                check=False,
                # `status` is read here so its five-line header can go out
                # ahead of the document; the engine's own stderr streams live.
                stdout=subprocess.PIPE if is_status else None,
                text=is_status,
            )
        if is_status:
            print_status_report(completed.stdout)
        return completed.returncode


def status_document(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        lines = text.splitlines()
        starts = [index for index, line in enumerate(lines) if line.strip() == "{"]
        if not starts:
            return None
        try:
            value = json.loads("\n".join(lines[starts[0] :]))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def print_status_report(stdout: str | None) -> None:
    """Five readable lines on stderr, then the engine's stdout byte for byte.

    The JSON document is what scripts parse (`staging_live_evidence`, the
    acceptance recorders), so it is passed through unchanged; the header is the
    part an administrator reads.
    """

    report = status_document(stdout or "")
    log_path = os.environ.get(ADMIN_LOG_ENVIRONMENT, "").strip()
    destination = "stdout" + (f" (also in {log_path})" if log_path else "")
    lines = (
        status_header_lines(report, json_destination=destination)
        if report is not None
        else ["status printed no JSON report; see the output below"]
    )
    print(
        "\n".join(f"gpu-fault-admin status: {line}" for line in lines), file=sys.stderr
    )
    sys.stderr.flush()
    if stdout:
        sys.stdout.write(stdout)
        sys.stdout.flush()


def _admin_config_site_identity(site: RenderedSite) -> dict[str, str]:
    return {
        "site_name": str(site.release_config["site_name"]),
        "aws_region": str(site.release_config["aws_region"]),
        "cpu_eks_arn": str(site.release_config["cpu_eks_arn"]),
    }


def _admin_config_candidate(
    arguments: argparse.Namespace,
    current: AdminConfig,
) -> tuple[AdminConfig, str]:
    """The target config: ``--file`` or ``<state-dir>/admin-config.yaml``.

    Without either on disk the canonical file is written from the current
    state and the command stops, so the administrator edits a complete
    document instead of authoring one from memory.
    """

    configured_file = cast(
        Path | None,
        getattr(arguments, "admin_config_file", None),
    )
    if configured_file is None:
        configured_file = admin_config_file_path(arguments.state_dir)
        if not configured_file.is_file():
            write_admin_config_file(configured_file, current, overwrite=False)
            raise AdminConfigError(
                f"created {configured_file}; edit it and rerun "
                f"{config_command(arguments.state_dir)}"
            )
    desired = load_admin_config_file(configured_file, base=current)
    return desired, f"file:{configured_file.expanduser().resolve()}"


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
    state_dir: Path,
    uncommitted_target_sha256: str | None = None,
) -> None:
    """Refuse to apply unless the live release is this signed one and is settled.

    ``uncommitted_target_sha256`` is the pending apply's config digest: the one
    uncommitted live release that may be resumed is a control-plane-only
    release of exactly that config (the live state's ``admin_config_sha256``).
    """

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
    finish = (
        f"finish it with gpu-fault-admin deploy --state-dir {state_dir} before "
        f"rerunning {config_command(state_dir)}"
    )
    if uncommitted_target_sha256 is None:
        raise SiteConfigError(f"live regional release is not committed; {finish}")
    if "rollback" in phase or state.get("rollback_result") is not None:
        raise SiteConfigError(f"live regional release is rolling back; {finish}")
    release_diff = state.get("release_diff")
    if (
        not isinstance(release_diff, dict)
        or release_diff.get("kind") != "CONTROL_PLANE_ONLY"
        or state.get("admin_config_sha256") != uncommitted_target_sha256
    ):
        raise SiteConfigError(
            "uncommitted live release is not the pending admin config apply; " + finish
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


def _approver_identity() -> str:
    """Who is applying: the STS caller ARN, else ``user@host``; never anonymous."""

    return operator_identity.resolve_operator_identity(
        fallback=operator_identity.local_operator_identity()
    )


def _site_aurora(site: RenderedSite) -> tuple[str, str]:
    """``(aws_region, cluster_id)`` of the site's control-plane Aurora."""

    return (
        str(site.release_config["aws_region"]),
        str(site.release_config["health"]["aurora_cluster_id"]),
    )


def _request_site_aurora(
    site: RenderedSite,
    *,
    expected: AuroraCapacityConfig,
    desired: AuroraCapacityConfig,
) -> dict[str, Any]:
    aws_region, cluster_id = _site_aurora(site)
    return request_aurora_capacity(
        aws_region=aws_region,
        cluster_id=cluster_id,
        expected=expected,
        desired=desired,
    )


def _await_site_aurora(
    site: RenderedSite,
    *,
    desired: AuroraCapacityConfig,
    requested: dict[str, Any],
) -> dict[str, Any]:
    """Finish a scale-up once the roles rolled: both instances at the new floor."""

    if not requested["scale_up"]:
        return requested
    aws_region, cluster_id = _site_aurora(site)
    after = await_aurora_capacity(
        aws_region=aws_region,
        cluster_id=cluster_id,
        desired=desired,
        modified=bool(requested["modified"]),
    )
    return {**requested, "after": after}


def _rollback_site_aurora(
    site: RenderedSite,
    changed: bool,
    current: AdminConfig,
    desired: AdminConfig,
) -> dict[str, Any] | None:
    if not changed:
        return None
    aws_region, cluster_id = _site_aurora(site)
    return reconcile_aurora_capacity(
        aws_region=aws_region,
        cluster_id=cluster_id,
        expected=desired.aurora,
        desired=current.aurora,
    )


def _fail_admin_config_apply(
    state_dir: Path,
    site: RenderedSite,
    apply: AdminConfigApply,
    *,
    release_id: str,
    error: str,
    restore_before: bool = True,
) -> AdminConfigError | None:
    """Undo the Aurora change and record the failure; a failed undo is returned.

    ``restore_before=False`` is the failure after the roles already run the
    target (the Aurora scale-up did not finish ramping): the window stays where
    it was asked to go, ``desired.json`` stays the target, and the pending
    record waits for the rerun to finish the wait.
    """

    rollback_result: dict[str, Any] | None = None
    rollback_error: Exception | None = None
    if restore_before:
        try:
            rollback_result = _rollback_site_aurora(
                site, apply.aurora_changed, apply.before, apply.desired
            )
        except Exception as exc:  # noqa: BLE001
            rollback_error = exc
    if rollback_error is not None:
        error += (
            "; Aurora rollback failed: "
            f"{type(rollback_error).__name__}: {rollback_error}"
        )
    complete_admin_config_apply(
        state_dir,
        config_sha256=apply.config_sha256,
        release_id=release_id,
        success=False,
        error=error,
        details=(
            {"aurora_rollback": rollback_result}
            if rollback_result is not None
            else None
        ),
        restore_before=restore_before,
    )
    return AdminConfigError(error) if rollback_error is not None else None


def _roll_admin_config(
    arguments: argparse.Namespace,
    site: RenderedSite,
    site_file: Path,
    apply: AdminConfigApply,
    *,
    release_id: str,
    staging_only: bool,
) -> tuple[int, dict[str, Any] | None]:
    """Aurora modify, role rollout, then the scale-up wait; undone on failure.

    The window change is issued first and the roles roll while a scale-up's
    instances ramp: the two are independent and the ramp is the slow part. A
    failure before the roles are live rolls Aurora back and restores the
    previous ``desired.json``; a failure after (the ramp did not finish in
    time) keeps the target, which is what the roles now run, and leaves the
    pending record for the rerun to finish waiting.
    """

    aurora: dict[str, Any] | None = None
    rolled = False
    try:
        if apply.aurora_changed:
            aurora = _request_site_aurora(
                site,
                expected=apply.before.aurora,
                desired=apply.desired.aurora,
            )
        returncode = _run_automatic_release(
            repository_root=site.repository_root,
            site_file=site_file,
            state_dir=arguments.state_dir,
            staging_only_release=staging_only,
        )
        rolled = returncode == 0
        if rolled and aurora is not None:
            aurora = _await_site_aurora(
                site,
                desired=apply.desired.aurora,
                requested=aurora,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if rolled:
            error += (
                "; the control-plane roles already run the new config, rerun "
                f"{config_command(arguments.state_dir)} to finish waiting for Aurora"
            )
        failure = _fail_admin_config_apply(
            arguments.state_dir,
            site,
            apply,
            release_id=release_id,
            error=error,
            restore_before=not rolled,
        )
        if failure is not None or rolled:
            raise AdminConfigError(error) from exc
        raise
    if returncode:
        failure = _fail_admin_config_apply(
            arguments.state_dir,
            site,
            apply,
            release_id=release_id,
            error=f"release-deploy exited with status {returncode}",
        )
        if failure is not None:
            raise failure
        return returncode, None
    return 0, aurora


def _apply_admin_config(
    arguments: argparse.Namespace,
    site: RenderedSite,
    site_file: Path,
) -> int:
    """The locked apply: verify, record, roll, complete, then normalise the YAML."""

    state_dir = arguments.state_dir
    release_identity = _current_release_metadata(site.repository_root)
    site_identity = _admin_config_site_identity(site)
    current = load_desired_admin_config(state_dir, migrate_legacy=True)
    desired, source = _admin_config_candidate(arguments, current)
    staging_only = bool(release_identity["staging_only"])
    verify_prebuilt_release(
        CommandRunner(),
        repository_root=site.repository_root,
        state_dir=state_dir,
        staging_only=staging_only,
    )
    pending = matching_pending_admin_config_apply(
        state_dir,
        site_identity=site_identity,
        release_identity=release_identity,
        desired=desired,
    )
    _validate_live_release_identity(
        site,
        release_identity,
        state_dir=state_dir,
        uncommitted_target_sha256=(
            str(pending["desired_config_sha256"]) if pending is not None else None
        ),
    )
    apply = begin_admin_config_apply(
        state_dir,
        site_identity=site_identity,
        release_identity=release_identity,
        desired=desired,
        source=source,
        approver_identity=_approver_identity(),
        reference=arguments.reference,
    )
    release_id = str(release_identity["release_id"])
    audit = {
        "config_sha256": apply.config_sha256,
        "reference": apply.reference,
        "approver_identity": apply.approver_identity,
    }
    if apply.no_op:
        print(json.dumps({"status": "NOOP", **audit}, indent=2, sort_keys=True))
        return 0
    returncode, aurora = _roll_admin_config(
        arguments,
        site,
        site_file,
        apply,
        release_id=release_id,
        staging_only=staging_only,
    )
    if returncode:
        return returncode
    result = complete_admin_config_apply(
        state_dir,
        config_sha256=apply.config_sha256,
        release_id=release_id,
        success=True,
        details={"aurora": aurora} if aurora is not None else None,
    )
    # The administrator's file is normalised only now, when desired.json and
    # the live site agree with it; a failed apply leaves their edit untouched.
    write_admin_config_file(admin_config_file_path(state_dir), desired, overwrite=True)
    print(
        json.dumps(
            {
                "status": "APPLIED",
                "release_id": release_id,
                "affected_roles": apply.affected_roles,
                "affected_resources": ["aurora"] if apply.aurora_changed else [],
                "history": apply.history,
                "audit": str(result),
                **audit,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_admin_config(arguments: argparse.Namespace) -> int:
    site_file = _managed_site_file(arguments, command="config")
    assert site_file is not None
    site = load_site(site_file)
    if getattr(arguments, "config_action", None) == "spare":
        return warm_spare.run_config_spare_command(arguments, site=site)
    if arguments.dry_run:
        # Local only: no signature check, no cluster read, nothing written.
        current = load_desired_admin_config(arguments.state_dir)
        desired, source = _admin_config_candidate(arguments, current)
        plan = admin_config_change_plan(current, desired, source=source)
        print(json.dumps({"status": "DRY_RUN", **plan}, indent=2, sort_keys=True))
        return 0
    with administrator_operation_lock(arguments.state_dir):
        return _apply_admin_config(arguments, site, site_file)


def _run_deploy(arguments: argparse.Namespace) -> int:
    """``gpu-fault-admin deploy`` without ``-f``; the decisions live in
    ``deploy_command``, the collaborators are looked up here so one module is
    the place a test replaces them."""

    return run_deploy(
        arguments,
        hooks=DeployHooks(
            run_source_deploy=run_source_deploy,
            bootstrap_from_arns=bootstrap_from_arns,
            run_automatic_release=_run_automatic_release,
            join_clusters=join_clusters,
            run_rollback=run_rollback,
            approve_profile_plan_inline=approve_profile_plan_inline,
            discover_cluster=discover_cluster,
            load_site=load_site,
            release_consent_environment=release_consent_environment,
            grafana_environment=grafana_environment,
            grafana_request_fields=grafana_request_fields,
        ),
    )


def run(arguments: argparse.Namespace) -> int:
    if arguments.command == "config":
        return _run_admin_config(arguments)
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
    if arguments.command == "rotate-token":
        return run_rotate_token_command(
            arguments,
            site=load_site(
                cast(Path, _managed_site_file(arguments, command="rotate-token")),
                repository_root=arguments.repo_root,
            ),
        )
    if arguments.command == "submit-remediation":
        return run_submit_remediation_command(
            arguments,
            site=load_site(
                cast(Path, _managed_site_file(arguments, command="submit-remediation")),
                repository_root=arguments.repo_root,
            ),
        )
    if arguments.command == "collector-outbox":
        return run_collector_outbox_command(
            arguments,
            site=load_site(
                cast(Path, _managed_site_file(arguments, command="collector-outbox")),
                repository_root=arguments.repo_root,
            ),
        )
    if arguments.command in READONLY_COMMANDS:
        site_file = _managed_site_file(
            arguments,
            command=arguments.command,
        )
        assert site_file is not None
        return _run_readonly_managed_command(arguments, site_file)
    site_file = getattr(arguments, "file", None)
    if arguments.command == "deploy" and site_file is None:
        return _run_deploy(arguments)
    if site_file is None:
        raise SiteConfigError(f"{arguments.command} requires -f site.yaml")
    site = load_site(site_file, repository_root=arguments.repo_root)
    if arguments.command == "deploy" and getattr(arguments, "alert_email", None):
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
    rollout = (
        site.repository_root
        / "deploy/control-plane/regional/rollout-regional-release.sh"
    )
    with materialized_release_config(site) as config:
        environment = {
            **effective_environment(site),
            **quick_validation_evidence_environment(arguments),
            **release_history_environment(arguments),
            **release_consent_environment(arguments),
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
        return 0


def _run_reporting_failures(arguments: argparse.Namespace) -> int:
    try:
        enforce_deploy_host_state_dir(arguments)
        return run(arguments)
    except (
        AdminConfigError,
        BootstrapError,
        OSError,
        ProfileApprovalError,
        SiteConfigError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        # A child that is one of our own drivers (``make``, a nested CLI, a
        # Python script) has already printed its cause; this only adds a line
        # for foreign commands and hands the child's exit status through.
        return report_failure("gpu-fault-admin", exc)


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
            # Only the process that opened the log has a path here: a nested
            # invocation inherits the tee and stays silent, so one failure is
            # announced once rather than once per layer.
            announce(f"gpu-fault-admin: full output in {log_path}")
        return status


if __name__ == "__main__":
    raise SystemExit(main())
