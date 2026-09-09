"""``gpu-fault-admin deploy`` without ``-f``: the one administrator command.

The verb covers five things with one parser: the first bootstrap (all four
inputs), the upgrade (``--state-dir`` alone), adding a GPU cluster (the managed
ARNs plus the new one), rolling back one step (``--rollback``) and approving a
pending Runtime Profile plan and continuing (``--approve-profile-plan``). The
CLI module keeps the parser and the dispatch; the decisions live here.

Every collaborator the command shells out to or that touches AWS arrives in
:class:`DeployHooks`, looked up by the CLI at call time, so the CLI module
remains the one place a test replaces them.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gpu_fault.admin.bootstrap_checkpoint import load_hyperpod_hints
from gpu_fault.admin.bootstrap_common import BootstrapRequest, CommandRunner
from gpu_fault.admin.bootstrap_site import load_existing_site
from gpu_fault.admin.cluster_join import JoinClusterRequest
from gpu_fault.admin.config_file import initialize_desired_admin_config
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.site import SiteConfigError

# Written by ``scripts/staging_deploy.py`` (``staging_state_hygiene``) after a
# successful source deploy; ``mode`` says whether a release was applied.
SOURCE_DEPLOY_SUCCESS_STATE = "source-deploy-success.json"


@dataclass(frozen=True)
class DeployHooks:
    """The command's collaborators, passed in by the CLI module."""

    run_source_deploy: Callable[..., int]
    bootstrap_from_arns: Callable[..., Any]
    run_automatic_release: Callable[..., int]
    join_clusters: Callable[..., Any]
    run_rollback: Callable[..., int]
    approve_profile_plan_inline: Callable[..., Mapping[str, Any]]
    discover_cluster: Callable[..., Any]
    load_site: Callable[..., Any]
    release_consent_environment: Callable[[argparse.Namespace], dict[str, str]]
    grafana_environment: Callable[[argparse.Namespace], dict[str, str]]
    grafana_request_fields: Callable[[argparse.Namespace], dict[str, Any]]


def managed_site_deploy_inputs(
    existing: Mapping[str, Any], state_dir: Path
) -> tuple[str, set[str], dict[str, str], str | None]:
    """``(cpu eks arn, cpu aliases, {gpu alias -> managed eks arn}, admin email)``.

    A cluster may be named by its EKS ARN or its HyperPod ARN; the site records
    the EKS ARN and the bootstrap checkpoint remembers which HyperPod ARN
    discovery mapped it to, so both spellings resolve without an AWS call.
    """

    spec = existing.get("spec")
    cpu_value = spec.get("cpu") if isinstance(spec, Mapping) else None
    clusters = spec.get("clusters") if isinstance(spec, Mapping) else None
    if not isinstance(cpu_value, Mapping) or not isinstance(clusters, list):
        raise SiteConfigError("existing site has no valid cluster identity")
    cpu_eks = str(cpu_value.get("eksArn") or "")
    hints = load_hyperpod_hints(state_dir)
    aliases: dict[str, str] = {}
    for item in clusters:
        eks = str(item.get("eksClusterArn") or "") if isinstance(item, Mapping) else ""
        if not eks:
            continue
        aliases[eks] = eks
        if hints.get(eks):
            aliases[hints[eks]] = eks
    cpu_aliases = {cpu_eks} | ({hints[cpu_eks]} if hints.get(cpu_eks) else set())
    notifications = spec.get("notifications") if isinstance(spec, Mapping) else None
    email = (
        notifications.get("adminEmail") if isinstance(notifications, Mapping) else None
    )
    return cpu_eks, cpu_aliases, aliases, (str(email) if email else None)


def resolve_deploy_identity(
    arguments: argparse.Namespace, state_dir: Path, *, hooks: DeployHooks
) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
    """``(cpu arn, gpu arns, admin email, gpu delta)`` for this deploy.

    A first deploy needs all three inputs on the command. On a managed site
    they are optional and read from ``site.yaml``; given, the CPU must be the
    site's, and the GPU set must be the managed set or a strict superset of
    it -- the extra ARNs are the delta the deploy joins after the release. A
    subset is refused naming ``remove-cluster``. A HyperPod ARN the checkpoint
    does not know costs one discovery read to map to its EKS cluster.
    """

    existing = load_existing_site(state_dir)
    cpu_arn = str(getattr(arguments, "cpu_cluster_arn", None) or "")
    gpu_arns = tuple(str(value) for value in getattr(arguments, "gpu_cluster_arn", []))
    admin_email = str(getattr(arguments, "alert_email", None) or "")
    if existing is None:
        if not cpu_arn or not gpu_arns:
            raise SiteConfigError(
                "deploy requires --cpu-cluster-arn and at least one --gpu-cluster-arn"
            )
        if not admin_email:
            raise SiteConfigError("deploy requires --admin-email")
        return cpu_arn, gpu_arns, admin_email, ()
    cpu_eks, cpu_aliases, aliases, site_email = managed_site_deploy_inputs(
        existing, state_dir
    )
    if cpu_arn and cpu_arn not in cpu_aliases:
        raise SiteConfigError(
            "requested CPU cluster identity differs from the existing site; "
            "the CPU control plane of a site cannot change"
        )
    managed = sorted(set(aliases.values()))
    delta: tuple[str, ...] = ()
    if gpu_arns:
        resolved = {}
        for arn in gpu_arns:
            if arn in aliases:
                resolved[arn] = aliases[arn]
            elif arn.startswith("arn:aws:sagemaker:"):
                resolved[arn] = str(
                    hooks.discover_cluster(
                        CommandRunner(),
                        cluster_arn=arn,
                        role="gpu",
                        context="gpu-fault-admin-deploy",
                    ).eks_arn
                )
            else:
                resolved[arn] = arn
        covered = set(resolved.values())
        missing = [eks for eks in managed if eks not in covered]
        if missing:
            raise SiteConfigError(
                "requested cluster identity differs from the existing site: the "
                "site manages GPU clusters the command omits ("
                + ", ".join(missing)
                + "); deploy accepts the managed set or a superset of it, use "
                "remove-cluster to detach a cluster"
            )
        delta = tuple(arn for arn in gpu_arns if resolved[arn] not in managed)
    else:
        gpu_arns = tuple(managed)
    admin_email = admin_email or (site_email or "")
    if not admin_email:
        raise SiteConfigError(
            "managed site records no administrator email; pass --admin-email"
        )
    return cpu_arn or cpu_eks, gpu_arns, admin_email, delta


def refuse_rollback_options(arguments: argparse.Namespace) -> None:
    present = [
        flag
        for flag, value in (
            ("--cpu-cluster-arn", getattr(arguments, "cpu_cluster_arn", None)),
            ("--gpu-cluster-arn", getattr(arguments, "gpu_cluster_arn", None)),
            ("--admin-email", getattr(arguments, "alert_email", None)),
            (
                "--approve-profile-plan",
                getattr(arguments, "approve_profile_plan", None),
            ),
            (
                "--accept-schema-change",
                getattr(arguments, "accept_schema_change", False),
            ),
            (
                "--accept-schema-change-without-snapshot",
                getattr(arguments, "accept_schema_change_without_snapshot", False),
            ),
            (
                "--supersede-failed-transaction",
                getattr(arguments, "supersede_failed_transaction", False),
            ),
        )
        if value
    ]
    if present:
        raise SiteConfigError(
            "deploy --rollback takes no other deploy option; remove "
            + ", ".join(present)
        )


def run_deploy_rollback(
    arguments: argparse.Namespace, state_dir: Path, *, hooks: DeployHooks
) -> int:
    refuse_rollback_options(arguments)
    site_file = state_dir / "site.yaml"
    if not site_file.is_file():
        raise SiteConfigError(
            "deploy --rollback requires a managed site under --state-dir"
        )
    return int(
        hooks.run_rollback(
            hooks.load_site(site_file, repository_root=arguments.repo_root),
            state_dir=state_dir,
        )
    )


def approve_pending_profile_plan(
    arguments: argparse.Namespace, state_dir: Path, *, hooks: DeployHooks
) -> None:
    """``--approve-profile-plan SHA --reference REF``: approve, then deploy on."""

    plan_sha256 = getattr(arguments, "approve_profile_plan", None)
    reference = getattr(arguments, "reference", None)
    if not plan_sha256:
        if reference:
            raise SiteConfigError(
                "--reference is only used with --approve-profile-plan"
            )
        return
    if not reference:
        raise SiteConfigError("--approve-profile-plan requires --reference")
    record = hooks.approve_profile_plan_inline(
        state_dir, plan_sha256=str(plan_sha256), reference=str(reference)
    )
    print(
        f"gpu-fault-admin: Runtime Profile plan {record['plan_sha256']} approved at "
        f"{record['approved_at']} by {record['approver_identity']} "
        f"(reference {record['reference']}); continuing the deploy",
        file=sys.stderr,
        flush=True,
    )


def source_deploy_applied_release(state_dir: Path) -> bool:
    """Whether the source deploy that just returned ran an application release."""

    try:
        value = json.loads(
            (state_dir / SOURCE_DEPLOY_SUCCESS_STATE).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("mode") == "APPLICATION_RELEASE"


def join_gpu_clusters(
    site_file: Path,
    gpu_cluster_arns: Sequence[str],
    *,
    repository_root: Path | None,
    hooks: DeployHooks,
) -> None:
    site = hooks.load_site(site_file, repository_root=repository_root)
    result = hooks.join_clusters(
        tuple(
            JoinClusterRequest(site=site, gpu_cluster_arn=gpu_cluster_arn)
            for gpu_cluster_arn in gpu_cluster_arns
        )
    )
    if result:
        print(json.dumps(result, indent=2, sort_keys=True))


def run_public_deploy(
    arguments: argparse.Namespace,
    state_dir: Path,
    *,
    cpu_cluster_arn: str,
    gpu_cluster_arns: tuple[str, ...],
    admin_email: str,
    delta: tuple[str, ...],
    hooks: DeployHooks,
) -> int:
    """The operator's hop: source preparation, then the GPU delta if any.

    An application release joins the delta itself right after the release
    (``pending_gpu_cluster_arns``); a deploy that applied no release -- the
    site would NOOP -- skips the 40-minute rollout and joins the delta directly.
    """

    status = int(
        hooks.run_source_deploy(
            cpu_cluster_arn=cpu_cluster_arn,
            gpu_cluster_arns=gpu_cluster_arns,
            state_dir=arguments.state_dir,
            admin_email=admin_email,
            impact_base=getattr(arguments, "impact_base", "origin/main"),
            current_directory=Path.cwd(),
            extra_environment={
                **hooks.release_consent_environment(arguments),
                **hooks.grafana_environment(arguments),
            },
            wait_for_email_confirmation=int(
                getattr(arguments, "wait_for_email_confirmation", 0) or 0
            ),
        )
    )
    if status or not delta or source_deploy_applied_release(state_dir):
        return status
    join_gpu_clusters(state_dir / "site.yaml", delta, repository_root=None, hooks=hooks)
    return 0


def run_prepared_deploy(
    arguments: argparse.Namespace,
    *,
    cpu_cluster_arn: str,
    gpu_cluster_arns: tuple[str, ...],
    admin_email: str,
    hooks: DeployHooks,
) -> int:
    """The inner hop inside the prepared snapshot: bootstrap, release, join."""

    repository_root = (arguments.repo_root or Path.cwd()).resolve()
    staging_only = bool(getattr(arguments, "staging_only_release", False))
    with administrator_operation_lock(arguments.state_dir):
        bootstrap_result = hooks.bootstrap_from_arns(
            BootstrapRequest(
                cpu_cluster_arn=cpu_cluster_arn,
                gpu_cluster_arns=gpu_cluster_arns,
                repository_root=repository_root,
                state_dir=arguments.state_dir.expanduser(),
                alert_email=admin_email,
                staging_only_release=staging_only,
                impact_base=getattr(arguments, "impact_base", "origin/main"),
                **hooks.grafana_request_fields(arguments),
            )
        )
    status = int(
        hooks.run_automatic_release(
            repository_root=repository_root,
            site_file=bootstrap_result.site_file,
            state_dir=arguments.state_dir,
            staging_only_release=staging_only,
        )
    )
    if status:
        return status
    if bootstrap_result.pending_gpu_cluster_arns:
        join_gpu_clusters(
            bootstrap_result.site_file,
            bootstrap_result.pending_gpu_cluster_arns,
            repository_root=repository_root,
            hooks=hooks,
        )
    return 0


def run_deploy(arguments: argparse.Namespace, *, hooks: DeployHooks) -> int:
    """Dispatch the one deploy command to its action."""

    if arguments.state_dir is None:
        raise SiteConfigError("deploy requires --state-dir")
    state_dir = arguments.state_dir.expanduser().resolve()
    if getattr(arguments, "rollback", False):
        return run_deploy_rollback(arguments, state_dir, hooks=hooks)
    approve_pending_profile_plan(arguments, state_dir, hooks=hooks)
    # The admin-config refusal ("existing sites must use gpu-fault-admin
    # config") is about the operator's input and is reported before any
    # identity comparison, so a wrong --config is never masked by an ARN typo.
    initialize_desired_admin_config(
        arguments.state_dir,
        config_file=getattr(arguments, "admin_config_file", None),
        permit_change=not (state_dir / "site.yaml").is_file(),
    )
    cpu_cluster_arn, gpu_cluster_arns, admin_email, delta = resolve_deploy_identity(
        arguments, state_dir, hooks=hooks
    )
    if not getattr(arguments, "prepared_source_release", False):
        return run_public_deploy(
            arguments,
            state_dir,
            cpu_cluster_arn=cpu_cluster_arn,
            gpu_cluster_arns=gpu_cluster_arns,
            admin_email=admin_email,
            delta=delta,
            hooks=hooks,
        )
    return run_prepared_deploy(
        arguments,
        cpu_cluster_arn=cpu_cluster_arn,
        gpu_cluster_arns=gpu_cluster_arns,
        admin_email=admin_email,
        hooks=hooks,
    )
