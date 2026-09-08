"""The inputs and first-minute checks of one staging deploy.

Split from ``staging_deploy.py``: what the command needs to know before the
source scan, the gates and the build -- the cluster ARNs and the administrator
email (from the command or from ``site.yaml``), the SES/SNS confirmation check,
and the consent refusal the outer hop can already make from the quick status
report and a reused release manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence, cast

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.bootstrap import discover_cluster
from gpu_fault.admin.bootstrap_checkpoint import load_hyperpod_hints
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapState,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_site import site_identifier
from gpu_fault.admin.deploy_consent import consent_refusal, load_release_manifest
from gpu_fault.admin.notification_precheck import (
    EmailConfirmation,
    await_email_confirmations,
    check_email_confirmations,
    email_confirmation_refusal,
)

if __package__:
    from scripts.staging_state_hygiene import SourceCheckout, StagingDeployError
else:
    from staging_state_hygiene import SourceCheckout, StagingDeployError


def managed_site_inputs(state_dir: Path) -> dict[str, object] | None:
    """The deploy inputs a managed site already knows, or None without a site.

    ``deploy --state-dir X`` is the whole upgrade command: the CPU ARN, the GPU
    ARNs and the administrator email are read from ``site.yaml`` when the
    command does not give them. Read as YAML rather than through ``load_site``
    so a site whose recorded repository snapshot was pruned still answers.
    """

    site_file = state_dir / "site.yaml"
    if not site_file.is_file():
        return None
    try:
        document = yaml.safe_load(site_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StagingDeployError("managed site.yaml is unreadable") from exc
    spec = document.get("spec") if isinstance(document, Mapping) else None
    spec = spec if isinstance(spec, Mapping) else {}
    cpu = spec.get("cpu")
    cpu_arn = str(cpu.get("eksArn") or "") if isinstance(cpu, Mapping) else ""
    clusters = spec.get("clusters")
    gpu_arns = tuple(
        str(item["eksClusterArn"])
        for item in (clusters if isinstance(clusters, list) else [])
        if isinstance(item, Mapping) and item.get("eksClusterArn")
    )
    notifications = spec.get("notifications")
    admin_email = (
        notifications.get("adminEmail") if isinstance(notifications, Mapping) else None
    )
    auto_rollback = spec.get("autoRollback")
    # A site the fields cannot be read from (a stub, an older document) supplies
    # nothing; the command's own inputs are then required as on a first deploy.
    return {
        "cpu_cluster_arn": cpu_arn or None,
        "gpu_cluster_arns": gpu_arns,
        "admin_email": str(admin_email) if admin_email else None,
        "auto_rollback": auto_rollback if isinstance(auto_rollback, bool) else True,
    }


def resolve_deploy_inputs(
    arguments: argparse.Namespace, *, state_dir: Path
) -> tuple[str, tuple[str, ...], str]:
    """``(cpu_cluster_arn, gpu_cluster_arns, admin_email)``: the command, else the site.

    A first deploy (no ``site.yaml``) needs all three on the command; an upgrade
    needs none. The resolved values are written back onto ``arguments`` so the
    apply hands the inner ``gpu-fault-admin deploy`` a complete command.
    """

    site = managed_site_inputs(state_dir)
    cpu_arn = str(getattr(arguments, "cpu_cluster_arn", None) or "").strip()
    gpu_arns = tuple(
        str(value) for value in (getattr(arguments, "gpu_cluster_arn", None) or [])
    )
    admin_email = str(getattr(arguments, "admin_email", None) or "").strip()
    if site is not None:
        cpu_arn = cpu_arn or str(site["cpu_cluster_arn"] or "")
        gpu_arns = gpu_arns or cast(tuple[str, ...], site["gpu_cluster_arns"])
        admin_email = admin_email or str(site["admin_email"] or "")
    if not cpu_arn or not gpu_arns:
        raise StagingDeployError(
            "first deploy requires --cpu-cluster-arn and at least one --gpu-cluster-arn"
            if site is None
            else "managed site records no cluster identity; pass --cpu-cluster-arn "
            "and --gpu-cluster-arn"
        )
    if not admin_email:
        raise StagingDeployError(
            "first deploy requires --admin-email"
            if site is None
            else "managed site records no administrator email; pass --admin-email"
        )
    arguments.cpu_cluster_arn = cpu_arn
    arguments.gpu_cluster_arn = list(gpu_arns)
    arguments.admin_email = admin_email
    return cpu_arn, gpu_arns, admin_email


def deploy_rerun_command(
    *,
    state_dir: Path,
    first_deploy: bool,
    cpu_cluster_arn: str,
    gpu_cluster_arns: Sequence[str],
    admin_email: str,
) -> str:
    """The public command the operator reruns: one parameter on a managed site."""

    if not first_deploy:
        return f"gpu-fault-admin deploy --state-dir {state_dir}"
    parts = ["gpu-fault-admin deploy", f"--cpu-cluster-arn {cpu_cluster_arn}"]
    parts.extend(f"--gpu-cluster-arn {arn}" for arn in gpu_cluster_arns)
    parts.append(f"--state-dir {state_dir}")
    parts.append(f"--admin-email {admin_email}")
    return " ".join(parts)


def precheck_email_confirmations(
    *,
    state_dir: Path,
    cpu_cluster_arn: str,
    admin_email: str,
    wait_minutes: int,
    rerun_command: str,
    runner: CommandRunner | None = None,
) -> EmailConfirmation:
    """The first-minute SES/SNS check; raises with both addresses when unconfirmed.

    Discovers the CPU cluster (the site id and Region come from it), sends the
    two confirmation mails once, and stops -- exit status 2 -- with one message
    naming both addresses and the rerun line. ``wait_minutes`` polls instead.
    """

    active_runner = runner or CommandRunner()
    try:
        cpu = discover_cluster(
            active_runner,
            cluster_arn=cpu_cluster_arn,
            role="cpu",
            context="gpu-fault-admin-precheck",
            hyperpod_hints=load_hyperpod_hints(state_dir),
        )
        site_id = site_identifier(cpu, ())
        state = BootstrapState(state_dir / "bootstrap-state.json", site_id=site_id)

        def check() -> EmailConfirmation:
            return check_email_confirmations(
                active_runner,
                cpu=cpu,
                site_id=site_id,
                admin_email=admin_email,
                state=state,
            )

        result = await_email_confirmations(check, wait_minutes=wait_minutes)
    except BootstrapError as exc:
        raise StagingDeployError(f"email notification check failed: {exc}") from exc
    print(
        "+ email notifications: " + json.dumps(result.as_dict(), sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    if not result.confirmed:
        raise StagingDeployError(
            email_confirmation_refusal(result, rerun_command=rerun_command)
        )
    return result


def live_release_block(report: Mapping[str, object] | None) -> dict[str, object]:
    live = (report or {}).get("live_release")
    return dict(live) if isinstance(live, Mapping) else {}


def next_deploy_block(report: Mapping[str, object] | None) -> dict[str, object]:
    value = (report or {}).get("next_deploy")
    return dict(value) if isinstance(value, Mapping) else {}


def early_consent_refusal(
    report: Mapping[str, object] | None,
    *,
    source: SourceCheckout,
    auto_rollback: bool,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """The consent refusal the outer hop can already make, or None.

    The candidate is known here only when the prepared snapshot carries a built
    release (``dist/current-release.json`` -- a reuse); a snapshot still to be
    built is checked by the inner hop the moment the build returns. The live
    schema version is the configured release's when the site would otherwise
    NOOP (configured equals live); anything else is left to the inner hop.
    """

    if report is None:
        return None
    manifest = load_release_manifest(
        source.repository_root / "dist/current-release.json"
    )
    if manifest is None:
        return None
    live = live_release_block(report)
    configured = report.get("configured_release")
    configured = configured if isinstance(configured, Mapping) else {}
    live_schema = (
        configured.get("database_schema_version")
        if next_deploy_block(report).get("kind") == "NOOP"
        else None
    )
    candidate_schema = manifest.get("database_schema_version")
    return consent_refusal(
        live_phase=str(live.get("phase") or ""),
        live_release_id=str(live.get("release_id") or ""),
        candidate_release_id=str(manifest.get("release_id") or ""),
        live_schema_version=live_schema if isinstance(live_schema, int) else None,
        candidate_schema_version=(
            candidate_schema if isinstance(candidate_schema, int) else None
        ),
        auto_rollback=auto_rollback,
        acceptance_recorded=isinstance(live.get("schema_change_acceptance"), Mapping),
        environment=environment,
    )
