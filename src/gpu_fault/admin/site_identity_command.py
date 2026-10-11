"""``gpu-fault-admin site-id``: the site id a CPU cluster derives, printed up front.

The ownership label (``cluster_scope.OWNERSHIP_LABEL``) carries the site id
as its value, and the administrator applies it to the GPU node groups before
the first ``deploy`` -- which is before any state directory exists. The id is
deterministic (``bootstrap_site.site_identifier``: Region, HyperPod name and
the first eight hex digits of the HyperPod ARN's digest), so this read-only
verb derives it from the CPU cluster ARN alone with the same discovery the
deploy will run, and prints the label and the provider commands that apply
it. Given ``--state-dir`` of a managed site it also confirms the recorded
site name is that id, so a script labelling node groups for an existing site
cannot use another site's id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from gpu_fault.admin.bootstrap import discover_cluster
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_ownership import label_remedy
from gpu_fault.admin.bootstrap_site import load_existing_site, site_identifier
from gpu_fault.cluster_scope import (
    HYPERPOD_CLUSTER_LABEL,
    OWNERSHIP_LABEL,
    ClusterProvider,
    ownership_node_selector,
)

__all__ = ["add_site_id_command", "run_site_id_command", "site_id_document"]

COMMAND = "site-id"


def add_site_id_command(commands: Any) -> None:
    parser = commands.add_parser(
        COMMAND,
        usage=f"gpu-fault-admin {COMMAND} --cpu-cluster-arn CPU_ARN [--state-dir STATE_DIR]",
        help=(
            "print the site id a CPU cluster derives and the ownership label "
            f"({OWNERSHIP_LABEL}=<site id>) to put on the GPU node groups before "
            "the first deploy"
        ),
        description=(
            "Read-only: derives the deterministic site id of the CPU control plane "
            "named by --cpu-cluster-arn (the same discovery deploy runs) and prints "
            "the ownership label value and the provider commands that apply it to "
            "EKS managed node groups, HyperPod instance groups and Karpenter "
            "NodePools. With --state-dir of a managed site the recorded site name "
            "must be that id."
        ),
    )
    parser.add_argument(
        "--cpu-cluster-arn",
        required=True,
        metavar="CPU_ARN",
        help="the CPU control plane's EKS or HyperPod cluster ARN",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        metavar="STATE_DIR",
        help=(
            "a managed site's private state directory: its recorded site name "
            "must equal the derived id (also what a bound deploy host requires)"
        ),
    )


def site_id_document(
    cpu: ClusterIdentity, *, site_id: str, state_dir: Path | None
) -> dict[str, Any]:
    """The JSON the verb prints: id, label, selectors, how to apply, cross-check."""

    hyperpod_group = "<instance group>"
    document: dict[str, Any] = {
        "site_id": site_id,
        "ownership_label": {"key": OWNERSHIP_LABEL, "value": site_id},
        "node_selectors": {
            ClusterProvider.EKS_MANAGED.value: ownership_node_selector(
                site_id, provider=ClusterProvider.EKS_MANAGED
            ),
            ClusterProvider.HYPERPOD.value: (
                f"{OWNERSHIP_LABEL}={site_id},{HYPERPOD_CLUSTER_LABEL}=<HyperPod name>"
            ),
        },
        "apply": {
            "eks_managed_node_group": label_remedy(
                ClusterProvider.EKS_MANAGED,
                cluster="<EKS cluster>",
                site_id=site_id,
                group="<node group>",
            ),
            "hyperpod_instance_group": label_remedy(
                ClusterProvider.HYPERPOD,
                cluster="<HyperPod cluster>",
                site_id=site_id,
                group=hyperpod_group,
            ),
            "karpenter_nodepool": (
                f"spec.template.metadata.labels: {{{OWNERSHIP_LABEL}: {site_id}}}"
            ),
        },
        "cpu_cluster": {
            "input_arn": cpu.input_arn,
            "hyperpod_arn": cpu.hyperpod_arn,
            "hyperpod_name": cpu.hyperpod_name,
            "eks_arn": cpu.eks_arn,
            "region": cpu.region,
        },
    }
    if state_dir is not None:
        recorded = load_existing_site(state_dir.expanduser().resolve())
        name = str(((recorded or {}).get("metadata") or {}).get("name") or "")
        document["recorded_site"] = {
            "state_dir": str(state_dir),
            "site_name": name or None,
            "matches": bool(name) and name == site_id,
        }
    return document


def run_site_id_command(
    arguments: argparse.Namespace,
    *,
    runner: CommandRunner | None = None,
    discover: Callable[..., ClusterIdentity] = discover_cluster,
) -> int:
    """Print the site id document; exit 1 when a managed site records another id."""

    cpu = discover(
        runner or CommandRunner(),
        cluster_arn=str(arguments.cpu_cluster_arn),
        role="cpu",
        context="gpu-fault-admin-site-id",
    )
    if cpu.provider != ClusterProvider.HYPERPOD.value:
        raise BootstrapError(
            "the CPU control plane must be a HyperPod EKS cluster; "
            f"{arguments.cpu_cluster_arn} has no HyperPod orchestrator"
        )
    site_id = site_identifier(cpu, [])
    state_dir = getattr(arguments, "state_dir", None)
    document = site_id_document(cpu, site_id=site_id, state_dir=state_dir)
    print(
        f"gpu-fault-admin {COMMAND}: {site_id} (label {OWNERSHIP_LABEL}={site_id})",
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    recorded = document.get("recorded_site")
    if (
        isinstance(recorded, dict)
        and recorded.get("site_name")
        and not recorded.get("matches")
    ):
        print(
            f"gpu-fault-admin {COMMAND}: {state_dir} records site "
            f"{recorded['site_name']}, not {site_id}; the ARN is not this site's "
            "CPU control plane",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0
