"""Refuse a first deploy bootstrap would refuse, before anything is recorded.

The public ``deploy`` records its request (``initial-deploy-request.json``)
first and the email check then creates the site's SNS topic and subscription;
bootstrap discovers the clusters only later, inside the prepared snapshot. A
first deploy that names GPU clusters of both providers was refused there, after
the request had pinned the cluster set: the corrected rerun with one provider
was refused as a conflicting request, and the operator had to start over in a
new state directory (mixed-site BOOT-029, stage 1) -- although the README said
the shape is refused before anything is created.

This runs bootstrap's own discovery (``discover_bootstrap_scope``) read-only on
the request before it is recorded, so every refusal that discovery makes on a
first deploy -- providers mixed, a GPU node group without the site's ownership
label or labelled for another site, a non-GPU instance type, node repair on,
``NodeRecovery`` not ``None``, a cluster that is not ACTIVE, a plain EKS CPU
cluster -- and the two identity checks bootstrap makes right after it (one
account and Region; no GPU cluster named twice, e.g. by its EKS and its
HyperPod ARN) stop the command while the state directory holds no request, no
bootstrap checkpoint and no AWS resource. Bootstrap keeps its own checks.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Sequence

from gpu_fault.admin.bootstrap_checkpoint import load_hyperpod_hints
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_site import discover_bootstrap_scope, require_same_scope

NOTHING_CREATED = (
    "first deploy refused before anything was recorded or created (no "
    "initial-deploy-request.json, no bootstrap checkpoint, no AWS resource); "
    "rerun with corrected inputs"
)
RECORDED_REQUEST = (
    "this state directory already recorded that first deploy request "
    "(initial-deploy-request.json) and resumes only the same clusters: fix "
    "the clusters and rerun, or start the first deploy in a new empty state "
    "directory"
)


def precheck_first_deploy(
    state_dir: Path,
    cpu_arn: str,
    gpu_arns: Sequence[str],
    recorded: bool = False,
    *,
    discover: Callable[..., ClusterIdentity],
    runner: CommandRunner | None = None,
) -> None:
    """Raise ``BootstrapError`` for a first deploy bootstrap would refuse.

    ``recorded`` says the request is a resumed one (an earlier deploy wrote
    it), which the refusal cannot claim was never recorded. ``discover`` is
    the CLI's ``discover_cluster``; only describe/list calls are made, the GPU
    clusters in parallel, as bootstrap does.
    """

    try:
        _existing, cpu, gpu_clusters = discover_bootstrap_scope(
            request=BootstrapRequest(
                cpu_cluster_arn=cpu_arn,
                gpu_cluster_arns=tuple(gpu_arns),
                repository_root=state_dir,
                state_dir=state_dir,
            ),
            runner=runner or CommandRunner(),
            discover=partial(discover, hyperpod_hints=load_hyperpod_hints(state_dir)),
            alias=lambda _arn, role, index: f"gpu-fault-precheck-{role}-{index}",
        )
        require_same_scope(cpu, gpu_clusters)
        seen: dict[tuple[str, str], str] = {}
        for cluster in gpu_clusters:
            key = (cluster.eks_arn, cluster.hyperpod_name)
            if key in seen:
                raise BootstrapError(
                    "initial deploy target contains duplicate GPU clusters: "
                    f"{seen[key]} and {cluster.input_arn} name the same cluster"
                )
            seen[key] = cluster.input_arn
    except BootstrapError as exc:
        raise BootstrapError(
            f"{exc}\n{RECORDED_REQUEST if recorded else NOTHING_CREATED}"
        ) from exc
