from __future__ import annotations

import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.cluster_join_state import complete_step, step_done
from gpu_fault.admin.cluster_join_context import clear_stale_installer_annotations

if TYPE_CHECKING:
    from gpu_fault.admin.cluster_join import JoinClusterRequest


def prepare_local_inputs(
    request: JoinClusterRequest,
    *,
    runner: CommandRunner,
    target: ClusterIdentity,
    cluster_id: str,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    gpu_kubeconfig = join._gpu_kubeconfig(request.site)
    cpu_kubeconfig = (
        Path(str(request.site.release_config["cpu_kubeconfig"])).expanduser().resolve()
    )
    if gpu_kubeconfig == cpu_kubeconfig:
        raise join.JoinTargetIdentityError(
            "GPU join must not modify the CPU kubeconfig"
        )
    secure = state_dir / "secure"
    secure.mkdir(mode=0o700, parents=True, exist_ok=True)
    site_secure = request.site.source.parent / "secure"
    site_secure.mkdir(mode=0o700, parents=True, exist_ok=True)
    token_file = site_secure / f"{cluster_id}.token"
    if token_file.exists() and not step_done(state, "LOCAL_INPUTS_STARTED"):
        expected_token = {
            "path": str(token_file),
            "cluster_id": cluster_id,
            "eks_arn": target.eks_arn,
            "hyperpod_arn": target.hyperpod_arn,
        }
        if state.get("retained_cluster_token") != expected_token:
            raise join.JoinTargetIdentityError(
                "existing cluster token has no matching join ownership"
            )
    if not step_done(state, "LOCAL_INPUTS_STARTED") and gpu_kubeconfig.is_file():
        contexts = runner.run(
            [
                "kubectl",
                "--kubeconfig",
                str(gpu_kubeconfig),
                "config",
                "get-contexts",
                "-o",
                "name",
            ]
        )
        if target.context in contexts.splitlines():
            raise join.JoinTargetIdentityError(
                "GPU join refuses an existing kubeconfig alias"
            )
    local = dict(
        (state.get("evidence") or {}).get("LOCAL_INPUTS_STARTED")
        or {
            "gpu_kubeconfig": str(gpu_kubeconfig),
            "token_file": str(token_file),
            "fleet_master_file": str(secure / "fleet-master"),
            "namespace_creation_started": False,
            "nodes": [],
        }
    )
    complete_step(state_path, state, "LOCAL_INPUTS_STARTED", local)
    gpu_kubeconfig.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with join._KUBECONFIG_THREAD_LOCK:
        runner.run(
            [
                "aws",
                "eks",
                "update-kubeconfig",
                "--region",
                target.region,
                "--name",
                target.eks_name,
                "--kubeconfig",
                str(gpu_kubeconfig),
                "--alias",
                target.context,
            ],
            mutate=True,
            capture=False,
        )
    if gpu_kubeconfig.exists():
        gpu_kubeconfig.chmod(0o600)
    namespace_uid = join.probe_join_namespace(
        runner,
        site=request.site,
        kubeconfig=gpu_kubeconfig,
        context=target.context,
    )
    if namespace_uid is not None and namespace_uid != local.get("namespace_uid"):
        raise BootstrapError("GPU join refuses an existing or replaced namespace")
    if namespace_uid is None:
        local["namespace_creation_started"] = True
        complete_step(state_path, state, "LOCAL_INPUTS_STARTED", local)
        join.ensure_namespace(
            runner,
            kubeconfig=gpu_kubeconfig,
            context=target.context,
            namespace=str(request.site.release_config["namespace"]),
        )
        namespace_uid = join.probe_join_namespace(
            runner,
            site=request.site,
            kubeconfig=gpu_kubeconfig,
            context=target.context,
        )
        if namespace_uid is None:
            raise BootstrapError("created GPU join namespace is not observable")
        local["namespace_uid"] = namespace_uid
        complete_step(state_path, state, "LOCAL_INPUTS_STARTED", local)
    # The cluster token survives registered rollback; the transaction's fleet
    # master copy is disposable and must never remove the site's own copy.
    join.write_secret(token_file, secrets.token_hex(32))
    fleet_master_file = join._ensure_base_secrets(
        runner,
        cpu_kubeconfig=cpu_kubeconfig,
        namespace=str(request.site.release_config["namespace"]),
        secure_dir=secure,
    )
    ca_file = join._shared_ca_file(request.site)
    node_uids: dict[str, str] = {}
    nodes = join._list_nodes(
        runner, kubeconfig=gpu_kubeconfig, target=target, identities=node_uids
    )
    clear_stale_installer_annotations(
        request.site,
        {
            "context": target.context,
            "hyperpod_cluster_name": target.hyperpod_name,
            "expected_node_uids": node_uids,
        },
        nodes,
        kubeconfig=gpu_kubeconfig,
    )
    complete_step(
        state_path,
        state,
        "LOCAL_INPUTS_READY",
        {
            **local,
            "fleet_master_file": str(fleet_master_file),
            "ca_file": str(ca_file),
            "nodes": nodes,
            "node_uids": node_uids,
        },
    )
