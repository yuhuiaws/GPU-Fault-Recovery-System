from __future__ import annotations

import json
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
from gpu_fault.admin.cluster_kubeconfig import (
    KubeconfigAliasRefused,
    reclaim_released_alias,
)
from gpu_fault.admin.site import RenderedSite, effective_environment
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
    alias_replaced: dict[str, Any] | None = None
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
            # A cluster joined, removed and joined again derives the identical
            # alias. The alias is taken over only when it provably names this
            # EKS cluster (endpoint and CA) and the site has released the
            # cluster; anything else is still the fail-closed refusal.
            try:
                alias_replaced = reclaim_released_alias(
                    runner,
                    site=request.site,
                    target=target,
                    cluster_id=cluster_id,
                    kubeconfig=gpu_kubeconfig,
                )
            except KubeconfigAliasRefused as exc:
                raise join.JoinTargetIdentityError(
                    f"GPU join refuses an existing kubeconfig alias: {exc}"
                ) from exc
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
    if alias_replaced is not None:
        local["kubeconfig_alias_replaced"] = alias_replaced
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
    keep_primary_current_context(runner, request.site, gpu_kubeconfig, target.context)
    verify_allowed_namespaces_exist(
        runner,
        site=request.site,
        kubeconfig=gpu_kubeconfig,
        context=target.context,
        cluster_id=cluster_id,
        namespaces=request.allowed_namespaces,
    )
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
            "provider": target.provider,
            "node_selector": target.node_selector,
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


def verify_allowed_namespaces_exist(
    runner: CommandRunner,
    *,
    site: RenderedSite,
    kubeconfig: Path,
    context: str,
    cluster_id: str,
    namespaces: tuple[str, ...],
) -> None:
    """Every allowed workload namespace must already exist on the GPU cluster.

    The release renders a Role/RoleBinding per allowed namespace for the
    completion watcher and applies them with a server dry-run; a namespace
    that does not exist fails that apply deep inside the join-cluster release
    (live 2026-10-04: ``namespaces "training" not found`` on gpu-eks-cluster,
    one rollback later). This is the first read against the GPU cluster, before
    the join creates anything on it, so the operator gets the answer and the
    remedy up front. The control-plane namespace is the join's own to create.
    """

    own = str(site.release_config["namespace"])
    missing: list[str] = []
    for namespace in sorted(set(namespaces) - {own}):
        output = runner.run(
            [
                "kubectl",
                "--kubeconfig",
                str(kubeconfig),
                "--context",
                context,
                "get",
                "namespace",
                namespace,
                "--ignore-not-found",
                "-o",
                "json",
            ],
            env=effective_environment(site),
        )
        document = json.loads(output) if output.strip() else None
        name = (
            (document.get("metadata") or {}).get("name")
            if isinstance(document, dict) and document.get("kind") == "Namespace"
            else None
        )
        if name != namespace:
            missing.append(namespace)
    if missing:
        listed = ", ".join(missing)
        raise BootstrapError(
            f"GPU join refuses {cluster_id}: allowed workload namespace(s) do not "
            f"exist on the cluster: {listed}. Create them first "
            f"(kubectl --context {context} create namespace <name>) or join with "
            "--allowed-namespace <name> naming only namespaces that exist; the "
            f"control-plane namespace {own} is created by the join itself"
        )


def keep_primary_current_context(
    runner: CommandRunner, site: RenderedSite, kubeconfig: Path, joining: str
) -> None:
    """``aws eks update-kubeconfig`` makes the new alias the current-context;
    the shared GPU kubeconfig keeps pointing at the site's primary cluster.

    Live 2026-10-04: after the plain EKS join, ``gpu.kubeconfig``'s
    current-context was the EKS cluster, so any tool reading KUBECONFIG without
    ``--context`` silently targeted the wrong cluster. The join's own commands
    always pass ``--context``; the default stays with the first managed cluster.
    """

    clusters = site.release_config.get("clusters") or []
    primary = str(clusters[0]["context"]) if clusters else ""
    if not primary or primary == joining:
        return
    prefix = ["kubectl", "--kubeconfig", str(kubeconfig), "config"]
    contexts = runner.run(
        [*prefix, "get-contexts", "-o", "name"], env=effective_environment(site)
    )
    if primary not in str(contexts).splitlines():
        return
    runner.run(
        [*prefix, "use-context", primary], env=effective_environment(site), mutate=True
    )
