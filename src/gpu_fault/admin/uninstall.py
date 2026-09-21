from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.aws_cleanup import (
    ResourceCleaner,
    delete_priority,
    is_aurora_resource,
)
from gpu_fault.admin.bootstrap_common import (
    Arn,
    BootstrapError,
    CommandRunner,
)
from gpu_fault.admin.execution import cleanup_deadline, run_command as bounded_command
from gpu_fault.admin.installation_lifecycle import (
    bind_retained_database,
    uninstall_state as _uninstall_state,
)
from gpu_fault.admin.membership_lock import (
    membership_operation_lock,
    reload_site_for_mutation,
)
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.admin.resource_registry import (
    LegacyInstallationRegistryMissing,
    build_legacy_installation_snapshot,
    fetch_installation_resource_registry,
    load_installation_resource_snapshot,
    sync_installation_resource_snapshot,
    sync_installation_resource_snapshot_direct,
    write_installation_resource_snapshot,
)
from gpu_fault.admin.site import RenderedSite, materialized_release_config
from gpu_fault.admin.uninstall_spares import release_declared_spares
from gpu_fault.admin.uninstall_override import (
    cleanup_shell_environment,
    record_repository_root_override,
)
from gpu_fault.admin.uninstall_types import (
    UNINSTALL_PHASES as UNINSTALL_PHASES,
    CpuDisposition as CpuDisposition,
    UninstallRequest as UninstallRequest,
)
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)

LEGACY_EXTERNAL_RESOURCE_TYPES = frozenset(
    {
        "cpu_eks",
        "cpu_hyperpod",
        "gpu_eks",
        "gpu_hyperpod",
        "ec2_subnet",
        "internet_gateway",
        "iam_oidc_provider",
        "eks_addon",
    }
)
DETACH_RESOURCE_TYPES = frozenset(
    {
        "ec2_route_table_association",
        "route53_vpc_association",
        "sns_subscription",
        "sqs_policy_binding",
        "eks_pod_identity_association",
        "iam_oidc_provider",
    }
)
CPU_CLUSTER_BOUND_RESOURCE_KEYS = frozenset(
    {
        "aws/eks/pod-identity-agent",
    }
)


def _required_confirmation(disposition: CpuDisposition) -> str:
    return (
        "DELETE_CPU_CONTROL_PLANE" if disposition == "delete" else "UNINSTALL_GPU_FAULT"
    )


def _run_cleanup(
    request: UninstallRequest,
    runner: CommandRunner,
    state_file: Path,
) -> None:
    with materialized_release_config(request.site) as config:
        environment = cleanup_shell_environment(request, Path(config))
        try:
            runner.run(
                [
                    str(
                        request.site.repository_root
                        / "deploy/control-plane/regional/prepare-clean-redeploy.sh"
                    ),
                    "--config",
                    str(config),
                    "--scope",
                    "all",
                    "--mode",
                    "reset",
                    "--node-mode",
                    "uninstall",
                    "--state-file",
                    str(state_file),
                    "--confirm-reset",
                    "RESET_GPU_FAULT_INSTALLATION",
                    "--execute",
                ],
                cwd=request.site.repository_root,
                env=environment,
                mutate=True,
                capture=False,
            )
        except BaseException as exc:
            if state_file.is_file():
                try:
                    with cleanup_deadline("uninstall node cleanup", 120):
                        runner.run(
                            [
                                "python3",
                                str(
                                    request.site.repository_root
                                    / "deploy/control-plane/tools/cleanup_kubernetes.py"
                                ),
                                "--config",
                                str(config),
                                "--state-file",
                                str(state_file),
                                "--timeout-seconds",
                                "90",
                                "cleanup-owned",
                            ],
                            cwd=request.site.repository_root,
                            env=environment,
                            mutate=True,
                            timeout_seconds=100,
                        )
                except BaseException as cleanup_error:
                    exc.add_note(
                        "temporary node cleanup remains unverified: "
                        + type(cleanup_error).__name__
                    )
            raise


def _kubectl_exists(command: list[str], *, description: str) -> bool:
    try:
        result = bounded_command(
            [*command, "--ignore-not-found", "-o", "name", "--request-timeout=30s"],
            timeout_seconds=45,
        )
    except (subprocess.TimeoutExpired, TimeoutError):
        raise BootstrapError(
            f"Kubernetes cleanup verification timed out for {description}"
        ) from None
    if result.returncode:
        # Authentication helpers can also report "not found". Only a successful
        # --ignore-not-found response is evidence of Kubernetes object absence.
        raise BootstrapError(
            f"Kubernetes cleanup verification failed for {description} "
            f"(exit {result.returncode})"
        )
    return bool(result.stdout.strip())


def _kubectl_prefix(
    site: RenderedSite,
    *,
    plane: str,
    context: str,
) -> list[str]:
    if plane == "cpu":
        return [
            "kubectl",
            "--kubeconfig",
            str(site.release_config["cpu_kubeconfig"]),
        ]
    command = ["kubectl"]
    gpu_kubeconfig = site.release_config.get("gpu_kubeconfig") or site.environment.get(
        "KUBECONFIG"
    )
    if gpu_kubeconfig:
        command.extend(["--kubeconfig", str(gpu_kubeconfig)])
    command.extend(["--context", context])
    return command


def verify_installed_registry_cleanup(
    site: RenderedSite,
    cleanup_state: dict[str, Any],
    *,
    cpu_deleted: bool = False,
) -> dict[str, int]:
    config = site.release_config
    namespace = str(config["namespace"])
    inventory = cleanup_state["inventory_snapshot"]
    checked = 0
    cluster_contexts = [str(item["context"]) for item in config["clusters"]]
    for plane, section in (("cpu", inventory["cpu"]), ("gpu", inventory["gpu"])):
        if plane == "cpu" and cpu_deleted:
            checked += len(section["resources"])
            continue
        contexts = ["cpu"] if plane == "cpu" else cluster_contexts
        for context in contexts:
            prefix = _kubectl_prefix(
                site,
                plane=plane,
                context=context,
            )
            if (
                plane == "gpu"
                and "by_context" in section
                and context not in section["by_context"]
            ):
                raise BootstrapError("cleanup inventory lacks a selected GPU context")
            resources = section.get("by_context", {}).get(context, section)["resources"]
            grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
            namespaces = {namespace}
            for resource in resources:
                resource_namespace = (
                    str(resource.get("namespace") or namespace)
                    if resource["scope"] == "namespaced"
                    else ""
                )
                if resource_namespace:
                    namespaces.add(resource_namespace)
                grouped[(resource_namespace, resource["kind"])].add(resource["name"])
            for target_namespace in sorted(namespaces):
                registry_description = (
                    f"{plane}:{context}:{target_namespace}:"
                    "configmap/gpu-fault-installed-resources"
                )
                if _kubectl_exists(
                    [
                        *prefix,
                        "-n",
                        target_namespace,
                        "get",
                        "configmap",
                        "gpu-fault-installed-resources",
                    ],
                    description=registry_description,
                ):
                    raise BootstrapError(
                        f"installed resource registry still exists: {registry_description}"
                    )
                if target_namespace == namespace and _kubectl_exists(
                    [*prefix, "get", "namespace", namespace],
                    description=f"{plane}:{context}:namespace/{namespace}",
                ):
                    raise BootstrapError(
                        f"solution namespace still exists: {plane}:{context}:{namespace}"
                    )
            for (resource_namespace, kind), names in sorted(grouped.items()):
                command = [*prefix]
                if resource_namespace:
                    command.extend(["-n", resource_namespace])
                command.extend(["get", kind, *sorted(names)])
                description = f"{plane}:{context}:{resource_namespace}:{kind}"
                if _kubectl_exists(command, description=description):
                    raise BootstrapError(
                        f"installed resource still exists: {description}"
                    )
            checked += len(resources)
    return {"registered_kubernetes_resources_verified_absent": checked}


def _sealed_snapshot(
    site_id: str,
    resources: list[InstallationResource],
) -> InstallationResourceSnapshot:
    snapshot = InstallationResourceSnapshot(
        site_id=site_id,
        resources=resources,
    )
    return snapshot.model_copy(update={"source_sha256": snapshot.digest()})


def _status_snapshot(
    snapshot: InstallationResourceSnapshot,
    *,
    status_for: Callable[
        [InstallationResource],
        InstallationResourceStatus,
    ],
) -> InstallationResourceSnapshot:
    now = datetime.now(timezone.utc)
    return _sealed_snapshot(
        snapshot.site_id,
        [
            resource.model_copy(
                update={
                    "status": status_for(resource),
                    "updated_at": now,
                    "error": None,
                }
            )
            for resource in snapshot.resources
        ],
    )


def _effective_policy(
    resource: InstallationResource,
    *,
    cpu_disposition: CpuDisposition,
    reset_database: bool = False,
    aurora_cluster_policy: InstallationResourceDeletePolicy | None = None,
) -> InstallationResourceDeletePolicy:
    if resource.resource_type in {"gpu_eks", "gpu_hyperpod"}:
        return InstallationResourceDeletePolicy.PRESERVE
    if resource.resource_type in {"cpu_eks", "cpu_hyperpod"}:
        return (
            InstallationResourceDeletePolicy.DELETE
            if cpu_disposition == "delete"
            else InstallationResourceDeletePolicy.PRESERVE
        )
    if (
        resource.resource_key in CPU_CLUSTER_BOUND_RESOURCE_KEYS
        and cpu_disposition == "delete"
    ):
        return InstallationResourceDeletePolicy.DELETE
    if is_aurora_resource(resource):
        # A reinstall keeps the database and everything the cluster stands on
        # (subnet group, security group, parameter group, managed secret); the
        # next deploy adopts the existing cluster.
        return (
            InstallationResourceDeletePolicy.PRESERVE
            if cpu_disposition == "keep" and not reset_database
            else InstallationResourceDeletePolicy.DELETE
        )
    if (
        resource.ownership is InstallationResourceOwnership.REUSED
        and resource.resource_type not in LEGACY_EXTERNAL_RESOURCE_TYPES
    ):
        if resource.resource_type in DETACH_RESOURCE_TYPES:
            return InstallationResourceDeletePolicy.DETACH
        return InstallationResourceDeletePolicy.DELETE
    return resource.delete_policy


def _terminal_status(
    policy: InstallationResourceDeletePolicy,
) -> InstallationResourceStatus:
    if policy is InstallationResourceDeletePolicy.DELETE:
        return InstallationResourceStatus.DELETED
    if policy is InstallationResourceDeletePolicy.DETACH:
        return InstallationResourceStatus.DETACHED
    return InstallationResourceStatus.PRESERVED


def _terminal_resource(
    resource: InstallationResource,
    *,
    policy: InstallationResourceDeletePolicy,
    now: datetime,
) -> InstallationResource:
    updates: dict[str, object] = {
        "status": _terminal_status(policy),
        "updated_at": now,
        "error": None,
    }
    return resource.model_copy(update=updates)


def _execution_resource(
    resource: InstallationResource,
    *,
    cpu_disposition: CpuDisposition,
    reset_database: bool = False,
    dns_binding: dict[str, Any] | None = None,
) -> InstallationResource:
    # This copy is an execution plan, never a registry update. ResourceCleaner
    # still requires live ownership/binding proof for legacy REUSED resources.
    return resource.model_copy(
        update={
            "delete_policy": _effective_policy(
                resource,
                cpu_disposition=cpu_disposition,
                reset_database=reset_database,
            ),
            "attributes": {
                **resource.attributes,
                **(
                    {"uninstall_dns_binding": json.dumps(dns_binding, sort_keys=True)}
                    if dns_binding is not None
                    else {}
                ),
            },
        }
    )


def _execution_dependencies(
    snapshot: InstallationResourceSnapshot,
) -> dict[str, set[str]]:
    by_key = {resource.resource_key: resource for resource in snapshot.resources}
    dependencies = {key: set(resource.dependencies) for key, resource in by_key.items()}
    role_key = "aws/iam/lbc/role"
    policy_key = "aws/iam/lbc/policy"
    role = by_key.get(role_key)
    policy = by_key.get(policy_key)
    if role is None or policy is None or role_key not in dependencies[policy_key]:
        return dependencies
    if (
        role.provider != "aws"
        or policy.provider != "aws"
        or role.resource_type != "iam_role"
        or policy.resource_type != "iam_policy"
    ):
        raise BootstrapError("legacy LBC attachment has invalid resource types")
    if policy_key in dependencies[role_key]:
        raise BootstrapError("installation resource dependencies contain a cycle")
    # Legacy registry rows reverse the attachment edge. The role consumes the
    # policy and detaches it on deletion; stored dependencies remain immutable.
    dependencies[policy_key].remove(role_key)
    dependencies[role_key].add(policy_key)
    return dependencies


def _transition(
    path: Path,
    state: dict[str, Any],
    phase: str,
    **updates: object,
) -> None:
    if phase not in UNINSTALL_PHASES or state.get("phase") not in UNINSTALL_PHASES:
        raise BootstrapError("invalid uninstall phase")
    if UNINSTALL_PHASES.index(phase) < UNINSTALL_PHASES.index(state["phase"]):
        raise BootstrapError("uninstall phase cannot move backwards")
    state.update(updates)
    state["phase"] = phase
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(path, state)


def _reached(state: dict[str, Any], phase: str) -> bool:
    return UNINSTALL_PHASES.index(state["phase"]) >= UNINSTALL_PHASES.index(phase)


def _validate_registry(
    request: UninstallRequest, snapshot: InstallationResourceSnapshot
) -> None:
    snapshot.require_source_binding()
    config = request.site.release_config
    cpu_arn = Arn.parse(str(config["cpu_eks_arn"]))
    eks_arns = {("cpu_eks", cpu_arn.resource_name): str(config["cpu_eks_arn"])}
    if snapshot.site_id != request.site.registry_site_id:
        raise BootstrapError("installation registry belongs to another site")
    expected = {
        ("cpu_eks", cpu_arn.resource_name),
        ("cpu_hyperpod", str(config["cpu_hyperpod_cluster_name"])),
        ("aurora_cluster", str(config["health"]["aurora_cluster_id"])),
    }
    for cluster in config["clusters"]:
        gpu_name = Arn.parse(cluster["eks_cluster_arn"]).resource_name
        if (
            gpu_name == cpu_arn.resource_name
            or cluster["hyperpod_cluster_name"] == config["cpu_hyperpod_cluster_name"]
        ):
            raise BootstrapError("CPU and GPU cluster identities overlap")
        eks_arns[("gpu_eks", gpu_name)] = str(cluster["eks_cluster_arn"])
        expected.update(
            {
                ("gpu_eks", gpu_name),
                ("gpu_hyperpod", str(cluster["hyperpod_cluster_name"])),
            }
        )
    cluster_types = {
        "cpu_eks",
        "cpu_hyperpod",
        "gpu_eks",
        "gpu_hyperpod",
        "aurora_cluster",
    }
    actual = [
        (resource.resource_type, resource.resource_id)
        for resource in snapshot.resources
        if resource.resource_type in cluster_types
    ]
    historical_gpu = {
        (resource.resource_type, resource.resource_id)
        for resource in snapshot.resources
        if resource.resource_type in {"gpu_eks", "gpu_hyperpod"}
        and resource.ownership is InstallationResourceOwnership.EXTERNAL
        and resource.delete_policy is InstallationResourceDeletePolicy.PRESERVE
    }
    if (
        not expected.issubset(actual)
        or set(actual) - expected - historical_gpu
        or len(actual) != len(set(actual))
    ):
        raise BootstrapError("installation registry cluster identity differs from site")
    by_key = {resource.resource_key: resource for resource in snapshot.resources}
    execution_dependencies = _execution_dependencies(snapshot)
    physical: set[tuple[str, str, str]] = set()
    for resource in snapshot.resources:
        if (
            resource.region != config["aws_region"]
            or resource.account_id != cpu_arn.account
        ):
            raise BootstrapError(
                "installation registry account or Region differs from site"
            )
        expected_arn = eks_arns.get((resource.resource_type, resource.resource_id))
        if expected_arn is not None and resource.resource_arn != expected_arn:
            raise BootstrapError("installation registry EKS ARN differs from site")
        policy = _effective_policy(
            resource,
            cpu_disposition=request.cpu_disposition,
            reset_database=request.reset_database,
        )
        if resource.resource_type not in DETACH_RESOURCE_TYPES:
            kind = resource.resource_type
            if kind in {"cpu_eks", "gpu_eks", "cpu_hyperpod", "gpu_hyperpod"}:
                kind = kind.partition("_")[2]
            identity = resource.provider, kind, resource.resource_id
            if identity in physical:
                raise BootstrapError(
                    "installation registry aliases a physical resource"
                )
            physical.add(identity)
        if (
            is_aurora_resource(resource)
            and policy is InstallationResourceDeletePolicy.DELETE
            and resource.ownership is InstallationResourceOwnership.EXTERNAL
        ):
            raise BootstrapError("Aurora deletion lacks recorded solution ownership")
        for dependency_key in execution_dependencies[resource.resource_key]:
            dependency = by_key.get(dependency_key)
            if dependency is None:
                raise BootstrapError(
                    f"missing installation dependency: {dependency_key}"
                )
            dependency_policy = _effective_policy(
                dependency,
                cpu_disposition=request.cpu_disposition,
                reset_database=request.reset_database,
            )
            if (
                policy is InstallationResourceDeletePolicy.PRESERVE
                and dependency_policy is not InstallationResourceDeletePolicy.PRESERVE
            ):
                raise BootstrapError(
                    f"retained resource depends on a deletion target: {dependency_key}"
                )
            if (
                is_aurora_resource(resource)
                and not is_aurora_resource(dependency)
                and dependency_policy is not InstallationResourceDeletePolicy.PRESERVE
            ):
                raise BootstrapError(
                    "Aurora dependency cannot be deleted before Aurora"
                )
    pending = set(by_key)
    while pending:
        ready = {key for key in pending if not execution_dependencies[key] & pending}
        if not ready:
            raise BootstrapError("installation resource dependencies contain a cycle")
        pending -= ready


def _load_or_export_registry(
    request: UninstallRequest,
    state_dir: Path,
) -> InstallationResourceSnapshot:
    path = state_dir / "installation-resources-before.json"
    plan_path = state_dir / "installation-resources-delete-plan.json"
    legacy = False
    if path.is_file():
        snapshot = load_installation_resource_snapshot(path)
    else:
        try:
            snapshot = fetch_installation_resource_registry(request.site, output=path)
        except LegacyInstallationRegistryMissing:
            legacy = True
            snapshot = build_legacy_installation_snapshot(request.site)
            write_installation_resource_snapshot(request.site, snapshot, path=path)
    _validate_registry(request, snapshot)
    if plan_path.is_file():
        plan = load_installation_resource_snapshot(plan_path)
        if [item.immutable_identity() for item in plan.resources] != [
            item.immutable_identity() for item in snapshot.resources
        ]:
            raise BootstrapError(
                "installation deletion plan differs from exported registry"
            )
        return snapshot
    pending = _status_snapshot(
        snapshot,
        status_for=lambda resource: (
            InstallationResourceStatus.DELETE_PENDING
            if _effective_policy(
                resource,
                cpu_disposition=request.cpu_disposition,
                reset_database=request.reset_database,
            )
            is not InstallationResourceDeletePolicy.PRESERVE
            else InstallationResourceStatus.ACTIVE
        ),
    )
    if legacy:
        sync_installation_resource_snapshot_direct(request.site, pending)
    else:
        try:
            sync_installation_resource_snapshot(request.site, pending)
        except LegacyInstallationRegistryMissing:
            sync_installation_resource_snapshot_direct(request.site, pending)
    write_installation_resource_snapshot(
        request.site,
        pending,
        path=plan_path,
    )
    return snapshot


def _delete_non_aurora_resources(
    cleaner: ResourceCleaner,
    snapshot: InstallationResourceSnapshot,
    *,
    cpu_disposition: CpuDisposition,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    execution_dependencies = _execution_dependencies(snapshot)
    candidates = [
        resource
        for resource in snapshot.resources
        if not is_aurora_resource(resource)
        and not (
            cpu_disposition == "delete"
            and resource.resource_key in CPU_CLUSTER_BOUND_RESOURCE_KEYS
        )
        and resource.resource_type
        not in {
            "cpu_eks",
            "cpu_hyperpod",
            "gpu_eks",
            "gpu_hyperpod",
        }
        and _effective_policy(
            resource,
            cpu_disposition=cpu_disposition,
        )
        is not InstallationResourceDeletePolicy.PRESERVE
    ]
    remaining = {resource.resource_key: resource for resource in candidates}
    batches: list[list[InstallationResource]] = []
    while remaining:
        dependencies = {
            dependency
            for key in remaining
            for dependency in execution_dependencies[key]
        }
        ready = [
            resource for key, resource in remaining.items() if key not in dependencies
        ]
        if not ready:
            raise BootstrapError("installation deletion dependencies contain a cycle")
        priority = min(delete_priority(resource) for resource in ready)
        batch = [
            resource for resource in ready if delete_priority(resource) == priority
        ]
        batches.append(batch)
        for resource in batch:
            del remaining[resource.resource_key]
    for batch in batches:
        with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
            futures = {
                executor.submit(
                    copy_context().run,
                    cleaner.delete,
                    _execution_resource(
                        resource,
                        cpu_disposition=cpu_disposition,
                        dns_binding=state.get("dns_bindings", {}).get(
                            resource.resource_key
                        ),
                    ),
                ): resource
                for resource in batch
            }
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        _transition(
            state_path,
            state,
            "NON_AURORA_DELETE_IN_PROGRESS",
            last_completed_priority=delete_priority(batch[0]),
        )


def _verify_resources(
    cleaner: ResourceCleaner,
    resources: list[InstallationResource],
    *,
    cpu_disposition: CpuDisposition,
    reset_database: bool = False,
    aurora_cluster_policy: InstallationResourceDeletePolicy | None = None,
    cpu_deleted: bool = False,
) -> list[InstallationResource]:
    now = datetime.now(timezone.utc)
    verified = []
    for resource in resources:
        policy = _effective_policy(
            resource,
            cpu_disposition=cpu_disposition,
            reset_database=reset_database,
            aurora_cluster_policy=aurora_cluster_policy,
        )
        exists = (
            False
            if (
                cpu_deleted
                and (
                    resource.resource_key in CPU_CLUSTER_BOUND_RESOURCE_KEYS
                    or resource.resource_type == "helm_release"
                )
            )
            else cleaner.exists(resource)
        )
        if policy is InstallationResourceDeletePolicy.PRESERVE:
            if not exists:
                raise BootstrapError(
                    f"preserved resource is missing: {resource.resource_key}"
                )
        elif exists:
            raise BootstrapError(
                f"deleted or detached resource still exists: {resource.resource_key}"
            )
        verified.append(
            _terminal_resource(
                resource,
                policy=policy,
                now=now,
            )
        )
    return verified


def _delete_or_verify_cpu(
    cleaner: ResourceCleaner,
    snapshot: InstallationResourceSnapshot,
    *,
    disposition: CpuDisposition,
    delete: bool = True,
    binding: dict[str, Any] | None = None,
) -> None:
    by_type = {
        resource.resource_type: resource
        for resource in snapshot.resources
        if resource.resource_type in {"cpu_eks", "cpu_hyperpod"}
    }
    if set(by_type) != {"cpu_eks", "cpu_hyperpod"}:
        raise BootstrapError(
            "installation registry does not contain both CPU cluster records"
        )
    if disposition == "delete" and delete:
        cleaner.delete_cpu_cluster(
            *(
                by_type[kind].model_copy(
                    update={
                        "attributes": {**by_type[kind].attributes, **(binding or {})}
                    }
                )
                for kind in ("cpu_hyperpod", "cpu_eks")
            )
        )
    for resource in by_type.values():
        exists = cleaner.exists(resource)
        if disposition == "delete" and exists:
            raise BootstrapError(
                f"CPU cluster resource still exists: {resource.resource_key}"
            )
        if disposition == "keep" and not exists:
            raise BootstrapError(
                f"CPU cluster resource is missing: {resource.resource_key}"
            )


def _verify_gpu_clusters(
    cleaner: ResourceCleaner,
    snapshot: InstallationResourceSnapshot,
    *,
    allow_empty: bool = False,
) -> int:
    resources = [
        resource
        for resource in snapshot.resources
        if resource.resource_type in {"gpu_eks", "gpu_hyperpod"}
    ]
    if not resources and not allow_empty:
        raise BootstrapError("installation registry contains no GPU clusters")
    for resource in resources:
        if not cleaner.exists(resource):
            raise BootstrapError(
                f"GPU cluster was not preserved: {resource.resource_key}"
            )
    return len(resources)


def _delete_aurora_last(
    cleaner: ResourceCleaner,
    snapshot: InstallationResourceSnapshot,
    request: UninstallRequest,
    state: dict[str, Any],
) -> InstallationResource | None:
    if state.get("phase") not in {
        "READY_TO_DELETE_AURORA",
        "AURORA_DELETE_IN_PROGRESS",
    }:
        raise BootstrapError("Aurora cleanup requires completed non-Aurora cleanup")
    aurora = [
        resource for resource in snapshot.resources if is_aurora_resource(resource)
    ]
    cluster = next(
        (resource for resource in aurora if resource.resource_type == "aurora_cluster"),
        None,
    )
    if cluster is None:
        raise BootstrapError(
            "installation registry does not contain the Aurora cluster"
        )
    cluster_policy = _effective_policy(
        cluster,
        cpu_disposition=request.cpu_disposition,
        reset_database=request.reset_database,
    )
    if cluster_policy is InstallationResourceDeletePolicy.PRESERVE:
        inconsistent = [
            resource.resource_key
            for resource in aurora
            if _effective_policy(
                resource,
                cpu_disposition=request.cpu_disposition,
                reset_database=request.reset_database,
                aurora_cluster_policy=cluster_policy,
            )
            is not InstallationResourceDeletePolicy.PRESERVE
        ]
        if inconsistent:
            raise BootstrapError(
                "preserved Aurora cluster has deletable child resources: "
                + ", ".join(inconsistent)
            )
        return None
    retained = cleaner.delete_aurora(
        _execution_resource(
            cluster,
            cpu_disposition=request.cpu_disposition,
            reset_database=request.reset_database,
        ).model_copy(
            update={
                "attributes": {
                    **cluster.attributes,
                    **state.get("aurora_binding", {}),
                }
            }
        ),
        final_snapshot_policy=request.final_snapshot_policy,
        final_snapshot_identifier=state["final_snapshot_identifier"],
    )
    for resource in aurora:
        if resource.resource_type in {
            "aurora_cluster",
            "aurora_instance",
            "rds_managed_secret",
        }:
            cleaner.wait_absent(resource, timeout_seconds=3600)
            continue
        policy = _effective_policy(
            resource,
            cpu_disposition=request.cpu_disposition,
            reset_database=request.reset_database,
            aurora_cluster_policy=cluster_policy,
        )
        if policy is InstallationResourceDeletePolicy.PRESERVE:
            continue
        cleaner.delete(
            _execution_resource(
                resource,
                cpu_disposition=request.cpu_disposition,
                reset_database=request.reset_database,
            )
        )
    if not retained:
        return None
    now = datetime.now(timezone.utc)
    return InstallationResource(
        site_id=snapshot.site_id,
        resource_key="aws/aurora/final-snapshot",
        resource_type="rds_snapshot",
        resource_id=retained,
        region=cluster.region,
        account_id=cluster.account_id,
        ownership=InstallationResourceOwnership.CREATED,
        delete_policy=InstallationResourceDeletePolicy.PRESERVE,
        status=InstallationResourceStatus.PRESERVED,
        attributes={
            "cluster_id": cluster.resource_id,
            **state.get("aurora_binding", {}),
        },
        created_at=now,
        updated_at=now,
    )


def _cleanup_document(
    request: UninstallRequest,
    runner: CommandRunner,
    state_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    cleanup_state = state_dir / "kubernetes-cleanup.json"
    if _reached(state, "KUBERNETES_VERIFIED") and not cleanup_state.is_file():
        raise BootstrapError("verified Kubernetes cleanup state is missing")
    if not _reached(state, "KUBERNETES_VERIFIED"):
        release_declared_spares(
            request,
            state_dir / "state.json",
            state,
            reference=f"uninstall/{request.confirmation}",
        )
        _run_cleanup(request, runner, cleanup_state)
    with materialized_release_config(request.site) as config:
        runner.run(
            [
                "python3",
                str(
                    request.site.repository_root
                    / "deploy/control-plane/tools/cleanup_state.py"
                ),
                "verify",
                "--path",
                str(cleanup_state),
                "--config",
                str(config),
                "--scope",
                "all",
                "--mode",
                "reset",
                "--node-mode",
                "uninstall",
            ],
            cwd=request.site.repository_root,
            env=cleanup_shell_environment(request, Path(config)),
        )
        target_arguments = [
            "python3",
            str(
                request.site.repository_root
                / "deploy/control-plane/tools/cleanup_kubernetes.py"
            ),
            "--config",
            str(config),
            "--state-file",
            str(cleanup_state),
            "--timeout-seconds",
            "120",
        ]
        if request.cpu_disposition == "delete" and _reached(
            state, "CPU_DELETE_IN_PROGRESS"
        ):
            target_arguments.append("--skip-cpu")
        runner.run(
            [*target_arguments, "verify-targets"],
            cwd=request.site.repository_root,
            env=cleanup_shell_environment(request, Path(config)),
        )
    cleanup_document = json.loads(cleanup_state.read_text(encoding="utf-8"))
    if (
        cleanup_document.get("phase") != "CLEANUP_COMPLETED"
        or cleanup_document.get("status") != "COMPLETED"
    ):
        raise BootstrapError("Kubernetes cleanup did not reach CLEANUP_COMPLETED")
    if (
        _reached(state, "KUBERNETES_VERIFIED")
        and state.get("cleanup_sha256") != (cleanup_document["content_sha256"])
    ):
        raise BootstrapError("verified Kubernetes cleanup snapshot changed")
    return cast(dict[str, Any], cleanup_document)


def _prepare_aurora_boundary(
    request: UninstallRequest,
    cleaner: ResourceCleaner,
    snapshot: InstallationResourceSnapshot,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    non_aurora = [
        resource for resource in snapshot.resources if not is_aurora_resource(resource)
    ]
    if not _reached(state, "NON_AURORA_VERIFIED"):
        _delete_non_aurora_resources(
            cleaner,
            snapshot,
            cpu_disposition=request.cpu_disposition,
            state_path=state_path,
            state=state,
        )
        # Helm and other CPU Kubernetes resources must be proved gone while
        # the cluster's API is still available.
        _verify_resources(
            cleaner,
            [
                resource
                for resource in non_aurora
                if resource.resource_type not in {"cpu_eks", "cpu_hyperpod"}
                and resource.resource_key not in CPU_CLUSTER_BOUND_RESOURCE_KEYS
            ],
            cpu_disposition=request.cpu_disposition,
        )
        _transition(state_path, state, "NON_AURORA_VERIFIED")
    if not _reached(state, "CPU_VERIFIED"):
        _transition(state_path, state, "CPU_DELETE_IN_PROGRESS")
        _delete_or_verify_cpu(
            cleaner,
            snapshot,
            disposition=request.cpu_disposition,
            binding=state.get("cpu_binding"),
        )
        _transition(state_path, state, "CPU_VERIFIED")
    else:
        _delete_or_verify_cpu(
            cleaner, snapshot, disposition=request.cpu_disposition, delete=False
        )
    verified_non_aurora = _verify_resources(
        cleaner,
        non_aurora,
        cpu_disposition=request.cpu_disposition,
        reset_database=request.reset_database,
        cpu_deleted=request.cpu_disposition == "delete",
    )
    aurora_pending = [
        resource.model_copy(
            update={
                "status": (
                    InstallationResourceStatus.DELETE_PENDING
                    if _effective_policy(
                        resource,
                        cpu_disposition=request.cpu_disposition,
                        reset_database=request.reset_database,
                    )
                    is not InstallationResourceDeletePolicy.PRESERVE
                    else InstallationResourceStatus.PRESERVED
                ),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        for resource in snapshot.resources
        if is_aurora_resource(resource)
    ]
    pre_aurora = _sealed_snapshot(
        snapshot.site_id,
        [*verified_non_aurora, *aurora_pending],
    )
    if not _reached(state, "READY_TO_DELETE_AURORA"):
        path = write_installation_resource_snapshot(
            request.site,
            pre_aurora,
            path=state_path.parent / "installation-resources-pre-aurora-delete.json",
        )
        _transition(
            state_path,
            state,
            "READY_TO_DELETE_AURORA",
            pre_aurora_sha256=pre_aurora.digest(),
            pre_aurora_registry=str(path),
        )
    else:
        recorded = load_installation_resource_snapshot(
            state_path.parent / "installation-resources-pre-aurora-delete.json"
        )
        if recorded.digest() != state.get("pre_aurora_sha256"):
            raise BootstrapError("pre-Aurora cleanup snapshot changed")


def _finish_uninstall(
    request: UninstallRequest,
    cleaner: ResourceCleaner,
    snapshot: InstallationResourceSnapshot,
    state_path: Path,
    state: dict[str, Any],
    *,
    kubernetes_result: dict[str, int],
    gpu_records: int,
) -> dict[str, Any]:
    aurora_cluster = next(
        resource
        for resource in snapshot.resources
        if resource.resource_type == "aurora_cluster"
    )
    aurora_policy = _effective_policy(
        aurora_cluster,
        cpu_disposition=request.cpu_disposition,
        reset_database=request.reset_database,
    )
    if state["phase"] == "COMPLETED":
        final = load_installation_resource_snapshot(
            state_path.parent / "installation-resources-final.json"
        )
        if final.digest() != state.get("final_registry_sha256"):
            raise BootstrapError("completed uninstall snapshot changed")
        final_snapshot = next(
            (
                item
                for item in final.resources
                if item.resource_key == "aws/aurora/final-snapshot"
            ),
            None,
        )
    else:
        _transition(state_path, state, "AURORA_DELETE_IN_PROGRESS")
        final_snapshot = _delete_aurora_last(cleaner, snapshot, request, state)
    verified = _verify_resources(
        cleaner,
        snapshot.resources,
        cpu_disposition=request.cpu_disposition,
        reset_database=request.reset_database,
        aurora_cluster_policy=aurora_policy,
        cpu_deleted=request.cpu_disposition == "delete",
    )
    if final_snapshot is not None:
        binding = state.get("aurora_binding", {})
        if (
            final_snapshot.resource_id != state["final_snapshot_identifier"]
            or not isinstance(binding.get("db_cluster_resource_id"), str)
            or not binding["db_cluster_resource_id"]
        ):
            raise BootstrapError("retained final snapshot lacks its original binding")
        if not cleaner.exists(
            final_snapshot.model_copy(
                update={"attributes": {**final_snapshot.attributes, **binding}}
            )
        ):
            raise BootstrapError(
                f"retained final snapshot is missing: {final_snapshot.resource_id}"
            )
        verified.append(final_snapshot)
    elif (
        aurora_policy is InstallationResourceDeletePolicy.DELETE
        and request.final_snapshot_policy == "retain"
    ):
        raise BootstrapError("retained final snapshot is missing from uninstall state")
    final = _sealed_snapshot(snapshot.site_id, verified)
    final_path = state_path.parent / "installation-resources-final.json"
    residuals = [
        resource.resource_key
        for resource in verified
        if resource.status
        not in {
            InstallationResourceStatus.DELETED,
            InstallationResourceStatus.DETACHED,
            InstallationResourceStatus.PRESERVED,
        }
    ]
    if residuals:
        raise BootstrapError(
            "installation registry has non-terminal resources: " + ", ".join(residuals)
        )
    deleted = sum(
        resource.status is InstallationResourceStatus.DELETED for resource in verified
    )
    detached = sum(
        resource.status is InstallationResourceStatus.DETACHED for resource in verified
    )
    preserved = sum(
        resource.status is InstallationResourceStatus.PRESERVED for resource in verified
    )
    if state["phase"] != "COMPLETED":
        write_installation_resource_snapshot(request.site, final, path=final_path)
        _transition(
            state_path,
            state,
            "COMPLETED",
            final_registry=str(final_path),
            final_registry_sha256=final.digest(),
            delete_policy_residuals=0,
        )
    return {
        "cpu_cluster": request.cpu_disposition,
        "gpu_clusters": "preserved",
        "gpu_cluster_records_verified": gpu_records,
        "aurora_cluster": (
            "deleted"
            if aurora_policy is InstallationResourceDeletePolicy.DELETE
            else "preserved"
        ),
        "database_reset": request.reset_database,
        "aurora_deleted_last": (
            aurora_policy is InstallationResourceDeletePolicy.DELETE
        ),
        "aurora_final_snapshot": (
            final_snapshot.resource_id if final_snapshot is not None else None
        ),
        "registry_entries_deleted": deleted,
        "registry_entries_detached": detached,
        "registry_entries_preserved": preserved,
        "delete_policy_residuals": 0,
        "final_registry": str(final_path),
        **kubernetes_result,
    }


def _bind_aurora_deletion(
    request: UninstallRequest,
    cleaner: ResourceCleaner,
    snapshot: InstallationResourceSnapshot,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    if request.cpu_disposition == "keep" and not request.reset_database:
        return
    if "aurora_binding" in state:
        binding = state["aurora_binding"]
        if (
            not isinstance(binding, dict)
            or not isinstance(binding.get("db_cluster_resource_id"), str)
            or not binding["db_cluster_resource_id"]
        ):
            raise BootstrapError("invalid saved Aurora incarnation binding")
        return
    if _reached(state, "AURORA_DELETE_IN_PROGRESS"):
        raise BootstrapError("started Aurora deletion lacks an incarnation checkpoint")
    cluster = next(
        item for item in snapshot.resources if item.resource_type == "aurora_cluster"
    )
    binding = cleaner.prepare_aurora_delete(
        _execution_resource(
            cluster,
            cpu_disposition=request.cpu_disposition,
            reset_database=request.reset_database,
        ),
        final_snapshot_policy=request.final_snapshot_policy,
        final_snapshot_identifier=state["final_snapshot_identifier"],
    )
    _transition(state_path, state, state["phase"], aurora_binding=binding)


def _bind_cpu_deletion(
    request: UninstallRequest,
    cleaner: ResourceCleaner,
    snapshot: InstallationResourceSnapshot,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    if request.cpu_disposition != "delete":
        return
    if "cpu_binding" not in state:
        if _reached(state, "KUBERNETES_VERIFIED"):
            raise BootstrapError("started CPU cleanup lacks an incarnation binding")
        by_type = {item.resource_type: item for item in snapshot.resources}
        binding = cleaner.prepare_cpu_delete(
            by_type["cpu_hyperpod"], by_type["cpu_eks"]
        )
        _transition(state_path, state, state["phase"], cpu_binding=binding)
    binding = state["cpu_binding"]
    if (
        not isinstance(binding, dict)
        or set(binding) != {"cpu_eks_created_at", "cpu_hyperpod_arn"}
        or any(
            not isinstance(value, str) or not value.strip()
            for value in binding.values()
        )
    ):
        raise BootstrapError("invalid saved CPU incarnation binding")


def _bind_dns_deletion(
    request: UninstallRequest,
    cleaner: ResourceCleaner,
    snapshot: InstallationResourceSnapshot,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    records = [
        _execution_resource(item, cpu_disposition=request.cpu_disposition)
        for item in snapshot.resources
        if item.resource_type == "route53_record"
        and _effective_policy(item, cpu_disposition=request.cpu_disposition)
        is not InstallationResourceDeletePolicy.PRESERVE
    ]
    if not records:
        return
    if "dns_bindings" not in state:
        if _reached(state, "KUBERNETES_VERIFIED"):
            raise BootstrapError("started DNS cleanup lacks an NLB target binding")
        bindings = {
            item.resource_key: cleaner.prepare_dns_delete(item) for item in records
        }
        _transition(state_path, state, state["phase"], dns_bindings=bindings)
    if not isinstance(state["dns_bindings"], dict) or set(state["dns_bindings"]) != {
        item.resource_key for item in records
    }:
        raise BootstrapError("saved DNS cleanup binding differs from its registry")


def _uninstall_locked(
    request: UninstallRequest,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    required = _required_confirmation(request.cpu_disposition)
    if request.confirmation != required:
        raise BootstrapError(f"uninstall requires --confirm {required}")
    active_runner = runner or CommandRunner()
    state_dir = request.site.source.parent / "uninstall"
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    resumed = state_path.is_file()
    state = _uninstall_state(request, state_path)
    record_repository_root_override(request, state_path, state, resumed=resumed)
    if _reached(state, "REGISTRY_EXPORTED") and not all(
        (state_dir / name).is_file()
        for name in (
            "installation-resources-before.json",
            "installation-resources-delete-plan.json",
        )
    ):
        raise BootstrapError("exported uninstall registry is missing")
    snapshot = _load_or_export_registry(request, state_dir)
    cleaner = ResourceCleaner(request.site)
    cleaner.validate_supported(snapshot.resources)
    policies = {
        item.resource_key: _effective_policy(
            item,
            cpu_disposition=request.cpu_disposition,
            reset_database=request.reset_database,
        ).value
        for item in snapshot.resources
    }
    if not _reached(state, "REGISTRY_EXPORTED"):
        _transition(
            state_path,
            state,
            "REGISTRY_EXPORTED",
            registry_sha256=snapshot.digest(),
            effective_policies=policies,
        )
    elif state.get("registry_sha256") != snapshot.digest():
        raise BootstrapError("exported uninstall registry changed")
    elif state.get("effective_policies") != policies:
        raise BootstrapError("authorized uninstall policies changed")
    _bind_aurora_deletion(request, cleaner, snapshot, state_path, state)
    bind_retained_database(request, active_runner, state_path, state)
    _bind_cpu_deletion(request, cleaner, snapshot, state_path, state)
    _bind_dns_deletion(request, cleaner, snapshot, state_path, state)
    _verify_gpu_clusters(
        cleaner, snapshot, allow_empty=not request.site.release_config["clusters"]
    )
    cleanup_document = _cleanup_document(request, active_runner, state_dir, state)
    cpu_deletion_started = request.cpu_disposition == "delete" and _reached(
        state, "CPU_DELETE_IN_PROGRESS"
    )
    kubernetes_result = verify_installed_registry_cleanup(
        request.site, cleanup_document, cpu_deleted=cpu_deletion_started
    )
    if not _reached(state, "KUBERNETES_VERIFIED"):
        _transition(
            state_path,
            state,
            "KUBERNETES_VERIFIED",
            cleanup_sha256=cleanup_document["content_sha256"],
        )
    _prepare_aurora_boundary(request, cleaner, snapshot, state_path, state)
    gpu_records = _verify_gpu_clusters(
        cleaner, snapshot, allow_empty=not request.site.release_config["clusters"]
    )
    bind_retained_database(request, active_runner, state_path, state)
    return _finish_uninstall(
        request,
        cleaner,
        snapshot,
        state_path,
        state,
        kubernetes_result=kubernetes_result,
        gpu_records=gpu_records,
    )


def uninstall(
    request: UninstallRequest,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    with membership_operation_lock(request.site):
        current = reload_site_for_mutation(request.site)
        active = replace(request, site=current)
        try:
            return _uninstall_locked(active, runner=runner)
        except ProcessSupervisionLost:
            path = current.source.parent / "uninstall" / "state.json"
            state = _uninstall_state(active, path)
            state["supervision_lost"] = True
            write_json_atomic(path, state)
            raise
