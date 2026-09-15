from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, cast

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.aws_cleanup import (
    ResourceCleaner,
    delete_priority,
    is_aurora_resource,
)
from gpu_fault.admin.aws_commands import FinalSnapshotPolicy
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    CommandRunner,
    safe_name,
)
from gpu_fault.admin.membership_lock import (
    membership_operation_lock,
    reload_site_for_mutation,
)
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
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)

CpuDisposition = Literal["keep", "delete"]
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


@dataclass(frozen=True)
class UninstallRequest:
    """One uninstall.

    ``cpu_disposition="keep"`` is a reinstall: the CPU cluster and the Aurora
    cluster stay, so the site's incident, workflow and registry records (and
    the quarantine taints keyed on those incident ids) survive to the next
    ``deploy``, which adopts the existing cluster. ``reset_database`` is the
    explicit opt-in to wipe Aurora on a reinstall. ``cpu_disposition="delete"``
    is retirement: Aurora goes with the CPU cluster, and
    ``final_snapshot_policy`` decides whether a final snapshot stays for audit.
    """

    site: RenderedSite
    cpu_disposition: CpuDisposition
    confirmation: str
    final_snapshot_policy: FinalSnapshotPolicy = "retain"
    reset_database: bool = False

    def __post_init__(self) -> None:
        if (
            self.cpu_disposition == "keep"
            and self.final_snapshot_policy == "skip"
            and not self.reset_database
        ):
            # A plain reinstall keeps the Aurora cluster, so there is nothing
            # to snapshot; a reinstall that resets the database deletes it and
            # may skip the audit snapshot like a retirement does.
            raise BootstrapError(
                "--aurora-final-snapshot skip is only valid with --cpu-cluster "
                "delete or with --reset-database; a reinstall keeps the Aurora "
                "cluster"
            )
        if self.cpu_disposition == "delete" and self.reset_database:
            raise BootstrapError(
                "--reset-database is only valid with --cpu-cluster keep; "
                "--cpu-cluster delete already deletes the Aurora cluster"
            )


def _required_confirmation(disposition: CpuDisposition) -> str:
    return (
        "DELETE_CPU_CONTROL_PLANE" if disposition == "delete" else "UNINSTALL_GPU_FAULT"
    )


def archive_unfinished_cleanup_state(path: Path, document: Mapping[str, Any]) -> Path:
    """Move a Kubernetes cleanup record that never reached CLEANUP_COMPLETED aside.

    ``prepare-clean-redeploy.sh`` refuses an existing state file and its state
    tool refuses a phase moving backwards, so a rerun after a failed sweep
    (live 2026-09-12: the drain timed out at QUEUES_DRAINED) could never start
    over the same file. The failed record stays beside the new one as evidence.
    """

    stamp = (
        str(document.get("updated_at") or datetime.now(timezone.utc).isoformat())
        .replace("-", "")
        .replace(":", "")[:15]
    )
    archive = path.with_name(f"{path.stem}.failed-{stamp}{path.suffix}")
    path.replace(archive)
    return archive


def _run_cleanup(
    request: UninstallRequest,
    runner: CommandRunner,
    state_file: Path,
) -> None:
    with materialized_release_config(request.site) as config:
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
            env={**os.environ, **request.site.environment},
            mutate=True,
            capture=False,
        )


def _kubectl_exists(command: list[str], *, description: str) -> bool:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode == 0:
        return True
    message = (result.stdout or "") + "\n" + (result.stderr or "")
    if any(
        pattern in message.lower()
        for pattern in (
            "notfound",
            "not found",
            "the server could not find the requested resource",
        )
    ):
        return False
    raise BootstrapError(
        f"Kubernetes cleanup verification failed for {description}: "
        f"{result.stderr.strip()}"
    )


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
) -> dict[str, int]:
    config = site.release_config
    namespace = str(config["namespace"])
    inventory = cleanup_state["inventory_snapshot"]
    checked = 0
    cluster_contexts = [str(item["context"]) for item in config["clusters"]]
    for plane, section in (("cpu", inventory["cpu"]), ("gpu", inventory["gpu"])):
        contexts = ["cpu"] if plane == "cpu" else cluster_contexts
        for context in contexts:
            prefix = _kubectl_prefix(
                site,
                plane=plane,
                context=context,
            )
            for resource in section["resources"]:
                command = [*prefix]
                if resource["scope"] == "namespaced":
                    command.extend(["-n", namespace])
                command.extend(["get", resource["kind"], resource["name"]])
                description = f"{plane}:{context}:{resource['kind']}/{resource['name']}"
                if _kubectl_exists(command, description=description):
                    raise BootstrapError(
                        f"installed resource still exists: {description}"
                    )
                checked += 1
            registry_command = [
                *prefix,
                "-n",
                namespace,
                "get",
                "configmap",
                "gpu-fault-installed-resources",
            ]
            registry_description = (
                f"{plane}:{context}:configmap/gpu-fault-installed-resources"
            )
            if _kubectl_exists(
                registry_command,
                description=registry_description,
            ):
                raise BootstrapError(
                    f"installed resource registry still exists: {registry_description}"
                )
            namespace_command = [
                *prefix,
                "get",
                "namespace",
                namespace,
            ]
            namespace_description = f"{plane}:{context}:namespace/{namespace}"
            if _kubectl_exists(
                namespace_command,
                description=namespace_description,
            ):
                raise BootstrapError(
                    f"solution namespace still exists: {namespace_description}"
                )
    return {"registered_kubernetes_resources_verified_absent": checked}


def _sealed_snapshot(
    site_id: str,
    resources: list[InstallationResource],
) -> InstallationResourceSnapshot:
    snapshot = InstallationResourceSnapshot(
        site_id=site_id,
        resources=resources,
    )
    return cast(
        InstallationResourceSnapshot,
        snapshot.model_copy(update={"source_sha256": snapshot.digest()}),
    )


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
    if (
        is_aurora_resource(resource)
        and cpu_disposition == "keep"
        and not reset_database
    ):
        # A reinstall keeps the database and everything the cluster stands on
        # (subnet group, security group, parameter group, managed secret); the
        # next deploy adopts the existing cluster.
        return InstallationResourceDeletePolicy.PRESERVE
    if (
        resource.resource_type in {"aurora_instance", "rds_managed_secret"}
        and aurora_cluster_policy is not None
    ):
        return aurora_cluster_policy
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
    if (
        resource.ownership is InstallationResourceOwnership.REUSED
        and resource.resource_type not in LEGACY_EXTERNAL_RESOURCE_TYPES
    ):
        updates["ownership"] = InstallationResourceOwnership.CREATED
        updates["delete_policy"] = policy
    return cast(InstallationResource, resource.model_copy(update=updates))


def _uninstall_state(
    request: UninstallRequest,
    path: Path,
) -> tuple[dict[str, Any], bool]:
    """Load an in-progress uninstall or start a fresh session.

    The second element is ``True`` when an existing ``state.json`` was
    resumed and ``False`` when a brand-new session was created. A prior
    session that ran to COMPLETED has its ``state.json`` renamed aside
    (``state.consumed-*``), so a later uninstall of a redeployed site lands
    here with no ``state.json`` and must be treated as fresh -- never a
    resume that trusts the previous session's working artifacts.
    """

    if path.is_file():
        value = cast(
            dict[str, Any],
            json.loads(path.read_text(encoding="utf-8")),
        )
        expected = {
            "site_id": request.site.release_config["site_name"],
            "cpu_disposition": request.cpu_disposition,
            "final_snapshot_policy": request.final_snapshot_policy,
            "reset_database": request.reset_database,
        }
        for key, item in expected.items():
            # A state file written before ``reset_database`` existed resumes as
            # a plain reinstall.
            if value.get(key, False if key == "reset_database" else None) != item:
                raise BootstrapError(f"uninstall state {path} conflicts on {key}")
        return value, True
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    snapshot_identifier = safe_name(
        (f"{request.site.release_config['site_name']}-uninstall-{timestamp}"),
        maximum=63,
    )
    value = {
        "site_id": request.site.release_config["site_name"],
        "cpu_disposition": request.cpu_disposition,
        "final_snapshot_policy": request.final_snapshot_policy,
        "reset_database": request.reset_database,
        "final_snapshot_identifier": snapshot_identifier,
        "phase": "STARTED",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(path, value)
    return value, False


def _archive_prior_session_artifacts(state_dir: Path) -> list[Path]:
    """Move a previous uninstall session's working files aside on a fresh run.

    ``_load_or_export_registry`` and the Kubernetes-cleanup gate both reuse a
    file whenever it merely ``is_file()`` -- the resume path that lets a
    crashed session pick up where it stopped. But once a session reaches
    COMPLETED its ``state.json`` is consumed, so the next uninstall of a
    site that was redeployed in between starts with no ``state.json`` yet
    finds a stale ``installation-resources-before.json`` (a registry snapshot
    of the retired install) and a ``kubernetes-cleanup.json`` still marked
    CLEANUP_COMPLETED. Trusting them makes the run skip the real cleanup and
    then fail verification against resources the redeploy recreated (live
    2026-09-15: ``installed resource still exists: cpu:cpu:deployment/
    gpu-fault-api-ha``). A fresh session must re-export and re-clean, so
    rename the prior session's artifacts aside as evidence.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived: list[Path] = []
    names = (
        "installation-resources-before.json",
        "installation-resources-delete-plan.json",
        "installation-resources-pre-aurora-delete.json",
        "installation-resources-final.json",
        "kubernetes-cleanup.json",
    )
    for name in names:
        base = state_dir / name
        for candidate in (base, base.with_name(base.name + ".sha256")):
            if candidate.is_file():
                archive = candidate.with_name(f"{candidate.name}.superseded-{stamp}")
                candidate.replace(archive)
                archived.append(archive)
    return archived


def _transition(
    path: Path,
    state: dict[str, Any],
    phase: str,
    **updates: object,
) -> None:
    state.update(updates)
    state["phase"] = phase
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(path, state)


def _load_or_export_registry(
    request: UninstallRequest,
    state_dir: Path,
) -> InstallationResourceSnapshot:
    path = state_dir / "installation-resources-before.json"
    if path.is_file():
        return load_installation_resource_snapshot(path)
    legacy = False
    try:
        snapshot = fetch_installation_resource_registry(
            request.site,
            output=path,
        )
    except LegacyInstallationRegistryMissing:
        legacy = True
        snapshot = build_legacy_installation_snapshot(request.site)
        write_installation_resource_snapshot(
            request.site,
            snapshot,
            path=path,
        )
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
        path=state_dir / "installation-resources-delete-plan.json",
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
    priorities = sorted({delete_priority(resource) for resource in candidates})
    for priority in priorities:
        batch = [
            resource for resource in candidates if delete_priority(resource) == priority
        ]
        with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
            futures = {
                executor.submit(cleaner.delete, resource): resource
                for resource in batch
            }
            for future in as_completed(futures):
                future.result()
        _transition(
            state_path,
            state,
            "NON_AURORA_DELETE_IN_PROGRESS",
            last_completed_priority=priority,
        )


def _verify_resources(
    cleaner: ResourceCleaner,
    resources: list[InstallationResource],
    *,
    cpu_disposition: CpuDisposition,
    reset_database: bool = False,
    aurora_cluster_policy: InstallationResourceDeletePolicy | None = None,
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
                cpu_disposition == "delete"
                and resource.resource_key in CPU_CLUSTER_BOUND_RESOURCE_KEYS
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
    if disposition == "delete":
        cleaner.delete_cpu_cluster(
            by_type["cpu_hyperpod"],
            by_type["cpu_eks"],
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
) -> int:
    resources = [
        resource
        for resource in snapshot.resources
        if resource.resource_type in {"gpu_eks", "gpu_hyperpod"}
    ]
    if not resources:
        raise BootstrapError("installation registry contains no GPU clusters")
    for resource in resources:
        if not cleaner.exists(resource):
            raise BootstrapError(
                f"GPU cluster was not preserved: {resource.resource_key}"
            )
    return len(resources)


class AuroraDeletion:
    """The Aurora phase of one uninstall, run beside the non-Aurora deletes.

    The cluster deletion depends only on the Kubernetes cleanup (no Pod may
    still hold a connection), not on Route53, the NLB, IAM or ECR, and its
    wait dominated the live uninstall (18 of 24 min, 2026-09-12). ``start``
    therefore issues it on a worker thread as soon as the cleanup is verified;
    the non-Aurora deletes run meanwhile; ``finish`` joins the thread, then
    deletes what the cluster stood on (managed secret, subnet group, security
    group, parameter group) -- RDS refuses those while a cluster uses them.

    A rerun after a crash needs nothing from this object: ``delete_aurora``
    finds the cluster and its instances already ``deleting`` (or gone) and
    only waits. The thread is a daemon so a Ctrl-C does not hang on the wait;
    the state file is written by the caller's thread only.
    """

    def __init__(
        self,
        cleaner: ResourceCleaner,
        snapshot: InstallationResourceSnapshot,
        request: UninstallRequest,
        state: Mapping[str, Any],
    ) -> None:
        self._cleaner = cleaner
        self._request = request
        self._final_snapshot_identifier = str(state["final_snapshot_identifier"])
        self.resources = [
            resource for resource in snapshot.resources if is_aurora_resource(resource)
        ]
        cluster = next(
            (
                resource
                for resource in self.resources
                if resource.resource_type == "aurora_cluster"
            ),
            None,
        )
        if cluster is None:
            raise BootstrapError(
                "installation registry does not contain the Aurora cluster"
            )
        self.cluster = cluster
        self.policy = _effective_policy(
            cluster,
            cpu_disposition=request.cpu_disposition,
            reset_database=request.reset_database,
        )
        self._thread: threading.Thread | None = None
        self._retained: str | None = None
        self._failure: BaseException | None = None

    def _child_policy(
        self, resource: InstallationResource
    ) -> InstallationResourceDeletePolicy:
        return _effective_policy(
            resource,
            cpu_disposition=self._request.cpu_disposition,
            reset_database=self._request.reset_database,
            aurora_cluster_policy=self.policy,
        )

    def _run(self) -> None:
        try:
            self._retained = self._cleaner.delete_aurora(
                self.cluster,
                final_snapshot_policy=self._request.final_snapshot_policy,
                final_snapshot_identifier=self._final_snapshot_identifier,
            )
        except BaseException as exc:  # handed to the caller by ``wait``/``join``
            self._failure = exc

    def start(self) -> None:
        """Issue the cluster deletion (protection off, readers, writer, cluster)."""

        if self.policy is InstallationResourceDeletePolicy.PRESERVE:
            inconsistent = [
                resource.resource_key
                for resource in self.resources
                if self._child_policy(resource)
                is not InstallationResourceDeletePolicy.PRESERVE
            ]
            if inconsistent:
                raise BootstrapError(
                    "preserved Aurora cluster has deletable child resources: "
                    + ", ".join(inconsistent)
                )
            return
        self._thread = threading.Thread(
            target=self._run, name="aurora-delete", daemon=True
        )
        self._thread.start()

    def wait(self) -> BaseException | None:
        """Block until the cluster deletion ended; return its failure, if any."""

        if self._thread is not None:
            self._thread.join()
        return self._failure

    def join(self) -> str | None:
        """Block until the cluster is gone; return the retained snapshot id."""

        failure = self.wait()
        if failure is not None:
            raise failure
        return self._retained

    def finish(self) -> InstallationResource | None:
        """Join, delete the cluster's dependents, describe the final snapshot."""

        retained = self.join()
        if self.policy is InstallationResourceDeletePolicy.PRESERVE:
            return None
        for resource in self.resources:
            if resource.resource_type in {
                "aurora_cluster",
                "aurora_instance",
                "rds_managed_secret",
            }:
                self._cleaner.wait_absent(resource, timeout_seconds=3600)
                continue
            if (
                self._child_policy(resource)
                is InstallationResourceDeletePolicy.PRESERVE
            ):
                continue
            self._cleaner.delete(resource)
        if not retained:
            return None
        now = datetime.now(timezone.utc)
        return InstallationResource(
            site_id=self.cluster.site_id,
            resource_key="aws/aurora/final-snapshot",
            resource_type="rds_snapshot",
            resource_id=retained,
            region=self.cluster.region,
            account_id=self.cluster.account_id,
            ownership=InstallationResourceOwnership.CREATED,
            delete_policy=InstallationResourceDeletePolicy.PRESERVE,
            status=InstallationResourceStatus.PRESERVED,
            created_at=now,
            updated_at=now,
        )


def _delete_non_aurora_phase(
    cleaner: ResourceCleaner,
    snapshot: InstallationResourceSnapshot,
    request: UninstallRequest,
    *,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> int:
    """Delete and verify everything but Aurora; return the GPU cluster count.

    Runs while ``AuroraDeletion`` waits on the cluster. Nothing here reads the
    cluster: the PKI secret is the solution's own, the IAM policy that read the
    managed master secret only has to be detached from roles no Pod uses any
    more, and the resources the cluster stands on (subnet group, its security
    group, parameter group) are keyed ``aws/aurora/`` and so belong to the
    Aurora phase, after the join.
    """

    _delete_non_aurora_resources(
        cleaner,
        snapshot,
        cpu_disposition=request.cpu_disposition,
        state_path=state_path,
        state=state,
    )
    _delete_or_verify_cpu(
        cleaner,
        snapshot,
        disposition=request.cpu_disposition,
    )
    gpu_records = _verify_gpu_clusters(cleaner, snapshot)
    non_aurora = [
        resource for resource in snapshot.resources if not is_aurora_resource(resource)
    ]
    verified_non_aurora = _verify_resources(
        cleaner,
        non_aurora,
        cpu_disposition=request.cpu_disposition,
        reset_database=request.reset_database,
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
    write_installation_resource_snapshot(
        request.site,
        pre_aurora,
        path=state_dir / "installation-resources-pre-aurora-delete.json",
    )
    _transition(state_path, state, "READY_TO_DELETE_AURORA")
    return gpu_records


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
    state, resumed = _uninstall_state(request, state_path)
    if not resumed:
        _archive_prior_session_artifacts(state_dir)
    snapshot = _load_or_export_registry(request, state_dir)
    cleaner = ResourceCleaner(request.site)
    cleaner.validate_supported(snapshot.resources)
    _transition(state_path, state, "REGISTRY_EXPORTED")

    cleanup_state = state_dir / "kubernetes-cleanup.json"
    cleanup_complete = False
    if cleanup_state.exists():
        previous_cleanup = json.loads(cleanup_state.read_text(encoding="utf-8"))
        cleanup_complete = (
            previous_cleanup.get("phase") == "CLEANUP_COMPLETED"
            and previous_cleanup.get("status") == "COMPLETED"
        )
        if not cleanup_complete:
            archive_unfinished_cleanup_state(cleanup_state, previous_cleanup)
    if not cleanup_complete:
        _run_cleanup(request, active_runner, cleanup_state)
    cleanup_document = json.loads(cleanup_state.read_text(encoding="utf-8"))
    if (
        cleanup_document.get("phase") != "CLEANUP_COMPLETED"
        or cleanup_document.get("status") != "COMPLETED"
    ):
        raise BootstrapError("Kubernetes cleanup did not reach CLEANUP_COMPLETED")
    kubernetes_result = verify_installed_registry_cleanup(
        request.site,
        cleanup_document,
    )
    _transition(state_path, state, "KUBERNETES_VERIFIED")

    aurora = AuroraDeletion(cleaner, snapshot, request, state)
    aurora.start()
    try:
        gpu_records = _delete_non_aurora_phase(
            cleaner,
            snapshot,
            request,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
        )
    except Exception as failure:
        # Never orphan the Aurora thread: wait for it, then report the failure
        # that stopped the uninstall (and the Aurora one, if it failed too).
        aurora_failure = aurora.wait()
        if aurora_failure is not None:
            raise BootstrapError(
                f"{failure}; the Aurora deletion also failed: {aurora_failure}"
            ) from failure
        raise
    final_snapshot = aurora.finish()
    aurora_policy = aurora.policy
    verified = _verify_resources(
        cleaner,
        snapshot.resources,
        cpu_disposition=request.cpu_disposition,
        reset_database=request.reset_database,
        aurora_cluster_policy=aurora_policy,
    )
    if final_snapshot is not None:
        if not cleaner.exists(final_snapshot):
            raise BootstrapError(
                f"retained final snapshot is missing: {final_snapshot.resource_id}"
            )
        verified.append(final_snapshot)
    final = _sealed_snapshot(snapshot.site_id, verified)
    final_path = write_installation_resource_snapshot(
        request.site,
        final,
        path=state_dir / "installation-resources-final.json",
    )
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
    _transition(
        state_path,
        state,
        "COMPLETED",
        final_registry=str(final_path),
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


def uninstall(
    request: UninstallRequest,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    with membership_operation_lock(request.site):
        current = reload_site_for_mutation(request.site)
        return _uninstall_locked(replace(request, site=current), runner=runner)
