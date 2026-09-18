"""Checked native uninstall adapter and independent live readback."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault.admin.aws_cleanup import ResourceProbe
from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.admin.execution import deployment_deadline, operation_budget
from gpu_fault.admin.operation_lock import (
    SITE_OPERATION_LOCK_FD_ENV,
    site_operation_lock,
)
from gpu_fault.admin.process_supervisor import (
    ProcessSupervisionLost,
    ensure_supervision_safe,
)
from gpu_fault.admin.uninstall import (
    CPU_CLUSTER_BOUND_RESOURCE_KEYS,
    UninstallRequest,
    uninstall,
    verify_installed_registry_cleanup,
)
from scripts.e2e.regional.acceptance_supervision import (
    record_supervision_loss,
    require_supervision_clear,
)
from scripts.e2e.regional.boot032_contract import Settings, cluster_specs, require
from scripts.e2e.regional.boot032_journal import (
    cleanup_complete,
    export_receipts,
    final_receipts,
    native_state,
    pause_proof,
    reached,
    required_native_state,
)
from scripts.e2e.regional.boot032_observe import (
    cluster_observation,
    initial_binding,
    installed_inventory,
    inventory_uids,
    kubernetes_uid,
    node_observation,
    observe_site,
    runtime_observation,
)
from scripts.e2e.regional.live_driver_guard import details_sha256


class RestartRequired(RuntimeError):
    """A supervised command and native proof completed before the controlled pause."""


@contextmanager
def checked_locks(settings: Settings, deadline: datetime) -> Iterator[None]:
    seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
    require(seconds > 0, "approved maintenance window has ended")
    try:
        require_supervision_clear(settings.run_dir)
        ensure_supervision_safe()
        with deployment_deadline(
            "BOOT-032 full uninstall", seconds, recovery_seconds=0
        ):
            with ExitStack() as stack:
                descriptors = {
                    root: stack.enter_context(site_operation_lock(root, wait=False))
                    for root in sorted(
                        {
                            settings.target.source.parent,
                            settings.protected.source.parent,
                        },
                        key=str,
                    )
                }
                previous = os.environ.get(SITE_OPERATION_LOCK_FD_ENV)
                os.environ[SITE_OPERATION_LOCK_FD_ENV] = str(
                    descriptors[settings.target.source.parent]
                )
                try:
                    yield
                finally:
                    if previous is None:
                        os.environ.pop(SITE_OPERATION_LOCK_FD_ENV, None)
                    else:
                        os.environ[SITE_OPERATION_LOCK_FD_ENV] = previous
    except ProcessSupervisionLost:
        record_supervision_loss()
        raise


class ProofPauseRunner(CommandRunner):
    def __init__(
        self, settings: Settings, binding: dict[str, Any], *, pause: bool
    ) -> None:
        super().__init__()
        self.settings = settings
        self.binding = binding
        self.pause = pause

    def run(
        self,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        capture: bool = True,
        sensitive: bool = False,
        mutate: bool = False,
        timeout_seconds: float | None = None,
    ) -> str:
        require(
            self.settings.inputs() == self.binding["inputs"],
            "site, context or configuration changed during native execution",
        )
        cleanup_script = (
            self.settings.target.repository_root
            / "deploy/control-plane/regional/prepare-clean-redeploy.sh"
        )
        if arguments and Path(arguments[0]) == cleanup_script:
            export_receipts(
                self.settings,
                self.binding,
                required_native_state(self.settings),
            )
        output = super().run(
            arguments,
            input_text=input_text,
            env=env,
            cwd=cwd,
            capture=capture,
            sensitive=sensitive,
            mutate=mutate,
            timeout_seconds=timeout_seconds,
        )
        tool = (
            self.settings.target.repository_root
            / "deploy/control-plane/tools/cleanup_kubernetes.py"
        )
        if (
            self.pause
            and len(arguments) > 2
            and Path(arguments[1]) == tool
            and arguments[-1] == "verify-targets"
        ):
            cleanup = cleanup_complete(self.settings, self.binding)
            verify_installed_registry_cleanup(self.settings.target, cleanup)
            pause_proof(self.settings, self.binding)
            ensure_supervision_safe()
            self.pause = False
            raise RestartRequired(
                "native Kubernetes proof completed; restart the case process"
            )
        return output


class NativeBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def initial(self) -> dict[str, Any]:
        require(
            not self.settings.native_dir.exists(),
            "pre-existing uninstall state cannot be adopted by BOOT-032",
        )
        return initial_binding(self.settings)

    def check(self, binding: dict[str, Any]) -> None:
        settings = self.settings
        require(
            settings.inputs() == binding["inputs"],
            "approved two-site input binding changed",
        )
        require(
            observe_site(settings.protected) == binding["protected"],
            "protected accepted-site identity or resource inventory changed",
        )
        state = native_state(settings)
        if state is None:
            current_target = observe_site(
                settings.target, fixture_id=settings.fixture_id
            )
            require(
                current_target["nodes"] == binding["target"]["nodes"],
                "approved GPU node or installed unit identity changed",
            )
            require(
                current_target == binding["target"],
                "approved sacrificial target runtime, registry or object identity changed",
            )
            return
        cleanup = None
        cleanup_path = settings.native_dir / "kubernetes-cleanup.json"
        if state is not None and cleanup_path.exists():
            from scripts.e2e.regional.boot032_journal import cleanup_receipt

            cleanup = cleanup_receipt(settings, binding)
        if cleanup is None:
            require(
                node_observation(settings.target) == binding["target"]["nodes"],
                "approved GPU node or installed unit identity changed",
            )
            inventory = installed_inventory(settings.target)
            require(
                runtime_observation(settings.target) == binding["target"]["runtime"]
                and details_sha256(inventory) == binding["target"]["inventory_sha256"]
                and inventory_uids(settings.target, inventory)
                == binding["target"]["resource_uids"],
                "target runtime or installed inventory changed before native cleanup",
            )
        for spec in cluster_specs(settings.target):
            expected = binding["target"]["clusters"][spec["context"]]
            deleting_cpu = (
                spec["plane"] == "cpu"
                and state is not None
                and reached(state, "CPU_DELETE_IN_PROGRESS")
            )
            current = cluster_observation(
                settings.target,
                spec,
                fixture_id=settings.fixture_id,
                cpu_deletion_started=deleting_cpu,
            )
            if deleting_cpu:
                state = required_native_state(settings)
                require(
                    state["cpu_binding"]
                    == {
                        "cpu_eks_created_at": expected["eks_created_at"],
                        "cpu_hyperpod_arn": expected["hyperpod_arn"],
                    },
                    "saved CPU incarnation binding differs from approval",
                )
                if reached(state, "CPU_VERIFIED"):
                    require(
                        current == {"eks_absent": True, "hyperpod_absent": True},
                        "deleted CPU cluster is present or was recreated",
                    )
                else:
                    require(
                        all(
                            current[key] == expected[key]
                            for key in (
                                "eks_arn",
                                "eks_created_at",
                                "hyperpod_arn",
                                "hyperpod_name",
                            )
                            if key in current
                        ),
                        "CPU incarnation changed during deletion",
                    )
                continue
            allowed_absence = (
                cleanup is not None
                and cleanup.get("phase") == "CLEANUP_COMPLETED"
                and cleanup.get("status") == "COMPLETED"
            )
            desired = (
                {**expected, "namespace_uid": None} if allowed_absence else expected
            )
            require(
                current == desired,
                "physical cluster, Kubernetes context or namespace changed",
            )
        if cleanup is not None and cleanup.get("status") == "COMPLETED":
            for spec in cluster_specs(settings.target)[1:]:
                for name, uid in (
                    cleanup.get("node_targets", {})
                    .get("gpu:" + spec["context"], {})
                    .items()
                ):
                    require(
                        kubernetes_uid(settings.target, spec, "node", name) == uid,
                        "surviving GPU node identity changed after cleanup",
                    )
        if state is not None and reached(state, "REGISTRY_EXPORTED"):
            export_receipts(settings, binding, state)

    def invoke(self, binding: dict[str, Any], *, pause: bool) -> dict[str, Any]:
        self.check(binding)
        # This public entry retains native source reloading, confirmation, locks,
        # registry policies, process supervision and every destructive preflight.
        with operation_budget("admin/uninstall"):
            result = uninstall(
                UninstallRequest(
                    site=self.settings.target,
                    cpu_disposition="delete",
                    confirmation="DELETE_CPU_CONTROL_PLANE",
                    final_snapshot_policy="retain",
                ),
                runner=ProofPauseRunner(self.settings, binding, pause=pause),
            )
        ensure_supervision_safe()
        return result

    def paused(self, binding: dict[str, Any]) -> dict[str, Any]:
        self.check(binding)
        ensure_supervision_safe()
        return pause_proof(self.settings, binding)

    def final(self, binding: dict[str, Any], pause: dict[str, Any]) -> dict[str, Any]:
        self.check(binding)
        snapshot, cleanup = final_receipts(self.settings, binding, pause)
        probe = ResourceProbe(self.settings.target)
        readbacks = {}
        for resource in snapshot.resources:
            expected = resource.status.value == "PRESERVED"
            inherited_cpu_proof = (
                resource.resource_type == "helm_release"
                or resource.resource_key in CPU_CLUSTER_BOUND_RESOURCE_KEYS
            )
            if inherited_cpu_proof:
                require(
                    not expected,
                    "deleted CPU has a supposedly preserved bound resource",
                )
                observed = False
                source = "verified CPU absence and sealed pre-CPU cleanup"
            else:
                observed = probe.exists(resource)
                source = "independent native resource readback"
            require(
                type(observed) is bool and observed is expected,
                "final resource deletion or preservation is unproved",
            )
            readbacks[resource.resource_key] = {
                "expected_present": expected,
                "observed_present": observed,
                "proof": source,
            }
        kubernetes = verify_installed_registry_cleanup(
            self.settings.target,
            cleanup,
            cpu_deleted=True,
        )
        return {
            "native_phase": "COMPLETED",
            "final_registry_sha256": snapshot.digest(),
            "resource_readbacks": readbacks,
            "kubernetes": kubernetes,
            "cpu_deleted": True,
            "gpu_clusters_preserved": len(cluster_specs(self.settings.target)) - 1,
            "protected_site_unchanged": True,
            "node_cleanup_contexts": sorted(cleanup["node_targets"]),
        }
