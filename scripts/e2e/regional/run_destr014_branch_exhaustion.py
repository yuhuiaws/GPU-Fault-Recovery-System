#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-014 live acceptance runner.

One two-node PyTorchJob (node-b + node-c, 16 GPUs) faults both nodes inside the
multi-node aggregation window so they join one job DAG. node-b's RESET_GPU is
made to FAIL at the reset step (a GPU device holder armed after this drill's
VERIFY_NO_GPU_CLIENTS succeeds); its branch escalates in place to a *real*
HyperPod reboot that succeeds and the branch ends normally. node-c's branch
resolves RESTART_NODE, but its Node Agent is disabled (without --now) before the
reboot so no new boot id is reported: RESTART_NODE waits to its managed-recovery
timeout and retains BLOCKED/NEEDS_OPERATOR. A timeout does not authorize a
replacement rung. The join never runs and the job is not restarted.
Confirmed no-start failure and replacement exhaustion are separate local
contracts; this live drill proves an unknown-reboot hold. Even when that
safety assertion passes, final acceptance remains BLOCKED until operator
reconciliation and cleanup are complete.

Two env windows shorten the wait. The managed-recovery timeout is a
control-worker setting (``execution/config.py`` derives the RESTART_NODE and
REPLACE_NODE waiting caps from it), so it is compressed through the
control-plane window; the executor window lowers only the GPU client verify
attempts. Attempts 1-7 (2026-09-08) set both on the executor Deployment, where
nothing reads the timeout, and the control plane kept its 1800 s default.

The runner defaults to ``--plan``; ``--execute`` needs ``--confirm
DESTR014_EXECUTE``. The verdict functions are pure and unit-tested; the runner
records digests only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import control_plane_env_window as control_window  # noqa: E402
from scripts.e2e.regional import executor_env_window as env_window  # noqa: E402
from scripts.e2e.regional import run_destr018_lifetime_deadline as destr018  # noqa: E402
from scripts.e2e.regional.run_destr015_parallel_branch_join import (  # noqa: E402
    cluster_arbiter_placement,
)
from scripts.e2e.regional.destr014_preflight import (  # noqa: E402
    BUDGET_HEADROOM as BUDGET_HEADROOM,
    _control_env,
    budget_headroom,
    executor_env_snapshot,
    managed_recovery_errors,
    preflight_errors,
)
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost  # noqa: E402
from scripts.e2e.regional import destr014_recovery as recovery  # noqa: E402
from scripts.e2e.regional import destr014_release as release  # noqa: E402
from scripts.e2e.regional.destr014_cleanup import (  # noqa: E402
    _cleanup as _cleanup,
    _ledger as _ledger,
    _restore_isolated_nodes as _restore_isolated_nodes,
    recreate_probe as recreate_probe,
    resume_hold_review,
    wait_agent_unit,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    install_abort_signals,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.site_profile import (  # noqa: E402
    install_site_profile,
    supplied_flags,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    QUARANTINE_TAINT as QUARANTINE_TAINT,
    WarmSpareLiveFixture,
)

yaml = importlib.import_module("yaml")

DESTRUCTIVE_PROBE = Path(__file__).with_name("probes") / "destructive_node_probe.py"
DESTR014_PROBE = Path(__file__).with_name("probes") / "destr014_node_probe.py"
DEFAULT_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "two-node-replace-pytorchjob.yaml"
)
CASE_ID = "GF-REGIONAL-DESTR-014"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-008"
CONFIRMATION = "DESTR014_EXECUTE"
VARIANTS = ("unknown-reboot",)

from scripts.e2e.regional.destr014_verdicts import (  # noqa: E402,F401
    canonical_digest as _digest,
    case_digest as case_digest,
    evidence_components as evidence_components,
    identity_digest as identity_digest,
    plan_identity as plan_identity,
    CONTAINMENT_ALLOWANCE_SECONDS,
    EXHAUSTION_PREFIX,
    FORBIDDEN_EVENTS,
    REBOOT_ALLOWANCE_SECONDS,
    REBOOT_EVENTS,
    VALIDATION_ALLOWANCE_SECONDS,
    _branch_executions,
    _first,
    _step_node,
    budget_headroom_errors,
    cloudtrail_errors,
    estimated_duration_seconds,
    follow_up_errors,
    host_errors,
    injection_errors,
    lifetime_errors,
    operator_hold_reasons,
    product_hold_reasons,
    quarantine_taint_value,
    reset_reached_commit,
    schedulability_errors,
    step_transitions,
    workflow_errors,
    workload_errors,
)


def render_two_node_manifest(
    source: Path,
    destination: Path,
    *,
    master_node: str,
    worker_node: str,
) -> Path:
    if master_node == worker_node:
        raise ValueError("master and worker nodes must differ")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    replicas = document["spec"]["pytorchReplicaSpecs"]
    replicas["Master"]["template"]["spec"]["nodeName"] = master_node
    replicas["Worker"]["template"]["spec"]["nodeName"] = worker_node
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    destination.chmod(0o600)
    return destination


def derived_identity(run_dir: Path, attempt: int) -> tuple[str, str]:
    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{CASE_ID}".encode()
    ).hexdigest()[:12]
    job_id = f"destr014-{suffix}"
    return job_id, f"{job_id}-a001"


# --------------------------------------------------------------------------- #
# Settings / plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    site_file: Path
    manifest: Path
    hyperpod_cluster: str
    host_probe_image: str
    fault_node: str
    fault_pci_bdf: str
    fault_device: str
    sibling_node: str
    sibling_pci_bdf: str
    job_id: str
    attempt_id: str
    verify_max_attempts: int
    managed_recovery_timeout_seconds: int
    variant: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_TRAINING_MANIFEST": str(self.manifest),
            "GPU_FAULT_HYPERPOD_CLUSTER_NAME": self.hyperpod_cluster,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_FAULT_NODE": self.fault_node,
            "GPU_FAULT_SIBLING_NODE": self.sibling_node,
            "GPU_FAULT_TEST_JOB_ID": self.job_id,
            "GPU_FAULT_TEST_ATTEMPT_ID": self.attempt_id,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
    return _settings(arguments, strict=True)


def release_settings(
    arguments: argparse.Namespace, identity: release.AttemptIdentity
) -> Settings:
    """Settings for ``--release``: the device identities the injection needs
    (PCI BDFs, device file) are not required to release a hold, and the node
    pair plus job/attempt id are the attempt's plan's, already checked against
    whatever the operator typed (`release.bind_attempt_identity`)."""

    return _settings(arguments, strict=False, identity=identity)


def _settings(
    arguments: argparse.Namespace,
    *,
    strict: bool,
    identity: release.AttemptIdentity | None = None,
) -> Settings:
    def device(value: str, label: str) -> str:
        return required(value, label) if strict else value.strip()

    default_job, default_attempt = derived_identity(
        arguments.run_dir, arguments.attempt
    )
    if identity is None:
        fault_node = required(
            arguments.fault_node or os.getenv("GPU_FAULT_FAULT_NODE", ""),
            "fault node",
        )
        sibling_node = required(
            arguments.sibling_node or os.getenv("GPU_FAULT_SIBLING_NODE", ""),
            "sibling node",
        )
        job_id = arguments.job_id.strip() or default_job
        attempt_id = arguments.attempt_id.strip() or default_attempt
    else:
        # The plan's words, not the profile's or the environment's defaults.
        fault_node, sibling_node = identity.fault_node, identity.sibling_node
        job_id, attempt_id = identity.job_id, identity.attempt_id
    problems = managed_recovery_errors(int(arguments.managed_recovery_timeout_seconds))
    if problems:
        raise RegionalFixtureError(
            "managed recovery timeout is not a value the control plane accepts: "
            + "; ".join(problems)
        )
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir
            / "cases"
            / PREDECESSOR_CASE_ID
            / f"{PREDECESSOR_CASE_ID}.json"
        ).resolve()
    )
    return Settings(
        regional=settings_from_arguments(arguments),
        site_file=Path(
            required(
                arguments.site_file or os.getenv("GPU_FAULT_SITE_FILE", ""),
                "regional site file",
            )
        )
        .expanduser()
        .resolve(),
        manifest=Path(arguments.manifest).expanduser().resolve(),
        hyperpod_cluster=required(
            arguments.hyperpod_cluster
            or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", ""),
            "HyperPod cluster name",
        ),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        fault_node=fault_node,
        fault_pci_bdf=device(arguments.fault_pci_bdf, "fault PCI BDF"),
        fault_device=device(arguments.fault_device, "fault GPU device"),
        sibling_node=sibling_node,
        sibling_pci_bdf=device(arguments.sibling_pci_bdf, "sibling PCI BDF"),
        job_id=job_id,
        attempt_id=attempt_id,
        verify_max_attempts=int(arguments.verify_max_attempts),
        managed_recovery_timeout_seconds=int(
            arguments.managed_recovery_timeout_seconds
        ),
        variant=arguments.variant,
        predecessor_path=predecessor,
    )


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    identity = plan_identity(preflight)
    return {
        "risk": "destructive-provider-reboot",
        "variant": settings.variant,
        "predecessor": preflight.get("predecessor"),
        "fault_node": settings.fault_node,
        "sibling_node": settings.sibling_node,
        "job_id": settings.job_id,
        "attempt_id": settings.attempt_id,
        "manifest": str(settings.manifest),
        "verify_max_attempts": settings.verify_max_attempts,
        "managed_recovery_timeout_seconds": settings.managed_recovery_timeout_seconds,
        "recovery_safeguard": {
            "kind": "persistent-host-systemd",
            "helper_sha256": hashlib.sha256(recovery.PROBE.read_bytes()).hexdigest(),
            "recovery_seconds": recovery.RECOVERY_SECONDS,
            "hold_seconds": recovery_hold_seconds(settings, preflight),
            "requires_independent_arm_ack_before_disable": True,
            "resume": "cleanup-only",
        },
        "mutation": (
            "submit one two-node 16-GPU PyTorchJob, fault both nodes into one job "
            "DAG; node-b RESET_GPU is held to failure and escalates to a real "
            "reboot that succeeds; node-c reboots but its Agent is disabled so "
            "RESTART_NODE reaches its confirmation timeout; the physical outcome "
            "remains unknown, no replacement is authorized, and the workflow "
            "retains BLOCKED/NEEDS_OPERATOR without restarting the job"
        ),
        "proof_scope": {
            "live": "unconfirmed reboot retains operator hold, not replacement exhaustion",
            "confirmed_failure_replacement": "separate deterministic contract",
            "formal_pass_requires_operator_reconciliation_and_complete_cleanup": True,
        },
        "preflight_identity": identity,
        "preflight_identity_digest": identity_digest(identity),
        "stop_conditions": [
            "DESTR-008 predecessor evidence is not PASS",
            "a warm spare is declared (the sibling REPLACE must exhaust)",
            "reboot disabled, NodeRecovery not None, or rung ceiling below 2",
            "a target node is busy, tainted, or carries a GPU workload",
            "the estimated duration does not fit the job workflow lifetime",
            "the two XID injections do not join one DAG workflow",
            "independent recovery arm is absent, stale, or changed",
            "automatic Agent recovery fires before the scenario is complete",
            "workflow does not retain BLOCKED/NEEDS_OPERATOR for unknown reboot",
            "CloudTrail shows other than two reboots or any replace/delete",
            "cleanup cannot disarm the holder, restore the agent, un-quarantine "
            "the sibling, or close the executor env window",
        ],
        "rollback": {
            "disarm_the_gpu_device_holder": True,
            "restore_the_sibling_node_agent_to_its_baseline": True,
            "prove_recovery_stopped_before_removing_owned_units": True,
            "unknown_physical_outcomes_forbid_automatic_isolation_restore": True,
            "close_the_executor_env_window_to_baseline": True,
            "close_the_control_plane_env_window_to_baseline": True,
            "delete_the_test_workload_after_async_quiescence": True,
            "never_call_provider_replace_or_delete": True,
        },
    }


# --------------------------------------------------------------------------- #
# Live preflight (not unit-tested; delegates every assertion to the pure funcs)
# --------------------------------------------------------------------------- #
def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    if not settings.site_file.is_file() or not settings.manifest.is_file():
        raise RegionalFixtureError("site file or training manifest does not exist")
    regional = RegionalLiveFixture(settings.regional)
    warm = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    fault = warm.node_snapshot(settings.fault_node)
    sibling = warm.node_snapshot(settings.sibling_node)
    state = warm.store_snapshot()
    fault_state = regional.store_snapshot(node=settings.fault_node)
    state["profile"] = fault_state.get("profile")
    state["release_id"] = fault_state.get("release_id")
    control_env = _control_env(regional)
    arbiter_pods, dns_nodes = cluster_arbiter_placement(regional)
    from scripts.e2e.regional.warm_spare_fixture import agent_by_node

    result: dict[str, Any] = {
        "release_id": state.get("release_id"),
        "fault_node": regional.node_snapshot(settings.fault_node),
        "sibling_node": regional.node_snapshot(settings.sibling_node),
        "arbiter_pods": arbiter_pods,
        "dns_nodes": dns_nodes,
        "store": state,
        "provider_inventory": warm.provider_inventory(),
        "cpu_blast": regional.cpu_blast_snapshot(),
        "predecessor": predecessor_evidence(
            settings.predecessor_path, PREDECESSOR_CASE_ID
        ),
        "control_env": control_env,
        "budget": budget_headroom(regional, settings),
        "recovery_agent": {
            key: (agent_by_node(state, settings.sibling_node) or {}).get(key)
            for key in (
                "node_instance_id",
                "artifact_sha256",
                "installer_bundle_sha256",
                "runtime_profile_version",
            )
        },
    }
    result["errors"] = preflight_errors(
        fault_node=settings.fault_node,
        sibling_node=settings.sibling_node,
        fault=fault,
        sibling=sibling,
        fault_agent=agent_by_node(state, settings.fault_node) or {},
        sibling_agent=agent_by_node(state, settings.sibling_node) or {},
        spare_nodes=warm.spare_nodes(),
        cluster=warm.cluster_recovery(),
        executor_env=executor_env_snapshot(regional),
        reboot_probe_errors=[],
        control_env=control_env,
        budget=result["budget"],
        fault_workloads=regional.business_workloads(settings.fault_node),
        sibling_workloads=regional.business_workloads(settings.sibling_node),
        open_workflows=[],
        gpu_workloads=regional.gpu_workloads(),
        predecessor=result["predecessor"],
        tests={"passed": True},
        verify_max_attempts=settings.verify_max_attempts,
        managed_recovery_timeout_seconds=settings.managed_recovery_timeout_seconds,
        arbiter_pods=arbiter_pods,
        dns_nodes=dns_nodes,
    )
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the guarded DESTR-014 branch-exhaustion acceptance."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--hyperpod-cluster", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--fault-node", default="")
    value.add_argument("--fault-pci-bdf", default="")
    value.add_argument("--fault-device", default="")
    value.add_argument("--sibling-node", default="")
    value.add_argument("--sibling-pci-bdf", default="")
    value.add_argument("--job-id", default="")
    value.add_argument("--attempt-id", default="")
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument("--verify-max-attempts", type=int, default=6)
    value.add_argument("--managed-recovery-timeout-seconds", type=int, default=600)
    value.add_argument("--variant", choices=VARIANTS, default="unknown-reboot")
    value.add_argument(
        "--release",
        action="store_true",
        help=(
            "release a held attempt (no new plan; the node pair and job identity "
            "are bound from the attempt's recorded plan, typed --fault-node/"
            "--sibling-node only have to agree): confirm-node-action for the "
            "rebooted node, --disposition restore, validated restore of every "
            "isolated node, workflow-reconcile, delete the drill job, disarm "
            "the holder, close both "
            f"env windows, close the drill incidents; requires --confirm "
            f"{release.RELEASE_CONFIRMATION}"
        ),
    )
    return value


def _host_probe(
    settings: Settings, node: str, run_id: str, case_dir: Path
) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            state_directory=case_dir / "host-probes",
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=DESTR014_PROBE,
            active_deadline_seconds=3600,
        )
    )


def _destructive_probe(
    settings: Settings, node: str, run_id: str, case_dir: Path
) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            state_directory=case_dir / "host-probes",
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=node,
            image=settings.host_probe_image,
            case_id=f"{CASE_ID}-inject",
            run_id=run_id,
            probe_script=DESTRUCTIVE_PROBE,
            active_deadline_seconds=3600,
        )
    )


def _recovery_probe(
    settings: Settings, run_id: str, case_dir: Path
) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            state_directory=case_dir / "host-probes",
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.sibling_node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=recovery.PROBE,
            active_deadline_seconds=7200,
        )
    )


def recovery_hold_seconds(settings: Settings, preflight: dict[str, Any]) -> int:
    return (
        estimated_duration_seconds(
            verify_max_attempts=settings.verify_max_attempts,
            poll_interval_seconds=float(
                (preflight.get("control_env") or {}).get("poll_interval_seconds") or 5
            ),
            managed_recovery_timeout_seconds=settings.managed_recovery_timeout_seconds,
        )
        + 120
    )


@dataclass
class _LiveRun:
    """Everything one execution owns: fixtures, probes and the flags cleanup
    reads. Kept on one object so the phases below stay short and share no
    module state."""

    settings: Settings
    case_dir: Path
    attempt: int
    maintenance_window_end: datetime
    preflight: dict[str, Any]
    regional: RegionalLiveFixture
    warm: WarmSpareLiveFixture
    run_id: str
    workload: ManagedWorkloadFixture
    prewarm: ImagePrewarmFixture
    env_baseline: Path
    control_env_baseline: Path
    fault_probe: Any
    sibling_probe: Any
    inject_fault: Any
    inject_sibling: Any
    incident_id: str = ""
    follow_up_incident_id: str = ""
    env_opened: bool = False
    control_env_opened: bool = False
    holder_armed: bool = False
    agent_disabled: bool = False
    marker: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_uids: set[str] = field(default_factory=set)
    fault_baseline: dict[str, Any] = field(default_factory=dict)
    sibling_baseline: dict[str, Any] = field(default_factory=dict)
    sibling_agent_during: dict[str, Any] = field(default_factory=dict)
    journal: recovery.RunJournal | None = None
    recovery_window: recovery.AgentRecoveryWindow | None = None
    resumed: bool = False
    injection_started: bool = False
    physical_outcome_unknown: bool = False
    # The two sources of the hold, recorded for the operator: the product's
    # (None until the workflow was read) and the sibling's host proof.
    product_hold_reasons: list[str] | None = None
    hold_reasons: list[str] | None = None
    sibling_reboot_proof: dict[str, Any] = field(default_factory=dict)


def _checkpoint(run: _LiveRun) -> None:
    if run.journal is not None:
        run.journal.checkpoint(
            env_opened=run.env_opened,
            control_env_opened=run.control_env_opened,
            holder_armed=run.holder_armed,
            agent_disabled=run.agent_disabled,
            injection_started=run.injection_started,
            physical_outcome_unknown=run.physical_outcome_unknown,
            hold_reasons=run.hold_reasons,
            sibling_reboot_proof=run.sibling_reboot_proof,
            incident_id=run.incident_id,
            follow_up_incident_id=run.follow_up_incident_id,
            marker=run.marker,
            started_at=run.started_at.isoformat(),
        )


def _prepare_live_run(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> _LiveRun:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    planned = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    run_id = f"destr014-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    identity = planned["details"]["preflight_identity"]
    journal = recovery.RunJournal(
        case_dir / "destr014-recovery-journal.json",
        {
            "environment": settings.environment(),
            "run_id": run_id,
            "attempt": attempt,
            "release_id": identity["release_id"],
            "plan_sha256": _digest(planned),
            "cluster_id": settings.regional.cluster_id,
            "node": settings.sibling_node,
            "node_uid": identity["sibling_node_uid"],
            "boot_id": identity["sibling_node_boot_id"],
        },
    )
    journal.acquire()
    try:
        if journal.resumed:
            preflight = journal.data["run"].get("preflight")
            if not isinstance(preflight, dict):
                raise RegionalFixtureError("DESTR-014 setup journal is incomplete")
        else:
            preflight = read_only_preflight(settings, case_dir)
            if preflight["errors"]:
                raise RegionalFixtureError(
                    "preflight failed: " + "; ".join(preflight["errors"])
                )
            if planned["details"]["preflight_identity_digest"] != identity_digest(
                plan_identity(preflight)
            ):
                raise RegionalFixtureError(
                    "DESTR-014 plan identity drifted before execution"
                )
            journal.checkpoint(preflight=preflight)
        return _build_live_run(
            settings,
            case_dir,
            attempt,
            maintenance_window_end,
            preflight,
            run_id,
            journal,
        )
    except BaseException:
        journal.close()
        raise


def _build_live_run(
    settings: Settings,
    case_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
    preflight: dict[str, Any],
    run_id: str,
    journal: recovery.RunJournal,
) -> _LiveRun:
    regional = RegionalLiveFixture(settings.regional)
    pinned = case_dir / "pinned-workload.yaml"
    if not journal.resumed:
        render_two_node_manifest(
            settings.manifest,
            pinned,
            master_node=settings.fault_node,
            worker_node=settings.sibling_node,
        )
    workload = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=pinned,
            site_file=settings.site_file,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            restart_budget=1,
            expected_pods=2,
            expected_gpu_count=16,
        ),
    )
    run = _LiveRun(
        settings=settings,
        case_dir=case_dir,
        attempt=attempt,
        maintenance_window_end=maintenance_window_end,
        preflight=preflight,
        regional=regional,
        warm=WarmSpareLiveFixture(regional, settings.hyperpod_cluster),
        run_id=run_id,
        workload=workload,
        prewarm=ImagePrewarmFixture(regional, case_id=CASE_ID, run_id=run_id),
        env_baseline=case_dir / "executor-env-window.json",
        control_env_baseline=case_dir / "control-plane-env-window.json",
        fault_probe=_host_probe(settings, settings.fault_node, run_id, case_dir),
        sibling_probe=_host_probe(settings, settings.sibling_node, run_id, case_dir),
        inject_fault=_destructive_probe(
            settings, settings.fault_node, f"{run_id}-b", case_dir
        ),
        inject_sibling=_destructive_probe(
            settings, settings.sibling_node, f"{run_id}-c", case_dir
        ),
        journal=journal,
        recovery_window=recovery.AgentRecoveryWindow(
            journal,
            _recovery_probe(settings, run_id, case_dir),
        ),
        resumed=journal.resumed,
    )
    if journal.resumed:
        saved = journal.data["run"]
        for name in (
            "env_opened",
            "control_env_opened",
            "holder_armed",
            "agent_disabled",
            "injection_started",
            "physical_outcome_unknown",
        ):
            value = saved.get(
                name,
                bool(saved.get("injection_started"))
                if name == "physical_outcome_unknown"
                else False,
            )
            if type(value) is not bool:
                raise RegionalFixtureError("DESTR-014 cleanup intent is malformed")
            setattr(run, name, value)
        for name in ("incident_id", "follow_up_incident_id", "marker"):
            value = saved.get(name, "")
            if not isinstance(value, str):
                raise RegionalFixtureError("DESTR-014 cleanup identity is malformed")
            setattr(run, name, value)
        run.started_at = datetime.fromisoformat(
            saved.get("started_at") or run.started_at.isoformat()
        )
        if isinstance(saved.get("hold_reasons"), list):
            run.hold_reasons = [str(item) for item in saved["hold_reasons"]]
        if isinstance(saved.get("sibling_reboot_proof"), dict):
            run.sibling_reboot_proof = dict(saved["sibling_reboot_proof"])
    return run


def _arm_and_inject(run: _LiveRun) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open the control-plane and executor env windows, start the job, arm
    both node-side failures, inject the two faults and wait until both land in
    one DAG."""

    settings, case_dir, run_id = run.settings, run.case_dir, run.run_id
    # The managed-recovery window is read by the control-worker (it derives the
    # RESTART_NODE/REPLACE_NODE waiting caps from it); opening it rolls every
    # worker replica, so it goes first and before anything is injected.
    if datetime.now(timezone.utc) >= run.maintenance_window_end:
        raise RegionalFixtureError("maintenance window ended before configuration")
    run.control_env_opened = True
    _checkpoint(run)
    control_report = control_window.open_window(
        control_window.Settings(
            baseline=run.control_env_baseline, rollout_timeout_seconds=600
        ),
        run.regional,
        control_window.survey(run.regional),
        {
            control_window.MANAGED_RECOVERY_VARIABLE: str(
                settings.managed_recovery_timeout_seconds
            )
        },
    )
    write_json_atomic(
        case_dir / "control-plane-env-window-open.json",
        control_window.without_survey(control_report),
    )
    if datetime.now(timezone.utc) >= run.maintenance_window_end:
        raise RegionalFixtureError(
            "maintenance window ended before executor configuration"
        )
    run.env_opened = True
    _checkpoint(run)
    env_report = env_window.open_window(
        env_window.Settings(baseline=run.env_baseline, rollout_timeout_seconds=300),
        run.regional,
        env_window.survey(run.regional),
        {
            "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": str(
                settings.verify_max_attempts
            ),
        },
    )
    write_json_atomic(case_dir / "env-window-open.json", env_report)
    run.prewarm.create([settings.fault_node, settings.sibling_node])
    run.workload.submit()
    source = run.workload.wait_running(timeout_seconds=900)
    run.source_uids = {str(item["uid"]) for item in source["pods"]}
    write_json_atomic(case_dir / "workload-source.json", source)
    run.fault_probe.create()
    run.sibling_probe.create()
    run.inject_fault.create()
    run.inject_sibling.create()
    run.fault_baseline = run.fault_probe.execute("snapshot", "--run-id", run_id)
    run.sibling_baseline = run.sibling_probe.execute("snapshot", "--run-id", run_id)
    write_json_atomic(case_dir / "fault-baseline.json", run.fault_baseline)
    write_json_atomic(case_dir / "sibling-baseline.json", run.sibling_baseline)
    if datetime.now(timezone.utc) >= run.maintenance_window_end:
        raise RegionalFixtureError("maintenance window ended before injection")
    if run.recovery_window is None or run.journal is None:
        raise RegionalFixtureError(
            "independent Node Agent recovery safeguard is missing"
        )
    restore_at = int(time.time()) + recovery_hold_seconds(settings, run.preflight)
    expires_at = restore_at + recovery.RECOVERY_SECONDS
    if expires_at > run.maintenance_window_end.timestamp():
        raise RegionalFixtureError(
            "maintenance window cannot contain bounded Agent recovery"
        )
    binding = recovery.make_binding(
        scope=run.journal.scope,
        owner=uuid4().hex,
        agent=run.preflight["recovery_agent"],
        restore_at=restore_at,
        expires_at=expires_at,
    )
    run.recovery_window.arm(binding)
    run.agent_disabled = True
    _checkpoint(run)
    run.recovery_window.disable()
    run.sibling_agent_during = run.sibling_probe.execute("snapshot", "--run-id", run_id)
    run.holder_armed = True
    _checkpoint(run)
    run.fault_probe.execute(
        "arm-holder",
        "--device",
        settings.fault_device,
        "--drill-id",
        run_id,
        "--after-ledger-op",
        "VERIFY_NO_GPU_CLIENTS",
        "--max-hold-seconds",
        "900",
        "--run-id",
        run_id,
        "--probe-script",
        run.fault_probe.host_script,
    )
    run.started_at = datetime.now(timezone.utc)
    run.marker = f"destr014-{int(time.time())}-a{run.attempt}"
    run.injection_started = True
    run.physical_outcome_unknown = True
    _checkpoint(run)
    run.inject_sibling.execute(
        "write-xid79",
        "--marker",
        run.marker,
        "--drill-id",
        f"{run_id}-c",
        "--pci-bdf",
        settings.sibling_pci_bdf,
    )
    run.inject_fault.execute(
        "write-xid46",
        "--marker",
        run.marker,
        "--drill-id",
        f"{run_id}-b",
        "--pci-bdf",
        settings.fault_pci_bdf,
    )
    fault_state = run.regional.wait_for_workflow(
        node=settings.fault_node,
        marker=run.marker,
        observed_after=run.started_at,
        case_dir=case_dir,
        timeout_seconds=300,
        job_id=settings.job_id,
        attempt_id=settings.attempt_id,
        hyperpod_cluster=settings.hyperpod_cluster,
        terminal=False,
    )
    sibling_state = run.regional.wait_for_workflow(
        node=settings.sibling_node,
        marker=run.marker,
        observed_after=run.started_at,
        case_dir=case_dir,
        timeout_seconds=300,
        hyperpod_cluster=settings.hyperpod_cluster,
        terminal=False,
    )
    run.incident_id = str((fault_state.get("incident") or {}).get("incident_id") or "")
    _checkpoint(run)
    return fault_state, sibling_state


def _observe_until_terminal(run: _LiveRun, initial: dict[str, Any]) -> dict[str, Any]:
    """Poll the job workflow, recording every step transition, until it
    reaches a terminal status or the observation budget runs out."""

    settings = run.settings
    transitions: dict[str, str] = {}
    timeline: list[dict[str, Any]] = []
    deadline = time.monotonic() + 2400
    state = initial
    while time.monotonic() < deadline:
        state = run.regional.store_snapshot(
            node=settings.fault_node,
            marker=run.marker,
            observed_after=run.started_at,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            hyperpod_cluster=settings.hyperpod_cluster,
            queue_attempts=1,
        )
        workflow = state.get("workflow") or {}
        transitions, changes = step_transitions(
            transitions, workflow.get("step_executions") or []
        )
        timeline.extend(changes)
        write_json_atomic(
            run.case_dir / "step-timeline.json", {"transitions": timeline}
        )
        if workflow.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED"}:
            break
        time.sleep(5)
    write_json_atomic(run.case_dir / "workflow-state.json", state)
    return state


def _control_plane_errors(
    run: _LiveRun, state: dict[str, Any]
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    settings = run.settings
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    run.incident_id = str(incident.get("incident_id") or "")
    errors = workflow_errors(
        workflow,
        incident,
        fault_node=settings.fault_node,
        sibling_node=settings.sibling_node,
        failure_reason=workflow.get("terminal_failure_reason")
        or state.get("workflow_failure_reason"),
        reboot_outcome="unknown",
    )
    # A missing/partial read is not a known failure. The product's share of
    # the hold is read here and persisted before any cleanup; the sibling's
    # physical proof only exists once the data plane has its Agent back.
    run.product_hold_reasons = product_hold_reasons(state)
    run.physical_outcome_unknown = run.physical_outcome_unknown or bool(
        run.product_hold_reasons
    )
    # The support escalation that follows an exhausted branch lives on its own
    # incident/workflow pair keyed by the failed workflow's id; read that pair
    # (attempt 4, 2026-09-08, crashed here on a store_snapshot kwarg that never
    # existed, before any verdict was written).
    follow_up = run.regional.cpu_python(
        destr018.ESCALATION_CHAIN, str(workflow.get("request_id") or "")
    )
    run.follow_up_incident_id = str(
        (follow_up.get("incident") or {}).get("incident_id") or ""
    )
    _checkpoint(run)
    if any(
        step.get("operation") in {"RESTART_NODE", "REPLACE_NODE", "RESTART_WORKLOAD"}
        for step in (follow_up.get("workflow") or {}).get("official_steps") or []
    ):
        errors.append("unknown reboot generated an executable recovery successor")
    return errors, workflow, incident


def _data_plane_errors(
    run: _LiveRun, state: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Node, scheduler, provider and workload verdicts after the workflow
    ended. Restores the sibling's agent on the way (its own verdict reads
    the before/after snapshots)."""

    settings, run_id = run.settings, run.run_id
    errors: list[str] = []
    run.regional.wait_node_ready(settings.fault_node, timeout_seconds=1800)
    run.regional.wait_node_ready(settings.sibling_node, timeout_seconds=1800)
    # Both branches may have rebooted their node; a reboot leaves the host
    # probe Pod Failed and exec into it is refused (attempt 5, 2026-09-08).
    recreate_probe(run.fault_probe)
    recreate_probe(run.sibling_probe)
    holder = run.fault_probe.execute("holder-status", "--run-id", run_id)
    fault_after = run.fault_probe.execute("snapshot", "--run-id", run_id)
    sibling_after = run.sibling_probe.execute("snapshot", "--run-id", run_id)
    if run.recovery_window is None:
        raise RegionalFixtureError("independent Agent recovery binding is missing")
    restored = run.recovery_window.restore()
    write_json_atomic(run.case_dir / "sibling-recovery-restored.json", restored)
    run.agent_disabled = False
    _checkpoint(run)
    # ``systemctl start`` returns before the unit reports active; judge the
    # settled state, not the first sample (attempt 9, 2026-09-09).
    sibling_agent_after = wait_agent_unit(run.sibling_probe, run_id)
    write_json_atomic(run.case_dir / "sibling-agent-after.json", sibling_agent_after)
    # The sibling's physical outcome: a new boot with its Agent restored, read
    # from the host record and this snapshot. It settles the runner's share of
    # the hold; the product's share stays whatever the workflow read said.
    run.sibling_reboot_proof = recovery.reboot_restore_proof(
        restored,
        run.recovery_window.binding,
        agent_unit=sibling_agent_after.get("agent_unit") or {},
    )
    run.hold_reasons = operator_hold_reasons(
        run.product_hold_reasons or [], run.sibling_reboot_proof
    )
    run.physical_outcome_unknown = bool(run.hold_reasons)
    _checkpoint(run)
    errors.extend(
        host_errors(
            fault_baseline={
                "boot_id": run.fault_baseline.get("boot_id"),
                "ledger": _ledger(run.fault_baseline),
            },
            fault_after={
                "boot_id": fault_after.get("boot_id"),
                "ledger": _ledger(fault_after),
            },
            sibling_baseline={"boot_id": run.sibling_baseline.get("boot_id")},
            sibling_after={"boot_id": sibling_after.get("boot_id")},
            holder_status=holder,
            sibling_agent_during=(run.sibling_agent_during.get("agent_unit") or {}),
            sibling_agent_after=(sibling_agent_after.get("agent_unit") or {}),
            reset_reached_commit=reset_reached_commit(
                state.get("workflow") or {}, fault_node=settings.fault_node
            ),
        )
    )
    nodes = {
        settings.fault_node: run.regional.node_snapshot(settings.fault_node),
        settings.sibling_node: run.regional.node_snapshot(settings.sibling_node),
    }
    write_json_atomic(run.case_dir / "nodes-after.json", nodes)
    errors.extend(
        schedulability_errors(
            {
                key: {**value, "taints": value.get("taints") or []}
                for key, value in nodes.items()
            },
            fault_node=settings.fault_node,
            sibling_node=settings.sibling_node,
            # The escalation's QUARANTINE runs under the support-after incident,
            # so that incident's digest is the taint value a correct run leaves.
            incident_id=run.follow_up_incident_id or run.incident_id,
        )
    )
    provider = run.regional.provider_events(run.started_at, datetime.now(timezone.utc))
    write_json_atomic(run.case_dir / "provider-events.json", {"events": provider})
    errors.extend(
        cloudtrail_errors(provider, settings.fault_node, settings.sibling_node)
    )
    provider_before = run.preflight["provider_inventory"]
    if run.warm.provider_inventory()["count"] != provider_before["count"]:
        errors.append("HyperPod node count changed")
    if run.regional.cpu_blast_snapshot() != run.preflight["cpu_blast"]:
        errors.append("control-plane EKS state differs from baseline")
    errors.extend(
        workload_errors(
            pods=run.workload.pods(),
            restart_budget=state.get("restart_budget") or {},
            source_uids=run.source_uids,
        )
    )
    return errors, provider


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    """Drive one attempt; any existing journal resumes cleanup, never injection."""

    run = _prepare_live_run(settings, run_dir, attempt, maintenance_window_end)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "variant": settings.variant,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    journal = getattr(run, "journal", None)
    cleanup_permitted = True
    supervision_lost = False
    try:
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError("maintenance window ended before case setup")
        if run.resumed:
            cleanup_permitted = False
            _resume_identity_guard(run)
            resume_hold_review(run)
            cleanup_permitted = True
            raise RegionalFixtureError(
                "resumed DESTR-014 attempt is cleanup-only; scenario was not rerun"
            )
        fault_state, sibling_state = _arm_and_inject(run)
        errors = injection_errors(fault_state, sibling_state)
        state = _observe_until_terminal(run, fault_state)
        control_errors, workflow, incident = _control_plane_errors(run, state)
        errors.extend(control_errors)
        data_errors, provider = _data_plane_errors(run, state)
        errors.extend(data_errors)
        details = {
            "preflight_identity": plan_identity(run.preflight),
            "workflow": workflow,
            "incident": incident,
            "provider_events": provider,
        }
        components = evidence_components(details)
        held = run.physical_outcome_unknown
        result.update(
            {
                "verdict": ("BLOCKED" if held else "PASS") if not errors else "FAIL",
                "scenario_verdict": "PASS" if not errors else "FAIL",
                "operator_review_required": held,
                "hold_reasons": run.hold_reasons,
                "sibling_reboot_proof": run.sibling_reboot_proof,
                "proof_scope": "unknown-reboot-hold",
                "errors": errors,
                "incident_id": run.incident_id,
                "case_digest": case_digest(components),
                "components": components,
            }
        )
    except ProcessSupervisionLost:
        supervision_lost = True
        cleanup_permitted = False
        result["error"] = (
            "command supervision lost; independent recovery and review required"
        )
        if journal is not None:
            journal.data["supervision_lost"] = True
            journal.data["phase"] = "RECOVERY_REQUIRED"
            journal.save()
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # A run that died before the verdict block still reports a uniform
        # shape: readers (and the cov95 harness) index "errors" unconditionally.
        result.setdefault("errors", [])
        if journal is not None and journal.data["phase"] == "CLOSED":
            cleanup: dict[str, Any] = {"errors": [], "already_closed": True}
        elif not cleanup_permitted:
            cleanup = {
                "errors": ["cleanup deferred: identity or supervision is unproven"]
            }
        else:
            try:
                cleanup = _cleanup(
                    regional=run.regional,
                    warm=run.warm,
                    workload=run.workload,
                    prewarm=run.prewarm,
                    fault_probe=run.fault_probe,
                    sibling_probe=run.sibling_probe,
                    inject_fault=run.inject_fault,
                    inject_sibling=run.inject_sibling,
                    settings=settings,
                    run_id=run.run_id,
                    incident_id=run.incident_id,
                    env_baseline=run.env_baseline,
                    env_opened=run.env_opened,
                    control_env_baseline=run.control_env_baseline,
                    control_env_opened=run.control_env_opened,
                    holder_armed=run.holder_armed,
                    agent_disabled=run.agent_disabled,
                    recovery_window=getattr(run, "recovery_window", None),
                    operator_hold=getattr(run, "physical_outcome_unknown", False),
                    hold_reasons=getattr(run, "hold_reasons", None) or (),
                    product_hold_reasons=getattr(run, "product_hold_reasons", None),
                    profile_version=str(
                        (run.preflight["store"].get("profile") or {}).get(
                            "profile_version"
                        )
                        or ""
                    ),
                )
            except ProcessSupervisionLost:
                supervision_lost = True
                cleanup = {"errors": ["cleanup stopped: command supervision lost"]}
                if journal is not None:
                    journal.data["supervision_lost"] = True
        result["cleanup"] = cleanup
        if cleanup["errors"]:
            result["verdict"] = (
                "BLOCKED"
                if result.get("scenario_verdict") == "PASS"
                and cleanup.get("operator_hold_preserved")
                else "FAIL"
            )
        if journal is not None:
            try:
                journal.data["phase"] = (
                    "CLOSED"
                    if not cleanup["errors"] and not supervision_lost
                    else "RECOVERY_REQUIRED"
                )
                journal.save()
            finally:
                journal.close()
    # A journal armed on a boot the node has since replaced was retired aside
    # when this attempt acquired its own journal; carry the lineage into the
    # case result so cleanup/verdict evidence keeps the pointer to its archive.
    if journal is not None:
        retired = (journal.data.get("run") or {}).get("retired_journals")
        if retired:
            result["retired_journals"] = retired
    write_json_atomic(run.case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def _resume_identity_guard(run: _LiveRun) -> None:
    if run.regional.release_id() != run.preflight["release_id"]:
        raise RegionalFixtureError("DESTR-014 cleanup release identity changed")
    for snapshot_name, node in (
        ("fault_node", run.settings.fault_node),
        ("sibling_node", run.settings.sibling_node),
    ):
        expected = (run.preflight.get(snapshot_name) or {}).get("uid")
        if not expected or run.regional.node_snapshot(node).get("uid") != expected:
            raise RegionalFixtureError("DESTR-014 cleanup Node UID changed")
    if run.injection_started and not run.incident_id:
        snapshot = run.regional.store_snapshot(
            node=run.settings.fault_node,
            marker=run.marker,
            observed_after=run.started_at,
            job_id=run.settings.job_id,
            attempt_id=run.settings.attempt_id,
            hyperpod_cluster=run.settings.hyperpod_cluster,
        )
        run.incident_id = str((snapshot.get("incident") or {}).get("incident_id") or "")
        _checkpoint(run)


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def build_release_context(
    settings: Settings, inputs: release.ReleaseInputs
) -> release.ReleaseContext:
    """The live fixtures a release needs, bound to the attempt's own records."""

    case_dir = inputs.case_dir
    regional = RegionalLiveFixture(settings.regional)
    pinned = case_dir / "pinned-workload.yaml"
    # By identity, not by the run's ownership nonce: see release.PinnedWorkload.
    workload = (
        release.PinnedWorkload(regional, pinned, settings.job_id)
        if pinned.is_file()
        else None
    )
    return release.ReleaseContext(
        settings=settings,
        inputs=inputs,
        regional=regional,
        warm=WarmSpareLiveFixture(regional, settings.hyperpod_cluster),
        state_dir=settings.site_file.parent,
        fault_probe=_host_probe(settings, settings.fault_node, inputs.run_id, case_dir),
        workload=workload,
        admin=release.run_admin,
    )


def run_release(arguments: argparse.Namespace, argv: list[str] | None = None) -> int:
    """``--release``: the documented operator handling of a held attempt.

    ``argv`` is the command line as typed (``sys.argv[1:]`` by default), before
    the site profile filled its unsupplied flags: the node pair and job identity
    come from the attempt's plan, and only flags present there are compared.
    """

    if arguments.plan or arguments.execute:
        raise RegionalFixtureError("--release is its own mode; drop --plan/--execute")
    if arguments.confirm != release.RELEASE_CONFIRMATION:
        raise RegionalFixtureError(
            f"--release requires --confirm {release.RELEASE_CONFIRMATION}"
        )
    inputs = release.load_release_inputs(arguments.run_dir / "cases" / CASE_ID)
    identity = release.bind_attempt_identity(
        inputs.identity,
        arguments,
        supplied_flags(sys.argv[1:] if argv is None else argv),
    )
    settings = release_settings(arguments, identity)
    context = build_release_context(settings, inputs)
    report = release.release_hold(context)
    print(json.dumps(report, sort_keys=True, default=str))
    return 0 if report["released"] else 1


def main() -> int:
    argv = sys.argv[1:]
    if "--release" in argv:
        install_site_profile()
        arguments = parser().parse_args(argv)
        os.umask(0o077)
        install_abort_signals()
        return run_release(arguments, argv)
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
