#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpu_fault.admin.operation_lock import site_operation_lock  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    processor_queue_backlog,
    write_json_atomic,
)
from scripts.e2e.regional.destr008_admission import require_admission_api  # noqa: E402
from scripts.e2e.regional.destr008_capabilities import read_capabilities  # noqa: E402
from scripts.e2e.regional.destr008_contract import (  # noqa: E402
    ALERT_SCENARIOS as ALERT_SCENARIOS,
    BOUND_MARGIN_SECONDS as BOUND_MARGIN_SECONDS,
    EXPECTED_REASON as EXPECTED_REASON,
    SERVICE_UNIT as SERVICE_UNIT,
    WORKFLOW_WAIT_SECONDS as WORKFLOW_WAIT_SECONDS,
    agent_by_node as agent_by_node,
    bound_errors as bound_errors,
    capability as capability,
    instance_type as instance_type,
    observation_gpu_uuids as observation_gpu_uuids,
    profile_errors as profile_errors,
    scenario_admission_errors as scenario_admission_errors,
    scenario_errors as scenario_errors,
    terminal_execution as terminal_execution,
    wait_timeout_seconds as wait_timeout_seconds,
)
from scripts.e2e.regional.destr008_gpu_holder import (  # noqa: E402
    BoundedGpuHolderFixture as GpuHolderFixture,
)
from scripts.e2e.regional.destr008_controller_lock import controller_ownership  # noqa: E402
from scripts.e2e.regional.destr008_journal import (  # noqa: E402
    SCENARIOS as SCENARIOS,
    ExecutionJournal,
    ExecutionRecord,
    ScenarioState,
    retire_completed_execution,
)
from scripts.e2e.regional.destr008_node_restore import (  # noqa: E402
    audit_scenario_nodes as audit_scenario_nodes,
    restore_fault_node as restore_fault_node,
)
from scripts.e2e.regional.destr008_safety import ShortageSafety  # noqa: E402
from scripts.e2e.regional.fixture_ownership import (  # noqa: E402
    file_digest,
    fixture_binding,
)
from scripts.e2e.regional.probes.destr008_cancellation_protocol import (  # noqa: E402
    Plan,
    digest as plan_digest,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    record_focused_tests,
    reusable_focused_tests,
    run_standard_case,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
    render_node_pinned_manifest,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    HYPERPOD_HEALTH_LABEL,
    INSTANCE_GROUP_LABEL,
    PROVIDER_REPLACE_EVENTS,
    QUARANTINE_TAINT,
    SPARE_LABEL,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    NodeMutationFixture,
    NodePatch,
    WarmSpareLiveFixture,
    WarmSpareServiceFixture,
)

DEFAULT_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "single-node-warm-spare-pytorchjob.yaml"
)
CASE_ID = "GF-REGIONAL-DESTR-008"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-003"
CONFIRMATION = "DESTR008_RUN_WARM_SPARE_SHORTAGE_MATRIX"
# kubelet serves the probe's own `kubectl exec` channel, so its stop is handed
# to a systemd timer that fires after the probe has already answered.
SERVICE_STOP_DELAY_SECONDS = {
    "kubernetes-not-ready": 15,
    "agent-unavailable": 0,
}
SERVICE_RESTORE_SECONDS = 420


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    site_file: Path
    manifest: Path
    hyperpod_cluster: str
    fault_node: str
    spare_node: str
    host_probe_image: str
    scenarios: tuple[str, ...]
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_TRAINING_MANIFEST": str(self.manifest),
            "GPU_FAULT_HYPERPOD_CLUSTER_NAME": self.hyperpod_cluster,
            "GPU_FAULT_FAULT_NODE": self.fault_node,
            "GPU_FAULT_SPARE_NODE": self.spare_node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_DESTR008_SCENARIOS": ",".join(self.scenarios),
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
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
    selected = tuple(arguments.scenario or SCENARIOS)
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
        fault_node=required(
            arguments.fault_node or os.getenv("GPU_FAULT_FAULT_NODE", ""),
            "fault node",
        ),
        spare_node=required(
            arguments.spare_node or os.getenv("GPU_FAULT_SPARE_NODE", ""),
            "spare node",
        ),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        scenarios=selected,
        predecessor_path=predecessor,
    )


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/hyperpod/test_hyperpod_spares.py::"
        "test_no_declared_spare_pool_keeps_existing_replace_behavior",
        "tests/hyperpod/test_hyperpod_spares.py::"
        "test_incompatible_spare_does_not_satisfy_target_group",
        "tests/hyperpod/test_hyperpod_spares.py::"
        "test_insufficient_healthy_spares_blocks_and_alerts_admin",
        "tests/hyperpod/test_hyperpod_spares.py::"
        "test_active_gpu_resource_pod_blocks_spare_allocation",
        "tests/hyperpod/test_hyperpod_spares.py::"
        "test_concurrent_reservation_conflict_rolls_back_partial_set",
        "tests/hyperpod/test_hyperpod_spares.py::"
        "test_shortage_reason_names_each_rejected_candidate_gate",
        "tests/execution/test_node_action.py::"
        "test_hyperpod_replace_is_blocked_when_spares_are_insufficient",
    ]
    completed = RegionalLiveFixture.run(
        command,
        cwd=ROOT,
        check=False,
        timeout=300,
    )
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
    *,
    reusable_tests: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not settings.site_file.is_file() or not settings.manifest.is_file():
        raise RegionalFixtureError("site file or training manifest does not exist")
    regional = RegionalLiveFixture(settings.regional)
    warm = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    fault = warm.node_snapshot(settings.fault_node)
    spare = warm.node_snapshot(settings.spare_node)
    state = warm.store_snapshot()
    fault_state = regional.store_snapshot(node=settings.fault_node)
    state["profile"] = fault_state.get("profile")
    state["release_id"] = fault_state.get("release_id")
    state["queue"] = fault_state.get("queue")
    state["remote_commands"] = fault_state.get("remote_commands")
    cluster = warm.cluster_recovery()
    executor_env = warm.executor_environment()
    synthetic_gates = warm.synthetic_replacement_gates()
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **regional.evidence_identity(),
    )
    tests = (
        {**reusable_tests, "reused": True}
        if reusable_tests is not None
        else focused_tests(case_dir)
    )
    errors = []
    if not predecessor["valid"]:
        errors.append("DESTR-003 predecessor evidence is not PASS")
    if set(settings.scenarios) - set(SCENARIOS):
        errors.append("unknown DESTR-008 scenario selected")
    if settings.fault_node == settings.spare_node:
        errors.append("fault and spare nodes are identical")
    if fault["ready"] != "True" or fault["unschedulable"]:
        errors.append("fault node is not Ready and schedulable")
    if fault["labels"].get(SPARE_LABEL) == "true":
        errors.append("fault node is labeled as a spare")
    if any(item.get("key") == QUARANTINE_TAINT for item in fault["taints"]) or any(
        fault["annotations"].get(key)
        for key in (
            "gpu-fault.io/incident-id",
            "gpu-fault.io/fencing-token",
            "gpu-fault.io/previous-unschedulable",
        )
    ):
        errors.append("fault node has pre-existing quarantine ownership")
    if spare["ready"] != "True" or not spare["unschedulable"]:
        errors.append("spare node is not Ready and cordoned")
    if spare["labels"].get(SPARE_LABEL) != "true":
        errors.append("spare node is not declared by the spare label")
    if spare["labels"].get(HYPERPOD_HEALTH_LABEL) != "Schedulable":
        errors.append("spare HyperPod health label is not Schedulable")
    if spare["annotations"].get(SPARE_RESERVATION_ANNOTATION):
        errors.append("spare node already has a reservation")
    if spare["annotations"].get(SPARE_POOL_STATE_ANNOTATION) not in {
        None,
        "AVAILABLE",
    }:
        errors.append("spare pool state is not AVAILABLE")
    if warm.spare_nodes() != [settings.spare_node]:
        errors.append("declared spare set does not exactly match --spare-node")
    if fault["labels"].get(INSTANCE_GROUP_LABEL) != spare["labels"].get(
        INSTANCE_GROUP_LABEL
    ) or instance_type(fault) != instance_type(spare):
        errors.append("fault and spare topology do not match at baseline")
    active_targets = [
        item
        for item in regional.gpu_workloads()
        if item.get("node") in {settings.fault_node, settings.spare_node}
    ]
    if active_targets:
        errors.append("fault or spare node already has a GPU workload")
    for node in (settings.fault_node, settings.spare_node):
        agent = agent_by_node(state, node)
        if agent is None or agent.get("lifecycle_state") != "ACTIVE":
            errors.append(f"{node} does not have exactly one ACTIVE Agent")
    errors.extend(profile_errors(state.get("profile")))
    capabilities: dict[str, Any] = {}
    try:
        capabilities = read_capabilities(regional)
        if set(settings.scenarios) & {*SERVICE_UNIT, "active-gpu-pod"}:
            require_admission_api(regional)
    except Exception as exc:
        # Keep the failing command's own words: a swallowed RegionalCommandFailed
        # left the 2026-09-18 refusal undiagnosable.
        errors.append(
            "independent shortage safeguards unavailable: "
            f"{type(exc).__name__}: {str(exc)[:300]}"
        )
    errors.extend(
        scenario_admission_errors(settings.scenarios, capabilities=capabilities)
    )
    if (state.get("profile") or {}).get("warnings"):
        errors.append("runtime profile has warnings")
    if processor_queue_backlog(state.get("queue")):
        errors.append("processor queue is not empty")
    commands = state.get("remote_commands")
    if not isinstance(commands, dict) or "open_by_cluster" not in commands:
        errors.append("remote command queue state is unknown")
    elif commands["open_by_cluster"]:
        errors.append("remote command queue is not empty")
    if cluster.get("status") != "InService" or cluster.get("node_recovery") != "None":
        errors.append("HyperPod cluster is not InService with NodeRecovery=None")
    if len(executor_env) < 1 or any(
        item.get("spare_failover") != "true"
        or item.get("remote_state") != "true"
        or item.get("allow_replace") != "false"
        for item in executor_env
    ):
        errors.append("executor warm-spare safety environment is inconsistent")
    if len(synthetic_gates) < 1 or any(
        str(item.get("enabled") or "").lower() != "true" for item in synthetic_gates
    ):
        errors.append(
            "synthetic replacement test route is not enabled on every API Pod"
        )
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        "release_id": state.get("release_id"),
        "fault_node": fault,
        "spare_node": spare,
        "cluster": cluster,
        "provider_inventory": warm.provider_inventory(),
        "store": state,
        "executor_environment": executor_env,
        "synthetic_replacement_gates": synthetic_gates,
        "predecessor": predecessor,
        "focused_tests": tests,
        "activation_inhibition": capabilities,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "selected_scenarios": list(settings.scenarios),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def scenario_identity(run_dir: Path, attempt: int, scenario: str) -> tuple[str, str]:
    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{scenario}".encode()
    ).hexdigest()[:10]
    job_id = f"destr008-{suffix}"
    return job_id, f"{job_id}-a001"


def wait_observation(
    warm: WarmSpareLiveFixture,
    *,
    job_id: str,
    attempt_id: str,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        state = warm.store_snapshot(job_id=job_id, attempt_id=attempt_id)
        last = state.get("observations") or []
        if len(last) == 1:
            observation = last[0]
            if (
                observation.get("workload_phase") == "RUNNING"
                and len(observation_gpu_uuids(observation)) == 8
            ):
                return cast(dict[str, Any], observation)
        time.sleep(5)
    raise RegionalFixtureError(f"8-GPU attempt observation did not appear: {last}")


class ScenarioFixture:
    def __init__(
        self,
        settings: Settings,
        warm: WarmSpareLiveFixture,
        *,
        scenario: str,
        run_id: str,
        state_directory: Path,
    ) -> None:
        self.settings = settings
        self.warm = warm
        self.scenario = scenario
        self.run_id = run_id
        self.state_directory = state_directory
        self.node_mutation: NodeMutationFixture | None = None
        self.service: WarmSpareServiceFixture | None = None
        self.holder: GpuHolderFixture | None = None
        self.plan: Plan | None = None
        self.maintenance_expires_at: datetime | None = None
        # When the scenario's shortage stops holding on its own: the service
        # failsafe timer, or the GPU holder's sleep. None for the label and
        # annotation scenarios, which hold until restored.
        self.bound_at: datetime | None = None
        self.bound_label = ""

    def bind_safety(self, plan: Plan, maintenance_expires_at: datetime) -> None:
        if (
            self.plan is not None
            or plan.run_id != self.run_id
            or plan.fault_node != self.settings.fault_node
            or plan.spare_node != self.settings.spare_node
            or plan.cluster_id != self.settings.regional.cluster_id
        ):
            raise RegionalFixtureError("scenario fixture safety binding differs")
        self.plan = plan
        self.maintenance_expires_at = maintenance_expires_at

    def bound_inputs(self) -> dict[str, Any]:
        if self.plan is None or self.maintenance_expires_at is None:
            raise RegionalFixtureError("bounded fixture has no independent safety plan")
        return {
            "state_directory": self.state_directory,
            "node_uid": self.plan.fence.node_uid,
            "plan_sha256": plan_digest(self.plan),
            "release_id": self.plan.release_id,
        }

    def service_fixture(self) -> WarmSpareServiceFixture:
        if self.maintenance_expires_at is None:
            raise RegionalFixtureError("service fixture maintenance expiry is missing")
        return WarmSpareServiceFixture(
            self.warm,
            node=self.settings.spare_node,
            image=self.settings.host_probe_image,
            case_id=CASE_ID,
            run_id=self.run_id,
            maintenance_expires_at=self.maintenance_expires_at,
            **self.bound_inputs(),
        )

    def apply(self) -> dict[str, Any]:
        if self.scenario == "no-spare":
            self.node_mutation = NodeMutationFixture(
                self.warm,
                self.settings.spare_node,
                label_keys=(SPARE_LABEL,),
            )
            self.node_mutation.apply(
                NodePatch(labels={SPARE_LABEL: None}, annotations={})
            )
            return {"removed_spare_label": True}
        if self.scenario == "topology-mismatch":
            self.node_mutation = NodeMutationFixture(
                self.warm,
                self.settings.spare_node,
                label_keys=(INSTANCE_GROUP_LABEL,),
            )
            mismatch = f"acceptance-mismatch-{self.run_id[-12:]}"
            self.node_mutation.apply(
                NodePatch(
                    labels={INSTANCE_GROUP_LABEL: mismatch},
                    annotations={},
                )
            )
            return {"instance_group": mismatch}
        if self.scenario == "reserved-by-other":
            self.node_mutation = NodeMutationFixture(
                self.warm,
                self.settings.spare_node,
                annotation_keys=(SPARE_RESERVATION_ANNOTATION,),
            )
            other = f"incident-other-{self.run_id[-12:]}"
            self.node_mutation.apply(
                NodePatch(
                    labels={},
                    annotations={SPARE_RESERVATION_ANNOTATION: other},
                )
            )
            return {"reservation": other}
        if self.scenario == "active-gpu-pod":
            # Named now, created in `apply_late`: the holder's sleep is the
            # bound on this shortage, and spending it while the workload is
            # still being scheduled let REPLACE_NODE find a free spare.
            self.holder = GpuHolderFixture(
                self.warm,
                node=self.settings.spare_node,
                run_id=self.run_id,
                **self.bound_inputs(),
            )
            return {"holder_pod": self.holder.name, "armed": "late"}
        if self.scenario in SERVICE_UNIT:
            self.service = self.service_fixture()
            self.service.create()
            return {"service": SERVICE_UNIT[self.scenario], "host_probe": "created"}
        raise ValueError(f"unknown scenario: {self.scenario}")

    def apply_late(self) -> dict[str, Any]:
        """Arm the bounded shortages once the workload is already running.

        Label and annotation mutations hold indefinitely, so `apply()` makes
        them before the workload starts. A stopped systemd unit and a GPU
        holder Pod do not: the unit is bounded by the failsafe timer that
        restores it, the holder by its own sleep. Submitting the workload and
        waiting for its observation first can take minutes, which would burn
        most of that window before the workflow ever reaches `REPLACE_NODE`
        -- and a shortage that lapses mid-workflow lets REPLACE_NODE succeed,
        which is a real failover. So both happen here, immediately before the
        replacement signal, and `bound_at` records when each stops holding.
        """

        if self.holder is not None:
            self.holder.create()
            self.bound_at = self.holder.deadline_at
            self.bound_label = "GPU holder Pod"
            return {
                "holder_pod": self.holder.name,
                "holder_deadline_at": (
                    self.bound_at.isoformat() if self.bound_at else None
                ),
            }
        if self.service is None:
            return {}
        unit = SERVICE_UNIT[self.scenario]
        stopped = self.service.stop(
            unit,
            restore_seconds=SERVICE_RESTORE_SECONDS,
            delay_seconds=SERVICE_STOP_DELAY_SECONDS[self.scenario],
        )
        self.bound_at = self.service.failsafe_at
        self.bound_label = f"{unit} failsafe"
        if self.scenario == "kubernetes-not-ready":
            self.warm.wait_node_ready(
                self.settings.spare_node,
                ready=False,
                timeout_seconds=180,
            )
        else:
            self.warm.wait_fleet_readiness(
                self.settings.spare_node,
                ready=False,
                timeout_seconds=180,
            )
        return {
            "service": unit,
            "stopped": stopped,
            "failsafe_at": self.bound_at.isoformat() if self.bound_at else None,
        }

    def restore(self) -> dict[str, Any]:
        errors = []
        result: dict[str, Any] = {}
        if self.holder is not None:
            try:
                residual = self.holder.cleanup()
                result["holder_residual"] = residual
                if residual:
                    errors.append("GPU holder Pod remains")
            except Exception as exc:
                errors.append(f"holder cleanup: {type(exc).__name__}: {exc}")
        if self.service is not None:
            try:
                if self.scenario == "kubernetes-not-ready":
                    # Nothing can be executed on the node until kubelet is
                    # back, and only the failsafe timer can bring it back, so
                    # this wait has to outlast the failsafe by a real margin.
                    self.warm.wait_node_ready(
                        self.settings.spare_node,
                        ready=True,
                        timeout_seconds=SERVICE_RESTORE_SECONDS + 180,
                    )
                result["service_restore"] = self.service.restore()
                if self.scenario == "agent-unavailable":
                    self.warm.wait_fleet_readiness(
                        self.settings.spare_node,
                        ready=True,
                        timeout_seconds=180,
                    )
            except Exception as exc:
                errors.append(f"service restore: {type(exc).__name__}: {exc}")
            try:
                residuals = self.service.cleanup()
                result["probe_residuals"] = residuals
                if any(residuals.values()):
                    errors.append("service probe resources remain")
            except Exception as exc:
                errors.append(f"probe cleanup: {type(exc).__name__}: {exc}")
        if self.node_mutation is not None:
            try:
                result["node_restore"] = self.node_mutation.restore()
            except Exception as exc:
                errors.append(f"node restore: {type(exc).__name__}: {exc}")
        result["errors"] = errors
        return result

    def resume_cleanup(self) -> dict[str, Any]:
        if self.scenario == "active-gpu-pod":
            self.holder = GpuHolderFixture(
                self.warm,
                node=self.settings.spare_node,
                run_id=self.run_id,
                **self.bound_inputs(),
            )
            residual = self.holder.cleanup()
            return {"errors": ["GPU holder remains"] if residual else []}
        if self.scenario in SERVICE_UNIT:
            self.service = self.service_fixture()
            if self.scenario == "kubernetes-not-ready":
                self.warm.wait_node_ready(
                    self.settings.spare_node,
                    ready=True,
                    timeout_seconds=SERVICE_RESTORE_SECONDS + 180,
                )
            report = self.service.resume_cleanup()
            self.warm.wait_fleet_readiness(
                self.settings.spare_node, ready=True, timeout_seconds=180
            )
            return {"errors": [], "service": report}
        raise RegionalFixtureError(
            "interrupted metadata shortage requires its original mutation owner"
        )

    def close(self) -> None:
        if self.service is not None:
            self.service.close()


def recover_incident_id(warm: WarmSpareLiveFixture, event_id: str) -> str:
    """The incident the injected event opened, when the scenario lost track.

    ``incident_id`` is only learned from the terminal store snapshot; a wait
    that timed out never produced one, and a cleanup keyed on an empty id
    released no spare and restored nothing on a node it had just quarantined.
    """

    if not event_id:
        return ""
    state = warm.store_snapshot(event_id=event_id)
    return str((state.get("incident") or {}).get("incident_id") or "")


def replacement_payload(
    settings: Settings,
    *,
    event_id: str,
    job_id: str,
    attempt_id: str,
    observation: dict[str, Any],
    observed_at: datetime,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "cluster_id": settings.regional.cluster_id,
        "node_id": settings.fault_node,
        "observed_at": observed_at.isoformat(),
        "runtime_profile_version": observation["runtime_profile_version"],
        "job_id": job_id,
        "attempt_id": attempt_id,
        "affected_workload_ids": observation["workload_ids"],
        "gpu_uuids": observation_gpu_uuids(observation),
        "reason": "synthetic warm-spare shortage acceptance trigger",
        "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY",
        "activation_forbidden": True,
        "synthetic": True,
    }


@dataclass(frozen=True)
class ScenarioResources:
    directory: Path
    job_id: str
    attempt_id: str
    workload: ManagedWorkloadFixture
    fixture: ScenarioFixture


def prepare_scenario_resources(
    settings: Settings,
    warm: WarmSpareLiveFixture,
    regional: RegionalLiveFixture,
    directory: Path,
    run_dir: Path,
    attempt: int,
    scenario: str,
) -> ScenarioResources:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    job_id, attempt_id = scenario_identity(run_dir, attempt, scenario)
    pinned = render_node_pinned_manifest(
        settings.manifest, directory / "pinned-workload.yaml", node=settings.fault_node
    )
    workload = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=pinned,
            site_file=settings.site_file,
            job_id=job_id,
            attempt_id=attempt_id,
            restart_budget=1,
            expected_pods=1,
            expected_gpu_count=8,
        ),
        state_path=directory / "workload-owner.json",
    )
    fixture = ScenarioFixture(
        settings,
        warm,
        scenario=scenario,
        run_id=job_id,
        state_directory=directory / "host-probes",
    )
    return ScenarioResources(directory, job_id, attempt_id, workload, fixture)


def cleanup_scenario(
    settings: Settings,
    warm: WarmSpareLiveFixture,
    resources: ScenarioResources,
    safety: ShortageSafety | None,
    result: dict[str, Any],
    *,
    incident_id: str,
    event_id: str,
    profile_version: str,
) -> None:
    cleanup_safe = True
    if safety is not None:
        proof = safety.finish()
        result["independent_safety_cleanup"] = proof
        if proof["errors"]:
            result["verdict"] = "FAIL"
        if proof["quiescent"] is not True:
            cleanup_safe = False
            result["cleanup_deferred"] = True
    if not incident_id:
        try:
            incident_id = recover_incident_id(warm, event_id)
            result["incident_id_recovered_from_event"] = bool(incident_id)
        except Exception as exc:
            result["incident_lookup_error"] = f"{type(exc).__name__}: {exc}"
    if event_id:
        try:
            if not incident_id:
                raise RegionalFixtureError(
                    "replacement outcome has no incident identity"
                )
            warm.wait_incident_idle(incident_id)
        except Exception as exc:
            cleanup_safe = False
            result["cleanup_quiescence_error"] = f"{type(exc).__name__}: {exc}"
            result["cleanup_deferred"] = True
            result["verdict"] = "FAIL"
    if cleanup_safe:
        try:
            resources.workload.delete()
        except Exception as exc:
            result["workload_cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    scenario_cleanup = (
        resources.fixture.restore()
        if cleanup_safe
        else {"errors": ["shortage restore deferred while commands may still act"]}
    )
    result["scenario_cleanup"] = scenario_cleanup
    if scenario_cleanup["errors"]:
        result["verdict"] = "FAIL"
    fault_cleanup = (
        restore_fault_node(
            warm,
            settings=settings,
            incident_id=incident_id,
            profile_version=profile_version,
        )
        if cleanup_safe
        else {"errors": ["fault restore deferred while commands may still act"]}
    )
    result["fault_cleanup"] = fault_cleanup
    if fault_cleanup["errors"]:
        result["verdict"] = "FAIL"
    audit_scenario_nodes(warm, settings, result)
    result["cleanup_complete"] = bool(
        cleanup_safe
        and not scenario_cleanup["errors"]
        and not fault_cleanup["errors"]
        and "workload_cleanup_error" not in result
        and not result.get("postflight_errors")
        and "postflight_error" not in result
        and (
            safety is None
            or result["independent_safety_cleanup"].get("retired") is True
        )
    )
    try:
        resources.fixture.close()
    except Exception as exc:
        result["fixture_close_error"] = type(exc).__name__
        result["cleanup_complete"] = False
        result["verdict"] = "FAIL"


def run_scenario(
    settings: Settings,
    *,
    warm: WarmSpareLiveFixture,
    regional: RegionalLiveFixture,
    case_dir: Path,
    run_dir: Path,
    attempt: int,
    scenario: str,
    maintenance_window_end: datetime,
    provider_baseline: dict[str, Any],
    profile_version: str,
    capabilities: dict[str, Any] | None = None,
    release_id: str = "",
    spare_uid: str = "",
    journal: ExecutionJournal | None = None,
) -> dict[str, Any]:
    directory = case_dir / "scenarios" / scenario
    with controller_ownership(directory / "scenario-owner.json"):
        resources = prepare_scenario_resources(
            settings, warm, regional, directory, run_dir, attempt, scenario
        )
        return _run_scenario(
            settings,
            warm=warm,
            regional=regional,
            resources=resources,
            scenario=scenario,
            maintenance_window_end=maintenance_window_end,
            provider_baseline=provider_baseline,
            profile_version=profile_version,
            capabilities=capabilities,
            release_id=release_id,
            spare_uid=spare_uid,
            journal=journal,
        )


def require_maintenance_window(expires_at: datetime, *, before: str) -> None:
    if datetime.now(timezone.utc) >= expires_at:
        raise RegionalFixtureError(f"maintenance window ended before {before}")


def _run_scenario(
    settings: Settings,
    *,
    warm: WarmSpareLiveFixture,
    regional: RegionalLiveFixture,
    resources: ScenarioResources,
    scenario: str,
    maintenance_window_end: datetime,
    provider_baseline: dict[str, Any],
    profile_version: str,
    capabilities: dict[str, Any] | None,
    release_id: str,
    spare_uid: str,
    journal: ExecutionJournal | None,
) -> dict[str, Any]:
    scenario_dir, job_id, attempt_id = (
        resources.directory,
        resources.job_id,
        resources.attempt_id,
    )
    workload, fixture = resources.workload, resources.fixture
    result: dict[str, Any] = {
        "scenario": scenario,
        "verdict": "FAIL",
        "synthetic_trigger": True,
    }
    incident_id = ""
    event_id = ""
    planned_event_id = f"destr008-{scenario}-{job_id}-event"
    bounded = scenario in {*SERVICE_UNIT, "active-gpu-pod"}
    safety: ShortageSafety | None = None
    try:
        admission = scenario_admission_errors([scenario], capabilities=capabilities)
        if admission:
            raise RegionalFixtureError("; ".join(admission))
        require_maintenance_window(maintenance_window_end, before="shortage setup")
        if not bounded and journal is not None:
            journal.stage(scenario, "fixture_started")
        mutation = {} if bounded else fixture.apply()
        if mutation:
            write_json_atomic(scenario_dir / "scenario-mutation.json", mutation)
        require_maintenance_window(maintenance_window_end, before="workload submission")
        if journal is not None:
            journal.stage(scenario, "workload_started")
        submission = workload.submit()
        write_json_atomic(scenario_dir / "submission.json", submission)
        workload.wait_running(timeout_seconds=900)
        observation = wait_observation(
            warm,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        if observation.get("runtime_profile_version") != profile_version:
            raise RegionalFixtureError(
                "workload observation profile differs from the approved preflight"
            )
        if bounded:
            if not release_id or not spare_uid:
                raise RegionalFixtureError(
                    "bounded shortage has no release or Node UID binding"
                )
            if journal is not None:
                journal.stage(scenario, "safety_started")
            safety = ShortageSafety(
                regional,
                run_id=job_id,
                attempt_id=attempt_id,
                event_id=planned_event_id,
                fault_node=settings.fault_node,
                spare_node=settings.spare_node,
                spare_uid=spare_uid,
                release_id=release_id,
                directory=scenario_dir / "safety",
            )
            window = (
                GpuHolderFixture.HOLD_SECONDS
                if scenario == "active-gpu-pod"
                else SERVICE_RESTORE_SECONDS
            ) - BOUND_MARGIN_SECONDS
            safety.arm(
                observation,
                window_seconds=window,
                maintenance_window_end=maintenance_window_end,
            )
            safety.admit_fixture()
            if safety.plan is None:
                raise RegionalFixtureError("independent cancellation plan is missing")
            fixture.bind_safety(safety.plan, maintenance_window_end)
            if journal is not None:
                journal.stage(scenario, "fixture_started")
            mutation = fixture.apply()
            write_json_atomic(scenario_dir / "scenario-mutation.json", mutation)
        require_maintenance_window(
            maintenance_window_end, before=f"scenario {scenario}"
        )
        late = fixture.apply_late()
        if late:
            mutation = {**mutation, "late": late}
            write_json_atomic(scenario_dir / "scenario-mutation.json", mutation)
        if safety is not None:
            safety.require_bound(fixture.bound_at, margin=BOUND_MARGIN_SECONDS)
        require_maintenance_window(maintenance_window_end, before="replacement trigger")
        if journal is not None:
            journal.stage(scenario, "post_started")
        claim_id = safety.before_post() if safety is not None else ""
        event_id = planned_event_id
        started_at = datetime.now(timezone.utc)
        injection = warm.post_synthetic_replacement(
            replacement_payload(
                settings,
                event_id=event_id,
                job_id=job_id,
                attempt_id=attempt_id,
                observation=observation,
                observed_at=started_at,
            )
        )
        write_json_atomic(scenario_dir / "injection.json", injection)
        if injection.get("status") != 200:
            raise RegionalFixtureError(
                f"synthetic replacement endpoint failed: {injection}"
            )
        if safety is not None:
            safety.acknowledge(claim_id, injection)
            incident_id = str(injection["body"]["incident_ids"][0])
        wait_seconds = wait_timeout_seconds(
            fixture.bound_at, datetime.now(timezone.utc)
        )
        if safety is not None:
            wait_seconds = min(wait_seconds, safety.remaining_seconds())
        state = warm.wait_for_workflow(
            event_id=event_id,
            job_id=job_id,
            attempt_id=attempt_id,
            case_dir=scenario_dir,
            timeout_seconds=wait_seconds,
        )
        incident_id = str((state.get("incident") or {}).get("incident_id") or "")
        state["fault_node"] = warm.node_snapshot(settings.fault_node)
        state["spare_node"] = warm.node_snapshot(settings.spare_node)
        write_json_atomic(scenario_dir / "workflow-state.json", state)
        errors = scenario_errors(state, settings, scenario, event_id=event_id)
        errors.extend(
            bound_errors(
                terminal_execution(state.get("workflow") or {}, "REPLACE_NODE"),
                bound_at=fixture.bound_at,
                label=fixture.bound_label,
            )
        )
        provider_after = warm.provider_inventory()
        if provider_after != provider_baseline:
            errors.append("HyperPod provider inventory changed")
        ended_at = datetime.now(timezone.utc)
        provider_events = regional.provider_events(started_at, ended_at)
        if any(
            item["event_name"] in PROVIDER_REPLACE_EVENTS for item in provider_events
        ):
            errors.append("provider replace/delete mutation appeared")
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "event_id": event_id,
                "incident_id": incident_id,
                "workflow_request_id": (
                    (state.get("workflow") or {}).get("request_id")
                ),
                "shortage_bound_at": (
                    fixture.bound_at.isoformat() if fixture.bound_at else None
                ),
                "provider_events": provider_events,
                # The negative claim cannot be proven inside CloudTrail's lag;
                # DESTR-013 re-reads the whole run's window later.
                "provider_events_provisional": regional.provider_events_provisional(
                    ended_at
                ),
                "notifications": state.get("notifications") or [],
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_scenario(
            settings,
            warm,
            resources,
            safety,
            result,
            incident_id=incident_id,
            event_id=event_id,
            profile_version=profile_version,
        )
    write_json_atomic(scenario_dir / f"{scenario}.json", result)
    return result


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    details: dict[str, Any] = {
        "risk": "destructive-warm-spare",
        "predecessor": preflight["predecessor"],
        "fault_node": settings.fault_node,
        "spare_node": settings.spare_node,
        "scenarios": list(settings.scenarios),
        "complete_matrix": set(settings.scenarios) == set(SCENARIOS),
        "synthetic_trigger": True,
        "mutation": (
            "run each selected warm-spare shortage scenario with a unique "
            "node-pinned workload and real failed REPLACE_NODE workflow; "
            "S3 stops kubelet and S6 stops Node Agent once the workload is "
            "already Running and only after arming a host-side automatic "
            "restore timer, and kubelet's stop is handed to a systemd timer "
            "because the host probe answers over kubelet's own exec channel"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "fault_node_uid": preflight["fault_node"]["uid"],
            "spare_node_uid": preflight["spare_node"]["uid"],
            "runtime_profile_version": (state.get("profile") or {}).get(
                "profile_version"
            ),
            "provider_inventory_sha256": preflight["provider_inventory"]["sha256"],
        },
        "stop_conditions": [
            "DESTR-003 predecessor evidence is not PASS",
            "fault/spare baseline or release drift",
            "scenario mutation cannot arm or verify rollback",
            "STOP_WORKLOADS does not complete before REPLACE_NODE failure",
            "failure reason or notification count differs from the scenario",
            "a marker other than the injected finding's own, a provider "
            "replace/delete, or a spare allocation remains",
            "fault-node validation/restore cleanup fails",
        ],
        "rollback": {
            "restore_each_scenario_before_the_next": True,
            "kubelet_and_agent_have_host_side_failsafe_restore": True,
            "delete_each_unique_test_workload": True,
            "release_only_incident_owned_spare_state": True,
            "restore_fault_node_via_validation_first_workflow": True,
            "revoke_producer_and_wait_for_complete_command_quiescence": True,
            "keep_independent_inhibition_when_cleanup_is_unproven": True,
        },
        "independent_safety": {
            "activation_forbidden_on_every_replacement": True,
            "old_executor_claims_are_refused": True,
            "bounded_scenarios_require_cpu_watchdog_and_admission_fence": True,
            "guard_refusal_is_not_a_shortage_pass": True,
        },
        "preflight": preflight,
    }
    record_focused_tests(details, preflight.get("focused_tests") or {})
    return details


def verify_plan_identity(
    case_dir: Path,
    preflight: dict[str, Any],
) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = {
        "release_id": preflight["release_id"],
        "fault_node_uid": preflight["fault_node"]["uid"],
        "spare_node_uid": preflight["spare_node"]["uid"],
        "runtime_profile_version": (preflight["store"].get("profile") or {}).get(
            "profile_version"
        ),
        "provider_inventory_sha256": preflight["provider_inventory"]["sha256"],
    }
    if current != planned:
        raise RegionalFixtureError(f"DESTR-008 plan drifted: {planned} != {current}")


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    directory = settings.site_file.parent.resolve()
    information = directory.stat()
    if (
        directory == Path("/")
        or not stat.S_ISDIR(information.st_mode)
        or information.st_uid != os.getuid()
        or information.st_mode & 0o077
    ):
        raise RegionalFixtureError(
            "DESTR-008 requires the canonical private administrator site directory"
        )
    with site_operation_lock(directory, wait=False):
        return _execute_case(settings, run_dir, attempt, maintenance_window_end)


def execution_binding(
    settings: Settings, regional: RegionalLiveFixture, run_dir: Path
) -> dict[str, Any]:
    return fixture_binding(
        regional,
        purpose=CASE_ID,
        inputs={
            "run_dir": str(run_dir.resolve()),
            "site_sha256": file_digest(settings.site_file),
            "manifest_sha256": file_digest(settings.manifest),
            "runner_sha256": file_digest(Path(__file__)),
            "contract_sha256": file_digest(
                Path(__file__).with_name("destr008_contract.py")
            ),
            "hyperpod_cluster": settings.hyperpod_cluster,
            "fault_node": settings.fault_node,
            "spare_node": settings.spare_node,
            "host_probe_image": settings.host_probe_image,
            "scenarios": list(settings.scenarios),
        },
    )


def _execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    regional = RegionalLiveFixture(settings.regional)
    warm = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    journal_path = case_dir / "execution-owner.json"

    def binding() -> dict[str, Any]:
        return execution_binding(settings, regional, run_dir)

    # A finished execution that owns nothing can neither be resumed for cleanup
    # nor grant re-execution, so it is archived and this attempt starts fresh.
    # An unfinished one still resumes as cleanup only, under its bound identity.
    retired = retire_completed_execution(case_dir, journal_path, binding=binding)
    if journal_path.exists() or journal_path.is_symlink():
        from scripts.e2e.regional.destr008_resume import resume_execution

        journal = ExecutionJournal(journal_path, binding)
        cleanup_report = resume_execution(
            settings,
            regional=regional,
            warm=warm,
            journal=journal,
            case_dir=case_dir,
            run_dir=run_dir,
        )
        write_json_atomic(case_dir / f"cleanup-attempt-{attempt}.json", cleanup_report)
        print(json.dumps(cleanup_report, sort_keys=True))
        return 1
    if (case_dir / "scenarios").exists() or (case_dir / "prewarm-owner.json").exists():
        raise RegionalFixtureError(
            "prior DESTR-008 artifacts have no execution owner; reconciliation required"
        )
    preflight = read_only_preflight(
        settings,
        case_dir,
        reusable_tests=reusable_focused_tests(case_dir / "plan.json"),
    )
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)
    profile_version = str(
        (preflight["store"].get("profile") or {}).get("profile_version") or ""
    )
    journal = ExecutionJournal(
        journal_path,
        binding,
        initial=ExecutionRecord(
            schema_version=1,
            binding=binding(),
            attempt=attempt,
            maintenance_expires_at=int(maintenance_window_end.timestamp()),
            release_id=preflight["release_id"],
            profile_version=profile_version,
            fault_uid=preflight["fault_node"]["uid"],
            spare_uid=preflight["spare_node"]["uid"],
            scenarios={scenario: ScenarioState() for scenario in settings.scenarios},
        ),
    )
    prewarm = ImagePrewarmFixture(
        regional,
        case_id=CASE_ID,
        run_id=scenario_identity(run_dir, attempt, "prewarm")[0],
        state_path=case_dir / "prewarm-owner.json",
    )
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "synthetic_trigger": True,
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "selected_scenarios": list(settings.scenarios),
        **regional.evidence_identity(),
        "focused_tests_reused": bool(
            (preflight.get("focused_tests") or {}).get("reused")
        ),
        **({"retired_execution": retired} if retired is not None else {}),
    }
    scenario_results = []
    try:
        require_maintenance_window(maintenance_window_end, before="image prewarm")
        journal.start_prewarm()
        prewarm.create([settings.fault_node, settings.spare_node])
        for scenario in settings.scenarios:
            journal.start_scenario(scenario)
            current = run_scenario(
                settings,
                warm=warm,
                regional=regional,
                case_dir=case_dir,
                run_dir=run_dir,
                attempt=attempt,
                scenario=scenario,
                maintenance_window_end=maintenance_window_end,
                provider_baseline=preflight["provider_inventory"],
                profile_version=profile_version,
                capabilities=preflight["activation_inhibition"],
                release_id=preflight["release_id"],
                spare_uid=preflight["spare_node"]["uid"],
                journal=journal,
            )
            if current.get("cleanup_complete") is True:
                journal.cleaned_scenario(scenario)
            else:
                current = {
                    **current,
                    "verdict": "FAIL",
                    "cleanup_contract_error": "scenario resource closure is unproven",
                }
            scenario_results.append(current)
            if current["verdict"] != "PASS":
                break
        complete = set(settings.scenarios) == set(SCENARIOS)
        errors = []
        if not complete:
            errors.append("selected scenarios do not cover the complete matrix")
        if len(scenario_results) != len(settings.scenarios):
            errors.append("scenario matrix stopped before all selected scenarios")
        if any(item["verdict"] != "PASS" for item in scenario_results):
            errors.append("one or more shortage scenarios failed")
        cpu_after = regional.cpu_blast_snapshot()
        if cpu_after != preflight["cpu_blast"]:
            errors.append("control-plane EKS state differs from baseline")
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "scenarios": scenario_results,
                "complete_matrix": complete,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            residuals = prewarm.cleanup()
        except Exception as exc:
            residuals = {"cleanup_error": True}
            result["prewarm_cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
        result["prewarm_residuals"] = residuals
        if any(residuals.values()):
            result["verdict"] = "FAIL"
        else:
            journal.cleaned_prewarm()
            if not any(
                item.state == "STARTED" for item in journal.record.scenarios.values()
            ):
                journal.complete()
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the guarded DESTR-008 warm-spare shortage matrix."
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
    value.add_argument("--fault-node", default="")
    value.add_argument("--spare-node", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--scenario", action="append", choices=SCENARIOS, default=[])
    value.add_argument("--predecessor-evidence", default="")
    return value


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
