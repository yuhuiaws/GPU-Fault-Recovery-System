#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
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
    install_abort_signals,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    HYPERPOD_HEALTH_LABEL,
    INSTANCE_GROUP_LABEL,
    INSTANCE_TYPE_LABELS,
    PROVIDER_REPLACE_EVENTS,
    QUARANTINE_TAINT,
    SPARE_LABEL,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    GpuHolderFixture,
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
SCENARIOS = (
    "no-spare",
    "topology-mismatch",
    "kubernetes-not-ready",
    "reserved-by-other",
    "active-gpu-pod",
    "agent-unavailable",
)
EXPECTED_REASON = {
    "no-spare": (
        "warm-spare replacement is required; "
        "provider node replacement API fallback is disabled"
    ),
    "topology-mismatch": "insufficient healthy HyperPod spares",
    "kubernetes-not-ready": "Kubernetes node is not Ready",
    "reserved-by-other": "reserved by incident",
    "active-gpu-pod": "active GPU resource pods exist",
    "agent-unavailable": "node agent is not fleet-ready",
}
ALERT_SCENARIOS = set(SCENARIOS) - {"no-spare"}
SERVICE_UNIT = {
    "kubernetes-not-ready": "kubelet.service",
    "agent-unavailable": "gpu-fault-node-agent.service",
}
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


def instance_type(snapshot: dict[str, Any]) -> str | None:
    return next(
        (
            snapshot["labels"].get(key)
            for key in INSTANCE_TYPE_LABELS
            if snapshot["labels"].get(key)
        ),
        None,
    )


def agent_by_node(state: dict[str, Any], node: str) -> dict[str, Any] | None:
    matches = [
        item for item in state.get("agents") or [] if item.get("node_id") == node
    ]
    return cast(dict[str, Any], matches[0]) if len(matches) == 1 else None


def capability(profile: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    for item in (profile or {}).get("capabilities", []):
        if isinstance(item, dict) and item.get("capability") == name:
            return cast(dict[str, Any], item)
    return None


def profile_errors(profile: dict[str, Any] | None) -> list[str]:
    expected = {
        "nodeReplace": ("gpu-fault-hyperpod-adapter", "regional-cluster-executor"),
        "workloadStop": (
            "gpu-fault-kubernetes-adapter",
            "regional-cluster-executor",
        ),
        "workloadRestart": (
            "gpu-fault-kubernetes-adapter",
            "regional-cluster-executor",
        ),
    }
    errors = []
    for name, (owner, adapter) in expected.items():
        item = capability(profile, name)
        if item is None:
            errors.append(f"runtime profile has no {name} capability")
        elif (
            item.get("mode") != "OWN"
            or item.get("owner") != owner
            or item.get("adapter") != adapter
        ):
            errors.append(f"{name} has the wrong owner or adapter")
    return errors


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
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
    cluster = warm.cluster_recovery()
    executor_env = warm.executor_environment()
    synthetic_gates = warm.synthetic_replacement_gates()
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
    )
    tests = focused_tests(case_dir)
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
    if (state.get("profile") or {}).get("warnings"):
        errors.append("runtime profile has warnings")
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


def observation_gpu_uuids(observation: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(gpu_uuid)
            for container in observation.get("containers", [])
            for gpu_uuid in container.get("gpu_uuids", [])
        }
    )


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


def terminal_execution(
    workflow: dict[str, Any],
    operation: str,
) -> dict[str, Any] | None:
    matches = [
        item
        for item in workflow.get("step_executions", [])
        if item.get("operation") == operation
        and item.get("status") in {"SUCCEEDED", "FAILED"}
    ]
    return cast(dict[str, Any], matches[-1]) if matches else None


class ScenarioFixture:
    def __init__(
        self,
        settings: Settings,
        warm: WarmSpareLiveFixture,
        *,
        scenario: str,
        run_id: str,
    ) -> None:
        self.settings = settings
        self.warm = warm
        self.scenario = scenario
        self.run_id = run_id
        self.node_mutation: NodeMutationFixture | None = None
        self.service: WarmSpareServiceFixture | None = None
        self.holder: GpuHolderFixture | None = None

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
            self.holder = GpuHolderFixture(
                self.warm,
                node=self.settings.spare_node,
                run_id=self.run_id,
            )
            self.holder.create()
            return {"holder_pod": self.holder.name}
        if self.scenario in SERVICE_UNIT:
            self.service = WarmSpareServiceFixture(
                self.warm,
                node=self.settings.spare_node,
                image=self.settings.host_probe_image,
                case_id=CASE_ID,
                run_id=self.run_id,
            )
            self.service.create()
            return {"service": SERVICE_UNIT[self.scenario], "host_probe": "created"}
        raise ValueError(f"unknown scenario: {self.scenario}")

    def apply_late(self) -> dict[str, Any]:
        """Take the spare's service down once the workload is already running.

        Label, annotation and Pod mutations hold indefinitely, so `apply()`
        makes them before the workload starts. A stopped systemd unit does
        not: it is bounded by the failsafe timer that restores it. Submitting
        the workload and waiting for its observation first can take minutes,
        which would burn most of that window before the workflow ever reaches
        `REPLACE_NODE` -- so the stop happens here, immediately before the
        replacement signal, and the window only has to cover the workflow.
        """

        if self.service is None:
            return {}
        unit = SERVICE_UNIT[self.scenario]
        stopped = self.service.stop(
            unit,
            restore_seconds=SERVICE_RESTORE_SECONDS,
            delay_seconds=SERVICE_STOP_DELAY_SECONDS[self.scenario],
        )
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
        return {"service": unit, "stopped": stopped}

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
        "synthetic": True,
    }


def scenario_errors(
    state: dict[str, Any],
    settings: Settings,
    scenario: str,
    *,
    event_id: str,
) -> list[str]:
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    errors = []
    if workflow.get("status") != "FAILED":
        errors.append("shortage workflow is not FAILED")
    stop = terminal_execution(workflow, "STOP_WORKLOADS")
    if stop is None or stop.get("status") != "SUCCEEDED":
        errors.append("STOP_WORKLOADS did not execute successfully")
    replace = terminal_execution(workflow, "REPLACE_NODE")
    if replace is None or replace.get("status") != "FAILED":
        errors.append("REPLACE_NODE did not execute and fail")
    elif EXPECTED_REASON[scenario] not in str(replace.get("error") or ""):
        errors.append("REPLACE_NODE failure reason does not match the scenario")
    replace_step = next(
        (
            item
            for item in workflow.get("official_steps", [])
            if item.get("operation") == "REPLACE_NODE"
        ),
        None,
    )
    if (replace_step or {}).get("parameters") != {
        "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"
    }:
        errors.append("REPLACE_NODE lacks HEALTHY_WARM_SPARE_ONLY")
    alerts = [
        item
        for item in state.get("notifications") or []
        if "hyperpod-spare-insufficient" in str(item.get("deduplication_key") or "")
    ]
    expected_alerts = 1 if scenario in ALERT_SCENARIOS else 0
    if len(alerts) != expected_alerts:
        errors.append(
            f"expected {expected_alerts} spare-insufficient alerts, got {len(alerts)}"
        )
    # The injected finding's own marker belongs to this incident, and has to:
    # a marker pointing at an incident nobody persisted lets a terminal attempt
    # inside the marker window open a second recovery for the same fault, which
    # is what re-quarantined the fault node minutes into the next scenario. What
    # the shortage path must not do is add any *other* marker -- handing the node
    # to a different replacement mechanism once warm-spare replacement failed.
    marker_ids = sorted(
        str(item.get("marker_id") or "") for item in state.get("markers") or []
    )
    if marker_ids != [f"marker-{event_id}"]:
        errors.append(
            "incident markers are not exactly the injected finding's own marker: "
            f"{marker_ids}"
        )
    fault = state.get("fault_node") or {}
    if not fault.get("unschedulable") or not any(
        item.get("key") == QUARANTINE_TAINT for item in fault.get("taints") or []
    ):
        errors.append("fault node is not kept quarantined")
    incident_id = str(incident.get("incident_id") or "")
    spare = state.get("spare_node") or {}
    if spare.get("annotations", {}).get(SPARE_RESERVATION_ANNOTATION) == incident_id:
        errors.append("failed scenario left an incident spare reservation")
    if spare.get("annotations", {}).get(SPARE_POOL_STATE_ANNOTATION) == "ALLOCATED":
        errors.append("failed scenario left the spare ALLOCATED")
    return errors


def restore_fault_node(
    warm: WarmSpareLiveFixture,
    *,
    settings: Settings,
    incident_id: str,
    profile_version: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}
    if not incident_id:
        return result
    try:
        warm.release_spares([settings.spare_node], incident_id)
    except Exception as exc:
        result["errors"].append(f"spare rollback: {type(exc).__name__}: {exc}")
    try:
        warm.reactivate_agent(settings.fault_node)
        warm.wait_agent_active(settings.fault_node)
    except Exception as exc:
        result["errors"].append(f"agent cleanup: {type(exc).__name__}: {exc}")
    try:
        fault = warm.node_snapshot(settings.fault_node)
        owner = str(fault["annotations"].get("gpu-fault.io/incident-id") or "")
        result["quarantine_owner"] = owner or None
        if owner and owner != incident_id:
            # A shortage that blocks recovery is escalated by the product
            # itself: the escalation engine opens a successor support incident
            # over the same node, re-quarantines it and files a ticket, so the
            # node ends up owned by that incident rather than by ours. Cleanup
            # that recognised only its own incident restored nothing and
            # recorded no error at all, and the case then failed in postflight
            # for a condition the cleanup had already seen and skipped.
            result["successor_incident"] = owner
        if owner:
            warm.wait_incident_idle(owner)
            created = warm.create_restore_workflow(
                incident_id=owner,
                node=settings.fault_node,
                profile_version=profile_version,
                reason="DESTR-008 scenario cleanup",
            )
            restored = warm.wait_workflow_id(str(created["workflow_request_id"]))
            result["restore_workflow"] = restored
            if restored.get("status") != "SUCCEEDED":
                result["errors"].append("fault-node restore workflow failed")
    except Exception as exc:
        result["errors"].append(f"fault restore: {type(exc).__name__}: {exc}")
    return result


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
) -> dict[str, Any]:
    scenario_dir = case_dir / "scenarios" / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    job_id, attempt_id = scenario_identity(run_dir, attempt, scenario)
    pinned = render_node_pinned_manifest(
        settings.manifest,
        scenario_dir / "pinned-workload.yaml",
        node=settings.fault_node,
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
    )
    fixture = ScenarioFixture(
        settings,
        warm,
        scenario=scenario,
        run_id=f"destr008-{scenario}-{attempt}",
    )
    result: dict[str, Any] = {
        "scenario": scenario,
        "verdict": "FAIL",
        "synthetic_trigger": True,
    }
    incident_id = ""
    try:
        mutation = fixture.apply()
        write_json_atomic(scenario_dir / "scenario-mutation.json", mutation)
        submission = workload.submit()
        write_json_atomic(scenario_dir / "submission.json", submission)
        workload.wait_running(timeout_seconds=900)
        observation = wait_observation(
            warm,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                f"maintenance window ended before scenario {scenario}"
            )
        late = fixture.apply_late()
        if late:
            mutation = {**mutation, "late": late}
            write_json_atomic(scenario_dir / "scenario-mutation.json", mutation)
        event_id = f"destr008-{scenario}-{int(time.time())}"
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
        state = warm.wait_for_workflow(
            event_id=event_id,
            job_id=job_id,
            attempt_id=attempt_id,
            case_dir=scenario_dir,
            timeout_seconds=900,
        )
        incident_id = str((state.get("incident") or {}).get("incident_id") or "")
        state["fault_node"] = warm.node_snapshot(settings.fault_node)
        state["spare_node"] = warm.node_snapshot(settings.spare_node)
        write_json_atomic(scenario_dir / "workflow-state.json", state)
        errors = scenario_errors(state, settings, scenario, event_id=event_id)
        provider_after = warm.provider_inventory()
        if provider_after != provider_baseline:
            errors.append("HyperPod provider inventory changed")
        provider_events = regional.provider_events(
            started_at,
            datetime.now(timezone.utc),
        )
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
                "provider_events": provider_events,
                "notifications": state.get("notifications") or [],
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            workload.delete()
        except Exception as exc:
            result["workload_cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
        scenario_cleanup = fixture.restore()
        result["scenario_cleanup"] = scenario_cleanup
        if scenario_cleanup["errors"]:
            result["verdict"] = "FAIL"
        fault_cleanup = restore_fault_node(
            warm,
            settings=settings,
            incident_id=incident_id,
            profile_version=profile_version,
        )
        result["fault_cleanup"] = fault_cleanup
        if fault_cleanup["errors"]:
            result["verdict"] = "FAIL"
        try:
            result["final_fault_node"] = warm.node_snapshot(settings.fault_node)
            result["final_spare_node"] = warm.node_snapshot(settings.spare_node)
            fault = result["final_fault_node"]
            spare = result["final_spare_node"]
            if (
                fault["ready"] != "True"
                or fault["unschedulable"]
                or any(item.get("key") == QUARANTINE_TAINT for item in fault["taints"])
                or any(
                    fault["annotations"].get(key)
                    for key in (
                        "gpu-fault.io/incident-id",
                        "gpu-fault.io/fencing-token",
                        "gpu-fault.io/previous-unschedulable",
                    )
                )
            ):
                result["verdict"] = "FAIL"
                result.setdefault("postflight_errors", []).append(
                    "fault node did not return to Ready/schedulable/unowned"
                )
            if (
                spare["ready"] != "True"
                or not spare["unschedulable"]
                or spare["labels"].get(SPARE_LABEL) != "true"
                or spare["annotations"].get(SPARE_RESERVATION_ANNOTATION)
                or spare["annotations"].get(SPARE_POOL_STATE_ANNOTATION)
                not in {None, "AVAILABLE"}
            ):
                result["verdict"] = "FAIL"
                result.setdefault("postflight_errors", []).append(
                    "spare node did not return to the available cordoned pool"
                )
        except Exception as exc:
            result["postflight_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    write_json_atomic(scenario_dir / f"{scenario}.json", result)
    return result


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    return {
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
        },
        "preflight": preflight,
    }


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
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)
    regional = RegionalLiveFixture(settings.regional)
    warm = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    prewarm = ImagePrewarmFixture(
        regional,
        case_id=CASE_ID,
        run_id=f"destr008-{attempt}",
    )
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "synthetic_trigger": True,
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "selected_scenarios": list(settings.scenarios),
    }
    scenario_results = []
    try:
        prewarm.create([settings.fault_node, settings.spare_node])
        profile_version = str(
            (preflight["store"].get("profile") or {}).get("profile_version") or ""
        )
        for scenario in settings.scenarios:
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
            )
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


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = configure(arguments)
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.execute:
        preflight = read_only_preflight(settings, case_dir)
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=CASE_ID,
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            environment=settings.environment(),
            details=plan_details(settings, preflight),
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 1
    deadline = authorize_execution(
        arguments,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=settings.environment(),
    )
    return execute_case(
        settings,
        arguments.run_dir,
        arguments.attempt,
        deadline,
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
