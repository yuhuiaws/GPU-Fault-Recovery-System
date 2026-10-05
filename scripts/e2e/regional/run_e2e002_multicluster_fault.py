#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import (  # noqa: E402
    run_destr009_workload_restart as workload_case,
    training_topology,
)
from scripts.e2e.regional import (  # noqa: E402
    run_workload_acceptance as workload_acceptance,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
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
)
from scripts.e2e.regional.multi_cluster_fixture import (  # noqa: E402
    ClusterTarget,
    MultiClusterSettings,
    container_status_errors,
    control_plane_container_statuses,
    registration_snapshot,
    registrations_are_distinct_physical_clusters,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    predecessor_evidence,
    required,
    run_case_main,
)
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records  # noqa: E402

CASE_ID = "GF-REGIONAL-E2E-002"
# ISO-001 is the same-identity, single-injection isolation case; E2E-002 is
# its two-injection successor. ISO-006 (whole-cluster offline) proves nothing
# this case builds on. The execution order lists ISO-001 before E2E-002.
PREDECESSOR_CASE_ID = "GF-REGIONAL-ISO-001"
CONFIRMATION = "E2E002_RESTART_SAME_JOB_IN_TWO_CLUSTERS"
MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "xid11-three-node-pytorchjob.yaml"
)


def cluster_topologies(
    settings: Settings, preflight: Mapping[str, Any]
) -> dict[str, training_topology.TrainingTopology]:
    """Each cluster's Master/Worker shape, derived from its own node inventory.

    The pair cases demanded the fixtures' native three GPU nodes on both sides
    and refused a two-node plain-EKS cluster that trains on two nodes (live
    2026-10-05: ``both clusters require at least three GPU nodes``, ISO-007
    behind it). The shape is per cluster: HyperPod's three nodes derive the
    native topology and the fixture is used verbatim, the two-node cluster gets
    a rewritten copy (``training_topology.render_topology_manifest``), and each
    side's expected Pods, GPUs and observation world size follow its own
    derivation -- the same contract WORKLOAD-001/002, E2E-001 and DESTR-009/012
    already run on.
    """

    return {
        settings.multi.cluster_a.cluster_id: (
            training_topology.derive_training_topology(preflight["nodes_a"])
        ),
        settings.multi.cluster_b.cluster_id: (
            training_topology.derive_training_topology(preflight["nodes_b"])
        ),
    }


def topology_evidence(
    topologies: Mapping[str, training_topology.TrainingTopology],
) -> dict[str, dict[str, Any]]:
    return {cluster_id: item.evidence() for cluster_id, item in topologies.items()}


@dataclass(frozen=True)
class Settings:
    multi: MultiClusterSettings
    site_file: Path
    job_id: str
    attempt_id: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.multi.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_SHARED_JOB_ID": self.job_id,
            "GPU_FAULT_SHARED_ATTEMPT_ID": self.attempt_id,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def derived_identity(
    run_dir: Path, attempt: int, *, case_id: str = CASE_ID
) -> tuple[str, str]:
    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{case_id}".encode()
    ).hexdigest()[:12]
    job_id = f"{case_id.removeprefix('GF-REGIONAL-').replace('-', '').lower()}-{suffix}"
    return job_id, f"{job_id}-a001"


def configure(
    arguments: argparse.Namespace,
    *,
    case_id: str = CASE_ID,
    predecessor_case_id: str = PREDECESSOR_CASE_ID,
) -> Settings:
    job_id, attempt_id = derived_identity(
        arguments.run_dir, arguments.attempt, case_id=case_id
    )
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir
            / "cases"
            / predecessor_case_id
            / f"{predecessor_case_id}.json"
        ).resolve()
    )
    multi = MultiClusterSettings(
        cpu_kubeconfig=Path(
            required(
                arguments.cpu_kubeconfig
                or os.getenv("GPU_FAULT_CONTROL_KUBECONFIG", ""),
                "CPU kubeconfig",
            )
        )
        .expanduser()
        .resolve(),
        namespace=arguments.namespace,
        region=required(
            arguments.region
            or os.getenv("AWS_REGION", "")
            or os.getenv("AWS_DEFAULT_REGION", ""),
            "AWS Region",
        ),
        cluster_a=ClusterTarget(
            cluster_id=required(arguments.cluster_a, "cluster A ID"),
            gpu_kubeconfig=Path(
                required(arguments.gpu_a_kubeconfig, "cluster A kubeconfig")
            )
            .expanduser()
            .resolve(),
            gpu_context=required(arguments.gpu_a_context, "cluster A context"),
        ),
        cluster_b=ClusterTarget(
            cluster_id=required(arguments.cluster_b, "cluster B ID"),
            gpu_kubeconfig=Path(
                required(arguments.gpu_b_kubeconfig, "cluster B kubeconfig")
            )
            .expanduser()
            .resolve(),
            gpu_context=required(arguments.gpu_b_context, "cluster B context"),
        ),
    )
    configured_job = arguments.job_id.strip() or job_id
    return Settings(
        multi=multi,
        site_file=Path(
            required(
                arguments.site_file or os.getenv("GPU_FAULT_SITE_FILE", ""),
                "regional site file",
            )
        )
        .expanduser()
        .resolve(),
        job_id=configured_job,
        attempt_id=arguments.attempt_id.strip()
        or (f"{configured_job}-a001" if arguments.job_id else attempt_id),
        predecessor_path=predecessor,
    )


def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
    """Run the focused pytest, or reuse the plan's result in ``--execute``."""

    if reuse:
        recorded = reusable_focused_tests(case_dir / "plan.json")
        if recorded is not None:
            return {**recorded, "focused_tests_reused": True}
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/regional/test_regional_control_plane.py::"
        "test_attempt_index_is_cluster_scoped",
        "tests/regional/test_regional_control_plane.py::"
        "test_failed_remote_restart_leaves_the_reservation_to_terminalization",
    ]
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=300)
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


EXECUTOR_ID_ENV = ("GPU_FAULT_CLUSTER_EXECUTOR_ID", "GPU_FAULT_EXECUTOR_ID")


def executor_identities(fixture: RegionalLiveFixture) -> list[str]:
    """The identities a cluster's executor may lease commands under.

    The lease owner is the executor's ``GPU_FAULT_CLUSTER_EXECUTOR_ID``
    (``cluster_executor/bootstrap.py``): when the Deployment sets it -- literally
    or from the Pod name -- that value alone; otherwise the executor's default
    ``<cluster_id>/<hostname>``, where the hostname is the Pod name, so both the
    prefixed form and the bare Pod name are accepted. The stock data-plane
    manifest sets neither variable, which is why reading only an explicit
    ``GPU_FAULT_EXECUTOR_ID`` refused every executor Pod as "missing" (live
    2026-10-05, E2E-002 and ISO-007 on the mixed site). Two different literals
    on one Pod are ambiguous.
    """

    document = json.loads(
        fixture.kubectl(
            "gpu", "get", "pod", "-l", "app=gpu-fault-cluster-executor", "-o", "json"
        )
    )
    cluster_id = str(
        getattr(getattr(fixture, "settings", None), "cluster_id", "") or ""
    )
    identities: set[str] = set()
    ready = ready_pod_records(document)
    if not ready or len(ready) != len(document["items"]):
        raise RegionalFixtureError(
            "executor identity inventory is not completely Ready"
        )
    for item in document["items"]:
        pod_name = str(item["metadata"]["name"])
        selected: set[str] = set()
        for container in item.get("spec", {}).get("containers", []):
            for entry in container.get("env", []):
                if entry.get("name") not in EXECUTOR_ID_ENV:
                    continue
                if entry.get("value"):
                    selected.add(str(entry["value"]))
                elif (entry.get("valueFrom") or {}).get("fieldRef", {}).get(
                    "fieldPath"
                ) == "metadata.name":
                    selected.add(pod_name)
                else:
                    raise RegionalFixtureError(
                        "executor lease identity is missing or ambiguous"
                    )
        if len(selected) > 1:
            raise RegionalFixtureError(
                "executor lease identity is missing or ambiguous"
            )
        if not selected:
            selected = {pod_name, *([f"{cluster_id}/{pod_name}"] if cluster_id else [])}
        identities.update(selected)
    return sorted(identity for identity in identities if identity)


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
    *,
    reuse_focused_tests: bool = False,
    predecessor_case_id: str = PREDECESSOR_CASE_ID,
) -> dict[str, Any]:
    a = settings.multi.regional(settings.multi.cluster_a)
    b = settings.multi.regional(settings.multi.cluster_b)
    registrations = registration_snapshot(a, settings.multi)
    identity = a.evidence_identity()
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        predecessor_case_id,
        release_id=identity["release_id"],
    )
    nodes_a = a.gpu_nodes()
    nodes_b = b.gpu_nodes()
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    errors = []
    if not predecessor["valid"]:
        errors.append(f"{predecessor_case_id} predecessor evidence is not PASS")
    if len(registrations) != 2 or any(
        not item.get("enabled") or item.get("synthetic") for item in registrations
    ):
        errors.append("two enabled physical cluster registrations are required")
    elif not registrations_are_distinct_physical_clusters(registrations):
        errors.append("cluster registrations do not identify two physical clusters")
    topologies: dict[str, dict[str, Any]] = {}
    for cluster_id, nodes in (
        (settings.multi.cluster_a.cluster_id, nodes_a),
        (settings.multi.cluster_b.cluster_id, nodes_b),
    ):
        try:
            topologies[cluster_id] = training_topology.derive_training_topology(
                nodes
            ).evidence()
        except training_topology.TrainingTopologyError as exc:
            errors.append(f"{cluster_id}: {exc}")
    if a.gpu_workloads() or b.gpu_workloads():
        errors.append("a target cluster already has GPU workloads")
    if not settings.site_file.is_file() or not MANIFEST.is_file():
        errors.append("site or training manifest is missing")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        **identity,
        "registrations": registrations,
        "nodes_a": nodes_a,
        "nodes_b": nodes_b,
        "training_topology": topologies,
        "predecessor": predecessor,
        "cpu_pods": a.ready_pods("cpu", "gpu-fault-api-ha"),
        "cpu_containers": control_plane_container_statuses(a),
        "executor_identities": {
            settings.multi.cluster_a.cluster_id: executor_identities(a),
            settings.multi.cluster_b.cluster_id: executor_identities(b),
        },
        "focused_tests": tests,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def workload_settings(
    settings: Settings,
    target: ClusterTarget,
) -> workload_case.Settings:
    fixture = settings.multi.regional(target)
    return workload_case.Settings(
        regional=fixture.settings,
        site_file=settings.site_file,
        manifest=MANIFEST,
        job_id=settings.job_id,
        attempt_id=settings.attempt_id,
        predecessor_path=settings.predecessor_path,
    )


def managed_workload(
    settings: Settings,
    target: ClusterTarget,
    *,
    topology: training_topology.TrainingTopology,
    case_dir: Path,
    restart_budget: int = 1,
) -> ManagedWorkloadFixture:
    """This cluster's workload in its own shape: the fixture itself on three
    native nodes, else a copy rewritten for ``topology`` beside the evidence."""

    fixture = settings.multi.regional(target)
    return ManagedWorkloadFixture(
        fixture,
        ManagedWorkloadSettings(
            manifest=training_topology.render_topology_manifest(
                MANIFEST,
                topology,
                case_dir / target.cluster_id / "topology-workload.yaml",
            ),
            site_file=settings.site_file,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            restart_budget=restart_budget,
            expected_pods=topology.node_count,
            expected_gpu_count=topology.world_size,
        ),
    )


def notification_errors(
    states: list[dict[str, Any]],
    cluster_ids: list[str],
    registrations: list[dict[str, Any]],
    *,
    denied_cluster_id: str | None = None,
    snapshots: list[Callable[[], dict[str, Any]]] | None = None,
) -> list[str]:
    """The two emails belong to their own cluster's incident, and only to it.

    ``snapshots`` re-reads one cluster's store while its ACTION_COMPLETED
    delivery is still in flight (the result row lands after the workflow is
    terminal); without it every state is judged as read.
    """

    errors = []
    names_by_cluster = {
        str(item.get("cluster_id")): {
            str(item.get("cluster_id")),
            str(item.get("hyperpod_cluster_name") or ""),
        }
        for item in registrations
    }
    seen: dict[str, str] = {}
    for index, (state, cluster_id) in enumerate(zip(states, cluster_ids, strict=True)):
        if cluster_id != denied_cluster_id and snapshots is None:
            errors.extend(workload_acceptance.notification_errors(state))
        elif cluster_id != denied_cluster_id:
            errors.extend(
                workload_acceptance.wait_for_notification_results(
                    state, snapshot=snapshots[index]
                )
            )
        incident = state.get("incident") or {}
        if incident.get("cluster_id") != cluster_id:
            errors.append(f"{cluster_id}: incident is not scoped to the cluster")
        notifications = state.get("notifications") or []
        if not notifications:
            errors.append(f"{cluster_id}: no notification was recorded")
        for entry in notifications:
            notification = entry.get("notification") or {}
            delivery = entry.get("result") or {}
            notification_id = str(notification.get("notification_id") or "")
            if not notification_id:
                errors.append(f"{cluster_id}: notification identity is missing")
            if notification.get("incident_id") != incident.get("incident_id"):
                errors.append(f"{cluster_id}: notification names another incident")
            if str(notification.get("cluster_name") or "") not in names_by_cluster.get(
                cluster_id, {cluster_id}
            ):
                errors.append(f"{cluster_id}: notification cluster_name is foreign")
            if notification_id in seen and seen[notification_id] != cluster_id:
                errors.append("a notification appears under both clusters")
            if delivery.get("status") != "SENT" or not delivery.get(
                "provider_message_id"
            ):
                errors.append(f"{cluster_id}: notification delivery is not proven")
            seen[notification_id] = cluster_id
    return errors


def command_scope_errors(
    states: list[dict[str, Any]],
    cluster_ids: list[str],
    identities: dict[str, list[str]],
) -> list[str]:
    """Each remote command was leased by its own cluster's executor."""

    errors = []
    for state, cluster_id in zip(states, cluster_ids, strict=True):
        incident = state.get("incident") or {}
        workflow = state.get("workflow") or {}
        if (
            not incident.get("incident_id")
            or incident.get("cluster_id") != cluster_id
            or not workflow.get("request_id")
            or workflow.get("incident_id") != incident.get("incident_id")
        ):
            errors.append(f"{cluster_id}: workflow/incident binding is incomplete")
        known = identities.get(cluster_id) or []
        if not known:
            errors.append(f"{cluster_id}: executor identities were not observed")
        commands = state.get("commands") or []
        if not commands:
            errors.append(f"{cluster_id}: no remote command was recorded")
        for item in commands:
            if item.get("cluster_id") != cluster_id:
                errors.append("remote command crossed cluster scope")
            if (
                not item.get("command_id")
                or item.get("workflow_request_id") != workflow.get("request_id")
                or item.get("incident_id") != incident.get("incident_id")
            ):
                errors.append(
                    f"{cluster_id}: command workflow/incident binding differs"
                )
            owner = str(item.get("last_lease_owner") or item.get("lease_owner") or "")
            if not owner:
                errors.append(
                    f"{cluster_id}: command {item.get('command_id')} was never leased"
                )
                continue
            if owner not in known:
                errors.append(
                    f"{cluster_id}: command leased by unknown executor {owner!r}"
                )
    return errors


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    topologies = preflight.get("training_topology") or {}
    shapes = ", ".join(
        f"{cluster_id}: {item['world_size']} GPUs on {item['node_count']} nodes"
        for cluster_id, item in sorted(topologies.items())
    )
    details = {
        "risk": "live-workload-restart",
        "predecessor": preflight["predecessor"],
        "training_topology": topologies,
        "cluster_ids": [
            settings.multi.cluster_a.cluster_id,
            settings.multi.cluster_b.cluster_id,
        ],
        "shared_job_id": settings.job_id,
        "shared_attempt_id": settings.attempt_id,
        "mutation": (
            "submit the same managed job/attempt identity to two physical GPU "
            "clusters and inject XID11 into both within one 60-second window"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "cluster_a_node_uids": sorted(
                str(item["uid"]) for item in preflight["nodes_a"]
            ),
            "cluster_b_node_uids": sorted(
                str(item["uid"]) for item in preflight["nodes_b"]
            ),
        },
        "stop_conditions": [
            "ISO-001 predecessor evidence is not PASS",
            "fewer than two enabled physical clusters",
            "either workload or its derived-topology GPU observation fails"
            + (f" ({shapes})" if shapes else ""),
            "injections are more than 60 seconds apart",
            "incident, command, budget or notification crosses cluster scope",
            "a control-plane container restarts",
            "cleanup leaves either workload behind",
        ],
        "rollback": {
            "delete_both_training_workloads": True,
            "delete_both_image_prewarm_sets": True,
            "no_node_or_provider_mutation": True,
        },
        "preflight": preflight,
    }
    record_focused_tests(details, preflight["focused_tests"])
    return details


@dataclass
class RecoveryBranches:
    settings: Settings
    targets: tuple[ClusterTarget, ClusterTarget]
    fixtures: list[RegionalLiveFixture]
    workloads: list[ManagedWorkloadFixture]
    prewarms: list[ImagePrewarmFixture]
    submission_started: list[bool]
    injection_contexts: list[dict[str, Any] | None]
    case_dir: Path
    expect_a_budget_denial: bool
    sources: list[dict[str, Any]]
    topologies: dict[str, training_topology.TrainingTopology] = field(
        default_factory=dict
    )

    def topology(self, index: int) -> training_topology.TrainingTopology:
        return self.topologies.get(
            self.targets[index].cluster_id, training_topology.NATIVE_TOPOLOGY
        )

    def prepare(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Prewarm, submit, wait Running and wait for the observation of this
        cluster's whole world (its derived topology). The two clusters are
        independent, so they run side by side: serially this phase cost twice
        the slowest cluster."""

        fixture, workload, prewarm, target = (
            self.fixtures[index],
            self.workloads[index],
            self.prewarms[index],
            self.targets[index],
        )
        topology = self.topology(index)
        nodes = list(topology.candidate_nodes) or [
            str(item["name"])
            for item in fixture.gpu_nodes()
            if item["ready"] == "True" and not item["unschedulable"]
        ]
        prewarm.create(nodes)
        self.submission_started[index] = True
        workload.submit()
        source = workload.wait_running(timeout_seconds=900)
        observation = workload_case.wait_observation(
            fixture,
            workload_settings(self.settings, target),
            node=str(source["pods"][0]["node"]),
            expected_gpu_count=topology.world_size,
        )
        if workload_acceptance.source_observation_errors(
            observation,
            source,
            cluster_id=fixture.settings.cluster_id,
            job_id=self.settings.job_id,
            attempt_id=self.settings.attempt_id,
        ):
            raise RegionalFixtureError(
                "source Observation is not fresh and workload-bound"
            )
        if type(observation.get("restart_budget")) is not int or observation[
            "restart_budget"
        ] != (0 if self.expect_a_budget_denial and index == 0 else 1):
            raise RegionalFixtureError(
                "source Observation restart budget differs from this scenario"
            )
        return source, observation

    def settle(self, index: int, injected_at: datetime, payload: dict[str, Any]) -> Any:
        fixture, workload, source = (
            self.fixtures[index],
            self.workloads[index],
            self.sources[index],
        )
        state = fixture.wait_for_workflow(
            node=str(source["pods"][0]["node"]),
            marker=str(payload["record_id"]),
            observed_after=injected_at,
            case_dir=self.case_dir / str(fixture.settings.cluster_id),
            timeout_seconds=1200,
            job_id=self.settings.job_id,
            attempt_id=self.settings.attempt_id,
        )
        context = self.injection_contexts[index]
        if context is not None and (state.get("workflow") or {}).get("request_id"):
            context["workflow_request_ids"] = [state["workflow"]["request_id"]]
        if self.expect_a_budget_denial and index == 0:
            restarted = {"pods": workload.pods()}
        else:
            workload.authorize_restart(state)
            restarted = workload.wait_restarted(
                {str(item["uid"]) for item in source["pods"]}, timeout_seconds=900
            )
        return state, restarted

    def cleanup(self, result: dict[str, Any]) -> None:
        for index, workload in enumerate(self.workloads):
            if not self.submission_started[index]:
                continue
            errors = workload_acceptance.cleanup_workload(
                regional=self.fixtures[index],
                workload=workload,
                case_dir=self.case_dir / self.targets[index].cluster_id,
                job_id=self.settings.job_id,
                attempt_id=self.settings.attempt_id,
                injection=self.injection_contexts[index],
                result=result,
                label=f"cluster_{index}",
            )
            if errors:
                result["errors"].extend(
                    f"workload cleanup failed: {error}" for error in errors
                )
                result["verdict"] = "FAIL"
        for prewarm in self.prewarms:
            try:
                residuals = prewarm.cleanup()
                if any(residuals.values()):
                    result["errors"].append("prewarm Pods remain")
                    result["verdict"] = "FAIL"
            except Exception as exc:
                result["errors"].append(
                    f"prewarm cleanup failed: {type(exc).__name__}: {exc}"
                )
                result["verdict"] = "FAIL"


def verify_cluster_recoveries(
    *,
    settings: Settings,
    fixtures: list[RegionalLiveFixture],
    cluster_ids: list[str],
    sources: list[dict[str, Any]],
    states: list[dict[str, Any]],
    targets_after: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    injection_starts: dict[int, datetime],
    preflight: dict[str, Any],
    result: dict[str, Any],
    expect_a_budget_denial: bool,
    topologies: Mapping[str, training_topology.TrainingTopology] | None = None,
) -> None:
    for index, state in enumerate(states):
        if expect_a_budget_denial and index == 0:
            result["errors"].extend(
                budget_denial_errors(
                    state,
                    cluster_id=cluster_ids[0],
                    job_id=settings.job_id,
                    pods=targets_after[0]["pods"],
                    source_uids={str(pod["uid"]) for pod in sources[0]["pods"]},
                )
            )
        else:
            world = (
                (topologies or {})
                .get(cluster_ids[index], training_topology.NATIVE_TOPOLOGY)
                .world_size
            )
            result["errors"].extend(
                workload_case.workflow_errors(state, expected_gpu_count=world)
            )
    workflow_ids = {(state.get("workflow") or {}).get("request_id") for state in states}
    incident_ids = {
        (state.get("incident") or {}).get("incident_id") for state in states
    }
    if len(workflow_ids) != 2 or len(incident_ids) != 2:
        result["errors"].append("incident/workflow identities are not isolated")
    for fixture, state in zip(fixtures, states, strict=True):
        index = fixtures.index(fixture)
        result["errors"].extend(
            workload_acceptance.recovery_identity_errors(
                state,
                cluster_id=fixture.settings.cluster_id,
                job_id=settings.job_id,
                attempt_id=settings.attempt_id,
                node=payloads[index]["node_id"],
                marker=payloads[index]["record_id"],
                observed_after=injection_starts[index],
                restart_withheld=expect_a_budget_denial and index == 0,
            )
        )
        budget = state.get("restart_budget") or {}
        expected_count = 0 if expect_a_budget_denial and index == 0 else 1
        if budget.get("restart_count") != expected_count:
            result["errors"].append(
                f"{fixture.settings.cluster_id} restart count differs from {expected_count}"
            )
        if (
            budget.get("cluster_id") != fixture.settings.cluster_id
            or budget.get("job_id") != settings.job_id
        ):
            result["errors"].append("restart budget crossed cluster/job scope")
    result["errors"].extend(
        command_scope_errors(
            states, cluster_ids, preflight.get("executor_identities") or {}
        )
    )

    def store_reader(index: int) -> Callable[[], dict[str, Any]]:
        # The same identity ``settle`` handed ``wait_for_workflow``; resolved
        # only when a delivery result is still in flight.
        return lambda: fixtures[index].store_snapshot(
            node=str(sources[index]["pods"][0]["node"]),
            marker=str(payloads[index]["record_id"]),
            observed_after=injection_starts[index],
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            queue_attempts=1,
        )

    result["errors"].extend(
        notification_errors(
            states,
            cluster_ids,
            preflight["registrations"],
            denied_cluster_id=cluster_ids[0] if expect_a_budget_denial else None,
            snapshots=[store_reader(index) for index in range(len(fixtures))],
        )
    )
    if expect_a_budget_denial:
        for fixture, baseline in zip(
            fixtures, (preflight["nodes_a"], preflight["nodes_b"]), strict=True
        ):
            if fixture.gpu_nodes() != baseline:
                result["errors"].append(
                    "GPU node state changed in the workload-only isolation case"
                )


def inject_cluster_faults(
    fixtures: list[RegionalLiveFixture],
    payloads: list[dict[str, Any]],
    injection_contexts: list[dict[str, Any] | None],
    *,
    maintenance_window_end: datetime,
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, datetime]]:
    injection_starts: dict[int, datetime] = {}

    def inject(index: int) -> dict[str, Any]:
        started = datetime.now(timezone.utc)
        if started >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before injection"
            )
        injection_starts[index] = started
        payloads[index]["observed_at"] = started.isoformat()
        injection_contexts[index] = {
            "node": payloads[index]["node_id"],
            "marker": payloads[index]["record_id"],
            "observed_after": started,
        }
        return fixtures[index].post_xid_event(payloads[index])

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending_injections = [
            pool.submit(inject, index) for index in range(len(fixtures))
        ]
        injections = []
        for index, future in enumerate(pending_injections):
            try:
                injections.append(future.result())
            except Exception as exc:
                injections.append({"error_type": type(exc).__name__})
                result["errors"].append(f"cluster {index} injection receipt failed")
    result["injections"] = injections
    result["injection_started_at"] = {
        str(index): started.isoformat() for index, started in injection_starts.items()
    }
    if (
        max(injection_starts.values()) - min(injection_starts.values())
    ).total_seconds() > 60:
        result["errors"].append("concurrent injections exceeded 60 seconds")
    return injections, injection_starts


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
    *,
    case_id: str = CASE_ID,
    predecessor_case_id: str = PREDECESSOR_CASE_ID,
    expect_a_budget_denial: bool = False,
    preflight_reader: Callable[..., dict[str, Any]] | None = None,
) -> int:
    case_dir = run_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = (
        preflight_reader(settings, case_dir, reuse_focused_tests=True)
        if preflight_reader is not None
        else read_only_preflight(
            settings,
            case_dir,
            reuse_focused_tests=True,
            predecessor_case_id=predecessor_case_id,
        )
    )
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise RegionalFixtureError("approved maintenance window has ended")
    targets = (settings.multi.cluster_a, settings.multi.cluster_b)
    cluster_ids = [target.cluster_id for target in targets]
    fixtures = [settings.multi.regional(target) for target in targets]
    topologies = cluster_topologies(settings, preflight)
    workloads = [
        managed_workload(
            settings,
            target,
            topology=topologies[target.cluster_id],
            case_dir=case_dir,
            restart_budget=0 if expect_a_budget_denial and index == 0 else 1,
        )
        for index, target in enumerate(targets)
    ]
    prewarms = [
        ImagePrewarmFixture(
            fixture,
            case_id=case_id,
            run_id=f"{case_id.lower()}-{index}-{attempt}",
        )
        for index, fixture in enumerate(fixtures)
    ]
    result: dict[str, Any] = {
        "case_id": case_id,
        "attempt": attempt,
        **fixtures[0].evidence_identity(),
        "cluster_ids": cluster_ids,
        "training_topology": topology_evidence(topologies),
        "verdict": "FAIL",
        "errors": [],
    }
    submission_started = [False, False]
    injection_contexts: list[dict[str, Any] | None] = [None, None]

    branches = RecoveryBranches(
        settings=settings,
        targets=targets,
        fixtures=fixtures,
        workloads=workloads,
        prewarms=prewarms,
        submission_started=submission_started,
        injection_contexts=injection_contexts,
        case_dir=case_dir,
        expect_a_budget_denial=expect_a_budget_denial,
        sources=[],
        topologies=topologies,
    )
    sources: list[dict[str, Any]] = []
    try:
        baseline_states = [
            workload_acceptance.workload_store(
                fixture, job_id=settings.job_id, attempt_id=settings.attempt_id
            )
            for fixture in fixtures
        ]
        workload_acceptance.assert_clean_identity_baseline(
            baseline_states, job_id=settings.job_id, attempt_id=settings.attempt_id
        )
        result["baseline_states"] = baseline_states
        containers_before = preflight["cpu_containers"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            prepared = list(pool.map(branches.prepare, range(len(targets))))
        sources = [item[0] for item in prepared]
        branches.sources = sources
        observations = [item[1] for item in prepared]
        injected_at = datetime.now(timezone.utc)
        payloads = [
            workload_case.xid11_payload(
                workload_settings(settings, target),
                case_id=case_id,
                marker=f"{case_id.lower()}-{index}-{int(time.time())}",
                node=str(source["pods"][0]["node"]),
                product=workload_case.normalize_product(
                    fixture.node_metadata(str(source["pods"][0]["node"])).get("product")
                ),
                observation=observation,
                observed_at=injected_at,
            )
            for index, (fixture, source, observation, target) in enumerate(
                zip(fixtures, sources, observations, targets, strict=True)
            )
        ]
        injections, injection_starts = inject_cluster_faults(
            fixtures,
            payloads,
            injection_contexts,
            maintenance_window_end=maintenance_window_end,
            result=result,
        )
        # Both workflows run at once; waiting for them one after the other
        # would have serialised two 20-minute bounds and, worse, watched the
        # second cluster only after the first had already settled.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                index: pool.submit(
                    branches.settle, index, injection_starts[index], payloads[index]
                )
                for index in sorted(injection_starts)
            }
            settled: list[tuple[dict[str, Any], dict[str, Any]]] = []
            result["cluster_recovery_outcomes"] = {}
            for index, settlement in futures.items():
                try:
                    outcome = settlement.result()
                    settled.append(outcome)
                    result["cluster_recovery_outcomes"][targets[index].cluster_id] = (
                        outcome
                    )
                except Exception as exc:
                    result["cluster_recovery_outcomes"][targets[index].cluster_id] = {
                        "error_type": type(exc).__name__
                    }
                    result["errors"].append(
                        f"cluster {index} recovery failed: {type(exc).__name__}"
                    )
            if len(settled) != len(targets):
                raise RegionalFixtureError(
                    "both injected branches were not observed to completion"
                )
        states = [item[0] for item in settled]
        targets_after = [item[1] for item in settled]
        verify_cluster_recoveries(
            settings=settings,
            fixtures=fixtures,
            cluster_ids=cluster_ids,
            sources=sources,
            states=states,
            targets_after=targets_after,
            payloads=payloads,
            injection_starts=injection_starts,
            preflight=preflight,
            result=result,
            expect_a_budget_denial=expect_a_budget_denial,
            topologies=topologies,
        )
        containers_after = control_plane_container_statuses(fixtures[0])
        result["errors"].extend(
            container_status_errors(containers_before, containers_after)
        )
        result.update(
            {
                "verdict": "PASS" if not result["errors"] else "FAIL",
                "injections": injections,
                "states": states,
                "target_workloads": targets_after,
                "cpu_containers_after": containers_after,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        branches.cleanup(result)
    write_json_atomic(case_dir / f"{case_id}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def budget_denial_errors(
    state: dict[str, Any],
    *,
    cluster_id: str,
    job_id: str,
    pods: list[dict[str, Any]],
    source_uids: set[str],
) -> list[str]:
    errors = []
    workflow = state.get("workflow") or {}
    failed = [
        step
        for step in workflow.get("step_executions") or []
        if step.get("operation") == "RESTART_WORKLOAD"
    ]
    details = (failed[0].get("details") or {}) if len(failed) == 1 else {}
    if (
        workflow.get("status") != "FAILED"
        or len(failed) != 1
        or failed[0].get("status") != "FAILED"
        or details.get("reason") != "RESTART_BUDGET_EXHAUSTED"
        or any(
            type(details.get(key)) is not int or details[key] != 0
            for key in ("restart_budget", "restart_count")
        )
        or type((state.get("restart_budget") or {}).get("budget")) is not int
        or (state.get("restart_budget") or {})["budget"] != 0
    ):
        errors.append(
            "A did not reach the precise zero-budget terminal recovery failure"
        )
    if any(
        step.get("operation") == "RESTART_WORKLOAD"
        for command in state.get("commands") or []
        for step in [
            command.get("step") or {},
            *[entry.get("step") or {} for entry in command.get("batched_steps") or []],
        ]
    ):
        errors.append("A issued a remote restart despite its zero budget")
    if not source_uids or any(
        pod.get("phase") not in {"Succeeded", "Failed"}
        or pod.get("uid") not in source_uids
        for pod in pods
    ):
        errors.append("A workload is not stopped after its rejected recovery")
    notices = [
        entry
        for entry in state.get("notifications") or []
        if (entry.get("notification") or {}).get("notification_id")
        == details.get("notification_id")
    ]
    if (
        not details.get("notification_id")
        or len(notices) != 1
        or (notices[0].get("notification") or {}).get("deduplication_key")
        != f"{cluster_id}/{job_id}/restart-budget-exhausted/0"
        or (notices[0].get("result") or {}).get("status") != "SENT"
        or not (notices[0].get("result") or {}).get("provider_message_id")
    ):
        errors.append("A budget-denial notification is not causally delivered")
    return errors


def parser(
    *,
    confirmation: str = CONFIRMATION,
    description: str = "Run E2E-002 concurrent same-identity faults in two clusters.",
) -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=description)
    add_live_arguments(value, confirmation=confirmation)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--region", default="")
    value.add_argument("--cluster-a", required=True)
    value.add_argument("--gpu-a-kubeconfig", required=True)
    value.add_argument("--gpu-a-context", required=True)
    value.add_argument("--cluster-b", required=True)
    value.add_argument("--gpu-b-kubeconfig", required=True)
    value.add_argument("--gpu-b-context", required=True)
    value.add_argument("--site-file", default="")
    value.add_argument("--job-id", default="")
    value.add_argument("--attempt-id", default="")
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
