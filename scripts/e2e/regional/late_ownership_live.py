"""Live assembly of the STOP/queued-Agent race and independent node witnesses."""

from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, get_args

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied, check_stop
from scripts.e2e.regional.late_ownership_contract import (
    AcceptanceScope,
    CaseId,
    CleanupReceipt,
    DecisionReceipt,
    MutationReceipt,
    NodeIdentity,
    Participant,
    QuiescenceReceipt,
    RecheckPermit,
    StopReceipt,
    WitnessEnd,
    WitnessStart,
    WorkloadIdentity,
    ProcessIdentity,
    Scenario,
)
from scripts.e2e.regional.late_ownership_control import cpu_program
from scripts.e2e.regional.late_ownership_ordinary import (
    read_ordinary_completion,
    require_external_ordinary,
)
from scripts.e2e.regional.late_ownership_probe_bundle import probe_program, stdin_loader
from scripts.e2e.regional.late_ownership_resources import OwnedMutation
from scripts.e2e.regional.late_ownership_stream import ProbeStream, open_executor_stream
from scripts.e2e.regional.late_ownership_witness_fixture import NodeWitness
from scripts.e2e.regional.managed_workload_fixture import read_resource
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
    component_python,
)
from scripts.e2e.regional.late_ownership_runner import run_case


def executor_identity(regional: RegionalLiveFixture) -> dict[str, Any]:
    pods = regional.ready_pods("gpu", "gpu-fault-cluster-executor")
    if not pods:
        raise BoundaryDenied("no Ready local Executor exists")
    selected = pods[0]
    name = str(selected["name"])
    document = json.loads(regional.kubectl("gpu", "get", "pod", name, "-o", "json"))
    metadata = document.get("metadata") or {}
    try:
        current = ready_pod_records({"items": [document]})
    except RegionalFixtureError:
        raise BoundaryDenied(
            "Executor Pod identity or readiness is incomplete"
        ) from None
    containers = [
        item
        for item in (document.get("status") or {}).get("containerStatuses", [])
        if item.get("name") == "executor" and item.get("ready") is True
    ]
    if (
        document.get("apiVersion") != "v1"
        or document.get("kind") != "Pod"
        or metadata.get("name") != name
        or metadata.get("namespace") != regional.settings.namespace
        or metadata.get("uid") != selected.get("uid")
        or not metadata.get("uid")
        or metadata.get("deletionTimestamp")
        or len(current) != 1
        or len(containers) != 1
        or not containers[0].get("containerID")
        or not isinstance(containers[0].get("imageID"), str)
        or not containers[0]["imageID"].strip()
    ):
        raise BoundaryDenied("Executor process ownership is incomplete")
    owner = executor_owner_binding(regional, document, containers[0]["imageID"])
    return {
        "name": name,
        "uid": metadata["uid"],
        "container_id": containers[0]["containerID"],
        "image_id": containers[0]["imageID"],
        **owner,
    }


def executor_owner_binding(
    regional: RegionalLiveFixture, pod: dict[str, Any], image_id: str
) -> dict[str, Any]:
    namespace = regional.settings.namespace
    owners = pod["metadata"].get("ownerReferences") or []
    if (
        len(owners) != 1
        or owners[0].get("apiVersion") != "apps/v1"
        or owners[0].get("kind") != "ReplicaSet"
        or owners[0].get("controller") is not True
        or not owners[0].get("name")
        or not owners[0].get("uid")
    ):
        raise BoundaryDenied("Executor Pod has no exact ReplicaSet owner")
    replica = json.loads(
        regional.kubectl("gpu", "get", "replicaset", owners[0]["name"], "-o", "json")
    )
    deployment = json.loads(
        regional.kubectl(
            "gpu", "get", "deployment", "gpu-fault-cluster-executor", "-o", "json"
        )
    )
    for resource, kind, name in (
        (replica, "ReplicaSet", owners[0]["name"]),
        (deployment, "Deployment", "gpu-fault-cluster-executor"),
    ):
        meta = resource.get("metadata") or {}
        if (
            resource.get("apiVersion") != "apps/v1"
            or resource.get("kind") != kind
            or meta.get("name") != name
            or meta.get("namespace") != namespace
            or not meta.get("uid")
            or meta.get("deletionTimestamp")
        ):
            raise BoundaryDenied("Executor controller identity is incomplete")
    replica_meta = replica["metadata"]
    root = replica_meta.get("ownerReferences") or []
    if (
        replica_meta["uid"] != owners[0]["uid"]
        or len(root) != 1
        or root[0].get("apiVersion") != "apps/v1"
        or root[0].get("kind") != "Deployment"
        or root[0].get("controller") is not True
        or root[0].get("name") != deployment["metadata"]["name"]
        or root[0].get("uid") != deployment["metadata"]["uid"]
    ):
        raise BoundaryDenied("Executor ReplicaSet belongs to another deployment")
    images = []
    for spec in (
        pod["spec"],
        replica["spec"]["template"]["spec"],
        deployment["spec"]["template"]["spec"],
    ):
        containers = [
            item
            for item in spec.get("containers", [])
            if item.get("name") == "executor"
        ]
        if len(containers) != 1 or not containers[0].get("image"):
            raise BoundaryDenied("Executor image is not bound to its controller")
        images.append(containers[0]["image"])
    if len(set(images)) != 1 or (
        "@sha256:" in images[0]
        and image_id.rsplit("sha256:", 1)[-1] != images[0].rsplit("sha256:", 1)[-1]
    ):
        raise BoundaryDenied("Executor running image differs from its deployment pin")
    return {
        "deployment_uid": deployment["metadata"]["uid"],
        "replicaset_uid": replica_meta["uid"],
        "image": images[0],
    }


def inspect_protocol(settings: Any, regional: RegionalLiveFixture) -> dict[str, Any]:
    identity = executor_identity(regional)
    program, digest = probe_program("executor")
    raw = regional.kubectl(
        "gpu",
        "exec",
        "-i",
        identity["name"],
        "-c",
        "executor",
        "--",
        component_python("gpu"),
        "-I",
        "-u",
        "-c",
        stdin_loader(program),
        input_text=program
        + json.dumps(
            {
                "inspect_only": True,
                "cluster_id": settings.regional.cluster_id,
                "nodes": list(settings.nodes),
            }
        )
        + "\n",
        timeout=120,
    )
    value = json.loads(raw.splitlines()[-1])
    if executor_identity(regional) != identity:
        raise BoundaryDenied("Executor was replaced during ownership preflight")
    if (
        not isinstance(value, dict)
        or value.get("protocol_ready") is not True
        or value.get("cluster_id") != settings.regional.cluster_id
        or value.get("nodes") != list(settings.nodes)
    ):
        raise BoundaryDenied("the deployed final-ownership protocol is unavailable")
    return {"executor": identity, "bundle_sha256": digest, "protocol_ready": True}


def require_companion_directory(run_dir: Path, predecessor: Path) -> None:
    root = run_dir.expanduser().resolve()
    source = predecessor.expanduser().resolve()
    if source.is_relative_to(root):
        raise ValueError(
            "companion requires external ordinary predecessor evidence and a separate run directory"
        )
    for parent in (root, *root.parents):
        if any(
            (parent / "cases" / case_id / f"{case_id}.json").exists()
            for case_id in get_args(CaseId)
        ):
            raise ValueError("companion run directory overlaps ordinary case evidence")


def read_only_preflight(
    settings: Any, case_dir: Path, *, reuse_focused_tests: bool = False
) -> dict[str, Any]:
    from scripts.e2e.regional import run_destr015_parallel_branch_join as normal

    require_external_ordinary(
        case_dir.parent.parent, settings.ordinary_destr015_evidence
    )
    result = normal.read_only_preflight(
        settings.base, case_dir, reuse_focused_tests=reuse_focused_tests
    )
    if result["errors"]:
        return result
    try:
        ordinary_proof = read_ordinary_completion(
            settings.ordinary_destr015_evidence,
            preflight=result,
            cluster_id=settings.base.regional.cluster_id,
            nodes=settings.base.nodes,
        )
    except Exception as exc:
        result["errors"].append(
            f"ordinary DESTR-015 evidence preflight: {type(exc).__name__}"
        )
        return result
    try:
        protocol = inspect_protocol(
            settings.base, RegionalLiveFixture(settings.base.regional)
        )
        result["late_ownership"] = {
            **protocol,
            "case_id": settings.case_id,
            "scenario": settings.scenario,
            "ordinary_destr015": ordinary_proof,
        }
    except Exception as exc:
        result["errors"].append(
            f"final ownership protocol preflight: {type(exc).__name__}"
        )
    return result


def workflow_inputs(
    run: Any, scope: AcceptanceScope
) -> tuple[WorkflowRequest, FaultIncident]:
    workload_id = f"{scope.workload.namespace}/pytorchjob/{scope.workload.name}"
    steps = [
        WorkflowStepSpec(
            operation=WorkflowOperation.STOP_WORKLOADS,
            execution_owner="gpu-fault-kubernetes-adapter",
            node_ids=list(run.settings.nodes),
            workload_ids=[workload_id],
            parameters={"termination_initiator_incident_id": scope.incident_id},
        )
    ]
    for node in run.settings.nodes:
        gpu = next(
            item["uuid"]
            for item in run.baselines[node]["gpu_inventory"]
            if item["pci_bdf"] == run.bdf[node]
        )
        for operation in (
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTORE_GPU_SERVICES,
            WorkflowOperation.RESTORE_SCHEDULING,
        ):
            owner = (
                "gpu-fault-kubernetes-adapter"
                if operation
                in {
                    WorkflowOperation.MARK_UNSCHEDULABLE,
                    WorkflowOperation.RESTORE_SCHEDULING,
                }
                else "gpu-fault-node-agent"
            )
            steps.append(
                WorkflowStepSpec(
                    operation=operation,
                    execution_owner=owner,
                    node_ids=[node],
                    gpu_uuids=[gpu],
                    workload_ids=[workload_id],
                )
            )
    workflow = WorkflowRequest(
        request_id=scope.workflow_id,
        incident_id=scope.incident_id,
        status=WorkflowStatus.RUNNING,
        fencing_token=scope.fencing_token,
        execution_epoch=scope.execution_epoch,
        official_action="RESET_GPU",
        runtime_profile_version=scope.runtime_profile,
        official_steps=steps,
    )
    incident = FaultIncident(
        incident_id=scope.incident_id,
        event_id=f"acceptance-{scope.challenge}",
        event_type="LATE_OWNERSHIP_ACCEPTANCE",
        cluster_id=scope.cluster_id,
        node_ids=list(run.settings.nodes),
        job_id=run.settings.job_id,
        attempt_id=scope.workload.attempt_id,
        state=IncidentState.ACTION_PENDING,
        fencing_token=scope.fencing_token,
        workflow_request_id=scope.workflow_id,
        official_action="RESET_GPU",
        policy_source="ACCEPTANCE",
        policy_version="physical-late-ownership/v1",
        drill_id=scope.run_id,
    )
    return workflow, incident


def build_scope(
    run: Any, *, case_id: CaseId, scenario: Scenario
) -> tuple[AcceptanceScope, dict[str, Any], dict[str, Any]]:
    source = read_resource(run.regional, "pytorchjob", run.workload.name)
    if source is None:
        raise BoundaryDenied("owned training workload disappeared")
    identity = executor_identity(run.regional)
    namespace = json.loads(
        run.regional.kubectl(
            "gpu", "get", "namespace", run.settings.regional.namespace, "-o", "json"
        )
    )
    pods = run.workload.pods()
    nodes = tuple(
        NodeIdentity(
            name=name,
            uid=run.preflight["nodes"][name]["uid"],
            boot_id=run.baselines[name]["boot_id"],
        )
        for name in run.settings.nodes
    )
    if len(nodes) != 2:
        raise BoundaryDenied("physical acceptance requires exactly two nodes")
    if len(pods) != 2 or {str(item["uid"]) for item in pods} != run.source_uids:
        raise BoundaryDenied(
            "source Pod UID set changed before the physical experiment"
        )
    node_uids = {node.name: node.uid for node in nodes}
    owner_refs = source["metadata"].get("ownerReferences") or []
    if owner_refs:
        raise BoundaryDenied(
            "the owned test workload unexpectedly has a controller owner"
        )
    challenge = secrets.token_hex(32)
    from scripts.e2e.regional.live_driver_guard import source_digest

    participants = [
        Participant(
            pod_uid=str(pod["uid"]),
            owner_uid=source["metadata"]["uid"],
            node_uid=node_uids[str(pod["node"])],
        )
        for pod in pods
    ]
    scope = AcceptanceScope(
        case_id=case_id,
        scenario=scenario,
        run_id=run.run_id,
        challenge=challenge,
        source_sha256=source_digest(),
        release_id=run.preflight["release_id"],
        runtime_profile=run.preflight["store"]["profile"]["profile_version"],
        region=run.settings.regional.region,
        context=run.settings.regional.gpu_context,
        cluster_id=run.settings.regional.cluster_id,
        namespace_uid=namespace["metadata"]["uid"],
        executor_uid=identity["uid"],
        workflow_id=f"late-ownership-{challenge[:24]}",
        incident_id=f"late-ownership-{challenge[:24]}",
        fencing_token=1,
        execution_epoch=1,
        nodes=(nodes[0], nodes[1]),
        workload=WorkloadIdentity(
            namespace=run.settings.regional.namespace,
            name=run.workload.name,
            uid=source["metadata"]["uid"],
            owner_uid=source["metadata"]["uid"],
            attempt_id=run.settings.attempt_id,
        ),
        participants=(participants[0], participants[1]),
        maintenance_start=datetime.now(timezone.utc) - timedelta(seconds=1),
        maintenance_end=run.maintenance_window_end,
    )
    return scope, source, identity


class LiveBoundaryIO:
    evidence_mode: Literal["LIVE"] = "LIVE"

    def __init__(
        self,
        run: Any,
        scope: AcceptanceScope,
        source: dict[str, Any],
        identity: dict[str, Any],
    ) -> None:
        self.run = run
        self.scope = scope
        self.regional = run.regional
        self.identity = identity
        self.directory = run.case_dir / "late-ownership" / scope.scenario
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if (self.directory / "scope.json").exists():
            raise BoundaryDenied("this experiment already has an ownership journal")
        write_json_atomic(self.directory / "scope.json", scope.model_dump(mode="json"))
        image = source["spec"]["pytorchReplicaSpecs"]["Worker"]["template"]["spec"][
            "containers"
        ][0]["image"]
        self.mutation = OwnedMutation(self.regional, scope, source, image)
        self.mutation.journal_path = self.directory / "owned-mutation.json"
        self.workflow, self.incident = workflow_inputs(run, scope)
        self.gpu: ProbeStream | None = None
        self.nodes: dict[str, NodeWitness] = {}
        self.producer: ProcessIdentity | None = None
        self.quiet = False
        self.revoked = False
        self.holder_created = False
        self.node_actions_started = False

    def check_identity(self) -> None:
        if executor_identity(self.regional) != self.identity:
            raise BoundaryDenied("the live Executor UID/container changed")
        namespace = json.loads(
            self.regional.kubectl(
                "gpu", "get", "namespace", self.scope.workload.namespace, "-o", "json"
            )
        )
        if namespace["metadata"]["uid"] != self.scope.namespace_uid:
            raise BoundaryDenied("the workload namespace was replaced")
        for node in self.scope.nodes:
            if self.regional.node_snapshot(node.name).get("uid") != node.uid:
                raise BoundaryDenied("an approved Node UID changed")

    def preflight(self, scope: AcceptanceScope) -> None:
        if scope != self.scope:
            raise BoundaryDenied("live driver scope changed")
        self.check_identity()
        self.regional.verify_runtime_identity(
            self.run.preflight["runtime_identity"],
            evidence_path=self.directory / "runtime-before.json",
            stage="before physical late-ownership acceptance",
        )

    def arm_witnesses(
        self, scope: AcceptanceScope
    ) -> tuple[WitnessStart, WitnessStart]:
        self.holder_created = True
        held = self.regional.cpu_python(
            cpu_program(),
            json.dumps(
                {
                    "action": "hold",
                    "workflow": self.workflow.model_dump(mode="json"),
                    "incident": self.incident.model_dump(mode="json"),
                    "run_id": scope.run_id,
                    "window_end": scope.maintenance_end.isoformat(),
                }
            ),
            attempts=1,
        )
        self.workflow = WorkflowRequest.model_validate(held["workflow"])
        program, _digest = probe_program("executor")
        raw = open_executor_stream(
            kubeconfig=str(self.run.settings.regional.gpu_kubeconfig),
            context=scope.context,
            namespace=scope.workload.namespace,
            pod=self.identity["name"],
            python=component_python("gpu"),
            program=program,
        )
        self.gpu = ProbeStream(
            raw,
            scope_sha256=scope.digest(),
            check_identity=self.check_identity,
            deadline=time.monotonic()
            + (scope.maintenance_end - datetime.now(timezone.utc)).total_seconds()
            + 180,
            holder_check=self.check_holder,
        )
        raw.write_stdin(
            json.dumps(
                {
                    "scope": scope.model_dump(mode="json"),
                    "workflow": self.workflow.model_dump(mode="json"),
                    "incident": self.incident.model_dump(mode="json"),
                }
            )
            + "\n"
        )
        ready = self.gpu.receive("ready")
        self.producer = ProcessIdentity.model_validate(ready["producer"])
        write_json_atomic(self.directory / "executor-process.json", ready)
        for node in scope.nodes:
            probe = self.run.probes[node.name]
            probe._check_target()
            witness = NodeWitness(
                self.regional, probe.settings, scope, node, self.directory
            )
            self.nodes[node.name] = witness
            witness.create()
            if witness.request("status", {}).get("phase") != "ATTACHED":
                raise BoundaryDenied("node witness is not independently attached")
        self.gpu.send("calibrate", {})
        self.gpu.receive("calibrated")
        starts = []
        for node in scope.nodes:
            starts.append(
                WitnessStart.model_validate_json(
                    json.dumps(
                        self.nodes[node.name].request("calibrated", {})["receipt"]
                    )
                )
            )
        return starts[0], starts[1]

    def check_holder(self, request: dict[str, Any]) -> dict[str, Any]:
        expected = {
            "workflow_id": self.scope.workflow_id,
            "fencing_token": self.scope.fencing_token,
            "execution_epoch": self.scope.execution_epoch,
        }
        nonce = request.get("nonce")
        if (
            any(request.get(key) != value for key, value in expected.items())
            or not isinstance(nonce, str)
            or len(nonce) != 64
        ):
            raise BoundaryDenied("CPU holder check escaped the owned workflow")
        value: dict[str, Any] = self.regional.cpu_python(
            cpu_program(),
            json.dumps(
                {
                    "action": "check",
                    "workflow": self.workflow.model_dump(mode="json"),
                    "run_id": self.scope.run_id,
                }
            ),
            attempts=1,
        )
        if (
            any(
                value.get(key) != expected_value
                for key, expected_value in expected.items()
            )
            or value.get("holder_valid") is not True
            or value.get("lifetime_deadline_at")
            != self.scope.maintenance_end.isoformat()
        ):
            raise BoundaryDenied("CPU holder check was not acknowledged")
        return {**value, "nonce": nonce}

    def _gpu(self) -> ProbeStream:
        if self.gpu is None:
            raise BoundaryDenied("Executor probe was not armed")
        return self.gpu

    def client_observation(
        self, node: NodeIdentity, pod_uids: list[str]
    ) -> dict[str, Any]:
        observed = self.nodes[node.name].request("clients", {"pod_uids": pod_uids})[
            "observation"
        ]
        if (
            not isinstance(observed, dict)
            or observed.get("scope_sha256") != self.scope.digest()
            or observed.get("node") != node.model_dump(mode="json")
            or observed.get("pod_uids") != pod_uids
            or observed.get("source") != "nvidia-compute-apps+proc-cgroup/v1"
            or not isinstance(observed.get("observations"), list)
        ):
            raise BoundaryDenied("physical client observation is unbound")
        return observed

    def stop_at_boundary(
        self, scope: AcceptanceScope, starts: tuple[WitnessStart, WitnessStart]
    ) -> StopReceipt:
        self.node_actions_started = True
        self._gpu().send(
            "begin",
            {"witness_starts": [item.model_dump(mode="json") for item in starts]},
        )
        contained = self._gpu().receive("contained")
        if not contained.get("stop_idempotency_key"):
            raise BoundaryDenied("product STOP did not acknowledge containment")
        for node in scope.nodes:
            observed = self.client_observation(
                node, [item.pod_uid for item in scope.participants]
            )
            write_json_atomic(
                self.directory / f"source-clients-{node.name}.json", observed
            )
            if observed["observations"]:
                raise BoundaryDenied("source workload GPU clients remain after STOP")
        self._gpu().send("continue-boundary", {})
        stop = StopReceipt.model_validate_json(json.dumps(self._gpu().receive("stop")))
        if stop.stop_command_id != contained["stop_idempotency_key"]:
            raise BoundaryDenied("physical boundary changed the completed STOP receipt")
        check_stop(scope, stop)
        if stop.producer != self.producer or stop.witness_start_sha256 != tuple(
            item.digest() for item in starts
        ):
            raise BoundaryDenied(
                "physical STOP is not bound to the armed producer and witnesses"
            )
        self.mutation.acknowledge_stop(stop)
        write_json_atomic(
            self.directory / "queued-stop.json", stop.model_dump(mode="json")
        )
        return stop

    def mutate_owned_target(
        self, scope: AcceptanceScope, stop: StopReceipt
    ) -> MutationReceipt:
        self.check_identity()
        payload: dict[str, Any] = {}
        if scope.scenario == "ownership-drift":
            self.mutation.change_owner()
        elif scope.scenario == "late-sibling":
            uid = self.mutation.late_sibling()
            observed = self.client_observation(scope.nodes[1], [uid])
            write_json_atomic(self.directory / "late-sibling-clients.json", observed)
            if not observed["observations"]:
                raise BoundaryDenied("late sibling has no physical GPU client")
            payload = {"sibling_uid": uid, "physical_client_verified": True}
        self._gpu().send("observe-mutation", payload)
        return MutationReceipt.model_validate_json(
            json.dumps(self._gpu().receive("mutation"))
        )

    def recheck(self, scope: AcceptanceScope, permit: RecheckPermit) -> DecisionReceipt:
        self._gpu().send("recheck", permit.model_dump(mode="json"))
        return DecisionReceipt.model_validate_json(
            json.dumps(self._gpu().receive("decision"))
        )

    def quiesce(
        self, scope: AcceptanceScope, decision: DecisionReceipt
    ) -> QuiescenceReceipt:
        self._gpu().send("quiesce", {})
        if (
            self._gpu().receive("services-restored").get("node_commands_terminal")
            is not True
        ):
            raise BoundaryDenied(
                "node actions did not drain before service restoration"
            )
        for node in scope.nodes:
            self.regional.kubectl(
                "gpu",
                "wait",
                "--for=condition=Ready",
                f"node/{node.name}",
                "--timeout=60s",
                timeout=70,
            )
            observed = self.run.probes[node.name].execute(
                "snapshot",
                "--run-id",
                self.run.run_id,
                timeout=60,
            )
            write_json_atomic(
                self.directory / f"restored-host-{node.name}.json", observed
            )
            before = self.run.baselines[node.name]
            if (
                observed.get("boot_id") != node.boot_id
                or len(observed.get("gpu_inventory") or [])
                != len(before["gpu_inventory"])
                or observed.get("quiesce_states")
                or observed.get("gpu_fault_timers") != before.get("gpu_fault_timers")
                or any(
                    state.get("ActiveState") == "active"
                    and (observed.get("services", {}).get(unit) or {}).get(
                        "ActiveState"
                    )
                    != "active"
                    for unit, state in before.get("services", {}).items()
                )
            ):
                raise BoundaryDenied(
                    "node health/service baseline did not return before scheduling"
                )
        self._gpu().send("restore-scheduling", {})
        drained = self._gpu().receive("actions-drained")
        self.workflow = WorkflowRequest.model_validate(drained["workflow"])
        final = self.regional.cpu_python(
            cpu_program(),
            json.dumps(
                {
                    "action": "finish",
                    "workflow": drained["workflow"],
                    "run_id": scope.run_id,
                }
            ),
            attempts=1,
        )
        self._gpu().send("confirm-terminal", final)
        quiet = QuiescenceReceipt.model_validate_json(
            json.dumps(self._gpu().receive("quiescence"))
        )
        if (
            quiet.scope_sha256 != scope.digest()
            or quiet.executor_uid != scope.executor_uid
            or quiet.producer != self.producer
            or quiet.decision_sha256 != decision.digest()
            or quiet.sequence <= decision.sequence
            or quiet.open_commands
            or quiet.pending_callbacks
            or not quiet.workflow_terminal
            or not quiet.gate_revoked
        ):
            raise BoundaryDenied("Executor quiescence acknowledgement is unbound")
        write_json_atomic(
            self.directory / "quiescence.json", quiet.model_dump(mode="json")
        )
        self.quiet = True
        return quiet

    def finish_witnesses(
        self, scope: AcceptanceScope, quiet: QuiescenceReceipt
    ) -> tuple[WitnessEnd, WitnessEnd]:
        results = []
        for node in scope.nodes:
            results.append(
                WitnessEnd.model_validate_json(
                    json.dumps(
                        self.nodes[node.name].request(
                            "finish", quiet.model_dump(mode="json")
                        )["receipt"]
                    )
                )
            )
        return results[0], results[1]

    def revoke(self, scope: AcceptanceScope) -> None:
        if not self.node_actions_started:
            self.revoked = True
            return
        if self.gpu is None:
            self.revoked = True
            return
        self._gpu().send("revoke" if self.quiet else "abort", {})
        if (
            not self.quiet
            or self._gpu().receive("revoked").get("gate_revoked") is not True
        ):
            raise BoundaryDenied(
                "live probe revocation has no quiescence acknowledgement"
            )
        self.revoked = True

    def cleanup(self, scope: AcceptanceScope) -> CleanupReceipt:
        try:
            setup_only = not self.node_actions_started
            if setup_only and self.holder_created:
                self.regional.cpu_python(
                    cpu_program(),
                    json.dumps(
                        {
                            "action": "finish",
                            "workflow": self.workflow.model_dump(mode="json"),
                            "run_id": scope.run_id,
                        }
                    ),
                    attempts=1,
                )
            if not self.revoked or (
                not setup_only and (not self.quiet or self.producer is None)
            ):
                raise BoundaryDenied(
                    "physical cleanup is deferred until action quiescence is proven"
                )
            self.mutation.cleanup()
            self.run.workload.delete()
            self.run.workload_submitted = False
            if not setup_only:
                self._gpu().send("finish", {})
                self._gpu().receive("finished")
                self._gpu().finish()
            for name, witness in self.nodes.items():
                if (
                    not setup_only
                    and witness.request("exit", {}).get("exited") is not True
                ):
                    raise BoundaryDenied("node witness exit was not acknowledged")
                witness.cleanup(self.run.probes[name])
            if any(self.run.prewarm.cleanup().values()):
                raise BoundaryDenied("prewarm resources remain")
            for probe in self.run.probes.values():
                if any(probe.cleanup().values()):
                    raise BoundaryDenied("host probe resources remain")
            self.regional.verify_runtime_identity(
                self.run.preflight["runtime_identity"],
                evidence_path=self.directory / "runtime-after.json",
                stage="after physical late-ownership acceptance",
            )
            from scripts.e2e.regional.late_ownership_barrier import process_identity

            return CleanupReceipt(
                scope_sha256=scope.digest(),
                producer=self.producer or process_identity(os.getpid()),
                executor_uid=scope.executor_uid,
                gate_revoked=True,
                callbacks_drained=True,
                commands_terminal=True,
                owned_resources_absent=True,
                workload_identity_preserved=True,
                nodes_restored=True,
            )
        finally:
            if self.gpu is not None:
                self.gpu.close()


def execute_case(
    settings: Any, run_dir: Path, attempt: int, maintenance_window_end: datetime
) -> int:
    from scripts.e2e.regional import run_destr015_parallel_branch_join as original

    require_companion_directory(run_dir, settings.base.predecessor_path)
    require_external_ordinary(run_dir, settings.ordinary_destr015_evidence)
    preflight = read_only_preflight(
        settings, run_dir / "cases" / settings.case_id, reuse_focused_tests=True
    )
    run = original._prepare_live_run(
        settings.base,
        run_dir,
        attempt,
        maintenance_window_end,
        case_id=settings.case_id,
        preflight_result=preflight,
    )
    io: LiveBoundaryIO | None = None
    try:
        original._start_job_and_probes(run)
        scope, source, identity = build_scope(
            run, case_id=settings.case_id, scenario=settings.scenario
        )
        io = LiveBoundaryIO(run, scope, source, identity)
        result = run_case(scope, io)
        write_json_atomic(io.directory / "result.json", result.summary())
        if result.evidence is not None:
            write_json_atomic(
                io.directory / "receipts.json", result.evidence.model_dump(mode="json")
            )
        print(json.dumps(result.summary(), sort_keys=True))
        return 0 if result.verdict == "PASS" else 1
    finally:
        if io is None or not io.holder_created:
            original._cleanup(run)
