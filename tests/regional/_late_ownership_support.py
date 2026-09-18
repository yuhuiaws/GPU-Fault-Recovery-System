"""Local, credential-free receipt builders for late-ownership tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.e2e.regional.late_ownership_contract import (
    AcceptanceEvidence,
    AcceptanceScope,
    CleanupReceipt,
    DecisionReceipt,
    MutationReceipt,
    NodeIdentity,
    Participant,
    PhysicalAction,
    ProcessIdentity,
    QuiescenceReceipt,
    RecheckPermit,
    Scenario,
    StopReceipt,
    WitnessEnd,
    WitnessStart,
    WorkloadIdentity,
)

DIGEST = "a" * 64


def scope(scenario: Scenario = "ownership-drift") -> AcceptanceScope:
    now = datetime.now(timezone.utc)
    return AcceptanceScope(
        case_id="GF-REGIONAL-DESTR-015",
        scenario=scenario,
        run_id="late-ownership-local",
        challenge="b" * 64,
        source_sha256=DIGEST,
        release_id="release-local",
        runtime_profile="profile-local",
        region="test-region",
        context="local-context",
        cluster_id="local-cluster",
        namespace_uid="namespace-original",
        executor_uid="executor-original",
        workflow_id="workflow-local",
        incident_id="incident-local",
        fencing_token=7,
        execution_epoch=3,
        nodes=(
            NodeIdentity(name="node-a", uid="node-uid-a", boot_id="boot-a"),
            NodeIdentity(name="node-b", uid="node-uid-b", boot_id="boot-b"),
        ),
        workload=WorkloadIdentity(
            namespace="training-local",
            name="local-job",
            uid="workload-original",
            owner_uid="owner-original",
            attempt_id="attempt-original",
        ),
        participants=(
            Participant(
                pod_uid="pod-original-a",
                owner_uid="owner-original",
                node_uid="node-uid-a",
            ),
            Participant(
                pod_uid="pod-original-b",
                owner_uid="owner-original",
                node_uid="node-uid-b",
            ),
        ),
        maintenance_start=now - timedelta(seconds=10),
        maintenance_end=now + timedelta(minutes=10),
    )


def process(pid: int = 200, boot: str = "executor-boot") -> ProcessIdentity:
    return ProcessIdentity(pid=pid, start_ticks=1234, uid=1000, boot_id=boot)


def mutation_for(binding: AcceptanceScope, stop: StopReceipt) -> MutationReceipt:
    workload = binding.workload
    participants = binding.participants
    sibling = None
    sibling_node = None
    if binding.scenario == "ownership-drift":
        workload = workload.model_copy(update={"owner_uid": "owner-replaced"})
    if binding.scenario == "late-sibling":
        sibling = "pod-late-b"
        sibling_node = binding.nodes[1].uid
        participants = (
            *participants,
            Participant(
                pod_uid=sibling, owner_uid=workload.owner_uid, node_uid=sibling_node
            ),
        )
    return MutationReceipt(
        scope_sha256=binding.digest(),
        executor_uid=binding.executor_uid,
        producer=stop.producer,
        stop_sha256=stop.digest(),
        sequence=stop.sequence + 1,
        workload=workload,
        participants=participants,
        resource_version="42",
        live_sibling_pod_uid=sibling,
        live_sibling_node_uid=sibling_node,
        sibling_gpu_client_observed=sibling is not None,
    )


def evidence(scenario: Scenario = "ownership-drift") -> AcceptanceEvidence:
    binding = scope(scenario)
    digest = binding.digest()
    starts = tuple(
        WitnessStart(
            scope_sha256=digest,
            executor_uid=binding.executor_uid,
            producer=process(300 + index, node.boot_id),
            node=node,
            witness_id=f"witness-{index}",
            sequence=1,
            calibration_execs=1,
            observer="linux-exec-trace",
            tracee=process(400 + index, node.boot_id),
            executable_path="/usr/bin/true",
            executable_sha256=DIGEST,
        )
        for index, node in enumerate(binding.nodes)
    )
    stop = StopReceipt(
        scope_sha256=digest,
        executor_uid=binding.executor_uid,
        producer=process(),
        boundary_id="c" * 64,
        stop_command_id="stop-command-local",
        queued_command_id="reset-command-local",
        agent_generation=1,
        agent_boundary="AGENT_PRE_SPAWN",
        sequence=10,
        workload=binding.workload,
        participants=binding.participants,
        absent_pod_uids=tuple(item.pod_uid for item in binding.participants),
        empty_client_node_uids=tuple(item.uid for item in binding.nodes),
        witness_start_sha256=tuple(item.digest() for item in starts),
        hardware_submitted=False,
        gate_closed=True,
    )
    mutation = mutation_for(binding, stop)
    permit = RecheckPermit(
        scope_sha256=digest,
        boundary_id=stop.boundary_id,
        stop_sha256=stop.digest(),
        mutation_sha256=mutation.digest(),
    )
    decision = DecisionReceipt(
        scope_sha256=digest,
        executor_uid=binding.executor_uid,
        producer=stop.producer,
        permit_sha256=permit.digest(),
        sequence=12,
        decision={
            "unchanged-owner": "ALLOWED",
            "ownership-drift": "STOP_OWNERSHIP_DRIFT",
            "late-sibling": "STOP_PARTICIPANTS_CHANGED",
        }[scenario],
        checked_node_uids=tuple(item.uid for item in binding.nodes),
        hardware_submitted=scenario == "unchanged-owner",
        restart_submitted=scenario == "unchanged-owner",
        product_guard="kubernetes-stop-ownership/v1",
    )
    quiet = QuiescenceReceipt(
        scope_sha256=digest,
        executor_uid=binding.executor_uid,
        producer=stop.producer,
        decision_sha256=decision.digest(),
        sequence=13,
        open_commands=0,
        pending_callbacks=0,
        workflow_terminal=True,
        gate_revoked=True,
    )
    ends = tuple(
        WitnessEnd(
            scope_sha256=digest,
            executor_uid=binding.executor_uid,
            producer=start.producer,
            node=start.node,
            witness_id=start.witness_id,
            start_sha256=start.digest(),
            quiescence_sha256=quiet.digest(),
            sequence=2,
            tracee=start.tracee,
            executable_path=start.executable_path,
            executable_sha256=start.executable_sha256,
            lost_events=0,
            trace_complete=True,
            trace_sha256=DIGEST,
            trace_bytes=120,
            exec_events=2,
            actions=(
                (
                    PhysicalAction(
                        operation="RESET_GPU",
                        argv_sha256=DIGEST,
                        pid=500 + index,
                        started_ns=100,
                        ended_ns=200,
                        returncode=0,
                    ),
                )
                if scenario == "unchanged-owner"
                else ()
            ),
        )
        for index, start in enumerate(starts)
    )
    cleanup = CleanupReceipt(
        scope_sha256=digest,
        executor_uid=binding.executor_uid,
        producer=stop.producer,
        gate_revoked=True,
        callbacks_drained=True,
        commands_terminal=True,
        owned_resources_absent=True,
        workload_identity_preserved=True,
        nodes_restored=True,
    )
    return AcceptanceEvidence(
        scope=binding,
        witness_starts=starts,
        stop=stop,
        mutation=mutation,
        permit=permit,
        decision=decision,
        quiescence=quiet,
        witness_ends=ends,
        cleanup=cleanup,
    )
