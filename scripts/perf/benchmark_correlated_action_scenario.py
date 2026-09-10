from __future__ import annotations

import json
import os
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import error, request
from urllib.parse import urlencode


OWNERS = [
    "gpu-fault-kubernetes-adapter",
    "gpu-fault-node-agent",
    "gpu-fault-hyperpod-adapter",
    "gpu-fault-validation-adapter",
    "gpu-fault-control-plane",
]
TERMINAL_COMMANDS = {"SUCCEEDED", "FAILED", "CANCELLED"}


class Client:
    def __init__(self, base_url: str, cluster_id: str, token: str, ca_file: str):
        self.base_url = base_url.rstrip("/")
        self.cluster_id = cluster_id
        self.token = token
        self.context = ssl.create_default_context(cafile=ca_file)

    def post(self, path: str, payload: dict) -> dict:
        value = request.Request(
            self.base_url + path,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "X-GPU-Fault-Cluster-ID": self.cluster_id,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(value, context=self.context, timeout=30) as response:
            return json.loads(response.read() or b"{}")

    def get(self, path: str) -> dict:
        value = request.Request(
            self.base_url + path,
            headers={
                "Authorization": f"Bearer {self.token}",
                "X-GPU-Fault-Cluster-ID": self.cluster_id,
            },
            method="GET",
        )
        with request.urlopen(value, context=self.context, timeout=30) as response:
            return json.loads(response.read() or b"{}")


def attempt_identity(
    run_id: str,
    offset: int,
    lane: str = "primary",
) -> dict[str, str]:
    suffix = f"{run_id}-c{offset:03d}-{lane}"
    return {
        "job_id": f"corr-job-{suffix}",
        "attempt_id": f"corr-attempt-{suffix}",
        "workload_id": f"training/PyTorchJob/corr-{suffix}",
        "node_id": f"corr-node-{suffix}",
        "gpu_uuid": f"GPU-corr-{offset:03d}",
    }


def observation_payload(
    cluster_id: str,
    identity: dict[str, str],
    observed_at: datetime,
    profile_version: str = "hyperpod-v1",
) -> dict:
    return {
        "cluster_id": cluster_id,
        "environment": "hyperpod-eks",
        "job_id": identity["job_id"],
        "attempt_id": identity["attempt_id"],
        "workload_phase": "RUNNING",
        "observed_at": observed_at.isoformat(),
        "started_at": (observed_at - timedelta(minutes=1)).isoformat(),
        "expected_critical_ranks": 1,
        "workload_ids": [identity["workload_id"]],
        "restart_budget": 1,
        "runtime_profile_version": profile_version,
        "containers": [
            {
                "pod_uid": f"pod-{identity['attempt_id']}",
                "pod_name": f"pod-{identity['attempt_id']}",
                "container_name": "trainer",
                "role": "worker",
                "rank": 0,
                "node_id": identity["node_id"],
                "gpu_count": 8,
                "gpu_uuids": [identity["gpu_uuid"]],
                "terminated": False,
            }
        ],
    }


def heartbeat_payload(
    cluster_id: str,
    identity: dict[str, str],
    observed_at: datetime,
) -> dict:
    return {
        "heartbeat_id": f"heartbeat-{identity['attempt_id']}",
        "cluster_id": cluster_id,
        "attempt_id": identity["attempt_id"],
        "rank": 0,
        "observed_at": observed_at.isoformat(),
        "node_id": identity["node_id"],
        "pod_uid": f"pod-{identity['attempt_id']}",
        "container_name": "trainer",
        "gpu_uuids": [identity["gpu_uuid"]],
        "step": 1,
        "samples_per_second": 1.0,
        "loss": 1.0,
        "labels": {"drill_id": "perf-capacity"},
    }


def xid_payload(
    *,
    xid: int,
    event_id: str,
    cluster_id: str,
    identity: dict[str, str],
    observed_at: datetime,
    profile_version: str,
) -> dict:
    return {
        "event_id": event_id,
        "cluster_id": cluster_id,
        "node_id": identity["node_id"],
        "observed_at": observed_at.isoformat(),
        "source_event_time": observed_at.isoformat(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "xid": xid,
        "gpu_uuid": identity["gpu_uuid"],
        "product": "H100",
        "driver_branch": 575,
        "cuda_version": "12.9",
        "runtime_profile_version": profile_version,
        "workload_state": "ACTIVE",
        "affected_workload_ids": [identity["workload_id"]],
        "drill_id": "perf-capacity",
        "synthetic": True,
    }


def sxid_payload(
    *,
    event_id: str,
    cluster_id: str,
    identity: dict[str, str],
    observed_at: datetime,
    profile_version: str,
) -> dict:
    return {
        "event_id": event_id,
        "cluster_id": cluster_id,
        "node_id": identity["node_id"],
        "observed_at": observed_at.isoformat(),
        "source_event_time": observed_at.isoformat(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "sxid": 11001,
        "classification": "FATAL",
        "classification_source": "NVIDIA_FABRIC_MANAGER_CATALOG",
        "link_scope": "TRUNK",
        "link_scope_source": "TRUSTED_NVSWITCH_TOPOLOGY",
        "product": "H200",
        "fabric_partition": f"{cluster_id}/{identity['node_id']}/local-nvswitch",
        "participating_gpu_uuids": [identity["gpu_uuid"]],
        "runtime_profile_version": profile_version,
        "workload_state": "ACTIVE",
        "affected_workload_ids": [identity["workload_id"]],
        "drill_id": "perf-capacity",
        "synthetic": True,
    }


def claim_payload(executor_id: str) -> dict:
    return {
        "executor_id": executor_id,
        "executor_protocol_version": int(os.environ["EXECUTOR_PROTOCOL_VERSION"]),
        "executor_artifact_sha256": os.environ["EXECUTOR_ARTIFACT_SHA256"],
        "executor_compatibility_digest": os.environ["EXECUTOR_COMPATIBILITY_DIGEST"],
        "execution_owners": OWNERS,
        "max_commands": 1,
        "lease_seconds": int(os.getenv("ACTION_LEASE_SECONDS", "120")),
    }


def command_result(
    client: Client,
    command: dict,
    *,
    failed: bool,
    executor_id: str,
) -> None:
    operation = command["step"]["operation"]
    details = {
        "simulated": True,
        "operation": operation,
        "executor_id": executor_id,
        "node_results": {
            node: {
                "status": "FAILED" if failed else "SUCCEEDED",
                **({"error": "synthetic reset failure"} if failed else {}),
            }
            for node in command["step"].get("node_ids") or []
        },
        **(
            {
                "failed_nodes": list(command["step"].get("node_ids") or []),
                "node_failures": {
                    node: ["synthetic reset failure"]
                    for node in command["step"].get("node_ids") or []
                },
            }
            if failed
            else {}
        ),
    }
    client.post(
        f"/v1/regional/executors/{command['command_id']}/result",
        {
            "lease_token": command["lease_token"],
            "status": "FAILED" if failed else "SUCCEEDED",
            "status_source": "correlated-action-simulator",
            "details": details,
            **({"error": "synthetic reset failure"} if failed else {}),
        },
    )


def post_attempt(
    client: Client,
    *,
    cluster_id: str,
    identity: dict[str, str],
    observed_at: datetime,
    profile_version: str,
) -> None:
    client.post(
        "/v1/workload-observations",
        observation_payload(
            cluster_id,
            identity,
            observed_at,
            profile_version=profile_version,
        ),
    )
    client.post(
        "/v1/training-progress",
        heartbeat_payload(cluster_id, identity, observed_at),
    )


def post_aggregated_weak_event(
    client: Client,
    *,
    cluster_id: str,
    identity: dict[str, str],
    run_id: str,
    offset: int,
    observed_at: datetime,
    profile_version: str,
) -> dict[str, object]:
    weak_event_id = f"corr-live-{run_id}-c{offset:03d}-weak"
    aggregate_event_id = f"corr-live-{run_id}-c{offset:03d}-aggregate"
    weak = client.post(
        "/v1/gpu-events/xid",
        xid_payload(
            xid=48,
            event_id=weak_event_id,
            cluster_id=cluster_id,
            identity=identity,
            observed_at=observed_at,
            profile_version=profile_version,
        ),
    )
    weak_workflow_id = str(weak.get("workflow_request_id") or "")
    if not weak_workflow_id:
        raise RuntimeError("weak event did not create a workflow")
    aggregate = client.post(
        "/v1/gpu-events/xid",
        xid_payload(
            xid=48,
            event_id=aggregate_event_id,
            cluster_id=cluster_id,
            identity=identity,
            observed_at=observed_at + timedelta(milliseconds=100),
            profile_version=profile_version,
        ),
    )
    shared_incident = bool(weak.get("incident_id")) and (
        aggregate.get("incident_id") == weak.get("incident_id")
    )
    shared_workflow = aggregate.get("workflow_request_id") == weak_workflow_id
    if not shared_incident or not shared_workflow:
        raise RuntimeError(
            "same-rank event did not aggregate into the weak incident/workflow"
        )
    return {
        "weak_event_id": weak_event_id,
        "aggregate_event_id": aggregate_event_id,
        "weak_workflow_id": weak_workflow_id,
        "aggregate_shared_incident": shared_incident,
        "aggregate_shared_workflow": shared_workflow,
    }


def incident_ownership(client: Client, incident_id: str) -> dict:
    return client.get(
        "/v1/regional/executors/incident-ownership?"
        + urlencode({"incident_id": incident_id})
    )


def ownership_succeeded(report: dict) -> bool:
    return bool(report.get("terminal")) and report.get("workflow_status") == "SUCCEEDED"


@dataclass
class ScenarioState:
    executor_id: str
    strong_workflow_id: str = ""
    primary_reboot_workflow_id: str = ""
    primary_reboot_incident_id: str = ""
    reset_workflow_id: str = ""
    reset_reboot_workflow_id: str = ""
    reset_reboot_incident_id: str = ""
    strong_sent: bool = False
    strong_in_record: bool = False
    primary_reset_failed: bool = False
    primary_reboot_succeeded: bool = False
    reset_attempt_started: bool = False
    reset_gpu_failed: bool = False
    reset_reboot_succeeded: bool = False
    primary_containment: set[str] = field(default_factory=set)
    completed_ids: set[str] = field(default_factory=set)
    duplicate_claims: int = 0
    claim_errors: int = 0
    result_errors: int = 0
    ownership_errors: int = 0
    idle_started: float | None = None
    last_ownership_poll: float = 0.0
    operations: dict[str, int] = field(default_factory=dict)


def _poll_ownership(client: Client, state: ScenarioState) -> None:
    if state.primary_reboot_incident_id and not state.primary_reboot_succeeded:
        primary = incident_ownership(client, state.primary_reboot_incident_id)
        if primary.get("terminal") and not ownership_succeeded(primary):
            raise RuntimeError(
                "primary reboot workflow reached a non-success terminal "
                f"state: {primary.get('workflow_status')}"
            )
        state.primary_reboot_succeeded = ownership_succeeded(primary)
    if state.reset_reboot_incident_id and not state.reset_reboot_succeeded:
        reset_report = incident_ownership(client, state.reset_reboot_incident_id)
        if reset_report.get("terminal") and not ownership_succeeded(reset_report):
            raise RuntimeError(
                "reset reboot workflow reached a non-success terminal "
                f"state: {reset_report.get('workflow_status')}"
            )
        state.reset_reboot_succeeded = ownership_succeeded(reset_report)


def _start_reset_attempt(
    client: Client,
    state: ScenarioState,
    *,
    cluster_id: str,
    reset_identity: dict[str, str],
    reset_event_id: str,
    profile_version: str,
) -> None:
    reset_now = datetime.now(timezone.utc)
    post_attempt(
        client,
        cluster_id=cluster_id,
        identity=reset_identity,
        observed_at=reset_now,
        profile_version=profile_version,
    )
    reset = client.post(
        "/v1/gpu-events/xid",
        xid_payload(
            xid=48,
            event_id=reset_event_id,
            cluster_id=cluster_id,
            identity=reset_identity,
            observed_at=reset_now,
            profile_version=profile_version,
        ),
    )
    state.reset_workflow_id = str(reset.get("workflow_request_id") or "")
    if not state.reset_workflow_id:
        raise RuntimeError("reset-only event did not create a workflow")
    state.reset_attempt_started = True
    state.idle_started = None


def _handle_scenario_command(
    client: Client,
    command: dict,
    state: ScenarioState,
    *,
    weak_workflow_id: str,
    strong_event_id: str,
    cluster_id: str,
    primary_identity: dict[str, str],
    observed_at: datetime,
    profile_version: str,
) -> None:
    command_id = command["command_id"]
    workflow = command["workflow"]
    workflow_id = workflow["request_id"]
    operation = command["step"]["operation"]
    if command_id in state.completed_ids:
        state.duplicate_claims += 1
    fail_primary_reset = (
        bool(state.strong_workflow_id)
        and workflow_id == state.strong_workflow_id
        and operation == "RESET_ALL_GPUS_NVSWITCHES"
        and not state.primary_reset_failed
    )
    fail_reset_gpu = (
        bool(state.reset_workflow_id)
        and workflow_id == state.reset_workflow_id
        and operation == "RESET_GPU"
        and not state.reset_gpu_failed
    )
    try:
        command_result(
            client,
            command,
            failed=fail_primary_reset or fail_reset_gpu,
            executor_id=state.executor_id,
        )
    except Exception:
        state.result_errors += 1
        time.sleep(0.1)
        return
    state.completed_ids.add(command_id)
    state.operations[operation] = state.operations.get(operation, 0) + 1
    state.primary_reset_failed = state.primary_reset_failed or fail_primary_reset
    state.reset_gpu_failed = state.reset_gpu_failed or fail_reset_gpu
    if workflow_id == weak_workflow_id and operation in {
        "MARK_UNSCHEDULABLE",
        "STOP_WORKLOADS",
    }:
        state.primary_containment.add(operation)
    if (
        workflow_id == weak_workflow_id
        and not state.strong_sent
        and state.primary_containment == {"MARK_UNSCHEDULABLE", "STOP_WORKLOADS"}
    ):
        strong = client.post(
            "/v1/gpu-events/sxid",
            sxid_payload(
                event_id=strong_event_id,
                cluster_id=cluster_id,
                identity=primary_identity,
                observed_at=observed_at + timedelta(seconds=1),
                profile_version=profile_version,
            ),
        )
        state.strong_workflow_id = str(strong.get("workflow_request_id") or "")
        if not state.strong_workflow_id:
            raise RuntimeError("strong event did not join or create a workflow")
        # A clean boundary (containment done, nothing physical in flight)
        # preempts inside the weak record: the fabric reset becomes the
        # node's successor branch under the same workflow id. A preemption
        # that lands on an in-flight physical step is a separate successor
        # record (DESTR-016's shape). Both are accepted; the audit tells them
        # apart.
        state.strong_in_record = state.strong_workflow_id == weak_workflow_id
        state.strong_sent = True
    if operation != "RESTART_NODE":
        return
    predecessor = workflow.get("predecessor_workflow_id")
    incident_id = str((command.get("incident") or {}).get("incident_id") or "")
    if predecessor == state.strong_workflow_id or workflow_id == (
        f"workflow-reboot-after-{state.strong_workflow_id}"
    ):
        state.primary_reboot_workflow_id = workflow_id
        state.primary_reboot_incident_id = incident_id
    if predecessor == state.reset_workflow_id or workflow_id == (
        f"workflow-reboot-after-{state.reset_workflow_id}"
    ):
        state.reset_reboot_workflow_id = workflow_id
        state.reset_reboot_incident_id = incident_id


def _scenario_output(
    state: ScenarioState,
    *,
    cluster_id: str,
    weak_event_id: str,
    aggregate_event_id: str,
    strong_event_id: str,
    reset_event_id: str,
    weak_workflow_id: str,
    aggregate_shared_incident: bool,
    aggregate_shared_workflow: bool,
) -> dict:
    return {
        "cluster_id": cluster_id,
        "weak_event_id": weak_event_id,
        "aggregate_event_id": aggregate_event_id,
        "strong_event_id": strong_event_id,
        "reset_event_id": reset_event_id,
        "weak_workflow_id": weak_workflow_id,
        "aggregate_shared_incident": aggregate_shared_incident,
        "aggregate_shared_workflow": aggregate_shared_workflow,
        "strong_workflow_id": state.strong_workflow_id,
        "primary_reboot_workflow_id": state.primary_reboot_workflow_id,
        "primary_reboot_incident_id": state.primary_reboot_incident_id,
        "primary_reboot_succeeded": state.primary_reboot_succeeded,
        "reset_workflow_id": state.reset_workflow_id,
        "reset_reboot_workflow_id": state.reset_reboot_workflow_id,
        "reset_reboot_incident_id": state.reset_reboot_incident_id,
        "reset_reboot_succeeded": state.reset_reboot_succeeded,
        "strong_sent": state.strong_sent,
        "strong_in_record": state.strong_in_record,
        "primary_reset_failed": state.primary_reset_failed,
        "reset_attempt_started": state.reset_attempt_started,
        "reset_gpu_failed": state.reset_gpu_failed,
        "completed_commands": len(state.completed_ids),
        "duplicate_claims": state.duplicate_claims,
        "claim_errors": state.claim_errors,
        "result_errors": state.result_errors,
        "ownership_errors": state.ownership_errors,
        "operations": state.operations,
        "idle_terminal": (
            state.primary_reboot_succeeded and state.reset_reboot_succeeded
        ),
    }


def run_scenario(
    client: Client,
    *,
    cluster_id: str,
    run_id: str,
    offset: int,
    profile_version: str,
) -> dict:
    primary_identity = attempt_identity(run_id, offset, "preempt")
    reset_identity = attempt_identity(run_id, offset, "reset")
    now = datetime.now(timezone.utc)
    post_attempt(
        client,
        cluster_id=cluster_id,
        identity=primary_identity,
        observed_at=now,
        profile_version=profile_version,
    )
    aggregated = post_aggregated_weak_event(
        client,
        cluster_id=cluster_id,
        identity=primary_identity,
        run_id=run_id,
        offset=offset,
        observed_at=now,
        profile_version=profile_version,
    )
    weak_event_id = str(aggregated["weak_event_id"])
    aggregate_event_id = str(aggregated["aggregate_event_id"])
    weak_workflow_id = str(aggregated["weak_workflow_id"])
    aggregate_shared_incident = bool(aggregated["aggregate_shared_incident"])
    aggregate_shared_workflow = bool(aggregated["aggregate_shared_workflow"])
    strong_event_id = f"corr-live-{run_id}-c{offset:03d}-strong"
    reset_event_id = f"corr-live-{run_id}-c{offset:03d}-reset"

    state = ScenarioState(executor_id=f"correlated-action-{offset:03d}")
    deadline = time.monotonic() + float(os.getenv("SCENARIO_MAX_SECONDS", "600"))

    while time.monotonic() < deadline:
        try:
            claim = client.post(
                "/v1/regional/executors/claim",
                claim_payload(state.executor_id),
            )
        except (error.HTTPError, error.URLError, OSError):
            state.claim_errors += 1
            time.sleep(0.1)
            continue
        commands = claim.get("commands") or []
        if not commands:
            if state.idle_started is None:
                state.idle_started = time.monotonic()
            now_monotonic = time.monotonic()
            if now_monotonic - state.last_ownership_poll >= 1:
                try:
                    _poll_ownership(client, state)
                except (error.HTTPError, error.URLError, OSError):
                    state.ownership_errors += 1
                state.last_ownership_poll = now_monotonic
            if state.primary_reboot_succeeded and not state.reset_attempt_started:
                _start_reset_attempt(
                    client,
                    cluster_id=cluster_id,
                    state=state,
                    reset_identity=reset_identity,
                    reset_event_id=reset_event_id,
                    profile_version=profile_version,
                )
            if state.reset_reboot_succeeded:
                break
            time.sleep(0.1)
            continue
        state.idle_started = None
        _handle_scenario_command(
            client,
            commands[0],
            state,
            weak_workflow_id=weak_workflow_id,
            strong_event_id=strong_event_id,
            cluster_id=cluster_id,
            primary_identity=primary_identity,
            observed_at=now,
            profile_version=profile_version,
        )

    return _scenario_output(
        state,
        cluster_id=cluster_id,
        weak_event_id=weak_event_id,
        aggregate_event_id=aggregate_event_id,
        strong_event_id=strong_event_id,
        reset_event_id=reset_event_id,
        weak_workflow_id=weak_workflow_id,
        aggregate_shared_incident=aggregate_shared_incident,
        aggregate_shared_workflow=aggregate_shared_workflow,
    )


def scenario_succeeded(output: dict) -> bool:
    return bool(
        output["aggregate_shared_incident"]
        and output["aggregate_shared_workflow"]
        and output["strong_sent"]
        and output["primary_reset_failed"]
        and output["primary_reboot_workflow_id"]
        and output["primary_reboot_succeeded"]
        and output["reset_attempt_started"]
        and output["reset_gpu_failed"]
        and output["reset_reboot_workflow_id"]
        and output["reset_reboot_succeeded"]
        and not output["duplicate_claims"]
        and not output["claim_errors"]
        and not output["result_errors"]
        and not output["ownership_errors"]
        and output["idle_terminal"]
    )


def main() -> None:
    registrations = json.loads(Path(os.environ["CLUSTERS_FILE"]).read_text())
    offset = int(os.environ["CLUSTER_OFFSET"])
    registration = registrations[offset]
    output = run_scenario(
        Client(
            os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
            registration["cluster_id"],
            registration["token"],
            os.environ["SSL_CERT_FILE"],
        ),
        cluster_id=registration["cluster_id"],
        run_id=os.environ["CORRELATED_ACTION_RUN_ID"],
        offset=offset,
        profile_version=os.getenv("RUNTIME_PROFILE_VERSION", "hyperpod-v1"),
    )
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    if not scenario_succeeded(output):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
