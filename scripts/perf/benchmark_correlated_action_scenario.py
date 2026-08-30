from __future__ import annotations

import json
import os
import ssl
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import error, request


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
        "restart_budget": 0,
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
    weak_event_id = f"corr-live-{run_id}-c{offset:03d}-weak"
    strong_event_id = f"corr-live-{run_id}-c{offset:03d}-strong"
    reset_event_id = f"corr-live-{run_id}-c{offset:03d}-reset"
    weak = client.post(
        "/v1/gpu-events/xid",
        xid_payload(
            xid=48,
            event_id=weak_event_id,
            cluster_id=cluster_id,
            identity=primary_identity,
            observed_at=now,
            profile_version=profile_version,
        ),
    )
    weak_workflow_id = str(weak.get("workflow_request_id") or "")
    if not weak_workflow_id:
        raise RuntimeError("weak event did not create a workflow")

    executor_id = f"correlated-action-{offset:03d}"
    strong_workflow_id = ""
    primary_reboot_workflow_id = ""
    reset_workflow_id = ""
    reset_reboot_workflow_id = ""
    strong_sent = False
    primary_reset_failed = False
    reset_attempt_started = False
    reset_gpu_failed = False
    primary_containment: set[str] = set()
    completed_ids: set[str] = set()
    duplicate_claims = 0
    claim_errors = 0
    result_errors = 0
    idle_started: float | None = None
    deadline = time.monotonic() + float(os.getenv("SCENARIO_MAX_SECONDS", "600"))
    operations: dict[str, int] = {}

    while time.monotonic() < deadline:
        try:
            claim = client.post(
                "/v1/regional/executors/claim",
                claim_payload(executor_id),
            )
        except (error.HTTPError, error.URLError, OSError):
            claim_errors += 1
            time.sleep(0.1)
            continue
        commands = claim.get("commands") or []
        if not commands:
            if idle_started is None:
                idle_started = time.monotonic()
            if (
                reset_gpu_failed
                and reset_reboot_workflow_id
                and time.monotonic() - idle_started >= 20
            ):
                break
            if (
                primary_reboot_workflow_id
                and not reset_attempt_started
                and time.monotonic() - idle_started >= 2
            ):
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
                reset_workflow_id = str(reset.get("workflow_request_id") or "")
                if not reset_workflow_id:
                    raise RuntimeError("reset-only event did not create a workflow")
                reset_attempt_started = True
                idle_started = None
            time.sleep(0.1)
            continue
        idle_started = None
        command = commands[0]
        command_id = command["command_id"]
        workflow = command["workflow"]
        workflow_id = workflow["request_id"]
        operation = command["step"]["operation"]
        if command_id in completed_ids:
            duplicate_claims += 1
        try:
            fail_primary_reset = (
                bool(strong_workflow_id)
                and workflow_id == strong_workflow_id
                and operation == "RESET_ALL_GPUS_NVSWITCHES"
                and not primary_reset_failed
            )
            fail_reset_gpu = (
                bool(reset_workflow_id)
                and workflow_id == reset_workflow_id
                and operation == "RESET_GPU"
                and not reset_gpu_failed
            )
            command_result(
                client,
                command,
                failed=fail_primary_reset or fail_reset_gpu,
                executor_id=executor_id,
            )
        except Exception:
            result_errors += 1
            time.sleep(0.1)
            continue
        completed_ids.add(command_id)
        operations[operation] = operations.get(operation, 0) + 1
        if fail_primary_reset:
            primary_reset_failed = True
        if fail_reset_gpu:
            reset_gpu_failed = True
        if workflow_id == weak_workflow_id and operation in {
            "MARK_UNSCHEDULABLE",
            "STOP_WORKLOADS",
        }:
            primary_containment.add(operation)
        if (
            workflow_id == weak_workflow_id
            and not strong_sent
            and primary_containment == {"MARK_UNSCHEDULABLE", "STOP_WORKLOADS"}
        ):
            strong = client.post(
                "/v1/gpu-events/sxid",
                sxid_payload(
                    event_id=strong_event_id,
                    cluster_id=cluster_id,
                    identity=primary_identity,
                    observed_at=now + timedelta(seconds=1),
                    profile_version=profile_version,
                ),
            )
            strong_workflow_id = str(strong.get("workflow_request_id") or "")
            if not strong_workflow_id or strong_workflow_id == weak_workflow_id:
                raise RuntimeError("strong event did not create a successor")
            strong_sent = True
        if operation == "RESTART_NODE":
            predecessor = workflow.get("predecessor_workflow_id")
            if predecessor == strong_workflow_id or workflow_id == (
                f"workflow-reboot-after-{strong_workflow_id}"
            ):
                primary_reboot_workflow_id = workflow_id
            if predecessor == reset_workflow_id or workflow_id == (
                f"workflow-reboot-after-{reset_workflow_id}"
            ):
                reset_reboot_workflow_id = workflow_id

    return {
        "cluster_id": cluster_id,
        "weak_event_id": weak_event_id,
        "strong_event_id": strong_event_id,
        "reset_event_id": reset_event_id,
        "weak_workflow_id": weak_workflow_id,
        "strong_workflow_id": strong_workflow_id,
        "primary_reboot_workflow_id": primary_reboot_workflow_id,
        "reset_workflow_id": reset_workflow_id,
        "reset_reboot_workflow_id": reset_reboot_workflow_id,
        "strong_sent": strong_sent,
        "primary_reset_failed": primary_reset_failed,
        "reset_attempt_started": reset_attempt_started,
        "reset_gpu_failed": reset_gpu_failed,
        "completed_commands": len(completed_ids),
        "duplicate_claims": duplicate_claims,
        "claim_errors": claim_errors,
        "result_errors": result_errors,
        "operations": operations,
        "idle_terminal": idle_started is not None,
    }


def scenario_succeeded(output: dict) -> bool:
    return bool(
        output["strong_sent"]
        and output["primary_reset_failed"]
        and output["primary_reboot_workflow_id"]
        and output["reset_attempt_started"]
        and output["reset_gpu_failed"]
        and output["reset_reboot_workflow_id"]
        and not output["duplicate_claims"]
        and not output["claim_errors"]
        and not output["result_errors"]
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
