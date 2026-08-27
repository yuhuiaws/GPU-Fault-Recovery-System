from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import gpu_fault.cluster_executor as cluster_executor
from gpu_fault.cluster_executor import (
    ClusterActionExecutor,
    ClusterExecutorError,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import (
    AgentHeartbeatReporter,
    NodeActionCommand,
    NodeActionExecutor,
    NodeActionLedger,
    SignedNodeAction,
)


def main(endpoint: str, node_id: str) -> None:
    with urlopen(endpoint.rstrip("/") + "/healthz", timeout=5) as response:
        health = json.loads(response.read())
    assert health == {"status": "ok"}

    now = datetime.now(timezone.utc)
    command = NodeActionCommand(
        command_id="audit-p2-invalid-signature",
        workflow_request_id="audit-p2-workflow",
        incident_id="audit-p2-incident",
        fencing_token=1,
        operation=WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        node_id=node_id,
        issued_at=now,
        expires_at=now + timedelta(minutes=1),
    )
    request = Request(
        endpoint.rstrip("/") + "/v1/node-actions/submit",
        data=SignedNodeAction(
            command=command,
            signature="0" * 64,
        )
        .model_dump_json()
        .encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urlopen(request, timeout=5)
    except HTTPError as error:
        rejection = json.loads(error.read())
        assert error.code == 401
        assert rejection["detail"]["code"] == "INVALID_SIGNATURE"
        assert rejection["detail"]["retryable"] is False
    else:
        raise AssertionError("invalid signature was accepted")

    class FailingClient:
        def claim(self, *_args, **_kwargs):
            raise ClusterExecutorError("expired token")

    class Adapter:
        owner = "gpu-fault-node-agent"

    sleeps = []
    original_sleep = cluster_executor.time.sleep

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 4:
            raise StopIteration

    cluster_executor.time.sleep = sleep
    try:
        executor = ClusterActionExecutor(
            FailingClient(),
            [Adapter()],
            executor_id="audit-p2",
            allowed_namespaces={"gpu-fault-system"},
            poll_seconds=2,
            claim_backoff_max_seconds=8,
        )
        try:
            executor.run()
        except StopIteration:
            pass
        else:
            raise AssertionError("executor backoff probe did not stop")
    finally:
        cluster_executor.time.sleep = original_sleep
    assert sleeps == [2, 4, 8, 8]

    versions = iter(["policy-v1", "policy-v2"])
    reports = []
    reporter = AgentHeartbeatReporter(
        control_plane_url="https://control.example",
        secret="x" * 32,
        cluster_id="audit-cluster",
        node_id=node_id,
        endpoint=endpoint,
        agent_version="0.10.0",
        artifact_sha256="a" * 64,
        policy_version="initial",
        policy_version_provider=lambda: next(versions),
        runtime_profile_version="audit-profile",
        config_digest="b" * 64,
        allowed_operations=[WorkflowOperation.VERIFY_NO_GPU_CLIENTS],
        boot_id="audit-boot",
        sender=lambda _url, envelope: (
            reports.append(envelope.heartbeat.policy_version) or {}
        ),
    )
    reporter.report_once()
    reporter.report_once()
    assert reports == ["policy-v1", "policy-v2"]

    nested = "Pass"
    for _ in range(66):
        nested = {"nested": nested}
    try:
        NodeActionExecutor._dcgm_diagnostic_statuses(nested)
    except ValueError as error:
        assert "maximum depth" in str(error)
    else:
        raise AssertionError("deep DCGM JSON was accepted")

    with tempfile.TemporaryDirectory() as directory:
        agent = NodeActionExecutor(
            secret="x" * 32,
            node_ids={node_id},
            allowed_operations={WorkflowOperation.RESET_GPU},
            reset_enabled=True,
            single_gpu_reset_supported=False,
            ledger=NodeActionLedger(f"{directory}/node-actions.db"),
        )
        try:
            agent._reset_gpu(["GPU-a"])
        except RuntimeError as error:
            assert "single-GPU reset is not supported" in str(error)
        else:
            raise AssertionError("unsupported single-GPU reset was accepted")

    print(
        "PASS",
        {
            "health": health,
            "error_code": rejection["detail"]["code"],
            "claim_backoff": sleeps,
            "policy_versions": reports,
            "dcgm_depth_rejected": True,
            "single_gpu_reset_gate": True,
        },
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2])
