"""The silence scan refreshes a silent channel, never a collector that reports.

Live 2026-10-29 (GF-REGIONAL-COLLECT-019, release R13i, both clusters): a
``nvidia-smi`` that hangs past the host collector's 15 s timeout makes every
host batch carry the timeout and then the open circuit breaker -- the
collector is up, posting on time and naming its own failure (ARCH-G5). The
silence scan read the first such batch as UNKNOWN (``COLLECTION_ERROR``) and
requested the node's hourly observation refresh; its TRIGGER_HEALTH_SNAPSHOT
restarted the metrics and the host collector 94 s (HyperPod) and 108 s
(plain EKS) into the hang. The restart cannot clear an error the source keeps
raising, it throws away the breaker the collector had built up (three more
rounds of 2 x 15 s timeouts) and it interrupts the healthy metrics collector
beside it; the refresh's VALIDATE_HOST then waits out its cap for a clean
host batch that the hang never delivers. R12 had no refresh at all.

The observation record is unchanged -- a reporting collector's error is still
UNKNOWN for validation and spare admission -- but only a channel that has
delivered no batch at all, clean or erroring, within its silence threshold
asks for the refresh, and the refresh waits for those channels only.
"""

from __future__ import annotations

from datetime import timedelta

from gpu_fault.app import notify_silent_collectors
from gpu_fault.collector_requirements import required_collectors_for_agent
from gpu_fault.fleet import AgentRecord
from gpu_fault.health_recovery_contract import HEALTH_CONTRACT
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.regional import RegionalClusterRegistration
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests.execution._health_policy_e2e_support import HealthRig

#: What the shipped host collector posts while ``nvidia-smi`` hangs: the
#: timeout first, then the open breaker (live HOST_TELEMETRY rows, R13i).
HUNG_NVIDIA_SMI = {
    "timeout": [
        "_gpu_utilization: CollectorError: nvidia-smi timed out after 15s",
        "_gpu_inventory: CollectorError: nvidia-smi timed out after 15s",
    ],
    "breaker": [
        "_gpu_utilization: CollectorError: nvidia-smi circuit breaker open "
        "(3 consecutive timeouts); GPU queries skipped this round",
        "_gpu_inventory: CollectorError: nvidia-smi circuit breaker open "
        "(3 consecutive timeouts); GPU queries skipped this round",
    ],
}


def _register(rig: HealthRig) -> AgentRecord:
    # The refresh is one request per node and wall-clock hour; start one
    # minute into the next hour so a scenario never straddles two buckets.
    rig.clock = (rig.clock + timedelta(hours=1)).replace(
        minute=1, second=0, microsecond=0
    )
    rig.observe()
    rig.store.save_regional_cluster(
        RegionalClusterRegistration(
            cluster_id="cluster-a",
            region="us-west-2",
            hyperpod_cluster_name="hp-a",
            eks_cluster_arn="arn:aws:eks:us-west-2:1:cluster/a",
            token_sha256="a" * 64,
            agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
        )
    )
    agent = AgentRecord(
        cluster_id="cluster-a",
        node_id="node-a",
        endpoint="https://127.0.0.1",
        agent_version="test",
        artifact_sha256="a" * 64,
        policy_version="test",
        runtime_profile_version="simulated-v1",
        config_digest="b" * 64,
        allowed_operations=[Op.TRIGGER_HEALTH_SNAPSHOT],
        first_seen_at=rig.clock,
        last_seen_at=rig.clock,
        lease_expires_at=rig.clock + timedelta(hours=1),
    )
    rig.store.save_agent(agent)
    return agent


def _report(
    rig: HealthRig,
    agent: AgentRecord,
    *,
    erroring: dict[CollectorKind, list[str]] | None = None,
    silent: frozenset[CollectorKind] = frozenset(),
) -> None:
    """One round of every required channel at the rig's clock: a clean batch,
    a batch that names its collection errors, or nothing at all."""

    for collector in required_collectors_for_agent(agent):
        if collector in silent:
            continue
        errors = (erroring or {}).get(collector, [])
        rig.store.save_collector_status(
            CollectorStatus(
                cluster_id="cluster-a",
                node_id="node-a",
                collector=collector,
                observed_at=rig.clock,
                ingested_at=rig.clock,
                last_success_at=None if errors else rig.clock,
                last_error_at=rig.clock if errors else None,
                sample_count=1,
                errors=list(errors),
            )
        )


def _scan(rig: HealthRig) -> None:
    # The product's own thresholds: inventory 180 s, metrics/host 420 s.
    notify_silent_collectors(
        rig.context,
        observed_at=rig.clock,
        silent_after_seconds={},
        alert_interval_seconds=3600,
    )


def _observation(rig: HealthRig, collector: CollectorKind) -> dict[str, str]:
    [sample] = [
        item
        for item in rig.store.list_telemetry_metrics_latest("cluster-a", "node-a")
        if item.name == "collector_observation_valid" and item.device == collector.value
    ]
    return dict(sample.labels)


def test_a_host_collector_reporting_a_hung_nvidia_smi_is_unknown_but_not_refreshed(
    tmp_path, monkeypatch
):
    rig = HealthRig(tmp_path, monkeypatch)
    try:
        agent = _register(rig)
        monkeypatch.setattr(
            rig.context.advisory_notifications, "send", lambda _id: None
        )
        _report(rig, agent)
        _scan(rig)
        assert rig.store.list_workflows() == []

        rig.advance(45)
        _report(
            rig,
            agent,
            erroring={CollectorKind.HOST_TELEMETRY: HUNG_NVIDIA_SMI["timeout"]},
        )
        _scan(rig)
        assert _observation(rig, CollectorKind.HOST_TELEMETRY) == {
            "state": "UNKNOWN",
            "reason": "COLLECTION_ERROR",
        }, "a reporting collector's own error is still UNKNOWN evidence"
        # The hang outlasts the host channel's 420 s threshold: the last
        # success ages out while every batch keeps arriving on time.
        for _ in range(10):
            rig.advance(45)
            _report(
                rig,
                agent,
                erroring={CollectorKind.HOST_TELEMETRY: HUNG_NVIDIA_SMI["breaker"]},
            )
            _scan(rig)
        assert _observation(rig, CollectorKind.HOST_TELEMETRY)["state"] == "UNKNOWN"
        assert rig.store.list_workflows() == [], (
            "the refresh's TRIGGER_HEALTH_SNAPSHOT restarts a collector that is "
            "up and naming its own failure: "
            + repr(
                [
                    [step.operation.value for step in workflow.official_steps]
                    for workflow in rig.store.list_workflows()
                ]
            )
        )
        assert rig.boundary.calls == [], rig.boundary.calls
    finally:
        rig.store.close()


def test_a_silent_channel_is_refreshed_alone_beside_an_erroring_one(
    tmp_path, monkeypatch
):
    rig = HealthRig(tmp_path, monkeypatch)
    try:
        agent = _register(rig)
        monkeypatch.setattr(
            rig.context.advisory_notifications, "send", lambda _id: None
        )
        _report(rig, agent)
        _scan(rig)
        # The metrics collector stops posting while nvidia-smi hangs for the
        # host collector, which keeps reporting its breaker.
        stalled = frozenset({CollectorKind.GPU_INVENTORY, CollectorKind.GPU_METRICS})
        for _ in range(5):
            rig.advance(60)
            _report(
                rig,
                agent,
                erroring={CollectorKind.HOST_TELEMETRY: HUNG_NVIDIA_SMI["breaker"]},
                silent=stalled,
            )
            _scan(rig)
        [workflow] = rig.store.list_workflows()
        assert [step.operation for step in workflow.official_steps] == [
            Op.FREEZE_EVIDENCE,
            Op.TRIGGER_HEALTH_SNAPSHOT,
            Op.VALIDATE_HOST,
        ], workflow.official_steps
        contract = workflow.official_steps[-1].parameters[HEALTH_CONTRACT]
        assert contract["collectors"] == [CollectorKind.GPU_INVENTORY.value], (
            "the refresh waits for the silent channel, not for the error the "
            f"host collector keeps reporting: {contract}"
        )
    finally:
        rig.store.close()
