from datetime import timedelta

from gpu_fault.app import notify_silent_collectors
from gpu_fault.collector_requirements import required_collectors_for_agent
from gpu_fault.fleet import AgentRecord
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowStatus
from gpu_fault.regional import RegionalClusterRegistration
from gpu_fault.telemetry import CollectorStatus
from tests.execution._health_policy_e2e_support import HealthRig
from tests.hyperpod._cov95_runtime_spare_health import HealthHarness


def test_not_ready_spare_does_not_invent_hardware_reboot():
    harness = HealthHarness(ready=False, failure_threshold=2)
    first = harness.scan()
    second = harness.scan()
    assert first["state"] == second["state"] == "SUSPECT", (first, second)
    assert harness.store.list_workflows() == [], "readiness loss is not hardware proof"
    assert harness.node["spec"]["unschedulable"] is True, harness.node


def test_silent_collector_refresh_is_bounded_and_requires_fresh_success(
    tmp_path, monkeypatch
):
    rig = HealthRig(tmp_path, monkeypatch)
    try:
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
        sent = []
        monkeypatch.setattr(rig.context.advisory_notifications, "send", sent.append)
        for _ in range(2):
            notify_silent_collectors(
                rig.context,
                observed_at=rig.clock,
                silent_after_seconds=420,
                alert_interval_seconds=3600,
            )
        [workflow] = rig.store.list_workflows()
        assert rig.execute(workflow).status is WorkflowStatus.RUNNING
        assert rig.boundary.calls == [Op.TRIGGER_HEALTH_SNAPSHOT], rig.boundary.calls
        rig.advance(1)
        for collector in required_collectors_for_agent(agent):
            rig.store.save_collector_status(
                CollectorStatus(
                    cluster_id="cluster-a",
                    node_id="node-a",
                    collector=collector,
                    observed_at=rig.clock,
                    ingested_at=rig.clock,
                    last_success_at=rig.clock,
                    sample_count=1,
                )
            )
        outcome = rig.execute(workflow)
        assert outcome.status is WorkflowStatus.SUCCEEDED, outcome
        assert rig.boundary.calls == [Op.TRIGGER_HEALTH_SNAPSHOT], rig.boundary.calls
        assert rig.core.node["spec"]["unschedulable"] is False, rig.core.node
    finally:
        rig.store.close()
