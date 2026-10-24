from datetime import timedelta

from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from gpu_fault.models import EfaTrafficSignal, RecoveryAction
from tests._builders import build_store
from tests.host_health._support import NOW, efa_telemetry, observe_running_attempt


def test_unknown_traffic_breaks_zero_streak_without_forgetting_attempt_baseline():
    store = build_store()
    policy = NodeHealthPolicy(store)
    for offset, rate, valid in (
        (0, 2_000_000, True),
        (130, 0, True),
        (200, 0, True),
        (700, 0, False),
        (740, 0, True),
    ):
        at = NOW + timedelta(seconds=offset)
        observe_running_attempt(store, at)
        batch = efa_telemetry(f"sample-{offset}", rate, at)
        if not valid:
            batch = batch.model_copy(
                update={
                    "samples": [
                        HostMetricSample(name="efa_traffic_observation_valid", value=0)
                    ]
                }
            )
        findings = policy.evaluate_metrics(batch)
        assert not any(
            item.diagnostic_parameters["diagnostic_reason"]
            == "EFA_TRAFFIC_HUNG_SUSPECTED"
            for item in findings
        ), findings
    state = store.get_efa_traffic_state("cluster-a/node-a")
    assert state.signal is EfaTrafficSignal.ZERO_PENDING
    assert state.zero_since == NOW + timedelta(seconds=740)
    assert state.had_active_traffic is True


def test_unknown_inventory_failure_is_diagnostic_not_reboot():
    policy = NodeHealthPolicy(build_store())
    batch = efa_telemetry("unknown-inventory", 0, NOW).model_copy(
        update={"samples": [HostMetricSample(name="efa_inventory_mismatch", value=1)]}
    )
    finding = policy.evaluate_metrics(batch)[0]
    assert finding.recommended_action is RecoveryAction.RUN_DIAGNOSTICS
    assert finding.diagnostic_parameters["failure_mode"] == "INVENTORY_UNAVAILABLE"
