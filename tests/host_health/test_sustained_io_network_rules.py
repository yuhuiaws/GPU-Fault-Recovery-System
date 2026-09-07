"""Disk saturation and packet drops need a sustain window, not one sample.

Siblings of the TCP-retransmit flapper: ``network_drops_delta >= 1``,
``disk_io_util_percent >= 98`` and ``disk_io_await_ms >= 100`` used to mint a
RUN_DIAGNOSTICS workflow from a single 15 s sample. A congested second or one
slow flush is not a node fault; the same reading held for a window is.
"""

from __future__ import annotations

from datetime import timedelta

from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from gpu_fault.models import RecoveryAction
from tests._builders import build_store, host_telemetry_batch
from tests.host_health._support import NOW


def _evaluate(policy: NodeHealthPolicy, name: str, value: float, offset: float):
    return policy.evaluate_metrics(
        host_telemetry_batch(
            f"{name}-{int(offset)}",
            NOW + timedelta(seconds=offset),
            [HostMetricSample(name=name, value=value, device="dm-0")],
        )
    )


def test_single_sample_breaches_do_not_mint_a_finding() -> None:
    for name, value in (
        ("network_drops_delta", 1),
        ("disk_io_util_percent", 99),
        ("disk_io_await_ms", 250),
    ):
        policy = NodeHealthPolicy(build_store())
        findings = _evaluate(policy, name, value, 0)
        assert findings == [], (name, findings)


def test_breach_held_for_the_sustain_window_mints_one_finding() -> None:
    for name, value, window in (
        ("network_drops_delta", 1, NodeHealthPolicy.NETWORK_DROPS_SUSTAIN_SECONDS),
        ("disk_io_util_percent", 99, NodeHealthPolicy.DISK_IO_SUSTAIN_SECONDS),
        ("disk_io_await_ms", 250, NodeHealthPolicy.DISK_IO_SUSTAIN_SECONDS),
    ):
        policy = NodeHealthPolicy(build_store())
        assert _evaluate(policy, name, value, 0) == [], name
        assert _evaluate(policy, name, value, window / 2) == [], name
        findings = _evaluate(policy, name, value, window)
        assert [item.metric_name for item in findings] == [name], (name, findings)
        assert findings[0].recommended_action is RecoveryAction.RUN_DIAGNOSTICS, name
        # Re-emission until the deliverer latches ``notified`` is the
        # existing P0-38B contract shared by every rule; only the first
        # emission's timing is this rule's business.


def test_breach_that_clears_before_the_window_never_fires() -> None:
    policy = NodeHealthPolicy(build_store())
    assert _evaluate(policy, "network_drops_delta", 3, 0) == []
    assert _evaluate(policy, "network_drops_delta", 0, 15) == []
    assert _evaluate(policy, "network_drops_delta", 2, 30) == []
    assert _evaluate(policy, "network_drops_delta", 0, 45) == [], (
        "drops that stopped before the window still minted a finding"
    )


def test_error_counter_rules_stay_immediate() -> None:
    for name in ("network_errors_delta", "rdma_errors_delta"):
        policy = NodeHealthPolicy(build_store())
        findings = _evaluate(policy, name, 1, 0)
        assert [item.metric_name for item in findings] == [name], (name, findings)
