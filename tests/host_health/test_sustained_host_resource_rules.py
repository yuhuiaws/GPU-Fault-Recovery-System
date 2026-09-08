"""CPU saturation, load and memory pressure need a sustain window, not a sample.

``cpu_usage_percent >= 98``, ``load1_per_cpu >= 2`` and
``memory_used_percent >= 95`` minted a WARNING ``RUN_DIAGNOSTICS`` finding --
and so an incident, a marker and a ``VALIDATE_HOST`` workflow -- from one 15 s
sample. A data-loader burst or a checkpoint flush is not a node fault; the same
reading held for two minutes is. Siblings of the disk-IO rule, and gated the
same way so the host collector's edge filter still ships the breach.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from gpu_fault.models import RecoveryAction, Severity
from tests._builders import build_store, host_telemetry_batch
from tests.host_health._support import NOW

RULES = (
    ("cpu_usage_percent", 99.0, 40.0),
    ("load1_per_cpu", 3.0, 0.5),
    ("memory_used_percent", 97.0, 60.0),
)


def _evaluate(policy: NodeHealthPolicy, name: str, value: float, offset: float):
    return policy.evaluate_metrics(
        host_telemetry_batch(
            f"{name}-{int(offset)}",
            NOW + timedelta(seconds=offset),
            [HostMetricSample(name=name, value=value)],
        )
    )


def test_host_resource_rules_share_the_120s_sustain_window() -> None:
    assert NodeHealthPolicy.HOST_RESOURCE_SUSTAIN_SECONDS == 120.0
    for name, _, _ in RULES:
        assert NodeHealthPolicy.METRIC_RULE_SUSTAIN_SECONDS[name] == 120.0, name
        assert name in NodeHealthPolicy.METRIC_RULES, (
            f"{name} must stay in METRIC_RULES so the collector ships the breach"
        )


@pytest.mark.parametrize(("name", "high", "normal"), RULES, ids=[r[0] for r in RULES])
def test_a_breach_that_clears_after_60s_never_fires(name, high, normal) -> None:
    policy = NodeHealthPolicy(build_store())

    assert _evaluate(policy, name, high, 0) == []
    assert _evaluate(policy, name, high, 30) == []
    assert _evaluate(policy, name, high, 60) == []
    assert _evaluate(policy, name, normal, 75) == [], (
        f"{name} that fell back before the window still minted a finding"
    )
    assert _evaluate(policy, name, normal, 200) == []


@pytest.mark.parametrize(("name", "high", "normal"), RULES, ids=[r[0] for r in RULES])
def test_a_breach_held_for_130s_mints_one_run_diagnostics_finding(
    name, high, normal
) -> None:
    policy = NodeHealthPolicy(build_store())

    assert _evaluate(policy, name, high, 0) == []
    assert _evaluate(policy, name, high, 60) == []
    findings = _evaluate(policy, name, high, 130)

    assert [item.metric_name for item in findings] == [name], (name, findings)
    assert findings[0].recommended_action is RecoveryAction.RUN_DIAGNOSTICS
    assert findings[0].severity is Severity.WARNING


def test_swap_and_filesystem_rules_stay_immediate() -> None:
    policy = NodeHealthPolicy(build_store())

    swap = _evaluate(policy, "swap_used_percent", 90, 0)
    assert [item.metric_name for item in swap] == ["swap_used_percent"], swap
    assert "swap_used_percent" not in NodeHealthPolicy.METRIC_RULE_SUSTAIN_SECONDS
