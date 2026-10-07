"""EKS node auto repair: plain EKS's counterpart of HyperPod ``NodeRecovery``."""

from __future__ import annotations

import pytest

from gpu_fault import eks_node_repair as repair


@pytest.mark.parametrize(
    ("nodegroup", "state"),
    [
        ({"nodegroupName": "gpu"}, "unset"),
        ({"nodeRepairConfig": None}, "unset"),
        ({"nodeRepairConfig": {}}, "unset"),
        ({"nodeRepairConfig": {"enabled": None}}, "unset"),
        ({"nodeRepairConfig": {"enabled": False}}, "disabled"),
        ({"nodeRepairConfig": {"enabled": True}}, "enabled"),
        ({"nodeRepairConfig": {"enabled": "true"}}, "disabled"),
    ],
)
def test_node_repair_state_reads_the_opt_in_field_literally(nodegroup, state) -> None:
    assert repair.node_repair_state(nodegroup) == state, (
        "only a boolean true is enabled; the absent opt-in field is unset, not disabled"
    )


def test_aggregate_node_repair_fails_closed_on_any_enabled_group() -> None:
    assert repair.aggregate_node_repair([]) == "unset"
    assert repair.aggregate_node_repair(["disabled", "disabled"]) == "disabled"
    assert repair.aggregate_node_repair(["disabled", "unset"]) == "unset"
    assert repair.aggregate_node_repair(["disabled", "enabled", "unset"]) == "enabled"


def test_node_repair_error_speaks_the_node_recovery_guards_language() -> None:
    assert repair.node_repair_error("gpu", "gpu-eks", "disabled") is None
    assert repair.node_repair_error("gpu", "gpu-eks", "unset") is None
    message = repair.node_repair_error("gpu", "gpu-eks", "enabled")
    assert message == (
        "EKS node group gpu of gpu-eks has node auto repair enabled "
        "(nodeRepairConfig.enabled=true); the GPU node groups of an eks-managed "
        "cluster must set it false -- as a HyperPod cluster must set "
        "NodeRecovery=None -- or it races this solution's own recovery"
    )
