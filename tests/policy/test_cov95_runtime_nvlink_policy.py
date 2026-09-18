from __future__ import annotations

import pytest

from gpu_fault.models import RecoveryAction
from gpu_fault.policy import (
    ActionDisposition,
    DynamicRecoveryAction,
    GpuFaultPolicyEngine,
    load_sxid_policy,
    load_xid_policy,
)
from tests.orchestration._cov95_runtime_builder import xid


@pytest.fixture(scope="module")
def catalogs():
    return load_xid_policy(), load_sxid_policy()


@pytest.mark.parametrize(
    ("driver", "intr_info", "action"),
    [
        (570, 0x0C2, RecoveryAction.NO_ACTION),
        (570, 0x4C2, RecoveryAction.RESET_GPU),
        (580, 0x006, RecoveryAction.NO_ACTION),
        (580, 0x086, RecoveryAction.RESET_GPU),
    ],
)
@pytest.mark.parametrize("error_status", [1, 2, 4])
def test_nvlink5_rxpipe_secondary_pattern_selects_the_pinned_reset_action(
    catalogs, driver, intr_info, action, error_status
):
    event = xid(
        event_id=f"rxpipe-{driver}-{intr_info}-{error_status}",
        xid=145,
        product="B100",
        driver_branch=driver,
        intr_info=intr_info,
        error_status=error_status,
    )
    decision = GpuFaultPolicyEngine(*catalogs).evaluate_xid(event)
    assert decision.action is action, decision
    assert decision.matched_decode_rules == ["RLW_RXPIPE"], decision
    assert decision.disposition is (
        ActionDisposition.EXECUTABLE
        if action is RecoveryAction.RESET_GPU
        else ActionDisposition.MONITOR_ONLY
    ), decision


def test_nvlink5_dynamic_decode_uses_the_supplied_companion_recovery_action(catalogs):
    event = xid(
        xid=145,
        product="B100",
        driver_branch=580,
        intr_info=4,
        error_status=1,
        workload_state="ACTIVE",
    )
    companion = event.model_copy(
        update={
            "event_id": "companion-154",
            "xid": 154,
            "xid_154_action": DynamicRecoveryAction.RESET_GPU,
        }
    )
    result = GpuFaultPolicyEngine(*catalogs).evaluate_xid(
        event, companion_events=[companion], xid154_window_closed=False
    )
    assert result.action is RecoveryAction.RESET_GPU, result
    assert result.disposition is ActionDisposition.EXECUTABLE, result
    assert result.matched_decode_rules == ["RLW_REMAP"], result


@pytest.mark.parametrize(("driver", "intr_info"), [(570, 0x4C2), (580, 0x086)])
@pytest.mark.parametrize("error_status", [None, 3])
def test_secondary_nvlink_pattern_still_requires_an_exact_error_status(
    catalogs, driver, intr_info, error_status
):
    result = GpuFaultPolicyEngine(*catalogs).evaluate_xid(
        xid(
            xid=145,
            product="B100",
            driver_branch=driver,
            intr_info=intr_info,
            error_status=error_status,
        )
    )
    assert result.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE, result
    assert result.action is None and result.requires_operator, result
    assert result.matched_decode_rules == [], result


def test_unrecognized_injected_decode_action_cannot_authorize_recovery(catalogs):
    policy, sxid = catalogs
    rule = policy.nvlink5.decode_rules[0].model_copy(
        update={"recovery_action": "UNSUPPORTED_RECOVERY"}
    )
    policy = policy.model_copy(
        update={"nvlink5": policy.nvlink5.model_copy(update={"decode_rules": [rule]})}
    )
    result = GpuFaultPolicyEngine(policy, sxid).evaluate_xid(
        xid(xid=144, product="B100", driver_branch=580, intr_info=1, error_status=1)
    )
    assert result.disposition is ActionDisposition.BLOCKED_WORKFLOW, result
    assert result.action is None and result.requires_operator, result
    assert "conflicting recovery actions" in result.reasons[0], result
