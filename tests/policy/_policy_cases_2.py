from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.models import RecoveryAction
from gpu_fault.policy import (
    ActionDisposition,
    ActionSource,
    Containment,
    DynamicRecoveryAction,
    GpuFaultPolicyEngine,
    XidEvent,
)
from tests.policy._support import (
    NOW,
    _catalog_matrix_cases,
    _engine,
    _expected_xid_outcome,
    _nvlink5_decode_cases,
    _nvlink5_error_status_values,
    _nvlink5_expected_action,
    _pinned_catalogs,
    _version_gated_rules,
    xid,
)

# Independent copy of policy.DIRECT_ACTION_MAP. Written out again on
# purpose: importing the engine's map would make "the four direct
# actions are mapped exactly" a tautology.


def test_unowned_restart_vm_workflow_remains_fail_closed() -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(
        xid(151, event_id="blocked-151", product="H100")
    )

    assert decision.official_action == "RESTART_VM"
    assert decision.action is None
    assert decision.safety_action is RecoveryAction.QUARANTINE
    assert decision.disposition is ActionDisposition.BLOCKED_WORKFLOW


@pytest.mark.parametrize(
    ("registers", "expected_action", "expected_disposition"),
    [
        (
            [1 << 8, 0, 0, 0, 0, 0, 0],
            RecoveryAction.RESET_GPU,
            ActionDisposition.EXECUTABLE,
        ),
        (
            [1 << 0, 0, 0, 0, 1 << 20, 0, 0],
            RecoveryAction.NO_ACTION,
            ActionDisposition.MONITOR_ONLY,
        ),
        (
            [0, 0, 0, 1 << 18, 0, 0, 0],
            RecoveryAction.ESCALATE_OPERATOR,
            ActionDisposition.EXECUTABLE,
        ),
        (
            [0, 1, 0, 0, 0, 0, 0],
            RecoveryAction.ESCALATE_OPERATOR,
            ActionDisposition.EXECUTABLE,
        ),
    ],
)
def test_xid74_register_workflow_is_deterministic(
    registers, expected_action, expected_disposition
) -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(
        xid(
            74,
            event_id=f"xid74-{registers}",
            registers=registers,
            nvlink_link_id=2,
            nvlink_occurrence_counts={
                f"register{register_index + 1}.bit{bit}": 1
                for register_index, value in enumerate(registers)
                for bit in range(value.bit_length())
                if value & (1 << bit)
            },
        )
    )

    assert decision.action is expected_action
    assert decision.disposition is expected_disposition
    if registers[3] & (1 << 18):
        assert decision.containment is Containment.FABRIC_PARTITION
        assert "fabric reset" in decision.reasons[0]


@pytest.mark.parametrize(
    ("registers", "expected_action", "expected_disposition", "expected_rule"),
    [
        pytest.param(
            [1 << 0, 0, 0, 0, 0, 0, 0],
            RecoveryAction.NO_ACTION,
            ActionDisposition.MONITOR_ONLY,
            "safe_ignore",
            id="safe-only",
        ),
        pytest.param(
            [0, 0, 0, 0, 1 << 20, 0, 0],
            RecoveryAction.NO_ACTION,
            ActionDisposition.MONITOR_ONLY,
            "corrected_threshold",
            id="corrected-only",
        ),
        pytest.param(
            [0, 0, 0, 0, 0, 0, 0],
            RecoveryAction.ESCALATE_OPERATOR,
            ActionDisposition.EXECUTABLE,
            None,
            id="all-zero",
        ),
        pytest.param(
            [1 << 1, 0, 0, 0, 0, 0, 0],
            RecoveryAction.ESCALATE_OPERATOR,
            ActionDisposition.EXECUTABLE,
            "secondary",
            id="secondary-only",
        ),
        pytest.param(
            [1 << 21, 0, 0, 0, 0, 0, 0],
            RecoveryAction.ESCALATE_OPERATOR,
            ActionDisposition.EXECUTABLE,
            "marginal_channel",
            id="marginal-channel",
        ),
        pytest.param(
            [0, 0, 1 << 13, 0, 0, 0, 0],
            RecoveryAction.ESCALATE_OPERATOR,
            ActionDisposition.EXECUTABLE,
            "unexpected_production",
            id="unexpected-production",
        ),
        pytest.param(
            [0, 0, 0, 1 << 18, 0, 0, 0],
            RecoveryAction.ESCALATE_OPERATOR,
            ActionDisposition.EXECUTABLE,
            "fabric_reset_required",
            id="fabric-reset",
        ),
        pytest.param(
            [0, 1, 0, 0, 0, 0, 0],
            RecoveryAction.ESCALATE_OPERATOR,
            ActionDisposition.EXECUTABLE,
            None,
            id="unknown-register",
        ),
        pytest.param(
            [1 << 4, 0, 0, 0, 0, 0, 0],
            RecoveryAction.NO_ACTION,
            ActionDisposition.MONITOR_ONLY,
            "ecc_parity",
            id="ecc-parity",
        ),
        pytest.param(
            [1 << 27, 0, 0, 0, 0, 0, 0],
            RecoveryAction.NO_ACTION,
            ActionDisposition.MONITOR_ONLY,
            "report_if_repeated",
            id="report-if-repeated",
        ),
    ],
)
def test_xid74_register_branch_matrix(
    registers, expected_action, expected_disposition, expected_rule
) -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(
        xid(
            74,
            event_id="xid74-register-branch",
            registers=registers,
            nvlink_link_id=2,
            nvlink_occurrence_counts={
                f"register{register_index + 1}.bit{bit}": 1
                for register_index, value in enumerate(registers)
                for bit in range(value.bit_length())
                if value & (1 << bit)
            },
        )
    )

    assert decision.action is expected_action
    assert decision.disposition is expected_disposition
    if expected_rule is None:
        assert not decision.matched_decode_rules
    else:
        assert any(
            item.endswith(f":{expected_rule}") for item in decision.matched_decode_rules
        )


def test_xid74_link_scoped_category_fails_closed_without_link() -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(
        xid(74, event_id="xid74-link-unknown", registers=[1 << 4, 0, 0, 0, 0, 0, 0])
    )

    assert decision.action is None
    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert "explicit NVLink identity" in decision.reasons[0]


def test_xid74_hopper_decoder_is_not_applied_to_a100() -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(
        xid(
            74,
            event_id="xid74-a100",
            product="A100",
            registers=[1 << 8, 0, 0, 0, 0, 0, 0],
        )
    )

    assert decision.action is RecoveryAction.ESCALATE_OPERATOR
    assert "refusing to apply" in decision.reasons[0]


@pytest.mark.parametrize(
    ("index", "rule", "branch", "is_v1", "intr_info", "error_status"),
    _nvlink5_decode_cases(),
)
def test_every_nvlink5_decode_rule_resolves_as_the_catalog_says(
    index: int, rule, branch: int, is_v1: bool, intr_info: int, error_status: int
) -> None:
    """Cover all 95 pinned XID 144-150 decode rules, both branches.

    Every rule is driven on both sides of ``driverBoundary`` (so the
    v1 and v2 bit patterns are each exercised) and once per declared
    Error Status value. The expected official action is recomputed
    from the pinned catalog by the independent helpers above -- the
    engine's own decode path is never consulted to build it.
    """

    rules = _pinned_catalogs()[0].nvlink5.decode_rules
    event = xid(
        rule.xid,
        event_id=f"nvlink5-{index}-{is_v1}-{error_status}",
        product="B100",
        driver_branch=branch,
        intr_info=intr_info,
        error_status=error_status,
    )

    decision = _engine().evaluate_xid(event)

    assert decision.official_action == _nvlink5_expected_action(
        rules, rule.xid, intr_info, error_status, is_v1
    )
    assert rule.subcode_name in decision.matched_decode_rules
    assert decision.disposition is not (ActionDisposition.BLOCKED_MISSING_EVIDENCE)


def test_nvlink5_decode_table_has_no_unreachable_rule() -> None:
    """Every pinned rule must be reachable from its own pattern.

    A rule whose bit pattern is shadowed by an earlier, broader rule
    would silently never contribute a recovery action. Asserted here
    as a table-level invariant so a future catalog regeneration that
    introduces a dead row fails loudly.
    """

    rules = _pinned_catalogs()[0].nvlink5.decode_rules
    engine = _engine()
    boundary = _pinned_catalogs()[0].nvlink5.driver_boundary
    unreachable = []
    for is_v1 in (True, False):
        branch = boundary - 5 if is_v1 else boundary
        for index, rule in enumerate(rules):
            pattern = rule.v1_pattern if is_v1 else rule.v2_pattern
            intr_info = int(pattern.replace("-", "0"), 2)
            for error_status in _nvlink5_error_status_values(rule.error_status):
                decision = engine.evaluate_xid(
                    xid(
                        rule.xid,
                        event_id=(f"reach-{index}-{is_v1}-{error_status}"),
                        product="B100",
                        driver_branch=branch,
                        intr_info=intr_info,
                        error_status=error_status,
                    )
                )
                if rule.subcode_name not in (decision.matched_decode_rules):
                    unreachable.append((rule.xid, rule.subcode_name, is_v1))

    assert unreachable == []
    assert len(rules) == 95


# Independent copy of policy.DIRECT_ACTION_MAP. Written out again on
# purpose: importing the engine's map would make "the four direct
# actions are mapped exactly" a tautology.


@pytest.mark.parametrize(("rule", "family"), _catalog_matrix_cases())
def test_every_catalog_rule_resolves_as_the_catalog_says(rule, family: str) -> None:
    """Cover every XID in the pinned catalog on every product column.

    Running only the H100 rows would miss the dangerous direction:
    executing an action on a product the catalog never approved it for.
    ``catalog_supports_product`` decides applicability against each
    rule's own ``products`` column, not a global switch, so the matrix
    is the only way to pin that down.
    """

    event = XidEvent(
        event_id=f"matrix-{rule.xid}-{family}",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW,
        xid=rule.xid,
        gpu_uuid="GPU-a",
        product=family,
        driver_branch=999,
        cuda_version="99.9",
    )

    decision = _engine().evaluate_xid(event)
    expected_official, expected_disposition, expected_action = _expected_xid_outcome(
        rule, family
    )

    assert decision.official_action == expected_official
    assert decision.disposition is expected_disposition
    assert decision.action is expected_action
    if expected_disposition is ActionDisposition.NOT_APPLICABLE:
        # Fail-closed: an inapplicable rule must never yield an
        # executable action, and must be quarantined rather than
        # silently dropped.
        assert decision.action is None
        assert decision.safety_action is RecoveryAction.QUARANTINE


@pytest.mark.parametrize(("rule", "min_driver", "min_cuda"), _version_gated_rules())
def test_version_gate_branches_are_exact(
    rule, min_driver: int | None, min_cuda: str | None
) -> None:
    """Every reporting-compatibility gate, in both directions.

    The last assertion is the important one: a gate written with ``<=``
    instead of ``<`` would reject a *real* fault reported by a driver
    that exactly meets the documented minimum, and silently drop it as
    NOT_APPLICABLE. Asserting only "older is rejected" would pass for
    the buggy comparison too.
    """

    engine = _engine()
    family = rule.products[0]

    def evaluate(suffix: str, **overrides):
        payload = {"driver_branch": min_driver, "cuda_version": min_cuda}
        payload.update(overrides)
        return engine.evaluate_xid(
            XidEvent(
                event_id=f"gate-{rule.xid}-{suffix}",
                cluster_id="cluster-a",
                node_id="node-a",
                observed_at=NOW,
                xid=rule.xid,
                gpu_uuid="GPU-a",
                product=family,
                **payload,
            )
        )

    if min_driver is not None:
        missing_driver = evaluate("no-driver", driver_branch=None)
        assert missing_driver.disposition is (
            ActionDisposition.BLOCKED_MISSING_EVIDENCE
        )
        assert missing_driver.requires_operator
        older_driver = evaluate("old-driver", driver_branch=min_driver - 1)
        assert older_driver.disposition is (ActionDisposition.NOT_APPLICABLE)
        assert older_driver.action is None

    if min_cuda is not None:
        missing_cuda = evaluate("no-cuda", cuda_version=None)
        assert missing_cuda.disposition is (ActionDisposition.BLOCKED_MISSING_EVIDENCE)
        major, minor = (int(part) for part in min_cuda.split("."))
        older_cuda = f"{major}.{minor - 1}" if minor > 0 else f"{major - 1}.9"
        assert (
            evaluate("old-cuda", cuda_version=older_cuda).disposition
            is ActionDisposition.NOT_APPLICABLE
        )

    at_boundary = evaluate("boundary")
    assert at_boundary.disposition is not (ActionDisposition.NOT_APPLICABLE)


def test_xid_154_covers_every_official_recovery_action() -> None:
    """All six driver-reported XID 154 labels, plus the open window."""

    engine = _engine()
    expected = {
        DynamicRecoveryAction.IGNORE: (
            ActionDisposition.MONITOR_ONLY,
            RecoveryAction.NO_ACTION,
            Containment.GPU,
            [],
        ),
        DynamicRecoveryAction.DRAIN_P2P: (
            ActionDisposition.EXECUTABLE,
            RecoveryAction.STOP_WORKLOAD,
            Containment.ALL_APPLICATIONS,
            [RecoveryAction.STOP_WORKLOAD],
        ),
        DynamicRecoveryAction.DRAIN_AND_RESET: (
            ActionDisposition.EXECUTABLE,
            RecoveryAction.RESET_GPU,
            Containment.GPU,
            [RecoveryAction.MARK_UNSCHEDULABLE, RecoveryAction.STOP_WORKLOAD],
        ),
        DynamicRecoveryAction.RESTART_APP: (
            ActionDisposition.EXECUTABLE,
            RecoveryAction.RESTART_WORKLOAD,
            Containment.GPU,
            [],
        ),
        DynamicRecoveryAction.RESET_GPU: (
            ActionDisposition.EXECUTABLE,
            RecoveryAction.RESET_GPU,
            Containment.GPU,
            [],
        ),
        DynamicRecoveryAction.RESTART_BM: (
            ActionDisposition.EXECUTABLE,
            RecoveryAction.REBOOT_NODE,
            Containment.GPU,
            [],
        ),
    }
    assert set(expected) == set(DynamicRecoveryAction)

    for label, (disposition, action, containment, pre_actions) in expected.items():
        decision = engine.evaluate_xid(
            xid(154, event_id=f"xid154-{label.value}", xid_154_action=label)
        )

        assert decision.official_action == label.value
        assert decision.disposition is disposition
        assert decision.action is action
        assert decision.containment is containment
        assert decision.pre_actions == pre_actions
        assert decision.source is ActionSource.NVIDIA_XID_154

    pending = engine.evaluate_xid(
        xid(
            154,
            event_id="xid154-open-window",
            xid_154_action=DynamicRecoveryAction.RESET_GPU,
        ),
        xid154_window_closed=False,
    )

    assert pending.disposition is (ActionDisposition.PENDING_CORRELATION)
    assert pending.action is None


def test_xid_45_and_48_workflow_branches_are_exact() -> None:
    """Both correlation workflows, in every documented branch."""

    engine = _engine()

    solo45 = engine.evaluate_xid(xid(45, event_id="w45-solo"))
    assert solo45.official_action == "RESTART_FM"
    assert solo45.disposition is ActionDisposition.EXECUTABLE
    assert solo45.action is None
    assert solo45.containment is Containment.APPLICATION

    open45 = engine.evaluate_xid(
        xid(45, event_id="w45-open"), xid45_window_closed=False
    )
    assert open45.official_action == "WORKFLOW_XID_45"
    assert open45.disposition is (ActionDisposition.PENDING_CORRELATION)

    companion = xid(
        95, event_id="w45-companion-95", observed_at=NOW + timedelta(seconds=5)
    )
    followed = engine.evaluate_xid(
        xid(45, event_id="w45-not-solo"), companion_events=[companion]
    )
    assert followed.official_action == "RESET_GPU"
    assert followed.action is RecoveryAction.RESET_GPU
    assert followed.containment is Containment.ALL_APPLICATIONS
    assert followed.correlated_event_id == "w45-companion-95"

    solo48 = engine.evaluate_xid(xid(48, event_id="w48-solo"))
    assert solo48.official_action == "RESET_GPU"
    assert solo48.pre_actions == []

    for code in (63, 64):
        drained = engine.evaluate_xid(
            xid(48, event_id=f"w48-with-{code}"),
            companion_events=[
                xid(
                    code,
                    event_id=f"w48-companion-{code}",
                    observed_at=NOW + timedelta(seconds=3),
                )
            ],
        )
        assert drained.official_action == "DRAIN_AND_RESET"
        assert drained.action is RecoveryAction.RESET_GPU
        assert drained.pre_actions == [RecoveryAction.MARK_UNSCHEDULABLE]


def test_containment_exceptions_are_pinned() -> None:
    """XID 94/95 are the only per-XID containment exceptions.

    ``_xid_containment`` special-cases exactly these two codes; a
    "simplification" that always returned GPU scope would let an
    uncontained XID 95 be treated as a single-GPU fault.
    """

    engine = _engine()

    contained = engine.evaluate_xid(xid(94, event_id="contain-94"))
    assert contained.containment is Containment.APPLICATION
    assert contained.pre_actions == []

    uncontained = engine.evaluate_xid(xid(95, event_id="contain-95"))
    assert uncontained.containment is Containment.ALL_APPLICATIONS
    assert uncontained.pre_actions == [
        RecoveryAction.MARK_UNSCHEDULABLE,
        RecoveryAction.STOP_WORKLOAD,
    ]

    ordinary = engine.evaluate_xid(xid(46, event_id="contain-46"))
    assert ordinary.containment is Containment.GPU
