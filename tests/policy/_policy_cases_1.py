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
    NvlinkDecodeRule,
    SxidClassification,
    SxidLinkScope,
    XidEvent,
    catalog_product_family,
    load_sxid_policy,
    load_xid_policy,
    parse_xid154_action,
)
from tests._builders import build_sxid_event
from tests.policy._support import NOW, _engine, xid

# Independent copy of policy.DIRECT_ACTION_MAP. Written out again on
# purpose: importing the engine's map would make "the four direct
# actions are mapped exactly" a tautology.


def test_nvlink_decode_rule_rejects_malformed_catalog_values() -> None:
    base = {
        "xid": 144,
        "subcodeName": "test",
        "v1Pattern": "-" * 32,
        "v2Pattern": "-" * 32,
        "errorStatus": "0x1",
        "recoveryAction": "RESET_GPU",
    }
    with pytest.raises(ValueError, match="32 binary"):
        NvlinkDecodeRule.model_validate({**base, "v1Pattern": "-" * 31})
    with pytest.raises(ValueError, match="hexadecimal"):
        NvlinkDecodeRule.model_validate({**base, "errorStatus": "not-hex"})


def test_unknown_xid_is_quarantined_and_idempotent() -> None:
    engine = GpuFaultPolicyEngine()
    event = xid(999, event_id="unknown")

    first = engine.evaluate_xid(event)
    second = engine.evaluate_xid(event)

    assert first.action is None
    assert first.safety_action is RecoveryAction.QUARANTINE
    assert first.source is ActionSource.SITE_SAFETY
    assert first.disposition is ActionDisposition.SITE_SAFETY
    assert first.requires_operator, "expected first.requires_operator to be truthy"
    assert first.marker.active, "expected first.marker.active to be truthy"
    assert second.duplicate, "expected second.duplicate to be truthy"
    assert second.marker.marker_id == first.marker.marker_id


def test_xid_154_dynamic_action_takes_precedence() -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(
        xid(63, event_id="dynamic", xid_154_action=DynamicRecoveryAction.RESTART_BM)
    )

    assert decision.action is RecoveryAction.REBOOT_NODE
    assert decision.source is ActionSource.NVIDIA_XID_154
    assert decision.official_action == "RESTART_BM"
    assert "XID 154" in decision.reasons[0]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("None", DynamicRecoveryAction.IGNORE),
        ("Drain P2P", DynamicRecoveryAction.DRAIN_P2P),
        ("Drain and Reset", DynamicRecoveryAction.DRAIN_AND_RESET),
        ("GPU Reset Required", DynamicRecoveryAction.RESET_GPU),
        ("Node Reboot Required", DynamicRecoveryAction.RESTART_BM),
    ],
)
def test_xid154_parser_accepts_only_official_labels(label, expected) -> None:
    message = (
        "NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action "
        f"changed from 0x0 (None) to 0x2 ({label})"
    )

    assert parse_xid154_action(message) is expected


def test_xid154_without_explicit_label_fails_closed() -> None:
    event = xid(
        154,
        event_id="xid154-missing-label",
        raw_message=(
            "NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed to 0x2"
        ),
    )

    decision = GpuFaultPolicyEngine().evaluate_xid(event)

    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.action is None


def test_xid_45_uses_correlated_primary_action() -> None:
    primary = xid(79, event_id="primary")
    companion = xid(45, event_id="companion", observed_at=NOW + timedelta(seconds=5))

    decision = GpuFaultPolicyEngine().evaluate_xid(
        companion, companion_events=[primary, companion]
    )

    assert decision.action is RecoveryAction.REBOOT_NODE
    assert "companion XID 79" in decision.reasons[0]


def test_xid_95_is_uncontained_and_stops_all_workloads() -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(xid(95, event_id="xid-95"))

    assert decision.containment is Containment.ALL_APPLICATIONS
    assert decision.action is RecoveryAction.RESET_GPU
    assert decision.pre_actions == [
        RecoveryAction.MARK_UNSCHEDULABLE,
        RecoveryAction.STOP_WORKLOAD,
    ]
    assert decision.marker.official_action == "RESET_GPU"
    assert decision.marker.policy_source == "NVIDIA_CATALOG"


def test_nvlink5_decodes_driver_specific_bits_but_does_not_guess() -> None:
    intr_info = int("------000000----------0000100001".replace("-", "0"), 2)
    event = xid(
        144,
        event_id="nvlink5",
        product="B100",
        driver_branch=570,
        intr_info=intr_info,
        error_status=2,
    )

    decision = GpuFaultPolicyEngine().evaluate_xid(event)

    assert decision.action is RecoveryAction.RESET_GPU
    assert decision.official_action == "RESET_GPU"
    assert decision.matched_decode_rules == ["SAW_MVB"]
    assert decision.disposition is ActionDisposition.EXECUTABLE


def test_repeated_ignore_xid_does_not_invent_site_escalation() -> None:
    engine = GpuFaultPolicyEngine()
    first = engine.evaluate_xid(xid(63, event_id="repeat-1"))
    second = engine.evaluate_xid(
        xid(63, event_id="repeat-2", observed_at=NOW + timedelta(hours=1))
    )
    third = engine.evaluate_xid(
        xid(63, event_id="repeat-3", observed_at=NOW + timedelta(hours=2))
    )

    assert first.action is RecoveryAction.NO_ACTION
    assert not first.marker.active, "expected first.marker.active to be falsy"
    assert second.action is RecoveryAction.NO_ACTION
    assert third.action is RecoveryAction.NO_ACTION
    assert third.official_action == "IGNORE"


@pytest.mark.parametrize(
    ("product", "family"),
    [
        ("A800", "A100"),
        ("NVIDIA H200 NVL", "H100"),
        ("GH200", "H100"),
        ("B200", "B100"),
        ("NVIDIA GB300 NVL", "GB200"),
    ],
)
def test_catalog_product_columns_represent_families(product: str, family: str) -> None:
    assert catalog_product_family(product) == family
    decision = GpuFaultPolicyEngine().evaluate_xid(
        xid(63, event_id=f"xid-63-{product}", product=product)
    )

    assert decision.official_action == "IGNORE"
    assert decision.action is RecoveryAction.NO_ACTION
    assert decision.disposition is ActionDisposition.MONITOR_ONLY


def test_catalog_family_matching_preserves_per_rule_scope() -> None:
    h200 = GpuFaultPolicyEngine().evaluate_xid(
        xid(136, event_id="h-family", product="NVIDIA H200 NVL")
    )
    b200 = GpuFaultPolicyEngine().evaluate_xid(
        xid(136, event_id="b-not-h-family", product="B200")
    )
    gb300 = GpuFaultPolicyEngine().evaluate_xid(
        xid(121, event_id="gb-family", product="GB300")
    )

    assert h200.action is RecoveryAction.RESET_GPU
    assert b200.disposition is ActionDisposition.NOT_APPLICABLE
    assert gb300.action is RecoveryAction.NO_ACTION


@pytest.mark.parametrize("code", [74, 80, 136])
def test_hopper_only_xids_do_not_apply_to_blackwell(code: int) -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(
        xid(code, event_id=f"hopper-only-{code}-on-b200", product="B200")
    )

    assert decision.disposition is ActionDisposition.NOT_APPLICABLE
    assert decision.action is None


@pytest.mark.parametrize(
    "code", [*range(126, 136), 139, *range(144, 151), 155, *range(159, 167), 170]
)
def test_blackwell_only_xids_do_not_apply_to_hopper(code: int) -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(
        xid(code, event_id=f"blackwell-only-{code}-on-h200", product="H200")
    )

    assert decision.disposition is ActionDisposition.NOT_APPLICABLE
    assert decision.action is None


@pytest.mark.parametrize(
    "code", [*range(126, 136), 139, *range(144, 151), 155, *range(159, 167), 170]
)
def test_blackwell_only_xids_are_applicable_to_b200(code: int) -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(
        xid(code, event_id=f"blackwell-only-{code}-on-b200", product="B200")
    )

    assert decision.disposition is not ActionDisposition.NOT_APPLICABLE


def test_fatal_access_sxid_requires_participating_gpu_reset() -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            "sxid-access",
            NOW,
            11001,
            SxidClassification.FATAL,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            link_scope=SxidLinkScope.ACCESS,
            link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
            participating_gpu_uuids=["GPU-a", "GPU-b"],
        )
    )

    assert decision.action is RecoveryAction.RESET_GPU
    assert decision.containment is Containment.GPU
    assert decision.marker.scope.gpu_uuids == ["GPU-a", "GPU-b"]


def test_fatal_trunk_sxid_is_not_mapped_to_single_gpu_reset() -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            "sxid-trunk",
            NOW,
            11001,
            SxidClassification.FATAL,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            link_scope=SxidLinkScope.TRUNK,
            link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
            fabric_partition="fabric-a",
            participating_gpu_uuids=["GPU-a", "GPU-b"],
        )
    )

    assert decision.action is None
    assert decision.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
    assert decision.disposition is ActionDisposition.EXECUTABLE
    assert decision.containment is Containment.FABRIC_PARTITION
    assert not decision.requires_operator, (
        "expected decision.requires_operator to be falsy"
    )


def test_fatal_trunk_sxid_requires_complete_fabric_scope() -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            "sxid-trunk-missing-inventory",
            NOW,
            11001,
            SxidClassification.FATAL,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            link_scope=SxidLinkScope.TRUNK,
            link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
            fabric_partition="fabric-a",
        )
    )

    assert decision.action is None
    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert "complete node GPU inventory" in decision.reasons[0]
    assert decision.requires_operator, (
        "expected decision.requires_operator to be truthy"
    )


def test_sxid_catalog_is_complete_and_pinned() -> None:
    policy = load_sxid_policy()
    rules = {rule.sxid: rule for rule in policy.rules}

    assert policy.coverage == "NVIDIA_FM_TABLES_21_24"
    assert len(rules) == 92
    assert rules[20001].classification is (SxidClassification.NON_FATAL)
    assert rules[12020].classification is (SxidClassification.ALWAYS_FATAL)
    assert rules[19084].official_action == ("RESET_ALL_GPUS_AND_NVSWITCHES")
    assert rules[10003].official_action == ("RESET_ALL_GPUS_AND_NVSWITCHES")


@pytest.mark.parametrize(
    "product", ["B100", "B200", "B300", "GB200", "NVIDIA GB300 NVL"]
)
def test_sxid_is_not_applicable_to_blackwell_products(product: str) -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            f"sxid-blackwell-{product}",
            NOW,
            22013,
            SxidClassification.NON_FATAL,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            product=product,
        )
    )

    assert decision.disposition is ActionDisposition.NOT_APPLICABLE
    assert decision.action is None
    assert decision.safety_action is RecoveryAction.QUARANTINE
    assert "Blackwell-family" in decision.reasons[0]


@pytest.mark.parametrize("product", ["A100", "H100", "H200", "GH200"])
def test_sxid_remains_applicable_to_hopper_and_earlier(product: str) -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            f"sxid-pre-blackwell-{product}",
            NOW,
            22013,
            SxidClassification.NON_FATAL,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            product=product,
        )
    )

    assert decision.disposition is ActionDisposition.MONITOR_ONLY
    assert decision.action is RecoveryAction.NO_ACTION


def test_sxid_with_explicit_unknown_product_fails_closed() -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            "sxid-unknown-product",
            NOW,
            22013,
            SxidClassification.NON_FATAL,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            product="X999",
        )
    )

    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.action is None
    assert decision.safety_action is RecoveryAction.QUARANTINE
    assert "product X999 is unknown" in decision.reasons[0]


def test_sxid_20001_fatal_conflict_fails_closed() -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            "sxid-20001-severity-conflict",
            NOW,
            20001,
            SxidClassification.FATAL,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            link_scope=SxidLinkScope.TRUNK,
            link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
            fabric_partition="fabric-a",
            participating_gpu_uuids=["GPU-a", "GPU-b"],
        )
    )

    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.action is None
    assert decision.safety_action is RecoveryAction.QUARANTINE
    assert decision.official_action == "IGNORE"
    assert "catalog=NON_FATAL, observed=FATAL" in decision.reasons[0]


def test_always_fatal_sxid_uses_host_restart_branch() -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            "sxid-12020",
            NOW,
            12020,
            SxidClassification.ALWAYS_FATAL,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            product="H200",
        )
    )

    assert decision.disposition is ActionDisposition.EXECUTABLE
    assert decision.official_action == "RESTART_BM"
    assert decision.action is RecoveryAction.REBOOT_NODE
    assert decision.containment is Containment.NODE


def test_unknown_always_fatal_claim_fails_closed() -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            "sxid-unknown-always-fatal",
            NOW,
            99999,
            SxidClassification.ALWAYS_FATAL,
            "NVIDIA_FABRIC_MANAGER_RUNTIME",
        )
    )

    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.action is None
    assert decision.safety_action is RecoveryAction.QUARANTINE


@pytest.mark.parametrize("code", [10003, 19084])
def test_code_specific_sxid_requires_full_fabric_reset(code: int) -> None:
    classification = (
        SxidClassification.FATAL if code == 10003 else SxidClassification.NON_FATAL
    )
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            f"sxid-{code}",
            NOW,
            code,
            classification,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            fabric_partition="fabric-a",
            participating_gpu_uuids=["GPU-a", "GPU-b"],
        )
    )

    assert decision.disposition is ActionDisposition.EXECUTABLE
    assert decision.official_action == ("RESET_ALL_GPUS_AND_NVSWITCHES")


@pytest.mark.parametrize("code", [11004, 12028])
def test_virtualization_only_sxid_fails_closed_on_hyperpod(code: int) -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            f"sxid-{code}",
            NOW,
            code,
            SxidClassification.NON_FATAL,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            product="H200",
        )
    )

    assert decision.disposition is ActionDisposition.BLOCKED_WORKFLOW
    assert decision.official_action == "RESTART_VM"
    assert decision.safety_action is RecoveryAction.QUARANTINE


def test_sxid_raw_link_scope_cannot_authorize_reset() -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            "sxid-untrusted-scope",
            NOW,
            11001,
            SxidClassification.FATAL,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            link_scope=SxidLinkScope.TRUNK,
            link_scope_source="RAW_MESSAGE",
            fabric_partition="fabric-a",
            participating_gpu_uuids=["GPU-a", "GPU-b"],
        )
    )

    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.official_action == "RESOLVE_TRUSTED_LINK_SCOPE"


@pytest.mark.parametrize(
    ("code", "investigatory_action"),
    [
        (20012, "CHECK_LINK_MECHANICAL_CONNECTIONS"),
        (10004, "CHECK_SYSTEM_COOLING"),
        (10005, "VERIFY_THERMAL_EVENT_CLEARED"),
    ],
)
def test_code_specific_investigatory_actions(
    code: int, investigatory_action: str
) -> None:
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            f"sxid-{code}",
            NOW,
            code,
            SxidClassification.NON_FATAL,
            "NVIDIA_FABRIC_MANAGER_CATALOG",
        )
    )

    assert decision.investigatory_action == investigatory_action
    if code == 20012:
        assert decision.official_action == "CHECK_MECHANICALS"
        assert decision.action is RecoveryAction.ESCALATE_OPERATOR
    else:
        assert decision.disposition is ActionDisposition.MONITOR_ONLY


def test_policy_is_generated_from_complete_pinned_catalog() -> None:
    policy = load_xid_policy()

    assert policy.coverage == "FULL_OFFICIAL_ARTIFACT"
    assert len(policy.catalog_rules) == 172
    assert len(policy.nvlink5.decode_rules) == 95
    assert policy.source_sha256 == (
        "7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5"
    )
    assert policy.generated_sha256 == (
        "ffb82509abb574577db3c5d759cec94df40ea25edaf4691e7bd4ba0a12ba0462"
    )
    assert policy.mapping_version == ("nvidia-xid-catalog-610/sha256:ffb82509abb57457")


def test_mechanical_workflow_uses_exact_approved_executor() -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(xid(54, event_id="mechanicals"))

    assert decision.official_action == "CHECK_MECHANICALS"
    assert decision.action is None
    assert decision.safety_action is None
    assert decision.disposition is ActionDisposition.EXECUTABLE
    assert decision.requires_operator, (
        "expected decision.requires_operator to be truthy"
    )


def test_xid_48_exact_workflow_handles_solo_and_63_companion() -> None:
    solo_event = xid(48, event_id="xid-48-solo")
    solo = GpuFaultPolicyEngine().evaluate_xid(
        solo_event, companion_events=[solo_event]
    )
    sibling = xid(63, event_id="xid-63")
    combined_event = xid(
        48, event_id="xid-48-combined", observed_at=NOW + timedelta(seconds=1)
    )
    combined = GpuFaultPolicyEngine().evaluate_xid(
        combined_event, companion_events=[sibling, combined_event]
    )

    assert solo.official_action == "RESET_GPU"
    assert solo.action is RecoveryAction.RESET_GPU
    assert combined.official_action == "DRAIN_AND_RESET"
    assert combined.pre_actions == [RecoveryAction.MARK_UNSCHEDULABLE]


def test_xid_48_companion_branch_never_weakens_the_solo_branch() -> None:
    """Whichever XID 48 branch is taken, containment is equivalent.

    The Catalog bucket is a co-occurrence test ("Solo: RESET_GPU / w/
    63 or 64: DRAIN_AND_RESET") and specifies no ordering, but the
    engine decides XID 48 in place at ingest, so whichever companion
    happens to be in the store wins. That is only acceptable while the
    two branches are equally containing: same action, same containment,
    and a cordon in both. Asserting the labels alone would pass even if
    someone dropped RESET_GPU from the solo branch.
    """
    engine = GpuFaultPolicyEngine()
    solo = engine.evaluate_xid(xid(48, event_id="w48-order-solo"))

    for code in (63, 64):
        companion = engine.evaluate_xid(
            xid(48, event_id=f"w48-order-{code}"),
            companion_events=[
                xid(
                    code,
                    event_id=f"w48-order-companion-{code}",
                    observed_at=NOW + timedelta(seconds=3),
                )
            ],
        )
        assert companion.action is RecoveryAction.RESET_GPU
        assert solo.action is companion.action
        assert solo.containment is companion.containment
        assert solo.severity == companion.severity
        assert solo.requires_operator == companion.requires_operator
        # The cordon must be present either way: explicitly via
        # pre_actions, or implicitly because RESET_GPU always compiles
        # MARK_UNSCHEDULABLE. Never neither.
        assert companion.pre_actions == [RecoveryAction.MARK_UNSCHEDULABLE]


def test_xid_48_only_merges_into_a_reset_bearing_companion() -> None:
    """XID 48 may share an incident with 64, never with 63.

    ``correlated_event_id`` makes the orchestrator adopt the
    companion's incident and workflow, so it is only safe when that
    companion actually carries an equivalent reset. XID 64 resolves to
    RESET_GPU; XID 63 resolves to IGNORE and would swallow the reset.
    """
    engine = _engine()

    with_63 = engine.evaluate_xid(
        xid(48, event_id="w48-merge-63"),
        companion_events=[
            xid(63, event_id="w48-companion-63", observed_at=NOW + timedelta(seconds=3))
        ],
    )
    assert with_63.official_action == "DRAIN_AND_RESET"
    assert with_63.correlated_event_id is None

    with_64 = engine.evaluate_xid(
        xid(48, event_id="w48-merge-64"),
        companion_events=[
            xid(64, event_id="w48-companion-64", observed_at=NOW + timedelta(seconds=3))
        ],
    )
    assert with_64.correlated_event_id == "w48-companion-64"

    # With both present the reset-bearing one must win, not the newest.
    with_both = engine.evaluate_xid(
        xid(48, event_id="w48-merge-both"),
        companion_events=[
            xid(64, event_id="w48-both-64", observed_at=NOW + timedelta(seconds=2)),
            xid(63, event_id="w48-both-63", observed_at=NOW + timedelta(seconds=8)),
        ],
    )
    assert with_both.correlated_event_id == "w48-both-64"


def test_xid_64_backlinks_the_earlier_xid_48_on_the_same_gpu() -> None:
    """The 48/64 pair must dedupe whichever one lands first.

    ``_workflow_xid_48`` only looks forward, so before this XID 64
    resolved to a bare RESET_GPU and compiled a second eight-step reset
    of a GPU that XID 48 had already scheduled one for. Backlinking is
    the alternative to holding XID 48 open for the companion window,
    which would delay containment of an uncorrectable error.
    """
    engine = _engine()
    earlier_48 = xid(48, event_id="backlink-48")

    backlinked = engine.evaluate_xid(
        xid(64, event_id="backlink-64", observed_at=NOW + timedelta(seconds=6)),
        companion_events=[earlier_48],
    )
    assert backlinked.action is RecoveryAction.RESET_GPU
    assert backlinked.correlated_event_id == "backlink-48"

    # A later XID 48 is the forward case: it merges into this 64
    # instead, so 64 must not claim it and build a cycle.
    leading_64 = engine.evaluate_xid(
        xid(64, event_id="leading-64"),
        companion_events=[
            xid(48, event_id="trailing-48", observed_at=NOW + timedelta(seconds=6))
        ],
    )
    assert leading_64.correlated_event_id is None

    # XID 63 owns no workflow to adopt, so it must never be a target.
    with_63_only = engine.evaluate_xid(
        xid(64, event_id="backlink-63-only"),
        companion_events=[
            xid(63, event_id="ignored-63", observed_at=NOW - timedelta(seconds=6))
        ],
    )
    assert with_63_only.correlated_event_id is None

    # A different GPU is a different fault, not a companion.
    other_gpu = engine.evaluate_xid(
        xid(64, event_id="backlink-other-gpu", observed_at=NOW + timedelta(seconds=6)),
        companion_events=[
            xid(48, event_id="other-gpu-48").model_copy(
                update={"gpu_uuid": "GPU-different", "pci_bdf": "0000:99:00.0"}
            )
        ],
    )
    assert other_gpu.correlated_event_id is None


def test_check_uvm_requires_evidence_and_follows_official_branch() -> None:
    engine = GpuFaultPolicyEngine()
    missing = engine.evaluate_xid(xid(159, event_id="uvm-missing", product="B100"))
    used = engine.evaluate_xid(
        xid(159, event_id="uvm-used", product="B100", uvm_in_use=True)
    )

    assert missing.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert used.official_action == "RESET_GPU"
    assert used.action is RecoveryAction.RESET_GPU


def test_catalog_minimum_version_is_enforced() -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(
        xid(95, event_id="old-driver", driver_branch=550, cuda_version="12.6")
    )

    assert decision.action is None
    assert decision.disposition is ActionDisposition.NOT_APPLICABLE


def test_all_direct_catalog_actions_are_exactly_normalized() -> None:
    policy = load_xid_policy()
    expected = {
        "IGNORE": RecoveryAction.NO_ACTION,
        "RESTART_APP": RecoveryAction.RESTART_WORKLOAD,
        "RESET_GPU": RecoveryAction.RESET_GPU,
        "RESTART_BM": RecoveryAction.REBOOT_NODE,
    }

    for rule in policy.catalog_rules:
        if rule.immediate_action not in expected or not rule.products:
            continue
        event = XidEvent(
            event_id=f"direct-{rule.xid}",
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=NOW,
            xid=rule.xid,
            gpu_uuid="GPU-a",
            product=rule.products[0],
            driver_branch=999,
            cuda_version="99.9",
        )
        decision = GpuFaultPolicyEngine(policy).evaluate_xid(event)

        assert decision.official_action == rule.immediate_action
        assert decision.action is expected[rule.immediate_action]


def test_approved_official_workflows_are_executable_without_remapping() -> None:
    cases = [
        (54, "H100", "CHECK_MECHANICALS"),
        (78, "H100", "UPDATE_SWFW"),
        (142, "GB200", "CONTACT_SUPPORT"),
    ]

    for code, product, official_action in cases:
        decision = GpuFaultPolicyEngine().evaluate_xid(
            xid(code, event_id=f"blocked-{code}", product=product)
        )

        assert decision.official_action == official_action
        assert decision.action is None
        assert decision.safety_action is None
        assert decision.disposition is ActionDisposition.EXECUTABLE
