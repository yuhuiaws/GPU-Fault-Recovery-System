"""GF-GENERAL-LIVE-032 unknown error: category-module verdict tests.

Driven over fakes: a ``FakeProbe`` stands in for the kmsg-mode replay host and a
``FakeRegional`` answers both the XID policy-decision read and the NVIDIA_KERNEL
activity read, so the unknown-code -> SITE_SAFETY grading and the dedup of a
repeated event are checked on the product's own ``FaultPolicyDecision`` shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.e2e.regional import general_live_unknown as mod
from scripts.e2e.regional.general_live_registry import coverage_rows
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def _spec(variant: str) -> object:
    return next(item for item in mod.VARIANTS if item.variant == variant)


def _host_script() -> dict[str, object]:
    return {
        "install-records": {"records": 1},
        "open-window": {"after": {"ActiveState": "active"}},
        "arm": {"armed": True, "fixture": "/run/.../kmsg.fixture"},
        "status": {"injected": {"fixture_lines": 5}},
        "close-window": {"clean": True},
    }


def _cpu_python(
    decisions: list[dict[str, object]],
    *,
    sxid_decisions: list[dict[str, object]] | None = None,
    workflows: list[dict[str, object]] | None = None,
):
    def responder(script: str, *arguments: str) -> dict[str, object]:
        if "list_xid_events" in script:
            return {"decisions": decisions}
        if "get_xid_policy_decision" in script:
            return {"decisions": sxid_decisions or []}
        return {
            "incidents": [],
            "workflows": workflows or [],
            "markers": [],
            "notifications": [],
            "batches": [],
            "metric_latest": [],
            "collector_statuses": [],
        }

    return responder


def _decision(**overrides: object) -> dict[str, object]:
    base = {
        "event_id": "xid-987-1",
        "xid": mod.UNKNOWN_XID,
        "disposition": "SITE_SAFETY",
        "action": None,
        "requires_operator": True,
        "incident_id": "inc-xid-987",
        "workflow_request_id": None,
        "duplicate": False,
        "reasons": ["XID is absent from the pinned NVIDIA Catalog"],
    }
    base.update(overrides)
    return base


def _ctx(variant: str, *, decisions: list[dict[str, object]], tmp_path: Path):
    spec = _spec(variant)
    fake = FakeProbe(_host_script())
    regional = FakeRegional(cpu_python=_cpu_python(decisions))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-kmsg@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    return ctx


def test_unknown_xid_is_site_safety_without_action(tmp_path: Path) -> None:
    ctx = _ctx("unknown_xid", decisions=[_decision()], tmp_path=tmp_path)
    result = mod.unknown_xid(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay", result


def test_unknown_xid_fails_if_it_earns_an_nvidia_action(tmp_path: Path) -> None:
    ctx = _ctx(
        "unknown_xid",
        decisions=[_decision(disposition="EXECUTABLE", action="RESET_GPU")],
        tmp_path=tmp_path,
    )
    result = mod.unknown_xid(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(
        "recovery action" in error or "SITE_SAFETY" in error
        for error in result["errors"]
    ), result["errors"]


def test_unknown_xid_inconclusive_without_injection(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantInconclusive

    spec = _spec("unknown_xid")
    fake = FakeProbe({**_host_script(), "status": {"injected": {"fixture_lines": 0}}})
    regional = FakeRegional(cpu_python=_cpu_python([]))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-kmsg@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    try:
        mod.unknown_xid(ctx)
    except VariantInconclusive as exc:
        assert "no reviewed records" in str(exc), str(exc)
    else:
        raise AssertionError("an empty kmsg fixture must be INCONCLUSIVE")


def test_duplicate_event_isolates_once(tmp_path: Path) -> None:
    decisions = [
        _decision(event_id="xid-987-1"),
        _decision(event_id="xid-987-2", duplicate=True, incident_id="inc-xid-987"),
    ]
    ctx = _ctx("duplicate", decisions=decisions, tmp_path=tmp_path)
    result = mod.duplicate(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_duplicate_fails_on_second_isolation(tmp_path: Path) -> None:
    decisions = [
        _decision(event_id="xid-987-1", incident_id="inc-a"),
        _decision(event_id="xid-987-2", incident_id="inc-b"),
    ]
    ctx = _ctx("duplicate", decisions=decisions, tmp_path=tmp_path)
    result = mod.duplicate(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("isolated" in error for error in result["errors"]), result["errors"]


def test_variants_match_the_coverage_row() -> None:
    expected = set(coverage_rows()["GF-GENERAL-LIVE-032"])
    assert {item.variant for item in mod.VARIANTS} == expected, (
        "032 registers every coverage variant"
    )
    implemented = {item.variant for item in mod.VARIANTS if item.execute is not None}
    assert implemented == {"unknown_xid", "duplicate", "missing_topology"}, (
        "032 implements the reachable kmsg variants"
    )
    deferred = [item for item in mod.VARIANTS if item.deferred]
    assert len(deferred) == 5, (
        "the other five 032 variants stay in the denominator as NOT_RUN"
    )
    version = _spec("missing_version")
    assert version.deferred.startswith("FIXTURE_NOT_READY") and (
        "kernel collector" in version.deferred
        and "gpu-fault-host-collector.service only" in version.deferred
    ), "missing_version names the unit whose driver-version query matters"
    assert version.lanes == ("eks", "hp")
    topology = _spec("missing_topology")
    assert topology.lanes == ("eks", "hp") and topology.approvals == (
        mod.KERNEL_SOURCE_REPLAY,
    )


# --------------------------------------------------------------------------- #
# missing_topology: a fatal SXID with no trusted port-scope evidence
# --------------------------------------------------------------------------- #
def _sxid_decision(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "event_id": f"kernel-log-r1-sxid-{mod.MISSING_TOPOLOGY_SXID}-abc-cluster-deadbeef",
        "incident_id": "inc-sxid",
        "incident_state": "QUARANTINED",
        "disposition": "BLOCKED_MISSING_EVIDENCE",
        "official_action": "RESET_ALL_GPUS_AND_NVSWITCHES",
        "investigatory_action": None,
        "action": None,
        "safety_action": "QUARANTINE",
        "pre_actions": ["MARK_UNSCHEDULABLE", "STOP_WORKLOAD"],
        "requires_operator": True,
        "workflow_request_id": "wf-sxid",
        "duplicate": False,
        "reasons": ["fatal SXID requires a trusted access/trunk port-scope mapping"],
    }
    base.update(overrides)
    return base


def _quarantine_workflow(*operations: str) -> dict[str, object]:
    return {
        "request_id": "wf-sxid",
        "incident_id": "inc-sxid",
        "status": "SUCCEEDED",
        "created_at": "2026-10-24T00:00:30+00:00",
        "official_steps": [
            {"operation": item, "parameters": {}} for item in operations
        ],
        "step_executions": [],
    }


def _topology_ctx(
    decisions: list[dict[str, object]],
    workflows: list[dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, list[list[dict[str, object]]]]:
    spec = _spec("missing_topology")
    fake = FakeProbe(_host_script())
    regional = FakeRegional(
        cpu_python=_cpu_python([], sxid_decisions=decisions, workflows=workflows)
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-kmsg@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )
    restored: list[list[dict[str, object]]] = []

    def restore(
        context: object, incidents: list[dict[str, object]], *, reason: str
    ) -> list:
        restored.append(list(incidents))
        return [
            {"incident_id": item["incident_id"], "status": "SUCCEEDED"}
            for item in incidents
        ]

    monkeypatch.setattr(mod, "restore_incidents", restore)
    return ctx, restored


def test_missing_topology_is_blocked_with_the_gap_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, restored = _topology_ctx(
        [_sxid_decision()],
        [
            _quarantine_workflow(
                "FREEZE_EVIDENCE",
                "COLLECT_DIAGNOSTIC_BUNDLE",
                "MARK_UNSCHEDULABLE",
                "QUARANTINE",
            )
        ],
        tmp_path,
        monkeypatch,
    )
    result = mod.missing_topology(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay" and result["incidents"] == [
        "inc-sxid"
    ]
    assert restored == [[{"incident_id": "inc-sxid"}]], "the quarantine is handed back"
    assert "Link" not in mod.MISSING_TOPOLOGY_LINE.split(",", 2)[-1].replace("link", "")
    assert str(mod.MISSING_TOPOLOGY_SXID) in mod.MISSING_TOPOLOGY_LINE


def test_missing_topology_fails_when_the_reset_was_granted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    granted = _sxid_decision(
        disposition="EXECUTABLE",
        action=None,
        requires_operator=False,
        reasons=[
            "fatal trunk SXID requires stopping Fabric Manager and coordinated reset"
        ],
    )
    ctx, _restored = _topology_ctx(
        [granted],
        [
            _quarantine_workflow(
                "FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "RESET_ALL_GPUS_NVSWITCHES"
            )
        ],
        tmp_path,
        monkeypatch,
    )
    result = mod.missing_topology(ctx)
    assert result["verdict"] == "FAIL", result
    joined = "\n".join(result["errors"])
    assert (
        "expected BLOCKED_MISSING_EVIDENCE" in joined and "port-scope mapping" in joined
    ), joined
    assert (
        "RESET_ALL_GPUS_NVSWITCHES" in joined
        and "did not require an operator" in joined
    ), joined


def test_missing_topology_fails_without_a_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, restored = _topology_ctx([], [], tmp_path, monkeypatch)
    result = mod.missing_topology(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("did not reach a FaultPolicyDecision" in e for e in result["errors"]), (
        result
    )
    assert restored == [], "nothing to restore without an incident"


def test_the_reviewed_sxid_line_is_blocked_by_the_real_parser_and_engine() -> None:
    """kmsg filter -> nvidia_logs parser -> topology service -> policy engine."""

    from datetime import datetime, timezone

    from gpu_fault.collectors.logs.fabric_manager import SXID_SUMMARY_PATTERN
    from gpu_fault.models import RecoveryAction
    from gpu_fault.nvidia_logs import NvidiaKernelLogEvent, NvidiaLogNormalizer
    from gpu_fault.policy.catalog import load_sxid_policy
    from gpu_fault.policy.engine import GpuFaultPolicyEngine
    from gpu_fault.policy.models import (
        NVIDIA_ALWAYS_FATAL_SXIDS,
        ActionDisposition,
        SxidClassification,
        SxidLinkScope,
    )
    from gpu_fault.telemetry import NvSwitchPortTopologyService
    from tests._builders import build_store

    assert SXID_SUMMARY_PATTERN.search(mod.MISSING_TOPOLOGY_LINE), (
        "the kernel collector forwards Fabric Manager summary lines only"
    )
    rule = next(
        item
        for item in load_sxid_policy().rules
        if item.sxid == mod.MISSING_TOPOLOGY_SXID
    )
    assert rule.classification is SxidClassification.FATAL
    assert rule.official_action == "SCOPE_DEPENDENT_RESET"
    assert mod.MISSING_TOPOLOGY_SXID not in NVIDIA_ALWAYS_FATAL_SXIDS
    source = NvidiaKernelLogEvent(
        cluster_id="gpu-eks-cluster",
        node_id=NODE,
        record_id="r1",
        observed_at=datetime(2026, 10, 24, tzinfo=timezone.utc),
        message=mod.MISSING_TOPOLOGY_LINE,
        product="H200",
        driver_branch=580,
        runtime_profile_version="simulated-v1",
    )
    (event,) = NvidiaLogNormalizer().normalize_kernel(source).sxid_events
    assert event.sxid == mod.MISSING_TOPOLOGY_SXID
    assert event.classification is SxidClassification.FATAL
    assert event.port is None and event.switch_id == "0"
    assert event.link_scope is SxidLinkScope.UNKNOWN
    store = build_store()
    resolved = NvSwitchPortTopologyService(store).resolve(event)
    assert resolved.link_scope is SxidLinkScope.UNKNOWN, (
        "no trusted nvswitch_port_topology and no port: the H200 invariant cannot apply"
    )
    decision = GpuFaultPolicyEngine().evaluate_sxid(resolved)
    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
    assert decision.action is None and decision.requires_operator
    assert decision.safety_action is RecoveryAction.QUARANTINE
    assert any(mod.TOPOLOGY_REASON in reason for reason in decision.reasons), (
        decision.reasons
    )
    view = {
        "event_id": decision.event_id,
        "incident_id": None,
        "disposition": decision.disposition.value,
        "official_action": decision.official_action,
        "action": None,
        "requires_operator": decision.requires_operator,
        "reasons": list(decision.reasons),
    }
    assert mod.missing_topology_errors([view]) == []
    linked = source.model_copy(
        update={"message": mod.MISSING_TOPOLOGY_LINE + " Link 3"}
    )
    (with_link,) = NvidiaLogNormalizer().normalize_kernel(linked).sxid_events
    assert (
        NvSwitchPortTopologyService(store).resolve(with_link).link_scope
        is SxidLinkScope.ACCESS
    ), (
        "a line naming a link would get ACCESS from the H200 invariant: the replay must not name one"
    )
